# Copyright 2026 Scott Weeden
# Author: Scott Weeden
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
=============================================================================
muzero/chassis.py - Kaggriculture MuZero Neural Network Chassis
=============================================================================
Neural architecture components:
  1. ResNetBlock2D: Pre-activation residual block preserving 10x10 spatial grid.
  2. RepresentationNetwork (h_θ): Encodes 10x10 spatial observation tensor [Batch, 28, 10, 10]
     into root latent state s_0 [Batch, 64, 10, 10], L2-normalized.
  3. DynamicsNetwork (g_θ): Predicts next latent state s^k and immediate reward
     distribution logits (601 atoms) from (s^{k-1}, action_plane).
  4. PredictionNetwork (f_θ) / FactorizedKaggriculturePredictionHead:
     Evaluates latent state s^k and predicts factorized policy priors
     (Farmer, Hands, Market) and expected state value (601 atoms).
     Supports candidate joint action sampling for Sampled MCTS.
  5. KaggricultureMuZeroChassis: Unified container for h_θ, g_θ, and f_θ.
  6. SimSiamProjectionPredictionHead & SimSiamConsistencyLoss:
     Asymmetric self-supervised consistency module with explicit stop-gradients.
  7. compute_muzero_unroll_loss_with_consistency: K-step unroll loss calculator.
=============================================================================
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatial_constants import ACTION_CHANNELS


# ============================================================================
# 1. RESNET & CHASSIS NETWORKS (h_theta, g_theta, f_theta)
# ============================================================================

class ResNetBlock2D(nn.Module):
    """Pre-activation ResNet block preserving 10x10 spatial grid dimensions."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class SpatialRepresentationNetwork(nn.Module):
    """
    h_theta: Encodes raw 10x10 spatial observation tensor o_t into root latent state s_0.
    Input:  [Batch, C_obs=28, 10, 10]
    Output: Latent state tensor s_0 [Batch, C_latent=64, 10, 10]
    """

    def __init__(self, in_channels: int = 28, latent_channels: int = 64, num_blocks: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.ModuleList([ResNetBlock2D(latent_channels) for _ in range(num_blocks)])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        s_0 = self.stem(obs)
        for block in self.res_blocks:
            s_0 = block(s_0)
        s_0 = F.normalize(s_0, p=2, dim=1)
        return s_0


class SpatialDynamicsNetwork(nn.Module):
    """
    g_theta: Predicts next latent state s^k and immediate reward r^k given (s^{k-1}, a^k).
    Inputs:
        s_prev: Latent state tensor [Batch, C_latent=64, 10, 10]
        action_plane: One-hot / spatial action encoding plane [Batch, C_action=16, 10, 10]
    Outputs:
        s_next: Transitioned latent state [Batch, C_latent=64, 10, 10]
        reward_logits: Categorical reward distribution logits [Batch, 601]
    """

    def __init__(
        self,
        latent_channels: int = 64,
        action_channels: int = ACTION_CHANNELS,
        num_blocks: int = 3,
        support_size: int = 601,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.action_channels = action_channels
        self.support_size = support_size
        in_channels = latent_channels + action_channels
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.ModuleList([ResNetBlock2D(latent_channels) for _ in range(num_blocks)])

        self.reward_head = nn.Sequential(
            nn.Conv2d(latent_channels, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(16 * 10 * 10, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, support_size),
        )

    def forward(self, s_prev: torch.Tensor, action_plane: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if s_prev.dim() == 3:
            s_prev = s_prev.unsqueeze(0)
        if action_plane.dim() == 3:
            action_plane = action_plane.unsqueeze(0)

        x = torch.cat([s_prev, action_plane], dim=1)
        s_next = self.stem(x)
        for block in self.res_blocks:
            s_next = block(s_next)

        s_next = F.normalize(s_next, p=2, dim=1)
        reward_logits = self.reward_head(s_next)

        return s_next, reward_logits


class SpatialPredictionNetwork(nn.Module):
    """
    f_theta: Predicts factorized policy priors p^k and value v^k from latent state s^k.
    Supports candidate joint action sampling for Sampled MCTS.
    """

    def __init__(
        self,
        latent_channels: int = 64,
        num_farmer_actions: int = 15,
        num_hand_assignments: int = 32,
        num_market_orders: int = 20,
        support_size: int = 601,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.num_farmer_actions = num_farmer_actions
        self.num_hand_assignments = num_hand_assignments
        self.num_market_orders = num_market_orders
        self.support_size = support_size

        self.trunk = nn.Sequential(
            nn.Conv2d(latent_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 10 * 10, 256),
            nn.ReLU(inplace=True),
        )

        self.farmer_head = nn.Linear(256, num_farmer_actions)
        self.hands_head = nn.Linear(256, num_hand_assignments)
        self.market_head = nn.Linear(256, num_market_orders)
        self.value_head = nn.Linear(256, support_size)

    def forward(self, s_k: torch.Tensor) -> Dict[str, torch.Tensor]:
        if s_k.dim() == 2:
            # Handles flattened 2D tensor: reshape back to spatial [Batch, latent_channels, 10, 10]
            if s_k.shape[1] == self.latent_channels * 10 * 10:
                s_k = s_k.view(-1, self.latent_channels, 10, 10)
            elif s_k.shape[1] == self.latent_channels:
                s_k = s_k.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 10, 10)
            else:
                raise ValueError(f"Unexpected 2D latent state shape: {s_k.shape}")
        elif s_k.dim() == 3:
            s_k = s_k.unsqueeze(0)

        features = self.trunk(s_k)
        return {
            "logits_farmer": self.farmer_head(features),
            "logits_hands": self.hands_head(features),
            "logits_market": self.market_head(features),
            "value_logits": self.value_head(features),
        }

    def sample_joint_actions(
        self,
        s_k: torch.Tensor,
        num_samples: int = 16,
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        preds = self.forward(s_k)

        logits_f = preds["logits_farmer"]
        logits_h = preds["logits_hands"]
        logits_m = preds["logits_market"]

        if masks is not None:
            if "farmer" in masks:
                logits_f = logits_f.masked_fill(masks["farmer"] == 0, -1e9)
            if "hands" in masks:
                logits_h = logits_h.masked_fill(masks["hands"] == 0, -1e9)
            if "market" in masks:
                logits_m = logits_m.masked_fill(masks["market"] == 0, -1e9)

        dist_f = torch.distributions.Categorical(logits=logits_f)
        dist_h = torch.distributions.Categorical(logits=logits_h)
        dist_m = torch.distributions.Categorical(logits=logits_m)

        a_f_raw = dist_f.sample((num_samples,))
        a_h_raw = dist_h.sample((num_samples,))
        a_m_raw = dist_m.sample((num_samples,))

        log_prob = (
            dist_f.log_prob(a_f_raw)
            + dist_h.log_prob(a_h_raw)
            + dist_m.log_prob(a_m_raw)
        ).transpose(0, 1)

        a_f = a_f_raw.transpose(0, 1)
        a_h = a_h_raw.transpose(0, 1)
        a_m = a_m_raw.transpose(0, 1)

        joint_actions = torch.stack([a_f, a_h, a_m], dim=-1)

        return joint_actions, log_prob


# Compatibility aliases matching docs/prioritized_muzero_system.py and docs/FACTORIZED_PREDICTION_HEAD.md
FactorizedKaggriculturePredictionHead = SpatialPredictionNetwork


class KaggricultureMuZeroChassis(nn.Module):
    """
    Unified Container holding Representation (h), Dynamics (g), and Prediction (f).
    Provides discrete scalar conversion helpers for value and reward support distributions.
    """

    def __init__(
        self,
        obs_channels: int = 28,
        latent_channels: int = 64,
        action_channels: int = ACTION_CHANNELS,
        num_farmer_actions: int = 15,
        num_hand_assignments: int = 32,
        num_market_orders: int = 20,
        support_size: int = 601,
    ):
        super().__init__()
        self.obs_channels = obs_channels
        self.latent_channels = latent_channels
        self.action_channels = action_channels
        self.support_size = support_size
        self.register_buffer("support", torch.linspace(-300, 300, support_size))

        self.representation = SpatialRepresentationNetwork(obs_channels, latent_channels)
        self.dynamics = SpatialDynamicsNetwork(latent_channels, action_channels, support_size=support_size)
        self.prediction = SpatialPredictionNetwork(
            latent_channels, num_farmer_actions, num_hand_assignments, num_market_orders, support_size
        )

    def initial_inference(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        s_0 = self.representation(obs)
        preds = self.prediction(s_0)
        value_scalar = self._logits_to_scalar(preds["value_logits"])
        return s_0, preds, value_scalar

    def recurrent_inference(
        self,
        s_prev: torch.Tensor,
        action_plane: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        s_next, reward_logits = self.dynamics(s_prev, action_plane)
        preds = self.prediction(s_next)
        reward_scalar = self._logits_to_scalar(reward_logits)
        value_scalar = self._logits_to_scalar(preds["value_logits"])
        return s_next, reward_scalar, preds, value_scalar

    def _logits_to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        support: torch.Tensor = getattr(self, "support")
        return torch.sum(probs * support, dim=-1)


# ============================================================================
# 2. SIMSIAM SELF-SUPERVISED CONSISTENCY MODULE
# ============================================================================

class SimSiamProjectionPredictionHead(nn.Module):
    """Asymmetric Projection and Prediction heads for SimSiam-style latent consistency."""

    def __init__(self, latent_dim: int = 64, proj_dim: int = 64, pred_dim: int = 32):
        super().__init__()
        self.proj_head = nn.Sequential(
            nn.Linear(latent_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
        )
        self.pred_head = nn.Sequential(
            nn.Linear(proj_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, proj_dim),
        )

    def project(self, latent_state: torch.Tensor) -> torch.Tensor:
        if latent_state.dim() > 2:
            latent_state = latent_state.mean(dim=[-2, -1])  # Global average pool if spatial
        return self.proj_head(latent_state)

    def predict(self, projected_state: torch.Tensor) -> torch.Tensor:
        return self.pred_head(projected_state)


class SimSiamConsistencyLoss(nn.Module):
    """Computes Negative Cosine Similarity with explicit Stop-Gradient on target."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        unrolled_latent: torch.Tensor,
        target_observation_embedding: torch.Tensor,
        simsiam_head: SimSiamProjectionPredictionHead,
    ) -> torch.Tensor:
        p_k = simsiam_head.predict(simsiam_head.project(unrolled_latent))

        with torch.no_grad():
            z_k_target = simsiam_head.project(target_observation_embedding).detach()

        p_k_norm = F.normalize(p_k, dim=-1, eps=self.eps)
        z_k_norm = F.normalize(z_k_target, dim=-1, eps=self.eps)

        cosine_sim = (p_k_norm * z_k_norm).sum(dim=-1)
        return -cosine_sim.mean()


def compute_muzero_unroll_loss_with_consistency(
    model_representation: nn.Module,
    model_dynamics: nn.Module,
    model_prediction: nn.Module,
    simsiam_head: SimSiamProjectionPredictionHead,
    simsiam_loss_fn: SimSiamConsistencyLoss,
    observations: torch.Tensor,
    actions: torch.Tensor,
    target_policies: torch.Tensor,
    target_values: torch.Tensor,
    target_rewards: torch.Tensor,
    consistency_weight: float = 0.25,
    unroll_steps: int = 5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Executes a K-step MuZero unroll loop incorporating policy, value, reward,
    and SimSiam self-supervised consistency losses.
    """
    total_policy_loss = torch.tensor(0.0, device=observations.device)
    total_value_loss = torch.tensor(0.0, device=observations.device)
    total_reward_loss = torch.tensor(0.0, device=observations.device)
    total_consistency_loss = torch.tensor(0.0, device=observations.device)

    def _extract_val(pred_dict: Dict[str, torch.Tensor], default_tensor: torch.Tensor) -> torch.Tensor:
        if "value" in pred_dict:
            v = pred_dict["value"]
            return v.squeeze(-1) if (v.dim() > 1 and v.shape[-1] == 1) else v
        if "value_logits" in pred_dict:
            v_logits = pred_dict["value_logits"]
            support = torch.linspace(-300, 300, v_logits.shape[-1], device=v_logits.device)
            return torch.sum(F.softmax(v_logits, dim=-1) * support, dim=-1)
        return default_tensor.new_zeros(default_tensor.shape[0])

    def _extract_reward(r: torch.Tensor) -> torch.Tensor:
        if r.dim() > 1 and r.shape[-1] > 1:
            support = torch.linspace(-300, 300, r.shape[-1], device=r.device)
            return torch.sum(F.softmax(r, dim=-1) * support, dim=-1)
        return r.squeeze(-1) if (r.dim() > 1 and r.shape[-1] == 1) else r

    s_current = model_representation(observations[:, 0])
    pred_0 = model_prediction(s_current)
    if isinstance(pred_0, dict) and "policy_logits" in pred_0:
        total_policy_loss = total_policy_loss + F.cross_entropy(pred_0["policy_logits"], target_policies[:, 0])
    elif isinstance(pred_0, dict) and "logits_farmer" in pred_0:
        total_policy_loss = total_policy_loss + F.cross_entropy(pred_0["logits_farmer"], target_policies[:, 0])
    total_value_loss = total_value_loss + F.mse_loss(_extract_val(pred_0, s_current), target_values[:, 0])

    for k in range(1, unroll_steps + 1):
        a_k = actions[:, k - 1]
        s_unrolled, r_predicted = model_dynamics(s_current, a_k)
        pred_k = model_prediction(s_unrolled)
        if isinstance(pred_k, dict) and "policy_logits" in pred_k:
            total_policy_loss = total_policy_loss + F.cross_entropy(pred_k["policy_logits"], target_policies[:, k])
        elif isinstance(pred_k, dict) and "logits_farmer" in pred_k:
            total_policy_loss = total_policy_loss + F.cross_entropy(pred_k["logits_farmer"], target_policies[:, k])

        total_value_loss = total_value_loss + F.mse_loss(_extract_val(pred_k, s_unrolled), target_values[:, k])
        total_reward_loss = total_reward_loss + F.mse_loss(_extract_reward(r_predicted), target_rewards[:, k - 1])

        target_z_k = model_representation(observations[:, k])
        c_loss_k = simsiam_loss_fn(s_unrolled, target_z_k, simsiam_head)
        total_consistency_loss = total_consistency_loss + c_loss_k
        s_current = s_unrolled

    mean_policy_loss = total_policy_loss / (unroll_steps + 1)
    mean_value_loss = total_value_loss / (unroll_steps + 1)
    mean_reward_loss = total_reward_loss / unroll_steps
    mean_consistency_loss = total_consistency_loss / unroll_steps

    total_loss = (
        mean_policy_loss
        + 0.25 * mean_value_loss
        + mean_reward_loss
        + consistency_weight * mean_consistency_loss
    )

    metrics = {
        "loss_total": total_loss.item(),
        "loss_policy": mean_policy_loss.item(),
        "loss_value": mean_value_loss.item(),
        "loss_reward": mean_reward_loss.item(),
        "loss_consistency": mean_consistency_loss.item(),
    }

    return total_loss, metrics
