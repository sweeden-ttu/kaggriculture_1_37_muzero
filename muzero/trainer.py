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
muzero/trainer.py - MuZero Trainer with PER, EMA Target Network, and Reanalyze
=============================================================================
Trainer classes:
  - PrioritizedMuZeroTrainer: Unified training engine coordinating PER,
    Importance Sampling (IS) weights, Polyak EMA target network updates,
    multi-objective factorized loss, and asynchronous MCTS Reanalysis passes.
  - MuZeroTrainer: Uniform trajectory batching variant with CosineAnnealingLR.
=============================================================================
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .buffer import MuZeroTrajectoryBuffer, PrioritizedMuZeroBuffer
from .chassis import (
    KaggricultureMuZeroChassis,
    SimSiamConsistencyLoss,
    SimSiamProjectionPredictionHead,
)
from .mcts import SampledMuZeroMCTS
from .spatial_constants import joint_action_to_plane, soft_cross_entropy


class PrioritizedMuZeroTrainer:
    """Unified Trainer coordinating PER, IS-weighted loss, EMA Target Model, and Reanalyze."""

    def __init__(
        self,
        online_model: KaggricultureMuZeroChassis,
        simsiam_head: SimSiamProjectionPredictionHead,
        buffer: PrioritizedMuZeroBuffer,
        mcts_engine: SampledMuZeroMCTS,
        lr: float = 1e-3,
        ema_tau: float = 0.995,
        consistency_weight: float = 0.25,
        policy_weight: float = 1.0,
        value_weight: float = 0.25,
        reward_weight: float = 1.0,
        cql_weight: float = 0.0,
        grad_clip: float = 5.0,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.online_model = online_model.to(self.device)

        self.target_model = copy.deepcopy(online_model).to(self.device)
        for param in self.target_model.parameters():
            param.requires_grad = False

        self.simsiam_head = simsiam_head.to(self.device)
        self.simsiam_loss_fn = SimSiamConsistencyLoss().to(self.device)
        self.buffer = buffer
        self.mcts_engine = mcts_engine

        self.ema_tau = ema_tau
        self.consistency_weight = consistency_weight
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.reward_weight = reward_weight
        self.cql_weight = cql_weight
        self.grad_clip = grad_clip

        joint_params = list(self.online_model.parameters()) + list(self.simsiam_head.parameters())
        self.optimizer = AdamW(joint_params, lr=lr, weight_decay=1e-4)

    def update_target_network(self):
        with torch.no_grad():
            for p_online, p_target in zip(self.online_model.parameters(), self.target_model.parameters()):
                p_target.data.copy_(self.ema_tau * p_target.data + (1.0 - self.ema_tau) * p_online.data)

    def _action_tuple_to_plane(self, action_tuples: torch.Tensor) -> torch.Tensor:
        return joint_action_to_plane(action_tuples, device=self.device)

    def train_step(self, batch_size: int = 4, k_steps: int = 3, global_step: int = 0) -> Dict[str, float]:
        self.online_model.train()
        self.simsiam_head.train()

        current_beta = self.buffer.update_beta(global_step)

        # 1. Sample PER Batch with IS Weights
        batch, slice_keys, is_weights = self.buffer.sample_prioritized_k_step_batch(
            batch_size=batch_size, k_steps=k_steps
        )

        obs_batch = batch["obs"].to(self.device)
        actions_batch = batch["actions"].to(self.device)
        rewards_batch = batch["rewards"].to(self.device)
        target_values = batch["values"].to(self.device)
        target_p_f = batch["target_policy_farmer"].to(self.device)
        target_p_h = batch["target_policy_hands"].to(self.device)
        target_p_m = batch["target_policy_market"].to(self.device)
        is_weights = is_weights.to(self.device)

        total_policy_loss = torch.zeros(batch_size, device=self.device)
        total_value_loss = torch.zeros(batch_size, device=self.device)
        total_reward_loss = torch.zeros(batch_size, device=self.device)
        total_consistency_loss = 0.0

        # 2. Root Pass (k = 0)
        s_current, root_preds, root_val_scalar = self.online_model.initial_inference(obs_batch[:, 0])

        total_policy_loss = total_policy_loss + (
            soft_cross_entropy(root_preds["logits_farmer"], target_p_f[:, 0])
            + soft_cross_entropy(root_preds["logits_hands"], target_p_h[:, 0])
            + soft_cross_entropy(root_preds["logits_market"], target_p_m[:, 0])
        )
        total_value_loss = total_value_loss + F.mse_loss(root_val_scalar, target_values[:, 0], reduction="none")
        value_errors = torch.abs(root_val_scalar - target_values[:, 0])

        # 3. Recurrent Unroll Loop (k = 1 ... K)
        for k in range(1, k_steps + 1):
            action_plane = self._action_tuple_to_plane(actions_batch[:, k - 1])
            s_unrolled, r_predicted, preds_k, v_predicted = self.online_model.recurrent_inference(
                s_current, action_plane
            )

            total_policy_loss = total_policy_loss + (
                soft_cross_entropy(preds_k["logits_farmer"], target_p_f[:, k])
                + soft_cross_entropy(preds_k["logits_hands"], target_p_h[:, k])
                + soft_cross_entropy(preds_k["logits_market"], target_p_m[:, k])
            )
            total_value_loss = total_value_loss + F.mse_loss(v_predicted, target_values[:, k], reduction="none")
            total_reward_loss = total_reward_loss + F.mse_loss(r_predicted, rewards_batch[:, k - 1], reduction="none")

            step_val_error = torch.abs(v_predicted - target_values[:, k])
            value_errors = torch.max(value_errors, step_val_error)

            with torch.no_grad():
                target_z_k = self.target_model.representation(obs_batch[:, k])

            c_loss_k = self.simsiam_loss_fn(s_unrolled, target_z_k, self.simsiam_head)
            total_consistency_loss = total_consistency_loss + c_loss_k
            s_current = s_unrolled

        mean_p_loss = total_policy_loss / (k_steps + 1)
        mean_v_loss = total_value_loss / (k_steps + 1)
        mean_r_loss = total_reward_loss / k_steps
        mean_c_loss = total_consistency_loss / k_steps

        # Composite Per-Sample Loss (consistency is batch-mean scalar)
        per_sample_loss = (
            self.policy_weight * mean_p_loss
            + self.value_weight * mean_v_loss
            + self.reward_weight * mean_r_loss
        )
        weighted_loss = (is_weights * per_sample_loss.unsqueeze(1)).mean()
        weighted_loss = weighted_loss + self.consistency_weight * mean_c_loss

        # 4. Backward & Step
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), self.grad_clip)
        self.optimizer.step()

        self.update_target_network()

        # 5. Update Priorities in Replay Buffer
        fresh_td_errors = value_errors.detach().cpu().numpy()
        self.buffer.update_slice_priorities(slice_keys, fresh_td_errors)

        return {
            "loss_total": weighted_loss.item(),
            "loss_policy": mean_p_loss.mean().item(),
            "loss_value": mean_v_loss.mean().item(),
            "loss_reward": mean_r_loss.mean().item(),
            "loss_consistency": float(mean_c_loss.item() if torch.is_tensor(mean_c_loss) else mean_c_loss),
            "beta": current_beta,
            "mean_td_error": float(np.mean(fresh_td_errors)),
        }

    def run_reanalyze_pass(self, num_trajectories: int = 1):
        if not self.buffer.trajectories:
            return
        sampled_indices = random.sample(
            range(len(self.buffer.trajectories)),
            min(num_trajectories, len(self.buffer.trajectories)),
        )
        for idx in sampled_indices:
            self.buffer.reanalyze_trajectory(idx, self.mcts_engine, self.device)


class MuZeroTrainer:
    """Complete Trainer coordinating gradient updates, target network EMA, and Reanalysis."""

    def __init__(
        self,
        online_model: KaggricultureMuZeroChassis,
        simsiam_head: SimSiamProjectionPredictionHead,
        buffer: MuZeroTrajectoryBuffer,
        mcts_engine: SampledMuZeroMCTS,
        lr: float = 3e-4,
        ema_tau: float = 0.995,
        consistency_weight: float = 0.25,
        policy_weight: float = 1.0,
        value_weight: float = 0.25,
        reward_weight: float = 1.0,
        cql_weight: float = 0.0,
        grad_clip: float = 5.0,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.online_model = online_model.to(self.device)

        self.target_model = copy.deepcopy(online_model).to(self.device)
        for param in self.target_model.parameters():
            param.requires_grad = False

        self.simsiam_head = simsiam_head.to(self.device)
        self.simsiam_loss_fn = SimSiamConsistencyLoss().to(self.device)
        self.buffer = buffer
        self.mcts_engine = mcts_engine

        self.ema_tau = ema_tau
        self.consistency_weight = consistency_weight
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.reward_weight = reward_weight
        self.cql_weight = cql_weight
        self.grad_clip = grad_clip

        joint_params = list(self.online_model.parameters()) + list(self.simsiam_head.parameters())
        self.optimizer = AdamW(joint_params, lr=lr, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=1e-5)

    def update_target_network(self):
        with torch.no_grad():
            for p_online, p_target in zip(self.online_model.parameters(), self.target_model.parameters()):
                p_target.data.copy_(self.ema_tau * p_target.data + (1.0 - self.ema_tau) * p_online.data)

    def _action_tuple_to_plane(self, action_tuples: torch.Tensor) -> torch.Tensor:
        return joint_action_to_plane(action_tuples, device=self.device)

    def train_step(self, batch_size: int = 4, k_steps: int = 3) -> Dict[str, float]:
        self.online_model.train()
        self.simsiam_head.train()

        batch = self.buffer.sample_k_step_batch(batch_size=batch_size, k_steps=k_steps)

        obs_batch = batch["obs"].to(self.device)
        actions_batch = batch["actions"].to(self.device)
        rewards_batch = batch["rewards"].to(self.device)
        target_values = batch["values"].to(self.device)
        target_p_f = batch["target_policy_farmer"].to(self.device)
        target_p_h = batch["target_policy_hands"].to(self.device)
        target_p_m = batch["target_policy_market"].to(self.device)

        total_policy_loss = torch.tensor(0.0, device=self.device)
        total_value_loss = torch.tensor(0.0, device=self.device)
        total_reward_loss = torch.tensor(0.0, device=self.device)
        total_consistency_loss = torch.tensor(0.0, device=self.device)

        s_current, root_preds, root_val_scalar = self.online_model.initial_inference(obs_batch[:, 0])

        total_policy_loss = total_policy_loss + (
            soft_cross_entropy(root_preds["logits_farmer"], target_p_f[:, 0]).mean()
            + soft_cross_entropy(root_preds["logits_hands"], target_p_h[:, 0]).mean()
            + soft_cross_entropy(root_preds["logits_market"], target_p_m[:, 0]).mean()
        )
        total_value_loss = total_value_loss + F.mse_loss(root_val_scalar, target_values[:, 0])

        for k in range(1, k_steps + 1):
            action_plane = self._action_tuple_to_plane(actions_batch[:, k - 1])
            s_unrolled, r_predicted, preds_k, v_predicted = self.online_model.recurrent_inference(
                s_current, action_plane
            )

            total_policy_loss = total_policy_loss + (
                soft_cross_entropy(preds_k["logits_farmer"], target_p_f[:, k]).mean()
                + soft_cross_entropy(preds_k["logits_hands"], target_p_h[:, k]).mean()
                + soft_cross_entropy(preds_k["logits_market"], target_p_m[:, k]).mean()
            )
            total_value_loss = total_value_loss + F.mse_loss(v_predicted, target_values[:, k])
            total_reward_loss = total_reward_loss + F.mse_loss(r_predicted, rewards_batch[:, k - 1])

            with torch.no_grad():
                target_z_k = self.target_model.representation(obs_batch[:, k])

            c_loss_k = self.simsiam_loss_fn(s_unrolled, target_z_k, self.simsiam_head)
            total_consistency_loss = total_consistency_loss + c_loss_k
            s_current = s_unrolled

        mean_p_loss = total_policy_loss / (k_steps + 1)
        mean_v_loss = total_value_loss / (k_steps + 1)
        mean_r_loss = total_reward_loss / k_steps
        mean_c_loss = total_consistency_loss / k_steps

        total_loss = (
            self.policy_weight * mean_p_loss
            + self.value_weight * mean_v_loss
            + self.reward_weight * mean_r_loss
            + self.consistency_weight * mean_c_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()

        self.update_target_network()

        return {
            "loss_total": total_loss.item(),
            "loss_policy": mean_p_loss.item(),
            "loss_value": mean_v_loss.item(),
            "loss_reward": mean_r_loss.item(),
            "loss_consistency": mean_c_loss.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def run_reanalyze_pass(self, num_trajectories: int = 1):
        if not self.buffer.trajectories:
            return
        sampled_indices = random.sample(
            range(len(self.buffer.trajectories)),
            min(num_trajectories, len(self.buffer.trajectories)),
        )
        for idx in sampled_indices:
            self.buffer.reanalyze_trajectory(idx, self.mcts_engine, self.device)
