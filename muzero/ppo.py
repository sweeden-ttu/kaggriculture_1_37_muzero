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
==============================================================================
ppo.py - PPO Microcontroller Macrocontroller for Kaggriculture
==============================================================================
A Proximal Policy Optimization (PPO) actor-critic controller over the same
8-option macro-action space used by the MuZero MCTS planner. The PPO policy
is the "microcontroller": it issues one macro-option per in-game day, mapping
the 32-dim encoded observation directly to a masked action distribution —
no tree search required at inference time.

Roles in the refactored stack:
  * Alternative / complementary controller to LatentMacroOptionMCTS inside
    the Phase-4 continuous self-play loop (``--controller ppo|hybrid``).
  * Offline hyperparameter tuning target: the offline grid search sweeps
    PPO dials (lr, gamma, clip_epsilon, entropy_coef, gae_lambda) alongside
    the MuZero loss-weight dials.
  * Trained with the Adam optimizer (Adaptive Moment Estimation engine);
    rewards are wealth deltas scaled by REWARD_SCALE, identical to the
    MuZero reward convention.

Rollouts use the analytical economy (``economy.meta_economic_step``) as a
fast O(1) environment, mirroring the Phase-4 self-play harvest. The module
never imports the Kaggle harness — it is pure torch + analytical states.
==============================================================================
"""

from __future__ import annotations

__author__ = "Scott Weeden"
__license__ = "Apache-2.0"

import math
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import encode_macro_state, legal_macro_options, wealth
from .types import (
    DEFAULT_PRICES,
    NUM_MACRO_OPTIONS,
    OBS_DIM,
    REWARD_SCALE,
    MacroOption,
    MacroState,
)

DEFAULT_PPO_CHECKPOINT = "ppo_checkpoints.pt"
DEFAULT_PPO_ADAPTERS_DIR = "artifacts/ppo"


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------


@dataclass
class PPOHyperParams:
    """PPO microcontroller macrocontroller hyperparameters (fixed dials, not gradients)."""

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 64
    hidden_dim: int = 128
    num_actions: int = NUM_MACRO_OPTIONS
    obs_dim: int = OBS_DIM

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Actor-critic network
# ---------------------------------------------------------------------------


class ActorCritic(nn.Module):
    """PPO microcontroller: shared trunk → policy head (8 macro-options) + value head."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden_dim: int = 128, num_actions: int = NUM_MACRO_OPTIONS):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.apply(_init_weights)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        h = self.trunk(obs)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def act(
        self,
        obs: torch.Tensor,
        legal_options: Optional[Sequence[MacroOption]] = None,
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sample (or argmax) a macro-option, optionally masked to legal options."""
        logits, value = self.forward(obs)
        if legal_options is not None and len(legal_options) > 0:
            mask = torch.full((self.num_actions,), float("-inf"), device=logits.device)
            mask[[int(o) for o in legal_options]] = 0.0
            logits = logits + mask
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            action = int(torch.argmax(probs, dim=-1).item())
        else:
            action = int(torch.multinomial(probs, num_samples=1).item())
        log_prob = torch.log(probs.squeeze(0)[action] + 1e-10)
        return action, log_prob, value.squeeze(0)

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Log-probs of taken actions, state values, and policy entropy for a batch."""
        logits, values = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return action_log_probs, values, entropy


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# Rollout buffer with GAE
# ---------------------------------------------------------------------------


class PPOBuffer:
    """On-policy rollout storage with Generalized Advantage Estimation."""

    def __init__(self) -> None:
        self.obs: List[torch.Tensor] = []
        self.actions: List[int] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.values: List[torch.Tensor] = []
        self.dones: List[float] = []
        self.advantages: Optional[torch.Tensor] = None
        self.returns: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.obs)

    def add(self, obs, action, log_prob, reward, value, done) -> None:
        self.obs.append(obs)
        self.actions.append(int(action))
        self.log_probs.append(log_prob)
        self.rewards.append(float(reward))
        self.values.append(value)
        self.dones.append(float(done))

    def compute_gae(self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        """GAE-λ advantages and discounted returns (Schulman et al., 2016)."""
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.stack([v.detach() for v in self.values])
        dones = torch.tensor(self.dones, dtype=torch.float32)

        advantages = torch.zeros_like(rewards)
        gae = 0.0
        next_value = float(last_value)
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * nonterminal - values[t]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[t] = gae
            next_value = values[t]
        self.advantages = advantages
        self.returns = advantages + values

    def get(self) -> Dict[str, torch.Tensor]:
        assert self.advantages is not None and self.returns is not None
        return {
            "obs": torch.stack([o.detach() for o in self.obs]),
            "actions": torch.tensor(self.actions, dtype=torch.long),
            "log_probs": torch.stack([lp.detach() for lp in self.log_probs]),
            "advantages": self.advantages,
            "returns": self.returns,
        }


# ---------------------------------------------------------------------------
# PPO training (Adam / Adaptive Moment Estimation engine)
# ---------------------------------------------------------------------------


def train_ppo(
    model: ActorCritic,
    buffer: PPOBuffer,
    hp: PPOHyperParams,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    """Clipped-surrogate PPO update over the collected rollout buffer."""
    if len(buffer) == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)

    batch = buffer.get()
    adv = batch["advantages"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    n = len(buffer)
    idx = torch.randperm(n)
    policy_losses: List[float] = []
    value_losses: List[float] = []
    entropies: List[float] = []

    model.train()
    for _ in range(hp.ppo_epochs):
        for start in range(0, n, hp.batch_size):
            mb = idx[start : start + hp.batch_size]
            mb_obs = batch["obs"][mb]
            mb_actions = batch["actions"][mb]
            mb_old_log_probs = batch["log_probs"][mb]
            mb_adv = adv[mb]
            mb_returns = batch["returns"][mb]

            new_log_probs, values, entropy = model.evaluate(mb_obs, mb_actions)
            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - hp.clip_epsilon, 1.0 + hp.clip_epsilon) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, mb_returns)
            entropy_mean = entropy.mean()

            loss = policy_loss + hp.value_coef * value_loss - hp.entropy_coef * entropy_mean

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp.max_grad_norm)
            optimizer.step()

            policy_losses.append(float(policy_loss.detach()))
            value_losses.append(float(value_loss.detach()))
            entropies.append(float(entropy_mean.detach()))

    return {
        "policy_loss": sum(policy_losses) / max(1, len(policy_losses)),
        "value_loss": sum(value_losses) / max(1, len(value_losses)),
        "entropy": sum(entropies) / max(1, len(entropies)),
        "total_loss": sum(policy_losses) / max(1, len(policy_losses))
        + hp.value_coef * (sum(value_losses) / max(1, len(value_losses))),
    }


# ---------------------------------------------------------------------------
# Rollouts against the analytical economy
# ---------------------------------------------------------------------------


def initial_macro_state(day: int = 0, capital: float = 1000.0, unlocked: Sequence[str] = ("NW",)) -> MacroState:
    """Sensible opening MacroState for PPO self-play rollouts."""
    return MacroState(
        day=day,
        capital=capital,
        unlocked=tuple(unlocked),
        shed={"WHEAT": 0.0, "MELON": 0.0, "STRAWBERRY": 0.0, "MILK": 0.0, "WOOL": 0.0},
        prices=dict(DEFAULT_PRICES),
        active_animals=1,
    )


def ppo_rollout_episode(
    model: ActorCritic,
    hp: PPOHyperParams,
    initial_state: Optional[MacroState] = None,
    max_days: int = 30,
    deterministic: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """One self-play episode: PPO picks a macro-option per day; economy steps the state."""
    from .economy import meta_economic_step

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    state = initial_state if initial_state is not None else initial_macro_state()
    buffer = PPOBuffer()
    total_reward = 0.0

    for _ in range(max_days):
        obs = encode_macro_state(state)
        legal = legal_macro_options(state)
        action, log_prob, value = model.act(obs, legal_options=legal, deterministic=deterministic)
        option = MacroOption(action)
        next_state = meta_economic_step(state, option)
        reward = (wealth(next_state) - wealth(state)) * REWARD_SCALE
        done = 1.0 if next_state.day >= max_days else 0.0
        buffer.add(obs, action, log_prob, reward, value, done)
        total_reward += reward
        state = next_state
        if done:
            break

    with torch.no_grad():
        last_obs = encode_macro_state(state)
        _, last_value = model.forward(last_obs)
    buffer.compute_gae(float(last_value.item()), gamma=hp.gamma, gae_lambda=hp.gae_lambda)
    return {"buffer": buffer, "total_reward": total_reward, "final_state": state}


def ppo_rollout_batch(
    model: ActorCritic,
    hp: PPOHyperParams,
    n_episodes: int = 8,
    max_days: int = 30,
    base_seed: int = 10101,
) -> Tuple[PPOBuffer, float]:
    """Aggregate several episodes into one buffer; returns (buffer, mean episode reward).

    Each episode's GAE advantages/returns are computed at episode end and
    concatenated into the merged buffer (advantages are episode-local).
    """
    merged = PPOBuffer()
    rewards: List[float] = []
    adv_parts: List[torch.Tensor] = []
    ret_parts: List[torch.Tensor] = []
    for ep in range(n_episodes):
        res = ppo_rollout_episode(model, hp, max_days=max_days, seed=base_seed + ep)
        ep_buf = res["buffer"]
        for i in range(len(ep_buf)):
            merged.add(
                ep_buf.obs[i],
                ep_buf.actions[i],
                ep_buf.log_probs[i],
                ep_buf.rewards[i],
                ep_buf.values[i],
                ep_buf.dones[i],
            )
        if ep_buf.advantages is not None:
            adv_parts.append(ep_buf.advantages)
        if ep_buf.returns is not None:
            ret_parts.append(ep_buf.returns)
        rewards.append(res["total_reward"])
    if adv_parts:
        merged.advantages = torch.cat(adv_parts)
    if ret_parts:
        merged.returns = torch.cat(ret_parts)
    return merged, sum(rewards) / max(1, len(rewards))


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def save_ppo_checkpoint(
    model: ActorCritic,
    path: str,
    meta: Optional[Dict[str, Any]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    atomic: bool = True,
) -> str:
    payload = {
        "state_dict": model.state_dict(),
        "obs_dim": model.obs_dim,
        "num_actions": model.num_actions,
        "hidden_dim": model.trunk[0].out_features,
        "meta": meta or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if atomic:
        tmp = f"{path}.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)
    else:
        torch.save(payload, path)
    return path


def load_ppo_checkpoint(
    model: Optional[ActorCritic] = None,
    path: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[ActorCritic, Optional[str]]:
    path = path or DEFAULT_PPO_CHECKPOINT
    payload = torch.load(path, map_location=device)
    state_dict = payload.get("state_dict", payload)
    obs_dim = payload.get("obs_dim", OBS_DIM)
    num_actions = payload.get("num_actions", NUM_MACRO_OPTIONS)
    hidden_dim = payload.get("hidden_dim", 128)
    if model is None:
        model = ActorCritic(obs_dim=obs_dim, hidden_dim=hidden_dim, num_actions=num_actions)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, payload.get("meta")


def resolve_ppo_checkpoint(explicit: Optional[str] = None) -> Optional[str]:
    candidates = [explicit, DEFAULT_PPO_CHECKPOINT, os.path.join("artifacts", "ppo", DEFAULT_PPO_CHECKPOINT)]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None
