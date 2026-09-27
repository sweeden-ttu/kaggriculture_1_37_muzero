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
muzero/buffer.py - Trajectory Replay Buffer with TD-Error Prioritization
=============================================================================
Episode-based trajectory buffer supporting:
  - GameTrajectory: Full-episode trajectory storage.
  - MuZeroTrajectoryBuffer: K-step unroll batching, n-step bootstrapped value
    targets, factorized policy targets, and MCTS Reanalyze target refreshing.
  - PrioritizedMuZeroBuffer: TD-error Prioritized Experience Replay (PER),
    Importance Sampling (IS) weights, and dynamic slice priority updates.
=============================================================================
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


class GameTrajectory:
    """Stores a single complete game episode trajectory."""

    def __init__(self):
        self.observations: List[np.ndarray] = []
        self.actions: List[Tuple[int, int, int]] = []
        self.rewards: List[float] = []
        self.policies: List[Dict[Tuple[int, int, int], float]] = []
        self.values: List[float] = []
        self.priorities: List[float] = []

    def __len__(self) -> int:
        return len(self.actions)

    def append_step(
        self,
        obs: np.ndarray,
        action: Tuple[int, int, int],
        reward: float,
        policy_dict: Dict[Tuple[int, int, int], float],
        value: float,
        priority: float = 1.0,
    ):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.policies.append(policy_dict)
        self.values.append(value)
        self.priorities.append(priority)

    def finalize(self, final_obs: np.ndarray):
        """Appends the terminal observation at step T (T actions, T+1 observations)."""
        self.observations.append(final_obs)


class MuZeroTrajectoryBuffer:
    """
    Trajectory-Based Replay Buffer supporting K-step unroll batching,
    n-step value bootstrapping, and Reanalyze target refreshing.
    """

    def __init__(
        self,
        max_trajectories: int = 1000,
        discount: float = 0.997,
        n_steps: int = 5,
        k_steps: int = 5,
        num_farmer_actions: int = 15,
        num_hand_assignments: int = 32,
        num_market_orders: int = 20,
    ):
        self.max_trajectories = max_trajectories
        self.discount = discount
        self.n_steps = n_steps
        self.k_steps = k_steps
        self.num_farmer = num_farmer_actions
        self.num_hands = num_hand_assignments
        self.num_market = num_market_orders
        self.trajectories: List[GameTrajectory] = []

    def save_trajectory(self, trajectory: GameTrajectory):
        """Adds a completed episode trajectory to the buffer with FIFO eviction."""
        if len(self.trajectories) >= self.max_trajectories:
            self.trajectories.pop(0)
        self.trajectories.append(trajectory)

    def _compute_n_step_target(self, trajectory: GameTrajectory, index: int) -> float:
        """Computes n-step bootstrapped value target z_t = sum(gamma^j * r_{t+j}) + gamma^n * v_{t+n}."""
        target_val = 0.0
        trajectory_len = len(trajectory)

        for j in range(self.n_steps):
            if index + j < trajectory_len:
                target_val += (self.discount ** j) * trajectory.rewards[index + j]
            else:
                break

        bootstrap_idx = index + self.n_steps
        if bootstrap_idx < trajectory_len:
            target_val += (self.discount ** self.n_steps) * trajectory.values[bootstrap_idx]

        return target_val

    def _convert_policy_dict_to_factorized_targets(
        self,
        policy_dict: Dict[Tuple[int, int, int], float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Converts joint tuple visit probabilities into marginal factorized target distributions."""
        target_f = np.zeros(self.num_farmer, dtype=np.float32)
        target_h = np.zeros(self.num_hands, dtype=np.float32)
        target_m = np.zeros(self.num_market, dtype=np.float32)

        for (f_act, h_act, m_act), prob in policy_dict.items():
            target_f[f_act % self.num_farmer] += prob
            target_h[h_act % self.num_hands] += prob
            target_m[m_act % self.num_market] += prob

        target_f /= max(1e-6, float(target_f.sum()))
        target_h /= max(1e-6, float(target_h.sum()))
        target_m /= max(1e-6, float(target_m.sum()))

        return target_f, target_h, target_m

    def sample_k_step_batch(
        self,
        batch_size: int = 16,
        k_steps: int = 5,
    ) -> Dict[str, torch.Tensor]:
        """Samples a uniform batch of K-step unroll trajectory slices."""
        valid_trajectories = [t for t in self.trajectories if len(t) >= k_steps]
        if not valid_trajectories:
            raise ValueError(f"Buffer contains no trajectories with length >= {k_steps}")

        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_values = []
        batch_policy_f = []
        batch_policy_h = []
        batch_policy_m = []

        for _ in range(batch_size):
            traj = random.choice(valid_trajectories)
            max_start = max(0, len(traj) - k_steps - 1) if len(traj) > k_steps else 0
            start_idx = random.randint(0, max_start)

            slice_obs = []
            slice_actions = []
            slice_rewards = []
            slice_values = []
            slice_p_f, slice_p_h, slice_p_m = [], [], []

            for k in range(k_steps + 1):
                curr_idx = start_idx + k
                slice_obs.append(traj.observations[min(curr_idx, len(traj.observations) - 1)])

                val_target = self._compute_n_step_target(traj, curr_idx)
                slice_values.append(val_target)

                p_dict = traj.policies[curr_idx] if curr_idx < len(traj.policies) else traj.policies[-1]
                pf, ph, pm = self._convert_policy_dict_to_factorized_targets(p_dict)
                slice_p_f.append(pf)
                slice_p_h.append(ph)
                slice_p_m.append(pm)

                if k < k_steps:
                    slice_actions.append(traj.actions[curr_idx])
                    slice_rewards.append(traj.rewards[curr_idx])

            batch_obs.append(np.array(slice_obs))
            batch_actions.append(np.array(slice_actions))
            batch_rewards.append(np.array(slice_rewards))
            batch_values.append(np.array(slice_values))
            batch_policy_f.append(np.array(slice_p_f))
            batch_policy_h.append(np.array(slice_p_h))
            batch_policy_m.append(np.array(slice_p_m))

        return {
            "obs": torch.from_numpy(np.array(batch_obs)).float(),
            "actions": torch.from_numpy(np.array(batch_actions)).long(),
            "rewards": torch.from_numpy(np.array(batch_rewards)).float(),
            "values": torch.from_numpy(np.array(batch_values)).float(),
            "target_policy_farmer": torch.from_numpy(np.array(batch_policy_f)).float(),
            "target_policy_hands": torch.from_numpy(np.array(batch_policy_h)).float(),
            "target_policy_market": torch.from_numpy(np.array(batch_policy_m)).float(),
        }

    def reanalyze_trajectory(self, traj_idx: int, mcts_engine: Any, device: torch.device):
        """Re-runs MCTS search on an existing stored trajectory to refresh stale targets."""
        traj = self.trajectories[traj_idx]
        for t in range(len(traj)):
            obs_tensor = torch.from_numpy(traj.observations[t]).unsqueeze(0).to(device)
            best_act, fresh_policy_dict, fresh_val = mcts_engine.search(obs_tensor, add_dirichlet_noise=False)
            traj.policies[t] = fresh_policy_dict
            traj.values[t] = fresh_val


class PrioritizedMuZeroBuffer(MuZeroTrajectoryBuffer):
    """Trajectory-Based Replay Buffer supporting TD-Error PER & IS Weights."""

    def __init__(
        self,
        max_trajectories: int = 1000,
        discount: float = 0.997,
        n_steps: int = 5,
        k_steps: int = 5,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        eps: float = 1e-5,
        num_farmer_actions: int = 15,
        num_hand_assignments: int = 32,
        num_market_orders: int = 20,
    ):
        super().__init__(
            max_trajectories=max_trajectories,
            discount=discount,
            n_steps=n_steps,
            k_steps=k_steps,
            num_farmer_actions=num_farmer_actions,
            num_hand_assignments=num_hand_assignments,
            num_market_orders=num_market_orders,
        )
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps
        self.frame_count = 0

    def update_beta(self, current_frame: int) -> float:
        self.frame_count = current_frame
        fraction = min(1.0, current_frame / float(self.beta_frames))
        self.beta = self.beta_start + fraction * (1.0 - self.beta_start)
        return self.beta

    def sample_prioritized_k_step_batch(
        self,
        batch_size: int = 8,
        k_steps: int = 5,
    ) -> Tuple[Dict[str, torch.Tensor], List[Tuple[int, int]], torch.Tensor]:
        valid_slices: List[Tuple[int, int, float]] = []

        for t_idx, traj in enumerate(self.trajectories):
            if len(traj) >= k_steps:
                for s_idx in range(len(traj) - k_steps):
                    prio = traj.priorities[s_idx] if s_idx < len(traj.priorities) else 1.0
                    valid_slices.append((t_idx, s_idx, prio))

        if not valid_slices:
            raise ValueError(f"No valid trajectory slices of length >= {k_steps}")

        N = len(valid_slices)
        priorities = np.array([p for _, _, p in valid_slices], dtype=np.float32)

        probs = np.power(priorities + self.eps, self.alpha)
        probs /= probs.sum()

        sampled_indices = np.random.choice(N, size=batch_size, p=probs, replace=True)
        sampled_slices = [valid_slices[idx] for idx in sampled_indices]

        sampled_probs = probs[sampled_indices]
        weights = np.power(N * sampled_probs, -self.beta)
        weights /= weights.max()
        is_weights_tensor = torch.from_numpy(weights).float().unsqueeze(1)

        batch_obs, batch_actions, batch_rewards, batch_values = [], [], [], []
        batch_p_f, batch_p_h, batch_p_m = [], [], []
        batch_slice_keys: List[Tuple[int, int]] = []

        for t_idx, s_idx, _ in sampled_slices:
            traj = self.trajectories[t_idx]
            batch_slice_keys.append((t_idx, s_idx))

            slice_obs, slice_actions, slice_rewards, slice_values = [], [], [], []
            slice_pf, slice_ph, slice_pm = [], [], []

            for k in range(k_steps + 1):
                curr_idx = s_idx + k
                slice_obs.append(traj.observations[curr_idx])

                val_target = self._compute_n_step_target(traj, curr_idx)
                slice_values.append(val_target)

                pf, ph, pm = self._convert_policy_dict_to_factorized_targets(traj.policies[curr_idx])
                slice_pf.append(pf)
                slice_ph.append(ph)
                slice_pm.append(pm)

                if k < k_steps:
                    slice_actions.append(traj.actions[curr_idx])
                    slice_rewards.append(traj.rewards[curr_idx])

            batch_obs.append(np.array(slice_obs))
            batch_actions.append(np.array(slice_actions))
            batch_rewards.append(np.array(slice_rewards))
            batch_values.append(np.array(slice_values))
            batch_p_f.append(np.array(slice_pf))
            batch_p_h.append(np.array(slice_ph))
            batch_p_m.append(np.array(slice_pm))

        batch_dict = {
            "obs": torch.from_numpy(np.array(batch_obs)).float(),
            "actions": torch.from_numpy(np.array(batch_actions)).long(),
            "rewards": torch.from_numpy(np.array(batch_rewards)).float(),
            "values": torch.from_numpy(np.array(batch_values)).float(),
            "target_policy_farmer": torch.from_numpy(np.array(batch_p_f)).float(),
            "target_policy_hands": torch.from_numpy(np.array(batch_p_h)).float(),
            "target_policy_market": torch.from_numpy(np.array(batch_p_m)).float(),
        }

        return batch_dict, batch_slice_keys, is_weights_tensor

    def update_slice_priorities(self, slice_keys: List[Tuple[int, int]], td_errors: np.ndarray):
        for (t_idx, s_idx), error in zip(slice_keys, td_errors):
            if t_idx < len(self.trajectories):
                traj = self.trajectories[t_idx]
                if s_idx < len(traj.priorities):
                    traj.priorities[s_idx] = float(error) + self.eps
