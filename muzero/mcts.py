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
muzero/mcts.py - Pure-Latent Sampled Monte Carlo Tree Search
=============================================================================
HARD INVARIANT — search is 100% latent space:

  1. Observation → h_θ (representation) **exactly once** at the root.
  2. Every expansion uses g_θ (dynamics) + f_θ (prediction) on stored latents.
  3. No environment / simulator / analytical economy steps inside the tree.
  4. No re-encoding of observations mid-search.
  5. Stored node states are detached latent tensors only.

Connects KaggricultureMuZeroChassis to Sampled MCTS:
  - MinMaxStats: Global Q-value tracking and PUCT score normalization.
  - MCTSNode: Search tree node holding visit counts, values, and latent state.
  - SampledMuZeroMCTS: Factorized candidate sampling, Dirichlet noise,
    latent recurrent unrolling, PUCT selection, discounted backpropagation.
=============================================================================
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .chassis import KaggricultureMuZeroChassis
from .spatial_constants import joint_action_to_plane

# Public contract flag — callers / tests may assert this.
PURE_LATENT_MCTS: bool = True


class MinMaxStats:
    """Tracks global minimum and maximum Q-values across the search tree."""

    def __init__(
        self,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        if min_value is not None and minimum is None:
            minimum = min_value
        if max_value is not None and maximum is None:
            maximum = max_value
        self.minimum = float(minimum) if minimum is not None else float("inf")
        self.maximum = float(maximum) if maximum is not None else -float("inf")

    def update(self, value: float) -> None:
        if math.isnan(value):
            return
        v = float(value)
        if v < self.minimum:
            self.minimum = v
        if v > self.maximum:
            self.maximum = v

    def normalize(self, value: float, eps: float = 0.0) -> float:
        if self.maximum > self.minimum and math.isfinite(self.minimum) and math.isfinite(self.maximum):
            denom = max(self.maximum - self.minimum, eps) if eps > 0 else (self.maximum - self.minimum)
            return (value - self.minimum) / denom
        return value

    def clear(self) -> None:
        self.minimum = float("inf")
        self.maximum = -float("inf")

    def is_valid(self) -> bool:
        return self.maximum > self.minimum and math.isfinite(self.minimum) and math.isfinite(self.maximum)


class MCTSNode:
    """Node in the pure-latent Monte Carlo search tree (stores s only, never obs)."""

    def __init__(self, prior: float = 0.0, action_tuple: Optional[Tuple[int, int, int]] = None):
        self.prior = prior
        self.action_tuple = action_tuple
        self.visit_count = 0
        self.value_sum = 0.0
        self.reward = 0.0
        self.latent_state: Optional[torch.Tensor] = None  # detached s ∈ latent space
        self.children: Dict[Tuple[int, int, int], MCTSNode] = {}

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count


def _detach_latent(s: torch.Tensor) -> torch.Tensor:
    """Store only detached latents in the tree (no autograd / no obs tensors)."""
    if not isinstance(s, torch.Tensor):
        raise TypeError("latent state must be a torch.Tensor")
    # Latents are [B, C, H, W] (spatial) or [B, D]; never raw obs (obs_channels=28).
    return s.detach()


class SampledMuZeroMCTS:
    """Pure-latent Sampled MuZero MCTS (h once at root; g/f only thereafter).

    Parameters ``pure_latent`` is always True — analytical / env rollouts are refused.
    """

    pure_latent: bool = True

    def __init__(
        self,
        chassis: KaggricultureMuZeroChassis,
        c1: float = 1.25,
        c2: float = 19652.0,
        discount: float = 0.997,
        num_samples: int = 8,
        num_simulations: int = 25,
        dirichlet_alpha: float = 0.25,
        exploration_fraction: float = 0.25,
        *,
        pure_latent: bool = True,
    ):
        if not pure_latent:
            raise ValueError(
                "SampledMuZeroMCTS requires pure_latent=True "
                "(no environment or analytical dynamics inside the tree)"
            )
        self.chassis = chassis
        self.c1 = c1
        self.c2 = c2
        self.discount = discount
        self.num_samples = num_samples
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction
        # Diagnostics last search (for tests / telemetry)
        self.last_search_stats: Dict[str, int] = {
            "representation_calls": 0,
            "dynamics_calls": 0,
            "prediction_sample_calls": 0,
        }

    def _puct_score(self, parent: MCTSNode, child: MCTSNode, min_max: MinMaxStats) -> float:
        pb_c = math.log((parent.visit_count + self.c2 + 1.0) / self.c2) + self.c1
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1.0)

        prior_score = pb_c * child.prior
        value_score = 0.0
        if child.visit_count > 0:
            value_score = min_max.normalize(child.reward + self.discount * child.value)

        return prior_score + value_score

    def _select_child(self, node: MCTSNode, min_max: MinMaxStats) -> Tuple[Tuple[int, int, int], MCTSNode]:
        best_score = -1e9
        best_action: Optional[Tuple[int, int, int]] = None
        best_child: Optional[MCTSNode] = None

        for action_tuple, child in node.children.items():
            score = self._puct_score(node, child, min_max)
            if score > best_score:
                best_score = score
                best_action = action_tuple
                best_child = child

        assert best_action is not None and best_child is not None
        return best_action, best_child

    def _action_tuple_to_plane(
        self,
        action_tuple: Tuple[int, int, int],
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Encode discrete joint action as latent-space action planes (not env cmds)."""
        return joint_action_to_plane(action_tuple, batch_size=batch_size, device=device)

    def _expand_from_latent(
        self,
        latent: torch.Tensor,
        *,
        num_samples: int,
        masks: Optional[Dict[str, torch.Tensor]],
        add_dirichlet_noise: bool,
    ) -> Tuple[List[Tuple[int, int, int]], np.ndarray]:
        """Sample factorized joint actions from f_θ(latent) only."""
        joint_actions, joint_log_probs = self.chassis.prediction.sample_joint_actions(
            latent, num_samples=num_samples, masks=masks
        )
        self.last_search_stats["prediction_sample_calls"] += 1
        priors = F.softmax(joint_log_probs, dim=-1).squeeze(0).detach().cpu().numpy()

        if add_dirichlet_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * num_samples)
            priors = (1.0 - self.exploration_fraction) * priors + self.exploration_fraction * noise

        actions: List[Tuple[int, int, int]] = []
        for k in range(num_samples):
            actions.append(
                (
                    int(joint_actions[0, k, 0].item()),
                    int(joint_actions[0, k, 1].item()),
                    int(joint_actions[0, k, 2].item()),
                )
            )
        return actions, priors

    @torch.no_grad()
    def search_from_latent(
        self,
        latent: torch.Tensor,
        *,
        root_value: Optional[float] = None,
        add_dirichlet_noise: bool = True,
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Tuple[int, int, int], Dict[Tuple[int, int, int], float], float]:
        """Run MCTS starting from an existing latent (no representation call)."""
        self.chassis.eval()
        self.last_search_stats = {
            "representation_calls": 0,
            "dynamics_calls": 0,
            "prediction_sample_calls": 0,
        }
        device = latent.device
        min_max = MinMaxStats()

        s_0 = _detach_latent(latent)
        if root_value is None:
            preds = self.chassis.prediction(s_0)
            root_value = float(self.chassis._logits_to_scalar(preds["value_logits"]).item())
            self.last_search_stats["prediction_sample_calls"] += 0  # value eval only

        root = MCTSNode()
        root.latent_state = s_0
        root.visit_count = 1
        root.value_sum = float(root_value)
        min_max.update(root.value_sum)

        actions, priors = self._expand_from_latent(
            s_0, num_samples=self.num_samples, masks=masks, add_dirichlet_noise=add_dirichlet_noise
        )
        for k, a_tuple in enumerate(actions):
            root.children[a_tuple] = MCTSNode(prior=float(priors[k]), action_tuple=a_tuple)

        for _ in range(self.num_simulations):
            node = root
            search_path = [root]

            while node.is_expanded:
                _, node = self._select_child(node, min_max)
                search_path.append(node)

            parent_node = search_path[-2]
            action_tuple = node.action_tuple
            assert action_tuple is not None
            assert parent_node.latent_state is not None
            # Pure latent transition: g_θ(s, a_plane) — never env.step / analytical teacher
            action_plane = self._action_tuple_to_plane(action_tuple, batch_size=1, device=device)
            s_next, r_scalar, _preds, v_scalar = self.chassis.recurrent_inference(
                parent_node.latent_state, action_plane
            )
            self.last_search_stats["dynamics_calls"] += 1

            node.latent_state = _detach_latent(s_next)
            node.reward = float(r_scalar.item())
            leaf_value = float(v_scalar.item())

            child_actions, child_priors = self._expand_from_latent(
                node.latent_state,
                num_samples=self.num_samples,
                masks=masks,
                add_dirichlet_noise=False,
            )
            for k, c_tuple in enumerate(child_actions):
                node.children[c_tuple] = MCTSNode(prior=float(child_priors[k]), action_tuple=c_tuple)

            value = leaf_value
            for path_node in reversed(search_path):
                path_node.visit_count += 1
                path_node.value_sum += value
                value = path_node.reward + self.discount * value
                min_max.update(value)

        total_visits = sum(child.visit_count for child in root.children.values())
        action_probabilities = {
            a_tuple: child.visit_count / max(1, total_visits)
            for a_tuple, child in root.children.items()
        }
        best_action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        return best_action, action_probabilities, root.value

    @torch.no_grad()
    def search(
        self,
        obs_tensor: torch.Tensor,
        add_dirichlet_noise: bool = True,
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Tuple[int, int, int], Dict[Tuple[int, int, int], float], float]:
        """Encode observation with h_θ once, then pure-latent search via g_θ/f_θ."""
        self.chassis.eval()
        # Single representation call at the root — the only place obs enters search.
        s_0, _root_preds, root_value_scalar = self.chassis.initial_inference(obs_tensor)
        best, probs, val = self.search_from_latent(
            s_0,
            root_value=float(root_value_scalar.item()),
            add_dirichlet_noise=add_dirichlet_noise,
            masks=masks,
        )
        self.last_search_stats["representation_calls"] = 1
        return best, probs, val
