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

"""Legacy MLP / 8-MacroOption MuZero (packaging & ablation only).

Production training uses spatial Sampled MuZero
(``muzero.chassis.KaggricultureMuZeroChassis``). Prefer ``--arch macro`` only
when deliberately exercising this stack.
"""
from __future__ import annotations

__author__ = "Scott Weeden"
__license__ = "Apache-2.0"


import math
import os
from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import (
    decode_macro_state,
    encode_macro_state,
    encode_observation,
    legal_macro_options,
    wealth,
)
from .types import (
    BASE_CROP_PRICES,
    DEFAULT_PRICES,
    HIDDEN_DIM,
    LAND_BUFFERS,
    LAND_ORDER,
    LAND_PRICES,
    MACRO_OPTION_NAMES,
    NUM_MACRO_OPTIONS,
    OBS_DIM,
    REWARD_SCALE,
    SUPPORT_SIZE,
    DiscreteSupport,
    MacroAction,
    MacroOption,
    MacroState,
    MinMaxStats,
    MuZeroMCTSNode,
    SearchResult,
)

# IntEnum still needed for any remaining local helpers; MacroOption comes from types.


# Encoding / legality / wealth live in encoding.py (re-exported below for callers).
# Analytical teacher (meta_economic_step, generate_analytical_*) lives in economy.py
# and must only be imported by phase1_bootstrap.


def scale_gradient(tensor: torch.Tensor, scale: float = 0.5) -> torch.Tensor:
    """Scales gradient during backpropagation while keeping forward values unchanged."""
    return tensor * scale + tensor.detach() * (1.0 - scale)


# ---------------------------------------------------------------------------
# Neural networks: h_θ, g_θ, f_θ
# ---------------------------------------------------------------------------

class RepresentationNetwork(nn.Module):
    """h_θ: observation features → latent hidden state s0."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.net(obs)


class DynamicsNetwork(nn.Module):
    """
    g_θ: (s_k, a_{k+1}) → (r_{k+1}, s_{k+1}).
    Action is one-hot concatenated with latent state.
    Reward head outputs discrete support logits or scalar.
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        num_actions: int = NUM_MACRO_OPTIONS,
        support_size: int = SUPPORT_SIZE,
        support: Optional[DiscreteSupport] = None,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.support_size = support_size
        self.support = support or (DiscreteSupport() if support_size > 1 else None)
        self.joint = nn.Sequential(
            nn.Linear(hidden_dim + num_actions, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.reward_head = nn.Linear(128, support_size)
        self.next_state_head = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.Tanh(),
        )

    def forward_logits(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (reward_logits, next_latent)."""
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        if action.dim() == 0:
            action = action.unsqueeze(0)
        a_oh = F.one_hot(action.long(), num_classes=self.num_actions).float()
        if a_oh.dim() == 1:
            a_oh = a_oh.unsqueeze(0)
        h = self.joint(torch.cat([latent, a_oh], dim=-1))
        reward_logits = self.reward_head(h)
        next_latent = self.next_state_head(h)
        return reward_logits, next_latent

    def forward(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (reward_scalar, next_latent)."""
        reward_logits, next_latent = self.forward_logits(latent, action)
        if self.support_size > 1 and self.support is not None:
            reward_scalar = self.support.support_to_scalar(reward_logits)
        else:
            reward_scalar = reward_logits.squeeze(-1)
        return reward_scalar, next_latent


class PredictionNetwork(nn.Module):
    """
    f_θ: latent s → (policy logits π, scalar value v).
    Policy head mirrors COMAActor; value head supports discrete distributions.
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        num_actions: int = NUM_MACRO_OPTIONS,
        support_size: int = SUPPORT_SIZE,
        support: Optional[DiscreteSupport] = None,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.support_size = support_size
        self.support = support or (DiscreteSupport() if support_size > 1 else None)
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )
        self.value_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, support_size),
        )

    def forward_logits(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (policy_logits, value_logits)."""
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        feat = self.shared(latent)
        policy_logits = self.policy_head(feat)
        value_logits = self.value_head(feat)
        return policy_logits, value_logits

    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (policy_logits, value_scalar)."""
        policy_logits, value_logits = self.forward_logits(latent)
        if self.support_size > 1 and self.support is not None:
            value_scalar = self.support.support_to_scalar(value_logits)
        else:
            value_scalar = value_logits.squeeze(-1)
        return policy_logits, value_scalar


class MuZeroNetwork(nn.Module):
    """Combined MuZero model: representation + dynamics + prediction."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_actions: int = NUM_MACRO_OPTIONS,
        support_size: int = SUPPORT_SIZE,
        support: Optional[DiscreteSupport] = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.support_size = support_size
        self.support = support or (DiscreteSupport() if support_size > 1 else None)
        self.representation = RepresentationNetwork(obs_dim, hidden_dim)
        self.dynamics = DynamicsNetwork(
            hidden_dim, num_actions, support_size=support_size, support=self.support
        )
        self.prediction = PredictionNetwork(
            hidden_dim, num_actions, support_size=support_size, support=self.support
        )

    def initial_inference(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """h_θ(o) then f_θ(s0) → (latent, policy_probs, value_scalar)."""
        latent = self.representation(obs)
        logits, value = self.prediction(latent)
        policy = F.softmax(logits, dim=-1)
        return latent, policy, value

    def recurrent_inference(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """g_θ(s,a) then f_θ(s') → (reward_scalar, next_latent, policy_probs, value_scalar)."""
        reward, next_latent = self.dynamics(latent, action)
        logits, value = self.prediction(next_latent)
        policy = F.softmax(logits, dim=-1)
        return reward, next_latent, policy, value

    def initial_inference_logits(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (latent, policy_logits, value_logits)."""
        latent = self.representation(obs)
        policy_logits, value_logits = self.prediction.forward_logits(latent)
        return latent, policy_logits, value_logits

    def recurrent_inference_logits(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (reward_logits, next_latent, policy_logits, value_logits)."""
        reward_logits, next_latent = self.dynamics.forward_logits(latent, action)
        policy_logits, value_logits = self.prediction.forward_logits(next_latent)
        return reward_logits, next_latent, policy_logits, value_logits

    @torch.no_grad()
    def predict_policy_value(self, obs: torch.Tensor) -> Tuple[torch.Tensor, float]:
        self.eval()
        _, policy, value = self.initial_inference(obs)
        val = float(value.squeeze(0).item()) if isinstance(value, torch.Tensor) else float(value)
        return policy.squeeze(0), val

    @torch.no_grad()
    def predict_target_value_and_support(
        self,
        obs_or_latent: torch.Tensor,
        is_latent: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predicts scalar value and categorical support distribution (if enabled)
        using this network (e.g. for target model θ^- in Priority 1).
        Returns (scalar_value, categorical_distribution).
        """
        self.eval()
        if not is_latent:
            latent = self.representation(obs_or_latent)
        else:
            latent = obs_or_latent

        _, v_logits = self.prediction.forward_logits(latent)
        if self.support_size > 1 and self.support is not None:
            scalar = self.support.support_to_scalar(v_logits, is_logits=True)
            support_dist = F.softmax(v_logits, dim=-1)
            return scalar, support_dist
        else:
            scalar = v_logits.squeeze(-1)
            return scalar, None

    @torch.no_grad()
    def predict_value(self, obs: torch.Tensor) -> float:
        self.eval()
        latent = self.representation(obs)
        _, value = self.prediction(latent)
        return float(value.squeeze(0).item()) if isinstance(value, torch.Tensor) else float(value)



# ---------------------------------------------------------------------------
# Latent MCTS with P-UCT
# (SearchResult / MinMaxStats / MuZeroMCTSNode live in types.py — single source of truth)
# ---------------------------------------------------------------------------


def _gumbel_noise(n: int, deterministic: bool = True) -> torch.Tensor:
    """Standard Gumbel(0,1) samples or zero noise for deterministic evaluation."""
    if deterministic:
        return torch.zeros(n)
    u = torch.rand(n).clamp(1e-8, 1.0 - 1e-8)
    return -torch.log(-torch.log(u))



class LatentMacroOptionMCTS:
    """
    Monte Carlo Tree Search strictly inside MuZero latent space over MacroOption.

    Canonical invariants (Towards Kaggriculture MuZero §3):
      1. Transitions use learned g_θ exclusively — never the analytical economy.
      2. Leaf values / priors come from f_θ only.
      3. Domain knowledge is limited to a root action mask from the observation;
         no MacroState is stored or stepped inside the tree.

    When n_simulations ≤ gumbel_threshold (default 32), root uses Gumbel-Top-k
    sequential selection (Danihelka et al.) for timeout-safe few-sim search.
    """

    def __init__(
        self,
        model: MuZeroNetwork,
        c_puct: float = 1.5,
        discount: float = 0.99,
        max_depth: int = 15,
        use_analytical_dynamics: bool = False,
        value_norm: float = 100_000.0,
        gumbel_threshold: int = 32,
        time_budget_ms: Optional[float] = None,
        target_model: Optional[MuZeroNetwork] = None,
    ):
        if use_analytical_dynamics:
            import warnings
            warnings.warn(
                "LatentMacroOptionMCTS ignores use_analytical_dynamics; "
                "search is latent g_θ only.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.model = model
        self.c_puct = c_puct
        self.discount = discount
        self.max_depth = max_depth
        self.use_analytical_dynamics = False  # hard-locked: latent only
        self.value_norm = value_norm
        self.gumbel_threshold = gumbel_threshold
        self.time_budget_ms = time_budget_ms
        self.target_model = target_model

    @torch.no_grad()
    def search(
        self,
        root_state: MacroState,
        n_simulations: int = 50,
        temperature: float = 0.0,
        add_dirichlet_noise: bool = False,
        dirichlet_alpha: float = 0.25,
        dirichlet_fraction: float = 0.25,
    ) -> SearchResult:
        import time as _time

        self.model.eval()
        t0 = _time.perf_counter()
        # Observation → h_θ only. Root legality mask from obs; tree stays latent.
        obs = encode_macro_state(root_state)
        root_latent, root_policy, root_value = self.model.initial_inference(obs)
        root_latent = root_latent.squeeze(0)
        root_policy = root_policy.squeeze(0)
        root_legal = legal_macro_options(root_state)

        root = MuZeroMCTSNode(
            latent=root_latent,
            prior=1.0,
            macro_state=None,
        )
        self._expand(root, root_policy, legal=root_legal)

        # Inject Dirichlet noise at root node for active exploration in self-play (Mode A)
        if add_dirichlet_noise and len(root.children) > 0:
            import numpy as _np
            actions = list(root.children.keys())
            noise = _np.random.dirichlet([dirichlet_alpha] * len(actions))
            for a, n_val in zip(actions, noise):
                root.children[a].prior = (
                    (1.0 - dirichlet_fraction) * root.children[a].prior
                    + dirichlet_fraction * float(n_val)
                )

        min_max_stats = MinMaxStats()

        use_gumbel = n_simulations <= self.gumbel_threshold and len(root.children) > 0
        gumbel_scores: Optional[Dict[MacroOption, float]] = None
        if use_gumbel:
            actions = list(root.children.keys())
            logits = torch.tensor(
                [math.log(max(1e-8, root.children[a].prior)) for a in actions],
                dtype=torch.float32,
            )
            _det = (not add_dirichlet_noise) and temperature <= 1e-4
            noise = _gumbel_noise(len(actions), deterministic=_det)
            scores = (logits + noise).tolist()
            gumbel_scores = {a: float(s) for a, s in zip(actions, scores)}

        for sim_i in range(n_simulations):
            if self.time_budget_ms is not None:
                elapsed_ms = (_time.perf_counter() - t0) * 1000.0
                if elapsed_ms >= self.time_budget_ms and sim_i > 0:
                    break

            node = root
            path: List[MuZeroMCTSNode] = [node]
            depth = 0

            # 1. Selection (latent tree only — no day/wealth env terminals)
            while node.is_expanded() and depth < self.max_depth:
                if node is root and use_gumbel and gumbel_scores is not None:
                    unvisited = [a for a, c in node.children.items() if c.visit_count == 0]
                    if unvisited:
                        best_a = max(unvisited, key=lambda a: gumbel_scores[a])
                        node = node.children[best_a]
                    else:
                        use_stats = min_max_stats.is_valid()
                        q_min: float = 0.0
                        q_range: float = 1.0
                        if not use_stats:
                            q_vals = [c.q_value for c in node.children.values() if c.visit_count > 0]
                            q_min = min(q_vals) if q_vals else 0.0
                            q_max = max(q_vals) if q_vals else 1.0
                            q_range = max(1e-4, q_max - q_min)
                        best_a = None
                        best_s = -float("inf")
                        for a, child in node.children.items():
                            if use_stats:
                                q_norm = min_max_stats.normalize(child.q_value)
                            else:
                                q_norm = (child.q_value - q_min) / q_range
                            s = gumbel_scores[a] + 2.0 * q_norm - 0.05 * child.visit_count
                            if s > best_s:
                                best_s = s
                                best_a = a
                        assert best_a is not None
                        node = node.children[best_a]
                else:
                    _, node = node.select_child(self.c_puct, min_max_stats=min_max_stats)
                path.append(node)
                depth += 1

            leaf_value = self._evaluate_and_expand(node)
            self._backup(path, leaf_value, min_max_stats=min_max_stats)

        visit_counts = {a: c.visit_count for a, c in root.children.items()}
        policy_target = torch.zeros(NUM_MACRO_OPTIONS, dtype=torch.float32)
        total = sum(visit_counts.values()) or 1
        for a, n in visit_counts.items():
            policy_target[int(a)] = float(n) / float(total)

        if temperature > 1e-4 and visit_counts:
            import numpy as _np
            actions = list(visit_counts.keys())
            counts = _np.array([visit_counts[a] for a in actions], dtype=_np.float64)
            if counts.sum() > 0:
                counts_exp = counts ** (1.0 / max(1e-4, temperature))
                c_sum = counts_exp.sum()
                if c_sum > 0 and _np.isfinite(c_sum):
                    probs = counts_exp / c_sum
                    chosen_idx = _np.random.choice(len(actions), p=probs)
                    best = actions[chosen_idx]
                else:
                    best = max(visit_counts.items(), key=lambda kv: kv[1])[0]
            else:
                best = max(visit_counts.items(), key=lambda kv: kv[1])[0]
        elif use_gumbel and gumbel_scores is not None and visit_counts:
            best = max(
                visit_counts.keys(),
                key=lambda a: gumbel_scores.get(a, 0.0) + root.children[a].q_value,
            )
        else:
            best = max(visit_counts.items(), key=lambda kv: kv[1])[0]

        return SearchResult(
            best_option=best,
            visit_counts=visit_counts,
            root_value=root.q_value if root.visit_count else float(root_value.item()),
            policy_target=policy_target,
        )

    def _expand(
        self,
        node: MuZeroMCTSNode,
        policy: torch.Tensor,
        legal: Optional[Sequence[MacroOption]] = None,
    ) -> None:
        """Expand with network priors. Root may pass an observation legality mask."""
        if legal is None:
            legal = list(MacroOption)
        masked = torch.zeros_like(policy)
        for opt in legal:
            masked[int(opt)] = policy[int(opt)]
        m_sum = float(masked.detach().sum().item())
        if m_sum <= 0:
            for opt in legal:
                masked[int(opt)] = 1.0 / max(1, len(list(legal)))
        else:
            masked = masked / m_sum

        for opt in legal:
            child_latent, child_reward = self._transition(node, opt)
            node.children[opt] = MuZeroMCTSNode(
                latent=child_latent,
                prior=float(masked[int(opt)].item()),
                parent=node,
                action=opt,
                macro_state=None,
                reward=child_reward,
            )

    def _transition(
        self,
        node: MuZeroMCTSNode,
        option: MacroOption,
    ) -> Tuple[torch.Tensor, float]:
        """Latent dynamics only: (r, s') = g_θ(s, a)."""
        action_t = torch.tensor(int(option), dtype=torch.long)
        reward_t, next_latent, _, _ = self.model.recurrent_inference(node.latent, action_t)
        return next_latent.squeeze(0), float(reward_t.squeeze(0).item())

    def _evaluate_and_expand(self, node: MuZeroMCTSNode) -> float:
        """Leaf value from f_θ(s) only — no analytical wealth blend."""
        eval_model = self.target_model if self.target_model is not None else self.model
        if not node.is_expanded():
            logits, _ = self.model.prediction(node.latent)
            policy = F.softmax(logits.squeeze(0), dim=-1)
            self._expand(node, policy)  # interior: full macro set via g_θ
            _, value = eval_model.prediction(node.latent)
            return float(value.squeeze(0).item())

        _, value = eval_model.prediction(node.latent)
        return float(value.squeeze(0).item())

    def _backup(
        self,
        path: List[MuZeroMCTSNode],
        leaf_value: float,
        min_max_stats: Optional[MinMaxStats] = None,
    ) -> None:
        """Backup discounted return: G = r_{k+1} + γ r_{k+2} + ... + γ^n v."""
        g = leaf_value
        for node in reversed(path):
            g = node.reward + self.discount * g
            node.visit_count += 1
            node.value_sum += g
            if min_max_stats is not None:
                min_max_stats.update(node.q_value)


_META_HANDS_BY_DAY = ((3, 4), (7, 6), (14, 8), (22, 9), (30, 10))


def _default_hands(day: int, unlocked_count: int) -> int:
    base = 4
    for max_day, hands in _META_HANDS_BY_DAY:
        if day <= max_day:
            base = hands
            break
    if unlocked_count >= 3:
        base = min(10, base + 1)
    elif unlocked_count == 1 and day <= 8:
        base = min(base, 6)
    return base


# ---------------------------------------------------------------------------
# Planner facade used by expansion policy / agent factories
# ---------------------------------------------------------------------------

class MuZeroPlanner:
    """
    High-level MuZero planner for expansion gating and daily option selection.
    Strict latent dynamics (g_θ only) — matches BASELINES latent-only MCTS.
    Supports centralized MinMaxStats and EMA target networks θ^-.
    """

    def __init__(
        self,
        model: Optional[MuZeroNetwork] = None,
        target_model: Optional[MuZeroNetwork] = None,
        n_simulations: int = 40,
        c_puct: float = 1.5,
        use_analytical_dynamics: bool = False,
        device: str = "cpu",
        gumbel_threshold: int = 32,
        time_budget_ms: Optional[float] = None,
    ):
        self.model = model or MuZeroNetwork()
        self.model.to(device)
        self.model.eval()
        self.target_model = target_model
        if self.target_model is not None:
            self.target_model.to(device)
            self.target_model.eval()
        self.n_simulations = n_simulations
        self.device = device
        self.mcts = LatentMacroOptionMCTS(
            model=self.model,
            target_model=self.target_model,
            c_puct=c_puct,
            use_analytical_dynamics=use_analytical_dynamics,
            gumbel_threshold=gumbel_threshold,
            time_budget_ms=time_budget_ms,
        )


    def load_weights(self, path: str) -> None:
        model, _ = load_muzero_checkpoint(self.model, path=path, device=self.device)
        self.model = model
        self.mcts.model = model
        self.model.eval()

    def search(
        self,
        money: float,
        day: int,
        unlocked: Sequence[str],
        prices: Optional[Dict[str, float]] = None,
        shed: Optional[Dict[str, float]] = None,
        n_simulations: Optional[int] = None,
        temperature: float = 0.0,
        add_dirichlet_noise: bool = False,
        dirichlet_alpha: float = 0.25,
        dirichlet_fraction: float = 0.25,
    ) -> SearchResult:
        root = MacroState(
            day=day,
            capital=money,
            unlocked=tuple(unlocked),
            shed=dict(shed or {}),
            prices=dict(prices or DEFAULT_PRICES),
        )
        return self.mcts.search(
            root,
            n_simulations=n_simulations or self.n_simulations,
            temperature=temperature,
            add_dirichlet_noise=add_dirichlet_noise,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_fraction=dirichlet_fraction,
        )

    def evaluate_expansion(
        self,
        money: float,
        day: int,
        unlocked: Sequence[str],
        prices: Optional[Dict[str, float]] = None,
        shed: Optional[Dict[str, float]] = None,
        n_simulations: Optional[int] = None,
        temperature: float = 0.0,
        add_dirichlet_noise: bool = False,
        dirichlet_alpha: float = 0.25,
        dirichlet_fraction: float = 0.25,
    ) -> Optional[str]:
        """
        Returns the land quadrant MuZero prefers to unlock today, or None.
        Only fires when the best option is an EXPAND_* and capital clears buffer.
        """
        if day > 24:
            return None
        next_target = None
        for q in LAND_ORDER:
            if q not in unlocked:
                next_target = q
                break
        if next_target is None:
            return None

        # Affordability uses engine land list prices only (no invented SE meta buffer).
        cost = LAND_PRICES[next_target] + LAND_BUFFERS[next_target]
        if money < cost:
            return None

        result = self.search(
            money=money,
            day=day,
            unlocked=unlocked,
            prices=prices,
            shed=shed,
            n_simulations=n_simulations,
            temperature=temperature,
            add_dirichlet_noise=add_dirichlet_noise,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_fraction=dirichlet_fraction,
        )
        expand_map = {
            MacroOption.EXPAND_NE: "NE",
            MacroOption.EXPAND_SW: "SW",
            MacroOption.EXPAND_SE: "SE",
            MacroOption.EXPANSION_PREPARATION: next_target if next_target != "SE" else None,
        }
        chosen = expand_map.get(result.best_option)
        if chosen != next_target:
            # Also accept if expand visits dominate pass/survival
            expand_opt = {
                "NE": MacroOption.EXPAND_NE,
                "SW": MacroOption.EXPAND_SW,
                "SE": MacroOption.EXPAND_SE,
            }[next_target]
            expand_visits = result.visit_counts.get(expand_opt, 0)
            prep_visits = result.visit_counts.get(MacroOption.EXPANSION_PREPARATION, 0)
            hold_visits = (
                result.visit_counts.get(MacroOption.PASS, 0)
                + result.visit_counts.get(MacroOption.FRONT_LOAD_SURVIVAL, 0)
            )
            if next_target == "SE":
                # Require a clearer expand majority before unlocking SE
                if expand_visits <= hold_visits + prep_visits:
                    return None
            elif expand_visits + prep_visits > hold_visits:
                return next_target
            return None
        return chosen

    def recommend_option(
        self,
        money: float,
        day: int,
        unlocked: Sequence[str],
        **kwargs: Any,
    ) -> MacroOption:
        return self.search(money=money, day=day, unlocked=unlocked, **kwargs).best_option

    def recommend_hire_target(
        self,
        money: float,
        day: int,
        unlocked: Sequence[str],
        base_target: int,
        **kwargs: Any,
    ) -> int:
        """Bump hire target when MuZero prefers HIRE_HANDS (meta 9–10 crew)."""
        opt = self.recommend_option(money=money, day=day, unlocked=unlocked, **kwargs)
        meta_floor = _default_hands(day, len(unlocked))
        target = max(base_target, meta_floor)
        if opt == MacroOption.HIRE_HANDS:
            target = min(10, max(target, meta_floor + 1))
        return target

    def prefer_trickle_sales(
        self,
        money: float,
        day: int,
        unlocked: Sequence[str],
        **kwargs: Any,
    ) -> bool:
        """True when MuZero selects TRICKLE_LIQUIDATION (metered premium lots)."""
        return self.recommend_option(money=money, day=day, unlocked=unlocked, **kwargs) == (
            MacroOption.TRICKLE_LIQUIDATION
        )


# ---------------------------------------------------------------------------
# Priority 1: EMA Target Network Utilities (Polyak averaging & n-step returns)
# ---------------------------------------------------------------------------

def create_target_network(model: MuZeroNetwork) -> MuZeroNetwork:
    """Create a frozen copy of the network to serve as target model θ^- (Priority 1)."""
    import copy
    target = copy.deepcopy(model)
    target.eval()
    for p in target.parameters():
        p.requires_grad = False
    return target


def update_target_network(
    model: MuZeroNetwork,
    target_model: MuZeroNetwork,
    tau: float = 0.005,
) -> None:
    """
    Polyak target network update (Priority 1):
    θ^- ← (1 - tau) * θ^- + tau * θ (default 0.995 / 0.005).

    Uses named-parameter matching so LoRA-adapted online nets stay aligned with
    identically structured targets (skips any stray unmatched keys).
    """
    with torch.no_grad():
        online = dict(model.named_parameters())
        for name, p_tgt in target_model.named_parameters():
            p = online.get(name)
            if p is None or p.shape != p_tgt.shape:
                continue
            p_tgt.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


def compute_target_values(
    target_model: MuZeroNetwork,
    obs_seq: torch.Tensor,
    rewards: torch.Tensor,
    discount: float = 0.99,
) -> torch.Tensor:
    """
    Computes value targets using EMA target network θ^- for bootstrapping (Priority 1, Figure S2).
    For a trajectory slice of length K (obs_seq: (B, K+1, OBS_DIM), rewards: (B, K)):
    Bootstraps v(s_K) from target_model, then discounts backwards to produce z_0..z_K.
    """
    with torch.no_grad():
        target_model.eval()
        B = obs_seq.shape[0]
        K = rewards.shape[1]
        terminal_obs = obs_seq[:, -1]
        v_boot, _ = target_model.predict_target_value_and_support(terminal_obs)
        if v_boot.dim() == 0:
            v_boot = v_boot.unsqueeze(0)

        targets = torch.zeros(B, K + 1, device=obs_seq.device, dtype=torch.float32)
        targets[:, -1] = v_boot

        for k in range(K - 1, -1, -1):
            targets[:, k] = rewards[:, k] + discount * targets[:, k + 1]

        return targets


# ---------------------------------------------------------------------------
# Priority 2: Data-Driven Curriculum through True Prioritized Experience Replay
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """
    Data-driven Prioritized Experience Replay buffer (PER, Priority 2, Appendix G).

    Replaces hard-coded frequency heuristics with priority sampling based on TD-error |v_i - z_i|.
    - Sampling probability: P(i) ∝ (|v_i - z_i| + ε)^α
    - Importance sampling weights: w_i = (1/N * 1/P(i))^β (normalized by max w)
    Default hyperparameters: α = 1.0, β = 1.0, ε = 1e-5.
    """

    def __init__(
        self,
        capacity: int = 50000,
        alpha: float = 1.0,
        beta: float = 1.0,
        eps: float = 1e-5,
    ):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)
        self.buffer: List[Any] = []
        self.priorities: List[float] = []
        self.pos: int = 0
        self.max_priority: float = 1.0

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, item: Any, priority: Optional[float] = None) -> None:
        """Add an item to the buffer with initial priority (default: max_priority)."""
        p = float(priority) if priority is not None and priority > 0 else self.max_priority
        p = max(p, self.eps)

        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
            self.priorities.append(p)
        else:
            self.buffer[self.pos] = item
            self.priorities[self.pos] = p

        self.pos = (self.pos + 1) % self.capacity
        if p > self.max_priority:
            self.max_priority = p

    def add_batch(self, items: Sequence[Any], priorities: Optional[Sequence[float]] = None) -> None:
        if priorities is None:
            for item in items:
                self.add(item)
        else:
            for item, p in zip(items, priorities):
                self.add(item, priority=p)

    def sample(self, batch_size: int) -> Tuple[List[Any], List[int], torch.Tensor]:
        """
        Sample a batch of items with probability P(i) ∝ p_i^α.
        Returns (batch_items, indices, normalized_is_weights).
        """
        import numpy as np

        N = len(self.buffer)
        if N == 0:
            raise ValueError("Cannot sample from an empty PrioritizedReplayBuffer.")

        k = min(batch_size, N)
        p_arr = np.array(self.priorities, dtype=np.float64)
        if self.alpha != 1.0:
            p_arr = p_arr ** self.alpha

        p_sum = p_arr.sum()
        if p_sum <= 0 or not np.isfinite(p_sum):
            probs = np.full(N, 1.0 / N, dtype=np.float64)
        else:
            probs = p_arr / p_sum

        indices = np.random.choice(N, size=k, replace=(k > N), p=probs).tolist()
        batch_items = [self.buffer[idx] for idx in indices]

        p_sampled = np.maximum(probs[indices], 1e-12)
        weights = (1.0 / (N * p_sampled)) ** self.beta
        max_w = np.max(weights)
        if max_w > 0:
            weights = weights / max_w
        weights_t = torch.tensor(weights, dtype=torch.float32)

        return batch_items, indices, weights_t

    def update_priorities(self, indices: Sequence[int], td_errors: Sequence[float] | torch.Tensor) -> None:
        """Update priorities: p_i = (|TD_error| + ε)^α."""
        if isinstance(td_errors, torch.Tensor):
            errors = td_errors.detach().cpu().abs().numpy().flatten().tolist()
        else:
            errors = [abs(float(e)) for e in td_errors]

        for idx, err in zip(indices, errors):
            if 0 <= idx < len(self.priorities):
                p = (float(err) + self.eps) ** self.alpha
                self.priorities[idx] = p
                if p > self.max_priority:
                    self.max_priority = p


# ---------------------------------------------------------------------------
# Priority 3: Full Trajectory K-Step Reanalyze
# ---------------------------------------------------------------------------

def reanalyze_trajectory_slice(
    planner: MuZeroPlanner,
    obs_slice: torch.Tensor,
    policy_slice: torch.Tensor,
    value_slice: torch.Tensor,
    target_model: Optional[MuZeroNetwork] = None,
    sims: int = 16,
    reanalyze_steps: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Full Trajectory K-Step Reanalyze (Priority 3).
    Re-executes MCTS at selected points k ∈ [0, K] (including interior points k > 0)
    using the latest network parameters, utilizing target model θ^- for bootstrapping.
    Returns (fresh_policies, fresh_values).
    """
    import random

    seq_len = obs_slice.shape[0]
    pol_out = policy_slice.clone()
    val_out = value_slice.clone()

    if reanalyze_steps is None:
        # Reanalyze root k=0 plus random interior point k > 0
        interior = [random.randint(1, seq_len - 1)] if seq_len > 1 else []
        reanalyze_steps = [0] + interior

    for k in reanalyze_steps:
        if k < 0 or k >= seq_len:
            continue
        state = decode_macro_state(obs_slice[k])
        result = planner.mcts.search(state, n_simulations=sims)
        pol_out[k] = result.policy_target

        if target_model is not None:
            with torch.no_grad():
                v_boot, _ = target_model.predict_target_value_and_support(obs_slice[k].unsqueeze(0))
                val_out[k] = 0.5 * result.root_value + 0.5 * float(v_boot.item())
        else:
            val_out[k] = result.root_value

    return pol_out, val_out


# ---------------------------------------------------------------------------
# K-Step Unroll Loss with Gradient Scaling & PER Weights (Appendix G)
# ---------------------------------------------------------------------------

def muzero_k_step_unroll_loss(
    model: MuZeroNetwork,
    obs_seq: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    value_targets: torch.Tensor,
    policy_targets: Optional[torch.Tensor] = None,
    k_steps: int = 5,
    dynamics_grad_scale: float = 0.5,
    c_l2: float = 1e-4,
    cons_w: float = 0.5,
    weights: Optional[torch.Tensor] = None,
    target_model: Optional[MuZeroNetwork] = None,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    reward_weight: float = 1.0,
    cql_weight: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    K-step unrolled MuZero loss with gradient scaling, discrete support distributions,
    configurable loss scaling factors, CQL regularization, and Importance Sampling weights.
    """
    effective_k = min(k_steps, actions.shape[1])
    root_obs = obs_seq[:, 0] if obs_seq.dim() == 3 else obs_seq
    scale_head = 1.0 / float(max(1, effective_k))

    w = None
    if weights is not None:
        w = weights.to(actions.device).float()
        if w.dim() > 1:
            w = w.squeeze()

    # 1. Root inference
    s, p0_logits, v0_logits = model.initial_inference_logits(root_obs)
    reward_logits_list = []
    latent_list = [s]
    policy_logits_list = [p0_logits]
    value_logits_list = [v0_logits]

    # 2. Recurrent unrolling over K steps with gradient scaling (1/2 at dynamics input)
    curr_s = s
    for k in range(effective_k):
        curr_s_scaled = scale_gradient(curr_s, scale=dynamics_grad_scale)
        r_logits, next_s, p_logits, v_logits = model.recurrent_inference_logits(
            curr_s_scaled, actions[:, k]
        )
        reward_logits_list.append(r_logits)
        latent_list.append(next_s)
        policy_logits_list.append(p_logits)
        value_logits_list.append(v_logits)
        curr_s = next_s

    # 3. Categorical Support or MSE Losses
    support = getattr(model, "support", None)
    use_support = model.support_size > 1 and support is not None

    # (a) Reward loss: scaled by 1/K
    loss_r_sum = torch.tensor(0.0, device=actions.device)
    for k in range(effective_k):
        r_pred = reward_logits_list[k]
        r_tgt = rewards[:, k]
        if support is not None:
            tgt_dist = support.scalar_to_support(r_tgt)
            loss_r_k_raw = F.cross_entropy(r_pred, tgt_dist, reduction="none")
        else:
            loss_r_k_raw = F.mse_loss(r_pred.squeeze(-1), r_tgt, reduction="none")
        if w is not None:
            loss_r_k = (loss_r_k_raw * w).mean()
        else:
            loss_r_k = loss_r_k_raw.mean()
        loss_r_sum = loss_r_sum + loss_r_k
    loss_reward = loss_r_sum * scale_head

    # (b) Value loss: scaled by 1/K + TD-error extraction for PER
    loss_v_sum = torch.tensor(0.0, device=actions.device)
    td_errors_list = []
    for k in range(effective_k + 1):
        v_pred = value_logits_list[k]
        v_tgt = value_targets[:, k]
        if support is not None:
            tgt_dist = support.scalar_to_support(v_tgt)
            loss_v_k_raw = F.cross_entropy(v_pred, tgt_dist, reduction="none")
            v_pred_scalar = support.support_to_scalar(v_pred, is_logits=True)
        else:
            loss_v_k_raw = F.mse_loss(v_pred.squeeze(-1), v_tgt, reduction="none")
            v_pred_scalar = v_pred.squeeze(-1)
        if w is not None:
            loss_v_k = (loss_v_k_raw * w).mean()
        else:
            loss_v_k = loss_v_k_raw.mean()
        loss_v_sum = loss_v_sum + loss_v_k
        td_errors_list.append(torch.abs(v_pred_scalar - v_tgt))
    loss_value = loss_v_sum * scale_head
    td_errors = torch.stack(td_errors_list, dim=-1).mean(dim=-1)

    # (c) Policy loss: scaled by 1/K
    loss_p_sum = torch.tensor(0.0, device=actions.device)
    for k in range(effective_k + 1):
        p_pred = policy_logits_list[k]
        if policy_targets is not None:
            p_tgt = policy_targets[:, k]
            if p_tgt.dim() == 1:
                p_tgt = F.one_hot(p_tgt.long(), num_classes=NUM_MACRO_OPTIONS).float()
            loss_p_k_raw = -(p_tgt * F.log_softmax(p_pred, dim=-1)).sum(dim=-1)
        else:
            act_k = actions[:, min(k, effective_k - 1)]
            loss_p_k_raw = F.cross_entropy(p_pred, act_k.long(), reduction="none")
        if w is not None:
            loss_p_k = (loss_p_k_raw * w).mean()
        else:
            loss_p_k = loss_p_k_raw.mean()
        loss_p_sum = loss_p_sum + loss_p_k
    loss_policy = loss_p_sum * scale_head

    # (d) Latent state consistency loss: scaled by 1/K
    loss_cons = torch.tensor(0.0, device=actions.device)
    if obs_seq.dim() == 3 and cons_w > 0.0:
        with torch.no_grad():
            for k in range(1, effective_k + 1):
                tgt_latent = model.representation(obs_seq[:, k])
                loss_cons_k_raw = F.mse_loss(latent_list[k], tgt_latent, reduction="none").mean(dim=-1)
                if w is not None:
                    loss_cons_k = (loss_cons_k_raw * w).mean()
                else:
                    loss_cons_k = loss_cons_k_raw.mean()
                loss_cons = loss_cons + loss_cons_k
        loss_cons = loss_cons * scale_head

    # (e) Conservative Q-Learning (CQL) / Value Equivalence Penalty (Mode B)
    loss_cql = torch.tensor(0.0, device=actions.device)
    if cql_weight > 0.0:
        cql_penalties = []
        for k in range(effective_k + 1):
            v_pred_k = value_logits_list[k]
            if support is not None:
                v_pred_s = support.support_to_scalar(v_pred_k, is_logits=True)
            else:
                v_pred_s = v_pred_k.squeeze(-1)
            v_tgt_k = value_targets[:, k]
            pen = F.relu(v_pred_s - v_tgt_k).pow(2)
            cql_penalties.append(pen.mean())
        loss_cql = torch.stack(cql_penalties).mean() * scale_head

    # (f) L2 regularization parameter c * ||θ||^2
    l2_reg = torch.tensor(0.0, device=actions.device)
    if c_l2 > 0.0:
        for p in model.parameters():
            if p.requires_grad:
                l2_reg = l2_reg + torch.sum(p ** 2)

    total = (
        reward_weight * loss_reward
        + value_weight * loss_value
        + policy_weight * loss_policy
        + cons_w * loss_cons
        + cql_weight * loss_cql
        + c_l2 * l2_reg
    )

    return {
        "total": total,
        "reward": loss_reward.detach(),
        "value": loss_value.detach(),
        "policy": loss_policy.detach(),
        "consistency": loss_cons.detach(),
        "cql": loss_cql.detach(),
        "l2": l2_reg.detach(),
        "td_errors": td_errors.detach(),
    }



def muzero_bootstrap_loss(
    model: MuZeroNetwork,
    obs: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_obs: torch.Tensor,
    policy_targets: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    One-step MuZero loss:
      L = L_reward + L_value + L_policy(+ BC) + L_ latent consistency
    """
    latent, policy, value = model.initial_inference(obs)
    reward_pred, next_latent, next_policy, next_value = model.recurrent_inference(latent, actions)

    # Target value ≈ immediate reward + discounted next network value (stopgrad)
    with torch.no_grad():
        _, _, boot_v = model.initial_inference(next_obs)
        boot_scalar = boot_v if not isinstance(boot_v, torch.Tensor) else boot_v.squeeze(-1)
        value_target = rewards + 0.99 * boot_scalar

    support = getattr(model, "support", None)
    if model.support_size > 1 and support is not None:
        r_logits, _ = model.dynamics.forward_logits(latent, actions)
        _, v_logits = model.prediction.forward_logits(latent)
        loss_reward = F.cross_entropy(r_logits, support.scalar_to_support(rewards))
        loss_value = F.cross_entropy(v_logits, support.scalar_to_support(value_target))
    else:
        loss_reward = F.mse_loss(reward_pred, rewards)
        loss_value = F.mse_loss(value, value_target)

    target_latent = model.representation(next_obs)
    loss_dyn = F.mse_loss(next_latent, target_latent.detach())

    if policy_targets is None:
        # Behavioral prior: uniform over batch actions as one-hot BC
        policy_targets = F.one_hot(actions, num_classes=NUM_MACRO_OPTIONS).float()
    loss_policy = -(policy_targets * torch.log(policy + 1e-8)).sum(dim=-1).mean()

    total = loss_reward + loss_value + loss_dyn + 0.5 * loss_policy
    return {
        "total": total,
        "reward": loss_reward.detach(),
        "value": loss_value.detach(),
        "dynamics": loss_dyn.detach(),
        "policy": loss_policy.detach(),
    }



# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

DEFAULT_MUZERO_CHECKPOINT: str = "muzero_checkpoints.pt"


def resolve_muzero_checkpoint(explicit: Optional[str] = None) -> Optional[str]:
    """Return the first existing MuZero weight path, or None."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(pkg_dir)
    candidates = [
        explicit,
        DEFAULT_MUZERO_CHECKPOINT,
        os.path.join(root_dir, "artifacts", DEFAULT_MUZERO_CHECKPOINT),
        os.path.join(pkg_dir, DEFAULT_MUZERO_CHECKPOINT),
        os.path.join(root_dir, DEFAULT_MUZERO_CHECKPOINT),
        "/workspace/artifacts/muzero_checkpoints.pt",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def save_muzero_checkpoint(
    model: MuZeroNetwork,
    path: str,
    meta: Optional[Dict[str, Any]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    champion_score: Optional[float] = None,
    atomic: bool = True,
) -> str:
    """
    Persist MuZero state_dict (+ optional training metadata, optimizer state, and champion score).
    Guarantees atomic writes via temporary file and atomic rename to prevent corrupted checkpoints.
    """
    payload: Dict[str, Any] = {
        "state_dict": model.state_dict(),
        "obs_dim": getattr(model, "obs_dim", OBS_DIM),
        "hidden_dim": getattr(model, "hidden_dim", HIDDEN_DIM),
        "num_actions": getattr(model, "num_actions", NUM_MACRO_OPTIONS),
        "support_size": getattr(model, "support_size", 1),
        "meta": meta or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if champion_score is not None:
        payload["champion_score"] = float(champion_score)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if atomic:
        tmp_path = path + ".tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    else:
        torch.save(payload, path)
    return path


def load_muzero_checkpoint(
    model: Optional[MuZeroNetwork] = None,
    path: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[MuZeroNetwork, Optional[str]]:
    """
    Load weights into a MuZeroNetwork. Accepts either a raw state_dict or the
    wrapped payload from save_muzero_checkpoint(). Returns (model, loaded_path).
    Automatically adapts support_size from checkpoint.
    """
    resolved = resolve_muzero_checkpoint(path)
    if resolved is None:
        return model or MuZeroNetwork(), None
    payload = torch.load(resolved, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        sd = payload["state_dict"]
        ckpt_supp_size = payload.get("support_size")
    else:
        sd = payload
        ckpt_supp_size = None

    if ckpt_supp_size is None:
        rew_wt = sd.get("dynamics.reward_head.weight")
        ckpt_supp_size = rew_wt.shape[0] if rew_wt is not None else 1

    if model is None:
        ckpt_hidden = int(payload.get("hidden_dim", HIDDEN_DIM)) if isinstance(payload, dict) else HIDDEN_DIM
        model = MuZeroNetwork(hidden_dim=ckpt_hidden, support_size=ckpt_supp_size)
        model.load_state_dict(sd)
    elif getattr(model, "support_size", 1) != ckpt_supp_size or getattr(model, "hidden_dim", HIDDEN_DIM) != int(
        (payload.get("hidden_dim") if isinstance(payload, dict) else None) or model.hidden_dim
    ):
        # Transfer matching parameters (e.g. representation, dynamics trunk, policy head)
        model_dict = model.state_dict()
        transferred = {
            k: v for k, v in sd.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }
        model_dict.update(transferred)
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(sd)

    model.to(device)
    model.eval()
    return model, resolved


def muzero_mcts_distill_loss(
    model: MuZeroNetwork,
    obs: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_obs: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    MuZero fine-tune loss combining:
      - policy KL toward MCTS visit distribution
      - value MSE toward MCTS / wealth targets
      - dynamics bootstrap (reward + latent consistency)
    """
    latent, policy, value = model.initial_inference(obs)
    reward_pred, next_latent, _, _ = model.recurrent_inference(latent, actions)

    loss_policy = -(policy_targets * torch.log(policy + 1e-8)).sum(dim=-1).mean()
    loss_value = F.mse_loss(value, value_targets)
    loss_reward = F.mse_loss(reward_pred, rewards)
    with torch.no_grad():
        target_latent = model.representation(next_obs)
    loss_dyn = F.mse_loss(next_latent, target_latent)

    total = loss_policy + loss_value + loss_reward + loss_dyn
    return {
        "total": total,
        "policy": loss_policy.detach(),
        "value": loss_value.detach(),
        "reward": loss_reward.detach(),
        "dynamics": loss_dyn.detach(),
    }


def muzero_consistency_loss(
    model: MuZeroNetwork,
    obs: torch.Tensor,
    actions: torch.Tensor,
    next_obs: torch.Tensor,
) -> torch.Tensor:
    """EfficientZero-style latent consistency: ‖h(o') − g(h(o), a)‖₂."""
    with torch.no_grad():
        target = model.representation(next_obs)
    latent = model.representation(obs)
    _, pred_next = model.dynamics(latent, actions)
    return F.mse_loss(pred_next, target)
