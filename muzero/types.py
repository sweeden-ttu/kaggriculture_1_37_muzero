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

"""Core types, enums, constants, and tree node structures for MuZero.

Shared observation/state types live here so ``encoding`` and ``_core`` never
need to import the Phase-1 analytical teacher (``economy``).
"""
from __future__ import annotations

__author__ = "Scott Weeden"
__license__ = "Apache-2.0"

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Macro state / action dataclasses (tensor-encoding domain, not the teacher)
# ---------------------------------------------------------------------------

POLICY_FIBONACCI: Tuple[int, ...] = (1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144)

LAND_ORDER: Tuple[str, ...] = ("NE", "SW", "SE")
LAND_PRICES: Dict[str, float] = {"NE": 1000.0, "SW": 2000.0, "SE": 4000.0}
LAND_BUFFERS: Dict[str, float] = {"NE": 500.0, "SW": 500.0, "SE": 800.0}

DEFAULT_PRICES: Dict[str, float] = {
    "WHEAT": 35.0,
    "CARROT": 40.0,
    "TOMATO": 50.0,
    "STRAWBERRY": 90.0,
    "MELON": 280.0,
    "EGG": 70.0,
    "MILK": 256.0,
    "WOOL": 240.0,
}


@dataclass(frozen=True)
class MacroAction:
    expansion_target: Optional[str]
    target_hands: int


@dataclass
class MacroState:
    day: int
    capital: float
    unlocked: Tuple[str, ...]
    shed: Dict[str, float]
    prices: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PRICES))
    active_animals: int = 1

    def copy(self) -> "MacroState":
        return MacroState(
            day=self.day,
            capital=self.capital,
            unlocked=self.unlocked,
            shed=dict(self.shed),
            prices=dict(self.prices),
            active_animals=self.active_animals,
        )


# ---------------------------------------------------------------------------
# Macro-option action abstraction
# ---------------------------------------------------------------------------


class MacroOption(IntEnum):
    """High-level daily macro options for latent MCTS (not tile primitives)."""

    FRONT_LOAD_SURVIVAL = 0
    EXPANSION_PREPARATION = 1
    TRICKLE_LIQUIDATION = 2
    EXPAND_NE = 3
    EXPAND_SW = 4
    EXPAND_SE = 5
    HIRE_HANDS = 6
    PASS = 7


MACRO_OPTION_NAMES: Dict[int, str] = {int(o): o.name for o in MacroOption}
NUM_MACRO_OPTIONS: int = len(MacroOption)
OBS_DIM: int = 32
# Canonical latent dimension (d=256) per REFACTOR_PLAN / README § meta
HIDDEN_DIM: int = 256
REWARD_SCALE: float = 1.0 / 1000.0  # align with DuelingRainbowDQN

# Discrete support half-width B: bins = {-B, ..., B} → size 2B+1 = 601
SUPPORT_B: int = 300
SUPPORT_SIZE: int = 2 * SUPPORT_B + 1  # 601

BASE_CROP_PRICES: Dict[str, float] = {
    "WHEAT": 25.0,
    "CARROT": 35.0,
    "TOMATO": 60.0,
    "STRAWBERRY": 120.0,
    "MELON": 250.0,
}


class DiscreteSupport:
    """Value-equivalent discrete support with invertible transform φ.

    Encodes scalars over categorical bins ``{-B, ..., B}`` (default B=300 → 601
    bins) so reward/value heads train with stable cross-entropy instead of MSE.

      φ(x) = sign(x) · (√(|x| + 1) − 1 + ε·|x|)
      φ⁻¹(y) = exact quadratic inverse
    """

    def __init__(
        self,
        min_val: int = -SUPPORT_B,
        max_val: int = SUPPORT_B,
        eps: float = 0.001,
    ):
        self.min_val = int(min_val)
        self.max_val = int(max_val)
        self.eps = float(eps)
        self.num_supports = self.max_val - self.min_val + 1
        self.values = torch.linspace(self.min_val, self.max_val, self.num_supports)

    def phi(self, x: torch.Tensor) -> torch.Tensor:
        ax = torch.abs(x)
        return torch.sign(x) * (torch.sqrt(ax + 1.0) - 1.0 + self.eps * ax)

    def phi_inv(self, y: torch.Tensor) -> torch.Tensor:
        ay = torch.abs(y.double())
        u = (torch.sqrt(1.0 + 4.0 * self.eps * (ay + 1.0 + self.eps)) - 1.0) / (2.0 * self.eps)
        rec = torch.sign(y.double()) * (u * u - 1.0)
        return rec.to(y.dtype)

    def scalar_to_support(self, scalar: torch.Tensor) -> torch.Tensor:
        """Projects a scalar target onto the two adjacent discrete supports."""
        transformed = self.phi(scalar).clamp(self.min_val, self.max_val)
        idx_float = transformed - self.min_val
        lower_idx = idx_float.floor().long().clamp(0, self.num_supports - 1)
        upper_idx = idx_float.ceil().long().clamp(0, self.num_supports - 1)
        upper_weight = idx_float - lower_idx.float()
        lower_weight = 1.0 - upper_weight

        probs = torch.zeros(
            *scalar.shape, self.num_supports, dtype=torch.float32, device=scalar.device
        )
        probs.scatter_add_(-1, lower_idx.unsqueeze(-1), lower_weight.unsqueeze(-1))
        diff_mask = lower_idx != upper_idx
        if diff_mask.any():
            probs.scatter_add_(-1, upper_idx.unsqueeze(-1), upper_weight.unsqueeze(-1))
        return probs

    def support_to_scalar(
        self, logits_or_probs: torch.Tensor, is_logits: bool = True
    ) -> torch.Tensor:
        """Converts distribution logits or probabilities into an expected scalar."""
        if is_logits:
            probs = torch.softmax(logits_or_probs, dim=-1)
        else:
            probs = logits_or_probs
        vals = self.values.to(probs.device, probs.dtype)
        transformed = torch.sum(probs * vals, dim=-1)
        return self.phi_inv(transformed)


@dataclass
class SearchResult:
    """Output summary of an MCTS search."""

    best_option: MacroOption
    visit_counts: Dict[MacroOption, int]
    root_value: float
    policy_target: torch.Tensor  # visit-count distribution for training


class MinMaxStats:
    """
    Centralized tree-wide min-max tracking for MuZero pUCT normalization (Equation 5, Appendix B).

    Unlike local sibling scaling which creates inconsistent exploration boundaries,
    MinMaxStats tracks global minimum and maximum Q-values across the entire search tree.
    Updated during the Backup phase whenever a new Q(s, a) is calculated.
    """

    def __init__(self, min_value: Optional[float] = None, max_value: Optional[float] = None):
        self.minimum = float(min_value) if min_value is not None else float("inf")
        self.maximum = float(max_value) if max_value is not None else -float("inf")

    def update(self, value: float) -> None:
        """Ingests a node evaluation / Q-value during Backup phase."""
        if math.isnan(value):
            return
        v = float(value)
        if v < self.minimum:
            self.minimum = v
        if v > self.maximum:
            self.maximum = v

    def normalize(self, value: float) -> float:
        """
        Normalizes Q(s, a) into [0, 1] relative to tree-wide min/max (Equation 5).
        Returns value as-is if tree has not established a range (maximum <= minimum).
        """
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

    def clear(self) -> None:
        self.minimum = float("inf")
        self.maximum = -float("inf")

    def is_valid(self) -> bool:
        return self.maximum > self.minimum and math.isfinite(self.minimum) and math.isfinite(self.maximum)


class MuZeroMCTSNode:
    """Tree node storing latent state (+ optional analytical MacroState)."""

    def __init__(
        self,
        latent: torch.Tensor,
        prior: float = 1.0,
        parent: Optional["MuZeroMCTSNode"] = None,
        action: Optional[MacroOption] = None,
        macro_state: Optional[MacroState] = None,
        reward: float = 0.0,
    ):
        self.latent = latent
        self.prior = prior
        self.parent = parent
        self.action = action
        self.macro_state = macro_state
        self.reward = reward
        self.children: Dict[MacroOption, MuZeroMCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def select_child(
        self,
        c_puct: float = 1.5,
        min_max_stats: Optional[MinMaxStats] = None,
    ) -> Tuple[MacroOption, "MuZeroMCTSNode"]:
        """P-UCT(a) = Q_norm(s,a) + P(s,a) * √N(s) / (1 + N(s,a)) with tree-wide or unvisited exploration."""
        for act, child in self.children.items():
            if child.visit_count == 0:
                return act, child

        use_tree_stats = min_max_stats is not None and min_max_stats.is_valid()
        q_min: float = 0.0
        q_range: float = 1.0
        if not use_tree_stats:
            q_vals = [c.q_value for c in self.children.values() if c.visit_count > 0]
            q_min = min(q_vals) if q_vals else 0.0
            q_max = max(q_vals) if q_vals else 1.0
            q_range = max(1e-4, q_max - q_min)

        best_score = -float("inf")
        best_action: Optional[MacroOption] = None
        best_child: Optional[MuZeroMCTSNode] = None
        sqrt_total = math.sqrt(max(1, self.visit_count))

        for act, child in self.children.items():
            if use_tree_stats:
                assert min_max_stats is not None
                q_norm = min_max_stats.normalize(child.q_value)
            else:
                q_norm = (child.q_value - q_min) / q_range
            u = c_puct * child.prior * (sqrt_total / (1.0 + child.visit_count))
            score = q_norm + u
            if score > best_score:
                best_score = score
                best_action = act
                best_child = child

        assert best_action is not None and best_child is not None
        return best_action, best_child

