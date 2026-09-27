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

"""Spatial Sampled MuZero is the production stack.

Macro-option MLP MuZero lives in ``muzero.legacy_macro`` for packaging/ablation.
This module re-exports legacy helpers still needed by hybrid packaging and PPO.
"""
from __future__ import annotations

from .legacy_macro import (  # noqa: F401
    DEFAULT_MUZERO_CHECKPOINT,
    DynamicsNetwork,
    LatentMacroOptionMCTS,
    MinMaxStats as MacroMinMaxStats,
    MuZeroMCTSNode,
    MuZeroNetwork,
    MuZeroPlanner,
    PredictionNetwork,
    PrioritizedReplayBuffer,
    RepresentationNetwork,
    SearchResult,
    compute_target_values,
    create_target_network,
    load_muzero_checkpoint,
    muzero_bootstrap_loss,
    muzero_consistency_loss,
    muzero_k_step_unroll_loss,
    muzero_mcts_distill_loss,
    reanalyze_trajectory_slice,
    resolve_muzero_checkpoint,
    save_muzero_checkpoint,
    scale_gradient,
    update_target_network,
)

# Prefer spatial MinMaxStats name collision avoidance for direct `_core` users.
MinMaxStats = MacroMinMaxStats

__all__ = [
    "DEFAULT_MUZERO_CHECKPOINT",
    "DynamicsNetwork",
    "LatentMacroOptionMCTS",
    "MinMaxStats",
    "MuZeroMCTSNode",
    "MuZeroNetwork",
    "MuZeroPlanner",
    "PredictionNetwork",
    "PrioritizedReplayBuffer",
    "RepresentationNetwork",
    "SearchResult",
    "compute_target_values",
    "create_target_network",
    "load_muzero_checkpoint",
    "muzero_bootstrap_loss",
    "muzero_consistency_loss",
    "muzero_k_step_unroll_loss",
    "muzero_mcts_distill_loss",
    "reanalyze_trajectory_slice",
    "resolve_muzero_checkpoint",
    "save_muzero_checkpoint",
    "scale_gradient",
    "update_target_network",
]
