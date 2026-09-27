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

"""Kaggriculture MuZero — spatial Sampled MuZero is the production API.

Canonical stack:
  - ``KaggricultureMuZeroChassis`` (spatial ResNet h/g/f, factorized heads)
  - ``SampledMuZeroMCTS``
  - ``PrioritizedMuZeroBuffer`` / ``PrioritizedMuZeroTrainer``
  - ``encode_spatial_observation`` → ``[28, 10, 10]``

Legacy MLP / 8-``MacroOption`` MuZero remains under ``muzero.legacy_macro`` and is
re-exported for packaging / ablation (``MuZeroNetwork``, ``LatentMacroOptionMCTS``, …).
"""
from __future__ import annotations

__author__ = "Scott Weeden"
__license__ = "Apache-2.0"

from .spatial_constants import (
    ACTION_CHANNELS,
    DEFAULT_SPATIAL_CHECKPOINT,
    LATENT_CHANNELS,
    NUM_FARMER_ACTIONS,
    NUM_HAND_ASSIGNMENTS,
    NUM_MARKET_ORDERS,
    OBS_CHANNELS,
    joint_action_to_plane,
    soft_cross_entropy,
)
from .chassis import (
    FactorizedKaggriculturePredictionHead,
    KaggricultureMuZeroChassis,
    ResNetBlock2D,
    SimSiamConsistencyLoss,
    SimSiamProjectionPredictionHead,
    SpatialDynamicsNetwork,
    SpatialPredictionNetwork,
    SpatialRepresentationNetwork,
    compute_muzero_unroll_loss_with_consistency,
)
from .buffer import (
    GameTrajectory,
    MuZeroTrajectoryBuffer,
    PrioritizedMuZeroBuffer,
)
from .mcts import MCTSNode, MinMaxStats, PURE_LATENT_MCTS, SampledMuZeroMCTS
from .observation_encoder import (
    KaggricultureObservationEncoder,
    encode_spatial_observation,
)
from .action_translate import label_expert_joint_action, translate_joint_action
from .spatial_checkpoint import (
    assert_spatial_checkpoint_loadable,
    load_spatial_checkpoint,
    promote_spatial_champion,
    save_spatial_checkpoint,
)
from .trainer import MuZeroTrainer, PrioritizedMuZeroTrainer
from .deterministic import DEFAULT_DETERMINISTIC_SEED, set_deterministic_mode

# Legacy macro stack (packaging / ablation)
from .legacy_macro import (
    DEFAULT_MUZERO_CHECKPOINT,
    DynamicsNetwork,
    LatentMacroOptionMCTS,
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
from .encoding import (
    decode_macro_state,
    encode_macro_state,
    encode_observation,
    legal_macro_options,
    wealth,
)
from .lora import (
    DEFAULT_LORA_RANK,
    LoRALinear,
    export_lora_state,
    has_lora,
    inject_lora,
    load_lora_adapters,
    lora_parameters,
    merge_lora,
    save_lora_adapters,
)
from .ppo import (
    DEFAULT_PPO_CHECKPOINT,
    ActorCritic,
    PPOBuffer,
    PPOHyperParams,
    load_ppo_checkpoint,
    ppo_rollout_batch,
    ppo_rollout_episode,
    resolve_ppo_checkpoint,
    save_ppo_checkpoint,
    train_ppo,
)
from .types import (
    DEFAULT_PRICES,
    HIDDEN_DIM,
    LAND_BUFFERS,
    LAND_ORDER,
    LAND_PRICES,
    MACRO_OPTION_NAMES,
    NUM_MACRO_OPTIONS,
    OBS_DIM,
    POLICY_FIBONACCI,
    REWARD_SCALE,
    SUPPORT_B,
    SUPPORT_SIZE,
    DiscreteSupport,
    MacroAction,
    MacroOption,
    MacroState,
)

# Spatial aliases matching docs/prioritized_muzero_system.py naming
# (do not shadow legacy MLP RepresentationNetwork used by packaging)
EncoderNetwork = SpatialRepresentationNetwork
DecoderNetwork = SpatialPredictionNetwork

__all__ = [
    "ACTION_CHANNELS",
    "ActorCritic",
    "DEFAULT_DETERMINISTIC_SEED",
    "DEFAULT_LORA_RANK",
    "DEFAULT_MUZERO_CHECKPOINT",
    "DEFAULT_PPO_CHECKPOINT",
    "DEFAULT_PRICES",
    "DEFAULT_SPATIAL_CHECKPOINT",
    "DiscreteSupport",
    "DecoderNetwork",
    "DynamicsNetwork",
    "EncoderNetwork",
    "FactorizedKaggriculturePredictionHead",
    "GameTrajectory",
    "HIDDEN_DIM",
    "KaggricultureMuZeroChassis",
    "KaggricultureObservationEncoder",
    "LAND_BUFFERS",
    "LAND_ORDER",
    "LAND_PRICES",
    "LATENT_CHANNELS",
    "LatentMacroOptionMCTS",
    "LoRALinear",
    "MACRO_OPTION_NAMES",
    "MacroAction",
    "MacroOption",
    "MacroState",
    "MCTSNode",
    "MinMaxStats",
    "MuZeroMCTSNode",
    "MuZeroNetwork",
    "MuZeroPlanner",
    "MuZeroTrajectoryBuffer",
    "MuZeroTrainer",
    "NUM_FARMER_ACTIONS",
    "NUM_HAND_ASSIGNMENTS",
    "NUM_MACRO_OPTIONS",
    "NUM_MARKET_ORDERS",
    "OBS_CHANNELS",
    "OBS_DIM",
    "POLICY_FIBONACCI",
    "PPOBuffer",
    "PPOHyperParams",
    "PURE_LATENT_MCTS",
    "PredictionNetwork",
    "PrioritizedMuZeroBuffer",
    "PrioritizedMuZeroTrainer",
    "PrioritizedReplayBuffer",
    "REWARD_SCALE",
    "RepresentationNetwork",
    "ResNetBlock2D",
    "SUPPORT_B",
    "SUPPORT_SIZE",
    "SampledMuZeroMCTS",
    "SearchResult",
    "SimSiamConsistencyLoss",
    "SimSiamProjectionPredictionHead",
    "SpatialDynamicsNetwork",
    "SpatialPredictionNetwork",
    "SpatialRepresentationNetwork",
    "assert_spatial_checkpoint_loadable",
    "compute_muzero_unroll_loss_with_consistency",
    "compute_target_values",
    "create_target_network",
    "decode_macro_state",
    "encode_macro_state",
    "encode_observation",
    "encode_spatial_observation",
    "export_lora_state",
    "has_lora",
    "inject_lora",
    "joint_action_to_plane",
    "label_expert_joint_action",
    "legal_macro_options",
    "load_lora_adapters",
    "load_muzero_checkpoint",
    "load_ppo_checkpoint",
    "load_spatial_checkpoint",
    "lora_parameters",
    "merge_lora",
    "muzero_bootstrap_loss",
    "muzero_consistency_loss",
    "muzero_k_step_unroll_loss",
    "muzero_mcts_distill_loss",
    "ppo_rollout_batch",
    "ppo_rollout_episode",
    "promote_spatial_champion",
    "reanalyze_trajectory_slice",
    "resolve_muzero_checkpoint",
    "resolve_ppo_checkpoint",
    "save_lora_adapters",
    "save_muzero_checkpoint",
    "save_ppo_checkpoint",
    "save_spatial_checkpoint",
    "scale_gradient",
    "set_deterministic_mode",
    "soft_cross_entropy",
    "train_ppo",
    "translate_joint_action",
    "update_target_network",
    "wealth",
]
