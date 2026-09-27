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
================================================================================
COMPLETE PRIORITIZED MUZERO SYSTEM FOR KAGGRICULTURE (muzero/prioritized_system.py)
================================================================================
Features:
1. KaggricultureMuZeroChassis: Representation (h), Dynamics (g), Prediction (f)
   with Factorized Policy Heads (Farmer, Hands, Market) & Discrete 601-Atom Support.
2. SimSiam Consistency: Asymmetric Projector/Predictor with Stop-Gradients.
3. Sampled MCTS Engine: Global MinMaxStats, P-UCT Search, & Dirichlet Noise.
4. Prioritized Experience Replay (PER) Trajectory Buffer: TD-Error Priority
   Sampling, Importance Sampling (IS) Weights, & n-Step Bootstrap Targets.
5. Reanalyze Engine & Polyak EMA Target Network Integration.
================================================================================
"""

from __future__ import annotations

from .buffer import (
    GameTrajectory,
    MuZeroTrajectoryBuffer,
    PrioritizedMuZeroBuffer,
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
from .mcts import (
    MCTSNode,
    MinMaxStats,
    SampledMuZeroMCTS,
)
from .observation_encoder import (
    KaggricultureObservationEncoder,
)
from .trainer import (
    MuZeroTrainer,
    PrioritizedMuZeroTrainer,
)

# Canonical aliases matching docs/prioritized_muzero_system.py
RepresentationNetwork = SpatialRepresentationNetwork
DynamicsNetwork = SpatialDynamicsNetwork
PredictionNetwork = SpatialPredictionNetwork

__all__ = [
    "DynamicsNetwork",
    "FactorizedKaggriculturePredictionHead",
    "GameTrajectory",
    "KaggricultureMuZeroChassis",
    "KaggricultureObservationEncoder",
    "MCTSNode",
    "MinMaxStats",
    "MuZeroTrajectoryBuffer",
    "MuZeroTrainer",
    "PredictionNetwork",
    "PrioritizedMuZeroBuffer",
    "PrioritizedMuZeroTrainer",
    "RepresentationNetwork",
    "ResNetBlock2D",
    "SampledMuZeroMCTS",
    "SimSiamConsistencyLoss",
    "SimSiamProjectionPredictionHead",
    "SpatialDynamicsNetwork",
    "SpatialPredictionNetwork",
    "SpatialRepresentationNetwork",
    "compute_muzero_unroll_loss_with_consistency",
]
