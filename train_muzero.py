#!/usr/bin/env python3
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
train_muzero.py (Authoritative Training Entrypoint -> pipelines.train_muzero)
=============================================================================
Four-phase offline & self-play trainer for Kaggriculture MuZero:
  - Phase 1: Analytical Bootstrap (latent dynamics consistency)
  - Phase 2: MCTS Distillation (lookahead reanalysis)
  - Phase 3: Expert Behavior Cloning (PER trajectory sampling)
  - Phase 4: Continuous Re-analysis (LoRA-only adapter updates)
=============================================================================
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from pipelines.phases.phase1_bootstrap import phase1_analytical_bootstrap as phase1
from pipelines.phases.phase2_distill import phase2_mcts_distill as phase2
from pipelines.phases.phase3_bc import (
    build_expert_trajectory_pool,
    discover_expert_replays,
    phase3_expert_behavior_cloning as phase3,
    trajectories_from_replay,
    transitions_from_replay,
)
from pipelines.phases.phase4_reanalysis import phase4_reanalysis as phase4
from pipelines.low_rank_adaptation import (
    compute_low_rank_adaptation_scheduled_weights,
    compute_separate_scheduled_weights,
    validate_loss_weights_rule,
)
from pipelines.train_muzero import main, smoke_expand

__all__ = [
    "build_expert_trajectory_pool",
    "compute_low_rank_adaptation_scheduled_weights",
    "compute_separate_scheduled_weights",
    "discover_expert_replays",
    "main",
    "phase1",
    "phase2",
    "phase3",
    "phase4",
    "smoke_expand",
    "trajectories_from_replay",
    "transitions_from_replay",
    "validate_loss_weights_rule",
]

if __name__ == "__main__":
    main()
