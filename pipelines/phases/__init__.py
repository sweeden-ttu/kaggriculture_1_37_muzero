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

"""Four-phase MuZero training pipeline stages."""
from __future__ import annotations

from .phase1_bootstrap import phase1_analytical_bootstrap
from .phase2_distill import phase2_mcts_distill
from .phase3_bc import phase3_expert_behavior_cloning
from .phase4_continuous_loop import (
    execute_candidate_trial,
    generate_candidate_hyperparams,
    main as phase4_continuous_loop_main,
    run_phase4_continuous_loop,
)
from .phase4_reanalysis import (
    harvest_selfplay_trajectories,
    phase4_reanalysis,
    route_base_snapshot,
    route_lora_adapter,
)
from .phase_spatial import (
    harvest_spatial_selfplay,
    phase_spatial_bc,
    phase_spatial_bootstrap,
    phase_spatial_mcts_distill,
    phase_spatial_reanalyze,
    save_spatial_checkpoint,
)
from .phase_spatial_loop import (
    main as phase_spatial_loop_main,
    run_spatial_continuous_loop,
)

__all__ = [
    "phase1_analytical_bootstrap",
    "phase2_mcts_distill",
    "phase3_expert_behavior_cloning",
    "phase4_reanalysis",
    "run_phase4_continuous_loop",
    "phase4_continuous_loop_main",
    "harvest_selfplay_trajectories",
    "harvest_spatial_selfplay",
    "phase_spatial_bc",
    "phase_spatial_bootstrap",
    "phase_spatial_mcts_distill",
    "phase_spatial_reanalyze",
    "save_spatial_checkpoint",
    "run_spatial_continuous_loop",
    "phase_spatial_loop_main",
    "route_base_snapshot",
    "route_lora_adapter",
    "execute_candidate_trial",
    "generate_candidate_hyperparams",
]
