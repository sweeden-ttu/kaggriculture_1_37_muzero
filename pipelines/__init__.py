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

"""Training and reinforcement learning pipeline domain.

Organized by training phases:
  - Phase 1: Analytical bootstrap (latent dynamics & representation consistency)
  - Phase 2: MCTS policy/value distillation with K-step lookahead reanalysis
  - Phase 3: Expert behavior cloning with Prioritized Experience Replay (PER)
  - Phase 4: Continuous self-play / re-analysis loop (LoRA adapters + champion gate)
"""
from __future__ import annotations

from .learn_from_replay import main as learn_from_replay_main
from .phases.phase1_bootstrap import phase1_analytical_bootstrap
from .phases.phase2_distill import phase2_mcts_distill
from .phases.phase3_bc import phase3_expert_behavior_cloning
from .phases.phase4_continuous_loop import (
    main as phase4_continuous_loop_main,
    run_phase4_continuous_loop,
)
from .self_improving_loop import main as self_improving_loop_main
from .train_muzero import main as train_muzero_main


def __getattr__(name: str):
    if name in (
        "generate_candidate_hyperparams",
        "hybrid_trainer_main",
        "run_hybrid_trainer_loop",
        "run_offline_sweep_trial",
        "run_online_selfplay_trial",
    ):
        from . import hybrid_trainer

        return getattr(hybrid_trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "generate_candidate_hyperparams",
    "hybrid_trainer_main",
    "learn_from_replay_main",
    "phase1_analytical_bootstrap",
    "phase2_mcts_distill",
    "phase3_expert_behavior_cloning",
    "phase4_continuous_loop_main",
    "run_hybrid_trainer_loop",
    "run_offline_sweep_trial",
    "run_online_selfplay_trial",
    "run_phase4_continuous_loop",
    "self_improving_loop_main",
    "train_muzero_main",
]
