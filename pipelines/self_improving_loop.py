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

"""Compatibility facade → ``pipelines.phases.phase4_continuous_loop``.

The continuous self-improving RL loop is curriculum **Phase 4**
(``run_phase4_continuous_loop`` + ``phase4_reanalysis``). This module re-exports
the Phase 4 API so ``train.sh`` / ``boost.sh`` / root ``self_improving_loop.py``
keep working unchanged.
"""
from __future__ import annotations

from .phases.phase4_continuous_loop import (
    append_history,
    execute_candidate_trial,
    generate_candidate_hyperparams,
    main,
    next_seed,
    run_phase4_continuous_loop,
    run_evaluation_episodes,
    run_online_selfplay_data_generation,
    setup_baseline_champion,
)

# Historic name kept for callers that imported ``phase4_continuous_loop``.
phase4_continuous_loop = run_phase4_continuous_loop

__all__ = [
    "append_history",
    "execute_candidate_trial",
    "generate_candidate_hyperparams",
    "main",
    "next_seed",
    "phase4_continuous_loop",
    "run_phase4_continuous_loop",
    "run_evaluation_episodes",
    "run_online_selfplay_data_generation",
    "setup_baseline_champion",
]

if __name__ == "__main__":
    raise SystemExit(main())
