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

"""Evaluation and tournament benchmark domain.

Organized by opponent tiers:
  - Tier 1: Direct heuristic baselines (submission_2000, submission_2500, submission_thirst)
  - Tier 2: Hybrid chassis direct baselines (submission_hybrid_direct, submission_beforegap)
  - Tier 3: Autonomous MuZero neural agents (submission_muzero, champion snapshots)
"""
from __future__ import annotations

from .harness import (
    SANDBOX_ROOT,
    SubprocessAgent,
    eval_match_worker,
    play_single_game,
    short_name,
)
from .tiers import BASELINE_TIERS, BaselineTier, classify_agent_tier
from .benchmarks import (
    find_baselines,
    gate_candidate_vs_2033,
    resolve_2033_champion,
    run_baseline_tournament,
)
from .league import (
    DEFAULT_LEAGUE_STATE,
    LeagueTracker,
    PFSPLeagueConfig,
    PFSPSampler,
    demo_sample_opponents,
    discover_agent_packages,
    register_packages,
    run_pfsp_league,
    run_round_robin_league,
    select_matchmaking_opponent,
)

__all__ = [
    "BASELINE_TIERS",
    "BaselineTier",
    "DEFAULT_LEAGUE_STATE",
    "LeagueTracker",
    "PFSPLeagueConfig",
    "PFSPSampler",
    "SANDBOX_ROOT",
    "SubprocessAgent",
    "classify_agent_tier",
    "demo_sample_opponents",
    "discover_agent_packages",
    "eval_match_worker",
    "find_baselines",
    "gate_candidate_vs_2033",
    "play_single_game",
    "register_packages",
    "resolve_2033_champion",
    "run_baseline_tournament",
    "run_pfsp_league",
    "run_round_robin_league",
    "select_matchmaking_opponent",
    "short_name",
]
