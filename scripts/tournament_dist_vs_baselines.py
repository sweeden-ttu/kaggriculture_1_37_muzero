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
tournament_dist_vs_baselines.py (Compatibility Facade -> evaluation.runner)
=============================================================================
Runs a seeded head-to-head tournament evaluating the refactored dist
against each agent in baselines/ with rotated seeds and alternating player seats.
Delegates to the canonical evaluation domain module.
=============================================================================
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from evaluation.benchmarks import find_baselines, run_baseline_tournament as run_tournament
from evaluation.harness import (
    SANDBOX_ROOT,
    SubprocessAgent,
    WORKER_SCRIPT,
    eval_match_worker,
    play_single_game,
    short_name,
)
from evaluation.runner import main

__all__ = [
    "SANDBOX_ROOT",
    "SubprocessAgent",
    "WORKER_SCRIPT",
    "eval_match_worker",
    "find_baselines",
    "main",
    "play_single_game",
    "run_tournament",
    "short_name",
]

if __name__ == "__main__":
    sys.exit(main())
