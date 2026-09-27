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
boost_loop.py
=============
Boost Mode Autonomous MuZero Training Loop.

Uses the same champion models and snapshots as train.sh, but trains under the
inverted boost loss schedule where loss weights start at the high end (e.g. 0.897)
and progress downward.
"""
from __future__ import annotations

import os
import sys

# Ensure execution in boost schedule mode across all child subprocesses
os.environ["MUZERO_SCHEDULE_MODE"] = "boost"

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from pipelines.adaptive_moment_estimation import (
    apply_adaptive_moment_estimation_schedule,
    apply_boost_schedule,
)
apply_adaptive_moment_estimation_schedule()

from pipelines.self_improving_loop import main

if __name__ == "__main__":
    print("\n" + "=" * 90)
    print("  MUZERO BOOST SCHEDULE TRAINING ENGINE INITIALIZED")
    print("  - Training Fuel : Top-Tier Replays (Raw Data Fuel -> Loss & Gradients)")
    print("  - Optimizer     : Adam (Applies Updates to Weights / LoRA Adapters)")
    print("  - Schedule Dial : Inverted Boost Schedule (Starts High -> Progresses Downward)")
    print("  - Model State   : Trained Champion Foundation Weights (Base Frozen under LoRA)")
    print("  - Pipeline      : Behavioral Cloning -> Bootstrapping -> MCTS Distillation")
    print("  - Head Weights  : Discrete Policy and Value Head Weight Files Saved")
    print("=" * 90 + "\n")
    sys.exit(main())
