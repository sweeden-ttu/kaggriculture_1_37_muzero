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
tournament_sub_vs_base.py (Compatibility Facade -> evaluation.runner)
=============================================================================
N-Game Tournament: submission.zip vs baseline.zip.
Delegates to the canonical evaluation domain module.
=============================================================================
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from evaluation.harness import (
    SANDBOX_ROOT,
    SubprocessAgent,
    WORKER_SCRIPT,
    eval_match_worker,
    extract_agent,
    play_single_game,
    short_name,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Head-to-head N-game match")
    parser.add_argument("--submission", default="dist/submission.zip", help="Agent under test (.zip or .py)")
    parser.add_argument("--baseline", default="baselines/submission_hybrid_direct.zip", help="Baseline opponent (.zip)")
    parser.add_argument("--games", type=int, default=10, help="Number of games")
    parser.add_argument("--seed", type=int, default=10101, help="Starting seed")
    parser.add_argument("--save-replays", action="store_true", help="Save match replays >= $80k")
    args = parser.parse_args()

    from evaluation.runner import main as runner_main
    sys.argv = [
        sys.argv[0],
        "h2h",
        "--agent-a", args.submission,
        "--agent-b", args.baseline,
        "--games", str(args.games),
        "--seed", str(args.seed),
    ]
    if args.save_replays:
        sys.argv.append("--save-replays")
    return runner_main()


if __name__ == "__main__":
    sys.exit(main())
