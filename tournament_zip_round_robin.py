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
tournament_zip_round_robin.py (Compatibility Facade -> evaluation.runner)
=============================================================================
All-vs-all round-robin league across discovered agent zip packages.
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

from evaluation.harness import SubprocessAgent, WORKER_SCRIPT, short_name
from evaluation.league import discover_agent_packages, run_round_robin_league


def main() -> int:
    parser = argparse.ArgumentParser(description="All-vs-all round-robin league")
    parser.add_argument("--dirs", nargs="+", default=["dist", "baselines"], help="Directories to scan for agent zips")
    parser.add_argument("--games-per-pair", type=int, default=2, help="Games per pair (seat alternating)")
    parser.add_argument("--seed", type=int, default=10101, help="Starting seed")
    parser.add_argument("--save-replays", action="store_true", help="Save match replays >= $80k")
    args = parser.parse_args()

    from evaluation.runner import main as runner_main
    sys.argv = [
        sys.argv[0],
        "league",
        "--dirs", *args.dirs,
        "--games-per-pair", str(args.games_per_pair),
        "--seed", str(args.seed),
    ]
    if args.save_replays:
        sys.argv.append("--save-replays")
    return runner_main()


if __name__ == "__main__":
    sys.exit(main())
