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
tournament_top3_vs_2033.py - Head-to-Head Tournament: Top 3 Candidates vs Kaggle 2033
=============================================================================
Evaluates the top 3 podium winners:
  - submission_boost_cand010.zip
  - submission_boost_cand012.zip
  - submission_train_cand037.zip
Against the authoritative Kaggle-scored gold standard:
  - /Users/sweeden/muzero/baselines/submission_2033.zip
Across balanced alternating seats with 8 parallel workers.
=============================================================================
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from evaluation.candidates_vs_baselines import run_candidates_vs_baselines_tournament, print_tournament_summary
from evaluation.harness import short_name

BASELINES_DIR = os.path.join(HERE, "baselines")
STAGING_DIR = "/Volumes/TRAINREPLAYBOOST/muzero/artifacts/new_baselines_staging"
REPLAYS_DIR = os.path.join(HERE, "replays")
REPORT_PATH = os.path.join(HERE, "artifacts", "tournament_top3_vs_2033.json")


def resolve_candidate(name: str) -> str:
    """Finds candidate package in local baselines or staging directory."""
    fn = f"{name}.zip" if not name.endswith(".zip") else name
    local_p = os.path.join(BASELINES_DIR, fn)
    if os.path.isfile(local_p):
        return local_p
    staging_p = os.path.join(STAGING_DIR, fn)
    if os.path.isfile(staging_p):
        return staging_p
    raise FileNotFoundError(f"Cannot find candidate package {fn}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Top 3 Candidates vs Kaggle 2033 Baseline")
    parser.add_argument("--games-per-candidate", type=int, default=6, help="Games per candidate (balanced P0/P1)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker processes")
    parser.add_argument("--seed", type=int, default=203301, help="Starting RNG seed")
    parser.add_argument("--report", default=REPORT_PATH, help="Output JSON path")
    parser.add_argument("--baseline", default=os.path.join(BASELINES_DIR, "submission_2033.zip"))
    args = parser.parse_args()

    candidates = [
        resolve_candidate("submission_boost_cand010"),
        resolve_candidate("submission_boost_cand012"),
        resolve_candidate("submission_train_cand037"),
    ]

    baseline_path = os.path.abspath(args.baseline)
    assert os.path.isfile(baseline_path), f"Baseline file not found: {baseline_path}"

    print("=" * 100)
    print("  SHOWDOWN TOURNAMENT: TOP 3 PODIUM CANDIDATES VS KAGGLE 2033 GOLD STANDARD")
    print("=" * 100)
    print(f"  Reference Baseline: {short_name(baseline_path)} ({baseline_path})")
    print(f"  Candidates ({len(candidates)}): {', '.join(short_name(c) for c in candidates)}")
    print(f"  Games per Matchup:  {args.games_per_candidate} (Balanced Alternating Seats P0 <-> P1)")
    print(f"  Total Games:        {len(candidates) * args.games_per_candidate}")
    print(f"  Parallel Workers:   {args.workers}")
    print(f"  Base Seed:          {args.seed}")
    print("=" * 100)

    report = run_candidates_vs_baselines_tournament(
        candidates=candidates,
        baselines=[baseline_path],
        games_per_matchup=args.games_per_candidate,
        base_seed=args.seed,
        n_workers=args.workers,
        save_replay_dir=REPLAYS_DIR,
        min_replay_score=80000.0,
        quiet=False,
    )

    print_tournament_summary(report, saved_records=[])

    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[Saved tournament report] → {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
