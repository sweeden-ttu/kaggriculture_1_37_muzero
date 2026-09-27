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

"""Local buoy-proxy gate: candidate must beat a personal-best / baseline zip.

Calibrated toward clearing a public ~3200 buoy by requiring H2H dominance vs a
reference agent (default: strongest local baseline or --buoy-agent zip).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _resolve_buoy_agent(explicit: Optional[str]) -> str:
    if explicit and os.path.isfile(explicit):
        return explicit
    candidates = [
        os.path.join(HERE, "artifacts", "champion_submission.zip"),
        os.path.join(HERE, "dist", "head_pl_submission.zip"),
    ]
    baselines = os.path.join(HERE, "baselines")
    if os.path.isdir(baselines):
        zips = sorted(
            [
                os.path.join(baselines, f)
                for f in os.listdir(baselines)
                if f.endswith(".zip") and "champion" in f.lower()
            ],
            key=os.path.getmtime,
            reverse=True,
        )
        candidates = zips + candidates
        scored = sorted(
            [
                os.path.join(baselines, f)
                for f in os.listdir(baselines)
                if f.startswith("submission_") and f.endswith(".zip")
            ]
        )
        candidates.extend(reversed(scored))
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError("No buoy reference agent found; pass --buoy-agent")


def run_buoy_gate(
    candidate: str,
    buoy_agent: str,
    *,
    games: int = 4,
    win_rate_threshold: float = 0.55,
) -> Dict[str, Any]:
    from evaluation.harness import play_single_game

    results: List[Dict[str, Any]] = []
    wins = 0
    losses = 0
    draws = 0
    for g in range(games):
        a_is_p0 = g % 2 == 0
        outcome = play_single_game(candidate, buoy_agent, seed=1000 + g, a_is_p0=a_is_p0)
        cand_r = float(outcome.get("score_a", 0) or 0)
        opp_r = float(outcome.get("score_b", 0) or 0)
        if cand_r > opp_r:
            wins += 1
            verdict = "win"
        elif cand_r < opp_r:
            losses += 1
            verdict = "loss"
        else:
            draws += 1
            verdict = "draw"
        results.append(
            {
                "game": g,
                "a_is_p0": a_is_p0,
                "cand_reward": cand_r,
                "opp_reward": opp_r,
                "verdict": verdict,
            }
        )
        print(f"  game {g}: {verdict} cand={cand_r:.1f} opp={opp_r:.1f}")

    decided = max(1, wins + losses)
    win_rate = wins / decided
    passed = win_rate >= win_rate_threshold and wins > losses
    return {
        "candidate": candidate,
        "buoy_agent": buoy_agent,
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "threshold": win_rate_threshold,
        "passed": passed,
        "results": results,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Buoy-proxy H2H gate for spatial submissions")
    p.add_argument(
        "--candidate",
        default=os.path.join(HERE, "dist", "head_muzero_submission.zip"),
        help="Candidate agent zip/py",
    )
    p.add_argument("--buoy-agent", default=None, help="Personal-best / buoy reference zip")
    p.add_argument("--games", type=int, default=4)
    p.add_argument("--win-rate", type=float, default=0.55, help="Min win rate vs buoy (excl draws)")
    p.add_argument("--out", default=os.path.join(HERE, "artifacts", "buoy_gate_report.json"))
    args = p.parse_args()

    if not os.path.isfile(args.candidate):
        print(f"[buoy-gate] candidate missing: {args.candidate}")
        return 2
    buoy = _resolve_buoy_agent(args.buoy_agent)
    print(f"[buoy-gate] candidate={args.candidate}")
    print(f"[buoy-gate] buoy={buoy} games={args.games} threshold={args.win_rate}")
    report = run_buoy_gate(
        args.candidate, buoy, games=args.games, win_rate_threshold=args.win_rate
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[buoy-gate] passed={report['passed']} win_rate={report['win_rate']:.3f}")
    print(f"[buoy-gate] wrote {args.out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
