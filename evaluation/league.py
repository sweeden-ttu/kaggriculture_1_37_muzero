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

"""League evaluation: round-robin tournaments + PFSP tracker re-exports.

PFSP classes live in ``evaluation.league_tracker`` (docs/League_Tracker.md).
This module keeps the all-vs-all round-robin runner and re-exports the tracker API.
"""
from __future__ import annotations

import os
import time
import zipfile
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence

from .harness import play_single_game, short_name
from .league_tracker import (  # noqa: F401 — public re-exports
    DEFAULT_LEAGUE_STATE,
    EXPLOITER_RESET_WINRATE,
    LeagueTracker,
    PFSPLeagueConfig,
    PFSPSampler,
    demo_sample_opponents,
    register_packages,
    run_pfsp_league,
    select_matchmaking_opponent,
)

SKIP_ZIP_NAMES = {"kaggriculture.zip"}

__all__ = [
    "DEFAULT_LEAGUE_STATE",
    "EXPLOITER_RESET_WINRATE",
    "LeagueTracker",
    "PFSPLeagueConfig",
    "PFSPSampler",
    "demo_sample_opponents",
    "discover_agent_packages",
    "register_packages",
    "run_pfsp_league",
    "run_round_robin_league",
    "select_matchmaking_opponent",
]


def discover_agent_packages(search_dirs: Sequence[str]) -> List[str]:
    """Finds all valid agent zip archives with main.py in the specified directories."""
    found: List[str] = []
    seen = set()
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith(".zip") or name in SKIP_ZIP_NAMES:
                continue
            path = os.path.abspath(os.path.join(d, name))
            if path in seen:
                continue
            try:
                with zipfile.ZipFile(path, "r") as zf:
                    names = zf.namelist()
            except zipfile.BadZipFile:
                continue
            if any(n == "main.py" or n.endswith("/main.py") for n in names):
                seen.add(path)
                found.append(path)
    return found


def run_round_robin_league(
    agent_paths: Sequence[str],
    *,
    games_per_pair: int = 2,
    base_seed: int = 10101,
    episode_steps: int = 720,
    save_replay_dir: Optional[str] = None,
    min_replay_score: float = 80000.0,
    quiet: bool = False,
    tracker: Optional[LeagueTracker] = None,
    tracker_state_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Executes a round-robin league across all discovered agents with alternating seats.

    When ``tracker`` / ``tracker_state_path`` is set, also records PFSP win matrices.
    """
    pairs = list(combinations(agent_paths, 2))
    total_matches = len(pairs) * games_per_pair

    if tracker is None and tracker_state_path:
        tracker = LeagueTracker.load(tracker_state_path)
    if tracker is not None:
        register_packages(tracker, agent_paths)

    if not quiet:
        print("=" * 100)
        print(f"  ALL-VS-ALL ROUND-ROBIN LEAGUE ({len(agent_paths)} agents, {total_matches} total games)")
        print("=" * 100)

    leaderboard: Dict[str, Dict[str, Any]] = {
        p: {
            "path": p,
            "name": short_name(p),
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "points": 0.0,
            "score_sum": 0.0,
            "games": 0,
        }
        for p in agent_paths
    }

    match_log: List[Dict[str, Any]] = []
    game_idx = 0
    t0 = time.time()

    for p_a, p_b in pairs:
        for g in range(games_per_pair):
            game_idx += 1
            a_is_p0 = (g % 2 == 0)
            seed = base_seed + game_idx
            res = play_single_game(
                p_a,
                p_b,
                seed=seed,
                a_is_p0=a_is_p0,
                episode_steps=episode_steps,
                save_replay_dir=save_replay_dir,
                min_replay_score=min_replay_score,
            )

            st_a = leaderboard[p_a]
            st_b = leaderboard[p_b]
            st_a["games"] += 1
            st_b["games"] += 1
            st_a["score_sum"] += res["score_a"]
            st_b["score_sum"] += res["score_b"]

            if res["winner"] == "A":
                st_a["wins"] += 1
                st_a["points"] += 1.0
                st_b["losses"] += 1
                if tracker is not None:
                    tracker.record_match_outcome(p_a, p_b, 1.0)
            elif res["winner"] == "B":
                st_b["wins"] += 1
                st_b["points"] += 1.0
                st_a["losses"] += 1
                if tracker is not None:
                    tracker.record_match_outcome(p_a, p_b, 0.0)
            else:
                st_a["ties"] += 1
                st_b["ties"] += 1
                st_a["points"] += 0.5
                st_b["points"] += 0.5
                if tracker is not None:
                    tracker.record_match_outcome(p_a, p_b, 0.5)

            match_log.append({
                "game": game_idx,
                "a": p_a,
                "b": p_b,
                "winner": res["winner"],
                "score_a": res["score_a"],
                "score_b": res["score_b"],
                "seed": seed,
            })
            if not quiet:
                print(
                    f"  [{game_idx:>3}/{total_matches}] "
                    f"{short_name(p_a)} vs {short_name(p_b)}  "
                    f"${res['score_a']:,.0f} / ${res['score_b']:,.0f}  "
                    f"→ {res['winner']}"
                )

    rows = sorted(leaderboard.values(), key=lambda r: (-r["points"], -r["score_sum"]))
    for rank, r in enumerate(rows, start=1):
        r["rank"] = rank
        r["avg_score"] = r["score_sum"] / max(1, r["games"])
        r["win_rate_pct"] = 100.0 * r["wins"] / max(1, r["games"])

    state_path = None
    if tracker is not None:
        state_path = tracker.save(tracker_state_path or DEFAULT_LEAGUE_STATE)

    report: Dict[str, Any] = {
        "mode": "round_robin",
        "agents": len(agent_paths),
        "games": total_matches,
        "elapsed_s": time.time() - t0,
        "leaderboard": rows,
        "match_log": match_log,
        "tracker_state": state_path,
    }

    if not quiet:
        print("=" * 100)
        print(f"  {'Rank':>4} | {'Name':<28} | {'Pts':>6} | {'W-L-T':^11} | {'WR%':>5} | {'Avg $':>12}")
        print("-" * 100)
        for r in rows:
            wlt = f"{r['wins']}-{r['losses']}-{r['ties']}"
            print(
                f"  {r['rank']:>4} | {r['name']:<28} | {r['points']:>6.1f} | "
                f"{wlt:^11} | {r['win_rate_pct']:>5.1f}% | ${r['avg_score']:>11,.0f}"
            )
        if state_path:
            print(f"  PFSP tracker → {state_path}")
        print("=" * 100)

    return report
