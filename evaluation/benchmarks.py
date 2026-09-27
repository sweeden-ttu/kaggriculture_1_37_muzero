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

"""Benchmark tournament evaluation against tiered opponent baselines."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from typing import Any, Dict, List, Optional, Sequence

from .harness import eval_match_worker, play_single_game, short_name
from .tiers import BaselineTier, classify_agent_tier

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINES_DIR = os.path.join(HERE, "baselines")


def find_baselines(baselines_input: str = BASELINES_DIR) -> List[str]:
    """Find baseline archives in a directory or validate individual files."""
    if os.path.isfile(baselines_input):
        return [os.path.abspath(baselines_input)]
    if not os.path.isdir(baselines_input):
        raise FileNotFoundError(f"Baselines path not found: {baselines_input}")

    found: List[str] = []
    for fn in sorted(os.listdir(baselines_input)):
        if fn.endswith(".zip") and not fn.startswith("."):
            found.append(os.path.abspath(os.path.join(baselines_input, fn)))
    if not found:
        raise RuntimeError(f"No .zip baselines found in {baselines_input}")
    return found


def run_baseline_tournament(
    agent_path: str,
    baseline_paths: Sequence[str],
    *,
    games_per_baseline: int = 4,
    base_seed: int = 10101,
    episode_steps: int = 720,
    save_replay_dir: Optional[str] = None,
    min_replay_score: float = 80000.0,
    n_workers: int = 4,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Runs a round-robin tournament of agent_path against each baseline opponent."""
    agent_label = short_name(agent_path)
    total_matches = len(baseline_paths) * games_per_baseline

    if not quiet:
        print("=" * 100)
        print(f"  TOURNAMENT BENCHMARK: {agent_label} vs {len(baseline_paths)} Opponent Baselines")
        print("=" * 100)
        print(f"  Agent Under Test:   {agent_path}")
        print(f"  Baselines:          {', '.join(short_name(b) for b in baseline_paths)}")
        print(f"  Games/Baseline:     {games_per_baseline} (Rotated Seats P0 ↔ P1)")
        print(f"  Total Matches:      {total_matches} (Workers: {n_workers})")
        print(f"  Base Seed:          {base_seed}")
        print("-" * 100)

    tasks: List[Dict[str, Any]] = []
    game_idx = 0
    for b_path in baseline_paths:
        for g in range(games_per_baseline):
            game_idx += 1
            is_p0 = (g % 2 == 0)
            seed = base_seed + game_idx
            tasks.append({
                "agent_a": agent_path,
                "agent_b": b_path,
                "seed": seed,
                "a_is_p0": is_p0,
                "episode_steps": episode_steps,
                "save_replay_dir": save_replay_dir,
                "min_replay_score": min_replay_score,
            })

    t_start = time.time()
    if n_workers > 1 and len(tasks) > 1:
        with mp.Pool(processes=min(n_workers, len(tasks))) as pool:
            results = pool.map(eval_match_worker, tasks)
    else:
        results = [eval_match_worker(t) for t in tasks]
    elapsed = round(time.time() - t_start, 2)

    # Process and aggregate per baseline and tier
    matches: List[Dict[str, Any]] = []
    stats_per_baseline: Dict[str, Dict[str, Any]] = {}
    stats_per_tier: Dict[str, Dict[str, Any]] = {
        "Tier 1 (Heuristic)": {"wins": 0, "losses": 0, "ties": 0, "score_sum": 0.0, "opp_score_sum": 0.0, "matches": 0},
        "Tier 2 (Hybrid)": {"wins": 0, "losses": 0, "ties": 0, "score_sum": 0.0, "opp_score_sum": 0.0, "matches": 0},
        "Tier 3 (Neural)": {"wins": 0, "losses": 0, "ties": 0, "score_sum": 0.0, "opp_score_sum": 0.0, "matches": 0},
    }

    for b_path in baseline_paths:
        b_label = short_name(b_path)
        stats_per_baseline[b_label] = {
            "baseline": b_label,
            "baseline_path": b_path,
            "tier": classify_agent_tier(b_label).name,
            "games": 0,
            "agent_wins": 0,
            "agent_losses": 0,
            "ties": 0,
            "agent_score_sum": 0.0,
            "base_score_sum": 0.0,
            "delta_sum": 0.0,
        }

    for res in results:
        b_label = short_name(res["task"]["agent_b"])
        b_tier = classify_agent_tier(b_label)
        tier_key = (
            "Tier 1 (Heuristic)" if b_tier == BaselineTier.TIER_1_HEURISTIC else
            ("Tier 2 (Hybrid)" if b_tier == BaselineTier.TIER_2_HYBRID else "Tier 3 (Neural)")
        )

        st = stats_per_baseline[b_label]
        st["games"] += 1
        st["agent_score_sum"] += res["score_a"]
        st["base_score_sum"] += res["score_b"]
        st["delta_sum"] += res["delta_a"]

        t_st = stats_per_tier[tier_key]
        t_st["matches"] += 1
        t_st["score_sum"] += res["score_a"]
        t_st["opp_score_sum"] += res["score_b"]

        if res["winner"] == "A":
            st["agent_wins"] += 1
            t_st["wins"] += 1
        elif res["winner"] == "B":
            st["agent_losses"] += 1
            t_st["losses"] += 1
        else:
            st["ties"] += 1
            t_st["ties"] += 1

        matches.append(res)
        if not quiet:
            w_str = f"AGENT ({agent_label})" if res["winner"] == "A" else (f"BASE ({b_label})" if res["winner"] == "B" else "TIE")
            print(
                f"  Match {len(matches):>2}: vs {b_label:<25} | Seat: {res['a_seat']} | "
                f"Score: ${res['score_a']:>11,.0f} | Opp: ${res['score_b']:>11,.0f} | Δ: {res['delta_a']:>+11,.0f} | {w_str}"
            )

    total_wins = sum(s["agent_wins"] for s in stats_per_baseline.values())
    total_losses = sum(s["agent_losses"] for s in stats_per_baseline.values())
    total_ties = sum(s["ties"] for s in stats_per_baseline.values())
    total_games = len(matches)

    avg_score = sum(m["score_a"] for m in matches) / max(1, total_games)
    avg_opp = sum(m["score_b"] for m in matches) / max(1, total_games)
    avg_delta = sum(m["delta_a"] for m in matches) / max(1, total_games)
    win_pct = (100.0 * total_wins / total_games) if total_games > 0 else 0.0

    report = {
        "agent": agent_label,
        "agent_path": agent_path,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "total_games": total_games,
            "wins": total_wins,
            "losses": total_losses,
            "ties": total_ties,
            "win_rate_pct": win_pct,
            "avg_score": avg_score,
            "avg_opponent_score": avg_opp,
            "avg_delta": avg_delta,
            "elapsed_seconds": elapsed,
        },
        "per_baseline": stats_per_baseline,
        "per_tier": stats_per_tier,
        "matches": matches,
    }

    if not quiet:
        print("=" * 100)
        print(f"  BENCHMARK SUMMARY: {total_wins} Wins - {total_losses} Losses - {total_ties} Ties ({win_pct:.1f}% Win Rate)")
        print(f"  Average Score: ${avg_score:,.0f} | Avg Delta: {avg_delta:+,.0f} | Time: {elapsed:.1f}s")
        print("=" * 100)

    return report


CHAMPION_2033_NAME = "submission_2033"


def resolve_2033_champion(
    baselines_input: str = BASELINES_DIR,
    fallback_readonly: str = "/Volumes/BASELINES/muzero/baselines",
) -> str:
    """Locate the benchmark 2033 ELO champion archive."""
    candidates = [
        os.path.join(baselines_input, f"{CHAMPION_2033_NAME}.zip"),
        os.path.join(baselines_input, f"{CHAMPION_2033_NAME}.tar.gz"),
        os.path.join(fallback_readonly, f"{CHAMPION_2033_NAME}.zip"),
        os.path.join(fallback_readonly, f"{CHAMPION_2033_NAME}.tar.gz"),
    ]
    if os.path.isfile(baselines_input) and "2033" in os.path.basename(baselines_input):
        return os.path.abspath(baselines_input)
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    # Fall back to directory scan
    for root in (baselines_input, fallback_readonly):
        if not os.path.isdir(root):
            continue
        for fn in os.listdir(root):
            if "2033" in fn and (fn.endswith(".zip") or fn.endswith(".tar.gz")):
                return os.path.abspath(os.path.join(root, fn))
    raise FileNotFoundError(
        f"2033 champion baseline not found under {baselines_input} or {fallback_readonly}"
    )


def gate_candidate_vs_2033(
    agent_path: str,
    *,
    baselines_dir: str = BASELINES_DIR,
    games: int = 4,
    base_seed: int = 20330,
    n_workers: int = 4,
    require_win_rate: float = 0.5,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Automated promotion gate: head-to-head vs the 2033 ELO champion.

    Returns a report with ``passed`` True iff the candidate win rate against
    submission_2033 is >= ``require_win_rate``.
    """
    champion = resolve_2033_champion(baselines_dir)
    if not quiet:
        print(f"[gate] Candidate vs 2033 champion: {short_name(champion)}")
    report = run_baseline_tournament(
        agent_path=agent_path,
        baseline_paths=[champion],
        games_per_baseline=games,
        base_seed=base_seed,
        n_workers=n_workers,
        quiet=quiet,
    )
    win_rate = float(report["summary"]["win_rate_pct"]) / 100.0
    passed = win_rate >= float(require_win_rate)
    report["gate"] = {
        "champion": short_name(champion),
        "champion_path": champion,
        "require_win_rate": require_win_rate,
        "win_rate": win_rate,
        "passed": passed,
    }
    if not quiet:
        status = "PASS" if passed else "FAIL"
        print(
            f"[gate] {status}: win_rate={win_rate:.1%} "
            f"(threshold={require_win_rate:.0%}) vs {short_name(champion)}"
        )
    return report
