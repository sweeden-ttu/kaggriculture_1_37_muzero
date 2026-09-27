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
candidates_vs_baselines.py - Master Cross-Evaluation Tournament
=============================================================================
Places official Kaggle-scored baselines (from /Volumes/BASELINES/muzero/baselines)
against the league of operational candidates (from /Volumes/TRAINREPLAYBOOST/muzero/baselines),
evaluates competitive head-to-head performance across alternating player seats,
determines the qualifying winners, and promotes/saves them to /Users/sweeden/muzero/baselines.
=============================================================================
"""
from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing as mp
import os
import shutil
import sys
import tarfile
import time
import zipfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .harness import eval_match_worker, play_single_game, short_name
from .tiers import BaselineTier, classify_agent_tier

# Canonical workspace defaults
DEFAULT_BASELINES_DIR = "/Volumes/BASELINES/muzero/baselines"
DEFAULT_CANDIDATES_DIR = "/Volumes/TRAINREPLAYBOOST/muzero/baselines"
DEFAULT_OUTPUT_DIR = "/Users/sweeden/muzero/baselines"
DEFAULT_REPORT_PATH = "/Users/sweeden/muzero/artifacts/tournament_candidates_vs_baselines.json"
DEFAULT_REPLAY_DIR = "/Users/sweeden/muzero/replays"

KNOWN_OFFICIAL_BASELINES = {
    "submission_2033",
    "submission_2028",
    "submission_2025",
    "submission_2021",
    "submission_0771",
}


def _is_valid_agent_archive(path: str) -> bool:
    """Verifies that an archive or python script contains an executable agent."""
    if not os.path.isfile(path):
        return False
    if path.endswith(".zip"):
        try:
            with zipfile.ZipFile(path, "r") as zf:
                names = zf.namelist()
            return any(n == "main.py" or n.endswith("/main.py") for n in names)
        except zipfile.BadZipFile:
            return False
    if path.endswith((".tar.gz", ".tgz", ".tar")):
        try:
            with tarfile.open(path, "r:*") as tf:
                names = tf.getnames()
            return any(n == "main.py" or n.endswith("/main.py") for n in names)
        except Exception:
            return False
    if path.endswith(".py"):
        return True
    return False


def discover_baselines(baselines_dir: str = DEFAULT_BASELINES_DIR, include_main: bool = False) -> List[str]:
    """Finds all authoritative reference baselines in the specified baseline directory."""
    if not os.path.isdir(baselines_dir):
        raise FileNotFoundError(f"Baselines directory not found: {baselines_dir}")

    found: List[str] = []
    for fn in sorted(os.listdir(baselines_dir)):
        if fn.startswith("."):
            continue
        p = os.path.abspath(os.path.join(baselines_dir, fn))
        if fn.endswith((".zip", ".tar.gz", ".tgz")):
            if _is_valid_agent_archive(p):
                found.append(p)
        elif include_main and fn == "main.py":
            found.append(p)

    if not found:
        raise RuntimeError(f"No valid baseline packages discovered in {baselines_dir}")
    return found


def discover_candidates(candidates_dir: str = DEFAULT_CANDIDATES_DIR, pattern: Optional[str] = None) -> List[str]:
    """Finds candidate packages in the candidate directory, isolating candidate submissions."""
    if not os.path.isdir(candidates_dir):
        raise FileNotFoundError(f"Candidates directory not found: {candidates_dir}")

    all_archives: List[str] = []
    cand_archives: List[str] = []

    for fn in sorted(os.listdir(candidates_dir)):
        if fn.startswith("."):
            continue
        p = os.path.abspath(os.path.join(candidates_dir, fn))
        if fn.endswith((".zip", ".tar.gz", ".tgz")) and _is_valid_agent_archive(p):
            all_archives.append(p)
            s_name = short_name(p)
            # Match candidate patterns (e.g. cand, boost, train)
            if pattern:
                if pattern.lower() in fn.lower():
                    cand_archives.append(p)
            else:
                if "cand" in fn.lower() or s_name not in KNOWN_OFFICIAL_BASELINES:
                    cand_archives.append(p)

    # If specific candidates were filtered, return them; otherwise return all valid archives
    res = cand_archives if cand_archives else all_archives
    if not res:
        raise RuntimeError(f"No valid candidate packages discovered in {candidates_dir}")
    return res


def build_bipartite_match_tasks(
    candidates: Sequence[str],
    baselines: Sequence[str],
    games_per_matchup: int = 2,
    base_seed: int = 10101,
    episode_steps: int = 720,
    save_replay_dir: Optional[str] = None,
    min_replay_score: float = 80000.0,
) -> List[Dict[str, Any]]:
    """Builds balanced, seat-alternating match tasks for candidates vs baselines."""
    tasks: List[Dict[str, Any]] = []
    task_idx = 0
    for cand in candidates:
        for base in baselines:
            for g in range(games_per_matchup):
                task_idx += 1
                seed = base_seed + task_idx
                # Alternating seats: even index -> cand=P0, base=P1; odd index -> base=P0, cand=P1
                cand_is_p0 = (g % 2 == 0)
                tasks.append({
                    "task_idx": task_idx,
                    "candidate_path": cand,
                    "baseline_path": base,
                    "agent_a": cand if cand_is_p0 else base,
                    "agent_b": base if cand_is_p0 else cand,
                    "seed": seed,
                    "a_is_p0": True,
                    "cand_is_p0": cand_is_p0,
                    "episode_steps": episode_steps,
                    "save_replay_dir": save_replay_dir,
                    "min_replay_score": min_replay_score,
                })
    return tasks


def run_candidates_vs_baselines_tournament(
    candidates: Sequence[str],
    baselines: Sequence[str],
    *,
    games_per_matchup: int = 2,
    base_seed: int = 10101,
    episode_steps: int = 720,
    n_workers: int = 4,
    save_replay_dir: Optional[str] = None,
    min_replay_score: float = 80000.0,
    save_criteria: str = "win-rate",
    min_win_rate: float = 0.50,
    top_k: Optional[int] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """
    Executes cross-tournament matches between all baselines and candidates,
    aggregating head-to-head metrics and ranking candidates.
    """
    total_matchups = len(candidates) * len(baselines)
    total_games = total_matchups * games_per_matchup
    t0 = time.time()

    if not quiet:
        print("=" * 110)
        print("  MASTER TOURNAMENT: OFFICIAL KAGGLE BASELINES VS OPERATIONAL CANDIDATES")
        print("=" * 110)
        print(f"  Reference Baselines ({len(baselines)}): {', '.join(short_name(b) for b in baselines)}")
        print(f"  Candidate League    ({len(candidates)}): {', '.join(short_name(c) for c in candidates)}")
        print(f"  Games per Matchup:   {games_per_matchup} (Balanced Alternating Seats P0 <-> P1)")
        print(f"  Total Games:         {total_games} across {total_matchups} pairings")
        print(f"  Parallel Workers:    {n_workers}")
        print(f"  Base Seed:           {base_seed}")
        print(f"  Qualification:       save_criteria={save_criteria} (min_win_rate={min_win_rate*100:.1f}%, top_k={top_k})")
        print("-" * 110)

    tasks = build_bipartite_match_tasks(
        candidates=candidates,
        baselines=baselines,
        games_per_matchup=games_per_matchup,
        base_seed=base_seed,
        episode_steps=episode_steps,
        save_replay_dir=save_replay_dir,
        min_replay_score=min_replay_score,
    )

    # Execute matches
    if n_workers > 1 and len(tasks) > 1:
        pool_size = min(n_workers, len(tasks), max(1, os.cpu_count() or 4))
        with mp.Pool(processes=pool_size) as pool:
            match_results = pool.map(eval_match_worker, tasks)
    else:
        match_results = [eval_match_worker(t) for t in tasks]

    elapsed = round(time.time() - t0, 2)

    # Initialize candidate statistics
    candidate_stats: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        c_name = short_name(c)
        candidate_stats[c] = {
            "path": c,
            "name": c_name,
            "tier": classify_agent_tier(c_name).name,
            "games": 0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "points": 0.0,
            "win_rate_pct": 0.0,
            "points_pct": 0.0,
            "score_sum": 0.0,
            "opp_score_sum": 0.0,
            "avg_score": 0.0,
            "opp_avg_score": 0.0,
            "net_margin": 0.0,
            "h2h_vs_baselines": {
                short_name(b): {"games": 0, "wins": 0, "losses": 0, "ties": 0, "score_sum": 0.0, "opp_score_sum": 0.0}
                for b in baselines
            },
        }

    # Initialize baseline defensive statistics
    baseline_stats: Dict[str, Dict[str, Any]] = {}
    for b in baselines:
        b_name = short_name(b)
        baseline_stats[b] = {
            "path": b,
            "name": b_name,
            "tier": classify_agent_tier(b_name).name,
            "games": 0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "points": 0.0,
            "win_rate_pct": 0.0,
            "score_sum": 0.0,
            "opp_score_sum": 0.0,
            "avg_score": 0.0,
            "opp_avg_score": 0.0,
        }

    # Process match logs
    match_logs: List[Dict[str, Any]] = []
    for res in match_results:
        task = res["task"]
        c_path = task["candidate_path"]
        b_path = task["baseline_path"]
        c_name = short_name(c_path)
        b_name = short_name(b_path)
        cand_is_p0 = task["cand_is_p0"]

        cand_score = res["score_a"] if cand_is_p0 else res["score_b"]
        base_score = res["score_b"] if cand_is_p0 else res["score_a"]

        if cand_score > base_score:
            outcome = "CANDIDATE_WIN"
        elif base_score > cand_score:
            outcome = "BASELINE_WIN"
        else:
            outcome = "TIE"

        # Update candidate
        cs = candidate_stats[c_path]
        cs["games"] += 1
        cs["score_sum"] += cand_score
        cs["opp_score_sum"] += base_score

        # Update baseline
        bs = baseline_stats[b_path]
        bs["games"] += 1
        bs["score_sum"] += base_score
        bs["opp_score_sum"] += cand_score

        # Update H2H subrecord
        h2h = cs["h2h_vs_baselines"][b_name]
        h2h["games"] += 1
        h2h["score_sum"] += cand_score
        h2h["opp_score_sum"] += base_score

        if outcome == "CANDIDATE_WIN":
            cs["wins"] += 1
            cs["points"] += 1.0
            bs["losses"] += 1
            h2h["wins"] += 1
        elif outcome == "BASELINE_WIN":
            cs["losses"] += 1
            bs["wins"] += 1
            bs["points"] += 1.0
            h2h["losses"] += 1
        else:
            cs["ties"] += 1
            cs["points"] += 0.5
            bs["ties"] += 1
            bs["points"] += 0.5
            h2h["ties"] += 1

        match_logs.append({
            "task_idx": task["task_idx"],
            "seed": res["seed"],
            "candidate": c_name,
            "baseline": b_name,
            "candidate_seat": "P0" if cand_is_p0 else "P1",
            "candidate_score": cand_score,
            "baseline_score": base_score,
            "delta": cand_score - base_score,
            "outcome": outcome,
            "saved_replay": res.get("saved_replay"),
            "elapsed_sec": res.get("elapsed_sec", 0.0),
        })

    # Finalize candidate aggregates
    for cs in candidate_stats.values():
        g = max(1, cs["games"])
        cs["win_rate_pct"] = round((cs["wins"] / g) * 100.0, 2)
        cs["points_pct"] = round((cs["points"] / g) * 100.0, 2)
        cs["avg_score"] = round(cs["score_sum"] / g, 2)
        cs["opp_avg_score"] = round(cs["opp_score_sum"] / g, 2)
        cs["net_margin"] = round(cs["avg_score"] - cs["opp_avg_score"], 2)

    # Finalize baseline aggregates
    for bs in baseline_stats.values():
        g = max(1, bs["games"])
        bs["win_rate_pct"] = round((bs["wins"] / g) * 100.0, 2)
        bs["avg_score"] = round(bs["score_sum"] / g, 2)
        bs["opp_avg_score"] = round(bs["opp_score_sum"] / g, 2)

    # Rank candidates by points (descending), then avg_score (descending)
    ranked_candidates = sorted(
        candidate_stats.values(),
        key=lambda x: (x["points"], x["avg_score"], x["net_margin"]),
        reverse=True,
    )

    for rank, entry in enumerate(ranked_candidates, 1):
        entry["rank"] = rank

        # Determine winner qualification based on policy
        is_winner = False
        if save_criteria == "win-rate":
            is_winner = (entry["win_rate_pct"] >= (min_win_rate * 100.0))
        elif save_criteria == "positive-margin":
            is_winner = (entry["net_margin"] > 0.0)
        elif save_criteria == "all-positive":
            is_winner = (entry["win_rate_pct"] >= (min_win_rate * 100.0) and entry["net_margin"] > 0.0)
        elif save_criteria == "top-k":
            limit = top_k if top_k is not None else 4
            is_winner = (rank <= limit)
        elif save_criteria == "all":
            is_winner = True
        else:
            is_winner = (entry["win_rate_pct"] >= (min_win_rate * 100.0))

        entry["is_winner"] = is_winner
        entry["qualification_status"] = "WINNER" if is_winner else "REJECTED"

    ranked_baselines = sorted(
        baseline_stats.values(),
        key=lambda x: (x["points"], x["avg_score"]),
        reverse=True,
    )

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {
            "baselines_dir": os.path.abspath(DEFAULT_BASELINES_DIR if not baselines else os.path.dirname(baselines[0])),
            "candidates_dir": os.path.abspath(DEFAULT_CANDIDATES_DIR if not candidates else os.path.dirname(candidates[0])),
            "games_per_matchup": games_per_matchup,
            "base_seed": base_seed,
            "n_workers": n_workers,
            "episode_steps": episode_steps,
            "save_criteria": save_criteria,
            "min_win_rate": min_win_rate,
            "top_k": top_k,
        },
        "summary": {
            "total_candidates": len(candidates),
            "total_baselines": len(baselines),
            "total_matchups": total_matchups,
            "total_games": total_games,
            "elapsed_seconds": elapsed,
            "qualifying_winners_count": sum(1 for c in ranked_candidates if c["is_winner"]),
        },
        "candidates": ranked_candidates,
        "baselines": ranked_baselines,
        "matches": match_logs,
    }


def save_tournament_winners(
    report: Dict[str, Any],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Copies all qualifying winning candidate archives to the specified target directory."""
    os.makedirs(output_dir, exist_ok=True)
    saved: List[Dict[str, Any]] = []

    for cand in report["candidates"]:
        if not cand.get("is_winner", False):
            continue
        src = cand["path"]
        fn = os.path.basename(src)
        dst = os.path.join(output_dir, fn)

        record = {
            "name": cand["name"],
            "rank": cand["rank"],
            "source": src,
            "destination": dst,
            "win_rate_pct": cand["win_rate_pct"],
            "avg_score": cand["avg_score"],
            "net_margin": cand["net_margin"],
            "dry_run": dry_run,
        }

        if not dry_run:
            shutil.copy2(src, dst)
            cand["saved_path"] = dst
            cand["saved_status"] = "SAVED"
        else:
            cand["saved_path"] = dst
            cand["saved_status"] = "DRY_RUN_QUALIFIED"

        saved.append(record)

    return saved


def print_tournament_summary(report: Dict[str, Any], saved_records: Sequence[Dict[str, Any]]) -> None:
    """Prints a clear, informative summary table of candidates, baselines, and promoted winners."""
    print("\n" + "=" * 115)
    print("  TOURNAMENT CANDIDATE LEADERBOARD vs OFFICIAL BASELINES")
    print("=" * 115)
    header = f"{'Rank':<5} {'Candidate':<28} {'Record (W-L-T)':<16} {'Win %':<8} {'Avg Score':<15} {'Base Avg':<15} {'Net Margin':<14} {'Status'}"
    print(header)
    print("-" * 115)

    for c in report["candidates"]:
        rec = f"{c['wins']}-{c['losses']}-{c['ties']}"
        win_pct = f"{c['win_rate_pct']:.1f}%"
        avg_s = f"${c['avg_score']:>11,.2f}"
        base_s = f"${c['opp_avg_score']:>11,.2f}"
        margin_sign = "+" if c["net_margin"] >= 0 else ""
        margin = f"{margin_sign}${c['net_margin']:>9,.2f}"
        status = "[WINNER -> SAVED]" if c["is_winner"] else "[REJECTED]"
        print(f"{c['rank']:<5} {c['name']:<28} {rec:<16} {win_pct:<8} {avg_s:<15} {base_s:<15} {margin:<14} {status}")

    print("=" * 115)

    print("\n" + "=" * 80)
    print("  OFFICIAL BASELINES DEFENSIVE PERFORMANCE")
    print("=" * 80)
    print(f"{'Baseline':<26} {'Tier':<18} {'Record vs Cand':<16} {'Win %':<8} {'Avg Score'}")
    print("-" * 80)
    for b in report["baselines"]:
        rec = f"{b['wins']}-{b['losses']}-{b['ties']}"
        win_pct = f"{b['win_rate_pct']:.1f}%"
        avg_s = f"${b['avg_score']:>10,.2f}"
        print(f"{b['name']:<26} {b['tier']:<18} {rec:<16} {win_pct:<8} {avg_s}")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("  PROMOTION AUDIT: CANDIDATES SAVED TO BASELINES DIRECTORY")
    print("=" * 80)
    if not saved_records:
        print("  [Notice] No candidates satisfied the qualification threshold.")
    else:
        for s in saved_records:
            prefix = "[DRY-RUN]" if s.get("dry_run") else "[PROMOTED]"
            print(f"  {prefix} #{s['rank']} {s['name']} (Win%: {s['win_rate_pct']}%, Margin: ${s['net_margin']:+,.2f})")
            print(f"            Destination: {s['destination']}")
    print("=" * 80 + "\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Master Tournament: Official Baselines vs League of Candidates with Winner Promotion"
    )
    parser.add_argument(
        "--baselines-dir",
        default=DEFAULT_BASELINES_DIR,
        help=f"Directory containing authoritative Kaggle baselines (default: {DEFAULT_BASELINES_DIR})",
    )
    parser.add_argument(
        "--candidates-dir",
        default=DEFAULT_CANDIDATES_DIR,
        help=f"Directory containing league of candidate archives (default: {DEFAULT_CANDIDATES_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination directory to save qualifying winners (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--games-per-matchup",
        type=int,
        default=2,
        help="Number of games per candidate-baseline pairing with alternating seats (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10101,
        help="Starting seed for reproducible deterministic matches (default: 10101)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 4),
        help="Number of parallel subprocess evaluation workers (default: 4)",
    )
    parser.add_argument(
        "--save-criteria",
        choices=["win-rate", "positive-margin", "all-positive", "top-k", "all"],
        default="win-rate",
        help="Criteria to qualify a candidate as a winner for promotion (default: win-rate)",
    )
    parser.add_argument(
        "--min-win-rate",
        type=float,
        default=0.50,
        help="Minimum win rate vs baselines (0.0 to 1.0) required to qualify under win-rate policy (default: 0.50)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Limit number of winners to top K ranked candidates when using top-k criteria",
    )
    parser.add_argument(
        "--include-main",
        action="store_true",
        help="Include standalone main.py in baselines directory if present",
    )
    parser.add_argument(
        "--candidate-pattern",
        default=None,
        help="Substring filter for candidate files in candidates-dir (e.g. 'cand')",
    )
    parser.add_argument(
        "--save-replays",
        action="store_true",
        help="Harvest self-play replays >= $80k to replays directory",
    )
    parser.add_argument(
        "--replay-dir",
        default=DEFAULT_REPLAY_DIR,
        help=f"Directory to harvest high-scoring replays (default: {DEFAULT_REPLAY_DIR})",
    )
    parser.add_argument(
        "--report",
        default=DEFAULT_REPORT_PATH,
        help=f"JSON file path to save full tournament telemetry (default: {DEFAULT_REPORT_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute tournament and compute qualifications without copying files to output-dir",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    try:
        baselines = discover_baselines(args.baselines_dir, include_main=args.include_main)
        candidates = discover_candidates(args.candidates_dir, pattern=args.candidate_pattern)
    except Exception as e:
        print(f"[Error] Discovery failed: {e}", file=sys.stderr)
        return 2

    replay_dir = args.replay_dir if args.save_replays else None

    report = run_candidates_vs_baselines_tournament(
        candidates=candidates,
        baselines=baselines,
        games_per_matchup=args.games_per_matchup,
        base_seed=args.seed,
        n_workers=args.workers,
        save_replay_dir=replay_dir,
        save_criteria=args.save_criteria,
        min_win_rate=args.min_win_rate,
        top_k=args.top_k,
    )

    saved_records = save_tournament_winners(
        report=report,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )

    print_tournament_summary(report, saved_records)

    if args.report:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
            report["saved_winners"] = saved_records
            with open(args.report, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"[Report Saved] -> {args.report}")
        except Exception as e:
            print(f"[Warning] Failed to write report to {args.report}: {e}", file=sys.stderr)

    winners_count = report["summary"]["qualifying_winners_count"]
    return 0 if winners_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
