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

"""Unified CLI entrypoint for tournament evaluation and benchmarking."""
from __future__ import annotations

import argparse
import json
import os
import sys

from .benchmarks import find_baselines, run_baseline_tournament
from .harness import play_single_game, short_name
from .league import (
    DEFAULT_LEAGUE_STATE,
    LeagueTracker,
    PFSPLeagueConfig,
    demo_sample_opponents,
    discover_agent_packages,
    register_packages,
    run_pfsp_league,
    run_round_robin_league,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINES_DIR = os.path.join(HERE, "baselines")
DEFAULT_DIST = os.path.join(HERE, "dist", "main.py")
DEFAULT_SAVE = os.path.join(HERE, "artifacts", "tournament_report.json")
DEFAULT_LEAGUE_SAVE = os.path.join(HERE, "artifacts", "league_report.json")
DEFAULT_PFSP_SAVE = os.path.join(HERE, "artifacts", "league_pfsp_report.json")
LEAGUE_COMMANDS = ("benchmark", "h2h", "league", "league-pfsp", "league-sample", "league-status")


def main() -> int:
    parser = argparse.ArgumentParser(description="Kaggriculture Tournament & Benchmark Suite")
    subparsers = parser.add_subparsers(dest="command", help="Evaluation mode")

    # 1. Benchmark vs Baselines
    p_base = subparsers.add_parser("benchmark", help="Benchmark an agent against baseline tiers")
    p_base.add_argument("--agent", "--dist", default=DEFAULT_DIST, help="Agent archive (.zip) or script (.py)")
    p_base.add_argument("--baselines", default=BASELINES_DIR, help="Directory containing baseline zips or single zip")
    p_base.add_argument("--games-per-baseline", type=int, default=4, help="Games per baseline (seat alternating)")
    p_base.add_argument("--seed", type=int, default=10101, help="Starting seed")
    p_base.add_argument("--save", default=DEFAULT_SAVE, help="Output JSON path")
    p_base.add_argument("--save-replays", action="store_true", help="Save match replays >= $80k")
    p_base.add_argument("--workers", type=int, default=4, help="Parallel evaluation workers")

    # 2. Head-to-Head Match
    p_h2h = subparsers.add_parser("h2h", help="Head-to-head match between two agents")
    p_h2h.add_argument("--agent-a", required=True, help="Agent A path (.zip or .py)")
    p_h2h.add_argument("--agent-b", required=True, help="Agent B path (.zip or .py)")
    p_h2h.add_argument("--games", type=int, default=10, help="Number of games (alternates P0/P1)")
    p_h2h.add_argument("--seed", type=int, default=10101, help="Starting seed")
    p_h2h.add_argument("--save-replays", action="store_true", help="Save match replays >= $80k")
    p_h2h.add_argument("--workers", type=int, default=4, help="Parallel evaluation workers")

    # 3. Round-Robin League
    p_league = subparsers.add_parser("league", help="All-vs-all round-robin league over packages")
    p_league.add_argument("--dirs", nargs="+", default=["dist", "baselines"], help="Directories to scan for agent zips")
    p_league.add_argument("--games-per-pair", type=int, default=2, help="Games per pair (seat alternating)")
    p_league.add_argument("--seed", type=int, default=10101, help="Starting seed")
    p_league.add_argument("--save", default=DEFAULT_LEAGUE_SAVE, help="Output JSON path")
    p_league.add_argument("--save-replays", action="store_true", help="Save match replays >= $80k")
    p_league.add_argument(
        "--tracker-state",
        default=DEFAULT_LEAGUE_STATE,
        help="Persist PFSP win matrices alongside round-robin results",
    )
    p_league.add_argument("--no-tracker", action="store_true", help="Skip PFSP tracker persistence")

    # 4. PFSP League (docs/League_Tracker.md)
    p_pfsp = subparsers.add_parser(
        "league-pfsp",
        help="Prioritized Fictitious Self-Play league (hard/var sampling + roles)",
    )
    p_pfsp.add_argument("--dirs", nargs="+", default=["dist", "baselines"], help="Directories to scan for agent zips")
    p_pfsp.add_argument("--active", default=None, help="Active agent zip/path (default: first discovered / current main)")
    p_pfsp.add_argument(
        "--role",
        choices=["main", "main_exploiter", "league_exploiter"],
        default="main",
        help="AlphaStar league role for matchmaking ratios",
    )
    p_pfsp.add_argument("--games", type=int, default=8, help="Number of PFSP-sampled matches to play")
    p_pfsp.add_argument("--seed", type=int, default=20201, help="Starting seed")
    p_pfsp.add_argument("--save", default=DEFAULT_PFSP_SAVE, help="Output JSON report path")
    p_pfsp.add_argument("--tracker-state", default=DEFAULT_LEAGUE_STATE, help="LeagueTracker JSON state path")
    p_pfsp.add_argument("--save-replays", action="store_true", help="Save match replays >= $80k")

    # 5. PFSP dry-run sampling (no env games)
    p_sample = subparsers.add_parser(
        "league-sample",
        help="Sample PFSP matchmaking decisions without playing games",
    )
    p_sample.add_argument("--dirs", nargs="+", default=["dist", "baselines"])
    p_sample.add_argument("--active", default=None)
    p_sample.add_argument("--role", choices=["main", "main_exploiter", "league_exploiter"], default="main")
    p_sample.add_argument("--n", type=int, default=40, help="Number of sample draws")
    p_sample.add_argument("--seed", type=int, default=0)
    p_sample.add_argument("--tracker-state", default=DEFAULT_LEAGUE_STATE)

    # 6. League tracker status
    p_status = subparsers.add_parser("league-status", help="Print saved LeagueTracker leaderboard")
    p_status.add_argument("--tracker-state", default=DEFAULT_LEAGUE_STATE)

    # Compatibility: allow running without subcommand (defaults to benchmark)
    if len(sys.argv) > 1 and sys.argv[1] not in (*LEAGUE_COMMANDS, "-h", "--help"):
        # Interpret as benchmark arguments
        args, rest = parser.parse_known_args(["benchmark"] + sys.argv[1:])
    else:
        args = parser.parse_args()

    if args.command in (None, "benchmark"):
        agent_path = getattr(args, "agent", DEFAULT_DIST)
        baselines_arg = getattr(args, "baselines", BASELINES_DIR)
        baseline_files = find_baselines(baselines_arg)
        report = run_baseline_tournament(
            agent_path=agent_path,
            baseline_paths=baseline_files,
            games_per_baseline=getattr(args, "games_per_baseline", 4),
            base_seed=getattr(args, "seed", 10101),
            save_replay_dir=os.path.join(HERE, "replays") if getattr(args, "save_replays", False) else None,
            n_workers=getattr(args, "workers", 4),
        )
        save_path = getattr(args, "save", DEFAULT_SAVE)
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"[Saved report] → {save_path}")
        return 0 if report["summary"]["win_rate_pct"] >= 50.0 else 1

    if args.command == "h2h":
        print(f"\nHead-to-Head: {short_name(args.agent_a)} vs {short_name(args.agent_b)} ({args.games} games)")
        wins_a, wins_b, ties = 0, 0, 0
        score_a_sum, score_b_sum = 0.0, 0.0
        workers = getattr(args, "workers", 4)
        if workers > 1:
            import multiprocessing as mp
            from .harness import eval_match_worker
            tasks = [
                {
                    "agent_a": args.agent_a,
                    "agent_b": args.agent_b,
                    "seed": args.seed + i + 1,
                    "a_is_p0": (i % 2 == 0),
                    "save_replay_dir": os.path.join(HERE, "replays") if args.save_replays else None,
                    "min_replay_score": 80000.0,
                }
                for i in range(args.games)
            ]
            with mp.Pool(processes=min(workers, args.games)) as pool:
                results = pool.map(eval_match_worker, tasks)
        else:
            results = []
            for i in range(args.games):
                a_is_p0 = (i % 2 == 0)
                seed = args.seed + i + 1
                res = play_single_game(
                    args.agent_a,
                    args.agent_b,
                    seed=seed,
                    a_is_p0=a_is_p0,
                    save_replay_dir=os.path.join(HERE, "replays") if args.save_replays else None,
                )
                results.append(res)

        for i, res in enumerate(results):
            score_a_sum += res["score_a"]
            score_b_sum += res["score_b"]
            if res["winner"] == "A":
                wins_a += 1
            elif res["winner"] == "B":
                wins_b += 1
            else:
                ties += 1
            w_str = short_name(args.agent_a) if res["winner"] == "A" else (short_name(args.agent_b) if res["winner"] == "B" else "TIE")
            print(f"  Game {i+1:>2}: {res['a_seat']} | {short_name(args.agent_a)}: ${res['score_a']:>10,.0f} | {short_name(args.agent_b)}: ${res['score_b']:>10,.0f} | Winner: {w_str}")
        print("-" * 75)
        print(f"Result: {short_name(args.agent_a)} {wins_a}W - {wins_b}L - {ties}T | Avg Score: ${score_a_sum/args.games:,.0f} vs ${score_b_sum/args.games:,.0f}")
        return 0

    if args.command == "league":
        agents = discover_agent_packages(args.dirs)
        if len(agents) < 2:
            print(f"Need at least 2 agent zip packages in {args.dirs}, found {len(agents)}")
            return 1
        report = run_round_robin_league(
            agents,
            games_per_pair=args.games_per_pair,
            base_seed=args.seed,
            save_replay_dir=os.path.join(HERE, "replays") if args.save_replays else None,
            tracker_state_path=None if args.no_tracker else args.tracker_state,
        )
        if args.save:
            os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
            with open(args.save, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"[Saved league report] → {args.save}")
        return 0

    if args.command == "league-pfsp":
        agents = discover_agent_packages(args.dirs)
        if len(agents) < 1:
            print(f"Need at least 1 agent zip in {args.dirs}, found {len(agents)}")
            return 1
        active = args.active or agents[0]
        if args.active is None and len(agents) < 2:
            print("PFSP league works best with ≥2 agents in --dirs (self-play only otherwise)")
        cfg = PFSPLeagueConfig(
            games=args.games,
            role=args.role,
            active_id=active,
            base_seed=args.seed,
            save_replay_dir=os.path.join(HERE, "replays") if args.save_replays else None,
            state_path=args.tracker_state,
        )
        report = run_pfsp_league(agents, config=cfg)
        if args.save:
            os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
            with open(args.save, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"[Saved PFSP report] → {args.save}")
        return 0

    if args.command == "league-sample":
        agents = discover_agent_packages(args.dirs)
        tracker = LeagueTracker.load(args.tracker_state)
        if agents:
            register_packages(tracker, agents, main_id=args.active or agents[0])
        if len(tracker) < 1:
            print("No checkpoints in tracker and no agent zips found.")
            return 1
        active = os.path.abspath(args.active) if args.active else (tracker.current_main or tracker.checkpoints[0])
        demo = demo_sample_opponents(
            tracker, active, n=args.n, role=args.role, seed=args.seed
        )
        print(json.dumps(demo, indent=2))
        return 0

    if args.command == "league-status":
        tracker = LeagueTracker.load(args.tracker_state)
        if len(tracker) == 0:
            print(f"Empty LeagueTracker at {args.tracker_state}")
            print("Run: make league-pfsp   or   make league-rr")
            return 0
        print(f"LeagueTracker ({len(tracker)} checkpoints) ← {args.tracker_state}")
        print(f"Current main: {short_name(tracker.current_main or '')}")
        print(f"{'Rank':>4} | {'Name':<28} | {'Pts':>6} | {'Games':>6} | Role")
        for row in tracker.leaderboard():
            flag = " *" if row["is_main"] else ""
            print(
                f"{row['rank']:>4} | {row['name']:<28} | {row['points']:>6.1f} | "
                f"{row['games']:>6.0f} | {row['role']}{flag}"
            )
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
