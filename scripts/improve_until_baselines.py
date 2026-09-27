#!/usr/bin/env python3
"""Improvement cycle: train MuZero on expert replays → graft onto hybrid_direct → league.

Exit 0 on SUCCESS: dist wins more games than it loses against every baseline zip.

Note: full replay-constant distillation into the hybrid chassis (package_submission)
is *not* used here — that path regressed vs baselines. Expert signal goes into
Phase-3 BC / the MuZero gate instead.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(HERE, "artifacts")
STATUS = os.path.join(ART, "loop_status.json")
SEED_FILE = os.path.join(ART, "loop_seed.txt")
BASELINES = os.path.join(HERE, "baselines")
DIST_MAIN = os.path.join(HERE, "dist", "main.py")
TOURNAMENT_SAVE = os.path.join(ART, "tournament_dist_vs_baselines.json")


def run(cmd: list[str], cwd: str | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=cwd or HERE)


def next_seed(base: int = 9900) -> int:
    seed = base
    if os.path.isfile(SEED_FILE):
        try:
            seed = int(open(SEED_FILE).read().strip()) + 17
        except Exception:
            seed = base
    open(SEED_FILE, "w").write(str(seed))
    return seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--games-per-baseline", type=int, default=6)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--phase1-steps", type=int, default=150)
    parser.add_argument("--phase2-steps", type=int, default=40)
    parser.add_argument("--phase3-epochs", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=1, help="Number of self-improvement iterations (default: 1)")
    parser.add_argument("--continuous", action="store_true", help="Run self-improving loop indefinitely")
    parser.add_argument("--schedule", choices=["adaptive", "curriculum", "fixed"], default="adaptive")
    args = parser.parse_args()

    # If running multiple iterations or continuous mode, delegate to self_improving_loop
    if args.continuous or args.iterations > 1:
        from scripts.self_improving_loop import main as loop_main
        sys.argv = [
            sys.argv[0],
            "--iterations", str(args.iterations),
            "--games-per-baseline", str(args.games_per_baseline),
            "--phase1-steps", str(args.phase1_steps),
            "--phase2-steps", str(args.phase2_steps),
            "--phase3-epochs", str(args.phase3_epochs),
            "--schedule", args.schedule,
        ]
        if args.continuous:
            sys.argv.append("--continuous")
        if args.seed is not None:
            sys.argv.extend(["--base-seed", str(args.seed)])
        return loop_main()

    os.makedirs(ART, exist_ok=True)
    t0 = time.time()
    py = sys.executable
    seed = args.seed if args.seed is not None else next_seed()
    train_seed = (seed * 10007) % 2147483647

    if not args.skip_train:
        run(
            [
                py,
                "train_muzero.py",
                "--phase1-steps",
                str(args.phase1_steps),
                "--phase2-steps",
                str(args.phase2_steps),
                "--phase3-epochs",
                str(args.phase3_epochs),
                "--batch",
                "64",
                "--sims",
                "16",
                "--consistency-weight",
                "0.5",
                "--seed",
                str(train_seed),
            ]
        )

    run([py, "scripts/build_muzero_on_hybrid.py", "--with-weights"])

    # tournament_dist exits 1 when overall wins < losses; still write SUCCESS ourselves.
    cmd = [
        py,
        "scripts/tournament_dist_vs_baselines.py",
        "--dist",
        DIST_MAIN,
        "--baselines",
        BASELINES,
        "--games-per-baseline",
        str(args.games_per_baseline),
        "--seed",
        str(seed),
        "--save",
        TOURNAMENT_SAVE,
        "--save-replays",
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.call(cmd, cwd=HERE)

    report = json.load(open(TOURNAMENT_SAVE))
    per = report["per_baseline"]
    failures = []
    for name, stats in per.items():
        wins = int(stats["dist_wins"])
        losses = int(stats["dist_losses"])
        if wins <= losses:
            failures.append(
                {
                    "baseline": name,
                    "wins": wins,
                    "losses": losses,
                    "ties": int(stats.get("ties", 0)),
                    "avg_delta": stats.get("avg_delta"),
                }
            )

    success = len(failures) == 0 and len(per) >= 1
    payload = {
        "success": success,
        "seed": seed,
        "elapsed_seconds": round(time.time() - t0, 2),
        "overall": report["overall"],
        "per_baseline": per,
        "failures": failures,
        "note": (
            "SUCCESS requires wins > losses vs every baselines/*.zip; "
            "agent = hybrid_direct + MuZero gate trained on replays/*.json winners"
        ),
    }
    json.dump(payload, open(STATUS, "w"), indent=2)

    print("=" * 72)
    print(f"  seed={seed} SUCCESS={success} failures={len(failures)}")
    for name, s in per.items():
        print(
            f"  vs {name:<28} {s['dist_wins']}-{s['dist_losses']}-{s.get('ties', 0)}  "
            f"Δ=${s.get('avg_delta', 0):+,.0f}"
        )
    print("=" * 72)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
