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

"""Held-out validation of the best sweep configurations.

    python -m arena.validate --agent baselines/champion_main_20260927_051846.py \
        --log artifacts/arena/sweep_pool.jsonl --top 3 --seeds 21-28 --fixed-draw \
        --opponents baselines/champion_main_20260927_051846.py dist/pool/champ_route12.py

Re-plays the top configurations (by the sweep objective) on seeds the sweep
never saw, against the given opponents, both seats, and prints the paired
summary per configuration next to the unmodified agent.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from arena.harness import paired_tasks, play_many, spec_name  # noqa: E402
from arena.gate import paired_summary, format_summary  # noqa: E402
from arena.run import parse_seeds  # noqa: E402
from arena.sweep import load_log, score_of  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate sweep configurations on held-out seeds")
    ap.add_argument("--agent", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--opponents", nargs="+", required=True)
    ap.add_argument("--seeds", default="21-28")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--fixed-draw", action="store_true")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    rows = load_log(args.log)
    ranked = sorted((r for r in rows if r.get("config")), key=score_of, reverse=True)[: args.top]
    seeds = parse_seeds(args.seeds)
    configs = [("baseline", {})] + [(f"trial{r['trial']}", r["config"]) for r in ranked]
    report = []
    for name, cfg in configs:
        spec = {"path": args.agent, "overrides": cfg, "name": name}
        results = []
        per_opponent = {}
        for opp in args.opponents:
            opp_results = play_many(paired_tasks(spec, opp, seeds, fixed_draw=args.fixed_draw),
                                    workers=args.workers, progress=False)
            results.extend(opp_results)
            per_opponent[spec_name(opp)] = paired_summary(opp_results)
        s = paired_summary(results)
        print(format_summary(name, "pool", s))
        for opp_name, os_ in per_opponent.items():
            print("   " + format_summary(name, opp_name, os_))
        report.append({"name": name, "config": cfg, "summary": s, "per_opponent": per_opponent})
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
