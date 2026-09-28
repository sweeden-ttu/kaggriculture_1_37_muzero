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

"""One-dial-at-a-time ablation of a configuration against an opponent.

    python -m arena.ablate --agent baselines/champion.py --log artifacts/arena/sweep_pool.jsonl \
        --trial 37 --opponent baselines/champion.py --seeds 21-28 --fixed-draw

Plays the full configuration, then each dial alone (others at their defaults),
on the same paired seeds, and prints the paired delta per dial so a candidate
can carry only the dials that earn their place.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from arena.harness import paired_tasks, play_many  # noqa: E402
from arena.gate import paired_summary, format_summary  # noqa: E402
from arena.run import parse_seeds  # noqa: E402
from arena.sweep import load_log  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-dial ablation")
    ap.add_argument("--agent", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--trial", type=int, required=True)
    ap.add_argument("--opponent", required=True)
    ap.add_argument("--seeds", default="21-28")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--fixed-draw", action="store_true")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    rows = load_log(args.log)
    cfg = next(r["config"] for r in rows if r["trial"] == args.trial)
    seeds = parse_seeds(args.seeds)
    report = []
    variants = [("full", cfg)] + [(k, {k: v}) for k, v in cfg.items()]
    for name, sub in variants:
        spec = {"path": args.agent, "overrides": sub, "name": name}
        results = play_many(paired_tasks(spec, args.opponent, seeds, fixed_draw=args.fixed_draw),
                            workers=args.workers, progress=False)
        s = paired_summary(results)
        print(format_summary(f"{name}={sub.get(name, '')}" if name != "full" else "full", "opp", s), flush=True)
        report.append({"name": name, "overrides": sub, "summary": s})
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
