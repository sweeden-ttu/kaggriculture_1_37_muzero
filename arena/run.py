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

"""CLI: paired-seed matches between a candidate and one or more opponents.

    python -m arena.run --a dist/candidate.py --b baselines/champion_main_20260927_051846.py \
        --seeds 1-16 --workers 4 --attribution --save artifacts/arena/candidate_vs_champion.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from arena.harness import paired_tasks, play_many, save_results, spec_name  # noqa: E402
from arena.gate import paired_summary, format_summary  # noqa: E402
from arena.attribution import format_attribution  # noqa: E402


def parse_seeds(text: str):
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def parse_spec(text: str, overrides_json: str | None):
    if overrides_json:
        return {"path": text, "overrides": json.loads(overrides_json)}
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description="Kaggriculture paired-seed arena")
    ap.add_argument("--a", required=True, help="candidate agent path (or pass/random/starter)")
    ap.add_argument("--b", required=True, nargs="+", help="one or more opponent agent paths")
    ap.add_argument("--a-overrides", default=None, help="JSON dict of module attribute overrides for A")
    ap.add_argument("--b-overrides", default=None, help="JSON dict of module attribute overrides for B")
    ap.add_argument("--seeds", default="1-8", help="e.g. 1-16 or 3,7,11")
    ap.add_argument("--one-seat", action="store_true", help="only play A as P0")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--attribution", action="store_true", help="attribute revenue per product (slower)")
    ap.add_argument("--fixed-draw", action="store_true", help="decouple the town shop draw from farm state (common random numbers)")
    ap.add_argument("--save", default=None, help="JSON output path")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    a = parse_spec(args.a, args.a_overrides)
    all_results = []
    for b_path in args.b:
        b = parse_spec(b_path, args.b_overrides)
        tasks = paired_tasks(a, b, seeds, both_seats=not args.one_seat, attribution=args.attribution,
                             fixed_draw=args.fixed_draw)
        results = play_many(tasks, workers=args.workers, progress=not args.quiet)
        all_results.extend(results)
        s = paired_summary(results)
        print(format_summary(spec_name(a), spec_name(b), s))
        if args.attribution:
            # aggregate mean revenue per product for A and B
            import collections
            agg = {"A": collections.defaultdict(float), "B": collections.defaultdict(float)}
            n = 0
            for r in results:
                if r.error or not r.attribution:
                    continue
                n += 1
                for side in ("A", "B"):
                    for k, v in r.attribution[side]["sells"].items():
                        agg[side][k] += v["dollars"]
            if n:
                for side in ("A", "B"):
                    items = ", ".join(f"{k} ${v / n:,.0f}" for k, v in sorted(agg[side].items(), key=lambda kv: -kv[1]))
                    print(f"   mean revenue {side}: {items}")
        for r in results:
            if r.error:
                print(f"   ERROR seed {r.seed}: {r.error.splitlines()[0]}")
    if args.save:
        save_results(all_results, args.save)
        print(f"saved {len(all_results)} results to {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
