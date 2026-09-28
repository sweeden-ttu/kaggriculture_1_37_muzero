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

"""Population-based dial search against an opponent pool (random search, then CEM).

    python -m arena.sweep --agent dist/candidate_species_ev.py --dials arena/dials_species.json \
        --opponents baselines/champion_main_20260927_051846.py --seeds 11-26 --trials 200 \
        --log artifacts/arena/sweep_species.jsonl

Each trial samples one configuration of module-global overrides, plays paired
games (both seats, common seeds) against every opponent, and appends a JSON
line with the configuration and the paired summary. After ``--warmup`` random
trials, new configurations are sampled around the elite (cross-entropy style).
The log is append-only, so a sweep can be resumed or extended at any time.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from arena.harness import paired_tasks, play_many  # noqa: E402
from arena.gate import paired_summary  # noqa: E402
from arena.run import parse_seeds  # noqa: E402


def sample_dial(spec: Dict[str, Any], rng: random.Random, centre: Any = None, spread: float = 1.0) -> Any:
    kind = spec.get("type", "float")
    if kind == "choice":
        choices = spec["choices"]
        if centre is not None and rng.random() < 0.7:
            return centre
        return rng.choice(choices)
    lo, hi = float(spec["low"]), float(spec["high"])
    log = bool(spec.get("log", False))
    if centre is None:
        if log:
            x = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        else:
            x = rng.uniform(lo, hi)
    else:
        c = float(centre)
        if log:
            width = (math.log(hi) - math.log(lo)) * 0.25 * spread
            x = math.exp(min(math.log(hi), max(math.log(lo), rng.gauss(math.log(max(c, lo)), width))))
        else:
            width = (hi - lo) * 0.25 * spread
            x = min(hi, max(lo, rng.gauss(c, width)))
    if kind == "int":
        return int(round(x))
    return round(x, 4)


def sample_config(dials: Dict[str, Dict[str, Any]], rng: random.Random, elite: List[Dict[str, Any]] | None,
                  spread: float) -> Dict[str, Any]:
    cfg = {}
    parent = rng.choice(elite) if elite else None
    for name, spec in dials.items():
        centre = parent["config"].get(name) if parent else None
        cfg[name] = sample_dial(spec, rng, centre, spread)
    return cfg


def load_log(path: str) -> List[Dict[str, Any]]:
    rows = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return rows


def score_of(row: Dict[str, Any]) -> float:
    """Objective: mean paired money delta, lightly penalised for losses."""
    s = row["summary"]
    return s["mean_delta"] + 2000.0 * (s["win_rate"] - 0.5)


def main() -> int:
    ap = argparse.ArgumentParser(description="Dial sweep against an opponent pool")
    ap.add_argument("--agent", required=True)
    ap.add_argument("--dials", required=True, help="JSON file: {name: {type, low, high, log|choices}}")
    ap.add_argument("--opponents", nargs="+", required=True)
    ap.add_argument("--seeds", default="11-18")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=30, help="random trials before CEM sampling")
    ap.add_argument("--elite", type=int, default=8)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--log", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline", action="store_true", help="first trial uses the unmodified agent")
    ap.add_argument("--fixed-draw", action="store_true", help="decouple the town shop draw from farm state")
    args = ap.parse_args()

    with open(args.dials, "r", encoding="utf-8") as f:
        dials = json.load(f)
    seeds = parse_seeds(args.seeds)
    rng = random.Random(args.seed + int(time.time()) % 100000)
    rows = load_log(args.log)
    os.makedirs(os.path.dirname(os.path.abspath(args.log)) or ".", exist_ok=True)
    print(f"resuming with {len(rows)} logged trials")
    done = len(rows)
    while done < args.trials:
        if args.baseline and not rows:
            cfg: Dict[str, Any] = {}
        else:
            ranked = sorted(rows, key=score_of, reverse=True)
            elite = ranked[: args.elite] if len(rows) >= args.warmup else None
            spread = max(0.3, 1.0 - 0.5 * (len(rows) - args.warmup) / max(1, args.trials - args.warmup)) if elite else 1.0
            cfg = sample_config(dials, rng, elite, spread)
        spec = {"path": args.agent, "overrides": cfg, "name": f"trial{done}"}
        t0 = time.time()
        results = []
        for opp in args.opponents:
            results.extend(play_many(paired_tasks(spec, opp, seeds, fixed_draw=args.fixed_draw),
                                     workers=args.workers, progress=False))
        summary = paired_summary(results)
        row = {"trial": done, "config": cfg, "summary": summary, "elapsed": round(time.time() - t0, 1),
               "opponents": [os.path.basename(o) for o in args.opponents], "seeds": args.seeds}
        with open(args.log, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        rows.append(row)
        best = max(rows, key=score_of)
        print(f"trial {done:4d} delta {summary['mean_delta']:+8,.0f} win {summary['win_rate']:.0%} "
              f"({summary['games']} games, {row['elapsed']}s) cfg={cfg}  | best trial {best['trial']} "
              f"delta {best['summary']['mean_delta']:+,.0f} win {best['summary']['win_rate']:.0%}", flush=True)
        done += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
