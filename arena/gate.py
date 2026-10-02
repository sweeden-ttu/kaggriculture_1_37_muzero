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

"""Paired-seed statistics for candidate-vs-opponent gating."""
from __future__ import annotations

import math
from typing import Any, Dict, Sequence

from .harness import GameResult


def wilson_interval(wins: int, n: int, z: float = 1.96):
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / denom, (centre + half) / denom)


def paired_summary(results: Sequence[GameResult]) -> Dict[str, Any]:
    ok = [r for r in results if not r.error]
    errors = [r for r in results if r.error]
    n = len(ok)
    wins = sum(1 for r in ok if r.winner == "A")
    losses = sum(1 for r in ok if r.winner == "B")
    ties = n - wins - losses
    deltas = [r.delta for r in ok]
    mean_delta = sum(deltas) / n if n else 0.0
    sd = math.sqrt(sum((d - mean_delta) ** 2 for d in deltas) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    lo, hi = wilson_interval(wins, n)
    return {
        "games": n,
        "errors": len(errors),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / n if n else 0.0,
        "win_rate_ci95": (lo, hi),
        "mean_score_a": sum(r.score_a for r in ok) / n if n else 0.0,
        "mean_score_b": sum(r.score_b for r in ok) / n if n else 0.0,
        "mean_delta": mean_delta,
        "delta_se": se,
        "delta_t": mean_delta / se if se > 0 else 0.0,
        "min_delta": min(deltas) if deltas else 0.0,
        "max_delta": max(deltas) if deltas else 0.0,
    }


def format_summary(name_a: str, name_b: str, s: Dict[str, Any]) -> str:
    lo, hi = s["win_rate_ci95"]
    return (f"{name_a} vs {name_b}: {s['wins']}W-{s['losses']}L-{s['ties']}T of {s['games']} "
            f"(win {s['win_rate']:.0%}, CI95 {lo:.0%}-{hi:.0%})  "
            f"A ${s['mean_score_a']:,.0f}  B ${s['mean_score_b']:,.0f}  "
            f"delta ${s['mean_delta']:+,.0f} +/- {s['delta_se']:,.0f} (t={s['delta_t']:.1f}, "
            f"range {s['min_delta']:+,.0f}..{s['max_delta']:+,.0f})"
            + (f"  errors={s['errors']}" if s['errors'] else ""))
