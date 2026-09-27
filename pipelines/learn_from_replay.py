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

"""Learn policy constants and distillation parameters from Kaggle episode JSONs."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

LAND_ORDER = ("NE", "SW", "SE")
LAND_PRICES = {"NE": 1000.0, "SW": 2000.0, "SE": 4000.0}
LAND_BUFFERS = {"NE": 500.0, "SW": 500.0, "SE": 800.0}
PREMIUM = ("STRAWBERRY", "MILK", "TOMATO", "MELON", "WOOL", "EGG")
NON_SEED = (
    "WHEAT", "MELON", "CARROT", "TOMATO", "STRAWBERRY",
    "EGG", "EGGS", "MILK", "WOOL", "FERTILIZER",
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPLAYS_DIR = os.path.join(HERE, "replays")
DEFAULT_ARTIFACTS_DIR = "/Volumes/BASELINES/muzero/artifacts"
DEFAULT_ENGINE_BASE = os.path.join(
    DEFAULT_ARTIFACTS_DIR, "engine_policy_start_1.32.7.json"
)
REPLAY_LESSONS_PATH = os.path.join(HERE, "hybrid_chassis", "layers", "replay_lessons.py")
CANONICAL_REPLAY_RE = re.compile(r"^[0-9]+\.json$")

# Keys written into _REPLAY_POLICY (embed target). Meta/debug fields are omitted.
_EMBED_KEYS = (
    "sw_day_min",
    "sw_money_floor",
    "skip_se",
    "tomato_seed_start_day",
    "tomato_seed_end_day",
    "tomato_seed_total_target",
    "tomato_seed_price",
    "carrot_post_ne",
    "carrot_post_ne_qty",
    "carrot_seed_price",
    "carrot_post_ne_day_end",
    "carrot_late_start_day",
    "carrot_late_end_day",
    "carrot_late_target",
    "melon_wave2_day",
    "melon_wave2_qty",
    "melon_seed_price",
    "geese_target",
    "early_hire_target",
    "early_cow_qty",
    "early_sheep_qty",
    "neml_step0",
    "fert_trap",
    "premium_first_on_shed_cap",
    "premium_first_items",
    "liq_hours",
    "premium_items",
    "hire_unlocked_min",
    "hire_day_end",
    "hire_target",
    "shed_force_sell_at",
    "shed_target_after",
    "clamp_sell_to_shed",
)

_DEFAULT_CONSENSUS = {
    "carrot_late_start_day": 20,
    "carrot_late_end_day": 26,
    "carrot_late_target": 100,
    "melon_wave2_day": 10,
    "melon_wave2_qty": 4,
    "melon_seed_price": 100,
    "geese_target": 2,
    "early_hire_target": 5,
    "early_cow_qty": 2,
    "early_sheep_qty": 2,
    "neml_step0": True,
    "fert_trap": True,
}


def _farm(obs: Dict[str, Any], seat: int) -> Dict[str, Any]:
    farms = obs.get("farms") or []
    return farms[seat] if seat < len(farms) else {}


def distill_replay(path: str, seat: int = 1) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        episode = json.load(f)
    steps: Sequence[Any] = episode.get("steps") or []
    if not steps:
        raise ValueError(f"No steps in {path}")

    expansions: List[Dict[str, Any]] = []
    tomato_seed_buys: List[Dict[str, Any]] = []
    carrot_seed_buys: List[Dict[str, Any]] = []
    carrot_post_ne: List[Dict[str, Any]] = []
    carrot_late: List[Dict[str, Any]] = []
    melon_seed_buys: List[Dict[str, Any]] = []
    geese_buys = 0
    hire_by_day: Dict[int, int] = defaultdict(int)
    sell_qty = Counter()
    sell_liq_qty = Counter()
    shed_peaks: List[int] = []
    premium_first_dumps: List[Dict[str, Any]] = []
    prev_u: Optional[List[str]] = None
    ne_unlocked = False
    first_sw_affordable: Optional[Dict[str, Any]] = None

    for t, step in enumerate(steps):
        st = step[seat]
        obs = st["observation"]
        farm = _farm(obs, seat)
        day = int(obs.get("day", t // 24))
        hour = int(obs.get("hour", t % 24))
        unlocked = list(farm.get("unlocked_quadrants") or ["NW"])
        money = float(farm.get("money", 0.0))
        hands = len(farm.get("hands") or [])
        hire_by_day[day] = max(hire_by_day[day], hands)

        shed = (obs.get("private") or {}).get("shed") or {}
        non_seed = sum(int(shed.get(k, 0) or 0) for k in NON_SEED)
        if hour == 0:
            shed_peaks.append(non_seed)

        if prev_u is not None and unlocked != prev_u:
            added = [q for q in unlocked if q not in prev_u]
            for q in added:
                expansions.append({
                    "land": q,
                    "step": t,
                    "day": day,
                    "hour": hour,
                    "money": money,
                    "hands": hands,
                })
                if q == "NE":
                    ne_unlocked = True
        prev_u = unlocked
        if "NE" in unlocked:
            ne_unlocked = True

        if (
            first_sw_affordable is None
            and "NE" in unlocked
            and "SW" not in unlocked
            and money >= LAND_PRICES["SW"] + LAND_BUFFERS["SW"]
        ):
            first_sw_affordable = {
                "step": t, "day": day, "hour": hour, "money": money,
            }

        market_orders = list((st.get("action") or {}).get("market") or [])
        if hour in (1, 2) and market_orders:
            prev_shed = {}
            prev_nons = non_seed
            if t > 0:
                prev_obs = steps[t - 1][seat]["observation"]
                prev_shed = (prev_obs.get("private") or {}).get("shed") or {}
                prev_nons = sum(int(prev_shed.get(k, 0) or 0) for k in NON_SEED)
            first_sell = next(
                (o for o in market_orders if o and o[0] == "SELL" and len(o) >= 3),
                None,
            )
            if first_sell and first_sell[1] in PREMIUM and prev_nons >= 85:
                stock = int(prev_shed.get(first_sell[1], 0) or 0)
                qty = int(first_sell[2])
                if stock > 0 and qty >= max(1, int(0.8 * stock)):
                    premium_first_dumps.append({
                        "step": t,
                        "day": day,
                        "hour": hour,
                        "item": first_sell[1],
                        "qty": qty,
                        "stock": stock,
                        "nons": prev_nons,
                    })

        for order in market_orders:
            if not order:
                continue
            op = order[0]
            if op == "BUY_SEED" and len(order) >= 2 and order[1] == "TOMATO":
                qty = int(order[2]) if len(order) > 2 else 1
                tomato_seed_buys.append({
                    "step": t, "day": day, "hour": hour, "qty": qty, "money": money,
                })
            if op == "BUY_SEED" and len(order) >= 2 and order[1] == "MELON":
                qty = int(order[2]) if len(order) > 2 else 1
                melon_seed_buys.append({
                    "step": t, "day": day, "hour": hour, "qty": qty, "money": money,
                })
            if op == "BUY_SEED" and len(order) >= 2 and order[1] == "CARROT":
                qty = int(order[2]) if len(order) > 2 else 1
                entry = {
                    "step": t, "day": day, "hour": hour, "qty": qty, "money": money,
                }
                carrot_seed_buys.append(entry)
                if ne_unlocked and "SW" not in unlocked:
                    carrot_post_ne.append(entry)
                if day >= 20:
                    carrot_late.append(entry)
                if any(o and o[0] == "BUY_LAND" for o in market_orders):
                    if entry not in carrot_post_ne:
                        carrot_post_ne.append(entry)
            if op == "BUY_ANIMAL" and len(order) >= 2 and order[1] == "GOOSE":
                geese_buys += int(order[2]) if len(order) > 2 else 1
            if op == "SELL" and len(order) >= 3:
                item, qty = order[1], int(order[2])
                sell_qty[item] += qty
                if hour in (1, 2) or day >= 27:
                    sell_liq_qty[item] += qty

    sw = next((e for e in expansions if e["land"] == "SW"), None)
    ne = next((e for e in expansions if e["land"] == "NE"), None)

    liq_share = {}
    for item in PREMIUM:
        tot = sell_qty.get(item, 0)
        liq = sell_liq_qty.get(item, 0)
        liq_share[item] = round(liq / tot, 3) if tot else 0.0

    hire_peak_post_sw = max(
        (hire_by_day[d] for d in range(9, 15) if d in hire_by_day), default=10
    )

    tomato_start_day = tomato_seed_buys[0]["day"] if tomato_seed_buys else 10
    tomato_total = sum(b["qty"] for b in tomato_seed_buys)
    carrot_post_ne_qty = sum(b["qty"] for b in carrot_post_ne)
    carrot_burst = max((b["qty"] for b in carrot_post_ne), default=0)
    carrot_late_total = sum(b["qty"] for b in carrot_late)
    melon_total = sum(b["qty"] for b in melon_seed_buys)
    melon_late = sum(b["qty"] for b in melon_seed_buys if b["day"] >= 10)
    se = next((e for e in expansions if e["land"] == "SE"), None)

    sw_day = int(sw["day"]) if sw else 8
    sw_day_min = min(sw_day, int(first_sw_affordable["day"])) if first_sw_affordable is not None else sw_day

    basename = os.path.basename(path)
    rewards = episode.get("rewards") or [0.0, 0.0]
    has_carrot = bool(carrot_post_ne)
    carrot_qty = int(carrot_burst or carrot_post_ne_qty or 12) if has_carrot else 0
    tomato_target = max(4, min(30, int(tomato_total))) if tomato_total > 0 else 0

    policy = {
        "source_episode": basename,
        "expert_seat": seat,
        "final_rewards": rewards,
        "expansions": expansions,
        "sw_day_min": sw_day_min,
        "sw_money_floor": float(LAND_PRICES["SW"] + LAND_BUFFERS["SW"]),
        "ne_day_min": int(ne["day"]) if ne else 6,
        "skip_se": (se is None),
        "tomato_seed_start_day": int(tomato_start_day),
        "tomato_seed_end_day": 18,
        "tomato_seed_total_target": tomato_target,
        "tomato_seed_price": 50,
        "carrot_post_ne": has_carrot,
        "carrot_post_ne_qty": carrot_qty,
        "carrot_seed_price": 10,
        "carrot_post_ne_day_end": 12,
        "carrot_late_start_day": 20,
        "carrot_late_end_day": 26,
        "carrot_late_target": int(max(40, min(125, carrot_late_total))) if carrot_late_total else 100,
        "melon_wave2_day": 10,
        "melon_wave2_qty": int(max(3, min(6, melon_late))) if melon_late else 4,
        "melon_seed_price": 100,
        "melon_seed_total": int(melon_total),
        "geese_target": int(min(4, geese_buys)) if geese_buys else 2,
        "early_hire_target": 5,
        "early_cow_qty": 2,
        "early_sheep_qty": 2,
        "neml_step0": True,
        "fert_trap": True,
        "premium_first_on_shed_cap": True,
        "premium_first_items": list(PREMIUM),
        "liq_hours": [1, 2],
        "liq_share_expert": liq_share,
        "premium_items": list(PREMIUM),
        "hire_unlocked_min": 2,
        "hire_day_end": 14,
        "hire_target": int(max(11, hire_peak_post_sw)),
        "shed_force_sell_at": 85,
        "shed_target_after": 75,
        "clamp_sell_to_shed": True,
        "first_sw_affordable": first_sw_affordable,
        "premium_first_dumps": premium_first_dumps[:8],
        "carrot_post_ne_buys": carrot_post_ne[:12],
    }
    return policy


def _expert_score(policy: Dict[str, Any]) -> float:
    rewards = policy.get("final_rewards") or [0.0, 0.0]
    seat = int(policy.get("expert_seat", 0))
    if seat < len(rewards):
        return float(rewards[seat] or 0.0)
    return float(max(rewards) if rewards else 0.0)


def _median_int(values: Sequence[float], default: int) -> int:
    if not values:
        return default
    return int(round(statistics.median(values)))


def _median_float(values: Sequence[float], default: float) -> float:
    if not values:
        return default
    return float(statistics.median(values))


def _mode_bool(values: Sequence[bool], default: bool) -> bool:
    if not values:
        return default
    return Counter(bool(v) for v in values).most_common(1)[0][0]


def _mode_list(values: Sequence[Any], default: List[Any]) -> List[Any]:
    if not values:
        return list(default)
    serialized = [json.dumps(v) for v in values]
    best = Counter(serialized).most_common(1)[0][0]
    return json.loads(best)


def _load_engine_base(engine_base_path: Optional[str]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Load engine_policy_start_*.json replay_policy_start dials as merge base."""
    base = dict(_DEFAULT_CONSENSUS)
    meta: Dict[str, Any] = {
        "engine_base_path": None,
        "engine_version": None,
        "engine_source_sha256": None,
    }
    if not engine_base_path or not os.path.isfile(engine_base_path):
        return base, meta
    with open(engine_base_path, "r", encoding="utf-8") as f:
        engine = json.load(f)
    start = engine.get("replay_policy_start") or {}
    for key, value in start.items():
        if key in _EMBED_KEYS or key in base or key.endswith("_orders"):
            base[key] = value
    meta["engine_base_path"] = os.path.abspath(engine_base_path)
    meta["engine_version"] = engine.get("engine_version")
    meta["engine_source_sha256"] = engine.get("source_sha256")
    meta["engine_policy_id"] = engine.get("policy_id")
    return base, meta


def merge_expert_policies(
    policy_paths: Sequence[str],
    min_score: float = 80000.0,
    engine_base_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Median/mode consensus over embed keys, layered on engine policy start."""
    loaded: List[Dict[str, Any]] = []
    for path in policy_paths:
        with open(path, "r", encoding="utf-8") as f:
            policy = json.load(f)
        if _expert_score(policy) < min_score:
            continue
        loaded.append(policy)

    if not loaded:
        raise ValueError(
            f"No policies with expert score >= {min_score} among {len(policy_paths)} files"
        )

    loaded.sort(key=_expert_score, reverse=True)
    consensus, engine_meta = _load_engine_base(engine_base_path)
    consensus.update(engine_meta)
    consensus["merge_layers"] = ["engine_policy_start", "gold_replay_consensus"]
    consensus["n_policies"] = len(loaded)
    consensus["min_score"] = float(min_score)
    consensus["source_episodes"] = [p.get("source_episode") for p in loaded]
    consensus["scores"] = [_expert_score(p) for p in loaded]

    def _d(key: str, fallback: Any) -> Any:
        return consensus.get(key, fallback)

    # Gold-replay overlay (median/mode) on top of engine base dials.
    consensus["sw_day_min"] = _median_int(
        [p.get("sw_day_min", _d("sw_day_min", 8)) for p in loaded],
        int(_d("sw_day_min", 8)),
    )
    consensus["sw_money_floor"] = _median_float(
        [float(p.get("sw_money_floor", _d("sw_money_floor", 2500.0))) for p in loaded],
        float(_d("sw_money_floor", 2500.0)),
    )
    consensus["skip_se"] = _mode_bool(
        [bool(p.get("skip_se", _d("skip_se", True))) for p in loaded],
        bool(_d("skip_se", True)),
    )
    consensus["tomato_seed_start_day"] = _median_int(
        [p.get("tomato_seed_start_day", _d("tomato_seed_start_day", 10)) for p in loaded],
        int(_d("tomato_seed_start_day", 10)),
    )
    consensus["tomato_seed_end_day"] = _median_int(
        [p.get("tomato_seed_end_day", _d("tomato_seed_end_day", 18)) for p in loaded],
        int(_d("tomato_seed_end_day", 18)),
    )
    consensus["tomato_seed_total_target"] = _median_int(
        [p.get("tomato_seed_total_target", _d("tomato_seed_total_target", 0)) for p in loaded],
        int(_d("tomato_seed_total_target", 0)),
    )
    consensus["tomato_seed_price"] = int(_d("tomato_seed_price", 50))
    consensus["carrot_post_ne"] = _mode_bool(
        [bool(p.get("carrot_post_ne", _d("carrot_post_ne", True))) for p in loaded],
        bool(_d("carrot_post_ne", True)),
    )
    consensus["carrot_post_ne_qty"] = _median_int(
        [p.get("carrot_post_ne_qty", _d("carrot_post_ne_qty", 12)) for p in loaded],
        int(_d("carrot_post_ne_qty", 12)),
    )
    consensus["carrot_seed_price"] = int(_d("carrot_seed_price", 10))
    consensus["carrot_post_ne_day_end"] = _median_int(
        [p.get("carrot_post_ne_day_end", _d("carrot_post_ne_day_end", 12)) for p in loaded],
        int(_d("carrot_post_ne_day_end", 12)),
    )
    consensus["carrot_late_start_day"] = _median_int(
        [p.get("carrot_late_start_day", _d("carrot_late_start_day", 20)) for p in loaded],
        int(_d("carrot_late_start_day", 20)),
    )
    consensus["carrot_late_end_day"] = _median_int(
        [p.get("carrot_late_end_day", _d("carrot_late_end_day", 26)) for p in loaded],
        int(_d("carrot_late_end_day", 26)),
    )
    consensus["carrot_late_target"] = _median_int(
        [p.get("carrot_late_target", _d("carrot_late_target", 100)) for p in loaded],
        int(_d("carrot_late_target", 100)),
    )
    consensus["melon_wave2_day"] = _median_int(
        [p.get("melon_wave2_day", _d("melon_wave2_day", 10)) for p in loaded],
        int(_d("melon_wave2_day", 10)),
    )
    consensus["melon_wave2_qty"] = _median_int(
        [p.get("melon_wave2_qty", _d("melon_wave2_qty", 4)) for p in loaded],
        int(_d("melon_wave2_qty", 4)),
    )
    consensus["melon_seed_price"] = int(_d("melon_seed_price", 100))
    consensus["geese_target"] = _median_int(
        [p.get("geese_target", _d("geese_target", 2)) for p in loaded],
        int(_d("geese_target", 2)),
    )
    consensus["early_hire_target"] = _median_int(
        [p.get("early_hire_target", _d("early_hire_target", 5)) for p in loaded],
        int(_d("early_hire_target", 5)),
    )
    consensus["early_cow_qty"] = _median_int(
        [p.get("early_cow_qty", _d("early_cow_qty", 2)) for p in loaded],
        int(_d("early_cow_qty", 2)),
    )
    consensus["early_sheep_qty"] = _median_int(
        [p.get("early_sheep_qty", _d("early_sheep_qty", 2)) for p in loaded],
        int(_d("early_sheep_qty", 2)),
    )
    # Keep engine-grounded behavioral flags unless replay policies say otherwise.
    consensus["neml_step0"] = bool(_d("neml_step0", True))
    consensus["fert_trap"] = bool(_d("fert_trap", True))
    consensus["premium_first_on_shed_cap"] = bool(_d("premium_first_on_shed_cap", True))
    consensus["premium_first_items"] = _mode_list(
        [p.get("premium_first_items") or _d("premium_first_items", list(PREMIUM)) for p in loaded],
        list(_d("premium_first_items", list(PREMIUM))),
    )
    consensus["liq_hours"] = _mode_list(
        [p.get("liq_hours") or _d("liq_hours", [1, 2]) for p in loaded],
        list(_d("liq_hours", [1, 2])),
    )
    consensus["premium_items"] = _mode_list(
        [p.get("premium_items") or _d("premium_items", list(PREMIUM)) for p in loaded],
        list(_d("premium_items", list(PREMIUM))),
    )
    consensus["hire_unlocked_min"] = _median_int(
        [p.get("hire_unlocked_min", _d("hire_unlocked_min", 2)) for p in loaded],
        int(_d("hire_unlocked_min", 2)),
    )
    consensus["hire_day_end"] = _median_int(
        [p.get("hire_day_end", _d("hire_day_end", 14)) for p in loaded],
        int(_d("hire_day_end", 14)),
    )
    consensus["hire_target"] = _median_int(
        [p.get("hire_target", _d("hire_target", 11)) for p in loaded],
        int(_d("hire_target", 11)),
    )
    consensus["shed_force_sell_at"] = _median_int(
        [p.get("shed_force_sell_at", _d("shed_force_sell_at", 85)) for p in loaded],
        int(_d("shed_force_sell_at", 85)),
    )
    consensus["shed_target_after"] = _median_int(
        [p.get("shed_target_after", _d("shed_target_after", 75)) for p in loaded],
        int(_d("shed_target_after", 75)),
    )
    consensus["clamp_sell_to_shed"] = bool(_d("clamp_sell_to_shed", True))
    if "neml_step0_orders" in consensus:
        # Preserve engine NeML opening sequence for embed consumers that read it.
        pass
    return consensus


def _format_policy_literal(policy: Dict[str, Any]) -> str:
    embed = {k: policy[k] for k in _EMBED_KEYS if k in policy}
    body = json.dumps(embed, indent=4)
    # json uses true/false/null; Python needs True/False/None
    body = body.replace("true", "True").replace("false", "False").replace("null", "None")
    return f"_REPLAY_POLICY = {body}\n"


def embed_policy_into_replay_lessons(
    consensus: Dict[str, Any],
    lessons_path: str = REPLAY_LESSONS_PATH,
) -> None:
    with open(lessons_path, "r", encoding="utf-8") as f:
        src = f.read()
    pattern = re.compile(
        r"_REPLAY_POLICY\s*=\s*\{.*?\n\}",
        re.DOTALL,
    )
    replacement = _format_policy_literal(consensus).rstrip("\n")
    new_src, n = pattern.subn(replacement, src, count=1)
    if n != 1:
        raise RuntimeError(f"Failed to locate _REPLAY_POLICY block in {lessons_path}")
    with open(lessons_path, "w", encoding="utf-8") as f:
        f.write(new_src)


def _list_replay_paths(canonical_only: bool) -> List[str]:
    paths = sorted(glob.glob(os.path.join(REPLAYS_DIR, "*.json")))
    if canonical_only:
        paths = [
            p for p in paths
            if CANONICAL_REPLAY_RE.match(os.path.basename(p))
        ]
    return paths


def _winner_seat(rewards: Sequence[Any], override: Optional[int]) -> int:
    if override is not None:
        return int(override)
    if len(rewards) >= 2:
        return 0 if float(rewards[0] or 0) >= float(rewards[1] or 0) else 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill Kaggle replay → policy JSON / embed consensus into replay_lessons"
    )
    parser.add_argument("--replay", default=None, help="Episode JSON path")
    parser.add_argument("--seat", type=int, default=None, help="Expert seat")
    parser.add_argument("--all", action="store_true", help="Distill all replays in replays/")
    parser.add_argument(
        "--canonical-only",
        action="store_true",
        default=True,
        help="Only distill canonical ^[0-9]+\\.json$ (default on with --all)",
    )
    parser.add_argument(
        "--include-selfplay",
        action="store_true",
        help="Include selfplay_*.json (disables --canonical-only)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=80000.0,
        help="Minimum expert seat score for --embed consensus (default 80000)",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=DEFAULT_ARTIFACTS_DIR,
        help=f"Output directory for ep*_policy.json (default {DEFAULT_ARTIFACTS_DIR})",
    )
    parser.add_argument(
        "--engine-base",
        default=DEFAULT_ENGINE_BASE,
        help=(
            "Engine policy start JSON to layer under gold-replay consensus "
            f"(default {DEFAULT_ENGINE_BASE})"
        ),
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Merge ep*_policy.json → consensus JSON and rewrite replay_lessons._REPLAY_POLICY",
    )
    parser.add_argument("--out", default=None, help="Output policy JSON (single replay)")
    args = parser.parse_args()

    artifacts_dir = os.path.abspath(args.artifacts_dir)
    os.makedirs(artifacts_dir, exist_ok=True)
    canonical_only = bool(args.canonical_only) and not bool(args.include_selfplay)
    engine_base = args.engine_base

    def _write_consensus(policy_paths: Sequence[str]) -> Dict[str, Any]:
        consensus = merge_expert_policies(
            policy_paths,
            min_score=args.min_score,
            engine_base_path=engine_base,
        )
        consensus_path = os.path.join(artifacts_dir, "replay_policy_consensus.json")
        with open(consensus_path, "w", encoding="utf-8") as f:
            json.dump(consensus, f, indent=2)
        eng_ver = consensus.get("engine_version") or "unknown"
        versioned = os.path.join(
            artifacts_dir, f"replay_policy_merged_{eng_ver}.json"
        )
        with open(versioned, "w", encoding="utf-8") as f:
            json.dump(consensus, f, indent=2)
        if args.embed:
            embed_policy_into_replay_lessons(consensus)
            print(
                f"Embedded consensus from {consensus['n_policies']} policies "
                f"(min_score={args.min_score}) → {REPLAY_LESSONS_PATH}"
            )
        print(f"Wrote {consensus_path}")
        print(f"Wrote {versioned}")
        return consensus

    if args.embed and not args.all and args.replay is None:
        # Embed-only path: merge existing policies from artifacts-dir
        policy_paths = sorted(glob.glob(os.path.join(artifacts_dir, "ep*_policy.json")))
        if not policy_paths:
            raise SystemExit(f"No ep*_policy.json under {artifacts_dir}; run --all first")
        _write_consensus(policy_paths)
        return

    if args.replay:
        target_files = [args.replay]
    elif args.all:
        target_files = _list_replay_paths(canonical_only=canonical_only)
    else:
        parser.error("Provide --replay, --all, or --embed")
        return

    if not target_files:
        print(f"No replays found in {REPLAYS_DIR} (canonical_only={canonical_only})")
        return

    written: List[str] = []
    for path in target_files:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        rewards = d.get("rewards") or [
            a.get("reward", 0) for a in (d.get("steps") or [[]])[-1]
        ]
        seat = _winner_seat(
            rewards,
            args.seat if args.seat is not None and len(target_files) == 1 else None,
        )
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = (
            args.out
            if (args.out and len(target_files) == 1)
            else os.path.join(artifacts_dir, f"ep{stem}_policy.json")
        )
        policy = distill_replay(path, seat=seat)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(policy, f, indent=2)
        written.append(out_path)
        score = float(rewards[seat]) if seat < len(rewards) else 0.0
        print(f"Wrote {out_path} (seat {seat}, score {score:.0f})")

    # Always merge gold distill onto engine base after --all (embed optional).
    if args.all or args.embed:
        policy_paths = written or sorted(
            glob.glob(os.path.join(artifacts_dir, "ep*_policy.json"))
        )
        _write_consensus(policy_paths)


if __name__ == "__main__":
    main()
