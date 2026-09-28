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

"""Exact Kaggriculture market model (self-contained; safe to inline in a submission).

Price curves, shop demand and animal/crop production schedules are transcribed
from kaggle-environments 1.32.7 ``kaggriculture.py`` (Apache-2.0, Kaggle and the
kaggle-environments contributors). Only the pure functions are reproduced.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
CROPS = {
    "WHEAT":      {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}
ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP",    "first_yield_day": 4, "interval": 1, "max_held": 4, "product": "EGG"},
    "COW":   {"cost": 400, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
}
PRODUCT_OF = {a: d["product"] for a, d in ANIMALS.items()}
MARKET_I0 = 10000
PRICE_FLOOR = 1
HINGE_GAIN = 8.0
MARKET_PARAMS = {
    "WHEAT":      {"base":  25, "I0": MARKET_I0, "T": 400, "below_func": "sqrt",   "below_target": 0.80, "above_func": "log",    "above_target": 0.20},
    "CARROT":     {"base":  35, "I0": MARKET_I0, "T": 450, "below_func": "hinge",  "below_target": 1.00, "above_func": "sqrt",   "above_target": 0.70},
    "TOMATO":     {"base":  60, "I0": MARKET_I0, "T": 200, "below_func": "hinge",  "below_target": 0.40, "above_func": "sqrt",   "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "I0": MARKET_I0, "T": 100, "below_func": "sqrt",   "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":      {"base": 250, "I0": MARKET_I0, "T": 300, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.60},
    "EGG":        {"base":  50, "I0": MARKET_I0, "T": 332, "below_func": "hinge",  "below_target": 0.40, "above_func": "log",    "above_target": 0.20},
    "MILK":       {"base": 160, "I0": MARKET_I0, "T": 122, "below_func": "sqrt",   "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":       {"base": 200, "I0": MARKET_I0, "T": 105, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.20},
    "FERTILIZER": {"base": 100, "I0": MARKET_I0, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}
SHOPS = {
    "BAKERY":         ["EGG", "WHEAT"],
    "PIZZA_SHOP":     ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT":    ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE":     ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE":       ["CARROT"],
    "SMOOTHIE_SHOP":  ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}
TOWN_CENTER_PRODUCTS = [p for p in PRODUCTS if p != "FERTILIZER"]
MAX_SHOP_INSTANCES = 8
SHOP_UNLOCK_INTERVAL_DAYS = 3
SHOP_SELL_INTERVAL = 4
CENTER_SELL_INTERVAL = 24
TURNS_PER_DAY = 24
EPISODE_STEPS = 720


def _shape(func: str, x: float, T: Optional[float] = None) -> float:
    x = max(0.0, x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    if func == "log10":
        return math.log10(1.0 + x)
    if func == "hinge":
        if not T or T <= 0:
            return x
        u = x / T
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    return x


def market_price(item: str, inventory: float, params: Optional[Dict[str, Any]] = None) -> int:
    p = (params or MARKET_PARAMS)[item]
    base, I0, T = p["base"], p["I0"], p["T"]
    if inventory < I0:
        f = p["below_func"]
        amp = p["below_target"] * base / _shape(f, T, T)
        price = base + amp * _shape(f, I0 - inventory, T)
    else:
        f = p["above_func"]
        amp = p["above_target"] * base / _shape(f, T, T)
        price = base - amp * _shape(f, inventory - I0, T)
    return max(PRICE_FLOOR, int(round(price)))


def price_at_offset(item: str, offset: float) -> int:
    """Price when inventory = I0 + offset (offset > 0 is a glut)."""
    return market_price(item, MARKET_I0 + offset)


def glut_offset_for_price(item: str, min_price: int) -> int:
    """Largest inventory offset (>= 0) whose price is still >= ``min_price``."""
    lo, hi = 0, 5000
    if price_at_offset(item, 0) < min_price:
        return -1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if price_at_offset(item, mid) >= min_price:
            lo = mid
        else:
            hi = mid - 1
    return lo


# ---------------------------------------------------------------------------
# Town demand
# ---------------------------------------------------------------------------
def shop_tick_demand(shops: Iterable[str]) -> Dict[str, int]:
    """Units of each product consumed per 4-turn shop tick by the given shop instances."""
    out = {p: 0 for p in PRODUCTS}
    for name in shops:
        products = SHOPS.get(name)
        if not products:
            continue
        mult = 2 if len(products) == 1 else 1
        for p in products:
            out[p] += mult
    return out


EXPECTED_TICK_DEMAND_PER_UNLOCK: Dict[str, float] = {
    p: sum((2 if len(v) == 1 else 1) for v in SHOPS.values() if p in v) / len(SHOPS) for p in PRODUCTS
}


def unlock_days(known_count: int) -> List[int]:
    """Days on which future shop instances unlock, given ``known_count`` already unlocked."""
    days = []
    for k in range(known_count + 1, MAX_SHOP_INSTANCES + 1):
        days.append(k * SHOP_UNLOCK_INTERVAL_DAYS)
    return days


def consumption_by_step(shops: Sequence[str], step_from: int, step_to: int = EPISODE_STEPS,
                        expected_future: bool = True) -> Dict[str, List[float]]:
    """Per-product consumption at each step in [step_from, step_to).

    Deterministic part: current shop instances (tick every 4 steps) and the town
    center (every 24 steps). Expected part: each future unlock adds the mean
    per-tick demand over the eight shop types, from its unlock day onward.
    """
    n = max(0, step_to - step_from)
    out = {p: [0.0] * n for p in PRODUCTS}
    if n == 0:
        return out
    tick = shop_tick_demand(shops)
    future = unlock_days(len(shops)) if expected_future else []
    for i in range(n):
        s = step_from + i
        if s % SHOP_SELL_INTERVAL == 0:
            day = s // TURNS_PER_DAY
            n_future = sum(1 for d in future if day >= d)
            for p in PRODUCTS:
                out[p][i] += tick[p] + n_future * EXPECTED_TICK_DEMAND_PER_UNLOCK[p]
        if s % CENTER_SELL_INTERVAL == 0:
            for p in TOWN_CENTER_PRODUCTS:
                out[p][i] += 1.0
    return out


def cumulative_consumption(shops: Sequence[str], step_from: int, step_to: int = EPISODE_STEPS,
                           expected_future: bool = True) -> Dict[str, float]:
    per = consumption_by_step(shops, step_from, step_to, expected_future)
    return {p: sum(v) for p, v in per.items()}


def daily_consumption(shops: Sequence[str], day_from: int, expected_future: bool = True) -> Dict[str, List[float]]:
    """Per-product consumption per day from ``day_from`` to day 29 inclusive."""
    per = consumption_by_step(shops, day_from * TURNS_PER_DAY, EPISODE_STEPS, expected_future)
    days = 30 - day_from
    out = {}
    for p, v in per.items():
        out[p] = [sum(v[d * TURNS_PER_DAY:(d + 1) * TURNS_PER_DAY]) for d in range(days)]
    return out


# ---------------------------------------------------------------------------
# Production forecasts from public tiles
# ---------------------------------------------------------------------------
def animal_production_days(animal: str, placed_day: int, from_day: int, to_day: int = 29,
                           cared: bool = True) -> Dict[int, int]:
    """Units produced on each day in [from_day, to_day] by one animal placed on ``placed_day``.

    Production lands at the end-of-day refresh; the care bonus banks one unit per
    fed-and-cared day and pays out on the next production day, so a cared animal
    yields ``interval`` units per production instead of one (capped by max_held).
    """
    a = ANIMALS[animal]
    out: Dict[int, int] = {}
    banked = 0
    for day in range(placed_day, to_day + 1):
        next_day = day + 1
        since = next_day - placed_day - a["first_yield_day"]
        produced = 0
        if since >= 0 and since % a["interval"] == 0:
            produced = 1 + (banked if cared else 0)
            banked = 0
        if cared:
            banked += 1
        produced = min(produced, a["max_held"])
        if produced and day >= from_day:
            out[day] = out.get(day, 0) + produced
    return out


def crop_production_days(crop: str, planted_day: int, from_day: int, fertilized: bool = False) -> Dict[int, int]:
    """Harvestable units by day for a crop watered daily (one-time crops: at max_yield_day)."""
    c = CROPS[crop]
    out: Dict[int, int] = {}
    if not c["ongoing"]:
        # One-time crop watered daily: yield climbs by 1 (2 fertilized) per day in
        # the bonus window; a sensible farmer harvests the day the cap is reached
        # (or at max_yield_day if the cap is never reached).
        units = 1
        window_start = (c["max_yield_day"] + 1) // 2
        harvest_age = c["max_yield_day"]
        for age in range(0, c["max_yield_day"] + 1):
            if window_start <= age <= c["max_yield_day"]:
                units = min(c["max_yield"], units + (2 if fertilized else 1))
                if units >= c["max_yield"]:
                    harvest_age = age
                    break
        day = planted_day + harvest_age
        if day >= from_day and day <= 29:
            out[day] = units
        return out
    count = 0
    for day in range(planted_day, 30):
        since = day + 1 - planted_day - c["first_yield_day"]
        if since < 0 or since % c["interval"] != 0:
            continue
        count += 1
        if count > c["max_yield"]:
            break
        if day >= from_day:
            out[day] = out.get(day, 0) + (2 if fertilized else 1)
    return out


def farm_production_forecast(farm: Dict[str, Any], day: int, cared: bool = True) -> Dict[str, Dict[int, int]]:
    """Expected units per product per day from the visible tiles of a farm."""
    out: Dict[str, Dict[int, int]] = {p: {} for p in PRODUCTS}
    for row in farm.get("tiles", []):
        for tile in row:
            if not isinstance(tile, dict):
                continue
            animal = tile.get("animal")
            if animal in ANIMALS:
                prod = PRODUCT_OF[animal]
                held = int(tile.get("yield_units", 0) or 0)
                if held:
                    out[prod][day] = out[prod].get(day, 0) + held
                for d, u in animal_production_days(animal, int(tile.get("placed_day", day)), day, cared=cared).items():
                    out[prod][d] = out[prod].get(d, 0) + u
                continue
            if tile.get("kind") == "PLANT":
                crop = tile.get("crop")
                if crop not in CROPS:
                    continue
                held = int(tile.get("yield_units", 0) or 0)
                planted = int(tile.get("planted_day", day))
                fert = int(tile.get("fertilized_until_day", -1) or -1) >= day
                if CROPS[crop]["ongoing"]:
                    if held:
                        out[crop][day] = out[crop].get(day, 0) + held
                    for d, u in crop_production_days(crop, planted, day + 1, fert).items():
                        out[crop][d] = out[crop].get(d, 0) + u
                else:
                    for d, u in crop_production_days(crop, planted, day, fert).items():
                        out[crop][d] = out[crop].get(d, 0) + u
    return out


def count_animals(farm: Dict[str, Any]) -> Dict[str, int]:
    out = {a: 0 for a in ANIMALS}
    for row in farm.get("tiles", []):
        for tile in row:
            if isinstance(tile, dict) and tile.get("animal") in ANIMALS:
                out[tile["animal"]] += 1
    return out


def count_structures(farm: Dict[str, Any]) -> Dict[str, int]:
    out = {"COOP": 0, "PASTURE": 0}
    for row in farm.get("tiles", []):
        for tile in row:
            if isinstance(tile, dict) and tile.get("kind") in out and not tile.get("animal"):
                out[tile["kind"]] += 1
    return out


# ---------------------------------------------------------------------------
# Rival sales recovery (public information only)
# ---------------------------------------------------------------------------
def recover_rival_sales(prev_inventory: Dict[str, int], inventory: Dict[str, int],
                        own_sold: Dict[str, int], own_bought: Dict[str, int],
                        consumption: Dict[str, float]) -> Dict[str, int]:
    """Units the rival sold this step (at prices above the floor), per product.

    inventory' = inventory + own_sold + rival_sold - own_bought - rival_bought - consumption.
    Only WHEAT and FERTILIZER can be bought; rival buys of those are indistinguishable
    from fewer sells, so those two products are lower bounds.
    """
    out = {}
    for p in PRODUCTS:
        delta = int(inventory.get(p, 0)) - int(prev_inventory.get(p, 0))
        est = delta - int(own_sold.get(p, 0)) + int(own_bought.get(p, 0)) + int(round(consumption.get(p, 0.0)))
        out[p] = max(0, est)
    return out


# ---------------------------------------------------------------------------
# Price-path valuation
# ---------------------------------------------------------------------------
def value_of_units_on_path(product: str, inventory_now: int, day_now: int,
                           consumption_per_day: Sequence[float],
                           base_sales_per_day: Sequence[float],
                           extra_sales_per_day: Sequence[float]) -> float:
    """Dollars earned by ``extra_sales_per_day`` sold on top of ``base_sales_per_day``.

    The book starts at ``inventory_now``; each day it loses consumption and gains
    everybody's sales (units that sell at the $1 floor are not added, as in the
    engine). Sales within a day are priced unit by unit.
    """
    inv = float(inventory_now)
    total = 0.0
    n = len(consumption_per_day)
    for i in range(n):
        inv -= consumption_per_day[i]
        base = int(round(base_sales_per_day[i])) if i < len(base_sales_per_day) else 0
        extra = int(round(extra_sales_per_day[i])) if i < len(extra_sales_per_day) else 0
        # Interleave: base and extra units share the book; price the extra units at
        # the average position (base sold first is pessimistic, last is optimistic).
        units = base + extra
        if units <= 0:
            continue
        prices = []
        for k in range(units):
            pr = market_price(product, inv)
            prices.append(pr)
            if pr > 1:
                inv += 1
        if extra:
            # extra units take the tail half of the day's queue on average
            avg = sum(prices) / units
            total += avg * extra
    return total
