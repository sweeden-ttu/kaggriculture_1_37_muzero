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

"""Per-product revenue attribution by exact replay of the engine's lockstep market.

The engine records, for step ``t``, the observation *before* the actions and the
actions in ``steps[t + 1][i].action``. We replay ``_process_market`` on a copy of
the market with each player's shed (from the private observation) so that
phantom SELL quantities are clamped exactly as the engine clamps them.
"""
from __future__ import annotations

import collections
import copy
from typing import Any, Dict, List

from kaggle_environments.envs.kaggriculture import kaggriculture as K


def _orders_of(action: Any) -> List[List[Any]]:
    if not isinstance(action, dict):
        return []
    m = action.get("market", [])
    return list(m) if isinstance(m, list) else []


def replay_market_step(market: Dict[str, Any], sheds: List[Dict[str, int]], moneys: List[float],
                       orders_by_player: List[List[List[Any]]], hires_today: List[int]):
    """Return (sells, buys) per player: {item: [units, dollars]}. Mutates copies only."""
    inv = dict(market["inventory"])
    params = market.get("params")
    sheds = [dict(s) for s in sheds]
    moneys = list(moneys)
    hires = list(hires_today)
    queues = [q[:10] for q in orders_by_player]
    sells = [collections.defaultdict(lambda: [0, 0.0]) for _ in queues]
    buys = [collections.defaultdict(lambda: [0, 0.0]) for _ in queues]
    max_len = max((len(q) for q in queues), default=0)
    for i in range(max_len):
        ostates = [K._parse_order(q[i]) if i < len(q) else None for q in queues]
        for pid, o in enumerate(ostates):
            if o is None:
                continue
            if o["type"] == "HIRE":
                cost = K._hire_cost(hires[pid])
                if moneys[pid] >= cost:
                    moneys[pid] -= cost
                    hires[pid] += 1
                    buys[pid]["HIRE"][0] += 1
                    buys[pid]["HIRE"][1] += cost
                ostates[pid] = None
            elif o["type"] == "BUY_LAND":
                buys[pid]["BUY_LAND"][0] += 1  # cost resolved by caller from unlocked count
                ostates[pid] = None
        guard = 0
        while True:
            guard += 1
            if guard > 100_000:
                break
            quoted: List[Any] = [None] * len(queues)
            for pid, o in enumerate(ostates):
                if o is None or o["remaining"] <= 0:
                    continue
                op, item = o["type"], o["item"]
                if op == "SELL" and item in K.PRODUCTS:
                    quoted[pid] = ("SELL", item, K.market_price(item, inv[item], params), o)
                elif op == "BUY_PRODUCT" and item in ("WHEAT", "FERTILIZER"):
                    quoted[pid] = ("BUY_PRODUCT", item, K.market_price(item, inv[item] - 1, params), o)
                elif op == "BUY_SEED" and item in K.CROPS:
                    quoted[pid] = ("BUY_SEED", item, K.CROPS[item]["seed"], o)
                elif op == "BUY_ANIMAL" and item in K.ANIMALS:
                    quoted[pid] = ("BUY_ANIMAL", item, K.ANIMALS[item]["cost"], o)
                else:
                    ostates[pid] = None
            if all(q is None for q in quoted):
                break
            committed_any = False
            for pid, q in enumerate(quoted):
                if q is None:
                    continue
                op, item, price, o = q
                ok = False
                if op == "SELL":
                    if sheds[pid].get(item, 0) > 0:
                        sheds[pid][item] -= 1
                        moneys[pid] += price
                        if price > 1:
                            inv[item] += 1
                        sells[pid][item][0] += 1
                        sells[pid][item][1] += price
                        ok = True
                elif op == "BUY_PRODUCT":
                    if moneys[pid] >= price and sum(sheds[pid].values()) < 100:
                        moneys[pid] -= price
                        sheds[pid][item] = sheds[pid].get(item, 0) + 1
                        inv[item] -= 1
                        buys[pid][item][0] += 1
                        buys[pid][item][1] += price
                        ok = True
                elif op == "BUY_SEED":
                    if moneys[pid] >= price:
                        moneys[pid] -= price
                        buys[pid][item][0] += 1
                        buys[pid][item][1] += price
                        ok = True
                elif op == "BUY_ANIMAL":
                    if moneys[pid] >= price and sum(sheds[pid].values()) < 100:
                        moneys[pid] -= price
                        sheds[pid][item] = sheds[pid].get(item, 0) + 1
                        buys[pid][item][0] += 1
                        buys[pid][item][1] += price
                        ok = True
                if ok:
                    o["remaining"] -= 1
                    committed_any = True
                else:
                    ostates[pid] = None
            if not committed_any:
                break
    return sells, buys


def _shed_at_market_time(farm: Dict[str, Any], private: Dict[str, Any], action: Any, step: int) -> Dict[str, int]:
    """Apply one player's recorded unit actions to copies of farm/private and return the shed."""
    farm_c = copy.deepcopy(farm)
    priv_c = copy.deepcopy(private)
    board = len(farm_c["tiles"])
    day = step // 24
    farmer_action = action.get("farmer", ["PASS"]) if isinstance(action, dict) else ["PASS"]
    hands = action.get("hands", []) if isinstance(action, dict) else []
    if not isinstance(hands, list):
        hands = []
    unit_actions = [farmer_action, *hands]
    demand: Dict[str, int] = {}
    for a in unit_actions:
        if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT":
            demand[a[1]] = demand.get(a[1], 0) + 1
    seeds = priv_c.get("seeds", {})
    blocked = {c for c, k in demand.items() if k > seeds.get(c, 0)}

    def allowed(a):
        if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT" and a[1] in blocked:
            return ["PASS"]
        return a

    try:
        K._apply_unit_action(farm_c, priv_c, 0, allowed(farmer_action), board, day, 24, 100)
        for h, ha in enumerate(hands):
            K._apply_unit_action(farm_c, priv_c, h + 1, allowed(ha), board, day, 24, 100)
    except Exception:
        pass
    return dict(priv_c["shed"])


def attribute_game(steps: List[List[Any]]) -> Dict[str, Any]:
    """Aggregate per-product sells/buys per player over a finished episode."""
    n = 2
    tot_sell = [collections.defaultdict(lambda: [0, 0.0]) for _ in range(n)]
    tot_buy = [collections.defaultdict(lambda: [0, 0.0]) for _ in range(n)]
    land_cost = [0.0] * n
    price_trace: Dict[str, List[int]] = collections.defaultdict(list)
    for t in range(len(steps) - 1):
        obs0 = steps[t][0].observation
        market = obs0["market"]
        if t % 24 == 0:
            for p in K.PRODUCTS:
                price_trace[p].append(int(market["prices"][p]))
        moneys = [float(obs0["farms"][i]["money"]) for i in range(n)]
        hires = [int(obs0["farms"][i].get("hires_today", 0)) for i in range(n)]
        orders = [_orders_of(steps[t + 1][i].action) for i in range(n)]
        # Unit actions (harvest, DROP, PLACE) run before the market inside one
        # engine step, so the shed the market sees is the pre-step shed plus
        # same-step deposits. Reproduce that with the engine's own functions.
        sheds = [_shed_at_market_time(obs0["farms"][i], steps[t][i].observation["private"],
                                      steps[t + 1][i].action, t) for i in range(n)]
        sells, buys = replay_market_step(market, sheds, moneys, orders, hires)
        for pid in range(n):
            for k, v in sells[pid].items():
                tot_sell[pid][k][0] += v[0]
                tot_sell[pid][k][1] += v[1]
            for k, v in buys[pid].items():
                tot_buy[pid][k][0] += v[0]
                tot_buy[pid][k][1] += v[1]
            if buys[pid].get("BUY_LAND", [0])[0]:
                before = len(obs0["farms"][pid]["unlocked_quadrants"])
                after = len(steps[t + 1][0].observation["farms"][pid]["unlocked_quadrants"])
                for k in range(before, after):
                    land_cost[pid] += K.LAND_PRICES[k - 1]
    last = steps[-1][0].observation
    out: Dict[str, Any] = {"town": list(last["town"]["unlocked_shops"]), "final_prices": dict(last["market"]["prices"]),
                           "price_trace": dict(price_trace)}
    for pid in range(n):
        farm = last["farms"][pid]
        tiles = collections.Counter()
        for row in farm["tiles"]:
            for tile in row:
                if isinstance(tile, dict):
                    tiles[tile.get("animal") or tile.get("crop") or tile.get("kind")] += 1
                elif tile is None:
                    tiles["empty"] += 1
                else:
                    tiles["LOCKED"] += 1
        revenue = sum(v[1] for v in tot_sell[pid].values())
        costs = sum(v[1] for v in tot_buy[pid].values()) + land_cost[pid]
        out[f"P{pid}"] = {
            "revenue": revenue,
            "costs": costs,
            "land_cost": land_cost[pid],
            "net_check": 3000.0 + revenue - costs,
            "final_money": float(farm["money"]),
            "sells": {k: {"units": v[0], "dollars": v[1], "avg": v[1] / max(1, v[0])} for k, v in tot_sell[pid].items()},
            "buys": {k: {"units": v[0], "dollars": v[1]} for k, v in tot_buy[pid].items()},
            "final_tiles": dict(tiles),
            "unlocked": list(farm["unlocked_quadrants"]),
        }
    return out


def format_attribution(attr: Dict[str, Any], key: str) -> str:
    p = attr[key]
    lines = [f"revenue ${p['revenue']:,.0f}  costs ${p['costs']:,.0f}  net-check ${p['net_check']:,.0f}  final ${p['final_money']:,.0f}"]
    for k, v in sorted(p["sells"].items(), key=lambda kv: -kv[1]["dollars"]):
        lines.append(f"    SELL {k:11s} {v['units']:5d} units  ${v['dollars']:>9,.0f}  avg ${v['avg']:6.1f}")
    for k, v in sorted(p["buys"].items(), key=lambda kv: -kv[1]["dollars"]):
        lines.append(f"    BUY  {k:11s} {v['units']:5d} units  ${v['dollars']:>9,.0f}")
    lines.append(f"    tiles {p['final_tiles']} unlocked {p['unlocked']}")
    return "\n".join(lines)
