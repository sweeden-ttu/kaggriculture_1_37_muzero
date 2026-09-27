#!/usr/bin/env python3
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

"""
=============================================================================
economic_diagnostics.py — Kaggriculture Economic & Labor Diagnostics
=============================================================================
Loads a replay (or the latest self-play), extracts the economic state at a
given step, and prints:

  1. Current economic snapshot at step N (capital, day, unlocked, shed, hands)
  2. Predicted labor tradeoff cost-benefit analysis
  3. Macro-level opportunity-cost predictions
  4. Estimated optimal hands (crew size maximizing net income)
  5. Current marginal cost of adding another hand
  6. Opportunity cost of unlocking each land quadrant
  7. Minimum liquid inventory + cash threshold for adding another hemisphere
  8. Market commodity prices and available shops

Usage:
  python scripts/economic_diagnostics.py                      # latest replay, final step
  python scripts/economic_diagnostics.py --replay R.json --step 360
  python scripts/economic_diagnostics.py --seat 0 --step 120
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from muzero.types import (  # noqa: E402
    DEFAULT_PRICES,
    LAND_BUFFERS,
    LAND_ORDER,
    LAND_PRICES,
    POLICY_FIBONACCI,
)
from muzero.economy import (  # noqa: E402
    META_HANDS_BY_DAY,
    META_SE_EXTRA_BUFFER,
    META_TRICKLE_BATCH,
    QUADRANT_DAILY_PRODUCE,
    QUADRANT_DAILY_SEED_COST,
    _default_hands,
    _next_expansion_target,
    option_to_macro_action,
)
from muzero.types import MacroAction, MacroOption, MacroState  # noqa: E402


# ---------------------------------------------------------------------------
# Replay loading & state extraction
# ---------------------------------------------------------------------------


def find_latest_replay(replays_dir: str) -> Optional[str]:
    files = glob.glob(os.path.join(replays_dir, "*.json"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def extract_state_at_step(
    replay: Dict[str, Any],
    step: int,
    seat: int = 0,
) -> Dict[str, Any]:
    """Extract the economic snapshot for `seat` at `step` from a replay."""
    steps = replay.get("steps", [])
    if step >= len(steps):
        step = len(steps) - 1
    seat_data = steps[step][seat] if seat < len(steps[step]) else steps[step][0]
    obs = seat_data.get("observation", {})
    farms = obs.get("farms", [{}, {}])
    farm = farms[seat] if seat < len(farms) else {}
    pvt = obs.get("private", {})
    market = obs.get("market", {})
    return {
        "step": int(obs.get("step", step)),
        "day": int(obs.get("day", step // 24)),
        "hour": int(obs.get("hour", step % 24)),
        "seat": seat,
        "capital": float(farm.get("money", 0.0)),
        "hands": len(farm.get("hands", [])) if isinstance(farm.get("hands"), list) else int(farm.get("hands", 0)),
        "unlocked": list(farm.get("unlocked_quadrants") or ["NW"]),
        "shed": dict(pvt.get("shed", {})),
        "seeds": dict(pvt.get("seeds", {})),
        "market_prices": dict(market.get("prices", {})),
        "market_inventory": dict(market.get("inventory", {})),
        "reward": seat_data.get("reward", 0.0),
    }


def build_macro_state(snap: Dict[str, Any]) -> MacroState:
    return MacroState(
        day=snap["day"],
        capital=snap["capital"],
        unlocked=tuple(snap["unlocked"]),
        shed=dict(snap["shed"]),
        prices=dict(snap.get("market_prices") or DEFAULT_PRICES),
        active_animals=1,
    )


# ---------------------------------------------------------------------------
# Labor economics
# ---------------------------------------------------------------------------


def labor_cost(hands: int) -> float:
    """Total daily labor cost for `hands` workers (Fibonacci-scaled)."""
    return float(sum(POLICY_FIBONACCI[min(h, len(POLICY_FIBONACCI) - 1)] for h in range(hands)))


def marginal_hand_cost(current_hands: int) -> float:
    """Cost of adding the next hand (current_hands is 0-indexed count)."""
    idx = min(current_hands, len(POLICY_FIBONACCI) - 1)
    return float(POLICY_FIBONACCI[idx])


def daily_produce_value(state: MacroState, hands: Optional[int] = None) -> float:
    """Gross daily produce value across unlocked quadrants."""
    total = 0.0
    for q in state.unlocked:
        for item, qty in QUADRANT_DAILY_PRODUCE.get(q, {}).items():
            price = state.prices.get(item, DEFAULT_PRICES.get(item, 25.0))
            total += qty * price
    return total


def daily_seed_cost(state: MacroState) -> float:
    return float(sum(QUADRANT_DAILY_SEED_COST.get(q, 15.0) for q in state.unlocked))


def optimal_hands(state: MacroState, max_hands: int = 12) -> Tuple[int, float]:
    """Sweep hand counts and return (best_hands, best_net_daily_income)."""
    best_h, best_net = 0, -float("inf")
    for h in range(1, max_hands + 1):
        # Productivity saturates: each hand works a slice of available tiles.
        productivity = min(1.0, h / max(1, _default_hands(state.day, len(state.unlocked))))
        gross = daily_produce_value(state) * productivity
        net = gross - labor_cost(h) - daily_seed_cost(state)
        if net > best_net:
            best_net = net
            best_h = h
    return best_h, best_net


# ---------------------------------------------------------------------------
# Land economics
# ---------------------------------------------------------------------------


def land_expansion_analysis(state: MacroState) -> List[Dict[str, Any]]:
    """Opportunity-cost analysis for unlocking each remaining quadrant."""
    results: List[Dict[str, Any]] = []
    next_q = _next_expansion_target(state)
    for q in LAND_ORDER:
        if q in state.unlocked:
            continue
        price = LAND_PRICES[q]
        buffer = LAND_BUFFERS[q]
        meta_buffer = META_SE_EXTRA_BUFFER if q == "SE" else 0.0
        total_cost = price + buffer + meta_buffer
        marginal_produce = sum(
            qty * state.prices.get(item, DEFAULT_PRICES.get(item, 25.0))
            for item, qty in QUADRANT_DAILY_PRODUCE.get(q, {}).items()
        )
        marginal_seed = QUADRANT_DAILY_SEED_COST.get(q, 15.0)
        net_daily = marginal_produce - marginal_seed
        payback_days = total_cost / net_daily if net_daily > 0 else float("inf")
        affordable = state.capital >= total_cost
        results.append({
            "quadrant": q,
            "price": price,
            "buffer": buffer,
            "meta_buffer": meta_buffer,
            "total_cost": total_cost,
            "marginal_daily_produce": marginal_produce,
            "marginal_daily_seed": marginal_seed,
            "net_daily_income": net_daily,
            "payback_days": payback_days,
            "affordable": affordable,
            "is_next": q == next_q,
        })
    return results


# ---------------------------------------------------------------------------
# Expansion threshold
# ---------------------------------------------------------------------------


def minimum_expansion_threshold(state: MacroState) -> Dict[str, Any]:
    """Minimum liquid inventory + cash for adding another hemisphere (quadrant).

    Land unlock requires: land price + buffer + necessary liquid cash for
    labor + inventory BEFORE spinning off a new hemisphere farmer agent.
    """
    next_q = _next_expansion_target(state)
    if next_q is None:
        return {"next_quadrant": None, "minimum_cash": 0.0, "minimum_liquid_inventory": 0.0}

    price = LAND_PRICES[next_q]
    buffer = LAND_BUFFERS[next_q]
    meta_buffer = META_SE_EXTRA_BUFFER if next_q == "SE" else 0.0
    base_need = price + buffer + meta_buffer

    # New hemisphere farmer agent startup costs:
    # - Initial labor cash: wages for the new agent's crew for the first days
    # - Initial inventory: seeds + feed to make the new quadrant productive
    new_quad = state.unlocked + (next_q,)
    new_hands = _default_hands(state.day, len(new_quad))
    new_labor = labor_cost(new_hands)
    new_seeds = QUADRANT_DAILY_SEED_COST.get(next_q, 15.0)

    # Liquid cash for labor: new agent needs wages for sustain_days
    sustain_days = 3
    labor_cash_need = new_labor * sustain_days

    # Inventory needed before spin-off: seeds for planting + wheat feed buffer
    seed_inventory_need = new_seeds * sustain_days
    feed_inventory_need = 15.0  # wheat reserve for animals
    inventory_need = seed_inventory_need + feed_inventory_need

    # Safety margin for unexpected costs
    safety_margin = 500.0

    min_cash = base_need + labor_cash_need + safety_margin
    min_liquid_inventory = inventory_need

    return {
        "next_quadrant": next_q,
        "land_price": price,
        "buffer": buffer,
        "meta_buffer": meta_buffer,
        "base_need": base_need,
        "sustain_days": sustain_days,
        "new_agent_hands": new_hands,
        "new_agent_labor_per_day": new_labor,
        "labor_cash_need": labor_cash_need,
        "seed_inventory_need": seed_inventory_need,
        "feed_inventory_need": feed_inventory_need,
        "safety_margin": safety_margin,
        "minimum_cash": min_cash,
        "minimum_liquid_inventory": min_liquid_inventory,
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def print_report(snap: Dict[str, Any], state: MacroState) -> None:
    W = 78
    print("=" * W)
    print("  KAGGRICULTURE ECONOMIC & LABOR DIAGNOSTICS")
    print("=" * W)

    # 1. Current economic snapshot
    print(f"\n[1] CURRENT ECONOMIC STATE AT STEP {snap['step']}")
    print("-" * W)
    print(f"  Day {snap['day']:>2} | Hour {snap['hour']:>2} | Seat {snap['seat']} | Step {snap['step']}")
    print(f"  Capital (cash on hand):       ${snap['capital']:>12,.2f}")
    print(f"  Hands (crew size):            {snap['hands']:>12d}")
    print(f"  Unlocked quadrants:           {', '.join(snap['unlocked']):>12}")
    shed = snap["shed"]
    shed_total = sum(shed.values())
    print(f"  Shed inventory (total items): {shed_total:>12.1f}")
    for item, qty in sorted(shed.items(), key=lambda kv: -kv[1]):
        if qty > 0:
            price = state.prices.get(item, DEFAULT_PRICES.get(item, 25.0))
            print(f"      {item:<14} {qty:>8.1f} units @ ${price:>7.2f} = ${qty * price:>10,.2f}")

    # 2. Labor tradeoff
    print(f"\n[2] LABOR TRADEOFF COST-BENEFIT ANALYSIS")
    print("-" * W)
    cur_hands = snap["hands"]
    cur_labor = labor_cost(cur_hands)
    prod_val = daily_produce_value(state)
    seed_cost = daily_seed_cost(state)
    print(f"  Current hands:                {cur_hands:>12d}")
    print(f"  Daily labor cost (Fibonacci): ${cur_labor:>12,.2f}")
    print(f"  Daily gross produce value:    ${prod_val:>12,.2f}")
    print(f"  Daily seed cost:              ${seed_cost:>12,.2f}")
    print(f"  Net daily income (current):   ${prod_val - cur_labor - seed_cost:>12,.2f}")
    print()
    print(f"  {'Hands':>6} | {'Labor $':>9} | {'Gross $':>9} | {'Net $':>9} | {'Marginal $':>10} | Verdict")
    print(f"  {'-' * 6}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 10}-+-{'-' * 20}")
    for h in range(max(1, cur_hands - 1), min(cur_hands + 4, 13)):
        productivity = min(1.0, h / max(1, _default_hands(state.day, len(state.unlocked))))
        gross = prod_val * productivity
        net = gross - labor_cost(h) - seed_cost
        mc = marginal_hand_cost(h - 1) if h > 0 else 0.0
        verdict = "OPTIMAL" if h == cur_hands else ("add" if net > (prod_val * min(1.0, cur_hands / max(1, _default_hands(state.day, len(state.unlocked)))) - labor_cost(cur_hands) - seed_cost) else "drop")
        marker = " <-- current" if h == cur_hands else ""
        print(f"  {h:>6} | {labor_cost(h):>9,.0f} | {gross:>9,.0f} | {net:>9,.0f} | {mc:>10,.0f} | {verdict}{marker}")

    # 3. Opportunity costs (macro-level)
    print(f"\n[3] MACRO-LEVEL OPPORTUNITY COST PREDICTIONS")
    print("-" * W)
    next_q = _next_expansion_target(state)
    premium = sum(state.shed.get(k, 0.0) for k in ("MELON", "MILK", "WOOL", "STRAWBERRY", "EGG", "TOMATO"))
    options: List[Tuple[str, float]] = []
    if next_q:
        cost = LAND_PRICES[next_q] + LAND_BUFFERS[next_q]
        marginal = sum(
            qty * state.prices.get(item, DEFAULT_PRICES.get(item, 25.0))
            for item, qty in QUADRANT_DAILY_PRODUCE.get(next_q, {}).items()
        ) - QUADRANT_DAILY_SEED_COST.get(next_q, 15.0)
        options.append((f"EXPAND_{next_q}", marginal))
    options.append(("HIRE_HANDS", prod_val * 0.15 - marginal_hand_cost(cur_hands)))
    options.append(("TRICKLE_LIQUIDATION", premium * 0.1 if premium >= META_TRICKLE_BATCH else 0.0))
    options.append(("PASS", 0.0))
    options.sort(key=lambda kv: -kv[1])
    print(f"  {'Macro-option':<24} | {'Expected daily value $':>20} | Opportunity cost $")
    print(f"  {'-' * 24}-+-{'-' * 20}-+-{'-' * 20}")
    best_val = options[0][1] if options else 0.0
    for name, val in options:
        oc = best_val - val
        print(f"  {name:<24} | {val:>20,.2f} | {oc:>20,.2f}")

    # 4. Optimal hands
    print(f"\n[4] ESTIMATED OPTIMAL HANDS")
    print("-" * W)
    best_h, best_net = optimal_hands(state)
    meta_h = _default_hands(state.day, len(state.unlocked))
    print(f"  Meta schedule hands (day {state.day}, {len(state.unlocked)} quadrants): {meta_h}")
    print(f"  Economically optimal hands (max net income):                  {best_h}")
    print(f"  Net daily income at optimal:                                ${best_net:,.2f}")
    print(f"  Current hands:                                               {cur_hands}")
    delta = best_h - cur_hands
    if delta > 0:
        print(f"  Recommendation: ADD {delta} hand(s)")
    elif delta < 0:
        print(f"  Recommendation: DROP {abs(delta)} hand(s)")
    else:
        print(f"  Recommendation: HOLD current crew size")

    # 5. Cost of adding another hand
    print(f"\n[5] CURRENT COST OF ADDING ANOTHER HAND")
    print("-" * W)
    mc = marginal_hand_cost(cur_hands)
    print(f"  Marginal labor cost (hand #{cur_hands + 1}):  ${mc:>10,.2f}/day")
    print(f"  Projected marginal productivity gain:        {prod_val * 0.15:>10,.2f}/day")
    print(f"  Net marginal benefit:                        {prod_val * 0.15 - mc:>10,.2f}/day")
    if prod_val * 0.15 > mc:
        print(f"  Verdict: PROFITABLE to add hand #{cur_hands + 1}")
    else:
        print(f"  Verdict: NOT profitable to add hand #{cur_hands + 1}")

    # 6. Land unlock opportunity costs
    print(f"\n[6] OPPORTUNITY COST OF UNLOCKING LAND")
    print("-" * W)
    land_analysis = land_expansion_analysis(state)
    print(f"  {'Quad':>5} | {'Price $':>9} | {'Total $':>9} | {'Net daily $':>10} | {'Payback d':>9} | {'Affordable':>10}")
    print(f"  {'-' * 5}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 10}-+-{'-' * 9}-+-{'-' * 10}")
    for la in land_analysis:
        marker = " (next)" if la["is_next"] else ""
        afford = "YES" if la["affordable"] else "NO"
        payback = f"{la['payback_days']:.1f}" if la["payback_days"] != float("inf") else "inf"
        print(f"  {la['quadrant']:>5} | {la['price']:>9,.0f} | {la['total_cost']:>9,.0f} | "
              f"{la['net_daily_income']:>10,.2f} | {payback:>9} | {afford:>10}{marker}")

    # 7. Minimum expansion threshold
    print(f"\n[7] MINIMUM LIQUID INVENTORY + CASH FOR NEXT HEMISPHERE")
    print("-" * W)
    thresh = minimum_expansion_threshold(state)
    if thresh["next_quadrant"] is None:
        print("  All quadrants unlocked — no expansion threshold applicable.")
    else:
        print(f"  Next quadrant:                    {thresh['next_quadrant']}")
        print(f"  Land price:                      ${thresh['land_price']:>10,.2f}")
        print(f"  Buffer:                          ${thresh['buffer']:>10,.2f}")
        if thresh["meta_buffer"] > 0:
            print(f"  Meta SE extra buffer:            ${thresh['meta_buffer']:>10,.2f}")
        print(f"  Base expansion need:             ${thresh['base_need']:>10,.2f}")
        print()
        print(f"  New hemisphere farmer agent startup:")
        print(f"    Crew hands:                    {thresh['new_agent_hands']:>10d}")
        print(f"    Labor cash ({thresh['sustain_days']}d wages):        ${thresh['labor_cash_need']:>10,.2f}")
        print(f"    Seed inventory:                {thresh['seed_inventory_need']:>10.1f} units")
        print(f"    Feed inventory (wheat):        {thresh['feed_inventory_need']:>10.1f} units")
        print(f"    Safety margin:                 ${thresh['safety_margin']:>10,.2f}")
        print(f"  ---------------------------------------------")
        print(f"  MINIMUM CASH ON HAND:            ${thresh['minimum_cash']:>10,.2f}")
        print(f"  MINIMUM LIQUID INVENTORY:        {thresh['minimum_liquid_inventory']:>10.1f} items")
        print(f"  Current capital:                 ${snap['capital']:>10,.2f}")
        if snap["capital"] >= thresh["minimum_cash"]:
            print(f"  Status: READY for expansion")
        else:
            short = thresh["minimum_cash"] - snap["capital"]
            print(f"  Status: NEED ${short:,.2f} more cash")

    # 8. Market prices and shops
    print(f"\n[8] MARKET PRICES & AVAILABLE SHOPS")
    print("-" * W)
    prices = snap.get("market_prices") or {k: state.prices.get(k, v) for k, v in DEFAULT_PRICES.items()}
    print(f"  {'Commodity':<14} | {'Price $':>9} | {'Base $':>9} | {'Spot/Base':>10} | {'Available':>10}")
    print(f"  {'-' * 14}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 10}-+-{'-' * 10}")
    for item in sorted(prices.keys()):
        p = float(prices[item])
        base = float(DEFAULT_PRICES.get(item, p))
        ratio = p / base if base > 0 else 1.0
        avail = snap.get("market_inventory", {}).get(item, 0)
        print(f"  {item:<14} | {p:>9,.2f} | {base:>9,.2f} | {ratio:>9.2f}x | {avail:>10,}")

    # 9. Future demand & harvest-time price predictions
    print(f"\n[9] PREDICTED FUTURE DEMAND & HARVEST-TIME PRICES")
    print("-" * W)
    days_remaining = max(0, 30 - snap["day"])
    print(f"  Days remaining in episode: {days_remaining}")
    print()
    print(f"  {'Commodity':<14} | {'Current $':>9} | {'Harvest $':>9} | {'Demand/units':>12} | {'Demand $':>10} | {'Trend':>8}")
    print(f"  {'-' * 14}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 12}-+-{'-' * 10}-+-{'-' * 8}")

    # Animal herd reward collection (daily recurring)
    herd_items = {"MILK": 0.4, "WOOL": 0.35, "EGG": 0.8}
    # Crop harvest schedules (day of harvest)
    harvest_schedule = {"WHEAT": 10, "CARROT": 12, "TOMATO": 15, "STRAWBERRY": 20, "MELON": 10}

    for item in sorted(prices.keys()):
        p = float(prices[item])
        base = float(DEFAULT_PRICES.get(item, p))
        ratio = p / base if base > 0 else 1.0

        # Predict harvest-time price: mean-reversion toward base with premium decay
        if item in harvest_schedule:
            harvest_day = harvest_schedule[item]
            days_to_harvest = max(0, harvest_day - snap["day"])
            # Price premium decays as harvest approaches (supply increases)
            premium_factor = 1.0 + 0.3 * (days_to_harvest / max(1, harvest_day))
            predicted_price = base * premium_factor
            demand_units = QUADRANT_DAILY_PRODUCE.get("NW", {}).get(item, 0) * days_remaining
            trend = "up" if ratio > 1.1 else ("down" if ratio < 0.9 else "flat")
        elif item in herd_items:
            # Herd rewards: steady demand, price tracks base with small premium
            predicted_price = base * 1.05
            demand_units = herd_items[item] * days_remaining * max(1, len(state.unlocked))
            trend = "stable"
        else:
            predicted_price = base
            demand_units = 0.0
            trend = "n/a"

        demand_dollars = demand_units * predicted_price
        print(f"  {item:<14} | {p:>9,.2f} | {predicted_price:>9,.2f} | {demand_units:>12.1f} | {demand_dollars:>10,.2f} | {trend:>8}")

    print()
    total_demand = sum(
        (QUADRANT_DAILY_PRODUCE.get("NW", {}).get(item, 0) * days_remaining * float(prices.get(item, DEFAULT_PRICES.get(item, 25)))
         if item in harvest_schedule else
         (herd_items.get(item, 0) * days_remaining * max(1, len(state.unlocked)) * float(DEFAULT_PRICES.get(item, 25)))
         for item in prices)
    )
    print(f"  Total predicted future demand value: ${total_demand:,.2f}")

    print("\n" + "=" * W)
    print("  END OF DIAGNOSTICS")
    print("=" * W)


def main() -> int:
    parser = argparse.ArgumentParser(description="Kaggriculture economic & labor diagnostics")
    parser.add_argument("--replay", default=None, help="Replay JSON path (default: latest in replays/)")
    parser.add_argument("--step", type=int, default=None, help="Step to analyze (default: final step)")
    parser.add_argument("--seat", type=int, default=0, help="Seat/player to analyze")
    parser.add_argument("--replays-dir", default=os.path.join(HERE, "replays"))
    args = parser.parse_args()

    replay_path = args.replay or find_latest_replay(args.replays_dir)
    if not replay_path or not os.path.isfile(replay_path):
        print(f"Error: no replay found at {replay_path or args.replays_dir}")
        return 1

    with open(replay_path, "r", encoding="utf-8") as f:
        replay = json.load(f)

    total_steps = len(replay.get("steps", []))
    step = args.step if args.step is not None else total_steps - 1
    step = max(0, min(step, total_steps - 1))

    snap = extract_state_at_step(replay, step, seat=args.seat)
    state = build_macro_state(snap)
    print(f"Replay: {replay_path}")
    print(f"Episode rewards: {replay.get('rewards', [])}")
    print()
    print_report(snap, state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
