"""Replay-distilled lesson layer (merged ep 113379227 + 113388398).

Wraps the upstream hybrid agent. Policy is embedded (no __file__ / filesystem
reads) so Kaggle loaders that exec main.py without __file__ still import cleanly.

Regenerate embedded constants via:
  python learn_from_replay.py --embed
  python package_submission.py
"""
# Merged winner lessons: early SW, tomato + carrot seeds, morning premium liq,
# hire bump, shed near-cap sells, phantom-sell clamp, SE soft-skip.
_REPLAY_POLICY = {
    "sw_day_min": 8,
    "sw_money_floor": 2500.0,
    "skip_se": False,
    "tomato_seed_start_day": 10,
    "tomato_seed_end_day": 18,
    "tomato_seed_total_target": 11,
    "tomato_seed_price": 50,
    "carrot_post_ne": False,
    "carrot_post_ne_qty": 0,
    "carrot_seed_price": 20,
    "carrot_post_ne_day_end": 12,
    "carrot_late_start_day": 20,
    "carrot_late_end_day": 26,
    "carrot_late_target": 40,
    "melon_wave2_day": 10,
    "melon_wave2_qty": 4,
    "melon_seed_price": 80,
    "geese_target": 4,
    "early_hire_target": 5,
    "early_cow_qty": 2,
    "early_sheep_qty": 2,
    "neml_step0": True,
    "fert_trap": True,
    "premium_first_on_shed_cap": True,
    "premium_first_items": [
        "STRAWBERRY",
        "MILK",
        "TOMATO",
        "MELON",
        "WOOL",
        "EGG"
    ],
    "liq_hours": [
        1,
        2
    ],
    "premium_items": [
        "STRAWBERRY",
        "MILK",
        "TOMATO",
        "MELON",
        "WOOL",
        "EGG"
    ],
    "hire_unlocked_min": 2,
    "hire_day_end": 14,
    "hire_target": 12,
    "shed_force_sell_at": 85,
    "shed_target_after": 75,
    "clamp_sell_to_shed": True
}

_CFG = dict(_REPLAY_POLICY)
_REPORT = dict(
    sw_injected=0,
    tomato_seeds=0,
    carrot_seeds=0,
    liq_boosts=0,
    hire_boosts=0,
    shed_forced=0,
    premium_first=0,
    fert_clamped=0,
    errors=0,
)
_PARENT = agent
_NON_SEED = (
    "WHEAT", "MELON", "CARROT", "TOMATO", "STRAWBERRY",
    "EGG", "EGGS", "MILK", "WOOL", "FERTILIZER",
)
_FIB = (1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144)


def _market(action):
    return [list(o) for o in (action.get("market") or [])]


def _slots(orders):
    return max(0, 10 - len(orders))


def _has(orders, op, item=None):
    for o in orders:
        if not o or o[0] != op:
            continue
        if item is None or (len(o) > 1 and o[1] == item):
            return True
    return False


def _free_slot(orders, prefer_drop=("FERTILIZER", "WHEAT")):
    if _slots(orders) > 0:
        return True
    for i in range(len(orders) - 1, -1, -1):
        o = orders[i]
        if o and o[0] == "SELL" and len(o) > 1 and o[1] in prefer_drop:
            orders.pop(i)
            return True
        if o and o[0] == "BUY_PRODUCT" and len(o) > 1 and o[1] == "FERTILIZER":
            orders.pop(i)
            return True
        if o and o[0] == "BUY_SEED" and len(o) > 1 and o[1] == "WHEAT":
            orders.pop(i)
            return True
    if orders:
        orders.pop()
        return True
    return False


def _clamp_sells(orders, shed):
    if not _CFG.get("clamp_sell_to_shed", True):
        return orders, 0
    clamped = 0
    out = []
    for o in orders:
        if o and o[0] == "SELL" and len(o) >= 3:
            item, qty = o[1], int(o[2])
            stock = int(shed.get(item, 0) or 0)
            if qty > stock:
                clamped += 1
                if stock <= 0:
                    continue
                o = [o[0], item, stock]
        out.append(o)
    return out, clamped


def _force_shed(orders, shed):
    thr = int(_CFG.get("shed_force_sell_at", 85))
    target = int(_CFG.get("shed_target_after", 75))
    total = sum(int(shed.get(k, 0) or 0) for k in _NON_SEED)
    if total < thr:
        return orders, 0
    premium = set(_CFG.get("premium_items") or [])
    stacks = sorted(
        [(k, int(shed.get(k, 0) or 0)) for k in _NON_SEED if int(shed.get(k, 0) or 0) > 0],
        key=lambda kv: (0 if kv[0] in premium else 1, -kv[1]),
    )
    already = {o[1]: int(o[2]) for o in orders if o and o[0] == "SELL" and len(o) >= 3}
    cur, forced = total, 0
    for item, qty in stacks:
        if cur < target or not _free_slot(orders):
            break
        avail = max(0, qty - already.get(item, 0))
        if avail <= 0:
            continue
        need = min(avail, cur - target + 1)
        if item in already:
            for o in orders:
                if o and o[0] == "SELL" and len(o) >= 3 and o[1] == item:
                    o[2] = int(o[2]) + need
                    break
        else:
            orders.append(["SELL", item, need])
        already[item] = already.get(item, 0) + need
        cur -= need
        forced += 1
    return orders, forced


def _premium_first(orders, shed, day, hour):
    if not _CFG.get("premium_first_on_shed_cap", True):
        return orders, 0
    thr = int(_CFG.get("shed_force_sell_at", 85))
    total = sum(int(shed.get(k, 0) or 0) for k in _NON_SEED)
    if total < thr:
        return orders, 0
    if hour not in (_CFG.get("liq_hours") or [1, 2]) and day < 27:
        return orders, 0
    premium = list(_CFG.get("premium_first_items") or _CFG.get("premium_items") or [])
    stacks = sorted(
        [(k, int(shed.get(k, 0) or 0)) for k in premium if int(shed.get(k, 0) or 0) > 0],
        key=lambda kv: -kv[1],
    )
    if not stacks:
        return orders, 0
    item, stock = stacks[0]
    orders = [o for o in orders if not (o and o[0] == "SELL" and len(o) >= 3 and o[1] == item)]
    _free_slot(orders)
    if len(orders) >= 10:
        _free_slot(orders)
    orders.insert(0, ["SELL", item, stock])
    return orders, 1


def _liq_boost(orders, shed, day, hour):
    if hour not in (_CFG.get("liq_hours") or [1, 2]) and day < 27:
        return orders, 0
    boosts = 0
    for item in _CFG.get("premium_items") or []:
        if not _free_slot(orders) and not _has(orders, "SELL", item):
            break
        stock = int(shed.get(item, 0) or 0)
        if stock <= 0:
            continue
        existing = next(
            (o for o in orders if o and o[0] == "SELL" and len(o) >= 3 and o[1] == item),
            None,
        )
        if existing is not None:
            if int(existing[2]) < stock:
                existing[2] = stock
                boosts += 1
        elif _slots(orders) > 0 or _free_slot(orders):
            orders.append(["SELL", item, stock])
            boosts += 1
    return orders, boosts


def _tomato_seeds(orders, seeds, money, day, bought):
    start = int(_CFG.get("tomato_seed_start_day", 10))
    end = int(_CFG.get("tomato_seed_end_day", 18))
    target = int(_CFG.get("tomato_seed_total_target", 15))
    price = int(_CFG.get("tomato_seed_price", 50))
    if day < start or day > end or bought >= target:
        return orders, 0, bought
    if _has(orders, "BUY_SEED", "TOMATO") or not _free_slot(orders):
        return orders, 0, bought
    on_hand = int(seeds.get("TOMATO", 0) or 0)
    need = min(4, target - bought, max(0, 3 - on_hand))
    need = min(need, int(max(0.0, money - 200.0) // max(1, price)))
    if need <= 0:
        return orders, 0, bought
    orders.append(["BUY_SEED", "TOMATO", need])
    return orders, need, bought + need


def _carrot_post_ne(orders, farm, seeds, money, day, bought):
    if not _CFG.get("carrot_post_ne", True):
        return orders, 0, bought
    unlocked = list(farm.get("unlocked_quadrants") or ["NW"])
    if "NE" not in unlocked or day > int(_CFG.get("carrot_post_ne_day_end", 12)):
        return orders, 0, bought
    target = int(_CFG.get("carrot_post_ne_qty", 12))
    if bought >= target or _has(orders, "BUY_SEED", "CARROT"):
        return orders, 0, bought
    if not _free_slot(orders):
        return orders, 0, bought
    price = int(_CFG.get("carrot_seed_price", 10))
    on_hand = int(seeds.get("CARROT", 0) or 0)
    burst = 12 if ("SW" not in unlocked and day <= 8) else 4
    need = min(burst, target - bought, max(0, burst - on_hand))
    need = min(need, int(max(0.0, money - 200.0) // max(1, price)))
    if need <= 0:
        return orders, 0, bought
    insert_at = 0
    for i, o in enumerate(orders):
        if o and o[0] == "BUY_LAND":
            insert_at = i + 1
            break
    orders.insert(insert_at, ["BUY_SEED", "CARROT", need])
    return orders, need, bought + need


def _early_sw(orders, farm, money, day):
    unlocked = list(farm.get("unlocked_quadrants") or ["NW"])
    if "SW" in unlocked or "NE" not in unlocked:
        return orders, 0
    if day < int(_CFG.get("sw_day_min", 8)):
        return orders, 0
    if money < float(_CFG.get("sw_money_floor", 2500.0)):
        return orders, 0
    if _has(orders, "BUY_LAND"):
        return orders, 0
    if not _free_slot(orders):
        return orders, 0
    orders.insert(0, ["BUY_LAND"])
    return orders, 1


def _hire_bump(orders, farm, day, money):
    unlocked = list(farm.get("unlocked_quadrants") or ["NW"])
    if len(unlocked) < int(_CFG.get("hire_unlocked_min", 2)):
        return orders, 0
    if day > int(_CFG.get("hire_day_end", 14)) or day > 27:
        return orders, 0
    target = int(_CFG.get("hire_target", 13))
    hands = len(farm.get("hands") or [])
    hires_today = int(farm.get("hires_today", 0) or 0)
    queued = sum(1 for o in orders if o and o[0] == "HIRE")
    have = max(hands, hires_today) + queued
    added = 0
    while have < target and _slots(orders) > 0:
        cost = _FIB[min(len(_FIB) - 1, hires_today + added)]
        if money < cost + 15:
            break
        orders.append(["HIRE"])
        money -= cost
        added += 1
        have += 1
    return orders, added


def agent(observation, configuration=None):
    action = _PARENT(observation, configuration)
    try:
        seat = int(observation["player"])
        step = int(observation.get("step", 0))
        day = int(observation.get("day", step // 24))
        hour = int(observation.get("hour", step % 24))
        farm = observation["farms"][seat]
        private = observation.get("private") or {}
        shed = dict(private.get("shed") or {})
        seeds = dict(private.get("seeds") or {})
        money = float(farm.get("money", 0.0))

        if step == 0:
            for k in list(_REPORT.keys()):
                if k != "errors":
                    _REPORT[k] = 0
            _REPORT["_tomato_bought"] = 0
            _REPORT["_carrot_bought"] = 0

        orders = _market(action)

        orders, n = _clamp_sells(orders, shed)
        _REPORT["fert_clamped"] += n

        orders, n = _premium_first(orders, shed, day, hour)
        _REPORT["premium_first"] += n

        orders, n = _force_shed(orders, shed)
        _REPORT["shed_forced"] += n

        orders, n = _liq_boost(orders, shed, day, hour)
        _REPORT["liq_boosts"] += n

        orders, n = _early_sw(orders, farm, money, day)
        _REPORT["sw_injected"] += n

        bought_t = int(_REPORT.get("_tomato_bought", 0))
        orders, n, bought_t = _tomato_seeds(orders, seeds, money, day, bought_t)
        _REPORT["_tomato_bought"] = bought_t
        _REPORT["tomato_seeds"] += n

        bought_c = int(_REPORT.get("_carrot_bought", 0))
        orders, n, bought_c = _carrot_post_ne(orders, farm, seeds, money, day, bought_c)
        _REPORT["_carrot_bought"] = bought_c
        _REPORT["carrot_seeds"] += n

        orders, n = _hire_bump(orders, farm, day, money)
        _REPORT["hire_boosts"] += n

        unlocked = list(farm.get("unlocked_quadrants") or ["NW"])
        if _CFG.get("skip_se") and "SW" in unlocked and "SE" not in unlocked:
            orders = [o for o in orders if not (o and o[0] == "BUY_LAND")]

        action = dict(action, market=orders[:10])
    except Exception:
        _REPORT["errors"] += 1
    return action


agent.telemetry = _REPORT
agent = globals().pop("agent")
kaggle_submission_agent = agent
