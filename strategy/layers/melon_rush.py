# ---------------------------------------------------------------------------
# MELON RUSH layer (Kaggriculture MuZero repo, Apache-2.0). Own implementation.
#
# Every tape in the meta plants 12 melons on day 0, waters them daily, harvests
# on day 10 once the yield hits 6, and dumps them at day 10 hours 9-15. Melon
# has no shop demand, so in a mirror both players sell 72 units into the same
# glut at once and split the price decline. A melon fertilized on day 6 (or on
# day 7 before its watering visit) gains +2 per watered day instead of +1 and
# reaches 6 by the end of day 8, so it can be harvested at the day-9 watering
# visit (same step, no path change) and sold on day 9-10 before the rival's
# dump: the first 72 units fetch ~$240 instead of ~$198, and the rival's units
# then land on a glutted book.
#
# Mechanics: hold collected fertilizer instead of selling it on days 5-6, hire
# one extra hand (fib cost ~$21) at the last tape hire hour of days 6 and 7,
# walk it from the shed to each melon tile with fertilizer, fertilize, then
# replace WATER by HARVEST on any melon already at max yield, and sell melons
# as soon as they reach the shed from day 8. Everything is tunable for sweeps.
# ---------------------------------------------------------------------------
_MR_ENABLED = True
_MR_HOLD_FERT_FROM = 0        # optional: steps in which SELL FERTILIZER is held back (off by default:
_MR_HOLD_FERT_TO = 0          # the hand picks fertilizer up at hour 1, before the tape's morning sale)
_MR_HIRE_DAYS = (6, 7)        # days on which one extra hand may be hired
_MR_MIN_CASH_AFTER_HIRE = 60  # keep this much cash after the hire (the day-6 wool sale funds NE at h6)
_MR_SELL_FROM_DAY = 8         # sell melons in the shed from this day
_MR_MAX_FERT = 12
_MR_STATE = {}
_MR_REPORT = {"mr_hires": 0, "mr_fert": 0, "mr_early_harvests": 0, "mr_sells": 0, "mr_held_fert": 0,
              "mr_hand_lost": 0, "mr_errors": 0}
_MR_ACCESS = ((4, 4), (5, 4), (4, 5), (5, 5))
_MR_MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}


def _mr_fib(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _mr_walk(pos, target):
    """One move toward target (x first, then y); None when already there."""
    dx, dy = target[0] - pos[0], target[1] - pos[1]
    if dx > 0:
        return ["EAST"]
    if dx < 0:
        return ["WEST"]
    if dy > 0:
        return ["SOUTH"]
    if dy < 0:
        return ["NORTH"]
    return None


def _mr_melons(farm, day):
    """Melon tiles not yet fertilized for today, as (x, y)."""
    out = []
    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == "MELON":
                if int(tile.get("fertilized_until_day", -1)) < day:
                    out.append((x, y))
    return out


def _mr_tape_hires_by_hour(seat, day):
    """Hours of the day at which the native tape issues HIRE orders."""
    native = _IMPL.chassis.players.get(seat)
    if not native or native.get("route") not in _IMPL.chassis.routes:
        return []
    tape = _IMPL.chassis.routes[native["route"]]
    hours = []
    for h in range(24):
        t = day * 24 + h
        if t < len(tape) and isinstance(tape[t], dict):
            for o in tape[t].get("market", []) or []:
                if o and o[0] == "HIRE":
                    hours.append(h)
    return hours


_MR_PARENT = agent
del agent


def agent(observation, configuration=None):
    action = _MR_PARENT(observation, configuration)
    if not _MR_ENABLED:
        return action
    try:
        seat = int(observation["player"])
        step = int(observation["step"])
        day, hour = step // 24, step % 24
        st = _MR_STATE.get(seat)
        if st is None or step <= st["step"]:
            st = _MR_STATE[seat] = {"step": -1, "hand": None, "hand_day": -1, "hire_step": -1, "expected": 0,
                                    "carry": 0, "targets": [], "hired_days": set()}
            if step == 0:
                _MR_REPORT.update(mr_hires=0, mr_fert=0, mr_early_harvests=0, mr_sells=0, mr_held_fert=0,
                                  mr_hand_lost=0, mr_errors=0)
        st["step"] = step
        if not isinstance(action, dict):
            return action
        farm = observation["farms"][seat]
        private = observation.get("private") or {}
        shed = dict(private.get("shed") or {})
        inventories = list(private.get("inventories") or [])
        market = [list(o) for o in (action.get("market") or []) if o]
        hands = [list(c) if c else ["PASS"] for c in (action.get("hands") or [])]
        farmer = list(action.get("farmer") or ["PASS"])
        changed = False

        # 1. keep collected fertilizer for the melons
        if _MR_HOLD_FERT_TO > _MR_HOLD_FERT_FROM and _MR_HOLD_FERT_FROM <= step < _MR_HOLD_FERT_TO:
            kept = [o for o in market if not (len(o) >= 2 and o[0] == "SELL" and o[1] == "FERTILIZER")]
            if len(kept) != len(market):
                _MR_REPORT["mr_held_fert"] += 1
                market = kept
                changed = True

        # 2. hire one extra hand after the tape's last hire of the day
        if day in _MR_HIRE_DAYS and day not in st["hired_days"] and st["hand_day"] != day:
            tape_hours = _mr_tape_hires_by_hour(seat, day)
            last_tape_hire = max(tape_hours) if tape_hours else -1
            unfertilized = _mr_melons(farm, day)
            fert_available = int(shed.get("FERTILIZER", 0))
            # hire only once every tape hire of the day is in, so tape hand indices stay valid
            if hour >= last_tape_hire and hour <= 10 and unfertilized:
                orders_hires = sum(1 for o in market if o[0] == "HIRE")
                n_before = int(farm.get("hires_today", 0)) + orders_hires
                cost = _mr_fib(n_before)
                spend = 0.0
                for o in market:
                    if len(o) >= 3 and o[0] == "BUY_SEED":
                        spend += int(o[2]) * {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}.get(o[1], 0)
                    elif len(o) >= 3 and o[0] == "BUY_ANIMAL":
                        spend += int(o[2]) * {"GOOSE": 300, "COW": 400, "SHEEP": 500}.get(o[1], 0)
                    elif len(o) >= 3 and o[0] == "BUY_PRODUCT":
                        spend += int(o[2]) * (int(observation["market"]["prices"].get(o[1], 0)) + 5)
                    elif o[0] == "BUY_LAND":
                        spend += {1: 1000, 2: 2000, 3: 4000}.get(len(farm.get("unlocked_quadrants") or ["NW"]), 4000)
                    elif o[0] == "HIRE":
                        spend += 0  # counted via n_before
                for k in range(int(farm.get("hires_today", 0)), n_before):
                    spend += _mr_fib(k)
                cash = float(farm.get("money", 0.0)) - spend
                reserve = _MR_MIN_CASH_AFTER_HIRE
                if cash - cost >= reserve and len(market) < 10:
                    market.append(["HIRE"])
                    st["hand_day"] = day
                    st["hire_step"] = step
                    st["expected"] = len(farm.get("hands") or []) + orders_hires + 1
                    st["hand"] = st["expected"] - 1
                    st["carry"] = 0
                    st["targets"] = list(unfertilized)
                    st["hired_days"].add(day)
                    _MR_REPORT["mr_hires"] += 1
                    changed = True

        # 3. drive our hand for the rest of its day
        if st["hand_day"] == day and st["hand"] is not None and step > st.get("hire_step", -1):
            real_hands = farm.get("hands") or []
            if len(real_hands) != st["expected"]:
                _MR_REPORT["mr_hand_lost"] += 1
                st["hand"] = None
            else:
                idx = st["hand"]
                pos = (int(real_hands[idx][0]), int(real_hands[idx][1]))
                inv = inventories[idx + 1] if idx + 1 < len(inventories) else {}
                carry = int(inv.get("FERTILIZER", 0))
                targets = [t for t in st["targets"] if t in set(_mr_melons(farm, day))]
                st["targets"] = targets
                cmd = ["PASS"]
                if targets:
                    if carry <= 0:
                        avail = int(shed.get("FERTILIZER", 0))
                        if avail > 0:
                            if pos in _MR_ACCESS:
                                cmd = ["PICKUP", "FERTILIZER", min(avail, len(targets), _MR_MAX_FERT)]
                            else:
                                home = min(_MR_ACCESS, key=lambda p: abs(p[0] - pos[0]) + abs(p[1] - pos[1]))
                                cmd = _mr_walk(pos, home) or ["PASS"]
                    else:
                        if pos in targets:
                            cmd = ["FERTILIZE"]
                            _MR_REPORT["mr_fert"] += 1
                        else:
                            nearest = min(targets, key=lambda p: abs(p[0] - pos[0]) + abs(p[1] - pos[1]))
                            cmd = _mr_walk(pos, nearest) or ["PASS"]
                elif carry > 0 and pos not in _MR_ACCESS and hour >= 20:
                    home = min(_MR_ACCESS, key=lambda p: abs(p[0] - pos[0]) + abs(p[1] - pos[1]))
                    cmd = _mr_walk(pos, home) or ["PASS"]
                elif carry > 0 and pos in _MR_ACCESS:
                    cmd = ["PLACE", "FERTILIZER", carry]
                while len(hands) < idx + 1:
                    hands.append(["PASS"])
                hands[idx] = cmd
                changed = True

        # 4. harvest melons already at max yield instead of watering them
        positions = [tuple(farm["farmer"])] + [tuple(h) for h in (farm.get("hands") or [])]
        units = [farmer] + hands
        for i in range(min(len(units), len(positions))):
            if st["hand"] is not None and st["hand_day"] == day and i == st["hand"] + 1:
                continue
            if units[i] == ["WATER"]:
                x, y = positions[i]
                tile = farm["tiles"][y][x]
                if isinstance(tile, dict) and tile.get("crop") == "MELON" and int(tile.get("yield_units", 0)) >= 6:
                    units[i] = ["HARVEST"]
                    _MR_REPORT["mr_early_harvests"] += 1
                    changed = True
        farmer, hands = units[0], units[1:]

        # 5. sell melons as soon as they reach the shed (before the rival's dump)
        if day >= _MR_SELL_FROM_DAY and step < 700:
            try:
                stock = int(projected_shed(dict(action, farmer=farmer, hands=hands, market=market), FarmView(observation)).get("MELON", 0))
            except Exception:
                stock = int(shed.get("MELON", 0))
            selling = sum(int(o[2]) for o in market if len(o) >= 3 and o[0] == "SELL" and o[1] == "MELON")
            if stock > selling and len(market) < 10 and int(observation["market"]["prices"].get("MELON", 0)) >= 60:
                market.insert(0, ["SELL", "MELON", stock - selling])
                _MR_REPORT["mr_sells"] += 1
                changed = True

        if changed:
            action = dict(action)
            action["farmer"], action["hands"], action["market"] = farmer, hands, market[:10]
    except Exception:
        _MR_REPORT["mr_errors"] += 1
    return action


agent.telemetry = _MR_REPORT
kaggle_submission_agent = agent
