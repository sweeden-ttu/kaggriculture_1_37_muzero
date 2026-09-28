# ---------------------------------------------------------------------------
# SPECIES layer (Kaggriculture MuZero repo, Apache-2.0): re-decide every
# pasture-animal purchase on the tape by expected season margin under the exact
# market model. Cows and sheep share one choreography (pasture, feed, care,
# harvest, collect), so the swap is safe at any purchase; later PICKUP/PLACE
# orders are re-targeted by what the shed and the unit's inventory really hold.
#
# The rival's visible herd, the shops drawn so far, the expected future draw and
# the tape's own planned purchases all enter the price path, and the value of a
# unit includes the price change it inflicts on the rival's future units (a win
# is a *margin*, not a total). Appended after every other layer, so it sees the
# final action of the stack. Tunables are module globals for arena sweeps.
# ---------------------------------------------------------------------------
_SP_MODE = "ev"            # "tape" (never deviate) | "cow" | "sheep" | "ev"
_SP_MIN_GAIN = 300.0       # EV advantage ($) needed to deviate from the tape species
_SP_FIRST_STEP = 24        # purchases before this step are left to the tape (day-0 hedge)
_SP_CARE = 0.8             # fraction of the daily care bonus actually realised
_SP_FUTURE = 1.0           # weight on expected demand from shops not yet drawn
_SP_RIVAL_WEIGHT = 1.0     # weight on price damage inflicted on the rival's future units
_SP_MIRROR_SIM = 0.9       # tile similarity above which the rival is assumed to run our tape
_SP_PASTURE = ("COW", "SHEEP")
_SP_SPEC = {
    "GOOSE": {"cost": 300, "first": 4, "interval": 1, "max_held": 4, "product": "EGG"},
    "COW": {"cost": 400, "first": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "first": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
}
_SP_SHOPS = {
    "BAKERY": ("EGG", "WHEAT"), "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"), "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"), "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"), "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}
_SP_STATE = {}
_SP_REPORT = {"sp_decisions": [], "sp_rewrites": 0, "sp_errors": 0}


def _sp_price(item, inventory, params=None):
    return _r37_market_price(item, int(round(inventory)), params)


def _sp_params(obs):
    params = {key: dict(v) for key, v in _R37_MARKET_PARAMS.items()}
    for key, patch in ((obs.get("market") or {}).get("params") or {}).items():
        if key in params and isinstance(patch, dict):
            params[key].update(patch)
    return params


def _sp_schedule(species, placed_day, day_from):
    """Units per day >= day_from from one animal placed on placed_day (care bonus banked
    from the placement day, paid on production days, capped by max_held)."""
    spec = _SP_SPEC[species]
    out = {}
    banked = 0.0
    for day in range(placed_day, 30):
        since = day + 1 - placed_day - spec["first"]
        produced = 0.0
        if since >= 0 and since % spec["interval"] == 0:
            produced = min(float(spec["max_held"]), 1.0 + banked)
            banked = 0.0
        banked += _SP_CARE
        if produced and day >= day_from:
            out[day] = out.get(day, 0.0) + produced
    return out


def _sp_daily_demand(item, shops, day_now):
    """Units of item consumed per day for days day_now+1..29: known shops, town center,
    and the expected demand of shops not yet drawn (weighted by _SP_FUTURE)."""
    known = 0.0
    for name in shops:
        products = _SP_SHOPS.get(name, ())
        if item in products:
            known += 6.0 * (2 if len(products) == 1 else 1)
    per_unlock = sum(6.0 * (2 if len(p) == 1 else 1) for p in _SP_SHOPS.values() if item in p) / len(_SP_SHOPS)
    demand = {}
    for d in range(day_now + 1, 30):
        opened = sum(1 for k in range(len(shops) + 1, 9) if d >= 3 * k)
        demand[d] = known + 1.0 + _SP_FUTURE * per_unlock * opened
    return demand


def _sp_supply(obs, species, st):
    """(own, rival) units per day of the species' product already committed."""
    seat = int(obs["player"])
    step = int(obs["step"])
    day = step // 24
    own, rival = {}, {}
    for idx, farm in enumerate(obs["farms"]):
        target = own if idx == seat else rival
        for row in farm["tiles"]:
            for tile in row:
                if isinstance(tile, dict) and tile.get("animal") == species:
                    for d, u in _sp_schedule(species, int(tile.get("placed_day", day)), day + 1).items():
                        target[d] = target.get(d, 0.0) + u
    private = obs.get("private") or {}
    held = int((private.get("shed") or {}).get(species, 0))
    held += sum(int(inv.get(species, 0)) for inv in (private.get("inventories") or []))
    held += int(st.get("bought", {}).get(species, 0))
    for _ in range(held):
        for d, u in _sp_schedule(species, day + 1, day + 1).items():
            own[d] = own.get(d, 0.0) + u
    native = _IMPL.chassis.players.get(seat)
    mirror = _r37_similarity(obs) >= _SP_MIRROR_SIM
    if native and native.get("route") in _IMPL.chassis.routes:
        for t in range(step + 1, 696):
            tape = _IMPL.chassis.routes[2 if t >= 648 else native["route"]]
            if t >= len(tape) or not isinstance(tape[t], dict):
                continue
            for order in tape[t].get("market", []) or []:
                if len(order) >= 3 and order[0] == "BUY_ANIMAL" and order[1] == species:
                    for _ in range(max(0, int(order[2]))):
                        for d, u in _sp_schedule(species, t // 24 + 1, day + 1).items():
                            own[d] = own.get(d, 0.0) + u
                            if mirror:
                                rival[d] = rival.get(d, 0.0) + u
    return own, rival


def _sp_ev(species, k, obs, st):
    """Expected season margin of k new animals of `species` bought now."""
    spec = _SP_SPEC[species]
    item = spec["product"]
    day = int(obs["step"]) // 24
    params = _sp_params(obs)
    own, rival = _sp_supply(obs, species, st)
    shops = list((obs.get("town") or {}).get("unlocked_shops") or [])
    demand = _sp_daily_demand(item, shops, day)
    new = {d: u * k for d, u in _sp_schedule(species, day + 1, day + 1).items()}
    inv0 = float(obs["market"]["inventory"][item])

    def path(with_new):
        inv = inv0
        prices, revenue = {}, 0.0
        for d in range(day + 1, 30):
            inv -= demand.get(d, 0.0)
            for _ in range(int(round(own.get(d, 0.0) + rival.get(d, 0.0)))):
                if _sp_price(item, inv, params) > 1:
                    inv += 1
            prices[d] = _sp_price(item, inv, params)
            if with_new:
                for _ in range(int(round(new.get(d, 0.0)))):
                    price = _sp_price(item, inv, params)
                    revenue += price
                    if price > 1:
                        inv += 1
        return prices, revenue

    base_prices, _ = path(False)
    new_prices, revenue = path(True)
    swing = 0.0
    for d, p_new in new_prices.items():
        dp = p_new - base_prices[d]
        swing += dp * own.get(d, 0.0) - _SP_RIVAL_WEIGHT * dp * rival.get(d, 0.0)
    return revenue + swing - spec["cost"] * k


def _sp_choose(obs, tape_species, k, st):
    if _SP_MODE == "tape":
        return tape_species, {}
    if _SP_MODE in ("cow", "sheep"):
        return _SP_MODE.upper(), {}
    evs = {s: _sp_ev(s, k, obs, st) for s in _SP_PASTURE}
    best = max(_SP_PASTURE, key=lambda s: evs[s])
    if best != tape_species and evs[best] - evs[tape_species] < _SP_MIN_GAIN:
        best = tape_species
    return best, evs


_SP_PARENT = agent
del agent


def agent(observation, configuration=None):
    action = _SP_PARENT(observation, configuration)
    try:
        seat = int(observation["player"])
        step = int(observation["step"])
        st = _SP_STATE.get(seat)
        if st is None or step <= st["step"]:
            st = _SP_STATE[seat] = {"step": -1, "bought": {}}
            if step == 0:
                _SP_REPORT.update(sp_decisions=[], sp_rewrites=0, sp_errors=0)
        st["step"] = step
        if not isinstance(action, dict):
            return action
        farm = observation["farms"][seat]
        private = observation.get("private") or {}
        shed = dict(private.get("shed") or {})
        inventories = list(private.get("inventories") or [])
        market = [list(o) for o in (action.get("market") or [])]
        changed = False
        cash = float(farm.get("money", 0.0))
        # 1. purchases: re-decide species
        st["bought"] = {}
        for order in market:
            if len(order) >= 3 and order[0] == "BUY_ANIMAL" and order[1] in _SP_PASTURE and int(order[2]) > 0:
                k = int(order[2])
                tape_species = order[1]
                if step >= _SP_FIRST_STEP:
                    chosen, evs = _sp_choose(observation, tape_species, k, st)
                else:
                    chosen, evs = tape_species, {}
                if chosen != tape_species:
                    need = _SP_SPEC[chosen]["cost"] * k
                    if cash >= need + 50:
                        order[1] = chosen
                        changed = True
                        _SP_REPORT["sp_rewrites"] += 1
                    else:
                        chosen = tape_species
                st["bought"][chosen] = st["bought"].get(chosen, 0) + k
                _SP_REPORT["sp_decisions"].append((step, tape_species, chosen, k, {s: int(v) for s, v in evs.items()}))
            if len(order) >= 3 and order[0] in ("BUY_ANIMAL", "BUY_SEED", "BUY_PRODUCT"):
                q = max(0, int(order[2]))
                if order[0] == "BUY_ANIMAL":
                    cash -= q * _SP_SPEC.get(order[1], {}).get("cost", 0)
                elif order[0] == "BUY_SEED":
                    cash -= q * {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}.get(order[1], 0)
                else:
                    cash -= q * int((observation["market"]["prices"] or {}).get(order[1], 0))
            elif order and order[0] == "BUY_LAND":
                cash -= {1: 1000, 2: 2000, 3: 4000}.get(len(farm.get("unlocked_quadrants") or ["NW"]), 4000)
        # 2. PICKUP / PLACE: re-target by real stock
        units = [list(action.get("farmer") or ["PASS"])] + [list(c) for c in (action.get("hands") or [])]
        for i, cmd in enumerate(units):
            if len(cmd) >= 2 and cmd[0] == "PICKUP" and cmd[1] in _SP_PASTURE:
                n = max(1, int(cmd[2]) if len(cmd) > 2 else 1)
                if shed.get(cmd[1], 0) < n:
                    alt = max(_SP_PASTURE, key=lambda s: shed.get(s, 0))
                    if shed.get(alt, 0) > shed.get(cmd[1], 0):
                        units[i] = ["PICKUP", alt] + cmd[2:]
                        changed = True
                        _SP_REPORT["sp_rewrites"] += 1
                shed[units[i][1]] = shed.get(units[i][1], 0) - min(n, shed.get(units[i][1], 0))
            elif len(cmd) >= 2 and cmd[0] == "PLACE" and cmd[1] in _SP_PASTURE:
                inv = inventories[i] if i < len(inventories) else {}
                if int(inv.get(cmd[1], 0)) <= 0:
                    alt = max(_SP_PASTURE, key=lambda s: int(inv.get(s, 0)))
                    if int(inv.get(alt, 0)) > 0:
                        units[i] = ["PLACE", alt] + cmd[2:]
                        changed = True
                        _SP_REPORT["sp_rewrites"] += 1
        if changed:
            action = dict(action)
            action["farmer"], action["hands"], action["market"] = units[0], units[1:], market
    except Exception:
        _SP_REPORT["sp_errors"] += 1
    return action


agent.telemetry = _SP_REPORT
kaggle_submission_agent = agent
