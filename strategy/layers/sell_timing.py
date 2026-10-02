# ---------------------------------------------------------------------------
# SELL TIMING layer (Kaggriculture MuZero repo, Apache-2.0). Own implementation.
#
# The tape sells premium goods (milk, wool, strawberry) the moment they reach
# the shed. In a two-tape match both players dump into the same book at the
# same hours, and the price the rival's dump leaves behind is what we get.
# This layer re-times our sales of those goods with the exact price curve and
# three public signals: town consumption (known from the unlocked shops), the
# rival's recent sales (recovered from market inventory moves: inventory' =
# inventory + both sales - our buys - consumption, so their units are
# ours to read), and the rival's *pending* supply (yield_units sitting on
# their visible animal and strawberry tiles, which they harvest within a day).
#
# Each step, for each premium product with stock in the shed: value of selling
# everything now vs. value of selling it after h steps (h in _ST_HORIZONS),
# with the book moved by consumption, the rival's rate and pending units over
# those steps. Hold when the best deferred value beats now by _ST_MARGIN,
# unless the shed is nearly full, the hold has lasted _ST_MAX_HOLD steps, or
# the season is ending; otherwise make sure the stock is on sale this step.
# ---------------------------------------------------------------------------
_ST_ENABLED = False          # measured: -$634 (seed 11) and -$233 (seed 13) vs the mirror; the rival takes the recovery
_ST_PRODUCTS = ("MILK", "WOOL", "STRAWBERRY")
_ST_MARGIN = 0.06            # deferred value must beat selling now by this fraction
_ST_HORIZONS = (6, 12, 24)   # steps ahead considered for a deferred sale
_ST_MAX_HOLD = 48            # never hold one product longer than this many steps
_ST_SHED_CAP = 88            # do not hold when the projected shed is at or above this
_ST_LAST_STEP = 690          # no holding from here; the tape's terminal liquidation follows
_ST_RATE_WINDOW = 48         # steps of recovered rival sales averaged into a rate
_ST_PENDING_WEIGHT = 1.0     # weight on the rival's visible unharvested units
_ST_MIN_PRICE = 2            # never add a sale below this quote
_ST_STATE = {}
_ST_REPORT = {"st_holds": 0, "st_releases": 0, "st_added_sales": 0, "st_units_held": 0, "st_errors": 0}
_ST_ANIMAL_PRODUCT = {"COW": "MILK", "SHEEP": "WOOL", "GOOSE": "EGG"}
_ST_ALL_PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")


def _st_params(obs):
    params = {key: dict(v) for key, v in _R37_MARKET_PARAMS.items()}
    for key, patch in ((obs.get("market") or {}).get("params") or {}).items():
        if key in params and isinstance(patch, dict):
            params[key].update(patch)
    return params


def _st_consumption(shops, step_from, steps):
    """Per-product units consumed by the town over [step_from, step_from + steps)."""
    per_tick = {p: 0.0 for p in _ST_ALL_PRODUCTS}
    for name in shops:
        products = _HD2_SHOP_TYPES.get(name, ())
        mult = 2.0 if len(products) == 1 else 1.0
        for p in products:
            per_tick[p] += mult
    out = {p: 0.0 for p in _ST_ALL_PRODUCTS}
    for s in range(step_from, step_from + steps):
        if s % 4 == 0:
            for p in _ST_ALL_PRODUCTS:
                out[p] += per_tick[p]
        if s % 24 == 0:
            for p in _ST_ALL_PRODUCTS:
                if p != "FERTILIZER":
                    out[p] += 1.0
    return out


def _st_sale_value(item, inventory, units, params):
    """Dollars from selling `units` one by one starting at `inventory`."""
    total = 0.0
    inv = float(inventory)
    for _ in range(int(units)):
        price = _r37_market_price(item, int(round(inv)), params)
        total += price
        if price > 1:
            inv += 1
    return total


def _st_rival_pending(obs, seat):
    """Unharvested units of each premium product visible on the rival's tiles."""
    rival = obs["farms"][1 - seat]
    out = {p: 0 for p in _ST_PRODUCTS}
    for row in rival.get("tiles", []):
        for tile in row:
            if not isinstance(tile, dict):
                continue
            animal = tile.get("animal")
            if animal in _ST_ANIMAL_PRODUCT and _ST_ANIMAL_PRODUCT[animal] in out:
                out[_ST_ANIMAL_PRODUCT[animal]] += max(0, int(tile.get("yield_units", 0) or 0))
            elif tile.get("kind") == "PLANT" and tile.get("crop") in out:
                out[tile["crop"]] += max(0, int(tile.get("yield_units", 0) or 0))
    return out


def _st_update_rival_rate(obs, st, seat, step):
    """Recover the rival's sales since the last step from public inventory moves."""
    market = obs["market"]["inventory"]
    prev = st.get("prev_inventory")
    if prev is not None and step == st.get("prev_step", -2) + 1:
        shops = list((obs.get("town") or {}).get("unlocked_shops") or [])
        cons = _st_consumption(shops, step - 1, 1)
        own = st.get("own_last", {})
        for p in _ST_PRODUCTS:
            delta = int(market.get(p, 0)) - int(prev.get(p, 0))
            rival_units = max(0, delta - int(own.get(p, 0)) + int(round(cons.get(p, 0.0))))
            hist = st["rival_hist"].setdefault(p, [])
            hist.append(rival_units)
            if len(hist) > _ST_RATE_WINDOW:
                del hist[: len(hist) - _ST_RATE_WINDOW]
    st["prev_inventory"] = dict(market)
    st["prev_step"] = step


def _st_rival_rate(st, p):
    hist = st["rival_hist"].get(p, [])
    if not hist:
        return 0.0
    return sum(hist) / float(len(hist))


_ST_PARENT = agent
del agent


def agent(observation, configuration=None):
    action = _ST_PARENT(observation, configuration)
    if not _ST_ENABLED:
        return action
    try:
        seat = int(observation["player"])
        step = int(observation["step"])
        st = _ST_STATE.get(seat)
        if st is None or step <= st["step"]:
            st = _ST_STATE[seat] = {"step": -1, "prev_inventory": None, "prev_step": -2, "own_last": {},
                                    "rival_hist": {}, "hold_since": {}}
            if step == 0:
                _ST_REPORT.update(st_holds=0, st_releases=0, st_added_sales=0, st_units_held=0, st_errors=0)
        st["step"] = step
        _st_update_rival_rate(observation, st, seat, step)
        if not isinstance(action, dict):
            return action
        market = [list(o) for o in (action.get("market") or []) if o]
        params = _st_params(observation)
        shops = list((observation.get("town") or {}).get("unlocked_shops") or [])
        inventory = observation["market"]["inventory"]
        prices = observation["market"]["prices"]
        try:
            projected = projected_shed(dict(action, market=market), FarmView(observation))
        except Exception:
            projected = dict((observation.get("private") or {}).get("shed") or {})
        shed_total = sum(max(0, int(v)) for v in projected.values())
        pending = _st_rival_pending(observation, seat)
        changed = False
        own_sold = {}
        for p in _ST_PRODUCTS:
            stock = max(0, int(projected.get(p, 0)))
            selling = sum(max(0, int(o[2])) for o in market if len(o) >= 3 and o[0] == "SELL" and o[1] == p)
            if stock <= 0:
                st["hold_since"].pop(p, None)
                continue
            inv_now = int(inventory.get(p, 0))
            value_now = _st_sale_value(p, inv_now, stock, params)
            best_h, best_value = 0, value_now
            can_hold = (step < _ST_LAST_STEP and shed_total < _ST_SHED_CAP
                        and step - st["hold_since"].get(p, step) < _ST_MAX_HOLD)
            if can_hold:
                rate = _st_rival_rate(st, p)
                for h in _ST_HORIZONS:
                    cons = _st_consumption(shops, step, h).get(p, 0.0)
                    rival = rate * h + _ST_PENDING_WEIGHT * pending.get(p, 0) * min(1.0, h / 24.0)
                    inv_h = inv_now - cons + rival
                    value_h = _st_sale_value(p, inv_h, stock, params)
                    if value_h > best_value:
                        best_h, best_value = h, value_h
            hold = best_h > 0 and best_value >= value_now * (1.0 + _ST_MARGIN)
            if hold:
                st["hold_since"].setdefault(p, step)
                if selling > 0:
                    market = [o for o in market if not (len(o) >= 3 and o[0] == "SELL" and o[1] == p)]
                    _ST_REPORT["st_holds"] += 1
                    _ST_REPORT["st_units_held"] += min(stock, selling)
                    changed = True
            else:
                if p in st["hold_since"]:
                    st["hold_since"].pop(p, None)
                    _ST_REPORT["st_releases"] += 1
                if selling < stock and int(prices.get(p, 0)) >= _ST_MIN_PRICE and len(market) < 10:
                    market.insert(0, ["SELL", p, stock - selling])
                    _ST_REPORT["st_added_sales"] += 1
                    changed = True
                own_sold[p] = stock
            if not hold:
                own_sold[p] = min(stock, max(selling, own_sold.get(p, 0)))
        st["own_last"] = own_sold
        if changed:
            action = dict(action)
            action["market"] = market[:10]
    except Exception:
        _ST_REPORT["st_errors"] += 1
    return action


agent.telemetry = _ST_REPORT
kaggle_submission_agent = agent
