"""Outer V7 layer stack and final production submission entrypoints"""

# ==== counter D: exact best-response ordering against a copy of ourselves (shiiin9, 2026-09-18) ====
# Supersedes layer A's fixed rule.  Against a V48 clone the rival's market list is exactly the
# list this stack produces before D touches it (same tape, same production, same market layers),
# so `_v44y_factor_margin` - V48's own per-unit lockstep evaluator - scores any ordering of our
# own list exactly.  V48 already uses it, but only permutes contiguous runs of 2-6 SELLs; it never
# moves a SELL past a HIRE/BUY_SEED/BUY_ANIMAL/BUY_LAND, which is where layer A found its wins.
#
# Search space: the slots held by SELLs and fixed-price orders.  SELLs may take any of them (any
# order); the fixed-price orders keep their relative order in the slots that are left.  Held fixed:
# BUY_PRODUCT (the turn-0/1 openings depend on its index), wash SELLs of an item the list also
# buys, and V48's deliberate empty slots.  Selling earlier only adds cash before a purchase, so a
# fixed-price order never moves earlier than it already is.
# Budgeted: at most _CXD_BUDGET scored orderings per turn, and the original ordering scores 0, so
# a change needs a strictly positive gain.

# ===========================================================================
# 2965 v7 ENHANCED MICROSTRUCTURE PIPELINE
# Integrates cha22's Milk Shield, MPX Front-Running, Late Seed Cap,
# Merge Compaction, and Queue Hole-Closure on top of V59 Order-Book Permutation
# and Risk-Aware Feed Guard.
# ===========================================================================

_V7_BASE_AGENT = agent

# ---------------------------------------------------------------------------
# 1. SHIELD-MILK: fund failing animal/land buys only in milk-shop worlds
# ---------------------------------------------------------------------------
_V7_SM_ANIMAL = {"COW": 400, "SHEEP": 500, "GOOSE": 300}
_V7_SM_LAND = (1000, 2000, 4000)
_V7_SM_MILK_SHOP = ("PIZZA_SHOP", "ICE_CREAM_SHOP", "SMOOTHIE_SHOP")
_V7_SM_PRODUCTS = ("MELON", "STRAWBERRY", "MILK", "WOOL", "EGG", "TOMATO", "CARROT", "WHEAT", "FERTILIZER")
_V7_SM_REPORT = dict(sm_shields=0, sm_units=0, sm_errors=0)
_V7_SM_PARENT = _V7_BASE_AGENT

def _v7_sm_apply(observation, action):
    if not isinstance(action, dict):
        return action
    shops = list((observation.get("town") or {}).get("unlocked_shops") or [])
    if not any(s in _V7_SM_MILK_SHOP for s in shops):
        return action
    market = [list(o) for o in (action.get("market") or [])]
    if not market:
        return action
    seat = int(observation["player"])
    farm = observation["farms"][seat]
    shed = dict((observation.get("private") or {}).get("shed") or {})
    prices = observation["market"].get("prices") or {}
    quads = len(farm.get("unlocked_quadrants") or [])
    cash = float(farm.get("money", 0) or 0)
    sold = {}
    for o in market:
        if not o:
            continue
        if o[0] == "SELL" and len(o) >= 3:
            item = o[1]
            take = min(max(0, int(o[2])), int(shed.get(item, 0)) - sold.get(item, 0))
            if take > 0:
                cash += take * float(prices.get(item, 0))
                sold[item] = sold.get(item, 0) + take
        elif o[0] == "BUY_ANIMAL" and len(o) >= 3:
            cash -= _V7_SM_ANIMAL.get(o[1], 400) * max(0, int(o[2]))
        elif o[0] == "BUY_SEED" and len(o) >= 3:
            cash -= {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}.get(o[1], 100) * max(0, int(o[2]))
        elif o[0] == "BUY_PRODUCT" and len(o) >= 3:
            cash -= float(prices.get(o[1], 0)) * max(0, int(o[2]))
        elif o[0] == "BUY_LAND":
            extra = quads - 1
            cash -= _V7_SM_LAND[extra] if 0 <= extra < 3 else 4000
    need = -cash
    if need <= 0:
        return action
    candidates = []
    for item in _V7_SM_PRODUCTS:
        price = float(prices.get(item, 0))
        avail = int(shed.get(item, 0)) - sold.get(item, 0)
        if price > 1 and avail > 0:
            candidates.append((price, item, avail))
    candidates.sort(reverse=True)
    added = []
    for price, item, avail in candidates:
        if need <= 0 or len(market) + len(added) >= 10:
            break
        q = min(avail, int(need / price) + 1)
        if q <= 0:
            continue
        added.append(["SELL", item, q])
        need -= q * price
        _V7_SM_REPORT["sm_units"] += q
    if not added:
        return action
    _V7_SM_REPORT["sm_shields"] += 1
    idx = next((i for i, o in enumerate(market) if o and str(o[0]).startswith("BUY")), len(market))
    market = market[:idx] + added + market[idx:]
    return dict(action, market=market[:10])

def v7_sm_agent(observation, configuration=None):
    if int(observation.get("step", 0)) == 0:
        _V7_SM_REPORT.update(sm_shields=0, sm_units=0, sm_errors=0)
    action = _V7_SM_PARENT(observation, configuration)
    try:
        return _v7_sm_apply(observation, action)
    except Exception:
        _V7_SM_REPORT["sm_errors"] += 1
        return action

# ---------------------------------------------------------------------------
# 2. MPX: Microstructure Price Defense (Front-runs crashes in milk/wool/berries)
# ---------------------------------------------------------------------------
_V7_MPX_ITEMS = ("MILK", "STRAWBERRY", "WOOL")
_V7_MPX_HOURS = tuple(range(12, 23))
_V7_MPX_REPORT = dict(mpx_fires=0, mpx_units=0, mpx_errors=0)
_V7_MPX_HIST = {}
_V7_MPX_PARENT = v7_sm_agent

def _v7_mpx_draw_units(item, step):
    draw = 0
    if step % 4 == 0:
        draw += 1
    if step % 24 == 0:
        draw += 1
    return draw

def _v7_mpx_apply(observation, action):
    step = int(observation["step"])
    if step < 144 or step >= 696 or (step % 24) not in _V7_MPX_HOURS:
        return action
    player = int(observation["player"])
    market = observation["market"]
    inv_all = market.get("inventory") or {}
    prices = market.get("prices") or {}
    hist = _V7_MPX_HIST.setdefault(player, {})
    prev = hist.get("prev")
    inv_now = {i: int(inv_all.get(i, 0)) for i in _V7_MPX_ITEMS}
    if prev and prev["step"] == step - 1:
        for i in _V7_MPX_ITEMS:
            d = inv_now[i] - prev["inv"][i] + _v7_mpx_draw_units(i, step - 1) - prev["own"].get(i, 0)
            hist.setdefault(i, []).append(max(0, d))
            if len(hist[i]) > 12:
                del hist[i][:6]
    hist["prev"] = {"step": step, "inv": inv_now, "own": {}}
    market_orders = [list(o) for o in (action.get("market") or [])]
    for o in market_orders:
        if len(o) >= 3 and o[0] == "SELL" and o[1] in _V7_MPX_ITEMS:
            hist["prev"]["own"][o[1]] = hist["prev"]["own"].get(o[1], 0) + int(o[2])
    already = {o[1] for o in market_orders if len(o) > 1 and o[0] == "SELL"}
    native = _IMPL.chassis.players.get(player)
    if not native or native.get("route") not in _IMPL.chassis.routes:
        return action
    stock = projected_shed(action, FarmView(observation))
    added = False
    for item in _V7_MPX_ITEMS:
        if item in already or len(market_orders) >= 10:
            continue
        avail = int(stock.get(item, 0))
        if avail <= 0:
            continue
        p_now = int(prices.get(item, 0))
        if p_now <= 1:
            continue
        rival = hist.get(item) or []
        rival_avg = (sum(rival[-4:]) / len(rival[-4:])) if rival else 0.0
        planned = 6
        try:
            inv = int(inv_all.get(item, 0))
            p_cur = float(_r37_market_price(item, inv))
            inv_next = inv + rival_avg + planned - _v7_mpx_draw_units(item, step)
            p_next = float(_r37_market_price(item, max(0, int(inv_next))))
        except Exception:
            continue
        if p_next < p_cur - 0.5:
            take = min(avail, max(1, planned // 2))
            if take <= 0:
                continue
            market_orders.insert(0, ["SELL", item, take])
            _V7_MPX_REPORT["mpx_fires"] += 1
            _V7_MPX_REPORT["mpx_units"] += take
            hist["prev"]["own"][item] = hist["prev"]["own"].get(item, 0) + take
            added = True
    if not added:
        return action
    return dict(action, market=market_orders[:10])

def v7_mpx_agent(observation, configuration=None):
    if int(observation.get("step", 0)) == 0:
        _V7_MPX_REPORT.update(mpx_fires=0, mpx_units=0, mpx_errors=0)
        _V7_MPX_HIST.clear()
    action = _V7_MPX_PARENT(observation, configuration)
    try:
        return _v7_mpx_apply(observation, action)
    except Exception:
        _V7_MPX_REPORT["mpx_errors"] += 1
        return action

# ---------------------------------------------------------------------------
# 3. CXD: V59 Order-Book Permutation Search
# ---------------------------------------------------------------------------
import itertools as _v7_cxd_it
_V7_CXD_HOST = v7_mpx_agent
_V7_CXD_FIXED = ('HIRE', 'BUY_SEED', 'BUY_ANIMAL', 'BUY_LAND')
_V7_CXD_BUDGET = 800
_V7_CXD_FROM = 0
_V7_CXD_REPORT = {'cxd_turns': 0, 'cxd_gain': 0.0, 'cxd_evals': 0, 'cxd_budget_hits': 0, 'cxd_errors': 0}
_V7_CXD_MODELS = []
_V7_CXD_PARENT_ORDERS = []

def _v7_cxd_candidates(orders, slots, sells, fixed):
    for positions in _v7_cxd_it.permutations(slots, len(sells)):
        out = list(orders)
        rest = [i for i in slots if i not in positions]
        for i, order in zip(positions, sells):
            out[i] = order
        for i, order in zip(rest, fixed):
            out[i] = order
        yield out

def _v7_cxd_reorder(obs, action):
    market = action.get('market') or []
    if len(market) < 2:
        return action
    orders = [list(o) if isinstance(o, (list, tuple)) else o for o in market]
    bought = {o[1] for o in orders if o and len(o) > 1 and o[0] == 'BUY_PRODUCT'}
    slots, sells, fixed = [], [], []
    for i, o in enumerate(orders):
        if not o:
            continue
        if o[0] in _V7_CXD_FIXED:
            slots.append(i); fixed.append(o)
        elif o[0] == 'SELL' and len(o) > 1 and o[1] not in bought:
            slots.append(i); sells.append(o)
    if not sells or len(slots) < 2:
        return action
    params = _v44y_params(obs)
    stock = {k: max(0, int(v)) for k, v in projected_shed(action, FarmView(obs)).items()}
    inv0 = {k: int(v) for k, v in obs['market']['inventory'].items()}
    _V7_CXD_PARENT_ORDERS[:] = [list(o) for o in orders if o]
    models = [m for m in _V7_CXD_MODELS if m] or [orders]
    margins = [_v44y_factor_margin(m, inv0, stock, params) for m in models]

    def margin(cand):
        return min(f(cand) for f in margins)
    base = best = margin(orders)
    best_orders = None
    evals = 0
    for cand in _v7_cxd_candidates(orders, slots, sells, fixed):
        if cand == orders:
            continue
        evals += 1
        if evals > _V7_CXD_BUDGET:
            _V7_CXD_REPORT['cxd_budget_hits'] += 1
            break
        value = margin(cand)
        if value > best + 0.5:
            best, best_orders = value, cand
    _V7_CXD_REPORT['cxd_evals'] += evals
    if best_orders is None:
        return action
    _V7_CXD_REPORT['cxd_turns'] += 1
    _V7_CXD_REPORT['cxd_gain'] += best - base
    return dict(action, market=best_orders)

def v7_cxd_agent(observation, configuration=None):
    action = _V7_CXD_HOST(observation, configuration)
    try:
        if int(observation.get('step', 0)) == 0:
            _V7_CXD_REPORT.update(cxd_turns=0, cxd_gain=0.0, cxd_evals=0, cxd_budget_hits=0, cxd_errors=0)
        if int(observation.get('step', 0)) >= _V7_CXD_FROM:
            return _v7_cxd_reorder(observation, action)
    except Exception:
        _V7_CXD_REPORT['cxd_errors'] += 1
    return action

# ---------------------------------------------------------------------------
# 4. E402: Late Seed Cap (caps unharvestable late seeds after step 624)
# ---------------------------------------------------------------------------
_V7_E402_PARENT = v7_cxd_agent
_V7_E402_CACHE = {}
_V7_E402_REPORT = dict(cut_units=0, saved_cost=0, changed_turns=0, errors=0)

def _v7_e402_remaining(native, step):
    route = native['route']; key = (route, step)
    if key in _V7_E402_CACHE: return _V7_E402_CACHE[key]
    need = 0
    for t in range(step + 1, 719):
        tape = _IMPL.chassis.routes[2 if t >= 648 else route]
        act = tape[t]
        need += sum(1 for c in [act.get('farmer') or ['PASS']] + list(act.get('hands') or [])
                    if len(c) > 1 and c[0] == 'PLANT' and c[1] in ('WHEAT', 'CARROT'))
    _V7_E402_CACHE[key] = need
    return need

def v7_e402_agent(observation, configuration=None):
    action = _V7_E402_PARENT(observation, configuration)
    try:
        step = int(observation['step']); seat = int(observation['player'])
        if step == 0:
            _V7_E402_CACHE.clear()
            for k in _V7_E402_REPORT: _V7_E402_REPORT[k] = 0
        if step < 624: return action
        market = action.get('market', [])
        if not any(len(o) >= 3 and o[0] == 'BUY_SEED' and o[1] in ('WHEAT', 'CARROT') for o in market): return action
        native = _IMPL.chassis.players[seat]
        remaining = _v7_e402_remaining(native, step)
        remaining += sum(1 for queue in native['pending'].values() for pos, c in queue
                         if len(c) > 1 and c[0] == 'PLANT' and c[1] in ('WHEAT', 'CARROT'))
        units = [action.get('farmer') or ['PASS']] + list(action.get('hands') or [])
        available = {p: max(0, int(observation['private']['seeds'].get(p, 0)) - sum(c[:2] == ['PLANT', p] for c in units)) for p in ('WHEAT', 'CARROT')}
        out = []; changed = False
        for o in market:
            if len(o) >= 3 and o[0] == 'BUY_SEED' and o[1] in available:
                p = o[1]; qty = max(0, int(o[2])); keep = min(qty, max(0, remaining - available[p])); available[p] += keep
                if keep < qty:
                    cut = qty - keep; changed = True
                    _V7_E402_REPORT['cut_units'] += cut; _V7_E402_REPORT['saved_cost'] += cut * (10 if p == 'WHEAT' else 20)
                    o = [o[0], p, keep] if keep else []
            out.append(o)
        if changed:
            _V7_E402_REPORT['changed_turns'] += 1
            action = dict(action, market=out)
    except Exception:
        _V7_E402_REPORT['errors'] += 1
    return action

# ---------------------------------------------------------------------------
# 5. MERGE: Compaction of duplicate market orders
# ---------------------------------------------------------------------------
_V7_MG_REPORT = dict(mg_turns=0, mg_merged=0, mg_errors=0)
_V7_MG_PARENT = v7_e402_agent

def _v7_mg_apply(observation, action):
    if not isinstance(action, dict):
        return action
    market = [list(o) for o in (action.get("market") or [])]
    if len(market) < 2:
        return action
    seen = {}
    out = []
    changed = False
    for o in market:
        if o and len(o) >= 3 and o[0] in ("SELL", "BUY_PRODUCT", "BUY_SEED"):
            it = (o[0], o[1])
            if it in seen:
                out[seen[it]][2] = int(out[seen[it]][2]) + int(o[2])
                changed = True
                continue
            seen[it] = len(out)
        out.append(o)
    if not changed:
        return action
    _V7_MG_REPORT["mg_turns"] += 1
    _V7_MG_REPORT["mg_merged"] += len(market) - len(out)
    return dict(action, market=out)

def v7_mg_agent(observation, configuration=None):
    if int(observation.get("step", 0)) == 0:
        _V7_MG_REPORT.update(mg_turns=0, mg_merged=0, mg_errors=0)
    action = _V7_MG_PARENT(observation, configuration)
    try:
        return _v7_mg_apply(observation, action)
    except Exception:
        _V7_MG_REPORT["mg_errors"] += 1
        return action

# ---------------------------------------------------------------------------
# 6. IG: Queue Hole-Closure (removes gaps and pulls sells forward)
# ---------------------------------------------------------------------------
_V7_IG_CASH = frozenset(("CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"))
_V7_IG_REPORT = {"queue_changed_turns": 0, "zeroed_orders": 0, "pulled_orders": 0, "pulled_slots": 0, "errors": 0}
_V7_IG_PARENT = v7_mg_agent

def _v7_ig_close_queue(observation, action):
    if not isinstance(action, dict):
        return action
    market = action.get("market") or []
    if len(market) < 2:
        return action
    projected = dict(projected_shed(action, FarmView(observation)))
    remaining = {item: max(0, int(projected.get(item, 0))) for item in _V7_IG_CASH}
    revised = []
    zeroed = 0
    for raw in market:
        order = list(raw) if isinstance(raw, (list, tuple)) else raw
        if (isinstance(order, list) and len(order) >= 3 and order[0] == "SELL" and order[1] in _V7_IG_CASH):
            requested = max(0, int(order[2]))
            executed = min(requested, remaining[order[1]])
            remaining[order[1]] -= executed
            if executed <= 0:
                revised.append([])
                zeroed += 1
            else:
                revised.append(order)
        else:
            revised.append(order)
    holes = []
    pulled = 0
    distance = 0
    for index, order in enumerate(revised):
        if not order:
            holes.append(index)
            continue
        movable = (isinstance(order, list) and len(order) >= 3 and order[0] == "SELL" and order[1] in _V7_IG_CASH and int(order[2]) > 0)
        if not movable or not holes:
            continue
        target = holes.pop(0)
        revised[target] = order
        revised[index] = []
        holes.append(index)
        pulled += 1
        distance += index - target
    if revised == market:
        return action
    _V7_IG_REPORT["queue_changed_turns"] += 1
    _V7_IG_REPORT["zeroed_orders"] += zeroed
    _V7_IG_REPORT["pulled_orders"] += pulled
    _V7_IG_REPORT["pulled_slots"] += distance
    return dict(action, market=revised)

def v7_ig_agent(observation, configuration=None):
    if int(observation.get("step", 0)) == 0:
        for key in _V7_IG_REPORT:
            _V7_IG_REPORT[key] = 0
    action = _V7_IG_PARENT(observation, configuration)
    try:
        return _v7_ig_close_queue(observation, action)
    except Exception:
        _V7_IG_REPORT["errors"] += 1
        return action

# ---------------------------------------------------------------------------
# 7. HR: Risk-Aware Feed Guard (Protects livestock herd from starvation)
# ---------------------------------------------------------------------------
_V7_HR_PARENT = v7_ig_agent
_V7_HR_REPORT = {'changed': 0, 'extra_units': 0, 'errors': 0}

def v7_hr_agent(observation, configuration=None):
    action = _V7_HR_PARENT(observation, configuration)
    step = int(observation['step']); seat = int(observation['player'])
    if step == 0: _V7_HR_REPORT.update(changed=0, extra_units=0, errors=0)
    if not 192 <= step < 696: return action
    command = action.get('farmer') or ['PASS']
    if command[:2] != ['PICKUP', 'WHEAT']: return action
    try:
        farm = observation['farms'][seat]
        if tuple(farm['farmer']) not in ((4,4),(5,4),(4,5),(5,5)): return action
        native = _IMPL.chassis.players[seat]
        count = 0; needed = 0
        pos = tuple(farm['farmer'])
        for t in range(step + 1, min((step//24+1)*24, 719)):
            a = _IMPL.chassis.routes[2 if t >= 648 else native['route']][t]
            c = a.get('farmer') or ['PASS']
            if c[0] == 'DROP' or c[:2] in (['PICKUP','WHEAT'], ['PLACE','WHEAT']): break
            if c[0] in MOVES:
                dx,dy=MOVES[c[0]]
                pos=(max(0,min(9,pos[0]+dx)),max(0,min(9,pos[1]+dy)))
            if c == ['FEED']:
                tile=farm['tiles'][pos[1]][pos[0]]
                if isinstance(tile,dict) and tile.get('animal') and not tile.get('fed_today'):
                    count += 1
                    if int(tile.get('consecutive_unfed',0)) >= 1:
                        needed=count
        requested = int(command[2]) if len(command) >= 3 else 1
        carried = int(observation['private']['inventories'][0].get('WHEAT',0))
        extra = min(2, max(0, needed-carried-requested))
        others = sum(max(0,int(c[2]) if len(c)>2 else 1) for c in action.get('hands',[]) if c[:2]==['PICKUP','WHEAT'])
        spare = max(0,int(observation['private']['shed'].get('WHEAT',0))-requested-others-2)
        extra = min(extra,spare)
        if extra:
            _V7_HR_REPORT['changed'] += 1; _V7_HR_REPORT['extra_units'] += extra
            return dict(action,farmer=['PICKUP','WHEAT',requested+extra])
    except Exception: _V7_HR_REPORT['errors'] += 1
    return action

# ---------------------------------------------------------------------------
# Final Production Entrypoints
# ---------------------------------------------------------------------------
def kaggle_submission_agent(observation, configuration=None):
    return v7_hr_agent(observation, configuration)

agent = kaggle_submission_agent
