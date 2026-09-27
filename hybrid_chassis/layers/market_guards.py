"""Market liquidation guards and hole-closure rules (ALT, CL, PG, E402, E410, IG, Ledger)"""
_V13V_PARENT = agent


def agent(observation, configuration=None):
    if int(observation["step"]) == 0:
        for k in _V13V_REPORT:
            _V13V_REPORT[k] = 0
    return _V13V_PARENT(observation, configuration)


agent.telemetry = _V13V_REPORT
agent = globals().pop("agent")


# Bounded alternative productive openings. Dmitrii Gluzdov, 2026, Apache-2.0.
# The upstream The 2945 Farm source and its notices are preserved above.
_ALT_MODE = 'HybridOpening'
_ALT_RAW = copy.deepcopy(_IMPL.chassis.routes[0][:96])
_ALT_CT = dict(CT_TABLE)
_ALT_STATE = {}
_ALT_REPORT = {}

def _alt_install(mode):
    global CT_TABLE
    tape=_IMPL.chassis.routes[0]
    tape[:96]=copy.deepcopy(_ALT_RAW)
    CT_TABLE=dict(_ALT_CT)
    if mode=='Original': return
    if mode=='HybridOpening':
        tape[0]['market']=list(tape[0]['market'])+[['BUY_SEED','WHEAT',1]]
    else:
        CT_TABLE={}
        tape[0]['market']=[['BUY_PRODUCT','WHEAT',5],['BUY_SEED','WHEAT',1]]
        tape[1]['market']=[o for o in tape[1]['market'] if not (len(o)>=3 and o[0] in ('BUY_PRODUCT','SELL') and o[1]=='WHEAT')]
    # Day-zero idle hand 1 walks from (5,4) to the future (2,4) pasture.
    for step,command in {2:['WEST'],3:['WEST'],4:['WEST'],5:['PLANT','WHEAT'],6:['WATER']}.items():
        assert tape[step]['hands'][1]==['PASS']
        tape[step]['hands'][1]=command
    # On day one, water instead of building the not-yet-needed pasture.
    assert tape[29]['hands'][2]==['BUILD_PASTURE']
    tape[29]['hands'][2]=['WATER']
    # Day-two idle hand 0: water, harvest, restore pasture, deliver.
    commands=[['WEST'],['WEST'],['WEST'],['WATER'],['HARVEST'],['BUILD_PASTURE'],['EAST'],['EAST'],['DROP']]
    for step,command in zip(range(49,58),commands):
        assert tape[step]['hands'][0]==['PASS']
        tape[step]['hands'][0]=command
    if mode=='TomatoInsteadOfCow':
        tape[0]['market'].append(['BUY_SEED','TOMATO',1])
        cow=next(o for o in tape[1]['market'] if o[:2]==['BUY_ANIMAL','COW'])
        assert cow[2]==2; cow[2]=1
        # The first carrier still owns the bought cow; the other cow's site
        # becomes one tomato plant, with its worker route left in place.
        assert tape[2]['hands'][4][:2]==['PICKUP','COW']
        assert tape[3]['hands'][4]==['BUILD_PASTURE']
        assert tape[4]['hands'][4][:2]==['PLACE','COW']
        tape[2]['hands'][4]=['PASS']
        tape[3]['hands'][4]=['PASS']
        tape[4]['hands'][4]=['PLANT','TOMATO']
        tape[5]['hands'][4]=['WATER']
    assert max(len(a.get('market',[])) for a in tape[:96])<=10

def _alt_sell_extra(action,item,n):
    if n<=0:return action
    orders=[list(o) for o in action.get('market',[])]
    sell=next((o for o in orders if len(o)>=3 and o[:2]==['SELL',item]),None)
    if sell is not None:sell[2]+=n
    elif len(orders)<10:orders.append(['SELL',item,n])
    else:return action
    return dict(action,market=orders)

_ALT_PARENT=agent
def agent(observation,configuration=None):
    seat,step=int(observation['player']),int(observation['step'])
    state=_ALT_STATE.get(seat)
    if state is None or step<=state['step']:
        mode = _ALT_MODE
        if mode == 'Mixed':
            mode = 'EarlyCycle'
        _alt_install(mode)
        state=_ALT_STATE[seat]={'step':-1,'mode':mode}
        _ALT_REPORT.clear()
        _ALT_REPORT.update(selected_opening=mode,temporary_crop_seen=0,temporary_crop_harvested=0,
                           restored_pasture_seen=0,delivered_extra_wheat=0,tomato_seen=0,
                           tomato_water_requests=0,tomato_harvest_requests=0,tomato_units_harvest_requested=0,
                           extension_errors=0)
    action=_ALT_PARENT(observation,configuration)
    try:
        mode=state['mode']
        if mode!='Original':
            farm=observation['farms'][seat];private=observation['private']
            site=farm['tiles'][4][2]
            if step==6:_ALT_REPORT['temporary_crop_seen']=int(isinstance(site,dict) and site.get('crop')=='WHEAT')
            if step==54:
                _ALT_REPORT['temporary_crop_harvested']=int(private['inventories'][1].get('WHEAT',0))
            if step==55:_ALT_REPORT['restored_pasture_seen']=int(isinstance(site,dict) and site.get('kind')=='PASTURE')
            if step==57 and len(farm['hands'])>=1:
                if tuple(farm['hands'][0])==(4,4) and action.get('hands',[[]])[0]==['DROP']:
                    n=int(private['inventories'][1].get('WHEAT',0))
                    action=_alt_sell_extra(action,'WHEAT',n)
                    _ALT_REPORT['delivered_extra_wheat']=n
            if mode=='TomatoInsteadOfCow':
                tomato=farm['tiles'][4][4]
                if isinstance(tomato,dict) and tomato.get('crop')=='TOMATO' and tomato.get('planted_day')==0:
                    _ALT_REPORT['tomato_seen']=1
                    units=[list(action.get('farmer') or ['PASS'])]+[list(c) for c in action.get('hands',[])]
                    positions=[tuple(farm['farmer'])]+[tuple(p) for p in farm['hands']]
                    watered=bool(tomato.get('watered_today'));yield_left=int(tomato.get('yield_units',0))
                    for actor,pos in enumerate(positions):
                        if pos!=(4,4) or actor>=len(units):continue
                        if units[actor][0] not in ('FEED','CARE','COLLECT_FERTILIZER','HARVEST','WATER','PASS'):continue
                        if not watered:
                            units[actor]=['WATER'];watered=True
                            _ALT_REPORT['tomato_water_requests']+=1
                        elif yield_left>0:
                            units[actor]=['HARVEST']
                            _ALT_REPORT['tomato_harvest_requests']+=1
                            _ALT_REPORT['tomato_units_harvest_requested']+=yield_left
                            yield_left=0
                        else:units[actor]=['PASS']
                    action=dict(action,farmer=units[0],hands=units[1:])
                # Tomatoes are cash produce, never a feed or route input. Sell
                # only observed shed stock beyond any parent's existing sale.
                stock=int(private['shed'].get('TOMATO',0))
                scheduled=sum(int(o[2]) for o in action.get('market',[]) if len(o)>=3 and o[:2]==['SELL','TOMATO'])
                action=_alt_sell_extra(action,'TOMATO',max(0,stock-scheduled))
    except Exception:
        _ALT_REPORT['extension_errors']+=1
    state['step']=step
    return action
agent.telemetry=_ALT_REPORT
agent=globals().pop('agent')

kaggle_submission_agent = agent

# Apache-2.0. Dmitrii Gluzdov: mature the temporary wheat one extra day.
_CL_INSTALL = _alt_install
_CL_REPORT = dict(cl_harvested=0, cl_pasture=0, cl_delivered=0, cl_errors=0)

def _alt_install(mode):
    _CL_INSTALL(mode)
    if mode not in ('EarlyCycle','HybridOpening'):
        return
    tape = _IMPL.chassis.routes[0]
    # Day-two watering remains at52; wait one more day before harvesting.
    for step in range(53,58):
        tape[step]['hands'][0] = ['PASS']
    # Hand0 finishes its original day-three work at(0,4), then idles.
    commands = [['EAST'],['EAST'],['WATER'],['HARVEST'],['BUILD_PASTURE'],['EAST'],['EAST'],['DROP']]
    for step, command in zip(range(84,92), commands):
        assert tape[step]['hands'][0] == ['PASS']
        tape[step]['hands'][0] = command

_CL_PARENT = agent
def agent(observation, configuration=None):
    step = int(observation['step'])
    if step == 0:
        for key in _CL_REPORT: _CL_REPORT[key] = 0
    action = _CL_PARENT(observation, configuration)
    try:
        seat = int(observation['player'])
        farm, private = observation['farms'][seat], observation['private']
        if step == 88:
            _CL_REPORT['cl_harvested'] = int(private['inventories'][1].get('WHEAT',0))
        if step == 89:
            tile = farm['tiles'][4][2]
            _CL_REPORT['cl_pasture'] = int(isinstance(tile,dict) and tile.get('kind')=='PASTURE')
        if step == 91 and farm['hands'] and tuple(farm['hands'][0]) == (4,4) and action['hands'][0] == ['DROP']:
            amount = int(private['inventories'][1].get('WHEAT',0))
            action = _alt_sell_extra(action,'WHEAT',amount)
            _CL_REPORT['cl_delivered'] = amount
    except Exception:
        _CL_REPORT['cl_errors'] += 1
    return action
agent.telemetry = _CL_REPORT
agent = globals().pop('agent')
kaggle_submission_agent = agent

# Correctly last-bound visible-price sale guard experiment; threshold=31.
_PG_PARENT=agent
def final_price_guard(observation, configuration=None):
    action=_PG_PARENT(observation, configuration)
    try:
        if int(observation["step"])==91:
            price=float(observation.get("market",{}).get("prices",{}).get("WHEAT",0))
            if price < 31:
                orders=[list(o) for o in action.get("market",[]) if not (len(o)>=3 and o[0]=="SELL" and o[1]=="WHEAT")]
                action=dict(action, market=orders)
    except Exception:
        pass
    return action
final_price_guard.telemetry={"sale_price_threshold":31}
kaggle_submission_agent=final_price_guard

# EXP389 frozen market-race reservation horizon.
V9_RACE_DEFAULT = 41
V9_RACE_MAX = 48

# EXP402: cap late seed purchases by an upper bound on all remaining planting.
# No price-sensitive product orders are moved. Keep empty market slots so the
# opponent's simultaneous market interactions retain their original positions.
_E402_PARENT=final_price_guard
_E402_CACHE={}
_E402_REPORT=dict(cut_units=0,saved_cost=0,changed_turns=0,errors=0)

def _e402_remaining(native,step):
    route=native['route'];key=(route,step)
    if key in _E402_CACHE:return _E402_CACHE[key]
    need=0
    for t in range(step+1,719):
        tape=_IMPL.chassis.routes[2 if t>=648 else route]
        act=tape[t]
        need+=sum(1 for c in [act.get('farmer') or ['PASS']]+list(act.get('hands') or [])
                  if len(c)>1 and c[0]=='PLANT' and c[1] in ('WHEAT','CARROT'))
    _E402_CACHE[key]=need
    return need

def e402_agent(observation,configuration=None):
    action=_E402_PARENT(observation,configuration)
    try:
        step=int(observation['step']);seat=int(observation['player'])
        if step==0:
            _E402_CACHE.clear()
            for k in _E402_REPORT:_E402_REPORT[k]=0
        if step<624:return action
        market=action.get('market',[])
        if not any(len(o)>=3 and o[0]=='BUY_SEED' and o[1] in ('WHEAT','CARROT') for o in market):return action
        native=_IMPL.chassis.players[seat]
        remaining=_e402_remaining(native,step)
        # Queued retries can outlive their original schedule; reserve for them too.
        remaining+=sum(1 for queue in native['pending'].values() for pos,c in queue
                       if len(c)>1 and c[0]=='PLANT' and c[1] in ('WHEAT','CARROT'))
        units=[action.get('farmer') or ['PASS']]+list(action.get('hands') or [])
        available={p:max(0,int(observation['private']['seeds'].get(p,0))-sum(c[:2]==['PLANT',p] for c in units)) for p in ('WHEAT','CARROT')}
        out=[];changed=False
        for o in market:
            if len(o)>=3 and o[0]=='BUY_SEED' and o[1] in available:
                p=o[1];qty=max(0,int(o[2]));keep=min(qty,max(0,remaining-available[p]));available[p]+=keep
                if keep<qty:
                    cut=qty-keep;changed=True
                    _E402_REPORT['cut_units']+=cut;_E402_REPORT['saved_cost']+=cut*(10 if p=='WHEAT' else 20)
                    if p=='CARROT':
                        st=_CA_STATE.get(seat)
                        if st is not None:st['spare_carrot']=max(0,st.get('spare_carrot',0)-cut)
                    o=[o[0],p,keep] if keep else []
            out.append(o)
        if changed:
            _E402_REPORT['changed_turns']+=1
            action=dict(action,market=out)
    except Exception:_E402_REPORT['errors']+=1
    return action

e402_agent.telemetry=_E402_REPORT
kaggle_submission_agent=e402_agent

# EXP410: do not consume another fertilizer where it cannot improve the planned harvest.
_E410_REPORT=dict(skips=0,covered=0,capped=0,errors=0)
_E410_PARENT=e402_agent
def e410_agent(observation,configuration=None):
    action=_E410_PARENT(observation,configuration)
    try:
        step=int(observation['step']);seat=int(observation['player']);day=step//24
        if step==0:
            for k in _E410_REPORT:_E410_REPORT[k]=0
        units=[action.get('farmer') or ['PASS']]+list(action.get('hands') or [])
        if not any(c==['FERTILIZE'] for c in units):return action
        farm,private=_PLANNER_NS['_clone_state'](observation['farms'][seat],observation['private'])
        positions=[farm['farmer']]+list(farm['hands']);changed=False
        native=_IMPL.chassis.players[seat]
        expected=max(len(a.get('hands',[])) for a in _v219_native_day(native,day))
        reactive=set(_R51_INPUT_STATES.get(seat,{}).get('workers',{}))
        for i,cmd in enumerate(units[:len(positions)]):
            pos=tuple(positions[i]);tile=farm['tiles'][pos[1]][pos[0]]
            if cmd==['FERTILIZE'] and isinstance(tile,dict) and tile.get('crop') in ('WHEAT','CARROT') and private['inventories'][i].get('FERTILIZER',0)>0:
                until=int(tile.get('fertilized_until_day',-1));covered=until>=day+2;skip=covered
                if not skip and i<=expected and i not in reactive:
                    visits=_ca_visits(observation,action,pos,min(718,(int(tile['planted_day'])+6)*24),start=step+1)
                    kw=dict(y0=int(tile['yield_units']),watered_day=day if tile.get('watered_today') else -1,now_step=step)
                    old=_ca_yield_path(tile['crop'],int(tile['planted_day']),visits,fert_until=until,**kw)[0]
                    new=_ca_yield_path(tile['crop'],int(tile['planted_day']),visits,fert_until=max(until,day+2),**kw)[0]
                    skip=old>0 and old==new
                if skip:
                    units[i]=cmd=['PASS'];changed=True
                    _E410_REPORT['skips']+=1;_E410_REPORT['covered' if covered else 'capped']+=1
            _PLANNER_NS['_apply_unit_action'](farm,private,i,cmd,len(farm['tiles']),day,24,100)
        if changed:return dict(action,farmer=units[0],hands=units[1:])
    except Exception:
        _E410_REPORT['errors']+=1
    return action


# Bridge: set agent to outermost v56 layer for IG to wrap
agent = e410_agent

# Final conservative closure for the productive-idle opening and market queue.
_IG_CASH = frozenset((
    "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL",
))
_IG_REPORT = {
    "opening_repairs": 0,
    "queue_changed_turns": 0,
    "zeroed_orders": 0,
    "pulled_orders": 0,
    "pulled_slots": 0,
    "errors": 0,
}


def _ig_standard(configuration):
    if configuration is None or not hasattr(configuration, "get"):
        return True
    return all(configuration.get(key, expected) == expected for key, expected in (
        ("boardSize", 10),
        ("turnsPerDay", 24),
        ("shedCapacity", 100),
        ("maxMarketOrdersPerTurn", 10),
    ))


def _ig_guard_opening(observation, action):
    """Restore the parent pasture command if the temporary wheat never exists."""
    if int(observation.get("step", -1)) != 29 or not isinstance(action, dict):
        return action
    seat = int(observation["player"])
    state = _ALT_STATE.get(seat)
    if not state or state.get("mode") != "HybridOpening":
        return action
    farm = observation["farms"][seat]
    site = farm["tiles"][4][2]
    valid_wheat = (
        isinstance(site, dict) and site.get("kind") == "PLANT" and
        site.get("crop") == "WHEAT" and int(site.get("planted_day", -1)) == 0
    )
    if valid_wheat:
        return action
    hands = [list(command) for command in (action.get("hands") or [])]
    if len(hands) <= 2 or hands[2] != ["WATER"]:
        return action
    hands[2] = ["BUILD_PASTURE"]
    _IG_REPORT["opening_repairs"] += 1
    return dict(action, hands=hands)


def _ig_close_queue(observation, action):
    """Turn final cash-sale no-ops into holes and fill holes from the right."""
    if not isinstance(action, dict):
        return action
    market = action.get("market") or []
    if len(market) < 2:
        return action
    projected = dict(projected_shed(action, FarmView(observation)))
    remaining = {item: max(0, int(projected.get(item, 0))) for item in _IG_CASH}
    revised = []
    zeroed = 0
    for raw in market:
        order = list(raw) if isinstance(raw, (list, tuple)) else raw
        if (isinstance(order, list) and len(order) >= 3 and
                order[0] == "SELL" and order[1] in _IG_CASH):
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
        movable = (
            isinstance(order, list) and len(order) >= 3 and
            order[0] == "SELL" and order[1] in _IG_CASH and int(order[2]) > 0
        )
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
    _IG_REPORT["queue_changed_turns"] += 1
    _IG_REPORT["zeroed_orders"] += zeroed
    _IG_REPORT["pulled_orders"] += pulled
    _IG_REPORT["pulled_slots"] += distance
    return dict(action, market=revised)


_IG_PARENT = agent
del agent


def agent(observation, configuration=None):
    if int(observation.get("step", 0)) == 0:
        for key in _IG_REPORT:
            _IG_REPORT[key] = 0
    action = _IG_PARENT(observation, configuration)
    try:
        if _ig_standard(configuration):
            action = _ig_guard_opening(observation, action)
            action = _ig_close_queue(observation, action)
    except Exception:
        _IG_REPORT["errors"] += 1
    return action


agent.telemetry = _IG_REPORT
agent = globals().pop("agent")

kaggle_submission_agent = agent

_LEDGER_PARENT = agent
_LEDGER_TELEMETRY = {}
def agent(observation, configuration=None):
    action = _LEDGER_PARENT(observation, configuration)
    _LEDGER_TELEMETRY.clear()
    for prefix in ("_IG", "_CL", "_E402", "_E410", "_ADV", "_ALT", "_R148"):
        report = globals().get(prefix + "_REPORT", {})
        for key, value in report.items():
            if isinstance(value, (int, float, str)):
                _LEDGER_TELEMETRY[prefix + ":" + key] = value
    return action
agent.telemetry = _LEDGER_TELEMETRY
submission_v57 = agent
