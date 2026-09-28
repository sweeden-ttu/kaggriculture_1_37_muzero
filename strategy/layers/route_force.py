# ---------------------------------------------------------------------------
# ROUTE FORCE (arena opponent variant): pin the shop-router's tape choice to
# _RF_ROUTE for days 6-26 so the champion can play as any of its 41 tapes.
# Used only to build a diverse local opponent pool; never submitted.
# ---------------------------------------------------------------------------
_RF_ROUTE = 0
_RF_ORIGINAL_ROUTER = _IMPL.chassis.router


def _rf_router(observation, step, state):
    route = _RF_ORIGINAL_ROUTER(observation, step, state)
    if 144 <= step < 648 and _RF_ROUTE in _IMPL.chassis.routes:
        state["route"] = _RF_ROUTE
        return _RF_ROUTE
    return route


_IMPL.chassis.router = _rf_router
