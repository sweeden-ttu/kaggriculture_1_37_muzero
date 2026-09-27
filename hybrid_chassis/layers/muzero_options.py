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

"""MuZero strategy + Kaggle chassis shop-router / pathfinder.

Roles:
  - **MuZero** (SampledMCTS): sole strategic decision maker for market / expand /
    hire / liquidation (and optional work intent).
  - **Kaggle chassis** (``_IMPL``): shop router + path finder only — selects the
    route tape from town shops and supplies farmer/hands locomotion. Chassis
    market orders from the tape are discarded so it never sets strategy.
"""
from __future__ import annotations

import os
import sys

__author__ = "Scott Weeden"
__license__ = "Apache-2.0"

_MZ_REPORT = {
    "step0_arb": 0,
    "se_denied": 0,
    "trickle_sells": 0,
    "mcts_calls": 0,
    "joint_applies": 0,
    "pathfind_calls": 0,
    "sanitized": 0,
    "errors": 0,
    "load_error": None,
    "arch": None,
    "decision_maker": "spatial_muzero",
    "chassis_role": "shop_router_pathfinder",
}

_NEML_STEP0_ORDERS = [
    ["BUY_PRODUCT", "WHEAT", 9],
    ["SELL", "WHEAT", 9],
    ["BUY_PRODUCT", "WHEAT", 11],
    ["SELL", "WHEAT", 6],
    ["BUY_SEED", "WHEAT", 1],
]

_PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}

_MZ_CHASSIS = None
_MZ_MCTS = None
_MZ_ENCODER = None
_MZ_TRANSLATE = None
_MZ_LOAD_ERROR = None

_LAND_COST = {"NE": 1000.0, "SW": 2000.0, "SE": 4000.0}
_MARKET_SELL = 2

# Farmer ops that are work intents (not locomotion) — MuZero may override path tape
_WORK_OPS = {"PLANT", "WATER", "HARVEST", "DIG", "PICKUP", "DROP", "PLACE"}


def _candidate_ckpt_paths(here: str):
    return [
        os.path.join(here, "muzero_spatial_checkpoints.pt"),
        os.path.join(here, "artifacts", "muzero_spatial_checkpoints.pt"),
        os.path.join(here, "..", "artifacts", "muzero_spatial_checkpoints.pt"),
        os.path.join(here, "..", "..", "artifacts", "muzero_spatial_checkpoints.pt"),
        os.path.join(here, "muzero_checkpoints_champion.pt"),
        os.path.join(here, "artifacts", "muzero_checkpoints_champion.pt"),
        os.path.join(here, "..", "artifacts", "muzero_checkpoints_champion.pt"),
        os.path.join(here, "muzero_checkpoints.pt"),
        os.path.join(here, "artifacts", "muzero_checkpoints.pt"),
        os.path.join(here, "..", "artifacts", "muzero_checkpoints.pt"),
    ]


def _load_spatial_stack():
    global _MZ_CHASSIS, _MZ_MCTS, _MZ_ENCODER, _MZ_TRANSLATE, _MZ_LOAD_ERROR
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    parent = os.path.dirname(here)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    import torch
    from muzero import (
        KaggricultureMuZeroChassis,
        KaggricultureObservationEncoder,
        SampledMuZeroMCTS,
        translate_joint_action,
    )
    from muzero.spatial_checkpoint import load_spatial_checkpoint

    ckpt_path = None
    for cp in _candidate_ckpt_paths(here):
        if os.path.isfile(cp):
            ckpt_path = cp
            break

    chassis = KaggricultureMuZeroChassis()
    if ckpt_path is None:
        _MZ_LOAD_ERROR = "no spatial/champion checkpoint found"
        _MZ_REPORT["load_error"] = _MZ_LOAD_ERROR
        _MZ_CHASSIS = chassis
        _MZ_MCTS = SampledMuZeroMCTS(chassis, num_samples=4, num_simulations=8)
        _MZ_ENCODER = KaggricultureObservationEncoder()
        _MZ_TRANSLATE = translate_joint_action
        _MZ_REPORT["arch"] = "spatial_untrained"
        return

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = blob.get("arch") if isinstance(blob, dict) else None
    if arch is not None and arch not in ("KaggricultureMuZeroChassis", "spatial"):
        raise RuntimeError(
            f"MuZero graft refused macro/foreign checkpoint {ckpt_path!r} (arch={arch!r}). "
            "Promote a spatial Sampled MuZero checkpoint before packaging."
        )

    load_spatial_checkpoint(chassis, ckpt_path, strict=False)
    chassis.eval()
    sims = int(os.environ.get("MUZERO_SPATIAL_SIMS", "50"))
    samples = int(os.environ.get("MUZERO_SPATIAL_SAMPLES", "8"))
    _MZ_CHASSIS = chassis
    _MZ_MCTS = SampledMuZeroMCTS(chassis, num_samples=samples, num_simulations=sims)
    _MZ_ENCODER = KaggricultureObservationEncoder()
    _MZ_TRANSLATE = translate_joint_action
    _MZ_REPORT["arch"] = "spatial"
    _MZ_REPORT["ckpt"] = ckpt_path


try:
    _load_spatial_stack()
except Exception as e:
    _MZ_LOAD_ERROR = str(e)
    _MZ_REPORT["load_error"] = _MZ_LOAD_ERROR
    _MZ_CHASSIS = None
    _MZ_MCTS = None


# Discard layered heuristic *policy* agents; keep ``_IMPL`` for shop routing / pathfinding.
if "agent" in globals():
    globals().pop("agent")


def _blank_action():
    return {"farmer": ["PASS"], "hands": [], "market": []}


def _chassis_pathfind(observation, configuration=None):
    """Shop router + path finder: farmer/hands from route tape only (no market)."""
    impl = globals().get("_IMPL")
    if impl is None:
        return _blank_action()
    try:
        chassis = getattr(impl, "chassis", None)
        if chassis is not None:
            raw = chassis.act(observation, configuration)
            seat = int(observation.get("player", 0))
            route_id = (chassis.players.get(seat) or {}).get("route")
        else:
            raw = impl(observation, configuration)
            route_id = None
        _MZ_REPORT["pathfind_calls"] += 1
        return {
            "farmer": list(raw.get("farmer") or ["PASS"]),
            "hands": [list(h) for h in (raw.get("hands") or [])],
            "market": [],  # never take strategic market from chassis tape
            "_chassis_route": route_id,
        }
    except Exception:
        _MZ_REPORT["errors"] += 1
        return _blank_action()


def _sanitize_legal(action, observation):
    """Legality rails only: clamp sells to shed, drop unaffordable/illegal land buys."""
    seat = int(observation.get("player", 0))
    farms = observation.get("farms", [{}, {}])
    farm = farms[seat] if seat < len(farms) else {}
    money = float(farm.get("money", 0.0) or 0.0)
    unlocked = set(farm.get("unlocked_quadrants") or ["NW"])
    shed = dict((observation.get("private") or {}).get("shed") or {})

    out = dict(action)
    orders = []
    changed = False
    for o in (action.get("market") or []):
        if not o:
            continue
        op = str(o[0]).upper()
        if op in ("SELL", "SELL_ITEM") and len(o) >= 2:
            item = o[1]
            have = int(shed.get(item, 0) or 0)
            qty = int(o[2]) if len(o) >= 3 else have
            qty = max(0, min(qty, have))
            if qty <= 0:
                changed = True
                continue
            if len(o) >= 3 and qty != int(o[2]):
                changed = True
            orders.append([op, item, qty])
        elif op == "BUY_LAND":
            target = str(o[1]).upper() if len(o) > 1 else ""
            cost = _LAND_COST.get(target)
            if cost is None or target in unlocked or money < cost:
                changed = True
                if target == "SE":
                    _MZ_REPORT["se_denied"] += 1
                continue
            orders.append(["BUY_LAND", target])
        else:
            orders.append(list(o))
        if len(orders) >= 10:
            break

    # Drop non-env PRIORITY tokens (not pathfinder hand lists)
    hands = action.get("hands") or []
    clean_hands = []
    for h in hands:
        if isinstance(h, (list, tuple)) and h and str(h[0]).upper() == "PRIORITY":
            changed = True
            continue
        clean_hands.append(list(h) if isinstance(h, (list, tuple)) else h)

    out["market"] = orders[:10]
    out["hands"] = clean_hands
    if "farmer" not in out or not out["farmer"]:
        out["farmer"] = ["PASS"]
        changed = True
    if changed:
        _MZ_REPORT["sanitized"] += 1
    return out


def _merge_path_and_strategy(path_action, strategy_action):
    """Chassis supplies locomotion; MuZero supplies market (+ work intent if any)."""
    farmer = list(path_action.get("farmer") or ["PASS"])
    hands = [list(h) for h in (path_action.get("hands") or [])]
    mz_farmer = list(strategy_action.get("farmer") or ["PASS"])
    # If MuZero issues a work intent, prefer it over a tape PASS (pathfinder still
    # owns MOVE chains from the shop route).
    if mz_farmer and str(mz_farmer[0]).upper() in _WORK_OPS:
        if not farmer or str(farmer[0]).upper() in ("PASS",):
            farmer = mz_farmer
    return {
        "farmer": farmer,
        "hands": hands,
        "market": [list(o) for o in (strategy_action.get("market") or []) if o],
        "_muzero_joint": strategy_action.get("_muzero_joint"),
        "_decision_maker": "spatial_muzero",
        "_chassis_role": "shop_router_pathfinder",
    }


def agent(observation, configuration=None):
    """MuZero decides strategy; chassis only shop-routes and pathfinds."""
    try:
        step = int(observation.get("step", 0))
        path = _chassis_pathfind(observation, configuration)

        # Fixed opening market script; farmer/hands still from shop route pathfinder
        if step == 0:
            strategy = dict(_blank_action(), market=list(_NEML_STEP0_ORDERS))
            _MZ_REPORT["step0_arb"] += 1
            return _sanitize_legal(_merge_path_and_strategy(path, strategy), observation)

        if _MZ_MCTS is None or _MZ_ENCODER is None or _MZ_TRANSLATE is None:
            _MZ_REPORT["errors"] += 1
            # Pathfinder-only fallback (no strategic market)
            return _sanitize_legal(dict(path), observation)

        obs_t = _MZ_ENCODER.encode(observation)
        if obs_t.dim() == 3:
            obs_t = obs_t.unsqueeze(0)
        best, _probs, _val = _MZ_MCTS.search(obs_t, add_dirichlet_noise=False)
        _MZ_REPORT["mcts_calls"] += 1
        f_act, h_act, m_act = int(best[0]), int(best[1]), int(best[2])

        strategy = _MZ_TRANSLATE(
            f_act, h_act, m_act, observation, base_action=_blank_action()
        )
        strategy["_muzero_joint"] = (f_act, h_act, m_act)
        _MZ_REPORT["joint_applies"] += 1
        if m_act == _MARKET_SELL:
            _MZ_REPORT["trickle_sells"] += 1

        return _sanitize_legal(_merge_path_and_strategy(path, strategy), observation)
    except Exception:
        _MZ_REPORT["errors"] += 1
        return dict(_PASS_ACTION)


agent.telemetry = _MZ_REPORT
kaggle_submission_agent = agent
