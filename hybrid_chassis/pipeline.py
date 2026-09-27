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

"""Canonical chassis pipeline: state-router + action mask + macro-option translator.

The hybrid chassis is no longer a heuristic decision-maker. This module:
  1. Routes observations through shop/opening route tables (state routing).
  2. Applies survival action masks (invalid logits → -inf).
  3. Translates MuZero macro-options into raw Kaggle environment actions.

Legacy reactive layer stacks remain available via ``build_legacy_pipeline()``
for packaging / ablation only.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .core.constants import (
    ANIMAL_COST,
    ANIMAL_STRUCTURE,
    DEFAULT_SETTINGS,
    FRONT_RUN_ITEMS,
    LAND_PRICES,
    LAST_ACT_STEP,
    MOVES,
    PASS_ACTION,
    PRODUCTS,
    SEED_PRICE,
)
from .core.helpers import (
    _fib,
    _get,
    _int,
    _is_noop,
    _shed_adjacent,
    _step_of,
    _tile_at,
)
from .core.view import FarmView, _View
from .core.chassis import Chassis, make_agent
from .routes.data import load_routes
from .routes.router import _SETTINGS, apply_opening_patch, route_selector
from .layers.manifest import layer_filenames
from .layers import action_masking

LAYERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "layers")

# MacroOption indices (aligned with muzero.types.MacroOption)
MACRO_FRONT_LOAD = 0
MACRO_EXPANSION_PREP = 1
MACRO_TRICKLE = 2
MACRO_EXPAND_NE = 3
MACRO_EXPAND_SW = 4
MACRO_EXPAND_SE = 5
MACRO_HIRE = 6
MACRO_PASS = 7


def translate_macro_option(
    option: int,
    observation: Dict[str, Any],
    base_action: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Map a MuZero macro-option index onto raw Kaggle farmer/market actions."""
    action: Dict[str, Any] = dict(base_action) if base_action else {
        "farmer": ["PASS"],
        "hands": [],
        "market": [],
    }
    market: List[List[Any]] = [list(o) for o in (action.get("market") or []) if o]
    seat = int(observation.get("player", 0))
    farms = observation.get("farms", [{}, {}])
    farm = farms[seat] if seat < len(farms) else {}
    unlocked = set(farm.get("unlocked_quadrants") or ["NW"])
    private = observation.get("private", {}) or {}
    shed = dict(private.get("shed", {}) or {})
    money = float(farm.get("money", 0.0))

    opt = int(option)
    if opt == MACRO_EXPAND_NE and "NE" not in unlocked and money >= 1000:
        market = [o for o in market if not (o and o[0] == "BUY_LAND")]
        market.append(["BUY_LAND", "NE"])
    elif opt == MACRO_EXPAND_SW and "SW" not in unlocked and money >= 2000:
        market = [o for o in market if not (o and o[0] == "BUY_LAND")]
        market.append(["BUY_LAND", "SW"])
    elif opt == MACRO_EXPAND_SE and "SE" not in unlocked and money >= 4000:
        market = [o for o in market if not (o and o[0] == "BUY_LAND")]
        market.append(["BUY_LAND", "SE"])
    elif opt == MACRO_HIRE:
        market.append(["HIRE"])
    elif opt == MACRO_TRICKLE:
        already = {o[1] for o in market if len(o) >= 2 and o[0] == "SELL"}
        for item in ("WOOL", "MILK", "STRAWBERRY", "MELON", "EGG", "TOMATO"):
            qty = int(shed.get(item, 0))
            if qty >= 1 and item not in already and len(market) < 10:
                market.append(["SELL", item, min(qty, 8)])
                already.add(item)
    elif opt == MACRO_FRONT_LOAD:
        # Survival: prefer watering / planting signal via empty market + PASS farmer
        # (downstream resource guards may still inject WATER when grafted).
        pass
    elif opt == MACRO_EXPANSION_PREP:
        # Hold capital: strip speculative buys, keep sells
        market = [o for o in market if not (o and o[0] in ("BUY_LAND", "HIRE", "BUY_ANIMAL"))]
    elif opt == MACRO_PASS:
        market = [o for o in market if o and o[0] == "SELL"]

    action["market"] = market
    action["_muzero_macro_option"] = opt
    return action


def translate_joint_action(
    farmer: int,
    hands: int,
    market: int,
    observation: Dict[str, Any],
    base_action: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Map a factorized (farmer, hands, market) joint action to a Kaggle action dict."""
    from muzero.action_translate import translate_joint_action as _translate

    return _translate(farmer, hands, market, observation, base_action=base_action)


def select_macro_under_mask(
    logits: Sequence[float],
    observation: Dict[str, Any],
) -> Tuple[int, List[bool], List[float]]:
    """Apply survival mask (−∞ logits) and return (best_option, mask, probs)."""
    mask = action_masking.build_action_mask(observation)
    masked_logits = action_masking.apply_logit_mask(list(logits), mask)
    probs = action_masking.masked_softmax_probs(masked_logits, mask)
    best = max(range(len(probs)), key=lambda i: probs[i])
    return best, mask, probs


def build_pipeline(
    *,
    legacy_layers: bool = False,
) -> Tuple[Callable[[Any, Any], Dict[str, Any]], Any]:
    """Build the canonical router + mask + macro translator pipeline.

    Args:
        legacy_layers: When True, also exec the full heuristic layer stack
            (packaging / ablation only). Default False for the reduced chassis.
    """
    routes, shop_routes = load_routes()
    apply_opening_patch(routes)

    def router(obs, step, state):
        return route_selector(obs, step, state, shop_routes)

    impl = make_agent(routes, router=router, **_SETTINGS)
    impl_any: Any = impl
    impl_any.chassis.diagnostics["terminal_rescue_errors"] = 0
    impl_any.chassis.diagnostics["action_masks_applied"] = 0

    def projected_shed(action, view):
        return impl_any.chassis._projected_shed(action, view)

    # Base agent from routes only (no heuristic planning)
    base_agent = impl_any

    def kaggle_submission_agent(observation, configuration=None):
        """MuZero decides strategy; chassis shop-routes and pathfinds units."""
        mask = action_masking.build_action_mask(observation)
        impl_any.chassis.diagnostics["action_masks_applied"] = (
            int(impl_any.chassis.diagnostics.get("action_masks_applied", 0)) + 1
        )

        blank = {"farmer": ["PASS"], "hands": [], "market": []}
        # Pathfind via shop router (farmer/hands only)
        try:
            path_raw = impl_any.chassis.act(observation, configuration)
            path_farmer = list(path_raw.get("farmer") or ["PASS"])
            path_hands = [list(h) for h in (path_raw.get("hands") or [])]
        except Exception:
            path_farmer, path_hands = ["PASS"], []

        spatial = getattr(impl_any, "spatial_mcts", None)
        if spatial is not None:
            try:
                from muzero import encode_spatial_observation, translate_joint_action

                obs_t = encode_spatial_observation(observation)
                if obs_t.dim() == 3:
                    obs_t = obs_t.unsqueeze(0)
                best, _, _ = spatial.search(obs_t, add_dirichlet_noise=False)
                strategy = translate_joint_action(
                    int(best[0]), int(best[1]), int(best[2]), observation, base_action=blank
                )
                action = {
                    "farmer": path_farmer,
                    "hands": path_hands,
                    "market": list(strategy.get("market") or []),
                    "_muzero_action_mask": mask,
                    "_decision_maker": "spatial_muzero",
                    "_chassis_role": "shop_router_pathfinder",
                }
                return action
            except Exception:
                pass

        option = MACRO_PASS
        planner = getattr(impl_any, "muzero_planner", None)
        if planner is not None:
            try:
                seat = int(observation.get("player", 0))
                farms = observation.get("farms", [{}, {}])
                farm = farms[seat] if seat < len(farms) else {}
                result = planner.search(
                    money=float(farm.get("money", 0.0)),
                    day=int(observation.get("day", 0)),
                    unlocked=list(farm.get("unlocked_quadrants") or ["NW"]),
                    shed=dict((observation.get("private") or {}).get("shed") or {}),
                    n_simulations=getattr(planner, "n_simulations", 16),
                )
                option = int(result.best_option)
            except Exception:
                option = MACRO_PASS

        if option < 0 or option >= len(mask) or not mask[option]:
            for i, ok in enumerate(mask):
                if ok:
                    option = i
                    break

        strategy = translate_macro_option(option, observation, base_action=blank)
        action = {
            "farmer": path_farmer,
            "hands": path_hands,
            "market": list(strategy.get("market") or []),
            "_muzero_action_mask": mask,
            "_decision_maker": "macro_muzero" if planner is not None else "masked_pass",
            "_chassis_role": "shop_router_pathfinder",
        }
        return action

    if legacy_layers:
        # Optional full heuristic stack for packaging compatibility
        context: Dict[str, Any] = {
            "copy": copy,
            "json": json,
            "_IMPL": impl,
            "FarmView": FarmView,
            "_View": _View,
            "projected_shed": projected_shed,
            "make_agent": make_agent,
            "Chassis": Chassis,
            "PRODUCTS": PRODUCTS,
            "SEED_PRICE": SEED_PRICE,
            "ANIMAL_COST": ANIMAL_COST,
            "ANIMAL_STRUCTURE": ANIMAL_STRUCTURE,
            "LAND_PRICES": LAND_PRICES,
            "MOVES": MOVES,
            "FRONT_RUN_ITEMS": FRONT_RUN_ITEMS,
            "LAST_ACT_STEP": LAST_ACT_STEP,
            "PASS_ACTION": PASS_ACTION,
            "DEFAULT_SETTINGS": DEFAULT_SETTINGS,
            "_get": _get,
            "_int": _int,
            "_fib": _fib,
            "_step_of": _step_of,
            "_shed_adjacent": _shed_adjacent,
            "_tile_at": _tile_at,
            "_is_noop": _is_noop,
            "agent": kaggle_submission_agent,
        }
        for fname in layer_filenames():
            fpath = os.path.join(LAYERS_DIR, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                code = f.read()
            layer_ctx = dict(context)
            layer_ctx["__name__"] = f"hybrid_chassis.layers.{fname[:-3]}"
            exec(code, layer_ctx)
            context.update(layer_ctx)
        return context["kaggle_submission_agent"] if "kaggle_submission_agent" in context else context["agent"], impl

    return kaggle_submission_agent, impl


def build_legacy_pipeline() -> Tuple[Callable[[Any, Any], Dict[str, Any]], Any]:
    """Full heuristic layer stack (packaging / ablation)."""
    return build_pipeline(legacy_layers=True)
