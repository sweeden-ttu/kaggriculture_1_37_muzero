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

"""Factorized joint-action ↔ Kaggle action translation (shipped with muzero/)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .spatial_constants import (
    NUM_FARMER_ACTIONS,
    NUM_HAND_ASSIGNMENTS,
    NUM_MARKET_ORDERS,
)

_FARMER_PASS = 0
_MARKET_NOOP = 0
_MARKET_HIRE = 1
_MARKET_SELL = 2
_MARKET_EXPAND_NE = 3
_MARKET_EXPAND_SW = 4
_MARKET_EXPAND_SE = 5

_PLANT_MAP = {
    "WHEAT": 7,
    "CARROT": 8,
    "TOMATO": 9,
    "STRAWBERRY": 10,
    "MELON": 11,
}
_MOVE_MAP = {"N": 1, "E": 2, "S": 3, "W": 4}


def translate_joint_action(
    farmer: int,
    hands: int,
    market: int,
    observation: Dict[str, Any],
    base_action: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Map a factorized (farmer, hands, market) joint action to a Kaggle action dict."""
    action: Dict[str, Any] = dict(base_action) if base_action else {
        "farmer": ["PASS"],
        "hands": [],
        "market": [],
    }
    market_orders: List[List[Any]] = [list(o) for o in (action.get("market") or []) if o]
    seat = int(observation.get("player", 0))
    farms = observation.get("farms", [{}, {}])
    farm = farms[seat] if seat < len(farms) else {}
    unlocked = set(farm.get("unlocked_quadrants") or ["NW"])
    private = observation.get("private", {}) or {}
    shed = dict(private.get("shed", {}) or {})
    money = float(farm.get("money", 0.0))

    f_act = int(farmer) % NUM_FARMER_ACTIONS
    h_act = int(hands) % NUM_HAND_ASSIGNMENTS
    m_act = int(market) % NUM_MARKET_ORDERS

    if f_act == _FARMER_PASS:
        action["farmer"] = ["PASS"]
    elif f_act in (1, 2, 3, 4):
        action["farmer"] = ["MOVE", ["N", "E", "S", "W"][f_act - 1]]
    elif f_act == 5:
        action["farmer"] = ["WATER"]
    elif f_act == 6:
        action["farmer"] = ["HARVEST"]
    elif f_act == 7:
        action["farmer"] = ["PLANT", "WHEAT"]
    elif f_act == 8:
        action["farmer"] = ["PLANT", "CARROT"]
    elif f_act == 9:
        action["farmer"] = ["PLANT", "TOMATO"]
    elif f_act == 10:
        action["farmer"] = ["PLANT", "STRAWBERRY"]
    elif f_act == 11:
        action["farmer"] = ["PLANT", "MELON"]
    else:
        action["farmer"] = ["PASS"]

    action["hands"] = [["PRIORITY", int(h_act)]] if h_act > 0 else list(action.get("hands") or [])

    if m_act == _MARKET_HIRE:
        market_orders.append(["HIRE"])
    elif m_act == _MARKET_SELL:
        already = {o[1] for o in market_orders if len(o) >= 2 and o[0] == "SELL"}
        for item in ("WOOL", "MILK", "STRAWBERRY", "MELON", "EGG", "TOMATO"):
            qty = int(shed.get(item, 0) or 0)
            if qty >= 1 and item not in already and len(market_orders) < 10:
                market_orders.append(["SELL", item, min(qty, 8)])
                already.add(item)
    elif m_act == _MARKET_EXPAND_NE and "NE" not in unlocked and money >= 1000:
        market_orders = [o for o in market_orders if not (o and o[0] == "BUY_LAND")]
        market_orders.append(["BUY_LAND", "NE"])
    elif m_act == _MARKET_EXPAND_SW and "SW" not in unlocked and money >= 2000:
        market_orders = [o for o in market_orders if not (o and o[0] == "BUY_LAND")]
        market_orders.append(["BUY_LAND", "SW"])
    elif m_act == _MARKET_EXPAND_SE and "SE" not in unlocked and money >= 4000:
        market_orders = [o for o in market_orders if not (o and o[0] == "BUY_LAND")]
        market_orders.append(["BUY_LAND", "SE"])

    action["market"] = market_orders[:10]
    action["_muzero_joint"] = (f_act, h_act, m_act)
    return action


def label_expert_joint_action(actions: Dict[str, Any]) -> Tuple[int, int, int]:
    """Map a Kaggle expert action dict to factorized (farmer, hands, market) indices."""
    farmer_cmd = actions.get("farmer") or ["PASS"]
    if isinstance(farmer_cmd, list) and farmer_cmd:
        op = str(farmer_cmd[0]).upper()
        if op == "MOVE" and len(farmer_cmd) > 1:
            f_act = _MOVE_MAP.get(str(farmer_cmd[1]).upper(), _FARMER_PASS)
        elif op == "WATER":
            f_act = 5
        elif op == "HARVEST":
            f_act = 6
        elif op == "PLANT" and len(farmer_cmd) > 1:
            f_act = _PLANT_MAP.get(str(farmer_cmd[1]).upper(), _FARMER_PASS)
        else:
            f_act = _FARMER_PASS
    else:
        f_act = _FARMER_PASS

    hands = actions.get("hands") or []
    if not hands:
        h_act = 0
    else:
        # Bucket by first hand opcode for a stable discrete label.
        first = hands[0] if isinstance(hands[0], (list, tuple)) else ["IDLE"]
        token = str(first[0]).upper() if first else "IDLE"
        h_act = (hash(token) % (NUM_HAND_ASSIGNMENTS - 1)) + 1

    market = actions.get("market") or []
    m_act = _MARKET_NOOP
    for ma in market:
        if not isinstance(ma, (list, tuple)) or not ma:
            continue
        op = str(ma[0]).upper()
        if op == "BUY_LAND":
            target = str(ma[1]).upper() if len(ma) > 1 else ""
            m_act = {"NE": _MARKET_EXPAND_NE, "SW": _MARKET_EXPAND_SW, "SE": _MARKET_EXPAND_SE}.get(
                target, _MARKET_NOOP
            )
            break
        if op == "HIRE":
            m_act = _MARKET_HIRE
            break
        if op in ("SELL", "SELL_ITEM"):
            item = str(ma[1]).upper() if len(ma) > 1 else ""
            if item in {"WOOL", "MILK", "STRAWBERRY", "MELON", "EGG", "EGGS", "TOMATO"}:
                m_act = _MARKET_SELL
                break

    return (
        int(f_act) % NUM_FARMER_ACTIONS,
        int(h_act) % NUM_HAND_ASSIGNMENTS,
        int(m_act) % NUM_MARKET_ORDERS,
    )
