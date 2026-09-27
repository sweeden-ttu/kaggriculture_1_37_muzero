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

"""Observation and MacroState feature vector encoders for MuZero.

Tensor serialization only — no analytical teacher (``economy``) imports.
"""
from __future__ import annotations

__author__ = "Scott Weeden"
__license__ = "Apache-2.0"

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .types import (
    BASE_CROP_PRICES,
    DEFAULT_PRICES,
    LAND_BUFFERS,
    LAND_ORDER,
    LAND_PRICES,
    MacroOption,
    MacroState,
    OBS_DIM,
)

# Observation-legality SE buffer (encoding / root mask — not the Phase-1 teacher).
META_SE_EXTRA_BUFFER: float = 2500.0


def _next_land_headroom(state: MacroState) -> float:
    """Headroom ratio relative to next affordable land quadrant buffer."""
    for q in LAND_ORDER:
        if q not in state.unlocked:
            need = LAND_PRICES[q] + LAND_BUFFERS[q]
            return max(-1.0, min(1.0, (state.capital - need) / need))
    return 1.0


def wealth(state: MacroState) -> float:
    """Terminal wealth estimate from capital + shed salvage at spot prices."""
    salvage = 0.0
    for item, qty in state.shed.items():
        if qty > 0:
            salvage += qty * state.prices.get(item, DEFAULT_PRICES.get(item, 25.0))
    return state.capital + salvage


def encode_macro_state(state: MacroState) -> torch.Tensor:
    """Compact 32-dim encoding of MacroState.

    Uses ln(1 + money) and relative spot prices P_t / P_0 as in the architecture guide.
    """
    prices = state.prices or DEFAULT_PRICES
    shed = state.shed or {}
    unlocked = set(state.unlocked)

    vec: List[float] = [
        math.log1p(max(0.0, state.capital)) / 10.0,
        float(state.day) / 30.0,
        float(len(unlocked)) / 4.0,
        1.0 if "NW" in unlocked else 0.0,
        1.0 if "NE" in unlocked else 0.0,
        1.0 if "SW" in unlocked else 0.0,
        1.0 if "SE" in unlocked else 0.0,
        float(state.active_animals) / 5.0,
        sum(shed.values()) / 100.0,
        shed.get("WHEAT", 0.0) / 50.0,
        shed.get("MELON", 0.0) / 20.0,
        shed.get("CARROT", 0.0) / 40.0,
        shed.get("TOMATO", 0.0) / 20.0,
        shed.get("STRAWBERRY", 0.0) / 20.0,
        shed.get("EGG", 0.0) / 30.0,
        shed.get("MILK", 0.0) / 20.0,
        shed.get("WOOL", 0.0) / 20.0,
        prices.get("WHEAT", 25.0) / BASE_CROP_PRICES["WHEAT"],
        prices.get("MELON", 250.0) / BASE_CROP_PRICES["MELON"],
        prices.get("CARROT", 35.0) / BASE_CROP_PRICES["CARROT"],
        prices.get("TOMATO", 60.0) / BASE_CROP_PRICES["TOMATO"],
        prices.get("STRAWBERRY", 120.0) / BASE_CROP_PRICES["STRAWBERRY"],
        prices.get("EGG", 70.0) / 70.0,
        prices.get("MILK", 256.0) / 256.0,
        prices.get("WOOL", 240.0) / 240.0,
        _next_land_headroom(state),
        float(state.day >= 27),
        float(sum(shed.values()) >= 70.0),
        float(state.day <= 8),
        float(state.day <= 18),
        float(state.day <= 23),
        float(state.capital >= 5000.0),
    ]
    assert len(vec) == OBS_DIM
    return torch.tensor(vec, dtype=torch.float32)


def encode_observation(obs: Dict[str, Any]) -> torch.Tensor:
    """Maps raw Kaggriculture observation dict into the 32-dim space."""
    p_idx = int(obs.get("player", 0))
    farms = obs.get("farms", [{}, {}])
    my_farm = farms[p_idx] if isinstance(farms, list) and len(farms) > p_idx else {}
    private = obs.get("private", {})
    market = obs.get("market", {})
    prices = dict(market.get("prices", {}) or {})
    shed = dict(private.get("shed", {}) or {})
    unlocked = tuple(my_farm.get("unlocked_quadrants", ["NW"]) or ["NW"])
    day = int(obs.get("day", 0))
    money = float(my_farm.get("money", 0.0))

    state = MacroState(
        day=day,
        capital=money,
        unlocked=unlocked,
        shed={k: float(v) for k, v in shed.items()},
        prices={**DEFAULT_PRICES, **{k: float(v) for k, v in prices.items()}},
        active_animals=1,
    )
    vec = encode_macro_state(state)
    vec[31] = float(obs.get("hour", 0)) / 24.0
    return vec


def decode_macro_state(vec: torch.Tensor | Sequence[float]) -> MacroState:
    """Invert the 32-dim encoding back into a MacroState."""
    if isinstance(vec, torch.Tensor):
        v = vec.detach().cpu().flatten().tolist()
    else:
        v = list(vec)

    capital = max(0.0, math.expm1(v[0] * 10.0))
    day = max(0, min(30, int(round(v[1] * 30.0))))

    unlocked = []
    for q, idx in [("NW", 3), ("NE", 4), ("SW", 5), ("SE", 6)]:
        if idx < len(v) and v[idx] > 0.5:
            unlocked.append(q)
    if not unlocked:
        unlocked = ["NW"]

    active_animals = max(0, int(round(v[7] * 5.0))) if len(v) > 7 else 1

    shed: Dict[str, float] = {}
    shed_items = [
        ("WHEAT", 9, 50.0),
        ("MELON", 10, 20.0),
        ("CARROT", 11, 40.0),
        ("TOMATO", 12, 20.0),
        ("STRAWBERRY", 13, 20.0),
        ("EGG", 14, 30.0),
        ("MILK", 15, 20.0),
        ("WOOL", 16, 20.0),
    ]
    for name, idx, scale in shed_items:
        if idx < len(v) and v[idx] > 0.0:
            shed[name] = max(0.0, round(v[idx] * scale, 2))

    prices = dict(DEFAULT_PRICES)
    price_items = [
        ("WHEAT", 17, BASE_CROP_PRICES["WHEAT"]),
        ("MELON", 18, BASE_CROP_PRICES["MELON"]),
        ("CARROT", 19, BASE_CROP_PRICES["CARROT"]),
        ("TOMATO", 20, BASE_CROP_PRICES["TOMATO"]),
        ("STRAWBERRY", 21, BASE_CROP_PRICES["STRAWBERRY"]),
        ("EGG", 22, 70.0),
        ("MILK", 23, 256.0),
        ("WOOL", 24, 240.0),
    ]
    for name, idx, base_p in price_items:
        if idx < len(v) and v[idx] > 0.0:
            prices[name] = max(1.0, round(v[idx] * base_p, 2))

    return MacroState(
        day=day,
        capital=capital,
        unlocked=tuple(unlocked),
        shed=shed,
        prices=prices,
        active_animals=active_animals,
    )


def _next_expansion_target(state: MacroState) -> Optional[str]:
    for q in LAND_ORDER:
        if q not in state.unlocked:
            return q
    return None


def legal_macro_options(state: MacroState) -> List[MacroOption]:
    """Root / observation legality mask from unlocks + capital (encoding layer)."""
    options: List[MacroOption] = [
        MacroOption.FRONT_LOAD_SURVIVAL,
        MacroOption.EXPANSION_PREPARATION,
        MacroOption.TRICKLE_LIQUIDATION,
        MacroOption.HIRE_HANDS,
        MacroOption.PASS,
    ]
    for opt, quad in (
        (MacroOption.EXPAND_NE, "NE"),
        (MacroOption.EXPAND_SW, "SW"),
        (MacroOption.EXPAND_SE, "SE"),
    ):
        if quad in state.unlocked:
            continue
        buffer = LAND_BUFFERS[quad]
        if quad == "SE":
            buffer = LAND_BUFFERS[quad] + META_SE_EXTRA_BUFFER
        cost = LAND_PRICES[quad] + buffer
        if state.capital >= cost and state.day <= 24:
            if _next_expansion_target(state) == quad:
                options.append(opt)
    return options
