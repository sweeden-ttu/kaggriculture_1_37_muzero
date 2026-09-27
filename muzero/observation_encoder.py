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

"""
=============================================================================
muzero/observation_encoder.py - Fixed Maximum-Grid 10x10 Observation Encoder
=============================================================================
Encodes raw Kaggriculture game observation dictionaries into fixed-size
maximum-grid (10x10) spatial feature tensors [Batch=1, C=28, H=10, W=10]:
  - Channel 0: Unlocked quadrant availability mask (NW, NE, SW, SE).
  - Channels 1..8: Tile entities (Empty, Weed, Wheat, Carrot, Tomato, Strawberry, Melon, Structure).
  - Channels 9..13: Crop growth & health (Watered today, Consecutive unwatered, Yield, Lifespan, Fertilized).
  - Channels 14..17: Livestock states (Animal present, Fed today, Consecutive unfed, Care bonus).
  - Channels 18..19: Unit positions (Farmer position, Hired hands positions).
  - Channels 20..27: Global context broadcast across 10x10 (Day, Hour, Log cash, Shed ratio, Market prices).
Also provides legal action masking across factorized heads:
  - Farmer head: 15 actions
  - Hands head: 32 assignments
  - Market head: 20 orders
=============================================================================
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch


QUAD_SLICES = {
    "NW": (slice(0, 5), slice(0, 5)),
    "NE": (slice(0, 5), slice(5, 10)),
    "SW": (slice(5, 10), slice(0, 5)),
    "SE": (slice(5, 10), slice(5, 10)),
}

CROP_CHANNEL_MAP = {
    "WHEAT": 3,
    "CARROT": 4,
    "TOMATO": 5,
    "STRAWBERRY": 6,
    "MELON": 7,
}

MAX_CROP_YIELDS = {
    "WHEAT": 2.0,
    "CARROT": 2.0,
    "TOMATO": 3.0,
    "STRAWBERRY": 4.0,
    "MELON": 1.0,
}

MAX_CROP_LIFESPANS = {
    "WHEAT": 8.0,
    "CARROT": 8.0,
    "TOMATO": 10.0,
    "STRAWBERRY": 12.0,
    "MELON": 14.0,
}


class KaggricultureObservationEncoder:
    """
    Fixed Maximum-Grid Spatial Feature Encoder for Kaggriculture.
    Encodes 10x10 boards into a 28-channel spatial tensor.
    """

    def __init__(self, board_size: int = 10, num_channels: int = 28):
        self.board_size = board_size
        self.num_channels = num_channels

    def encode(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Encodes an environment observation dict into a [1, 28, 10, 10] float tensor."""
        tensor = np.zeros((self.num_channels, self.board_size, self.board_size), dtype=np.float32)

        player_idx = int(obs.get("player", 0))
        farms = obs.get("farms", [{}, {}])
        farm = farms[player_idx] if player_idx < len(farms) else {}

        # ------------------------------------------------------------------
        # Channel 0: Unlocked Quadrant Mask Channel
        # ------------------------------------------------------------------
        unlocked = farm.get("unlocked_quadrants") or ["NW"]
        for q in unlocked:
            if q in QUAD_SLICES:
                r, c = QUAD_SLICES[q]
                tensor[0, r, c] = 1.0

        # ------------------------------------------------------------------
        # Channels 1..8: Tile Entity Channels & Channels 9..13: Crop Health
        # ------------------------------------------------------------------
        tiles = farm.get("tiles") or []
        for y in range(min(self.board_size, len(tiles))):
            row = tiles[y] if isinstance(tiles[y], (list, tuple)) else []
            for x in range(min(self.board_size, len(row))):
                tile = row[x]
                if tile == "LOCKED" or tile is None or not isinstance(tile, dict):
                    continue

                # Quadrant unlocked
                is_unlocked = tensor[0, y, x] > 0.5
                if not is_unlocked:
                    continue

                crop_type = tile.get("crop")
                structure = tile.get("structure")
                is_weed = tile.get("weed", False) or tile.get("type") == "WEED"

                if structure:
                    tensor[8, y, x] = 1.0  # Structure channel
                elif is_weed:
                    tensor[2, y, x] = 1.0  # Weed channel
                elif crop_type and str(crop_type).upper() in CROP_CHANNEL_MAP:
                    c_idx = CROP_CHANNEL_MAP[str(crop_type).upper()]
                    tensor[c_idx, y, x] = 1.0  # Specific crop type

                    # Crop growth & health features
                    if tile.get("watered_today", False):
                        tensor[9, y, x] = 1.0
                    unwatered_days = float(tile.get("consecutive_unwatered", 0))
                    tensor[10, y, x] = min(1.0, unwatered_days / 2.0)

                    yield_avail = float(tile.get("yield_available", 0.0))
                    max_y = MAX_CROP_YIELDS.get(str(crop_type).upper(), 2.0)
                    tensor[11, y, x] = min(1.0, yield_avail / max_y)

                    age = float(tile.get("age", 0.0))
                    max_l = MAX_CROP_LIFESPANS.get(str(crop_type).upper(), 10.0)
                    tensor[12, y, x] = min(1.0, age / max_l)

                    fert_remaining = float(tile.get("fertilized_days_remaining", 0))
                    tensor[13, y, x] = min(1.0, fert_remaining / 3.0)
                else:
                    tensor[1, y, x] = 1.0  # Empty arable land

        # ------------------------------------------------------------------
        # Channels 14..17: Livestock States
        # ------------------------------------------------------------------
        animals = farm.get("animals") or []
        if isinstance(animals, dict):
            # Aggregated dict of animal counts: e.g. {"COW": 2, "SHEEP": 1}
            animal_count = sum(int(v) for v in animals.values() if isinstance(v, (int, float)))
            if animal_count > 0:
                # Place density over structure / pasture tiles or NW quadrant
                pasture_mask = (tensor[8] > 0.5)
                if not np.any(pasture_mask):
                    pasture_mask = (tensor[0] > 0.5)
                tensor[14][pasture_mask] = min(1.0, float(animal_count) / 10.0)
                tensor[15][pasture_mask] = 1.0  # Assumed fed in aggregate representation
        elif isinstance(animals, list):
            for anim in animals:
                if isinstance(anim, dict):
                    ax = int(anim.get("x", 0))
                    ay = int(anim.get("y", 0))
                    if 0 <= ax < self.board_size and 0 <= ay < self.board_size:
                        tensor[14, ay, ax] = 1.0
                        if anim.get("fed_today", False):
                            tensor[15, ay, ax] = 1.0
                        unfed = float(anim.get("consecutive_unfed", 0))
                        tensor[16, ay, ax] = min(1.0, unfed / 2.0)
                        if anim.get("pending_care", False):
                            tensor[17, ay, ax] = 1.0

        # ------------------------------------------------------------------
        # Channels 18..19: Unit Positions
        # ------------------------------------------------------------------
        farmer = farm.get("farmer") or {}
        if isinstance(farmer, dict):
            fx = int(farmer.get("x", 0))
            fy = int(farmer.get("y", 0))
            if 0 <= fx < self.board_size and 0 <= fy < self.board_size:
                tensor[18, fy, fx] = 1.0

        hands = farm.get("hands") or []
        if isinstance(hands, list):
            for hand in hands:
                if isinstance(hand, dict):
                    hx = int(hand.get("x", 0))
                    hy = int(hand.get("y", 0))
                    if 0 <= hx < self.board_size and 0 <= hy < self.board_size:
                        tensor[19, hy, hx] += 1.0
        elif isinstance(hands, (int, float)):
            # Broadcast hands count if spatial coords are not separate
            tensor[19, :, :] = float(hands) / 12.0

        # ------------------------------------------------------------------
        # Channels 20..27: Global Context Broadcast across 10x10
        # ------------------------------------------------------------------
        step = int(obs.get("step", 0))
        day = int(obs.get("day", step // 24))
        hour = int(obs.get("hour", step % 24))
        money = float(farm.get("money", 0.0))

        private = obs.get("private", {}) or {}
        shed = dict(private.get("shed", {}) or farm.get("shed", {}) or {})
        shed_total = sum(float(v) for v in shed.values() if isinstance(v, (int, float)))

        market_prices = dict(obs.get("market_prices", {}) or {})
        market = obs.get("market", {}) or {}
        if isinstance(market, dict):
            market_prices = {**market_prices, **dict(market.get("prices", {}) or {})}

        day_norm = float(day) / 30.0
        hour_norm = float(hour) / 24.0
        cash_log = math.log10(max(1.0, money)) / 5.0
        shed_ratio = min(1.0, shed_total / 100.0)

        p_wheat = float(market_prices.get("WHEAT", 10.0)) / 10.0
        p_carrot = float(market_prices.get("CARROT", 15.0))
        p_tomato = float(market_prices.get("TOMATO", 15.0))
        p_c_t = (p_carrot + p_tomato) / 30.0

        p_straw = float(market_prices.get("STRAWBERRY", 25.0))
        p_melon = float(market_prices.get("MELON", 35.0))
        p_s_m = (p_straw + p_melon) / 60.0

        p_milk = float(market_prices.get("MILK", 25.0))
        p_wool = float(market_prices.get("WOOL", 25.0))
        p_egg = float(market_prices.get("EGG", 10.0))
        p_anim = (p_milk + p_wool + p_egg) / 60.0

        tensor[20, :, :] = day_norm
        tensor[21, :, :] = hour_norm
        tensor[22, :, :] = cash_log
        tensor[23, :, :] = shed_ratio
        tensor[24, :, :] = p_wheat
        tensor[25, :, :] = p_c_t
        tensor[26, :, :] = p_s_m
        tensor[27, :, :] = p_anim

        return torch.from_numpy(tensor).unsqueeze(0)

    def get_action_mask(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Constructs legal action masks across factorized sub-heads:
          - farmer: [15] float tensor (1.0 = legal, 0.0 = illegal)
          - hands:  [32] float tensor
          - market: [20] float tensor
        """
        player_idx = int(obs.get("player", 0))
        farms = obs.get("farms", [{}, {}])
        farm = farms[player_idx] if player_idx < len(farms) else {}
        money = float(farm.get("money", 0.0))
        unlocked = farm.get("unlocked_quadrants") or ["NW"]

        # 1. Farmer Action Mask (15 actions: e.g. movement, plant, water, harvest, pass)
        mask_f = torch.ones(15, dtype=torch.float32)

        # 2. Hands Assignment Mask (32 assignments: dispatch modes, queues)
        mask_h = torch.ones(32, dtype=torch.float32)

        # 3. Market Orders Mask (20 orders: land expansions, buying/selling)
        mask_m = torch.ones(20, dtype=torch.float32)

        # Land Expansion gating
        # Market orders 3, 4, 5 represent EXPAND_NE, EXPAND_SW, EXPAND_SE
        if "NE" in unlocked or money < 1000.0:
            mask_m[3] = 0.0
        if "SW" in unlocked or "NE" not in unlocked or money < 2000.0:
            mask_m[4] = 0.0
        if "SE" in unlocked or "SW" not in unlocked or money < 4000.0:
            mask_m[5] = 0.0

        return {
            "farmer": mask_f,
            "hands": mask_h,
            "market": mask_m,
        }


_DEFAULT_ENCODER = KaggricultureObservationEncoder()


def encode_spatial_observation(obs: Dict[str, Any]) -> torch.Tensor:
    """Encode a Kaggriculture obs dict to ``[28, 10, 10]`` (no batch dim)."""
    batched = _DEFAULT_ENCODER.encode(obs)
    return batched.squeeze(0)

