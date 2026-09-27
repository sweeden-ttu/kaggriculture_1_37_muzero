"""Core constants and game configuration for the Kaggriculture replay chassis."""
from __future__ import annotations

from typing import Dict, Tuple

PRODUCTS: Tuple[str, ...] = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"
)

SEED_PRICE: Dict[str, int] = {
    "WHEAT": 10,
    "CARROT": 20,
    "TOMATO": 50,
    "STRAWBERRY": 100,
    "MELON": 80,
}

ANIMAL_COST: Dict[str, int] = {
    "GOOSE": 300,
    "COW": 400,
    "SHEEP": 500,
}

ANIMAL_STRUCTURE: Dict[str, str] = {
    "GOOSE": "COOP",
    "COW": "PASTURE",
    "SHEEP": "PASTURE",
}

LAND_PRICES: Tuple[int, ...] = (1000, 2000, 4000)

MOVES: Dict[str, Tuple[int, int]] = {
    "NORTH": (0, -1),
    "SOUTH": (0, 1),
    "EAST": (1, 0),
    "WEST": (-1, 0),
}

FRONT_RUN_ITEMS: Tuple[str, ...] = ("MILK", "WOOL", "STRAWBERRY", "MELON")

LAST_ACT_STEP: int = 718

PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}

DEFAULT_SETTINGS = {
    "hand_align": True,
    "weed_repair": True,
    "sell_lead": True,
    "front_run": True,
    "budget_guard": True,
    "room_guard": True,
    "clamp_sells": True,
    "dead_stock": True,
    "terminal_liquidation": True,
    # Tunables
    "block_turns": 72,
    "shed_capacity": 100,
    "board_size": 10,
    "max_orders": 10,
    "turns_per_day": 24,
    "min_sell_price": 2,
}
