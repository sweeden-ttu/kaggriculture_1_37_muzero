"""Pure-Python utility functions and game engine mirroring helpers."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .constants import ANIMAL_STRUCTURE, MOVES


def _get(value: Any, key: Any, default: Any = None) -> Any:
    """Field access that works for dicts, Kaggle Structs, and attribute objects."""
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def _int(value: Any, default: int = 0) -> int:
    """Safely convert a value to an integer with a fallback default."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _fib(n: int) -> int:
    """Compute Fibonacci number for hiring cost scaling."""
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _step_of(observation: Any) -> int:
    """Extract or calculate the current simulation turn step (0..719)."""
    raw = _get(observation, "step")
    if raw is not None:
        return _int(raw)
    return _int(_get(observation, "day", 0)) * 24 + _int(_get(observation, "hour", 0))


def _shed_adjacent(pos: Sequence[int], board: int) -> bool:
    """Check if a coordinate (x, y) is adjacent to or on the central 2x2 shed."""
    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
        return False
    half = board // 2
    return pos[0] in (half - 1, half) and pos[1] in (half - 1, half)


def _tile_at(tiles: Sequence[Sequence[Any]], pos: Sequence[int]) -> Any:
    """Return the tile at pos (x, y), or 'LOCKED' if invalid/out-of-bounds."""
    try:
        x, y = int(pos[0]), int(pos[1])
        return tiles[y][x]
    except (TypeError, ValueError, IndexError):
        return "LOCKED"


def _is_noop(
    act: Sequence[Any],
    tile: Any,
    inv: Dict[str, Any],
    seeds: Dict[str, Any],
    pos: Sequence[int],
    board: int,
) -> bool:
    """Determine if the simulation engine will ignore `act`.

    Mirrors the engine's internal `_apply_unit_action` validation.
    Used by weed_repair and chore schedulers to detect wasted action turns.
    """
    if not act:
        return True

    op = act[0]
    x, y = pos[0], pos[1]

    # 1. Positional movements
    if op in MOVES:
        dx, dy = MOVES[op]
        return not (0 <= x + dx < board and 0 <= y + dy < board)

    if op == "PASS":
        return True

    # 2. Shed inventory transfers (require unit to stand adjacent to shed)
    adjacent = _shed_adjacent(pos, board)
    if op == "DROP":
        return (not adjacent) or (not inv)

    if op == "PICKUP":
        return not adjacent

    if op == "PLACE":
        item = act[1] if len(act) > 1 else None
        # Animal placement into matching vacant structure
        if (
            item in ANIMAL_STRUCTURE
            and isinstance(tile, dict)
            and _get(tile, "kind") == ANIMAL_STRUCTURE[item]
            and _get(tile, "animal") is None
        ):
            return _int(_get(inv, item, 0)) <= 0
        return (not adjacent) or _int(_get(inv, item, 0)) <= 0

    # 3. Operations on tiles
    if tile == "LOCKED":
        return True

    is_dict = isinstance(tile, dict)
    kind = _get(tile, "kind") if is_dict else None
    has_animal = is_dict and _get(tile, "animal") is not None

    if op == "PLANT":
        seed_item = act[1] if len(act) > 1 else None
        return tile is not None or _int(_get(seeds, seed_item, 0)) <= 0

    if op == "WATER":
        return kind != "PLANT" or bool(_get(tile, "watered_today"))

    if op == "HARVEST":
        return (not is_dict) or _int(_get(tile, "yield_units", 0)) <= 0

    if op == "FERTILIZE":
        return kind != "PLANT" or _int(_get(inv, "FERTILIZER", 0)) <= 0

    if op == "DIG":
        return tile is None or has_animal

    if op in ("BUILD_COOP", "BUILD_PASTURE"):
        return tile is not None

    if op == "FEED":
        return (
            (not has_animal)
            or bool(_get(tile, "fed_today"))
            or _int(_get(inv, "WHEAT", 0)) <= 0
        )

    if op == "COLLECT_FERTILIZER":
        return (not has_animal) or (not _get(tile, "fertilizer_available"))

    if op == "CARE":
        return (not has_animal) or bool(_get(tile, "cared_today"))

    return True
