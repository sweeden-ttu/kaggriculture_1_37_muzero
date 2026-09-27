"""Observation view snapshots for efficient, structured state access."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .constants import DEFAULT_SETTINGS
from .helpers import _get, _int


class _View:
    """Cheap per-step snapshot of everything the layers read from the observation."""

    def __init__(self, observation: Any, player: int, cfg: Dict[str, Any]):
        farms = list(_get(observation, "farms", []) or [])
        self.farm = farms[player] if player < len(farms) else {}
        self.rival = (
            farms[1 - player]
            if len(farms) >= 2 and 1 - player < len(farms)
            else {}
        )
        private = _get(observation, "private", {}) or {}
        self.shed = {
            k: max(0, _int(v))
            for k, v in dict(_get(private, "shed", {}) or {}).items()
        }
        self.seeds = dict(_get(private, "seeds", {}) or {})
        self.invs = [
            dict(i or {})
            for i in (_get(private, "inventories", []) or [])
        ]
        market = _get(observation, "market", {}) or {}
        self.prices = {
            k: _int(v)
            for k, v in dict(_get(market, "prices", {}) or {}).items()
        }
        self.money = float(_get(self.farm, "money", 0.0) or 0.0)
        self.tiles = _get(self.farm, "tiles", []) or []
        self.board = len(self.tiles) or cfg.get("board_size", 10)
        self.positions = [_get(self.farm, "farmer", None)] + [
            list(p) for p in (_get(self.farm, "hands", []) or [])
        ]
        self.hires_today = _int(_get(self.farm, "hires_today", 0))
        self.quadrants = len(list(_get(self.farm, "unlocked_quadrants", []) or []))

    def inv(self, idx: int) -> Dict[str, int]:
        """Return inventory dictionary of unit at `idx`."""
        return self.invs[idx] if idx < len(self.invs) else {}

    def in_hands(self, item: str) -> int:
        """Count total units of `item` carried across all hands/farmer."""
        return sum(max(0, _int(_get(inv, item, 0))) for inv in self.invs)


class FarmView(_View):
    """Convenience subclass for player-relative farm state."""

    def __init__(self, obs: Any, cfg: Optional[Dict[str, Any]] = None):
        config = cfg if cfg is not None else DEFAULT_SETTINGS
        super().__init__(obs, int(_get(obs, "player", 0)), config)
