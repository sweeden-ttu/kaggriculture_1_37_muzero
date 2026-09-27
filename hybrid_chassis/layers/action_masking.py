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

"""Hard survival action masks for MuZero policy logits.

Chassis layers own all environment constraint enforcement. Invalid macro-options
are zeroed in a boolean mask; callers set corresponding logits to -inf before
softmax so MuZero never samples lethal / illegal actions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

import math

try:
    import torch
except ImportError:  # pragma: no cover - packaging without torch still gets masks
    torch = None  # type: ignore

# MacroOption indices (must stay aligned with muzero.types.MacroOption)
FRONT_LOAD_SURVIVAL = 0
EXPANSION_PREPARATION = 1
TRICKLE_LIQUIDATION = 2
EXPAND_NE = 3
EXPAND_SW = 4
EXPAND_SE = 5
HIRE_HANDS = 6
PASS = 7
NUM_MACRO_OPTIONS = 8

LAND_PRICES = {"NE": 1000.0, "SW": 2000.0, "SE": 4000.0}
LAND_BUFFERS = {"NE": 500.0, "SW": 500.0, "SE": 800.0}

# Graft onto parent agent when executed inside the hybrid layer stack
_MASK_PARENT = globals().get("agent")


def survival_facts(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Extract survival-critical scalars from a raw Kaggriculture observation."""
    seat = int(observation.get("player", 0))
    farms = observation.get("farms", [{}, {}])
    farm = farms[seat] if seat < len(farms) else {}
    private = observation.get("private", {}) or {}
    shed = dict(private.get("shed", {}) or {})
    unlocked = list(farm.get("unlocked_quadrants") or ["NW"])
    money = float(farm.get("money", 0.0))
    day = int(observation.get("day", int(observation.get("step", 0)) // 24))
    tiles = farm.get("tiles") or []

    water_starved = False
    for row in tiles:
        for tile in row if isinstance(row, (list, tuple)) else []:
            if not isinstance(tile, dict):
                continue
            crop = tile.get("crop")
            if crop and not tile.get("watered_today", False):
                # Mid-day crop without water is a survival risk signal
                if int(observation.get("hour", 0)) >= 18:
                    water_starved = True
                    break
        if water_starved:
            break

    wheat = float(shed.get("WHEAT", 0))
    animals = farm.get("animals") or {}
    animal_count = 0
    if isinstance(animals, dict):
        animal_count = sum(int(v) for v in animals.values() if isinstance(v, (int, float)))
    elif isinstance(animals, list):
        animal_count = len(animals)

    seed_starved = animal_count > 0 and wheat < 1.0 and day < 27
    shed_total = sum(float(v) for v in shed.values())
    shed_critical = shed_total >= 85.0

    return {
        "money": money,
        "day": day,
        "unlocked": unlocked,
        "shed": shed,
        "water_starved": water_starved,
        "seed_starved": seed_starved,
        "shed_critical": shed_critical,
        "animal_count": animal_count,
    }


def legal_macro_mask_from_facts(facts: Dict[str, Any]) -> List[bool]:
    """Return length-8 boolean mask; False ⇒ action must be logit-masked to -inf."""
    mask = [True] * NUM_MACRO_OPTIONS
    unlocked: Sequence[str] = facts.get("unlocked") or ["NW"]
    money = float(facts.get("money", 0.0))
    day = int(facts.get("day", 0))

    # Already-unlocked quadrants cannot be expanded again
    if "NE" in unlocked:
        mask[EXPAND_NE] = False
    if "SW" in unlocked:
        mask[EXPAND_SW] = False
    if "SE" in unlocked:
        mask[EXPAND_SE] = False

    # Capital / buffer gates
    for opt, quad in ((EXPAND_NE, "NE"), (EXPAND_SW, "SW"), (EXPAND_SE, "SE")):
        if not mask[opt]:
            continue
        need = LAND_PRICES[quad] + LAND_BUFFERS[quad]
        if money < need or day > 24:
            mask[opt] = False

    # Survival: zero-water / seed starvation forces FRONT_LOAD_SURVIVAL
    if facts.get("water_starved") or facts.get("seed_starved"):
        for i in range(NUM_MACRO_OPTIONS):
            mask[i] = i == FRONT_LOAD_SURVIVAL
        return mask

    # Shed overflow: liquidation must remain available
    if facts.get("shed_critical"):
        mask[TRICKLE_LIQUIDATION] = True

    # Late game: stop hiring
    if day >= 28:
        mask[HIRE_HANDS] = False

    # Always allow PASS as a safe no-op when everything else is illegal
    if not any(mask):
        mask[PASS] = True
    return mask


def build_action_mask(observation: Dict[str, Any]) -> List[bool]:
    """Convenience: observation → length-8 legality mask."""
    return legal_macro_mask_from_facts(survival_facts(observation))


def apply_logit_mask(logits: Any, mask: Sequence[bool]) -> Any:
    """Set invalid action logits to -inf before softmax.

    Accepts a torch.Tensor or a Python list/tuple of floats.
    """
    if torch is not None and isinstance(logits, torch.Tensor):
        out = logits.clone()
        device = out.device
        mask_t = torch.tensor(list(mask), dtype=torch.bool, device=device)
        while mask_t.dim() < out.dim():
            mask_t = mask_t.unsqueeze(0)
        # Broadcast mask over batch dims
        if mask_t.shape != out.shape:
            mask_t = mask_t.expand_as(out)
        neg_inf = torch.tensor(float("-inf"), dtype=out.dtype, device=device)
        return torch.where(mask_t, out, neg_inf)

    out_list = [float(x) for x in logits]
    for i, ok in enumerate(mask):
        if i < len(out_list) and not ok:
            out_list[i] = float("-inf")
    return out_list


def masked_softmax_probs(logits: Any, mask: Sequence[bool]) -> List[float]:
    """Softmax over masked logits; returns a length-8 probability vector."""
    masked = apply_logit_mask(logits, mask)
    if torch is not None and isinstance(masked, torch.Tensor):
        probs = torch.softmax(masked.flatten()[:NUM_MACRO_OPTIONS], dim=-1)
        return [float(p) for p in probs.tolist()]
    xs = [float(x) for x in masked][:NUM_MACRO_OPTIONS]
    finite = [x for x in xs if math.isfinite(x)]
    if not finite:
        return [1.0 / NUM_MACRO_OPTIONS] * NUM_MACRO_OPTIONS
    m = max(finite)
    exps = [math.exp(x - m) if math.isfinite(x) else 0.0 for x in xs]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


# ---------------------------------------------------------------------------
# Layer graft: wrap parent agent, attach mask metadata for MuZero consumers
# ---------------------------------------------------------------------------

_MASK_REPORT = {"masks_applied": 0, "survival_forces": 0, "errors": 0}


def agent(observation, configuration=None):
    """Constraint-enforcer wrapper: runs parent then stamps action mask metadata."""
    parent = _MASK_PARENT
    if parent is None:
        action = {"farmer": ["PASS"], "hands": [], "market": []}
    else:
        action = parent(observation, configuration)
    try:
        facts = survival_facts(observation)
        mask = legal_macro_mask_from_facts(facts)
        _MASK_REPORT["masks_applied"] += 1
        if facts.get("water_starved") or facts.get("seed_starved"):
            _MASK_REPORT["survival_forces"] += 1
        # Attach non-breaking metadata for MuZero / telemetry consumers
        if isinstance(action, dict):
            action = dict(action)
            action["_muzero_action_mask"] = mask
            action["_muzero_survival"] = {
                "water_starved": bool(facts.get("water_starved")),
                "seed_starved": bool(facts.get("seed_starved")),
                "shed_critical": bool(facts.get("shed_critical")),
            }
    except Exception:
        _MASK_REPORT["errors"] += 1
    return action
