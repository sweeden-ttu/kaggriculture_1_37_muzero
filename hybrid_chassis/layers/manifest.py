"""Single source of truth for hybrid chassis layer stack order.

Used by both `pipeline.build_pipeline` (runtime exec) and
`builder.compile_standalone` (Kaggle single-file bundle).
"""
from __future__ import annotations

from typing import List, Tuple

# (filename under layers/, human title for compiled main comments)
LAYER_STACK: List[Tuple[str, str]] = [
    ("base_patches.py", "1. BASE PATCHES & EXPERIMENTAL LAYERS"),
    ("resource_guards.py", "2. RESOURCE & WAREHOUSE GUARDS"),
    ("action_masking.py", "3. SURVIVAL ACTION MASKING"),
    ("v9_expansion.py", "4. V9 EXPANSION LAYERS"),
    ("tactical_planners.py", "5. TACTICAL & HEURISTIC PLANNERS"),
    ("market_guards.py", "6. MARKET LIQUIDATION GUARDS"),
    ("v7_stack.py", "7. V7 OUTER STACK & PRODUCTION ENTRYPOINTS"),
    ("replay_lessons.py", "8. REPLAY-DISTILLED LESSONS"),
    ("muzero_options.py", "9. MUZERO DAY-BOUNDARY OPTIONS"),
]


def layer_filenames() -> List[str]:
    return [name for name, _ in LAYER_STACK]
