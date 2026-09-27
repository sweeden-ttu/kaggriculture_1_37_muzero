"""Standalone single-file Kaggle submission compiler for the hybrid chassis.

Bundles the modular package (core, routes, and layered guards) into a single,
dependency-free Python script suitable for Kaggle submission environments.
"""
from __future__ import annotations

import os
import re
from typing import Optional

from .layers.manifest import LAYER_STACK

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _strip_imports(code: str) -> str:
    """Strip package-internal, future, and typing imports from source blocks."""
    code = re.sub(r"from\s+\.+[a-zA-Z0-9_\.]*\s+import\s*(?:\([^)]*\)|[^\n]+)", "", code)
    code = re.sub(r"from\s+__future__\s+import\s+[^\n]+", "", code)
    code = re.sub(r"from\s+typing\s+import\s*(?:\([^)]*\)|[^\n]+)", "", code)
    code = re.sub(r"^import\s+copy\s*$", "", code, flags=re.MULTILINE)
    return code


def compile_standalone(output_path: Optional[str] = None, include_muzero: bool = True) -> str:
    """Compile the modular hybrid chassis into a single standalone submission script."""
    parts = []

    header_title = "Kaggriculture Master Hybrid Replay Chassis" + (" (MuZero Graft)" if include_muzero else " (Policy Lessons Control)")
    parts.append(
        f'"""{header_title}.\n'
        "Compiled from modular hybrid_chassis package.\n"
        '"""\n'
        "from __future__ import annotations\n\n"
        "import base64\n"
        "import copy\n"
        "import json\n"
        "import zlib\n\n"
    )

    core_blocks = [
        ("core/constants.py", "CORE CONSTANTS"),
        ("core/helpers.py", "CORE HELPERS"),
        ("core/view.py", "OBSERVATION VIEWS"),
        ("core/chassis.py", "REPLAY CHASSIS ENGINE"),
        ("routes/data.py", "ROUTE TAPES (COMPRESSED)"),
        ("routes/router.py", "ROUTE SELECTOR"),
    ]
    for rel, title in core_blocks:
        with open(os.path.join(BASE_DIR, rel), encoding="utf-8") as f:
            content = _strip_imports(f.read())
        parts.append(f"# {'=' * 75}\n# {title}\n# {'=' * 75}\n{content}\n\n")

    parts.append(
        """
# ---------------------------------------------------------------------------
# Chassis Instantiation (shop router + path finder)
# ---------------------------------------------------------------------------
_ROUTES, _R108_SHOP_ROUTES = load_routes()
apply_opening_patch(_ROUTES)

def _router(observation, step, state):
    return route_selector(observation, step, state, _R108_SHOP_ROUTES)

"""
    )
    if include_muzero:
        # MuZero owns strategy; chassis only routes shops and pathfinds units.
        parts.append(
            """
_PATHFIND_SETTINGS = dict(_SETTINGS)
_PATHFIND_SETTINGS.update({
    'hand_align': True,
    'weed_repair': True,
    'sell_lead': False,
    'front_run': False,
    'budget_guard': False,
    'room_guard': False,
    'clamp_sells': True,
    'dead_stock': False,
    'terminal_liquidation': False,
})
_IMPL = make_agent(_ROUTES, router=_router, **_PATHFIND_SETTINGS)
"""
        )
    else:
        parts.append(
            """
_IMPL = make_agent(_ROUTES, router=_router, **_SETTINGS)
"""
        )
    parts.append(
        """
_IMPL.chassis.diagnostics['terminal_rescue_errors'] = 0

def projected_shed(action, view):
    return _IMPL.chassis._projected_shed(action, view)

"""
    )

    for fname, title in LAYER_STACK:
        if not include_muzero and fname == "muzero_options.py":
            continue
        with open(os.path.join(BASE_DIR, "layers", fname), encoding="utf-8") as f:
            code = f.read()
        code = code.replace("from __future__ import annotations\n", "")
        parts.append(f"# {'=' * 75}\n# {title}\n# {'=' * 75}\n{code}\n\n")

    parts.append(
        """
# ---------------------------------------------------------------------------
# Final Export
# ---------------------------------------------------------------------------
agent = kaggle_submission_agent
"""
    )

    compiled_code = "".join(parts)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(compiled_code)
    return compiled_code


if __name__ == "__main__":
    dist_dir = os.path.join(BASE_DIR, "..", "dist")
    os.makedirs(dist_dir, exist_ok=True)
    compile_standalone(os.path.join(dist_dir, "main_pl.py"), include_muzero=False)
    compile_standalone(os.path.join(dist_dir, "main_mz.py"), include_muzero=True)
    compile_standalone(os.path.join(dist_dir, "compiled_submission.py"), include_muzero=True)
    print("Successfully compiled dist targets: main_pl.py, main_mz.py, compiled_submission.py")
