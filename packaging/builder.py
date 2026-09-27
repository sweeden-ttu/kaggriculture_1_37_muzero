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

"""Consolidated package builder for Kaggriculture submissions and baselines."""
from __future__ import annotations

import base64
import json
import os
import shutil
import zipfile
import zlib
from typing import Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BASE_CANDIDATES = [
    os.path.join(HERE, "baselines", "submission_hybrid_direct.zip"),
    os.path.join(HERE, "artifacts", "templates", "submission_hybrid_direct.zip"),
    os.path.join(HERE, "artifacts", "baselines_archive", "submission_hybrid_direct.zip"),
]
BASELINE_HYBRID = next((p for p in _BASE_CANDIDATES if os.path.isfile(p)), _BASE_CANDIDATES[0])
GATE_FILE = os.path.join(HERE, "hybrid_chassis", "layers", "muzero_options.py")
_CKPT_CANDIDATES = [
    os.path.join(HERE, "artifacts", "muzero_spatial_checkpoints.pt"),
    os.path.join(HERE, "artifacts", "muzero_checkpoints_champion.pt"),
    os.path.join(HERE, "artifacts", "muzero_checkpoints.pt"),
]
DEFAULT_CKPT = next((p for p in _CKPT_CANDIDATES if os.path.isfile(p)), _CKPT_CANDIDATES[0])
MUZERO_PKG = os.path.join(HERE, "muzero")
DIST_DIR = os.path.join(HERE, "dist")
DEFAULT_OUT_ZIP = os.path.join(DIST_DIR, "submission_muzero.zip")

MAIN_LICENSE_HEADER = """# Copyright 2026 Scott Weeden
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

from __future__ import annotations

__author__ = "Scott Weeden"
__license__ = "Apache-2.0"
"""

GATE_PREFIX = """
# =============================================================================
# MUZERO EXPAND GATE (grafted onto Hybrid Direct chassis)
# =============================================================================
"""


def _add_tree_to_zip(zf: zipfile.ZipFile, root: str, arc_root: str) -> None:
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
        for fn in files:
            if fn.endswith(".pyc") or fn.startswith("."):
                continue
            path = os.path.join(dirpath, fn)
            zf.write(path, arcname=os.path.join(arc_root, os.path.relpath(path, root)))


def build_hybrid_package(
    ckpt_path: str = DEFAULT_CKPT,
    out_zip: Optional[str] = None,
    out_py: Optional[str] = None,
    with_weights: bool = True,
) -> Tuple[str, str]:
    """Combines hybrid chassis with the MuZero options gate and neural weights."""
    base_hybrid = next((p for p in _BASE_CANDIDATES if os.path.isfile(p)), BASELINE_HYBRID)
    assert os.path.isfile(base_hybrid), f"Missing base: {base_hybrid}"
    assert os.path.isfile(GATE_FILE), f"Missing gate: {GATE_FILE}"

    with zipfile.ZipFile(base_hybrid, "r") as zf:
        base_main = zf.read("main.py").decode("utf-8")
    with open(GATE_FILE, "r", encoding="utf-8") as f:
        gate = f.read()

    base_main = base_main.replace("from __future__ import annotations", "# from __future__ import annotations (moved to top)")
    gate = gate.replace("from __future__ import annotations", "# from __future__ import annotations (moved to top)")

    random_snippet = (
        "            # Private RNG: no mutation of the engine or opponent's random state.\n"
        "            # Seed is used only for reproducible opening selection, never prediction.\n"
        "            import random as _opening_random\n"
        "            seed=(configuration or {}).get('seed',0)\n"
        "            bit=_opening_random.Random('productive-opening:'+str(seed)+':'+str(seat)).getrandbits(1)\n"
        "            mode='EarlyCycle' if bit else 'Original'"
    )
    if random_snippet in base_main:
        base_main = base_main.replace(random_snippet, "            mode='EarlyCycle'")

    combined = MAIN_LICENSE_HEADER + "\n" + base_main.rstrip() + "\n\n" + GATE_PREFIX + gate + "\n"
    target_py = out_py or os.path.join(DIST_DIR, "main.py")
    os.makedirs(os.path.dirname(os.path.abspath(target_py)), exist_ok=True)
    with open(target_py, "w", encoding="utf-8") as f:
        f.write(combined)

    target_zip = out_zip or DEFAULT_OUT_ZIP
    os.makedirs(os.path.dirname(os.path.abspath(target_zip)), exist_ok=True)
    targets_to_write = [target_zip]
    if out_zip is None:
        targets_to_write.append(os.path.join(DIST_DIR, "submission.zip"))

    include_weights = with_weights or os.path.isfile(ckpt_path)
    for zpath in targets_to_write:
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("main.py", combined)
            if include_weights and os.path.isfile(ckpt_path):
                zf.write(ckpt_path, arcname="artifacts/muzero_checkpoints_champion.pt")
                zf.write(ckpt_path, arcname="muzero_checkpoints_champion.pt")
                zf.write(ckpt_path, arcname="artifacts/muzero_checkpoints.pt")
                zf.write(ckpt_path, arcname="muzero_checkpoints.pt")
                for root, dirs, files in os.walk(MUZERO_PKG):
                    dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
                    for fn in files:
                        if fn.endswith(".pyc") or fn.startswith("."):
                            continue
                        path = os.path.join(root, fn)
                        zf.write(path, arcname=os.path.relpath(path, HERE))

    return target_py, target_zip


def embed_weights(
    zip_path: str,
    ckpt_path: str = DEFAULT_CKPT,
    main_path: Optional[str] = None,
) -> None:
    """Embeds MuZero weights into an existing submission zip without modifying main.py."""
    assert os.path.isfile(ckpt_path), f"Missing checkpoint: {ckpt_path}"
    main_bytes: Optional[bytes] = None
    if main_path and os.path.isfile(main_path):
        with open(main_path, "rb") as f:
            main_bytes = f.read()
    elif os.path.isfile(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            main_bytes = zf.read("main.py")
    else:
        raise FileNotFoundError(f"No main.py source for {zip_path}")

    os.makedirs(os.path.dirname(os.path.abspath(zip_path)) or ".", exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("main.py", main_bytes)
        zf.write(ckpt_path, arcname="artifacts/muzero_checkpoints.pt")
        zf.write(ckpt_path, arcname="muzero_checkpoints.pt")
        _add_tree_to_zip(zf, MUZERO_PKG, "muzero")


def build_thirst_baseline(
    replay_path: Optional[str] = None,
    out_zip: Optional[str] = None,
) -> str:
    """Builds submission_thirst.zip from replay 113565586."""
    src = replay_path or os.path.join(HERE, "replays", "113565586.json")
    dst = out_zip or os.path.join(HERE, "baselines", "submission_thirst.zip")
    assert os.path.isfile(src), f"Missing replay {src}"

    with open(src, "r", encoding="utf-8") as f:
        replay_data = json.load(f)

    steps = replay_data["steps"]
    thirst_tape = [s[1]["action"] for s in steps]
    raw_json = json.dumps(thirst_tape, separators=(",", ":"))
    compressed = zlib.compress(raw_json.encode("utf-8"), level=9)
    b85_tape = base64.b85encode(compressed).decode("ascii")

    main_content = f'''# Copyright 2026
# Thirst Baseline Agent distilled from Episode 113565586
from __future__ import annotations
import base64
import json
import zlib

_TAPE_B85 = "{b85_tape}"
_TAPE = json.loads(zlib.decompress(base64.b85decode(_TAPE_B85)).decode("utf-8"))

def agent(observation, configuration=None):
    step = int(observation.get("step", 0))
    if step >= len(_TAPE):
        return {{"farmer": ["PASS"], "hands": [], "market": []}}
    return dict(_TAPE[step])
'''
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("main.py", main_content)
    return dst


def compile_standalone_chassis(out_main: Optional[str] = None) -> str:
    """Compiles the modular hybrid_chassis into a standalone single-file submission."""
    from hybrid_chassis.builder import compile_standalone

    target = out_main or os.path.join(DIST_DIR, "main.py")
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    code = compile_standalone(target)
    print(f"Compiled hybrid chassis → {target} ({len(code):,} chars)")
    return target


def package_chassis_submission(
    main_path: Optional[str] = None,
    out_zip: Optional[str] = None,
) -> str:
    """Packages a standalone main.py into submission.zip."""
    src = main_path or os.path.join(DIST_DIR, "main.py")
    dst = out_zip or os.path.join(DIST_DIR, "submission.zip")
    assert os.path.isfile(src), f"Missing source: {src}"

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(src, arcname="main.py")
    print(f"Packaged {dst} ({os.path.getsize(dst):,} bytes)")
    return dst

