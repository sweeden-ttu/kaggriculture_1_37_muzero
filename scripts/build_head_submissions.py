#!/usr/bin/env python3
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

"""build_head_submissions.py
===========================
Builds the three canonical head submission packages:
  1. head_pl_submission.zip     (Control pure replay, 0% neural dependency)
  2. head_hybrid_submission.zip (Hybrid + spatial MuZero graft)
  3. head_muzero_submission.zip (Pure MuZero; requires loadable spatial weights)
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from hybrid_chassis.builder import compile_standalone
from muzero.spatial_checkpoint import assert_spatial_checkpoint_loadable

DIST = os.path.join(HERE, "dist")
MUZERO_PKG = os.path.join(HERE, "muzero")
CKPT_SPATIAL = os.path.join(HERE, "artifacts", "muzero_spatial_checkpoints.pt")
CKPT_STD = os.path.join(HERE, "artifacts", "muzero_checkpoints.pt")
CKPT_CHAMP = os.path.join(HERE, "artifacts", "muzero_checkpoints_champion.pt")
EXTERNAL_DIST = "/Volumes/TRAINREPLAYBOOST/muzero/dist"


def _resolve_spatial_ckpt(*candidates: str) -> str | None:
    """Prefer checkpoints that actually load into KaggricultureMuZeroChassis."""
    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            assert_spatial_checkpoint_loadable(path)
            return path
        except Exception as exc:
            print(f"  [skip] {os.path.relpath(path, HERE)} not spatial-loadable: {exc}")
    return None


def _add_muzero_tree(zf: zipfile.ZipFile) -> None:
    for root, dirs, files in os.walk(MUZERO_PKG):
        dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
        for fn in files:
            if fn.endswith(".pyc") or fn.startswith("."):
                continue
            p = os.path.join(root, fn)
            arcname = os.path.relpath(p, HERE)
            zf.write(p, arcname=arcname)


def _validate_zip_spatial_load(zip_path: str) -> None:
    """Extract zip to temp dir and confirm spatial chassis loads + one MCTS smoke."""
    import torch
    from muzero import KaggricultureMuZeroChassis, SampledMuZeroMCTS, encode_spatial_observation
    from muzero.spatial_checkpoint import load_spatial_checkpoint

    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(td)
        ckpt = None
        for name in (
            "muzero_spatial_checkpoints.pt",
            "artifacts/muzero_spatial_checkpoints.pt",
            "muzero_checkpoints_champion.pt",
            "artifacts/muzero_checkpoints_champion.pt",
        ):
            p = os.path.join(td, name)
            if os.path.isfile(p):
                ckpt = p
                break
        if ckpt is None:
            raise RuntimeError(f"{zip_path}: no spatial/champion checkpoint inside zip")
        chassis = KaggricultureMuZeroChassis()
        load_spatial_checkpoint(chassis, ckpt, strict=False)
        chassis.eval()
        obs = {
            "player": 0,
            "day": 5,
            "hour": 8,
            "farms": [
                {
                    "money": 2500.0,
                    "unlocked_quadrants": ["NW"],
                    "tiles": [[{} for _ in range(10)] for _ in range(10)],
                    "farmer": {"x": 2, "y": 2},
                    "hands": [],
                    "animals": {},
                },
                {},
            ],
            "private": {"shed": {}},
            "market": {"prices": {}},
        }
        obs_t = encode_spatial_observation(obs).unsqueeze(0)
        mcts = SampledMuZeroMCTS(chassis, num_samples=2, num_simulations=4)
        best, _, _ = mcts.search(obs_t, add_dirichlet_noise=False)
        assert best is not None and len(best) == 3
        print(f"  [validate] {os.path.basename(zip_path)} spatial load+MCTS OK joint={best}")


def build_head_submissions(*, validate: bool = True) -> None:
    os.makedirs(DIST, exist_ok=True)
    print("=" * 80)
    print("  BUILDING HEAD SUBMISSION TRINITY (spatial Sampled MuZero)")
    print("=" * 80)

    pl_py = os.path.join(DIST, "main_pl.py")
    compile_standalone(pl_py, include_muzero=False)

    hpm_py = os.path.join(DIST, "main_hpm.py")
    if not os.path.isfile(hpm_py) and os.path.isfile(os.path.join(DIST, "main_2033.py")):
        shutil.copyfile(os.path.join(DIST, "main_2033.py"), hpm_py)

    mz_py = os.path.join(DIST, "main_mz.py")
    compile_standalone(mz_py, include_muzero=True)
    shutil.copyfile(mz_py, os.path.join(DIST, "compiled_submission.py"))

    pl_zip = os.path.join(DIST, "head_pl_submission.zip")
    with zipfile.ZipFile(pl_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(pl_py, arcname="main.py")
    print(f"  [1/3] Built head_pl_submission.zip     ({os.path.getsize(pl_zip):>10,} B)")

    spatial_ckpt = _resolve_spatial_ckpt(CKPT_SPATIAL, CKPT_CHAMP, CKPT_STD)

    hybrid_zip = os.path.join(DIST, "head_hybrid_submission.zip")
    with zipfile.ZipFile(hybrid_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(hpm_py if os.path.isfile(hpm_py) else mz_py, arcname="main.py")
        if spatial_ckpt:
            zf.write(spatial_ckpt, arcname="artifacts/muzero_spatial_checkpoints.pt")
            zf.write(spatial_ckpt, arcname="muzero_spatial_checkpoints.pt")
            zf.write(spatial_ckpt, arcname="artifacts/muzero_checkpoints.pt")
            zf.write(spatial_ckpt, arcname="muzero_checkpoints.pt")
            print(f"       hybrid weights ← {os.path.relpath(spatial_ckpt, HERE)}")
        _add_muzero_tree(zf)
    print(f"  [2/3] Built head_hybrid_submission.zip ({os.path.getsize(hybrid_zip):>10,} B)")

    mz_zip = os.path.join(DIST, "head_muzero_submission.zip")
    if spatial_ckpt is None:
        raise RuntimeError(
            "No loadable spatial checkpoint for head_muzero_submission.zip. "
            "Run `make weights-spatial` (or promote a spatial ckpt) before packaging."
        )
    with zipfile.ZipFile(mz_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(mz_py, arcname="main.py")
        zf.write(spatial_ckpt, arcname="artifacts/muzero_spatial_checkpoints.pt")
        zf.write(spatial_ckpt, arcname="muzero_spatial_checkpoints.pt")
        zf.write(spatial_ckpt, arcname="artifacts/muzero_checkpoints_champion.pt")
        zf.write(spatial_ckpt, arcname="muzero_checkpoints_champion.pt")
        print(f"       muzero weights ← {os.path.relpath(spatial_ckpt, HERE)}")
        _add_muzero_tree(zf)
    print(f"  [3/3] Built head_muzero_submission.zip ({os.path.getsize(mz_zip):>10,} B)")

    if validate:
        _validate_zip_spatial_load(mz_zip)
        if spatial_ckpt:
            try:
                _validate_zip_spatial_load(hybrid_zip)
            except Exception as exc:
                print(f"  [warn] hybrid zip validate: {exc}")

    if os.path.isdir(os.path.dirname(EXTERNAL_DIST)):
        os.makedirs(EXTERNAL_DIST, exist_ok=True)
        for fn in [
            "head_pl_submission.zip",
            "head_hybrid_submission.zip",
            "head_muzero_submission.zip",
            "main_pl.py",
            "main_hpm.py",
            "main_mz.py",
        ]:
            src = os.path.join(DIST, fn)
            dst = os.path.join(EXTERNAL_DIST, fn)
            if os.path.isfile(src):
                shutil.copyfile(src, dst)
        print(f"  ✓ Synchronized all targets to {EXTERNAL_DIST}")

    print("=" * 80)


if __name__ == "__main__":
    build_head_submissions()
