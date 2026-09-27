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

"""Spatial checkpoint save/load (shipped with muzero/ for Kaggle packages)."""
from __future__ import annotations

import os
import shutil
from typing import Any, Dict, Optional

import torch

from .chassis import KaggricultureMuZeroChassis
from .spatial_constants import DEFAULT_SPATIAL_CHECKPOINT

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.path.join(HERE, "artifacts")
DEFAULT_SPATIAL_OUT = os.path.join(ARTIFACTS, DEFAULT_SPATIAL_CHECKPOINT)


def save_spatial_checkpoint(
    chassis: KaggricultureMuZeroChassis,
    path: str = DEFAULT_SPATIAL_OUT,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model_state_dict": chassis.state_dict(),
        "arch": "KaggricultureMuZeroChassis",
        "meta": meta or {"artifact_kind": "spatial_muzero"},
    }
    torch.save(payload, path)
    return path


def load_spatial_checkpoint(
    chassis: KaggricultureMuZeroChassis,
    path: str,
    device: str = "cpu",
    *,
    strict: bool = False,
) -> KaggricultureMuZeroChassis:
    blob = torch.load(path, map_location=device, weights_only=False)
    if isinstance(blob, dict) and "arch" in blob:
        arch = blob["arch"]
        if arch not in ("KaggricultureMuZeroChassis", "spatial"):
            raise RuntimeError(
                f"Checkpoint arch mismatch at {path}: expected KaggricultureMuZeroChassis, got {arch!r}"
            )
    state = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
    if not isinstance(state, dict):
        raise RuntimeError(f"Invalid spatial checkpoint payload at {path}")
    keys = set(state.keys())
    looks_macro = ("policy_head.weight" in keys or "value_head.weight" in keys) and not any(
        k.startswith("representation.") or k.startswith("dynamics.") or k.startswith("prediction.")
        for k in keys
    )
    if looks_macro:
        raise RuntimeError(
            f"Refusing to load MLP/macro weights into spatial chassis from {path}"
        )
    incompatible = chassis.load_state_dict(state, strict=strict)
    if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        matched = len(state) - len(incompatible.unexpected_keys)
        if matched < 5:
            raise RuntimeError(
                f"Failed to load spatial weights from {path}: "
                f"missing={incompatible.missing_keys[:5]} "
                f"unexpected={incompatible.unexpected_keys[:5]}"
            )
    chassis.to(device)
    return chassis


def promote_spatial_champion(
    spatial_path: str = DEFAULT_SPATIAL_OUT,
    *,
    std_path: Optional[str] = None,
    champ_path: Optional[str] = None,
) -> Dict[str, str]:
    std_path = std_path or os.path.join(ARTIFACTS, "muzero_checkpoints.pt")
    champ_path = champ_path or os.path.join(ARTIFACTS, "muzero_checkpoints_champion.pt")
    if not os.path.isfile(spatial_path):
        raise FileNotFoundError(f"No spatial checkpoint at {spatial_path}")
    probe = KaggricultureMuZeroChassis()
    load_spatial_checkpoint(probe, spatial_path, strict=False)
    shutil.copy2(spatial_path, std_path)
    shutil.copy2(spatial_path, champ_path)
    return {"spatial": spatial_path, "std": std_path, "champion": champ_path}


def assert_spatial_checkpoint_loadable(path: str) -> Dict[str, Any]:
    """Validate a checkpoint can load into KaggricultureMuZeroChassis; raise on failure."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    chassis = KaggricultureMuZeroChassis()
    load_spatial_checkpoint(chassis, path, strict=False)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    meta = blob.get("meta") if isinstance(blob, dict) else None
    return {"path": path, "arch": blob.get("arch") if isinstance(blob, dict) else None, "meta": meta}
