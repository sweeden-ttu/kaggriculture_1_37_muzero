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

"""Low-Rank Adaptation (LoRA) for MuZero dynamics and prediction heads.

Freezes base weights of g_θ / f_θ and injects trainable rank-r adapters:
  W_adapted = W_0 + (α / r) · (B @ A)

Adapters are applied exclusively to Linear layers inside DynamicsNetwork and
PredictionNetwork. Representation (h_θ) remains frozen during Phase 4 fine-tuning.
"""
from __future__ import annotations

__author__ = "Scott Weeden"
__license__ = "Apache-2.0"

import math
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_LORA_RANK: int = 16
DEFAULT_LORA_ALPHA: float = 16.0
LORA_ADAPTERS_DIR_NAME: str = "lora_adapters"


class LoRALinear(nn.Module):
    """Drop-in Linear with frozen base weights and low-rank residual adapters."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = DEFAULT_LORA_RANK,
        alpha: float = DEFAULT_LORA_ALPHA,
    ):
        super().__init__()
        if rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {rank}")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)

        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # A ~ N(0, 1/r), B = 0 so adapters start as identity residual
        self.lora_A = nn.Parameter(torch.zeros(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        # (α/r) * x @ A^T @ B^T
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + self.scaling * lora_out

    def merge(self) -> nn.Linear:
        """Bake adapters into a standard Linear for standalone deployment."""
        merged = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        with torch.no_grad():
            delta = self.scaling * (self.lora_B @ self.lora_A)
            merged.weight.copy_(self.weight + delta)
            if self.bias is not None and merged.bias is not None:
                merged.bias.copy_(self.bias)
        return merged

    def adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "lora_A": self.lora_A.detach().cpu().clone(),
            "lora_B": self.lora_B.detach().cpu().clone(),
            "rank": torch.tensor(self.rank),
            "alpha": torch.tensor(self.alpha),
            "in_features": torch.tensor(self.in_features),
            "out_features": torch.tensor(self.out_features),
        }


def _replace_linear_with_lora(
    module: nn.Module,
    rank: int,
    alpha: float,
    prefix: str = "",
) -> List[str]:
    """Recursively replace nn.Linear children with LoRALinear. Returns injected paths."""
    injected: List[str] = []
    for name, child in list(module.named_children()):
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            injected.append(path)
        elif isinstance(child, LoRALinear):
            continue
        else:
            injected.extend(_replace_linear_with_lora(child, rank, alpha, path))
    return injected


def inject_lora(
    model: nn.Module,
    rank: int = DEFAULT_LORA_RANK,
    alpha: float = DEFAULT_LORA_ALPHA,
    targets: Iterable[str] = ("dynamics", "prediction"),
) -> List[str]:
    """Freeze base weights and inject LoRA into dynamics / prediction Linears.

    Representation and any modules not listed in ``targets`` stay frozen and
    un-adapted. Returns list of module paths that received adapters.
    """
    # Freeze entire model first
    for p in model.parameters():
        p.requires_grad = False

    injected: List[str] = []
    for target_name in targets:
        submodule = getattr(model, target_name, None)
        if submodule is None:
            continue
        paths = _replace_linear_with_lora(submodule, rank=rank, alpha=alpha)
        injected.extend(f"{target_name}.{p}" for p in paths)

    # Unfreeze only LoRA parameters
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True

    return injected


def lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Collect trainable LoRA adapter parameters."""
    params: List[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.append(module.lora_A)
            params.append(module.lora_B)
    return params


def has_lora(model: nn.Module) -> bool:
    return any(isinstance(m, LoRALinear) for m in model.modules())


def merge_lora(model: nn.Module) -> nn.Module:
    """In-place merge of all LoRALinear layers back into plain nn.Linear."""
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, name, child.merge())
    return model


def export_lora_state(model: nn.Module) -> Dict[str, Any]:
    """Export adapter-only state for artifacts/lora_adapters/."""
    adapters: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            adapters[name] = module.adapter_state_dict()
    return {
        "adapters": adapters,
        "format": "muzero_lora_v1",
    }


def save_lora_adapters(
    model: nn.Module,
    path: str,
    meta: Optional[Dict[str, Any]] = None,
    atomic: bool = True,
) -> str:
    """Persist lightweight LoRA adapters (not full base weights)."""
    payload = export_lora_state(model)
    payload["meta"] = meta or {}
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if atomic:
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)
    else:
        torch.save(payload, path)
    return path


def load_lora_adapters(
    model: nn.Module,
    path: str,
    device: str = "cpu",
    rank: int = DEFAULT_LORA_RANK,
    alpha: float = DEFAULT_LORA_ALPHA,
) -> Tuple[nn.Module, str]:
    """Inject LoRA if needed and load adapter weights from disk."""
    if not has_lora(model):
        inject_lora(model, rank=rank, alpha=alpha)
    payload = torch.load(path, map_location=device)
    adapters = payload.get("adapters", {})
    name_to_module = dict(model.named_modules())
    for name, state in adapters.items():
        module = name_to_module.get(name)
        if not isinstance(module, LoRALinear):
            continue
        module.lora_A.data.copy_(state["lora_A"].to(device))
        module.lora_B.data.copy_(state["lora_B"].to(device))
    return model, path


def default_lora_adapters_dir(repo_root: Optional[str] = None) -> str:
    root = repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "artifacts", LORA_ADAPTERS_DIR_NAME)
