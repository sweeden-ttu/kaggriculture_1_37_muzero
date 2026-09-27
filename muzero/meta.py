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

"""Canonical MuZero hyperparameters (d=256, B=300→601, LoRA r=16).

No analytical economy imports — teacher logic lives exclusively in ``economy.py``.
"""
from __future__ import annotations

__author__ = "Scott Weeden"
__license__ = "Apache-2.0"

from .types import HIDDEN_DIM, SUPPORT_B, SUPPORT_SIZE

LORA_RANK: int = 16
LORA_ALPHA: float = 16.0

__all__ = (
    "HIDDEN_DIM",
    "SUPPORT_B",
    "SUPPORT_SIZE",
    "LORA_RANK",
    "LORA_ALPHA",
)
