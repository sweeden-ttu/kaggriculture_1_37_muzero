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

"""
pipelines/loss_schedule.py
==========================
Backward-compatible proxy module for Low-Rank Adaptation (LoRA) loss schedule.
Canonical implementation resides in pipelines/low_rank_adaptation.py.
"""
from __future__ import annotations

from .low_rank_adaptation import (
    compute_low_rank_adaptation_scheduled_weights,
    compute_separate_scheduled_weights,
    compute_softmax_weights,
    softmax_weights_high_low,
    validate_loss_weights_rule,
)

__all__ = [
    "validate_loss_weights_rule",
    "softmax_weights_high_low",
    "compute_softmax_weights",
    "compute_low_rank_adaptation_scheduled_weights",
    "compute_separate_scheduled_weights",
]
