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
pipelines/boost_schedule.py
===========================
Backward-compatible proxy module for Adaptive Moment Estimation (Adam) loss schedule.
Canonical implementation resides in pipelines/adaptive_moment_estimation.py.
"""
from __future__ import annotations

from .adaptive_moment_estimation import (
    apply_adaptive_moment_estimation_schedule,
    apply_boost_schedule,
    compute_adaptive_moment_scheduled_weights,
    compute_boost_scheduled_weights,
)

__all__ = [
    "compute_adaptive_moment_scheduled_weights",
    "compute_boost_scheduled_weights",
    "apply_adaptive_moment_estimation_schedule",
    "apply_boost_schedule",
]
