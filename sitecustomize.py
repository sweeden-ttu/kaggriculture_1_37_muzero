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
sitecustomize.py
================
Automatic environment hook for Kaggriculture MuZero.
When MUZERO_SCHEDULE_MODE == 'boost', dynamically enables the inverted boost schedule.
When unset or in standard mode, this hook is a zero-overhead no-op.
"""
from __future__ import annotations

import os

if os.environ.get("MUZERO_SCHEDULE_MODE") in ("boost", "adaptive_moment_estimation", "adam"):
    try:
        from pipelines.adaptive_moment_estimation import apply_adaptive_moment_estimation_schedule
        apply_adaptive_moment_estimation_schedule()
    except Exception:
        pass
