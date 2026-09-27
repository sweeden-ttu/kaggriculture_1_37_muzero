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

"""Opponent classification and tier taxonomy for Kaggriculture."""
from __future__ import annotations

import enum
import os
from typing import Dict, List, Optional


class BaselineTier(enum.Enum):
    """Evaluation tiers based on strategic autonomy and complexity."""
    TIER_1_HEURISTIC = 1   # Direct heuristic agents (submission_2000, 2500, thirst)
    TIER_2_HYBRID = 2      # Reactive chassis engines (hybrid_direct, beforegap)
    TIER_3_NEURAL = 3      # Autonomous MuZero agents (submission_muzero, champions)


BASELINE_TIERS: Dict[str, BaselineTier] = {
    # Tier 1: Fixed rules & hardcoded action schedules
    "submission_2000": BaselineTier.TIER_1_HEURISTIC,
    "submission_2500": BaselineTier.TIER_1_HEURISTIC,
    "submission_thirst": BaselineTier.TIER_1_HEURISTIC,
    "submission_0771": BaselineTier.TIER_1_HEURISTIC,

    # Tier 2: Priority-driven reactive state-machine chassis
    "submission_hybrid_direct": BaselineTier.TIER_2_HYBRID,
    "submission_beforegap": BaselineTier.TIER_2_HYBRID,
    "submission_2033": BaselineTier.TIER_2_HYBRID,

    # Tier 3: Neural value-equivalent macro-option planners
    "submission_muzero": BaselineTier.TIER_3_NEURAL,
    "submission_cand010": BaselineTier.TIER_3_NEURAL,
    "champion_submission": BaselineTier.TIER_3_NEURAL,
    "submission_2021": BaselineTier.TIER_3_NEURAL,
    "submission_2025": BaselineTier.TIER_3_NEURAL,
    "submission_2028": BaselineTier.TIER_3_NEURAL,
}


def classify_agent_tier(agent_identifier: str) -> BaselineTier:
    """Classifies an agent file path or name into its corresponding performance tier."""
    clean = os.path.basename(agent_identifier).lower()
    for key, tier in BASELINE_TIERS.items():
        if key in clean:
            return tier
    if "muzero" in clean or "champion" in clean or "cand" in clean:
        return BaselineTier.TIER_3_NEURAL
    if "hybrid" in clean:
        return BaselineTier.TIER_2_HYBRID
    return BaselineTier.TIER_1_HEURISTIC
