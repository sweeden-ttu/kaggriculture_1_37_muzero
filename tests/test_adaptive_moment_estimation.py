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

"""Tests for Adaptive Moment Estimation (Adam) and Low-Rank Adaptation (LoRA) loss schedules."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
from pipelines.adaptive_moment_estimation import (
    compute_adaptive_moment_scheduled_weights,
    compute_boost_scheduled_weights,
)
from pipelines.low_rank_adaptation import (
    compute_low_rank_adaptation_scheduled_weights,
    compute_separate_scheduled_weights,
    compute_softmax_weights,
    softmax_weights_high_low,
    validate_loss_weights_rule,
)


def test_softmax_weights_high_low_properties():
    """Verify smooth softmax transition, boundary calibration, and monotonicity."""
    high = 0.8970
    low = 0.1757

    # Direction down: starts from high goes to low
    assert abs(softmax_weights_high_low(high, low, progress=0.0, direction="down") - high) < 1e-5
    assert abs(softmax_weights_high_low(high, low, progress=1.0, direction="down") - low) < 1e-5

    # Direction up: starts from low goes to high
    assert abs(softmax_weights_high_low(high, low, progress=0.0, direction="up") - low) < 1e-5
    assert abs(softmax_weights_high_low(high, low, progress=1.0, direction="up") - high) < 1e-5

    # Verify smooth monotonicity across progress steps
    prev_down = high
    prev_up = low
    for step in range(1, 20):
        p = step / 20.0
        cur_down = softmax_weights_high_low(high, low, progress=p, direction="down")
        cur_up = softmax_weights_high_low(high, low, progress=p, direction="up")
        assert cur_down < prev_down, "downward softmax schedule must be monotonically decreasing"
        assert cur_up > prev_up, "upward softmax schedule must be monotonically increasing"
        prev_down = cur_down
        prev_up = cur_up


def test_adaptive_moment_regime1():
    """
    Verify Regime 1 (--consistency-weight 0.1 to 0.5):
      --policy-weight   starts from high goes to low
      --value-weight    starts from low goes to high
      --reward-weight   starts from high goes to low
      --cql-weight      starts from low goes to high
      --consistency-weight 0.1 to 0.5 (low to high)
    """
    base_w = {
        "consistency_weight": 0.3557,
        "policy_weight": 0.8970,
        "value_weight": 0.9637,
        "reward_weight": 0.7357,
        "cql_weight": 0.07857,
    }
    total_steps = 100

    w_start = compute_adaptive_moment_scheduled_weights(step=0, total_steps=total_steps, base_weights=base_w)
    w_end = compute_adaptive_moment_scheduled_weights(step=total_steps - 1, total_steps=total_steps, base_weights=base_w)

    assert w_start["policy_weight"] > w_end["policy_weight"]
    assert w_end["value_weight"] > w_start["value_weight"]
    assert w_start["reward_weight"] > w_end["reward_weight"]
    assert w_end["cql_weight"] > w_start["cql_weight"]
    assert w_end["consistency_weight"] > w_start["consistency_weight"]


def test_adaptive_moment_regime2():
    """
    Verify Regime 2 (if --consistency-weight 0.5 to 1.0):
      --policy-weight   starts from low goes to high
      --value-weight    starts high goes to low
      --reward-weight   starts from low goes to high
      --cql-weight      starts from high goes to low
      --consistency-weight starts high goes to 0.5
    """
    base_w = {
        "consistency_weight": 0.7557,
        "policy_weight": 0.8970,
        "value_weight": 0.9637,
        "reward_weight": 0.7357,
        "cql_weight": 0.07857,
    }
    total_steps = 100

    w_start = compute_adaptive_moment_scheduled_weights(step=0, total_steps=total_steps, base_weights=base_w)
    w_end = compute_adaptive_moment_scheduled_weights(step=total_steps - 1, total_steps=total_steps, base_weights=base_w)

    assert w_end["policy_weight"] > w_start["policy_weight"]
    assert w_start["value_weight"] > w_end["value_weight"]
    assert w_end["reward_weight"] > w_start["reward_weight"]
    assert w_start["cql_weight"] > w_end["cql_weight"]
    assert w_start["consistency_weight"] > w_end["consistency_weight"]


def test_low_rank_adaptation_regimes():
    """Verify compute_low_rank_adaptation_scheduled_weights matches compute_separate_scheduled_weights."""
    base_w = {
        "consistency_weight": 0.3557,
        "policy_weight": 0.8970,
        "value_weight": 0.9637,
        "reward_weight": 0.7357,
        "cql_weight": 0.07857,
    }
    w1 = compute_low_rank_adaptation_scheduled_weights(step=50, total_steps=100, base_weights=base_w)
    w2 = compute_separate_scheduled_weights(step=50, total_steps=100, base_weights=base_w)
    assert w1 == w2
