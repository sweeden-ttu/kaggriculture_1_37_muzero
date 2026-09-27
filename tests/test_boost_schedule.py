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

"""Tests for boost mode inverted loss schedule."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
from pipelines.boost_schedule import compute_boost_scheduled_weights
from pipelines.loss_schedule import validate_loss_weights_rule


def test_softmax_weights_high_low_properties():
    """Verify smooth softmax transition, boundary calibration, and monotonicity."""
    from pipelines.loss_schedule import softmax_weights_high_low

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


def test_regime1_consistency_0_1_to_0_5():
    """
    Verify Regime 1 (--consistency-weight 0.1 to 0.5):
      --policy-weight   starts from high goes to low
      --value-weight    starts from low goes to high
      --reward-weight   starts from high goes to low
      --cql-weight      starts from low goes to high
      --consistency-weight 0.1 to 0.5 (low to high)
    """
    base_w = {
        "consistency_weight": 0.3557,  # in [0.1, 0.5]
        "policy_weight": 0.8970,
        "value_weight": 0.9637,
        "reward_weight": 0.7357,
        "cql_weight": 0.07857,
    }
    total_steps = 100

    w_start = compute_boost_scheduled_weights(step=0, total_steps=total_steps, base_weights=base_w)
    w_end = compute_boost_scheduled_weights(step=total_steps - 1, total_steps=total_steps, base_weights=base_w)

    # Policy starts high, goes low
    assert w_start["policy_weight"] > w_end["policy_weight"], "Regime 1: policy starts high goes low"
    assert w_start["policy_weight"] >= 0.85
    assert w_end["policy_weight"] <= 0.25

    # Value starts low, goes high
    assert w_end["value_weight"] > w_start["value_weight"], "Regime 1: value starts low goes high"
    assert w_start["value_weight"] <= 0.25
    assert w_end["value_weight"] >= 0.90

    # Reward starts high, goes low
    assert w_start["reward_weight"] > w_end["reward_weight"], "Regime 1: reward starts high goes low"
    assert w_start["reward_weight"] >= 0.70
    assert w_end["reward_weight"] <= 0.20

    # CQL starts low, goes high
    assert w_end["cql_weight"] > w_start["cql_weight"], "Regime 1: cql starts low goes high"
    assert w_start["cql_weight"] <= 0.02
    assert w_end["cql_weight"] >= 0.07

    # Consistency starts low (0.1), goes high (0.5)
    assert w_end["consistency_weight"] > w_start["consistency_weight"], "Regime 1: consistency 0.1 to 0.5"
    assert w_start["consistency_weight"] <= 0.15
    assert w_end["consistency_weight"] >= 0.35


def test_regime2_consistency_0_5_to_1_0():
    """
    Verify Regime 2 (if --consistency-weight 0.5 to 1.0):
      --policy-weight   starts from low goes to high
      --value-weight    starts high goes to low
      --reward-weight   starts from low goes to high
      --cql-weight      starts from high goes to low
      --consistency-weight starts high goes to 0.5
    """
    base_w = {
        "consistency_weight": 0.7557,  # in [0.5, 1.0]
        "policy_weight": 0.8970,
        "value_weight": 0.9637,
        "reward_weight": 0.7357,
        "cql_weight": 0.07857,
    }
    total_steps = 100

    w_start = compute_boost_scheduled_weights(step=0, total_steps=total_steps, base_weights=base_w)
    w_end = compute_boost_scheduled_weights(step=total_steps - 1, total_steps=total_steps, base_weights=base_w)

    # Policy starts low, goes high
    assert w_end["policy_weight"] > w_start["policy_weight"], "Regime 2: policy starts low goes high"
    assert w_start["policy_weight"] <= 0.25
    assert w_end["policy_weight"] >= 0.85

    # Value starts high, goes low
    assert w_start["value_weight"] > w_end["value_weight"], "Regime 2: value starts high goes low"
    assert w_start["value_weight"] >= 0.90
    assert w_end["value_weight"] <= 0.25

    # Reward starts low, goes high
    assert w_end["reward_weight"] > w_start["reward_weight"], "Regime 2: reward starts low goes high"
    assert w_start["reward_weight"] <= 0.20
    assert w_end["reward_weight"] >= 0.70

    # CQL starts high, goes low
    assert w_start["cql_weight"] > w_end["cql_weight"], "Regime 2: cql starts high goes low"
    assert w_start["cql_weight"] >= 0.07
    assert w_end["cql_weight"] <= 0.02

    # Consistency starts high, goes to 0.5 (starts high, goes low)
    assert w_start["consistency_weight"] > w_end["consistency_weight"], "Regime 2: consistency starts high goes to 0.5"
    assert w_start["consistency_weight"] >= 0.70
    assert w_end["consistency_weight"] <= 0.55


def test_regimes_are_exact_opposite_gradients():
    """Verify that Regime 1 and Regime 2 test opposing gradients in all 5 weights."""
    total_steps = 100

    # Regime 1: consistency in [0.1, 0.5]
    r1_base = {
        "consistency_weight": 0.4357,
        "policy_weight": 0.8970,
        "value_weight": 0.9637,
        "reward_weight": 0.7357,
        "cql_weight": 0.07857,
    }
    r1_start = compute_boost_scheduled_weights(0, total_steps, r1_base)
    r1_end = compute_boost_scheduled_weights(total_steps - 1, total_steps, r1_base)

    # Regime 2: consistency in [0.5, 1.0]
    r2_base = {
        "consistency_weight": 0.7557,
        "policy_weight": 0.8970,
        "value_weight": 0.9637,
        "reward_weight": 0.7357,
        "cql_weight": 0.07857,
    }
    r2_start = compute_boost_scheduled_weights(0, total_steps, r2_base)
    r2_end = compute_boost_scheduled_weights(total_steps - 1, total_steps, r2_base)

    for key in ("consistency_weight", "policy_weight", "value_weight", "reward_weight", "cql_weight"):
        delta_r1 = r1_end[key] - r1_start[key]
        delta_r2 = r2_end[key] - r2_start[key]
        # Exact opposite signs: one goes up (+), the other goes down (-)
        assert delta_r1 * delta_r2 < 0, f"{key} must have opposite gradient directions between Regime 1 and Regime 2"


def test_opposing_gradient_hyperparameter_strategy():
    """Verify that gradient strategy tests weights going against each other in opposite directions."""
    from pipelines.self_improving_loop import generate_candidate_hyperparams

    # Test pairs of candidates across iterations 1 to 20
    for pair_idx in range(10):
        it_pos = 2 * pair_idx + 1
        it_neg = 2 * pair_idx + 2

        cfg_pos = generate_candidate_hyperparams(iteration=it_pos, strategy="gradient", base_iterations=719)
        cfg_neg = generate_candidate_hyperparams(iteration=it_neg, strategy="gradient", base_iterations=719)

        # Both configs must strictly satisfy distinctness rule
        validate_loss_weights_rule(
            consistency_weight=cfg_pos["consistency_weight"],
            policy_weight=cfg_pos["policy_weight"],
            value_weight=cfg_pos["value_weight"],
            reward_weight=cfg_pos["reward_weight"],
            cql_weight=cfg_pos["cql_weight"],
            auto_adjust=False,
        )
        validate_loss_weights_rule(
            consistency_weight=cfg_neg["consistency_weight"],
            policy_weight=cfg_neg["policy_weight"],
            value_weight=cfg_neg["value_weight"],
            reward_weight=cfg_neg["reward_weight"],
            cql_weight=cfg_neg["cql_weight"],
            auto_adjust=False,
        )

        info_pos = cfg_pos["gradient_info"]
        info_neg = cfg_neg["gradient_info"]

        assert info_pos["is_positive"] is True
        assert info_neg["is_positive"] is False
        assert info_pos["pair_index"] == pair_idx + 1
        assert info_neg["pair_index"] == pair_idx + 1
        assert info_pos["opposite_candidate_id"] == f"Cand-{it_neg:03d}"
        assert info_neg["opposite_candidate_id"] == f"Cand-{it_pos:03d}"

        # Verify weights moved in opposite directions between the pair
        for key in ("consistency_weight", "policy_weight", "value_weight", "reward_weight", "cql_weight"):
            delta_pos = cfg_pos[key] - 0.5  # relative shift from center
            delta_neg = cfg_neg[key] - 0.5
            # At least one weight must have opposite sign deltas across the pair
            # For the active pair keys, pos and neg must move in opposite directions:
            # (cfg_pos[key] - base) * (cfg_neg[key] - base) <= 0
        
        # Verify specific antagonistic pairs where weights go against each other:
        if pair_idx == 0:
            # Cand 1: consistency up (+), value down (-) -> they go against each other!
            # Cand 2: consistency down (-), value up (+) -> exact opposite!
            assert cfg_pos["consistency_weight"] > cfg_neg["consistency_weight"]
            assert cfg_pos["value_weight"] < cfg_neg["value_weight"]
        elif pair_idx == 1:
            # Cand 3: policy up (+), reward down (-) -> they go against each other!
            # Cand 4: policy down (-), reward up (+) -> exact opposite!
            assert cfg_pos["policy_weight"] > cfg_neg["policy_weight"]
            assert cfg_pos["reward_weight"] < cfg_neg["reward_weight"]
        elif pair_idx == 2:
            # Cand 5: consistency up (+), policy down (-) -> they go against each other!
            # Cand 6: consistency down (-), policy up (+) -> exact opposite!
            assert cfg_pos["consistency_weight"] > cfg_neg["consistency_weight"]
            assert cfg_pos["policy_weight"] < cfg_neg["policy_weight"]
        elif pair_idx == 3:
            # Cand 7: value up (+), reward down (-) -> they go against each other!
            # Cand 8: value down (-), reward up (+) -> exact opposite!
            assert cfg_pos["value_weight"] > cfg_neg["value_weight"]
            assert cfg_pos["reward_weight"] < cfg_neg["reward_weight"]
        elif pair_idx == 4:
            # Cand 9: cql up (+), policy down (-) -> they go against each other!
            # Cand 10: cql down (-), policy up (+) -> exact opposite!
            assert cfg_pos["cql_weight"] > cfg_neg["cql_weight"]
            assert cfg_pos["policy_weight"] < cfg_neg["policy_weight"]


def test_compute_softmax_weights_distribution():
    """Verify compute_softmax_weights scales to budget and reflects relative logits."""
    from pipelines.loss_schedule import compute_softmax_weights

    logits = {
        "consistency": -0.5,
        "policy": 1.2,
        "value": -0.2,
        "reward": 0.8,
        "cql": -1.8,
    }
    budget = 3.18067
    w = compute_softmax_weights(logits, total_scale=budget, temperature=1.0)

    # Sum of weights matches budget
    assert abs(sum(w.values()) - budget) < 1e-3

    # Policy has highest logit -> highest weight
    assert w["policy"] == max(w.values())
    # CQL has lowest logit -> lowest weight
    assert w["cql"] == min(w.values())


def test_separate_loss_schedule_regimes():
    """Verify compute_separate_scheduled_weights follows Regime 1 and Regime 2 with softmax."""
    from pipelines.loss_schedule import compute_separate_scheduled_weights

    total_steps = 100

    # Regime 1: consistency in [0.1, 0.5]
    r1 = {
        "consistency_weight": 0.4000,
        "policy_weight": 0.8970,
        "value_weight": 0.9637,
        "reward_weight": 0.7357,
        "cql_weight": 0.07857,
    }
    r1_start = compute_separate_scheduled_weights(0, total_steps, r1)
    r1_end = compute_separate_scheduled_weights(total_steps - 1, total_steps, r1)

    assert r1_start["policy_weight"] > r1_end["policy_weight"]  # starts high goes low
    assert r1_end["value_weight"] > r1_start["value_weight"]    # starts low goes high
    assert r1_start["reward_weight"] > r1_end["reward_weight"]  # starts high goes low
    assert r1_end["cql_weight"] > r1_start["cql_weight"]        # starts low goes high
    assert r1_end["consistency_weight"] > r1_start["consistency_weight"]  # 0.1 to 0.5

    # Regime 2: consistency in [0.5, 1.0]
    r2 = {
        "consistency_weight": 0.7000,
        "policy_weight": 0.8970,
        "value_weight": 0.9637,
        "reward_weight": 0.7357,
        "cql_weight": 0.07857,
    }
    r2_start = compute_separate_scheduled_weights(0, total_steps, r2)
    r2_end = compute_separate_scheduled_weights(total_steps - 1, total_steps, r2)

    assert r2_end["policy_weight"] > r2_start["policy_weight"]  # starts low goes high
    assert r2_start["value_weight"] > r2_end["value_weight"]    # starts high goes low
    assert r2_end["reward_weight"] > r2_start["reward_weight"]  # starts low goes high
    assert r2_start["cql_weight"] > r2_end["cql_weight"]        # starts high goes low
    assert r2_start["consistency_weight"] > r2_end["consistency_weight"]  # starts high goes to 0.5


