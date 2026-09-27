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
pipelines/adaptive_moment_estimation.py
=======================================
Adaptive Moment Estimation (Adam) loss schedule for Kaggriculture MuZero.

Implements antagonistic and opposing gradient schedules across the 5 loss weights
governed by the consistency regimes with smooth softmax transitions:
  - Regime 1: --consistency-weight in [0.1, 0.5]
      policy: starts high goes down
      value: starts low goes up
      reward: starts high goes down
      cql: starts low goes up
      consistency: progresses 0.1 to 0.5
  - Regime 2: --consistency-weight in [0.5, 1.0]
      policy: starts low goes up
      value: starts high goes down
      reward: starts low goes up
      cql: starts high goes down
      consistency: starts high goes to 0.5

Strictly enforces the distinctness rule at every step:
  consistency-weight != policy-weight != value-weight != reward-weight != cql-weight
"""
from __future__ import annotations

import sys
from typing import Any, Dict

from .low_rank_adaptation import softmax_weights_high_low, validate_loss_weights_rule


def compute_adaptive_moment_scheduled_weights(
    step: int,
    total_steps: int,
    base_weights: Dict[str, float],
    start_max: float = 0.9637,
    end_min: float = 0.1357,
) -> Dict[str, float]:
    """
    Computes Adaptive Moment Estimation (Adam) gradient schedules using softmax transitions between
    high and low anchors governed by consistency weight.

    Regime 1: --consistency-weight in [0.1, 0.5]:
      - consistency-weight: progresses from 0.1 to 0.5 (low -> high, up)
      - policy-weight: starts from high goes to low (down)
      - value-weight: starts from low goes to high (up)
      - reward-weight: starts from high goes to low (down)
      - cql-weight: starts from low goes to high (up)

    Regime 2: --consistency-weight in [0.5, 1.0]:
      - consistency-weight: in [0.5, 1.0] (starts high goes to 0.5, down)
      - policy-weight: starts from low goes to high (up)
      - value-weight: starts high goes to low (down)
      - reward-weight: starts from low goes to high (up)
      - cql-weight: starts from high goes to low (down)

    All transitions use softmax: w(p) = softmax([z_high, z_low]) @ [high, low].
    Maintains distinct, slightly different values at every step throughout training.
    """
    p = float(step) / float(max(1, total_steps - 1))
    p = max(0.0, min(1.0, p))

    # Determine consistency regime: 0.1 to 0.5 vs 0.5 to 1.0
    cons_base = float(base_weights.get("consistency_weight", 0.4357))
    is_low_cons_regime = (cons_base <= 0.50)

    # Base high and low anchors
    high_val = float(base_weights.get("value_weight", start_max))
    high_pol = float(base_weights.get("policy_weight", 0.8970))
    high_rew = float(base_weights.get("reward_weight", 0.7357))
    high_cql = max(float(base_weights.get("cql_weight", 0.07857)), 0.07857)

    low_rew = end_min                  # 0.1357
    low_pol = end_min + 0.0400         # 0.1757
    low_val = end_min + 0.0600         # 0.1957
    low_cql = round(end_min * 0.1, 5)  # 0.01357

    # Decoupled progression curves
    p_rew = min(1.0, p / 0.40)
    p_cons = min(1.0, p / 0.60)
    p_pol = 0.0 if p < 0.15 else min(1.0, (p - 0.15) / 0.70)
    p_val = p
    p_cql = min(1.0, p / 0.80)

    # Check for manual overrides if specified
    directions = base_weights.get("schedule_directions")
    dir_map = {}
    if isinstance(directions, dict):
        dir_map = directions
    elif isinstance(directions, str):
        for item in directions.split(","):
            if ":" in item:
                k, v = item.strip().split(":", 1)
                dir_map[k.strip()] = v.strip().lower()

    temperature = float(base_weights.get("schedule_temperature", 0.25))

    if is_low_cons_regime:
        # Regime 1: consistency 0.1 to 0.5
        # policy starts from high goes to low (down)
        # value starts from low goes to high (up)
        # reward starts from high goes to low (down)
        # cql starts from low goes to high (up)
        low_cons = 0.1000
        high_cons = min(0.50, max(0.3557, cons_base))
        dir_cons = dir_map.get("consistency", "up")
        dir_pol = dir_map.get("policy", "down")
        dir_val = dir_map.get("value", "up")
        dir_rew = dir_map.get("reward", "down")
        dir_cql = dir_map.get("cql", "up")
    else:
        # Regime 2: consistency 0.5 to 1.0
        # policy starts from low goes to high (up)
        # value starts high goes to low (down)
        # reward starts from low goes to high (up)
        # cql starts from high goes to low (down)
        high_cons = max(0.5057, min(1.0, cons_base if cons_base > 0.50 else 0.85))
        low_cons = 0.5000
        dir_cons = dir_map.get("consistency", "down")
        dir_pol = dir_map.get("policy", "up")
        dir_val = dir_map.get("value", "down")
        dir_rew = dir_map.get("reward", "up")
        dir_cql = dir_map.get("cql", "down")

    w_cons = softmax_weights_high_low(high_cons, low_cons, p_cons, dir_cons, temperature)
    w_pol = softmax_weights_high_low(high_pol, low_pol, p_pol, dir_pol, temperature)
    w_val = softmax_weights_high_low(high_val, low_val, p_val, dir_val, temperature)
    w_rew = softmax_weights_high_low(high_rew, low_rew, p_rew, dir_rew, temperature)
    w_cql = softmax_weights_high_low(high_cql, low_cql, p_cql, dir_cql, temperature)

    raw = {
        "consistency-weight": round(w_cons, 5),
        "policy-weight": round(w_pol, 5),
        "value-weight": round(w_val, 5),
        "reward-weight": round(w_rew, 5),
        "cql-weight": round(w_cql, 5),
    }

    adjusted = validate_loss_weights_rule(
        consistency_weight=raw["consistency-weight"],
        policy_weight=raw["policy-weight"],
        value_weight=raw["value-weight"],
        reward_weight=raw["reward-weight"],
        cql_weight=raw["cql-weight"],
        auto_adjust=True,
    )

    return {
        "consistency_weight": adjusted["consistency-weight"],
        "policy_weight": adjusted["policy-weight"],
        "value_weight": adjusted["value-weight"],
        "reward_weight": adjusted["reward-weight"],
        "cql_weight": adjusted["cql-weight"],
    }


# Backward-compatible alias for existing callers
compute_boost_scheduled_weights = compute_adaptive_moment_scheduled_weights


def apply_adaptive_moment_estimation_schedule() -> None:
    """
    Dynamically attaches the Adaptive Moment Estimation schedule to all relevant training phases.
    Ensures that when scheduled loss weighting is requested, the Adaptive Moment Estimation
    gradient schedule is executed.
    """
    import pipelines.low_rank_adaptation as lra
    lra.compute_low_rank_adaptation_scheduled_weights = compute_adaptive_moment_scheduled_weights
    lra.compute_separate_scheduled_weights = compute_adaptive_moment_scheduled_weights

    try:
        import pipelines.loss_schedule as ls
        ls.compute_separate_scheduled_weights = compute_adaptive_moment_scheduled_weights
        if hasattr(ls, "compute_low_rank_adaptation_scheduled_weights"):
            ls.compute_low_rank_adaptation_scheduled_weights = compute_adaptive_moment_scheduled_weights
    except ImportError:
        pass

    # Patch imported symbols in modules if already loaded
    for mod_name in (
        "pipelines.phases.phase1_bootstrap",
        "pipelines.phases.phase2_distill",
        "pipelines.train_muzero",
        "train_muzero",
    ):
        if mod_name in sys.modules:
            setattr(sys.modules[mod_name], "compute_separate_scheduled_weights", compute_adaptive_moment_scheduled_weights)
            setattr(sys.modules[mod_name], "compute_low_rank_adaptation_scheduled_weights", compute_adaptive_moment_scheduled_weights)


# Backward-compatible alias for existing callers
apply_boost_schedule = apply_adaptive_moment_estimation_schedule
