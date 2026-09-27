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
pipelines/low_rank_adaptation.py
================================
Low-Rank Adaptation (LoRA) decoupled loss weighting, scheduling, and distinctness validation
for Kaggriculture MuZero.

Implements the standard softmax-scheduled loss adaptation with consistency regimes:
  - Regime 1: --consistency-weight in [0.1, 0.5]
  - Regime 2: --consistency-weight in [0.5, 1.0]

Strictly enforces the distinctness rule at every step:
  consistency-weight != policy-weight != value-weight != reward-weight != cql-weight
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Union, overload


def validate_loss_weights_rule(
    consistency_weight: float,
    policy_weight: float,
    value_weight: float,
    reward_weight: float,
    cql_weight: float,
    tolerance: float = 1e-4,
    auto_adjust: bool = False,
) -> Dict[str, float]:
    """
    Enforces the MuZero training loss weight distinctness rule:
      1. 'consistency-weight' cannot be equivalent in value to 'policy-weight'
      2. 'policy-weight' cannot have equivalent value of 'value-weight'
      3. 'value-weight' cannot be equivalent in value to 'reward-weight'
      4. 'reward-weight' cannot be equivalent in value to 'cql-weight'
      5. All five loss weights must be slightly different and mutually non-equivalent.

    Args:
        consistency_weight: Latent state consistency weight (λ_cons)
        policy_weight: Policy cross-entropy / KL weight (λ_pol)
        value_weight: Value target bootstrapping weight (λ_val)
        reward_weight: Step transition reward weight (λ_rew)
        cql_weight: Conservative Q-learning penalty weight (λ_cql)
        tolerance: Maximum absolute difference to consider two weights equivalent.
        auto_adjust: If True, slightly jitters colliding weights deterministically
                     instead of raising ValueError.

    Returns:
        Dict[str, float] with validated (or adjusted) weights.

    Raises:
        ValueError: If any pair of weights is equivalent and auto_adjust is False.
    """
    weights = {
        "consistency-weight": float(consistency_weight),
        "policy-weight": float(policy_weight),
        "value-weight": float(value_weight),
        "reward-weight": float(reward_weight),
        "cql-weight": float(cql_weight),
    }

    if auto_adjust:
        adjusted = dict(weights)
        keys = list(adjusted.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k1, k2 = keys[i], keys[j]
                if abs(adjusted[k1] - adjusted[k2]) < tolerance:
                    adjusted[k2] = round(adjusted[k2] + 0.0157 * (j + 1), 5)
        return adjusted

    # Explicitly check chain of rules specified by user
    chain_pairs = [
        ("consistency-weight", "policy-weight"),
        ("policy-weight", "value-weight"),
        ("value-weight", "reward-weight"),
        ("reward-weight", "cql-weight"),
    ]
    for name1, name2 in chain_pairs:
        v1, v2 = weights[name1], weights[name2]
        if abs(v1 - v2) < tolerance:
            raise ValueError(
                f"Weight rule violation in train_muzero.py: '{name1}' ({v1:.5f}) cannot be equivalent in value "
                f"to '{name2}' ({v2:.5f}). The rule requires that 'consistency-weight' != 'policy-weight' "
                f"!= 'value-weight' != 'reward-weight' != 'cql-weight'. All 5 weights must have slightly different values."
            )

    # Generalize across all pairs among the 5 weights
    items = list(weights.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            name1, v1 = items[i]
            name2, v2 = items[j]
            if abs(v1 - v2) < tolerance:
                raise ValueError(
                    f"Weight rule violation in train_muzero.py: '{name1}' ({v1:.5f}) cannot be equivalent in value "
                    f"to '{name2}' ({v2:.5f}). All loss weights must have distinct, slightly different values."
                )

    return weights


def softmax_weights_high_low(
    high: float,
    low: float,
    progress: float,
    direction: str = "down",
    temperature: float = 0.25,
) -> float:
    """
    Computes a smooth softmax transition between high and low weights.

    Args:
        high: The high weight value.
        low: The low weight value.
        progress: Progress fraction in [0.0, 1.0].
        direction: Direction of transition:
            - 'down' (starts high, goes to low): high -> low
            - 'up'   (starts low, goes to high): low -> high
        temperature: Temperature parameter for the softmax transition (default 0.25).

    Returns:
        float: The scheduled weight at the given progress.
    """
    p = max(0.0, min(1.0, float(progress)))
    tau = max(1e-4, float(temperature))
    z_high = (1.0 - p) / tau
    z_low = p / tau
    m = max(z_high, z_low)
    e_high = math.exp(z_high - m)
    e_low = math.exp(z_low - m)
    pi_high = e_high / (e_high + e_low)

    # Boundary calibration to guarantee exact high/low endpoints at p=0 and p=1
    pi_0 = 1.0 / (1.0 + math.exp(-1.0 / tau))
    pi_1 = 1.0 - pi_0
    denom = pi_0 - pi_1
    alpha = max(0.0, min(1.0, (pi_high - pi_1) / denom if denom > 1e-9 else 1.0 - p))

    d = str(direction).lower()
    if d in ("down", "high_to_low", "-1", "-"):
        return low + (high - low) * alpha
    else:
        return low + (high - low) * (1.0 - alpha)


@overload
def compute_softmax_weights(
    logits: Dict[str, float],
    total_scale: float = 3.18067,
    temperature: float = 1.0,
) -> Dict[str, float]: ...


@overload
def compute_softmax_weights(
    logits: Sequence[float],
    total_scale: float = 3.18067,
    temperature: float = 1.0,
) -> List[float]: ...


def compute_softmax_weights(
    logits: Union[Dict[str, float], Sequence[float]],
    total_scale: float = 3.18067,
    temperature: float = 1.0,
) -> Union[Dict[str, float], List[float]]:
    """
    Computes softmax normalized weights scaled by total_scale.
    Useful for generating high/low weights from logit priorities.
    """
    tau = max(1e-4, float(temperature))
    if isinstance(logits, dict):
        keys = list(logits.keys())
        vals = [float(logits[k]) / tau for k in keys]
        m = max(vals)
        exp_v = [math.exp(v - m) for v in vals]
        s = sum(exp_v)
        probs = [ev / s for ev in exp_v]
        return {k: round(p * total_scale, 5) for k, p in zip(keys, probs)}
    else:
        vals = [float(v) / tau for v in logits]
        m = max(vals)
        exp_v = [math.exp(v - m) for v in vals]
        s = sum(exp_v)
        probs = [ev / s for ev in exp_v]
        return [round(p * total_scale, 5) for p in probs]


def compute_low_rank_adaptation_scheduled_weights(
    step: int,
    total_steps: int,
    base_weights: Dict[str, Any],
    start_min: float = 0.1357,
    end_max: float = 0.9637,
) -> Dict[str, float]:
    """
    Computes Low-Rank Adaptation (LoRA) gradient schedules for the loss weights using softmax
    transitions between high and low anchors governed by consistency weight.

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

    # Consistency regime: 0.1 to 0.5 vs 0.5 to 1.0
    cons_base = float(base_weights.get("consistency_weight", 0.4357))
    is_low_cons_regime = (cons_base <= 0.50)

    # Base high and low anchors
    high_val = float(base_weights.get("value_weight", end_max))
    high_pol = float(base_weights.get("policy_weight", 0.8970))
    high_rew = float(base_weights.get("reward_weight", 0.7357))
    high_cql = max(float(base_weights.get("cql_weight", 0.07857)), 0.07857)

    low_rew = start_min                  # 0.1357
    low_pol = start_min + 0.0400         # 0.1757
    low_val = start_min + 0.0600         # 0.1957
    low_cql = round(start_min * 0.1, 5)  # 0.01357

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
        # policy starts high goes to low (down)
        # value starts low goes to high (up)
        # reward starts high goes to low (down)
        # cql starts low goes to high (up)
        low_cons = 0.1000
        high_cons = min(0.50, max(0.3557, cons_base))
        dir_cons = dir_map.get("consistency", "up")
        dir_pol = dir_map.get("policy", "down")
        dir_val = dir_map.get("value", "up")
        dir_rew = dir_map.get("reward", "down")
        dir_cql = dir_map.get("cql", "up")
    else:
        # Regime 2: consistency 0.5 to 1.0
        # policy starts low goes to high (up)
        # value starts high goes to low (down)
        # reward starts low goes to high (up)
        # cql starts high goes to low (down)
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
compute_separate_scheduled_weights = compute_low_rank_adaptation_scheduled_weights
