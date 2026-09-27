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

"""Unit tests for Unified Online/Offline Hybrid Trainer & Hyperparameter Sweep."""
from __future__ import annotations

import os
import pytest

from pipelines.hybrid_trainer import (
    generate_candidate_hyperparams,
    evaluate_candidate_agent,
    run_hybrid_trainer_loop,
)
from pipelines.low_rank_adaptation import validate_loss_weights_rule


def test_hyperparameter_generators_rule_compliance():
    """Verify all sweep strategies generate distinct loss weight vectors conforming to governance."""
    strategies = ["adaptive", "grid", "random", "gradient"]
    for strat in strategies:
        for it in range(1, 7):
            cfg = generate_candidate_hyperparams(
                iteration=it,
                strategy=strat,
                base_iterations=100,
            )
            assert "consistency_weight" in cfg
            assert "policy_weight" in cfg
            assert "value_weight" in cfg
            assert "reward_weight" in cfg
            assert "cql_weight" in cfg

            # Governance: mutual distinctness of 5 loss dials
            weights = [
                cfg["consistency_weight"],
                cfg["policy_weight"],
                cfg["value_weight"],
                cfg["reward_weight"],
                cfg["cql_weight"],
            ]
            assert len(weights) == len(set(weights)), f"Strategy {strat} iteration {it} produced duplicate weights"
            assert validate_loss_weights_rule(
                cfg["consistency_weight"],
                cfg["policy_weight"],
                cfg["value_weight"],
                cfg["reward_weight"],
                cfg["cql_weight"],
            )


def test_evaluate_candidate_agent_seasonal_progression(tmp_path):
    """Verify seasonal progression evaluation produces valid scores, margins, and win percentages."""
    from muzero import KaggricultureMuZeroChassis, save_spatial_checkpoint

    model = KaggricultureMuZeroChassis()
    ckpt_path = str(tmp_path / "test_spatial_ckpt.pt")
    save_spatial_checkpoint(model, ckpt_path)

    metrics = evaluate_candidate_agent(
        candidate_ckpt=ckpt_path,
        champion_score=100000.0,
        eval_episodes=5,
        n_workers=1,
    )
    assert metrics["n_episodes"] == 5
    assert isinstance(metrics["avg_score"], float)
    assert isinstance(metrics["margin"], float)
    assert 0.0 <= metrics["win_pct"] <= 100.0


def test_hybrid_trainer_loop_smoke(tmp_path):
    """Run a tiny 1-step offline trial smoke test."""
    code = run_hybrid_trainer_loop(
        mode="offline",
        sweep_strategy="adaptive",
        iterations=1,
        eval_episodes=2,
        eval_workers=1,
        base_iterations=2,
        batch=4,
        sims=4,
    )
    assert code == 0
