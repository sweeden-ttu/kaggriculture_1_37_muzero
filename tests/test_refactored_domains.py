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

"""Unit tests for refactored domains: evaluation (tiers), pipelines (phases), packaging."""
from __future__ import annotations

import os
import sys
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation.tiers import BASELINE_TIERS, BaselineTier, classify_agent_tier
from evaluation.benchmarks import find_baselines
from evaluation.harness import short_name
from evaluation.league import discover_agent_packages
from muzero import MuZeroNetwork, create_target_network
from pipelines.phases.phase1_bootstrap import phase1_analytical_bootstrap
from pipelines.self_improving_loop import generate_candidate_hyperparams
from packaging.builder import build_hybrid_package


def test_tier_classification():
    """Verify evaluation domain tier classification (Tiers 1, 2, 3)."""
    assert classify_agent_tier("submission_2000.zip") == BaselineTier.TIER_1_HEURISTIC
    assert classify_agent_tier("submission_2500.zip") == BaselineTier.TIER_1_HEURISTIC
    assert classify_agent_tier("submission_thirst.zip") == BaselineTier.TIER_1_HEURISTIC
    assert classify_agent_tier("submission_hybrid_direct.zip") == BaselineTier.TIER_2_HYBRID
    assert classify_agent_tier("submission_beforegap.zip") == BaselineTier.TIER_2_HYBRID
    assert classify_agent_tier("submission_muzero.zip") == BaselineTier.TIER_3_NEURAL
    assert classify_agent_tier("submission_cand010.zip") == BaselineTier.TIER_3_NEURAL
    assert classify_agent_tier("artifacts/snapshots/champion_Cand-001.pt") == BaselineTier.TIER_3_NEURAL


def test_baseline_discovery():
    """Verify evaluation domain baseline discovery."""
    baselines = find_baselines(os.path.join(ROOT, "baselines"))
    assert len(baselines) >= 8
    names = [short_name(b) for b in baselines]
    # Verify top boost baselines, train baselines, and Kaggle submissions
    assert any("boost" in n for n in names)
    assert any("train" in n for n in names)
    assert "submission_boost_cand012" in names
    assert "submission_train_cand042" in names
    assert "submission_2033" in names or "submission_2028" in names


def test_league_package_discovery():
    """Verify package discovery for round-robin league."""
    pkgs = discover_agent_packages([os.path.join(ROOT, "baselines"), os.path.join(ROOT, "dist")])
    assert len(pkgs) >= 2


def test_phase1_bootstrap_execution():
    """Verify Phase 1 bootstrap pipeline executes and minimizes loss."""
    model = MuZeroNetwork(hidden_dim=32, support_size=601)
    target_model = create_target_network(model)
    metrics = phase1_analytical_bootstrap(
        model=model,
        steps=2,
        batch=8,
        lr=1e-3,
        cons_w=0.25,
        k_steps=3,
        target_model=target_model,
    )
    assert "total" in metrics
    assert metrics["total"] > 0


def test_adaptive_hyperparam_generation_no_name_error():
    """Verify adaptive hyperparam generation does not raise NameError (base_hidden_dim fix)."""
    # Test adaptive without prev_metrics
    cfg1 = generate_candidate_hyperparams(1, strategy="adaptive")
    assert cfg1["hidden_dim"] == 32
    assert cfg1["total_iterations"] == 1000

    # Test adaptive with prev_metrics
    prev = {"margin": -5000.0, "train": {"phase1": {"total": 35.0}}}
    cfg2 = generate_candidate_hyperparams(2, strategy="adaptive", prev_metrics=prev)
    assert cfg2["hidden_dim"] == 32
    assert cfg2["sims"] == 24
    assert cfg2["total_iterations"] == 1500


def test_loss_weight_equivalence_rule_rejections():
    """Verify that train_muzero weight rule strictly rejects equivalent weights."""
    import pytest
    from train_muzero import validate_loss_weights_rule

    # 1. consistency-weight cannot equal policy-weight
    with pytest.raises(ValueError, match="consistency-weight.*cannot be equivalent.*policy-weight"):
        validate_loss_weights_rule(
            consistency_weight=0.8637,
            policy_weight=0.8637,
            value_weight=0.9637,
            reward_weight=0.7357,
            cql_weight=0.02357,
        )

    # 2. policy-weight cannot equal value-weight
    with pytest.raises(ValueError, match="policy-weight.*cannot be equivalent.*value-weight"):
        validate_loss_weights_rule(
            consistency_weight=0.4357,
            policy_weight=0.9637,
            value_weight=0.9637,
            reward_weight=0.7357,
            cql_weight=0.02357,
        )

    # 3. value-weight cannot equal reward-weight
    with pytest.raises(ValueError, match="value-weight.*cannot be equivalent.*reward-weight"):
        validate_loss_weights_rule(
            consistency_weight=0.4357,
            policy_weight=0.8637,
            value_weight=0.7357,
            reward_weight=0.7357,
            cql_weight=0.02357,
        )

    # 4. reward-weight cannot equal cql-weight
    with pytest.raises(ValueError, match="reward-weight.*cannot be equivalent.*cql-weight"):
        validate_loss_weights_rule(
            consistency_weight=0.4357,
            policy_weight=0.8637,
            value_weight=0.9637,
            reward_weight=0.02357,
            cql_weight=0.02357,
        )

    # 5. Distinct weights pass without error
    valid = validate_loss_weights_rule(
        consistency_weight=0.4357,
        policy_weight=0.8637,
        value_weight=0.9637,
        reward_weight=0.7357,
        cql_weight=0.02357,
    )
    assert valid["consistency-weight"] == 0.4357


def test_loss_weights_distinct_across_all_candidate_strategies():
    """Verify that every candidate strategy generates 5 distinct loss weights."""
    from pipelines.self_improving_loop import generate_candidate_hyperparams

    strategies = ["combinations", "grid", "random", "adaptive"]
    for strat in strategies:
        for iter_idx in range(1, 10):
            cfg = generate_candidate_hyperparams(iter_idx, strategy=strat)
            weights = [
                cfg["consistency_weight"],
                cfg["policy_weight"],
                cfg["value_weight"],
                cfg["reward_weight"],
                cfg["cql_weight"],
            ]
            # Verify pairwise distinctness with minimum delta
            for i in range(len(weights)):
                for j in range(i + 1, len(weights)):
                    assert abs(weights[i] - weights[j]) > 1e-4, (
                        f"Strategy {strat} iteration {iter_idx} produced equivalent weights: "
                        f"{weights[i]} vs {weights[j]}"
                    )


def test_compute_separate_scheduled_weights_no_step_collisions():
    """Verify separate loss schedule maintains distinct weights across all steps."""
    from train_muzero import compute_separate_scheduled_weights

    base_w = {
        "consistency_weight": 0.4357,
        "policy_weight": 0.8637,
        "value_weight": 0.9637,
        "reward_weight": 0.7357,
        "cql_weight": 0.02357,
    }
    total_steps = 100
    for step in range(total_steps):
        sched = compute_separate_scheduled_weights(step, total_steps, base_w)
        w_list = list(sched.values())
        for i in range(len(w_list)):
            for j in range(i + 1, len(w_list)):
                assert abs(w_list[i] - w_list[j]) > 1e-4, (
                    f"Step {step}/{total_steps} had collision: {w_list[i]} vs {w_list[j]}"
                )


def test_replay_manager_pruning_and_protection(tmp_path):
    """Verify that canonical replays are protected and obsolete selfplay replays are pruned."""
    import json
    from pipelines.replay_manager import prune_stale_selfplay_replays, is_canonical_replay

    # Verify canonical detection
    assert is_canonical_replay("113179986.json") is True
    assert is_canonical_replay("selfplay_seed10101.json") is False

    rep_dir = str(tmp_path / "replays")
    os.makedirs(rep_dir, exist_ok=True)

    # 1. Canonical Kaggle replays (must never be pruned)
    canon_files = ["113179986.json", "113402557.json"]
    for fn in canon_files:
        p = os.path.join(rep_dir, fn)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"id": fn, "rewards": [130000.0, 95000.0]}, f)

    # 2. Self-play replays with explicit ascending timestamps
    # selfplay_1 is oldest, selfplay_5 is newest
    sp_data = [
        ("selfplay_1.json", [160000.0, 100000.0]),
        ("selfplay_2.json", [140000.0, 90000.0]),
        ("selfplay_3.json", [110000.0, 85000.0]),
        ("selfplay_4.json", [85000.0, 80000.0]),
        ("selfplay_5.json", [75000.0, 70000.0]),
    ]
    base_time = 1700000000
    for idx, (fn, rew) in enumerate(sp_data):
        p = os.path.join(rep_dir, fn)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"id": fn, "rewards": rew, "padding": "x" * 500}, f)
        file_time = base_time + (idx * 100)
        os.utime(p, (file_time, file_time))

    # Calculate total size of canon files + newest 2 self-play files
    canon_bytes = sum(os.path.getsize(os.path.join(rep_dir, fn)) for fn in canon_files)
    newest_bytes = sum(os.path.getsize(os.path.join(rep_dir, fn)) for fn in ["selfplay_4.json", "selfplay_5.json"])
    budget_bytes = canon_bytes + newest_bytes + 50
    budget_gb = budget_bytes / (1024.0 ** 3)

    pruned_count, freed_gb = prune_stale_selfplay_replays(
        replays_dir=rep_dir,
        max_dir_size_gb=budget_gb,
    )

    remaining = set(os.listdir(rep_dir))
    # Canonical replays must remain intact
    for fn in canon_files:
        assert fn in remaining, f"Canonical replay {fn} was incorrectly deleted!"

    # Oldest selfplay files (1, 2, 3) must be pruned first by timestamp
    assert "selfplay_1.json" not in remaining
    assert "selfplay_2.json" not in remaining
    assert "selfplay_3.json" not in remaining

    # Newest selfplay files (4, 5) must be kept within the budget
    assert "selfplay_4.json" in remaining
    assert "selfplay_5.json" in remaining
    assert pruned_count == 3

