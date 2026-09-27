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

"""Comprehensive test suite for the Complete Prioritized MuZero System (docs/)."""

from __future__ import annotations

import math
import os
import sys
import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from muzero import (
    FactorizedKaggriculturePredictionHead,
    GameTrajectory,
    KaggricultureMuZeroChassis,
    KaggricultureObservationEncoder,
    MCTSNode,
    MinMaxStats,
    MuZeroTrajectoryBuffer,
    MuZeroTrainer,
    PrioritizedMuZeroBuffer,
    PrioritizedMuZeroTrainer,
    ResNetBlock2D,
    SampledMuZeroMCTS,
    SimSiamConsistencyLoss,
    SimSiamProjectionPredictionHead,
    SpatialDynamicsNetwork,
    SpatialPredictionNetwork,
    SpatialRepresentationNetwork,
    compute_muzero_unroll_loss_with_consistency,
)
from evaluation.league import (
    LeagueTracker,
    PFSPSampler,
    select_matchmaking_opponent,
)


# ============================================================================
# 1. RESNET & CHASSIS BACKBONE TESTS
# ============================================================================

def test_resnet_block_2d():
    block = ResNetBlock2D(channels=64)
    x = torch.randn(2, 64, 10, 10)
    out = block(x)
    assert out.shape == (2, 64, 10, 10)
    assert torch.isfinite(out).all()


def test_spatial_representation_network():
    rep_net = SpatialRepresentationNetwork(in_channels=28, latent_channels=64, num_blocks=3)
    obs = torch.randn(2, 28, 10, 10)
    s_0 = rep_net(obs)
    assert s_0.shape == (2, 64, 10, 10)
    # Norm along channel dimension must be 1.0 due to F.normalize(p=2, dim=1)
    norms = torch.norm(s_0, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_spatial_dynamics_network():
    dyn_net = SpatialDynamicsNetwork(latent_channels=64, action_channels=16, num_blocks=3, support_size=601)
    s_prev = torch.randn(2, 64, 10, 10)
    action_plane = torch.zeros(2, 16, 10, 10)
    action_plane[:, 0, :, :] = 1.0

    s_next, reward_logits = dyn_net(s_prev, action_plane)
    assert s_next.shape == (2, 64, 10, 10)
    assert reward_logits.shape == (2, 601)
    norms = torch.norm(s_next, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_spatial_prediction_network_and_action_sampling():
    pred_net = SpatialPredictionNetwork(
        latent_channels=64,
        num_farmer_actions=15,
        num_hand_assignments=32,
        num_market_orders=20,
        support_size=601,
    )
    s_k = torch.randn(2, 64, 10, 10)
    preds = pred_net(s_k)

    assert preds["logits_farmer"].shape == (2, 15)
    assert preds["logits_hands"].shape == (2, 32)
    assert preds["logits_market"].shape == (2, 20)
    assert preds["value_logits"].shape == (2, 601)

    # Test joint action sampling without mask
    joint_actions, log_probs = pred_net.sample_joint_actions(s_k, num_samples=8)
    assert joint_actions.shape == (2, 8, 3)
    assert log_probs.shape == (2, 8)

    # Test action masking: mask out farmer action 0 and verify it is never sampled
    mask_farmer = torch.ones(2, 15)
    mask_farmer[:, 0] = 0.0
    masks = {"farmer": mask_farmer}
    joint_actions_masked, _ = pred_net.sample_joint_actions(s_k, num_samples=32, masks=masks)
    # Action 0 must never be sampled
    assert (joint_actions_masked[:, :, 0] != 0).all()


def test_kaggriculture_muzero_chassis_inference():
    chassis = KaggricultureMuZeroChassis(
        obs_channels=28,
        latent_channels=64,
        action_channels=16,
        support_size=601,
    )
    obs = torch.randn(2, 28, 10, 10)
    s_0, init_preds, value_scalar = chassis.initial_inference(obs)
    assert s_0.shape == (2, 64, 10, 10)
    assert value_scalar.shape == (2,)
    assert (-300.0 <= value_scalar).all() and (value_scalar <= 300.0).all()

    action_plane = torch.zeros(2, 16, 10, 10)
    action_plane[:, 2, :, :] = 1.0
    s_1, reward_scalar, k1_preds, value_1 = chassis.recurrent_inference(s_0, action_plane)
    assert s_1.shape == (2, 64, 10, 10)
    assert reward_scalar.shape == (2,)
    assert value_1.shape == (2,)


# ============================================================================
# 2. SIMSIAM CONSISTENCY TESTS
# ============================================================================

def test_simsiam_head_and_loss():
    head = SimSiamProjectionPredictionHead(latent_dim=64, proj_dim=64, pred_dim=32)
    loss_fn = SimSiamConsistencyLoss()

    s_unrolled = torch.randn(4, 64, 10, 10, requires_grad=True)
    target_embedding = torch.randn(4, 64, 10, 10)

    loss = loss_fn(s_unrolled, target_embedding, head)
    assert torch.isfinite(loss)
    # Cosine similarity is bounded in [-1.0, 1.0]
    assert -1.05 <= loss.item() <= 1.05

    # Backward pass verification
    loss.backward()
    assert s_unrolled.grad is not None
    assert torch.isfinite(s_unrolled.grad).all()


def test_compute_muzero_unroll_loss_with_consistency():
    rep = SpatialRepresentationNetwork(in_channels=28, latent_channels=64, num_blocks=1)
    dyn = SpatialDynamicsNetwork(latent_channels=64, action_channels=16, num_blocks=1, support_size=601)
    pred = SpatialPredictionNetwork(latent_channels=64, num_farmer_actions=15, num_hand_assignments=32, num_market_orders=20, support_size=601)
    simsiam_head = SimSiamProjectionPredictionHead(latent_dim=64, proj_dim=64, pred_dim=32)
    simsiam_loss = SimSiamConsistencyLoss()

    batch_size = 2
    unroll_steps = 3
    dummy_obs = torch.randn(batch_size, unroll_steps + 1, 28, 10, 10)
    dummy_actions = torch.zeros(batch_size, unroll_steps, 16, 10, 10)
    dummy_target_policies = F.softmax(torch.randn(batch_size, unroll_steps + 1, 15), dim=-1)
    dummy_target_values = torch.zeros(batch_size, unroll_steps + 1)
    dummy_target_rewards = torch.zeros(batch_size, unroll_steps)

    total_loss, metrics = compute_muzero_unroll_loss_with_consistency(
        rep, dyn, pred, simsiam_head, simsiam_loss,
        dummy_obs, dummy_actions, dummy_target_policies, dummy_target_values, dummy_target_rewards,
        consistency_weight=0.25, unroll_steps=unroll_steps,
    )
    assert torch.isfinite(total_loss)
    assert "loss_total" in metrics
    assert "loss_consistency" in metrics
    total_loss.backward()


# ============================================================================
# 3. SAMPLED MCTS SEARCH ENGINE TESTS
# ============================================================================

def test_min_max_stats():
    stats = MinMaxStats()
    assert stats.normalize(5.0) == 5.0
    stats.update(10.0)
    stats.update(20.0)
    assert abs(stats.normalize(10.0) - 0.0) < 1e-4
    assert abs(stats.normalize(20.0) - 1.0) < 1e-4
    assert abs(stats.normalize(15.0) - 0.5) < 1e-4


def test_sampled_mcts_search():
    chassis = KaggricultureMuZeroChassis()
    mcts = SampledMuZeroMCTS(chassis, num_samples=4, num_simulations=10)
    obs = torch.randn(1, 28, 10, 10)

    best_action, action_probs, root_val = mcts.search(obs, add_dirichlet_noise=True)
    assert isinstance(best_action, tuple)
    assert len(best_action) == 3
    assert len(action_probs) <= 4
    prob_sum = sum(action_probs.values())
    assert abs(prob_sum - 1.0) < 1e-5
    assert isinstance(root_val, float)
    assert math.isfinite(root_val)


# ============================================================================
# 4. BUFFER & PRIORITIZED EXPERIENCE REPLAY TESTS
# ============================================================================

def test_game_trajectory_and_uniform_buffer():
    traj = GameTrajectory()
    for t in range(10):
        obs = np.random.randn(28, 10, 10).astype(np.float32)
        act = (t % 15, t % 32, t % 20)
        rew = 1.0
        p_dict = {act: 1.0}
        val = 10.0
        traj.append_step(obs, act, rew, p_dict, val, priority=1.0)
    traj.finalize(np.random.randn(28, 10, 10).astype(np.float32))

    assert len(traj) == 10
    assert len(traj.observations) == 11

    buffer = MuZeroTrajectoryBuffer(max_trajectories=5, k_steps=3)
    buffer.save_trajectory(traj)
    assert len(buffer.trajectories) == 1

    batch = buffer.sample_k_step_batch(batch_size=2, k_steps=3)
    assert batch["obs"].shape == (2, 4, 28, 10, 10)
    assert batch["actions"].shape == (2, 3, 3)
    assert batch["target_policy_farmer"].shape == (2, 4, 15)
    assert batch["target_policy_hands"].shape == (2, 4, 32)
    assert batch["target_policy_market"].shape == (2, 4, 20)


def test_prioritized_muzero_buffer_and_is_weights():
    buffer = PrioritizedMuZeroBuffer(max_trajectories=10, alpha=0.6, beta_start=0.4, beta_frames=1000)

    for ep in range(3):
        traj = GameTrajectory()
        for t in range(8):
            obs = np.random.randn(28, 10, 10).astype(np.float32)
            act = (t % 15, t % 32, t % 20)
            rew = float(t)
            p_dict = {act: 0.8, ((act[0] + 1) % 15, act[1], act[2]): 0.2}
            val = float(t * 10)
            priority = 2.0 + t
            traj.append_step(obs, act, rew, p_dict, val, priority=priority)
        traj.finalize(np.random.randn(28, 10, 10).astype(np.float32))
        buffer.save_trajectory(traj)

    beta = buffer.update_beta(500)
    assert 0.4 < beta < 1.0

    batch, slice_keys, is_weights = buffer.sample_prioritized_k_step_batch(batch_size=2, k_steps=3)
    assert len(slice_keys) == 2
    assert is_weights.shape == (2, 1)
    assert (0.0 < is_weights).all() and (is_weights <= 1.0).all()

    # Priority update
    test_slice_keys = [(0, 0), (1, 1)]
    fresh_errors = np.array([0.5, 2.5], dtype=np.float32)
    buffer.update_slice_priorities(test_slice_keys, fresh_errors)
    assert abs(buffer.trajectories[0].priorities[0] - (0.5 + buffer.eps)) < 1e-4
    assert abs(buffer.trajectories[1].priorities[1] - (2.5 + buffer.eps)) < 1e-4


# ============================================================================
# 5. TRAINER ENGINE TESTS
# ============================================================================

def test_prioritized_muzero_trainer():
    chassis = KaggricultureMuZeroChassis()
    simsiam_head = SimSiamProjectionPredictionHead(latent_dim=64, proj_dim=64, pred_dim=32)
    buffer = PrioritizedMuZeroBuffer(max_trajectories=5)
    mcts_engine = SampledMuZeroMCTS(chassis, num_samples=4, num_simulations=5)

    # Populate mock trajectory
    traj = GameTrajectory()
    for t in range(8):
        obs = np.random.randn(28, 10, 10).astype(np.float32)
        act = (t % 15, t % 32, t % 20)
        rew = 5.0
        p_dict = {act: 1.0}
        val = 50.0
        traj.append_step(obs, act, rew, p_dict, val, priority=1.0)
    traj.finalize(np.random.randn(28, 10, 10).astype(np.float32))
    buffer.save_trajectory(traj)

    trainer = PrioritizedMuZeroTrainer(
        online_model=chassis,
        simsiam_head=simsiam_head,
        buffer=buffer,
        mcts_engine=mcts_engine,
        lr=1e-3,
        device="cpu",
    )

    metrics = trainer.train_step(batch_size=2, k_steps=3, global_step=100)
    assert "loss_total" in metrics
    assert "loss_policy" in metrics
    assert "loss_value" in metrics
    assert "loss_consistency" in metrics
    assert math.isfinite(metrics["loss_total"])

    # Test reanalyze pass
    trainer.run_reanalyze_pass(num_trajectories=1)


def test_muzero_trainer_uniform():
    chassis = KaggricultureMuZeroChassis()
    simsiam_head = SimSiamProjectionPredictionHead(latent_dim=64, proj_dim=64, pred_dim=32)
    buffer = MuZeroTrajectoryBuffer(max_trajectories=5)
    mcts_engine = SampledMuZeroMCTS(chassis, num_samples=4, num_simulations=5)

    traj = GameTrajectory()
    for t in range(8):
        obs = np.random.randn(28, 10, 10).astype(np.float32)
        act = (t % 15, t % 32, t % 20)
        rew = 2.0
        p_dict = {act: 1.0}
        val = 20.0
        traj.append_step(obs, act, rew, p_dict, val)
    traj.finalize(np.random.randn(28, 10, 10).astype(np.float32))
    buffer.save_trajectory(traj)

    trainer = MuZeroTrainer(
        online_model=chassis,
        simsiam_head=simsiam_head,
        buffer=buffer,
        mcts_engine=mcts_engine,
        lr=1e-3,
        device="cpu",
    )
    metrics = trainer.train_step(batch_size=2, k_steps=3)
    assert "loss_total" in metrics
    assert math.isfinite(metrics["loss_total"])


# ============================================================================
# 6. OBSERVATION ENCODER & ACTION MASKING TESTS
# ============================================================================

def test_kaggriculture_observation_encoder():
    encoder = KaggricultureObservationEncoder()
    mock_obs = {
        "player": 0,
        "step": 48,
        "day": 2,
        "hour": 0,
        "farms": [
            {
                "unlocked_quadrants": ["NW", "NE"],
                "money": 1500.0,
                "tiles": [
                    [{"crop": "WHEAT", "watered_today": True, "yield_available": 1.0, "age": 2.0}]
                    + [None] * 9
                ]
                + [[None] * 10 for _ in range(9)],
                "farmer": {"x": 2, "y": 2},
                "hands": 4,
                "animals": {"COW": 2},
            },
            {},
        ],
        "market_prices": {"WHEAT": 12.0, "CARROT": 14.0, "TOMATO": 16.0},
    }

    tensor = encoder.encode(mock_obs)
    assert tensor.shape == (1, 28, 10, 10)
    # Check NW quadrant unlocked (channel 0)
    assert tensor[0, 0, 0, 0].item() == 1.0
    # Check NE quadrant unlocked
    assert tensor[0, 0, 0, 7].item() == 1.0
    # Check SW quadrant locked
    assert tensor[0, 0, 7, 0].item() == 0.0

    # Check tile entity WHEAT at (0, 0)
    assert tensor[0, 3, 0, 0].item() == 1.0

    # Check action masks
    masks = encoder.get_action_mask(mock_obs)
    assert "farmer" in masks and masks["farmer"].shape == (15,)
    assert "hands" in masks and masks["hands"].shape == (32,)
    assert "market" in masks and masks["market"].shape == (20,)

    # NE is already unlocked, so EXPAND_NE (order index 3) must be 0.0
    assert masks["market"][3].item() == 0.0
    # SW requires $2000, money is $1500, so EXPAND_SW (order index 4) must be 0.0
    assert masks["market"][4].item() == 0.0


# ============================================================================
# 7. LEAGUE TRACKER & PFSP SAMPLER TESTS
# ============================================================================

def test_league_tracker_and_pfsp():
    league = LeagueTracker(smoothing=1.0)
    league.add_checkpoint("model_v1")
    league.add_checkpoint("model_v2")
    league.add_checkpoint("model_v3")

    # model_v1 beats model_v2, loses to model_v3
    league.record_match_outcome("model_v1", "model_v2", outcome=1.0)
    league.record_match_outcome("model_v1", "model_v3", outcome=0.0)

    # With Laplace smoothing:
    # vs model_v2: (1 + 1) / (1 + 2) = 2/3 ≈ 0.667
    # vs model_v3: (0 + 1) / (1 + 2) = 1/3 ≈ 0.333
    wr_v2 = league.get_win_rate("model_v1", "model_v2")
    wr_v3 = league.get_win_rate("model_v1", "model_v3")
    assert abs(wr_v2 - 2.0 / 3.0) < 1e-4
    assert abs(wr_v3 - 1.0 / 3.0) < 1e-4

    # PFSP sampler
    pfsp = PFSPSampler(league)
    opp_hard = pfsp.sample_opponent("model_v1", mode="hard", power=2.0)
    assert opp_hard in ("model_v2", "model_v3")

    opp_var = pfsp.sample_opponent("model_v1", mode="var")
    assert opp_var in ("model_v2", "model_v3")

    opp_uniform = pfsp.sample_opponent("model_v1", mode="uniform")
    assert opp_uniform in ("model_v2", "model_v3")

    # Matchmaking role selection
    opp, mode = select_matchmaking_opponent("main", "model_v1", pfsp)
    assert opp in ("model_v1", "model_v2", "model_v3")
    assert mode in ("self_play", "pfsp_hard", "pfsp_var")

    opp, mode = select_matchmaking_opponent("league_exploiter", "model_v1", pfsp)
    assert opp in ("model_v2", "model_v3")
    assert mode == "pfsp_hard"
