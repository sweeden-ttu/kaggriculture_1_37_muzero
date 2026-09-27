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

"""Spatial Sampled MuZero production contracts + legacy macro smoke."""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from muzero import (
    ACTION_CHANNELS,
    GameTrajectory,
    KaggricultureMuZeroChassis,
    PrioritizedMuZeroBuffer,
    PrioritizedMuZeroTrainer,
    SampledMuZeroMCTS,
    SimSiamConsistencyLoss,
    SimSiamProjectionPredictionHead,
    encode_spatial_observation,
    joint_action_to_plane,
)
from muzero.legacy_macro import MuZeroNetwork, LatentMacroOptionMCTS
from muzero.types import MacroState
from hybrid_chassis.pipeline import translate_joint_action


def _fixture_obs():
    return {
        "player": 0,
        "day": 5,
        "hour": 8,
        "farms": [
            {
                "money": 2500.0,
                "unlocked_quadrants": ["NW"],
                "tiles": [[{} for _ in range(10)] for _ in range(10)],
                "farmer": {"x": 2, "y": 2},
                "hands": [],
                "animals": {},
            },
            {},
        ],
        "private": {"shed": {"WHEAT": 3}},
        "market": {"prices": {"WHEAT": 35.0}},
    }


def test_encode_spatial_observation_shape():
    t = encode_spatial_observation(_fixture_obs())
    assert t.shape == (28, 10, 10)
    assert float(t[0, 0, 0]) == 1.0  # NW unlocked


def test_joint_action_plane_lossless():
    plane = joint_action_to_plane((14, 31, 19), batch_size=1)
    assert plane.shape == (1, ACTION_CHANNELS, 10, 10)
    assert float(plane[0, 14].mean()) == 1.0
    assert float(plane[0, 15 + 31].mean()) == 1.0
    assert float(plane[0, 15 + 32 + 19].mean()) == 1.0
    # No aliasing into low channels for high indices
    assert float(plane[0, 4].sum()) == 0.0


def test_chassis_initial_and_recurrent():
    chassis = KaggricultureMuZeroChassis()
    obs = encode_spatial_observation(_fixture_obs()).unsqueeze(0)
    s0, preds, v = chassis.initial_inference(obs)
    assert s0.shape == (1, 64, 10, 10)
    assert preds["logits_farmer"].shape[-1] == 15
    assert preds["logits_hands"].shape[-1] == 32
    assert preds["logits_market"].shape[-1] == 20
    plane = joint_action_to_plane((1, 2, 3), batch_size=1)
    s1, r, preds2, v2 = chassis.recurrent_inference(s0, plane)
    assert s1.shape == s0.shape
    assert r.shape[0] == 1


def test_sampled_mcts_search():
    chassis = KaggricultureMuZeroChassis()
    mcts = SampledMuZeroMCTS(chassis, num_samples=4, num_simulations=8)
    obs = encode_spatial_observation(_fixture_obs()).unsqueeze(0)
    best, probs, val = mcts.search(obs, add_dirichlet_noise=False)
    assert isinstance(best, tuple) and len(best) == 3
    assert len(probs) >= 1
    assert isinstance(val, float)


def test_sampled_mcts_is_pure_latent():
    """h_θ once at root; every expansion via g_θ only — no env / re-encode."""
    from muzero.mcts import PURE_LATENT_MCTS

    assert PURE_LATENT_MCTS is True
    chassis = KaggricultureMuZeroChassis()
    mcts = SampledMuZeroMCTS(chassis, num_samples=4, num_simulations=6, pure_latent=True)
    with pytest.raises(ValueError):
        SampledMuZeroMCTS(chassis, pure_latent=False)

    repr_calls = {"n": 0}
    dyn_calls = {"n": 0}
    _orig_repr = chassis.representation.forward
    _orig_dyn = chassis.dynamics.forward

    def _count_repr(x):
        repr_calls["n"] += 1
        return _orig_repr(x)

    def _count_dyn(s, a):
        dyn_calls["n"] += 1
        return _orig_dyn(s, a)

    chassis.representation.forward = _count_repr  # type: ignore[method-assign]
    chassis.dynamics.forward = _count_dyn  # type: ignore[method-assign]

    obs = encode_spatial_observation(_fixture_obs()).unsqueeze(0)
    best, probs, val = mcts.search(obs, add_dirichlet_noise=False)
    assert repr_calls["n"] == 1, f"expected 1 representation call, got {repr_calls['n']}"
    assert dyn_calls["n"] == 6, f"expected 6 dynamics calls (one per sim), got {dyn_calls['n']}"
    assert mcts.last_search_stats["representation_calls"] == 1
    assert mcts.last_search_stats["dynamics_calls"] == 6
    assert best in probs


def test_prioritized_trainer_step():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    chassis = KaggricultureMuZeroChassis()
    buffer = PrioritizedMuZeroBuffer(max_trajectories=8)
    for _ in range(2):
        traj = GameTrajectory()
        for _t in range(10):
            o = np.random.randn(28, 10, 10).astype(np.float32)
            a = (random.randint(0, 14), random.randint(0, 31), random.randint(0, 19))
            pol = {a: 0.8, ((a[0] + 1) % 15, a[1], a[2]): 0.2}
            traj.append_step(o, a, 1.0, pol, 10.0, priority=1.0)
        traj.finalize(np.random.randn(28, 10, 10).astype(np.float32))
        buffer.save_trajectory(traj)
    mcts = SampledMuZeroMCTS(chassis, num_samples=4, num_simulations=6)
    head = SimSiamProjectionPredictionHead(latent_dim=64)
    trainer = PrioritizedMuZeroTrainer(chassis, head, buffer, mcts, device="cpu")
    metrics = trainer.train_step(batch_size=2, k_steps=3, global_step=1)
    assert "loss_total" in metrics
    assert np.isfinite(metrics["loss_total"])
    trainer.run_reanalyze_pass(num_trajectories=1)


def test_translate_joint_action():
    act = translate_joint_action(0, 0, 3, _fixture_obs())
    assert "_muzero_joint" in act
    assert act["farmer"] == ["PASS"]
    # money 2500, NW only → EXPAND_NE legal
    assert any(o and o[0] == "BUY_LAND" and o[1] == "NE" for o in act["market"])


def test_label_expert_and_spatial_checkpoint_roundtrip(tmp_path):
    from muzero import label_expert_joint_action, save_spatial_checkpoint, load_spatial_checkpoint

    joint = label_expert_joint_action(
        {"farmer": ["PLANT", "MELON"], "hands": [], "market": [["BUY_LAND", "SE"]]}
    )
    assert joint[0] == 11
    assert joint[2] == 5
    chassis = KaggricultureMuZeroChassis()
    path = str(tmp_path / "spatial.pt")
    save_spatial_checkpoint(chassis, path, meta={"test": True})
    chassis2 = KaggricultureMuZeroChassis()
    load_spatial_checkpoint(chassis2, path, strict=False)


def test_legacy_macro_still_importable():
    net = MuZeroNetwork(hidden_dim=32, support_size=601)
    mcts = LatentMacroOptionMCTS(net, gumbel_threshold=8)
    state = MacroState(day=1, capital=1000.0, unlocked=("NW",), shed={})
    result = mcts.search(state, n_simulations=4)
    assert result.best_option is not None
