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

"""Spatial Sampled MuZero curriculum: BC → bootstrap → MCTS distill → reanalyze."""
from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from muzero import (
    GameTrajectory,
    KaggricultureMuZeroChassis,
    KaggricultureObservationEncoder,
    PrioritizedMuZeroBuffer,
    PrioritizedMuZeroTrainer,
    SampledMuZeroMCTS,
    SimSiamProjectionPredictionHead,
    encode_spatial_observation,
)
from muzero.action_translate import label_expert_joint_action
from muzero.spatial_checkpoint import (
    load_spatial_checkpoint,
    promote_spatial_champion,
    save_spatial_checkpoint,
)
from muzero.spatial_constants import (
    DEFAULT_SPATIAL_CHECKPOINT,
    LATENT_CHANNELS,
    NUM_FARMER_ACTIONS,
    NUM_HAND_ASSIGNMENTS,
    NUM_MARKET_ORDERS,
)

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS = os.path.join(HERE, "artifacts")
REPLAYS_DIR = os.path.join(HERE, "replays")
DEFAULT_SPATIAL_OUT = os.path.join(ARTIFACTS, DEFAULT_SPATIAL_CHECKPOINT)


def _fixture_obs(day: int = 5, money: float = 2000.0, unlocked: Optional[List[str]] = None) -> Dict[str, Any]:
    unlocked = unlocked or ["NW"]
    return {
        "player": 0,
        "day": day,
        "hour": 8,
        "step": day * 24 + 8,
        "farms": [
            {
                "money": money,
                "unlocked_quadrants": list(unlocked),
                "tiles": [[{} for _ in range(10)] for _ in range(10)],
                "animals": {},
                "farmer": {"x": 2, "y": 2},
                "hands": [],
            },
            {},
        ],
        "private": {"shed": {"WHEAT": 5.0, "MILK": 2.0}},
        "market": {"prices": {"WHEAT": 35.0, "MELON": 280.0, "MILK": 256.0}},
    }


def make_spatial_buffer(
    max_trajectories: int = 1000,
    alpha: float = 0.6,
    beta: float = 0.4,
    k_steps: int = 5,
) -> PrioritizedMuZeroBuffer:
    return PrioritizedMuZeroBuffer(
        max_trajectories=max_trajectories,
        alpha=alpha,
        beta_start=beta,
        k_steps=k_steps,
    )


def make_spatial_trainer(
    chassis: KaggricultureMuZeroChassis,
    buffer: PrioritizedMuZeroBuffer,
    *,
    lr: float = 1e-3,
    consistency_weight: float = 0.25,
    policy_weight: float = 1.0,
    value_weight: float = 0.25,
    reward_weight: float = 1.0,
    cql_weight: float = 0.0,
    sims: int = 16,
    num_samples: int = 8,
    ema_tau: float = 0.995,
    device: str = "cpu",
) -> PrioritizedMuZeroTrainer:
    mcts = SampledMuZeroMCTS(chassis, num_samples=num_samples, num_simulations=sims)
    simsiam = SimSiamProjectionPredictionHead(latent_dim=LATENT_CHANNELS)
    return PrioritizedMuZeroTrainer(
        online_model=chassis,
        simsiam_head=simsiam,
        buffer=buffer,
        mcts_engine=mcts,
        lr=lr,
        ema_tau=ema_tau,
        consistency_weight=consistency_weight,
        policy_weight=policy_weight,
        value_weight=value_weight,
        reward_weight=reward_weight,
        cql_weight=cql_weight,
        device=device,
    )


def train_on_buffer(
    trainer: PrioritizedMuZeroTrainer,
    steps: int,
    batch: int,
    k_steps: int = 3,
    label: str = "train",
    reanalyze_every: int = 0,
    reanalyze_trajs: int = 1,
) -> Dict[str, float]:
    """Run Adam train steps on the trainer's shared PER buffer."""
    if steps <= 0 or len(trainer.buffer.trajectories) == 0:
        return {}
    last: Dict[str, float] = {}
    t0 = time.time()
    for i in range(steps):
        last = trainer.train_step(batch_size=max(1, batch), k_steps=k_steps, global_step=i)
        if reanalyze_every > 0 and (i + 1) % reanalyze_every == 0:
            trainer.run_reanalyze_pass(num_trajectories=reanalyze_trajs)
        if (i + 1) % max(1, steps // 5) == 0 or i == steps - 1:
            print(
                f"  [{label}] step {i+1}/{steps} loss={last.get('loss_total', 0):.4f} "
                f"policy={last.get('loss_policy', 0):.4f} value={last.get('loss_value', 0):.4f} "
                f"cons={last.get('loss_consistency', 0):.4f} ({time.time()-t0:.1f}s)"
            )
    return last


def populate_synthetic_spatial_buffer(
    buffer: PrioritizedMuZeroBuffer,
    n_episodes: int = 4,
    steps_per_ep: int = 12,
    mcts: Optional[SampledMuZeroMCTS] = None,
) -> int:
    """Fill PER buffer with synthetic spatial trajectories (optional MCTS priors)."""
    encoder = KaggricultureObservationEncoder()
    for _ in range(n_episodes):
        traj = GameTrajectory()
        day = random.randint(0, 20)
        money = random.uniform(500.0, 12000.0)
        unlocked = ["NW"]
        if day >= 8:
            unlocked.append("NE")
        if day >= 14:
            unlocked.append("SW")
        for t in range(steps_per_ep):
            obs = _fixture_obs(day=min(30, day + t // 3), money=money, unlocked=unlocked)
            obs_t = encoder.encode(obs).squeeze(0).numpy().astype(np.float32)
            if mcts is not None:
                best, policy_dict, value = mcts.search(
                    torch.from_numpy(obs_t).unsqueeze(0),
                    add_dirichlet_noise=True,
                )
                action = best
            else:
                action = (
                    random.randint(0, NUM_FARMER_ACTIONS - 1),
                    random.randint(0, NUM_HAND_ASSIGNMENTS - 1),
                    random.randint(0, NUM_MARKET_ORDERS - 1),
                )
                alt = ((action[0] + 1) % NUM_FARMER_ACTIONS, action[1], action[2])
                policy_dict = {action: 0.7, alt: 0.3}
                value = random.uniform(-50.0, 150.0)
            reward = float(random.choice([0.0, 5.0, 10.0, 25.0]))
            money += reward * 10.0
            traj.append_step(obs_t, action, reward, policy_dict, float(value), priority=1.0)
        final = encoder.encode(_fixture_obs(day=min(30, day + 4), money=money, unlocked=unlocked))
        traj.finalize(final.squeeze(0).numpy().astype(np.float32))
        buffer.save_trajectory(traj)
    return n_episodes


def ingest_expert_spatial_trajectories(
    buffer: PrioritizedMuZeroBuffer,
    *,
    min_score: float = 80000.0,
    max_replays: int = 50,
    max_steps_per_ep: int = 240,
    stride: int = 3,
) -> int:
    """Load gold expert replays into the spatial PER buffer (BC targets)."""
    from .phase3_bc import discover_expert_replays

    experts = discover_expert_replays(min_score=min_score, limit=max_replays)
    if not experts:
        print("[spatial-bc] no expert replays found; skipping BC ingest")
        return 0

    encoder = KaggricultureObservationEncoder()
    n_traj = 0
    for path_or_name, seat in experts:
        path = path_or_name
        if not os.path.isfile(path):
            path = os.path.join(REPLAYS_DIR, os.path.basename(path_or_name))
        if not os.path.isfile(path):
            print(f"[spatial-bc] skip missing {path_or_name}")
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                episode = json.load(f)
        except Exception as exc:
            print(f"[spatial-bc] skip {path}: {exc}")
            continue

        steps = episode.get("steps") or []
        rewards = episode.get("rewards") or [0, 0]
        try:
            seat_reward = float(rewards[seat]) if seat < len(rewards) else 0.0
        except Exception:
            seat_reward = 0.0

        traj = GameTrajectory()
        prev_money: Optional[float] = None
        counted = 0
        for si in range(0, len(steps), max(1, stride)):
            if counted >= max_steps_per_ep:
                break
            step = steps[si]
            if not isinstance(step, list) or seat >= len(step):
                continue
            entry = step[seat]
            if not isinstance(entry, dict):
                continue
            obs = entry.get("observation")
            act = entry.get("action") or {}
            if not isinstance(obs, dict):
                continue
            try:
                obs_t = encoder.encode(obs).squeeze(0).numpy().astype(np.float32)
            except Exception:
                continue
            joint = label_expert_joint_action(act if isinstance(act, dict) else {})
            farms = obs.get("farms") or [{}, {}]
            farm = farms[seat] if seat < len(farms) else {}
            money = float(farm.get("money", 0.0) or 0.0)
            if prev_money is None:
                reward = 0.0
            else:
                reward = (money - prev_money) / 100.0
            prev_money = money
            # One-hot-ish soft prior on the expert joint (+ light alternate for soft CE)
            alt = ((joint[0] + 1) % NUM_FARMER_ACTIONS, joint[1], joint[2])
            policy = {joint: 0.85, alt: 0.15}
            value = seat_reward / 1000.0
            traj.append_step(obs_t, joint, float(reward), policy, float(value), priority=2.0)
            counted += 1

        if len(traj) < 2:
            continue
        traj.finalize(traj.observations[-1].copy())
        buffer.save_trajectory(traj)
        n_traj += 1

    print(f"[spatial-bc] ingested {n_traj} expert trajectories into PER buffer")
    return n_traj


def persist_selfplay_json(
    trajectories: List[Dict[str, Any]],
    *,
    replays_dir: str = REPLAYS_DIR,
    tag: str = "spatial",
) -> str:
    """Write harvested self-play episodes to replays/selfplay_*.json (mtime-prunable)."""
    os.makedirs(replays_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(replays_dir, f"selfplay_{tag}_{stamp}_{os.getpid()}.json")
    payload = {
        "schema": "spatial_selfplay_v1",
        "created_utc": stamp,
        "episodes": trajectories,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


def harvest_spatial_selfplay(
    chassis: KaggricultureMuZeroChassis,
    buffer: PrioritizedMuZeroBuffer,
    n_episodes: int = 4,
    steps_per_ep: int = 48,
    sims: int = 16,
    num_samples: int = 8,
    *,
    persist: bool = True,
    replays_dir: str = REPLAYS_DIR,
) -> int:
    """Self-play harvest using SampledMuZeroMCTS into the *same* PER buffer used for training."""
    mcts = SampledMuZeroMCTS(chassis, num_samples=num_samples, num_simulations=sims)
    n = populate_synthetic_spatial_buffer(
        buffer, n_episodes=n_episodes, steps_per_ep=steps_per_ep, mcts=mcts
    )
    if persist and n > 0:
        # Compact summary for disk (obs shapes only — full tensors live in PER).
        summary = [
            {
                "steps": steps_per_ep,
                "sims": sims,
                "source": "sampled_mcts_synthetic",
            }
            for _ in range(n)
        ]
        path = persist_selfplay_json(summary, replays_dir=replays_dir)
        print(f"[spatial-selfplay] persisted harvest stub → {path}")
    return n


def phase_spatial_bc(
    chassis: KaggricultureMuZeroChassis,
    buffer: PrioritizedMuZeroBuffer,
    *,
    steps: int,
    batch: int,
    lr: float,
    k_steps: int = 3,
    consistency_weight: float = 0.25,
    sims: int = 16,
    num_samples: int = 8,
    ema_tau: float = 0.995,
    min_score: float = 80000.0,
    max_expert_replays: int = 50,
    device: str = "cpu",
) -> Dict[str, float]:
    """Phase A: expert BC into shared PER buffer, then Adam train."""
    print(f"[Phase Spatial BC] steps={steps} batch={batch} k={k_steps}")
    n = ingest_expert_spatial_trajectories(
        buffer, min_score=min_score, max_replays=max_expert_replays
    )
    if n == 0:
        populate_synthetic_spatial_buffer(buffer, n_episodes=max(4, batch), steps_per_ep=max(k_steps + 2, 16))
    trainer = make_spatial_trainer(
        chassis,
        buffer,
        lr=lr,
        consistency_weight=consistency_weight,
        sims=sims,
        num_samples=num_samples,
        ema_tau=ema_tau,
        device=device,
    )
    return train_on_buffer(trainer, steps=steps, batch=batch, k_steps=k_steps, label="bc")


def phase_spatial_bootstrap(
    chassis: KaggricultureMuZeroChassis,
    steps: int,
    batch: int,
    lr: float,
    k_steps: int = 3,
    consistency_weight: float = 0.25,
    device: str = "cpu",
    *,
    buffer: Optional[PrioritizedMuZeroBuffer] = None,
    sims: int = 16,
    num_samples: int = 8,
    ema_tau: float = 0.995,
    seed_synthetic: bool = True,
) -> Dict[str, float]:
    """Phase B: dynamics/SimSiam bootstrap on the shared buffer (synthetic seed if empty)."""
    print(f"[Phase Spatial Bootstrap] steps={steps} batch={batch} k={k_steps}")
    if buffer is None:
        buffer = make_spatial_buffer(max_trajectories=max(64, batch * 4), k_steps=k_steps)
        seed_synthetic = True
    if seed_synthetic and len(buffer.trajectories) == 0:
        populate_synthetic_spatial_buffer(
            buffer, n_episodes=max(4, batch // 2), steps_per_ep=max(k_steps + 2, 16)
        )
    trainer = make_spatial_trainer(
        chassis,
        buffer,
        lr=lr,
        consistency_weight=consistency_weight,
        sims=sims,
        num_samples=num_samples,
        ema_tau=ema_tau,
        device=device,
    )
    last = train_on_buffer(trainer, steps=max(0, steps), batch=batch, k_steps=k_steps, label="bootstrap")
    if steps > 0:
        trainer.run_reanalyze_pass(num_trajectories=min(2, max(1, len(buffer.trajectories))))
    return last


def phase_spatial_mcts_distill(
    chassis: KaggricultureMuZeroChassis,
    buffer: PrioritizedMuZeroBuffer,
    *,
    steps: int,
    batch: int,
    lr: float,
    sims: int,
    k_steps: int = 3,
    consistency_weight: float = 0.25,
    num_samples: int = 8,
    ema_tau: float = 0.995,
    selfplay_episodes: int = 4,
    steps_per_ep: int = 48,
    device: str = "cpu",
) -> Dict[str, float]:
    """Phase C: harvest Sampled-MCTS self-play into shared buffer, then distill."""
    print(
        f"[Phase Spatial MCTS Distill] steps={steps} batch={batch} sims={sims} "
        f"selfplay={selfplay_episodes}"
    )
    if selfplay_episodes > 0:
        harvest_spatial_selfplay(
            chassis,
            buffer,
            n_episodes=selfplay_episodes,
            steps_per_ep=steps_per_ep,
            sims=sims,
            num_samples=num_samples,
            persist=True,
        )
    trainer = make_spatial_trainer(
        chassis,
        buffer,
        lr=lr,
        consistency_weight=consistency_weight,
        sims=sims,
        num_samples=num_samples,
        ema_tau=ema_tau,
        device=device,
    )
    return train_on_buffer(
        trainer,
        steps=max(0, steps),
        batch=batch,
        k_steps=k_steps,
        label="mcts-distill",
        reanalyze_every=max(5, steps // 4) if steps > 0 else 0,
        reanalyze_trajs=2,
    )


def phase_spatial_reanalyze(
    chassis: KaggricultureMuZeroChassis,
    buffer: PrioritizedMuZeroBuffer,
    *,
    steps: int,
    batch: int,
    lr: float,
    sims: int,
    k_steps: int = 3,
    consistency_weight: float = 0.25,
    num_samples: int = 8,
    ema_tau: float = 0.995,
    device: str = "cpu",
) -> Dict[str, float]:
    """Phase D: EMA target reanalyze + continued Adam on refreshed MCTS targets."""
    print(f"[Phase Spatial Reanalyze] steps={steps} batch={batch} sims={sims}")
    if len(buffer.trajectories) == 0:
        populate_synthetic_spatial_buffer(buffer, n_episodes=max(4, batch), steps_per_ep=max(k_steps + 2, 16))
    trainer = make_spatial_trainer(
        chassis,
        buffer,
        lr=lr,
        consistency_weight=consistency_weight,
        sims=sims,
        num_samples=num_samples,
        ema_tau=ema_tau,
        device=device,
    )
    # Refresh stale targets first
    n_refresh = min(8, len(buffer.trajectories))
    if n_refresh > 0:
        trainer.run_reanalyze_pass(num_trajectories=n_refresh)
    return train_on_buffer(
        trainer,
        steps=max(0, steps),
        batch=batch,
        k_steps=k_steps,
        label="reanalyze",
        reanalyze_every=max(3, steps // 5) if steps > 0 else 0,
        reanalyze_trajs=2,
    )


# Re-export checkpoint helpers from muzero package (single source of truth)
__all__ = [
    "make_spatial_buffer",
    "make_spatial_trainer",
    "train_on_buffer",
    "populate_synthetic_spatial_buffer",
    "ingest_expert_spatial_trajectories",
    "persist_selfplay_json",
    "harvest_spatial_selfplay",
    "phase_spatial_bc",
    "phase_spatial_bootstrap",
    "phase_spatial_mcts_distill",
    "phase_spatial_reanalyze",
    "save_spatial_checkpoint",
    "load_spatial_checkpoint",
    "promote_spatial_champion",
]
