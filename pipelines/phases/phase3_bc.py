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

"""Phase 3: Expert behavior cloning with Prioritized Experience Replay (PER)."""
from __future__ import annotations

import glob
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from muzero import (
    DEFAULT_PRICES,
    LAND_BUFFERS,
    LAND_ORDER,
    LAND_PRICES,
    NUM_MACRO_OPTIONS,
    REWARD_SCALE,
    MacroOption,
    MacroState,
    MuZeroNetwork,
    PrioritizedReplayBuffer,
    create_target_network,
    encode_macro_state,
    muzero_k_step_unroll_loss,
    update_target_network,
    wealth,
)

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPLAYS_DIR = os.path.join(HERE, "replays")


def discover_expert_replays(
    replays_dir: str = REPLAYS_DIR,
    min_score: float = 80000.0,
    limit: Optional[int] = None,
) -> List[Tuple[str, int]]:
    """Scan replays/ via replay_manager ingest for high-score expert seats."""
    from ..replay_manager import discover_expert_replay_seats

    return discover_expert_replay_seats(replays_dir, min_score=min_score, limit=limit)


def label_expert_macro_option(
    actions: Dict[str, Any],
    day: int,
    money: float,
    unlocked: Sequence[str],
    prev_unlocked: Optional[Sequence[str]] = None,
) -> int:
    market = actions.get("market", []) or []
    for ma in market:
        if isinstance(ma, list) and len(ma) >= 1 and ma[0] == "BUY_LAND":
            target = ma[1] if len(ma) > 1 else None
            if target not in ("NE", "SW", "SE"):
                prev_set = set(prev_unlocked or unlocked)
                added = [q for q in unlocked if q not in prev_set]
                if added:
                    target = added[0]
                else:
                    for q in LAND_ORDER:
                        if q not in prev_set:
                            target = q
                            break
                    else:
                        target = "NE"
            return {
                "NE": int(MacroOption.EXPAND_NE),
                "SW": int(MacroOption.EXPAND_SW),
                "SE": int(MacroOption.EXPAND_SE),
            }.get(str(target), int(MacroOption.EXPANSION_PREPARATION))
    for ma in market:
        if isinstance(ma, list) and ma and ma[0] == "HIRE":
            return int(MacroOption.HIRE_HANDS)
    high_value = {"MELON", "TOMATO", "EGG", "EGGS", "MILK", "WOOL", "STRAWBERRY"}
    for ma in market:
        if (
            isinstance(ma, list)
            and len(ma) >= 2
            and ma[0] in ("SELL", "SELL_ITEM")
            and ma[1] in high_value
        ):
            return int(MacroOption.TRICKLE_LIQUIDATION)
    next_q = next((q for q in LAND_ORDER if q not in unlocked), None)
    if next_q and money >= (LAND_PRICES[next_q] + LAND_BUFFERS[next_q]) * 0.85:
        return int(MacroOption.EXPANSION_PREPARATION)
    if day <= 6:
        return int(MacroOption.FRONT_LOAD_SURVIVAL)
    return int(MacroOption.PASS)


def transitions_from_replay(
    path: str, seat: int, episode_data: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    if episode_data is not None:
        episode = episode_data
    else:
        with open(path, "r", encoding="utf-8") as f:
            episode = json.load(f)
    steps = episode.get("steps") or []
    out: List[Dict[str, Any]] = []
    prev_obs = None
    prev_worth = None
    prev_action = int(MacroOption.PASS)
    prev_unlocked: List[str] = ["NW"]

    for t, step in enumerate(steps):
        st = step[seat]
        obs = st["observation"]
        farms = obs.get("farms") or []
        farm = farms[seat] if seat < len(farms) else {}
        day = int(obs.get("day", t // 24))
        money = float(farm.get("money", 0.0))
        unlocked = list(farm.get("unlocked_quadrants") or ["NW"])
        shed = {k: float(v) for k, v in ((obs.get("private") or {}).get("shed") or {}).items()}
        prices = dict((obs.get("market") or {}).get("prices") or {})
        state = MacroState(
            day=day,
            capital=money,
            unlocked=tuple(unlocked),
            shed=shed,
            prices=dict(prices) if prices else dict(DEFAULT_PRICES),
        )
        obs_vec = encode_macro_state(state)
        worth = wealth(state)

        if t > 0 and prev_obs is not None and prev_worth is not None:
            out.append(
                {
                    "obs": prev_obs,
                    "action": prev_action,
                    "reward": float((worth - prev_worth) * REWARD_SCALE),
                    "next_obs": obs_vec.clone(),
                    "worth": float(prev_worth),
                    "value": float(prev_worth * REWARD_SCALE),
                }
            )

        action_dict = st.get("action") or {}
        if not isinstance(action_dict, dict):
            action_dict = {}
        if t < len(steps) - 1:
            prev_action = label_expert_macro_option(
                action_dict, day, money, unlocked, prev_unlocked
            )
        prev_unlocked = unlocked
        prev_obs = obs_vec.clone()
        prev_worth = worth

    return out


def trajectories_from_replay(
    replay_file: str,
    seat: int,
    episode_data: Optional[Dict[str, Any]] = None,
    k_steps: int = 5,
) -> List[Dict[str, torch.Tensor]]:
    txs = transitions_from_replay(replay_file, seat, episode_data=episode_data)
    if len(txs) <= k_steps:
        return []
    slices: List[Dict[str, torch.Tensor]] = []
    stride = 2 if len(txs) > 200 else 1
    for t in range(0, len(txs) - k_steps, stride):
        obs_seq = torch.stack([txs[t + i]["obs"] for i in range(k_steps + 1)])
        actions = torch.tensor([txs[t + i]["action"] for i in range(k_steps)], dtype=torch.long)
        rewards = torch.tensor([txs[t + i]["reward"] for i in range(k_steps)], dtype=torch.float32)
        values = torch.tensor([txs[t + i].get("value", 0.0) for i in range(k_steps + 1)], dtype=torch.float32)
        policy_tgts = []
        for i in range(k_steps + 1):
            act_idx = txs[min(t + i, len(txs) - 1)]["action"]
            p_vec = torch.zeros(NUM_MACRO_OPTIONS)
            p_vec[act_idx] = 1.0
            policy_tgts.append(p_vec)
        policies = torch.stack(policy_tgts)
        slices.append({
            "obs": obs_seq,
            "actions": actions,
            "rewards": rewards,
            "values": values,
            "policies": policies,
        })
    return slices


def build_expert_trajectory_pool(
    replays: Sequence[Tuple[str, int]],
    k_steps: int = 5,
    max_trajectories_per_replay: Optional[int] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    replays_dir: str = REPLAYS_DIR,
) -> PrioritizedReplayBuffer:
    buffer = PrioritizedReplayBuffer(capacity=100000, alpha=alpha, beta=beta)
    _cache: Dict[str, Any] = {}
    for name, seat in replays:
        path = name if os.path.isabs(name) else os.path.join(replays_dir, name)
        if not os.path.isfile(path):
            continue
        if path not in _cache:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    _cache[path] = json.load(f)
            except Exception:
                continue
        trajs = trajectories_from_replay(path, seat=seat, episode_data=_cache[path], k_steps=k_steps)
        if max_trajectories_per_replay and len(trajs) > max_trajectories_per_replay:
            trajs = trajs[:max_trajectories_per_replay]
        for tr in trajs:
            init_p = float(tr["rewards"].abs().sum().item()) + 1.0
            buffer.add(tr, priority=init_p)
    return buffer


def phase3_expert_behavior_cloning(
    model: MuZeroNetwork,
    epochs: int,
    batch: int,
    lr: float,
    cons_w: float = 0.4357,
    k_steps: int = 5,
    dynamics_grad_scale: float = 0.5,
    c_l2: float = 1e-4,
    replays: Optional[Sequence[Tuple[str, int]]] = None,
    prebuilt_pool: Optional[Any] = None,
    target_model: Optional[MuZeroNetwork] = None,
    tau: float = 0.005,
    policy_weight: float = 0.8637,
    value_weight: float = 0.9637,
    reward_weight: float = 0.7357,
    cql_weight: float = 0.02357,
) -> Dict[str, Any]:
    """Phase 3: Behavior cloning over expert trajectory slices with Prioritized Experience Replay."""
    print(f"[Phase 3] Expert BC K={k_steps} with data-driven PER")
    if target_model is None:
        target_model = create_target_network(model)

    buffer: PrioritizedReplayBuffer
    if isinstance(prebuilt_pool, PrioritizedReplayBuffer):
        buffer = prebuilt_pool
    elif isinstance(prebuilt_pool, list) and len(prebuilt_pool) > 0:
        buffer = PrioritizedReplayBuffer(capacity=max(50000, len(prebuilt_pool)))
        buffer.add_batch(prebuilt_pool)
    else:
        buffer = build_expert_trajectory_pool(replays or [], k_steps=k_steps)

    if len(buffer) == 0:
        print("  no expert trajectory data; skipping Phase 3")
        return {"epochs": 0, "pool_size": 0, "avg_loss": 0.0}

    opt = optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    avg = 0.0
    steps_per_epoch = max(1, len(buffer) // batch)

    for ep in range(epochs):
        model.train()
        target_model.eval()
        total_loss = 0.0
        n_batches = 0

        for _ in range(steps_per_epoch):
            chunk, indices, is_weights = buffer.sample(batch)
            if len(chunk) < 4:
                continue
            obs_b = torch.stack([c["obs"] for c in chunk])
            act_b = torch.stack([c["actions"] for c in chunk])
            rew_b = torch.stack([c["rewards"] for c in chunk])
            val_b = torch.stack([c["values"] for c in chunk])
            pol_b = torch.stack([c["policies"] for c in chunk])

            losses = muzero_k_step_unroll_loss(
                model=model,
                obs_seq=obs_b,
                actions=act_b,
                rewards=rew_b,
                value_targets=val_b,
                policy_targets=pol_b,
                k_steps=k_steps,
                dynamics_grad_scale=dynamics_grad_scale,
                c_l2=c_l2,
                cons_w=cons_w,
                weights=is_weights,
                target_model=target_model,
                policy_weight=policy_weight,
                value_weight=value_weight,
                reward_weight=reward_weight,
                cql_weight=cql_weight,
            )
            opt.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            update_target_network(model, target_model, tau=tau)
            if "td_errors" in losses:
                buffer.update_priorities(indices, losses["td_errors"])

            total_loss += float(losses["total"].detach())
            n_batches += 1

        avg = total_loss / max(1, n_batches)
        print(f"  epoch {ep+1}/{epochs} avg_loss={avg:.4f}")

    print(f"  done in {time.time()-t0:.1f}s pool={len(buffer)}")
    return {"epochs": epochs, "pool_size": len(buffer), "avg_loss": avg}
