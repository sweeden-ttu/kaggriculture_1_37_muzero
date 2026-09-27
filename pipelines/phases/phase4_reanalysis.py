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

"""Phase 4: Continuous re-analysis with LoRA-only adapter updates.

Harvests fresh self-play trajectories under the current policy, re-evaluates
historical + self-play slices via Gumbel MCTS, refreshes policy/value targets,
and trains *only* LoRA adapters on frozen dynamics / prediction bases. Adapter
checkpoints land in ``artifacts/lora_adapters/``; frozen base weights stay in
``artifacts/snapshots/``.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from muzero import (
    DEFAULT_PRICES,
    LAND_BUFFERS,
    LAND_ORDER,
    LAND_PRICES,
    REWARD_SCALE,
    MacroOption,
    MacroState,
    MuZeroNetwork,
    MuZeroPlanner,
    PrioritizedReplayBuffer,
    create_target_network,
    encode_macro_state,
    legal_macro_options,
    muzero_k_step_unroll_loss,
    reanalyze_trajectory_slice,
    save_muzero_checkpoint,
    update_target_network,
    wealth,
)
from muzero.lora import (
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_RANK,
    default_lora_adapters_dir,
    has_lora,
    inject_lora,
    lora_parameters,
    save_lora_adapters,
)
from muzero.meta import LORA_ALPHA, LORA_RANK

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS = os.path.join(HERE, "artifacts")
SNAPSHOTS_DIR = os.path.join(ARTIFACTS, "snapshots")
LORA_ADAPTERS_DIR = default_lora_adapters_dir(HERE)
REPLAYS_DIR = os.path.join(HERE, "replays")


def route_base_snapshot(
    model: MuZeroNetwork,
    tag: str = "base",
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist frozen base weights into artifacts/snapshots/ only."""
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    path = os.path.join(SNAPSHOTS_DIR, f"snapshot_{tag}.pt")
    save_muzero_checkpoint(model, path=path, meta=meta or {"artifact_kind": "base_snapshot"}, atomic=True)
    return path


def route_lora_adapter(
    model: MuZeroNetwork,
    tag: str = "phase4",
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist lightweight LoRA adapters into artifacts/lora_adapters/ only."""
    os.makedirs(LORA_ADAPTERS_DIR, exist_ok=True)
    path = os.path.join(LORA_ADAPTERS_DIR, f"adapter_{tag}.pt")
    save_lora_adapters(model, path=path, meta=meta or {"artifact_kind": "lora_adapter"}, atomic=True)
    return path


def harvest_selfplay_trajectories(
    model: MuZeroNetwork,
    n_episodes: int = 4,
    k_steps: int = 5,
    sims: int = 16,
    replays_dir: str = REPLAYS_DIR,
    save_json: bool = True,
) -> List[Dict[str, torch.Tensor]]:
    """Continuous self-play harvest: MCTS policy rollouts → PER trajectory slices."""
    if n_episodes <= 0:
        return []
    planner = MuZeroPlanner(
        model=model,
        n_simulations=sims,
        gumbel_threshold=max(8, min(32, sims)),
        use_analytical_dynamics=False,
    )
    trajs: List[Dict[str, torch.Tensor]] = []
    os.makedirs(replays_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for ep in range(n_episodes):
            day = random.randint(0, 20)
            unlocked_n = 1 if day < 8 else (2 if day < 18 else 3)
            unlocked = tuple(["NW", "NE", "SW", "SE"][:unlocked_n])
            capital = random.uniform(800.0, 15000.0)
            for q in LAND_ORDER:
                if q not in unlocked:
                    capital = max(300.0, LAND_PRICES[q] + LAND_BUFFERS[q] + random.uniform(0, 2000))
                    break
            state = MacroState(
                day=day,
                capital=capital,
                unlocked=unlocked,
                shed={
                    "WHEAT": random.uniform(0, 30),
                    "MELON": random.uniform(0, 8),
                    "MILK": random.uniform(0, 10),
                    "WOOL": random.uniform(0, 8),
                },
                prices=dict(DEFAULT_PRICES),
                active_animals=random.randint(1, 6),
            )
            obs_seq = [encode_macro_state(state)]
            acts: List[int] = []
            rews: List[float] = []
            vals: List[float] = [wealth(state) * REWARD_SCALE]
            pols: List[torch.Tensor] = []
            curr = state
            episode_log: List[Dict[str, Any]] = []
            for _ in range(k_steps):
                result = planner.mcts.search(curr, n_simulations=sims)
                opt = result.best_option
                legal = legal_macro_options(curr)
                if opt not in legal:
                    opt = random.choice(legal)
                pols.append(result.policy_target.clone())
                acts.append(int(opt))
                nxt = curr.copy()
                nxt.day = min(30, curr.day + 1)
                if opt == MacroOption.EXPAND_NE and "NE" not in nxt.unlocked:
                    nxt.unlocked = tuple(sorted(set(nxt.unlocked) | {"NE"}))
                    nxt.capital = max(0.0, nxt.capital - LAND_PRICES["NE"])
                elif opt == MacroOption.EXPAND_SW and "SW" not in nxt.unlocked:
                    nxt.unlocked = tuple(sorted(set(nxt.unlocked) | {"SW"}))
                    nxt.capital = max(0.0, nxt.capital - LAND_PRICES["SW"])
                elif opt == MacroOption.EXPAND_SE and "SE" not in nxt.unlocked:
                    nxt.unlocked = tuple(sorted(set(nxt.unlocked) | {"SE"}))
                    nxt.capital = max(0.0, nxt.capital - LAND_PRICES["SE"])
                elif opt == MacroOption.TRICKLE_LIQUIDATION:
                    nxt.capital += 200.0
                rew = (wealth(nxt) - wealth(curr)) * REWARD_SCALE
                rews.append(rew)
                obs_seq.append(encode_macro_state(nxt))
                vals.append(wealth(nxt) * REWARD_SCALE)
                episode_log.append({"day": curr.day, "option": int(opt), "reward": rew})
                curr = nxt
            result = planner.mcts.search(curr, n_simulations=max(4, sims // 2))
            pols.append(result.policy_target.clone())
            trajs.append(
                {
                    "obs": torch.stack(obs_seq),
                    "actions": torch.tensor(acts, dtype=torch.long),
                    "rewards": torch.tensor(rews, dtype=torch.float32),
                    "values": torch.tensor(vals, dtype=torch.float32),
                    "policies": torch.stack(pols),
                }
            )
            if save_json:
                stub = {
                    "info": {"SelfPlay": True, "episode": ep, "source": "phase4_reanalysis"},
                    "rewards": [float(sum(rews) / max(REWARD_SCALE, 1e-8)), 0.0],
                    "steps_summary": episode_log,
                }
                out = os.path.join(replays_dir, f"selfplay_phase4_{int(time.time())}_{ep}.json")
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(stub, f)
    print(f"  [self-play] harvested {len(trajs)} episodes → PER (+replays stubs)")
    return trajs


def phase4_reanalysis(
    model: MuZeroNetwork,
    steps: int,
    batch: int,
    lr: float,
    sims: int = 16,
    cons_w: float = 0.4357,
    k_steps: int = 5,
    dynamics_grad_scale: float = 0.5,
    c_l2: float = 1e-4,
    trajectory_pool: Optional[Any] = None,
    target_model: Optional[MuZeroNetwork] = None,
    tau: float = 0.005,
    policy_weight: float = 0.8637,
    value_weight: float = 0.9637,
    reward_weight: float = 0.7357,
    cql_weight: float = 0.02357,
    lora_rank: int = LORA_RANK,
    lora_alpha: float = LORA_ALPHA,
    adapter_tag: str = "phase4",
    save_adapters: bool = True,
    selfplay_episodes: int = 4,
) -> Dict[str, Any]:
    """Phase 4: Self-play harvest + reanalyze PER; update LoRA adapters only."""
    print(
        f"[Phase 4] Self-play({selfplay_episodes}) + Re-analysis + LoRA (r={lora_rank}) "
        f"steps={steps} batch={batch} sims={sims}"
    )

    if not has_lora(model):
        injected = inject_lora(
            model, rank=lora_rank or DEFAULT_LORA_RANK, alpha=lora_alpha or DEFAULT_LORA_ALPHA
        )
        print(f"  injected LoRA into {len(injected)} linear projections")

    target_model = create_target_network(model)

    adapter_params = lora_parameters(model)
    if not adapter_params:
        raise RuntimeError("Phase 4 requires LoRA adapters but none were found after inject_lora()")

    opt = optim.Adam(adapter_params, lr=lr)
    planner = MuZeroPlanner(
        model=model,
        target_model=target_model,
        n_simulations=sims,
        gumbel_threshold=16,
        use_analytical_dynamics=False,
    )

    if isinstance(trajectory_pool, PrioritizedReplayBuffer):
        per_buffer = trajectory_pool
    elif trajectory_pool is not None and len(trajectory_pool) > 0:
        per_buffer = PrioritizedReplayBuffer(capacity=max(50000, len(trajectory_pool)))
        per_buffer.add_batch(list(trajectory_pool))
    else:
        per_buffer = PrioritizedReplayBuffer(capacity=50000)

    sp_trajs = harvest_selfplay_trajectories(
        model, n_episodes=selfplay_episodes, k_steps=k_steps, sims=sims
    )
    for tr in sp_trajs:
        init_p = float(tr["rewards"].abs().sum().item()) + 1.0
        per_buffer.add(tr, priority=init_p)

    if len(per_buffer) == 0:
        print("  no trajectory pool after self-play; skipping Phase 4 train")
        return {
            "steps": 0,
            "pool_size": 0,
            "avg_loss": 0.0,
            "adapter_path": None,
            "selfplay_episodes": len(sp_trajs),
        }

    t0 = time.time()
    last: Dict[str, float] = {}
    total_loss = 0.0
    n_ok = 0

    for i in range(steps):
        if selfplay_episodes > 0 and i > 0 and i % max(1, steps // 4) == 0:
            for tr in harvest_selfplay_trajectories(
                model,
                n_episodes=max(1, selfplay_episodes // 2),
                k_steps=k_steps,
                sims=sims,
                save_json=False,
            ):
                per_buffer.add(tr, priority=float(tr["rewards"].abs().sum().item()) + 1.0)

        chunk, indices, weights = per_buffer.sample(min(batch, len(per_buffer)))
        if len(chunk) < 2:
            continue
        obs_b = torch.stack([c["obs"] for c in chunk])
        act_b = torch.stack([c["actions"] for c in chunk])
        rew_b = torch.stack([c["rewards"] for c in chunk])
        val_b = torch.stack([c["values"] for c in chunk])
        pol_b = torch.stack([c["policies"] for c in chunk])

        model.eval()
        with torch.no_grad():
            n_re = min(len(obs_b), 16)
            for j in range(n_re):
                pol_b[j], val_b[j] = reanalyze_trajectory_slice(
                    planner=planner,
                    obs_slice=obs_b[j],
                    policy_slice=pol_b[j],
                    value_slice=val_b[j],
                    target_model=target_model,
                    sims=sims,
                )

        model.train()
        for p in model.parameters():
            p.requires_grad = False
        for p in adapter_params:
            p.requires_grad = True

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
            weights=weights,
            target_model=target_model,
            policy_weight=policy_weight,
            value_weight=value_weight,
            reward_weight=reward_weight,
            cql_weight=cql_weight,
        )
        opt.zero_grad()
        losses["total"].backward()
        nn.utils.clip_grad_norm_(adapter_params, 5.0)
        opt.step()
        update_target_network(model, target_model, tau=tau)

        if indices is not None and "td_errors" in losses:
            per_buffer.update_priorities(indices, losses["td_errors"])

        step_loss = float(losses["total"].detach())
        total_loss += step_loss
        n_ok += 1
        last = {
            "total": step_loss,
            "policy": float(losses["policy"]),
            "value": float(losses["value"]),
            "reward": float(losses["reward"]),
            "consistency": float(losses["consistency"]),
        }
        if (i + 1) % max(1, steps // 4) == 0 or i == 0:
            print(
                f"  step {i+1}/{steps} loss={last['total']:.4f} "
                f"π={last['policy']:.4f} v={last['value']:.4f} "
                f"(LoRA-only)"
            )

    adapter_path = None
    if save_adapters:
        adapter_path = route_lora_adapter(
            model,
            tag=adapter_tag,
            meta={
                "steps": steps,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "final_loss": last.get("total"),
                "selfplay_episodes": len(sp_trajs),
            },
        )
        print(f"  [artifact] LoRA adapters → {adapter_path}")

    avg = total_loss / max(1, n_ok)
    print(f"  done in {time.time()-t0:.1f}s avg_loss={avg:.4f} pool={len(per_buffer)}")
    return {
        "steps": steps,
        "pool_size": len(per_buffer),
        "avg_loss": avg,
        "final": last,
        "adapter_path": adapter_path,
        "selfplay_episodes": len(sp_trajs),
    }
