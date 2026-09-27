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

"""Phase 2: MCTS policy and value distillation with K-step trajectory reanalysis."""
from __future__ import annotations

import random
import time
from typing import Any, Dict, Optional

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
    MuZeroPlanner,
    PrioritizedReplayBuffer,
    compute_target_values,
    create_target_network,
    decode_macro_state,
    encode_macro_state,
    legal_macro_options,
    muzero_k_step_unroll_loss,
    update_target_network,
    wealth,
)
from ..low_rank_adaptation import (
    compute_low_rank_adaptation_scheduled_weights as compute_separate_scheduled_weights,
)


def _random_latent_seed_trajectories(n_trajectories: int, k_steps: int) -> Dict[str, torch.Tensor]:
    """Encoding-only seed batches when no expert PER pool is available.

    Does **not** import ``economy`` — Phase 2 must stay economy-free. States are
    randomly sampled MacroStates; actions use observation legality; rewards come
    from wealth deltas under a no-op hold (identity next-state) so MCTS reanalyze
    can still overwrite policy/value targets.
    """
    obs_all, acts_all, rews_all, vals_all, pol_all = [], [], [], [], []
    for _ in range(n_trajectories):
        day = random.randint(0, 25)
        unlocked_n = 1 if day < 8 else (2 if day < 18 else 3)
        unlocked = tuple(["NW", "NE", "SW", "SE"][:unlocked_n])
        capital = random.uniform(500.0, 20000.0)
        for q in LAND_ORDER:
            if q not in unlocked:
                capital = max(200.0, LAND_PRICES[q] + LAND_BUFFERS[q] + random.uniform(-100, 3000))
                break
        state = MacroState(
            day=day,
            capital=capital,
            unlocked=unlocked,
            shed={
                "WHEAT": random.uniform(0, 40),
                "MELON": random.uniform(0, 12),
                "STRAWBERRY": random.uniform(0, 20),
                "MILK": random.uniform(0, 15),
                "WOOL": random.uniform(0, 10),
            },
            prices=dict(DEFAULT_PRICES),
            active_animals=random.randint(1, 8),
        )
        obs_traj = [encode_macro_state(state)]
        acts_traj, rews_traj = [], []
        vals_traj = [wealth(state) * REWARD_SCALE]
        pol_traj = []
        curr = state
        for _ in range(k_steps):
            options = legal_macro_options(curr)
            opt = random.choice(options)
            opt_oh = torch.zeros(NUM_MACRO_OPTIONS)
            opt_oh[int(opt)] = 1.0
            pol_traj.append(opt_oh)
            # Hold state (no analytical step) — MCTS reanalyze supplies targets.
            nxt = curr
            rew = 0.0
            acts_traj.append(int(opt))
            rews_traj.append(rew)
            obs_traj.append(encode_macro_state(nxt))
            vals_traj.append(wealth(nxt) * REWARD_SCALE)
            curr = nxt
        opt_oh = torch.zeros(NUM_MACRO_OPTIONS)
        opt_oh[int(MacroOption.PASS)] = 1.0
        pol_traj.append(opt_oh)
        obs_all.append(torch.stack(obs_traj))
        acts_all.append(torch.tensor(acts_traj, dtype=torch.long))
        rews_all.append(torch.tensor(rews_traj, dtype=torch.float32))
        vals_all.append(torch.tensor(vals_traj, dtype=torch.float32))
        pol_all.append(torch.stack(pol_traj))
    return {
        "obs": torch.stack(obs_all),
        "actions": torch.stack(acts_all),
        "rewards": torch.stack(rews_all),
        "values": torch.stack(vals_all),
        "policies": torch.stack(pol_all),
    }


def phase2_mcts_distill(
    model: MuZeroNetwork,
    steps: int,
    batch: int,
    lr: float,
    sims: int,
    cons_w: float = 0.4357,
    k_steps: int = 5,
    dynamics_grad_scale: float = 0.5,
    c_l2: float = 1e-4,
    trajectory_pool: Optional[Any] = None,
    target_model: Optional[MuZeroNetwork] = None,
    tau: float = 0.005,
    reanalyze_interior: bool = True,
    policy_weight: float = 0.8637,
    value_weight: float = 0.9637,
    reward_weight: float = 0.7357,
    cql_weight: float = 0.02357,
    loss_schedule: str = "separate",
    schedule_directions: Optional[str] = None,
) -> Dict[str, float]:
    """Phase 2: Distill MCTS search targets into prediction networks via lookahead reanalysis."""
    pool_len = len(trajectory_pool) if trajectory_pool is not None else 0
    print(f"[Phase 2] MCTS distill K={k_steps} steps={steps} sims={sims} (reanalyse_pool={pool_len})")
    if target_model is None:
        target_model = create_target_network(model)

    planner = MuZeroPlanner(model=model, target_model=target_model, n_simulations=sims, gumbel_threshold=16)
    opt = optim.Adam(model.parameters(), lr=lr)

    if isinstance(trajectory_pool, PrioritizedReplayBuffer):
        per_buffer = trajectory_pool
    elif isinstance(trajectory_pool, list) and len(trajectory_pool) > 0:
        per_buffer = PrioritizedReplayBuffer(capacity=max(50000, len(trajectory_pool)))
        per_buffer.add_batch(trajectory_pool)
    else:
        per_buffer = PrioritizedReplayBuffer(capacity=50000)

    t0 = time.time()
    last: Dict[str, float] = {}

    for i in range(steps):
        use_pool = len(per_buffer) >= batch and (random.random() < 0.5 or pool_len > 0)
        indices = None
        weights = None

        if use_pool:
            chunk, indices, weights = per_buffer.sample(batch)
            obs_b = torch.stack([c["obs"] for c in chunk])
            act_b = torch.stack([c["actions"] for c in chunk])
            rew_b = torch.stack([c["rewards"] for c in chunk])
            val_b = torch.stack([c["values"] for c in chunk])
            pol_b = torch.stack([c["policies"] for c in chunk])
        else:
            trajs = _random_latent_seed_trajectories(n_trajectories=batch, k_steps=k_steps)
            obs_b = trajs["obs"]
            act_b = trajs["actions"]
            rew_b = trajs["rewards"]
            val_b = compute_target_values(target_model, obs_b, rew_b, discount=0.99)
            pol_b = trajs["policies"]

        # K-Step Reanalyze lookahead search
        model.eval()
        target_model.eval()
        with torch.no_grad():
            effective_k = min(k_steps, act_b.shape[1])
            n_reanalyze = min(len(obs_b), 16)
            for j in range(n_reanalyze):
                steps_to_re = [0]
                if reanalyze_interior and effective_k > 0:
                    steps_to_re.append(random.randint(1, effective_k))

                for k in steps_to_re:
                    state = decode_macro_state(obs_b[j, k])
                    result = planner.mcts.search(state, n_simulations=sims)
                    pol_b[j, k] = result.policy_target
                    v_boot, _ = target_model.predict_target_value_and_support(obs_b[j, k].unsqueeze(0))
                    val_b[j, k] = 0.5 * result.root_value + 0.5 * float(v_boot.item())

        model.train()
        if loss_schedule in ("separate", "low_rank_adaptation", "lora", "adaptive_moment_estimation", "adam"):
            cur_w = compute_separate_scheduled_weights(
                step=i,
                total_steps=steps,
                base_weights={
                    "consistency_weight": cons_w,
                    "policy_weight": policy_weight,
                    "value_weight": value_weight,
                    "reward_weight": reward_weight,
                    "cql_weight": cql_weight,
                    "schedule_directions": schedule_directions,
                },
            )
            step_cons = cur_w["consistency_weight"]
            step_pol = cur_w["policy_weight"]
            step_val = cur_w["value_weight"]
            step_rew = cur_w["reward_weight"]
            step_cql = cur_w["cql_weight"]
        else:
            step_cons = cons_w
            step_pol = policy_weight
            step_val = value_weight
            step_rew = reward_weight
            step_cql = cql_weight

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
            cons_w=step_cons,
            weights=weights,
            target_model=target_model,
            policy_weight=step_pol,
            value_weight=step_val,
            reward_weight=step_rew,
            cql_weight=step_cql,
        )
        opt.zero_grad()
        losses["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        update_target_network(model, target_model, tau=tau)
        if indices is not None and "td_errors" in losses:
            per_buffer.update_priorities(indices, losses["td_errors"])

        last = {
            "total": float(losses["total"].detach()),
            "policy": float(losses["policy"]),
            "value": float(losses["value"]),
            "reward": float(losses["reward"]),
            "consistency": float(losses["consistency"]),
        }
        if (i + 1) % max(1, steps // 4) == 0 or i == 0:
            print(
                f"  step {i+1}/{steps} loss={last['total']:.4f} "
                f"π={last['policy']:.4f} v={last['value']:.4f} rew={last['reward']:.4f} cons={last['consistency']:.4f}"
            )
    print(f"  done in {time.time()-t0:.1f}s final={last}")
    return last
