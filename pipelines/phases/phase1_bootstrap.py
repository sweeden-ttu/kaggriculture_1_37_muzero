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

"""Phase 1: Analytical bootstrap for self-consistent latent representation and dynamics."""
from __future__ import annotations

import time
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from muzero import (
    MuZeroNetwork,
    compute_target_values,
    create_target_network,
    muzero_k_step_unroll_loss,
    update_target_network,
)
from muzero.economy import generate_analytical_trajectories
from ..low_rank_adaptation import (
    compute_low_rank_adaptation_scheduled_weights as compute_separate_scheduled_weights,
)


def phase1_analytical_bootstrap(
    model: MuZeroNetwork,
    steps: int,
    batch: int,
    lr: float,
    cons_w: float = 0.4357,
    k_steps: int = 5,
    dynamics_grad_scale: float = 0.5,
    c_l2: float = 1e-4,
    target_model: Optional[MuZeroNetwork] = None,
    tau: float = 0.005,
    policy_weight: float = 0.8637,
    value_weight: float = 0.9637,
    reward_weight: float = 0.7357,
    cql_weight: float = 0.02357,
    loss_schedule: str = "separate",
    schedule_directions: Optional[str] = None,
) -> Dict[str, float]:
    """Phase 1: Analytical bootstrap using self-consistent value-equivalent dynamics."""
    print(f"[Phase 1] Analytical bootstrap K={k_steps} steps={steps} batch={batch} support_size={model.support_size}")
    if target_model is None:
        target_model = create_target_network(model)
    opt = optim.Adam(model.parameters(), lr=lr)
    model.train()
    target_model.eval()
    t0 = time.time()
    last: Dict[str, float] = {}

    for i in range(steps):
        trajs = generate_analytical_trajectories(n_trajectories=batch, k_steps=k_steps)
        # Compute value targets using EMA target network θ^- for bootstrapping
        val_targets = compute_target_values(
            target_model=target_model,
            obs_seq=trajs["obs"],
            rewards=trajs["rewards"],
            discount=0.99,
        )

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
            obs_seq=trajs["obs"],
            actions=trajs["actions"],
            rewards=trajs["rewards"],
            value_targets=val_targets,
            policy_targets=trajs["policies"],
            k_steps=k_steps,
            dynamics_grad_scale=dynamics_grad_scale,
            c_l2=c_l2,
            cons_w=step_cons,
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
        # Polyak averaging: θ^- ← (1 - tau) θ^- + tau θ
        update_target_network(model, target_model, tau=tau)

        last = {k: float(v) for k, v in losses.items() if k not in ("total", "td_errors")}
        last["total"] = float(losses["total"].detach())
        if (i + 1) % max(1, steps // 5) == 0 or i == 0:
            print(
                f"  step {i+1}/{steps} loss={last['total']:.4f} "
                f"rew={last.get('reward', 0):.4f} val={last.get('value', 0):.4f} pol={last.get('policy', 0):.4f}"
            )
    print(f"  done in {time.time()-t0:.1f}s final={last}")
    return last
