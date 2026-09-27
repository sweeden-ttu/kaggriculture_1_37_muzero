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

"""Master 4-Phase training runner for Kaggriculture MuZero."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Any, Dict, Optional

import torch

from muzero import (
    DEFAULT_MUZERO_CHECKPOINT,
    DEFAULT_SPATIAL_CHECKPOINT,
    HIDDEN_DIM,
    KaggricultureMuZeroChassis,
    LAND_BUFFERS,
    LAND_PRICES,
    MuZeroNetwork,
    MuZeroPlanner,
    SampledMuZeroMCTS,
    create_target_network,
    encode_spatial_observation,
    load_muzero_checkpoint,
    load_spatial_checkpoint,
    save_muzero_checkpoint,
    save_spatial_checkpoint,
)
from muzero.lora import merge_lora
from .adaptive_moment_estimation import (
    apply_adaptive_moment_estimation_schedule,
    compute_adaptive_moment_scheduled_weights,
)
from .low_rank_adaptation import (
    compute_low_rank_adaptation_scheduled_weights,
    compute_separate_scheduled_weights,
    validate_loss_weights_rule,
)
from .phases.phase1_bootstrap import phase1_analytical_bootstrap
from .phases.phase2_distill import phase2_mcts_distill
from .phases.phase3_bc import (
    build_expert_trajectory_pool,
    discover_expert_replays,
    phase3_expert_behavior_cloning,
)
from .phases.phase4_reanalysis import (
    phase4_reanalysis,
    route_base_snapshot,
    route_lora_adapter,
)
from .phases.phase_spatial import (
    make_spatial_buffer,
    phase_spatial_bc,
    phase_spatial_bootstrap,
    phase_spatial_mcts_distill,
    phase_spatial_reanalyze,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.path.join(HERE, "artifacts")
DEFAULT_OUT = os.path.join(ARTIFACTS, DEFAULT_MUZERO_CHECKPOINT)
DEFAULT_SPATIAL_OUT = os.path.join(ARTIFACTS, DEFAULT_SPATIAL_CHECKPOINT)
SNAPSHOTS_DIR = os.path.join(ARTIFACTS, "snapshots")
LORA_ADAPTERS_DIR = os.path.join(ARTIFACTS, "lora_adapters")


def smoke_expand_spatial(chassis: KaggricultureMuZeroChassis) -> None:
    mcts = SampledMuZeroMCTS(chassis, num_samples=4, num_simulations=8)
    obs = {
        "player": 0,
        "day": 8,
        "hour": 4,
        "farms": [
            {
                "money": LAND_PRICES["SW"] + LAND_BUFFERS["SW"] + 200.0,
                "unlocked_quadrants": ["NW", "NE"],
                "tiles": [[{} for _ in range(10)] for _ in range(10)],
                "farmer": {"x": 1, "y": 1},
                "hands": [],
                "animals": {},
            },
            {},
        ],
        "private": {"shed": {}},
        "market": {"prices": {}},
    }
    obs_t = encode_spatial_observation(obs).unsqueeze(0)
    best, probs, val = mcts.search(obs_t, add_dirichlet_noise=False)
    print(f"[smoke-spatial] joint={best} root_value={val:.4f} candidates={len(probs)}")


def smoke_expand(model: MuZeroNetwork) -> None:
    planner = MuZeroPlanner(
        model=model, n_simulations=16, gumbel_threshold=16, use_analytical_dynamics=False
    )
    state_money = LAND_PRICES["SW"] + LAND_BUFFERS["SW"] + 200.0
    result = planner.search(
        money=state_money,
        day=8,
        unlocked=["NW", "NE"],
        n_simulations=16,
    )
    target = planner.evaluate_expansion(
        money=state_money,
        day=8,
        unlocked=["NW", "NE"],
        n_simulations=16,
    )
    print(
        f"[smoke] day8 NE-unlocked ${state_money:.0f} → "
        f"best={result.best_option.name} expand={target}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Kaggriculture MuZero (Phases 1–4)")
    parser.add_argument(
        "--arch",
        choices=["spatial", "macro"],
        default="spatial",
        help="Production architecture: spatial Sampled MuZero (default) or legacy macro MLP",
    )
    # Competition / publication long-horizon defaults (match Makefile)
    parser.add_argument("--phase1-steps", type=int, default=1000)
    parser.add_argument("--phase2-steps", type=int, default=400)
    parser.add_argument("--phase3-epochs", type=int, default=400, help="Spatial: reanalyze steps; macro: BC epochs")
    parser.add_argument("--phase4-steps", type=int, default=1000, help="Spatial: MCTS distill steps; macro: reanalyze")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--sims", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--policy-weight", type=float, default=0.8637)
    parser.add_argument("--value-weight", type=float, default=0.9637)
    parser.add_argument("--reward-weight", type=float, default=0.7357)
    parser.add_argument("--cql-weight", type=float, default=0.02357)
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--consistency-weight", type=float, default=0.4357)
    parser.add_argument(
        "--loss-schedule",
        choices=["none", "separate", "low_rank_adaptation", "lora", "adaptive_moment_estimation", "adam"],
        default="low_rank_adaptation",
        help="Loss schedule strategy: low_rank_adaptation (LoRA / standard), adaptive_moment_estimation (Adam / boost), separate, or none",
    )
    parser.add_argument("--auto-adjust-weights", action="store_true")
    parser.add_argument("--k-steps", type=int, default=5)
    parser.add_argument(
        "--buffer-max",
        type=int,
        default=1000,
        help="Prioritized replay max trajectories (competition default 1000)",
    )
    parser.add_argument("--dynamics-grad-scale", type=float, default=0.5)
    parser.add_argument("--c-l2", type=float, default=1e-4)
    parser.add_argument("--support-size", type=int, default=601)
    parser.add_argument("--no-support", action="store_true")
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    parser.add_argument("--min-expert-score", type=float, default=80000.0)
    parser.add_argument("--max-expert-replays", type=int, default=50)
    parser.add_argument("--skip-phase2", action="store_true")
    parser.add_argument("--skip-phase3", action="store_true")
    parser.add_argument("--skip-phase4", action="store_true")
    parser.add_argument(
        "--spatial-phase",
        choices=["all", "bc", "bootstrap", "distill", "reanalyze"],
        default="all",
        help=(
            "Spatial curriculum slice: all (default), bc (expert BC), bootstrap (SimSiam), "
            "distill (Sampled-MCTS self-play), reanalyze (refresh MCTS targets)"
        ),
    )
    parser.add_argument("--use-lora", action="store_true", help="Enable Phase 4 LoRA re-analysis")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument(
        "--selfplay-episodes",
        type=int,
        default=50,
        help="Self-play harvest episodes for MCTS distill (competition default 50)",
    )
    parser.add_argument(
        "--gate-2033",
        action="store_true",
        help="After training, run automated H2H gate vs submission_2033 (requires --gate-agent)",
    )
    parser.add_argument("--gate-agent", type=str, default=None, help="Agent .py/.zip path for --gate-2033")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--summary-out", type=str, default=None)
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta", type=float, default=0.4)
    parser.add_argument("--no-interior-reanalyze", action="store_true")
    parser.add_argument(
        "--schedule-directions",
        type=str,
        default=None,
        help="Per-weight schedule directions (e.g. 'consistency:up,value:down')",
    )
    args = parser.parse_args()

    # Normalize schedule aliases: lora -> low_rank_adaptation, adam -> adaptive_moment_estimation
    if args.loss_schedule == "lora":
        args.loss_schedule = "low_rank_adaptation"
    elif args.loss_schedule == "adam":
        args.loss_schedule = "adaptive_moment_estimation"

    if args.loss_schedule == "adaptive_moment_estimation":
        apply_adaptive_moment_estimation_schedule()

    validated_weights = validate_loss_weights_rule(
        consistency_weight=args.consistency_weight,
        policy_weight=args.policy_weight,
        value_weight=args.value_weight,
        reward_weight=args.reward_weight,
        cql_weight=args.cql_weight,
        auto_adjust=args.auto_adjust_weights,
    )
    args.consistency_weight = validated_weights["consistency-weight"]
    args.policy_weight = validated_weights["policy-weight"]
    args.value_weight = validated_weights["value-weight"]
    args.reward_weight = validated_weights["reward-weight"]
    args.cql_weight = validated_weights["cql-weight"]

    print(
        f"[weights] Validated distinct loss weights: "
        f"cons={args.consistency_weight:.4f}, pol={args.policy_weight:.4f}, "
        f"val={args.value_weight:.4f}, rew={args.reward_weight:.4f}, "
        f"cql={args.cql_weight:.4f} (schedule={args.loss_schedule})"
    )

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"[seed] initialized random seed={args.seed}")

    # ------------------------------------------------------------------
    # Spatial Sampled MuZero (production default)
    # Curriculum: BC → bootstrap → MCTS distill (+selfplay) → reanalyze
    # ------------------------------------------------------------------
    if args.arch == "spatial":
        out_path = args.out if args.out != DEFAULT_OUT else DEFAULT_SPATIAL_OUT
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        chassis = KaggricultureMuZeroChassis(support_size=args.support_size)
        if not args.scratch and os.path.isfile(out_path):
            load_spatial_checkpoint(chassis, out_path, strict=False)
            print(f"[init-spatial] warm-start from {out_path}")
        else:
            print("[init-spatial] training KaggricultureMuZeroChassis from scratch")

        batch = max(1, args.batch)
        k_steps = max(1, min(args.k_steps, 5))
        buffer = make_spatial_buffer(
            max_trajectories=max(256, int(args.buffer_max)),
            alpha=float(args.per_alpha),
            beta=float(args.per_beta),
            k_steps=k_steps,
        )
        ema_tau = 0.995 if args.target_tau >= 0.5 else max(0.9, 1.0 - args.target_tau)

        # --spatial-phase selects a single curriculum slice (make phase-1 … phase-4).
        # Legacy --skip-phase* flags still apply when spatial_phase == "all".
        spat = args.spatial_phase
        run_bc = spat in ("all", "bc") and not args.skip_phase3
        run_boot = spat in ("all", "bootstrap") and not args.skip_phase2
        run_distill = spat in ("all", "distill") and not args.skip_phase4
        run_re = spat == "reanalyze" or (
            spat == "all"
            and (
                (not args.skip_phase3 and args.phase3_epochs > 0)
                or (not args.skip_phase4 and args.phase4_steps > 0)
            )
        )
        if spat != "all":
            print(f"[spatial] isolated phase → {spat}")

        # Phase A — expert BC (skip with --skip-phase3)
        p_bc: Dict[str, Any] = {}
        if run_bc:
            p_bc = phase_spatial_bc(
                chassis,
                buffer,
                steps=max(0, args.phase1_steps),
                batch=batch,
                lr=args.lr,
                k_steps=k_steps,
                consistency_weight=args.consistency_weight,
                sims=args.sims,
                ema_tau=ema_tau,
                min_score=args.min_expert_score,
                max_expert_replays=args.max_expert_replays,
            )
        elif spat == "all":
            print("[spatial] skip-phase3: skipping expert BC")

        # Phase B — dynamics / SimSiam bootstrap on shared buffer
        p_boot: Dict[str, Any] = {}
        if run_boot:
            p_boot = phase_spatial_bootstrap(
                chassis,
                steps=max(0, args.phase2_steps),
                batch=batch,
                lr=args.lr * 0.75,
                k_steps=k_steps,
                consistency_weight=args.consistency_weight,
                buffer=buffer,
                sims=args.sims,
                ema_tau=ema_tau,
                seed_synthetic=(len(buffer.trajectories) == 0),
            )
        elif spat == "all":
            print("[spatial] skip-phase2: skipping bootstrap")

        # Phase C — Sampled MCTS self-play harvest + distill
        p_distill: Dict[str, Any] = {}
        if run_distill:
            distill_steps = max(0, args.phase4_steps)
            p_distill = phase_spatial_mcts_distill(
                chassis,
                buffer,
                steps=distill_steps,
                batch=batch,
                lr=args.lr * 0.5,
                sims=args.sims,
                k_steps=k_steps,
                consistency_weight=args.consistency_weight,
                selfplay_episodes=args.selfplay_episodes,
                steps_per_ep=48,
                ema_tau=ema_tau,
            )
        elif spat == "all":
            print("[spatial] skip-phase4: skipping MCTS distill / self-play")

        # Phase D — reanalyze refresh
        p_re: Dict[str, Any] = {}
        if spat == "reanalyze":
            re_steps = max(1, args.phase3_epochs)
        elif spat == "all":
            re_steps = max(0, args.phase3_epochs) if not args.skip_phase3 else 0
            # When distill ran, still run a short reanalyze pass after distill
            if not args.skip_phase4 and args.phase4_steps > 0:
                re_steps = max(re_steps, max(5, args.phase4_steps // 2))
        else:
            re_steps = 0
        if run_re and re_steps > 0:
            p_re = phase_spatial_reanalyze(
                chassis,
                buffer,
                steps=re_steps,
                batch=batch,
                lr=args.lr * 0.25,
                sims=args.sims,
                k_steps=k_steps,
                consistency_weight=args.consistency_weight,
                ema_tau=ema_tau,
            )

        meta = {
            "arch": "spatial",
            "spatial_phase": spat,
            "bc": p_bc,
            "bootstrap": p_boot,
            "mcts_distill": p_distill,
            "reanalyze": p_re,
            "buffer_trajectories": len(buffer.trajectories),
            "sims": args.sims,
            "batch": batch,
            "k_steps": k_steps,
            "per_alpha": buffer.alpha,
            "per_beta": buffer.beta,
            "consistency_weight": args.consistency_weight,
            "seed": args.seed,
        }
        saved = save_spatial_checkpoint(chassis, path=out_path, meta=meta)
        print(f"[artifact] spatial checkpoint → {saved} (trajs={len(buffer.trajectories)})")
        smoke_expand_spatial(chassis)

        gate_report = None
        if args.gate_2033:
            agent_path = args.gate_agent
            if not agent_path:
                print("[gate] --gate-2033 set but --gate-agent missing; skipping")
            else:
                from evaluation.benchmarks import gate_candidate_vs_2033

                try:
                    gate_report = gate_candidate_vs_2033(agent_path, games=2, n_workers=2, quiet=False)
                    print(f"[gate] passed={gate_report.get('gate', {}).get('passed')}")
                except Exception as e:
                    print(f"[gate] failed: {e}")
                    gate_report = {"error": str(e)}

        if args.summary_out:
            summary = {
                "arch": "spatial",
                "out": saved,
                "meta": meta,
                "gate_2033": gate_report,
                "timestamp": time.time(),
            }
            os.makedirs(os.path.dirname(os.path.abspath(args.summary_out)) or ".", exist_ok=True)
            with open(args.summary_out, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"[summary] wrote training metrics to {args.summary_out}")
        return

    # ------------------------------------------------------------------
    # Legacy macro MLP path (--arch macro)
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    os.makedirs(LORA_ADAPTERS_DIR, exist_ok=True)
    support_sz = 1 if args.no_support else args.support_size
    model = MuZeroNetwork(hidden_dim=args.hidden_dim, support_size=support_sz)
    if not args.scratch:
        model, loaded = load_muzero_checkpoint(model, path=args.out)
        if loaded:
            print(
                f"[init] warm-start from {loaded} "
                f"(support_size={model.support_size}, hidden_dim={model.hidden_dim})"
            )
    else:
        print(
            f"[init] training fresh model from scratch "
            f"(support_size={model.support_size}, hidden_dim={model.hidden_dim})"
        )

    target_model = create_target_network(model)

    p1_metrics = phase1_analytical_bootstrap(
        model,
        args.phase1_steps,
        args.batch,
        args.lr,
        args.consistency_weight,
        k_steps=args.k_steps,
        dynamics_grad_scale=args.dynamics_grad_scale,
        c_l2=args.c_l2,
        target_model=target_model,
        tau=args.target_tau,
        policy_weight=args.policy_weight,
        value_weight=args.value_weight,
        reward_weight=args.reward_weight,
        cql_weight=args.cql_weight,
        loss_schedule=args.loss_schedule,
        schedule_directions=args.schedule_directions,
    )

    traj_pool = None
    if not args.skip_phase3 or not args.skip_phase2 or (args.use_lora and not args.skip_phase4):
        experts = discover_expert_replays(
            min_score=args.min_expert_score, limit=args.max_expert_replays
        )
        if experts:
            traj_pool = build_expert_trajectory_pool(
                experts,
                k_steps=args.k_steps,
                alpha=args.per_alpha,
                beta=args.per_beta,
            )
            print(
                f"[experts] extracted {len(traj_pool)} K={args.k_steps} trajectory slices "
                f"into PER buffer from {len(experts)} expert replays"
            )

    p2_metrics: Dict[str, Any] = {}
    if not args.skip_phase2:
        p2_metrics = phase2_mcts_distill(
            model,
            args.phase2_steps,
            min(args.batch, 32),
            args.lr * 0.5,
            args.sims,
            args.consistency_weight,
            k_steps=args.k_steps,
            dynamics_grad_scale=args.dynamics_grad_scale,
            c_l2=args.c_l2,
            trajectory_pool=traj_pool,
            target_model=target_model,
            tau=args.target_tau,
            reanalyze_interior=not args.no_interior_reanalyze,
            policy_weight=args.policy_weight,
            value_weight=args.value_weight,
            reward_weight=args.reward_weight,
            cql_weight=args.cql_weight,
            loss_schedule=args.loss_schedule,
            schedule_directions=args.schedule_directions,
        )

    p3_metrics: Dict[str, Any] = {}
    if not args.skip_phase3 and traj_pool is not None and len(traj_pool) > 0:
        p3_metrics = phase3_expert_behavior_cloning(
            model,
            args.phase3_epochs,
            args.batch,
            args.lr * 0.5,
            args.consistency_weight,
            k_steps=args.k_steps,
            dynamics_grad_scale=args.dynamics_grad_scale,
            c_l2=args.c_l2,
            prebuilt_pool=traj_pool,
            target_model=target_model,
            tau=args.target_tau,
            policy_weight=args.policy_weight,
            value_weight=args.value_weight,
            reward_weight=args.reward_weight,
            cql_weight=args.cql_weight,
        )

    base_snap = route_base_snapshot(
        model,
        tag="pre_phase4",
        meta={"hidden_dim": model.hidden_dim, "support_size": model.support_size},
    )
    print(f"[artifact] frozen base snapshot → {base_snap}")

    p4_metrics: Dict[str, Any] = {}
    run_phase4 = args.use_lora and not args.skip_phase4
    if run_phase4 and traj_pool is not None and len(traj_pool) > 0:
        p4_metrics = phase4_reanalysis(
            model,
            steps=args.phase4_steps,
            batch=min(args.batch, 32),
            lr=args.lr * 0.25,
            sims=args.sims,
            cons_w=args.consistency_weight,
            k_steps=args.k_steps,
            dynamics_grad_scale=args.dynamics_grad_scale,
            c_l2=args.c_l2,
            trajectory_pool=traj_pool,
            target_model=target_model,
            tau=args.target_tau,
            policy_weight=args.policy_weight,
            value_weight=args.value_weight,
            reward_weight=args.reward_weight,
            cql_weight=args.cql_weight,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            adapter_tag=f"train_{int(time.time())}",
            save_adapters=True,
            selfplay_episodes=args.selfplay_episodes,
        )
        merge_lora(model)

    meta = {
        "phase1_steps": args.phase1_steps,
        "phase2_steps": args.phase2_steps if not args.skip_phase2 else 0,
        "phase3_epochs": args.phase3_epochs if not args.skip_phase3 else 0,
        "phase4_steps": args.phase4_steps if run_phase4 else 0,
        "consistency_weight": args.consistency_weight,
        "policy_weight": args.policy_weight,
        "value_weight": args.value_weight,
        "reward_weight": args.reward_weight,
        "cql_weight": args.cql_weight,
        "loss_schedule": args.loss_schedule,
        "k_steps": args.k_steps,
        "dynamics_grad_scale": args.dynamics_grad_scale,
        "c_l2": args.c_l2,
        "support_size": support_sz,
        "hidden_dim": args.hidden_dim,
        "lora_rank": args.lora_rank if run_phase4 else None,
        "seed": args.seed,
        "base_snapshot": base_snap,
        "lora_adapter": p4_metrics.get("adapter_path"),
        "selfplay_episodes": p4_metrics.get("selfplay_episodes"),
    }
    save_muzero_checkpoint(model, path=args.out, meta=meta)
    print(f"[save] {args.out}")

    gate_report = None
    if args.gate_2033:
        agent_path = args.gate_agent
        if not agent_path:
            print("[gate] --gate-2033 set but --gate-agent missing; skipping")
        else:
            from evaluation.benchmarks import gate_candidate_vs_2033

            try:
                gate_report = gate_candidate_vs_2033(agent_path, games=2, n_workers=2, quiet=False)
                print(f"[gate] passed={gate_report.get('gate', {}).get('passed')}")
            except Exception as e:
                print(f"[gate] failed: {e}")
                gate_report = {"error": str(e)}

    if args.summary_out:
        summary_payload = {
            "meta": meta,
            "phase1": p1_metrics,
            "phase2": p2_metrics,
            "phase3": p3_metrics,
            "phase4": p4_metrics,
            "gate_2033": gate_report,
            "timestamp": time.time(),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_out)), exist_ok=True)
        with open(args.summary_out, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)
        print(f"[summary] wrote training metrics to {args.summary_out}")

    smoke_expand(model)


if __name__ == "__main__":
    main()
