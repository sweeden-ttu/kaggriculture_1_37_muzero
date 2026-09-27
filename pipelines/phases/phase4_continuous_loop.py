#!/usr/bin/env python3
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

"""Phase 4 continuous self-improving RL loop (REFACTOR_PLAN curriculum Phase 4).

Owns the autonomous outer loop that:
  1. Harvests / streams self-play into ``replays/`` (Mode A) or reanalyzes PER (Mode B)
  2. Trains via ``train_muzero.py`` with Phase 4 LoRA re-analysis
     (``phase4_reanalysis`` → adapters in ``artifacts/lora_adapters/``)
  3. Evaluates candidates and gates promotion vs the 2033 champion
  4. Atomically updates champion checkpoints and ``artifacts/snapshots/``

CLI entrypoints ``self_improving_loop.py`` / ``train.sh`` / ``boost.sh`` delegate here.
"""
from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from evaluation.benchmarks import find_baselines
from evaluation.harness import eval_match_worker, play_single_game
from muzero import MuZeroNetwork, load_muzero_checkpoint, save_muzero_checkpoint
from packaging.builder import build_hybrid_package
from pipelines.low_rank_adaptation import validate_loss_weights_rule
from pipelines.replay_manager import get_default_replay_budget_gb, prune_stale_selfplay_replays

ART = os.path.join(HERE, "artifacts")
REPLAYS = os.path.join(HERE, "replays")
BASELINES = os.path.join(HERE, "baselines")
SNAPSHOTS = os.path.join(ART, "snapshots")
POPULATION = os.path.join(ART, "population")

DIST_MAIN = os.path.join(HERE, "dist", "main.py")
CKPT = os.path.join(ART, "muzero_checkpoints.pt")
CHAMPION_CKPT = os.path.join(ART, "muzero_checkpoints_champion.pt")
CHAMPION_ZIP = os.path.join(ART, "champion_submission.zip")
CHAMPION_PY = os.path.join(ART, "champion_main.py")
CANDIDATE_ZIP = os.path.join(ART, "candidate_submission.zip")
CANDIDATE_PY = os.path.join(ART, "candidate_main.py")

SEED_FILE = os.path.join(ART, "loop_seed.txt")
STATUS_FILE = os.path.join(ART, "loop_status.json")
HISTORY_FILE = os.path.join(ART, "continuous_learning_history.json")
CHAMPION_HISTORY_LOG = os.path.join(ART, "champion_history.log")
TRAIN_SUMMARY = os.path.join(ART, "train_summary.json")

COMBINATION_PRESETS: List[Dict[str, Any]] = [
    {
        "name": "Combo-1-Agile-Exploration",
        "total_iterations": 100,
        "phase1_steps": 50,
        "phase2_steps": 25,
        "phase3_epochs": 4,
        "sims": 8,
        "consistency_weight": 0.3557,
        "policy_weight": 0.8137,
        "value_weight": 0.9637,
        "reward_weight": 0.7357,
        "cql_weight": 0.01357,
        "k_steps": 3,
        "hidden_dim": 32,
    },
    {
        "name": "Combo-2-Deep-MCTS-Distill",
        "total_iterations": 500,
        "phase1_steps": 150,
        "phase2_steps": 50,
        "phase3_epochs": 6,
        "sims": 16,
        "consistency_weight": 0.4357,
        "policy_weight": 0.8637,
        "value_weight": 0.9437,
        "reward_weight": 0.7157,
        "cql_weight": 0.02357,
        "k_steps": 4,
        "hidden_dim": 32,
    },
    {
        "name": "Combo-3-High-Capacity-Reanalyze",
        "total_iterations": 1000,
        "phase1_steps": 250,
        "phase2_steps": 80,
        "phase3_epochs": 8,
        "sims": 16,
        "consistency_weight": 0.5157,
        "policy_weight": 0.8837,
        "value_weight": 0.9637,
        "reward_weight": 0.7557,
        "cql_weight": 0.04357,
        "k_steps": 5,
        "hidden_dim": 64,
    },
    {
        "name": "Combo-4-Long-Horizon-Conservative",
        "total_iterations": 2500,
        "phase1_steps": 500,
        "phase2_steps": 120,
        "phase3_epochs": 10,
        "sims": 20,
        "consistency_weight": 0.6357,
        "policy_weight": 0.8437,
        "value_weight": 0.9737,
        "reward_weight": 0.7757,
        "cql_weight": 0.08357,
        "k_steps": 5,
        "hidden_dim": 64,
    },
]


def next_seed(base: int = 10101, stride: int = 37) -> int:
    seed = base
    if os.path.isfile(SEED_FILE):
        try:
            with open(SEED_FILE, "r", encoding="utf-8") as f:
                seed = int(f.read().strip()) + stride
        except Exception:
            seed = base
    with open(SEED_FILE, "w", encoding="utf-8") as f:
        f.write(str(seed))
    return seed


def run_evaluation_episodes(
    candidate_pkg: str,
    baseline_paths: Sequence[str],
    *,
    n_episodes: int = 20,
    base_seed: int = 20000,
    n_workers: int = 4,
    save_replays: bool = True,
    replays_dir: str = REPLAYS,
    min_replay_score: float = 80000.0,
) -> Dict[str, Any]:
    if not baseline_paths:
        raise ValueError("No baseline opponents provided for evaluation.")

    games_per_baseline = max(1, n_episodes // len(baseline_paths))
    tasks: List[Dict[str, Any]] = []
    game_idx = 0

    for b_path in baseline_paths:
        for g in range(games_per_baseline):
            if len(tasks) >= n_episodes:
                break
            game_idx += 1
            is_p0 = (g % 2 == 0)
            seed = base_seed + game_idx
            tasks.append({
                "agent_a": candidate_pkg,
                "agent_b": b_path,
                "seed": seed,
                "a_is_p0": is_p0,
                "episode_steps": 720,
                "save_replay_dir": replays_dir if save_replays else None,
                "min_replay_score": min_replay_score,
            })

    while len(tasks) < n_episodes:
        game_idx += 1
        b_path = baseline_paths[len(tasks) % len(baseline_paths)]
        is_p0 = (len(tasks) % 2 == 0)
        seed = base_seed + game_idx
        tasks.append({
            "agent_a": candidate_pkg,
            "agent_b": b_path,
            "seed": seed,
            "a_is_p0": is_p0,
            "episode_steps": 720,
            "save_replay_dir": replays_dir if save_replays else None,
            "min_replay_score": min_replay_score,
        })

    t0 = time.time()
    results: List[Dict[str, Any]] = []

    if n_workers > 1 and len(tasks) > 1:
        with mp.Pool(processes=min(n_workers, len(tasks))) as pool:
            results = pool.map(eval_match_worker, tasks)
    else:
        results = [eval_match_worker(t) for t in tasks]

    elapsed = round(time.time() - t0, 2)
    cand_scores = [float(r["score_a"]) for r in results]
    opp_scores = [float(r["score_b"]) for r in results]
    deltas = [float(r["delta_a"]) for r in results]

    wins = sum(1 for r in results if r["winner"] == "A")
    losses = sum(1 for r in results if r["winner"] == "B")
    ties = sum(1 for r in results if r["winner"] == "TIE")

    avg_score = sum(cand_scores) / max(1, len(cand_scores))
    avg_opp = sum(opp_scores) / max(1, len(opp_scores))
    avg_delta = sum(deltas) / max(1, len(deltas))
    win_pct = 100.0 * wins / max(1, len(results))

    return {
        "n_episodes": len(results),
        "avg_score": avg_score,
        "avg_opp_score": avg_opp,
        "avg_delta": avg_delta,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_pct": win_pct,
        "elapsed_seconds": elapsed,
        "matches": results,
    }


def setup_baseline_champion(
    base_seed: int = 10101,
    n_episodes: int = 10,
    n_workers: int = 4,
    save_replays: bool = True,
) -> Tuple[float, Optional[Dict[str, Any]]]:
    os.makedirs(ART, exist_ok=True)
    os.makedirs(SNAPSHOTS, exist_ok=True)
    os.makedirs(POPULATION, exist_ok=True)
    os.makedirs(REPLAYS, exist_ok=True)

    s_champ: Optional[float] = None
    optimizer_state = None

    if os.path.isfile(CHAMPION_CKPT):
        try:
            ckpt_data = torch.load(CHAMPION_CKPT, map_location="cpu")
            if isinstance(ckpt_data, dict):
                s_champ = ckpt_data.get("champion_score")
                optimizer_state = ckpt_data.get("optimizer_state_dict")
        except Exception as e:
            print(f"[Baseline Setup] Warning reading {CHAMPION_CKPT}: {e}")

    if not os.path.isfile(CHAMPION_CKPT):
        print(f"[Baseline Setup] Champion checkpoint not found. Initializing seed baseline...")
        if os.path.isfile(CKPT):
            shutil.copy2(CKPT, CHAMPION_CKPT)
        else:
            seed_net = MuZeroNetwork(hidden_dim=32, support_size=601)
            save_muzero_checkpoint(seed_net, CHAMPION_CKPT, meta={"seed_architecture": True}, atomic=True)

    if not os.path.isfile(CHAMPION_ZIP) or not os.path.isfile(CHAMPION_PY):
        build_hybrid_package(ckpt_path=CHAMPION_CKPT, out_zip=CHAMPION_ZIP, out_py=CHAMPION_PY)

    if s_champ is None:
        print("\n" + "=" * 90)
        print("  PHASE A: INITIALIZING BASELINE CHAMPION SCORE (10 Episodes)")
        print("=" * 90)
        baselines = find_baselines(BASELINES)
        eval_res = run_evaluation_episodes(
            candidate_pkg=CHAMPION_ZIP,
            baseline_paths=baselines,
            n_episodes=n_episodes,
            base_seed=base_seed,
            n_workers=n_workers,
            save_replays=save_replays,
        )
        s_champ = float(eval_res["avg_score"])
        ckpt_net, _ = load_muzero_checkpoint(path=CHAMPION_CKPT)
        save_muzero_checkpoint(
            ckpt_net,
            CHAMPION_CKPT,
            champion_score=s_champ,
            atomic=True,
        )
        print(f"[Baseline Setup] Determined S_champion = ${s_champ:,.0f} coins ({eval_res['win_pct']:.1f}% win rate across {n_episodes} games).")
        print(f"[Baseline Setup] Recorded to {CHAMPION_CKPT} (atomic).")
    else:
        print(f"[Baseline Setup] Loaded existing champion: S_champion = ${s_champ:,.0f} coins.")

    seed_snap = os.path.join(SNAPSHOTS, "snapshot_seed_champion.pt")
    if not os.path.isfile(seed_snap):
        shutil.copy2(CHAMPION_CKPT, seed_snap)

    return s_champ, optimizer_state


def generate_candidate_hyperparams(
    iteration: int,
    strategy: str,
    base_iterations: int = 1000,
    base_hidden_dim: int = 32,
    base_seed: int = 10101,
    stride: int = 37,
    prev_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    seed = next_seed(base=base_seed, stride=stride)
    train_seed = (seed * 10007) % 2147483647

    raw_cfg: Dict[str, Any]
    if strategy == "combinations":
        preset = dict(COMBINATION_PRESETS[(iteration - 1) % len(COMBINATION_PRESETS)])
        preset["iteration"] = iteration
        preset["seed"] = seed
        preset["train_seed"] = train_seed
        raw_cfg = preset

    elif strategy == "grid":
        grid_iterations = [500, 1000, 1500, 2500]
        grid_hidden_dims = [32, 64]
        grid_k_steps = [3, 4, 5]

        idx = iteration - 1

        # Every 3rd grid trial sweeps the PPO microcontroller macrocontroller dials
        # (lr, gamma, clip, entropy, gae_lambda) alongside the MuZero dials.
        if idx % 3 == 2:
            from pipelines.ppo_tuning import ppo_grid_configs

            ppo_configs = ppo_grid_configs(12)
            ppo_cfg = dict(ppo_configs[idx % len(ppo_configs)])
            ppo_cfg["iteration"] = iteration
            ppo_cfg["seed"] = seed
            ppo_cfg["train_seed"] = train_seed
            raw_cfg = ppo_cfg

        else:
            total_iters = grid_iterations[idx % len(grid_iterations)]
            h_dim = grid_hidden_dims[(idx // len(grid_iterations)) % len(grid_hidden_dims)]
            cons_w = round(0.3557 + 0.08 * (idx % 4), 4)
            cql_w = round(0.01357 + 0.025 * ((idx // 4) % 3), 5)
            pol_w = round(0.8137 + 0.04 * ((idx // 2) % 3), 4)
            val_w = round(0.9637 - 0.02 * (idx % 3), 4)
            rew_w = round(0.7057 + 0.03 * ((idx // 3) % 3), 4)
            k_step = grid_k_steps[idx % len(grid_k_steps)]

            p1 = max(50, int(0.50 * total_iters))
            p2 = max(25, int(0.30 * total_iters))
            p3 = max(4, int(0.20 * total_iters / 20))

            raw_cfg = {
                "name": f"Grid-Cand-{iteration}-I{total_iters}-H{h_dim}",
                "iteration": iteration,
                "seed": seed,
                "train_seed": train_seed,
                "total_iterations": total_iters,
                "phase1_steps": p1,
                "phase2_steps": p2,
                "phase3_epochs": p3,
                "sims": 20,
                "consistency_weight": cons_w,
                "policy_weight": pol_w,
                "value_weight": val_w,
                "reward_weight": rew_w,
                "cql_weight": cql_w,
                "k_steps": k_step,
                "hidden_dim": h_dim,
            }

    elif strategy == "random":
        rng = random.Random(seed)
        total_iters = rng.choice([500, 1000, 1500, 2500])
        h_dim = rng.choice([32, 64])
        cons_w = rng.choice([0.3357, 0.4357, 0.5357])
        cql_w = rng.choice([0.01357, 0.02357, 0.04357, 0.06357])
        pol_w = rng.choice([0.8137, 0.8637, 0.9037])
        val_w = rng.choice([0.9437, 0.9637, 0.9837])
        rew_w = rng.choice([0.6857, 0.7357, 0.7657])
        k_step = rng.choice([3, 4, 5])
        sims = rng.choice([12, 16, 20, 24])

        p1 = max(50, int(0.50 * total_iters))
        p2 = max(25, int(0.30 * total_iters))
        p3 = max(4, int(0.20 * total_iters / 20))

        raw_cfg = {
            "name": f"Random-Cand-{iteration}-I{total_iters}-H{h_dim}",
            "iteration": iteration,
            "seed": seed,
            "train_seed": train_seed,
            "total_iterations": total_iters,
            "phase1_steps": p1,
            "phase2_steps": p2,
            "phase3_epochs": p3,
            "sims": sims,
            "consistency_weight": cons_w,
            "policy_weight": pol_w,
            "value_weight": val_w,
            "reward_weight": rew_w,
            "cql_weight": cql_w,
            "k_steps": k_step,
            "hidden_dim": h_dim,
        }

    elif strategy == "gradient":
        # Opposing gradient strategy:
        # Tests the 5 weights going against each other in different directions.
        # Every trial 2k - 1 (primary direction) is directly paired with trial 2k (opposite direction).
        pair_idx = (iteration - 1) // 2
        in_pair = (iteration - 1) % 2
        is_pos = (in_pair == 0)
        sign = 1.0 if is_pos else -1.0
        opp_cand_id = f"Cand-{iteration + 1:03d}" if is_pos else f"Cand-{iteration - 1:03d}"

        # Distinct baseline centers for the 5 weights
        # In boost mode, centered around high representation fidelity
        c_cons = 0.4357
        c_pol = 0.8637
        c_val = 0.9537
        c_rew = 0.7257
        c_cql = 0.02857

        # Distinct step deltas for each weight
        d_cons = 0.0600
        d_pol = 0.0400
        d_val = 0.0250
        d_rew = 0.0350
        d_cql = 0.0125

        # Progressive scaling for deeper iterations
        scale = 1.0 + 0.15 * (pair_idx // 10)

        # 10 Antagonistic & Opposing Direction Patterns where weights go against each other:
        pat = pair_idx % 10

        if pat == 0:
            # Pair 1: Consistency vs Value (representation vs bootstrap)
            # Cand 1: +cons, -val | Cand 2: -cons, +val
            cons_w = c_cons + sign * d_cons * scale
            val_w = c_val - sign * d_val * scale
            pol_w = c_pol
            rew_w = c_rew
            cql_w = c_cql
            desc = "+cons / -val" if is_pos else "-cons / +val"
            sched_dirs = "consistency:up,value:down" if is_pos else "consistency:down,value:up"

        elif pat == 1:
            # Pair 2: Policy vs Reward (policy distillation vs 1-step reward)
            # Cand 3: +pol, -rew | Cand 4: -pol, +rew
            cons_w = c_cons
            val_w = c_val
            pol_w = c_pol + sign * d_pol * scale
            rew_w = c_rew - sign * d_rew * scale
            cql_w = c_cql
            desc = "+pol / -rew" if is_pos else "-pol / +rew"
            sched_dirs = "policy:up,reward:down" if is_pos else "policy:down,reward:up"

        elif pat == 2:
            # Pair 3: Consistency vs Policy (latent geometry vs policy target)
            # Cand 5: +cons, -pol | Cand 6: -cons, +pol
            cons_w = c_cons + sign * d_cons * scale
            val_w = c_val
            pol_w = c_pol - sign * d_pol * scale
            rew_w = c_rew
            cql_w = c_cql
            desc = "+cons / -pol" if is_pos else "-cons / +pol"
            sched_dirs = "consistency:up,policy:down" if is_pos else "consistency:down,policy:up"

        elif pat == 3:
            # Pair 4: Value vs Reward (long-term return vs transition reward)
            # Cand 7: +val, -rew | Cand 8: -val, +rew
            cons_w = c_cons
            val_w = c_val + sign * d_val * scale
            pol_w = c_pol
            rew_w = c_rew - sign * d_rew * scale
            cql_w = c_cql
            desc = "+val / -rew" if is_pos else "-val / +rew"
            sched_dirs = "value:up,reward:down" if is_pos else "value:down,reward:up"

        elif pat == 4:
            # Pair 5: CQL vs Policy (conservative penalty vs policy entropy)
            # Cand 9: +cql, -pol | Cand 10: -cql, +pol
            cons_w = c_cons
            val_w = c_val
            pol_w = c_pol - sign * d_pol * scale
            rew_w = c_rew
            cql_w = c_cql + sign * d_cql * scale
            desc = "+cql / -pol" if is_pos else "-cql / +pol"
            sched_dirs = "cql:up,policy:down" if is_pos else "cql:down,policy:up"

        elif pat == 5:
            # Pair 6: Multi-Weight Vector A (all 5 weights going against each other)
            # Cand 11: +cons, -pol, +val, -rew, +cql | Cand 12: opposite
            cons_w = c_cons + sign * d_cons * scale
            pol_w = c_pol - sign * d_pol * scale
            val_w = c_val + sign * d_val * scale
            rew_w = c_rew - sign * d_rew * scale
            cql_w = c_cql + sign * d_cql * scale
            desc = "+cons/-pol/+val/-rew/+cql" if is_pos else "-cons/+pol/-val/+rew/-cql"
            sched_dirs = "consistency:up,policy:down,value:up,reward:down,cql:up" if is_pos else "consistency:down,policy:up,value:down,reward:up,cql:down"

        elif pat == 6:
            # Pair 7: Multi-Weight Vector B
            # Cand 13: -cons, +pol, +val, -rew, -cql | Cand 14: opposite
            cons_w = c_cons - sign * d_cons * scale
            pol_w = c_pol + sign * d_pol * scale
            val_w = c_val + sign * d_val * scale
            rew_w = c_rew - sign * d_rew * scale
            cql_w = c_cql - sign * d_cql * scale
            desc = "-cons/+pol/+val/-rew/-cql" if is_pos else "+cons/-pol/-val/+rew/+cql"
            sched_dirs = "consistency:down,policy:up,value:up,reward:down,cql:down" if is_pos else "consistency:up,policy:down,value:down,reward:up,cql:up"

        elif pat == 7:
            # Pair 8: Multi-Weight Vector C
            # Cand 15: +cons, +pol, -val, -rew, +cql | Cand 16: opposite
            cons_w = c_cons + sign * d_cons * scale
            pol_w = c_pol + sign * d_pol * scale
            val_w = c_val - sign * d_val * scale
            rew_w = c_rew - sign * d_rew * scale
            cql_w = c_cql + sign * d_cql * scale
            desc = "+cons/+pol/-val/-rew/+cql" if is_pos else "-cons/-pol/+val/+rew/-cql"
            sched_dirs = "consistency:up,policy:up,value:down,reward:down,cql:up" if is_pos else "consistency:down,policy:down,value:up,reward:up,cql:down"

        elif pat == 8:
            # Pair 9: Multi-Weight Vector D
            # Cand 17: -cons, -pol, +val, +rew, -cql | Cand 18: opposite
            cons_w = c_cons - sign * d_cons * scale
            pol_w = c_pol - sign * d_pol * scale
            val_w = c_val + sign * d_val * scale
            rew_w = c_rew + sign * d_rew * scale
            cql_w = c_cql - sign * d_cql * scale
            desc = "-cons/-pol/+val/+rew/-cql" if is_pos else "+cons/+pol/-val/-rew/+cql"
            sched_dirs = "consistency:down,policy:down,value:up,reward:up,cql:down" if is_pos else "consistency:up,policy:up,value:down,reward:down,cql:up"

        else:
            # Pair 10: Multi-Weight Vector E
            # Cand 19: +cons, -pol, -val, +rew, +cql | Cand 20: opposite
            cons_w = c_cons + sign * d_cons * scale
            pol_w = c_pol - sign * d_pol * scale
            val_w = c_val - sign * d_val * scale
            rew_w = c_rew + sign * d_rew * scale
            cql_w = c_cql + sign * d_cql * scale
            desc = "+cons/-pol/-val/+rew/+cql" if is_pos else "-cons/+pol/+val/-rew/-cql"
            sched_dirs = "consistency:up,policy:down,value:down,reward:up,cql:up" if is_pos else "consistency:down,policy:up,value:up,reward:down,cql:down"

        total_iters = base_iterations
        p1 = max(50, int(0.50 * total_iters))
        p2 = max(25, int(0.30 * total_iters))
        p3 = max(4, int(0.20 * total_iters / 20))
        sims = 20

        cons_w = max(0.12, min(0.85, cons_w))
        pol_w = max(0.18, min(0.95, pol_w))
        val_w = max(0.20, min(0.99, val_w))
        rew_w = max(0.14, min(0.88, rew_w))
        cql_w = max(0.005, min(0.15, cql_w))

        raw_cfg = {
            "name": f"Gradient-Pair-{pair_idx+1:02d}-{'Pos' if is_pos else 'Neg'} ({desc})",
            "iteration": iteration,
            "seed": seed,
            "train_seed": train_seed,
            "total_iterations": total_iters,
            "phase1_steps": p1,
            "phase2_steps": p2,
            "phase3_epochs": p3,
            "sims": sims,
            "consistency_weight": round(cons_w, 4),
            "policy_weight": round(pol_w, 4),
            "value_weight": round(val_w, 4),
            "reward_weight": round(rew_w, 4),
            "cql_weight": round(cql_w, 5),
            "k_steps": 5,
            "hidden_dim": base_hidden_dim,
            "schedule_directions": sched_dirs,
            "gradient_info": {
                "pair_index": pair_idx + 1,
                "is_positive": is_pos,
                "vector_desc": desc,
                "opposite_candidate_id": opp_cand_id,
                "schedule_directions": sched_dirs,
            },
        }

    else:
        # Adaptive feedback strategy
        total_iters = base_iterations
        p1 = max(100, int(0.50 * total_iters))
        p2 = max(50, int(0.30 * total_iters))
        p3 = max(6, int(0.20 * total_iters / 20))
        sims = 20
        cons_w = round(0.4157 + 0.03 * (iteration % 4), 4)
        pol_w = round(0.8357 + 0.02 * (iteration % 3), 4)
        val_w = round(0.9637 - 0.015 * (iteration % 3), 4)
        rew_w = round(0.7157 + 0.025 * (iteration % 3), 4)
        cql_w = round(0.02357 + 0.015 * (iteration % 4), 5)
        h_dim = base_hidden_dim

        if prev_metrics:
            margin = prev_metrics.get("margin", 0.0)
            p1_loss = prev_metrics.get("train", {}).get("phase1", {}).get("total", 50.0)
            if p1_loss < 40.0:
                p2 = min(150, int(p2 * 1.3))
                sims = min(24, sims + 4)
            if margin < 0:
                total_iters = min(3000, int(total_iters * 1.5))
                p3 = min(14, p3 + 2)
                cql_w = min(0.08, cql_w + 0.01)

        raw_cfg = {
            "name": f"Adaptive-Cand-{iteration}",
            "iteration": iteration,
            "seed": seed,
            "train_seed": train_seed,
            "total_iterations": total_iters,
            "phase1_steps": p1,
            "phase2_steps": p2,
            "phase3_epochs": p3,
            "sims": sims,
            "consistency_weight": cons_w,
            "policy_weight": pol_w,
            "value_weight": val_w,
            "reward_weight": rew_w,
            "cql_weight": cql_w,
            "k_steps": 5,
            "hidden_dim": h_dim,
        }

    # Defensive guarantee: enforce distinctness rule on all generated weights
    validated = validate_loss_weights_rule(
        consistency_weight=raw_cfg.get("consistency_weight", 0.4357),
        policy_weight=raw_cfg.get("policy_weight", 0.8637),
        value_weight=raw_cfg.get("value_weight", 0.9637),
        reward_weight=raw_cfg.get("reward_weight", 0.7357),
        cql_weight=raw_cfg.get("cql_weight", 0.02357),
        auto_adjust=True,
    )
    raw_cfg["consistency_weight"] = validated["consistency-weight"]
    raw_cfg["policy_weight"] = validated["policy-weight"]
    raw_cfg["value_weight"] = validated["value-weight"]
    raw_cfg["reward_weight"] = validated["reward-weight"]
    raw_cfg["cql_weight"] = validated["cql-weight"]
    return raw_cfg


def run_online_selfplay_data_generation(
    candidate_pkg: str,
    snapshots_dir: str,
    champion_pkg: str,
    n_games: int = 8,
    base_seed: int = 10101,
    replays_dir: Optional[str] = REPLAYS,
    min_replay_score: float = 80000.0,
    n_workers: int = 8,
) -> Dict[str, Any]:
    snapshot_files = [
        os.path.join(snapshots_dir, f)
        for f in os.listdir(snapshots_dir)
        if f.endswith(".zip") or f.endswith(".pt")
    ] if os.path.isdir(snapshots_dir) else []

    opponents = [champion_pkg]
    for snap in snapshot_files:
        if snap.endswith(".zip"):
            opponents.append(snap)

    print("\n" + "=" * 90)
    print(f"  [Mode A] ONLINE SELF-PLAY DATA GENERATION ({n_games} games, {n_workers} parallel workers)")
    print(f"  Dirichlet noise Dir(α=0.25) + Dynamic Temperature Schedule (τ=1.0 for day≤10 → τ=0.0)")
    print("=" * 90)

    tasks: List[Dict[str, Any]] = []
    for i in range(n_games):
        opp = opponents[i % len(opponents)]
        match_seed = base_seed + i + 1
        cand_is_p0 = (i % 2 == 0)
        tasks.append({
            "agent_a": candidate_pkg,
            "agent_b": opp,
            "seed": match_seed,
            "a_is_p0": cand_is_p0,
            "episode_steps": 720,
            "save_replay_dir": replays_dir,
            "min_replay_score": min_replay_score,
            "env_a": {"MUZERO_ONLINE_EXPLORATION": "1", "MUZERO_DIRICHLET_ALPHA": "0.25"},
        })

    if n_workers > 1 and len(tasks) > 1:
        with mp.Pool(processes=min(n_workers, len(tasks))) as pool:
            matches = pool.map(eval_match_worker, tasks)
    else:
        matches = [eval_match_worker(t) for t in tasks]

    for i, res in enumerate(matches):
        winner_str = "CAND" if res["delta_a"] > 0 else ("OPP" if res["delta_a"] < 0 else "TIE")
        seat_str = res.get("a_seat", "P0" if (i % 2 == 0) else "P1")
        opp_name = os.path.basename(tasks[i]["agent_b"])
        print(f"  Match {i+1:>2}: {seat_str} vs {opp_name:<25} | Cand: ${res['score_a']:>11,.0f} | Opp: ${res['score_b']:>11,.0f} | Δ: {res['delta_a']:>+11,.0f} | Winner: {winner_str}")

    return {
        "games_played": len(matches),
        "matches": matches,
    }


def execute_candidate_trial(
    candidate_id: str,
    config: Dict[str, Any],
    mode: str,
    s_champion: float,
    champion_delta_threshold: float = 6000.0,
    eval_episodes: int = 20,
    n_workers: int = 8,
    save_replays: bool = True,
    py: str = sys.executable,
    selfplay_games: int = 8,
    loss_schedule: str = "separate",
) -> Dict[str, Any]:
    trial_start = time.time()
    seed = config["seed"]
    train_seed = config["train_seed"]
    total_iterations = config.get("total_iterations", config["phase1_steps"] + config["phase2_steps"])

    print("\n" + "#" * 90)
    print(f"  [TRIAL] CANDIDATE: {candidate_id} | Strategy: {config.get('name', 'Custom')}")
    print(f"  Seed: {seed} | Mode: {mode.upper()} | Target: S_champ (${s_champion:,.0f}) + ${champion_delta_threshold:,.0f} coins")
    grad_info = config.get("gradient_info")
    if grad_info:
        opp_note = f" (Opposite Tested in {grad_info['opposite_candidate_id']})" if grad_info.get("opposite_candidate_id") else ""
        print(f"  Gradient Test : {grad_info.get('vector_desc', 'Custom')}{opp_note}")
        if grad_info.get("schedule_directions"):
            print(f"  Schedule Dirs : {grad_info['schedule_directions']}")
    print(
        f"  Weights: Hidden={config.get('hidden_dim', 32)} | K={config.get('k_steps', 5)} | "
        f"λ_cons={config.get('consistency_weight', 0.4357):.4f} | λ_pol={config.get('policy_weight', 0.8637):.4f} | "
        f"λ_val={config.get('value_weight', 0.9637):.4f} | λ_rew={config.get('reward_weight', 0.7357):.4f} | "
        f"CQL={config.get('cql_weight', 0.02357):.4f}"
    )
    print(f"  Steps: P1={config['phase1_steps']} | P2={config['phase2_steps']} (sims={config['sims']}) | P3={config['phase3_epochs']} ep")
    print("#" * 90)

    # PPO microcontroller macrocontroller trials dispatch to the PPO tuning module.
    if config.get("controller") == "ppo":
        from pipelines.ppo_tuning import execute_ppo_trial

        return execute_ppo_trial(
            config,
            s_champion=s_champion,
            champion_delta_threshold=champion_delta_threshold,
            eval_episodes=eval_episodes,
            n_workers=n_workers,
            save_replays=save_replays,
            seed=seed,
        )

    selfplay_report: Optional[Dict[str, Any]] = None
    if mode in ("online", "hybrid"):
        if not os.path.isfile(CANDIDATE_ZIP):
            try:
                build_hybrid_package(ckpt_path=CHAMPION_CKPT, out_zip=CANDIDATE_ZIP, out_py=CANDIDATE_PY)
            except Exception as _be:
                raise
        selfplay_report = run_online_selfplay_data_generation(
            candidate_pkg=CANDIDATE_ZIP,
            snapshots_dir=SNAPSHOTS,
            champion_pkg=CHAMPION_ZIP,
            n_games=selfplay_games,
            base_seed=seed,
            replays_dir=REPLAYS if save_replays else None,
            min_replay_score=80000.0,
            n_workers=n_workers,
        )

    train_cmd = [
        py,
        "train_muzero.py",
        "--phase1-steps", str(config["phase1_steps"]),
        "--phase2-steps", str(config["phase2_steps"]),
        "--phase3-epochs", str(config["phase3_epochs"]),
        "--batch", "64",
        "--sims", str(config["sims"]),
        "--consistency-weight", str(config["consistency_weight"]),
        "--policy-weight", str(config.get("policy_weight", 0.8637)),
        "--value-weight", str(config.get("value_weight", 0.9637)),
        "--reward-weight", str(config.get("reward_weight", 0.7357)),
        "--cql-weight", str(config.get("cql_weight", 0.02357)),
        "--loss-schedule", str(loss_schedule),
        "--k-steps", str(config.get("k_steps", 5)),
        "--hidden-dim", str(config.get("hidden_dim", 32)),
        "--seed", str(train_seed),
        "--summary-out", TRAIN_SUMMARY,
        "--out", CKPT,
    ]
    # Phase 4 continuous re-analysis: actual LoRA adapters (r=16), distinct from
    # --loss-schedule low_rank_adaptation which only schedules loss weights.
    if mode in ("hybrid", "online"):
        phase4_steps = int(config.get("phase4_steps", max(10, config.get("phase2_steps", 30) // 2)))
        lora_rank = int(config.get("lora_rank", 16))
        train_cmd.extend([
            "--use-lora",
            "--phase4-steps", str(phase4_steps),
            "--lora-rank", str(lora_rank),
            "--lora-alpha", str(float(config.get("lora_alpha", 16.0))),
        ])
    if config.get("schedule_directions"):
        train_cmd.extend(["--schedule-directions", str(config["schedule_directions"])])
    print("+ " + " ".join(train_cmd), flush=True)
    ret = subprocess.call(train_cmd, cwd=HERE)
    if ret != 0:
        raise RuntimeError(f"train_muzero.py failed with exit code {ret}")

    train_summary = {}
    if os.path.isfile(TRAIN_SUMMARY):
        try:
            with open(TRAIN_SUMMARY) as f:
                train_summary = json.load(f)
        except Exception:
            pass

    build_hybrid_package(ckpt_path=CKPT, out_zip=CANDIDATE_ZIP, out_py=CANDIDATE_PY)

    print(f"\n[Evaluation Suite] Running E = {eval_episodes} deterministic episodes against benchmark distribution...")
    baselines = find_baselines(BASELINES)
    eval_report = run_evaluation_episodes(
        candidate_pkg=CANDIDATE_ZIP,
        baseline_paths=baselines,
        n_episodes=eval_episodes,
        base_seed=seed + 2000,
        n_workers=n_workers,
        save_replays=save_replays,
    )

    r_candidate = float(eval_report["avg_score"])
    win_pct = float(eval_report.get("win_pct", 0.0))
    avg_delta = float(eval_report.get("avg_delta", 0.0))
    margin = r_candidate - s_champion

    # Multimodal Promotion Evaluation:
    # 1. Absolute Threshold: Exceeds historical champion score target
    promoted_by_score = (r_candidate >= s_champion + champion_delta_threshold)
    # 2. Baseline Dominance: High win rate (>= 75.0%) and positive margin vs baseline pool
    promoted_by_winrate = (win_pct >= 75.0 and avg_delta > 0.0 and r_candidate >= 95000.0)
    # 3. Direct Head-to-Head: Outperformed the Champion in direct self-play matches
    promoted_by_h2h = False
    cand_sp_wins = 0
    opp_sp_wins = 0
    cand_sp_delta = 0.0
    if selfplay_report and selfplay_report.get("matches"):
        sp_matches = selfplay_report["matches"]
        cand_sp_wins = sum(1 for m in sp_matches if m.get("winner") == "A")
        opp_sp_wins = sum(1 for m in sp_matches if m.get("winner") == "B")
        cand_sp_delta = sum(m.get("delta_a", 0.0) for m in sp_matches)
        if (cand_sp_wins > opp_sp_wins or cand_sp_delta > 0.0) and win_pct >= 50.0 and r_candidate >= 95000.0:
            promoted_by_h2h = True

    is_promoted = bool(promoted_by_score or promoted_by_winrate or promoted_by_h2h)

    reasons = []
    if promoted_by_score:
        reasons.append(f"Score >= Target (${r_candidate:,.0f} >= ${s_champion + champion_delta_threshold:,.0f})")
    if promoted_by_winrate:
        reasons.append(f"Baseline Dominance ({win_pct:.1f}% Win Rate, Δ: {avg_delta:+,.0f})")
    if promoted_by_h2h:
        reasons.append(f"Head-to-Head Victory vs Champ ({cand_sp_wins}W-{opp_sp_wins}L, Net Δ: {cand_sp_delta:+,.0f})")

    # Phase 5 automated gate: head-to-head vs benchmark 2033 ELO champion
    gate_passed: Optional[bool] = None
    try:
        from evaluation.benchmarks import gate_candidate_vs_2033

        gate_report = gate_candidate_vs_2033(
            CANDIDATE_ZIP,
            games=2,
            n_workers=min(2, n_workers),
            require_win_rate=0.5,
            quiet=False,
        )
        gate_passed = bool(gate_report.get("gate", {}).get("passed"))
        if gate_passed:
            reasons.append("2033 Gate PASS")
        else:
            reasons.append("2033 Gate FAIL")
            is_promoted = False
    except FileNotFoundError as e:
        print(f"[gate] 2033 champion missing — skipping gate ({e})")
        gate_passed = None
    except Exception as e:
        print(f"[gate] 2033 gate error — blocking promotion ({e})")
        gate_passed = False
        is_promoted = False

    status_str = "PROMOTED" if is_promoted else "REJECTED"
    reason_str = " | ".join(reasons) if reasons else "Below promotion threshold"

    print(f"\n[{candidate_id}] Seed: {seed} | Iterations: {total_iterations} | Score: {r_candidate:,.0f} coins | Margin vs Champ: {margin:+,.0f} coins | Win%: {win_pct:.1f}% | Status: [{status_str}]\n")

    if is_promoted:
        print("★" * 90)
        print(f"  ★ PROMOTED TO NEW CHAMPION! Score: ${r_candidate:,.0f} coins")
        print(f"  Reason: {reason_str}")
        print(f"  Atomic update to {CHAMPION_CKPT}")
        print("★" * 90)

        cand_net, _ = load_muzero_checkpoint(path=CKPT)
        save_muzero_checkpoint(
            cand_net,
            CHAMPION_CKPT,
            champion_score=r_candidate,
            meta=config,
            atomic=True,
        )

        build_hybrid_package(ckpt_path=CHAMPION_CKPT, out_zip=CHAMPION_ZIP, out_py=CHAMPION_PY)
        build_hybrid_package(ckpt_path=CHAMPION_CKPT)

        snap_path = os.path.join(SNAPSHOTS, f"champion_{candidate_id}_{int(r_candidate)}.pt")
        save_muzero_checkpoint(cand_net, snap_path, champion_score=r_candidate, meta=config, atomic=True)

        log_entry = (
            f"[{datetime.datetime.now().isoformat()}] PROMOTED {candidate_id} | "
            f"Seed: {seed} | Iterations: {total_iterations} | Score: ${r_candidate:,.0f} | "
            f"Reason: {reason_str} | Margin: +${margin:,.0f} coins | Win%: {eval_report['win_pct']:.1f}% | "
            f"Weights: hidden={config.get('hidden_dim', 32)}, cons_w={config.get('consistency_weight', 0.4357)}, cql={config.get('cql_weight', 0.0236)}, K={config.get('k_steps', 5)}\n"
        )
        with open(CHAMPION_HISTORY_LOG, "a", encoding="utf-8") as f:
            f.write(log_entry)

        # Automatic disk space guard: prune older self-play replays keeping folder under workspace budget (SCRATCH: 10GB, READ+WRITE: 24GB)
        try:
            target_budget = get_default_replay_budget_gb(REPLAYS)
            pruned_count, freed_gb = prune_stale_selfplay_replays(
                replays_dir=REPLAYS,
                max_dir_size_gb=target_budget,
            )
            if pruned_count > 0:
                print(f"  [Disk Guard] Pruned {pruned_count} older self-play replays by timestamp (reclaimed {freed_gb:.2f} GB, folder <= {target_budget:.1f} GB).")
        except Exception as e:
            print(f"  [Disk Guard] Warning: Failed to prune replays: {e}")
    else:
        pop_path = os.path.join(POPULATION, f"candidate_{candidate_id}_{int(r_candidate)}.pt")
        shutil.copy2(CKPT, pop_path)
        print(f"  Candidate retained in secondary population pool at {pop_path}")

    trial_elapsed = round(time.time() - trial_start, 2)
    record = {
        "candidate_id": candidate_id,
        "status": status_str,
        "is_promoted": is_promoted,
        "score": r_candidate,
        "margin": margin,
        "s_champion_prior": s_champion,
        "s_champion_new": r_candidate if is_promoted else s_champion,
        "config": config,
        "eval_report": {
            "n_episodes": eval_report["n_episodes"],
            "avg_score": r_candidate,
            "win_pct": eval_report["win_pct"],
            "wins": eval_report["wins"],
            "losses": eval_report["losses"],
            "ties": eval_report["ties"],
            "elapsed_seconds": eval_report["elapsed_seconds"],
        },
        "train_summary": train_summary,
        "elapsed_seconds": trial_elapsed,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    status_payload = {
        "last_candidate": candidate_id,
        "status": status_str,
        "score": r_candidate,
        "margin": margin,
        "s_champion": record["s_champion_new"],
        "is_promoted": is_promoted,
        "config": config,
        "timestamp": record["timestamp"],
    }
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(status_payload, f, indent=2)

    return record


def append_history(record: Dict[str, Any]) -> None:
    history = []
    if os.path.isfile(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(record)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 4 continuous self-improving RL loop (curriculum) & hyperparameter engine"
    )
    parser.add_argument("--mode", choices=["online", "offline", "hybrid"], default="offline", help="Tuning mode: online (self-play search) or offline (dataset replay & reanalyse)")
    parser.add_argument("--iterations", type=int, default=20, help="Number of candidate trials (-1 for continuous, default: 20)")
    parser.add_argument("--continuous", action="store_true", help="Run indefinitely")
    parser.add_argument("--search-strategy", choices=["grid", "random", "combinations", "adaptive", "gradient"], default="gradient", help="Hyperparameter search matrix strategy (default: gradient)")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Deterministic evaluation episodes per candidate (default: 20)")
    parser.add_argument("--champion-delta-threshold", type=float, default=6000.0, help="Margin over previous champion to promote (default: 6000.0 coins)")
    parser.add_argument("--base-seed", type=int, default=10101, help="Starting RNG seed")
    parser.add_argument("--stride", type=int, default=37, help="Seed stride per candidate")
    parser.add_argument("--eval-workers", type=int, default=8, help="Parallel evaluation worker processes (default: 8)")
    parser.add_argument("--base-iterations", type=int, default=1000, help="Base training iteration budget (default: 1000)")
    parser.add_argument("--selfplay-games", type=int, default=8, help="Online self-play games generated per trial (default: 8)")
    parser.add_argument(
        "--loss-schedule",
        choices=["separate", "low_rank_adaptation", "lora", "adaptive_moment_estimation", "adam", "none"],
        default="low_rank_adaptation",
        help="Loss weighting schedule: low_rank_adaptation (LoRA / standard), adaptive_moment_estimation (Adam / boost), separate, or none",
    )
    parser.add_argument("--no-save-replays", action="store_true", help="Disable harvesting tournament replays")
    args = parser.parse_args()


    # Normalize schedule aliases: lora -> low_rank_adaptation, adam -> adaptive_moment_estimation
    if args.loss_schedule == "lora":
        args.loss_schedule = "low_rank_adaptation"
    elif args.loss_schedule == "adam":
        args.loss_schedule = "adaptive_moment_estimation"

    os.makedirs(ART, exist_ok=True)
    os.makedirs(REPLAYS, exist_ok=True)
    os.makedirs(SNAPSHOTS, exist_ok=True)
    os.makedirs(POPULATION, exist_ok=True)

    s_champion, _ = setup_baseline_champion(
        base_seed=args.base_seed,
        n_episodes=10,
        n_workers=args.eval_workers,
        save_replays=not args.no_save_replays,
    )

    max_trials = None if args.continuous or args.iterations < 0 else args.iterations
    print("=" * 90)
    print("  AUTONOMOUS MUZERO SELF-IMPROVEMENT PIPELINE INITIALIZED")
    print(f"  Mode:                       {args.mode.upper()}")
    print(f"  Search Strategy:            {args.search_strategy.upper()}")
    print(f"  Target Trials:              {'Continuous (∞)' if max_trials is None else max_trials}")
    print(f"  Base Training Iterations:   {args.base_iterations} iters")
    print(f"  Current Champion Score:     ${s_champion:,.0f} coins")
    print(f"  Promotion Threshold Target: +${args.champion_delta_threshold:,.0f} coins (>= ${s_champion + args.champion_delta_threshold:,.0f})")
    print(f"  Evaluation Suite:           E = {args.eval_episodes} deterministic episodes (parallel workers: {args.eval_workers})")
    print(f"  Loss Schedule:              {args.loss_schedule.upper()} (Decoupled 0.1357 -> 0.9637)")
    print("=" * 90)

    trial_idx = 1
    promotions = 0
    prev_record: Optional[Dict[str, Any]] = None

    try:
        while True:
            if max_trials is not None and trial_idx > max_trials:
                break

            cand_id = f"Cand-{trial_idx:03d}"
            config = generate_candidate_hyperparams(
                iteration=trial_idx,
                strategy=args.search_strategy,
                base_iterations=args.base_iterations,
                base_seed=args.base_seed,
                stride=args.stride,
                prev_metrics=prev_record,
            )

            record = execute_candidate_trial(
                candidate_id=cand_id,
                config=config,
                mode=args.mode,
                s_champion=s_champion,
                champion_delta_threshold=args.champion_delta_threshold,
                eval_episodes=args.eval_episodes,
                n_workers=args.eval_workers,
                save_replays=not args.no_save_replays,
                selfplay_games=args.selfplay_games,
                loss_schedule=args.loss_schedule,
            )

            append_history(record)

            # Opposing gradient pair comparison summary
            if prev_record and config.get("gradient_info") and prev_record.get("config", {}).get("gradient_info"):
                p_curr = config["gradient_info"].get("pair_index")
                p_prev = prev_record["config"]["gradient_info"].get("pair_index")
                if p_curr == p_prev:
                    s_prev = prev_record["score"]
                    s_curr = record["score"]
                    prev_id = prev_record["candidate_id"]
                    curr_id = record["candidate_id"]
                    delta_p = s_curr - s_prev
                    favored = curr_id if delta_p > 0 else (prev_id if delta_p < 0 else "TIE")
                    print("\n" + "=" * 90)
                    print(f"  [OPPOSING GRADIENT PAIR COMPARISON] Pair #{p_curr}")
                    print(f"  - {prev_id} ({prev_record['config']['gradient_info'].get('vector_desc')}): ${s_prev:,.0f} coins")
                    print(f"  - {curr_id} ({config['gradient_info'].get('vector_desc')}): ${s_curr:,.0f} coins")
                    print(f"  -> Outcome: {favored} favored by |Δ| = ${abs(delta_p):,.0f} coins")
                    print("=" * 90 + "\n")

            prev_record = record

            if record["is_promoted"]:
                promotions += 1
                s_champion = record["s_champion_new"]

            trial_idx += 1

    except KeyboardInterrupt:
        print("\n[Autonomous Loop Interrupted by User]")

    print(f"\nAutonomous pipeline completed {trial_idx - 1} trials. Total promotions: {promotions} (Final Champion: ${s_champion:,.0f} coins)")
    return 0


# Canonical Phase 4 curriculum entry name (REFACTOR_PLAN continuous self-play loop).
# Named run_* so it does not shadow the ``phase4_continuous_loop`` module in
# ``pipelines.phases.__init__``.
run_phase4_continuous_loop = main
# Back-compat alias (prefer importing the module, then ``.main`` / ``.run_phase4_continuous_loop``).
phase4_continuous_loop = main


if __name__ == "__main__":
    sys.exit(main())
