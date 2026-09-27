#!/usr/bin/env python3
"""
scripts/system_concurrency_benchmark.py
=============================================================================
System Benchmarking & Hardware Concurrency Profiler for Kaggriculture MuZero
=============================================================================
Measures:
  - Multi-worker throughput across 4, 8, 10, and 12 parallel worker processes
  - Real-time CPU per-core utilization (8 Performance cores + 4 Efficiency cores)
  - Process tree RSS memory consumption and system RAM headroom
  - Scaled 3-Phase training latency and convergence dynamics
=============================================================================
"""
import os
import sys
import time
import json
import threading
import multiprocessing as mp
from typing import List, Dict, Any, Tuple

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import psutil
import torch

from evaluation.harness import eval_match_worker
from pipelines.phases.phase1_bootstrap import phase1_analytical_bootstrap
from pipelines.phases.phase2_distill import phase2_mcts_distill
from pipelines.phases.phase3_bc import (
    discover_expert_replays,
    build_expert_trajectory_pool,
    phase3_expert_behavior_cloning,
)
from muzero import MuZeroNetwork, create_target_network

CAND_ZIP = os.path.join(HERE, "artifacts", "candidate_submission.zip")
CHAMP_ZIP = os.path.join(HERE, "artifacts", "champion_submission.zip")
REPLAYS = os.path.join(HERE, "replays")


class SystemMonitor:
    """Background sampler for CPU and Memory telemetry."""
    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self._running = False
        self._thread = None
        self.cpu_samples: List[float] = []
        self.percpu_samples: List[List[float]] = []
        self.ram_samples: List[float] = []
        self.rss_samples: List[float] = []
        self.p = psutil.Process(os.getpid())

    def _sample(self):
        # Initial call to prime psutil cpu_percent
        psutil.cpu_percent(interval=None, percpu=True)
        while self._running:
            try:
                per_cpu = psutil.cpu_percent(interval=None, percpu=True)
                total_cpu = psutil.cpu_percent(interval=None)
                vmem = psutil.virtual_memory()

                # Sum RSS of current process and all subprocess children
                total_rss = self.p.memory_info().rss
                for child in self.p.children(recursive=True):
                    try:
                        total_rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                self.cpu_samples.append(total_cpu)
                self.percpu_samples.append(per_cpu)
                self.ram_samples.append(vmem.percent)
                self.rss_samples.append(total_rss / (1024 * 1024))
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.cpu_samples.clear()
        self.percpu_samples.clear()
        self.ram_samples.clear()
        self.rss_samples.clear()
        self._running = True
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

        avg_cpu = sum(self.cpu_samples) / max(1, len(self.cpu_samples))
        peak_cpu = max(self.cpu_samples) if self.cpu_samples else 0.0
        peak_rss = max(self.rss_samples) if self.rss_samples else 0.0
        avg_rss = sum(self.rss_samples) / max(1, len(self.rss_samples))
        
        # Per core averages
        core_avgs = []
        if self.percpu_samples:
            num_cores = len(self.percpu_samples[0])
            for c in range(num_cores):
                core_avg = sum(sample[c] for sample in self.percpu_samples) / len(self.percpu_samples)
                core_avgs.append(round(core_avg, 1))

        return {
            "avg_cpu_pct": round(avg_cpu, 1),
            "peak_cpu_pct": round(peak_cpu, 1),
            "avg_rss_mb": round(avg_rss, 1),
            "peak_rss_mb": round(peak_rss, 1),
            "per_core_avg_pct": core_avgs,
        }


def run_worker_benchmark(workers_list: List[int], n_matches: int = 8) -> List[Dict[str, Any]]:
    print("\n" + "=" * 90)
    print(f"  SYSTEM TEST 1: PARALLEL EVALUATION WORKER SWEEP ({n_matches} matches)")
    print("=" * 90)
    
    results = []
    monitor = SystemMonitor(interval=0.2)

    for w in workers_list:
        tasks = []
        for i in range(n_matches):
            tasks.append({
                "agent_a": CAND_ZIP,
                "agent_b": CHAMP_ZIP,
                "seed": 20000 + i,
                "a_is_p0": (i % 2 == 0),
                "episode_steps": 720,
                "save_replay_dir": None,
                "min_replay_score": 80000.0,
            })

        print(f"\n  [Benchmarking] Workers = {w:>2} | Matches = {n_matches}...")
        monitor.start()
        t0 = time.time()
        with mp.Pool(processes=w) as pool:
            match_res = pool.map(eval_match_worker, tasks)
        elapsed = round(time.time() - t0, 2)
        telemetry = monitor.stop()

        throughput = round((n_matches / elapsed) * 60, 1)  # matches per min
        res_row = {
            "workers": w,
            "elapsed_sec": elapsed,
            "throughput_mpm": throughput,
            **telemetry
        }
        results.append(res_row)

        p_cores = telemetry["per_core_avg_pct"][:8]
        e_cores = telemetry["per_core_avg_pct"][8:] if len(telemetry["per_core_avg_pct"]) > 8 else []
        print(f"    Elapsed: {elapsed:>5.1f}s | Throughput: {throughput:>5.1f} games/min")
        print(f"    CPU Avg: {telemetry['avg_cpu_pct']:>5.1f}% (Peak: {telemetry['peak_cpu_pct']:>5.1f}%) | Peak RSS: {telemetry['peak_rss_mb']:>6.1f} MB")
        print(f"    P-Cores (1-8)  Avg: {p_cores}")
        if e_cores:
            print(f"    E-Cores (9-12) Avg: {e_cores}")

    return results


def run_parallel_selfplay_benchmark(n_games: int = 8, n_workers: int = 8) -> Dict[str, Any]:
    print("\n" + "=" * 90)
    print(f"  SYSTEM TEST 2: PARALLEL ONLINE SELF-PLAY DATA GENERATION ({n_games} games)")
    print(f"  Workers: {n_workers} | Exploration: Dir(α=0.25) + Dynamic Temp Schedule")
    print("=" * 90)

    monitor = SystemMonitor(interval=0.2)
    tasks = []
    for i in range(n_games):
        tasks.append({
            "agent_a": CAND_ZIP,
            "agent_b": CHAMP_ZIP,
            "seed": 30000 + i,
            "a_is_p0": (i % 2 == 0),
            "episode_steps": 720,
            "save_replay_dir": REPLAYS,
            "min_replay_score": 80000.0,
            "env_a": {"MUZERO_ONLINE_EXPLORATION": "1", "MUZERO_DIRICHLET_ALPHA": "0.25"},
        })

    monitor.start()
    t0 = time.time()
    with mp.Pool(processes=min(n_workers, len(tasks))) as pool:
        matches = pool.map(eval_match_worker, tasks)
    elapsed = round(time.time() - t0, 2)
    telemetry = monitor.stop()

    harvested = sum(1 for m in matches if m.get("saved_replay"))
    avg_cand = sum(m["score_a"] for m in matches) / len(matches)
    avg_opp = sum(m["score_b"] for m in matches) / len(matches)

    print(f"\n  [Parallel Self-Play Complete]")
    print(f"    Elapsed:      {elapsed}s for {n_games} games ({round(n_games / elapsed * 60, 1)} games/min)")
    print(f"    CPU Avg:      {telemetry['avg_cpu_pct']}% (Peak: {telemetry['peak_cpu_pct']}%)")
    print(f"    Peak RSS:     {telemetry['peak_rss_mb']} MB")
    print(f"    Avg Cand $:   ${avg_cand:,.0f} | Avg Opp $: ${avg_opp:,.0f}")
    print(f"    Replays >= $80k Saved: {harvested}/{n_games}")

    return {
        "n_games": n_games,
        "n_workers": n_workers,
        "elapsed_sec": elapsed,
        "harvested": harvested,
        **telemetry,
    }


def run_scaled_training_benchmark() -> Dict[str, Any]:
    print("\n" + "=" * 90)
    print("  SYSTEM TEST 3: SCALED 3-PHASE MUZERO TRAINING PIPELINE")
    print("  Steps: Phase 1 = 500 steps | Phase 2 = 100 steps (sims=20) | Phase 3 = 8 epochs")
    print("=" * 90)

    monitor = SystemMonitor(interval=0.2)
    model = MuZeroNetwork(hidden_dim=32, support_size=601)
    target_model = create_target_network(model)

    monitor.start()
    
    # Phase 1
    t0 = time.time()
    p1_metrics = phase1_analytical_bootstrap(
        model=model,
        steps=500,
        batch=64,
        lr=1e-3,
        cons_w=0.5,
        k_steps=5,
        target_model=target_model,
    )
    p1_time = round(time.time() - t0, 2)
    print(f"  [Phase 1] 500 steps completed in {p1_time}s | Final Total Loss: {p1_metrics.get('total', 0.0):.4f}")

    # Phase 2
    expert_files = discover_expert_replays(REPLAYS, min_score=80000.0, limit=50)
    traj_pool = build_expert_trajectory_pool(expert_files, k_steps=5)
    print(f"  [Pool Size] {len(traj_pool)} trajectory slices available for reanalyse")

    t0 = time.time()
    p2_metrics = phase2_mcts_distill(
        model=model,
        steps=100,
        batch=64,
        lr=5e-4,
        sims=20,
        cons_w=0.5,
        k_steps=5,
        trajectory_pool=traj_pool,
        target_model=target_model,
    )
    p2_time = round(time.time() - t0, 2)
    print(f"  [Phase 2] 100 MCTS steps (sims=20) completed in {p2_time}s | Final Loss: {p2_metrics.get('total', 0.0):.4f}")

    # Phase 3
    t0 = time.time()
    p3_metrics = phase3_expert_behavior_cloning(
        model=model,
        epochs=8,
        batch=64,
        lr=3e-4,
        cons_w=0.5,
        k_steps=5,
        prebuilt_pool=traj_pool,
        target_model=target_model,
    )
    p3_time = round(time.time() - t0, 2)
    print(f"  [Phase 3] 8 BC epochs completed in {p3_time}s | Final Epoch Loss: {p3_metrics.get('final_loss', 0.0):.4f}")

    telemetry = monitor.stop()
    total_time = round(p1_time + p2_time + p3_time, 2)

    print(f"\n  [Training Suite Summary]")
    print(f"    Total Wall Time: {total_time}s (P1: {p1_time}s, P2: {p2_time}s, P3: {p3_time}s)")
    print(f"    CPU Avg:         {telemetry['avg_cpu_pct']}% (Peak: {telemetry['peak_cpu_pct']}%)")
    print(f"    Peak RSS:        {telemetry['peak_rss_mb']} MB")

    return {
        "p1_time": p1_time,
        "p2_time": p2_time,
        "p3_time": p3_time,
        "total_time": total_time,
        **telemetry,
    }


def main():
    print("#" * 90)
    print("  KAGGICULTURE MUZERO HARDWARE CONCURRENCY & SYSTEM PROFILER")
    print("  Target Machine: Apple M2 Pro (12 Cores: 8 P-Cores, 4 E-Cores | 16 GB Unified RAM)")
    print("#" * 90)

    # 1. Sweep worker counts: 4 vs 8 vs 10 vs 12
    worker_sweep = run_worker_benchmark([4, 8, 10, 12], n_matches=8)

    # 2. Parallel selfplay generation
    selfplay_res = run_parallel_selfplay_benchmark(n_games=8, n_workers=8)

    # 3. Scaled training pipeline
    train_res = run_scaled_training_benchmark()

    # Save comprehensive report to artifacts
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "chip": "Apple M2 Pro",
            "total_cores": 12,
            "p_cores": 8,
            "e_cores": 4,
            "memory_gb": 16,
        },
        "worker_scaling": worker_sweep,
        "selfplay_benchmark": selfplay_res,
        "training_benchmark": train_res,
    }
    out_path = os.path.join(HERE, "artifacts", "hardware_concurrency_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 90)
    print(f"  BENCHMARK REPORT SAVED TO: {out_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
