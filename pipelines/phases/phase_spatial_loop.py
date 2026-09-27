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

"""Spatial continuous RL loop: self-play → train → eval → promote champion."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Optional

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from muzero import (
    DEFAULT_SPATIAL_CHECKPOINT,
    KaggricultureMuZeroChassis,
    load_spatial_checkpoint,
    promote_spatial_champion,
    save_spatial_checkpoint,
)
from pipelines.replay_manager import get_default_replay_budget_gb, prune_stale_selfplay_replays

ART = os.path.join(HERE, "artifacts")
REPLAYS = os.path.join(HERE, "replays")
SPATIAL_CKPT = os.path.join(ART, DEFAULT_SPATIAL_CHECKPOINT)
CHAMPION_SPATIAL = os.path.join(ART, "muzero_spatial_checkpoints_champion.pt")
STATUS_FILE = os.path.join(ART, "spatial_loop_status.json")
HISTORY_FILE = os.path.join(ART, "spatial_loop_history.json")
PYTHON = sys.executable


def _write_status(payload: Dict[str, Any]) -> None:
    os.makedirs(ART, exist_ok=True)
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _append_history(entry: Dict[str, Any]) -> None:
    hist = []
    if os.path.isfile(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                hist = json.load(f)
        except Exception:
            hist = []
    hist.append(entry)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(hist[-200:], f, indent=2)


def run_spatial_train_iteration(
    *,
    phase1_steps: int = 1000,
    phase2_steps: int = 400,
    phase3_epochs: int = 400,
    phase4_steps: int = 1000,
    selfplay_episodes: int = 50,
    batch: int = 64,
    sims: int = 50,
    lr: float = 1e-3,
    k_steps: int = 5,
    buffer_max: int = 1000,
    out: str = SPATIAL_CKPT,
    scratch: bool = False,
) -> Dict[str, Any]:
    cmd = [
        PYTHON,
        "-m",
        "pipelines.train_muzero",
        "--arch",
        "spatial",
        "--phase1-steps",
        str(phase1_steps),
        "--phase2-steps",
        str(phase2_steps),
        "--phase3-epochs",
        str(phase3_epochs),
        "--phase4-steps",
        str(phase4_steps),
        "--selfplay-episodes",
        str(selfplay_episodes),
        "--batch",
        str(batch),
        "--sims",
        str(sims),
        "--lr",
        str(lr),
        "--k-steps",
        str(k_steps),
        "--buffer-max",
        str(buffer_max),
        "--out",
        out,
        "--summary-out",
        os.path.join(ART, "train_summary_spatial.json"),
    ]
    if scratch:
        cmd.append("--scratch")
    print(f"[spatial-loop] train: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=HERE)
    return {
        "returncode": proc.returncode,
        "elapsed_s": time.time() - t0,
        "out": out,
    }


def promote_if_improved(
    candidate_path: str = SPATIAL_CKPT,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    """Atomic promote: validate load → copy spatial + packaging aliases + champion_spatial."""
    if not os.path.isfile(candidate_path):
        return {"promoted": False, "reason": "missing_candidate"}
    chassis = KaggricultureMuZeroChassis()
    load_spatial_checkpoint(chassis, candidate_path, strict=False)
    paths = promote_spatial_champion(candidate_path)
    os.makedirs(ART, exist_ok=True)
    shutil.copy2(candidate_path, CHAMPION_SPATIAL)
    paths["champion_spatial"] = CHAMPION_SPATIAL
    return {"promoted": True, "force": force, **paths}


def run_spatial_continuous_loop(
    *,
    iterations: int = 50,
    phase1_steps: int = 1000,
    phase2_steps: int = 400,
    phase3_epochs: int = 400,
    phase4_steps: int = 1000,
    selfplay_episodes: int = 50,
    batch: int = 64,
    sims: int = 50,
    lr: float = 1e-3,
    k_steps: int = 5,
    buffer_max: int = 1000,
    promote_every: int = 1,
    prune: bool = True,
) -> None:
    print(
        f"[spatial-loop] COMPETITION iterations={iterations} "
        f"sims={sims} batch={batch} selfplay={selfplay_episodes}"
    )
    for i in range(1, iterations + 1):
        _write_status({"iteration": i, "phase": "train", "ts": time.time()})
        result = run_spatial_train_iteration(
            phase1_steps=phase1_steps,
            phase2_steps=phase2_steps,
            phase3_epochs=phase3_epochs,
            phase4_steps=phase4_steps,
            selfplay_episodes=selfplay_episodes,
            batch=batch,
            sims=sims,
            lr=lr,
            k_steps=k_steps,
            buffer_max=buffer_max,
            scratch=(i == 1 and not os.path.isfile(SPATIAL_CKPT)),
        )
        if result["returncode"] != 0:
            print(f"[spatial-loop] train failed rc={result['returncode']}; stopping")
            _write_status({"iteration": i, "phase": "failed", "result": result})
            break

        promo = {"promoted": False}
        if i % max(1, promote_every) == 0:
            _write_status({"iteration": i, "phase": "promote", "ts": time.time()})
            try:
                promo = promote_if_improved(SPATIAL_CKPT)
                print(f"[spatial-loop] promote → {promo}")
            except Exception as exc:
                promo = {"promoted": False, "error": str(exc)}
                print(f"[spatial-loop] promote failed: {exc}")

        if prune:
            try:
                budget = get_default_replay_budget_gb()
                prune_stale_selfplay_replays(REPLAYS, max_dir_size_gb=budget)
            except Exception as exc:
                print(f"[spatial-loop] prune skipped: {exc}")

        entry = {"iteration": i, "train": result, "promote": promo, "ts": time.time()}
        _append_history(entry)
        _write_status({"iteration": i, "phase": "done", **entry})
        print(f"[spatial-loop] iteration {i}/{iterations} complete ({result['elapsed_s']:.1f}s)")


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Spatial Sampled MuZero continuous loop (competition defaults)")
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--phase1-steps", type=int, default=1000)
    p.add_argument("--phase2-steps", type=int, default=400)
    p.add_argument("--phase3-epochs", type=int, default=400, help="Reanalyze steps")
    p.add_argument("--phase4-steps", type=int, default=1000, help="MCTS distill steps")
    p.add_argument("--selfplay-episodes", type=int, default=50)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--sims", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--k-steps", type=int, default=5)
    p.add_argument("--buffer-max", type=int, default=1000)
    p.add_argument("--promote-every", type=int, default=1)
    p.add_argument("--no-prune", action="store_true")
    p.add_argument("--promote-only", action="store_true", help="Only promote existing spatial ckpt")
    args = p.parse_args(argv)

    if args.promote_only:
        print(promote_if_improved(force=True))
        return 0

    run_spatial_continuous_loop(
        iterations=args.iterations,
        phase1_steps=args.phase1_steps,
        phase2_steps=args.phase2_steps,
        phase3_epochs=args.phase3_epochs,
        phase4_steps=args.phase4_steps,
        selfplay_episodes=args.selfplay_episodes,
        batch=args.batch,
        sims=args.sims,
        lr=args.lr,
        k_steps=args.k_steps,
        buffer_max=args.buffer_max,
        promote_every=args.promote_every,
        prune=not args.no_prune,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
