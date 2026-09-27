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

"""Evaluation harness: Isolated subprocess agent sandbox and match execution."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import kaggle_environments

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SANDBOX_ROOT = os.path.join(HERE, "_eval_tournament_dist")

WORKER_SCRIPT = """
import sys
import os
import json

# Redirect stdout prints to stderr so IPC on stdout is pure JSON
real_stdout = sys.stdout
sys.stdout = sys.stderr

cwd = os.path.dirname(os.path.abspath(__file__))
if cwd not in sys.path:
    sys.path.insert(0, cwd)

import main

agent_fn = getattr(main, "agent", None) or getattr(main, "rl_agent", None) or getattr(main, "my_agent", None)
if not callable(agent_fn):
    agent_reg = getattr(main, "kaggle_submission_agent", None)
    if callable(agent_reg):
        agent_fn = agent_reg
    else:
        raise RuntimeError("Could not find callable agent in main.py")

real_stdout.write(json.dumps({"status": "READY"}) + "\\n")
real_stdout.flush()

while True:
    line = sys.stdin.readline()
    if not line:
        break
    line_str = line.strip()
    if not line_str:
        continue
    try:
        msg = json.loads(line_str)
        obs = msg.get("obs", {})
        cfg = msg.get("config", {})
        action = agent_fn(obs, cfg)
        response = json.dumps({"action": action})
    except Exception as e:
        response = json.dumps({"error": str(e), "action": {}})
    real_stdout.write(response + "\\n")
    real_stdout.flush()
"""


class SubprocessAgent:
    """Runs an agent in an isolated subprocess working directory with pipe IPC."""

    def __init__(self, agent_dir: str, name: str, env: Optional[Dict[str, str]] = None):
        self.agent_dir = agent_dir
        self.name = name
        env_dict = dict(os.environ)
        if env:
            env_dict.update(env)
        self.process = subprocess.Popen(
            [sys.executable, "-u", "worker.py"],
            cwd=self.agent_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env_dict,
        )
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        ready_line = self.process.stdout.readline()
        if not ready_line:
            err = self.process.stderr.read()
            raise RuntimeError(f"Agent {self.name} failed to initialize: {err}")
        ready = json.loads(ready_line)
        if ready.get("status") != "READY":
            raise RuntimeError(f"Unexpected init message from {self.name}: {ready}")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        req = json.dumps({"obs": obs, "config": config or {}}) + "\n"
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        self.process.stdin.write(req)
        self.process.stdin.flush()
        resp_line = self.process.stdout.readline()
        if not resp_line:
            err = self.process.stderr.read()
            raise RuntimeError(f"Agent {self.name} subprocess crashed: {err}")
        resp = json.loads(resp_line)
        if "error" in resp:
            raise RuntimeError(f"Agent {self.name} returned error: {resp['error']}")
        return resp.get("action", {})

    def close(self) -> None:
        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
            self.process.terminate()
            self.process.wait(timeout=2.0)
        except Exception:
            pass


def short_name(path: str) -> str:
    base = os.path.basename(path)
    for ext in (".zip", ".py", ".tar.gz"):
        if base.endswith(ext):
            return base[:-len(ext)]
    return base


def extract_agent(source_path: str, target_dir: str) -> None:
    os.makedirs(target_dir, exist_ok=True)
    if source_path.endswith(".zip"):
        with zipfile.ZipFile(source_path, "r") as zf:
            zf.extractall(target_dir)
    elif os.path.isfile(source_path):
        target_main = os.path.join(target_dir, "main.py")
        shutil.copy2(source_path, target_main)
        parent = os.path.dirname(os.path.abspath(source_path))
        for extra in ("artifacts", "muzero", "hybrid_chassis"):
            src_extra = os.path.join(parent, extra)
            dst_extra = os.path.join(target_dir, extra)
            if os.path.isdir(src_extra) and not os.path.exists(dst_extra):
                shutil.copytree(src_extra, dst_extra, dirs_exist_ok=True)
    elif os.path.isdir(source_path):
        shutil.copytree(source_path, target_dir, dirs_exist_ok=True)
    else:
        raise FileNotFoundError(f"Agent path not found: {source_path}")

    worker_file = os.path.join(target_dir, "worker.py")
    with open(worker_file, "w", encoding="utf-8") as f:
        f.write(WORKER_SCRIPT)


def play_single_game(
    agent_a_path: str,
    agent_b_path: str,
    seed: int,
    a_is_p0: bool,
    episode_steps: int = 720,
    save_replay_dir: Optional[str] = None,
    min_replay_score: float = 80000.0,
    env_a: Optional[Dict[str, str]] = None,
    env_b: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Execute one sealed match between two agents in isolated sandboxes."""
    match_dir = os.path.join(SANDBOX_ROOT, f"match_{os.getpid()}_{seed}_{time.time_ns()}")
    dir_a = os.path.join(match_dir, "a")
    dir_b = os.path.join(match_dir, "b")
    saved_replay: Optional[str] = None
    try:
        extract_agent(agent_a_path, dir_a)
        extract_agent(agent_b_path, dir_b)
        proc_a = SubprocessAgent(dir_a, short_name(agent_a_path), env=env_a)
        proc_b = SubprocessAgent(dir_b, short_name(agent_b_path), env=env_b)
        try:
            p0, p1 = (proc_a, proc_b) if a_is_p0 else (proc_b, proc_a)
            env = kaggle_environments.make(
                "kaggriculture",
                configuration={"episodeSteps": episode_steps, "seed": seed},
            )
            env.info["seed"] = seed
            env.configuration["seed"] = seed
            t0 = time.time()
            env.run([p0, p1])
            elapsed = round(time.time() - t0, 2)
            r0 = float(env.state[0].reward or 0.0)
            r1 = float(env.state[1].reward or 0.0)

            if save_replay_dir and max(r0, r1) >= min_replay_score:
                try:
                    os.makedirs(save_replay_dir, exist_ok=True)
                    tag_a = short_name(agent_a_path)
                    tag_b = short_name(agent_b_path)
                    ep_fn = f"selfplay_seed{seed}_{tag_a}_vs_{tag_b}.json"
                    ep_path = os.path.join(save_replay_dir, ep_fn)
                    ep_data = {
                        "id": ep_fn.replace(".json", ""),
                        "rewards": [r0, r1],
                        "configuration": dict(env.configuration),
                        "specification": dict(env.specification),
                        "steps": env.steps,
                    }
                    with open(ep_path, "w", encoding="utf-8") as f:
                        json.dump(ep_data, f)
                    saved_replay = ep_path
                except Exception as e:
                    print(f"  [warn] failed to save replay: {e}")
        finally:
            proc_a.close()
            proc_b.close()
    finally:
        shutil.rmtree(match_dir, ignore_errors=True)

    score_a = r0 if a_is_p0 else r1
    score_b = r1 if a_is_p0 else r0
    if score_a > score_b:
        winner = "A"
    elif score_b > score_a:
        winner = "B"
    else:
        winner = "TIE"

    return {
        "seed": seed,
        "a_seat": "P0" if a_is_p0 else "P1",
        "b_seat": "P1" if a_is_p0 else "P0",
        "score_a": score_a,
        "score_b": score_b,
        "delta_a": score_a - score_b,
        "winner": winner,
        "elapsed_sec": elapsed,
        "saved_replay": saved_replay,
    }


def eval_match_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level worker function for multiprocessing pools."""
    res = play_single_game(
        agent_a_path=task["agent_a"],
        agent_b_path=task["agent_b"],
        seed=task["seed"],
        a_is_p0=task["a_is_p0"],
        episode_steps=task.get("episode_steps", 720),
        save_replay_dir=task.get("save_replay_dir"),
        min_replay_score=task.get("min_replay_score", 80000.0),
        env_a=task.get("env_a"),
        env_b=task.get("env_b"),
    )
    res["task"] = task
    return res
