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

"""
Render full 720-step interactive visualizer for Cand-010 vs Cand-008 playoff.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import kaggle_environments

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from evaluation.harness import SubprocessAgent, extract_agent, short_name


def render_head_to_head(
    agent_a_path: str,
    agent_b_path: str,
    seed: int = 10119,
    a_is_p0: bool = True,
    steps: int = 720,
    out_html: str = "artifacts/cand010_vs_cand008_match.html",
    out_json: str = "artifacts/cand010_vs_cand008_replay.json",
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)

    match_dir = os.path.join(HERE, "_render_h2h_tmp")
    shutil.rmtree(match_dir, ignore_errors=True)
    os.makedirs(match_dir, exist_ok=True)

    dir_a = os.path.join(match_dir, "a")
    dir_b = os.path.join(match_dir, "b")

    print("=" * 80)
    print(f"  Rendering 720-Step Gameplay: {short_name(agent_a_path)} vs {short_name(agent_b_path)}")
    print(f"  Seed: {seed} | P0: {short_name(agent_a_path) if a_is_p0 else short_name(agent_b_path)} | Steps: {steps}")
    print("=" * 80)

    try:
        extract_agent(agent_a_path, dir_a)
        extract_agent(agent_b_path, dir_b)

        proc_a = SubprocessAgent(dir_a, short_name(agent_a_path))
        proc_b = SubprocessAgent(dir_b, short_name(agent_b_path))

        try:
            p0, p1 = (proc_a, proc_b) if a_is_p0 else (proc_b, proc_a)
            env = kaggle_environments.make(
                "kaggriculture",
                configuration={"episodeSteps": steps, "seed": seed},
                debug=True,
            )
            env.info["seed"] = seed
            env.configuration["seed"] = seed

            t0 = time.time()
            print("  Running 720-step simulation...")
            env.run([p0, p1])
            elapsed = time.time() - t0

            final = env.steps[-1]
            r0 = float(final[0].reward or 0.0)
            r1 = float(final[1].reward or 0.0)
            print(f"  Simulation complete in {elapsed:.1f}s.")
            print(f"  Player 0 ({short_name(agent_a_path) if a_is_p0 else short_name(agent_b_path)}): ${r0:,.0f} | status={final[0].status}")
            print(f"  Player 1 ({short_name(agent_b_path) if a_is_p0 else short_name(agent_a_path)}): ${r1:,.0f} | status={final[1].status}")

            # Dump replay JSON
            print(f"  Writing replay JSON → {out_json}...")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(env.toJSON(), f)

            # Dump interactive visualizer HTML
            print(f"  Rendering interactive visualizer HTML → {out_html}...")
            html_content = env.render(mode="html", width=1280, height=860)
            with open(out_html, "w", encoding="utf-8") as f:
                f.write(html_content)

            print(f"\n[SUCCESS] Interactive HTML visualizer written to:\n  file://{os.path.abspath(out_html)}")
            print(f"Replay JSON written to:\n  file://{os.path.abspath(out_json)}")

        finally:
            proc_a.close()
            proc_b.close()
    finally:
        shutil.rmtree(match_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-a", default="dist/champion_cand010.zip")
    parser.add_argument("--agent-b", default="dist/champion_cand008.zip")
    parser.add_argument("--seed", type=int, default=10119)
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--a-is-p0", action="store_true", default=True)
    parser.add_argument("--out-html", default=os.path.join(HERE, "artifacts", "cand010_vs_cand008_match.html"))
    parser.add_argument("--out-json", default=os.path.join(HERE, "artifacts", "cand010_vs_cand008_replay.json"))
    args = parser.parse_args()

    render_head_to_head(
        agent_a_path=args.agent_a,
        agent_b_path=args.agent_b,
        seed=args.seed,
        a_is_p0=args.a_is_p0,
        steps=args.steps,
        out_html=args.out_html,
        out_json=args.out_json,
    )
