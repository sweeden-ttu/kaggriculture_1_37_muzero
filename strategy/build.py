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

"""Build a single-file candidate by appending strategy layers to a base chassis.

    python -m strategy.build --layers species_planner --out dist/candidate_species.py

Layers live in ``strategy/layers/<name>.py`` and are written in the chassis'
own wrapping style (``_X_PARENT = agent; del agent; def agent(...)``). The
result is validated by importing it and playing a short game against ``pass``.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import uuid

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYER_DIR = os.path.join(HERE, "strategy", "layers")
DEFAULT_BASE = os.path.join(HERE, "baselines", "champion_main_20260927_051846.py")


def build(base: str, layers: list[str], out: str, settings: dict | None = None) -> str:
    with open(base, "r", encoding="utf-8") as f:
        src = f.read()
    if not src.endswith("\n"):
        src += "\n"
    parts = [src]
    for name in layers:
        path = name if name.endswith(".py") else os.path.join(LAYER_DIR, name + ".py")
        with open(path, "r", encoding="utf-8") as f:
            layer_src = f.read()
        parts.append(f"\n\n# ===== strategy layer: {os.path.basename(path)} =====\n")
        parts.append(layer_src if layer_src.endswith("\n") else layer_src + "\n")
    if settings:
        parts.append("\n# ===== build-time settings =====\n")
        for key, value in settings.items():
            parts.append(f"{key} = {value!r}\n")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    return out


def validate(path: str, steps: int = 120) -> tuple[float, float]:
    import kaggle_environments

    name = "candidate_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    fn = getattr(module, "agent", None) or getattr(module, "kaggle_submission_agent")
    env = kaggle_environments.make("kaggriculture", configuration={"episodeSteps": steps, "seed": 7})
    t0 = time.time()
    env.run([fn, "pass"])
    elapsed = time.time() - t0
    statuses = [s.status for s in env.state]
    if any(st == "ERROR" for st in statuses):
        raise RuntimeError(f"candidate errored: {statuses}")
    telemetry = getattr(fn, "telemetry", None)
    return float(env.state[0].reward or 0.0), elapsed, telemetry


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a single-file candidate from base + layers")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--layers", nargs="*", default=["species_planner"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="build-time overrides KEY=PYTHON_LITERAL")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()
    settings = {}
    for item in args.set:
        key, _, value = item.partition("=")
        settings[key.strip()] = eval(value, {})  # literals only, from the command line
    out = build(args.base, args.layers, args.out, settings)
    size = os.path.getsize(out)
    print(f"built {out} ({size:,} bytes) from {os.path.basename(args.base)} + {args.layers} {settings or ''}")
    if not args.no_validate:
        reward, elapsed, telemetry = validate(out)
        print(f"validated: 120-step game vs pass ok, money ${reward:,.0f}, {elapsed:.1f}s, telemetry={telemetry}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
