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

"""Hybrid chassis standalone compiler and local execution CLI."""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from hybrid_chassis import compile_standalone, get_agent


def kaggle_submission_agent(
    observation: Dict[str, Any],
    configuration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Authoritative Kaggle submission entrypoint."""
    return get_agent()(observation, configuration)


agent = kaggle_submission_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggriculture Hybrid Chassis Compiler & CLI")
    parser.add_argument(
        "--test",
        type=int,
        nargs="?",
        const=42,
        metavar="SEED",
        help="Run one evaluation game vs pass (default seed 42)",
    )
    parser.add_argument(
        "--compile",
        type=str,
        nargs="?",
        const=os.path.join(HERE, "dist", "compiled_submission.py"),
        metavar="OUTPUT_PATH",
        help="Compile modular package into a standalone submission script",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run microbenchmarks on chassis action speed",
    )
    args = parser.parse_args()

    if args.compile:
        output_path = compile_standalone(args.compile)
        print(f"Chassis compiled standalone → {output_path}")

    if args.test is not None:
        import kaggle_environments

        env = kaggle_environments.make("kaggriculture", configuration={"episodeSteps": 720})
        env.run([agent, lambda obs, cfg=None: {"farmer": ["PASS"], "hands": [], "market": []}])
        print(f"Test match complete. Rewards: {env.state[0].reward} vs {env.state[1].reward}")

    if args.benchmark:
        from hybrid_chassis.pipeline import build_legacy_pipeline
        build_legacy_pipeline()


if __name__ == "__main__":
    main()
