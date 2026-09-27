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

"""Build submission_muzero.zip = hybrid_direct main + MuZero expand gate.

Delegates to canonical packaging domain.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from packaging.builder import DEFAULT_CKPT, build_hybrid_package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-weights", action="store_true")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT, help="Path to checkpoint")
    parser.add_argument("--out-zip", type=str, default=None, help="Output zip path")
    parser.add_argument("--out-py", type=str, default=None, help="Output main.py path")
    args = parser.parse_args()

    target_py, target_zip = build_hybrid_package(
        ckpt_path=args.ckpt,
        out_zip=args.out_zip,
        out_py=args.out_py,
        with_weights=args.with_weights,
    )
    print(f"Built hybrid package: {target_py}, {target_zip}")


if __name__ == "__main__":
    main()
