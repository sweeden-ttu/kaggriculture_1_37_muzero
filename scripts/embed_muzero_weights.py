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

"""Bundle MuZero checkpoint into existing submission zip.

Delegates to canonical packaging domain.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from packaging.builder import DEFAULT_CKPT, embed_weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", default=os.path.join(HERE, "dist", "main.py"), help="Standalone main.py to pack")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="Checkpoint file path")
    parser.add_argument("--zips", nargs="+", default=[os.path.join(HERE, "dist", "submission.zip")])
    args = parser.parse_args()

    for zp in args.zips:
        embed_weights(zip_path=zp, ckpt_path=args.ckpt, main_path=args.main)


if __name__ == "__main__":
    main()
