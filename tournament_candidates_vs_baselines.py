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
=============================================================================
tournament_candidates_vs_baselines.py - Master Cross-Tournament CLI
=============================================================================
Places all reference baselines from /Volumes/BASELINES/muzero/baselines
against the league of operational candidates from /Volumes/TRAINREPLAYBOOST/muzero/baselines
and saves/promotes the winners to /Users/sweeden/muzero/baselines.

Usage Examples:
  # Standard cross-tournament (2 games/pair, alternating seats):
  python tournament_candidates_vs_baselines.py

  # Fast evaluation with 8 workers and top-4 promotion:
  python tournament_candidates_vs_baselines.py --workers 8 --top-k 4 --save-criteria top-k

  # Dry-run evaluation without overwriting destination files:
  python tournament_candidates_vs_baselines.py --dry-run
=============================================================================
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from evaluation.candidates_vs_baselines import main

if __name__ == "__main__":
    sys.exit(main())
