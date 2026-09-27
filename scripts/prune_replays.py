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
scripts/prune_replays.py
CLI tool to prune obsolete selfplay replays and reclaim disk space.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from pipelines.replay_manager import get_default_replay_budget_gb, prune_stale_selfplay_replays

DEFAULT_REPLAYS = os.path.join(HERE, "replays")


def main() -> None:
    default_budget = get_default_replay_budget_gb(DEFAULT_REPLAYS)
    parser = argparse.ArgumentParser(description=f"Prune older self-play replays to keep folder under size budget ({default_budget:.1f} GB).")
    parser.add_argument("--replays-dir", default=DEFAULT_REPLAYS, help="Directory containing replays")
    parser.add_argument("--max-size-gb", type=float, default=None, help="Maximum replays folder size budget in GB (default: 10.0 SCRATCH / 24.0 READ+WRITE)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate pruning without deleting files")
    args = parser.parse_args()

    max_size = args.max_size_gb if args.max_size_gb is not None else get_default_replay_budget_gb(args.replays_dir)
    print(f"[Prune Replays] Scanning {args.replays_dir} (target_max_size={max_size:.2f} GB, dry_run={args.dry_run})...")
    pruned, freed_gb = prune_stale_selfplay_replays(
        replays_dir=args.replays_dir,
        max_dir_size_gb=max_size,
        dry_run=args.dry_run,
    )
    action_str = "Would prune" if args.dry_run else "Successfully pruned"
    print(f"[Prune Replays] {action_str} {pruned} older self-play replays. Reclaimed {freed_gb:.2f} GB of disk space (folder now <= {args.max_size_gb:.2f} GB).")


if __name__ == "__main__":
    main()
