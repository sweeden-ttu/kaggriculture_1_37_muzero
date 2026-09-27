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

"""Replay storage, fast header metadata inspection, and disk space pruning."""
from __future__ import annotations

import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

REWARDS_REGEX = re.compile(r'"rewards"\s*:\s*\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]')


def extract_replay_header_info(path: str) -> Optional[Tuple[float, float]]:
    """
    Ultra-fast replay reward extractor.
    Reads the first 1024 bytes with regex to avoid parsing full 30MB+ JSON trees.
    Falls back to json.load only if regex fails.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            chunk = f.read(1024)
        m = REWARDS_REGEX.search(chunk)
        if m:
            return float(m.group(1)), float(m.group(2))
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            rew = data.get("rewards", [])
            if len(rew) >= 2:
                return float(rew[0]), float(rew[1])
    except Exception:
        return None
    return None


def is_canonical_replay(filename: str) -> bool:
    """Canonical Kaggle ground-truth replays have numeric filenames: e.g., 113179986.json."""
    base = os.path.basename(filename)
    name, ext = os.path.splitext(base)
    return ext == ".json" and name.isdigit()


def get_default_replay_budget_gb(replays_dir: Optional[str] = None) -> float:
    """
    Returns the target replay directory capacity limit based on workspace location:
      - /Volumes/TRAINREPLAYBOOST/muzero (READ+WRITE): 24.0 GB
      - /Users/sweeden/muzero (SCRATCH): 10.0 GB
      - Can be overridden via MAX_REPLAY_DIR_SIZE_GB environment variable.
    """
    env_val = os.environ.get("MAX_REPLAY_DIR_SIZE_GB")
    if env_val:
        try:
            return float(env_val)
        except ValueError:
            pass
    path_str = os.path.abspath(replays_dir or os.getcwd())
    if "TRAINREPLAYBOOST" in path_str:
        return 24.0
    return 10.0


def prune_stale_selfplay_replays(
    replays_dir: str,
    max_dir_size_gb: Optional[float] = None,
    dry_run: bool = False,
    **kwargs: Any,
) -> Tuple[int, float]:
    """
    Prunes older self-play replays strictly by file system update timestamps (mtime)
    to keep the replays directory under max_dir_size_gb (SCRATCH: 10.0 GB, READ+WRITE: 24.0 GB).

    Strict guarantees:
      1. Canonical Kaggle ground-truth replays (^[0-9]+.json$) are NEVER deleted.
      2. Deletes older non-canonical replays first by mtime ascending.
      3. Ceases deletion immediately once directory size is under max_dir_size_gb.

    Returns:
        Tuple of (pruned_count, freed_gb)
    """
    if max_dir_size_gb is None:
        max_dir_size_gb = get_default_replay_budget_gb(replays_dir)
    if not os.path.isdir(replays_dir):
        return 0, 0.0

    all_files = glob.glob(os.path.join(replays_dir, "*.json"))
    canonical_files: Set[str] = set()
    prune_candidates: List[Dict[str, Any]] = []
    total_bytes = 0

    for path in all_files:
        base = os.path.basename(path)
        try:
            size_bytes = os.path.getsize(path)
            mtime = os.path.getmtime(path)
        except OSError:
            size_bytes = 0
            mtime = 0.0

        total_bytes += size_bytes

        if is_canonical_replay(base):
            canonical_files.add(path)
        else:
            prune_candidates.append({
                "path": path,
                "mtime": mtime,
                "size_bytes": size_bytes,
            })

    max_bytes = int(max_dir_size_gb * (1024.0 ** 3))
    if total_bytes <= max_bytes:
        return 0, 0.0

    # Sort candidates strictly by file system update timestamp ascending (oldest first)
    prune_candidates.sort(key=lambda x: x["mtime"])

    pruned_count = 0
    freed_bytes = 0

    for rec in prune_candidates:
        if total_bytes <= max_bytes:
            break

        path = rec["path"]
        size = rec["size_bytes"]
        pruned_count += 1
        freed_bytes += size
        total_bytes -= size

        if not dry_run:
            try:
                os.remove(path)
            except OSError:
                pass

    freed_gb = freed_bytes / (1024.0 ** 3)
    return pruned_count, freed_gb


def discover_expert_replay_seats(
    replays_dir: str,
    min_score: float = 80000.0,
    limit: Optional[int] = None,
) -> List[Tuple[str, int]]:
    """Ingest expert `.json` replays: return (filename, seat) pairs by score.

    Winner seats are always included; runner-up seats are included when their
    reward meets ``min_score`` (high-score / “high-ELO” filter for PER BC).
    """
    found: List[Tuple[float, str, int]] = []
    for path in sorted(glob.glob(os.path.join(replays_dir, "*.json"))):
        rewards = extract_replay_header_info(path)
        if not rewards or len(rewards) < 2:
            continue
        base = os.path.basename(path)
        winner = 0 if rewards[0] >= rewards[1] else 1
        runner = 1 - winner
        found.append((float(rewards[winner]), base, winner))
        if float(rewards[runner]) >= min_score:
            found.append((float(rewards[runner]), base, runner))
    found.sort(key=lambda x: x[0], reverse=True)
    if limit is not None and limit > 0:
        found = found[:limit]
    return [(f[1], f[2]) for f in found]
