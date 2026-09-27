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

"""download_top_gold_replays.py
==============================
Downloads today's top-tier Gold Kaggle replays where BOTH players scored > 100,000 coin.

Workflow:
1. Backs up existing local replays to artifacts/replays_preclear_backup and clears replays/
2. Fetches the live leaderboard for competition 'kaggriculture'
3. Filters for teams in the Gold tier (Score >= 2793)
4. Discovers all submissions and episodes from today (UTC)
5. Downloads each candidate replay and inspects final rewards
6. Saves ONLY matches where BOTH player 0 and player 1 scored > 100,000 coin
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Sequence, Set, Tuple

import pandas as pd

KAGGLE_BIN = "/Users/sweeden/.local/bin/kaggle"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(HERE, "replays")
BACKUP_DIR = os.path.join(HERE, "artifacts", "replays_preclear_backup")
EXTERNAL_REPLAYS = "/Volumes/TRAINREPLAYBOOST/muzero/replays"
EXTERNAL_BACKUP_DIR = "/Volumes/TRAINREPLAYBOOST/muzero/artifacts/replays_preclear_backup"

COMPETITION_SLUG = "kaggriculture"
GOLD_TIER_THRESHOLD = 2793.0
WINNINGS_THRESHOLD = 100000.0


def run_kaggle_cmd(cmd_args: List[str]) -> str:
    cmd = [KAGGLE_BIN] + cmd_args
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return ""
    return res.stdout


def backup_and_clear_replays(clear_external: bool = True) -> None:
    os.makedirs(BACKUP_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    existing = [f for f in os.listdir(SAVE_DIR) if f.endswith(".json")]
    print(f"[Backup] Backing up {len(existing)} existing replays to {BACKUP_DIR}...")
    for f in existing:
        src = os.path.join(SAVE_DIR, f)
        dst = os.path.join(BACKUP_DIR, f)
        if os.path.exists(dst):
            os.remove(src)
        else:
            try:
                shutil.move(src, dst)
            except Exception:
                shutil.copyfile(src, dst)
                os.remove(src)
    print(f"[Clear] Cleared {len(existing)} replays from {SAVE_DIR}.")

    if clear_external and os.path.isdir(EXTERNAL_REPLAYS):
        os.makedirs(EXTERNAL_BACKUP_DIR, exist_ok=True)
        ext_existing = [f for f in os.listdir(EXTERNAL_REPLAYS) if f.endswith(".json")]
        print(f"[Backup External] Backing up {len(ext_existing)} external replays to {EXTERNAL_BACKUP_DIR}...")
        for f in ext_existing:
            src = os.path.join(EXTERNAL_REPLAYS, f)
            dst = os.path.join(EXTERNAL_BACKUP_DIR, f)
            if os.path.exists(dst):
                os.remove(src)
            else:
                try:
                    shutil.move(src, dst)
                except Exception:
                    shutil.copyfile(src, dst)
                    os.remove(src)
        print(f"[Clear External] Cleared {len(ext_existing)} replays from {EXTERNAL_REPLAYS}.")


def get_gold_tier_teams(elo_min: float = GOLD_TIER_THRESHOLD) -> List[Dict[str, Any]]:
    print(f"\n[Leaderboard] Fetching leaderboard for '{COMPETITION_SLUG}'...")
    raw = run_kaggle_cmd(["competitions", "leaderboard", COMPETITION_SLUG, "--show", "--page-size", "100", "-v"])
    lines = [l for l in raw.splitlines() if not l.startswith("Next Page Token")]
    if not lines:
        return []
    df = pd.read_csv(io.StringIO("\n".join(lines)))
    gold = df[df["score"] >= elo_min].copy()
    print(f"[Leaderboard] Found {len(gold)} Gold-tier teams (Score >= {elo_min}):")
    teams = []
    for _, row in gold.iterrows():
        t = {
            "team_id": int(row["teamId"]),
            "team_name": str(row["teamName"]),
            "score": float(row["score"]),
        }
        teams.append(t)
        print(f"  Team {t['team_id']:>8}: {t['team_name'][:25]:<25} | Elo: {t['score']:.1f}")
    return teams


def get_recent_episodes_for_team(team: Dict[str, Any], date_prefixes: Sequence[str]) -> List[int]:
    tid = team["team_id"]
    sub_raw = run_kaggle_cmd(["competitions", "team-submissions", str(tid), "-v"])
    if not sub_raw:
        return []
    lines = [l for l in sub_raw.splitlines() if not l.startswith("Next Page Token")]
    if not lines:
        return []
    try:
        sub_df = pd.read_csv(io.StringIO("\n".join(lines)))
    except Exception:
        return []

    episodes = []
    for _, srow in sub_df.head(2).iterrows():
        sid = int(srow["id"])
        ep_raw = run_kaggle_cmd(["competitions", "episodes", str(sid), "-v"])
        if not ep_raw:
            continue
        elines = [l for l in ep_raw.splitlines() if not l.startswith("Next Page Token")]
        if not elines:
            continue
        try:
            ep_df = pd.read_csv(io.StringIO("\n".join(elines)))
        except Exception:
            continue

        if "createTime" not in ep_df.columns:
            continue

        # Filter for recent completed public matches
        is_recent = ep_df["createTime"].astype(str).apply(lambda s: any(s.startswith(p) for p in date_prefixes))
        todays = ep_df[is_recent]
        for _, erow in todays.iterrows():
            ep_id = int(erow["id"])
            episodes.append(ep_id)

    return episodes


def inspect_replay_rewards(file_path: str) -> Tuple[float, float]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rew = data.get("rewards")
        if rew and len(rew) >= 2:
            return float(rew[0] or 0.0), float(rew[1] or 0.0)
        steps = data.get("steps")
        if steps and len(steps) > 0:
            last = steps[-1]
            return float(last[0].get("reward", 0.0) or 0.0), float(last[1].get("reward", 0.0) or 0.0)
    except Exception:
        pass
    return 0.0, 0.0


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download top-tier Gold Kaggle replays with dual >100k coin scores.")
    parser.add_argument("--target-count", type=int, default=50, help="Target number of qualified replays to download (default: 50)")
    parser.add_argument("--min-score", type=float, default=WINNINGS_THRESHOLD, help="Minimum coin score for BOTH players (default: 100,000)")
    parser.add_argument("--elo-min", type=float, default=GOLD_TIER_THRESHOLD, help="Minimum team leaderboard Elo (default: 2793.0)")
    parser.add_argument("--days", type=int, default=2, help="Number of past days (UTC) to search episodes from (default: 2)")
    parser.add_argument("--skip-clear", action="store_true", help="Skip clearing existing replays")
    parser.add_argument("--no-external-sync", action="store_true", help="Do not clear or sync to external volume")
    args = parser.parse_args()

    print("=" * 80)
    print("  KAGGRICULTURE TOP-TIER GOLD REPLAY INGESTION")
    print(f"  Gold Tier Threshold: Elo >= {args.elo_min:.1f}")
    print(f"  Winnings Threshold:  Both Players > ${args.min_score:,.0f} coin")
    print(f"  Target Qualified Count: {args.target_count}")
    print("=" * 80)

    # 1. Clear local replays directory
    if not args.skip_clear:
        backup_and_clear_replays(clear_external=not args.no_external_sync)
    else:
        print("[Clear] Skipped clearing replays directory (--skip-clear enabled).")

    # 2. Get Gold tier teams
    teams = get_gold_tier_teams(elo_min=args.elo_min)
    if not teams:
        print("Error: Could not retrieve Gold tier teams from leaderboard.")
        return

    # 3. Discover recent episodes across all Gold teams
    now_utc = datetime.now(timezone.utc)
    date_prefixes = [(now_utc - timedelta(days=d)).strftime("%Y-%m-%d") for d in range(args.days)]
    print(f"\n[Episodes Discovery] Searching matches played on {date_prefixes} (UTC)...")

    all_episodes: Set[int] = set()
    for t in teams:
        eps = get_recent_episodes_for_team(t, date_prefixes)
        all_episodes.update(eps)
        if eps:
            print(f"  Team {t['team_name'][:20]:<20}: {len(eps)} episode(s) found in window")

    print(f"\n[Total Discovery] Found {len(all_episodes)} unique episodes across Gold teams.")

    # 4. Download and filter replays
    kept_count = 0
    staging_dir = tempfile.mkdtemp(prefix="kagg_staging_")

    try:
        sorted_eps = sorted(list(all_episodes), reverse=True)
        print(f"\n[Downloading & Filtering] Inspecting episodes (filtering for both players > ${args.min_score:,.0f})...")

        for idx, ep_id in enumerate(sorted_eps, 1):
            target_file = os.path.join(SAVE_DIR, f"{ep_id}.json")
            if os.path.exists(target_file):
                continue

            # Download using kaggle CLI
            run_kaggle_cmd(["competitions", "replay", str(ep_id), "-p", staging_dir, "-q"])

            # Find downloaded file
            staged = [os.path.join(staging_dir, f) for f in os.listdir(staging_dir) if str(ep_id) in f]
            if not staged:
                continue

            staged_path = staged[0]
            r0, r1 = inspect_replay_rewards(staged_path)

            if r0 > args.min_score and r1 > args.min_score:
                shutil.move(staged_path, target_file)
                kept_count += 1
                print(f"  ✓ [{kept_count:>2}/{args.target_count}] Episode {ep_id}: P0=${r0:>8,.0f} | P1=${r1:>8,.0f} (QUALIFIED!)")
                if not args.no_external_sync and os.path.isdir(EXTERNAL_REPLAYS):
                    shutil.copyfile(target_file, os.path.join(EXTERNAL_REPLAYS, f"{ep_id}.json"))
                if kept_count >= args.target_count:
                    print(f"\n[Target Reached] Successfully acquired {kept_count} qualified top-tier replays.")
                    break
            else:
                os.remove(staged_path)
                print(f"  ✗ [{idx:>3}/{len(sorted_eps)}] Episode {ep_id}: P0=${r0:>8,.0f} | P1=${r1:>8,.0f} (Below filter)")

            # Gentle rate pacing
            time.sleep(0.3)

    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    print("\n" + "=" * 80)
    print(f"  REPLAY INGESTION COMPLETE")
    print(f"  Saved {kept_count} top-tier dual >${args.min_score:,.0f} episodes to {SAVE_DIR}")
    if not args.no_external_sync and os.path.isdir(EXTERNAL_REPLAYS):
        print(f"  Mirrored {kept_count} episodes to {EXTERNAL_REPLAYS}")
    print("=" * 80)


if __name__ == "__main__":
    main()

