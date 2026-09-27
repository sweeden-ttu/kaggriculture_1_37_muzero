#!/usr/bin/env python3
"""
survivor_elimination_tournament.py
==================================
Multi-Stage Weakest-Link Elimination Tournament for Kaggriculture baselines.
Uses the clean, deduplicated SCRATCH partition at /Users/sweeden/muzero/baselines.

Architecture:
  - Stage 1 (The Open Qualifier): All 21 clean contenders play a round-robin (1 game/pair).
    The Cut Line eliminates the weakest links (bottom 11 eliminated, top 10 advance).
  - Stage 2 (Survivor Semifinals): Top 10 survivors battle in a 2-games/pair round-robin
    with alternating seats. The Cut Line eliminates the bottom 6 (top 4 advance).
  - Stage 3 (Grand Championship Finals): Top 4 elite finalists clash in a decisive
    4-games/pair series (2 as P0, 2 as P1) to crown the undisputed champion.

Outputs:
  - JSON Report: artifacts/survivor_tournament.json
  - Interactive HTML Dashboard: artifacts/survivor_tournament_dashboard.html
"""
from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from evaluation.harness import eval_match_worker, extract_agent, short_name

BASELINES_DIR = "/Users/sweeden/muzero/baselines"
DEFAULT_JSON = os.path.join(HERE, "artifacts", "survivor_tournament.json")
DEFAULT_HTML = os.path.join(HERE, "artifacts", "survivor_tournament_dashboard.html")


def get_clean_roster(baselines_dir: str = BASELINES_DIR) -> List[Dict[str, Any]]:
    """Loads all unique, deduplicated contenders from baselines_dir."""
    roster = []
    for fn in sorted(os.listdir(baselines_dir)):
        if fn.startswith(".") or fn.startswith("._") or fn == ".DS_Store":
            continue
        p = os.path.join(baselines_dir, fn)
        if not (os.path.isfile(p) and (fn.endswith(".zip") or fn.endswith(".tar.gz") or fn == "main.py")):
            continue

        clean_name = short_name(fn)
        category = "Official Kaggle Baseline" if (fn.startswith("submission_20") or fn.startswith("submission_0771") or fn == "main.py") else "Candidate"
        tier = "Tier 1-2 (Kaggle Benchmark)" if category == "Official Kaggle Baseline" else ("Tier 3 (Neural / Boost)" if "boost" in fn else "Tier 3 (Neural / Train)")

        roster.append({
            "path": p,
            "filename": fn,
            "name": clean_name,
            "display_name": clean_name.replace("submission_", ""),
            "category": category,
            "tier": tier,
            "size": os.path.getsize(p),
        })

    return roster


def execute_matches(
    pairings: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    games_per_pair: int,
    base_seed: int,
    workers: int,
    stage_name: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Runs a batch of pairings and aggregates stats."""
    tasks = []
    game_idx = 0
    stats = {}

    for p_a, p_b in pairings:
        for a in (p_a, p_b):
            if a["name"] not in stats:
                stats[a["name"]] = {
                    "name": a["name"],
                    "display_name": a["display_name"],
                    "category": a["category"],
                    "tier": a["tier"],
                    "path": a["path"],
                    "wins": 0,
                    "losses": 0,
                    "ties": 0,
                    "points": 0.0,
                    "score_sum": 0.0,
                    "opp_score_sum": 0.0,
                    "games": 0,
                }

        for g in range(games_per_pair):
            game_idx += 1
            a_is_p0 = (g % 2 == 0)
            seed = base_seed + (game_idx * 23)
            tasks.append({
                "game_idx": game_idx,
                "agent_a": p_a["path"],
                "agent_b": p_b["path"],
                "a_name": p_a["name"],
                "b_name": p_b["name"],
                "a_is_p0": a_is_p0,
                "seed": seed,
                "episode_steps": 720,
            })

    print(f"\n--- {stage_name}: Running {len(tasks)} matches across {workers} workers ---")
    t0 = time.time()
    completed = 0
    match_logs = []

    with mp.Pool(processes=min(workers, len(tasks))) as pool:
        for res in pool.imap_unordered(eval_match_worker, tasks):
            completed += 1
            task = res["task"]
            name_a = task["a_name"]
            name_b = task["b_name"]
            st_a = stats[name_a]
            st_b = stats[name_b]

            s_a = res["score_a"]
            s_b = res["score_b"]
            w = res["winner"]

            st_a["games"] += 1
            st_b["games"] += 1
            st_a["score_sum"] += s_a
            st_b["score_sum"] += s_b
            st_a["opp_score_sum"] += s_b
            st_b["opp_score_sum"] += s_a

            if w == "A":
                st_a["wins"] += 1
                st_a["points"] += 1.0
                st_b["losses"] += 1
                w_str = name_a
            elif w == "B":
                st_b["wins"] += 1
                st_b["points"] += 1.0
                st_a["losses"] += 1
                w_str = name_b
            else:
                st_a["ties"] += 1
                st_b["ties"] += 1
                st_a["points"] += 0.5
                st_b["points"] += 0.5
                w_str = "TIE"

            match_logs.append({
                "game_idx": task["game_idx"],
                "seed": res["seed"],
                "agent_a": name_a,
                "agent_b": name_b,
                "score_a": s_a,
                "score_b": s_b,
                "winner": w_str,
                "elapsed_sec": res["elapsed_sec"],
            })

            if completed % 15 == 0 or completed == len(tasks):
                rate = completed / max(0.1, time.time() - t0)
                eta = (len(tasks) - completed) / max(0.01, rate)
                print(f"  [{stage_name}] {completed:>3}/{len(tasks)} ({completed/len(tasks)*100:5.1f}%) | {name_a[:20]:<20} vs {name_b[:20]:<20} → {w_str[:18]:<18} | ETA: {eta:>3.0f}s")

    for st in stats.values():
        st["avg_score"] = st["score_sum"] / max(1, st["games"])
        st["avg_margin"] = (st["score_sum"] - st["opp_score_sum"]) / max(1, st["games"])
        st["win_rate_pct"] = (100.0 * st["wins"] / max(1, st["games"]))

    return match_logs, stats


def run_survivor_tournament(baselines_dir: str = BASELINES_DIR, workers: int = 11) -> Dict[str, Any]:
    roster = get_clean_roster(baselines_dir)
    print("=" * 95)
    print(f"  SURVIVOR ELIMINATION TOURNAMENT (Weakest-Link Elimination Architecture)")
    print(f"  Source Partition: {baselines_dir}")
    print(f"  Total Clean Contenders: {len(roster)} | Parallel Workers: {workers}")
    print("=" * 95)

    tournament_start = time.time()

    # =========================================================================
    # STAGE 1: The Open Qualifier (21 Contenders, 1 Game/Pair)
    # =========================================================================
    stage1_pairs = list(combinations(roster, 2))
    stage1_matches, stage1_stats = execute_matches(
        stage1_pairs,
        games_per_pair=1,
        base_seed=10101,
        workers=workers,
        stage_name="Stage 1 (Qualifier)",
    )

    stage1_ranked = sorted(
        stage1_stats.values(),
        key=lambda x: (x["points"], x["avg_margin"], x["avg_score"]),
        reverse=True,
    )

    # Cut Line: Top 10 advance! Bottom 11 are eliminated as weakest links!
    CUT_OFF_1 = 10
    stage1_survivors = stage1_ranked[:CUT_OFF_1]
    stage1_eliminated = stage1_ranked[CUT_OFF_1:]

    print("\n" + "=" * 95)
    print("  STAGE 1 RESULTS: TOP 10 SURVIVORS ADVANCE | BOTTOM 11 ELIMINATED")
    print("=" * 95)
    for i, s in enumerate(stage1_survivors, 1):
        print(f"  ✓ SURVIVOR #{i:>2}: {s['name']:<28} | Pts: {s['points']:>4.1f} | Win%: {s['win_rate_pct']:>5.1f}% | Avg Margin: {s['avg_margin']:>+8,.0f}")
    print("-" * 95)
    for i, e in enumerate(stage1_eliminated, CUT_OFF_1 + 1):
        print(f"  ✗ ELIMINATED #{i:>2}: {e['name']:<26} | Pts: {e['points']:>4.1f} | Win%: {e['win_rate_pct']:>5.1f}% | Cause: Weakest Link")
    print("=" * 95)

    # =========================================================================
    # STAGE 2: The Semifinal Survivor Bracket (Top 10 Survivors, 2 Games/Pair)
    # =========================================================================
    roster_by_name = {a["name"]: a for a in roster}
    survivors_roster = [roster_by_name[s["name"]] for s in stage1_survivors]
    stage2_pairs = list(combinations(survivors_roster, 2))

    stage2_matches, stage2_stats = execute_matches(
        stage2_pairs,
        games_per_pair=2,
        base_seed=20202,
        workers=workers,
        stage_name="Stage 2 (Semifinals)",
    )

    stage2_ranked = sorted(
        stage2_stats.values(),
        key=lambda x: (x["points"], x["avg_margin"], x["avg_score"]),
        reverse=True,
    )

    # Cut Line: Top 4 advance to Grand Finals! Bottom 6 eliminated!
    CUT_OFF_2 = 4
    stage2_finalists = stage2_ranked[:CUT_OFF_2]
    stage2_eliminated = stage2_ranked[CUT_OFF_2:]

    print("\n" + "=" * 95)
    print("  STAGE 2 RESULTS: TOP 4 FINALISTS ADVANCE | BOTTOM 6 ELIMINATED")
    print("=" * 95)
    for i, s in enumerate(stage2_finalists, 1):
        print(f"  ★ FINALIST #{i}: {s['name']:<28} | Pts: {s['points']:>4.1f} | Win%: {s['win_rate_pct']:>5.1f}% | Avg Margin: {s['avg_margin']:>+8,.0f}")
    print("-" * 95)
    for i, e in enumerate(stage2_eliminated, CUT_OFF_2 + 1):
        print(f"  ✗ ELIMINATED #{i}: {e['name']:<26} | Pts: {e['points']:>4.1f} | Win%: {e['win_rate_pct']:>5.1f}% | Semifinal Knockout")
    print("=" * 95)

    # =========================================================================
    # STAGE 3: Grand Championship Finals (Top 4 Finalists, 4 Games/Pair)
    # =========================================================================
    finalists_roster = [roster_by_name[f["name"]] for f in stage2_finalists]
    finals_pairs = list(combinations(finalists_roster, 2))

    finals_matches, finals_stats = execute_matches(
        finals_pairs,
        games_per_pair=4,
        base_seed=30303,
        workers=workers,
        stage_name="Stage 3 (Finals)",
    )

    finals_ranked = sorted(
        finals_stats.values(),
        key=lambda x: (x["points"], x["avg_margin"], x["avg_score"]),
        reverse=True,
    )

    print("\n" + "=" * 95)
    print("  🏆 GRAND FINALS CHAMPIONSHIP PODIUM")
    print("=" * 95)
    medals = ["🥇 GOLD CHAMPION", "🥈 SILVER RUNNER-UP", "🥉 BRONZE 3RD PLACE", "4TH PLACE FINALIST"]
    for i, f in enumerate(finals_ranked):
        print(f"  {medals[i]:<20} | {f['name']:<28} | Pts: {f['points']:>4.1f} | Record: {f['wins']}-{f['losses']}-{f['ties']} | Avg Score: ${f['avg_score']:>8,.0f}")
    print("=" * 95)

    total_time = round(time.time() - tournament_start, 2)

    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "baselines_dir": baselines_dir,
        "total_contenders": len(roster),
        "total_games_played": len(stage1_matches) + len(stage2_matches) + len(finals_matches),
        "elapsed_seconds": total_time,
        "stage1": {
            "name": "The Open Qualifier",
            "matches_count": len(stage1_matches),
            "survivors": stage1_survivors,
            "eliminated": stage1_eliminated,
        },
        "stage2": {
            "name": "The Semifinal Survivor Bracket",
            "matches_count": len(stage2_matches),
            "finalists": stage2_finalists,
            "eliminated": stage2_eliminated,
        },
        "finals": {
            "name": "Grand Championship Finals",
            "matches_count": len(finals_matches),
            "podium": finals_ranked,
            "matches": finals_matches,
        },
    }


def render_html_dashboard(data: Dict[str, Any], out_path: str) -> None:
    finals = data["finals"]["podium"]
    s2_elim = data["stage2"]["eliminated"]
    s1_elim = data["stage1"]["eliminated"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Survivor Elimination Tournament Dashboard - Kaggriculture</title>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {{
  --bg-primary: #070a12;
  --bg-card: rgba(18, 24, 38, 0.75);
  --border-card: rgba(255, 255, 255, 0.08);
  --cyan: #00f0ff;
  --purple: #a855f7;
  --gold: #fbbf24;
  --silver: #cbd5e1;
  --bronze: #d97706;
  --green: #10b981;
  --red: #ef4444;
  --text-main: #f8fafc;
  --text-dim: #94a3b8;
  --text-muted: #64748b;
}}

* {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
  font-family: 'Outfit', sans-serif;
  background-color: var(--bg-primary);
  background-image: 
    radial-gradient(circle at 15% 15%, rgba(0, 240, 255, 0.05) 0%, transparent 45%),
    radial-gradient(circle at 85% 85%, rgba(168, 85, 247, 0.05) 0%, transparent 45%);
  color: var(--text-main);
  padding: 3rem 4rem;
  line-height: 1.5;
}}

.header {{
  border-bottom: 1px solid var(--border-card);
  padding-bottom: 1.5rem;
  margin-bottom: 3rem;
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
}}

.header h1 {{
  font-size: 2.6rem;
  font-weight: 800;
  background: linear-gradient(135deg, #ffffff 20%, var(--cyan) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  display: flex;
  align-items: center;
  gap: 0.8rem;
}}

.header p {{ color: var(--text-dim); margin-top: 0.4rem; font-size: 1.1rem; }}

.badge-row {{ display: flex; gap: 0.8rem; }}
.badge {{
  background: var(--bg-card);
  border: 1px solid var(--border-card);
  padding: 0.5rem 1rem;
  border-radius: 9999px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.85rem;
  color: var(--cyan);
}}

/* Podium */
.podium-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
  margin-bottom: 3.5rem;
}}

.podium-card {{
  background: var(--bg-card);
  backdrop-filter: blur(14px);
  border: 1px solid var(--border-card);
  border-radius: 1.25rem;
  padding: 2rem;
  position: relative;
  overflow: hidden;
  transition: transform 0.2s ease;
}}
.podium-card:hover {{ transform: translateY(-4px); }}

.gold {{ border-top: 4px solid var(--gold); }}
.silver {{ border-top: 4px solid var(--silver); }}
.bronze {{ border-top: 4px solid var(--bronze); }}

.podium-title {{
  font-size: 0.85rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-bottom: 0.5rem;
}}
.gold .podium-title {{ color: var(--gold); }}
.silver .podium-title {{ color: var(--silver); }}
.bronze .podium-title {{ color: var(--bronze); }}

.agent-name {{
  font-size: 1.35rem;
  font-weight: 700;
  margin-bottom: 1.25rem;
  word-break: break-all;
}}

.stat-grid {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.6rem;
  background: rgba(0, 0, 0, 0.35);
  border-radius: 0.75rem;
  padding: 1rem;
}}

.stat-box .lbl {{ font-size: 0.7rem; text-transform: uppercase; color: var(--text-muted); }}
.stat-box .val {{ font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 700; margin-top: 0.2rem; }}

/* Section Titles */
.section-title {{
  font-size: 1.6rem;
  font-weight: 700;
  margin-bottom: 1.25rem;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}}

.stage-card {{
  background: var(--bg-card);
  backdrop-filter: blur(14px);
  border: 1px solid var(--border-card);
  border-radius: 1.25rem;
  padding: 1.75rem;
  margin-bottom: 3.5rem;
}}

table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.95rem;
}}

th {{
  text-align: left;
  padding: 0.9rem 1rem;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-dim);
  border-bottom: 1px solid var(--border-card);
}}

td {{
  padding: 1rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.03);
}}

tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}

.mono {{ font-family: 'JetBrains Mono', monospace; }}

.badge-survivor {{
  background: rgba(16, 185, 129, 0.15);
  color: var(--green);
  border: 1px solid rgba(16, 185, 129, 0.3);
  padding: 0.25rem 0.65rem;
  border-radius: 0.4rem;
  font-size: 0.8rem;
  font-weight: 600;
}}

.badge-eliminated {{
  background: rgba(239, 68, 68, 0.15);
  color: var(--red);
  border: 1px solid rgba(239, 68, 68, 0.3);
  padding: 0.25rem 0.65rem;
  border-radius: 0.4rem;
  font-size: 0.8rem;
  font-weight: 600;
}}

</style>
</head>
<body>

<header class="header">
  <div>
    <h1>⚔️ Survivor Elimination Tournament</h1>
    <p>Multi-Stage Weakest-Link Pruning across Contenders in <code>{data['baselines_dir']}</code></p>
  </div>
  <div class="badge-row">
    <div class="badge">🎮 {data['total_games_played']} Matches</div>
    <div class="badge">🤖 {data['total_contenders']} Contenders</div>
    <div class="badge">⏱️ {data['elapsed_seconds']}s Runtime</div>
    <div class="badge">📅 {data['timestamp']}</div>
  </div>
</header>

<!-- Grand Finals Podium -->
<div class="podium-grid">
"""

    medals = ["gold", "silver", "bronze"]
    titles = ["🥇 Grand Champion", "🥈 Silver Runner-Up", "🥉 Bronze 3rd Place"]
    for i in range(min(3, len(finals))):
        f = finals[i]
        m = medals[i]
        t = titles[i]
        html += f"""
  <div class="podium-card {m}">
    <div class="podium-title">{t}</div>
    <div class="agent-name">{f['name']}</div>
    <div class="stat-grid">
      <div class="stat-box">
        <div class="lbl">Finals Pts</div>
        <div class="val">{f['points']:.1f}</div>
      </div>
      <div class="stat-box">
        <div class="lbl">Record</div>
        <div class="val">{f['wins']}-{f['losses']}-{f['ties']}</div>
      </div>
      <div class="stat-box">
        <div class="lbl">Avg Score</div>
        <div class="val">${f['avg_score']:,.0f}</div>
      </div>
    </div>
  </div>
"""

    html += f"""
</div>

<!-- STAGE 3: GRAND FINALS STANDINGS -->
<h2 class="section-title">🏆 Stage 3: Grand Championship Finals (Top 4 Elite Clash)</h2>
<div class="stage-card">
  <table>
    <thead>
      <tr>
        <th>Rank</th>
        <th>Contender</th>
        <th>Category</th>
        <th>Points</th>
        <th>Record (W-L-T)</th>
        <th>Win Rate</th>
        <th>Avg Margin</th>
        <th>Avg Score</th>
      </tr>
    </thead>
    <tbody>
"""

    for idx, f in enumerate(finals, 1):
        html += f"""
      <tr>
        <td class="mono" style="font-weight:700;">#{idx}</td>
        <td class="mono" style="color:var(--cyan); font-weight:600;">{f['name']}</td>
        <td>{f['tier']}</td>
        <td class="mono" style="font-weight:700;">{f['points']:.1f}</td>
        <td class="mono">{f['wins']}-{f['losses']}-{f['ties']}</td>
        <td class="mono">{f['win_rate_pct']:.1f}%</td>
        <td class="mono" style="color: {'var(--green)' if f['avg_margin'] >= 0 else 'var(--red)'};">{f['avg_margin']:+,.0f}</td>
        <td class="mono">${f['avg_score']:,.0f}</td>
      </tr>
"""

    html += f"""
    </tbody>
  </table>
</div>

<!-- STAGE 2: SEMIFINALS SURVIVORS & ELIMINATED -->
<h2 class="section-title">🔥 Stage 2: Semifinal Survivor Bracket (Top 10 Contenders)</h2>
<div class="stage-card">
  <table>
    <thead>
      <tr>
        <th>Status</th>
        <th>Contender</th>
        <th>Points</th>
        <th>Record</th>
        <th>Win Rate</th>
        <th>Avg Margin</th>
        <th>Avg Score</th>
        <th>Outcome</th>
      </tr>
    </thead>
    <tbody>
"""

    for f in data["stage2"]["finalists"]:
        html += f"""
      <tr>
        <td><span class="badge-survivor">ADVANCED</span></td>
        <td class="mono" style="font-weight:600;">{f['name']}</td>
        <td class="mono">{f['points']:.1f}</td>
        <td class="mono">{f['wins']}-{f['losses']}-{f['ties']}</td>
        <td class="mono">{f['win_rate_pct']:.1f}%</td>
        <td class="mono">{f['avg_margin']:+,.0f}</td>
        <td class="mono">${f['avg_score']:,.0f}</td>
        <td>Top 4 Finalist</td>
      </tr>
"""

    for e in s2_elim:
        html += f"""
      <tr>
        <td><span class="badge-eliminated">ELIMINATED</span></td>
        <td class="mono" style="color:var(--text-muted);">{e['name']}</td>
        <td class="mono">{e['points']:.1f}</td>
        <td class="mono">{e['wins']}-{e['losses']}-{e['ties']}</td>
        <td class="mono">{e['win_rate_pct']:.1f}%</td>
        <td class="mono">{e['avg_margin']:+,.0f}</td>
        <td class="mono">${e['avg_score']:,.0f}</td>
        <td>Semifinal Cut Line</td>
      </tr>
"""

    html += f"""
    </tbody>
  </table>
</div>

<!-- STAGE 1: GRAVEYARD OF ELIMINATED WEAK LINKS -->
<h2 class="section-title">🪦 Stage 1: The Qualifier Cut Line & Eliminated Weakest Links</h2>
<div class="stage-card">
  <table>
    <thead>
      <tr>
        <th>Status</th>
        <th>Contender</th>
        <th>Category</th>
        <th>Qualifier Pts</th>
        <th>Record</th>
        <th>Win Rate</th>
        <th>Net Margin</th>
        <th>Elimination Cause</th>
      </tr>
    </thead>
    <tbody>
"""

    for e in s1_elim:
        html += f"""
      <tr>
        <td><span class="badge-eliminated">KNOCKED OUT</span></td>
        <td class="mono" style="color:var(--text-muted);">{e['name']}</td>
        <td>{e['tier']}</td>
        <td class="mono">{e['points']:.1f}</td>
        <td class="mono">{e['wins']}-{e['losses']}-{e['ties']}</td>
        <td class="mono">{e['win_rate_pct']:.1f}%</td>
        <td class="mono" style="color:var(--red);">{e['avg_margin']:+,.0f}</td>
        <td>Sub-45% Win Rate / Negative Margin</td>
      </tr>
"""

    html += """
    </tbody>
  </table>
</div>

</body>
</html>
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n[Dashboard Rendered] Saved survivor tournament dashboard to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-Stage Weakest-Link Elimination Tournament")
    parser.add_argument("--baselines-dir", default=BASELINES_DIR, help="Baselines directory (default: scratch /Users/sweeden/muzero/baselines)")
    parser.add_argument("--workers", type=int, default=11, help="Parallel workers")
    parser.add_argument("--save-json", default=DEFAULT_JSON, help="JSON output file")
    parser.add_argument("--save-html", default=DEFAULT_HTML, help="HTML output file")
    args = parser.parse_args()

    results = run_survivor_tournament(baselines_dir=args.baselines_dir, workers=args.workers)

    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved JSON] Report saved to {args.save_json}")

    render_html_dashboard(results, args.save_html)
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
