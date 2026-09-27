#!/usr/bin/env python3
"""
tournament_baselines_volume.py
==============================
Executes an exhaustive round-robin head-to-head tournament for all baselines
under /Volumes/BASELINES/muzero/baselines/.

Discovers both official Kaggle scored standards and operational candidate packages,
runs fair alternating-seat matches in isolated sandboxes, compiles an N x N
head-to-head cross-matrix, and generates an interactive HTML dashboard.
"""
from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from evaluation.harness import eval_match_worker, extract_agent, short_name

BASELINES_DIR = "/Volumes/BASELINES/muzero/baselines"
DEFAULT_JSON = os.path.join(HERE, "artifacts", "tournament_baselines_volume.json")
DEFAULT_HTML = os.path.join(HERE, "artifacts", "tournament_baselines_volume_dashboard.html")


def discover_baselines_agents(base_dir: str) -> List[Dict[str, Any]]:
    """Discovers all valid agent archives and standalone scripts in base_dir."""
    agents = []
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    for fn in sorted(os.listdir(base_dir)):
        if fn.startswith(".") or fn.startswith("._") or fn == ".DS_Store":
            continue
        p = os.path.join(base_dir, fn)
        if not (os.path.isfile(p) and (fn.endswith(".zip") or fn.endswith(".tar.gz") or fn == "main.py")):
            continue

        # Tag category and role
        name = short_name(fn)
        if " 2" in name:
            name = name.replace(" 2", "_v2")
            clean_tag = f"{name} (Promoted Snapshot)"
        else:
            clean_tag = name

        category = "Candidate"
        tier = "Tier 3 (Neural / Boost)" if "boost" in fn else ("Tier 3 (Neural / Train)" if "train" in fn else "Tier 1-2 (Official Kaggle)")

        if fn.startswith("submission_20") or fn.startswith("submission_0771") or fn == "main.py":
            category = "Official Kaggle Baseline"
            tier = "Tier 1-2 (Kaggle Benchmark)"

        agents.append({
            "path": p,
            "filename": fn,
            "name": name,
            "display_name": clean_tag,
            "category": category,
            "tier": tier,
            "size": os.path.getsize(p),
        })

    return agents


def run_tournament(
    agents: List[Dict[str, Any]],
    games_per_pair: int = 2,
    base_seed: int = 10101,
    workers: int = 10,
    quiet: bool = False,
) -> Dict[str, Any]:
    agent_paths = [a["path"] for a in agents]
    agent_map = {a["path"]: a for a in agents}
    pairs = list(combinations(agent_paths, 2))
    total_matches = len(pairs) * games_per_pair

    if not quiet:
        print("=" * 105)
        print(f"  HEAD-TO-HEAD GRAND TOURNAMENT: /Volumes/BASELINES/muzero/baselines")
        print(f"  Total Agents: {len(agents)} | Total Pairings: {len(pairs)} | Total Games: {total_matches} ({games_per_pair}/pair) | Workers: {workers}")
        print("=" * 105)

    leaderboard: Dict[str, Dict[str, Any]] = {
        p: {
            "path": p,
            "name": agent_map[p]["name"],
            "display_name": agent_map[p]["display_name"],
            "category": agent_map[p]["category"],
            "tier": agent_map[p]["tier"],
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "points": 0.0,
            "score_sum": 0.0,
            "opp_score_sum": 0.0,
            "games": 0,
            "h2h_vs": {},  # opponent_name -> {"wins": x, "losses": y, "ties": z, "net_delta": d}
        }
        for p in agent_paths
    }

    # Initialize h2h matrix
    for p_a in agent_paths:
        for p_b in agent_paths:
            if p_a != p_b:
                name_b = agent_map[p_b]["name"]
                leaderboard[p_a]["h2h_vs"][name_b] = {"wins": 0, "losses": 0, "ties": 0, "score_a": 0.0, "score_b": 0.0, "net_delta": 0.0}

    tasks: List[Dict[str, Any]] = []
    game_idx = 0
    for p_a, p_b in pairs:
        for g in range(games_per_pair):
            game_idx += 1
            a_is_p0 = (g % 2 == 0)
            seed = base_seed + (game_idx * 17)
            tasks.append({
                "game_idx": game_idx,
                "agent_a": p_a,
                "agent_b": p_b,
                "a_is_p0": a_is_p0,
                "seed": seed,
                "episode_steps": 720,
                "save_replay_dir": None,
                "min_replay_score": 80000.0,
            })

    t0 = time.time()
    completed_matches = 0
    match_log: List[Dict[str, Any]] = []

    def handle_result(res: Dict[str, Any]) -> None:
        nonlocal completed_matches
        completed_matches += 1
        p_a = res["task"]["agent_a"]
        p_b = res["task"]["agent_b"]
        a_info = agent_map[p_a]
        b_info = agent_map[p_b]
        st_a = leaderboard[p_a]
        st_b = leaderboard[p_b]

        s_a = res["score_a"]
        s_b = res["score_b"]
        w = res["winner"]

        st_a["games"] += 1
        st_b["games"] += 1
        st_a["score_sum"] += s_a
        st_b["score_sum"] += s_b
        st_a["opp_score_sum"] += s_b
        st_b["opp_score_sum"] += s_a

        h2h_ab = st_a["h2h_vs"][b_info["name"]]
        h2h_ba = st_b["h2h_vs"][a_info["name"]]
        h2h_ab["score_a"] += s_a
        h2h_ab["score_b"] += s_b
        h2h_ab["net_delta"] += (s_a - s_b)
        h2h_ba["score_a"] += s_b
        h2h_ba["score_b"] += s_a
        h2h_ba["net_delta"] += (s_b - s_a)

        if w == "A":
            st_a["wins"] += 1
            st_a["points"] += 1.0
            st_b["losses"] += 1
            h2h_ab["wins"] += 1
            h2h_ba["losses"] += 1
            w_name = a_info["name"]
        elif w == "B":
            st_b["wins"] += 1
            st_b["points"] += 1.0
            st_a["losses"] += 1
            h2h_ab["losses"] += 1
            h2h_ba["wins"] += 1
            w_name = b_info["name"]
        else:
            st_a["ties"] += 1
            st_b["ties"] += 1
            st_a["points"] += 0.5
            st_b["points"] += 0.5
            h2h_ab["ties"] += 1
            h2h_ba["ties"] += 1
            w_name = "TIE"

        match_log.append({
            "game_idx": res["task"]["game_idx"],
            "seed": res["seed"],
            "agent_a": a_info["name"],
            "agent_b": b_info["name"],
            "p0": a_info["name"] if res["task"]["a_is_p0"] else b_info["name"],
            "score_a": s_a,
            "score_b": s_b,
            "net_delta_a": s_a - s_b,
            "winner": w_name,
            "elapsed_sec": res["elapsed_sec"],
        })

        if not quiet and (completed_matches % 10 == 0 or completed_matches == total_matches):
            rate = completed_matches / max(0.1, time.time() - t0)
            eta = (total_matches - completed_matches) / max(0.01, rate)
            print(f"  [{completed_matches:>3}/{total_matches}] ({completed_matches/total_matches*100:5.1f}%) | {a_info['name'][:22]:<22} (${s_a:>8,.0f}) vs {b_info['name'][:22]:<22} (${s_b:>8,.0f}) → {w_name:<16} | ETA: {eta:>4.0f}s")

    if workers > 1:
        with mp.Pool(processes=min(workers, len(tasks))) as pool:
            for r in pool.imap_unordered(eval_match_worker, tasks):
                handle_result(r)
    else:
        for t in tasks:
            r = eval_match_worker(t)
            handle_result(r)

    total_time = round(time.time() - t0, 2)

    # Compute rankings
    ranked = sorted(
        leaderboard.values(),
        key=lambda x: (x["points"], (x["score_sum"] - x["opp_score_sum"]) / max(1, x["games"]), x["score_sum"] / max(1, x["games"])),
        reverse=True,
    )

    for r_idx, entry in enumerate(ranked, 1):
        entry["rank"] = r_idx
        entry["avg_score"] = entry["score_sum"] / max(1, entry["games"])
        entry["avg_opp_score"] = entry["opp_score_sum"] / max(1, entry["games"])
        entry["avg_margin"] = (entry["score_sum"] - entry["opp_score_sum"]) / max(1, entry["games"])
        entry["win_rate_pct"] = (100.0 * entry["wins"] / max(1, entry["games"]))

    if not quiet:
        print("\n" + "=" * 105)
        print("  FINAL LEAGUE RANKINGS ACROSS ALL BASELINES")
        print("=" * 105)
        print(f"  {'Rank':>4} | {'Agent Name':<30} | {'Category':<16} | {'Points':>6} | {'W-L-T':^11} | {'Win %':>6} | {'Avg Margin':>11} | {'Avg Score':>11}")
        print("-" * 105)
        for r in ranked:
            wlt = f"{r['wins']}-{r['losses']}-{r['ties']}"
            print(f"  {r['rank']:>4} | {r['name']:<30} | {r['category'][:16]:<16} | {r['points']:>6.1f} | {wlt:^11} | {r['win_rate_pct']:>5.1f}% | {r['avg_margin']:>+10,.0f} | ${r['avg_score']:>10,.0f}")
        print("=" * 105)

    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "baselines_dir": BASELINES_DIR,
        "total_agents": len(agents),
        "total_games": total_matches,
        "games_per_pair": games_per_pair,
        "elapsed_seconds": total_time,
        "rankings": ranked,
        "matches": match_log,
    }


def render_html_dashboard(report: Dict[str, Any], out_path: str) -> None:
    """Generates an aesthetic, interactive HTML dashboard."""
    rankings = report["rankings"]
    matches = report["matches"]

    # Names list for matrix
    names = [r["name"] for r in rankings]

    # Build H2H lookup: name_a -> name_b -> data
    h2h_data = {r["name"]: r["h2h_vs"] for r in rankings}

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kaggriculture Baselines Grand Tournament Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {{
  --bg-primary: #0a0e17;
  --bg-card: rgba(22, 30, 49, 0.7);
  --bg-card-hover: rgba(30, 41, 67, 0.9);
  --border-card: rgba(255, 255, 255, 0.08);
  --border-accent: rgba(0, 240, 255, 0.3);
  --text-primary: #f0f4fc;
  --text-secondary: #94a3b8;
  --text-muted: #64748b;
  --cyan: #00f0ff;
  --purple: #8b5cf6;
  --gold: #f59e0b;
  --silver: #cbd5e1;
  --bronze: #d97706;
  --green: #10b981;
  --red: #ef4444;
}}

* {{
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}}

body {{
  font-family: 'Outfit', sans-serif;
  background-color: var(--bg-primary);
  background-image: 
    radial-gradient(circle at 10% 20%, rgba(0, 240, 255, 0.04) 0%, transparent 40%),
    radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.04) 0%, transparent 40%);
  color: var(--text-primary);
  line-height: 1.5;
  padding: 2.5rem 3rem;
  min-height: 100vh;
}}

.header {{
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  margin-bottom: 2.5rem;
  border-bottom: 1px solid var(--border-card);
  padding-bottom: 1.5rem;
}}

.header-title h1 {{
  font-size: 2.4rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  background: linear-gradient(135deg, #ffffff 0%, var(--cyan) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  display: flex;
  align-items: center;
  gap: 0.75rem;
}}

.header-title p {{
  color: var(--text-secondary);
  font-size: 1.05rem;
  margin-top: 0.3rem;
}}

.meta-badges {{
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
}}

.badge {{
  background: var(--bg-card);
  border: 1px solid var(--border-card);
  padding: 0.45rem 0.9rem;
  border-radius: 9999px;
  font-size: 0.85rem;
  font-family: 'JetBrains Mono', monospace;
  color: var(--cyan);
  display: flex;
  align-items: center;
  gap: 0.4rem;
}}

/* Podium Cards */
.podium-container {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 1.5rem;
  margin-bottom: 3rem;
}}

.podium-card {{
  background: var(--bg-card);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border-card);
  border-radius: 1.25rem;
  padding: 1.75rem;
  position: relative;
  overflow: hidden;
  transition: transform 0.2s ease, border-color 0.2s ease;
}}

.podium-card:hover {{
  transform: translateY(-4px);
  border-color: var(--border-accent);
}}

.card-gold {{ border-top: 4px solid var(--gold); }}
.card-silver {{ border-top: 4px solid var(--silver); }}
.card-bronze {{ border-top: 4px solid var(--bronze); }}

.podium-rank {{
  font-size: 0.85rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-bottom: 0.5rem;
}}
.card-gold .podium-rank {{ color: var(--gold); }}
.card-silver .podium-rank {{ color: var(--silver); }}
.card-bronze .podium-rank {{ color: var(--bronze); }}

.podium-name {{
  font-size: 1.35rem;
  font-weight: 700;
  word-break: break-all;
  margin-bottom: 1rem;
}}

.podium-stats {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.75rem;
  background: rgba(0, 0, 0, 0.25);
  border-radius: 0.75rem;
  padding: 1rem;
}}

.stat-item .label {{
  font-size: 0.75rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}}

.stat-item .val {{
  font-size: 1.15rem;
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
  color: var(--text-primary);
  margin-top: 0.2rem;
}}

/* Section Titles */
.section-title {{
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 1.25rem;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}}

/* Tables */
.table-card {{
  background: var(--bg-card);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border-card);
  border-radius: 1.25rem;
  padding: 1.5rem;
  margin-bottom: 3rem;
  overflow-x: auto;
}}

table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.95rem;
}}

th {{
  text-align: left;
  padding: 0.9rem 1rem;
  color: var(--text-secondary);
  font-weight: 600;
  border-bottom: 1px solid var(--border-card);
  text-transform: uppercase;
  font-size: 0.8rem;
  letter-spacing: 0.05em;
}}

td {{
  padding: 1rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.03);
}}

tr:hover td {{
  background: rgba(255, 255, 255, 0.02);
}}

.rank-badge {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 2rem;
  height: 2rem;
  border-radius: 50%;
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.85rem;
  background: rgba(255, 255, 255, 0.05);
}}

.rank-1 {{ background: rgba(245, 158, 11, 0.2); color: var(--gold); border: 1px solid var(--gold); }}
.rank-2 {{ background: rgba(203, 213, 225, 0.2); color: var(--silver); border: 1px solid var(--silver); }}
.rank-3 {{ background: rgba(217, 119, 6, 0.2); color: var(--bronze); border: 1px solid var(--bronze); }}

.agent-col {{
  font-family: 'JetBrains Mono', monospace;
  font-weight: 600;
  color: var(--text-primary);
}}

.tag-badge {{
  display: inline-block;
  font-size: 0.75rem;
  padding: 0.2rem 0.6rem;
  border-radius: 0.4rem;
  margin-left: 0.5rem;
  font-weight: 500;
  font-family: 'Outfit', sans-serif;
}}
.tag-kaggle {{ background: rgba(0, 240, 255, 0.15); color: var(--cyan); }}
.tag-boost {{ background: rgba(139, 92, 246, 0.15); color: var(--purple); }}
.tag-train {{ background: rgba(16, 185, 129, 0.15); color: var(--green); }}

.mono {{
  font-family: 'JetBrains Mono', monospace;
}}

/* Head to Head Matrix Grid */
.matrix-card {{
  background: var(--bg-card);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border-card);
  border-radius: 1.25rem;
  padding: 1.5rem;
  margin-bottom: 3rem;
  overflow-x: auto;
}}

.matrix-table {{
  font-size: 0.8rem;
  border-collapse: separate;
  border-spacing: 2px;
}}

.matrix-table th {{
  padding: 0.5rem;
  font-size: 0.75rem;
  text-align: center;
  max-width: 90px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}

.matrix-table td {{
  padding: 0.45rem;
  text-align: center;
  border: none;
  border-radius: 4px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
}}

.cell-self {{
  background: rgba(255, 255, 255, 0.05);
  color: var(--text-muted);
}}

.cell-win {{
  background: rgba(16, 185, 129, 0.25);
  color: #34d399;
  border: 1px solid rgba(16, 185, 129, 0.4);
}}

.cell-loss {{
  background: rgba(239, 68, 68, 0.25);
  color: #f87171;
  border: 1px solid rgba(239, 68, 68, 0.4);
}}

.cell-tie {{
  background: rgba(245, 158, 11, 0.2);
  color: #fbbf24;
  border: 1px solid rgba(245, 158, 11, 0.3);
}}

/* Search Filter */
.filter-bar {{
  display: flex;
  gap: 1rem;
  margin-bottom: 1.5rem;
}}

.search-input {{
  background: var(--bg-card);
  border: 1px solid var(--border-card);
  color: var(--text-primary);
  padding: 0.6rem 1rem;
  border-radius: 0.6rem;
  font-family: 'Outfit', sans-serif;
  font-size: 0.95rem;
  width: 320px;
}}
.search-input:focus {{
  outline: none;
  border-color: var(--cyan);
}}

</style>
</head>
<body>

<header class="header">
  <div class="header-title">
    <h1>⚔️ Kaggriculture Grand Baselines Tournament</h1>
    <p>Comprehensive Head-to-Head Evaluation across all Baselines in <code>/Volumes/BASELINES/muzero/baselines</code></p>
  </div>
  <div class="meta-badges">
    <div class="badge">🎮 {report['total_games']} Games</div>
    <div class="badge">🤖 {report['total_agents']} Agents</div>
    <div class="badge">⏱️ {report['elapsed_seconds']}s Runtime</div>
    <div class="badge">📅 {report['timestamp']}</div>
  </div>
</header>

<!-- Podium Top 3 -->
<div class="podium-container">
"""

    medals = ["gold", "silver", "bronze"]
    for i in range(min(3, len(rankings))):
        r = rankings[i]
        m = medals[i]
        html += f"""
  <div class="podium-card card-{m}">
    <div class="podium-rank">#{r['rank']} Podium Champion</div>
    <div class="podium-name">{r['name']}</div>
    <div class="podium-stats">
      <div class="stat-item">
        <div class="label">Points</div>
        <div class="val">{r['points']:.1f}</div>
      </div>
      <div class="stat-item">
        <div class="label">Win Rate</div>
        <div class="val">{r['win_rate_pct']:.1f}%</div>
      </div>
      <div class="stat-item">
        <div class="label">Avg Score</div>
        <div class="val">${r['avg_score']:,.0f}</div>
      </div>
    </div>
  </div>
"""

    html += """
</div>

<!-- Standings Table -->
<h2 class="section-title">🏆 League Standings</h2>
<div class="table-card">
  <table id="standingsTable">
    <thead>
      <tr>
        <th style="width: 60px;">Rank</th>
        <th>Agent Package</th>
        <th>Category / Tier</th>
        <th>Points</th>
        <th>Record (W-L-T)</th>
        <th>Win Rate</th>
        <th>Avg Margin</th>
        <th>Avg Score</th>
      </tr>
    </thead>
    <tbody>
"""

    for r in rankings:
        rk_cls = f"rank-{r['rank']}" if r["rank"] <= 3 else ""
        cat_tag = "tag-kaggle" if "Kaggle" in r["category"] else ("tag-boost" if "boost" in r["name"] else "tag-train")
        html += f"""
      <tr>
        <td><span class="rank-badge {rk_cls}">{r['rank']}</span></td>
        <td class="agent-col">{r['name']}</td>
        <td><span class="tag-badge {cat_tag}">{r['tier']}</span></td>
        <td class="mono" style="font-weight: 700; color: var(--cyan);">{r['points']:.1f}</td>
        <td class="mono">{r['wins']}-{r['losses']}-{r['ties']}</td>
        <td class="mono">{r['win_rate_pct']:.1f}%</td>
        <td class="mono" style="color: {'var(--green)' if r['avg_margin'] >= 0 else 'var(--red)'};">{r['avg_margin']:+,.0f}</td>
        <td class="mono" style="font-weight: 600;">${r['avg_score']:,.0f}</td>
      </tr>
"""

    html += """
    </tbody>
  </table>
</div>

<!-- Head-to-Head Cross Matrix -->
<h2 class="section-title">📊 Head-to-Head Matchup Matrix</h2>
<p style="color: var(--text-secondary); margin-bottom: 1rem;">Row agent vs Column opponent: shows W-L-T record and net score margin (Row minus Column).</p>
<div class="matrix-card">
  <table class="matrix-table">
    <thead>
      <tr>
        <th style="text-align: left; position: sticky; left: 0; background: var(--bg-card); z-index: 2;">Row vs Col</th>
"""

    for col_name in names:
        short_col = col_name.replace("submission_", "").replace("boost_", "b_").replace("train_", "t_")
        html += f'<th title="{col_name}">{short_col}</th>'

    html += """
      </tr>
    </thead>
    <tbody>
"""

    for row_name in names:
        short_row = row_name.replace("submission_", "").replace("boost_", "b_").replace("train_", "t_")
        html += f'<tr><td style="text-align: left; font-weight: 600; position: sticky; left: 0; background: var(--bg-card); z-index: 1;">{short_row}</td>'
        for col_name in names:
            if row_name == col_name:
                html += '<td class="cell-self">—</td>'
            else:
                cell = h2h_data[row_name].get(col_name, {"wins": 0, "losses": 0, "ties": 0, "net_delta": 0.0})
                w = cell["wins"]
                l = cell["losses"]
                t = cell["ties"]
                d = cell["net_delta"]
                rec = f"{w}-{l}" + (f"-{t}" if t > 0 else "")
                delta_str = f"{d:+,.0f}"

                if w > l:
                    cls = "cell-win"
                elif l > w:
                    cls = "cell-loss"
                else:
                    cls = "cell-tie"

                html += f'<td class="{cls}" title="{row_name} vs {col_name}: {rec} (Δ {delta_str})">{rec}<br><span style="font-size:0.65rem; opacity:0.8;">{delta_str}</span></td>'
        html += '</tr>'

    html += """
    </tbody>
  </table>
</div>

<!-- Match Logs Recent Sample -->
<h2 class="section-title">📜 Individual Match Log (Sample of Head-to-Head Clashes)</h2>
<div class="table-card">
  <table>
    <thead>
      <tr>
        <th>Game #</th>
        <th>Agent A</th>
        <th>Agent B</th>
        <th>P0 Seat</th>
        <th>Score A</th>
        <th>Score B</th>
        <th>Winner</th>
        <th>Seed</th>
      </tr>
    </thead>
    <tbody>
"""

    for m in matches[:60]:
        html += f"""
      <tr>
        <td class="mono">#{m['game_idx']}</td>
        <td class="mono">{m['agent_a']}</td>
        <td class="mono">{m['agent_b']}</td>
        <td class="mono">{m['p0']}</td>
        <td class="mono">${m['score_a']:,.0f}</td>
        <td class="mono">${m['score_b']:,.0f}</td>
        <td class="mono" style="font-weight: 700; color: var(--cyan);">{m['winner']}</td>
        <td class="mono">{m['seed']}</td>
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
    print(f"\n[Dashboard Rendered] Saved interactive HTML to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Head-to-head tournament for /Volumes/BASELINES/muzero/baselines")
    parser.add_argument("--games-per-pair", type=int, default=2, help="Games per pairing with alternating seats (default: 2)")
    parser.add_argument("--seed", type=int, default=10101, help="Base seed")
    parser.add_argument("--workers", type=int, default=10, help="Parallel worker processes (default: 10)")
    parser.add_argument("--save-json", default=DEFAULT_JSON, help="JSON output file")
    parser.add_argument("--save-html", default=DEFAULT_HTML, help="HTML dashboard output file")
    parser.add_argument("--official-only", action="store_true", help="Only run official Kaggle scored standards")
    args = parser.parse_args()

    agents = discover_baselines_agents(BASELINES_DIR)
    if args.official_only:
        agents = [a for a in agents if a["category"] == "Official Kaggle Baseline"]

    print(f"Discovered {len(agents)} eligible baselines in {BASELINES_DIR}:")
    for a in agents:
        print(f"  - {a['name']:30s} | {a['category']:24s} | {a['size']:,} bytes")

    report = run_tournament(
        agents,
        games_per_pair=args.games_per_pair,
        base_seed=args.seed,
        workers=args.workers,
        quiet=False,
    )

    # Save JSON
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Saved JSON] Report written to {args.save_json}")

    # Render HTML Dashboard
    render_html_dashboard(report, args.save_html)

    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
