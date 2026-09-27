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
render_tournament_dashboard.py - Interactive Tournament Visualizer & Dashboard
=============================================================================
Generates a modern, dark-mode interactive HTML visualization of the
all-vs-all tournament results, pairwise cross-tables, match logs, and
replays from artifacts/league_staged_candidates.json.
=============================================================================
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(HERE, "artifacts", "league_staged_candidates.json")
DEFAULT_OUTPUT = os.path.join(HERE, "artifacts", "tournament_dashboard.html")
REPLAY_HTML_REL = "finals_cand010_vs_cand012_replay.html"


def generate_dashboard_html(data: Dict[str, Any]) -> str:
    rankings = data.get("rankings", [])
    matches = data.get("matches", [])
    total_games = data.get("total_games", len(matches))
    elapsed_seconds = data.get("elapsed_seconds", 0)

    # Compute pairwise matrix
    agents = [r["name"] for r in rankings]
    pairwise = defaultdict(lambda: defaultdict(lambda: {"w": 0, "l": 0, "t": 0, "sa": 0, "sb": 0}))

    for m in matches:
        a = m["agent_a"]
        b = m["agent_b"]
        pairwise[a][b]["sa"] += m["score_a"]
        pairwise[a][b]["sb"] += m["score_b"]
        pairwise[b][a]["sa"] += m["score_b"]
        pairwise[b][a]["sb"] += m["score_a"]

        if m["winner"] == "A":
            pairwise[a][b]["w"] += 1
            pairwise[b][a]["l"] += 1
        elif m["winner"] == "B":
            pairwise[a][b]["l"] += 1
            pairwise[b][a]["w"] += 1
        else:
            pairwise[a][b]["t"] += 1
            pairwise[b][a]["t"] += 1

    # Format JSON data to embed for client-side interactivity
    json_rankings = json.dumps(rankings)
    json_matches = json.dumps(matches)
    json_agents = json.dumps(agents)
    json_pairwise = json.dumps(pairwise)

    # Candidate short badges and origins
    def get_origin(name: str) -> str:
        if "boost" in name:
            return "Inverted Boost (Adam)"
        if "train" in name:
            return "Standard Train (LoRA)"
        return "Baseline"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Kaggriculture MuZero League Tournament Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-base: #090d16;
      --bg-surface: #111726;
      --bg-card: rgba(22, 30, 49, 0.7);
      --bg-card-hover: rgba(30, 41, 67, 0.85);
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-accent: rgba(56, 189, 248, 0.25);
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --gold: #fbbf24;
      --gold-glow: rgba(251, 191, 36, 0.2);
      --silver: #cbd5e1;
      --silver-glow: rgba(203, 213, 225, 0.15);
      --bronze: #d97706;
      --bronze-glow: rgba(217, 119, 6, 0.2);
      --accent-cyan: #38bdf8;
      --accent-emerald: #34d399;
      --accent-rose: #f43f5e;
      --accent-indigo: #818cf8;
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: radial-gradient(circle at 50% 0%, #172554 0%, var(--bg-base) 65%);
      color: var(--text-main);
      font-family: var(--font-sans);
      min-height: 100vh;
      padding: 2.5rem 1.5rem;
      line-height: 1.5;
    }}

    .container {{
      max-width: 1400px;
      margin: 0 auto;
    }}

    /* Header */
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      margin-bottom: 2.5rem;
      border-bottom: 1px solid var(--border-subtle);
      padding-bottom: 1.5rem;
    }}
    .title-group h1 {{
      font-size: 2.4rem;
      font-weight: 800;
      letter-spacing: -0.03em;
      background: linear-gradient(135deg, #ffffff 40%, var(--accent-cyan) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.35rem;
    }}
    .subtitle {{
      color: var(--text-muted);
      font-size: 0.95rem;
    }}
    .meta-badges {{
      display: flex;
      gap: 0.75rem;
    }}
    .meta-badge {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      padding: 0.4rem 0.85rem;
      border-radius: 9999px;
      font-size: 0.8rem;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }}
    .meta-badge strong {{
      color: var(--text-main);
    }}

    /* Podium Section */
    .podium-section {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 1.5rem;
      margin-bottom: 3rem;
      align-items: end;
    }}
    .podium-card {{
      background: var(--bg-card);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-subtle);
      border-radius: 1.25rem;
      padding: 1.75rem;
      position: relative;
      overflow: hidden;
      transition: transform 0.25s ease, box-shadow 0.25s ease;
    }}
    .podium-card:hover {{
      transform: translateY(-4px);
    }}
    .podium-card.rank-1 {{
      border-color: var(--gold);
      box-shadow: 0 10px 35px var(--gold-glow);
      order: 2;
      padding-top: 2.25rem;
      padding-bottom: 2.25rem;
    }}
    .podium-card.rank-2 {{
      border-color: var(--silver);
      box-shadow: 0 8px 25px var(--silver-glow);
      order: 1;
    }}
    .podium-card.rank-3 {{
      border-color: var(--bronze);
      box-shadow: 0 8px 25px var(--bronze-glow);
      order: 3;
    }}
    .crown-badge {{
      position: absolute;
      top: 1rem;
      right: 1.25rem;
      font-size: 1.8rem;
    }}
    .podium-rank {{
      font-size: 0.85rem;
      text-transform: uppercase;
      font-weight: 700;
      letter-spacing: 0.08em;
      margin-bottom: 0.5rem;
    }}
    .rank-1 .podium-rank {{ color: var(--gold); }}
    .rank-2 .podium-rank {{ color: var(--silver); }}
    .rank-3 .podium-rank {{ color: var(--bronze); }}

    .podium-title {{
      font-size: 1.2rem;
      font-weight: 700;
      font-family: var(--font-mono);
      margin-bottom: 0.25rem;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .podium-method {{
      font-size: 0.8rem;
      color: var(--text-dim);
      margin-bottom: 1.25rem;
    }}
    .podium-stats {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
      border-top: 1px solid var(--border-subtle);
      padding-top: 1rem;
    }}
    .stat-item .label {{
      font-size: 0.7rem;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .stat-item .val {{
      font-size: 1.25rem;
      font-weight: 700;
      font-family: var(--font-mono);
      color: var(--text-main);
    }}

    /* Action Banner */
    .banner-card {{
      background: linear-gradient(135deg, rgba(30, 58, 138, 0.4) 0%, rgba(15, 23, 42, 0.6) 100%);
      border: 1px solid rgba(56, 189, 248, 0.3);
      border-radius: 1.25rem;
      padding: 1.5rem 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 3rem;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }}
    .banner-text h3 {{
      font-size: 1.25rem;
      font-weight: 700;
      margin-bottom: 0.35rem;
      color: #fff;
    }}
    .banner-text p {{
      color: var(--text-muted);
      font-size: 0.9rem;
    }}
    .btn {{
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      background: linear-gradient(135deg, var(--accent-cyan) 0%, #0284c7 100%);
      color: #03132e;
      font-weight: 700;
      font-size: 0.9rem;
      padding: 0.75rem 1.4rem;
      border-radius: 0.75rem;
      text-decoration: none;
      transition: all 0.2s ease;
      box-shadow: 0 4px 15px rgba(56, 189, 248, 0.35);
    }}
    .btn:hover {{
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(56, 189, 248, 0.5);
    }}

    /* Section Cards */
    .section-card {{
      background: var(--bg-card);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-subtle);
      border-radius: 1.25rem;
      padding: 2rem;
      margin-bottom: 3rem;
    }}
    .section-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.5rem;
    }}
    .section-title {{
      font-size: 1.35rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }}

    /* Standings Table */
    .table-wrapper {{
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      text-align: left;
    }}
    th {{
      font-size: 0.75rem;
      text-transform: uppercase;
      color: var(--text-dim);
      letter-spacing: 0.06em;
      padding: 0.85rem 1rem;
      border-bottom: 1px solid var(--border-subtle);
    }}
    td {{
      padding: 1rem;
      font-size: 0.92rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }}
    tr:hover td {{
      background: rgba(255, 255, 255, 0.02);
    }}
    .rank-num {{
      font-weight: 800;
      font-family: var(--font-mono);
      font-size: 1.1rem;
      width: 3rem;
    }}
    .agent-cell {{
      font-family: var(--font-mono);
      font-weight: 600;
      color: var(--text-main);
    }}
    .tag {{
      display: inline-block;
      font-size: 0.7rem;
      font-family: var(--font-sans);
      padding: 0.2rem 0.55rem;
      border-radius: 0.4rem;
      font-weight: 600;
      margin-left: 0.5rem;
    }}
    .tag-boost {{ background: rgba(56, 189, 248, 0.15); color: var(--accent-cyan); }}
    .tag-train {{ background: rgba(52, 211, 153, 0.15); color: var(--accent-emerald); }}
    .val-points {{
      font-family: var(--font-mono);
      font-size: 1.1rem;
      font-weight: 700;
      color: var(--accent-cyan);
    }}
    .val-score {{
      font-family: var(--font-mono);
      font-weight: 600;
    }}

    /* Pairwise Matrix */
    .matrix-table {{
      font-family: var(--font-mono);
      font-size: 0.82rem;
    }}
    .matrix-table th, .matrix-table td {{
      padding: 0.65rem 0.5rem;
      text-align: center;
    }}
    .matrix-table th:first-child, .matrix-table td:first-child {{
      text-align: left;
      font-weight: 600;
      min-width: 140px;
    }}
    .cell-diag {{
      background: rgba(255, 255, 255, 0.02);
      color: var(--text-dim);
    }}
    .cell-win {{
      background: rgba(52, 211, 153, 0.15);
      color: var(--accent-emerald);
      font-weight: 700;
      border-radius: 0.35rem;
    }}
    .cell-loss {{
      background: rgba(244, 63, 94, 0.12);
      color: var(--accent-rose);
      border-radius: 0.35rem;
    }}
    .cell-tie {{
      background: rgba(251, 191, 36, 0.15);
      color: var(--gold);
      border-radius: 0.35rem;
    }}

    /* Matches List */
    .filter-bar {{
      display: flex;
      gap: 1rem;
      margin-bottom: 1.25rem;
    }}
    .search-input, .select-input {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      padding: 0.6rem 1rem;
      border-radius: 0.6rem;
      font-size: 0.85rem;
      font-family: var(--font-sans);
    }}
    .search-input {{ flex: 1; }}
    .search-input:focus, .select-input:focus {{
      outline: none;
      border-color: var(--accent-cyan);
    }}
    .match-item {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(255, 255, 255, 0.015);
      border: 1px solid rgba(255, 255, 255, 0.04);
      padding: 0.85rem 1.25rem;
      border-radius: 0.75rem;
      margin-bottom: 0.5rem;
      font-size: 0.88rem;
    }}
    .match-players {{
      display: flex;
      align-items: center;
      gap: 1rem;
      flex: 1;
    }}
    .player-entry {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
      min-width: 210px;
      font-family: var(--font-mono);
    }}
    .player-entry.winner {{
      font-weight: 700;
      color: var(--accent-emerald);
    }}
    .seat-badge {{
      font-size: 0.65rem;
      padding: 0.15rem 0.4rem;
      border-radius: 0.3rem;
      background: rgba(255, 255, 255, 0.08);
      color: var(--text-muted);
    }}
    .score-badge {{
      margin-left: auto;
      font-family: var(--font-mono);
    }}
    .vs-sep {{
      color: var(--text-dim);
      font-weight: 700;
      font-size: 0.75rem;
    }}

    /* Footer */
    .footer {{
      text-align: center;
      margin-top: 4rem;
      color: var(--text-dim);
      font-size: 0.82rem;
      border-top: 1px solid var(--border-subtle);
      padding-top: 2rem;
    }}
  </style>
</head>
<body>
  <div class="container">
    <!-- Header -->
    <header class="header">
      <div class="title-group">
        <h1>Kaggriculture Tournament League</h1>
        <p class="subtitle">Operational Candidates Championship &bull; All-vs-All Round-Robin Matrix</p>
      </div>
      <div class="meta-badges">
        <div class="meta-badge">🏆 Total Games: <strong>{total_games}</strong></div>
        <div class="meta-badge">⚡ Completed in: <strong>{elapsed_seconds:.1f}s</strong></div>
        <div class="meta-badge">🖥️ Workers: <strong>8 Parallel</strong></div>
      </div>
    </header>

    <!-- Podium -->
    <div class="podium-section">
      <!-- 2nd Place -->
      <div class="podium-card rank-2">
        <div class="crown-badge">🥈</div>
        <div class="podium-rank">2nd Place &bull; Runner-Up</div>
        <div class="podium-title" title="{rankings[1]['name']}">{rankings[1]['name']}</div>
        <div class="podium-method">{get_origin(rankings[1]['name'])}</div>
        <div class="podium-stats">
          <div class="stat-item">
            <div class="label">Points</div>
            <div class="val">{rankings[1]['points']:.1f}</div>
          </div>
          <div class="stat-item">
            <div class="label">Record</div>
            <div class="val">{rankings[1]['wins']}-{rankings[1]['losses']}-{rankings[1]['ties']}</div>
          </div>
          <div class="stat-item">
            <div class="label">Win Rate</div>
            <div class="val">{rankings[1]['win_rate_pct']:.1f}%</div>
          </div>
          <div class="stat-item">
            <div class="label">Avg Revenue</div>
            <div class="val">${rankings[1]['avg_score']:,.0f}</div>
          </div>
        </div>
      </div>

      <!-- 1st Place (Champion) -->
      <div class="podium-card rank-1">
        <div class="crown-badge">👑</div>
        <div class="podium-rank">1st Place &bull; Undefeated Champion</div>
        <div class="podium-title" title="{rankings[0]['name']}">{rankings[0]['name']}</div>
        <div class="podium-method">{get_origin(rankings[0]['name'])}</div>
        <div class="podium-stats">
          <div class="stat-item">
            <div class="label">Points</div>
            <div class="val" style="color: var(--gold);">{rankings[0]['points']:.1f}</div>
          </div>
          <div class="stat-item">
            <div class="label">Record</div>
            <div class="val">{rankings[0]['wins']}-{rankings[0]['losses']}-{rankings[0]['ties']}</div>
          </div>
          <div class="stat-item">
            <div class="label">Win Rate</div>
            <div class="val" style="color: var(--gold);">{rankings[0]['win_rate_pct']:.1f}%</div>
          </div>
          <div class="stat-item">
            <div class="label">Avg Revenue</div>
            <div class="val">${rankings[0]['avg_score']:,.0f}</div>
          </div>
        </div>
      </div>

      <!-- 3rd Place -->
      <div class="podium-card rank-3">
        <div class="crown-badge">🥉</div>
        <div class="podium-rank">3rd Place &bull; Standard Leader</div>
        <div class="podium-title" title="{rankings[2]['name']}">{rankings[2]['name']}</div>
        <div class="podium-method">{get_origin(rankings[2]['name'])}</div>
        <div class="podium-stats">
          <div class="stat-item">
            <div class="label">Points</div>
            <div class="val">{rankings[2]['points']:.1f}</div>
          </div>
          <div class="stat-item">
            <div class="label">Record</div>
            <div class="val">{rankings[2]['wins']}-{rankings[2]['losses']}-{rankings[2]['ties']}</div>
          </div>
          <div class="stat-item">
            <div class="label">Win Rate</div>
            <div class="val">{rankings[2]['win_rate_pct']:.1f}%</div>
          </div>
          <div class="stat-item">
            <div class="label">Avg Revenue</div>
            <div class="val">${rankings[2]['avg_score']:,.0f}</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Spotlight Banner for Match Replay -->
    <div class="banner-card">
      <div class="banner-text">
        <h3>🎮 Interactive Finals Replay: Cand-010 vs Cand-012</h3>
        <p>Watch the complete 720-step visual gameplay simulation with interactive controls, hand movements, farm plots, and cash curves.</p>
      </div>
      <a href="{REPLAY_HTML_REL}" target="_blank" class="btn">
        <span>Launch Game Visualizer</span>
        <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8.636 3.5a.5.5 0 0 0-.5-.5H1.5A1.5 1.5 0 0 0 0 4.5v10A1.5 1.5 0 0 0 1.5 16h10a1.5 1.5 0 0 0 1.5-1.5V7.864a.5.5 0 0 0-1 0V14.5a.5.5 0 0 1-.5.5h-10a.5.5 0 0 1-.5-.5v-10a.5.5 0 0 1 .5-.5h6.636a.5.5 0 0 0 .5-.5z"/><path fill-rule="evenodd" d="M16 .5a.5.5 0 0 0-.5-.5h-5a.5.5 0 0 0 0 1h3.793L6.146 9.146a.5.5 0 1 0 .708.708L15 1.707V5.5a.5.5 0 0 0 1 0v-5z"/></svg>
      </a>
    </div>

    <!-- Standings Table -->
    <div class="section-card">
      <div class="section-header">
        <h2 class="section-title">📊 Official Standings</h2>
      </div>
      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Candidate Agent</th>
              <th>Origin</th>
              <th>Points</th>
              <th>Record (W-L-T)</th>
              <th>Win %</th>
              <th>Average Score</th>
              <th>Total Revenue</th>
            </tr>
          </thead>
          <tbody>"""

    for r in rankings:
        rk = r["rank"]
        nm = r["name"]
        pts = r["points"]
        wlt = f"{r['wins']}W - {r['losses']}L - {r['ties']}T"
        win_pct = f"{r['win_rate_pct']:.1f}%"
        avg_sc = f"${r['avg_score']:,.2f}"
        tot_sc = f"${r['score_sum']:,.0f}"
        tag_cls = "tag-boost" if "boost" in nm else "tag-train"
        tag_text = "Boost Schedule" if "boost" in nm else "Train Schedule"
        medal = "🥇 " if rk == 1 else ("🥈 " if rk == 2 else ("🥉 " if rk == 3 else ""))

        html += f"""
            <tr>
              <td class="rank-num">{medal}{rk}</td>
              <td class="agent-cell">{nm}</td>
              <td><span class="tag {tag_cls}">{tag_text}</span></td>
              <td class="val-points">{pts:.1f}</td>
              <td>{wlt}</td>
              <td>{win_pct}</td>
              <td class="val-score">{avg_sc}</td>
              <td class="val-score">{tot_sc}</td>
            </tr>"""

    html += """
          </tbody>
        </table>
      </div>
    </div>

    <!-- Pairwise Matrix Section -->
    <div class="section-card">
      <div class="section-header">
        <h2 class="section-title">⚔️ Pairwise Head-to-Head Cross-Table</h2>
        <span style="font-size: 0.8rem; color: var(--text-muted);">Row vs Column (W - L - T)</span>
      </div>
      <div class="table-wrapper">
        <table class="matrix-table">
          <thead>
            <tr>
              <th>Candidate</th>"""

    for a in agents:
        s_alias = a.replace("submission_", "").replace("cand", "c")
        html += f"""<th title="{a}">{s_alias}</th>"""

    html += """
            </tr>
          </thead>
          <tbody>"""

    for a in agents:
        a_alias = a.replace("submission_", "")
        html += f"""<tr><td title="{a}">{a_alias}</td>"""
        for b in agents:
            if a == b:
                html += """<td class="cell-diag">—</td>"""
            else:
                rec = pairwise[a][b]
                w = rec["w"]
                l = rec["l"]
                t = rec["t"]
                txt = f"{w}-{l}" + (f"-{t}" if t > 0 else "")
                if w > l:
                    cls = "cell-win"
                elif l > w:
                    cls = "cell-loss"
                else:
                    cls = "cell-tie"
                html += f"""<td class="{cls}" title="{a} vs {b}: {w}W {l}L {t}T">{txt}</td>"""
        html += """</tr>"""

    html += """
          </tbody>
        </table>
      </div>
    </div>

    <!-- Matches Log Section -->
    <div class="section-card">
      <div class="section-header">
        <h2 class="section-title">📜 Match Results & Replay Index (56 Games)</h2>
      </div>
      <div class="filter-bar">
        <input type="text" id="searchInput" class="search-input" placeholder="Search candidate, match #, seed..." onkeyup="filterMatches()">
        <select id="outcomeSelect" class="select-input" onchange="filterMatches()">
          <option value="ALL">All Outcomes</option>
          <option value="DECISIVE">Decisive Wins</option>
          <option value="TIE">Ties</option>
        </select>
      </div>
      <div id="matchesContainer">"""

    for m in matches:
        g_idx = m["game_idx"]
        seed = m["seed"]
        a_name = m["agent_a"]
        b_name = m["agent_b"]
        sa = m["score_a"]
        sb = m["score_b"]
        w = m["winner"]
        
        a_win = "winner" if w == "A" else ""
        b_win = "winner" if w == "B" else ""
        outcome_tag = "TIE" if w == "TIE" else "DECISIVE"

        html += f"""
        <div class="match-item" data-outcome="{outcome_tag}" data-text="{g_idx} {seed} {a_name} {b_name}">
          <div style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-dim); width: 85px;">
            Match #{g_idx:02d}
          </div>
          <div class="match-players">
            <div class="player-entry {a_win}">
              <span class="seat-badge">P0</span>
              <span>{a_name}</span>
              <span class="score-badge">${sa:,.0f}</span>
            </div>
            <div class="vs-sep">vs</div>
            <div class="player-entry {b_win}">
              <span class="seat-badge">P1</span>
              <span>{b_name}</span>
              <span class="score-badge">${sb:,.0f}</span>
            </div>
          </div>
          <div style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-muted); width: 110px; text-align: right;">
            Seed: {seed}
          </div>
        </div>"""

    html += """
      </div>
    </div>

    <!-- Footer -->
    <footer class="footer">
      <p>Kaggriculture MuZero Operational Candidate League &bull; Automated Benchmark Suite &bull; Generated """ + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    </footer>
  </div>

  <script>
    function filterMatches() {
      const q = document.getElementById('searchInput').value.toLowerCase();
      const outcome = document.getElementById('outcomeSelect').value;
      const items = document.querySelectorAll('.match-item');

      items.forEach(el => {
        const text = el.getAttribute('data-text').toLowerCase();
        const oc = el.getAttribute('data-outcome');

        const matchesQuery = !q || text.includes(q);
        const matchesOutcome = outcome === 'ALL' || oc === outcome;

        if (matchesQuery && matchesOutcome) {
          el.style.display = 'flex';
        } else {
          el.style.display = 'none';
        }
      });
    }
  </script>
</body>
</html>
"""
    return html


def main() -> int:
    parser = argparse.ArgumentParser(description="Render interactive tournament dashboard HTML")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to input league json report")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to output HTML dashboard")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input report not found at {args.input}")
        return 1

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    html_content = generate_dashboard_html(data)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html_content)

    print("=" * 80)
    print("  TOURNAMENT DASHBOARD RENDERED SUCCESSFULLY")
    print("=" * 80)
    print(f"  Input League Data:   {args.input}")
    print(f"  Output Dashboard:    {args.output}")
    print(f"  Dashboard URL:       file://{os.path.abspath(args.output)}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
