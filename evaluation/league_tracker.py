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

"""Prioritized Fictitious Self-Play (PFSP) league tracker.

Implements the AlphaStar-style matchmaking math from ``docs/League_Tracker.md``:

* ``LeagueTracker`` — checkpoint registry + Laplace-smoothed win matrices
* ``PFSPSampler`` — hard / var / uniform opponent sampling
* ``select_matchmaking_opponent`` — main / main_exploiter / league_exploiter roles
* ``run_pfsp_league`` — play matches under PFSP sampling and persist state
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .harness import play_single_game, short_name

DEFAULT_LEAGUE_STATE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "artifacts",
    "league_tracker_state.json",
)

EXPLOITER_RESET_WINRATE = 0.70


class LeagueTracker:
    """Tracks policy checkpoints and empirical win rates between league members."""

    def __init__(self, smoothing: float = 1.0):
        self.checkpoints: List[str] = []
        self.wins_matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.games_matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.smoothing = float(smoothing)
        self.current_main: Optional[str] = None
        self.roles: Dict[str, str] = {}  # checkpoint_id → role label
        self.meta: Dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.checkpoints)

    def add_checkpoint(self, checkpoint_id: str, role: Optional[str] = None) -> None:
        """Registers a new frozen model checkpoint into the league (idempotent)."""
        if checkpoint_id in self.checkpoints:
            if role:
                self.roles[checkpoint_id] = role
            return

        self.checkpoints.append(checkpoint_id)
        n = len(self.checkpoints)
        new_wins = np.zeros((n, n), dtype=np.float32)
        new_games = np.zeros((n, n), dtype=np.float32)
        if n > 1:
            new_wins[: n - 1, : n - 1] = self.wins_matrix
            new_games[: n - 1, : n - 1] = self.games_matrix
        self.wins_matrix = new_wins
        self.games_matrix = new_games
        if role:
            self.roles[checkpoint_id] = role
        if self.current_main is None:
            self.current_main = checkpoint_id

    def set_current_main(self, checkpoint_id: str) -> None:
        if checkpoint_id not in self.checkpoints:
            self.add_checkpoint(checkpoint_id, role="main")
        self.current_main = checkpoint_id
        self.roles[checkpoint_id] = "main"

    def record_match_outcome(self, agent_id: str, opponent_id: str, outcome: float) -> None:
        """Records game result: outcome = 1.0 (win), 0.5 (draw), 0.0 (loss)."""
        if agent_id not in self.checkpoints:
            self.add_checkpoint(agent_id)
        if opponent_id not in self.checkpoints:
            self.add_checkpoint(opponent_id)
        i = self.checkpoints.index(agent_id)
        j = self.checkpoints.index(opponent_id)
        outcome = float(np.clip(outcome, 0.0, 1.0))
        self.wins_matrix[i, j] += outcome
        self.wins_matrix[j, i] += 1.0 - outcome
        self.games_matrix[i, j] += 1.0
        self.games_matrix[j, i] += 1.0

    def get_win_rate(self, agent_id: str, opponent_id: str) -> float:
        """Empirical win rate with Laplace smoothing."""
        if agent_id not in self.checkpoints or opponent_id not in self.checkpoints:
            return 0.5
        i = self.checkpoints.index(agent_id)
        j = self.checkpoints.index(opponent_id)
        games = float(self.games_matrix[i, j])
        wins = float(self.wins_matrix[i, j])
        return (wins + self.smoothing) / (games + 2.0 * self.smoothing)

    def should_reset_exploiter(
        self,
        exploiter_id: str,
        main_id: Optional[str] = None,
        threshold: float = EXPLOITER_RESET_WINRATE,
        min_games: float = 10.0,
    ) -> bool:
        """True when exploiter beats main in ≥ threshold of matches (population-collapse guard)."""
        main_id = main_id or self.current_main
        if not main_id or exploiter_id not in self.checkpoints or main_id not in self.checkpoints:
            return False
        i = self.checkpoints.index(exploiter_id)
        j = self.checkpoints.index(main_id)
        if float(self.games_matrix[i, j]) < min_games:
            return False
        return self.get_win_rate(exploiter_id, main_id) >= threshold

    def leaderboard(self) -> List[Dict[str, Any]]:
        """Points = sum of win matrix row; sorted descending."""
        rows: List[Dict[str, Any]] = []
        for i, cid in enumerate(self.checkpoints):
            points = float(self.wins_matrix[i].sum())
            games = float(self.games_matrix[i].sum())
            rows.append({
                "id": cid,
                "name": short_name(cid) if os.path.sep in cid or cid.endswith(".zip") else cid,
                "role": self.roles.get(cid, "member"),
                "points": points,
                "games": games,
                "is_main": cid == self.current_main,
            })
        rows.sort(key=lambda r: (-r["points"], -r["games"], r["name"]))
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        return rows

    def to_dict(self) -> Dict[str, Any]:
        return {
            "smoothing": self.smoothing,
            "checkpoints": list(self.checkpoints),
            "wins_matrix": self.wins_matrix.tolist(),
            "games_matrix": self.games_matrix.tolist(),
            "current_main": self.current_main,
            "roles": dict(self.roles),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LeagueTracker":
        tracker = cls(smoothing=float(data.get("smoothing", 1.0)))
        tracker.checkpoints = list(data.get("checkpoints") or [])
        n = len(tracker.checkpoints)
        wins = np.array(data.get("wins_matrix") or [], dtype=np.float32)
        games = np.array(data.get("games_matrix") or [], dtype=np.float32)
        if wins.size == 0:
            wins = np.zeros((n, n), dtype=np.float32)
            games = np.zeros((n, n), dtype=np.float32)
        tracker.wins_matrix = wins.reshape(n, n) if n else np.zeros((0, 0), dtype=np.float32)
        tracker.games_matrix = games.reshape(n, n) if n else np.zeros((0, 0), dtype=np.float32)
        tracker.current_main = data.get("current_main")
        tracker.roles = dict(data.get("roles") or {})
        tracker.meta = dict(data.get("meta") or {})
        return tracker

    def save(self, path: str = DEFAULT_LEAGUE_STATE) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str = DEFAULT_LEAGUE_STATE) -> "LeagueTracker":
        if not os.path.isfile(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class PFSPSampler:
    """Prioritized Fictitious Self-Play opponent sampler."""

    def __init__(self, league: LeagueTracker):
        self.league = league

    def sample_opponent(
        self,
        active_agent_id: str,
        mode: str = "hard",
        power: float = 2.0,
        rng: Optional[np.random.Generator] = None,
    ) -> str:
        """
        Sample an opponent for ``active_agent_id``.

        Modes:
          - hard:    f(x)=(1-x)^p  (focus on losses)
          - var:     f(x)=x(1-x)   (focus on ~50% matchups)
          - uniform: f(x)=1
        """
        rng = rng or np.random.default_rng()
        checkpoints = self.league.checkpoints
        if not checkpoints or (len(checkpoints) == 1 and checkpoints[0] == active_agent_id):
            return active_agent_id

        candidates = [c for c in checkpoints if c != active_agent_id]
        if not candidates:
            return active_agent_id

        win_rates = np.array(
            [self.league.get_win_rate(active_agent_id, c) for c in candidates],
            dtype=np.float64,
        )

        if mode == "hard":
            weights = np.power(np.maximum(0.0, 1.0 - win_rates), float(power))
        elif mode == "var":
            weights = win_rates * (1.0 - win_rates)
        elif mode == "uniform":
            weights = np.ones_like(win_rates)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        total = float(weights.sum())
        if total <= 1e-8:
            probs = np.ones(len(candidates), dtype=np.float64) / len(candidates)
        else:
            probs = weights / total

        idx = int(rng.choice(len(candidates), p=probs))
        return candidates[idx]


def select_matchmaking_opponent(
    agent_role: str,
    active_agent_id: str,
    pfsp: PFSPSampler,
    *,
    current_main_id: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[str, str]:
    """
    Role-specific AlphaStar matchmaking ratios.

    Returns ``(opponent_id, match_type)``.
    """
    rng = rng or np.random.default_rng()
    rand = float(rng.random())
    current_main = current_main_id or pfsp.league.current_main or active_agent_id

    if agent_role == "main":
        # 35% self-play, 50% hard PFSP, 15% var PFSP
        if rand < 0.35:
            return active_agent_id, "self_play"
        if rand < 0.85:
            return pfsp.sample_opponent(active_agent_id, mode="hard", power=2.0, rng=rng), "pfsp_hard"
        return pfsp.sample_opponent(active_agent_id, mode="var", rng=rng), "pfsp_var"

    if agent_role == "main_exploiter":
        # 50% vs current main, 50% var PFSP
        if rand < 0.50:
            return current_main, "direct_exploit"
        return pfsp.sample_opponent(active_agent_id, mode="var", rng=rng), "pfsp_var"

    if agent_role == "league_exploiter":
        return pfsp.sample_opponent(active_agent_id, mode="hard", power=3.0, rng=rng), "pfsp_hard"

    return active_agent_id, "fallback"


@dataclass
class PFSPLeagueConfig:
    games: int = 8
    role: str = "main"
    active_id: Optional[str] = None
    base_seed: int = 20201
    episode_steps: int = 720
    save_replay_dir: Optional[str] = None
    min_replay_score: float = 80000.0
    state_path: str = DEFAULT_LEAGUE_STATE
    quiet: bool = False


def register_packages(
    tracker: LeagueTracker,
    agent_paths: Sequence[str],
    *,
    main_id: Optional[str] = None,
) -> LeagueTracker:
    """Register zip/py agent paths as league checkpoints."""
    for path in agent_paths:
        abs_path = os.path.abspath(path)
        tracker.add_checkpoint(abs_path, role="member")
    if main_id:
        tracker.set_current_main(os.path.abspath(main_id))
    elif tracker.current_main is None and agent_paths:
        tracker.set_current_main(os.path.abspath(agent_paths[0]))
    return tracker


def run_pfsp_league(
    agent_paths: Sequence[str],
    *,
    config: Optional[PFSPLeagueConfig] = None,
    tracker: Optional[LeagueTracker] = None,
) -> Dict[str, Any]:
    """
    Play ``config.games`` matches with PFSP matchmaking, updating the tracker.

    Outcome encoding: active agent win → 1.0, draw → 0.5, loss → 0.0.
    """
    cfg = config or PFSPLeagueConfig()
    tracker = tracker or LeagueTracker.load(cfg.state_path)
    register_packages(tracker, agent_paths, main_id=cfg.active_id)
    pfsp = PFSPSampler(tracker)

    active = os.path.abspath(cfg.active_id) if cfg.active_id else tracker.current_main
    if not active:
        raise ValueError("No active agent / current_main set for PFSP league")
    if active not in tracker.checkpoints:
        tracker.add_checkpoint(active, role=cfg.role)

    match_log: List[Dict[str, Any]] = []
    t0 = time.time()
    if not cfg.quiet:
        print("=" * 100)
        print(
            f"  PFSP LEAGUE  role={cfg.role}  active={short_name(active)}  "
            f"games={cfg.games}  pool={len(tracker)}"
        )
        print("=" * 100)

    for g in range(cfg.games):
        opponent, match_type = select_matchmaking_opponent(
            cfg.role,
            active,
            pfsp,
            current_main_id=tracker.current_main,
        )
        # Self-play still runs a real match (same agent both seats) for data harvest
        a_is_p0 = (g % 2 == 0)
        seed = cfg.base_seed + g + 1
        res = play_single_game(
            active,
            opponent,
            seed=seed,
            a_is_p0=a_is_p0,
            episode_steps=cfg.episode_steps,
            save_replay_dir=cfg.save_replay_dir,
            min_replay_score=cfg.min_replay_score,
        )
        if res["winner"] == "A":
            outcome = 1.0
        elif res["winner"] == "B":
            outcome = 0.0
        else:
            outcome = 0.5
        tracker.record_match_outcome(active, opponent, outcome)
        entry = {
            "game": g + 1,
            "active": active,
            "opponent": opponent,
            "match_type": match_type,
            "outcome": outcome,
            "score_a": res["score_a"],
            "score_b": res["score_b"],
            "winner": res["winner"],
            "seed": seed,
        }
        match_log.append(entry)
        if not cfg.quiet:
            print(
                f"  [{g+1:>3}/{cfg.games}] {match_type:<14} "
                f"{short_name(active)} vs {short_name(opponent)}  "
                f"out={outcome:.1f}  "
                f"${res['score_a']:,.0f} / ${res['score_b']:,.0f}"
            )

    # Population-collapse tip: flag exploiters that dominate main
    reset_flags = []
    for cid, role in tracker.roles.items():
        if role.endswith("exploiter") and tracker.should_reset_exploiter(cid):
            reset_flags.append(cid)

    state_path = tracker.save(cfg.state_path)
    board = tracker.leaderboard()
    report = {
        "mode": "pfsp",
        "role": cfg.role,
        "active": active,
        "games": cfg.games,
        "elapsed_s": time.time() - t0,
        "state_path": state_path,
        "match_log": match_log,
        "leaderboard": board,
        "exploiter_reset_candidates": reset_flags,
        "current_main": tracker.current_main,
    }

    if not cfg.quiet:
        print("-" * 100)
        print(f"  Saved tracker → {state_path}")
        print(f"  {'Rank':>4} | {'Name':<28} | {'Pts':>6} | {'Games':>6} | Role")
        for row in board[:12]:
            flag = " *" if row["is_main"] else ""
            print(
                f"  {row['rank']:>4} | {row['name']:<28} | {row['points']:>6.1f} | "
                f"{row['games']:>6.0f} | {row['role']}{flag}"
            )
        if reset_flags:
            print(f"  Exploiter reset candidates (≥{EXPLOITER_RESET_WINRATE:.0%} vs main):")
            for cid in reset_flags:
                print(f"    - {short_name(cid)}")
        print("=" * 100)

    return report


def demo_sample_opponents(
    tracker: LeagueTracker,
    active_id: str,
    *,
    n: int = 20,
    role: str = "main",
    seed: int = 0,
) -> Dict[str, Any]:
    """Sample matchmaking decisions without playing games (sanity / dry-run)."""
    pfsp = PFSPSampler(tracker)
    rng = np.random.default_rng(seed)
    counts: Dict[str, int] = {}
    types: Dict[str, int] = {}
    for _ in range(n):
        opp, mtype = select_matchmaking_opponent(
            role, active_id, pfsp, current_main_id=tracker.current_main, rng=rng
        )
        counts[short_name(opp)] = counts.get(short_name(opp), 0) + 1
        types[mtype] = types.get(mtype, 0) + 1
    return {"samples": n, "role": role, "active": short_name(active_id), "opponents": counts, "match_types": types}
