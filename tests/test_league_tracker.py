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

"""Unit tests for LeagueTracker / PFSPSampler (docs/League_Tracker.md)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.league_tracker import (
    LeagueTracker,
    PFSPSampler,
    demo_sample_opponents,
    select_matchmaking_opponent,
)


def test_add_checkpoint_expands_matrices():
    league = LeagueTracker()
    league.add_checkpoint("a")
    league.add_checkpoint("b")
    assert league.wins_matrix.shape == (2, 2)
    assert league.games_matrix.shape == (2, 2)
    league.add_checkpoint("a")  # idempotent
    assert len(league) == 2


def test_laplace_win_rate_unplayed_is_half():
    league = LeagueTracker(smoothing=1.0)
    league.add_checkpoint("a")
    league.add_checkpoint("b")
    assert league.get_win_rate("a", "b") == pytest.approx(0.5)


def test_record_match_outcome_symmetric():
    league = LeagueTracker()
    league.add_checkpoint("a")
    league.add_checkpoint("b")
    league.record_match_outcome("a", "b", 1.0)
    assert league.games_matrix[0, 1] == 1.0
    assert league.wins_matrix[0, 1] == 1.0
    assert league.wins_matrix[1, 0] == 0.0
    # After one win: (1+1)/(1+2) = 2/3
    assert league.get_win_rate("a", "b") == pytest.approx(2.0 / 3.0)


def test_pfsp_hard_prefers_low_winrate(monkeypatch):
    league = LeagueTracker()
    for cid in ("main", "easy", "hard"):
        league.add_checkpoint(cid)
    # main dominates easy, loses to hard
    for _ in range(10):
        league.record_match_outcome("main", "easy", 1.0)
        league.record_match_outcome("main", "hard", 0.0)

    pfsp = PFSPSampler(league)
    # Force deterministic sample of highest-weight candidate
    counts = {"easy": 0, "hard": 0}
    rng = np.random.default_rng(0)
    for _ in range(200):
        opp = pfsp.sample_opponent("main", mode="hard", power=2.0, rng=rng)
        counts[opp] += 1
    assert counts["hard"] > counts["easy"]


def test_pfsp_var_prefers_even_matchups():
    league = LeagueTracker()
    for cid in ("main", "even", "stomped"):
        league.add_checkpoint(cid)
    for _ in range(20):
        league.record_match_outcome("main", "even", 0.5)
        league.record_match_outcome("main", "stomped", 1.0)

    pfsp = PFSPSampler(league)
    counts = {"even": 0, "stomped": 0}
    rng = np.random.default_rng(1)
    for _ in range(300):
        opp = pfsp.sample_opponent("main", mode="var", rng=rng)
        counts[opp] += 1
    assert counts["even"] > counts["stomped"]


def test_select_matchmaking_main_ratios():
    league = LeagueTracker()
    league.add_checkpoint("main", role="main")
    league.add_checkpoint("opp")
    league.set_current_main("main")
    pfsp = PFSPSampler(league)
    types = set()
    rng = np.random.default_rng(42)
    for _ in range(100):
        _, mtype = select_matchmaking_opponent("main", "main", pfsp, rng=rng)
        types.add(mtype)
    assert "self_play" in types
    assert "pfsp_hard" in types
    assert "pfsp_var" in types


def test_main_exploiter_resolves_current_main():
    league = LeagueTracker()
    league.add_checkpoint("champ", role="main")
    league.add_checkpoint("exploiter", role="main_exploiter")
    league.set_current_main("champ")
    pfsp = PFSPSampler(league)

    class _LowRand:
        def random(self):
            return 0.1  # < 0.50 → direct_exploit

    opp, mtype = select_matchmaking_opponent(
        "main_exploiter",
        "exploiter",
        pfsp,
        current_main_id="champ",
        rng=_LowRand(),  # type: ignore[arg-type]
    )
    assert opp == "champ"
    assert mtype == "direct_exploit"


def test_exploiter_reset_flag():
    league = LeagueTracker()
    league.add_checkpoint("main", role="main")
    league.add_checkpoint("exploiter", role="main_exploiter")
    league.set_current_main("main")
    for _ in range(12):
        league.record_match_outcome("exploiter", "main", 1.0)
    assert league.should_reset_exploiter("exploiter") is True


def test_save_load_roundtrip(tmp_path: Path):
    league = LeagueTracker(smoothing=1.5)
    league.add_checkpoint("a", role="main")
    league.add_checkpoint("b")
    league.record_match_outcome("a", "b", 1.0)
    league.set_current_main("a")
    path = tmp_path / "league.json"
    league.save(str(path))
    loaded = LeagueTracker.load(str(path))
    assert loaded.checkpoints == ["a", "b"]
    assert loaded.current_main == "a"
    assert loaded.smoothing == 1.5
    assert loaded.get_win_rate("a", "b") == pytest.approx(league.get_win_rate("a", "b"))
    assert json.loads(path.read_text())["roles"]["a"] == "main"


def test_demo_sample_opponents():
    league = LeagueTracker()
    league.add_checkpoint("/tmp/a.zip")
    league.add_checkpoint("/tmp/b.zip")
    league.set_current_main("/tmp/a.zip")
    demo = demo_sample_opponents(league, "/tmp/a.zip", n=30, role="main", seed=3)
    assert demo["samples"] == 30
    assert sum(demo["match_types"].values()) == 30
