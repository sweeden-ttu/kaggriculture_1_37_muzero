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

"""Process-isolated game execution.

An *agent spec* is either a path to a ``main.py``-style file (anything exposing
``agent`` or ``kaggle_submission_agent``), one of the engine's built-in names
(``pass``, ``random``, ``starter``), or a ``dict`` with keys ``path`` and
optional ``overrides`` (module attribute name -> value, applied after import so
tunable constants such as ``V9_HERD_MIN_WOOL`` can be swept without editing the
source).
"""
from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BUILTIN_AGENTS = ("pass", "random", "starter")
AgentSpec = Union[str, Dict[str, Any]]


@dataclass
class GameResult:
    seed: int
    a: str
    b: str
    a_is_p0: bool
    score_a: float
    score_b: float
    elapsed: float
    statuses: List[str]
    attribution: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def delta(self) -> float:
        return self.score_a - self.score_b

    @property
    def winner(self) -> str:
        if self.score_a > self.score_b:
            return "A"
        if self.score_b > self.score_a:
            return "B"
        return "TIE"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["delta"] = self.delta
        d["winner"] = self.winner
        return d


def spec_path(spec: AgentSpec) -> str:
    return spec["path"] if isinstance(spec, dict) else spec


def spec_name(spec: AgentSpec) -> str:
    if isinstance(spec, dict):
        base = os.path.basename(spec["path"])
        tag = spec.get("name")
        return tag or base
    return os.path.basename(spec) if os.path.sep in spec else spec


def load_agent_module(path: str, overrides: Optional[Dict[str, Any]] = None):
    """Import an agent file under a unique module name and apply attribute overrides."""
    name = "arena_agent_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import agent at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    agent_dir = os.path.dirname(os.path.abspath(path))
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)
    spec.loader.exec_module(module)
    for key, value in (overrides or {}).items():
        if not hasattr(module, key):
            raise AttributeError(f"{path}: no tunable named {key!r}")
        setattr(module, key, value)
    return module


def resolve_agent(spec: AgentSpec):
    if isinstance(spec, str) and spec in BUILTIN_AGENTS:
        return spec
    path = spec_path(spec)
    overrides = spec.get("overrides") if isinstance(spec, dict) else None
    module = load_agent_module(path, overrides)
    fn = getattr(module, "agent", None) or getattr(module, "kaggle_submission_agent", None)
    if not callable(fn):
        raise RuntimeError(f"{path}: no callable agent")
    return fn


def _run_game(task: Dict[str, Any]) -> Dict[str, Any]:
    """Worker entry: play one game and return a serialisable result dict."""
    import kaggle_environments
    from .attribution import attribute_game

    a, b, seed, a_is_p0 = task["a"], task["b"], int(task["seed"]), bool(task["a_is_p0"])
    episode_steps = int(task.get("episode_steps", 720))
    want_attr = bool(task.get("attribution", False))
    t0 = time.time()
    try:
        if task.get("fixed_draw"):
            install_fixed_draw()
        fa, fb = resolve_agent(a), resolve_agent(b)
        p0, p1 = (fa, fb) if a_is_p0 else (fb, fa)
        env = kaggle_environments.make(
            "kaggriculture", configuration={"episodeSteps": episode_steps, "seed": seed}
        )
        env.run([p0, p1])
        r0 = float(env.state[0].reward or 0.0)
        r1 = float(env.state[1].reward or 0.0)
        statuses = [str(s.status) for s in env.state]
        attribution = attribute_game(env.steps) if want_attr else None
        score_a, score_b = (r0, r1) if a_is_p0 else (r1, r0)
        res = GameResult(seed, spec_name(a), spec_name(b), a_is_p0, score_a, score_b,
                         round(time.time() - t0, 2), statuses, attribution)
        if want_attr and attribution is not None:
            # re-key attribution as A/B instead of P0/P1
            pa, pb = ("P0", "P1") if a_is_p0 else ("P1", "P0")
            res.attribution = {"A": attribution[pa], "B": attribution[pb], "town": attribution["town"],
                               "final_prices": attribution["final_prices"]}
    except Exception as exc:  # never let one broken game kill the pool
        import traceback
        res = GameResult(seed, spec_name(a), spec_name(b), a_is_p0, 0.0, 0.0,
                         round(time.time() - t0, 2), ["ERROR", "ERROR"], None,
                         error=f"{exc}\n{traceback.format_exc()}")
    return res.to_dict()


def play_game(a: AgentSpec, b: AgentSpec, seed: int, a_is_p0: bool = True, **kw) -> GameResult:
    d = _run_game({"a": a, "b": b, "seed": seed, "a_is_p0": a_is_p0, **kw})
    return _from_dict(d)


def _from_dict(d: Dict[str, Any]) -> GameResult:
    d = dict(d)
    d.pop("delta", None)
    d.pop("winner", None)
    return GameResult(**d)


def play_many(tasks: Iterable[Dict[str, Any]], workers: int = 4, progress: bool = True) -> List[GameResult]:
    """Run many games in parallel. Each worker process plays exactly one game and exits."""
    tasks = list(tasks)
    if not tasks:
        return []
    workers = max(1, min(workers, len(tasks)))
    ctx = mp.get_context("spawn")
    out: List[GameResult] = []
    with ctx.Pool(processes=workers, maxtasksperchild=1) as pool:
        for i, d in enumerate(pool.imap_unordered(_run_game, tasks, chunksize=1), 1):
            out.append(_from_dict(d))
            if progress:
                r = out[-1]
                tag = "ERR" if r.error else r.winner
                print(f"  [{i:3d}/{len(tasks)}] seed {r.seed:6d} {'A=P0' if r.a_is_p0 else 'A=P1'} "
                      f"{r.a[:28]:28s} {r.score_a:>9,.0f} vs {r.b[:28]:28s} {r.score_b:>9,.0f}  {tag} {r.elapsed}s",
                      flush=True)
    out.sort(key=lambda r: (r.seed, not r.a_is_p0))
    return out


def paired_tasks(a: AgentSpec, b: AgentSpec, seeds: Sequence[int], both_seats: bool = True, **kw) -> List[Dict[str, Any]]:
    tasks = []
    for s in seeds:
        tasks.append({"a": a, "b": b, "seed": int(s), "a_is_p0": True, **kw})
        if both_seats:
            tasks.append({"a": a, "b": b, "seed": int(s), "a_is_p0": False, **kw})
    return tasks


def save_results(results: Sequence[GameResult], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f)


# ---------------------------------------------------------------------------
# Fixed town draw (common random numbers for candidate comparisons)
# ---------------------------------------------------------------------------
def install_fixed_draw() -> None:
    """Decouple the shop draw (and each farm's weeds) from farm state.

    The engine's ``_end_of_day`` seeds one RNG per day and consumes one draw per
    empty tile on both farms for weeds before drawing the day's new shop, so any
    difference in either farm re-rolls every later shop. For A/B comparisons we
    want the town to depend on the seed only; weeds get a per-player stream.
    Real Kaggle games keep the coupling (a zero-mean re-roll), so this is only
    used inside the arena when ``fixed_draw`` is requested.
    """
    import random as _random
    from kaggle_environments.envs.kaggriculture import kaggriculture as K

    if getattr(K, "_arena_fixed_draw", False):
        return

    def _end_of_day(state, env, day):
        obs0 = state[0].observation
        cfg = env.configuration
        board_size = int(K.get(cfg, "boardSize", 10))
        turns_per_day = max(1, int(K.get(cfg, "turnsPerDay", 24)))
        weed_chance = float(K.get(cfg, "weedSpawnChance", 0.005))
        shed_cap = int(K.get(cfg, "shedCapacity", 100))
        shop_interval = max(1, int(K.get(cfg, "townShopUnlockInterval", 3)))
        seed = env.info.get("seed", 0)
        for player_id, farm in enumerate(obs0.farms):
            private = state[player_id].observation.private
            K._daily_refresh_plants(farm, day, turns_per_day)
            K._daily_refresh_animals(farm, day)
            rng_w = _random.Random(((seed * 1_000_003) ^ day) * 7 + 1000 + player_id)
            K._spawn_weeds(farm, board_size, weed_chance, rng_w)
            K._drop_inventories_to_shed(private, shed_cap)
            farm["farmer"] = list(K._default_spawn(board_size))
            farm["hands"] = []
            farm["hires_today"] = 0
            private["inventories"] = [{}]
        next_day = day + 1
        town = obs0.town
        if next_day > 0 and next_day % shop_interval == 0:
            if len(town["unlocked_shops"]) < K.MAX_SHOP_INSTANCES:
                rng_s = _random.Random((seed * 1_000_003) ^ day)
                town["unlocked_shops"].append(rng_s.choice(sorted(K.SHOPS)))

    K._end_of_day = _end_of_day
    K._arena_fixed_draw = True
