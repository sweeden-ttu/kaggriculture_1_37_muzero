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
==============================================================================
ppo_tuning.py - PPO Microcontroller Macrocontroller Tuning & Loop Integration
==============================================================================
Wires the PPO controller (``muzero.ppo``) into the Phase-4 continuous
self-play loop:

  * ``ppo_grid_configs``      — offline grid sweep over PPO dials (lr, gamma,
                               clip_epsilon, entropy_coef, gae_lambda) alongside
                               the MuZero loss-weight dials.
  * ``ppo_bc_init``          — behavior-clone initialization from expert replay
                               macro-labels so self-play does not start random.
  * ``train_ppo_controller`` — BC init + analytical-economy self-play fine-tune
                               with the Adam (Adaptive Moment Estimation) engine.
  * ``build_ppo_agent_package`` — compiles the hybrid chassis + PPO gate layer
                               (with embedded weights) into a standalone zip.
  * ``execute_ppo_trial``    — full trial: train → package → evaluate vs
                               champion → multimodal promotion gating.

The offline ``--search-strategy grid`` mode in ``phase4_continuous_loop`` and
``train.sh``/``boost.sh`` sweeps these PPO configs alongside MuZero configs.
==============================================================================
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import time
import zipfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from muzero.ppo import (  # noqa: E402
    DEFAULT_PPO_CHECKPOINT,
    ActorCritic,
    PPOBuffer,
    PPOHyperParams,
    load_ppo_checkpoint,
    ppo_rollout_batch,
    save_ppo_checkpoint,
    train_ppo,
)
from muzero.types import MacroOption  # noqa: E402

ART = os.path.join(HERE, "artifacts")
REPLAYS = os.path.join(HERE, "replays")
BASELINES = os.path.join(HERE, "baselines")
DIST = os.path.join(HERE, "dist")
SNAPSHOTS = os.path.join(ART, "snapshots")
PPO_DIR = os.path.join(ART, "ppo")
STATUS_FILE = os.path.join(ART, "loop_status.json")
CHAMPION_HISTORY_LOG = os.path.join(ART, "champion_history.log")

GATE_PREFIX = """
# =============================================================================
# PPO MICROCONTROLLER GATE (grafted onto Hybrid Direct chassis)
# =============================================================================
"""

# ---------------------------------------------------------------------------
# PPO gate layer source (modeled on hybrid_chassis/layers/muzero_options.py)
# ---------------------------------------------------------------------------

_LAYER_TEMPLATE = '''
import base64 as _ppo_b64
import os as _ppo_os
import tempfile as _ppo_tempfile

_PPO_CKPT_B64 = "{ckpt_b64}"
_PPO_REPORT = {{"expansions_blocked": 0, "trickle_sells": 0, "hire_pushes": 0, "errors": 0}}
_PPO_MODEL = None
_PPO_OPTION = None


def _ppo_actor_critic(obs_dim, hidden_dim, num_actions):
    import torch as _t
    import torch.nn as _nn

    class _AC(_nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = _nn.Sequential(
                _nn.Linear(obs_dim, hidden_dim), _nn.ReLU(),
                _nn.Linear(hidden_dim, hidden_dim), _nn.ReLU(),
            )
            self.policy_head = _nn.Linear(hidden_dim, num_actions)
            self.value_head = _nn.Linear(hidden_dim, 1)

        def forward(self, obs):
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            h = self.trunk(obs)
            return self.policy_head(h), self.value_head(h).squeeze(-1)

    return _AC()


def _ppo_load():
    global _PPO_MODEL, _PPO_OPTION
    if _PPO_MODEL is not None:
        return _PPO_MODEL
    try:
        import torch as _t
        data = _ppo_b64.b64decode(_PPO_CKPT_B64)
        fd, path = _ppo_tempfile.mkstemp(suffix=".pt")
        with _ppo_os.fdopen(fd, "wb") as f:
            f.write(data)
        payload = _t.load(path, map_location="cpu")
        _ppo_os.unlink(path)
        sd = payload.get("state_dict", payload)
        obs_dim = int(payload.get("obs_dim", 32))
        hidden_dim = int(payload.get("hidden_dim", 128))
        num_actions = int(payload.get("num_actions", 8))
        model = _ppo_actor_critic(obs_dim, hidden_dim, num_actions)
        model.load_state_dict(sd)
        model.eval()
        _PPO_MODEL = model
        _PPO_OPTION = _t
    except Exception:
        _PPO_MODEL = None
    return _PPO_MODEL


def _ppo_recommend(money, day, unlocked, shed):
    """Return the PPO macro-option index, or None when the controller is unavailable."""
    model = _ppo_load()
    if model is None:
        return None
    try:
        from muzero.encoding import encode_macro_state, legal_macro_options
        from muzero.types import MacroState, DEFAULT_PRICES

        state = MacroState(
            day=int(day), capital=float(money), unlocked=tuple(unlocked),
            shed=dict(shed), prices=dict(DEFAULT_PRICES), active_animals=1,
        )
        obs = encode_macro_state(state)
        legal = legal_macro_options(state)
        import torch as _t
        logits, _ = model(obs)
        if legal:
            mask = _t.full((8,), float("-inf"))
            mask[[int(o) for o in legal]] = 0.0
            logits = logits + mask
        probs = _t.softmax(logits, dim=-1)
        return int(_t.argmax(probs, dim=-1).item())
    except Exception:
        return None


_PPO_PARENT = agent

_NEML_STEP0_ORDERS = [
    ["BUY_PRODUCT", "WHEAT", 9], ["SELL", "WHEAT", 9],
    ["BUY_PRODUCT", "WHEAT", 11], ["SELL", "WHEAT", 6],
    ["BUY_SEED", "WHEAT", 1],
]

_EXPAND_BY_QUAD = {0: "NE", 1: "SW", 2: "SE"}


def agent(observation, configuration=None):
    action = _PPO_PARENT(observation, configuration)
    try:
        seat = int(observation.get("player", 0))
        farms = observation.get("farms", [{}, {}])
        farm = farms[seat] if seat < len(farms) else {}
        money = float(farm.get("money", 0.0))
        unlocked = list(farm.get("unlocked_quadrants") or ["NW"])
        step = int(observation.get("step", 0))
        day = int(observation.get("day", step // 24))

        if step == 0:
            action = dict(action, market=_NEML_STEP0_ORDERS)
            return action

        orders = [list(o) for o in (action.get("market") or []) if o]
        pvt = observation.get("private", {})
        shed = dict(pvt.get("shed", {}))

        rec = _ppo_recommend(money, day, unlocked, shed)
        if rec is None:
            action = dict(action, market=orders[:10])
            return action

        # Expansion gating: block BUY_LAND unless PPO picks that quadrant
        land_orders = [o for o in orders if o and o[0] == "BUY_LAND"]
        if land_orders:
            quad = _EXPAND_BY_QUAD.get(rec)
            allowed = None
            if rec in (3, 4, 5):
                allowed = {3: "NE", 4: "SW", 5: "SE"}.get(rec)
            if allowed is None:
                orders = [o for o in orders if not (o and o[0] == "BUY_LAND")]
                _PPO_REPORT["expansions_blocked"] += 1
            else:
                for o in land_orders:
                    if len(o) >= 3 and o[2] != allowed:
                        orders = [x for x in orders if x is not o]
                        _PPO_REPORT["expansions_blocked"] += 1

        # Trickle liquidation when PPO selects it
        if rec == 2 and len(orders) < 10:
            already_selling = {{o[1] for o in orders if len(o) >= 2 and o[0] == "SELL"}}
            wool = shed.get("WOOL", 0)
            milk = shed.get("MILK", 0)
            straw = shed.get("STRAWBERRY", 0)
            for item, stock in (("WOOL", wool), ("MILK", milk), ("STRAWBERRY", straw)):
                if stock >= 2 and item not in already_selling and len(orders) < 10:
                    qty = min(3, int(stock)) if day < 28 else int(stock)
                    orders.append(["SELL", item, qty])
                    _PPO_REPORT["trickle_sells"] += 1

        # Push hiring when PPO selects it
        if rec == 6 and len(orders) < 10:
            has_hire = any(o and o[0] == "HIRE" for o in orders)
            if not has_hire:
                orders.append(["HIRE", 1, 1])
                _PPO_REPORT["hire_pushes"] += 1

        action = dict(action, market=orders[:10])
    except Exception:
        _PPO_REPORT["errors"] += 1

    return action


agent.telemetry = _PPO_REPORT
agent = globals().pop("agent")
kaggle_submission_agent = agent
'''


def ppo_options_layer_source(ckpt_path: str) -> str:
    """Generate the PPO gate layer with the checkpoint embedded as base64."""
    with open(ckpt_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return _LAYER_TEMPLATE.format(ckpt_b64=b64)


# ---------------------------------------------------------------------------
# Package builder
# ---------------------------------------------------------------------------


def build_ppo_agent_package(
    ckpt_path: str,
    out_zip: Optional[str] = None,
    out_py: Optional[str] = None,
) -> str:
    """Compile hybrid chassis + PPO gate layer into a standalone submission zip."""
    from hybrid_chassis.builder import compile_standalone

    target_py = out_py or os.path.join(DIST, "main_ppo.py")
    os.makedirs(os.path.dirname(os.path.abspath(target_py)), exist_ok=True)
    compile_standalone(target_py)

    with open(target_py, "r", encoding="utf-8") as f:
        src = f.read()
    src = src.rstrip() + "\n\n" + GATE_PREFIX + ppo_options_layer_source(ckpt_path) + "\n"
    with open(target_py, "w", encoding="utf-8") as f:
        f.write(src)

    target_zip = out_zip or os.path.join(DIST, "head_ppo_submission.zip")
    os.makedirs(os.path.dirname(os.path.abspath(target_zip)), exist_ok=True)
    with zipfile.ZipFile(target_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(target_py, arcname="main.py")
    return target_zip


# ---------------------------------------------------------------------------
# Behavior-clone initialization from expert replays
# ---------------------------------------------------------------------------


def ppo_bc_init(
    model: ActorCritic,
    transitions: Sequence[Dict[str, Any]],
    hp: PPOHyperParams,
    epochs: int = 3,
    device: str = "cpu",
) -> Dict[str, float]:
    """Supervised policy BC on expert macro-labels (phase3_bc.label_expert_macro_option)."""
    from muzero.encoding import encode_observation

    labeled: List[Tuple[Any, int]] = []
    for tr in transitions:
        if tr.get("obs") is None or tr.get("action") is None:
            continue
        obs = tr["obs"]
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        try:
            labeled.append((obs, int(tr["action"])))
        except Exception:
            continue
    if not labeled:
        return {"bc_loss": 0.0, "bc_accuracy": 0.0, "n": 0}

    opt = torch.optim.Adam(model.parameters(), lr=hp.lr)
    model.train()
    last_loss = 0.0
    for _ in range(epochs):
        random.shuffle(labeled)
        for i in range(0, len(labeled), hp.batch_size):
            batch = labeled[i : i + hp.batch_size]
            obs = torch.stack([o for o, _ in batch])
            acts = torch.tensor([a for _, a in batch], dtype=torch.long)
            logits, _ = model(obs)
            loss = torch.nn.functional.cross_entropy(logits, acts)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.max_grad_norm)
            opt.step()
            last_loss = float(loss.detach())

    model.eval()
    correct = 0
    with torch.no_grad():
        for o, a in labeled:
            logits, _ = model(o.unsqueeze(0) if o.dim() == 1 else o)
            correct += int(torch.argmax(logits, dim=-1).item() == a)
    return {
        "bc_loss": last_loss,
        "bc_accuracy": correct / max(1, len(labeled)),
        "n": len(labeled),
    }


# ---------------------------------------------------------------------------
# PPO controller training (BC init + self-play fine-tune)
# ---------------------------------------------------------------------------


def train_ppo_controller(
    config: Dict[str, Any],
    *,
    hp: Optional[PPOHyperParams] = None,
    bc_transitions: Optional[Sequence[Dict[str, Any]]] = None,
    seed: int = 10101,
    out_ckpt: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[ActorCritic, Dict[str, Any]]:
    """Train the PPO microcontroller: expert BC init, then Adam self-play fine-tune."""
    hp = hp or PPOHyperParams()
    torch.manual_seed(seed)
    random.seed(seed)

    model = ActorCritic(obs_dim=hp.obs_dim, hidden_dim=hp.hidden_dim, num_actions=hp.num_actions)
    report: Dict[str, Any] = {"hyperparams": hp.to_dict(), "seed": seed}

    bc_stats = {"bc_loss": 0.0, "bc_accuracy": 0.0, "n": 0}
    if bc_transitions:
        bc_stats = ppo_bc_init(model, bc_transitions, hp, epochs=3)
    report["bc"] = bc_stats
    if verbose:
        print(f"  [PPO] BC init: loss={bc_stats['bc_loss']:.4f} acc={bc_stats['bc_accuracy']:.3f} n={bc_stats['n']}")

    opt = torch.optim.Adam(model.parameters(), lr=hp.lr)
    rollout_rounds = int(config.get("ppo_rounds", 20))
    episodes_per_round = int(config.get("ppo_episodes", 8))
    losses: List[float] = []
    rewards: List[float] = []

    for rnd in range(rollout_rounds):
        buffer, mean_rew = ppo_rollout_batch(
            model, hp, n_episodes=episodes_per_round, base_seed=seed + rnd * 100,
        )
        stats = train_ppo(model, buffer, hp, optimizer=opt)
        losses.append(stats["total_loss"])
        rewards.append(mean_rew)
        if verbose and (rnd % 5 == 0 or rnd == rollout_rounds - 1):
            print(f"  [PPO] round {rnd + 1}/{rollout_rounds}: loss={stats['total_loss']:.4f} "
                  f"entropy={stats['entropy']:.3f} ep_rew={mean_rew:+.4f}")

    report["final_loss"] = sum(losses) / max(1, len(losses))
    report["mean_episode_reward"] = sum(rewards) / max(1, len(rewards))
    report["rounds"] = rollout_rounds

    out_ckpt = out_ckpt or os.path.join(PPO_DIR, DEFAULT_PPO_CHECKPOINT)
    os.makedirs(os.path.dirname(os.path.abspath(out_ckpt)), exist_ok=True)
    save_ppo_checkpoint(model, out_ckpt, meta=config, optimizer=opt, atomic=True)
    report["ckpt"] = out_ckpt
    return model, report


# ---------------------------------------------------------------------------
# Offline grid search configs (PPO dials x MuZero dials)
# ---------------------------------------------------------------------------


def ppo_grid_configs(n_trials: int = 12) -> List[Dict[str, Any]]:
    """Deterministic grid over PPO hyperparameter dials for offline tuning."""
    lrs = [1e-4, 3e-4, 1e-3]
    gammas = [0.98, 0.99, 0.995]
    clips = [0.1, 0.2, 0.3]
    entropies = [0.005, 0.01, 0.02]
    gaes = [0.90, 0.95, 0.99]

    configs: List[Dict[str, Any]] = []
    trial = 0
    for lr in lrs:
        for gamma in gammas:
            for clip in clips:
                if trial >= n_trials:
                    break
                trial += 1
                configs.append({
                    "name": f"PPO-Grid-{trial:02d}",
                    "controller": "ppo",
                    "ppo_lr": lr,
                    "ppo_gamma": gamma,
                    "ppo_clip": clip,
                    "ppo_entropy": entropies[trial % len(entropies)],
                    "ppo_gae_lambda": gaes[trial % len(gaes)],
                    "ppo_rounds": 20,
                    "ppo_episodes": 8,
                    "hidden_dim": 64,
                    "ppo_epochs": 4,
                    "ppo_batch_size": 64,
                })
    return configs[:n_trials]


# ---------------------------------------------------------------------------
# Full PPO trial with champion promotion gating
# ---------------------------------------------------------------------------


def execute_ppo_trial(
    config: Dict[str, Any],
    s_champion: float,
    champion_delta_threshold: float = 6000.0,
    eval_episodes: int = 20,
    n_workers: int = 8,
    save_replays: bool = True,
    seed: int = 10101,
) -> Dict[str, Any]:
    """Train PPO controller, package it, evaluate vs champion, gate promotion."""
    from evaluation.benchmarks import find_baselines, gate_candidate_vs_2033
    from pipelines.phases.phase4_continuous_loop import run_evaluation_episodes

    trial_start = time.time()
    os.makedirs(PPO_DIR, exist_ok=True)
    hp = PPOHyperParams(
        lr=float(config.get("ppo_lr", 3e-4)),
        gamma=float(config.get("ppo_gamma", 0.99)),
        gae_lambda=float(config.get("ppo_gae_lambda", 0.95)),
        clip_epsilon=float(config.get("ppo_clip", 0.2)),
        entropy_coef=float(config.get("ppo_entropy", 0.01)),
        hidden_dim=int(config.get("hidden_dim", 64)),
        ppo_epochs=int(config.get("ppo_epochs", 4)),
        batch_size=int(config.get("ppo_batch_size", 64)),
    )
    config = dict(config)
    config.setdefault("name", "PPO-Controller")
    config.setdefault("seed", seed)

    print("\n" + "#" * 90)
    print(f"  [PPO TRIAL] {config['name']} | lr={hp.lr} gamma={hp.gamma} clip={hp.clip_epsilon} "
          f"ent={hp.entropy_coef} gae={hp.gae_lambda}")
    print(f"  Target: S_champ (${s_champion:,.0f}) + ${champion_delta_threshold:,.0f} coins")
    print("#" * 90)

    ckpt = os.path.join(PPO_DIR, f"ppo_{config['name'].replace(' ', '_')}.pt")
    model, train_report = train_ppo_controller(config, hp=hp, seed=seed, out_ckpt=ckpt)

    pkg_zip = os.path.join(DIST, f"head_ppo_{config['name'].replace(' ', '_')}.zip")
    build_ppo_agent_package(ckpt, out_zip=pkg_zip)

    print(f"\n[PPO Evaluation] Running E = {eval_episodes} episodes against benchmark distribution...")
    baselines = find_baselines(BASELINES)
    eval_report = run_evaluation_episodes(
        candidate_pkg=pkg_zip,
        baseline_paths=baselines,
        n_episodes=eval_episodes,
        base_seed=seed + 2000,
        n_workers=n_workers,
        save_replays=save_replays,
    )

    r_candidate = float(eval_report["avg_score"])
    win_pct = float(eval_report.get("win_pct", 0.0))
    avg_delta = float(eval_report.get("avg_delta", 0.0))
    margin = r_candidate - s_champion

    promoted_by_score = r_candidate >= s_champion + champion_delta_threshold
    promoted_by_winrate = win_pct >= 75.0 and avg_delta > 0.0 and r_candidate >= 95000.0
    is_promoted = bool(promoted_by_score or promoted_by_winrate)

    reasons = []
    if promoted_by_score:
        reasons.append(f"Score >= Target (${r_candidate:,.0f} >= ${s_champion + champion_delta_threshold:,.0f})")
    if promoted_by_winrate:
        reasons.append(f"Baseline Dominance ({win_pct:.1f}% Win Rate, delta: {avg_delta:+,.0f})")

    gate_passed: Optional[bool] = None
    try:
        gate_report = gate_candidate_vs_2033(pkg_zip, games=2, n_workers=min(2, n_workers),
                                             require_win_rate=0.5, quiet=False)
        gate_passed = bool(gate_report.get("gate", {}).get("passed"))
        reasons.append("2033 Gate PASS" if gate_passed else "2033 Gate FAIL")
        if not gate_passed:
            is_promoted = False
    except Exception as e:
        print(f"[gate] 2033 gate error — blocking promotion ({e})")
        gate_passed = False
        is_promoted = False

    status_str = "PROMOTED" if is_promoted else "REJECTED"
    print(f"\n[{config['name']}] Score: {r_candidate:,.0f} coins | Margin: {margin:+,.0f} | "
          f"Win%: {win_pct:.1f}% | Status: [{status_str}]")

    if is_promoted:
        champ_ckpt = os.path.join(PPO_DIR, f"ppo_champion_{int(r_candidate)}.pt")
        save_ppo_checkpoint(model, champ_ckpt, meta=config, atomic=True)
        print(f"  PPO champion saved -> {champ_ckpt}")
        with open(CHAMPION_HISTORY_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] PPO PROMOTED {config['name']} | "
                    f"Score: ${r_candidate:,.0f} | Margin: +${margin:,.0f} | "
                    f"lr={hp.lr} gamma={hp.gamma} clip={hp.clip_epsilon}\n")

    record = {
        "controller": "ppo",
        "config": config,
        "status": status_str,
        "is_promoted": is_promoted,
        "score": r_candidate,
        "margin": margin,
        "s_champion_prior": s_champion,
        "s_champion_new": r_candidate if is_promoted else s_champion,
        "train_report": train_report,
        "eval_report": {
            "n_episodes": eval_report["n_episodes"],
            "avg_score": r_candidate,
            "win_pct": eval_report["win_pct"],
            "wins": eval_report["wins"],
            "losses": eval_report["losses"],
            "ties": eval_report["ties"],
        },
        "gate_2033_passed": gate_passed,
        "elapsed_seconds": round(time.time() - trial_start, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="PPO microcontroller macrocontroller tuning")
    sub = parser.add_subparsers(dest="command")

    p_grid = sub.add_parser("grid", help="Run offline PPO hyperparameter grid search")
    p_grid.add_argument("--trials", type=int, default=12)
    p_grid.add_argument("--s-champion", type=float, default=99291.5)
    p_grid.add_argument("--eval-episodes", type=int, default=20)
    p_grid.add_argument("--workers", type=int, default=8)
    p_grid.add_argument("--seed", type=int, default=10101)

    p_train = sub.add_parser("train", help="Train a single PPO controller")
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--gamma", type=float, default=0.99)
    p_train.add_argument("--clip", type=float, default=0.2)
    p_train.add_argument("--entropy", type=float, default=0.01)
    p_train.add_argument("--gae", type=float, default=0.95)
    p_train.add_argument("--rounds", type=int, default=20)
    p_train.add_argument("--episodes", type=int, default=8)
    p_train.add_argument("--hidden", type=int, default=64)
    p_train.add_argument("--seed", type=int, default=10101)
    p_train.add_argument("--out", default=os.path.join(PPO_DIR, DEFAULT_PPO_CHECKPOINT))

    args = parser.parse_args()
    if args.command == "grid":
        configs = ppo_grid_configs(args.trials)
        results = []
        for cfg in configs:
            rec = execute_ppo_trial(
                cfg, s_champion=args.s_champion, eval_episodes=args.eval_episodes,
                n_workers=args.workers, seed=args.seed,
            )
            results.append(rec)
        print("\n" + "=" * 90)
        print("PPO GRID SEARCH COMPLETE")
        for rec in sorted(results, key=lambda r: r["score"], reverse=True):
            print(f"  {rec['config']['name']:<20} Score: ${rec['score']:>9,.0f} | "
                  f"Status: {rec['status']} | Margin: {rec['margin']:+,.0f}")
        return 0
    if args.command == "train":
        config = {
            "name": "PPO-CLI",
            "ppo_lr": args.lr, "ppo_gamma": args.gamma, "ppo_clip": args.clip,
            "ppo_entropy": args.entropy, "ppo_gae_lambda": args.gae,
            "ppo_rounds": args.rounds, "ppo_episodes": args.episodes,
            "hidden_dim": args.hidden,
        }
        _, report = train_ppo_controller(config, seed=args.seed, out_ckpt=args.out)
        print(json.dumps(report, indent=2))
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
