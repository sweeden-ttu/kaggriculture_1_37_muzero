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

"""Analytical daily economy — Phase 1 bootstrap teacher only.

Must only be imported by ``pipelines.phases.phase1_bootstrap`` (and tests of
that teacher). MCTS / ``_core`` must never call into this module — search stays
in latent space via g_θ.
"""
from __future__ import annotations

__author__ = "Scott Weeden"
__license__ = "Apache-2.0"

from typing import Dict, List, Optional, Tuple

import torch

from .encoding import encode_macro_state, legal_macro_options, wealth
from .types import (
    DEFAULT_PRICES,
    LAND_BUFFERS,
    LAND_ORDER,
    LAND_PRICES,
    NUM_MACRO_OPTIONS,
    POLICY_FIBONACCI,
    REWARD_SCALE,
    MacroAction,
    MacroOption,
    MacroState,
)

QUADRANT_DAILY_PRODUCE: Dict[str, Dict[str, float]] = {
    "NW": {"WHEAT": 2.5, "EGG": 0.8, "MILK": 0.4},
    "NE": {"CARROT": 4.5},
    "SW": {"TOMATO": 1.5, "STRAWBERRY": 1.0},
    "SE": {"MELON": 0.35},
}

QUADRANT_DAILY_SEED_COST: Dict[str, float] = {
    "NW": 15.0,
    "NE": 45.0,
    "SW": 50.0,
    "SE": 40.0,
}

META_HANDS_BY_DAY: Tuple[Tuple[int, int], ...] = (
    (0, 4),
    (8, 8),
    (16, 10),
    (27, 10),
    (30, 4),
)
META_SE_EXTRA_BUFFER: float = 2500.0
META_TRICKLE_BATCH: int = 8
META_COWS_TARGET: int = 8
META_SHEEP_TARGET: int = 4


class AnalyticalEconomicModel:
    """O(1) daily cashflow model used as Phase-1 supervised dynamics teacher."""

    @staticmethod
    def step(state: MacroState, action: MacroAction) -> MacroState:
        new_state = state.copy()
        new_state.day += 1

        if action.expansion_target and action.expansion_target not in new_state.unlocked:
            cost = LAND_PRICES.get(action.expansion_target, 0.0)
            if new_state.capital >= cost:
                new_state.capital -= cost
                new_state.unlocked = tuple(
                    sorted(list(new_state.unlocked) + [action.expansion_target])
                )

        labor_cost = 0.0
        for h in range(action.target_hands):
            fib_idx = min(len(POLICY_FIBONACCI) - 1, max(0, h))
            labor_cost += POLICY_FIBONACCI[fib_idx]
        new_state.capital = max(0.0, new_state.capital - labor_cost)

        for q in new_state.unlocked:
            for item, qty in QUADRANT_DAILY_PRODUCE.get(q, {}).items():
                new_state.shed[item] = new_state.shed.get(item, 0.0) + qty
            new_state.capital = max(
                0.0, new_state.capital - QUADRANT_DAILY_SEED_COST.get(q, 15.0)
            )

        total_items = sum(new_state.shed.values())
        is_endgame = new_state.day >= 27
        force_dump = total_items >= 85.0

        for item, qty in list(new_state.shed.items()):
            if qty <= 0:
                continue
            price = new_state.prices.get(item, DEFAULT_PRICES.get(item, 25.0))
            if item == "WHEAT" and new_state.active_animals > 0 and not is_endgame:
                sellable = max(0.0, qty - 15.0)
            else:
                sellable = qty
            if sellable > 0 and (
                is_endgame
                or force_dump
                or price >= DEFAULT_PRICES.get(item, 0.0) * 0.95
            ):
                new_state.capital += sellable * price
                new_state.shed[item] -= sellable

        return new_state

    @staticmethod
    def evaluate_terminal_wealth(state: MacroState) -> float:
        return wealth(state)


def _default_hands(day: int, unlocked_count: int) -> int:
    base = 4
    for max_day, hands in META_HANDS_BY_DAY:
        if day <= max_day:
            base = hands
            break
    if unlocked_count >= 3:
        base = min(10, base + 1)
    elif unlocked_count == 1 and day <= 8:
        base = min(base, 6)
    return base


def _next_expansion_target(state: MacroState) -> Optional[str]:
    for q in LAND_ORDER:
        if q not in state.unlocked:
            return q
    return None


def meta_prior_bias(option: MacroOption, state: MacroState) -> float:
    """Multiplicative prior boost for Phase-1 teacher labeling."""
    nxt = _next_expansion_target(state)
    if option == MacroOption.EXPAND_SE:
        return 0.15
    if option == MacroOption.EXPAND_NE and nxt == "NE":
        return 1.8 if state.day >= 6 else 1.2
    if option == MacroOption.EXPAND_SW and nxt == "SW":
        return 1.8 if state.day >= 8 else 1.2
    if option == MacroOption.HIRE_HANDS and state.day <= 27:
        return 1.5
    if option == MacroOption.TRICKLE_LIQUIDATION:
        premium = sum(
            state.shed.get(k, 0.0)
            for k in ("MELON", "MILK", "WOOL", "STRAWBERRY", "EGG", "TOMATO")
        )
        return 1.6 if premium >= META_TRICKLE_BATCH else 1.1
    if option == MacroOption.EXPANSION_PREPARATION and nxt in ("NE", "SW"):
        return 1.3
    if option == MacroOption.PASS and nxt == "SE":
        return 1.4
    return 1.0


def meta_expert_option(state: MacroState) -> MacroOption:
    """Behavioral cloning label: imitate live-ladder playbook choices."""
    nxt = _next_expansion_target(state)
    premium = sum(
        state.shed.get(k, 0.0)
        for k in ("MELON", "MILK", "WOOL", "STRAWBERRY", "TOMATO")
    )

    if nxt == "SE":
        need = LAND_PRICES["SE"] + LAND_BUFFERS["SE"] + META_SE_EXTRA_BUFFER
        if state.capital < need:
            if premium >= META_TRICKLE_BATCH:
                return MacroOption.TRICKLE_LIQUIDATION
            return MacroOption.HIRE_HANDS if state.day <= 27 else MacroOption.PASS

    if nxt == "NE" and state.day >= 6 and state.capital >= LAND_PRICES["NE"] + LAND_BUFFERS["NE"]:
        return MacroOption.EXPAND_NE
    if nxt == "SW" and state.day >= 8 and state.capital >= LAND_PRICES["SW"] + LAND_BUFFERS["SW"]:
        return MacroOption.EXPAND_SW
    if nxt == "SE" and state.capital >= LAND_PRICES["SE"] + LAND_BUFFERS["SE"] + META_SE_EXTRA_BUFFER:
        return MacroOption.EXPAND_SE

    if premium >= META_TRICKLE_BATCH and state.day >= 6:
        return MacroOption.TRICKLE_LIQUIDATION

    hands = _default_hands(state.day, len(state.unlocked))
    if state.day <= 27 and hands < 10:
        return MacroOption.HIRE_HANDS

    if nxt in ("NE", "SW") and state.capital >= 0.6 * (
        LAND_PRICES[nxt] + LAND_BUFFERS[nxt]
    ):
        return MacroOption.EXPANSION_PREPARATION

    if state.day <= 8:
        return MacroOption.FRONT_LOAD_SURVIVAL
    return MacroOption.PASS


def option_to_macro_action(option: MacroOption, state: MacroState) -> MacroAction:
    """Map MacroOption onto analytical daily MacroAction."""
    hands = _default_hands(state.day, len(state.unlocked))
    next_target = _next_expansion_target(state)

    if option == MacroOption.FRONT_LOAD_SURVIVAL:
        return MacroAction(expansion_target=None, target_hands=min(8, max(4, hands)))
    if option == MacroOption.EXPANSION_PREPARATION:
        exp = None
        if next_target in ("NE", "SW"):
            need = LAND_PRICES[next_target] + LAND_BUFFERS[next_target] + 500.0
            if state.capital >= need:
                exp = next_target
        elif next_target == "SE":
            need = LAND_PRICES["SE"] + LAND_BUFFERS["SE"] + META_SE_EXTRA_BUFFER
            if state.capital >= need:
                exp = "SE"
        return MacroAction(expansion_target=exp, target_hands=hands)
    if option == MacroOption.TRICKLE_LIQUIDATION:
        return MacroAction(expansion_target=None, target_hands=max(6, hands - 1))
    if option == MacroOption.EXPAND_NE:
        return MacroAction(
            expansion_target="NE" if "NE" not in state.unlocked else None,
            target_hands=max(hands, 8),
        )
    if option == MacroOption.EXPAND_SW:
        return MacroAction(
            expansion_target="SW" if "SW" not in state.unlocked else None,
            target_hands=max(hands, 9),
        )
    if option == MacroOption.EXPAND_SE:
        return MacroAction(
            expansion_target="SE" if "SE" not in state.unlocked else None,
            target_hands=hands,
        )
    if option == MacroOption.HIRE_HANDS:
        return MacroAction(expansion_target=None, target_hands=min(10, max(hands + 2, 9)))
    return MacroAction(expansion_target=None, target_hands=hands)


def meta_economic_step(state: MacroState, option: MacroOption) -> MacroState:
    """Analytical day-step with live-meta overlays for Phase-1 targets."""
    act = option_to_macro_action(option, state)
    nxt = AnalyticalEconomicModel.step(state, act)
    n_quad = len(nxt.unlocked)

    if n_quad >= 2 and nxt.day >= 8:
        cows = min(META_COWS_TARGET, 2 + 3 * (n_quad - 1))
        sheep = min(META_SHEEP_TARGET + 2, 2 * n_quad)
        animal_cash = cows * 80.0 + sheep * (200.0 / 3.0) + cows * 40.0
        straw_tiles = 6 if n_quad >= 2 else 0
        straw_cash = straw_tiles * 60.0
        labor_scale = 0.55 + 0.045 * min(10, act.target_hands)
        nxt.capital += (animal_cash + straw_cash) * labor_scale

    if option == MacroOption.EXPAND_SE:
        nxt.capital = max(0.0, nxt.capital - 1200.0)

    if option == MacroOption.TRICKLE_LIQUIDATION:
        nxt.capital += 350.0

    if option == MacroOption.HIRE_HANDS and n_quad >= 2:
        nxt.capital += 150.0

    return nxt


def generate_analytical_transitions(
    n_samples: int = 256,
    max_day: int = 28,
    meta_bc_frac: float = 0.7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Synthetic (obs, action, reward, next_obs) tuples for Phase-1 bootstrap."""
    import random

    obs_list: List[torch.Tensor] = []
    act_list: List[int] = []
    rew_list: List[float] = []
    next_list: List[torch.Tensor] = []

    for _ in range(n_samples):
        day = random.randint(0, max_day)
        if random.random() < 0.55:
            unlocked_n = 1 if day < 8 else (2 if day < 18 else 3)
        else:
            unlocked_n = random.randint(1, min(4, 1 + day // 8))
        unlocked = tuple(["NW", "NE", "SW", "SE"][:unlocked_n])
        capital = random.uniform(500.0, 20000.0)
        if random.random() < 0.4:
            nxt = None
            for q in LAND_ORDER:
                if q not in unlocked:
                    nxt = q
                    break
            if nxt:
                capital = LAND_PRICES[nxt] + LAND_BUFFERS[nxt] + random.uniform(-100, 3000)
                capital = max(200.0, capital)
        shed = {
            "WHEAT": random.uniform(0, 40),
            "MELON": random.uniform(0, 12),
            "STRAWBERRY": random.uniform(0, 20),
            "MILK": random.uniform(0, 15),
            "WOOL": random.uniform(0, 10),
        }
        state = MacroState(
            day=day,
            capital=capital,
            unlocked=unlocked,
            shed=shed,
            prices=dict(DEFAULT_PRICES),
            active_animals=random.randint(1, 8),
        )
        options = legal_macro_options(state)
        if random.random() < meta_bc_frac:
            opt = meta_expert_option(state)
            if opt not in options:
                opt = random.choice(options)
        else:
            opt = random.choice(options)
        nxt_state = meta_economic_step(state, opt)
        reward = (wealth(nxt_state) - wealth(state)) * REWARD_SCALE

        obs_list.append(encode_macro_state(state))
        act_list.append(int(opt))
        rew_list.append(reward)
        next_list.append(encode_macro_state(nxt_state))

    return (
        torch.stack(obs_list),
        torch.tensor(act_list, dtype=torch.long),
        torch.tensor(rew_list, dtype=torch.float32),
        torch.stack(next_list),
    )


def generate_analytical_trajectories(
    n_trajectories: int = 64,
    k_steps: int = 5,
    max_day: int = 25,
    meta_bc_frac: float = 0.85,
) -> Dict[str, torch.Tensor]:
    """Simulate analytical MacroState trajectories for K-step Phase-1 unrolls."""
    import random

    obs_all, acts_all, rews_all, vals_all, pol_all = [], [], [], [], []
    for _ in range(n_trajectories):
        day = random.randint(0, max_day)
        if random.random() < 0.55:
            unlocked_n = 1 if day < 8 else (2 if day < 18 else 3)
        else:
            unlocked_n = random.randint(1, min(4, 1 + day // 8))
        unlocked = tuple(["NW", "NE", "SW", "SE"][:unlocked_n])
        capital = random.uniform(500.0, 20000.0)
        if random.random() < 0.4:
            nxt = None
            for q in LAND_ORDER:
                if q not in unlocked:
                    nxt = q
                    break
            if nxt:
                capital = LAND_PRICES[nxt] + LAND_BUFFERS[nxt] + random.uniform(-100, 3000)
                capital = max(200.0, capital)

        shed = {
            "WHEAT": random.uniform(0, 40),
            "MELON": random.uniform(0, 12),
            "STRAWBERRY": random.uniform(0, 20),
            "MILK": random.uniform(0, 15),
            "WOOL": random.uniform(0, 10),
        }
        state = MacroState(
            day=day,
            capital=capital,
            unlocked=unlocked,
            shed=shed,
            prices=dict(DEFAULT_PRICES),
            active_animals=random.randint(1, 8),
        )

        obs_traj = [encode_macro_state(state)]
        acts_traj: List[int] = []
        rews_traj: List[float] = []
        vals_traj: List[float] = [wealth(state) * REWARD_SCALE]
        pol_traj: List[torch.Tensor] = []

        curr = state
        for _ in range(k_steps):
            options = legal_macro_options(curr)
            if random.random() < meta_bc_frac:
                opt = meta_expert_option(curr)
                if opt not in options:
                    opt = random.choice(options)
            else:
                opt = random.choice(options)

            opt_oh = torch.zeros(NUM_MACRO_OPTIONS)
            opt_oh[int(opt)] = 1.0
            pol_traj.append(opt_oh)

            nxt_state = meta_economic_step(curr, opt)
            rew = (wealth(nxt_state) - wealth(curr)) * REWARD_SCALE
            acts_traj.append(int(opt))
            rews_traj.append(rew)
            obs_traj.append(encode_macro_state(nxt_state))
            vals_traj.append(wealth(nxt_state) * REWARD_SCALE)
            curr = nxt_state

        final_opts = legal_macro_options(curr)
        opt_final = meta_expert_option(curr)
        if opt_final not in final_opts:
            opt_final = random.choice(final_opts)
        opt_oh = torch.zeros(NUM_MACRO_OPTIONS)
        opt_oh[int(opt_final)] = 1.0
        pol_traj.append(opt_oh)

        obs_all.append(torch.stack(obs_traj))
        acts_all.append(torch.tensor(acts_traj, dtype=torch.long))
        rews_all.append(torch.tensor(rews_traj, dtype=torch.float32))
        vals_all.append(torch.tensor(vals_traj, dtype=torch.float32))
        pol_all.append(torch.stack(pol_traj))

    return {
        "obs": torch.stack(obs_all),
        "actions": torch.stack(acts_all),
        "rewards": torch.stack(rews_all),
        "values": torch.stack(vals_all),
        "policies": torch.stack(pol_all),
    }


def bootstrap_muzero(
    model=None,
    steps: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
):
    """Quick analytical bootstrap so untrained MuZero is not random noise.

    Lazy-imports network/loss from ``_core`` so economy remains the sole teacher
    entry while avoiding a hard import cycle at module load.
    """
    import torch.nn as nn
    from ._core import MuZeroNetwork, muzero_bootstrap_loss

    model = model or MuZeroNetwork()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        obs, acts, rews, next_obs = generate_analytical_transitions(n_samples=batch_size)
        losses = muzero_bootstrap_loss(model, obs, acts, rews, next_obs)
        opt.zero_grad()
        losses["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    model.eval()
    return model
