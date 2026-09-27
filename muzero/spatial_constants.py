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

"""Spatial MuZero constants and lossless joint-action plane encoding."""
from __future__ import annotations

from typing import Sequence, Tuple, Union

import torch

OBS_CHANNELS: int = 28
LATENT_CHANNELS: int = 64
BOARD_SIZE: int = 10
NUM_FARMER_ACTIONS: int = 15
NUM_HAND_ASSIGNMENTS: int = 32
NUM_MARKET_ORDERS: int = 20
# Lossless packing: one broadcast plane per discrete sub-action (no % aliasing).
ACTION_CHANNELS: int = NUM_FARMER_ACTIONS + NUM_HAND_ASSIGNMENTS + NUM_MARKET_ORDERS  # 67
SUPPORT_SIZE: int = 601
SUPPORT_MIN: float = -300.0
SUPPORT_MAX: float = 300.0

DEFAULT_SPATIAL_CHECKPOINT: str = "muzero_spatial_checkpoints.pt"

FARMER_SLICE = slice(0, NUM_FARMER_ACTIONS)
HANDS_SLICE = slice(NUM_FARMER_ACTIONS, NUM_FARMER_ACTIONS + NUM_HAND_ASSIGNMENTS)
MARKET_SLICE = slice(
    NUM_FARMER_ACTIONS + NUM_HAND_ASSIGNMENTS,
    ACTION_CHANNELS,
)


def joint_action_to_plane(
    action_tuples: Union[Tuple[int, int, int], Sequence[int], torch.Tensor],
    *,
    batch_size: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Encode joint (farmer, hands, market) ints as [B, ACTION_CHANNELS, 10, 10] planes.

    Each discrete sub-action gets its own channel filled with 1.0 (broadcast over the board).
    """
    if isinstance(action_tuples, torch.Tensor):
        acts = action_tuples.long()
        if acts.dim() == 1:
            acts = acts.unsqueeze(0)
        device = device or acts.device
        b = acts.shape[0]
        plane = torch.zeros(b, ACTION_CHANNELS, BOARD_SIZE, BOARD_SIZE, device=device)
        for i in range(b):
            f_act = int(acts[i, 0].item()) % NUM_FARMER_ACTIONS
            h_act = int(acts[i, 1].item()) % NUM_HAND_ASSIGNMENTS
            m_act = int(acts[i, 2].item()) % NUM_MARKET_ORDERS
            plane[i, f_act, :, :] = 1.0
            plane[i, NUM_FARMER_ACTIONS + h_act, :, :] = 1.0
            plane[i, NUM_FARMER_ACTIONS + NUM_HAND_ASSIGNMENTS + m_act, :, :] = 1.0
        return plane

    f_act, h_act, m_act = int(action_tuples[0]), int(action_tuples[1]), int(action_tuples[2])
    b = 1 if batch_size is None else int(batch_size)
    device = device or torch.device("cpu")
    plane = torch.zeros(b, ACTION_CHANNELS, BOARD_SIZE, BOARD_SIZE, device=device)
    plane[:, f_act % NUM_FARMER_ACTIONS, :, :] = 1.0
    plane[:, NUM_FARMER_ACTIONS + (h_act % NUM_HAND_ASSIGNMENTS), :, :] = 1.0
    plane[:, NUM_FARMER_ACTIONS + NUM_HAND_ASSIGNMENTS + (m_act % NUM_MARKET_ORDERS), :, :] = 1.0
    return plane


def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """Per-sample CE against a soft target distribution (visit counts / MCTS π)."""
    log_p = torch.nn.functional.log_softmax(logits, dim=-1)
    return -(target_probs * log_p).sum(dim=-1)
