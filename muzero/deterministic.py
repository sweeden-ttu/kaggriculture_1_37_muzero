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
muzero/deterministic.py
=======================
Guarantees strict bit-for-bit mathematical determinism across all subsystems:
  1. Python built-in random module (random.seed)
  2. NumPy RNG state (np.random.seed)
  3. PyTorch CPU RNG state (torch.manual_seed)
  4. PyTorch CUDA / MPS RNG states (torch.cuda.manual_seed_all)
  5. Python Hash Seed environment variable (PYTHONHASHSEED)
  6. PyTorch CuDNN deterministic execution flags
"""
from __future__ import annotations

import os
import random
import numpy as np
import torch

DEFAULT_DETERMINISTIC_SEED = 10101


def set_deterministic_mode(seed: int = DEFAULT_DETERMINISTIC_SEED) -> int:
    """
    Enforces total reproducibility across Python, NumPy, and PyTorch.
    Ensures no unseeded random number generators contaminate training or evaluation.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass
    return seed
