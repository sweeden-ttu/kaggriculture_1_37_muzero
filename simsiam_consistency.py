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

"""Root alias for SimSiam consistency module (docs/Consistency_Loss.md & docs/Representation_Function.md)."""
from __future__ import annotations

from muzero.chassis import (
    SimSiamConsistencyLoss,
    SimSiamProjectionPredictionHead,
    compute_muzero_unroll_loss_with_consistency,
)

__all__ = [
    "SimSiamConsistencyLoss",
    "SimSiamProjectionPredictionHead",
    "compute_muzero_unroll_loss_with_consistency",
]
