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

"""Compatibility facade → ``pipelines.phases.phase4_continuous_loop``."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from pipelines.phases.phase4_continuous_loop import (
    execute_candidate_trial,
    generate_candidate_hyperparams,
    main,
    run_phase4_continuous_loop,
    run_evaluation_episodes,
    run_online_selfplay_data_generation,
    setup_baseline_champion,
)

if __name__ == "__main__":
    sys.exit(main())
