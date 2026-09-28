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

"""Fast in-process arena for Kaggriculture agents.

Every game runs in a fresh worker process (module-level agent state cannot leak
between games), seeds are paired across candidates so the town-shop draw is a
controlled variable, and per-product revenue is attributed by replaying the
engine's per-unit lockstep market exactly.
"""
from .harness import GameResult, play_game, play_many, load_agent_module
from .attribution import attribute_game
from .gate import paired_summary, wilson_interval

__all__ = [
    "GameResult",
    "play_game",
    "play_many",
    "load_agent_module",
    "attribute_game",
    "paired_summary",
    "wilson_interval",
]
