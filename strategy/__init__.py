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

"""Market-aware planning layers for the Kaggriculture chassis.

The engine's market is a known, cheap, deterministic model: per-product price
curves, per-shop consumption, and per-unit lockstep order matching are all
public code. These modules use that model directly (no learned dynamics) to
forecast demand, recover the rival's sales from public inventory moves, and
choose production so that both players' output does not glut the same book.
"""
