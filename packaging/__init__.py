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

"""Packaging, compilation, and submission archive distribution domain."""
from __future__ import annotations

import os as _os
import sys as _sys

# Enable namespace chaining with third-party packaging (e.g. packaging.version used by langsmith/pip/pytest)
for _p in _sys.path:
    if _p and _p not in (".", "") and not _p.startswith(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))):
        _candidate = _os.path.join(_p, "packaging")
        if _os.path.isdir(_candidate) and _candidate not in __path__:
            __path__.append(_candidate)

from .builder import (
    build_hybrid_package,
    build_thirst_baseline,
    embed_weights,
)

__all__ = [
    "build_hybrid_package",
    "build_thirst_baseline",
    "embed_weights",
]
