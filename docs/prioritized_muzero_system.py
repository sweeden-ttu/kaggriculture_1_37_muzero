# Copyright 2026 Scott Weeden
"""Thin pointer: prioritized spatial MuZero lives in ``muzero`` production modules.

Historical monolithic sketch: ``docs/adr/archive/prioritized_muzero_system.py``.
Import from ``muzero`` instead of this file.
"""
from muzero.prioritized_system import *  # noqa: F401,F403
from muzero.prioritized_system import __all__  # noqa: F401
