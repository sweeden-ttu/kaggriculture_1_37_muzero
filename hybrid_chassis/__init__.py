"""Kaggriculture Master Hybrid Replay Chassis Package.

A modularized, production-grade architecture refactored from the monolithic 7,375-line
Kaggle competition chassis.

Subpackages:
  - core: Constants, helpers, observation view, and replay chassis.
  - routes: Compressed base85 route tapes and shop routing.
  - layers: Reactive safety layers + replay-distilled lessons (`manifest.LAYER_STACK`).
  - pipeline: Runtime pipeline builder and agent instantiator.
  - builder: Single-file compiler for Kaggle competition submissions.
"""
from __future__ import annotations

from .builder import compile_standalone
from .pipeline import build_legacy_pipeline, build_pipeline

# Lazy/cached module-level agent instance
_CACHED_AGENT = None
_CACHED_IMPL = None


def get_agent():
    """Return the cached or newly built submission agent (full legacy layer stack)."""
    global _CACHED_AGENT, _CACHED_IMPL
    if _CACHED_AGENT is None:
        # Packaging / Kaggle entrypoints keep the full reactive stack.
        # Canonical reduced chassis: build_pipeline(legacy_layers=False).
        _CACHED_AGENT, _CACHED_IMPL = build_legacy_pipeline()
    return _CACHED_AGENT


def reset_agent_cache() -> None:
    """Drop cached agent (useful after rebuilding layers in-process)."""
    global _CACHED_AGENT, _CACHED_IMPL
    _CACHED_AGENT = None
    _CACHED_IMPL = None


def kaggle_submission_agent(observation, configuration=None):
    """Kaggle environment submission entrypoint."""
    return get_agent()(observation, configuration)


agent = kaggle_submission_agent

__all__ = [
    "agent",
    "kaggle_submission_agent",
    "get_agent",
    "reset_agent_cache",
    "build_pipeline",
    "build_legacy_pipeline",
    "compile_standalone",
]
