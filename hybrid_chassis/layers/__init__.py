"""Layer package for the hybrid chassis.

Runtime layers are `exec`'d by `pipeline.build_pipeline` in declaration order
from `manifest.LAYER_STACK`. They are not imported as normal Python modules
at agent runtime (except `manifest`).
"""
from .manifest import LAYER_STACK, layer_filenames

__all__ = ["LAYER_STACK", "layer_filenames"]
