"""
blackmarble: package for Black Marble processing.
This package provides a functional API for processing NASA Black Marble data,
including Landsat temporal compositing, VIIRS nighttime lights integration,
OSM road enhancement, and urban index calculation.
"""
from .config import PipelineConfig
from .pipeline import black_marble_pipeline, black_marble_pipeline_sync
from .typing import ArrayLike, BBox, BoolArray, CRSType, FloatArray, IntArray, UInt8Array

__version__ = "0.1.0"


__all__ = [
    "black_marble_pipeline",
    "black_marble_pipeline_sync",
    "PipelineConfig",
    # Type aliases
    "ArrayLike",
    "FloatArray",
    "IntArray", 
    "UInt8Array",
    "BoolArray",
    "BBox",
    "CRSType",
]
