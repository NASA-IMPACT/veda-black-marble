"""
blackmarble: package for Black Marble processing.
This package provides a functional API for processing NASA Black Marble data,
including Landsat temporal compositing, VIIRS nighttime lights integration,
OSM road enhancement, and urban index calculation.
"""

__version__ = "0.1.0"

# Export main pipeline function for convenience
from .pipeline import black_marble_pipeline
from .config import PipelineConfig

__all__ = [
    "black_marble_pipeline",
    "PipelineConfig",
]