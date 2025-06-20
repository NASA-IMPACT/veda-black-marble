"""Components for Black Marble pipeline.

This package provides modular, testable components for quality assessment,
spatial preprocessing, temporal processing, output generation, and enhancement
of satellite imagery data.
"""

from .base import QualityComponent, MaskResult, QualityStatistics, ComposableComponent
from .quality import CloudMasker, QualityCompositor, MaskStatistics
from .preprocessing import (
    ProjectionHandler,
    ResolutionResampler,
    BoundsCropper,
    SpatialAligner,
    SpatialResult,
)
from .temporal import (
    TemporalCompositor,
    SceneMerger,
    DateRangeSelector,
    CompositeMethod,
    TemporalResult,
)
from .output import (
    COGGenerator,
    MetadataGenerator,
    VisualizationEnhancer,
    OutputValidator,
    OutputMetadata,
    COGProfile,
)
from .enhancement import (
    RoadEnhancer,
    ContrastEnhancer,
    ColorMapper,
    CompositeEnhancer,
    EnhancementResult,
)

__all__ = [
    # Base classes
    "QualityComponent",
    "ComposableComponent",
    "MaskResult",
    "QualityStatistics",
    # Quality components
    "CloudMasker",
    "QualityCompositor",
    "MaskStatistics",
    # Preprocessing components
    "ProjectionHandler",
    "ResolutionResampler",
    "BoundsCropper",
    "SpatialAligner",
    "SpatialResult",
    # Temporal components
    "TemporalCompositor",
    "SceneMerger",
    "DateRangeSelector",
    "CompositeMethod",
    "TemporalResult",
    # Output components
    "COGGenerator",
    "MetadataGenerator",
    "VisualizationEnhancer",
    "OutputValidator",
    "OutputMetadata",
    "COGProfile",
    # Enhancement components
    "RoadEnhancer",
    "ContrastEnhancer",
    "ColorMapper",
    "CompositeEnhancer",
    "EnhancementResult",
]
