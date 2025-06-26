"""Enhancement module for Black Marble pipeline.

This module provides enhancement capabilities including:
- Contrast and dynamic range adjustments
- Urban and infrastructure feature enhancement
- Visualization preparation and colormap application
"""

from .contrast import (
    chain_enhancements,
    enhance_contrast,
    normalize_data,
)

from .features import (
    enhance_roads,
    enhance_urban_features,
)

from .visualize import (
    apply_inferno_colormap,
    create_false_color_composite,
)

__all__ = [
    "chain_enhancements",
    "enhance_contrast",
    "normalize_data",
    "enhance_roads",
    "enhance_urban_features",
    "apply_inferno_colormap",
    "create_false_color_composite",
]