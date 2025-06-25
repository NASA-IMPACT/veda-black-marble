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

from .visualize import (
    apply_inferno_colormap,
)

__all__ = [
    # Contrast functions (used by pipeline)
    "chain_enhancements",
    "enhance_contrast",
    "normalize_data",
    # Visualization functions (used by pipeline)
    "apply_inferno_colormap",
]