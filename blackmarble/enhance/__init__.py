"""Enhancement module for Black Marble pipeline.

This module provides enhancement capabilities including:
- Contrast and dynamic range adjustments
- Urban and infrastructure feature enhancement
- Visualization preparation and colormap application
"""

from .contrast import (
    adaptive_histogram_equalization,
    chain_enhancements,
    compute_enhancement_statistics,
    enhance_contrast,
    gamma_correction,
    histogram_matching,
    multi_scale_retinex,
    normalize_data,
)

from .features import (
    create_urban_gradient,
    detect_urban_corridors,
    enhance_infrastructure,
    enhance_roads,
    enhance_small_settlements,
    enhance_urban_boundaries,
    enhance_urban_features,
    suppress_blooming,
)

from .visualize import (
    add_north_arrow,
    add_scale_bar,
    apply_colormap,
    apply_inferno_colormap,
    blend_with_hillshade,
    create_comparison_view,
    create_false_color_composite,
    create_hillshade,
    generate_colorbar,
    prepare_for_web_display,
)

__all__ = [
    # Contrast functions
    "adaptive_histogram_equalization",
    "chain_enhancements",
    "compute_enhancement_statistics",
    "enhance_contrast",
    "gamma_correction",
    "histogram_matching",
    "multi_scale_retinex",
    "normalize_data",
    # Feature enhancement functions
    "create_urban_gradient",
    "detect_urban_corridors",
    "enhance_infrastructure",
    "enhance_roads",
    "enhance_small_settlements",
    "enhance_urban_boundaries",
    "enhance_urban_features",
    "suppress_blooming",
    # Visualization functions
    "add_north_arrow",
    "add_scale_bar",
    "apply_colormap",
    "apply_inferno_colormap",
    "blend_with_hillshade",
    "create_comparison_view",
    "create_false_color_composite",
    "create_hillshade",
    "generate_colorbar",
    "prepare_for_web_display",
]