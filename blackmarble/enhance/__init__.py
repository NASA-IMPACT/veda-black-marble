"""Enhancement module - ONLY WHAT'S USED."""

from .contrast import enhance_contrast
from .features import enhance_roads, enhance_urban_features
from .visualize import apply_inferno_colormap, create_false_color_composite

__all__ = [
    "enhance_contrast",
    "enhance_roads", 
    "enhance_urban_features",
    "apply_inferno_colormap",
    "create_false_color_composite",
]