"""
blackmarble: package for Black Marble processing.
This package organizes raster I/O, reprojection, coordinate utilities,
data download (Landsat, VIIRS), OSM processing, masking, and pipeline orchestration.
"""

__version__ = "0.1.0"

# Export the new processing functions for easier access
from .processing import (
    # Spatial transformation functions
    resample_image,
    reproject_image,
    crop_to_bounds,
    align_rasters,
    # Mask generation functions
    create_cloud_mask,
    create_quality_mask,
    create_aerosol_mask,
    create_saturation_mask,
    create_water_mask,
    create_viirs_quality_mask,
    combine_masks,
    dilate_mask,
    # Enhancement functions
    enhance_contrast,
    normalize_data,
    scale_to_byte,
    apply_colormap,
    enhance_roads,
    enhance_urban_features,
)

__all__ = [
    # Spatial transformation functions
    "resample_image",
    "reproject_image",
    "crop_to_bounds",
    "align_rasters",
    # Mask generation functions
    "create_cloud_mask",
    "create_quality_mask",
    "create_aerosol_mask",
    "create_saturation_mask",
    "create_water_mask",
    "create_viirs_quality_mask",
    "combine_masks",
    "dilate_mask",
    # Enhancement functions
    "enhance_contrast",
    "normalize_data",
    "scale_to_byte",
    "apply_colormap",
    "enhance_roads",
    "enhance_urban_features",
]
