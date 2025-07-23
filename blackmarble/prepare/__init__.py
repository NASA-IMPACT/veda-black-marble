"""Data preparation module for Black Marble pipeline.

This module handles data preprocessing tasks including:
- Quality assessment and masking
- Radiometric corrections and scaling
- Spatial reprojection and alignment
- Coordinate system transformations
- File I/O operations
"""

# Coordinates functions removed - entire coordinates.py was unused
# IO functions removed - use rasterio directly
# Quality functions removed - entire quality.py was unused
# Scaling functions removed - entire scaling.py was unused

from .diagnostic import save_diagnostic
from .landsat import process_landsat_date
from .landsat_qa import create_landsat_cloud_mask, print_qa_pixel_summary
from .spatial import (
    align_rasters,
    crop_to_bounds,
    reproject_image,
)

__all__ = [
    # Spatial functions (only the used ones)
    "align_rasters",
    "crop_to_bounds",
    "reproject_image",
    # Diagnostic functions
    "save_diagnostic",
    # Landsat functions
    "process_landsat_date",
    # Landsat QA functions (only the used ones)
    "create_landsat_cloud_mask",
    "print_qa_pixel_summary",
]
