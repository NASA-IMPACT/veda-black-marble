"""Window rounding utilities for correct pixel placement."""

import math
from typing import Any
from rasterio.windows import Window


def round_window_offsets_correctly(window: Window, transform: Any) -> Window:
    """Round window offsets correctly for north-up vs south-up rasters.
    
    For north-up rasters (negative pixel height):
    - col_off: use floor (leftmost pixel)
    - row_off: use ceil (topmost pixel)
    
    For south-up rasters (positive pixel height):
    - col_off: use floor (leftmost pixel)
    - row_off: use floor (topmost pixel)
    
    Args:
        window: rasterio Window with float offsets
        transform: rasterio Affine transform
        
    Returns:
        Window with integer offsets
    """
    if transform.e < 0:  # North-up (most common)
        # For north-up: ceil for rows to get topmost pixel
        row_off = int(math.ceil(window.row_off))
        col_off = int(math.floor(window.col_off))
    else:  # South-up (rare)
        # For south-up: floor for both
        row_off = int(math.floor(window.row_off))
        col_off = int(math.floor(window.col_off))
    
    # Round lengths using ceiling to ensure full coverage
    height = int(math.ceil(window.height))
    width = int(math.ceil(window.width))
    
    return Window(col_off=col_off, row_off=row_off, width=width, height=height)