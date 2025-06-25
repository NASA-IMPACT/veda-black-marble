"""Temporal analysis and compositing functions.

This module provides the median composite function used in the pipeline.
"""

from typing import Any
import numpy.typing as npt
import numpy as np

# Type aliases
ArrayLike = npt.NDArray[Any]


def create_landsat_median_composite(
    data_list: list[ArrayLike],
    scale_factor: float = 0.0000275,
    add_offset: float = -0.2
) -> ArrayLike:
    """Create temporal median composite from Landsat scenes.
    
    This is a simplified compositing function that:
    1. Stacks all input arrays
    2. Computes per-pixel median across time
    3. Handles NaN values from QA masking
    
    The median composite approach is robust to outliers from clouds,
    shadows, and other transient artifacts that weren't caught by QA.
    
    Args:
        data_list: List of 2D arrays (one per date) with NaN for masked pixels
        scale_factor: Scale factor for reflectance conversion
        add_offset: Offset for reflectance conversion
        
    Returns:
        Median composite array with same shape as inputs
    """
    
    if not data_list:
        raise ValueError("No data provided for compositing")
    
    # Find common shape (they might be slightly different)
    shapes = [d.shape for d in data_list]
    min_height = min(s[0] for s in shapes)
    min_width = min(s[1] for s in shapes)
    
    # Crop all arrays to common size
    cropped_data = []
    for data in data_list:
        cropped = data[:min_height, :min_width]
        cropped_data.append(cropped)
    
    # Stack arrays along new time dimension
    # Shape: (n_dates, height, width)
    stacked = np.stack(cropped_data, axis=0)
    
    # Compute median along time axis, ignoring NaN
    # nanmedian handles NaN values gracefully
    composite = np.nanmedian(stacked, axis=0)
    
    # Fill any remaining all-NaN pixels with 0
    composite = np.nan_to_num(composite, nan=0.0)
    
    return composite