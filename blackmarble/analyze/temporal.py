"""Temporal analysis and compositing functions.

This module provides the median composite function used in the pipeline.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

# Type aliases
ArrayLike = npt.NDArray[Any]


def create_landsat_median_composite(
    data_list: list[ArrayLike],
    scale_factor: float = 0.0000275,
    add_offset: float = -0.2,
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

    # Apply scaling to the raw DN values
    # This converts from raw digital numbers to surface reflectance
    scaled_stack = stacked * scale_factor + add_offset
    
    # DEBUG: Check scaling
    print(f"    Scale factor: {scale_factor}, Add offset: {add_offset}")
    print(f"    Raw DN range: [{np.nanmin(stacked):.1f}, {np.nanmax(stacked):.1f}]")
    print(f"    Scaled range: [{np.nanmin(scaled_stack):.3f}, {np.nanmax(scaled_stack):.3f}]")

    # Compute median along time axis, ignoring NaN
    # nanmedian handles NaN values gracefully
    composite = np.nanmedian(scaled_stack, axis=0)

    # Keep NaN values for pixels with no valid observations
    # This preserves the distinction between "no data" and "dark surface"
    # The downstream code should handle NaN appropriately

    return composite
