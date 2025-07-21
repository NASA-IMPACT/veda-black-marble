"""Temporal analysis and compositing functions.

This module provides the median composite function used in the pipeline.
"""

import warnings
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

# Type aliases
ArrayLike = npt.NDArray[Any]


def create_landsat_median_composite(
    data_list: list[ArrayLike],
    scale_factor: float = 0.0000275,
    add_offset: float = -0.2,
    method: str = "mean",
    log_coverage: bool = False,
) -> ArrayLike:
    """Create temporal composite from Landsat scenes.

    This compositing function:
    1. Stacks all input arrays
    2. Computes per-pixel composite across time
    3. Handles NaN values from QA masking

    Args:
        data_list: List of 2D arrays (one per date) with NaN for masked pixels
        scale_factor: Scale factor for reflectance conversion
        add_offset: Offset for reflectance conversion
        method: Compositing method - "median" (hard edges) or "mean" (smooth blending)

    Returns:
        Composite array with same shape as inputs

    Note:
        The "median" method picks one value at overlaps creating hard edges.
        The "mean" method blends values smoothly avoiding scene boundaries.
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
    
    # Log per-scene valid pixel coverage if requested
    if log_coverage:
        print("\n📊 Per-scene valid pixel coverage:")
        for i, data in enumerate(cropped_data):
            valid_pixels = np.sum(~np.isnan(data))
            total_pixels = data.size
            valid_pct = (valid_pixels / total_pixels * 100) if total_pixels > 0 else 0
            print(f"   Scene {i+1}: {valid_pct:.1f}% valid ({valid_pixels}/{total_pixels} pixels)")
        
        # Check if any scene has very low coverage
        all_valid_pcts = []
        for data in cropped_data:
            valid_pct = np.sum(~np.isnan(data)) / data.size * 100 if data.size > 0 else 0
            all_valid_pcts.append(valid_pct)
        
        if any(pct < 5.0 for pct in all_valid_pcts):
            print("   ⚠️  WARNING: Some scenes have <5% valid pixels - likely over-masking!")
        
        # Show overall stack coverage
        valid_in_stack = np.sum(~np.isnan(stacked))
        total_in_stack = stacked.size
        stack_valid_pct = (valid_in_stack / total_in_stack * 100) if total_in_stack > 0 else 0
        print(f"   Overall stack: {stack_valid_pct:.1f}% valid pixels")

    # Apply scaling to the raw DN values
    # This converts from raw digital numbers to surface reflectance
    scaled_stack = stacked * scale_factor + add_offset

    # Compute composite along time axis, ignoring NaN
    # Suppress warning about all-NaN slices - this is expected for pixels
    # with persistent clouds or outside scene boundaries
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN slice encountered")

        if method == "median":
            # Median is robust to outliers but creates hard edges at scene boundaries
            composite = np.nanmedian(scaled_stack, axis=0)
        elif method == "mean":
            # Mean blends overlapping scenes smoothly, avoiding visible boundaries
            composite = np.nanmean(scaled_stack, axis=0)
        else:
            raise ValueError(f"Unknown compositing method: {method}. Use 'median' or 'mean'.")

    # Keep NaN values for pixels with no valid observations
    # This preserves the distinction between "no data" and "dark surface"
    # The downstream code should handle NaN appropriately

    return composite


def create_index_temporal_composite(
    index_stack: list[ArrayLike],
    method: Literal["median", "mean", "percentile"] = "median",
    percentile: float = 50.0,
    mask_nan: bool = True,
    min_valid_observations: int = 1,
) -> ArrayLike:
    """Create temporal composite from pre-calculated spectral indices.
    
    Unlike create_landsat_median_composite, this expects indices
    (NDVI, NDWI) already calculated per date. This ensures mathematical
    correctness since median(ratio) ≠ ratio(median).
    
    Args:
        index_stack: List of 2D arrays containing pre-calculated indices
        method: Compositing method - "median", "mean", or "percentile"
        percentile: Percentile value if method="percentile" (0-100)
        mask_nan: Whether to mask output where insufficient observations
        min_valid_observations: Minimum valid observations required
        
    Returns:
        Composite array with same shape as inputs
        
    Note:
        For NDUI calculations, using 85th percentile NDVI better captures
        vegetation's light-dimming potential than median values.
    """
    if not index_stack:
        raise ValueError("No index data provided for compositing")
        
    # Find common shape
    shapes = [arr.shape for arr in index_stack]
    min_height = min(s[0] for s in shapes)
    min_width = min(s[1] for s in shapes)
    
    # Crop all arrays to common size
    cropped_stack = []
    for arr in index_stack:
        cropped = arr[:min_height, :min_width].astype(np.float32)  # Ensure float32
        cropped_stack.append(cropped)
    
    # Stack arrays along time dimension
    # Shape: (n_dates, height, width)
    stacked = np.stack(cropped_stack, axis=0)
    
    # Count valid observations per pixel
    valid_count = np.sum(~np.isnan(stacked), axis=0)
    
    # Compute composite along time axis
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN slice encountered")
        
        if method == "median":
            composite = np.nanmedian(stacked, axis=0)
        elif method == "mean":
            composite = np.nanmean(stacked, axis=0)
        elif method == "percentile":
            composite = np.nanpercentile(stacked, percentile, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'median', 'mean', or 'percentile'.")
    
    # Mask pixels with insufficient observations
    if mask_nan and min_valid_observations > 1:
        composite[valid_count < min_valid_observations] = np.nan
    
    return composite
