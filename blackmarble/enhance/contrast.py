"""Contrast enhancement and dynamic range adjustments.

This module provides functions for improving the visual quality and
analytical utility of processed imagery through contrast adjustments.
"""

from typing import Any, Literal
import numpy as np
import numpy.typing as npt

# Import existing implementations
from blackmarble.ported_functionality import (
    enhance_contrast as _enhance_contrast,
    chain_enhancements as _chain_enhancements,
)
from blackmarble.processing import normalize_data as _normalize_data

# Type aliases
ArrayLike = npt.NDArray[Any]
MaskArray = npt.NDArray[Any]


def enhance_contrast(
    data: ArrayLike,
    method: Literal["percentile", "minmax", "histogram", "adaptive"] = "percentile",
    percentiles: tuple[float, float] = (2.0, 98.0),
    output_range: tuple[float, float] = (0.0, 1.0),
    mask: MaskArray | None = None,
    target_mean: float = 0.5,
    target_std: float = 0.15
) -> ArrayLike:
    """Apply contrast enhancement to improve visualization.
    
    Args:
        data: Input data array
        method: Enhancement method to use
        percentiles: Percentiles for clipping (only for percentile method)
        output_range: Desired output range (min, max)
        mask: Optional mask for valid pixels
        target_mean: Target mean for adaptive method
        target_std: Target standard deviation for adaptive method
        
    Returns:
        Enhanced array with improved contrast
    """
    # Delegate to existing implementation in ported_functionality.py
    return _enhance_contrast(
        data=data,
        method=method,
        percentiles=percentiles,
        output_range=output_range,
        mask=mask,
        target_mean=target_mean,
        target_std=target_std
    )


def normalize_data(
    data: ArrayLike,
    output_min: float = 0.0,
    output_max: float = 1.0,
    input_min: float | None = None,
    input_max: float | None = None
) -> ArrayLike:
    """Normalize data to specified range.
    
    Args:
        data: Input data array
        output_min: Minimum of output range
        output_max: Maximum of output range
        input_min: Minimum of input range (auto-detected if None)
        input_max: Maximum of input range (auto-detected if None)
        
    Returns:
        Normalized data array
    """
    # Delegate to existing implementation in processing.py
    return _normalize_data(
        data=data,
        output_min=output_min,
        output_max=output_max,
        input_min=input_min,
        input_max=input_max
    )


def adaptive_histogram_equalization(
    data: ArrayLike,
    clip_limit: float = 0.03,
    tile_size: tuple[int, int] = (8, 8)
) -> ArrayLike:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Args:
        data: Input data array
        clip_limit: Clipping limit for contrast limiting
        tile_size: Size of tiles for local equalization
        
    Returns:
        Enhanced array with locally equalized contrast
    """
    # Simple implementation using percentile stretching per tile
    # For full CLAHE, would need cv2 or skimage
    h, w = data.shape[-2:]
    tile_h, tile_w = tile_size
    
    # Create output array
    output = np.zeros_like(data)
    
    # Process each tile
    for i in range(0, h, tile_h):
        for j in range(0, w, tile_w):
            # Extract tile
            tile_end_h = min(i + tile_h, h)
            tile_end_w = min(j + tile_w, w)
            
            if data.ndim == 2:
                tile = data[i:tile_end_h, j:tile_end_w]
                # Apply local enhancement
                enhanced_tile = _enhance_contrast(
                    tile, method="percentile", 
                    percentiles=(2, 98),
                    output_range=(0, 1)
                )
                output[i:tile_end_h, j:tile_end_w] = enhanced_tile
            else:
                # Handle multi-band data
                for b in range(data.shape[0]):
                    tile = data[b, i:tile_end_h, j:tile_end_w]
                    enhanced_tile = _enhance_contrast(
                        tile, method="percentile",
                        percentiles=(2, 98),
                        output_range=(0, 1)
                    )
                    output[b, i:tile_end_h, j:tile_end_w] = enhanced_tile
    
    return output


def gamma_correction(
    data: ArrayLike,
    gamma: float = 1.0,
    epsilon: float = 0.0
) -> ArrayLike:
    """Apply gamma correction for display adjustment.
    
    Args:
        data: Input data array (should be in [0, 1] range)
        gamma: Gamma value (< 1 brightens, > 1 darkens)
        epsilon: Small value to avoid issues with zero
        
    Returns:
        Gamma-corrected array
    """
    # Simple gamma correction implementation
    # Ensure data is in [0, 1] range
    data_normalized = np.clip(data, 0, 1)
    
    # Apply gamma correction with epsilon to avoid issues
    return np.power(data_normalized + epsilon, gamma)


def multi_scale_retinex(
    data: ArrayLike,
    scales: list[float] = [15, 80, 250],
    weights: list[float] | None = None
) -> ArrayLike:
    """Apply Multi-Scale Retinex enhancement.
    
    Args:
        data: Input data array
        scales: Gaussian kernel scales
        weights: Weights for each scale (equal if None)
        
    Returns:
        Enhanced array with improved local contrast
    """
    # Simple implementation using multiple contrast enhancements
    # For full MSR, would need scipy.ndimage for Gaussian filtering
    if weights is None:
        weights = [1.0 / len(scales)] * len(scales)
    
    # Ensure weights sum to 1
    weights_array = np.array(weights)
    weights_array = weights_array / weights_array.sum()
    
    # Apply multiple scales of enhancement
    result = np.zeros_like(data, dtype=np.float32)
    
    for scale, weight in zip(scales, weights_array):
        # Simulate different scales with different percentile ranges
        # Smaller scale = tighter percentiles (more local contrast)
        # Larger scale = wider percentiles (more global contrast)
        p_low = max(0.5, 5.0 - scale / 50.0)
        p_high = min(99.5, 95.0 + scale / 50.0)
        
        enhanced = _enhance_contrast(
            data,
            method="percentile",
            percentiles=(p_low, p_high),
            output_range=(0, 1)
        )
        result += weight * enhanced
    
    return np.clip(result, 0, 1)


def histogram_matching(
    source: ArrayLike,
    reference: ArrayLike,
    mask: MaskArray | None = None
) -> ArrayLike:
    """Match histogram of source to reference.
    
    Args:
        source: Source data to adjust
        reference: Reference data for histogram
        mask: Optional mask for valid pixels
        
    Returns:
        Source data with matched histogram
    """
    # Simple histogram matching implementation
    # Get valid data
    if mask is not None:
        source_valid = source[mask > 0]
        reference_valid = reference[mask > 0]
    else:
        source_valid = source[~np.isnan(source)]
        reference_valid = reference[~np.isnan(reference)]
    
    if len(source_valid) == 0 or len(reference_valid) == 0:
        return source
    
    # Compute CDFs
    source_values, source_indices = np.unique(source_valid, return_inverse=True)
    reference_values, reference_counts = np.unique(reference_valid, return_counts=True)
    
    source_cdf = np.cumsum(np.bincount(source_indices))
    source_cdf = source_cdf / source_cdf[-1]
    
    reference_cdf = np.cumsum(reference_counts)
    reference_cdf = reference_cdf / reference_cdf[-1]
    
    # Create mapping
    mapping = np.interp(source_cdf, reference_cdf, reference_values)
    
    # Apply mapping
    result = source.copy()
    if mask is not None:
        result[mask > 0] = mapping[source_indices]
    else:
        valid_mask = ~np.isnan(source)
        flat_source = source[valid_mask]
        mapped_values = np.interp(flat_source, source_values, mapping)
        result[valid_mask] = mapped_values
    
    return result


def chain_enhancements(
    data: ArrayLike,
    operations: list[tuple[Any, dict[str, Any]]]
) -> ArrayLike:
    """Chain multiple enhancement operations in sequence.
    
    Args:
        data: Input data array
        operations: List of (function, kwargs) tuples to apply
        
    Returns:
        Final enhanced data after all operations
    """
    # Delegate to existing implementation in ported_functionality.py
    return _chain_enhancements(data=data, operations=operations)


def compute_enhancement_statistics(
    original: ArrayLike,
    enhanced: ArrayLike,
    mask: MaskArray | None = None
) -> dict[str, Any]:
    """Compute statistics comparing original and enhanced data.
    
    Args:
        original: Original data array
        enhanced: Enhanced data array
        mask: Optional mask for valid pixels
        
    Returns:
        Dictionary with comparison statistics
    """
    # Compute comparison statistics
    if mask is not None:
        orig_valid = original[mask > 0]
        enh_valid = enhanced[mask > 0]
    else:
        orig_valid = original[~np.isnan(original)]
        enh_valid = enhanced[~np.isnan(enhanced)]
    
    if len(orig_valid) == 0 or len(enh_valid) == 0:
        return {
            "original_mean": np.nan,
            "original_std": np.nan,
            "enhanced_mean": np.nan,
            "enhanced_std": np.nan,
            "mean_change": np.nan,
            "std_change": np.nan,
            "dynamic_range_improvement": np.nan,
            "contrast_improvement_ratio": np.nan
        }
    
    # Basic statistics
    orig_mean = np.mean(orig_valid)
    orig_std = np.std(orig_valid)
    enh_mean = np.mean(enh_valid)
    enh_std = np.std(enh_valid)
    
    # Dynamic range
    orig_range = np.ptp(orig_valid)  # peak to peak
    enh_range = np.ptp(enh_valid)
    
    return {
        "original_mean": float(orig_mean),
        "original_std": float(orig_std),
        "enhanced_mean": float(enh_mean),
        "enhanced_std": float(enh_std),
        "mean_change": float(enh_mean - orig_mean),
        "std_change": float(enh_std - orig_std),
        "dynamic_range_improvement": float(enh_range / orig_range) if orig_range > 0 else np.nan,
        "contrast_improvement_ratio": float(enh_std / orig_std) if orig_std > 0 else np.nan
    }