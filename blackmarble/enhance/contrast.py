"""Contrast enhancement and dynamic range adjustments.

This module provides functions for improving the visual quality and
analytical utility of processed imagery through contrast adjustments.
"""

from typing import Any, Literal
import numpy as np
import numpy.typing as npt
from scipy import ndimage

# Type aliases
ArrayLike = npt.NDArray[Any]
MaskArray = npt.NDArray[Any]


def _histogram_equalize(image: ArrayLike, nbins: int = 256) -> ArrayLike:
    """Pure numpy implementation of histogram equalization.
    
    Args:
        image: Input image array (should be normalized to [0, 1])
        nbins: Number of histogram bins
        
    Returns:
        Equalized image in [0, 1] range
    """
    # Get image histogram
    image_flat = image.flatten()
    hist, bins = np.histogram(image_flat, bins=nbins, range=(0, 1))
    
    # Calculate cumulative distribution function
    cdf = hist.cumsum()
    cdf = cdf / float(cdf[-1])  # Normalize
    
    # Use linear interpolation of cdf to find new pixel values
    equalized_flat = np.interp(image_flat, bins[:-1], cdf)
    
    return equalized_flat.reshape(image.shape)


def _adaptive_histogram_equalize(image: ArrayLike, clip_limit: float = 0.03, nbins: int = 256) -> ArrayLike:
    """Simplified adaptive histogram equalization.
    
    This is a basic implementation that applies local histogram equalization
    with contrast limiting. For production use, consider the full CLAHE algorithm.
    
    Args:
        image: Input image (should be normalized to [0, 1])
        clip_limit: Clipping limit for contrast limiting
        nbins: Number of histogram bins
        
    Returns:
        Adaptively equalized image
    """
    # For simplicity, we'll use a global approach with contrast limiting
    # A full implementation would divide the image into tiles
    
    # Get histogram
    hist, bins = np.histogram(image.flatten(), bins=nbins, range=(0, 1))
    
    # Apply clipping
    clip_level = clip_limit * image.size / nbins
    excess = 0
    for i in range(nbins):
        if hist[i] > clip_level:
            excess += hist[i] - clip_level
            hist[i] = clip_level
    
    # Redistribute excess uniformly
    hist = hist + excess / nbins
    
    # Calculate CDF
    cdf = hist.cumsum()
    cdf = cdf / float(cdf[-1])
    
    # Apply transformation
    result = np.interp(image.flatten(), bins[:-1], cdf)
    return result.reshape(image.shape)


def enhance_contrast(
    data: ArrayLike,
    method: Literal["percentile", "minmax", "histogram", "adaptive"] = "percentile",
    percentiles: tuple[float, float] = (2.0, 98.0),
    output_range: tuple[float, float] = (0.0, 1.0),
    mask: MaskArray | None = None,
    target_mean: float = 0.5,
    target_std: float = 0.15
) -> ArrayLike:
    """Enhance contrast using various methods.
    
    Args:
        data: Input data array
        method: Enhancement method to use
        percentiles: Percentiles for percentile stretching
        output_range: Desired output range (min, max)
        mask: Optional mask for valid data
        target_mean: Target mean for histogram method
        target_std: Target std dev for histogram method
        
    Returns:
        Contrast-enhanced array
    """
    # Pure functional implementation
    result = data.copy()
    
    # Apply mask if provided
    if mask is not None:
        valid_data = data[mask]
    else:
        valid_data = data.ravel()
    
    if method == "percentile":
        # Percentile stretching
        p_low = np.percentile(valid_data, percentiles[0])
        p_high = np.percentile(valid_data, percentiles[1])
        
        if p_high > p_low:
            result = (data - p_low) / (p_high - p_low)
        else:
            result = np.zeros_like(data)
    
    elif method == "minmax":
        # Min-max stretching
        vmin = np.min(valid_data)
        vmax = np.max(valid_data)
        
        if vmax > vmin:
            result = (data - vmin) / (vmax - vmin)
        else:
            result = np.zeros_like(data)
    
    elif method == "histogram":
        # Histogram equalization with target statistics
        # First normalize to [0, 1]
        vmin = np.min(valid_data)
        vmax = np.max(valid_data)
        
        if vmax > vmin:
            normalized = (data - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(data)
        
        # Apply histogram equalization
        if mask is not None:
            # Create a copy for processing
            temp = normalized.copy()
            temp[mask] = _histogram_equalize(normalized[mask])
            result = temp
        else:
            result = _histogram_equalize(normalized)
        
        # Adjust to target statistics
        current_mean = np.mean(result[mask] if mask is not None else result)
        current_std = np.std(result[mask] if mask is not None else result)
        
        if current_std > 0:
            result = (result - current_mean) / current_std * target_std + target_mean
    
    else:  # adaptive
        # Adaptive histogram equalization
        # Convert to appropriate range for skimage
        vmin = np.min(valid_data)
        vmax = np.max(valid_data)
        
        if vmax > vmin:
            normalized = (data - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(data)
        
        # Apply adaptive equalization
        clip_limit = 0.03
        result = _adaptive_histogram_equalize(normalized, clip_limit=clip_limit)
    
    # Scale to output range
    out_min, out_max = output_range
    result = result * (out_max - out_min) + out_min
    
    # Ensure we stay within output range
    result = np.clip(result, out_min, out_max)
    
    return result


def adaptive_local_contrast(
    data: ArrayLike,
    window_size: int = 15,
    enhancement_factor: float = 1.5,
    preserve_brightness: bool = True
) -> ArrayLike:
    """Apply local contrast enhancement using adaptive methods.
    
    Args:
        data: Input data array
        window_size: Size of local window for enhancement
        enhancement_factor: Strength of enhancement
        preserve_brightness: Whether to preserve overall brightness
        
    Returns:
        Locally enhanced array
    """
    # Calculate local mean using uniform filter
    local_mean = ndimage.uniform_filter(data, size=window_size)
    
    # Calculate deviation from local mean
    deviation = data - local_mean
    
    # Enhance the deviation
    enhanced_deviation = deviation * enhancement_factor
    
    # Reconstruct the image
    enhanced = local_mean + enhanced_deviation
    
    if preserve_brightness:
        # Preserve original brightness
        original_mean = np.mean(data)
        enhanced_mean = np.mean(enhanced)
        if enhanced_mean != 0:
            enhanced = enhanced * (original_mean / enhanced_mean)
    
    return enhanced


def multi_scale_enhancement(
    data: ArrayLike,
    scales: list[int] = [3, 7, 15],
    weights: list[float] | None = None
) -> ArrayLike:
    """Enhance contrast at multiple scales.
    
    Args:
        data: Input data array
        scales: List of window sizes for different scales
        weights: Weights for each scale (defaults to equal)
        
    Returns:
        Multi-scale enhanced array
    """
    if weights is None:
        weights = [1.0 / len(scales)] * len(scales)
    
    # Ensure weights sum to 1
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]
    
    # Apply enhancement at each scale
    enhanced_scales = []
    for scale, weight in zip(scales, weights):
        enhanced = adaptive_local_contrast(data, window_size=scale, enhancement_factor=1.2)
        enhanced_scales.append(enhanced * weight)
    
    # Combine all scales
    result = np.sum(enhanced_scales, axis=0)
    
    return result


def sigmoid_correction(
    data: ArrayLike,
    midpoint: float = 0.5,
    slope: float = 10.0
) -> ArrayLike:
    """Apply sigmoid tone curve for S-curve contrast.
    
    Args:
        data: Input data array (should be in [0, 1])
        midpoint: Center of sigmoid curve
        slope: Steepness of curve
        
    Returns:
        Sigmoid-corrected array
    """
    # Apply sigmoid function
    result = 1.0 / (1.0 + np.exp(-slope * (data - midpoint)))
    
    # Normalize to maintain range
    result_min = 1.0 / (1.0 + np.exp(-slope * (0 - midpoint)))
    result_max = 1.0 / (1.0 + np.exp(-slope * (1 - midpoint)))
    
    if result_max > result_min:
        result = (result - result_min) / (result_max - result_min)
    
    return result


def gamma_correction(
    data: ArrayLike,
    gamma: float = 1.0,
    preserve_zeros: bool = True
) -> ArrayLike:
    """Apply gamma correction for brightness adjustment.
    
    Args:
        data: Input data array (should be in [0, 1])
        gamma: Gamma value (< 1 brightens, > 1 darkens)
        preserve_zeros: Whether to keep zero values as zero
        
    Returns:
        Gamma-corrected array
    """
    # Preserve zero mask if requested
    zero_mask = None
    if preserve_zeros:
        zero_mask = data == 0
    
    # Apply gamma correction
    result = np.power(np.clip(data, 0, 1), gamma)
    
    # Restore zeros if requested
    if preserve_zeros and zero_mask is not None:
        result[zero_mask] = 0
    
    return result


def clahe_enhancement(
    data: ArrayLike,
    clip_limit: float = 0.03,
    tile_size: tuple[int, int] = (8, 8)
) -> ArrayLike:
    """Apply Contrast Limited Adaptive Histogram Equalization.
    
    Args:
        data: Input data array
        clip_limit: Clipping limit for contrast limiting
        tile_size: Size of tiles for local processing
        
    Returns:
        CLAHE enhanced array
    """
    # Normalize to [0, 1] first
    vmin, vmax = np.min(data), np.max(data)
    if vmax > vmin:
        normalized = (data - vmin) / (vmax - vmin)
    else:
        return data
    
    # Apply CLAHE
    result = _adaptive_histogram_equalize(
        normalized,
        clip_limit=clip_limit,
        nbins=256
    )
    
    # Scale back to original range
    result = result * (vmax - vmin) + vmin
    
    return result


def chain_enhancements(
    data: ArrayLike,
    enhancements: list[dict[str, Any]]
) -> ArrayLike:
    """Apply multiple enhancements in sequence.
    
    Args:
        data: Input data array
        enhancements: List of enhancement configurations
        
    Returns:
        Result after applying all enhancements
    """
    result = data.copy()
    
    for config in enhancements:
        method = config.get("method", "percentile")
        
        if method == "gamma":
            result = gamma_correction(
                result,
                gamma=config.get("gamma", 1.0),
                preserve_zeros=config.get("preserve_zeros", True)
            )
        elif method == "sigmoid":
            result = sigmoid_correction(
                result,
                midpoint=config.get("midpoint", 0.5),
                slope=config.get("slope", 10.0)
            )
        elif method == "adaptive":
            result = adaptive_local_contrast(
                result,
                window_size=config.get("window_size", 15),
                enhancement_factor=config.get("enhancement_factor", 1.5)
            )
        elif method == "clahe":
            result = clahe_enhancement(
                result,
                clip_limit=config.get("clip_limit", 0.03),
                tile_size=config.get("tile_size", (8, 8))
            )
        else:
            # Default contrast enhancement
            result = enhance_contrast(
                result,
                method=method,
                percentiles=config.get("percentiles", (2.0, 98.0)),
                output_range=config.get("output_range", (0.0, 1.0))
            )
    
    return result


def normalize_data(
    data: ArrayLike,
    output_range: tuple[float, float] = (0.0, 1.0),
    input_range: tuple[float, float] | None = None,
    clip: bool = True
) -> ArrayLike:
    """Normalize data to specified range.
    
    Args:
        data: Input data array
        output_range: Desired output range
        input_range: Input range (auto-detected if None)
        clip: Whether to clip values to output range
        
    Returns:
        Normalized array
    """
    if input_range is None:
        vmin, vmax = np.nanmin(data), np.nanmax(data)
    else:
        vmin, vmax = input_range
    
    if vmax > vmin:
        # Scale to [0, 1]
        normalized = (data - vmin) / (vmax - vmin)
        
        # Scale to output range
        out_min, out_max = output_range
        result = normalized * (out_max - out_min) + out_min
    else:
        # Constant data
        result = np.full_like(data, output_range[0])
    
    if clip:
        result = np.clip(result, output_range[0], output_range[1])
    
    return result


def highlight_features(
    data: ArrayLike,
    threshold: float = 0.7,
    highlight_factor: float = 1.5,
    suppress_factor: float = 0.5
) -> ArrayLike:
    """Highlight bright features while suppressing dim areas.
    
    Args:
        data: Input data array (should be normalized)
        threshold: Threshold for feature detection
        highlight_factor: Enhancement for bright features
        suppress_factor: Suppression for dim areas
        
    Returns:
        Feature-highlighted array
    """
    # Create masks for different regions
    bright_mask = data > threshold
    dim_mask = data < (threshold * 0.5)
    mid_mask = ~(bright_mask | dim_mask)
    
    # Apply different factors to different regions
    result = data.copy()
    result[bright_mask] = np.minimum(data[bright_mask] * highlight_factor, 1.0)
    result[dim_mask] = data[dim_mask] * suppress_factor
    
    # Smooth transitions
    if np.any(mid_mask):
        # Linear interpolation in middle region
        mid_data = data[mid_mask]
        alpha = (mid_data - threshold * 0.5) / (threshold * 0.5)
        result[mid_mask] = (
            mid_data * suppress_factor * (1 - alpha) +
            mid_data * alpha
        )
    
    return result


def remove_outliers(
    data: ArrayLike,
    percentile_low: float = 0.1,
    percentile_high: float = 99.9,
    replacement: Literal["clip", "nan", "median"] = "clip"
) -> ArrayLike:
    """Remove or replace outlier values.
    
    Args:
        data: Input data array
        percentile_low: Lower percentile threshold
        percentile_high: Upper percentile threshold
        replacement: How to handle outliers
        
    Returns:
        Array with outliers handled
    """
    # Calculate thresholds
    p_low = np.percentile(data, percentile_low)
    p_high = np.percentile(data, percentile_high)
    
    # Create outlier mask
    outlier_mask = (data < p_low) | (data > p_high)
    
    result = data.copy()
    
    if replacement == "clip":
        result = np.clip(result, p_low, p_high)
    elif replacement == "nan":
        result[outlier_mask] = np.nan
    else:  # median
        median_val = np.median(data[~outlier_mask])
        result[outlier_mask] = median_val
    
    return result