"""Advanced functionality ported from component classes.

This module contains advanced processing functions that were previously
implemented as component classes. These functions provide sophisticated
algorithms for temporal processing, scene merging, enhancement, and output
generation, all in a functional style.
"""

import numpy as np
from typing import Optional, Union, Tuple, List, Dict, Any, Literal
from datetime import datetime
import numpy.typing as npt

# Type aliases for clarity
ArrayLike = npt.NDArray[Any]  # More flexible type to handle various dtypes
MaskArray = npt.NDArray[np.bool_]
Transform = Any  # rasterio.transform.Affine
CRS = Any  # rasterio.crs.CRS


def create_temporal_composite(
    data_stack: ArrayLike,
    dates: Optional[List[datetime]] = None,
    method: Literal["median", "mean", "max", "min", "percentile", "quality_weighted"] = "median",
    percentile: Optional[int] = None,
    min_valid_observations: int = 1,
    quality_masks: Optional[ArrayLike] = None,
    weights: Optional[ArrayLike] = None,
    nodata_value: Optional[float] = None,
    use_scipy_fill_gaps: bool = False
) -> Tuple[ArrayLike, ArrayLike, Dict[str, Any]]:
    """Create temporal composite from stacked time series data.
    
    Supports advanced gap filling using scipy's distance transform for
    high-quality results when use_scipy_fill_gaps=True.
    
    Args:
        data_stack: 3D array (time, height, width) or 4D (time, bands, height, width)
        dates: List of acquisition dates for each time step
        method: Compositing method to use
        percentile: Percentile value if method is "percentile" (0-100)
        min_valid_observations: Minimum observations for valid composite pixel
        quality_masks: Optional quality masks for each time step
        weights: Optional weights for quality-weighted composite
        nodata_value: Value representing no data (default: np.nan)
        use_scipy_fill_gaps: Enable advanced gap filling using scipy's
            distance_transform_edt for nearest neighbor interpolation
            
    Returns:
        Tuple of (composite, valid_count, coverage_stats) where:
        - composite: Composited array
        - valid_count: Number of valid observations per pixel
        - coverage_stats: Dictionary with temporal coverage statistics
        
    Notes:
        When use_scipy_fill_gaps=True, gaps are filled using the nearest
        valid pixel based on Euclidean distance, producing smooth results
        without artifacts.
    """
    if data_stack.ndim < 3:
        raise ValueError(f"Expected 3D or 4D array, got shape {data_stack.shape}")
    
    # Handle nodata values
    if nodata_value is None:
        nodata_value = np.nan
    
    if np.isnan(nodata_value):
        valid_mask = ~np.isnan(data_stack)
    else:
        valid_mask = data_stack != nodata_value
    
    # Apply quality masks if provided
    if quality_masks is not None:
        if quality_masks.shape[0] != data_stack.shape[0]:
            raise ValueError("Number of quality masks must match time dimension")
        # Expand quality masks to match data dimensions
        if data_stack.ndim == 4 and quality_masks.ndim == 3:
            quality_masks = quality_masks[:, np.newaxis, :, :]
        valid_mask = valid_mask & (quality_masks > 0)
    
    # Create composite based on method
    if method == "median":
        masked_data = np.ma.masked_where(~valid_mask, data_stack)
        composite = np.ma.median(masked_data, axis=0).filled(np.nan)
    elif method == "mean":
        masked_data = np.ma.masked_where(~valid_mask, data_stack)
        composite = np.ma.mean(masked_data, axis=0).filled(np.nan)
    elif method == "max":
        masked_data = np.ma.masked_where(~valid_mask, data_stack)
        composite = np.ma.max(masked_data, axis=0).filled(np.nan)
    elif method == "min":
        masked_data = np.ma.masked_where(~valid_mask, data_stack)
        composite = np.ma.min(masked_data, axis=0).filled(np.nan)
    elif method == "percentile":
        if percentile is None:
            raise ValueError("Percentile value required for percentile method")
        composite = _percentile_composite(data_stack, valid_mask, percentile)
    elif method == "quality_weighted":
        if weights is None:
            raise ValueError("Weights required for quality-weighted composite")
        composite = _quality_weighted_composite(data_stack, valid_mask, weights)
    else:
        raise ValueError(f"Unknown composite method: {method}")
    
    # Count valid observations per pixel
    if data_stack.ndim == 3:
        valid_count = np.sum(valid_mask, axis=0)
    else:
        # For multi-band, use minimum valid count across bands
        valid_count = np.min(np.sum(valid_mask, axis=(0, 1)), axis=0)
    
    # Create output mask based on minimum observations
    output_mask = valid_count >= min_valid_observations
    
    # Apply advanced gap filling if requested
    if use_scipy_fill_gaps and np.any(~output_mask):
        composite = _fill_gaps_scipy(composite, output_mask)
    
    # Calculate coverage statistics
    coverage_stats = _calculate_temporal_coverage(valid_mask, dates)
    
    return composite, valid_count, coverage_stats


def merge_scenes(
    scenes: List[ArrayLike],
    transforms: List[Transform],
    bounds: List[Tuple[float, float, float, float]],
    blend_method: Literal["feather", "max", "min", "mean"] = "feather",
    feather_distance: int = 10,
    priority_order: Optional[List[int]] = None,
    nodata: Optional[float] = None,
    use_scipy_feather: bool = False
) -> Tuple[ArrayLike, Transform, Tuple[float, float, float, float]]:
    """Merge multiple satellite scenes into a single mosaic.
    
    Supports advanced feather blending using scipy's distance transform
    for seamless mosaics when use_scipy_feather=True.
    
    Args:
        scenes: List of scene arrays to merge
        transforms: List of affine transforms for each scene
        bounds: List of (min_x, min_y, max_x, max_y) tuples
        blend_method: Method for blending overlapping scenes
        feather_distance: Distance for feathering in pixels (only for feather method)
        priority_order: Optional priority order for scenes (lower indices = higher priority)
        nodata: Value representing no data
        use_scipy_feather: Enable scipy-based feather blending for seamless transitions
        
    Returns:
        Tuple of (merged_data, merged_transform, merged_bounds)
        
    Notes:
        When use_scipy_feather=True and blend_method="feather", uses
        scipy.ndimage.distance_transform_edt to create smooth transitions
        between overlapping scenes, eliminating visible seams.
    """
    if not scenes:
        raise ValueError("No scenes provided for merging")
    
    if len(scenes) != len(transforms) or len(scenes) != len(bounds):
        raise ValueError("Number of scenes, transforms, and bounds must match")
    
    # Calculate output bounds
    min_x = min(b[0] for b in bounds)
    min_y = min(b[1] for b in bounds)
    max_x = max(b[2] for b in bounds)
    max_y = max(b[3] for b in bounds)
    merged_bounds = (min_x, min_y, max_x, max_y)
    
    # Use first transform's resolution
    res_x = abs(transforms[0].a)
    res_y = abs(transforms[0].e)
    
    # Calculate output dimensions
    width = int(np.ceil((max_x - min_x) / res_x))
    height = int(np.ceil((max_y - min_y) / res_y))
    
    # Create output transform
    from rasterio.transform import from_bounds
    merged_transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
    
    # Initialize output array
    shape = scenes[0].shape
    if blend_method == "max":
        fill_value = -np.inf
    elif blend_method == "min":
        fill_value = np.inf
    else:
        fill_value = nodata if nodata is not None else np.nan
    
    if len(shape) == 2:
        merged = np.full((height, width), fill_value, dtype=scenes[0].dtype)
    else:
        merged = np.full((shape[0], height, width), fill_value, dtype=scenes[0].dtype)
    
    # Apply priority order if specified
    scene_order = priority_order if priority_order is not None else list(range(len(scenes)))
    
    # Merge scenes based on method
    if blend_method == "feather" and use_scipy_feather:
        merged = _feather_blend_scipy(
            merged, scenes, transforms, merged_transform, scene_order, nodata, feather_distance
        )
    elif blend_method == "feather":
        # Simple feather blend without scipy
        merged = _simple_feather_blend(
            merged, scenes, transforms, merged_transform, scene_order, nodata
        )
    elif blend_method == "max":
        merged = _max_merge(merged, scenes, transforms, merged_transform, scene_order, nodata)
    elif blend_method == "min":
        merged = _min_merge(merged, scenes, transforms, merged_transform, scene_order, nodata)
    elif blend_method == "mean":
        merged = _mean_merge(merged, scenes, transforms, merged_transform, scene_order, nodata)
    else:
        raise ValueError(f"Unknown blend method: {blend_method}")
    
    return merged, merged_transform, merged_bounds


def select_by_date_range(
    scene_dates: List[datetime],
    scene_metadata: Optional[List[Dict[str, Any]]] = None,
    quality_scores: Optional[List[float]] = None,
    target_date: Optional[datetime] = None,
    date_range: Optional[Tuple[datetime, datetime]] = None,
    max_days_from_target: Optional[int] = None,
    seasons: Optional[List[Literal["spring", "summer", "fall", "winter"]]] = None,
    cloud_threshold: Optional[float] = None,
    min_scenes: int = 1,
    max_scenes: Optional[int] = None
) -> List[int]:
    """Select scene indices based on date and quality criteria.
    
    Filters and selects satellite scenes based on temporal criteria
    including date ranges, seasonal constraints, and cloud cover.
    
    Args:
        scene_dates: List of scene acquisition dates
        scene_metadata: Optional metadata for each scene
        quality_scores: Optional quality scores for ranking
        target_date: Target acquisition date
        date_range: Explicit date range (start, end)
        max_days_from_target: Maximum days from target date
        seasons: List of seasons to include
        cloud_threshold: Maximum cloud cover percentage
        min_scenes: Minimum number of scenes required
        max_scenes: Maximum number of scenes to select
        
    Returns:
        List of selected scene indices sorted by quality/proximity
        
    Notes:
        If initial filtering results in fewer than min_scenes, constraints
        are progressively relaxed to meet the minimum requirement.
    """
    if not scene_dates:
        return []
    
    # Start with all scenes
    candidates = list(range(len(scene_dates)))
    
    # Apply date range filter
    if date_range is not None:
        candidates = [
            i for i in candidates
            if date_range[0] <= scene_dates[i] <= date_range[1]
        ]
    
    # Apply target date filter
    if target_date is not None and max_days_from_target is not None:
        candidates = [
            i for i in candidates
            if abs((scene_dates[i] - target_date).days) <= max_days_from_target
        ]
    
    # Apply season filter
    if seasons is not None:
        candidates = [
            i for i in candidates
            if _get_season(scene_dates[i]) in seasons
        ]
    
    # Apply cloud cover filter
    if cloud_threshold is not None and scene_metadata is not None:
        candidates = [
            i for i in candidates
            if i < len(scene_metadata) and 
            scene_metadata[i].get("cloud_cover", 100) <= cloud_threshold
        ]
    
    # Check minimum scenes requirement
    if len(candidates) < min_scenes:
        # Relax constraints and try again
        candidates = list(range(len(scene_dates)))
        if date_range is not None:
            candidates = [
                i for i in candidates
                if date_range[0] <= scene_dates[i] <= date_range[1]
            ]
    
    # Sort by quality if scores provided
    if quality_scores is not None:
        candidates.sort(key=lambda i: quality_scores[i], reverse=True)
    elif target_date is not None:
        # Sort by distance from target date
        candidates.sort(key=lambda i: abs((scene_dates[i] - target_date).days))
    
    # Apply maximum scenes limit
    if max_scenes is not None and len(candidates) > max_scenes:
        candidates = candidates[:max_scenes]
    
    return candidates


def enhance_contrast(
    data: ArrayLike,
    method: Literal["percentile", "minmax", "histogram", "adaptive"] = "percentile",
    percentiles: Tuple[float, float] = (2.0, 98.0),
    output_range: Tuple[float, float] = (0.0, 1.0),
    mask: Optional[MaskArray] = None,
    target_mean: float = 0.5,
    target_std: float = 0.15
) -> ArrayLike:
    """Apply contrast enhancement to improve visualization.
    
    Provides various contrast enhancement techniques including
    percentile stretching and adaptive enhancement.
    
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
        
    Notes:
        - percentile: Clips data to specified percentiles
        - minmax: Stretches full data range
        - histogram: Equalizes histogram (similar to percentile)
        - adaptive: Adjusts to achieve target statistics
    """
    # Apply mask if provided
    if mask is not None:
        valid_data = data[mask > 0]
    else:
        valid_data = data[~np.isnan(data)]
    
    if len(valid_data) == 0:
        # No valid data, return zeros
        return np.zeros_like(data)
    
    # Apply enhancement based on method
    if method == "percentile":
        vmin = np.percentile(valid_data, percentiles[0])
        vmax = np.percentile(valid_data, percentiles[1])
        enhanced = _normalize_range(data, output_range[0], output_range[1], float(vmin), float(vmax))
    elif method == "minmax":
        vmin = np.min(valid_data)
        vmax = np.max(valid_data)
        enhanced = _normalize_range(data, output_range[0], output_range[1], float(vmin), float(vmax))
    elif method == "histogram":
        # Similar to percentile but with fixed 1% and 99%
        vmin = np.percentile(valid_data, 1.0)
        vmax = np.percentile(valid_data, 99.0)
        enhanced = _normalize_range(data, output_range[0], output_range[1], float(vmin), float(vmax))
    elif method == "adaptive":
        enhanced = _adaptive_contrast(data, valid_data, output_range, target_mean, target_std)
    else:
        raise ValueError(f"Unknown enhancement method: {method}")
    
    return enhanced


def apply_colormap(
    data: ArrayLike,
    colormap: Literal["grayscale", "viridis", "plasma", "inferno", "turbo"] = "grayscale",
    colormap_data: Optional[Dict[str, ArrayLike]] = None,
    scale_range: Optional[Tuple[float, float]] = None,
    alpha: Optional[Union[float, ArrayLike]] = None
) -> ArrayLike:
    """Apply colormap to data for visualization.
    
    Converts single-channel data to RGB(A) using specified colormap.
    
    Args:
        data: Input data array (2D)
        colormap: Name of standard colormap or "custom"
        colormap_data: Custom colormap as dict with 'r', 'g', 'b' arrays (256 values each)
        scale_range: Range to scale data before colormap (min, max)
        alpha: Optional alpha channel (scalar or array)
        
    Returns:
        RGB(A) array with shape (3 or 4, height, width) as uint8
        
    Notes:
        Standard colormaps are approximated. For exact matplotlib colormaps,
        provide colormap_data with the RGB values.
    """
    # Scale data to 0-255
    if scale_range is not None:
        scaled = _normalize_range(data, 0, 255, scale_range[0], scale_range[1])
    else:
        # Auto-scale based on data range
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            vmin, vmax = np.min(valid_data), np.max(valid_data)
            scaled = _normalize_range(data, 0, 255, float(vmin), float(vmax))
        else:
            scaled = np.zeros_like(data)
    
    # Handle NaN values
    valid_mask = ~np.isnan(scaled)
    indices = np.zeros_like(scaled, dtype=np.uint8)
    if np.any(valid_mask):
        indices[valid_mask] = np.clip(np.round(scaled[valid_mask]), 0, 255).astype(np.uint8)
    
    # Get colormap arrays
    if colormap_data is not None:
        cmap_r = colormap_data["r"]
        cmap_g = colormap_data["g"]
        cmap_b = colormap_data["b"]
    else:
        cmap_r, cmap_g, cmap_b = _get_standard_colormap(colormap)
    
    # Apply colormap
    r = cmap_r[indices]
    g = cmap_g[indices]
    b = cmap_b[indices]
    
    # Handle alpha channel
    if alpha is not None:
        if isinstance(alpha, (int, float)):
            # Scalar alpha
            alpha_channel = np.full_like(data, int(alpha * 255), dtype=np.uint8)
        else:
            # Array alpha
            alpha_channel = (alpha * 255).astype(np.uint8)
        
        # Set transparent where data is invalid
        alpha_channel[~valid_mask] = 0
        
        # Stack RGBA
        output = np.stack([r, g, b, alpha_channel], axis=0)
    else:
        # Stack RGB
        output = np.stack([r, g, b], axis=0)
    
    return output


def validate_output(
    data: ArrayLike,
    transform: Transform,
    min_valid_percentage: float = 0.1,
    expected_range: Optional[Tuple[float, float]] = None,
    expected_shape: Optional[Tuple[int, ...]] = None,
    check_crs: bool = True
) -> Dict[str, Any]:
    """Validate output data quality and properties.
    
    Performs comprehensive validation checks on output data to ensure
    it meets quality standards.
    
    Args:
        data: Output data array
        transform: Affine transform for georeferencing
        min_valid_percentage: Minimum percentage of valid (non-NaN) pixels
        expected_range: Expected data value range (min, max)
        expected_shape: Expected array shape
        check_crs: Whether to validate CRS properties
        
    Returns:
        Dictionary with validation results:
        - valid: Boolean indicating if all checks passed
        - issues: List of validation issues found
        - stats: Dictionary of computed statistics
    """
    issues = []
    
    # Check valid pixel percentage
    valid_mask = ~np.isnan(data) & (data != 0)
    valid_percentage = np.mean(valid_mask)
    
    if valid_percentage < min_valid_percentage:
        issues.append(
            f"Only {valid_percentage:.1%} valid pixels "
            f"(minimum required: {min_valid_percentage:.1%})"
        )
    
    # Check data range
    if expected_range is not None and np.any(valid_mask):
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        if data_min < expected_range[0] or data_max > expected_range[1]:
            issues.append(
                f"Data range [{data_min:.2f}, {data_max:.2f}] outside "
                f"expected range {expected_range}"
            )
    
    # Check shape
    if expected_shape is not None and data.shape != expected_shape:
        issues.append(
            f"Data shape {data.shape} does not match expected shape {expected_shape}"
        )
    
    # Check transform properties
    if check_crs and hasattr(transform, 'a'):
        # Check for reasonable pixel sizes
        pixel_width = abs(transform.a)
        pixel_height = abs(transform.e)
        
        if pixel_width <= 0 or pixel_height <= 0:
            issues.append("Invalid pixel dimensions in transform")
        elif pixel_width > 1.0 or pixel_height > 1.0:
            # Assuming degrees, check for unreasonable sizes
            issues.append(
                f"Pixel size ({pixel_width:.4f}, {pixel_height:.4f}) "
                "seems too large for geographic data"
            )
    
    # Compute statistics
    stats = {
        "valid_percentage": float(valid_percentage),
        "total_pixels": int(data.size),
        "valid_pixels": int(np.sum(valid_mask)),
        "data_range": (float(np.nanmin(data)), float(np.nanmax(data))) 
                      if np.any(valid_mask) else (np.nan, np.nan),
        "mean": float(np.nanmean(data)),
        "std": float(np.nanstd(data)),
        "shape": data.shape
    }
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "stats": stats
    }


def generate_metadata(
    data: ArrayLike,
    transform: Transform,
    crs: CRS,
    bounds: Tuple[float, float, float, float],
    processing_info: Optional[Dict[str, Any]] = None,
    band_names: Optional[List[str]] = None,
    acquisition_date: Optional[datetime] = None,
    platform: Optional[str] = None,
    instrument: Optional[str] = None,
    include_band_statistics: bool = True
) -> Dict[str, Any]:
    """Generate comprehensive metadata for output products.
    
    Creates metadata including spatial reference, statistics, and
    processing information.
    
    Args:
        data: Output data array
        transform: Affine transform
        crs: Coordinate reference system
        bounds: Bounding box (minx, miny, maxx, maxy)
        processing_info: Dictionary of processing steps and parameters
        band_names: Names for each band (if multi-band)
        acquisition_date: Data acquisition date
        platform: Satellite platform name
        instrument: Instrument name
        include_band_statistics: Whether to compute per-band statistics
        
    Returns:
        Metadata dictionary with spatial, temporal, and statistical information
    """
    # Determine number of bands
    if data.ndim == 2:
        n_bands = 1
        shape = data.shape
    else:
        n_bands = data.shape[0]
        shape = data.shape[1:]
    
    # Base metadata
    metadata = {
        "type": "raster",
        "crs": str(crs) if crs is not None else None,
        "transform": list(transform) if hasattr(transform, '__iter__') else str(transform),
        "bounds": {
            "west": bounds[0],
            "south": bounds[1],
            "east": bounds[2],
            "north": bounds[3]
        },
        "shape": {
            "height": shape[0],
            "width": shape[1],
            "bands": n_bands
        },
        "dtype": str(data.dtype)
    }
    
    # Add temporal information
    if acquisition_date is not None:
        metadata["properties"] = {
            "datetime": acquisition_date.isoformat(),
            "year": acquisition_date.year,
            "month": acquisition_date.month,
            "day": acquisition_date.day
        }
    else:
        metadata["properties"] = {}
    
    # Add platform/instrument
    if platform is not None:
        metadata["properties"]["platform"] = platform
    if instrument is not None:
        metadata["properties"]["instrument"] = instrument
    
    # Add processing information
    if processing_info is not None:
        metadata["processing"] = processing_info
    
    # Compute statistics
    if include_band_statistics:
        if n_bands == 1:
            # Single band statistics
            valid_mask = ~np.isnan(data)
            if np.any(valid_mask):
                metadata["statistics"] = {
                    "min": float(np.nanmin(data)),
                    "max": float(np.nanmax(data)),
                    "mean": float(np.nanmean(data)),
                    "std": float(np.nanstd(data)),
                    "valid_percentage": float(np.mean(valid_mask) * 100)
                }
        else:
            # Multi-band statistics
            metadata["band_statistics"] = []
            for i in range(n_bands):
                band_data = data[i]
                valid_mask = ~np.isnan(band_data)
                band_name = band_names[i] if band_names and i < len(band_names) else f"band_{i+1}"
                
                if np.any(valid_mask):
                    band_stats = {
                        "name": band_name,
                        "min": float(np.nanmin(band_data)),
                        "max": float(np.nanmax(band_data)),
                        "mean": float(np.nanmean(band_data)),
                        "std": float(np.nanstd(band_data)),
                        "valid_percentage": float(np.mean(valid_mask) * 100)
                    }
                else:
                    band_stats = {
                        "name": band_name,
                        "min": np.nan,
                        "max": np.nan,
                        "mean": np.nan,
                        "std": np.nan,
                        "valid_percentage": 0.0
                    }
                metadata["band_statistics"].append(band_stats)
    
    return metadata


def chain_enhancements(
    data: ArrayLike,
    operations: List[Tuple[Any, Dict[str, Any]]]  # Any to avoid callable type issues
) -> ArrayLike:
    """Chain multiple enhancement operations in sequence.
    
    Applies a series of enhancement functions to data, where each
    operation receives the output of the previous operation.
    
    Args:
        data: Input data array
        operations: List of (function, kwargs) tuples to apply in order
        
    Returns:
        Final enhanced data after all operations
        
    Example:
        >>> enhanced = chain_enhancements(data, [
        ...     (enhance_contrast, {"method": "percentile"}),
        ...     (apply_colormap, {"colormap": "viridis"}),
        ... ])
    """
    result = data.copy()
    
    for func, kwargs in operations:
        result = func(result, **kwargs)
    
    return result


# Helper functions (private)

def _percentile_composite(
    data: ArrayLike,
    valid_mask: ArrayLike,
    percentile: int
) -> ArrayLike:
    """Create percentile composite from masked data."""
    output_shape = data.shape[1:]
    composite = np.full(output_shape, np.nan)
    
    # Process each pixel
    if data.ndim == 3:
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                valid_data = data[valid_mask[:, i, j], i, j]
                if len(valid_data) > 0:
                    composite[i, j] = np.percentile(valid_data, percentile)
    else:
        # Multi-band case
        for b in range(output_shape[0]):
            for i in range(output_shape[1]):
                for j in range(output_shape[2]):
                    valid_data = data[valid_mask[:, b, i, j], b, i, j]
                    if len(valid_data) > 0:
                        composite[b, i, j] = np.percentile(valid_data, percentile)
    
    return composite


def _quality_weighted_composite(
    data: ArrayLike,
    valid_mask: ArrayLike,
    weights: ArrayLike
) -> ArrayLike:
    """Create quality-weighted composite."""
    # Ensure weights match data shape
    if weights.shape[0] != data.shape[0]:
        raise ValueError("Weights must match time dimension")
    
    # Expand weights to match data dimensions if needed
    if data.ndim == 4 and weights.ndim == 3:
        weights = weights[:, np.newaxis, :, :]
    elif data.ndim == 3 and weights.ndim == 1:
        weights = weights[:, np.newaxis, np.newaxis]
    
    # Apply valid mask to weights
    masked_weights = weights * valid_mask
    
    # Normalize weights
    weight_sum = np.sum(masked_weights, axis=0)
    weight_sum[weight_sum == 0] = 1  # Avoid division by zero
    
    # Compute weighted average
    weighted_data = data * masked_weights
    composite = np.sum(weighted_data, axis=0) / weight_sum
    
    return composite


def _fill_gaps_scipy(
    composite: ArrayLike,
    valid_mask: ArrayLike
) -> ArrayLike:
    """Fill gaps using scipy's distance transform."""
    from scipy.ndimage import distance_transform_edt
    
    filled = composite.copy()
    
    if composite.ndim == 2:
        # Find invalid pixels
        invalid_mask = valid_mask == 0
        
        if np.any(invalid_mask) and np.any(valid_mask):
            # Get nearest valid pixel indices
            indices = distance_transform_edt(
                invalid_mask,
                return_distances=False,
                return_indices=True
            )
            
            # Fill invalid pixels with nearest valid values
            invalid_coords = np.where(invalid_mask)
            if indices is not None:
                nearest_rows = indices[0][invalid_coords]
                nearest_cols = indices[1][invalid_coords]
                filled[invalid_coords] = composite[nearest_rows, nearest_cols]
    else:
        # Multi-band case
        for b in range(composite.shape[0]):
            band_valid = valid_mask if valid_mask.ndim == 2 else valid_mask[b]
            invalid_mask = band_valid == 0
            
            if np.any(invalid_mask) and np.any(band_valid):
                indices = distance_transform_edt(
                    invalid_mask,
                    return_distances=False,
                    return_indices=True
                )
                
                invalid_coords = np.where(invalid_mask)
                if indices is not None:
                    nearest_rows = indices[0][invalid_coords]
                    nearest_cols = indices[1][invalid_coords]
                    filled[b][invalid_coords] = composite[b][nearest_rows, nearest_cols]
    
    return filled


def _calculate_temporal_coverage(
    valid_mask: ArrayLike,
    dates: Optional[List[datetime]]
) -> Dict[str, Any]:
    """Calculate temporal coverage statistics."""
    coverage = {
        "temporal_range_days": 0,
        "observation_frequency": 0.0,
        "coverage_percentage": 0.0
    }
    
    if dates and len(dates) == valid_mask.shape[0]:
        # Calculate temporal range
        date_range = max(dates) - min(dates)
        coverage["temporal_range_days"] = date_range.days
        
        # Calculate observation frequency
        if coverage["temporal_range_days"] > 0:
            coverage["observation_frequency"] = len(dates) / coverage["temporal_range_days"]
        
        # Calculate overall coverage percentage
        total_possible = valid_mask.size
        total_valid = np.sum(valid_mask)
        coverage["coverage_percentage"] = 100.0 * total_valid / total_possible
    
    return coverage


def _feather_blend_scipy(
    output: ArrayLike,
    scenes: List[ArrayLike],
    transforms: List[Transform],
    out_transform: Transform,
    order: List[int],
    nodata: Optional[float],
    feather_distance: int
) -> ArrayLike:
    """Feather blend using scipy distance transform."""
    from scipy.ndimage import distance_transform_edt
    
    # Process scenes in priority order
    for idx in order:
        scene = scenes[idx]
        transform = transforms[idx]
        
        # Calculate offset in output array
        col_off = int((transform.c - out_transform.c) / out_transform.a)
        row_off = int((transform.f - out_transform.f) / out_transform.e)
        
        # Get valid data mask
        if nodata is None or np.isnan(nodata):
            valid_mask = ~np.isnan(scene)
        else:
            valid_mask = scene != nodata
        
        if scene.ndim > 2:
            valid_mask = np.all(valid_mask, axis=0)
        
        # Calculate distance from edges
        distance_array = distance_transform_edt(valid_mask)
        if distance_array is not None:
            distance = np.minimum(distance_array, feather_distance) / feather_distance
        else:
            # Fallback if distance transform fails
            distance = np.ones_like(valid_mask, dtype=float)
        
        # Apply feathered blend
        if scene.ndim == 2:
            _blend_2d_scene(output, scene, valid_mask, distance, row_off, col_off)
        else:
            _blend_3d_scene(output, scene, valid_mask, distance, row_off, col_off)
    
    return output


def _simple_feather_blend(
    output: ArrayLike,
    scenes: List[ArrayLike],
    transforms: List[Transform],
    out_transform: Transform,
    order: List[int],
    nodata: Optional[float]
) -> ArrayLike:
    """Simple feather blend without scipy."""
    # For simple blending, just use mean in overlap areas
    return _mean_merge(output, scenes, transforms, out_transform, order, nodata)


def _blend_2d_scene(
    output: ArrayLike,
    scene: ArrayLike,
    valid_mask: ArrayLike,
    distance: ArrayLike,
    row_off: int,
    col_off: int
) -> None:
    """Blend a 2D scene into output."""
    out_slice = output[
        row_off:row_off + scene.shape[0],
        col_off:col_off + scene.shape[1]
    ]
    
    # Check for overlapping data
    existing_valid = ~np.isnan(out_slice)
    blend_mask = valid_mask & existing_valid
    
    # Blend overlapping regions
    if np.any(blend_mask):
        out_slice[blend_mask] = (
            distance[blend_mask] * scene[blend_mask] +
            (1 - distance[blend_mask]) * out_slice[blend_mask]
        )
    
    # Copy non-overlapping regions
    new_data_mask = valid_mask & ~existing_valid
    out_slice[new_data_mask] = scene[new_data_mask]


def _blend_3d_scene(
    output: ArrayLike,
    scene: ArrayLike,
    valid_mask: ArrayLike,
    distance: ArrayLike,
    row_off: int,
    col_off: int
) -> None:
    """Blend a 3D (multi-band) scene into output."""
    for b in range(scene.shape[0]):
        out_slice = output[
            b,
            row_off:row_off + scene.shape[1],
            col_off:col_off + scene.shape[2]
        ]
        
        # Check for overlapping data
        existing_valid = ~np.isnan(out_slice)
        blend_mask = valid_mask & existing_valid
        
        # Blend overlapping regions
        if np.any(blend_mask):
            out_slice[blend_mask] = (
                distance[blend_mask] * scene[b][blend_mask] +
                (1 - distance[blend_mask]) * out_slice[blend_mask]
            )
        
        # Copy non-overlapping regions
        new_data_mask = valid_mask & ~existing_valid
        out_slice[new_data_mask] = scene[b][new_data_mask]


def _max_merge(
    output: ArrayLike,
    scenes: List[ArrayLike],
    transforms: List[Transform],
    out_transform: Transform,
    order: List[int],
    nodata: Optional[float]
) -> ArrayLike:
    """Merge using maximum values."""
    for idx in order:
        scene = scenes[idx]
        transform = transforms[idx]
        
        # Calculate offset
        col_off = int((transform.c - out_transform.c) / out_transform.a)
        row_off = int((transform.f - out_transform.f) / out_transform.e)
        
        # Get valid data mask
        if nodata is None or np.isnan(nodata):
            valid_mask = ~np.isnan(scene)
        else:
            valid_mask = scene != nodata
        
        # Apply maximum
        if scene.ndim == 2:
            out_slice = output[
                row_off:row_off + scene.shape[0],
                col_off:col_off + scene.shape[1]
            ]
            np.maximum(out_slice, scene, out=out_slice, where=valid_mask)
        else:
            for b in range(scene.shape[0]):
                out_slice = output[
                    b,
                    row_off:row_off + scene.shape[1],
                    col_off:col_off + scene.shape[2]
                ]
                band_valid = valid_mask if valid_mask.ndim == 2 else valid_mask[b]
                np.maximum(out_slice, scene[b], out=out_slice, where=band_valid)
    
    # Replace -inf with nodata
    output[output == -np.inf] = nodata if nodata is not None else np.nan
    return output


def _min_merge(
    output: ArrayLike,
    scenes: List[ArrayLike],
    transforms: List[Transform],
    out_transform: Transform,
    order: List[int],
    nodata: Optional[float]
) -> ArrayLike:
    """Merge using minimum values."""
    for idx in order:
        scene = scenes[idx]
        transform = transforms[idx]
        
        # Calculate offset
        col_off = int((transform.c - out_transform.c) / out_transform.a)
        row_off = int((transform.f - out_transform.f) / out_transform.e)
        
        # Get valid data mask
        if nodata is None or np.isnan(nodata):
            valid_mask = ~np.isnan(scene)
        else:
            valid_mask = scene != nodata
        
        # Apply minimum
        if scene.ndim == 2:
            out_slice = output[
                row_off:row_off + scene.shape[0],
                col_off:col_off + scene.shape[1]
            ]
            np.minimum(out_slice, scene, out=out_slice, where=valid_mask)
        else:
            for b in range(scene.shape[0]):
                out_slice = output[
                    b,
                    row_off:row_off + scene.shape[1],
                    col_off:col_off + scene.shape[2]
                ]
                band_valid = valid_mask if valid_mask.ndim == 2 else valid_mask[b]
                np.minimum(out_slice, scene[b], out=out_slice, where=band_valid)
    
    # Replace inf with nodata
    output[output == np.inf] = nodata if nodata is not None else np.nan
    return output


def _mean_merge(
    output: ArrayLike,
    scenes: List[ArrayLike],
    transforms: List[Transform],
    out_transform: Transform,
    order: List[int],
    nodata: Optional[float]
) -> ArrayLike:
    """Merge using mean values."""
    # Initialize sum and count arrays
    if output.ndim == 2:
        sum_array = np.zeros_like(output)
        count_array = np.zeros_like(output, dtype=int)
    else:
        sum_array = np.zeros_like(output)
        count_array = np.zeros(output.shape[1:], dtype=int)
    
    # Accumulate values
    for idx in order:
        scene = scenes[idx]
        transform = transforms[idx]
        
        # Calculate offset
        col_off = int((transform.c - out_transform.c) / out_transform.a)
        row_off = int((transform.f - out_transform.f) / out_transform.e)
        
        # Get valid data mask
        if nodata is None or np.isnan(nodata):
            valid_mask = ~np.isnan(scene)
        else:
            valid_mask = scene != nodata
        
        if scene.ndim == 2:
            sum_slice = sum_array[
                row_off:row_off + scene.shape[0],
                col_off:col_off + scene.shape[1]
            ]
            count_slice = count_array[
                row_off:row_off + scene.shape[0],
                col_off:col_off + scene.shape[1]
            ]
            
            sum_slice[valid_mask] += scene[valid_mask]
            count_slice[valid_mask] += 1
        else:
            for b in range(scene.shape[0]):
                sum_slice = sum_array[
                    b,
                    row_off:row_off + scene.shape[1],
                    col_off:col_off + scene.shape[2]
                ]
                band_valid = valid_mask if valid_mask.ndim == 2 else valid_mask[b]
                sum_slice[band_valid] += scene[b][band_valid]
            
            count_slice = count_array[
                row_off:row_off + scene.shape[1],
                col_off:col_off + scene.shape[2]
            ]
            if valid_mask.ndim == 2:
                count_slice[valid_mask] += 1
            else:
                count_slice[np.any(valid_mask, axis=0)] += 1
    
    # Calculate mean
    if output.ndim == 2:
        result = np.where(count_array > 0, sum_array / count_array, nodata if nodata is not None else np.nan)
    else:
        result = np.zeros_like(output)
        for b in range(output.shape[0]):
            result[b] = np.where(count_array > 0, sum_array[b] / count_array, nodata if nodata is not None else np.nan)
    
    return result


def _get_season(date: datetime) -> str:
    """Get season for a given date (Northern Hemisphere)."""
    month = date.month
    if month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    elif month in [9, 10, 11]:
        return "fall"
    else:
        return "winter"


def _normalize_range(
    data: ArrayLike,
    out_min: float,
    out_max: float,
    in_min: float,
    in_max: float
) -> ArrayLike:
    """Normalize data from input range to output range."""
    if in_max == in_min:
        return np.full_like(data, out_min)
    
    # Clip to input range
    clipped = np.clip(data, in_min, in_max)
    
    # Scale to output range
    scaled = (clipped - in_min) / (in_max - in_min)
    result = scaled * (out_max - out_min) + out_min
    
    return result


def _adaptive_contrast(
    data: ArrayLike,
    valid_data: ArrayLike,
    output_range: Tuple[float, float],
    target_mean: float,
    target_std: float
) -> ArrayLike:
    """Apply adaptive contrast enhancement."""
    # Calculate current statistics
    current_mean = np.mean(valid_data)
    current_std = np.std(valid_data)
    
    if current_std == 0:
        # No variation, return constant value
        return np.full_like(data, target_mean)
    
    # Calculate scaling to achieve target statistics
    scale = target_std / current_std
    shift = target_mean - (current_mean * scale)
    
    # Apply transformation
    enhanced = data * scale + shift
    
    # Clip to output range
    enhanced = np.clip(enhanced, output_range[0], output_range[1])
    
    return enhanced


def _get_standard_colormap(name: str) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Get standard colormap arrays."""
    if name == "grayscale":
        values = np.arange(256, dtype=np.uint8)
        return values, values, values
    elif name == "viridis":
        # Approximate viridis colormap
        r = np.linspace(68, 253, 256).astype(np.uint8)
        g = np.linspace(1, 231, 256).astype(np.uint8)
        b = np.linspace(84, 37, 256).astype(np.uint8)
        return r, g, b
    elif name == "plasma":
        # Approximate plasma colormap
        r = np.linspace(13, 240, 256).astype(np.uint8)
        g = np.linspace(8, 249, 256).astype(np.uint8)
        b = np.linspace(135, 33, 256).astype(np.uint8)
        return r, g, b
    elif name == "inferno":
        # Approximate inferno colormap
        r = np.linspace(0, 252, 256).astype(np.uint8)
        g = np.linspace(0, 255, 256).astype(np.uint8)
        b = np.linspace(4, 164, 256).astype(np.uint8)
        # Make first value black
        r[0] = g[0] = b[0] = 0
        return r, g, b
    elif name == "turbo":
        # Approximate turbo colormap
        # This is a rough approximation of the turbo colormap
        t = np.linspace(0, 1, 256)
        r = np.clip(255 * (1.5 - 4 * np.abs(t - 0.75)), 0, 255).astype(np.uint8)
        g = np.clip(255 * (1.5 - 4 * np.abs(t - 0.5)), 0, 255).astype(np.uint8)
        b = np.clip(255 * (1.5 - 4 * np.abs(t - 0.25)), 0, 255).astype(np.uint8)
        return r, g, b
    else:
        # Default to grayscale
        values = np.arange(256, dtype=np.uint8)
        return values, values, values