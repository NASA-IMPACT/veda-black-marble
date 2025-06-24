"""Temporal analysis and compositing functions.

This module provides functions for creating temporal composites,
merging scenes, and performing time series analysis.
"""

from typing import Any, Literal
from datetime import datetime
import numpy.typing as npt

# Type aliases
ArrayLike = npt.NDArray[Any]
Transform = Any  # rasterio.transform.Affine
BBox = tuple[float, float, float, float]


def create_landsat_median_composite(
    data_list: list[ArrayLike],
    scale_factor: float = 0.0000275,
    add_offset: float = -0.2
) -> ArrayLike:
    """Create median composite from multiple Landsat scenes.
    
    This function performs temporal compositing on pre-masked data:
    1. Stacks all input arrays (which already have NaN for masked pixels)
    2. Scales to reflectance
    3. Takes nanmedian to ignore masked/missing data
    4. Replaces remaining NaN with 0
    
    This is essential for filling gaps from clouds and missing data
    in Landsat time series.
    
    Args:
        data_list: List of 2D arrays (one per date) with NaN for masked pixels
        scale_factor: Scale factor for reflectance conversion
        add_offset: Offset for reflectance conversion
        
    Returns:
        Median composite array with same shape as inputs
    """
    import numpy as np
    
    if not data_list:
        raise ValueError("No data provided for compositing")
    
    # Find common shape (they might be slightly different)
    shapes = [d.shape for d in data_list]
    min_height = min(s[0] for s in shapes)
    min_width = min(s[1] for s in shapes)
    
    # Crop all to common size and scale to reflectance
    scaled_data = []
    for data in data_list:
        # Crop to common size
        cropped = data[:min_height, :min_width]
        # Scale to reflectance (NaN values remain NaN)
        scaled = cropped * scale_factor + add_offset
        scaled_data.append(scaled)
    
    # Stack all temporal data
    stacked = np.stack(scaled_data)
    
    # Take median ignoring NaN (already in the data from masking)
    median_composite = np.nanmedian(stacked, axis=0)
    
    # Replace remaining NaN with 0
    median_composite = np.where(np.isnan(median_composite), 0, median_composite)
    
    return median_composite


def create_temporal_composite(
    data_stack: ArrayLike,
    dates: list[datetime] | None = None,
    method: Literal["median", "mean", "max", "min", "percentile", "quality_weighted"] = "median",
    percentile: int | None = None,
    min_valid_observations: int = 1,
    quality_masks: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    nodata_value: float | None = None,
    use_scipy_fill_gaps: bool = False
) -> tuple[ArrayLike, ArrayLike, dict[str, Any]]:
    """Create temporal composite from stacked time series data.
    
    This is the primary function for creating temporal composites in the Black Marble
    pipeline. It combines multiple observations over time using various statistical
    methods to create a single representative image that reduces noise, fills gaps,
    and improves data quality.
    
    The function supports multiple compositing methods:
    - Median: Robust to outliers, preferred for most applications
    - Mean: Simple average, good for large sample sizes
    - Max/Min: Captures extreme values
    - Percentile: Flexible percentile-based compositing
    - Quality-weighted: Uses quality scores to weight observations
    
    Args:
        data_stack: Input time series as 3D array (time, height, width) or 
            4D array (time, bands, height, width) for multi-band data.
            Each time slice represents one observation.
        dates: List of acquisition dates for each time step.
            Must have same length as time dimension if provided.
            Used for temporal analysis and gap filling.
        method: Compositing method to use:
            - 'median': Median value (robust, recommended)
            - 'mean': Arithmetic mean
            - 'max': Maximum value
            - 'min': Minimum value
            - 'percentile': Specified percentile
            - 'quality_weighted': Weighted average using quality scores
        percentile: Percentile value (0-100) when method='percentile'.
            E.g., 75 for 75th percentile.
        min_valid_observations: Minimum number of valid observations required
            for a pixel to be included in the composite.
        quality_masks: Optional 3D boolean array (time, height, width) where
            True indicates good quality pixels to include in compositing.
        weights: Optional 3D array (time, height, width) of quality weights
            for 'quality_weighted' method. Higher values = better quality.
        nodata_value: Value representing no data. If None, uses np.nan.
        use_scipy_fill_gaps: Enable advanced gap filling using scipy distance
            transform for smoother interpolation of missing data.
            
    Returns:
        Tuple of (composite, valid_count, coverage_stats) where:
        - composite: Composite array with same spatial dimensions as input
        - valid_count: Number of valid observations used per pixel
        - coverage_stats: Dictionary with coverage statistics:
            - total_pixels: Total number of pixels
            - valid_pixels: Number of pixels with valid composite
            - coverage_percent: Percentage of valid pixels
            - mean_observations: Average observations per valid pixel
            - temporal_range: Date range if dates provided
            
    Raises:
        ValueError: If input dimensions don't match or invalid parameters.
        
    Example:
        >>> # Create median composite from monthly data
        >>> composite, counts, stats = create_temporal_composite(
        >>>     data_stack=monthly_radiance,  # (12, 1000, 1000)
        >>>     dates=monthly_dates,
        >>>     method='median',
        >>>     min_valid_observations=3,
        >>>     quality_masks=quality_stack
        >>> )
        >>> print(f"Coverage: {stats['coverage_percent']:.1f}%")
        
    Note:
        This function is a core component of the Black Marble algorithm,
        used to create stable composite images from temporal stacks.
    """
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    
    # Input validation
    if data_stack.ndim not in [3, 4]:
        raise ValueError(f"Data stack must be 3D or 4D, got {data_stack.ndim}D")
    
    if dates is not None and len(dates) != data_stack.shape[0]:
        raise ValueError(f"Dates length ({len(dates)}) must match time dimension ({data_stack.shape[0]})")
    
    if method == "percentile" and percentile is None:
        raise ValueError("Percentile value required when method='percentile'")
    
    if method == "quality_weighted" and weights is None:
        raise ValueError("Weights required when method='quality_weighted'")
    
    # Handle nodata value
    if nodata_value is None:
        nodata_value = np.nan
    
    # Get dimensions
    n_times = data_stack.shape[0]
    spatial_shape = data_stack.shape[1:] if data_stack.ndim == 3 else data_stack.shape[2:]
    
    # Apply quality masks if provided
    if quality_masks is not None:
        if quality_masks.shape != data_stack.shape[:3]:
            raise ValueError(f"Quality masks shape {quality_masks.shape} incompatible with data shape {data_stack.shape}")
        
        # Set poor quality pixels to nodata
        masked_data = data_stack.copy()
        if data_stack.ndim == 3:
            masked_data[~quality_masks] = nodata_value
        else:
            # For 4D data, broadcast mask across bands
            mask_4d = quality_masks[:, np.newaxis, :, :]
            masked_data[~mask_4d] = nodata_value
    else:
        masked_data = data_stack
    
    # Count valid observations per pixel
    if np.isnan(nodata_value):
        valid_obs_mask = ~np.isnan(masked_data)
    else:
        valid_obs_mask = masked_data != nodata_value
    
    if data_stack.ndim == 4:
        # For multi-band data, count valid observations across first band
        valid_count = np.sum(valid_obs_mask[:, 0, :, :], axis=0)
    else:
        valid_count = np.sum(valid_obs_mask, axis=0)
    
    # Create mask for pixels with sufficient observations
    sufficient_obs_mask = valid_count >= min_valid_observations
    
    # Initialize composite array
    if data_stack.ndim == 4:
        composite_shape = data_stack.shape[1:]
    else:
        composite_shape = data_stack.shape[1:]
    
    composite = np.full(composite_shape, nodata_value, dtype=np.float32)
    
    # Apply compositing method
    if data_stack.ndim == 4:
        # Process each band separately for multi-band data
        for band in range(data_stack.shape[1]):
            band_data = masked_data[:, band, :, :]
            band_composite = _composite_single_band(
                band_data, method, percentile, weights, sufficient_obs_mask, nodata_value
            )
            composite[band, :, :] = band_composite
    else:
        # Single band data
        composite = _composite_single_band(
            masked_data, method, percentile, weights, sufficient_obs_mask, nodata_value
        )
    
    # Apply gap filling if requested
    if use_scipy_fill_gaps and np.any(np.isnan(composite) if np.isnan(nodata_value) else composite == nodata_value):
        composite = _fill_gaps_scipy(composite, nodata_value)
    
    # Calculate coverage statistics
    total_pixels = np.prod(spatial_shape)
    valid_pixels = np.sum(sufficient_obs_mask)
    coverage_percent = (valid_pixels / total_pixels) * 100.0
    
    if valid_pixels > 0:
        mean_observations = float(np.mean(valid_count[sufficient_obs_mask]))
    else:
        mean_observations = 0.0
    
    coverage_stats = {
        'total_pixels': int(total_pixels),
        'valid_pixels': int(valid_pixels),
        'coverage_percent': float(coverage_percent),
        'mean_observations': mean_observations
    }
    
    # Add temporal range if dates provided
    if dates is not None:
        coverage_stats['temporal_range'] = (min(dates), max(dates))
    
    return composite, valid_count, coverage_stats


def _composite_single_band(
    data: ArrayLike,
    method: str,
    percentile: int | None,
    weights: ArrayLike | None,
    sufficient_obs_mask: ArrayLike,
    nodata_value: float
) -> ArrayLike:
    """Helper function to composite a single band of data."""
    import numpy as np
    
    composite = np.full(data.shape[1:], nodata_value, dtype=np.float32)
    
    # Only process pixels with sufficient observations
    valid_pixels = np.where(sufficient_obs_mask)
    
    for i, j in zip(valid_pixels[0], valid_pixels[1]):
        pixel_time_series = data[:, i, j]
        
        # Get valid observations for this pixel
        if np.isnan(nodata_value):
            valid_mask = ~np.isnan(pixel_time_series)
        else:
            valid_mask = pixel_time_series != nodata_value
        
        if not np.any(valid_mask):
            continue
        
        valid_values = pixel_time_series[valid_mask]
        
        # Apply compositing method
        if method == "median":
            composite[i, j] = np.median(valid_values)
        elif method == "mean":
            composite[i, j] = np.mean(valid_values)
        elif method == "max":
            composite[i, j] = np.max(valid_values)
        elif method == "min":
            composite[i, j] = np.min(valid_values)
        elif method == "percentile":
            if percentile is not None:
                composite[i, j] = np.percentile(valid_values, percentile)
        elif method == "quality_weighted":
            if weights is not None:
                pixel_weights = weights[:, i, j][valid_mask]
                if np.sum(pixel_weights) > 0:
                    composite[i, j] = np.average(valid_values, weights=pixel_weights)
                else:
                    composite[i, j] = np.mean(valid_values)
    
    return composite


def _fill_gaps_scipy(data: ArrayLike, nodata_value: float) -> ArrayLike:
    """Fill gaps using scipy distance transform."""
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    
    if data.ndim == 2:
        # Single band
        return _fill_gaps_2d(data, nodata_value)
    else:
        # Multi-band - process each band
        filled = data.copy()
        for band in range(data.shape[0]):
            filled[band] = _fill_gaps_2d(data[band], nodata_value)
        return filled


def _fill_gaps_2d(data: ArrayLike, nodata_value: float) -> ArrayLike:
    """Fill gaps in 2D array using nearest neighbor with distance weighting."""
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    
    # Create mask of missing data
    if np.isnan(nodata_value):
        missing_mask = np.isnan(data)
    else:
        missing_mask = data == nodata_value
    
    if not np.any(missing_mask):
        return data  # No gaps to fill
    
    filled = data.copy()
    valid_mask = ~missing_mask
    
    if not np.any(valid_mask):
        return data  # No valid data to interpolate from
    
    # Use distance transform for nearest neighbor interpolation
    result = distance_transform_edt(missing_mask, return_indices=True)
    distances, indices = result  # type: ignore
    
    # Fill missing pixels with nearest valid values
    filled[missing_mask] = data[tuple(indices[:, missing_mask])]
    
    return filled


def merge_scenes(
    scenes: list[ArrayLike],
    transforms: list[Transform],
    bounds: list[BBox],
    blend_method: Literal["feather", "max", "min", "mean"] = "feather",
    feather_distance: int = 10,
    priority_order: list[int] | None = None,
    nodata: float | None = None,
    use_scipy_feather: bool = False
) -> tuple[ArrayLike, Transform, BBox]:
    """Merge multiple satellite scenes into a single seamless mosaic.
    
    This function combines multiple overlapping satellite scenes into a single
    continuous image. It handles geometric alignment, overlap resolution, and
    edge blending to create professional-quality mosaics suitable for analysis.
    
    The function supports several blending methods:
    - Feather: Creates smooth transitions using distance-weighted blending
    - Max/Min: Takes maximum or minimum values in overlap areas
    - Mean: Averages overlapping pixels
    
    Args:
        scenes: List of 2D or 3D scene arrays to merge.
            All scenes must have same number of bands if 3D.
        transforms: List of rasterio Affine transforms for each scene.
            Must have same length as scenes list.
        bounds: List of (min_x, min_y, max_x, max_y) tuples in map coordinates.
            Must have same length as scenes list.
        blend_method: Method for handling overlapping areas:
            - 'feather': Distance-weighted smooth blending (recommended)
            - 'max': Take maximum value in overlaps
            - 'min': Take minimum value in overlaps
            - 'mean': Average overlapping pixels
        feather_distance: Distance in pixels for feathering transition zone.
            Larger values create smoother transitions but may blur features.
        priority_order: Optional list of scene indices in priority order.
            Higher priority scenes are preferred in overlaps.
            If None, uses input order as priority.
        nodata: Value representing no data. If None, uses np.nan.
        use_scipy_feather: Enable scipy-based advanced feathering.
            Provides smoother blending but requires scipy dependency.
            
    Returns:
        Tuple of (merged_data, merged_transform, merged_bounds) where:
        - merged_data: Merged array covering the union of all input scenes
        - merged_transform: Affine transform for the merged mosaic
        - merged_bounds: Geographic bounds of the merged mosaic
        
    Raises:
        ValueError: If input lists have different lengths or invalid parameters.
        
    Example:
        >>> # Merge overlapping VIIRS tiles
        >>> merged_data, transform, bounds = merge_scenes(
        >>>     scenes=[tile1_data, tile2_data],
        >>>     transforms=[tile1_transform, tile2_transform],
        >>>     bounds=[tile1_bounds, tile2_bounds],
        >>>     blend_method='feather',
        >>>     feather_distance=20
        >>> )
        >>> # Result is seamless mosaic covering both tiles
        
    Note:
        This function is essential for creating seamless regional mosaics
        from multiple satellite scenes in the Black Marble pipeline.
    """
    import numpy as np
    from rasterio.transform import Affine
    
    # Input validation
    n_scenes = len(scenes)
    if not (len(transforms) == len(bounds) == n_scenes):
        raise ValueError(f"All input lists must have same length: scenes={n_scenes}, transforms={len(transforms)}, bounds={len(bounds)}")
    
    if n_scenes == 0:
        raise ValueError("At least one scene required")
    
    if n_scenes == 1:
        # Single scene - return as is
        return scenes[0], transforms[0], bounds[0]
    
    # Handle nodata value
    if nodata is None:
        nodata = np.nan
    
    # Set priority order
    if priority_order is None:
        priority_order = list(range(n_scenes))
    
    # Calculate output bounds (union of all scenes)
    min_x = min(b[0] for b in bounds)
    min_y = min(b[1] for b in bounds)
    max_x = max(b[2] for b in bounds)
    max_y = max(b[3] for b in bounds)
    merged_bounds = (min_x, min_y, max_x, max_y)
    
    # Determine output resolution from first scene
    first_transform = transforms[0]
    pixel_size_x = abs(first_transform.a)
    pixel_size_y = abs(first_transform.e)
    
    # Calculate output dimensions
    output_width = int(np.ceil((max_x - min_x) / pixel_size_x))
    output_height = int(np.ceil((max_y - min_y) / pixel_size_y))
    
    # Create output transform
    merged_transform = Affine(
        pixel_size_x, 0.0, min_x,
        0.0, -pixel_size_y, max_y
    )
    
    # Determine output shape based on input scenes
    if scenes[0].ndim == 2:
        output_shape = (output_height, output_width)
        n_bands = 1
    else:
        n_bands = scenes[0].shape[0]
        output_shape = (n_bands, output_height, output_width)
    
    # Initialize output arrays
    merged_data = np.full(output_shape, nodata, dtype=np.float32)
    
    # Initialize feathering variables (used for type safety)
    weight_sum = np.zeros((output_height, output_width), dtype=np.float32)
    weighted_sum = np.zeros(output_shape, dtype=np.float32)
    
    # Process scenes in priority order
    for scene_idx in priority_order:
        scene = scenes[scene_idx]
        scene_transform = transforms[scene_idx]
        scene_bounds = bounds[scene_idx]
        
        # Calculate scene position in output grid
        col_offset = int((scene_bounds[0] - min_x) / pixel_size_x)
        row_offset = int((max_y - scene_bounds[3]) / pixel_size_y)
        
        # Get scene dimensions
        if scene.ndim == 2:
            scene_height, scene_width = scene.shape
        else:
            _, scene_height, scene_width = scene.shape
        
        # Calculate valid output region
        row_start = max(0, row_offset)
        row_end = min(output_height, row_offset + scene_height)
        col_start = max(0, col_offset)
        col_end = min(output_width, col_offset + scene_width)
        
        # Calculate corresponding scene region
        scene_row_start = max(0, -row_offset)
        scene_row_end = scene_row_start + (row_end - row_start)
        scene_col_start = max(0, -col_offset)
        scene_col_end = scene_col_start + (col_end - col_start)
        
        if row_start >= row_end or col_start >= col_end:
            continue  # No overlap
        
        # Extract scene data for this region
        if scene.ndim == 2:
            scene_data = scene[scene_row_start:scene_row_end, scene_col_start:scene_col_end]
        else:
            scene_data = scene[:, scene_row_start:scene_row_end, scene_col_start:scene_col_end]
        
        # Create valid data mask
        if np.isnan(nodata):
            if scene.ndim == 2:
                valid_mask = ~np.isnan(scene_data)
            else:
                valid_mask = ~np.isnan(scene_data[0])  # Use first band for mask
        else:
            if scene.ndim == 2:
                valid_mask = scene_data != nodata
            else:
                valid_mask = scene_data[0] != nodata
        
        if not np.any(valid_mask):
            continue  # No valid data in this scene
        
        # Apply blending method
        if blend_method == "feather":
            # Calculate distance weights for feathering
            if use_scipy_feather:
                weights = _calculate_feather_weights_scipy(valid_mask, feather_distance)
            else:
                weights = _calculate_feather_weights_simple(valid_mask, feather_distance)
            
            # Apply feather blending
            region_slice = (slice(row_start, row_end), slice(col_start, col_end))
            
            if scene.ndim == 2:
                weighted_sum[region_slice] += scene_data * weights
            else:
                for band in range(n_bands):
                    weighted_sum[band, row_start:row_end, col_start:col_end] += scene_data[band] * weights
            
            weight_sum[row_start:row_end, col_start:col_end] += weights
            
        else:
            # Simple blending methods
            region_slice = (slice(row_start, row_end), slice(col_start, col_end))
            
            if scene.ndim == 2:
                existing_data = merged_data[region_slice]
                existing_valid = ~(np.isnan(existing_data) if np.isnan(nodata) else existing_data == nodata)
                
                if blend_method == "max":
                    merged_data[region_slice] = np.where(
                        valid_mask & existing_valid,
                        np.maximum(existing_data, scene_data),
                        np.where(valid_mask, scene_data, existing_data)
                    )
                elif blend_method == "min":
                    merged_data[region_slice] = np.where(
                        valid_mask & existing_valid,
                        np.minimum(existing_data, scene_data),
                        np.where(valid_mask, scene_data, existing_data)
                    )
                elif blend_method == "mean":
                    merged_data[region_slice] = np.where(
                        valid_mask & existing_valid,
                        (existing_data + scene_data) / 2.0,
                        np.where(valid_mask, scene_data, existing_data)
                    )
            else:
                # Multi-band processing
                for band in range(n_bands):
                    band_slice = (band, slice(row_start, row_end), slice(col_start, col_end))
                    existing_data = merged_data[band_slice]
                    existing_valid = ~(np.isnan(existing_data) if np.isnan(nodata) else existing_data == nodata)
                    
                    if blend_method == "max":
                        merged_data[band_slice] = np.where(
                            valid_mask & existing_valid,
                            np.maximum(existing_data, scene_data[band]),
                            np.where(valid_mask, scene_data[band], existing_data)
                        )
                    elif blend_method == "min":
                        merged_data[band_slice] = np.where(
                            valid_mask & existing_valid,
                            np.minimum(existing_data, scene_data[band]),
                            np.where(valid_mask, scene_data[band], existing_data)
                        )
                    elif blend_method == "mean":
                        merged_data[band_slice] = np.where(
                            valid_mask & existing_valid,
                            (existing_data + scene_data[band]) / 2.0,
                            np.where(valid_mask, scene_data[band], existing_data)
                        )
    
    # Finalize feather blending
    if blend_method == "feather":
        valid_weights = weight_sum > 0
        if len(scenes) > 0 and scenes[0].ndim == 2:
            merged_data = np.where(valid_weights, weighted_sum / weight_sum, nodata)
        else:
            for band in range(n_bands):
                merged_data[band] = np.where(valid_weights, weighted_sum[band] / weight_sum, nodata)
    
    return merged_data, merged_transform, merged_bounds


def _calculate_feather_weights_simple(valid_mask: ArrayLike, feather_distance: int) -> ArrayLike:
    """Calculate simple feather weights using erosion-based approach."""
    import numpy as np
    from scipy.ndimage import binary_erosion
    
    weights = np.zeros_like(valid_mask, dtype=np.float32)
    
    # Create distance-based weights by iterative erosion
    current_mask = valid_mask.copy()
    for distance in range(feather_distance, 0, -1):
        eroded_mask = binary_erosion(current_mask)
        edge_mask = current_mask & ~eroded_mask
        weights[edge_mask] = distance / feather_distance
        current_mask = eroded_mask
    
    # Inner area gets full weight
    weights[current_mask] = 1.0
    
    return weights


def _calculate_feather_weights_scipy(valid_mask: ArrayLike, feather_distance: int) -> ArrayLike:
    """Calculate advanced feather weights using distance transform."""
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    
    # Calculate distance from edge
    distances = distance_transform_edt(valid_mask)
    
    # Create smooth weight transition
    max_distance = float(feather_distance)
    weights = np.minimum(np.asarray(distances) / max_distance, 1.0)
    
    # Smooth the transition with sigmoid-like function
    weights = weights ** 0.5  # Square root for smoother transition
    
    return weights.astype(np.float32)


def select_by_date_range(
    scene_dates: list[datetime],
    scene_metadata: list[dict[str, Any]] | None = None,
    quality_scores: list[float] | None = None,
    target_date: datetime | None = None,
    date_range: tuple[datetime, datetime] | None = None,
    max_days_from_target: int | None = None,
    seasons: list[Literal["spring", "summer", "fall", "winter"]] | None = None,
    cloud_threshold: float | None = None,
    min_scenes: int = 1,
    max_scenes: int | None = None
) -> list[int]:
    """Select scene indices based on date and quality criteria.
    
    This function implements intelligent scene selection for temporal compositing
    based on multiple criteria including dates, quality, and metadata. It enables
    flexible filtering and ranking of available scenes to optimize composite quality.
    
    The selection process applies filters in sequence:
    1. Date-based filtering (range, target, seasonal)
    2. Quality-based filtering (cloud cover, quality scores)
    3. Ranking and limiting to desired number of scenes
    
    Args:
        scene_dates: List of acquisition dates for each scene.
            Must be datetime objects.
        scene_metadata: Optional list of metadata dictionaries for each scene.
            May contain keys like 'cloud_cover', 'solar_zenith', etc.
        quality_scores: Optional list of quality scores (0-1) for ranking.
            Higher scores indicate better quality.
        target_date: Target acquisition date for temporal proximity filtering.
            Scenes closer to this date are preferred.
        date_range: Explicit date range as (start_date, end_date) tuple.
            Only scenes within this range are considered.
        max_days_from_target: Maximum days from target_date to include.
            Creates a symmetric window around target_date.
        seasons: List of seasons to include. Options:
            - 'spring': March-May (Northern Hemisphere)
            - 'summer': June-August
            - 'fall': September-November  
            - 'winter': December-February
        cloud_threshold: Maximum cloud cover percentage (0-100).
            Requires 'cloud_cover' key in scene_metadata.
        min_scenes: Minimum number of scenes to return.
            If fewer scenes pass filters, raises ValueError.
        max_scenes: Maximum number of scenes to return.
            If more scenes pass filters, returns highest quality ones.
            
    Returns:
        List of selected scene indices (0-based) in descending quality order.
        Indices correspond to positions in input scene_dates list.
        
    Raises:
        ValueError: If fewer than min_scenes pass filters or input validation fails.
        
    Example:
        >>> # Select best 5 scenes from summer near target date
        >>> selected_indices = select_by_date_range(
        >>>     scene_dates=acquisition_dates,
        >>>     scene_metadata=metadata_list,
        >>>     quality_scores=quality_list,
        >>>     target_date=datetime(2023, 7, 15),
        >>>     max_days_from_target=30,
        >>>     seasons=['summer'],
        >>>     cloud_threshold=20.0,
        >>>     max_scenes=5
        >>> )
        >>> print(f"Selected {len(selected_indices)} scenes")
        
    Note:
        This function is essential for optimizing temporal composite quality
        by selecting the most suitable input scenes.
    """
    import numpy as np
    from datetime import timedelta
    
    # Input validation
    n_scenes = len(scene_dates)
    if n_scenes == 0:
        raise ValueError("At least one scene date required")
    
    if scene_metadata is not None and len(scene_metadata) != n_scenes:
        raise ValueError(f"Metadata length ({len(scene_metadata)}) must match dates length ({n_scenes})")
    
    if quality_scores is not None and len(quality_scores) != n_scenes:
        raise ValueError(f"Quality scores length ({len(quality_scores)}) must match dates length ({n_scenes})")
    
    # Initialize candidate indices
    candidate_indices = list(range(n_scenes))
    
    # Apply date range filter
    if date_range is not None:
        start_date, end_date = date_range
        candidate_indices = [
            i for i in candidate_indices
            if start_date <= scene_dates[i] <= end_date
        ]
    
    # Apply target date proximity filter
    if target_date is not None and max_days_from_target is not None:
        max_delta = timedelta(days=max_days_from_target)
        candidate_indices = [
            i for i in candidate_indices
            if abs(scene_dates[i] - target_date) <= max_delta
        ]
    
    # Apply seasonal filter
    if seasons is not None:
        def get_season(date):
            month = date.month
            if month in [3, 4, 5]:
                return "spring"
            elif month in [6, 7, 8]:
                return "summer"
            elif month in [9, 10, 11]:
                return "fall"
            else:
                return "winter"
        
        candidate_indices = [
            i for i in candidate_indices
            if get_season(scene_dates[i]) in seasons
        ]
    
    # Apply cloud cover filter
    if cloud_threshold is not None and scene_metadata is not None:
        candidate_indices = [
            i for i in candidate_indices
            if scene_metadata[i].get('cloud_cover', 0) <= cloud_threshold
        ]
    
    # Check minimum scenes requirement
    if len(candidate_indices) < min_scenes:
        raise ValueError(
            f"Only {len(candidate_indices)} scenes pass filters, but {min_scenes} required. "
            f"Consider relaxing filter criteria."
        )
    
    # Calculate composite quality score for ranking
    scene_scores = []
    for i in candidate_indices:
        score = 0.0
        
        # Quality score component (40% weight)
        if quality_scores is not None:
            score += 0.4 * quality_scores[i]
        else:
            score += 0.4  # Default moderate score
        
        # Date proximity component (30% weight)
        if target_date is not None:
            days_diff = abs((scene_dates[i] - target_date).days)
            # Exponential decay with distance
            proximity_score = np.exp(-days_diff / 30.0)  # 30-day characteristic time
            score += 0.3 * proximity_score
        else:
            score += 0.3  # No preference
        
        # Cloud cover component (20% weight)
        if scene_metadata is not None and 'cloud_cover' in scene_metadata[i]:
            cloud_cover = scene_metadata[i]['cloud_cover']
            cloud_score = max(0, (100 - cloud_cover) / 100.0)
            score += 0.2 * cloud_score
        else:
            score += 0.2  # Default moderate score
        
        # Solar geometry component (10% weight)
        if scene_metadata is not None and 'solar_zenith' in scene_metadata[i]:
            solar_zenith = scene_metadata[i]['solar_zenith']
            # Prefer lower solar zenith angles (sun higher in sky)
            geometry_score = max(0, (90 - solar_zenith) / 90.0)
            score += 0.1 * geometry_score
        else:
            score += 0.1  # Default moderate score
        
        scene_scores.append((score, i))
    
    # Sort by score (descending)
    scene_scores.sort(reverse=True)
    
    # Apply max scenes limit
    if max_scenes is not None:
        scene_scores = scene_scores[:max_scenes]
    
    # Extract indices in quality order
    selected_indices = [i for score, i in scene_scores]
    
    return selected_indices


def composite_with_quality_weighting(
    data_stack: ArrayLike,
    quality_scores: ArrayLike,
    method: Literal["weighted_mean", "best_quality", "threshold_mean"] = "weighted_mean",
    threshold: float | None = None
) -> tuple[ArrayLike, ArrayLike]:
    """Create composite using quality scores for optimal pixel selection.
    
    This function creates temporal composites by leveraging quality information
    to weight or select the best pixels from multiple observations. It ensures
    that higher quality observations contribute more to the final composite.
    
    Quality scores can represent various metrics:
    - Overall scene quality (0-1)
    - Cloud-free probability
    - Atmospheric correction confidence
    - Geometric accuracy measures
    
    Args:
        data_stack: 3D array (time, height, width) or 4D (time, bands, height, width)
            containing stacked observations to composite.
        quality_scores: Quality scores as 1D array (time,) for scene-level scores
            or 3D array (time, height, width) for pixel-level scores.
            Range: [0, 1] where 1 = highest quality.
        method: Compositing method:
            - 'weighted_mean': Weight pixels by quality scores
            - 'best_quality': Select pixel from highest quality observation
            - 'threshold_mean': Average only pixels above quality threshold
        threshold: Quality threshold (0-1) for 'threshold_mean' method.
            Only observations with quality >= threshold are included.
            
    Returns:
        Tuple of (composite, composite_quality) where:
        - composite: Quality-weighted composite array
        - composite_quality: Quality map showing confidence in composite values
        
    Raises:
        ValueError: If input dimensions don't match or parameters are invalid.
        
    Example:
        >>> # Create quality-weighted composite
        >>> composite, quality_map = composite_with_quality_weighting(
        >>>     data_stack=monthly_data,       # (12, 1000, 1000)
        >>>     quality_scores=scene_quality,  # (12,) - scene-level scores
        >>>     method='weighted_mean'
        >>> )
        >>> 
        >>> # Use threshold-based compositing
        >>> composite, quality_map = composite_with_quality_weighting(
        >>>     data_stack=monthly_data,
        >>>     quality_scores=pixel_quality,  # (12, 1000, 1000) - pixel-level
        >>>     method='threshold_mean',
        >>>     threshold=0.7
        >>> )
        
    Note:
        This function is valuable for creating high-quality composites from
        datasets with variable observation quality.
    """
    import numpy as np
    
    # Input validation
    if data_stack.ndim not in [3, 4]:
        raise ValueError(
            f"Data stack must be 3D (time, height, width) or 4D (time, bands, height, width), "
            f"got {data_stack.ndim}D"
        )
    
    n_times = data_stack.shape[0]
    
    # Validate quality scores
    if np.any((quality_scores < 0) | (quality_scores > 1)):
        raise ValueError("Quality scores must be in range [0, 1]")
    
    # Handle quality scores dimensions
    if quality_scores.ndim == 1:
        # Scene-level quality scores - broadcast to all pixels
        if len(quality_scores) != n_times:
            raise ValueError(
                f"Scene-level quality scores length ({len(quality_scores)}) "
                f"must match time dimension ({n_times})"
            )
        
        # Broadcast to spatial dimensions
        if data_stack.ndim == 3:
            spatial_shape = data_stack.shape[1:]
            quality_weights = quality_scores[:, np.newaxis, np.newaxis]
            quality_weights = np.broadcast_to(quality_weights, data_stack.shape)
        else:
            spatial_shape = data_stack.shape[2:]
            quality_weights = quality_scores[:, np.newaxis, np.newaxis, np.newaxis]
            quality_weights = np.broadcast_to(quality_weights, data_stack.shape)
            
    elif quality_scores.ndim == 3:
        # Pixel-level quality scores
        if quality_scores.shape != data_stack.shape[:3]:
            raise ValueError(f"Quality scores shape {quality_scores.shape} incompatible with data shape {data_stack.shape}")
        
        if data_stack.ndim == 3:
            quality_weights = quality_scores
            spatial_shape = data_stack.shape[1:]
        else:
            # Broadcast across bands for 4D data
            quality_weights = quality_scores[:, np.newaxis, :, :]
            quality_weights = np.broadcast_to(quality_weights, data_stack.shape)
            spatial_shape = data_stack.shape[2:]
    else:
        raise ValueError(f"Quality scores must be 1D or 3D, got {quality_scores.ndim}D")
    
    # Validate method-specific parameters
    if method == "threshold_mean" and threshold is None:
        raise ValueError("Threshold required for 'threshold_mean' method")
    
    # Initialize output arrays
    if data_stack.ndim == 3:
        composite = np.zeros(spatial_shape, dtype=np.float32)
        composite_quality = np.zeros(spatial_shape, dtype=np.float32)
    else:
        composite = np.zeros(data_stack.shape[1:], dtype=np.float32)
        composite_quality = np.zeros(spatial_shape, dtype=np.float32)
    
    # Apply compositing method
    if method == "weighted_mean":
        # Weight observations by quality scores
        
        # Calculate weighted sum and weight sum
        valid_mask = ~np.isnan(data_stack) & (quality_weights > 0)
        
        if data_stack.ndim == 3:
            weighted_sum = np.nansum(data_stack * quality_weights * valid_mask, axis=0)
            weight_sum = np.sum(quality_weights * valid_mask, axis=0)
            
            # Calculate composite (avoid division by zero)
            nonzero_weights = weight_sum > 0
            composite[nonzero_weights] = weighted_sum[nonzero_weights] / weight_sum[nonzero_weights]
            composite[~nonzero_weights] = np.nan
            
            # Composite quality is normalized weight sum
            max_possible_weight = np.sum(quality_weights[:, 0, 0])  # Use corner pixel as reference
            if max_possible_weight > 0:
                composite_quality = weight_sum / max_possible_weight
        else:
            # Multi-band data
            for band in range(data_stack.shape[1]):
                band_data = data_stack[:, band, :, :]
                band_weights = quality_weights[:, band, :, :]
                band_valid = ~np.isnan(band_data) & (band_weights > 0)
                
                weighted_sum = np.nansum(band_data * band_weights * band_valid, axis=0)
                weight_sum = np.sum(band_weights * band_valid, axis=0)
                
                nonzero_weights = weight_sum > 0
                composite[band, nonzero_weights] = weighted_sum[nonzero_weights] / weight_sum[nonzero_weights]
                composite[band, ~nonzero_weights] = np.nan
            
            # Composite quality from first band weights
            first_band_weights = quality_weights[:, 0, :, :]
            first_band_valid = ~np.isnan(data_stack[:, 0, :, :]) & (first_band_weights > 0)
            weight_sum = np.sum(first_band_weights * first_band_valid, axis=0)
            max_possible_weight = np.sum(first_band_weights[:, 0, 0])
            if max_possible_weight > 0:
                composite_quality = weight_sum / max_possible_weight
    
    elif method == "best_quality":
        # Select pixel from highest quality observation
        
        if data_stack.ndim == 3:
            # Find index of highest quality observation per pixel
            # Set quality to 0 for invalid pixels
            masked_quality = np.where(np.isnan(data_stack), 0, quality_weights)
            best_indices = np.argmax(masked_quality, axis=0)
            
            # Extract best pixels
            composite = data_stack[best_indices, np.arange(spatial_shape[0])[:, np.newaxis], 
                                 np.arange(spatial_shape[1])]
            composite_quality = quality_weights[best_indices, np.arange(spatial_shape[0])[:, np.newaxis], 
                                              np.arange(spatial_shape[1])]
        else:
            # Multi-band data - use quality from first band to select best observation
            first_band_quality = quality_weights[:, 0, :, :]
            masked_quality = np.where(np.isnan(data_stack[:, 0, :, :]), 0, first_band_quality)
            best_indices = np.argmax(masked_quality, axis=0)
            
            # Extract best pixels for all bands
            for band in range(data_stack.shape[1]):
                composite[band] = data_stack[best_indices, band, 
                                           np.arange(spatial_shape[0])[:, np.newaxis], 
                                           np.arange(spatial_shape[1])]
            
            composite_quality = first_band_quality[best_indices, 
                                                  np.arange(spatial_shape[0])[:, np.newaxis], 
                                                  np.arange(spatial_shape[1])]
    
    elif method == "threshold_mean":
        # Average only observations above quality threshold
        
        threshold_mask = quality_weights >= threshold
        valid_mask = ~np.isnan(data_stack) & threshold_mask
        
        if data_stack.ndim == 3:
            # Count valid observations per pixel
            valid_count = np.sum(valid_mask, axis=0)
            
            # Calculate mean of valid observations
            composite = np.nansum(data_stack * valid_mask, axis=0) / np.maximum(valid_count, 1)
            
            # Set pixels with no valid observations to NaN
            composite[valid_count == 0] = np.nan
            
            # Composite quality is fraction of observations above threshold
            composite_quality = valid_count / n_times
        else:
            # Multi-band data
            for band in range(data_stack.shape[1]):
                band_valid = valid_mask[:, band, :, :]
                valid_count = np.sum(band_valid, axis=0)
                
                composite[band] = np.nansum(data_stack[:, band, :, :] * band_valid, axis=0) / np.maximum(valid_count, 1)
                composite[band][valid_count == 0] = np.nan
            
            # Composite quality from first band
            first_band_valid = valid_mask[:, 0, :, :]
            composite_quality = np.sum(first_band_valid, axis=0) / n_times
    
    return composite, composite_quality


def compute_temporal_statistics(
    data_stack: ArrayLike,
    dates: list[datetime] | None = None,
    compute_trend: bool = False
) -> dict[str, ArrayLike]:
    """Compute comprehensive temporal statistics for time series analysis.
    
    This function calculates pixel-wise temporal statistics across a time series
    stack, providing insights into temporal variability, trends, and data quality.
    The statistics are essential for understanding temporal dynamics and identifying
    anomalies or change patterns.
    
    Args:
        data_stack: 3D array (time, height, width) or 4D (time, bands, height, width)
            containing time series observations.
        dates: Optional list of datetime objects for each time step.
            Required for trend analysis. Must match time dimension length.
        compute_trend: Whether to compute temporal trend using linear regression.
            Requires dates parameter to be provided.
            
    Returns:
        Dictionary containing statistical arrays with spatial dimensions:
        - 'mean': Temporal mean across all observations
        - 'std': Temporal standard deviation
        - 'min': Temporal minimum value
        - 'max': Temporal maximum value
        - 'range': Temporal range (max - min)
        - 'cv': Coefficient of variation (std/mean) where mean != 0
        - 'count': Number of valid (non-NaN) observations per pixel
        - 'median': Temporal median (robust central tendency)
        - 'p25': 25th percentile
        - 'p75': 75th percentile
        - 'trend': Optional linear trend slope (if compute_trend=True)
        - 'trend_pvalue': Optional trend significance p-value
        
    Raises:
        ValueError: If dates don't match time dimension or trend requested without dates.
        
    Example:
        >>> # Compute temporal statistics for NDVI time series
        >>> stats = compute_temporal_statistics(
        >>>     data_stack=ndvi_monthly,  # (12, 1000, 1000)
        >>>     dates=monthly_dates,
        >>>     compute_trend=True
        >>> )
        >>> 
        >>> # Identify areas with high temporal variability
        >>> high_cv_areas = stats['cv'] > 0.5
        >>> 
        >>> # Find areas with significant positive trends
        >>> significant_trends = (
        >>>     (stats['trend'] > 0) & 
        >>>     (stats['trend_pvalue'] < 0.05)
        >>> )
        
    Note:
        This function is valuable for change detection, phenology analysis,
        and quality assessment of time series data.
    """
    import numpy as np
    
    # Input validation
    if data_stack.ndim not in [3, 4]:
        raise ValueError(f"Data stack must be 3D or 4D, got {data_stack.ndim}D")
    
    n_times = data_stack.shape[0]
    
    if dates is not None and len(dates) != n_times:
        raise ValueError(f"Dates length ({len(dates)}) must match time dimension ({n_times})")
    
    if compute_trend and dates is None:
        raise ValueError("Dates required for trend computation")
    
    # Determine output shape
    if data_stack.ndim == 3:
        output_shape = data_stack.shape[1:]
        is_multiband = False
    else:
        output_shape = data_stack.shape[1:]
        is_multiband = True
    
    # Initialize output dictionary
    stats = {}
    
    # Function to compute statistics for single band
    def compute_band_stats(band_data):
        """Compute statistics for a single band (2D spatial array over time)."""
        band_stats = {}
        
        # Count valid observations
        valid_mask = ~np.isnan(band_data)
        valid_count = np.sum(valid_mask, axis=0)
        
        # Basic statistics
        band_stats['mean'] = np.nanmean(band_data, axis=0)
        band_stats['std'] = np.nanstd(band_data, axis=0)
        band_stats['min'] = np.nanmin(band_data, axis=0)
        band_stats['max'] = np.nanmax(band_data, axis=0)
        band_stats['range'] = band_stats['max'] - band_stats['min']
        band_stats['count'] = valid_count
        band_stats['median'] = np.nanmedian(band_data, axis=0)
        
        # Percentiles
        band_stats['p25'] = np.nanpercentile(band_data, 25, axis=0)
        band_stats['p75'] = np.nanpercentile(band_data, 75, axis=0)
        
        # Coefficient of variation (handle division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            cv = band_stats['std'] / np.abs(band_stats['mean'])
            cv[band_stats['mean'] == 0] = np.inf
            cv[np.isnan(band_stats['mean'])] = np.nan
        band_stats['cv'] = cv
        
        # Trend analysis if requested
        if compute_trend and dates is not None:
            trend_slope, trend_pvalue = _compute_pixel_trends(band_data, dates)
            band_stats['trend'] = trend_slope
            band_stats['trend_pvalue'] = trend_pvalue
        
        return band_stats
    
    if is_multiband:
        # Multi-band data - compute statistics for each band
        n_bands = data_stack.shape[1]
        
        # Initialize arrays for each statistic
        stat_names = ['mean', 'std', 'min', 'max', 'range', 'count', 'median', 'p25', 'p75', 'cv']
        if compute_trend:
            stat_names.extend(['trend', 'trend_pvalue'])
        
        for stat_name in stat_names:
            stats[stat_name] = np.full(output_shape, np.nan, dtype=np.float32)
        
        # Compute statistics for each band
        for band in range(n_bands):
            band_data = data_stack[:, band, :, :]
            band_stats = compute_band_stats(band_data)
            
            for stat_name in stat_names:
                stats[stat_name][band, :, :] = band_stats[stat_name]
    else:
        # Single band data
        stats = compute_band_stats(data_stack)
    
    return stats


def _compute_pixel_trends(data: ArrayLike, dates: list[datetime]) -> tuple[ArrayLike, ArrayLike]:
    """Compute linear trend slope and p-value for each pixel."""
    import numpy as np
    from datetime import datetime
    
    # Convert dates to numeric values (days since first date)
    first_date = dates[0]
    numeric_dates = np.array([(d - first_date).days for d in dates], dtype=np.float32)
    
    # Initialize output arrays
    height, width = data.shape[1:]
    trend_slopes = np.full((height, width), np.nan, dtype=np.float32)
    trend_pvalues = np.full((height, width), np.nan, dtype=np.float32)
    
    # Compute trends pixel by pixel
    for i in range(height):
        for j in range(width):
            pixel_values = data[:, i, j]
            valid_mask = ~np.isnan(pixel_values)
            
            if np.sum(valid_mask) < 3:  # Need at least 3 points for trend
                continue
            
            valid_dates = numeric_dates[valid_mask]
            valid_values = pixel_values[valid_mask]
            
            # Simple linear regression
            n = len(valid_dates)
            sum_x = np.sum(valid_dates)
            sum_y = np.sum(valid_values)
            sum_xy = np.sum(valid_dates * valid_values)
            sum_x2 = np.sum(valid_dates * valid_dates)
            
            # Calculate slope
            denominator = n * sum_x2 - sum_x * sum_x
            if abs(denominator) < 1e-10:
                continue  # Avoid division by zero
            
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n
            
            # Calculate p-value (simplified)
            # Residual sum of squares
            y_pred = slope * valid_dates + intercept
            residuals = valid_values - y_pred
            rss = np.sum(residuals ** 2)
            
            if n > 2 and rss > 0:
                # Standard error of slope
                mse = rss / (n - 2)
                se_slope = np.sqrt(mse / np.sum((valid_dates - np.mean(valid_dates)) ** 2))
                
                # t-statistic
                t_stat = abs(slope / se_slope) if se_slope > 0 else 0
                
                # Approximate p-value (two-tailed)
                # This is a simplified approximation
                if t_stat > 2.57:  # 99% confidence
                    p_value = 0.01
                elif t_stat > 1.96:  # 95% confidence
                    p_value = 0.05
                elif t_stat > 1.65:  # 90% confidence
                    p_value = 0.10
                else:
                    p_value = 0.20
            else:
                p_value = 1.0
            
            trend_slopes[i, j] = slope
            trend_pvalues[i, j] = p_value
    
    return trend_slopes, trend_pvalues


def detect_change(
    before_composite: ArrayLike,
    after_composite: ArrayLike,
    threshold: float = 0.2,
    index_type: str = "ndvi"
) -> ArrayLike:
    """Detect significant changes between two temporal composites.
    
    This function performs change detection by comparing pixel values between
    two temporal composites (e.g., before/after images). It uses index-specific
    thresholds and change metrics to identify areas of significant change.
    
    The function supports different spectral indices with appropriate thresholds:
    - NDVI: Vegetation change detection
    - NDWI: Water change detection  
    - NDUI: Urban development detection
    - Radiance: General brightness changes
    
    Args:
        before_composite: Earlier temporal composite as reference.
            Should be same spatial dimensions as after_composite.
        after_composite: Later temporal composite for comparison.
            Change is calculated as (after - before).
        threshold: Absolute change threshold for significance.
            Index-specific default thresholds are applied if threshold=None.
        index_type: Type of spectral index being analyzed:
            - 'ndvi': Normalized Difference Vegetation Index
            - 'ndwi': Normalized Difference Water Index
            - 'ndui': Normalized Difference Urban Index
            - 'radiance': Raw radiance values
            - 'generic': Generic change detection
            
    Returns:
        Change detection mask as integer array with values:
        - 0: No significant change
        - 1: Significant positive change (increase)
        - -1: Significant negative change (decrease)
        
    Raises:
        ValueError: If input arrays have incompatible shapes.
        
    Example:
        >>> # Detect vegetation loss/gain
        >>> change_mask = detect_change(
        >>>     before_composite=ndvi_2020,
        >>>     after_composite=ndvi_2021,
        >>>     threshold=0.1,
        >>>     index_type='ndvi'
        >>> )
        >>> 
        >>> # Count changed pixels
        >>> vegetation_loss = np.sum(change_mask == -1)
        >>> vegetation_gain = np.sum(change_mask == 1)
        >>> print(f"Loss: {vegetation_loss}, Gain: {vegetation_gain} pixels")
        
    Note:
        Change detection is fundamental for monitoring land cover dynamics,
        urban expansion, and environmental changes.
    """
    import numpy as np
    
    # Input validation
    if before_composite.shape != after_composite.shape:
        raise ValueError(f"Composite shapes must match: {before_composite.shape} vs {after_composite.shape}")
    
    # Apply index-specific default thresholds if not provided
    if threshold is None:
        default_thresholds = {
            'ndvi': 0.1,    # 10% NDVI change
            'ndwi': 0.15,   # 15% NDWI change (water is more variable)
            'ndui': 0.05,   # 5% NDUI change (urban development)
            'radiance': 1.0, # 1 nW/cm²/sr radiance change
            'generic': 0.2   # 20% generic change
        }
        threshold = default_thresholds.get(index_type, 0.2)
    
    # Calculate change (after - before)
    change = after_composite - before_composite
    
    # Create masks for valid pixels (non-NaN in both images)
    valid_mask = ~np.isnan(before_composite) & ~np.isnan(after_composite)
    
    # Initialize change detection result
    change_mask = np.zeros_like(change, dtype=np.int8)
    
    # Apply index-specific change detection logic
    if index_type == 'ndvi':
        # NDVI change detection
        # Positive change: vegetation increase (greening)
        # Negative change: vegetation decrease (browning, deforestation)
        
        # Additional constraint: only consider areas with some vegetation
        # to avoid false positives in water/urban areas
        vegetation_baseline = np.maximum(before_composite, after_composite)
        vegetation_areas = vegetation_baseline > 0.1  # Minimum NDVI for vegetation
        
        positive_change = valid_mask & vegetation_areas & (change > threshold)
        negative_change = valid_mask & vegetation_areas & (change < -threshold)
        
    elif index_type == 'ndwi':
        # NDWI change detection
        # Positive change: water increase (flooding, new water bodies)
        # Negative change: water decrease (drought, drainage)
        
        positive_change = valid_mask & (change > threshold)
        negative_change = valid_mask & (change < -threshold)
        
    elif index_type == 'ndui':
        # NDUI change detection
        # Positive change: urban development (new construction)
        # Negative change: urban decline (demolition, abandonment)
        
        # Focus on areas with potential for urban development
        # Exclude water areas (negative NDWI) if NDWI data is available
        positive_change = valid_mask & (change > threshold)
        negative_change = valid_mask & (change < -threshold)
        
    elif index_type == 'radiance':
        # Radiance change detection
        # Positive change: brightness increase (new lights, development)
        # Negative change: brightness decrease (lights going out)
        
        # Use relative change for radiance to account for baseline differences
        baseline_radiance = np.maximum(before_composite, 0.1)  # Avoid division by zero
        relative_change = change / baseline_radiance
        
        # Significant relative change (e.g., 50% increase/decrease)
        relative_threshold = 0.5
        positive_change = valid_mask & (relative_change > relative_threshold)
        negative_change = valid_mask & (relative_change < -relative_threshold)
        
        # Also check absolute change for very bright areas
        absolute_positive = valid_mask & (change > threshold)
        absolute_negative = valid_mask & (change < -threshold)
        
        positive_change = positive_change | absolute_positive
        negative_change = negative_change | absolute_negative
        
    else:
        # Generic change detection
        positive_change = valid_mask & (change > threshold)
        negative_change = valid_mask & (change < -threshold)
    
    # Apply change detection results
    change_mask[positive_change] = 1   # Positive change
    change_mask[negative_change] = -1  # Negative change
    
    return change_mask


def fill_temporal_gaps(
    data_stack: ArrayLike,
    method: Literal["linear", "nearest", "spline"] = "linear",
    max_gap: int = 3
) -> ArrayLike:
    """Fill gaps in temporal data using intelligent interpolation.
    
    This function fills temporal gaps in time series data using various
    interpolation methods. It's designed to handle missing observations
    due to clouds, sensor issues, or data processing problems while
    preserving the temporal characteristics of the data.
    
    The function applies gap filling pixel-wise, ensuring that spatial
    patterns are preserved while temporal continuity is restored.
    
    Args:
        data_stack: 3D array (time, height, width) or 4D (time, bands, height, width)
            containing time series with gaps (NaN values).
        method: Interpolation method to use:
            - 'linear': Linear interpolation between valid points
            - 'nearest': Nearest neighbor interpolation
            - 'spline': Cubic spline interpolation (smoother)
        max_gap: Maximum gap size (number of consecutive missing observations)
            to attempt filling. Larger gaps are left unfilled.
            
    Returns:
        Gap-filled data stack with same shape as input.
        Gaps larger than max_gap remain as NaN.
        
    Raises:
        ValueError: If input is not 3D/4D or method is invalid.
        
    Example:
        >>> # Fill small gaps in monthly NDVI time series
        >>> filled_ndvi = fill_temporal_gaps(
        >>>     data_stack=ndvi_monthly,  # (12, 1000, 1000)
        >>>     method='linear',
        >>>     max_gap=2  # Fill gaps up to 2 months
        >>> )
        >>> 
        >>> # Count filled pixels
        >>> original_gaps = np.sum(np.isnan(ndvi_monthly))
        >>> remaining_gaps = np.sum(np.isnan(filled_ndvi))
        >>> filled_count = original_gaps - remaining_gaps
        >>> print(f"Filled {filled_count} gap pixels")
        
    Note:
        This function is valuable for creating more complete time series
        for analysis while avoiding artifacts from excessive interpolation.
    """
    import numpy as np
    from scipy.interpolate import interp1d
    
    # Input validation
    if data_stack.ndim not in [3, 4]:
        raise ValueError(f"Data stack must be 3D or 4D, got {data_stack.ndim}D")
    
    valid_methods = ["linear", "nearest", "spline"]
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}, got {method}")
    
    # Create copy for output
    filled_data = data_stack.copy()
    
    n_times = data_stack.shape[0]
    time_indices = np.arange(n_times)
    
    if data_stack.ndim == 3:
        # Single band data
        height, width = data_stack.shape[1:]
        
        # Process each pixel
        for i in range(height):
            for j in range(width):
                pixel_time_series = data_stack[:, i, j].copy()
                filled_pixel = _fill_pixel_gaps(
                    pixel_time_series, time_indices, method, max_gap
                )
                filled_data[:, i, j] = filled_pixel
    
    else:
        # Multi-band data
        n_bands, height, width = data_stack.shape[1:]
        
        # Process each pixel and band
        for band in range(n_bands):
            for i in range(height):
                for j in range(width):
                    pixel_time_series = data_stack[:, band, i, j].copy()
                    filled_pixel = _fill_pixel_gaps(
                        pixel_time_series, time_indices, method, max_gap
                    )
                    filled_data[:, band, i, j] = filled_pixel
    
    return filled_data


def _fill_pixel_gaps(
    pixel_values: ArrayLike, 
    time_indices: ArrayLike, 
    method: str, 
    max_gap: int
) -> ArrayLike:
    """Fill gaps in a single pixel time series."""
    import numpy as np
    from scipy.interpolate import interp1d
    
    # Find valid (non-NaN) observations
    valid_mask = ~np.isnan(pixel_values)
    
    if np.sum(valid_mask) < 2:
        # Need at least 2 valid points for interpolation
        return pixel_values
    
    valid_times = time_indices[valid_mask]
    valid_values = pixel_values[valid_mask]
    
    # Identify gaps and their sizes
    gap_mask = ~valid_mask
    gap_positions = np.where(gap_mask)[0]
    
    if len(gap_positions) == 0:
        # No gaps to fill
        return pixel_values
    
    # Group consecutive gap positions
    gap_groups = []
    current_group = [gap_positions[0]]
    
    for i in range(1, len(gap_positions)):
        if gap_positions[i] == gap_positions[i-1] + 1:
            current_group.append(gap_positions[i])
        else:
            gap_groups.append(current_group)
            current_group = [gap_positions[i]]
    gap_groups.append(current_group)
    
    # Fill gaps that are within size limit
    filled_values = pixel_values.copy()
    
    for gap_group in gap_groups:
        gap_size = len(gap_group)
        
        if gap_size > max_gap:
            continue  # Skip gaps that are too large
        
        gap_start = gap_group[0]
        gap_end = gap_group[-1]
        
        # Check if gap can be interpolated (not at edges)
        if gap_start == 0 or gap_end == len(pixel_values) - 1:
            # Gap at beginning or end - use nearest neighbor
            if gap_start == 0 and np.any(valid_mask):
                # Fill with first valid value
                first_valid_idx = np.where(valid_mask)[0][0]
                filled_values[gap_group] = pixel_values[first_valid_idx]
            elif gap_end == len(pixel_values) - 1 and np.any(valid_mask):
                # Fill with last valid value
                last_valid_idx = np.where(valid_mask)[0][-1]
                filled_values[gap_group] = pixel_values[last_valid_idx]
            continue
        
        # Interpolate gap
        try:
            if method == "nearest":
                # Nearest neighbor interpolation
                gap_times = time_indices[gap_group]
                
                for gap_time in gap_times:
                    # Find nearest valid time
                    distances = np.abs(valid_times - gap_time)
                    nearest_idx = np.argmin(distances)
                    filled_values[gap_time] = valid_values[nearest_idx]
            
            elif method == "linear":
                # Linear interpolation
                interp_func = interp1d(
                    valid_times, valid_values, 
                    kind='linear', 
                    bounds_error=False,
                    fill_value=np.nan
                )
                
                gap_times = time_indices[gap_group]
                interpolated = interp_func(gap_times)
                
                # Only fill if interpolation was successful
                valid_interp = ~np.isnan(interpolated)
                filled_values[np.array(gap_group)[valid_interp]] = interpolated[valid_interp]
            
            elif method == "spline":
                # Cubic spline interpolation
                if len(valid_values) >= 4:  # Need at least 4 points for cubic
                    interp_func = interp1d(
                        valid_times, valid_values,
                        kind='cubic',
                        bounds_error=False,
                        fill_value=np.nan
                    )
                else:
                    # Fall back to linear if insufficient points
                    interp_func = interp1d(
                        valid_times, valid_values,
                        kind='linear',
                        bounds_error=False,
                        fill_value=np.nan
                    )
                
                gap_times = time_indices[gap_group]
                interpolated = interp_func(gap_times)
                
                # Only fill if interpolation was successful
                valid_interp = ~np.isnan(interpolated)
                filled_values[np.array(gap_group)[valid_interp]] = interpolated[valid_interp]
        
        except Exception:
            # If interpolation fails, skip this gap
            continue
    
    return filled_values


def compute_phenology_metrics(
    ndvi_stack: ArrayLike,
    dates: list[datetime]
) -> dict[str, ArrayLike]:
    """Compute vegetation phenology metrics from NDVI time series.
    
    This function analyzes seasonal vegetation patterns by extracting key
    phenological events from NDVI time series. It identifies the timing
    of vegetation green-up, peak greenness, senescence, and dormancy.
    
    The analysis is particularly useful for:
    - Agricultural monitoring (crop phenology)
    - Forest health assessment
    - Climate change impact studies
    - Ecosystem productivity analysis
    
    Args:
        ndvi_stack: 3D NDVI array (time, height, width) with seasonal observations.
            Should span at least one complete growing season.
            Expected NDVI range: [-1, 1] with vegetation typically [0.2, 0.8].
        dates: List of acquisition dates corresponding to each time slice.
            Must have same length as time dimension. Should be chronologically ordered.
            
    Returns:
        Dictionary containing phenology metric arrays with spatial dimensions:
        - 'greenup_date': Day of year for start of growing season
        - 'peak_date': Day of year for maximum NDVI (peak greenness)
        - 'senescence_date': Day of year for start of senescence
        - 'dormancy_date': Day of year for end of growing season
        - 'growing_season_length': Length of growing season in days
        - 'peak_ndvi': Maximum NDVI value during growing season
        - 'min_ndvi': Minimum NDVI value (baseline/dormant vegetation)
        - 'ndvi_amplitude': Difference between peak and minimum NDVI
        - 'integrated_ndvi': Seasonal integral (proxy for productivity)
        
    Raises:
        ValueError: If input dimensions don't match or insufficient temporal data.
        
    Example:
        >>> # Analyze crop phenology from monthly NDVI
        >>> phenology = compute_phenology_metrics(
        >>>     ndvi_stack=monthly_ndvi,  # (12, 1000, 1000)
        >>>     dates=monthly_dates       # 12 datetime objects
        >>> )
        >>> 
        >>> # Find areas with early green-up
        >>> early_greenup = phenology['greenup_date'] < 90  # Before April 1
        >>> 
        >>> # Calculate growing season length statistics
        >>> mean_season_length = np.nanmean(phenology['growing_season_length'])
        >>> print(f"Average growing season: {mean_season_length:.1f} days")
        
    Note:
        This function requires at least 6 observations spanning a full
        growing season for reliable phenology extraction.
    """
    import numpy as np
    from datetime import datetime
    
    # Input validation
    if ndvi_stack.ndim != 3:
        raise ValueError(f"NDVI stack must be 3D (time, height, width), got {ndvi_stack.ndim}D")
    
    n_times = ndvi_stack.shape[0]
    if len(dates) != n_times:
        raise ValueError(f"Dates length ({len(dates)}) must match time dimension ({n_times})")
    
    if n_times < 6:
        raise ValueError(f"Need at least 6 observations for phenology analysis, got {n_times}")
    
    height, width = ndvi_stack.shape[1:]
    
    # Convert dates to day of year
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    
    # Initialize output arrays
    phenology_metrics = {
        'greenup_date': np.full((height, width), np.nan, dtype=np.float32),
        'peak_date': np.full((height, width), np.nan, dtype=np.float32),
        'senescence_date': np.full((height, width), np.nan, dtype=np.float32),
        'dormancy_date': np.full((height, width), np.nan, dtype=np.float32),
        'growing_season_length': np.full((height, width), np.nan, dtype=np.float32),
        'peak_ndvi': np.full((height, width), np.nan, dtype=np.float32),
        'min_ndvi': np.full((height, width), np.nan, dtype=np.float32),
        'ndvi_amplitude': np.full((height, width), np.nan, dtype=np.float32),
        'integrated_ndvi': np.full((height, width), np.nan, dtype=np.float32)
    }
    
    # Process each pixel
    for i in range(height):
        for j in range(width):
            pixel_ndvi = ndvi_stack[:, i, j]
            
            # Skip pixels with insufficient valid data
            valid_mask = ~np.isnan(pixel_ndvi)
            if np.sum(valid_mask) < 4:
                continue
            
            valid_ndvi = pixel_ndvi[valid_mask]
            valid_doy = day_of_year[valid_mask]
            
            # Skip if NDVI range is too small (likely non-vegetated)
            ndvi_range = np.max(valid_ndvi) - np.min(valid_ndvi)
            if ndvi_range < 0.1:  # Minimum vegetation signal
                continue
            
            try:
                pixel_metrics = _extract_pixel_phenology(valid_ndvi, valid_doy)
                
                # Store results
                for metric_name, value in pixel_metrics.items():
                    if metric_name in phenology_metrics:
                        phenology_metrics[metric_name][i, j] = value
                        
            except Exception:
                # Skip pixels where phenology extraction fails
                continue
    
    return phenology_metrics


def _extract_pixel_phenology(ndvi_values: ArrayLike, day_of_year: ArrayLike) -> dict[str, float]:
    """Extract phenology metrics for a single pixel."""
    import numpy as np
    
    # Sort by day of year to ensure chronological order
    sort_indices = np.argsort(day_of_year)
    sorted_doy = day_of_year[sort_indices]
    sorted_ndvi = ndvi_values[sort_indices]
    
    # Calculate basic statistics
    min_ndvi = float(np.min(sorted_ndvi))
    max_ndvi = float(np.max(sorted_ndvi))
    peak_idx = np.argmax(sorted_ndvi)
    peak_date = float(sorted_doy[peak_idx])
    
    # Calculate amplitude
    amplitude = max_ndvi - min_ndvi
    
    # Calculate integrated NDVI (trapezoidal rule approximation)
    # This serves as a proxy for seasonal productivity
    if len(sorted_doy) > 1:
        integrated_ndvi = float(np.trapz(sorted_ndvi, sorted_doy))
    else:
        integrated_ndvi = float(sorted_ndvi[0])
    
    # Define thresholds for phenological events
    # These are based on common phenology analysis methods
    ndvi_range = max_ndvi - min_ndvi
    
    # Green-up threshold: 20% of the way from minimum to maximum
    greenup_threshold = min_ndvi + 0.2 * ndvi_range
    
    # Senescence threshold: 80% of the way from minimum to maximum
    senescence_threshold = min_ndvi + 0.8 * ndvi_range
    
    # Dormancy threshold: 30% of the way from minimum to maximum
    dormancy_threshold = min_ndvi + 0.3 * ndvi_range
    
    # Find phenological dates
    greenup_date = np.nan
    senescence_date = np.nan
    dormancy_date = np.nan
    
    # Green-up: first crossing above threshold
    greenup_candidates = np.where(sorted_ndvi >= greenup_threshold)[0]
    if len(greenup_candidates) > 0:
        greenup_date = float(sorted_doy[greenup_candidates[0]])
    
    # Senescence: first crossing below threshold after peak
    senescence_candidates = np.where(
        (sorted_ndvi <= senescence_threshold) & 
        (np.arange(len(sorted_ndvi)) > peak_idx)
    )[0]
    if len(senescence_candidates) > 0:
        senescence_date = float(sorted_doy[senescence_candidates[0]])
    
    # Dormancy: first crossing below threshold after senescence
    if not np.isnan(senescence_date):
        senescence_idx = np.where(sorted_doy >= senescence_date)[0][0]
        dormancy_candidates = np.where(
            (sorted_ndvi <= dormancy_threshold) & 
            (np.arange(len(sorted_ndvi)) >= senescence_idx)
        )[0]
        if len(dormancy_candidates) > 0:
            dormancy_date = float(sorted_doy[dormancy_candidates[0]])
    
    # Calculate growing season length
    if not np.isnan(greenup_date) and not np.isnan(dormancy_date):
        growing_season_length = dormancy_date - greenup_date
        if growing_season_length < 0:  # Handle year boundary crossing
            growing_season_length += 365
    else:
        growing_season_length = np.nan
    
    return {
        'greenup_date': greenup_date,
        'peak_date': peak_date,
        'senescence_date': senescence_date,
        'dormancy_date': dormancy_date,
        'growing_season_length': growing_season_length,
        'peak_ndvi': max_ndvi,
        'min_ndvi': min_ndvi,
        'ndvi_amplitude': amplitude,
        'integrated_ndvi': integrated_ndvi
    }