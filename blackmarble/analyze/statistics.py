"""Statistical analysis functions for validation and reporting.

This module provides comprehensive statistical analysis capabilities that are
essential for quality assessment, validation, and scientific reporting throughout
the Black Marble pipeline. These functions generate the quantitative metrics
needed to ensure data quality and document processing results.

Statistical analysis serves multiple purposes: validating processing steps,
comparing results across regions, and providing metadata for data products.
"""

from typing import Any, Literal
import numpy.typing as npt

# Type aliases
ArrayLike = npt.NDArray[Any]
BBox = tuple[float, float, float, float]


def calculate_coverage_statistics(
    data: ArrayLike,
    mask: ArrayLike | None = None,
    region_labels: ArrayLike | None = None,
    compute_regional: bool = True
) -> dict[str, Any]:
    """Calculate comprehensive coverage statistics for data and masks.
    
    This function computes detailed statistics about data coverage, which is
    critical for quality assessment. In the Black Marble pipeline, coverage
    statistics determine whether sufficient valid data exists to produce
    reliable outputs for a given area.
    
    Coverage analysis is particularly important for cloud-prone regions where
    data availability may be limited, affecting the quality of temporal composites.
    
    Args:
        data: Input data array
        mask: Optional validity mask (True = valid pixel)
        region_labels: Optional array of region labels for regional stats
        compute_regional: Whether to compute per-region statistics
        
    Returns:
        Dictionary containing:
        - total_pixels: Total number of pixels
        - valid_pixels: Number of valid (non-masked) pixels
        - valid_percent: Percentage of valid pixels
        - nodata_pixels: Number of nodata pixels
        - nodata_percent: Percentage of nodata pixels
        - coverage_area_km2: Approximate area coverage in km²
        - regional_stats: Per-region statistics if requested
        - coverage_map: Binary coverage map
        - quality_flag: Overall quality assessment
        
    Example:
        >>> # Assess coverage after cloud masking
        >>> stats = calculate_coverage_statistics(
        >>>     landsat_data,
        >>>     mask=~cloud_mask,
        >>>     region_labels=admin_boundaries
        >>> )
        >>> print(f"Valid coverage: {stats['valid_percent']:.1f}%")
        
    Note:
        Coverage below 10% typically triggers warnings or alternative
        processing strategies in the pipeline.
    """
    import numpy as np
    
    # Ensure data is array
    data = np.asarray(data)
    total_pixels = data.size
    
    # Determine valid pixels based on mask and nodata
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != data.shape:
            raise ValueError(f"Mask shape {mask.shape} doesn't match data shape {data.shape}")
        valid_mask = mask.astype(bool)
    else:
        valid_mask = np.ones(data.shape, dtype=bool)
    
    # Handle nodata values (NaN or specific nodata value)
    nodata_mask = np.isnan(data) if np.issubdtype(data.dtype, np.floating) else False
    valid_mask = valid_mask & ~nodata_mask
    
    # Calculate basic statistics
    valid_pixels = int(np.sum(valid_mask))
    nodata_pixels = int(np.sum(nodata_mask))
    valid_percent = (valid_pixels / total_pixels * 100) if total_pixels > 0 else 0.0
    nodata_percent = (nodata_pixels / total_pixels * 100) if total_pixels > 0 else 0.0
    
    # Estimate coverage area (assuming ~1km pixels for rough estimate)
    # Real implementation would use actual pixel resolution
    pixel_area_km2 = 1.0  # Placeholder - should get from metadata
    coverage_area_km2 = valid_pixels * pixel_area_km2
    
    # Create coverage map
    coverage_map = valid_mask.astype(np.uint8)
    
    # Quality assessment based on coverage percentage
    if valid_percent >= 80:
        quality_flag = "excellent"
    elif valid_percent >= 50:
        quality_flag = "good"
    elif valid_percent >= 20:
        quality_flag = "fair"
    elif valid_percent >= 10:
        quality_flag = "poor"
    else:
        quality_flag = "insufficient"
    
    result = {
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "valid_percent": valid_percent,
        "nodata_pixels": nodata_pixels,
        "nodata_percent": nodata_percent,
        "coverage_area_km2": coverage_area_km2,
        "coverage_map": coverage_map,
        "quality_flag": quality_flag
    }
    
    # Compute regional statistics if requested
    if compute_regional and region_labels is not None:
        region_labels = np.asarray(region_labels)
        if region_labels.shape != data.shape:
            raise ValueError(f"Region labels shape {region_labels.shape} doesn't match data shape {data.shape}")
        
        unique_regions = np.unique(region_labels[region_labels != 0])  # Exclude 0 (background)
        regional_stats = {}
        
        for region_id in unique_regions:
            region_mask = region_labels == region_id
            region_valid = valid_mask & region_mask
            region_total = int(np.sum(region_mask))
            region_valid_count = int(np.sum(region_valid))
            
            regional_stats[int(region_id)] = {
                "total_pixels": region_total,
                "valid_pixels": region_valid_count,
                "valid_percent": (region_valid_count / region_total * 100) if region_total > 0 else 0.0,
                "coverage_area_km2": region_valid_count * pixel_area_km2
            }
        
        result["regional_stats"] = regional_stats
    
    return result


def compute_zonal_statistics(
    data: ArrayLike,
    zones: ArrayLike,
    stats_list: list[str] | None = None,
    percentiles: list[float] | None = None,
    weighted: bool = False,
    weights: ArrayLike | None = None
) -> dict[int, dict[str, float]]:
    """Compute statistics for each zone in a labeled array.
    
    This function calculates statistics within spatial zones, enabling
    analysis by administrative boundaries, land cover types, or other
    geographic divisions. It's essential for generating reports that
    summarize results by meaningful geographic units.
    
    Zonal statistics are used to create summary products like average
    nighttime lights by city or vegetation indices by agricultural district.
    
    Args:
        data: Data array to analyze
        zones: Zone labels array (integer labels)
        stats_list: List of statistics to compute:
            ['mean', 'std', 'min', 'max', 'sum', 'count', 'median']
            If None, computes all standard statistics
        percentiles: Additional percentiles to compute (e.g., [25, 75])
        weighted: Whether to compute weighted statistics
        weights: Weight array for weighted statistics
        
    Returns:
        Dictionary mapping zone_id to statistics dictionary
        
    Example:
        >>> # Calculate nighttime lights by district
        >>> district_stats = compute_zonal_statistics(
        >>>     ntl_data,
        >>>     zones=district_labels,
        >>>     stats_list=['mean', 'sum', 'max'],
        >>>     percentiles=[10, 90]
        >>> )
        >>> for district_id, stats in district_stats.items():
        >>>     print(f"District {district_id}: {stats['mean']:.2f}")
    """
    import numpy as np
    
    # Ensure arrays
    data = np.asarray(data)
    zones = np.asarray(zones)
    
    if data.shape != zones.shape:
        raise ValueError(f"Data shape {data.shape} doesn't match zones shape {zones.shape}")
    
    # Default statistics list
    if stats_list is None:
        stats_list = ['mean', 'std', 'min', 'max', 'sum', 'count', 'median']
    
    # Handle weights
    if weighted and weights is not None:
        weights = np.asarray(weights)
        if weights.shape != data.shape:
            raise ValueError(f"Weights shape {weights.shape} doesn't match data shape {data.shape}")
    
    # Get unique zones (excluding 0 which typically means no zone)
    unique_zones = np.unique(zones)
    unique_zones = unique_zones[unique_zones != 0]
    
    results = {}
    
    for zone_id in unique_zones:
        # Get mask for current zone
        zone_mask = zones == zone_id
        zone_data = data[zone_mask]
        
        # Skip if no valid data
        valid_data = zone_data[~np.isnan(zone_data)] if np.issubdtype(zone_data.dtype, np.floating) else zone_data
        if len(valid_data) == 0:
            continue
        
        zone_stats = {}
        
        # Compute requested statistics
        if 'count' in stats_list:
            zone_stats['count'] = float(len(valid_data))
        
        if weighted and weights is not None:
            zone_weights = weights[zone_mask]
            valid_weights = zone_weights[~np.isnan(zone_data)] if np.issubdtype(zone_data.dtype, np.floating) else zone_weights
            
            if 'mean' in stats_list:
                zone_stats['mean'] = float(np.average(valid_data, weights=valid_weights))
            if 'sum' in stats_list:
                zone_stats['sum'] = float(np.sum(valid_data * valid_weights))
        else:
            if 'mean' in stats_list:
                zone_stats['mean'] = float(np.mean(valid_data))
            if 'sum' in stats_list:
                zone_stats['sum'] = float(np.sum(valid_data))
        
        if 'std' in stats_list:
            zone_stats['std'] = float(np.std(valid_data))
        if 'min' in stats_list:
            zone_stats['min'] = float(np.min(valid_data))
        if 'max' in stats_list:
            zone_stats['max'] = float(np.max(valid_data))
        if 'median' in stats_list:
            zone_stats['median'] = float(np.median(valid_data))
        
        # Compute percentiles if requested
        if percentiles:
            for p in percentiles:
                zone_stats[f'p{int(p)}'] = float(np.percentile(valid_data, p))
        
        results[int(zone_id)] = zone_stats
    
    return results


def generate_histogram_analysis(
    data: ArrayLike,
    bins: int | ArrayLike = 256,
    range: tuple[float, float] | None = None,
    mask: ArrayLike | None = None,
    compute_metrics: bool = True
) -> dict[str, Any]:
    """Generate comprehensive histogram analysis of data distribution.
    
    This function produces detailed histogram analysis that reveals data
    distribution characteristics. Histogram analysis is crucial for
    identifying bimodal distributions, outliers, and appropriate scaling
    parameters for visualization.
    
    In the Black Marble pipeline, histograms guide contrast enhancement
    parameters and help identify processing artifacts.
    
    Args:
        data: Input data array
        bins: Number of bins or bin edges array
        range: Range for histogram (min, max). Auto-computed if None
        mask: Optional mask for valid pixels only
        compute_metrics: Whether to compute distribution metrics
        
    Returns:
        Dictionary containing:
        - counts: Histogram counts
        - edges: Bin edges
        - centers: Bin centers
        - cumulative: Cumulative distribution
        - percentiles: Key percentiles (1, 5, 25, 50, 75, 95, 99)
        - mode: Modal value(s)
        - skewness: Distribution skewness
        - kurtosis: Distribution kurtosis
        - entropy: Distribution entropy
        - visualization_range: Suggested range for display
        
    Example:
        >>> # Analyze NDVI distribution
        >>> hist = generate_histogram_analysis(
        >>>     ndvi_data,
        >>>     bins=100,
        >>>     range=(-1, 1)
        >>> )
        >>> plt.bar(hist['centers'], hist['counts'])
    """
    import numpy as np
    from scipy import stats
    
    # Prepare data
    data = np.asarray(data)
    
    # Apply mask if provided
    if mask is not None:
        mask = np.asarray(mask).astype(bool)
        valid_data = data[mask]
    else:
        valid_data = data.ravel()
    
    # Remove NaN values
    valid_data = valid_data[~np.isnan(valid_data)] if np.issubdtype(valid_data.dtype, np.floating) else valid_data
    
    if len(valid_data) == 0:
        raise ValueError("No valid data points for histogram analysis")
    
    # Compute histogram
    counts, edges = np.histogram(valid_data, bins=bins, range=range)
    centers = (edges[:-1] + edges[1:]) / 2
    
    # Compute cumulative distribution
    cumulative = np.cumsum(counts).astype(float)
    cumulative /= cumulative[-1]  # Normalize to [0, 1]
    
    result = {
        "counts": counts,
        "edges": edges,
        "centers": centers,
        "cumulative": cumulative
    }
    
    if compute_metrics:
        # Compute percentiles
        percentile_values = [1, 5, 25, 50, 75, 95, 99]
        percentiles = {}
        for p in percentile_values:
            percentiles[f"p{p}"] = float(np.percentile(valid_data, p))
        result["percentiles"] = percentiles
        
        # Find mode(s) - bin(s) with highest count
        mode_indices = np.where(counts == np.max(counts))[0]
        result["mode"] = centers[mode_indices].tolist()
        
        # Compute distribution metrics
        result["skewness"] = float(stats.skew(valid_data))
        result["kurtosis"] = float(stats.kurtosis(valid_data))
        
        # Compute entropy
        # Normalize counts for probability distribution
        prob = counts.astype(float)
        prob = prob[prob > 0]  # Remove zero bins
        prob /= prob.sum()
        result["entropy"] = float(-np.sum(prob * np.log(prob)))
        
        # Suggest visualization range (clip outliers)
        # Use 2nd and 98th percentiles for robust range
        viz_min = percentiles["p2"] if "p2" in percentiles else percentiles["p5"]
        viz_max = percentiles["p98"] if "p98" in percentiles else percentiles["p95"]
        result["visualization_range"] = (viz_min, viz_max)
        
        # Additional statistics
        result["mean"] = float(np.mean(valid_data))
        result["std"] = float(np.std(valid_data))
        result["data_range"] = (float(np.min(valid_data)), float(np.max(valid_data)))
    
    return result


def calculate_temporal_statistics(
    data_stack: ArrayLike,
    dates: list[Any] | None = None,
    compute_trend: bool = True,
    detrend: bool = False
) -> dict[str, ArrayLike]:
    """Calculate temporal statistics across a time series stack.
    
    This function computes statistics that characterize temporal behavior
    of pixels across multiple observations. These statistics are essential
    for identifying stable features, detecting changes, and assessing
    temporal data quality.
    
    Temporal statistics guide compositing decisions and help identify
    pixels with high temporal variability that may need special handling.
    
    Args:
        data_stack: 3D array (time, height, width)
        dates: Optional list of dates for trend analysis
        compute_trend: Whether to compute temporal trends
        detrend: Whether to remove trends from statistics
        
    Returns:
        Dictionary of statistical arrays:
        - temporal_mean: Mean value over time
        - temporal_std: Standard deviation over time
        - temporal_min: Minimum value over time
        - temporal_max: Maximum value over time
        - temporal_range: Range (max - min) over time
        - coefficient_variation: CV (std/mean) over time
        - trend_slope: Linear trend slope (if compute_trend=True)
        - trend_r2: Trend R-squared value
        - detrended_std: Std after detrending (if detrend=True)
        
    Note:
        High coefficient of variation indicates temporal instability,
        which may indicate land cover change or data quality issues.
    """
    import numpy as np
    from scipy import stats as scipy_stats
    
    # Ensure 3D array
    data_stack = np.asarray(data_stack)
    if data_stack.ndim != 3:
        raise ValueError(f"Expected 3D array (time, height, width), got shape {data_stack.shape}")
    
    n_times, height, width = data_stack.shape
    
    # Basic temporal statistics
    # Use nanmean, nanstd etc to handle missing data gracefully
    temporal_mean = np.nanmean(data_stack, axis=0)
    temporal_std = np.nanstd(data_stack, axis=0)
    temporal_min = np.nanmin(data_stack, axis=0)
    temporal_max = np.nanmax(data_stack, axis=0)
    temporal_range = temporal_max - temporal_min
    
    # Coefficient of variation (normalized variability)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        coefficient_variation = temporal_std / np.abs(temporal_mean)
        coefficient_variation[temporal_mean == 0] = np.nan
    
    result = {
        "temporal_mean": temporal_mean,
        "temporal_std": temporal_std,
        "temporal_min": temporal_min,
        "temporal_max": temporal_max,
        "temporal_range": temporal_range,
        "coefficient_variation": coefficient_variation
    }
    
    # Compute trends if requested
    if compute_trend:
        # Create time index
        if dates is not None and len(dates) == n_times:
            # Convert dates to numeric days from first date
            from datetime import datetime
            if isinstance(dates[0], datetime):
                time_numeric = np.array([(d - dates[0]).days for d in dates], dtype=float)
            else:
                # Assume already numeric
                time_numeric = np.array(dates, dtype=float)
        else:
            # Use simple time index
            time_numeric = np.arange(n_times, dtype=float)
        
        # Initialize trend arrays
        trend_slope = np.full((height, width), np.nan)
        trend_r2 = np.full((height, width), np.nan)
        
        # Compute pixel-wise trends
        for i in range(height):
            for j in range(width):
                pixel_series = data_stack[:, i, j]
                
                # Skip if too many NaN values
                valid_mask = ~np.isnan(pixel_series)
                if np.sum(valid_mask) < 3:  # Need at least 3 points
                    continue
                
                # Fit linear trend
                try:
                    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                        time_numeric[valid_mask], pixel_series[valid_mask]
                    )
                    trend_slope[i, j] = slope
                    trend_r2[i, j] = r_value ** 2
                except:
                    # Regression failed, leave as NaN
                    pass
        
        result["trend_slope"] = trend_slope
        result["trend_r2"] = trend_r2
        
        # Compute detrended statistics if requested
        if detrend:
            detrended_stack = np.full_like(data_stack, np.nan)
            
            for i in range(height):
                for j in range(width):
                    if not np.isnan(trend_slope[i, j]):
                        # Remove linear trend
                        pixel_series = data_stack[:, i, j]
                        trend_line = trend_slope[i, j] * time_numeric + \
                                    (temporal_mean[i, j] - trend_slope[i, j] * np.mean(time_numeric))
                        detrended_stack[:, i, j] = pixel_series - trend_line
            
            # Compute detrended std
            result["detrended_std"] = np.nanstd(detrended_stack, axis=0)
    
    return result


def compute_comparison_metrics(
    reference: ArrayLike,
    comparison: ArrayLike,
    metrics: list[str] | None = None,
    mask: ArrayLike | None = None
) -> dict[str, float]:
    """Compute comparison metrics between reference and comparison arrays.
    
    This function calculates various metrics for comparing two datasets,
    essential for validation, quality assessment, and change detection.
    It's used to compare outputs with reference data or to assess
    agreement between different processing methods.
    
    Args:
        reference: Reference/truth data array
        comparison: Data array to compare
        metrics: List of metrics to compute:
            ['rmse', 'mae', 'bias', 'correlation', 'r2', 'mape']
            If None, computes all metrics
        mask: Optional mask for valid pixels only
        
    Returns:
        Dictionary of computed metrics:
        - rmse: Root Mean Square Error
        - mae: Mean Absolute Error
        - bias: Mean bias (comparison - reference)
        - correlation: Pearson correlation coefficient
        - r2: Coefficient of determination
        - mape: Mean Absolute Percentage Error
        - max_error: Maximum absolute error
        - percentile_errors: Errors at key percentiles
        
    Example:
        >>> # Compare two processing methods
        >>> metrics = compute_comparison_metrics(
        >>>     reference=method1_output,
        >>>     comparison=method2_output,
        >>>     metrics=['rmse', 'correlation']
        >>> )
        >>> print(f"RMSE: {metrics['rmse']:.4f}")
    """
    import numpy as np
    from scipy import stats
    
    # Ensure arrays
    reference = np.asarray(reference)
    comparison = np.asarray(comparison)
    
    if reference.shape != comparison.shape:
        raise ValueError(f"Reference shape {reference.shape} doesn't match comparison shape {comparison.shape}")
    
    # Default metrics
    if metrics is None:
        metrics = ['rmse', 'mae', 'bias', 'correlation', 'r2', 'mape']
    
    # Apply mask and flatten
    if mask is not None:
        mask = np.asarray(mask).astype(bool)
        if mask.shape != reference.shape:
            raise ValueError(f"Mask shape {mask.shape} doesn't match data shape {reference.shape}")
        ref_flat = reference[mask]
        comp_flat = comparison[mask]
    else:
        ref_flat = reference.ravel()
        comp_flat = comparison.ravel()
    
    # Remove NaN pairs
    valid_mask = ~(np.isnan(ref_flat) | np.isnan(comp_flat))
    ref_valid = ref_flat[valid_mask]
    comp_valid = comp_flat[valid_mask]
    
    if len(ref_valid) == 0:
        raise ValueError("No valid data pairs for comparison")
    
    # Calculate errors
    errors = comp_valid - ref_valid
    abs_errors = np.abs(errors)
    
    result = {}
    
    # Basic error metrics
    if 'rmse' in metrics:
        result['rmse'] = float(np.sqrt(np.mean(errors ** 2)))
    
    if 'mae' in metrics:
        result['mae'] = float(np.mean(abs_errors))
    
    if 'bias' in metrics:
        result['bias'] = float(np.mean(errors))
    
    # Correlation metrics
    if 'correlation' in metrics:
        result['correlation'] = float(np.corrcoef(ref_valid, comp_valid)[0, 1])
    
    if 'r2' in metrics:
        # Coefficient of determination
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((ref_valid - np.mean(ref_valid)) ** 2)
        result['r2'] = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    if 'mape' in metrics:
        # Mean Absolute Percentage Error
        # Avoid division by zero
        nonzero_mask = ref_valid != 0
        if np.any(nonzero_mask):
            percentage_errors = np.abs(errors[nonzero_mask] / ref_valid[nonzero_mask]) * 100
            result['mape'] = float(np.mean(percentage_errors))
        else:
            result['mape'] = np.nan
    
    # Additional error statistics
    result['max_error'] = float(np.max(abs_errors))
    result['percentile_errors'] = {
        'p25': float(np.percentile(abs_errors, 25)),
        'p50': float(np.percentile(abs_errors, 50)),
        'p75': float(np.percentile(abs_errors, 75)),
        'p90': float(np.percentile(abs_errors, 90)),
        'p95': float(np.percentile(abs_errors, 95))
    }
    
    # Count of samples used
    result['n_samples'] = len(ref_valid)
    
    return result


def analyze_spatial_autocorrelation(
    data: ArrayLike,
    lag_distances: list[float] | None = None,
    method: Literal["moran", "geary", "variogram"] = "moran"
) -> dict[str, Any]:
    """Analyze spatial autocorrelation in raster data.
    
    This function quantifies spatial patterns and dependencies in data,
    which is important for understanding data quality and identifying
    processing artifacts. High spatial autocorrelation indicates smooth,
    realistic patterns, while low values may indicate noise or artifacts.
    
    Args:
        data: Input data array
        lag_distances: Distances for lag analysis
        method: Autocorrelation method:
            - 'moran': Moran's I statistic
            - 'geary': Geary's C statistic
            - 'variogram': Empirical variogram
            
    Returns:
        Dictionary containing:
        - global_statistic: Global autocorrelation value
        - local_statistics: Local autocorrelation map
        - significance: Statistical significance
        - lag_analysis: Autocorrelation by distance
        - clusters: Identified spatial clusters
        
    Note:
        Spatial autocorrelation analysis helps identify edge artifacts
        and validate the realism of processed outputs.
    """
    import numpy as np
    from scipy import ndimage
    from scipy.spatial.distance import cdist
    
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")
    
    height, width = data.shape
    
    # Handle NaN values
    valid_mask = ~np.isnan(data)
    data_mean = np.nanmean(data)
    
    result = {}
    
    if method == "moran":
        # Moran's I: measures similarity of neighboring values
        # I = N/W * sum(w_ij * (x_i - mean) * (x_j - mean)) / sum((x_i - mean)^2)
        
        # Create spatial weights matrix (queen contiguity)
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=float)
        
        # Compute local deviations
        deviations = data - data_mean
        deviations[~valid_mask] = 0
        
        # Compute spatially lagged deviations
        lagged = ndimage.convolve(deviations, kernel, mode='constant')
        neighbor_counts = ndimage.convolve(valid_mask.astype(float), kernel, mode='constant')
        
        # Avoid division by zero
        neighbor_counts[neighbor_counts == 0] = 1
        lagged /= neighbor_counts
        
        # Local Moran's I
        local_moran = deviations * lagged
        
        # Global Moran's I
        numerator = np.nansum(local_moran[valid_mask])
        denominator = np.nansum(deviations[valid_mask] ** 2)
        n_valid = np.sum(valid_mask)
        
        if denominator > 0 and n_valid > 0:
            global_moran = (n_valid / np.sum(neighbor_counts[valid_mask])) * (numerator / denominator)
        else:
            global_moran = np.nan
        
        result["global_statistic"] = float(global_moran)
        result["local_statistics"] = local_moran
        
        # Expected value under null hypothesis
        expected_i = -1.0 / (n_valid - 1) if n_valid > 1 else 0
        result["expected_value"] = expected_i
        
        # Identify clusters (simplified - real implementation would use significance testing)
        clusters = np.zeros_like(data, dtype=int)
        # High-High clusters
        clusters[(deviations > 0) & (lagged > 0)] = 1
        # Low-Low clusters
        clusters[(deviations < 0) & (lagged < 0)] = 2
        # High-Low outliers
        clusters[(deviations > 0) & (lagged < 0)] = 3
        # Low-High outliers
        clusters[(deviations < 0) & (lagged > 0)] = 4
        
        result["clusters"] = clusters
        
    elif method == "geary":
        # Geary's C: measures dissimilarity of neighboring values
        # C = (N-1)/(2W) * sum(w_ij * (x_i - x_j)^2) / sum((x_i - mean)^2)
        
        # Simplified implementation using local differences
        diff_right = np.zeros_like(data)
        diff_down = np.zeros_like(data)
        
        diff_right[:, :-1] = (data[:, 1:] - data[:, :-1]) ** 2
        diff_down[:-1, :] = (data[1:, :] - data[:-1, :]) ** 2
        
        # Handle NaN values
        diff_right[~valid_mask] = np.nan
        diff_down[~valid_mask] = np.nan
        
        # Global Geary's C
        numerator = np.nansum(diff_right) + np.nansum(diff_down)
        denominator = np.nansum((data[valid_mask] - data_mean) ** 2)
        n_pairs = np.sum(~np.isnan(diff_right)) + np.sum(~np.isnan(diff_down))
        
        if denominator > 0 and n_pairs > 0:
            global_geary = (n_valid - 1) * numerator / (2 * n_pairs * denominator)
        else:
            global_geary = np.nan
        
        result["global_statistic"] = float(global_geary)
        result["expected_value"] = 1.0  # Expected value for Geary's C
        
    elif method == "variogram":
        # Empirical variogram: variance as function of distance
        if lag_distances is None:
            # Default lag distances
            max_dist = np.sqrt(height**2 + width**2) / 4
            lag_distances = np.linspace(1, max_dist, 10)
        
        # Sample points for efficiency (use all for small arrays)
        n_points = min(1000, np.sum(valid_mask))
        if n_points < np.sum(valid_mask):
            indices = np.random.choice(np.sum(valid_mask), n_points, replace=False)
            y_coords, x_coords = np.where(valid_mask)
            sample_y = y_coords[indices]
            sample_x = x_coords[indices]
            sample_values = data[sample_y, sample_x]
        else:
            sample_y, sample_x = np.where(valid_mask)
            sample_values = data[valid_mask]
        
        # Compute pairwise distances
        coords = np.column_stack([sample_y, sample_x])
        distances = cdist(coords, coords)
        
        # Compute semivariance for each lag
        lag_semivariance = []
        lag_counts = []
        
        for i, lag in enumerate(lag_distances):
            # Find pairs within lag distance
            if i == 0:
                mask = distances <= lag
            else:
                mask = (distances > lag_distances[i-1]) & (distances <= lag)
            
            # Compute semivariance
            if np.any(mask):
                i_idx, j_idx = np.where(mask)
                # Only use upper triangle to avoid double counting
                valid_pairs = i_idx < j_idx
                i_idx = i_idx[valid_pairs]
                j_idx = j_idx[valid_pairs]
                
                if len(i_idx) > 0:
                    squared_diffs = (sample_values[i_idx] - sample_values[j_idx]) ** 2
                    semivar = np.mean(squared_diffs) / 2
                    lag_semivariance.append(float(semivar))
                    lag_counts.append(len(i_idx))
                else:
                    lag_semivariance.append(np.nan)
                    lag_counts.append(0)
            else:
                lag_semivariance.append(np.nan)
                lag_counts.append(0)
        
        result["lag_distances"] = lag_distances.tolist()
        result["lag_semivariance"] = lag_semivariance
        result["lag_counts"] = lag_counts
        
        # Estimate range (distance at which semivariance plateaus)
        # Simplified - just find where semivariance reaches 95% of maximum
        max_semivar = np.nanmax(lag_semivariance)
        if not np.isnan(max_semivar):
            threshold = 0.95 * max_semivar
            range_idx = np.where(np.array(lag_semivariance) >= threshold)[0]
            if len(range_idx) > 0:
                result["range"] = float(lag_distances[range_idx[0]])
            else:
                result["range"] = float(lag_distances[-1])
        else:
            result["range"] = np.nan
    
    return result


def generate_quality_metrics_summary(
    data: ArrayLike,
    mask: ArrayLike | None = None,
    reference: ArrayLike | None = None,
    comprehensive: bool = True
) -> dict[str, Any]:
    """Generate comprehensive quality metrics summary.
    
    This function produces a complete quality assessment report combining
    multiple statistical analyses. It serves as the primary quality
    documentation for data products and processing validation.
    
    Args:
        data: Data array to assess
        mask: Optional quality mask
        reference: Optional reference data for comparison
        comprehensive: Whether to include all available metrics
        
    Returns:
        Comprehensive metrics dictionary including:
        - basic_stats: Mean, std, min, max, etc.
        - distribution_metrics: Histogram analysis results
        - spatial_metrics: Spatial autocorrelation results
        - quality_scores: Overall quality assessment scores
        - comparison_metrics: Metrics vs reference (if provided)
        - recommendations: Quality-based recommendations
        
    Note:
        This summary is typically included in product metadata
        and quality assessment reports.
    """
    import numpy as np
    
    data = np.asarray(data)
    
    # Apply mask if provided
    if mask is not None:
        mask = np.asarray(mask).astype(bool)
        valid_data = data[mask]
    else:
        valid_data = data.ravel()
    
    # Remove NaN values
    valid_data = valid_data[~np.isnan(valid_data)]
    
    result = {}
    
    # Basic statistics
    if len(valid_data) > 0:
        basic_stats = {
            "mean": float(np.mean(valid_data)),
            "std": float(np.std(valid_data)),
            "min": float(np.min(valid_data)),
            "max": float(np.max(valid_data)),
            "median": float(np.median(valid_data)),
            "n_valid": len(valid_data),
            "n_total": data.size,
            "valid_percent": float(len(valid_data) / data.size * 100)
        }
    else:
        basic_stats = {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "n_valid": 0,
            "n_total": data.size,
            "valid_percent": 0.0
        }
    result["basic_stats"] = basic_stats
    
    if comprehensive and len(valid_data) > 0:
        # Distribution metrics
        try:
            distribution = generate_histogram_analysis(
                data,
                bins=100,
                mask=mask,
                compute_metrics=True
            )
            result["distribution_metrics"] = {
                "skewness": distribution.get("skewness", np.nan),
                "kurtosis": distribution.get("kurtosis", np.nan),
                "entropy": distribution.get("entropy", np.nan),
                "percentiles": distribution.get("percentiles", {}),
                "visualization_range": distribution.get("visualization_range", (np.nan, np.nan))
            }
        except:
            result["distribution_metrics"] = {"error": "Failed to compute distribution metrics"}
        
        # Spatial metrics (only for 2D data)
        if data.ndim == 2:
            try:
                spatial = analyze_spatial_autocorrelation(data, method="moran")
                result["spatial_metrics"] = {
                    "moran_i": spatial.get("global_statistic", np.nan),
                    "expected_i": spatial.get("expected_value", np.nan),
                    "spatial_pattern": "clustered" if spatial.get("global_statistic", 0) > 0 else "dispersed"
                }
            except:
                result["spatial_metrics"] = {"error": "Failed to compute spatial metrics"}
        
        # Comparison metrics if reference provided
        if reference is not None:
            try:
                comparison = compute_comparison_metrics(
                    reference=reference,
                    comparison=data,
                    mask=mask
                )
                result["comparison_metrics"] = comparison
            except:
                result["comparison_metrics"] = {"error": "Failed to compute comparison metrics"}
    
    # Quality scores and assessment
    quality_scores = {}
    
    # Data completeness score
    completeness = basic_stats["valid_percent"] / 100
    quality_scores["completeness"] = completeness
    
    # Distribution quality (penalize extreme skewness/kurtosis)
    if "distribution_metrics" in result and "skewness" in result["distribution_metrics"]:
        skew = abs(result["distribution_metrics"]["skewness"])
        kurt = abs(result["distribution_metrics"]["kurtosis"])
        # Normal distribution has skewness=0, kurtosis=0
        distribution_score = 1.0 / (1.0 + 0.1 * skew + 0.05 * kurt)
        quality_scores["distribution"] = distribution_score
    
    # Spatial quality (higher autocorrelation = smoother, more realistic)
    if "spatial_metrics" in result and "moran_i" in result["spatial_metrics"]:
        moran = result["spatial_metrics"]["moran_i"]
        if not np.isnan(moran):
            # Moran's I typically ranges from -1 to 1
            spatial_score = (moran + 1) / 2  # Normalize to [0, 1]
            quality_scores["spatial_coherence"] = spatial_score
    
    # Overall quality score (weighted average)
    if quality_scores:
        weights = {"completeness": 0.4, "distribution": 0.3, "spatial_coherence": 0.3}
        overall_score = 0.0
        total_weight = 0.0
        
        for metric, score in quality_scores.items():
            if metric in weights and not np.isnan(score):
                overall_score += weights[metric] * score
                total_weight += weights[metric]
        
        if total_weight > 0:
            quality_scores["overall"] = overall_score / total_weight
        else:
            quality_scores["overall"] = np.nan
    
    result["quality_scores"] = quality_scores
    
    # Generate recommendations
    recommendations = []
    
    if completeness < 0.5:
        recommendations.append("Low data completeness - consider alternative data sources or relaxing quality criteria")
    
    if "distribution_metrics" in result and "skewness" in result["distribution_metrics"]:
        if abs(result["distribution_metrics"]["skewness"]) > 2:
            recommendations.append("Highly skewed distribution - consider log transformation or robust statistics")
    
    if "spatial_metrics" in result and "moran_i" in result["spatial_metrics"]:
        if result["spatial_metrics"]["moran_i"] < 0:
            recommendations.append("Negative spatial autocorrelation detected - check for processing artifacts")
    
    if "comparison_metrics" in result and "rmse" in result["comparison_metrics"]:
        if result["comparison_metrics"]["correlation"] < 0.7:
            recommendations.append("Low correlation with reference - review processing parameters")
    
    if not recommendations:
        recommendations.append("Data quality appears satisfactory")
    
    result["recommendations"] = recommendations
    
    return result


def calculate_uncertainty_metrics(
    data: ArrayLike,
    observation_count: ArrayLike | None = None,
    quality_scores: ArrayLike | None = None
) -> dict[str, ArrayLike]:
    """Calculate uncertainty metrics for processed data.
    
    This function estimates uncertainty in derived products based on
    input data characteristics and processing quality. Uncertainty
    quantification is essential for scientific applications and
    helps users understand data limitations.
    
    Args:
        data: Processed data array
        observation_count: Number of observations per pixel
        quality_scores: Quality scores per pixel
        
    Returns:
        Dictionary containing:
        - uncertainty_estimate: Per-pixel uncertainty values
        - confidence_interval_lower: Lower confidence bound
        - confidence_interval_upper: Upper confidence bound
        - uncertainty_sources: Breakdown by source
        
    Note:
        Uncertainty metrics are crucial for scientific analysis
        and should be propagated through processing chains.
    """
    import numpy as np
    
    data = np.asarray(data)
    
    # Initialize uncertainty components
    measurement_uncertainty = np.full_like(data, np.nan, dtype=float)
    sampling_uncertainty = np.full_like(data, np.nan, dtype=float)
    processing_uncertainty = np.full_like(data, np.nan, dtype=float)
    
    # Base measurement uncertainty (5% of value as default)
    # In real implementation, this would come from sensor specifications
    valid_mask = ~np.isnan(data)
    measurement_uncertainty[valid_mask] = np.abs(data[valid_mask]) * 0.05
    
    # Sampling uncertainty based on observation count
    if observation_count is not None:
        observation_count = np.asarray(observation_count)
        if observation_count.shape != data.shape:
            raise ValueError(f"Observation count shape {observation_count.shape} doesn't match data shape {data.shape}")
        
        # Higher uncertainty with fewer observations
        # Using standard error approximation: sigma / sqrt(n)
        base_sigma = np.nanstd(data) if np.any(valid_mask) else 1.0
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            sampling_uncertainty = base_sigma / np.sqrt(observation_count)
            sampling_uncertainty[observation_count == 0] = np.nan
    else:
        # Assume moderate sampling uncertainty if no count provided
        sampling_uncertainty[valid_mask] = np.nanstd(data[valid_mask]) * 0.3
    
    # Processing uncertainty based on quality scores
    if quality_scores is not None:
        quality_scores = np.asarray(quality_scores)
        if quality_scores.shape != data.shape:
            raise ValueError(f"Quality scores shape {quality_scores.shape} doesn't match data shape {data.shape}")
        
        # Lower quality = higher uncertainty
        # Assuming quality scores are normalized [0, 1]
        quality_factor = 1.0 - quality_scores
        processing_uncertainty = quality_factor * np.nanstd(data) * 0.2
    else:
        # Assume low processing uncertainty if no quality info
        processing_uncertainty[valid_mask] = np.nanstd(data[valid_mask]) * 0.1
    
    # Combine uncertainty sources (root sum of squares)
    # This assumes independent error sources
    total_uncertainty = np.sqrt(
        measurement_uncertainty**2 + 
        sampling_uncertainty**2 + 
        processing_uncertainty**2
    )
    
    # Calculate confidence intervals (assuming normal distribution)
    # 95% confidence interval = ±1.96 * uncertainty
    z_score_95 = 1.96
    confidence_interval_lower = data - z_score_95 * total_uncertainty
    confidence_interval_upper = data + z_score_95 * total_uncertainty
    
    # Create uncertainty breakdown
    uncertainty_sources = {
        "measurement": measurement_uncertainty,
        "sampling": sampling_uncertainty,
        "processing": processing_uncertainty
    }
    
    # Calculate relative contributions
    total_var = total_uncertainty**2
    relative_contributions = {}
    for source, unc in uncertainty_sources.items():
        with np.errstate(divide='ignore', invalid='ignore'):
            contrib = (unc**2 / total_var) * 100  # Percentage
            contrib[total_var == 0] = np.nan
        relative_contributions[f"{source}_percent"] = contrib
    
    return {
        "uncertainty_estimate": total_uncertainty,
        "confidence_interval_lower": confidence_interval_lower,
        "confidence_interval_upper": confidence_interval_upper,
        "uncertainty_sources": uncertainty_sources,
        "relative_contributions": relative_contributions,
        "confidence_level": 0.95
    }