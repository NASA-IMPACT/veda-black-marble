"""Quality assessment and composite quality mask generation.

This module provides comprehensive quality assessment functions that are critical
for ensuring data integrity throughout the Black Marble pipeline. Quality assessment
occurs at multiple stages and influences temporal compositing, gap filling, and
final product validation.

The quality system serves as the foundation for identifying valid pixels,
detecting processing artifacts, and generating confidence metrics that guide
downstream analysis decisions.
"""

from typing import Any, Literal
import numpy.typing as npt
from datetime import datetime

# Type aliases
ArrayLike = npt.NDArray[Any]
MaskArray = npt.NDArray[Any]  # Boolean or float mask


def assess_data_quality(
    data: ArrayLike,
    quality_flags: ArrayLike | None = None,
    valid_range: tuple[float, float] | None = None,
    nodata_value: float | None = None,
    check_outliers: bool = True,
    outlier_threshold: float = 3.0,
) -> dict[str, Any]:
    """Perform comprehensive quality assessment on input data.

    This function evaluates data quality across multiple dimensions including
    valid data coverage, value distributions, and potential artifacts. It's
    essential for identifying problematic data early in the pipeline before
    expensive processing operations.

    The assessment results guide decisions about whether to proceed with
    processing, which pixels to include in composites, and what quality
    warnings to propagate to users.

    Args:
        data: Input data array to assess. Can be 2D or 3D.
            For 3D arrays, assessment is performed on flattened data.
        quality_flags: Optional quality flag array from data provider.
            Should have same spatial dimensions as data.
            Non-zero values indicate quality issues.
        valid_range: Expected valid range for data values (min, max).
            Values outside this range are flagged as invalid.
        nodata_value: Value representing no data/fill value.
            If None, uses np.nan for floating point or 0 for integer data.
        check_outliers: Whether to identify statistical outliers using
            z-score analysis. Outliers may indicate processing errors.
        outlier_threshold: Number of standard deviations for outlier detection.
            Common values: 2.0 (95%), 2.5 (99%), 3.0 (99.7%).

    Returns:
        Quality assessment dictionary containing:
        - valid_pixel_count: Number of valid pixels
        - valid_pixel_percent: Percentage of valid pixels
        - mean: Mean of valid pixels
        - std: Standard deviation of valid pixels
        - min: Minimum valid value
        - max: Maximum valid value
        - outlier_count: Number of detected outliers
        - outlier_locations: Indices of outlier pixels as tuple (row, col)
        - quality_score: Overall quality score (0-100)
        - warnings: List of quality warnings
        - recommendations: Suggested actions based on quality

    Raises:
        ValueError: If quality_flags shape doesn't match data shape.

    Example:
        >>> # Assess Landsat band quality
        >>> quality = assess_data_quality(
        >>>     band_data,
        >>>     valid_range=(0, 1),
        >>>     check_outliers=True
        >>> )
        >>> if quality['quality_score'] < 50:
        >>>     warnings.warn("Poor data quality detected")
    """
    import numpy as np

    # Input validation - ensure quality flags align with data
    if quality_flags is not None and quality_flags.shape != data.shape:
        raise ValueError(
            f"Quality flags shape {quality_flags.shape} must match data shape {data.shape}"
        )
    
    # Validate outlier threshold
    if outlier_threshold <= 0:
        raise ValueError(f"Outlier threshold must be positive, got {outlier_threshold}")

    # Handle nodata value
    if nodata_value is None:
        if np.issubdtype(data.dtype, np.floating):
            nodata_value = np.nan
        else:
            nodata_value = 0

    # Create validity mask through sequential filtering
    # Step 1: Check for nodata values
    # Handle both NaN and explicit nodata values appropriately
    if np.isnan(nodata_value):
        valid_mask = ~np.isnan(data)
    else:
        valid_mask = data != nodata_value

    # Step 2: Apply valid range constraints if provided
    if valid_range is not None:
        min_val, max_val = valid_range
        if min_val >= max_val:
            raise ValueError(f"Invalid range: min ({min_val}) must be less than max ({max_val})")
        # Create mask for values within expected physical range
        range_mask = (data >= min_val) & (data <= max_val)
        valid_mask = valid_mask & range_mask

    # Step 3: Apply quality flags if provided
    if quality_flags is not None:
        # Assume non-zero flags indicate problems
        flag_mask = quality_flags == 0
        valid_mask = valid_mask & flag_mask

    # Extract valid data for statistics
    valid_data = data[valid_mask]
    total_pixels = data.size
    valid_pixel_count = valid_data.size

    # Initialize results dictionary
    results = {
        "valid_pixel_count": int(valid_pixel_count),
        "valid_pixel_percent": float((valid_pixel_count / total_pixels) * 100)
        if total_pixels > 0
        else 0.0,
        "warnings": [],
        "recommendations": [],
    }

    # Compute basic statistics for valid pixels
    if valid_pixel_count > 0:
        results.update(
            {
                "mean": float(np.mean(valid_data)),
                "std": float(np.std(valid_data)),
                "min": float(np.min(valid_data)),
                "max": float(np.max(valid_data)),
            }
        )
    else:
        # No valid data available
        results.update({"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan})
        results["warnings"].append("No valid pixels found")
        results["recommendations"].append("Check input data and quality parameters")

    # Outlier detection using z-score method
    outlier_locations = []
    outlier_count = 0

    if (
        check_outliers and valid_pixel_count > 2
    ):  # Need at least 3 points for meaningful std
        # Calculate z-scores for all data points
        data_mean = results["mean"]
        data_std = results["std"]

        if data_std > 0:  # Avoid division by zero
            # Calculate z-scores only for valid pixels to identify outliers within valid data
            z_scores = np.abs((valid_data - data_mean) / data_std)
            outlier_mask_1d = z_scores > outlier_threshold

            # Map outliers back to original array coordinates
            if np.any(outlier_mask_1d):
                # Get indices of valid pixels
                valid_indices = np.where(valid_mask)

                # Get indices of outliers within valid pixels
                outlier_indices_1d = np.where(outlier_mask_1d)[0]

                # Map back to 2D coordinates
                if data.ndim == 2:
                    outlier_rows = valid_indices[0][outlier_indices_1d]
                    outlier_cols = valid_indices[1][outlier_indices_1d]
                    outlier_locations = list(
                        zip(outlier_rows.tolist(), outlier_cols.tolist())
                    )
                else:
                    # For higher dimensional data, store flat indices
                    outlier_flat_indices = np.where(valid_mask.flatten())[0][
                        outlier_indices_1d
                    ]
                    outlier_locations = outlier_flat_indices.tolist()

                outlier_count = len(outlier_locations)

    results.update(
        {"outlier_count": outlier_count, "outlier_locations": outlier_locations}
    )

    # Calculate overall quality score (0-100)
    quality_score = _calculate_quality_score(
        valid_pixel_percent=results["valid_pixel_percent"],
        outlier_percent=(outlier_count / total_pixels * 100) if total_pixels > 0 else 0,
        data_completeness=valid_pixel_count > 0,
        has_outliers=outlier_count > 0,
    )
    results["quality_score"] = quality_score

    # Generate quality warnings and recommendations
    _add_quality_warnings_and_recommendations(results)

    return results


def _calculate_quality_score(
    valid_pixel_percent: float,
    outlier_percent: float,
    data_completeness: bool,
    has_outliers: bool,
) -> float:
    """Calculate overall quality score from individual quality metrics.

    This is a helper function that combines multiple quality indicators
    into a single score using empirically-derived weights.
    """
    # Start with base score
    score = 100.0

    # Penalize low data coverage (most critical factor)
    coverage_penalty = max(
        0, (100 - valid_pixel_percent) * 0.8
    )  # 80% weight on coverage
    score -= coverage_penalty

    # Penalize high outlier rates (moderate factor)
    outlier_penalty = min(outlier_percent * 2.0, 20.0)  # Cap at 20 point penalty
    score -= outlier_penalty

    # Severe penalty for no data
    if not data_completeness:
        score = min(score, 10.0)  # Cap at 10 if no valid data

    # Moderate penalty for having outliers
    if has_outliers:
        score -= 5.0

    return max(0.0, score)  # Ensure non-negative


def _add_quality_warnings_and_recommendations(results: dict[str, Any]) -> None:
    """Add contextual warnings and recommendations based on quality metrics.

    This helper function analyzes the computed quality metrics and adds
    human-readable warnings and actionable recommendations. The warnings
    help identify specific quality issues while recommendations provide
    concrete steps to improve data quality.
    """
    valid_percent = results["valid_pixel_percent"]
    outlier_count = results["outlier_count"]
    quality_score = results["quality_score"]

    # Coverage warnings - critical for determining usability
    if valid_percent < 50:
        results["warnings"].append(f"Low data coverage ({valid_percent:.1f}%)")
        results["recommendations"].append(
            "Consider relaxing quality filters or using gap filling"
        )
    elif valid_percent < 80:
        results["warnings"].append(f"Moderate data coverage ({valid_percent:.1f}%)")
        results["recommendations"].append("Monitor data quality in final products")

    # Outlier warnings
    if outlier_count > 0:
        outlier_percent = (
            (outlier_count / results["valid_pixel_count"]) * 100
            if results["valid_pixel_count"] > 0
            else 0
        )
        if outlier_percent > 5:
            results["warnings"].append(f"High outlier rate ({outlier_percent:.1f}%)")
            results["recommendations"].append(
                "Review processing parameters and input data quality"
            )
        else:
            results["warnings"].append(f"Outliers detected ({outlier_count} pixels)")
            results["recommendations"].append(
                "Consider outlier removal for sensitive analyses"
            )

    # Overall quality warnings
    if quality_score < 30:
        results["warnings"].append("Very poor overall quality")
        results["recommendations"].append(
            "Do not use for analysis - investigate data issues"
        )
    elif quality_score < 60:
        results["warnings"].append("Poor overall quality")
        results["recommendations"].append(
            "Use with caution - apply additional quality filters"
        )
    elif quality_score < 80:
        results["warnings"].append("Moderate overall quality")
        results["recommendations"].append(
            "Acceptable for analysis with quality considerations"
        )

    # Statistical warnings
    import numpy as np  # Import needed for np.isnan

    if "std" in results and not np.isnan(results["std"]):
        if results["std"] == 0:
            results["warnings"].append("No data variation (constant values)")
            results["recommendations"].append(
                "Verify data is not corrupted or incorrectly processed"
            )
        elif (
            "mean" in results and results["std"] / abs(results["mean"]) > 2.0
        ):  # High coefficient of variation
            results["warnings"].append("High data variability")
            results["recommendations"].append(
                "Consider whether high variability is expected for this dataset"
            )


def compute_quality_metrics(
    data_stack: ArrayLike,
    dates: list[datetime] | None = None,
    compute_temporal_stability: bool = True,
    compute_spatial_coherence: bool = True,
) -> dict[str, ArrayLike]:
    """Compute multi-dimensional quality metrics for data stack.

    This function generates quality metrics that capture both temporal and
    spatial aspects of data quality. These metrics are crucial for the
    temporal compositing process, helping identify which observations are
    most reliable at each pixel location.

    The metrics enable quality-weighted compositing and help identify
    areas with persistent data quality issues that may need special handling.

    Args:
        data_stack: 3D array (time, height, width) of observations.
            Each time slice represents one observation period.
        dates: Optional list of acquisition dates for temporal analysis.
            If provided, used for temporal gap analysis.
            Must match time dimension length.
        compute_temporal_stability: Calculate temporal consistency metrics.
            Includes standard deviation, range, and trend analysis.
        compute_spatial_coherence: Calculate spatial smoothness metrics.
            Includes gradient analysis and edge detection.

    Returns:
        Dictionary of quality metric arrays with spatial dimensions (height, width):
        - observation_count: Number of valid observations per pixel
        - temporal_std: Temporal standard deviation per pixel
        - temporal_range: Temporal range (max-min) per pixel
        - temporal_cv: Coefficient of variation (std/mean) per pixel
        - spatial_gradient: Spatial gradient magnitude
        - edge_density: Density of edges/discontinuities in neighborhood
        - composite_quality: Overall quality score per pixel (0-100)
        - confidence: Confidence level for each pixel (0-1)
        - data_availability: Fraction of time periods with valid data

    Raises:
        ValueError: If input dimensions are invalid or dates don't match time dimension.

    Example:
        >>> # Compute quality metrics for monthly time series
        >>> metrics = compute_quality_metrics(
        >>>     data_stack=monthly_ndvi,  # (12, 1000, 1000)
        >>>     dates=monthly_dates,
        >>>     compute_temporal_stability=True,
        >>>     compute_spatial_coherence=True
        >>> )
        >>> # Use confidence for quality weighting
        >>> high_confidence = metrics['confidence'] > 0.8

    Note:
        These metrics feed into quality-weighted temporal compositing
        and help identify pixels that need gap filling or enhancement.
    """
    import numpy as np
    from scipy.ndimage import sobel, generic_filter

    # Input validation
    if data_stack.ndim != 3:
        raise ValueError(
            f"Data stack must be 3D (time, height, width), got {data_stack.ndim}D "
            f"with shape {data_stack.shape}"
        )

    n_times, height, width = data_stack.shape
    
    # Validate minimum data requirements
    if n_times < 2:
        raise ValueError(
            f"Need at least 2 time steps for temporal analysis, got {n_times}"
        )

    if dates is not None and len(dates) != n_times:
        raise ValueError(
            f"Dates length ({len(dates)}) must match time dimension ({n_times})"
        )

    # Initialize output dictionary with spatial arrays
    metrics = {}

    # Basic data availability metrics
    # Count valid (non-NaN) observations per pixel
    valid_mask = ~np.isnan(data_stack)
    observation_count = np.sum(valid_mask, axis=0).astype(np.float32)
    data_availability = observation_count / n_times

    metrics["observation_count"] = observation_count
    metrics["data_availability"] = data_availability

    # Temporal stability metrics (computed pixel-wise)
    if compute_temporal_stability:
        # Initialize temporal metric arrays
        temporal_std = np.full((height, width), np.nan, dtype=np.float32)
        temporal_range = np.full((height, width), np.nan, dtype=np.float32)
        temporal_cv = np.full((height, width), np.nan, dtype=np.float32)

        # Compute temporal statistics for each pixel
        for i in range(height):
            for j in range(width):
                pixel_time_series = data_stack[:, i, j]
                valid_obs = pixel_time_series[~np.isnan(pixel_time_series)]

                if (
                    len(valid_obs) >= 2
                ):  # Need at least 2 observations for meaningful stats
                    # Standard deviation across time - measures temporal stability
                    temporal_std[i, j] = np.std(valid_obs)

                    # Range across time - captures full extent of variation
                    temporal_range[i, j] = np.max(valid_obs) - np.min(valid_obs)

                    # Coefficient of variation (normalized variability)
                    # CV = std/mean, useful for comparing variability across different magnitudes
                    mean_val = np.mean(valid_obs)
                    if abs(mean_val) > 1e-6:  # Avoid division by zero
                        temporal_cv[i, j] = temporal_std[i, j] / abs(mean_val)
                    else:
                        temporal_cv[i, j] = np.inf  # Indicate problematic case (zero mean)

        metrics["temporal_std"] = temporal_std
        metrics["temporal_range"] = temporal_range
        metrics["temporal_cv"] = temporal_cv

    # Spatial coherence metrics
    if compute_spatial_coherence:
        # Compute spatial gradients using latest available data
        # Use the temporal mean for spatial analysis to reduce noise
        mean_data = np.nanmean(data_stack, axis=0)

        # Calculate spatial gradients using Sobel operators
        # Handle NaN values by temporarily filling with zeros
        mean_filled = np.nan_to_num(mean_data, nan=0.0)

        grad_x = sobel(mean_filled, axis=1)  # Horizontal gradient
        grad_y = sobel(mean_filled, axis=0)  # Vertical gradient

        # Gradient magnitude
        spatial_gradient = np.sqrt(grad_x**2 + grad_y**2).astype(np.float32)

        # Restore NaN values where original data was missing
        spatial_gradient[np.isnan(mean_data)] = np.nan

        metrics["spatial_gradient"] = spatial_gradient

        # Edge density using local standard deviation
        # High edge density indicates lots of local variation
        def local_std(patch):
            """Calculate standard deviation in local neighborhood.
            
            High local variation indicates edges, boundaries, or noise.
            This helps identify areas that may need special processing.
            """
            valid_patch = patch[~np.isnan(patch)]
            return np.std(valid_patch) if len(valid_patch) > 1 else 0.0

        # Apply local standard deviation filter
        edge_density = generic_filter(
            mean_data,
            function=local_std,
            size=3,  # 3x3 neighborhood
            mode="constant",
            cval=np.nan,
        ).astype(np.float32)

        metrics["edge_density"] = edge_density

    # Compute composite quality score (0-100) for each pixel
    composite_quality = _compute_pixel_quality_scores(
        observation_count=observation_count,
        data_availability=data_availability,
        temporal_std=metrics.get("temporal_std"),
        temporal_cv=metrics.get("temporal_cv"),
        spatial_gradient=metrics.get("spatial_gradient"),
        n_times=n_times,
    )

    metrics["composite_quality"] = composite_quality

    # Compute confidence scores (0-1) based on quality metrics
    confidence = _compute_pixel_confidence(
        composite_quality=composite_quality,
        observation_count=observation_count,
        data_availability=data_availability,
    )

    metrics["confidence"] = confidence

    return metrics


def _compute_pixel_quality_scores(
    observation_count: ArrayLike,
    data_availability: ArrayLike,
    temporal_std: ArrayLike | None,
    temporal_cv: ArrayLike | None,
    spatial_gradient: ArrayLike | None,
    n_times: int,
) -> ArrayLike:
    """Compute composite quality scores for each pixel.

    This helper function combines multiple quality indicators into a single
    score using empirically-derived weights optimized for satellite data.
    """
    import numpy as np

    # Start with perfect quality score
    quality = np.full_like(observation_count, 100.0, dtype=np.float32)

    # Factor 1: Data availability (40% weight)
    # Penalize pixels with few observations
    availability_score = (data_availability * 100) * 0.4
    quality = quality * (availability_score / 40.0)  # Normalize to 0-1 range

    # Factor 2: Temporal stability (30% weight)
    if temporal_cv is not None:
        # Lower coefficient of variation = higher quality
        # Use sigmoid function to map CV to quality score
        cv_quality = 1.0 / (
            1.0 + np.exp(temporal_cv - 1.0)
        )  # Sigmoid centered at CV=1.0
        cv_quality = np.nan_to_num(cv_quality, nan=0.5)  # Handle NaN/inf values
        quality *= 0.3 * cv_quality + 0.7  # Apply 30% weight

    # Factor 3: Spatial coherence (20% weight)
    if spatial_gradient is not None:
        # Normalize gradient values (high gradients reduce quality)
        # Use 95th percentile for robust normalization
        valid_gradients = spatial_gradient[~np.isnan(spatial_gradient)]
        if len(valid_gradients) > 0:
            grad_95th = np.percentile(valid_gradients, 95)
            if grad_95th > 0:
                normalized_grad = np.clip(spatial_gradient / grad_95th, 0, 1)
                spatial_quality = 1.0 - normalized_grad  # High gradient = low quality
                spatial_quality = np.nan_to_num(spatial_quality, nan=0.5)
                quality *= 0.2 * spatial_quality + 0.8  # Apply 20% weight

    # Factor 4: Minimum observation threshold (10% weight)
    # Severe penalty for very few observations
    min_obs_penalty = np.where(
        observation_count
        < max(2, n_times * 0.1),  # Need at least 2 obs or 10% of total
        0.9,  # 90% penalty
        1.0,  # No penalty
    )
    quality *= min_obs_penalty

    # Ensure quality stays in valid range
    quality = np.clip(quality, 0.0, 100.0)

    return quality


def _compute_pixel_confidence(
    composite_quality: ArrayLike,
    observation_count: ArrayLike,
    data_availability: ArrayLike,
) -> ArrayLike:
    """Compute confidence scores (0-1) from quality metrics.

    Confidence is a user-friendly metric that indicates reliability
    for end-user applications and visualization.
    """
    import numpy as np

    # Convert quality score (0-100) to confidence (0-1)
    base_confidence = composite_quality / 100.0

    # Apply additional confidence modifiers

    # Boost confidence for high data availability
    availability_boost = np.where(
        data_availability > 0.8,
        1.1,  # 10% boost for high availability
        1.0,
    )

    # Reduce confidence for very low observation counts
    obs_penalty = np.where(
        observation_count < 3,
        0.8,  # 20% penalty for very few observations
        1.0,
    )

    confidence = base_confidence * availability_boost * obs_penalty

    # Ensure confidence stays in valid range
    confidence = np.clip(confidence, 0.0, 1.0)

    return confidence


def create_composite_quality_mask(
    masks: dict[str, ArrayLike],
    weights: dict[str, float] | None = None,
    combination_method: Literal["and", "or", "weighted"] = "and",
    dilation_radius: int = 0,
) -> tuple[ArrayLike, dict[str, Any]]:
    """Create composite quality mask from multiple input masks.

    This function combines multiple quality indicators (clouds, shadows,
    saturation, etc.) into a single comprehensive quality mask. This is
    critical for the prepare phase where we need a unified view of pixel
    validity across all quality dimensions.

    The composite mask determines which pixels participate in analysis
    and which require gap filling or exclusion from final products.

    Args:
        masks: Dictionary of mask_name -> mask_array where True = valid pixel.
               Common keys: 'cloud', 'shadow', 'saturated', 'snow', 'water'
               All masks must have identical spatial dimensions.
        weights: Optional weights for each mask (for weighted combination).
               Keys must match mask keys. Weights are normalized automatically.
               Higher weights give more influence to that mask.
        combination_method: How to combine masks:
            - 'and': Pixel is valid only if valid in ALL masks (strict)
            - 'or': Pixel is valid if valid in ANY mask (permissive)
            - 'weighted': Weighted combination of mask confidence values
        dilation_radius: Number of pixels to dilate masks (expands masked areas).
               Applied before combination. Useful for conservative masking.

    Returns:
        Tuple of (composite_mask, statistics) where:
        - composite_mask: Combined quality mask (True = valid pixel)
        - statistics: Dictionary with combination statistics:
            - mask_coverage: Percent coverage of each input mask
            - composite_coverage: Percent of valid pixels in composite
            - overlap_matrix: Overlap statistics between input masks
            - combination_efficiency: How much valid data remains after combination

    Raises:
        ValueError: If masks have different shapes or invalid parameters.

    Example:
        >>> # Combine multiple quality indicators
        >>> masks = {
        >>>     'cloud': cloud_mask,
        >>>     'shadow': shadow_mask,
        >>>     'saturated': saturation_mask
        >>> }
        >>> composite, stats = create_composite_quality_mask(
        >>>     masks,
        >>>     combination_method='and',
        >>>     dilation_radius=3
        >>> )
        >>> print(f"Final coverage: {stats['composite_coverage']:.1f}%")
    """
    import numpy as np
    from scipy.ndimage import binary_dilation

    # Input validation
    if not masks:
        raise ValueError("At least one mask must be provided")

    # Check that all masks have the same shape
    mask_shapes = [mask.shape for mask in masks.values()]
    reference_shape = mask_shapes[0]
    if not all(shape == reference_shape for shape in mask_shapes):
        raise ValueError(
            f"All masks must have same shape. Got: {dict(zip(masks.keys(), mask_shapes))}"
        )
    
    # Validate mask data types and values
    for name, mask in masks.items():
        if not np.issubdtype(mask.dtype, np.bool_) and not np.issubdtype(mask.dtype, np.number):
            raise ValueError(
                f"Mask '{name}' has invalid dtype {mask.dtype}. "
                f"Masks must be boolean or numeric arrays."
            )

    # Validate weights if provided
    if weights is not None:
        if set(weights.keys()) != set(masks.keys()):
            raise ValueError(
                f"Weight keys {set(weights.keys())} must match mask keys {set(masks.keys())}"
            )
        if any(w <= 0 for w in weights.values()):
            raise ValueError("All weights must be positive")

    # Convert masks to boolean and apply dilation if requested
    processed_masks = {}

    for name, mask in masks.items():
        # Convert to boolean (True = valid pixel)
        bool_mask = mask.astype(bool)

        # Apply dilation to expand masked (invalid) areas
        # This provides a conservative buffer around detected features
        # Note: We dilate the INVALID areas (False pixels), so we need to invert
        if dilation_radius > 0:
            # Create structure element (disk-like) for morphological operation
            # Larger radius = more conservative masking
            structure = np.ones(
                (2 * dilation_radius + 1, 2 * dilation_radius + 1), dtype=bool
            )

            # Dilate invalid areas (invert mask, dilate, invert back)
            invalid_mask = ~bool_mask
            dilated_invalid = binary_dilation(invalid_mask, structure=structure)
            bool_mask = ~dilated_invalid

        processed_masks[name] = bool_mask

    # Calculate individual mask statistics
    total_pixels = reference_shape[0] * reference_shape[1]
    mask_coverage = {}

    for name, mask in processed_masks.items():
        valid_pixels = np.sum(mask)
        coverage_percent = (valid_pixels / total_pixels) * 100
        mask_coverage[name] = coverage_percent

    # Combine masks according to specified method
    if combination_method == "and":
        # Pixel is valid only if valid in ALL masks (most restrictive)
        composite_mask = np.ones(reference_shape, dtype=bool)

        for mask in processed_masks.values():
            composite_mask = composite_mask & mask

    elif combination_method == "or":
        # Pixel is valid if valid in ANY mask (most permissive)
        composite_mask = np.zeros(reference_shape, dtype=bool)

        for mask in processed_masks.values():
            composite_mask = composite_mask | mask

    elif combination_method == "weighted":
        # Weighted combination of mask confidence values
        # This allows fine-tuned control over how different quality indicators
        # contribute to the final mask, useful when some indicators are more reliable
        if weights is None:
            # Use equal weights if none provided - democratic approach
            weights = {name: 1.0 for name in masks.keys()}

        # Normalize weights to sum to 1 for interpretable results
        total_weight = sum(weights.values())
        normalized_weights = {name: w / total_weight for name, w in weights.items()}

        # Calculate weighted sum
        weighted_sum = np.zeros(reference_shape, dtype=np.float32)

        for name, mask in processed_masks.items():
            weight = normalized_weights[name]
            # Convert boolean mask to float (1.0 = valid, 0.0 = invalid)
            weighted_sum += mask.astype(np.float32) * weight

        # Convert weighted sum to boolean using 0.5 threshold
        # This means pixel needs >50% weighted confidence to be valid
        composite_mask = weighted_sum > 0.5

    else:
        raise ValueError(f"Unknown combination method: {combination_method}")

    # Calculate composite statistics
    composite_valid_pixels = np.sum(composite_mask)
    composite_coverage = (composite_valid_pixels / total_pixels) * 100

    # Calculate overlap matrix (pairwise mask overlaps)
    overlap_matrix = _calculate_mask_overlap_matrix(processed_masks)

    # Calculate combination efficiency (how much data is retained)
    if combination_method == "and":
        # For AND, efficiency is the intersection relative to the union
        union_mask = np.zeros(reference_shape, dtype=bool)
        for mask in processed_masks.values():
            union_mask = union_mask | mask
        union_pixels = np.sum(union_mask)
        efficiency = (
            (composite_valid_pixels / union_pixels * 100) if union_pixels > 0 else 0
        )
    else:
        # For OR and weighted, efficiency is relative to best individual mask
        max_individual_coverage = max(mask_coverage.values()) if mask_coverage else 0
        efficiency = (
            (composite_coverage / max_individual_coverage * 100)
            if max_individual_coverage > 0
            else 0
        )

    # Compile statistics
    statistics = {
        "mask_coverage": mask_coverage,
        "composite_coverage": float(composite_coverage),
        "overlap_matrix": overlap_matrix,
        "combination_efficiency": float(efficiency),
        "total_pixels": int(total_pixels),
        "valid_pixels": int(composite_valid_pixels),
        "combination_method": combination_method,
        "dilation_radius": dilation_radius,
    }

    # Add normalized weights if they were computed
    if combination_method == "weighted":
        if weights is None:
            # Use equal weights if none provided
            normalized_weights = {name: 1.0 / len(masks) for name in masks.keys()}
        else:
            # Normalize weights to sum to 1 (already done above)
            total_weight = sum(weights.values())
            normalized_weights = {name: w / total_weight for name, w in weights.items()}

        statistics["normalized_weights"] = normalized_weights

    return composite_mask, statistics


def _calculate_mask_overlap_matrix(
    masks: dict[str, ArrayLike],
) -> dict[str, dict[str, float]]:
    """Calculate pairwise overlap statistics between masks.

    This helper function computes how much each pair of masks overlaps,
    which is useful for understanding mask relationships and debugging.
    """
    import numpy as np

    mask_names = list(masks.keys())
    overlap_matrix = {}

    for i, name1 in enumerate(mask_names):
        overlap_matrix[name1] = {}

        for name2 in mask_names:
            if name1 == name2:
                # Self-overlap is always 100%
                overlap_matrix[name1][name2] = 100.0
            else:
                mask1 = masks[name1]
                mask2 = masks[name2]

                # Calculate Jaccard index (intersection over union)
                intersection = np.sum(mask1 & mask2)
                union = np.sum(mask1 | mask2)

                if union > 0:
                    jaccard = (intersection / union) * 100
                else:
                    jaccard = 0.0

                overlap_matrix[name1][name2] = float(jaccard)

    return overlap_matrix


def flag_suspicious_values(
    data: ArrayLike,
    index_type: Literal["ndvi", "ndwi", "ndui", "radiance"],
    context_data: dict[str, ArrayLike] | None = None,
    flag_impossible: bool = True,
    flag_unlikely: bool = True,
) -> tuple[ArrayLike, dict[str, Any]]:
    """Flag pixels with suspicious or impossible values.

    This function applies domain-specific knowledge to identify values that
    are technically valid but physically impossible or highly unlikely. For
    example, NDVI > 0.9 in urban areas or negative radiance values.

    This validation is crucial for catching processing errors and ensuring
    scientific validity of derived products.

    Args:
        data: Data array to check. Should contain the specified index type.
        index_type: Type of data for applying appropriate validation rules:
            - 'ndvi': Normalized Difference Vegetation Index [-1, 1]
            - 'ndwi': Normalized Difference Water Index [-1, 1]
            - 'ndui': Normalized Difference Urban Index [0, 1]
            - 'radiance': Raw radiance values [0, inf)
        context_data: Additional context arrays for enhanced validation:
            - 'land_cover': Land cover classification
            - 'elevation': Elevation in meters
            - 'urban_mask': Boolean urban area mask
            - 'water_mask': Boolean water body mask
        flag_impossible: Flag physically impossible values (hard constraints).
        flag_unlikely: Flag statistically unlikely values (soft constraints).

    Returns:
        Tuple of (flag_mask, flag_statistics) where:
        - flag_mask: Boolean mask (True = suspicious pixel, False = normal)
        - flag_statistics: Dictionary with validation results:
            - impossible_count: Number of impossible values
            - unlikely_count: Number of unlikely values
            - impossible_percent: Percentage of impossible values
            - unlikely_percent: Percentage of unlikely values
            - severity_scores: Array of severity scores [0-3] per pixel
            - flagged_regions: Number of connected suspicious regions
            - validation_rules_applied: List of validation rules that were applied

    Raises:
        ValueError: If index_type is unsupported or context data has wrong shape.

    Example:
        >>> # Check NDUI for suspicious urban index values
        >>> flags, stats = flag_suspicious_values(
        >>>     ndui_data,
        >>>     index_type='ndui',
        >>>     context_data={'urban_mask': city_areas},
        >>>     flag_impossible=True,
        >>>     flag_unlikely=True
        >>> )
        >>> print(f"Found {stats['impossible_count']} impossible values")
    """
    import numpy as np
    from scipy.ndimage import label

    # Input validation
    if context_data is not None:
        for name, context_array in context_data.items():
            if context_array.shape != data.shape:
                raise ValueError(
                    f"Context data '{name}' shape {context_array.shape} must match data shape {data.shape}"
                )

    # Initialize flag mask and severity scores
    flag_mask = np.zeros_like(data, dtype=bool)
    severity_scores = np.zeros_like(
        data, dtype=np.int8
    )  # 0=normal, 1=unlikely, 2=impossible, 3=severe

    # Track applied validation rules
    validation_rules_applied = []

    # Apply index-specific validation rules
    if index_type == "ndvi":
        impossible_flags, unlikely_flags = _validate_ndvi(
            data, context_data, flag_impossible, flag_unlikely
        )
        validation_rules_applied.extend(["ndvi_range", "ndvi_context"])

    elif index_type == "ndwi":
        impossible_flags, unlikely_flags = _validate_ndwi(
            data, context_data, flag_impossible, flag_unlikely
        )
        validation_rules_applied.extend(["ndwi_range", "ndwi_context"])

    elif index_type == "ndui":
        impossible_flags, unlikely_flags = _validate_ndui(
            data, context_data, flag_impossible, flag_unlikely
        )
        validation_rules_applied.extend(["ndui_range", "ndui_context"])

    elif index_type == "radiance":
        impossible_flags, unlikely_flags = _validate_radiance(
            data, context_data, flag_impossible, flag_unlikely
        )
        validation_rules_applied.extend(["radiance_physical", "radiance_statistical"])

    else:
        raise ValueError(f"Unsupported index type: {index_type}")

    # Combine flags and assign severity scores
    if flag_impossible:
        flag_mask |= impossible_flags
        severity_scores[impossible_flags] = 2  # Impossible = severity 2

    if flag_unlikely:
        flag_mask |= unlikely_flags
        # Only assign unlikely severity if not already marked as impossible
        unlikely_only = unlikely_flags & ~impossible_flags
        severity_scores[unlikely_only] = 1  # Unlikely = severity 1

    # Check for extreme clustering (many suspicious pixels together)
    extreme_cluster_mask = _detect_extreme_clusters(flag_mask)
    flag_mask |= extreme_cluster_mask
    severity_scores[extreme_cluster_mask] = 3  # Extreme clustering = severity 3
    validation_rules_applied.append("clustering_analysis")

    # Calculate statistics
    total_pixels = data.size
    valid_pixels = ~np.isnan(data)
    total_valid = np.sum(valid_pixels)

    impossible_count = int(np.sum(impossible_flags)) if flag_impossible else 0
    unlikely_count = int(np.sum(unlikely_flags)) if flag_unlikely else 0
    extreme_count = int(np.sum(extreme_cluster_mask))

    # Calculate percentages relative to valid pixels
    impossible_percent = (
        (impossible_count / total_valid * 100) if total_valid > 0 else 0
    )
    unlikely_percent = (unlikely_count / total_valid * 100) if total_valid > 0 else 0

    # Count connected flagged regions using connected components
    labeled_result = label(flag_mask)
    flagged_regions = labeled_result[0]  # type: ignore
    n_regions = int(labeled_result[1])  # type: ignore

    flag_statistics = {
        "impossible_count": impossible_count,
        "unlikely_count": unlikely_count,
        "extreme_cluster_count": extreme_count,
        "impossible_percent": float(impossible_percent),
        "unlikely_percent": float(unlikely_percent),
        "severity_scores": severity_scores,
        "flagged_regions": int(n_regions),
        "validation_rules_applied": validation_rules_applied,
        "total_flagged": int(np.sum(flag_mask)),
        "total_valid_pixels": int(total_valid),
    }

    return flag_mask, flag_statistics


def _validate_ndvi(
    data: ArrayLike,
    context_data: dict[str, ArrayLike] | None,
    flag_impossible: bool,
    flag_unlikely: bool,
) -> tuple[ArrayLike, ArrayLike]:
    """Apply NDVI-specific validation rules.
    
    NDVI = (NIR - Red) / (NIR + Red) is mathematically bounded to [-1, 1]
    but real-world values have additional constraints based on physics
    and land cover characteristics.
    """
    import numpy as np

    impossible_flags = np.zeros_like(data, dtype=bool)
    unlikely_flags = np.zeros_like(data, dtype=bool)

    if flag_impossible:
        # NDVI must be in [-1, 1] range (mathematical constraint)
        impossible_flags |= (data < -1.0) | (data > 1.0)

        # NDVI values exactly 1.0 or -1.0 require one band to be zero
        # This is extremely rare in calibrated satellite data
        impossible_flags |= (data == 1.0) | (data == -1.0)

    if flag_unlikely:
        # Very high NDVI in non-vegetated areas is unlikely
        if context_data is not None and "urban_mask" in context_data:
            urban_mask = context_data["urban_mask"].astype(bool)
            unlikely_flags |= urban_mask & (data > 0.6)  # High NDVI in urban areas

        if context_data is not None and "water_mask" in context_data:
            water_mask = context_data["water_mask"].astype(bool)
            unlikely_flags |= water_mask & (data > 0.1)  # Vegetation over water

        # Statistically unlikely values
        unlikely_flags |= data > 0.95  # Extremely high vegetation
        unlikely_flags |= data < -0.5  # Very low values (usually water/bare soil)

    return impossible_flags, unlikely_flags


def _validate_ndwi(
    data: ArrayLike,
    context_data: dict[str, ArrayLike] | None,
    flag_impossible: bool,
    flag_unlikely: bool,
) -> tuple[ArrayLike, ArrayLike]:
    """Apply NDWI-specific validation rules."""
    import numpy as np

    impossible_flags = np.zeros_like(data, dtype=bool)
    unlikely_flags = np.zeros_like(data, dtype=bool)

    if flag_impossible:
        # NDWI must be in [-1, 1] range (mathematical constraint)
        impossible_flags |= (data < -1.0) | (data > 1.0)

    if flag_unlikely:
        # Very high NDWI in non-water areas is unlikely
        if context_data is not None and "urban_mask" in context_data:
            urban_mask = context_data["urban_mask"].astype(bool)
            unlikely_flags |= urban_mask & (
                data > 0.3
            )  # High water index in urban areas

        # High NDWI at high elevations (where water is less likely)
        if context_data is not None and "elevation" in context_data:
            elevation = context_data["elevation"]
            high_elevation = elevation > 3000  # Above 3000m
            unlikely_flags |= high_elevation & (data > 0.2)

    return impossible_flags, unlikely_flags


def _validate_ndui(
    data: ArrayLike,
    context_data: dict[str, ArrayLike] | None,
    flag_impossible: bool,
    flag_unlikely: bool,
) -> tuple[ArrayLike, ArrayLike]:
    """Apply NDUI-specific validation rules."""
    import numpy as np

    impossible_flags = np.zeros_like(data, dtype=bool)
    unlikely_flags = np.zeros_like(data, dtype=bool)

    if flag_impossible:
        # NDUI must be in [0, 1] range (by definition)
        impossible_flags |= (data < 0.0) | (data > 1.0)

    if flag_unlikely:
        # Very high NDUI in known non-urban areas
        if context_data is not None and "water_mask" in context_data:
            water_mask = context_data["water_mask"].astype(bool)
            unlikely_flags |= water_mask & (data > 0.1)  # Urban signal over water

        # Extremely high NDUI values are rare even in urban areas
        unlikely_flags |= data > 0.8  # Very high urban index

    return impossible_flags, unlikely_flags


def _validate_radiance(
    data: ArrayLike,
    context_data: dict[str, ArrayLike] | None,
    flag_impossible: bool,
    flag_unlikely: bool,
) -> tuple[ArrayLike, ArrayLike]:
    """Apply radiance-specific validation rules."""
    import numpy as np

    impossible_flags = np.zeros_like(data, dtype=bool)
    unlikely_flags = np.zeros_like(data, dtype=bool)

    if flag_impossible:
        # Radiance cannot be negative (physical constraint)
        impossible_flags |= data < 0.0

        # Extremely high radiance values (sensor saturation)
        impossible_flags |= data > 1000.0  # Adjust based on sensor specifications

    if flag_unlikely:
        # Very high radiance in non-urban areas
        if context_data is not None and "water_mask" in context_data:
            water_mask = context_data["water_mask"].astype(bool)
            unlikely_flags |= water_mask & (data > 10.0)  # Bright lights over water

        # Statistical outliers (beyond 99.9th percentile)
        valid_data = data[~np.isnan(data) & (data >= 0)]
        if len(valid_data) > 100:  # Need sufficient data for statistics
            p999 = np.percentile(valid_data, 99.9)
            unlikely_flags |= data > p999

    return impossible_flags, unlikely_flags


def _detect_extreme_clusters(
    flag_mask: ArrayLike, min_cluster_size: int = 100
) -> ArrayLike:
    """Detect extreme clusters of suspicious pixels.

    Large clusters of flagged pixels may indicate systematic processing
    errors rather than isolated data quality issues. This helps distinguish
    between random quality problems and systematic failures that require
    investigation.
    """
    import numpy as np
    from scipy.ndimage import label, sum as ndi_sum

    # Find connected components of flagged pixels
    labeled_result = label(flag_mask)
    labeled_regions = labeled_result[0]  # type: ignore
    n_regions = int(labeled_result[1])  # type: ignore

    extreme_clusters = np.zeros_like(flag_mask, dtype=bool)

    # Check each region for extreme size
    for region_id in range(1, n_regions + 1):
        region_mask = labeled_regions == region_id
        region_size = np.sum(region_mask)

        # Flag regions that are suspiciously large
        if region_size >= min_cluster_size:
            extreme_clusters |= region_mask

    return extreme_clusters


def compute_confidence_scores(
    data: ArrayLike,
    quality_metrics: dict[str, ArrayLike],
    observation_count: ArrayLike | None = None,
    temporal_stability: ArrayLike | None = None,
) -> ArrayLike:
    """Compute per-pixel confidence scores for analysis results.

    This function generates confidence scores that indicate the reliability
    of computed values at each pixel. These scores are essential for
    communicating data quality to end users and for quality-weighted
    processing operations.

    Confidence scores influence visualization (transparency), analysis
    (weighting), and validation (thresholds) throughout the pipeline.

    Args:
        data: Computed data values (e.g., indices, composites).
            Should not contain NaN values where confidence is computed.
        quality_metrics: Dictionary of quality metric arrays from compute_quality_metrics().
            Expected keys: 'composite_quality', 'data_availability'
            All arrays must match data shape.
        observation_count: Optional number of observations per pixel.
            Higher counts generally increase confidence.
            If None, uses data_availability from quality_metrics.
        temporal_stability: Optional temporal stability scores [0, 1].
            Lower values (more stable) increase confidence.
            If None, computed from quality_metrics if available.

    Returns:
        Confidence score array with values in [0, 1] where:
        - 1.0 = Highest confidence (reliable data)
        - 0.8-0.99 = High confidence
        - 0.5-0.79 = Moderate confidence
        - 0.2-0.49 = Low confidence (use with caution)
        - 0.0-0.19 = No confidence (exclude from analysis)

    Raises:
        ValueError: If quality_metrics are missing required keys or have wrong shapes.

    Example:
        >>> # Compute confidence for NDUI results
        >>> metrics = compute_quality_metrics(data_stack)
        >>> confidence = compute_confidence_scores(
        >>>     ndui_composite,
        >>>     metrics,
        >>>     observation_count=obs_count
        >>> )
        >>> # Mask low confidence areas
        >>> reliable_mask = confidence > 0.5

    Note:
        Low confidence areas may be masked in final products or
        highlighted in quality assessment reports.
    """
    import numpy as np

    # Input validation
    required_keys = {"composite_quality", "data_availability"}
    if not required_keys.issubset(quality_metrics.keys()):
        missing = required_keys - set(quality_metrics.keys())
        raise ValueError(f"Quality metrics missing required keys: {missing}")

    # Validate array shapes
    for key, metric_array in quality_metrics.items():
        if metric_array.shape != data.shape:
            raise ValueError(
                f"Quality metric '{key}' shape {metric_array.shape} must match data shape {data.shape}"
            )

    if observation_count is not None and observation_count.shape != data.shape:
        raise ValueError(
            f"Observation count shape {observation_count.shape} must match data shape {data.shape}"
        )

    if temporal_stability is not None and temporal_stability.shape != data.shape:
        raise ValueError(
            f"Temporal stability shape {temporal_stability.shape} must match data shape {data.shape}"
        )

    # Initialize confidence with base quality score (convert 0-100 to 0-1)
    base_confidence = quality_metrics["composite_quality"] / 100.0
    confidence = base_confidence.astype(np.float32)

    # Factor 1: Data availability boost (30% weight)
    # More available data = higher confidence in the result
    # Use square root to model diminishing returns - first observations
    # contribute more to confidence than later ones
    availability = quality_metrics["data_availability"]
    availability_boost = np.power(
        availability, 0.5
    )  # Square root for diminishing returns
    confidence *= 0.7 + 0.3 * availability_boost  # Boost up to 30%

    # Factor 2: Observation count boost (25% weight)
    if observation_count is not None:
        # Normalize observation count to [0, 1] using sigmoid
        # More observations = higher confidence, with diminishing returns
        normalized_obs = observation_count / (
            observation_count + 5.0
        )  # Sigmoid-like normalization
        obs_boost = np.power(normalized_obs, 0.3)  # Gentle curve
        confidence *= 0.75 + 0.25 * obs_boost  # Boost up to 25%

    # Factor 3: Temporal stability boost (20% weight)
    if temporal_stability is not None:
        # Temporal stability is assumed to be [0, 1] where lower = more stable
        # Convert to confidence boost: stable = high confidence
        stability_confidence = 1.0 - temporal_stability
        stability_boost = np.power(stability_confidence, 0.5)
        confidence *= 0.8 + 0.2 * stability_boost  # Boost up to 20%
    elif "temporal_cv" in quality_metrics:
        # Use coefficient of variation as proxy for temporal stability
        # Lower CV = more stable = higher confidence
        cv = quality_metrics["temporal_cv"]
        # Normalize CV to [0, 1] using sigmoid (CV > 1.0 is considered high)
        cv_normalized = 1.0 / (1.0 + np.exp(cv - 1.0))
        cv_boost = np.power(cv_normalized, 0.4)
        cv_boost = np.nan_to_num(cv_boost, nan=0.5)  # Handle NaN/inf
        confidence *= 0.8 + 0.2 * cv_boost

    # Factor 4: Data validity check (15% weight)
    # Penalize areas with invalid or suspicious data values
    valid_data_mask = ~np.isnan(data) & np.isfinite(data)
    confidence = np.where(
        valid_data_mask, confidence, confidence * 0.1
    )  # Severe penalty for invalid data

    # Factor 5: Spatial coherence boost (10% weight)
    if "spatial_gradient" in quality_metrics:
        # Lower gradients = more coherent = higher confidence
        gradient = quality_metrics["spatial_gradient"]
        valid_gradients = gradient[~np.isnan(gradient)]

        if len(valid_gradients) > 0:
            # Normalize gradient using robust statistics
            median_grad = np.median(valid_gradients)
            mad_grad = np.median(np.abs(valid_gradients - median_grad))

            if mad_grad > 0:
                # Z-score like normalization using median and MAD
                gradient_norm = np.abs(gradient - median_grad) / (
                    1.4826 * mad_grad
                )  # 1.4826 converts MAD to std estimate
                gradient_norm = np.nan_to_num(gradient_norm, nan=1.0)

                # Convert to confidence: lower gradient = higher confidence
                spatial_confidence = 1.0 / (1.0 + gradient_norm)  # Inverse relationship
                confidence *= 0.9 + 0.1 * spatial_confidence  # Boost up to 10%

    # Final adjustments and bounds checking

    # Apply conservative ceiling for noisy data
    # If base quality is very low, cap maximum achievable confidence
    low_quality_mask = base_confidence < 0.3
    confidence = np.where(low_quality_mask, np.minimum(confidence, 0.4), confidence)

    # Apply confidence floor - some minimum confidence for valid pixels
    valid_floor = 0.05  # Even poor data gets some minimal confidence
    confidence = np.where(valid_data_mask, np.maximum(confidence, valid_floor), 0.0)

    # Ensure confidence stays in [0, 1] range
    confidence = np.clip(confidence, 0.0, 1.0)

    # Set confidence to 0 for completely invalid areas
    confidence = np.where(valid_data_mask, confidence, 0.0)

    return confidence


def identify_processing_artifacts(
    data: ArrayLike, artifact_types: list[str] | None = None, window_size: int = 5
) -> dict[str, ArrayLike]:
    """Identify common processing artifacts in raster data.

    This function detects artifacts that can arise from processing steps
    such as resampling, mosaicking, or atmospheric correction. Early
    detection allows for corrective action or quality warnings.

    Common artifacts include striping, edge effects, resampling artifacts,
    and tile boundaries from mosaicking operations. Detection uses spatial
    pattern analysis and statistical outlier detection methods.

    Args:
        data: Data array to analyze (2D or 3D).
            Should be preprocessed to remove obvious invalid values.
            For 3D data, analysis is performed on each slice.
        artifact_types: List of artifact types to check for:
            - 'striping': Systematic row/column striping patterns
            - 'edges': Edge artifacts from processing boundaries
            - 'tiles': Tile boundary artifacts from mosaicking
            - 'resampling': Checkerboard/aliasing from resampling
            If None, checks for all supported types.
        window_size: Size of analysis window for local statistics.
            Larger windows detect broader artifacts but may miss fine details.
            Must be odd integer >= 3.

    Returns:
        Dictionary mapping artifact_type to boolean mask array where
        True indicates locations where artifacts were detected.
        All masks have same shape as input data.

    Raises:
        ValueError: If window_size is invalid or data has unsupported dimensions.

    Example:
        >>> # Check for processing artifacts
        >>> artifacts = identify_processing_artifacts(
        >>>     composite_data,
        >>>     artifact_types=['striping', 'tiles'],
        >>>     window_size=7
        >>> )
        >>> if artifacts['striping'].any():
        >>>     warnings.warn(f"Striping detected in {np.sum(artifacts['striping'])} pixels")
        >>> # Create overall artifact mask
        >>> any_artifacts = np.any(list(artifacts.values()), axis=0)

    Note:
        Detection algorithms use conservative thresholds to minimize false positives.
        Some artifacts may only be detectable with specific parameter tuning.
    """
    import numpy as np
    from scipy.ndimage import generic_filter, sobel, uniform_filter
    from scipy.stats import zscore

    # Input validation
    if data.ndim not in [2, 3]:
        raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")

    if window_size < 3 or window_size % 2 == 0:
        raise ValueError(f"Window size must be odd integer >= 3, got {window_size}")

    # Set default artifact types
    if artifact_types is None:
        artifact_types = ["striping", "edges", "tiles", "resampling"]

    # Validate artifact types
    supported_types = {"striping", "edges", "tiles", "resampling"}
    invalid_types = set(artifact_types) - supported_types
    if invalid_types:
        raise ValueError(
            f"Unsupported artifact types: {invalid_types}. Supported: {supported_types}"
        )

    # Initialize results dictionary
    artifacts = {}

    # Handle 3D data by processing each slice
    if data.ndim == 3:
        n_slices = data.shape[0]
        for artifact_type in artifact_types:
            # Initialize 3D mask
            artifacts[artifact_type] = np.zeros_like(data, dtype=bool)

        # Process each slice independently
        for i in range(n_slices):
            slice_data = data[i, :, :]
            slice_artifacts = identify_processing_artifacts(
                slice_data, artifact_types=artifact_types, window_size=window_size
            )

            # Combine results
            for artifact_type in artifact_types:
                artifacts[artifact_type][i, :, :] = slice_artifacts[artifact_type]

        return artifacts

    # Process 2D data
    height, width = data.shape

    # Create base mask for valid pixels (exclude NaN and infinite values)
    valid_mask = np.isfinite(data)

    # Detect striping artifacts
    if "striping" in artifact_types:
        striping_mask = _detect_striping_artifacts(data, valid_mask, window_size)
        artifacts["striping"] = striping_mask

    # Detect edge artifacts
    if "edges" in artifact_types:
        edge_mask = _detect_edge_artifacts(data, valid_mask, window_size)
        artifacts["edges"] = edge_mask

    # Detect tile boundary artifacts
    if "tiles" in artifact_types:
        tile_mask = _detect_tile_artifacts(data, valid_mask, window_size)
        artifacts["tiles"] = tile_mask

    # Detect resampling artifacts
    if "resampling" in artifact_types:
        resampling_mask = _detect_resampling_artifacts(data, valid_mask, window_size)
        artifacts["resampling"] = resampling_mask

    return artifacts


def _detect_striping_artifacts(
    data: ArrayLike, valid_mask: ArrayLike, window_size: int
) -> ArrayLike:
    """Detect systematic striping patterns in rows or columns.

    Striping artifacts appear as systematic differences between adjacent
    rows or columns, often caused by sensor calibration issues or
    scan line errors. Detection uses statistical outlier analysis on
    row/column means to identify anomalous stripes.
    """
    import numpy as np
    from scipy.stats import zscore

    # Initialize striping mask
    striping_mask = np.zeros_like(data, dtype=bool)

    # Detect horizontal striping (row-wise patterns)
    # Calculate row means for valid pixels only
    row_means = []
    valid_rows = []

    for i in range(data.shape[0]):
        row_data = data[i, :]
        row_valid = valid_mask[i, :]

        if np.sum(row_valid) > data.shape[1] * 0.1:  # Need at least 10% valid pixels
            row_mean = np.mean(row_data[row_valid])
            row_means.append(row_mean)
            valid_rows.append(i)

    if len(row_means) > 3:  # Need sufficient rows for analysis
        row_means = np.array(row_means)

        # Detect outlier rows using z-score
        row_z_scores = np.abs(zscore(row_means, nan_policy="omit"))
        outlier_rows = np.array(valid_rows)[
            row_z_scores > 2.5
        ]  # Conservative threshold

        # Mark outlier rows as having striping artifacts
        for row_idx in outlier_rows:
            striping_mask[row_idx, :] = valid_mask[row_idx, :]

    # Detect vertical striping (column-wise patterns)
    # Calculate column means for valid pixels only
    col_means = []
    valid_cols = []

    for j in range(data.shape[1]):
        col_data = data[:, j]
        col_valid = valid_mask[:, j]

        if np.sum(col_valid) > data.shape[0] * 0.1:  # Need at least 10% valid pixels
            col_mean = np.mean(col_data[col_valid])
            col_means.append(col_mean)
            valid_cols.append(j)

    if len(col_means) > 3:  # Need sufficient columns for analysis
        col_means = np.array(col_means)

        # Detect outlier columns using z-score
        col_z_scores = np.abs(zscore(col_means, nan_policy="omit"))
        outlier_cols = np.array(valid_cols)[
            col_z_scores > 2.5
        ]  # Conservative threshold

        # Mark outlier columns as having striping artifacts
        for col_idx in outlier_cols:
            striping_mask[:, col_idx] = valid_mask[:, col_idx]

    return striping_mask


def _detect_edge_artifacts(
    data: ArrayLike, valid_mask: ArrayLike, window_size: int
) -> ArrayLike:
    """Detect edge artifacts near data boundaries.

    Edge artifacts occur at the boundaries of processing regions,
    often due to edge effects in filtering or resampling operations.
    """
    import numpy as np
    from scipy.ndimage import sobel

    # Calculate spatial gradients
    data_filled = np.where(valid_mask, data, np.nan)
    data_filled = np.nan_to_num(data_filled, nan=0.0)

    # Compute gradients in both directions
    grad_x = sobel(data_filled, axis=1)
    grad_y = sobel(data_filled, axis=0)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Restore NaN where original data was invalid
    gradient_magnitude[~valid_mask] = np.nan

    # Define edge regions (pixels near boundaries)
    edge_width = max(2, window_size // 2)
    height, width = data.shape

    # Create edge region mask
    edge_region = np.zeros_like(data, dtype=bool)
    edge_region[:edge_width, :] = True  # Top edge
    edge_region[-edge_width:, :] = True  # Bottom edge
    edge_region[:, :edge_width] = True  # Left edge
    edge_region[:, -edge_width:] = True  # Right edge

    # Only consider valid pixels in edge regions
    edge_region &= valid_mask

    # Calculate gradient statistics for edge vs. interior regions
    interior_region = valid_mask & ~edge_region

    if np.sum(edge_region) > 0 and np.sum(interior_region) > 0:
        # Compare gradient distributions
        edge_gradients = gradient_magnitude[edge_region]
        interior_gradients = gradient_magnitude[interior_region]

        # Remove NaN values
        edge_gradients = edge_gradients[~np.isnan(edge_gradients)]
        interior_gradients = interior_gradients[~np.isnan(interior_gradients)]

        if len(edge_gradients) > 0 and len(interior_gradients) > 0:
            # Calculate threshold for edge artifacts
            # Use 95th percentile of interior gradients as reference
            interior_95th = np.percentile(interior_gradients, 95)

            # Flag edge pixels with gradients significantly higher than interior
            threshold = max(
                float(interior_95th) * 2.0,
                float(np.median(interior_gradients))
                + 3 * float(np.std(interior_gradients)),
            )

            # Create edge artifact mask
            edge_artifacts = edge_region & (gradient_magnitude > threshold)
            return edge_artifacts

    # Return empty mask if analysis couldn't be performed
    return np.zeros_like(data, dtype=bool)


def _detect_tile_artifacts(
    data: ArrayLike, valid_mask: ArrayLike, window_size: int
) -> ArrayLike:
    """Detect tile boundary artifacts from mosaicking operations.

    Tile artifacts appear as systematic discontinuities along regular
    grid patterns, typically at tile boundaries in mosaicked data.
    """
    import numpy as np
    from scipy.ndimage import uniform_filter

    # Common tile sizes to check for
    tile_sizes = [256, 512, 1024, 2048]  # Common raster tile sizes

    # Initialize tile artifact mask
    tile_mask = np.zeros_like(data, dtype=bool)

    height, width = data.shape

    # Check for artifacts at regular intervals corresponding to tile boundaries
    for tile_size in tile_sizes:
        # Skip if tile size is larger than data dimensions
        if tile_size >= min(height, width):
            continue

        # Check vertical tile boundaries (columns at multiples of tile_size)
        for col in range(tile_size, width, tile_size):
            if col < width - 1:  # Ensure we don't go out of bounds
                # Compare values on either side of the boundary
                left_col = col - 1
                right_col = col

                # Extract column data
                left_data = data[:, left_col]
                right_data = data[:, right_col]
                left_valid = valid_mask[:, left_col]
                right_valid = valid_mask[:, right_col]

                # Only analyze where both sides are valid
                both_valid = left_valid & right_valid

                if np.sum(both_valid) > height * 0.1:  # Need sufficient valid pixels
                    # Calculate differences across boundary
                    differences = np.abs(left_data[both_valid] - right_data[both_valid])

                    # Check if differences are systematically large
                    if len(differences) > 0:
                        median_diff = np.median(differences)
                        data_range = np.percentile(data[valid_mask], [5, 95])
                        expected_diff = (
                            data_range[1] - data_range[0]
                        ) * 0.05  # 5% of data range

                        # Flag as tile artifact if median difference is large
                        if median_diff > max(
                            expected_diff, np.std(data[valid_mask]) * 2
                        ):
                            # Mark pixels near this boundary
                            boundary_width = min(3, window_size // 2)
                            start_col = max(0, col - boundary_width)
                            end_col = min(width, col + boundary_width + 1)
                            tile_mask[:, start_col:end_col] |= valid_mask[
                                :, start_col:end_col
                            ]

        # Check horizontal tile boundaries (rows at multiples of tile_size)
        for row in range(tile_size, height, tile_size):
            if row < height - 1:  # Ensure we don't go out of bounds
                # Compare values on either side of the boundary
                top_row = row - 1
                bottom_row = row

                # Extract row data
                top_data = data[top_row, :]
                bottom_data = data[bottom_row, :]
                top_valid = valid_mask[top_row, :]
                bottom_valid = valid_mask[bottom_row, :]

                # Only analyze where both sides are valid
                both_valid = top_valid & bottom_valid

                if np.sum(both_valid) > width * 0.1:  # Need sufficient valid pixels
                    # Calculate differences across boundary
                    differences = np.abs(top_data[both_valid] - bottom_data[both_valid])

                    # Check if differences are systematically large
                    if len(differences) > 0:
                        median_diff = np.median(differences)
                        data_range = np.percentile(data[valid_mask], [5, 95])
                        expected_diff = (
                            data_range[1] - data_range[0]
                        ) * 0.05  # 5% of data range

                        # Flag as tile artifact if median difference is large
                        if median_diff > max(
                            expected_diff, np.std(data[valid_mask]) * 2
                        ):
                            # Mark pixels near this boundary
                            boundary_width = min(3, window_size // 2)
                            start_row = max(0, row - boundary_width)
                            end_row = min(height, row + boundary_width + 1)
                            tile_mask[start_row:end_row, :] |= valid_mask[
                                start_row:end_row, :
                            ]

    return tile_mask


def _detect_resampling_artifacts(
    data: ArrayLike, valid_mask: ArrayLike, window_size: int
) -> ArrayLike:
    """Detect resampling artifacts like checkerboard patterns or aliasing.

    Resampling artifacts appear as regular patterns that don't match
    the natural spatial structure of the data, often with characteristic
    frequencies related to the resampling operation.
    """
    import numpy as np
    from scipy.ndimage import generic_filter

    # Initialize resampling artifact mask
    resampling_mask = np.zeros_like(data, dtype=bool)

    # Detect checkerboard patterns using local variance analysis
    # Checkerboard patterns have high local variance in regular patterns

    def local_variance_pattern(patch):
        """Calculate variance pattern indicator for local patch.
        
        This function detects checkerboard-like patterns that are
        characteristic of resampling artifacts by analyzing the
        variance structure within local neighborhoods.
        """
        if len(patch) < 4:  # Need minimum patch size
            return 0.0

        # Reshape patch to 2D if possible (assuming square window)
        patch_size = int(np.sqrt(len(patch)))
        if patch_size * patch_size == len(patch):
            patch_2d = patch.reshape(patch_size, patch_size)

            # Check for checkerboard pattern
            # In checkerboard, alternating pixels have very different values
            if patch_size >= 2:
                # Extract alternating patterns
                even_even = patch_2d[::2, ::2].flatten()  # Even rows, even cols
                even_odd = patch_2d[::2, 1::2].flatten()  # Even rows, odd cols
                odd_even = patch_2d[1::2, ::2].flatten()  # Odd rows, even cols
                odd_odd = patch_2d[1::2, 1::2].flatten()  # Odd rows, odd cols

                # Remove NaN values
                patterns = [even_even, even_odd, odd_even, odd_odd]
                patterns = [p[~np.isnan(p)] for p in patterns]
                patterns = [p for p in patterns if len(p) > 0]

                if len(patterns) >= 2:
                    # Calculate variance between alternating patterns
                    pattern_means = [np.mean(p) for p in patterns]
                    if len(pattern_means) >= 2:
                        pattern_var = np.var(pattern_means)
                        return pattern_var

        return np.var(patch[~np.isnan(patch)]) if np.sum(~np.isnan(patch)) > 1 else 0.0

    # Apply local pattern analysis
    pattern_scores = generic_filter(
        data,
        function=local_variance_pattern,
        size=window_size,
        mode="constant",
        cval=np.nan,
    )

    # Restore NaN for invalid areas
    pattern_scores[~valid_mask] = np.nan

    # Identify areas with unusually high pattern scores
    valid_scores = pattern_scores[valid_mask & ~np.isnan(pattern_scores)]

    if len(valid_scores) > 10:  # Need sufficient data for statistics
        # Use robust threshold based on distribution
        score_75th = np.percentile(valid_scores, 75)
        score_99th = np.percentile(valid_scores, 99)

        # Conservative threshold to avoid false positives
        threshold = score_75th + 2 * (score_99th - score_75th)

        # Flag pixels with high pattern scores as potential resampling artifacts
        resampling_mask = valid_mask & (pattern_scores > threshold)

    return resampling_mask


def generate_quality_report(
    quality_assessments: list[dict[str, Any]],
    output_format: Literal["summary", "detailed", "json"] = "summary",
) -> str | dict[str, Any]:
    """Generate comprehensive quality report from assessments.

    This function consolidates multiple quality assessments into a
    human-readable report suitable for documentation, debugging, or
    quality assurance workflows.

    The report provides actionable insights about data quality issues,
    processing artifacts, and recommendations for improving results.

    Args:
        quality_assessments: List of assessment dictionaries from assess_data_quality().
            Each dictionary should contain quality metrics, warnings, and recommendations.
            Required keys: 'valid_pixel_percent', 'quality_score', 'warnings', 'recommendations'
        output_format: Report format:
            - 'summary': Brief text summary suitable for logs
            - 'detailed': Full text report with analysis
            - 'json': Structured JSON output for programmatic use

    Returns:
        Quality report as string (for 'summary'/'detailed') or dictionary (for 'json').

    Raises:
        ValueError: If quality_assessments is empty or contains invalid data.

    Example:
        >>> # Generate quality report from multiple assessments
        >>> assessments = [
        >>>     assess_data_quality(band1_data),
        >>>     assess_data_quality(band2_data)
        >>> ]
        >>> report = generate_quality_report(assessments, output_format='detailed')
        >>> print(report)

    Note:
        The report format is designed to be human-readable while providing
        actionable information for quality improvement workflows.
    """
    import json
    from datetime import datetime

    # Input validation
    if not quality_assessments:
        raise ValueError("Quality assessments list cannot be empty")

    # Validate required keys in each assessment
    required_keys = {
        "valid_pixel_percent",
        "quality_score",
        "warnings",
        "recommendations",
    }
    for i, assessment in enumerate(quality_assessments):
        missing_keys = required_keys - set(assessment.keys())
        if missing_keys:
            raise ValueError(f"Assessment {i} missing required keys: {missing_keys}")

    # Aggregate statistics across all assessments
    n_assessments = len(quality_assessments)

    # Calculate overall statistics
    valid_percentages = [a["valid_pixel_percent"] for a in quality_assessments]
    quality_scores = [a["quality_score"] for a in quality_assessments]

    overall_stats = {
        "total_assessments": n_assessments,
        "mean_valid_coverage": sum(valid_percentages) / n_assessments,
        "min_valid_coverage": min(valid_percentages),
        "max_valid_coverage": max(valid_percentages),
        "mean_quality_score": sum(quality_scores) / n_assessments,
        "min_quality_score": min(quality_scores),
        "max_quality_score": max(quality_scores),
        "assessments_with_poor_quality": sum(
            1 for score in quality_scores if score < 50
        ),
        "assessments_with_excellent_quality": sum(
            1 for score in quality_scores if score >= 80
        ),
    }

    # Aggregate warnings and recommendations
    all_warnings = []
    all_recommendations = []
    warning_counts = {}
    recommendation_counts = {}

    for assessment in quality_assessments:
        for warning in assessment.get("warnings", []):
            all_warnings.append(warning)
            warning_counts[warning] = warning_counts.get(warning, 0) + 1

        for recommendation in assessment.get("recommendations", []):
            all_recommendations.append(recommendation)
            recommendation_counts[recommendation] = (
                recommendation_counts.get(recommendation, 0) + 1
            )

    # Identify most common issues
    most_common_warnings = sorted(
        warning_counts.items(), key=lambda x: x[1], reverse=True
    )[:5]
    most_common_recommendations = sorted(
        recommendation_counts.items(), key=lambda x: x[1], reverse=True
    )[:5]

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Generate report based on format
    if output_format == "json":
        return {
            "timestamp": timestamp,
            "overall_statistics": overall_stats,
            "individual_assessments": quality_assessments,
            "warning_summary": {
                "total_warnings": len(all_warnings),
                "unique_warnings": len(warning_counts),
                "most_common": most_common_warnings,
            },
            "recommendation_summary": {
                "total_recommendations": len(all_recommendations),
                "unique_recommendations": len(recommendation_counts),
                "most_common": most_common_recommendations,
            },
            "quality_classification": _classify_overall_quality(overall_stats),
        }

    elif output_format == "summary":
        return _generate_summary_report(overall_stats, most_common_warnings, timestamp)

    elif output_format == "detailed":
        return _generate_detailed_report(
            overall_stats,
            quality_assessments,
            most_common_warnings,
            most_common_recommendations,
            timestamp,
        )

    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def _classify_overall_quality(stats: dict[str, Any]) -> dict[str, str]:
    """Classify overall quality based on aggregated statistics."""
    mean_score = stats["mean_quality_score"]
    min_score = stats["min_quality_score"]
    poor_count = stats["assessments_with_poor_quality"]
    total_count = stats["total_assessments"]

    # Overall quality classification
    if mean_score >= 80 and min_score >= 60:
        overall_quality = "Excellent"
        confidence = "High"
    elif mean_score >= 60 and poor_count / total_count <= 0.2:
        overall_quality = "Good"
        confidence = "Medium"
    elif mean_score >= 40:
        overall_quality = "Fair"
        confidence = "Medium"
    else:
        overall_quality = "Poor"
        confidence = "Low"

    # Coverage assessment
    mean_coverage = stats["mean_valid_coverage"]
    if mean_coverage >= 90:
        coverage_quality = "Excellent"
    elif mean_coverage >= 75:
        coverage_quality = "Good"
    elif mean_coverage >= 50:
        coverage_quality = "Fair"
    else:
        coverage_quality = "Poor"

    return {
        "overall_quality": overall_quality,
        "confidence_level": confidence,
        "coverage_quality": coverage_quality,
        "recommendation": _get_quality_recommendation(
            overall_quality, coverage_quality
        ),
    }


def _get_quality_recommendation(overall_quality: str, coverage_quality: str) -> str:
    """Generate actionable recommendation based on quality classification."""
    if overall_quality == "Excellent" and coverage_quality == "Excellent":
        return "Data is suitable for all analysis types with high confidence."
    elif overall_quality == "Good":
        return "Data is suitable for most analyses. Monitor quality metrics in final products."
    elif overall_quality == "Fair":
        return "Data is usable with caution. Apply additional quality filters and validation."
    else:
        return "Data quality is insufficient for reliable analysis. Investigate processing issues."


def _generate_summary_report(
    stats: dict[str, Any], warnings: list[tuple[str, int]], timestamp: str
) -> str:
    """Generate brief summary report."""
    classification = _classify_overall_quality(stats)

    summary = f"""Quality Assessment Summary ({timestamp})
===========================================

Overall Quality: {classification["overall_quality"]} ({classification["confidence_level"]} confidence)
Coverage Quality: {classification["coverage_quality"]}

Key Statistics:
- Total assessments: {stats["total_assessments"]}
- Mean quality score: {stats["mean_quality_score"]:.1f}/100
- Mean valid coverage: {stats["mean_valid_coverage"]:.1f}%
- Poor quality assessments: {stats["assessments_with_poor_quality"]}/{stats["total_assessments"]}

"""

    if warnings:
        summary += "Most Common Issues:\n"
        for warning, count in warnings[:3]:
            summary += f"- {warning} ({count} occurrences)\n"

    summary += f"\nRecommendation: {classification['recommendation']}"

    return summary


def _generate_detailed_report(
    stats: dict[str, Any],
    assessments: list[dict[str, Any]],
    warnings: list[tuple[str, int]],
    recommendations: list[tuple[str, int]],
    timestamp: str,
) -> str:
    """Generate comprehensive detailed report."""
    classification = _classify_overall_quality(stats)

    report = f"""Comprehensive Quality Assessment Report
====================================
Generated: {timestamp}

EXECUTIVE SUMMARY
================
Overall Quality: {classification["overall_quality"]} ({classification["confidence_level"]} confidence)
Coverage Quality: {classification["coverage_quality"]}
Recommendation: {classification["recommendation"]}

AGGREGATE STATISTICS
===================
Number of assessments: {stats["total_assessments"]}

Quality Scores:
- Mean: {stats["mean_quality_score"]:.1f}/100
- Range: {stats["min_quality_score"]:.1f} - {stats["max_quality_score"]:.1f}
- Poor quality (< 50): {stats["assessments_with_poor_quality"]} assessments
- Excellent quality (≥ 80): {stats["assessments_with_excellent_quality"]} assessments

Valid Data Coverage:
- Mean: {stats["mean_valid_coverage"]:.1f}%
- Range: {stats["min_valid_coverage"]:.1f}% - {stats["max_valid_coverage"]:.1f}%

DETAILED ASSESSMENT RESULTS
============================
"""

    # Add individual assessment details
    for i, assessment in enumerate(assessments):
        report += f"\nAssessment {i + 1}:\n"
        report += f"  Quality Score: {assessment['quality_score']:.1f}/100\n"
        report += f"  Valid Coverage: {assessment['valid_pixel_percent']:.1f}%\n"
        report += f"  Valid Pixels: {assessment.get('valid_pixel_count', 'N/A')}\n"

        if "mean" in assessment:
            report += f"  Mean Value: {assessment['mean']:.3f}\n"
            report += f"  Std Dev: {assessment.get('std', 0):.3f}\n"

        if assessment.get("outlier_count", 0) > 0:
            report += f"  Outliers: {assessment['outlier_count']} pixels\n"

        if assessment.get("warnings"):
            report += f"  Warnings: {', '.join(assessment['warnings'][:2])}\n"

    # Add issue analysis
    report += f"\nISSUE ANALYSIS\n==============\n"

    if warnings:
        report += "Most Common Warnings:\n"
        for warning, count in warnings[:5]:
            percentage = (count / stats["total_assessments"]) * 100
            report += f"  • {warning}: {count} assessments ({percentage:.1f}%)\n"

    if recommendations:
        report += "\nMost Common Recommendations:\n"
        for recommendation, count in recommendations[:5]:
            percentage = (count / stats["total_assessments"]) * 100
            report += f"  • {recommendation}: {count} assessments ({percentage:.1f}%)\n"

    # Add actionable insights
    report += f"\nACTIONABLE INSIGHTS\n===================\n"

    if stats["assessments_with_poor_quality"] > 0:
        poor_percentage = (
            stats["assessments_with_poor_quality"] / stats["total_assessments"]
        ) * 100
        report += f"• {poor_percentage:.1f}% of assessments have poor quality - investigate processing pipeline\n"

    if stats["min_valid_coverage"] < 50:
        report += f"• Minimum coverage is only {stats['min_valid_coverage']:.1f}% - consider gap filling or additional data\n"

    if (
        len(set(assessment["quality_score"] for assessment in assessments))
        > stats["total_assessments"] * 0.3
    ):
        report += "• High variability in quality scores - inconsistent processing or input data quality\n"

    return report


def calculate_quality_weighted_composite(
    data_stack: ArrayLike,
    quality_scores: ArrayLike,
    method: Literal["weighted_mean", "best_quality", "threshold"] = "weighted_mean",
    threshold: float | None = None,
) -> tuple[ArrayLike, ArrayLike]:
    """Create composite using quality scores for weighting.

    This function implements quality-aware compositing where observations
    with higher quality scores contribute more to the final composite.
    This is superior to simple median compositing for datasets with
    varying quality because it emphasizes high-quality observations
    while de-emphasizing or excluding poor-quality data.

    The compositing process preserves spatial structure while optimally
    combining temporal observations based on their individual quality
    assessment results.

    Args:
        data_stack: 3D array of observations (time, height, width).
            Each time slice represents one observation period.
            NaN values indicate missing or invalid data.
        quality_scores: 3D array of quality scores matching data_stack shape.
            Scores should be in [0, 1] range where 1.0 = highest quality.
            Can also accept (0, 100) range which will be normalized.
        method: Compositing method to use:
            - 'weighted_mean': Quality-weighted average (recommended)
            - 'best_quality': Select observation with highest quality score
            - 'threshold': Use only observations above quality threshold
        threshold: Quality threshold for 'threshold' method (0-1 or 0-100).
            If None and method='threshold', uses 0.5 (or 50 for 0-100 range).
            Observations below threshold are excluded from compositing.

    Returns:
        Tuple of (composite, composite_quality) where:
        - composite: Quality-weighted composite array (height, width)
        - composite_quality: Effective quality scores for composite (height, width)
          Represents the reliability of the composite at each pixel

    Raises:
        ValueError: If input arrays have incompatible shapes or invalid parameters.

    Example:
        >>> # Create quality-weighted composite from monthly data
        >>> monthly_composite, composite_quality = calculate_quality_weighted_composite(
        >>>     data_stack=monthly_ndvi,        # (12, 1000, 1000)
        >>>     quality_scores=monthly_quality, # (12, 1000, 1000)
        >>>     method='weighted_mean'
        >>> )
        >>> # Mask low-quality areas in final product
        >>> reliable_areas = composite_quality > 0.6
        >>> final_product = np.where(reliable_areas, monthly_composite, np.nan)

    Note:
        This is the preferred method for creating temporal composites
        when quality information is available. It produces better results
        than simple statistical compositing by leveraging quality metadata.
    """
    import numpy as np

    # Input validation
    if data_stack.ndim != 3:
        raise ValueError(
            f"Data stack must be 3D (time, height, width), got {data_stack.ndim}D"
        )

    if quality_scores.shape != data_stack.shape:
        raise ValueError(
            f"Quality scores shape {quality_scores.shape} must match data stack shape {data_stack.shape}"
        )

    n_times, height, width = data_stack.shape

    # Normalize quality scores to [0, 1] range if they appear to be in [0, 100] range
    max_quality = np.nanmax(quality_scores)
    if max_quality > 1.0:
        # Assume scores are in [0, 100] range
        quality_normalized = quality_scores / 100.0
        quality_range_100 = True
    else:
        quality_normalized = quality_scores.copy()
        quality_range_100 = False

    # Set default threshold if needed
    if method == "threshold" and threshold is None:
        threshold = 50.0 if quality_range_100 else 0.5
    elif threshold is not None and quality_range_100 and threshold <= 1.0:
        # Convert threshold to 0-100 range if needed
        threshold = threshold * 100.0
    elif threshold is not None and not quality_range_100 and threshold > 1.0:
        # Convert threshold to 0-1 range if needed
        threshold = threshold / 100.0

    # Validate threshold
    if threshold is not None:
        expected_range = "0-100" if quality_range_100 else "0-1"
        max_threshold = 100.0 if quality_range_100 else 1.0
        if threshold < 0 or threshold > max_threshold:
            raise ValueError(
                f"Threshold must be in {expected_range} range, got {threshold}"
            )

    # Initialize output arrays
    composite = np.full((height, width), np.nan, dtype=np.float32)
    composite_quality = np.full((height, width), np.nan, dtype=np.float32)

    # Process each pixel independently for maximum flexibility
    # This pixel-wise approach allows handling of irregular temporal coverage
    # and pixel-specific quality variations
    for i in range(height):
        for j in range(width):
            # Extract time series for this pixel
            pixel_data = data_stack[:, i, j]
            pixel_quality = quality_normalized[:, i, j]

            # Create mask for valid observations (not NaN in either data or quality)
            valid_mask = ~np.isnan(pixel_data) & ~np.isnan(pixel_quality)

            if not np.any(valid_mask):
                # No valid observations for this pixel
                continue

            valid_data = pixel_data[valid_mask]
            valid_quality = pixel_quality[valid_mask]

            # Apply quality-based compositing
            if method == "weighted_mean":
                pixel_composite, pixel_composite_quality = _weighted_mean_composite(
                    valid_data, valid_quality
                )

            elif method == "best_quality":
                pixel_composite, pixel_composite_quality = _best_quality_composite(
                    valid_data, valid_quality
                )

            elif method == "threshold":
                # Convert threshold to normalized range for comparison
                assert threshold is not None  # Should be set by validation above
                threshold_normalized = (
                    threshold / 100.0 if quality_range_100 else threshold
                )
                pixel_composite, pixel_composite_quality = _threshold_composite(
                    valid_data, valid_quality, threshold_normalized
                )

            else:
                raise ValueError(f"Unsupported compositing method: {method}")

            # Store results
            composite[i, j] = pixel_composite
            composite_quality[i, j] = pixel_composite_quality

    return composite, composite_quality


def _weighted_mean_composite(
    data: ArrayLike, quality: ArrayLike
) -> tuple[float, float]:
    """Compute quality-weighted mean for a single pixel time series.

    Uses quality scores as weights to compute a weighted average where
    higher quality observations contribute more to the final result.
    """
    import numpy as np

    # Ensure quality scores are non-negative
    quality_weights = np.maximum(quality, 0.0)

    # Avoid division by zero
    total_weight = np.sum(quality_weights)
    if total_weight == 0:
        return np.nan, np.nan

    # Calculate weighted mean
    weighted_sum = np.sum(data * quality_weights)
    composite_value = weighted_sum / total_weight

    # Calculate effective quality as weighted average of input qualities
    # This represents how reliable the composite is based on input quality distribution
    composite_quality = np.sum(quality * quality_weights) / total_weight

    return float(composite_value), float(composite_quality)


def _best_quality_composite(data: ArrayLike, quality: ArrayLike) -> tuple[float, float]:
    """Select observation with highest quality score for a single pixel.

    This method simply selects the observation with the highest quality
    score, which is appropriate when quality differences are large.
    """
    import numpy as np

    # Find observation with maximum quality
    best_idx = np.argmax(quality)

    composite_value = data[best_idx]
    composite_quality = quality[best_idx]

    return float(composite_value), float(composite_quality)


def _threshold_composite(
    data: ArrayLike, quality: ArrayLike, threshold: float
) -> tuple[float, float]:
    """Compute mean using only observations above quality threshold.

    This method excludes low-quality observations and computes a simple
    mean of the remaining high-quality observations.
    """
    import numpy as np

    # Apply quality threshold filter
    above_threshold = quality >= threshold

    if not np.any(above_threshold):
        # No observations meet the quality threshold
        return np.nan, np.nan

    # Use only high-quality observations
    high_quality_data = data[above_threshold]
    high_quality_scores = quality[above_threshold]

    # Compute simple mean of high-quality observations
    composite_value = np.mean(high_quality_data)

    # Composite quality is the mean quality of observations used
    composite_quality = np.mean(high_quality_scores)

    return float(composite_value), float(composite_quality)
