"""Output validation and quality assurance utilities.

This module provides comprehensive validation functions to ensure data products
meet quality standards before export. These validations are critical for
maintaining data integrity and catching processing errors early.

The validation pipeline serves as the final quality gate before data is
delivered to users, ensuring that all products meet specifications for
spatial alignment, data ranges, and format compliance.
"""

from typing import Any, Literal
import numpy.typing as npt

# Type aliases
ArrayLike = npt.NDArray[Any]
Transform = Any  # rasterio.transform.Affine
CRS = Any  # rasterio.crs.CRS
BBox = tuple[float, float, float, float]


def validate_for_cog(
    data: ArrayLike | str,
    transform: Transform | None = None,
    crs: CRS | None = None,
    check_tiling: bool = True,
    check_compression: bool = True,
    check_overviews: bool = True
) -> dict[str, Any]:
    """Validate that data meets Cloud-Optimized GeoTIFF requirements.
    
    This function performs comprehensive checks to ensure data can be
    successfully exported as a COG. It validates spatial referencing,
    data structure, and COG-specific requirements like tiling and overviews.
    This is a critical step before export to prevent invalid COG generation.
    
    Args:
        data: Data array or path to existing raster
        transform: Affine transform (required if data is array)
        crs: Coordinate reference system (required if data is array)
        check_tiling: Verify proper internal tiling structure
        check_compression: Check for valid compression settings
        check_overviews: Verify overview levels are appropriate
        
    Returns:
        Validation results dictionary containing:
        - valid: bool indicating if all checks passed
        - errors: List of validation errors found
        - warnings: List of non-critical issues
        - recommendations: Suggestions for optimization
        - statistics: Data statistics for quality assessment
        
    Note:
        This validation should be run before create_cog() to catch
        issues early and provide actionable feedback to users.
    """
    import numpy as np
    import rasterio
    import os
    
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "recommendations": [],
        "statistics": {}
    }
    
    if isinstance(data, str):
        # Validate existing file
        if not os.path.exists(data):
            result["valid"] = False
            result["errors"].append(f"File not found: {data}")
            return result
        
        try:
            with rasterio.open(data) as src:
                # Basic file info
                result["statistics"]["width"] = src.width
                result["statistics"]["height"] = src.height
                result["statistics"]["bands"] = src.count
                result["statistics"]["dtype"] = str(src.dtypes[0])
                result["statistics"]["crs"] = str(src.crs) if src.crs else None
                
                # Check CRS
                if not src.crs:
                    result["errors"].append("No coordinate reference system defined")
                    result["valid"] = False
                
                # Check transform
                if not src.transform or src.transform == rasterio.Affine.identity():
                    result["errors"].append("No valid geotransform defined")
                    result["valid"] = False
                
                # Check tiling
                if check_tiling:
                    if src.is_tiled:
                        blocksize = src.block_shapes[0]
                        result["statistics"]["tiled"] = True
                        result["statistics"]["blocksize"] = blocksize
                        
                        # Check block size
                        if blocksize[0] != blocksize[1]:
                            result["warnings"].append(f"Non-square tiles: {blocksize}")
                        elif blocksize[0] not in [256, 512, 1024]:
                            result["warnings"].append(f"Non-standard tile size: {blocksize[0]}")
                            result["recommendations"].append("Use 512x512 tiles for optimal cloud performance")
                    else:
                        result["statistics"]["tiled"] = False
                        result["warnings"].append("File is not tiled")
                        result["recommendations"].append("Enable tiling for cloud-optimized access")
                
                # Check compression
                if check_compression:
                    compression = src.compression
                    result["statistics"]["compression"] = compression.name if compression else "none"
                    
                    if not compression:
                        result["warnings"].append("No compression applied")
                        result["recommendations"].append("Use DEFLATE compression for optimal COG performance")
                    elif compression.name in ['JPEG', 'WEBP']:
                        result["errors"].append(f"Lossy compression '{compression.name}' not suitable for scientific data")
                        result["valid"] = False
                
                # Check overviews
                overviews = []
                if check_overviews:
                    overviews = src.overviews(1) if src.count > 0 else []
                    result["statistics"]["has_overviews"] = len(overviews) > 0
                    result["statistics"]["overview_levels"] = overviews
                    
                    if not overviews:
                        result["warnings"].append("No overviews present")
                        result["recommendations"].append("Add overviews (2, 4, 8, 16) for multi-resolution access")
                    else:
                        # Check for standard levels
                        standard_levels = [2, 4, 8, 16]
                        missing_levels = [l for l in standard_levels if l not in overviews]
                        if missing_levels:
                            result["recommendations"].append(f"Consider adding overview levels: {missing_levels}")
                
                # Check for COG layout
                # In a true COG, IFDs should be ordered: main image, then overviews
                # This is a simplified check
                if src.is_tiled and overviews:
                    result["statistics"]["likely_cog"] = True
                else:
                    result["statistics"]["likely_cog"] = False
                    if not result["errors"]:
                        result["recommendations"].append("Convert to Cloud-Optimized GeoTIFF for best performance")
                
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Error reading file: {str(e)}")
            
    else:
        # Validate array data
        data = np.asarray(data)
        
        # Check array properties
        if data.ndim not in [2, 3]:
            result["valid"] = False
            result["errors"].append(f"Invalid array dimensions: {data.ndim}. Expected 2D or 3D array.")
            return result
        
        # Check for required metadata
        if transform is None:
            result["valid"] = False
            result["errors"].append("Transform is required for array data")
        
        if crs is None:
            result["valid"] = False
            result["errors"].append("CRS is required for array data")
        
        # Statistics
        if data.ndim == 2:
            result["statistics"]["width"] = data.shape[1]
            result["statistics"]["height"] = data.shape[0]
            result["statistics"]["bands"] = 1
        else:
            result["statistics"]["width"] = data.shape[2]
            result["statistics"]["height"] = data.shape[1]
            result["statistics"]["bands"] = data.shape[0]
        
        result["statistics"]["dtype"] = str(data.dtype)
        
        # Check data type
        if data.dtype == np.float64:
            result["warnings"].append("Float64 data will be converted to Float32 to save space")
        
        # Check for NaN or infinite values
        if np.issubdtype(data.dtype, np.floating):
            if np.any(np.isnan(data)):
                nan_percent = (np.sum(np.isnan(data)) / data.size) * 100
                result["statistics"]["nan_percent"] = nan_percent
                if nan_percent > 50:
                    result["warnings"].append(f"High percentage of NaN values: {nan_percent:.1f}%")
            
            if np.any(np.isinf(data)):
                result["errors"].append("Infinite values detected in data")
                result["valid"] = False
        
        # Check data size
        size_mb = data.nbytes / (1024 * 1024)
        result["statistics"]["uncompressed_size_mb"] = size_mb
        
        if size_mb > 1000:
            result["warnings"].append(f"Large uncompressed size: {size_mb:.1f} MB")
            result["recommendations"].append("Consider using compression and tiling")
        
        # Array-specific recommendations
        result["recommendations"].append("Will create COG with DEFLATE compression and 512x512 tiles")
    
    return result


def validate_data_range(
    data: ArrayLike | str,
    expected_min: float | None = None,
    expected_max: float | None = None,
    valid_range: tuple[float, float] | None = None,
    nodata_value: float | None = None,
    check_distribution: bool = True
) -> dict[str, Any]:
    """Validate that data values fall within expected ranges.
    
    This function ensures data values are physically meaningful and
    within expected bounds. For Black Marble, this includes checking
    that NDVI is in [-1, 1], NDUI is in [0, 1], and radiance values
    are non-negative. This validation prevents nonsensical outputs
    from propagating to final products.
    
    Args:
        data: Data array or path to raster
        expected_min: Expected minimum value (inclusive)
        expected_max: Expected maximum value (inclusive)
        valid_range: Alternative tuple specification of (min, max)
        nodata_value: Value representing no data
        check_distribution: Analyze value distribution for anomalies
        
    Returns:
        Validation results containing:
        - valid: bool indicating if data is within range
        - actual_min: Observed minimum value
        - actual_max: Observed maximum value
        - out_of_range_count: Number of pixels outside range
        - out_of_range_percent: Percentage of pixels outside range
        - distribution_stats: Histogram and percentiles if requested
        - anomalies: Detected distribution anomalies
        
    Example:
        >>> # Validate NDVI is in proper range
        >>> result = validate_data_range(ndvi_data, -1.0, 1.0)
        >>> if not result["valid"]:
        >>>     print(f"Warning: {result['out_of_range_percent']:.1f}% pixels out of range")
    """
    import numpy as np
    import rasterio
    from scipy import stats
    
    # Load data
    if isinstance(data, str):
        with rasterio.open(data) as src:
            # Read first band for validation
            data_array = src.read(1)
            if nodata_value is None:
                nodata_value = src.nodata
    else:
        data_array = np.asarray(data)
        if data_array.ndim == 3:
            # Use first band for multi-band data
            data_array = data_array[0]
    
    # Set expected range
    if valid_range is not None:
        expected_min, expected_max = valid_range
    
    # Create mask for valid data
    valid_mask = np.ones_like(data_array, dtype=bool)
    
    # Exclude nodata
    if nodata_value is not None:
        valid_mask &= (data_array != nodata_value)
    
    # Exclude NaN
    if np.issubdtype(data_array.dtype, np.floating):
        valid_mask &= ~np.isnan(data_array)
    
    # Get valid data
    valid_data = data_array[valid_mask]
    
    result = {
        "valid": True,
        "total_pixels": data_array.size,
        "valid_pixels": len(valid_data),
        "anomalies": []
    }
    
    if len(valid_data) == 0:
        result["valid"] = False
        result["actual_min"] = None
        result["actual_max"] = None
        result["out_of_range_count"] = 0
        result["out_of_range_percent"] = 0
        result["anomalies"].append("No valid data found")
        return result
    
    # Calculate actual range
    actual_min = float(np.min(valid_data))
    actual_max = float(np.max(valid_data))
    
    result["actual_min"] = actual_min
    result["actual_max"] = actual_max
    
    # Check range if specified
    if expected_min is not None or expected_max is not None:
        out_of_range_mask = np.zeros_like(valid_data, dtype=bool)
        
        if expected_min is not None:
            out_of_range_mask |= (valid_data < expected_min)
            if actual_min < expected_min:
                result["valid"] = False
        
        if expected_max is not None:
            out_of_range_mask |= (valid_data > expected_max)
            if actual_max > expected_max:
                result["valid"] = False
        
        out_of_range_count = int(np.sum(out_of_range_mask))
        out_of_range_percent = (out_of_range_count / len(valid_data)) * 100
        
        result["out_of_range_count"] = out_of_range_count
        result["out_of_range_percent"] = out_of_range_percent
        
        # Tolerance for small percentage out of range
        if out_of_range_percent > 0 and out_of_range_percent < 0.1:
            result["valid"] = True  # Allow small percentage
            result["anomalies"].append(f"Small percentage ({out_of_range_percent:.3f}%) out of range - likely rounding errors")
    else:
        result["out_of_range_count"] = 0
        result["out_of_range_percent"] = 0
    
    # Check distribution if requested
    if check_distribution and len(valid_data) > 100:
        result["distribution_stats"] = {}
        
        # Basic statistics
        result["distribution_stats"]["mean"] = float(np.mean(valid_data))
        result["distribution_stats"]["std"] = float(np.std(valid_data))
        result["distribution_stats"]["median"] = float(np.median(valid_data))
        
        # Percentiles
        percentiles = [1, 5, 25, 50, 75, 95, 99]
        result["distribution_stats"]["percentiles"] = {}
        for p in percentiles:
            result["distribution_stats"]["percentiles"][f"p{p}"] = float(np.percentile(valid_data, p))
        
        # Distribution tests
        # Check for bimodal distribution
        hist, bins = np.histogram(valid_data, bins=50)
        hist_smooth = np.convolve(hist, np.ones(3)/3, mode='same')  # Smooth histogram
        
        # Find peaks
        peaks = []
        for i in range(1, len(hist_smooth)-1):
            if hist_smooth[i] > hist_smooth[i-1] and hist_smooth[i] > hist_smooth[i+1]:
                if hist_smooth[i] > np.max(hist_smooth) * 0.2:  # Significant peak
                    peaks.append(i)
        
        if len(peaks) > 1:
            result["anomalies"].append("Bimodal distribution detected")
        
        # Check for extreme skewness
        skewness = stats.skew(valid_data)
        if abs(skewness) > 2:
            result["anomalies"].append(f"Highly skewed distribution (skewness={skewness:.2f})")
        
        result["distribution_stats"]["skewness"] = float(skewness)
        
        # Check for outliers using IQR method
        q1 = np.percentile(valid_data, 25)
        q3 = np.percentile(valid_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = np.sum((valid_data < lower_bound) | (valid_data > upper_bound))
        outlier_percent = (outliers / len(valid_data)) * 100
        
        if outlier_percent > 5:
            result["anomalies"].append(f"High percentage of outliers ({outlier_percent:.1f}%)")
        
        result["distribution_stats"]["outlier_percent"] = outlier_percent
        
        # Check specific ranges for common indices
        if expected_min == -1 and expected_max == 1:
            # Likely NDVI
            if result["distribution_stats"]["mean"] < -0.5:
                result["anomalies"].append("Unusually low NDVI values - check for water/snow")
            elif result["distribution_stats"]["mean"] > 0.7:
                result["anomalies"].append("Unusually high NDVI values - check for saturation")
    
    return result


def validate_spatial_alignment(
    reference: str | Transform,
    candidates: list[str] | list[Transform],
    tolerance: float = 1e-6,
    check_resolution: bool = True,
    check_bounds: bool = True,
    check_crs: bool = True
) -> dict[str, Any]:
    """Validate spatial alignment between multiple rasters.
    
    This function ensures that multiple data layers are properly aligned
    for analysis. In the Black Marble pipeline, Landsat, VIIRS, and OSM
    data must be perfectly aligned after the prepare phase. Misalignment
    would cause incorrect index calculations and visual artifacts.
    
    Args:
        reference: Reference raster path or transform
        candidates: List of rasters to check against reference
        tolerance: Maximum allowed difference in coordinates
        check_resolution: Verify matching pixel sizes
        check_bounds: Verify overlapping spatial extents
        check_crs: Verify matching coordinate systems
        
    Returns:
        Alignment report containing:
        - aligned: bool indicating if all rasters align
        - misaligned: List of rasters that don't align
        - alignment_errors: Specific alignment issues found
        - transform_deltas: Numerical differences in transforms
        - recommendations: How to fix alignment issues
        
    Note:
        This validation is critical after the prepare phase to ensure
        all data layers can be properly combined in analysis phase.
    """
    import rasterio
    import numpy as np
    from rasterio.transform import Affine
    
    result = {
        "aligned": True,
        "misaligned": [],
        "alignment_errors": {},
        "transform_deltas": {},
        "recommendations": []
    }
    
    # Get reference metadata
    if isinstance(reference, str):
        with rasterio.open(reference) as src:
            ref_transform = src.transform
            ref_crs = src.crs
            ref_bounds = src.bounds
            ref_width = src.width
            ref_height = src.height
            ref_name = reference
    else:
        # Assume Transform object
        ref_transform = reference
        ref_crs = None
        ref_bounds = None
        ref_width = None
        ref_height = None
        ref_name = "reference_transform"
    
    # Extract transform components for comparison
    ref_pixel_width = ref_transform.a
    ref_pixel_height = -ref_transform.e  # Usually negative
    ref_origin_x = ref_transform.c
    ref_origin_y = ref_transform.f
    
    # Check each candidate
    for i, candidate in enumerate(candidates):
        if isinstance(candidate, str):
            candidate_name = candidate
            try:
                with rasterio.open(candidate) as src:
                    cand_transform = src.transform
                    cand_crs = src.crs
                    cand_bounds = src.bounds
                    cand_width = src.width
                    cand_height = src.height
            except Exception as e:
                result["aligned"] = False
                result["misaligned"].append(candidate_name)
                result["alignment_errors"][candidate_name] = [f"Cannot open file: {str(e)}"]
                continue
        else:
            # Assume Transform object
            cand_transform = candidate
            cand_crs = None
            cand_bounds = None
            cand_width = None
            cand_height = None
            candidate_name = f"candidate_{i}"
        
        errors = []
        
        # Check CRS
        if check_crs and ref_crs and cand_crs:
            if ref_crs != cand_crs:
                errors.append(f"CRS mismatch: {ref_crs} vs {cand_crs}")
                result["recommendations"].append(f"Reproject {candidate_name} to {ref_crs}")
        
        # Extract transform components
        cand_pixel_width = cand_transform.a
        cand_pixel_height = -cand_transform.e
        cand_origin_x = cand_transform.c
        cand_origin_y = cand_transform.f
        
        # Calculate deltas
        deltas = {
            "pixel_width_diff": abs(ref_pixel_width - cand_pixel_width),
            "pixel_height_diff": abs(ref_pixel_height - cand_pixel_height),
            "origin_x_diff": abs(ref_origin_x - cand_origin_x),
            "origin_y_diff": abs(ref_origin_y - cand_origin_y)
        }
        
        result["transform_deltas"][candidate_name] = deltas
        
        # Check resolution
        if check_resolution:
            if deltas["pixel_width_diff"] > tolerance:
                errors.append(f"Pixel width mismatch: {ref_pixel_width} vs {cand_pixel_width}")
            if deltas["pixel_height_diff"] > tolerance:
                errors.append(f"Pixel height mismatch: {ref_pixel_height} vs {cand_pixel_height}")
        
        # Check origin alignment
        # Origins should align to pixel grid
        x_misalignment = (cand_origin_x - ref_origin_x) % ref_pixel_width
        y_misalignment = (cand_origin_y - ref_origin_y) % ref_pixel_height
        
        # Handle negative modulo results
        if x_misalignment > ref_pixel_width / 2:
            x_misalignment = x_misalignment - ref_pixel_width
        if y_misalignment > ref_pixel_height / 2:
            y_misalignment = y_misalignment - ref_pixel_height
        
        if abs(x_misalignment) > tolerance:
            errors.append(f"X origin misalignment: {x_misalignment:.6f} pixels")
            result["recommendations"].append(f"Align {candidate_name} to reference pixel grid")
        
        if abs(y_misalignment) > tolerance:
            errors.append(f"Y origin misalignment: {y_misalignment:.6f} pixels")
        
        # Check bounds overlap
        if check_bounds and ref_bounds and cand_bounds:
            # Calculate intersection
            x_min = max(ref_bounds.left, cand_bounds.left)
            y_min = max(ref_bounds.bottom, cand_bounds.bottom)
            x_max = min(ref_bounds.right, cand_bounds.right)
            y_max = min(ref_bounds.top, cand_bounds.top)
            
            if x_min >= x_max or y_min >= y_max:
                errors.append("No spatial overlap with reference")
                result["recommendations"].append(f"Check spatial extent of {candidate_name}")
            else:
                # Calculate overlap percentage
                ref_area = (ref_bounds.right - ref_bounds.left) * (ref_bounds.top - ref_bounds.bottom)
                overlap_area = (x_max - x_min) * (y_max - y_min)
                overlap_percent = (overlap_area / ref_area) * 100
                
                if overlap_percent < 90:
                    errors.append(f"Limited overlap: {overlap_percent:.1f}%")
        
        # Check dimensions if available
        if ref_width and ref_height and cand_width and cand_height:
            if ref_width != cand_width or ref_height != cand_height:
                errors.append(f"Dimension mismatch: ({ref_width}x{ref_height}) vs ({cand_width}x{cand_height})")
                
                # Check if it's just a bounds issue
                if ref_bounds:
                    expected_width = int((ref_bounds.right - ref_bounds.left) / ref_pixel_width)
                    expected_height = int((ref_bounds.top - ref_bounds.bottom) / ref_pixel_height)
                    
                    if abs(expected_width - cand_width) <= 1 and abs(expected_height - cand_height) <= 1:
                        errors.append("Minor dimension difference - likely rounding error")
                    else:
                        result["recommendations"].append(f"Crop or pad {candidate_name} to match reference dimensions")
                else:
                    result["recommendations"].append(f"Crop or pad {candidate_name} to match reference dimensions")
        
        # Store errors
        if errors:
            result["aligned"] = False
            result["misaligned"].append(candidate_name)
            result["alignment_errors"][candidate_name] = errors
    
    # General recommendations if misaligned
    if not result["aligned"]:
        result["recommendations"].insert(0, "Use prepare.align_rasters() to ensure all layers match")
    
    return result


def check_output_quality(
    data: ArrayLike | str,
    mask: ArrayLike | str | None = None,
    min_valid_pixels: int = 100,
    max_nodata_percent: float = 90.0,
    check_patterns: bool = True,
    check_edges: bool = True
) -> dict[str, Any]:
    """Perform comprehensive quality checks on output data.
    
    This function runs a suite of quality checks to ensure the output
    meets minimum standards for distribution. It looks for common
    processing artifacts, edge effects, and data coverage issues that
    could indicate processing problems.
    
    Args:
        data: Output data array or path
        mask: Optional quality mask
        min_valid_pixels: Minimum number of valid pixels required
        max_nodata_percent: Maximum acceptable percentage of nodata
        check_patterns: Look for artificial patterns/striping
        check_edges: Check for edge artifacts from processing
        
    Returns:
        Quality report containing:
        - passed: bool indicating if quality is acceptable
        - valid_pixel_count: Number of valid data pixels
        - nodata_percent: Percentage of nodata pixels
        - coverage_percent: Percentage of area with valid data
        - artifacts_detected: List of detected artifacts
        - edge_quality: Assessment of edge handling
        - overall_score: Numeric quality score (0-100)
        
    Example:
        >>> # Check final output quality
        >>> quality = check_output_quality(enhanced_ntl, urban_mask)
        >>> if quality["overall_score"] < 80:
        >>>     warnings.warn("Output quality below threshold")
    """
    import numpy as np
    import rasterio
    from scipy import ndimage, signal
    
    # Load data
    if isinstance(data, str):
        with rasterio.open(data) as src:
            data_array = src.read(1)
            nodata_value = src.nodata
    else:
        data_array = np.asarray(data)
        if data_array.ndim == 3:
            data_array = data_array[0]
        nodata_value = None
    
    # Load mask if provided
    if mask is not None:
        if isinstance(mask, str):
            with rasterio.open(mask) as src:
                mask_array = src.read(1).astype(bool)
        else:
            mask_array = np.asarray(mask).astype(bool)
    else:
        mask_array = np.ones_like(data_array, dtype=bool)
    
    result = {
        "passed": True,
        "artifacts_detected": [],
        "quality_scores": {}
    }
    
    # Create valid data mask
    valid_mask = mask_array.copy()
    if nodata_value is not None:
        valid_mask &= (data_array != nodata_value)
    if np.issubdtype(data_array.dtype, np.floating):
        valid_mask &= ~np.isnan(data_array)
    
    # Calculate coverage statistics
    total_pixels = data_array.size
    valid_pixel_count = int(np.sum(valid_mask))
    nodata_percent = ((total_pixels - valid_pixel_count) / total_pixels) * 100
    coverage_percent = 100 - nodata_percent
    
    result["valid_pixel_count"] = valid_pixel_count
    result["nodata_percent"] = nodata_percent
    result["coverage_percent"] = coverage_percent
    
    # Check minimum requirements
    if valid_pixel_count < min_valid_pixels:
        result["passed"] = False
        result["artifacts_detected"].append(f"Insufficient valid pixels: {valid_pixel_count} < {min_valid_pixels}")
    
    if nodata_percent > max_nodata_percent:
        result["passed"] = False
        result["artifacts_detected"].append(f"Excessive nodata: {nodata_percent:.1f}% > {max_nodata_percent}%")
    
    # Quality scoring components
    scores = {}
    
    # Coverage score
    scores["coverage"] = min(100, coverage_percent)
    
    # Only perform additional checks if we have enough valid data
    if valid_pixel_count > min_valid_pixels:
        valid_data = data_array[valid_mask]
        
        # Check for patterns/striping
        if check_patterns:
            # Detect striping using FFT
            # Sample a subset for efficiency
            sample_size = min(data_array.shape[0], 1000)
            sample_data = data_array[:sample_size, :]
            
            # Compute row-wise variance
            row_variance = np.var(sample_data, axis=1)
            
            # Check for periodic patterns in variance
            if len(row_variance) > 10:
                # Autocorrelation to detect periodicity
                autocorr = signal.correlate(row_variance, row_variance, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / autocorr[0]  # Normalize
                
                # Look for peaks indicating periodicity
                peaks, properties = signal.find_peaks(autocorr[1:50], height=0.5)
                
                if len(peaks) > 0:
                    result["artifacts_detected"].append("Periodic striping pattern detected")
                    scores["pattern_free"] = 50
                else:
                    scores["pattern_free"] = 100
            else:
                scores["pattern_free"] = 100
            
            # Check for blockiness (tiling artifacts)
            # Compute local variance
            local_var = ndimage.generic_filter(data_array, np.var, size=8)
            
            # Look for regular grid patterns in variance
            var_gradient_x = np.diff(local_var, axis=1)
            var_gradient_y = np.diff(local_var, axis=0)
            
            # High gradients at regular intervals indicate blocks
            grad_threshold = np.percentile(np.abs(var_gradient_x[valid_mask[:-1, :]]), 95)
            
            if grad_threshold > 0:
                block_edges_x = np.abs(var_gradient_x) > grad_threshold
                block_edges_y = np.abs(var_gradient_y) > grad_threshold
                
                # Check for regular spacing
                if np.sum(block_edges_x) > 100:
                    x_positions = np.where(np.any(block_edges_x, axis=0))[0]
                    if len(x_positions) > 1:
                        spacings = np.diff(x_positions)
                        unique_spacings = np.unique(spacings)
                        
                        # Regular spacing indicates tiling
                        if len(unique_spacings) < 3 and np.max(unique_spacings) > 100:
                            result["artifacts_detected"].append("Tiling/block artifacts detected")
                            scores["block_free"] = 70
                        else:
                            scores["block_free"] = 100
                    else:
                        scores["block_free"] = 100
                else:
                    scores["block_free"] = 100
            else:
                scores["block_free"] = 100
        
        # Check edges
        if check_edges:
            # Extract edge regions
            edge_size = min(50, data_array.shape[0]//10, data_array.shape[1]//10)
            
            edges = {
                "top": data_array[:edge_size, :],
                "bottom": data_array[-edge_size:, :],
                "left": data_array[:, :edge_size],
                "right": data_array[:, -edge_size:]
            }
            
            edge_scores = []
            
            for edge_name, edge_data in edges.items():
                edge_valid = valid_mask[:edge_size, :] if edge_name == "top" else \
                           valid_mask[-edge_size:, :] if edge_name == "bottom" else \
                           valid_mask[:, :edge_size] if edge_name == "left" else \
                           valid_mask[:, -edge_size:]
                
                if np.any(edge_valid):
                    # Check for abrupt changes at edges
                    edge_values = edge_data[edge_valid]
                    if len(edge_values) > 0:
                        # Compare edge statistics to overall statistics
                        edge_mean = np.mean(edge_values)
                        overall_mean = np.mean(valid_data)
                        
                        # Large difference indicates edge artifacts
                        relative_diff = abs(edge_mean - overall_mean) / (abs(overall_mean) + 1e-10)
                        
                        if relative_diff > 0.5:
                            result["artifacts_detected"].append(f"{edge_name} edge shows discontinuity")
                            edge_scores.append(50)
                        else:
                            edge_scores.append(100)
                    else:
                        edge_scores.append(100)
                else:
                    # No valid data at edge
                    edge_scores.append(50)
            
            scores["edge_quality"] = np.mean(edge_scores) if edge_scores else 100
            result["edge_quality"] = "good" if scores["edge_quality"] > 80 else "poor"
        
        # Data distribution score
        # Check for reasonable value distribution
        data_std = np.std(valid_data)
        data_range = np.max(valid_data) - np.min(valid_data)
        
        if data_range > 0:
            # Coefficient of variation
            cv = data_std / np.mean(np.abs(valid_data)) if np.mean(np.abs(valid_data)) > 0 else 0
            
            # Reasonable CV indicates good data distribution
            if cv < 0.01:
                result["artifacts_detected"].append("Very low variation - possible constant fill")
                scores["distribution"] = 50
            elif cv > 10:
                result["artifacts_detected"].append("Extreme variation - possible outliers")
                scores["distribution"] = 70
            else:
                scores["distribution"] = 100
        else:
            result["artifacts_detected"].append("Constant values - no variation")
            scores["distribution"] = 0
    
    # Calculate overall score
    result["quality_scores"] = scores
    
    if scores:
        weights = {
            "coverage": 0.3,
            "pattern_free": 0.2,
            "block_free": 0.2,
            "edge_quality": 0.15,
            "distribution": 0.15
        }
        
        overall_score = 0
        total_weight = 0
        
        for component, weight in weights.items():
            if component in scores:
                overall_score += scores[component] * weight
                total_weight += weight
        
        if total_weight > 0:
            overall_score = overall_score / total_weight
        else:
            overall_score = 0
    else:
        overall_score = 0
    
    result["overall_score"] = overall_score
    
    # Final pass/fail based on score
    if overall_score < 60:
        result["passed"] = False
    
    return result


def validate_cog_structure(
    file_path: str,
    strict: bool = False
) -> dict[str, Any]:
    """Validate internal structure of Cloud-Optimized GeoTIFF.
    
    This function checks that an existing GeoTIFF file meets all COG
    specifications. It examines the internal file structure, IFD ordering,
    tile organization, and overview placement to ensure cloud-optimized
    access patterns will work efficiently.
    
    Args:
        file_path: Path to GeoTIFF file to validate
        strict: Apply strict COG validation rules
        
    Returns:
        COG validation report containing:
        - is_valid_cog: bool indicating COG compliance
        - errors: List of COG specification violations
        - warnings: List of suboptimal configurations
        - ifd_structure: Internal file directory analysis
        - tile_info: Tiling configuration details
        - overview_info: Overview level details
        
    Note:
        This should be run on any existing GeoTIFF before assuming
        it's a valid COG, and after COG generation to verify success.
    """
    import rasterio
    import os
    
    result = {
        "is_valid_cog": True,
        "errors": [],
        "warnings": [],
        "ifd_structure": {},
        "tile_info": {},
        "overview_info": {}
    }
    
    if not os.path.exists(file_path):
        result["is_valid_cog"] = False
        result["errors"].append(f"File not found: {file_path}")
        return result
    
    try:
        # Try to use rio-cogeo if available
        try:
            from rio_cogeo.cogeo import cog_validate, cog_info
            
            # Use rio-cogeo validation
            is_valid, errors, warnings = cog_validate(file_path, strict=strict)
            
            result["is_valid_cog"] = is_valid
            result["errors"] = errors
            result["warnings"] = warnings
            
            # Get additional COG info
            info = cog_info(file_path)
            if info:
                # info is likely a namedtuple or object, not a dict
                if hasattr(info, 'IFD'):
                    result["ifd_structure"] = info.IFD
                elif hasattr(info, '__dict__') and 'IFD' in info.__dict__:
                    result["ifd_structure"] = info.__dict__.get("IFD", {})
                    
                if hasattr(info, 'overviews'):
                    result["overview_info"] = info.overviews
                elif hasattr(info, '__dict__') and 'overviews' in info.__dict__:
                    result["overview_info"] = info.__dict__.get("overviews", {})
            
        except ImportError:
            # Fallback to manual validation
            with rasterio.open(file_path) as src:
                # Basic COG requirements
                
                # 1. Must be tiled
                if not src.is_tiled:
                    result["is_valid_cog"] = False
                    result["errors"].append("File is not tiled")
                else:
                    block_shapes = src.block_shapes
                    result["tile_info"]["block_shapes"] = block_shapes
                    result["tile_info"]["tiled"] = True
                    
                    # Check tile size
                    if block_shapes:
                        tile_width, tile_height = block_shapes[0]
                        result["tile_info"]["tile_size"] = (tile_width, tile_height)
                        
                        if tile_width != tile_height:
                            result["warnings"].append(f"Non-square tiles: {tile_width}x{tile_height}")
                        
                        if tile_width not in [256, 512, 1024]:
                            result["warnings"].append(f"Non-standard tile size: {tile_width}")
                
                # 2. Check compression
                compression = src.compression
                if compression:
                    result["tile_info"]["compression"] = compression.name
                    
                    # Check for lossy compression
                    if compression.name in ['JPEG', 'WEBP']:
                        if strict:
                            result["is_valid_cog"] = False
                            result["errors"].append(f"Lossy compression {compression.name} not suitable for COG")
                        else:
                            result["warnings"].append(f"Lossy compression {compression.name} may not be suitable for all use cases")
                else:
                    result["warnings"].append("No compression")
                
                # 3. Check overviews
                overviews = []
                for i in range(1, src.count + 1):
                    band_overviews = src.overviews(i)
                    if band_overviews:
                        overviews = band_overviews
                        break
                
                result["overview_info"]["levels"] = overviews
                result["overview_info"]["count"] = len(overviews)
                
                if not overviews:
                    result["warnings"].append("No overviews found")
                else:
                    # Check for power-of-2 overviews
                    expected = [2, 4, 8, 16]
                    missing = [ov for ov in expected if ov not in overviews]
                    if missing:
                        result["warnings"].append(f"Missing standard overview levels: {missing}")
                
                # 4. Check IFD ordering (simplified check)
                # In a COG, overviews should come after the main image
                # This is a simplified check - full validation requires parsing TIFF structure
                if src.is_tiled and overviews:
                    # Basic heuristic: file should be reasonably sized for a COG
                    file_size = os.path.getsize(file_path)
                    expected_min_size = (src.width * src.height * src.count * src.dtypes[0].itemsize) // 10
                    
                    if file_size < expected_min_size:
                        result["warnings"].append("File size suggests missing data or compression issues")
                
                # 5. Additional metadata
                if not isinstance(result["ifd_structure"], dict):
                    result["ifd_structure"] = {}
                result["ifd_structure"]["width"] = src.width
                result["ifd_structure"]["height"] = src.height
                result["ifd_structure"]["bands"] = src.count
                result["ifd_structure"]["dtype"] = str(src.dtypes[0])
                
                # Check for web optimization
                if src.driver == 'GTiff':
                    tags = src.tags()
                    if 'PREDICTOR' in tags:
                        result["tile_info"]["predictor"] = tags['PREDICTOR']
                    
                    # Check interleaving for multi-band
                    if src.count > 1:
                        interleave = src.interleaving
                        result["tile_info"]["interleave"] = interleave.name
                        if interleave.name != 'pixel':
                            result["warnings"].append(f"Interleaving '{interleave.name}' may not be optimal for web access")
        
        # Determine if it's a valid COG based on findings
        if not result["errors"] and len(result["warnings"]) <= 2:
            result["is_valid_cog"] = True
        
    except Exception as e:
        result["is_valid_cog"] = False
        result["errors"].append(f"Error validating file: {str(e)}")
    
    return result


def validate_export_config(
    config: dict[str, Any],
    data_shape: tuple[int, ...],
    data_dtype: Any
) -> dict[str, Any]:
    """Validate export configuration before processing.
    
    This function checks that export parameters are compatible with
    the data being exported. It prevents invalid configurations that
    would cause export to fail or produce incorrect results.
    
    Args:
        config: Export configuration dictionary
        data_shape: Shape of data to be exported
        data_dtype: Data type of array
        
    Returns:
        Configuration validation results:
        - valid: bool indicating if config is valid
        - adjusted_config: Config with corrections applied
        - incompatibilities: List of incompatible settings
        - suggestions: Recommended configuration changes
    """
    import numpy as np
    
    result = {
        "valid": True,
        "adjusted_config": config.copy(),
        "incompatibilities": [],
        "suggestions": []
    }
    
    # Validate data shape
    if len(data_shape) not in [2, 3]:
        result["valid"] = False
        result["incompatibilities"].append(f"Invalid data dimensions: {len(data_shape)}D. Expected 2D or 3D.")
        return result
    
    # Extract dimensions
    if len(data_shape) == 2:
        height, width = data_shape
        bands = 1
    else:
        bands, height, width = data_shape
    
    # Check compression settings
    compression = config.get("compress", "deflate")
    
    # Validate compression for data type
    if np.issubdtype(data_dtype, np.floating):
        if compression in ["jpeg", "webp"]:
            result["incompatibilities"].append(f"Compression '{compression}' is lossy and not suitable for floating point data")
            result["adjusted_config"]["compress"] = "deflate"
            result["suggestions"].append("Changed to DEFLATE compression for lossless storage")
    
    # Check tile size
    blocksize = config.get("blocksize", 512)
    
    if blocksize not in [128, 256, 512, 1024]:
        result["warnings"] = result.get("warnings", [])
        result["warnings"].append(f"Non-standard block size: {blocksize}")
        
        # Suggest appropriate block size
        if width < 512 or height < 512:
            result["adjusted_config"]["blocksize"] = 256
            result["suggestions"].append("Reduced block size to 256 for small image")
        elif width > 10000 or height > 10000:
            result["adjusted_config"]["blocksize"] = 1024
            result["suggestions"].append("Increased block size to 1024 for large image")
        else:
            result["adjusted_config"]["blocksize"] = 512
            result["suggestions"].append("Using standard block size of 512")
    
    # Check if tiling makes sense
    if config.get("tiled", True):
        if width < blocksize and height < blocksize:
            result["adjusted_config"]["tiled"] = False
            result["suggestions"].append("Disabled tiling for small image")
    
    # Validate overview levels
    overviews = config.get("overviews", None)
    
    if overviews is None:
        # Generate appropriate overviews
        min_dimension = min(width, height)
        suggested_overviews = []
        level = 2
        
        while min_dimension / level > 256:
            suggested_overviews.append(level)
            level *= 2
        
        if suggested_overviews:
            result["adjusted_config"]["overviews"] = suggested_overviews
            result["suggestions"].append(f"Added overview levels: {suggested_overviews}")
    else:
        # Validate provided overviews
        max_overview = max(overviews) if overviews else 1
        if min(width, height) / max_overview < 128:
            result["warnings"] = result.get("warnings", [])
            result["warnings"].append("Overview levels may be too aggressive")
    
    # Check output format compatibility
    output_format = config.get("driver", "GTiff")
    
    if output_format != "GTiff":
        result["incompatibilities"].append(f"Driver '{output_format}' may not support COG features")
        result["adjusted_config"]["driver"] = "GTiff"
    
    # Data type considerations
    if data_dtype == np.float64:
        result["suggestions"].append("Consider converting float64 to float32 to reduce file size")
        result["adjusted_config"]["dtype"] = "float32"
    
    # Multi-band considerations
    if bands > 1:
        interleave = config.get("interleave", "pixel")
        if interleave != "pixel" and bands <= 4:
            result["adjusted_config"]["interleave"] = "pixel"
            result["suggestions"].append("Using pixel interleaving for better multi-band performance")
    
    # Predictor settings for compression
    # CRITICAL: Predictor is essential for good compression with scientific data
    # - Predictor 3 stores differences between adjacent float values
    # - Can reduce file size by 50%+ for spatially correlated data like radiance
    if compression in ["deflate", "lzw"]:
        if "predictor" not in config:
            if np.issubdtype(data_dtype, np.floating):
                result["adjusted_config"]["predictor"] = 3  # Floating point predictor
            else:
                result["adjusted_config"]["predictor"] = 2  # Horizontal differencing
            result["suggestions"].append(f"Added predictor {result['adjusted_config']['predictor']} for better compression")
    
    # BigTIFF considerations
    estimated_size = width * height * bands * np.dtype(data_dtype).itemsize
    if estimated_size > 3.5e9:  # ~3.5 GB
        if "BIGTIFF" not in config:
            result["adjusted_config"]["BIGTIFF"] = "YES"
            result["suggestions"].append("Enabled BigTIFF for large file support")
    
    # Validate required parameters
    required_for_cog = ["transform", "crs"]
    for param in required_for_cog:
        if param not in config or config[param] is None:
            result["incompatibilities"].append(f"Missing required parameter: {param}")
            result["valid"] = False
    
    return result


def compare_with_reference(
    output_path: str,
    reference_path: str,
    metrics: list[str] = ["rmse", "mae", "correlation"],
    tolerance: dict[str, float] | None = None
) -> dict[str, Any]:
    """Compare output with reference data for validation.
    
    This function enables regression testing by comparing new outputs
    with known-good reference data. It's essential for ensuring that
    pipeline changes don't introduce unexpected differences in results.
    
    Args:
        output_path: Path to generated output
        reference_path: Path to reference/expected data
        metrics: List of comparison metrics to compute
        tolerance: Acceptable differences for each metric
        
    Returns:
        Comparison results containing:
        - matches_reference: bool indicating if within tolerance
        - metrics: Dict of computed comparison metrics
        - differences: Spatial map of differences
        - max_difference: Maximum pixel difference found
        - difference_histogram: Distribution of differences
        
    Note:
        Used for regression testing and quality assurance to ensure
        consistency across pipeline versions.
    """
    import numpy as np
    import rasterio
    from rasterio.warp import reproject, Resampling
    
    # Default tolerances if not provided
    if tolerance is None:
        tolerance = {
            "rmse": 0.01,
            "mae": 0.01,
            "correlation": 0.99,
            "max_difference": 0.05
        }
    
    result = {
        "matches_reference": True,
        "metrics": {},
        "max_difference": 0,
        "difference_histogram": {}
    }
    
    try:
        # Open both files
        with rasterio.open(output_path) as out_src, rasterio.open(reference_path) as ref_src:
            # Check if spatial alignment is needed
            need_reproject = (
                out_src.crs != ref_src.crs or
                out_src.transform != ref_src.transform or
                out_src.shape != ref_src.shape
            )
            
            if need_reproject:
                result["warning"] = "Files have different spatial properties - reprojecting for comparison"
                
                # Read reference data
                ref_data = ref_src.read(1)
                
                # Create array for reprojected output
                out_data = np.empty_like(ref_data)
                
                # Reproject output to match reference
                reproject(
                    source=rasterio.band(out_src, 1),
                    destination=out_data,
                    src_transform=out_src.transform,
                    src_crs=out_src.crs,
                    dst_transform=ref_src.transform,
                    dst_crs=ref_src.crs,
                    resampling=Resampling.bilinear
                )
            else:
                # Direct comparison
                out_data = out_src.read(1)
                ref_data = ref_src.read(1)
            
            # Handle nodata
            out_nodata = out_src.nodata
            ref_nodata = ref_src.nodata
            
            # Create valid mask
            valid_mask = np.ones_like(ref_data, dtype=bool)
            
            if out_nodata is not None:
                valid_mask &= (out_data != out_nodata)
            if ref_nodata is not None:
                valid_mask &= (ref_data != ref_nodata)
            
            # Also exclude NaN
            if np.issubdtype(out_data.dtype, np.floating):
                valid_mask &= ~np.isnan(out_data)
            if np.issubdtype(ref_data.dtype, np.floating):
                valid_mask &= ~np.isnan(ref_data)
            
            # Get valid data
            out_valid = out_data[valid_mask]
            ref_valid = ref_data[valid_mask]
            
            if len(out_valid) == 0:
                result["matches_reference"] = False
                result["error"] = "No valid pixels for comparison"
                return result
            
            # Calculate differences
            differences = out_valid - ref_valid
            abs_differences = np.abs(differences)
            
            # Store max difference
            result["max_difference"] = float(np.max(abs_differences))
            
            # Calculate requested metrics
            if "rmse" in metrics:
                rmse = float(np.sqrt(np.mean(differences ** 2)))
                result["metrics"]["rmse"] = rmse
                if rmse > tolerance.get("rmse", float('inf')):
                    result["matches_reference"] = False
            
            if "mae" in metrics:
                mae = float(np.mean(abs_differences))
                result["metrics"]["mae"] = mae
                if mae > tolerance.get("mae", float('inf')):
                    result["matches_reference"] = False
            
            if "correlation" in metrics:
                if np.std(out_valid) > 0 and np.std(ref_valid) > 0:
                    correlation = float(np.corrcoef(out_valid, ref_valid)[0, 1])
                    result["metrics"]["correlation"] = correlation
                    if correlation < tolerance.get("correlation", -1):
                        result["matches_reference"] = False
                else:
                    result["metrics"]["correlation"] = np.nan
            
            if "bias" in metrics:
                bias = float(np.mean(differences))
                result["metrics"]["bias"] = bias
            
            if "r2" in metrics:
                # Coefficient of determination
                ss_res = np.sum(differences ** 2)
                ss_tot = np.sum((ref_valid - np.mean(ref_valid)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                result["metrics"]["r2"] = float(r2)
            
            # Check max difference tolerance
            if result["max_difference"] > tolerance.get("max_difference", float('inf')):
                result["matches_reference"] = False
            
            # Create difference map
            diff_map = np.full_like(ref_data, np.nan)
            diff_map[valid_mask] = out_data[valid_mask] - ref_data[valid_mask]
            result["differences"] = diff_map
            
            # Difference histogram
            hist_bins = [-1, -0.1, -0.01, -0.001, 0, 0.001, 0.01, 0.1, 1]
            hist, _ = np.histogram(differences, bins=hist_bins)
            
            result["difference_histogram"] = {
                "bins": hist_bins,
                "counts": hist.tolist(),
                "percentiles": {
                    "p1": float(np.percentile(differences, 1)),
                    "p5": float(np.percentile(differences, 5)),
                    "p25": float(np.percentile(differences, 25)),
                    "p50": float(np.percentile(differences, 50)),
                    "p75": float(np.percentile(differences, 75)),
                    "p95": float(np.percentile(differences, 95)),
                    "p99": float(np.percentile(differences, 99))
                }
            }
            
            # Additional statistics
            result["comparison_stats"] = {
                "n_pixels": len(out_valid),
                "percent_different": float(np.sum(abs_differences > 1e-6) / len(differences) * 100),
                "std_difference": float(np.std(differences))
            }
            
    except Exception as e:
        result["matches_reference"] = False
        result["error"] = f"Error comparing files: {str(e)}"
    
    return result


def generate_validation_report(
    validation_results: list[dict[str, Any]],
    output_path: str | None = None,
    format: Literal["text", "json", "html"] = "text"
) -> str:
    """Generate comprehensive validation report from multiple checks.
    
    This function consolidates results from multiple validation functions
    into a single human-readable report. It provides an overall assessment
    of data quality and export readiness.
    
    Args:
        validation_results: List of validation result dictionaries
        output_path: Optional path to save report
        format: Output format for report
        
    Returns:
        Formatted validation report as string, or path if saved
        
    Example:
        >>> # Run multiple validations
        >>> results = []
        >>> results.append(validate_data_range(data, 0, 1))
        >>> results.append(check_output_quality(data))
        >>> report = generate_validation_report(results)
        >>> print(report)
    """
    import json
    import os
    from datetime import datetime
    
    # Consolidate results
    summary = {
        "report_date": datetime.now().isoformat(),
        "total_checks": len(validation_results),
        "passed": 0,
        "failed": 0,
        "warnings": 0,
        "overall_status": "PASS"
    }
    
    # Analyze results
    all_errors = []
    all_warnings = []
    all_recommendations = []
    
    for result in validation_results:
        # Determine check status
        if result.get("valid", result.get("passed", result.get("aligned", result.get("is_valid_cog", True)))):
            summary["passed"] += 1
        else:
            summary["failed"] += 1
            summary["overall_status"] = "FAIL"
        
        # Collect issues
        if "errors" in result:
            all_errors.extend(result["errors"])
        if "warnings" in result:
            all_warnings.extend(result["warnings"])
            summary["warnings"] += len(result["warnings"])
        if "recommendations" in result:
            all_recommendations.extend(result["recommendations"])
        if "anomalies" in result:
            all_warnings.extend(result["anomalies"])
        if "artifacts_detected" in result:
            all_errors.extend(result["artifacts_detected"])
    
    # Generate report based on format
    if format == "json":
        report_data = {
            "summary": summary,
            "errors": all_errors,
            "warnings": all_warnings,
            "recommendations": all_recommendations,
            "detailed_results": validation_results
        }
        
        report_content = json.dumps(report_data, indent=2, default=str)
    
    elif format == "html":
        # Generate HTML report
        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<title>Black Marble Validation Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; }",
            "h1, h2, h3 { color: #333; }",
            ".pass { color: green; }",
            ".fail { color: red; }",
            ".warning { color: orange; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            "</style>",
            "</head><body>",
            "<h1>Black Marble Export Validation Report</h1>",
            f"<p>Generated: {summary['report_date']}</p>",
            
            "<h2>Summary</h2>",
            f"<p>Total Checks: {summary['total_checks']}</p>",
            f"<p class='{'pass' if summary['overall_status'] == 'PASS' else 'fail'}'>",
            f"Overall Status: {summary['overall_status']}</p>",
            
            "<table>",
            "<tr><th>Status</th><th>Count</th></tr>",
            f"<tr><td>Passed</td><td>{summary['passed']}</td></tr>",
            f"<tr><td>Failed</td><td>{summary['failed']}</td></tr>",
            f"<tr><td>Warnings</td><td>{summary['warnings']}</td></tr>",
            "</table>"
        ]
        
        if all_errors:
            html_parts.extend([
                "<h2>Errors</h2>",
                "<ul>"
            ])
            for error in all_errors:
                html_parts.append(f"<li class='fail'>{error}</li>")
            html_parts.append("</ul>")
        
        if all_warnings:
            html_parts.extend([
                "<h2>Warnings</h2>",
                "<ul>"
            ])
            for warning in all_warnings:
                html_parts.append(f"<li class='warning'>{warning}</li>")
            html_parts.append("</ul>")
        
        if all_recommendations:
            html_parts.extend([
                "<h2>Recommendations</h2>",
                "<ul>"
            ])
            for rec in all_recommendations:
                html_parts.append(f"<li>{rec}</li>")
            html_parts.append("</ul>")
        
        html_parts.extend([
            "</body></html>"
        ])
        
        report_content = "\n".join(html_parts)
    
    else:  # text format
        # Generate text report
        lines = [
            "=" * 60,
            "BLACK MARBLE EXPORT VALIDATION REPORT",
            "=" * 60,
            f"Generated: {summary['report_date']}",
            "",
            "SUMMARY",
            "-" * 30,
            f"Total Checks: {summary['total_checks']}",
            f"Passed: {summary['passed']}",
            f"Failed: {summary['failed']}",
            f"Warnings: {summary['warnings']}",
            f"Overall Status: {summary['overall_status']}",
            ""
        ]
        
        if all_errors:
            lines.extend([
                "ERRORS",
                "-" * 30
            ])
            for i, error in enumerate(all_errors, 1):
                lines.append(f"{i}. {error}")
            lines.append("")
        
        if all_warnings:
            lines.extend([
                "WARNINGS",
                "-" * 30
            ])
            for i, warning in enumerate(all_warnings, 1):
                lines.append(f"{i}. {warning}")
            lines.append("")
        
        if all_recommendations:
            lines.extend([
                "RECOMMENDATIONS",
                "-" * 30
            ])
            for i, rec in enumerate(all_recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")
        
        # Add detailed results summary
        lines.extend([
            "DETAILED CHECK RESULTS",
            "-" * 30
        ])
        
        for i, result in enumerate(validation_results, 1):
            check_name = result.get("check_name", f"Check {i}")
            status = "PASS" if result.get("valid", result.get("passed", result.get("aligned", result.get("is_valid_cog", True)))) else "FAIL"
            
            lines.append(f"\n{i}. {check_name}: {status}")
            
            # Add key metrics
            if "metrics" in result:
                for metric, value in result["metrics"].items():
                    lines.append(f"   - {metric}: {value}")
            
            if "overall_score" in result:
                lines.append(f"   - Overall Score: {result['overall_score']:.1f}/100")
            
            if "coverage_percent" in result:
                lines.append(f"   - Coverage: {result['coverage_percent']:.1f}%")
        
        lines.extend([
            "",
            "=" * 60
        ])
        
        report_content = "\n".join(lines)
    
    # Save or return report
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report_content)
        return output_path
    else:
        return report_content