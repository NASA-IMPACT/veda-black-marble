"""Data validation and quality check functions for Black Marble processing.

This module provides functions to validate data quality, check ranges,
and ensure processing requirements are met throughout the pipeline.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional


def validate_reflectance_range(
    data: np.ndarray, 
    band_name: str = "band",
    valid_range: Tuple[float, float] = (0.0, 1.0)
) -> Dict[str, float]:
    """Validate that reflectance values are in expected range.
    
    Surface reflectance should typically be between 0 and 1, though
    some surfaces (snow, clouds) can exceed 1.0 slightly.
    
    Args:
        data: Array of reflectance values
        band_name: Name of the band for reporting
        valid_range: Expected valid range (min, max)
        
    Returns:
        Dictionary with validation metrics:
        - band: Band name
        - total_pixels: Total number of pixels
        - valid_pixels: Number of non-NaN pixels
        - valid_percentage: Percentage of valid pixels
        - min_value: Minimum value (excluding NaN)
        - max_value: Maximum value (excluding NaN)
        - mean_value: Mean value (excluding NaN)
        - out_of_range: Number of pixels outside valid range
        
    Example:
        >>> data = np.array([[0.1, 0.3], [0.5, 1.2]])
        >>> result = validate_reflectance_range(data, "B4")
        >>> print(f"{result['band']}: {result['out_of_range']} pixels out of range")
        B4: 1 pixels out of range
    """
    valid_mask = ~np.isnan(data)
    valid_data = data[valid_mask]
    
    out_of_range = 0
    if valid_data.size > 0:
        out_of_range = np.sum(
            (valid_data < valid_range[0]) | (valid_data > valid_range[1])
        )
    
    return {
        "band": band_name,
        "total_pixels": data.size,
        "valid_pixels": valid_mask.sum(),
        "valid_percentage": 100.0 * valid_mask.sum() / data.size,
        "min_value": np.min(valid_data) if valid_data.size > 0 else np.nan,
        "max_value": np.max(valid_data) if valid_data.size > 0 else np.nan,
        "mean_value": np.mean(valid_data) if valid_data.size > 0 else np.nan,
        "out_of_range": out_of_range
    }


def check_data_coverage(
    data: np.ndarray,
    min_valid_percentage: float = 10.0,
    mask: Optional[np.ndarray] = None
) -> Tuple[bool, float]:
    """Check if data has sufficient valid pixel coverage.
    
    This is used to ensure there's enough valid data to proceed
    with processing. Too few valid pixels may indicate issues with
    cloud cover, data availability, or processing errors.
    
    Args:
        data: Input array to check
        min_valid_percentage: Minimum percentage of valid pixels required
        mask: Optional mask array (True = valid pixels)
        
    Returns:
        Tuple of (is_valid, valid_percentage)
        
    Example:
        >>> data = np.array([[1, 2, np.nan], [4, np.nan, np.nan]])
        >>> is_valid, percentage = check_data_coverage(data, min_valid=50.0)
        >>> print(f"Valid: {is_valid}, Coverage: {percentage:.1f}%")
        Valid: True, Coverage: 50.0%
    """
    if mask is not None:
        valid_pixels = np.sum(mask)
        total_pixels = mask.size
    else:
        valid_pixels = np.sum(~np.isnan(data))
        total_pixels = data.size
    
    valid_percentage = 100.0 * valid_pixels / total_pixels
    return valid_percentage >= min_valid_percentage, valid_percentage


def validate_index_range(
    index_data: np.ndarray,
    index_name: str,
    expected_min: float = -1.0,
    expected_max: float = 1.0
) -> Dict[str, any]:
    """Validate that spectral index values are within expected range.
    
    Most normalized difference indices (NDVI, NDWI) should be in [-1, 1]
    by definition, but processing errors can sometimes produce values
    outside this range.
    
    Args:
        index_data: Array of index values (e.g., NDVI, NDWI)
        index_name: Name of the index for reporting
        expected_min: Expected minimum value
        expected_max: Expected maximum value
        
    Returns:
        Validation report dictionary with:
        - index: Index name
        - valid_pixels: Number of non-NaN pixels
        - total_pixels: Total number of pixels
        - min_value: Actual minimum value
        - max_value: Actual maximum value
        - out_of_range_count: Number of pixels outside expected range
        - out_of_range_percentage: Percentage of valid pixels outside range
        
    Example:
        >>> ndvi = np.array([[-0.5, 0.8], [1.2, np.nan]])
        >>> report = validate_index_range(ndvi, "NDVI")
        >>> print(f"{report['index']}: {report['out_of_range_count']} invalid")
        NDVI: 1 invalid
    """
    valid_mask = ~np.isnan(index_data)
    valid_data = index_data[valid_mask]
    
    out_of_range_count = 0
    out_of_range_percentage = 0.0
    
    if valid_data.size > 0:
        out_of_range_mask = (valid_data < expected_min) | (valid_data > expected_max)
        out_of_range_count = out_of_range_mask.sum()
        out_of_range_percentage = 100.0 * out_of_range_count / valid_data.size
    
    return {
        "index": index_name,
        "valid_pixels": valid_mask.sum(),
        "total_pixels": index_data.size,
        "min_value": np.min(valid_data) if valid_data.size > 0 else np.nan,
        "max_value": np.max(valid_data) if valid_data.size > 0 else np.nan,
        "out_of_range_count": out_of_range_count,
        "out_of_range_percentage": out_of_range_percentage
    }


def validate_nighttime_lights(
    ntl_data: np.ndarray,
    expected_max: float = 1.0,
    zero_threshold: float = 0.1
) -> Dict[str, float]:
    """Validate nighttime lights data quality.
    
    Checks for common issues with NTL data:
    - Excessive zero/near-zero values (might indicate processing error)
    - Values exceeding expected maximum
    - Data coverage
    
    Args:
        ntl_data: Normalized nighttime lights array (0-1)
        expected_max: Expected maximum value
        zero_threshold: Threshold below which values are considered "dark"
        
    Returns:
        Dictionary with validation metrics:
        - total_pixels: Total pixel count
        - valid_pixels: Non-NaN pixel count
        - dark_pixels: Pixels below zero_threshold
        - dark_percentage: Percentage of valid pixels that are dark
        - bright_pixels: Pixels above 50% of expected_max
        - max_value: Maximum value in data
        - mean_brightness: Mean of non-dark pixels
    """
    valid_mask = ~np.isnan(ntl_data)
    valid_data = ntl_data[valid_mask]
    
    if valid_data.size == 0:
        return {
            "total_pixels": ntl_data.size,
            "valid_pixels": 0,
            "dark_pixels": 0,
            "dark_percentage": 0.0,
            "bright_pixels": 0,
            "max_value": np.nan,
            "mean_brightness": np.nan
        }
    
    dark_mask = valid_data < zero_threshold
    dark_pixels = dark_mask.sum()
    bright_mask = valid_data > (expected_max * 0.5)
    bright_pixels = bright_mask.sum()
    
    non_dark_data = valid_data[~dark_mask]
    mean_brightness = np.mean(non_dark_data) if non_dark_data.size > 0 else 0.0
    
    return {
        "total_pixels": ntl_data.size,
        "valid_pixels": valid_mask.sum(),
        "dark_pixels": dark_pixels,
        "dark_percentage": 100.0 * dark_pixels / valid_data.size,
        "bright_pixels": bright_pixels,
        "max_value": np.max(valid_data),
        "mean_brightness": mean_brightness
    }


def validate_road_coverage(
    road_data: np.ndarray,
    min_road_percentage: float = 0.1,
    max_road_percentage: float = 50.0
) -> Dict[str, any]:
    """Validate road network coverage.
    
    Checks that road coverage is reasonable for the area.
    Too few roads might indicate download failure, too many
    might indicate an error in processing.
    
    Args:
        road_data: Road classification array (0=no road, 1-5=road types)
        min_road_percentage: Minimum expected road coverage (%)
        max_road_percentage: Maximum expected road coverage (%)
        
    Returns:
        Dictionary with validation metrics:
        - total_pixels: Total pixel count
        - road_pixels: Number of road pixels
        - road_percentage: Percentage of pixels with roads
        - is_valid: Whether coverage is within expected range
        - road_types: Count of each road type present
    """
    total_pixels = road_data.size
    road_mask = road_data > 0
    road_pixels = road_mask.sum()
    road_percentage = 100.0 * road_pixels / total_pixels
    
    # Count each road type
    road_types = {}
    for road_type in range(1, 6):
        count = np.sum(road_data == road_type)
        if count > 0:
            road_types[f"type_{road_type}"] = count
    
    is_valid = min_road_percentage <= road_percentage <= max_road_percentage
    
    return {
        "total_pixels": total_pixels,
        "road_pixels": road_pixels,
        "road_percentage": road_percentage,
        "is_valid": is_valid,
        "road_types": road_types
    }


def validate_processing_inputs(
    bands: Dict[str, np.ndarray],
    required_bands: List[str] = ["B3", "B4", "B5"],
    min_coverage: float = 10.0
) -> Tuple[bool, List[str]]:
    """Validate that all required inputs are present and valid.
    
    Comprehensive check before processing to ensure all required
    data is available and meets quality requirements.
    
    Args:
        bands: Dictionary of band arrays
        required_bands: List of required band names
        min_coverage: Minimum valid pixel coverage (%)
        
    Returns:
        Tuple of (all_valid, error_messages)
        
    Example:
        >>> bands = {"B3": np.ones((10, 10)), "B4": np.ones((10, 10))}
        >>> valid, errors = validate_processing_inputs(bands)
        >>> if not valid:
        ...     print(f"Validation failed: {errors}")
        Validation failed: ['Missing required band: B5']
    """
    errors = []
    
    # Check for missing bands
    for band_name in required_bands:
        if band_name not in bands:
            errors.append(f"Missing required band: {band_name}")
            continue
        
        # Check band coverage
        band_data = bands[band_name]
        is_valid, coverage = check_data_coverage(band_data, min_coverage)
        
        if not is_valid:
            errors.append(
                f"{band_name} has insufficient coverage: {coverage:.1f}% "
                f"(minimum {min_coverage}% required)"
            )
        
        # Check for all-NaN bands
        if np.all(np.isnan(band_data)):
            errors.append(f"{band_name} contains only NaN values")
    
    return len(errors) == 0, errors


def generate_quality_report(
    data_dict: Dict[str, np.ndarray],
    config: Optional[dict] = None
) -> Dict[str, any]:
    """Generate comprehensive quality report for processed data.
    
    Creates a summary of data quality metrics for all inputs and outputs,
    useful for debugging and quality assurance.
    
    Args:
        data_dict: Dictionary of arrays to validate
        config: Optional configuration with thresholds
        
    Returns:
        Dictionary with quality metrics for each array
        
    Example:
        >>> data = {
        ...     "ndvi": calculate_ndvi(red, nir),
        ...     "ntl": apply_lunar_correction(ntl, lunar, quality)
        ... }
        >>> report = generate_quality_report(data)
        >>> print(json.dumps(report, indent=2))
    """
    report = {}
    
    for name, data in data_dict.items():
        if data is None:
            report[name] = {"status": "missing"}
            continue
        
        # Basic statistics
        valid_mask = ~np.isnan(data)
        valid_data = data[valid_mask]
        
        stats = {
            "shape": data.shape,
            "dtype": str(data.dtype),
            "total_pixels": data.size,
            "valid_pixels": valid_mask.sum(),
            "valid_percentage": 100.0 * valid_mask.sum() / data.size,
        }
        
        if valid_data.size > 0:
            stats.update({
                "min": float(np.min(valid_data)),
                "max": float(np.max(valid_data)),
                "mean": float(np.mean(valid_data)),
                "std": float(np.std(valid_data)),
                "unique_values": int(np.unique(valid_data).size)
            })
        
        report[name] = stats
    
    return report