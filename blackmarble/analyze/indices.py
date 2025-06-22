"""Spectral indices and scientific calculations.

This module provides functions for calculating various spectral indices
and derived products used in the Black Marble algorithm.
"""

from typing import Any
import numpy.typing as npt

# Type aliases
ArrayLike = npt.NDArray[Any]


def calculate_ndvi(
    nir: ArrayLike,
    red: ArrayLike,
    epsilon: float = 0.000001
) -> ArrayLike:
    """Calculate Normalized Difference Vegetation Index (NDVI).
    
    Args:
        nir: Near-infrared band (Band 5 for Landsat 8/9)
        red: Red band (Band 4 for Landsat 8/9)
        epsilon: Small value to prevent division by zero
        
    Returns:
        NDVI array with values clamped to [-1, 1]
    """
    # Currently in bm_algorithms.py
    ...


def calculate_ndwi(
    green: ArrayLike,
    nir: ArrayLike,
    epsilon: float = 0.000001
) -> ArrayLike:
    """Calculate Normalized Difference Water Index (NDWI).
    
    Args:
        green: Green band (Band 3 for Landsat 8/9)
        nir: Near-infrared band (Band 5 for Landsat 8/9)
        epsilon: Small value to prevent division by zero
        
    Returns:
        NDWI array with values clamped to [-1, 1]
    """
    # Currently in bm_algorithms.py
    ...


def calculate_ndui(
    ntl: ArrayLike,
    ndvi: ArrayLike,
    ndwi: ArrayLike | None = None,
    ntl_floor: float = 0.1,
    ntl_ceiling: float = 10.0,
    ndui_floor: float = 0.02,
    epsilon: float = 0.00001
) -> ArrayLike:
    """Calculate Normalized Difference Urban Index (NDUI).
    
    Args:
        ntl: Nighttime light radiance values
        ndvi: NDVI values from calculate_ndvi()
        ndwi: Optional NDWI values for water masking
        ntl_floor: Minimum NTL threshold
        ntl_ceiling: Maximum NTL threshold for normalization
        ndui_floor: Minimum NDUI threshold
        epsilon: Small value to prevent division by zero
        
    Returns:
        NDUI array with values in [0, 1]
    """
    # Currently in bm_algorithms.py
    ...


def calculate_built_area_index(
    ndvi: ArrayLike,
    ndwi: ArrayLike,
    swir1: ArrayLike,
    swir2: ArrayLike
) -> ArrayLike:
    """Calculate Built-up Area Index (BAI).
    
    Args:
        ndvi: NDVI array
        ndwi: NDWI array
        swir1: SWIR1 band (Band 6 for Landsat 8/9)
        swir2: SWIR2 band (Band 7 for Landsat 8/9)
        
    Returns:
        Built-up area index
    """
    # Additional index that could be useful
    ...


def calculate_normalized_burn_ratio(
    nir: ArrayLike,
    swir2: ArrayLike,
    epsilon: float = 0.000001
) -> ArrayLike:
    """Calculate Normalized Burn Ratio (NBR).
    
    Args:
        nir: Near-infrared band
        swir2: SWIR2 band
        epsilon: Small value to prevent division by zero
        
    Returns:
        NBR array useful for distinguishing urban areas
    """
    # Can help distinguish urban from bare soil
    ...


def calculate_brightness_temperature(
    thermal_band: ArrayLike,
    metadata: dict[str, Any]
) -> ArrayLike:
    """Calculate brightness temperature from thermal band.
    
    Args:
        thermal_band: Thermal infrared band (Band 10 for Landsat 8/9)
        metadata: Metadata containing calibration constants
        
    Returns:
        Brightness temperature in Celsius
    """
    # For urban heat island analysis
    ...


def combine_indices_for_classification(
    ndvi: ArrayLike,
    ndwi: ArrayLike,
    ndui: ArrayLike,
    weights: dict[str, float] | None = None
) -> ArrayLike:
    """Combine multiple indices for land cover classification.
    
    Args:
        ndvi: NDVI array
        ndwi: NDWI array
        ndui: NDUI array
        weights: Optional weights for each index
        
    Returns:
        Combined classification index
    """
    # Weighted combination of indices
    ...


def calculate_index_statistics(
    index_array: ArrayLike,
    mask: ArrayLike | None = None
) -> dict[str, float]:
    """Calculate statistics for a spectral index.
    
    Args:
        index_array: Input index array
        mask: Optional mask for valid pixels
        
    Returns:
        Dictionary with min, max, mean, std, percentiles
    """
    # Statistical summary of indices
    ...