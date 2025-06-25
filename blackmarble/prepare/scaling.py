"""Data scaling and radiometric corrections.

This module handles converting digital numbers to physical units,
applying atmospheric corrections, and lunar illumination corrections.
"""

from typing import Any
import numpy as np
import numpy.typing as npt

# Type aliases
ArrayLike = npt.NDArray[Any]


def scale_landsat_bands(
    band_data: ArrayLike,
    scale_factor: float = 0.0000275,
    add_offset: float = -0.2,
    valid_range: tuple[float, float] | None = None
) -> ArrayLike:
    """Scale Landsat Level-2 surface reflectance bands to physical values.
    
    Args:
        band_data: Raw band digital numbers
        scale_factor: Multiplicative scale factor (default for Landsat 8/9)
        add_offset: Additive offset (default for Landsat 8/9)
        valid_range: Optional tuple of (min, max) valid DN values
        
    Returns:
        Scaled surface reflectance values
    """
    if valid_range is None:
        valid_range = (7273, 43636)
    
    # Mask invalid values
    band_masked = np.where(band_data < valid_range[0], 0, band_data)
    band_masked = np.where(band_masked > valid_range[1], 0, band_masked)
    
    # Apply scale and offset
    reflectance = band_masked * scale_factor + add_offset
    
    # Mask out-of-range DN values
    reflectance = np.where(band_masked == 0, np.nan, reflectance)
    
    return reflectance


def detect_if_already_scaled(
    band_data: ArrayLike,
    reflectance_threshold: float = 2.0
) -> bool:
    """Detect if Landsat data is already scaled to reflectance.
    
    Args:
        band_data: Input band data
        reflectance_threshold: Max expected reflectance value
        
    Returns:
        True if data appears to be already scaled
    """
    # Based on logic from calculations.py:248
    # Check if data is already reflectance (values between -0.5 and threshold)
    data_min = np.nanmin(band_data)
    data_max = np.nanmax(band_data)
    
    # Data is likely already reflectance if max is below threshold
    # and min is above typical negative reflectance values
    return data_max <= reflectance_threshold and data_min >= -0.5


def apply_lunar_correction(
    vnp_data: ArrayLike,
    lunar_irradiance: ArrayLike,
    moon_illumination_fraction: ArrayLike | None = None
) -> ArrayLike:
    """Apply lunar BRDF correction to VIIRS nighttime lights.
    
    Args:
        vnp_data: VIIRS Day/Night Band radiance
        lunar_irradiance: Calculated lunar irradiance at surface
        moon_illumination_fraction: Optional moon illumination fraction
        
    Returns:
        Lunar-corrected radiance values
    """
    # Basic correction: subtract lunar contribution
    corrected = vnp_data - lunar_irradiance
    
    # Ensure non-negative
    corrected = np.where(corrected < 0, 0, corrected)
    
    # Advanced correction could be implemented here using moon_illumination_fraction
    # and surface BRDF models
    
    return corrected


def calculate_ntli(
    vnp_radiance: ArrayLike,
    lunar_irradiance: ArrayLike,
    quality_flag: ArrayLike,
    valid_flags: list[int] | None = None
) -> ArrayLike:
    """Calculate Nighttime Light Index with lunar correction.
    
    Args:
        vnp_radiance: VIIRS DNB radiance values
        lunar_irradiance: Lunar irradiance values for correction
        quality_flag: Quality flag array from VIIRS
        valid_flags: List of valid quality flag values (default: [0, 2])
        
    Returns:
        Lunar-corrected nighttime light radiance
    """
    if valid_flags is None:
        valid_flags = [0, 2]
    
    # Handle fill values
    vnp_clean = np.where(vnp_radiance > 65530, 0, vnp_radiance)
    lunar_clean = np.where(lunar_irradiance > 65530, 0, lunar_irradiance)
    
    # Create quality mask
    quality_mask = np.isin(quality_flag, valid_flags)
    
    # Apply lunar correction
    lunar_corrected = vnp_clean - lunar_clean
    
    # Set invalid pixels to 0
    lunar_corrected = np.where(quality_mask, lunar_corrected, 0)
    lunar_corrected = np.where(lunar_corrected < 0, 0, lunar_corrected)
    
    return lunar_corrected


def apply_viirs_floor_ceiling(
    ntl_data: ArrayLike,
    floor_value: float = 0.1,
    ceiling_value: float = 10.0
) -> ArrayLike:
    """Apply floor and ceiling thresholds to VIIRS NTL data.
    
    Args:
        ntl_data: Nighttime lights data
        floor_value: Minimum threshold (set to 0 below this)
        ceiling_value: Maximum threshold (clip above this)
        
    Returns:
        Thresholded NTL data
    """
    # Based on logic from bm_algorithms.py:119 (NDUI calculation)
    
    # Apply ceiling first (clip values above ceiling)
    ntl_processed = np.where(ntl_data > ceiling_value, ceiling_value, ntl_data)
    
    # Apply floor (set values below floor to 0)
    ntl_processed = np.where(ntl_processed < floor_value, 0.0, ntl_processed)
    
    return ntl_processed


def normalize_to_range(
    data: ArrayLike,
    out_min: float = 0.0,
    out_max: float = 1.0,
    in_min: float | None = None,
    in_max: float | None = None
) -> ArrayLike:
    """Normalize data to specified output range.
    
    Args:
        data: Input data
        out_min: Minimum of output range
        out_max: Maximum of output range
        in_min: Minimum of input range (auto-detected if None)
        in_max: Maximum of input range (auto-detected if None)
        
    Returns:
        Normalized data
    """
    if in_min is None:
        in_min = float(np.nanmin(data))
    if in_max is None:
        in_max = float(np.nanmax(data))

    # Handle case where all values are the same
    if in_max == in_min:
        return np.full_like(data, (out_min + out_max) / 2)

    # Linear scaling
    scaled = (data - in_min) / (in_max - in_min)
    scaled = scaled * (out_max - out_min) + out_min

    return scaled


def scale_to_byte(
    data: ArrayLike,
    in_min: float | None = None,
    in_max: float | None = None
) -> ArrayLike:
    """Scale data to uint8 range [0, 255].
    
    Args:
        data: Input data
        in_min: Minimum input value (auto-detected if None)
        in_max: Maximum input value (auto-detected if None)
        
    Returns:
        Scaled uint8 array
    """
    # Handle NaN values
    valid_mask = ~np.isnan(data)

    if not np.any(valid_mask):
        # No valid data
        return np.full(data.shape, 0, dtype=np.uint8)

    # Auto-detect range if not provided
    if in_min is None:
        in_min = float(np.nanmin(data))
    if in_max is None:
        in_max = float(np.nanmax(data))

    # Normalize to 0-1
    if in_max > in_min:
        normalized = (data - in_min) / (in_max - in_min)
    else:
        normalized = np.zeros_like(data)

    # Scale to 0-255
    scaled = normalized * 255

    # Convert to uint8
    result = np.zeros(data.shape, dtype=np.uint8)
    result[valid_mask] = np.clip(np.round(scaled[valid_mask]), 0, 255).astype(np.uint8)
    result[~valid_mask] = 0  # nodata value

    return result