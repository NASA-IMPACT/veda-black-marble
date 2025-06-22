"""Data scaling and radiometric corrections.

This module handles converting digital numbers to physical units,
applying atmospheric corrections, and lunar illumination corrections.
"""

from typing import Any
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
    # Currently in bm_algorithms.py
    ...


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
    # Currently in calculations.py is_already_reflectance()
    ...


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
    # Currently in bm_algorithms.py
    ...


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
    # Currently in bm_algorithms.py
    ...


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
    # Logic from bm.py NDUI calculation
    ...


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
    # Currently in calculations.py
    ...


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
    # Currently in processing.py
    ...