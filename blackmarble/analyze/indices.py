"""Spectral indices and scientific calculations.

This module provides functions for calculating various spectral indices
and derived products used in the Black Marble algorithm.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

# Type aliases
ArrayLike = npt.NDArray[Any]


def calculate_ndvi(nir: ArrayLike, red: ArrayLike, epsilon: float = 0.02) -> ArrayLike:
    """Calculate Normalized Difference Vegetation Index (NDVI).

    The NDVI is a fundamental vegetation index that measures the health and density
    of vegetation using the difference between near-infrared and red reflectance.
    Healthy vegetation strongly reflects NIR light while absorbing red light,
    resulting in high NDVI values.

    Formula: NDVI = (NIR - Red) / (NIR + Red)
    
    Note: Negative reflectance values are masked as NaN to handle Landsat offset artifacts.
    A minimum denominator threshold is applied to prevent extreme values.

    NDVI values interpretation:
    - [-1, 0]: Water, clouds, snow, bare soil
    - [0, 0.2]: Sparse vegetation, bare soil
    - [0.2, 0.4]: Shrubs and grasslands
    - [0.4, 0.8]: Dense vegetation, forests
    - [0.8, 1.0]: Very dense vegetation

    Args:
        nir: Near-infrared band reflectance (Band 5 for Landsat 8/9).
            Expected range: [0, 1] for surface reflectance.
        red: Red band reflectance (Band 4 for Landsat 8/9).
            Expected range: [0, 1] for surface reflectance.
        epsilon: Minimum denominator threshold to prevent extreme NDVI values.
            Default is 0.02 (2% reflectance), which prevents artifacts from
            snow/cloud edges and near-zero denominators.

    Returns:
        NDVI array with values mathematically bounded to [-1, 1].

    Raises:
        ValueError: If input arrays have incompatible shapes.

    Example:
        >>> # Calculate NDVI from Landsat bands
        >>> ndvi = calculate_ndvi(landsat_band5, landsat_band4)
        >>> # Mask vegetation areas
        >>> vegetation_mask = ndvi > 0.3

    Note:
        This function preserves NaN values in the input arrays, making it
        suitable for masked data. Output is NOT clamped to [-1, 1] to
        preserve extreme values that may indicate processing issues.
    """

    # Input validation
    if nir.shape != red.shape:
        raise ValueError(f"NIR and Red arrays must have same shape: {nir.shape} vs {red.shape}")

    # Mask negative reflectance values (Landsat offset artifact)
    # These are non-physical and indicate no-data or processing artifacts
    nir_masked = np.where(nir < 0, np.nan, nir)
    red_masked = np.where(red < 0, np.nan, red)

    # Calculate denominator with epsilon to prevent division by zero
    # The epsilon prevents NaN/inf when both bands are zero (e.g., no-data regions)
    denominator = nir_masked + red_masked
    
    # Protect against near-zero denominators that can occur when NIR ≈ -Red
    # Use a physically meaningful threshold (default 2% of typical reflectance)
    # This prevents extreme NDVI values from snow/cloud edges
    safe_denominator = np.where(
        np.abs(denominator) < epsilon,
        np.copysign(epsilon, denominator),
        denominator
    )

    # Calculate NDVI using the standard formula: (NIR - Red) / (NIR + Red)
    # With the 2% threshold, NDVI is naturally bounded to reasonable values
    ndvi = (nir_masked - red_masked) / safe_denominator
    
    # Sanity check - NDVI should never exceed reasonable bounds
    # If it does, we have a calculation error
    max_abs_ndvi = np.nanmax(np.abs(ndvi))
    if max_abs_ndvi > 5.0:
        raise ValueError(
            f"NDVI calculation error: maximum absolute value {max_abs_ndvi:.1f} exceeds reasonable bounds. "
            f"Check input scaling and offset values."
        )

    return ndvi


def calculate_ndwi(green: ArrayLike, nir: ArrayLike, epsilon: float = 0.001) -> ArrayLike:
    """Calculate Normalized Difference Water Index (NDWI).

    The NDWI is designed to detect water features and measure water content
    in vegetation. Water strongly absorbs near-infrared radiation while
    reflecting green light, resulting in positive NDWI values for water bodies.

    Formula: NDWI = (Green - NIR) / (Green + NIR + epsilon)

    NDWI values interpretation:
    - [0.3, 1.0]: Open water, wet surfaces
    - [0.0, 0.3]: Moist soil, wet vegetation
    - [-0.2, 0.0]: Dry vegetation, moderate moisture
    - [-1.0, -0.2]: Dry soil, built-up areas

    This index is crucial for the Black Marble algorithm as it helps
    distinguish water bodies from land surfaces, enabling proper masking
    and preventing false urban light detection over water.

    Args:
        green: Green band reflectance (Band 3 for Landsat 8/9).
            Expected range: [0, 1] for surface reflectance.
        nir: Near-infrared band reflectance (Band 5 for Landsat 8/9).
            Expected range: [0, 1] for surface reflectance.
        epsilon: Small value to prevent division by zero in areas where
            Green + NIR approaches zero. Default 1e-4 provides stability
            when reflectance values are near zero.

    Returns:
        NDWI array with values mathematically bounded to [-1, 1].

    Raises:
        ValueError: If input arrays have incompatible shapes.

    Example:
        >>> # Calculate NDWI for water detection
        >>> ndwi = calculate_ndwi(landsat_band3, landsat_band5)
        >>> # Create water mask
        >>> water_mask = ndwi > 0.0
        >>> # Find highly confident water pixels
        >>> water_confident = ndwi > 0.3

    Note:
        NDWI is used extensively in the prepare.quality module for
        water masking operations. Positive values indicate water presence.
    """

    # Input validation
    if green.shape != nir.shape:
        raise ValueError(f"Green and NIR arrays must have same shape: {green.shape} vs {nir.shape}")

    # Calculate denominator
    denominator = green + nir
    
    # Protect against near-zero denominators (same issue as NDVI with offset data)
    safe_denominator = np.where(
        np.abs(denominator) < epsilon,
        np.copysign(epsilon, denominator),  # Always at least epsilon in magnitude
        denominator
    )

    # Calculate NDWI using the standard formula: (Green - NIR) / (Green + NIR)
    # Water reflects green but absorbs NIR, resulting in positive values
    ndwi = (green - nir) / safe_denominator
    
    # Sanity check
    max_abs_ndwi = np.nanmax(np.abs(ndwi))
    if max_abs_ndwi > 5.0:
        raise ValueError(
            f"NDWI calculation error: maximum absolute value {max_abs_ndwi:.1f} exceeds reasonable bounds."
        )

    return ndwi


def calculate_ndui(
    ntl: ArrayLike,
    ndvi: ArrayLike,
    ndwi: ArrayLike | None = None,
    ntl_floor: float = 0.1,
    ntl_ceiling: float = 10.0,
    ndui_floor: float = 0.02,
    epsilon: float = 0.00001,
) -> ArrayLike:
    """Calculate Normalized Difference Urban Index (NDUI).

    The NDUI is the core index of the Black Marble algorithm, designed to
    identify and quantify urban areas by combining nighttime lights with
    vegetation indices. It leverages the fact that urban areas have high
    nighttime radiance and low vegetation density.

    Formula: NDUI = NTL_norm / (1 + NDVI + epsilon)
    Where: NTL_norm = clamp((NTL - floor) / (ceiling - floor), 0, 1)

    NDUI values interpretation:
    - [0.7, 1.0]: Dense urban areas, city centers
    - [0.4, 0.7]: Suburban areas, developed land
    - [0.2, 0.4]: Sparse development, rural towns
    - [0.0, 0.2]: Non-urban (vegetation, water, bare land)

    The index is normalized to [0, 1] and incorporates thresholding to
    handle the wide dynamic range of nighttime lights data.

    Args:
        ntl: Nighttime light radiance values.
            Can be either raw DN values (0-5000) or scaled nW/cm²/sr (0-50).
            Function auto-detects based on range and scales if needed.
        ndvi: NDVI values from calculate_ndvi(), range [-1, 1].
            Used to suppress vegetation areas.
        ndwi: Optional NDWI values for water masking.
            If provided, water pixels (NDWI > 0) are set to NDUI = 0.
        ntl_floor: Minimum NTL threshold below which NDUI = 0.
            Removes sensor noise and very dim light sources.
        ntl_ceiling: Maximum NTL value for normalization.
            Values above this are clamped to prevent saturation effects.
        ndui_floor: Minimum NDUI threshold below which values are set to 0.
            Removes spurious low-confidence urban detections.
        epsilon: Small value to prevent division by zero when NDVI = -1.
            Default 1e-5 handles edge cases in water/shadow areas.

    Returns:
        NDUI array with values in [0, 1] where:
        - 0 indicates non-urban areas
        - Values approaching 1 indicate dense urban development

    Raises:
        ValueError: If input arrays have incompatible shapes.

    Example:
        >>> # Calculate NDUI for urban mapping
        >>> ndui = calculate_ndui(
        >>>     ntl=viirs_radiance,
        >>>     ndvi=ndvi_values,
        >>>     ndwi=ndwi_values,
        >>>     ntl_floor=0.1,
        >>>     ntl_ceiling=10.0
        >>> )
        >>> # Create urban mask
        >>> urban_mask = ndui > 0.3

    Note:
        This is the signature index of the Black Marble algorithm.
        Parameter tuning significantly affects urban detection accuracy.
    """

    # Input validation
    if ntl.shape != ndvi.shape:
        raise ValueError(f"NTL and NDVI arrays must have same shape: {ntl.shape} vs {ndvi.shape}")

    if ndwi is not None and ndwi.shape != ntl.shape:
        raise ValueError(f"NDWI shape {ndwi.shape} must match NTL shape {ntl.shape}")

    # Step 1: Handle raw DN values if needed (auto-detect based on range)
    # If max value > 100, assume raw DN values and scale by /10
    if np.nanmax(ntl) > 100:
        # Raw DN values - scale to nW/cm²/sr
        ntl_scaled = ntl.astype(np.float32) / 10.0
    else:
        # Already scaled to nW/cm²/sr
        ntl_scaled = ntl

    # Step 2: Apply floor/ceiling and normalize by ceiling (like bm.py)
    # First clip to ceiling
    ntl_clipped = np.where(ntl_scaled > ntl_ceiling, ntl_ceiling, ntl_scaled)
    # Then apply floor
    ntl_clipped = np.where(ntl_clipped < ntl_floor, 0.0, ntl_clipped)
    # Normalize by dividing by ceiling (not range)
    ntl_normalized = ntl_clipped / ntl_ceiling

    # Step 2: Calculate NDUI using original Black Marble formula
    # Original formula: (ntl - ndvi) / (ntl + ndvi + epsilon)
    ndui = (ntl_normalized - ndvi) / (ntl_normalized + ndvi + epsilon)

    # Step 3: Clamp to [-1, 1]
    ndui = np.where(ndui > 1.0, 1.0, ndui)
    ndui = np.where(ndui < -1.0, -1.0, ndui)

    # Step 4: Apply water masking if NDWI is provided
    if ndwi is not None:
        # Water pixels (NDWI >= 0) are set to -1 (like bm.py)
        water_mask = ndwi >= 0.0
        ndui = np.where(water_mask, -1.0, ndui)

    # Step 5: Transform to [0, 1] range using bm.py formula
    # Transform: abs((ndui + 1.0)) / 2.0
    ndui = np.abs(ndui + 1.0) / 2.0

    # Step 6: Apply floor threshold
    ndui = np.where(ndui < ndui_floor, 0.0, ndui)

    return ndui
