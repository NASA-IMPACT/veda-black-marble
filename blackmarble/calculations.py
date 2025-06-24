"""Pure calculation functions for Black Marble processing.

This module contains all the core calculations used in the Black Marble pipeline,
implemented as pure functions with no side effects. All functions work with
numpy arrays and return new arrays without modifying inputs.
"""

import numpy as np

from .config import NDUIConfig, VIIRSConfig, LandsatConfig


def calculate_ndvi(
    red: np.ndarray, nir: np.ndarray, epsilon: float = 1e-6
) -> np.ndarray:
    """Calculate Normalized Difference Vegetation Index (NDVI).

    NDVI = (NIR - Red) / (NIR + Red)

    NDVI ranges from -1 to 1, where:
    - High positive values (0.6-0.9) indicate dense vegetation
    - Low positive values (0.2-0.4) indicate sparse vegetation
    - Near zero values indicate bare soil
    - Negative values indicate water or snow

    Args:
        red: Red band (Band 4) reflectance values
        nir: Near-infrared band (Band 5) reflectance values
        epsilon: Small value to prevent division by zero

    Returns:
        NDVI values clipped to [-1, 1]

    Example:
        >>> red = np.array([[0.1, 0.2], [0.15, 0.1]])
        >>> nir = np.array([[0.5, 0.6], [0.55, 0.7]])
        >>> ndvi = calculate_ndvi(red, nir)
        >>> print(f"NDVI range: {ndvi.min():.2f} to {ndvi.max():.2f}")
        NDVI range: 0.50 to 0.75
    """
    ndvi = (nir - red) / (nir + red + epsilon)
    return np.clip(ndvi, -1.0, 1.0)


def calculate_ndwi(
    green: np.ndarray, nir: np.ndarray, epsilon: float = 1e-6
) -> np.ndarray:
    """Calculate Normalized Difference Water Index (NDWI).

    NDWI = (Green - NIR) / (Green + NIR)

    NDWI is used to identify water bodies and monitor changes in water content.
    - Positive values typically indicate water presence
    - Negative values indicate vegetation or soil

    Args:
        green: Green band (Band 3) reflectance values
        nir: Near-infrared band (Band 5) reflectance values
        epsilon: Small value to prevent division by zero

    Returns:
        NDWI values clipped to [-1, 1]

    Example:
        >>> green = np.array([[0.3, 0.1], [0.4, 0.2]])
        >>> nir = np.array([[0.1, 0.5], [0.1, 0.6]])
        >>> ndwi = calculate_ndwi(green, nir)
        >>> # First pixel: (0.3-0.1)/(0.3+0.1) = 0.5 (water)
        >>> # Second pixel: (0.1-0.5)/(0.1+0.5) = -0.67 (vegetation)
    """
    ndwi = (green - nir) / (green + nir + epsilon)
    return np.clip(ndwi, -1.0, 1.0)


def calculate_ndui(
    ntl: np.ndarray,
    ndvi: np.ndarray,
    ndwi: np.ndarray,
    roads: np.ndarray,
    config: NDUIConfig | None = None,
) -> np.ndarray:
    """Calculate Normalized Difference Urban Index (NDUI).

    NDUI combines nighttime lights with vegetation index to identify urban areas:
    NDUI = (NTL - NDVI) / (NTL + NDVI)

    The calculation includes:
    1. Basic NDUI calculation
    2. Water masking (set to 0 where water is detected)
    3. Floor threshold (set to 0 below minimum threshold)
    4. Road enhancement (multiply by factors for road pixels)

    Args:
        ntl: Normalized nighttime lights (0-1)
        ndvi: NDVI values (-1 to 1)
        ndwi: NDWI values (-1 to 1)
        roads: OSM road classification (0=no road, 1-5=road types)
        config: NDUI configuration (uses defaults if None)

    Returns:
        NDUI values in range [0, 1]

    Example:
        >>> ntl = np.array([[0.5, 0.8], [0.3, 0.6]])
        >>> ndvi = np.array([[0.4, 0.2], [0.5, 0.3]])
        >>> ndwi = np.array([[-0.2, 0.1], [-0.3, 0.2]])
        >>> roads = np.array([[0, 1], [0, 2]])
        >>> ndui = calculate_ndui(ntl, ndvi, ndwi, roads)
    """
    if config is None:
        config = NDUIConfig()

    # Basic NDUI calculation
    ndui = (ntl - ndvi) / (ntl + ndvi + config.epsilon)
    ndui = np.clip(ndui, -1.0, 1.0)

    # Apply water mask - water pixels become -1
    ndui = np.where(ndwi >= config.water_threshold, -1.0, ndui)

    # Transform from [-1, 1] to [0, 1] range
    ndui = (ndui + 1.0) / 2.0

    # Apply floor threshold
    ndui = np.where(ndui < config.floor, 0.0, ndui)

    # Road enhancement
    # Major roads (types 1 and 4)
    ndui = np.where(
        (roads == 1) | (roads == 4), ndui * config.road_enhancement_major, ndui
    )

    # Minor roads (types 2, 3, and 5)
    ndui = np.where(
        (roads == 2) | (roads == 3) | (roads == 5),
        ndui * config.road_enhancement_minor,
        ndui,
    )

    # Ensure final values are in [0, 1]
    return np.clip(ndui, 0.0, 1.0)


def apply_lunar_correction(
    ntl: np.ndarray,
    lunar_irradiance: np.ndarray,
    quality_flags: np.ndarray,
    config: VIIRSConfig | None = None,
) -> np.ndarray:
    """Apply lunar irradiance correction to nighttime lights.

    Corrects VIIRS nighttime lights for lunar illumination:
    1. Subtract lunar irradiance from raw radiance
    2. Apply quality mask (only keep high quality and twilight pixels)
    3. Convert units and apply floor/ceiling thresholds
    4. Normalize to 0-1 range

    Args:
        ntl: Raw nighttime lights radiance
        lunar_irradiance: Lunar irradiance values
        quality_flags: Quality flag array (0=high quality, 2=twilight)
        config: VIIRS configuration (uses defaults if None)

    Returns:
        Corrected and normalized NTL values (0-1)

    Example:
        >>> ntl = np.array([[100.0, 200.0], [150.0, 65535.0]])
        >>> lunar = np.array([[10.0, 20.0], [15.0, 0.0]])
        >>> quality = np.array([[0, 1], [2, 3]])
        >>> corrected = apply_lunar_correction(ntl, lunar, quality)
        >>> # First pixel: (100-10)/10 = 9.0 -> normalized
        >>> # Second pixel: quality=1, masked to 0
    """
    if config is None:
        config = VIIRSConfig()

    # Handle fill values - replace with 0
    ntl_clean = np.where(ntl > config.fill_value, 0.0, ntl).astype(np.float32)
    lunar_clean = np.where(
        lunar_irradiance > config.fill_value, 0.0, lunar_irradiance
    ).astype(np.float32)

    # Create quality mask - based on analysis, we need to EXCLUDE 0 and 2
    # because quality==0 always has VIIRS=0, and quality==2 max is 2.0
    # Using NOT(0 or 2) gives us the bright pixels we need
    quality_mask = ~np.isin(quality_flags, config.quality_valid_flags)
    quality_mask = quality_mask.astype(np.float32)

    # Apply lunar correction
    adjusted_ntl = (ntl_clean - lunar_clean) * quality_mask

    # Convert units (from instrument units to nW/cm²/sr)
    adjusted_ntl = adjusted_ntl / config.lunar_unit_conversion

    # Apply ceiling threshold
    adjusted_ntl = np.where(adjusted_ntl > config.ceiling, config.ceiling, adjusted_ntl)

    # Apply floor threshold
    adjusted_ntl = np.where(adjusted_ntl < config.floor, 0.0, adjusted_ntl)

    # Normalize to 0-1 range
    return adjusted_ntl / config.ceiling


def scale_landsat_to_reflectance(
    band_data: np.ndarray,
    config: LandsatConfig | None = None,
    nodata_value: float | None = None,
) -> np.ndarray:
    """Scale Landsat Collection 2 DN values to surface reflectance.

    Applies the standard Landsat Collection 2 scaling equation:
    Reflectance = DN * scale_factor + offset

    The function handles:
    1. Detection of already-scaled reflectance data
    2. NoData value handling
    3. Valid DN range checking
    4. Negative reflectance removal

    Args:
        band_data: Raw band data (DN values or reflectance)
        config: Landsat configuration (uses defaults if None)
        nodata_value: Value indicating no data (replaced with NaN)

    Returns:
        Surface reflectance values (NaN for invalid pixels)

    Example:
        >>> # DN values
        >>> dn = np.array([[10000, 20000], [30000, 40000]])
        >>> reflectance = scale_landsat_to_reflectance(dn)
        >>> # 10000 * 0.0000275 - 0.2 = 0.075
        >>> print(f"Reflectance: {reflectance[0,0]:.3f}")
        Reflectance: 0.075

        >>> # Already reflectance
        >>> refl = np.array([[0.1, 0.3], [0.5, 0.8]])
        >>> result = scale_landsat_to_reflectance(refl)
        >>> np.array_equal(result, refl)  # Should be unchanged
        True
    """
    if config is None:
        config = LandsatConfig()

    # Handle NoData values
    if nodata_value is not None:
        band_data = np.where(band_data == nodata_value, np.nan, band_data)

    # Check if data is already reflectance (values between -0.5 and 1.5)
    data_min = np.nanmin(band_data)
    data_max = np.nanmax(band_data)

    if data_max <= 1.5 and data_min >= -0.5:
        # Already reflectance, just ensure no negative values
        return np.where(band_data < 0, np.nan, band_data)

    # Apply scaling equation
    scaled = band_data * config.band_scale_factor + config.band_offset

    # Mark pixels outside valid DN range as invalid
    scaled = np.where(
        (band_data < config.valid_dn_min) | (band_data > config.valid_dn_max),
        np.nan,
        scaled,
    )

    # Remove negative reflectance values
    scaled = np.where(scaled < 0, np.nan, scaled)

    return scaled


def enhance_with_roads(
    base_layer: np.ndarray,
    roads: np.ndarray,
    major_factor: float = 1.3,
    minor_factor: float = 1.1,
) -> np.ndarray:
    """Enhance values along road networks.

    Multiplies base layer values by enhancement factors where roads are present.
    This is useful for enhancing urban signatures along transportation corridors.

    Road types:
    - Major roads (1, 4): Primary/secondary highways, trunk roads
    - Minor roads (2, 3, 5): Motorways, residential, service roads

    Args:
        base_layer: Base data layer to enhance
        roads: Road classification (0=no road, 1-5=road types)
        major_factor: Enhancement factor for major roads
        minor_factor: Enhancement factor for minor roads

    Returns:
        Enhanced array with same shape as input

    Example:
        >>> data = np.array([[0.5, 0.6], [0.4, 0.7]])
        >>> roads = np.array([[1, 0], [2, 0]])
        >>> enhanced = enhance_with_roads(data, roads)
        >>> enhanced[0, 0]  # 0.5 * 1.3 = 0.65
        0.65
    """
    # Create a copy to avoid modifying input
    enhanced = base_layer.copy()

    # Enhance major roads
    major_mask = (roads == 1) | (roads == 4)
    enhanced = np.where(major_mask, enhanced * major_factor, enhanced)

    # Enhance minor roads
    minor_mask = (roads == 2) | (roads == 3) | (roads == 5)
    enhanced = np.where(minor_mask, enhanced * minor_factor, enhanced)

    return enhanced


def normalize_to_range(
    data: np.ndarray,
    output_min: float = 0.0,
    output_max: float = 1.0,
    input_min: float | None = None,
    input_max: float | None = None,
) -> np.ndarray:
    """Normalize array values to specified range.

    Linear scaling from input range to output range:
    output = (data - input_min) / (input_max - input_min) * (output_max - output_min) + output_min

    Args:
        data: Input array to normalize
        output_min: Minimum value of output range
        output_max: Maximum value of output range
        input_min: Minimum value of input range (auto-detected if None)
        input_max: Maximum value of input range (auto-detected if None)

    Returns:
        Normalized array with values in [output_min, output_max]

    Example:
        >>> data = np.array([0, 50, 100])
        >>> normalized = normalize_to_range(data, 0, 1)
        >>> normalized
        array([0. , 0.5, 1. ])
    """
    if input_min is None:
        input_min = float(np.nanmin(data))
    if input_max is None:
        input_max = float(np.nanmax(data))

    # Handle case where all values are the same
    if input_max == input_min:
        return np.full_like(data, (output_min + output_max) / 2)

    # Linear scaling
    scaled = (data - input_min) / (input_max - input_min)
    scaled = scaled * (output_max - output_min) + output_min

    return scaled
