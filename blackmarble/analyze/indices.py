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
    
    The NDVI is a fundamental vegetation index that measures the health and density
    of vegetation using the difference between near-infrared and red reflectance.
    Healthy vegetation strongly reflects NIR light while absorbing red light,
    resulting in high NDVI values.
    
    Formula: NDVI = (NIR - Red) / (NIR + Red + epsilon)
    
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
        epsilon: Small value to prevent division by zero in areas where
            NIR + Red approaches zero. Default 1e-6 is sufficient for
            most surface reflectance values.
            
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
    import numpy as np
    
    # Input validation
    if nir.shape != red.shape:
        raise ValueError(f"NIR and Red arrays must have same shape: {nir.shape} vs {red.shape}")
    
    # Calculate denominator with epsilon to prevent division by zero
    # The epsilon prevents NaN/inf when both bands are zero (e.g., no-data regions)
    denominator = nir + red + epsilon
    
    # Calculate NDVI using the standard formula: (NIR - Red) / (NIR + Red)
    # No explicit clamping - let mathematical bounds apply naturally
    # Result is mathematically bounded to [-1, 1] when inputs are non-negative
    ndvi = (nir - red) / denominator
    
    return ndvi


def calculate_ndwi(
    green: ArrayLike,
    nir: ArrayLike,
    epsilon: float = 0.000001
) -> ArrayLike:
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
            Green + NIR approaches zero. Default 1e-6 handles edge cases.
            
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
    import numpy as np
    
    # Input validation
    if green.shape != nir.shape:
        raise ValueError(f"Green and NIR arrays must have same shape: {green.shape} vs {nir.shape}")
    
    # Calculate denominator with epsilon to prevent division by zero
    # Water absorbs NIR strongly, so denominator can approach zero over water
    denominator = green + nir + epsilon
    
    # Calculate NDWI using the standard formula: (Green - NIR) / (Green + NIR)
    # Water reflects green but absorbs NIR, resulting in positive values
    ndwi = (green - nir) / denominator
    
    return ndwi


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
        ntl: Nighttime light radiance values in nW/cm²/sr.
            Should be lunar-corrected VIIRS DNB radiance.
            Typical range: [0, 50] with most urban areas < 10.
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
    import numpy as np
    
    # Input validation
    if ntl.shape != ndvi.shape:
        raise ValueError(f"NTL and NDVI arrays must have same shape: {ntl.shape} vs {ndvi.shape}")
    
    if ndwi is not None and ndwi.shape != ntl.shape:
        raise ValueError(f"NDWI shape {ndwi.shape} must match NTL shape {ntl.shape}")
    
    # Step 1: Apply floor/ceiling and normalize by ceiling (like bm.py)
    # First clip to ceiling
    ntl_clipped = np.where(ntl > ntl_ceiling, ntl_ceiling, ntl)
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
    ndui = np.abs((ndui + 1.0)) / 2.0
    
    # Step 6: Apply floor threshold
    ndui = np.where(ndui < ndui_floor, 0.0, ndui)
    
    return ndui


def calculate_built_area_index(
    ndvi: ArrayLike,
    ndwi: ArrayLike,
    swir1: ArrayLike,
    swir2: ArrayLike
) -> ArrayLike:
    """Calculate Built-up Area Index (BAI).
    
    The BAI combines multiple spectral indices to identify built-up areas
    using the spectral characteristics of urban materials. It leverages
    the fact that built-up areas have low vegetation, low water content,
    and high reflectance in SWIR bands due to concrete and asphalt.
    
    Formula: BAI = (SWIR1 + SWIR2) / (1 + NDVI) - NDWI
    
    BAI values interpretation:
    - [0.5, 1.0]: High-confidence built-up areas
    - [0.2, 0.5]: Moderate built-up areas, mixed urban/vegetation
    - [0.0, 0.2]: Low built-up probability
    - [<0.0]: Non-built areas (vegetation, water)
    
    This index complements NDUI by using different spectral bands
    and can help validate urban detection results.
    
    Args:
        ndvi: NDVI array from calculate_ndvi(), range [-1, 1].
        ndwi: NDWI array from calculate_ndwi(), range [-1, 1].
        swir1: SWIR1 band reflectance (Band 6 for Landsat 8/9).
            Expected range: [0, 1] for surface reflectance.
        swir2: SWIR2 band reflectance (Band 7 for Landsat 8/9).
            Expected range: [0, 1] for surface reflectance.
            
    Returns:
        BAI array with values typically in [-1, 2] range.
        Higher values indicate higher probability of built-up areas.
        
    Raises:
        ValueError: If input arrays have incompatible shapes.
        
    Example:
        >>> # Calculate BAI for built-up area detection
        >>> bai = calculate_built_area_index(ndvi, ndwi, swir1, swir2)
        >>> # Create built-up mask
        >>> built_mask = bai > 0.3
        
    Note:
        BAI is useful for cross-validation with NDUI results
        and can help identify areas where NDUI may be uncertain.
    """
    import numpy as np
    
    # Input validation - all arrays must have matching dimensions
    shapes = [ndvi.shape, ndwi.shape, swir1.shape, swir2.shape]
    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError(
            f"All input arrays must have same shape. Got: "
            f"ndvi={ndvi.shape}, ndwi={ndwi.shape}, "
            f"swir1={swir1.shape}, swir2={swir2.shape}"
        )
    
    # Calculate SWIR component - built-up areas have high SWIR reflectance
    # due to concrete, asphalt, and other urban materials
    swir_component = swir1 + swir2
    
    # Normalize by vegetation (add 1 to NDVI to shift range from [-1,1] to [0,2])
    # This ensures denominator is always positive and weights by vegetation absence
    # Areas with more vegetation (high NDVI) get lower BAI values
    vegetation_normalized = swir_component / (1.0 + ndvi)
    
    # Subtract water component - water areas should have low BAI
    # High NDWI (water) reduces BAI, preventing false urban detection over water
    bai = vegetation_normalized - ndwi
    
    return bai


def calculate_normalized_burn_ratio(
    nir: ArrayLike,
    swir2: ArrayLike,
    epsilon: float = 0.000001
) -> ArrayLike:
    """Calculate Normalized Burn Ratio (NBR).
    
    The NBR was originally designed for burn severity mapping but is also
    useful for distinguishing urban areas from bare soil and natural surfaces.
    Urban materials and bare soil have different reflectance patterns in
    NIR vs SWIR2 bands, making NBR valuable for urban analysis.
    
    Formula: NBR = (NIR - SWIR2) / (NIR + SWIR2 + epsilon)
    
    NBR values interpretation:
    - [0.4, 1.0]: Healthy vegetation
    - [0.1, 0.4]: Sparse vegetation, grassland
    - [-0.1, 0.1]: Bare soil, urban areas, rock
    - [-0.4, -0.1]: Burned areas, water
    - [-1.0, -0.4]: Severe burns, clouds
    
    In urban mapping context, NBR helps distinguish between:
    - Urban areas (moderate negative values)
    - Bare soil (values near zero)
    - Water bodies (strongly negative values)
    
    Args:
        nir: Near-infrared band reflectance (Band 5 for Landsat 8/9).
            Expected range: [0, 1] for surface reflectance.
        swir2: SWIR2 band reflectance (Band 7 for Landsat 8/9).
            Expected range: [0, 1] for surface reflectance.
        epsilon: Small value to prevent division by zero.
            Default 1e-6 handles edge cases in very dark areas.
            
    Returns:
        NBR array with values mathematically bounded to [-1, 1].
        
    Raises:
        ValueError: If input arrays have incompatible shapes.
        
    Example:
        >>> # Calculate NBR for urban/soil discrimination
        >>> nbr = calculate_normalized_burn_ratio(nir_band, swir2_band)
        >>> # Identify potential urban areas
        >>> urban_candidates = (nbr > -0.2) & (nbr < 0.1)
        
    Note:
        NBR complements other indices in the urban detection workflow
        by providing additional spectral discrimination capability.
    """
    import numpy as np
    
    # Input validation
    if nir.shape != swir2.shape:
        raise ValueError(f"NIR and SWIR2 arrays must have same shape: {nir.shape} vs {swir2.shape}")
    
    # Calculate denominator with epsilon to prevent division by zero
    denominator = nir + swir2 + epsilon
    
    # Calculate NBR using the standard formula
    nbr = (nir - swir2) / denominator
    
    return nbr


def calculate_brightness_temperature(
    thermal_band: ArrayLike,
    metadata: dict[str, Any]
) -> ArrayLike:
    """Calculate brightness temperature from thermal infrared band.
    
    Brightness temperature is derived from thermal infrared radiance using
    Planck's law and sensor-specific calibration constants. This measurement
    is valuable for urban heat island analysis and can help validate urban
    area detection by identifying areas with elevated temperatures.
    
    The conversion follows the standard Landsat thermal calibration process:
    1. Convert DN to Top-of-Atmosphere radiance
    2. Apply Planck's function to get brightness temperature
    3. Convert from Kelvin to Celsius
    
    Args:
        thermal_band: Thermal infrared band digital numbers (Band 10 for Landsat 8/9).
            Should be raw DN values, not already converted to radiance.
        metadata: Metadata dictionary containing calibration constants:
            Required keys:
            - 'thermal_conversion_constant_1' or 'K1': Thermal constant K1
            - 'thermal_conversion_constant_2' or 'K2': Thermal constant K2
            - 'thermal_scale_factor' or 'mult': Multiplicative scale factor
            - 'thermal_add_offset' or 'add': Additive offset
            
    Returns:
        Brightness temperature array in degrees Celsius.
        Typical urban values: 15-45°C
        Typical rural values: 10-35°C
        
    Raises:
        ValueError: If required metadata keys are missing.
        KeyError: If thermal calibration constants are not found.
        
    Example:
        >>> # Calculate brightness temperature
        >>> metadata = {
        >>>     'K1': 774.8853,
        >>>     'K2': 1321.0789,
        >>>     'mult': 0.0003342,
        >>>     'add': 0.1
        >>> }
        >>> temp_celsius = calculate_brightness_temperature(thermal_dn, metadata)
        >>> # Identify urban heat islands
        >>> hot_areas = temp_celsius > 30.0
        
    Note:
        Urban areas typically exhibit higher brightness temperatures
        due to the urban heat island effect, making this useful for
        validating urban detection results.
    """
    import numpy as np
    
    # Extract calibration constants from metadata with fallback keys
    # Landsat metadata may use different key names depending on processing level
    try:
        k1 = metadata.get('thermal_conversion_constant_1', metadata.get('K1'))
        k2 = metadata.get('thermal_conversion_constant_2', metadata.get('K2'))
        mult = metadata.get('thermal_scale_factor', metadata.get('mult'))
        add = metadata.get('thermal_add_offset', metadata.get('add'))
        
        # Validate all required constants are present
        if any(x is None for x in [k1, k2, mult, add]):
            missing = []
            if k1 is None: missing.append('K1/thermal_conversion_constant_1')
            if k2 is None: missing.append('K2/thermal_conversion_constant_2')
            if mult is None: missing.append('mult/thermal_scale_factor')
            if add is None: missing.append('add/thermal_add_offset')
            raise KeyError(f"Missing thermal calibration constants: {missing}")
            
    except KeyError as e:
        raise ValueError(
            f"Required thermal calibration constants not found in metadata. "
            f"Need: K1, K2, mult, add. Available keys: {list(metadata.keys())}"
        ) from e
    
    # Step 1: Convert DN to Top-of-Atmosphere radiance
    # Radiance = DN * multiplicative_scale + additive_offset
    # This is the standard Landsat Collection 2 Level-1 conversion
    radiance = thermal_band * mult + add
    
    # Step 2: Convert radiance to brightness temperature using Planck's law
    # Rearranged Planck equation: T = K2 / ln((K1 / L) + 1)
    # Where L is spectral radiance, K1 and K2 are calibration constants
    # Add small epsilon to prevent log(0) or division by 0 in no-data areas
    epsilon = 1e-10
    radiance_safe = np.maximum(radiance, epsilon)
    
    # Calculate brightness temperature in Kelvin
    # K1 and K2 are pre-computed constants specific to the sensor
    temp_kelvin = k2 / np.log((k1 / radiance_safe) + 1.0)
    
    # Step 3: Convert from Kelvin to Celsius for practical interpretation
    temp_celsius = temp_kelvin - 273.15
    
    return temp_celsius


def combine_indices_for_classification(
    ndvi: ArrayLike,
    ndwi: ArrayLike,
    ndui: ArrayLike,
    weights: dict[str, float] | None = None
) -> ArrayLike:
    """Combine multiple spectral indices for enhanced land cover classification.
    
    This function creates a weighted composite index that leverages the
    complementary information from vegetation (NDVI), water (NDWI), and
    urban (NDUI) indices. The combined index provides better discrimination
    between land cover types than any single index alone.
    
    The default weighting emphasizes urban detection while maintaining
    sensitivity to vegetation and water features:
    - NDUI: 0.5 (primary urban signal)
    - NDVI: 0.3 (vegetation suppression)
    - NDWI: 0.2 (water discrimination)
    
    Args:
        ndvi: NDVI array from calculate_ndvi(), range [-1, 1].
        ndwi: NDWI array from calculate_ndwi(), range [-1, 1].
        ndui: NDUI array from calculate_ndui(), range [0, 1].
        weights: Optional dictionary specifying weights for each index.
            Default: {'ndui': 0.5, 'ndvi': 0.3, 'ndwi': 0.2}
            Weights should sum to 1.0 for interpretable results.
            
    Returns:
        Combined classification index with enhanced land cover discrimination.
        Values are normalized to approximately [0, 1] range where:
        - Values near 1: Strong urban signal
        - Values near 0: Non-urban (vegetation, water, bare soil)
        
    Raises:
        ValueError: If input arrays have incompatible shapes or weights don't sum to 1.
        
    Example:
        >>> # Create classification index with default weights
        >>> combined = combine_indices_for_classification(ndvi, ndwi, ndui)
        >>> # Create classification index with custom weights
        >>> custom_weights = {'ndui': 0.6, 'ndvi': 0.25, 'ndwi': 0.15}
        >>> combined = combine_indices_for_classification(
        >>>     ndvi, ndwi, ndui, weights=custom_weights
        >>> )
        >>> # Apply threshold for final classification
        >>> urban_mask = combined > 0.4
        
    Note:
        The combined index often provides more robust results than
        individual indices, especially in areas with mixed land cover.
    """
    import numpy as np
    
    # Input validation
    shapes = [ndvi.shape, ndwi.shape, ndui.shape]
    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError(f"All input arrays must have same shape: {shapes}")
    
    # Set default weights if not provided
    # Default weights prioritize urban detection while maintaining sensitivity
    # to vegetation and water features for comprehensive classification
    if weights is None:
        weights = {'ndui': 0.5, 'ndvi': 0.3, 'ndwi': 0.2}
    
    # Validate weights contain required indices
    required_keys = {'ndui', 'ndvi', 'ndwi'}
    if not set(weights.keys()) == required_keys:
        raise ValueError(
            f"Weights must contain exactly these keys: {required_keys}. "
            f"Got: {set(weights.keys())}"
        )
    
    # Ensure weights sum to 1.0 for interpretable results
    # Small tolerance allows for floating-point rounding errors
    weight_sum = sum(weights.values())
    if not np.isclose(weight_sum, 1.0, rtol=1e-3):
        raise ValueError(
            f"Weights must sum to 1.0 for normalized output, got {weight_sum:.4f}. "
            f"Current weights: {weights}"
        )
    
    # Normalize indices to [0, 1] range for consistent weighting
    # This ensures each index contributes proportionally to the final result
    
    # NDVI: [-1, 1] -> [0, 1] using linear transformation
    # 0 represents barren/urban, 1 represents dense vegetation
    ndvi_norm = (ndvi + 1.0) / 2.0
    
    # NDWI: [-1, 1] -> [0, 1] using linear transformation  
    # 0 represents dry surfaces, 1 represents water bodies
    ndwi_norm = (ndwi + 1.0) / 2.0
    
    # NDUI: [0, 1] -> [0, 1] (already normalized)
    # 0 represents non-urban, 1 represents dense urban
    ndui_norm = ndui
    
    # Calculate weighted combination of normalized indices
    # The result maintains [0, 1] range due to weight constraint
    combined = (
        weights['ndvi'] * ndvi_norm +
        weights['ndwi'] * ndwi_norm +
        weights['ndui'] * ndui_norm
    )
    
    return combined


def calculate_index_statistics(
    index_array: ArrayLike,
    mask: ArrayLike | None = None
) -> dict[str, float]:
    """Calculate comprehensive statistics for a spectral index array.
    
    This function computes essential statistical measures that help assess
    data quality, distribution characteristics, and provide insights for
    threshold selection in classification workflows.
    
    The statistics are calculated only on valid (non-NaN, non-masked) pixels
    to ensure accurate representation of the actual data distribution.
    
    Args:
        index_array: Input spectral index array (e.g., NDVI, NDWI, NDUI).
            Can contain NaN values which will be excluded from calculations.
        mask: Optional boolean mask array where True indicates valid pixels
            to include in statistics. If None, all non-NaN pixels are used.
            Must have same shape as index_array if provided.
            
    Returns:
        Dictionary containing statistical measures:
        - 'count': Number of valid pixels used in calculations
        - 'min': Minimum value
        - 'max': Maximum value
        - 'mean': Arithmetic mean
        - 'median': 50th percentile
        - 'std': Standard deviation
        - 'var': Variance
        - 'p25': 25th percentile
        - 'p75': 75th percentile
        - 'p95': 95th percentile
        - 'p99': 99th percentile
        - 'range': Max - Min
        - 'iqr': Interquartile range (P75 - P25)
        - 'cv': Coefficient of variation (std/mean) if mean != 0
        
    Raises:
        ValueError: If mask shape doesn't match index_array shape.
        RuntimeError: If no valid pixels are found for calculation.
        
    Example:
        >>> # Calculate statistics for NDUI
        >>> stats = calculate_index_statistics(ndui_array)
        >>> print(f"Mean NDUI: {stats['mean']:.3f}")
        >>> print(f"Urban threshold (P95): {stats['p95']:.3f}")
        >>> 
        >>> # Calculate statistics with quality mask
        >>> stats = calculate_index_statistics(ndvi_array, mask=quality_mask)
        >>> if stats['cv'] > 0.5:
        >>>     print("High variability detected in NDVI")
        
    Note:
        Percentile values (especially P95, P99) are useful for
        determining classification thresholds in urban detection workflows.
    """
    import numpy as np
    
    # Input validation
    if mask is not None and mask.shape != index_array.shape:
        raise ValueError(f"Mask shape {mask.shape} must match index array shape {index_array.shape}")
    
    # Create combined mask for valid pixels
    # Exclude NaN values which may represent no-data or masked regions
    valid_mask = ~np.isnan(index_array)
    
    # Apply additional user-provided mask if available
    if mask is not None:
        valid_mask = valid_mask & mask
    
    # Extract valid data for statistics calculation
    valid_data = index_array[valid_mask]
    
    # Check if we have any valid data
    if len(valid_data) == 0:
        raise RuntimeError(
            "No valid pixels found for statistics calculation. "
            "Check input data and mask for excessive NaN values or overly restrictive masking."
        )
    
    # Calculate basic statistics
    # Convert to float to ensure JSON serialization compatibility
    stats = {
        'count': len(valid_data),
        'min': float(np.min(valid_data)),
        'max': float(np.max(valid_data)),
        'mean': float(np.mean(valid_data)),
        'median': float(np.median(valid_data)),
        'std': float(np.std(valid_data, ddof=0)),  # Population std
        'var': float(np.var(valid_data, ddof=0))    # Population variance
    }
    
    # Calculate percentiles
    percentiles = [25, 75, 95, 99]
    p_values = np.percentile(valid_data, percentiles)
    
    stats.update({
        'p25': float(p_values[0]),
        'p75': float(p_values[1]),
        'p95': float(p_values[2]),
        'p99': float(p_values[3])
    })
    
    # Calculate derived statistics
    stats['range'] = stats['max'] - stats['min']
    stats['iqr'] = stats['p75'] - stats['p25']
    
    # Coefficient of variation (CV) - normalized measure of dispersion
    # CV = std / |mean|, useful for comparing variability across different scales
    if stats['mean'] != 0:
        stats['cv'] = stats['std'] / abs(stats['mean'])
    else:
        # Handle edge case where mean is zero
        stats['cv'] = np.inf if stats['std'] > 0 else 0.0
    
    return stats