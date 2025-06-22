"""Essential Black Marble algorithms ported from bm.py.

This module contains the core algorithms for processing Landsat and VIIRS data
to create Black Marble nighttime light products. All functions are implemented
in a functional style with clear type signatures and comprehensive docstrings.
"""

import numpy as np
from typing import Optional, Union, Tuple, List, Any
import numpy.typing as npt

# Type aliases for clarity
ArrayLike = npt.NDArray[Any]  # More flexible type to handle various dtypes
MaskArray = npt.NDArray[np.bool_]


def calculate_ndvi(
    nir: ArrayLike, 
    red: ArrayLike, 
    epsilon: float = 0.000001
) -> ArrayLike:
    """Calculate Normalized Difference Vegetation Index (NDVI).
    
    NDVI is used to identify vegetation areas. Higher values indicate
    more vegetation, which helps distinguish urban from vegetated areas.
    
    Args:
        nir: Near-infrared band (Band 5 for Landsat 8/9)
        red: Red band (Band 4 for Landsat 8/9)
        epsilon: Small value to prevent division by zero
        
    Returns:
        NDVI array with values clamped to [-1, 1]
        
    Notes:
        NDVI = (NIR - Red) / (NIR + Red)
        Values range from -1 to 1:
        - Water: negative values
        - Bare soil: 0 to 0.2
        - Vegetation: 0.2 to 1
    """
    # Calculate NDVI with epsilon to prevent division by zero
    ndvi = (nir - red) / (nir + red + epsilon)
    
    # Clamp values to valid range
    ndvi = np.where(ndvi > 1, 1.0, ndvi)
    ndvi = np.where(ndvi < -1, -1.0, ndvi)
    
    return ndvi


def calculate_ndwi(
    green: ArrayLike, 
    nir: ArrayLike, 
    epsilon: float = 0.000001
) -> ArrayLike:
    """Calculate Normalized Difference Water Index (NDWI).
    
    NDWI is used to identify water bodies. Positive values typically
    indicate the presence of water features.
    
    Args:
        green: Green band (Band 3 for Landsat 8/9)
        nir: Near-infrared band (Band 5 for Landsat 8/9)
        epsilon: Small value to prevent division by zero
        
    Returns:
        NDWI array with values clamped to [-1, 1]
        
    Notes:
        NDWI = (Green - NIR) / (Green + NIR)
        Values range from -1 to 1:
        - Water: positive values (typically > 0)
        - Land: negative values
    """
    # Calculate NDWI with epsilon to prevent division by zero
    ndwi = (green - nir) / (green + nir + epsilon)
    
    # Clamp values to valid range
    ndwi = np.where(ndwi > 1, 1.0, ndwi)
    ndwi = np.where(ndwi < -1, -1.0, ndwi)
    
    return ndwi


def calculate_ndui(
    ntl: ArrayLike,
    ndvi: ArrayLike,
    ndwi: Optional[ArrayLike] = None,
    ntl_floor: float = 0.1,
    ntl_ceiling: float = 10.0,
    ndui_floor: float = 0.02,
    epsilon: float = 0.00001
) -> ArrayLike:
    """Calculate Normalized Difference Urban Index (NDUI).
    
    NDUI combines nighttime lights with vegetation index to identify
    urban areas. It assumes urban areas have high nighttime lights
    and low vegetation.
    
    Args:
        ntl: Nighttime light radiance values
        ndvi: NDVI values from calculate_ndvi()
        ndwi: Optional NDWI values for water masking
        ntl_floor: Minimum NTL threshold
        ntl_ceiling: Maximum NTL threshold for normalization
        ndui_floor: Minimum NDUI threshold
        epsilon: Small value to prevent division by zero
        
    Returns:
        NDUI array with values in [0, 1], where higher values indicate
        more urban characteristics
        
    Notes:
        NDUI = (NTL - NDVI) / (NTL + NDVI)
        The index is then transformed to [0, 1] range and water-masked
    """
    # Preprocess NTL values
    ntl_processed = np.where(ntl > ntl_ceiling, ntl_ceiling, ntl)
    ntl_processed = np.where(ntl_processed < ntl_floor, 0.0, ntl_processed)
    ntl_processed = ntl_processed / ntl_ceiling
    
    # Calculate NDUI
    ndui = (ntl_processed - ndvi) / (ntl_processed + ndvi + epsilon)
    
    # Clamp to valid range
    ndui = np.where(ndui > 1.0, 1.0, ndui)
    ndui = np.where(ndui < -1.0, -1.0, ndui)
    
    # Apply water mask if provided
    if ndwi is not None:
        ndui = np.where(ndwi >= 0.0, -1.0, ndui)
    
    # Transform to [0, 1] range
    ndui = np.abs(ndui + 1.0) / 2.0
    
    # Apply floor threshold
    ndui = np.where(ndui < ndui_floor, 0.0, ndui)
    
    return ndui


def calculate_ntli(
    vnp_radiance: ArrayLike,
    lunar_irradiance: ArrayLike,
    quality_flag: ArrayLike,
    valid_flags: Optional[List[int]] = None
) -> ArrayLike:
    """Calculate Nighttime Light Index (NTLI) with lunar correction.
    
    Adjusts VIIRS nighttime light radiance by removing lunar illumination
    and applying quality flags.
    
    Args:
        vnp_radiance: VIIRS DNB radiance values
        lunar_irradiance: Lunar irradiance values for correction
        quality_flag: Quality flag array from VIIRS
        valid_flags: List of valid quality flag values (default: [0, 2])
        
    Returns:
        Lunar-corrected nighttime light radiance
        
    Notes:
        - Values > 65530 are considered fill values and set to 0
        - Only pixels with valid quality flags are processed
        - Result is lunar_corrected = vnp - lunar_irradiance
    """
    if valid_flags is None:
        valid_flags = [0, 2]
    
    # Handle fill values
    vnp_clean = np.where(vnp_radiance > 65530, 0, vnp_radiance)
    lunar_clean = np.where(lunar_irradiance > 65530, 0, lunar_irradiance)
    
    # Create quality mask
    quality_mask = np.isin(quality_flag, valid_flags).astype(float)
    
    # Apply lunar correction with quality mask
    ntli = vnp_clean * quality_mask - lunar_clean
    
    # Ensure non-negative values
    ntli = np.where(ntli < 0, 0, ntli)
    
    return ntli


def scale_landsat_bands(
    band_data: ArrayLike,
    scale_factor: float = 0.0000275,
    add_offset: float = -0.2,
    valid_range: Optional[Tuple[float, float]] = None
) -> ArrayLike:
    """Scale Landsat Level-2 surface reflectance bands to physical values.
    
    Converts digital numbers to surface reflectance using the scale
    factor and offset provided in Landsat metadata.
    
    Args:
        band_data: Raw band digital numbers
        scale_factor: Multiplicative scale factor (default for Landsat 8/9)
        add_offset: Additive offset (default for Landsat 8/9)
        valid_range: Optional tuple of (min, max) valid DN values
        
    Returns:
        Scaled surface reflectance values
        
    Notes:
        Default valid_range for Landsat 8/9 Level-2 is (7273, 43636)
        which corresponds to surface reflectance of ~0 to ~1
    """
    if valid_range is None:
        valid_range = (7273, 43636)
    
    # Mask invalid values
    band_masked = np.where(band_data < valid_range[0], 0, band_data)
    band_masked = np.where(band_masked > valid_range[1], 0, band_masked)
    
    # Apply scaling
    scaled = band_masked * scale_factor + add_offset
    
    return scaled


def apply_lunar_correction(
    vnp_data: ArrayLike,
    lunar_irradiance: ArrayLike,
    moon_illumination_fraction: Optional[ArrayLike] = None
) -> ArrayLike:
    """Apply lunar BRDF correction to VIIRS nighttime lights.
    
    Removes the effect of lunar illumination from VIIRS DNB observations
    to isolate artificial light sources.
    
    Args:
        vnp_data: VIIRS Day/Night Band radiance
        lunar_irradiance: Calculated lunar irradiance at surface
        moon_illumination_fraction: Optional moon illumination fraction
            for advanced correction (not used in basic implementation)
        
    Returns:
        Lunar-corrected radiance values
        
    Notes:
        This implements the basic correction. Advanced corrections could
        include BRDF modeling based on moon angle and surface type.
    """
    # Basic correction: subtract lunar contribution
    corrected = vnp_data - lunar_irradiance
    
    # Ensure non-negative
    corrected = np.where(corrected < 0, 0, corrected)
    
    # Advanced correction could be implemented here using moon_illumination_fraction
    # and surface BRDF models
    
    return corrected


def create_urban_mask(
    ndvi: ArrayLike,
    ndwi: ArrayLike,
    ndui: ArrayLike,
    ndvi_threshold: float = 0.2,
    ndwi_threshold: float = 0.0,
    ndui_threshold: float = 0.3
) -> MaskArray:
    """Create urban area mask from spectral indices.
    
    Combines multiple indices to identify urban pixels with high confidence.
    Urban areas are characterized by low vegetation, no water, and high
    urban index values.
    
    Args:
        ndvi: NDVI array from calculate_ndvi()
        ndwi: NDWI array from calculate_ndwi()
        ndui: NDUI array from calculate_ndui()
        ndvi_threshold: Maximum NDVI for urban (default: 0.2)
        ndwi_threshold: Maximum NDWI for urban (default: 0.0)
        ndui_threshold: Minimum NDUI for urban (default: 0.3)
        
    Returns:
        Boolean mask where True indicates urban pixels
    """
    # Urban criteria:
    # - Low vegetation (NDVI < threshold)
    # - Not water (NDWI < threshold)
    # - High urban index (NDUI > threshold)
    urban_mask = (
        (ndvi < ndvi_threshold) & 
        (ndwi < ndwi_threshold) & 
        (ndui > ndui_threshold)
    )
    
    return urban_mask


def stack_arrays_with_padding(
    arrays: List[ArrayLike]
) -> ArrayLike:
    """Stack arrays of different shapes by padding to the maximum shape.
    
    This function handles arrays with different dimensions by zero-padding
    smaller arrays to match the largest dimensions in the collection.
    
    Args:
        arrays: List of 2D arrays with potentially different shapes
        
    Returns:
        3D array of shape (n_arrays, max_height, max_width) with smaller
        arrays zero-padded
        
    Notes:
        Padding is applied to the bottom and right edges. Arrays are
        assumed to be aligned at the top-left corner.
    """
    if not arrays:
        raise ValueError("Cannot stack empty list of arrays")
    
    # Find maximum dimensions
    max_shape = np.max([arr.shape for arr in arrays], axis=0)
    
    # Create output array
    n_arrays = len(arrays)
    stacked = np.zeros((n_arrays, *max_shape), dtype=np.float64)
    
    # Fill with data
    for idx, arr in enumerate(arrays):
        stacked[idx, :arr.shape[0], :arr.shape[1]] = arr
    
    return stacked


def align_to_common_grid(
    arrays: List[ArrayLike],
    transforms: List[Tuple[float, ...]],
    reference_idx: int = 0
) -> List[Tuple[ArrayLike, Tuple[float, ...]]]:
    """Align multiple arrays to a common spatial grid.
    
    This is a placeholder for the full implementation which would use
    rasterio or GDAL for proper georeferenced alignment. The actual
    implementation is handled by the spatial processing functions.
    
    Args:
        arrays: List of 2D arrays to align
        transforms: List of affine transform tuples for each array
        reference_idx: Index of array to use as reference grid
        
    Returns:
        List of tuples (aligned_array, transform) all sharing the same
        spatial grid as the reference
        
    Notes:
        In practice, this functionality is provided by functions like
        align_rasters() in the processing module which use proper
        geospatial libraries.
    """
    # This is a conceptual implementation
    # Real implementation would use rasterio.warp.reproject or similar
    
    if not arrays or not transforms:
        raise ValueError("Arrays and transforms cannot be empty")
    
    if len(arrays) != len(transforms):
        raise ValueError("Arrays and transforms must have same length")
    
    if reference_idx >= len(arrays):
        raise ValueError(f"Invalid reference index {reference_idx}")
    
    # For now, return input unchanged
    # Real implementation would reproject to common grid
    return list(zip(arrays, transforms))


def compute_quality_flags(
    qa_pixel: ArrayLike,
    qa_aerosol: Optional[ArrayLike] = None,
    qa_radsat: Optional[ArrayLike] = None,
    cloud_values: Optional[List[int]] = None,
    aerosol_values: Optional[List[int]] = None
) -> MaskArray:
    """Compute comprehensive quality flags from Landsat QA bands.
    
    Combines multiple QA bands to create a single quality mask indicating
    pixels suitable for analysis.
    
    Args:
        qa_pixel: QA_PIXEL band containing cloud and other flags
        qa_aerosol: Optional QA_AEROSOL band for aerosol quality
        qa_radsat: Optional QA_RADSAT band for saturation flags
        cloud_values: List of QA_PIXEL values indicating clouds
        aerosol_values: List of QA_AEROSOL values indicating poor quality
        
    Returns:
        Boolean mask where True indicates good quality pixels
        
    Notes:
        Default cloud values exclude most conditions except:
        - 21824: Clear land
        - 21952: Clear water
        
        Default aerosol values indicate high quality:
        - 192, 194, 196, 228: Various levels of valid aerosol
    """
    if cloud_values is None:
        # Keep only clear land (21824) and clear water (21952)
        cloud_values = (
            list(range(1, 21823)) + 
            list(range(21825, 21951)) + 
            list(range(21953, 65535))
        )
    
    if aerosol_values is None:
        # These values indicate valid aerosol conditions
        aerosol_values = [192, 194, 196, 228]
    
    # Create cloud mask (True = good pixel)
    cloud_mask = ~np.isin(qa_pixel, cloud_values)
    
    # Combine with aerosol mask if provided
    if qa_aerosol is not None:
        aerosol_mask = np.isin(qa_aerosol, aerosol_values)
        quality_mask = cloud_mask & aerosol_mask
    else:
        quality_mask = cloud_mask
    
    # Add saturation check if provided
    if qa_radsat is not None:
        # Check if any bands are saturated (bits 1-7)
        # Bit 0 is "band 1 data quality" which we ignore
        saturation_mask = qa_radsat == 0
        quality_mask = quality_mask & saturation_mask
    
    return quality_mask