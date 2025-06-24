"""Data preparation module for Black Marble pipeline.

This module handles data preprocessing tasks including:
- Quality assessment and masking
- Radiometric corrections and scaling
- Spatial reprojection and alignment
- Coordinate system transformations
- File I/O operations
"""

from .coordinates import (
    calculate_ground_distance,
    calculate_pixel_area,
    calculate_transform,
    calculate_utm_zone as calculate_utm_zone_coords,  # CRITICAL FIX: Rename to avoid collision
    create_crs_transformer,
    get_crs_from_epsg,
    reproject_geometry,
    transform_bounds,
    validate_crs_compatibility,
)

from .io import (
    create_memory_efficient_reader,
    create_virtual_raster,
    detect_format,
    load_with_gdal,
    load_with_rasterio,
    read_metadata,
    setup_output_profile,
    validate_raster_stack,
)

from .quality import (
    combine_masks,
    compute_mask_statistics,
    compute_quality_flags,
    create_aerosol_mask,
    create_cloud_mask,
    create_quality_mask,
    create_saturation_mask,
    create_urban_mask,
    create_viirs_quality_mask,
    create_water_mask,
    dilate_mask,
)

from .scaling import (
    apply_lunar_correction,
    apply_viirs_floor_ceiling,
    calculate_ntli,
    detect_if_already_scaled,
    normalize_to_range,
    scale_landsat_bands,
    scale_to_byte,
)

from .spatial import (
    align_rasters,
    align_to_common_grid,
    bounds_to_utm,
    calculate_utm_zone as calculate_utm_zone_spatial,  # CRITICAL FIX: Rename to avoid collision
    convert_to_wgs84,
    crop_to_bounds,
    mosaic_tiles,
    reproject_image,
    resample_image,
    stack_arrays_with_padding,
)

from .diagnostic import (
    save_diagnostic,
)

from .landsat import (
    process_landsat_date,
)

from .landsat_qa import (
    QALevel,
    qa_mask,
    composite_masked,
    create_landsat_cloud_mask,
    decode_qa_pixel,
    print_qa_pixel_summary,
)

# CRITICAL FIX: Unified calculate_utm_zone function to resolve API inconsistency
#
# The original implementation had two conflicting versions:
# 1. coordinates.py: calculate_utm_zone(lon, lat) -> int
# 2. spatial.py: calculate_utm_zone(lat, lon) -> tuple[int, str]
#
# These had DIFFERENT parameter order and return types, which could cause:
# - Silent bugs from swapped lat/lon parameters
# - Type errors from expecting int vs tuple
# - Incorrect hemisphere handling (negative int vs 'N'/'S' string)
#
# Solution: Provide a unified API that:
# - Uses consistent (lon, lat) order matching geographic convention
# - Returns comprehensive information in a tuple
# - Maintains backward compatibility through the renamed imports
def calculate_utm_zone(
    lon: float,
    lat: float,
    return_hemisphere: bool = True
) -> int | tuple[int, str]:
    """Calculate appropriate UTM zone for given coordinates.
    
    This is the unified API that resolves the collision between
    coordinates.py and spatial.py versions.
    
    Args:
        lon: Longitude in degrees [-180, 180]
        lat: Latitude in degrees [-90, 90]
        return_hemisphere: If True, returns (zone, hemisphere).
            If False, returns zone number with negative for south.
            
    Returns:
        If return_hemisphere=True: Tuple of (zone_number, hemisphere)
            where zone_number is 1-60 and hemisphere is 'N' or 'S'
        If return_hemisphere=False: Zone number (1-60), negative if southern
        
    Example:
        >>> # Modern usage with hemisphere
        >>> zone, hemi = calculate_utm_zone(-74.0, 40.7)
        >>> print(f"UTM Zone {zone}{hemi}")  # "UTM Zone 18N"
        >>> 
        >>> # Legacy compatibility
        >>> zone_int = calculate_utm_zone(-74.0, -40.7, return_hemisphere=False)
        >>> print(zone_int)  # -18 (negative for south)
    """
    # Use the coordinates.py implementation which handles special cases
    zone_with_hemisphere = calculate_utm_zone_coords(lon, lat)
    
    if return_hemisphere:
        # Convert to (zone_number, hemisphere) format
        zone_num = abs(zone_with_hemisphere)
        hemisphere = 'S' if zone_with_hemisphere < 0 else 'N'
        return zone_num, hemisphere
    else:
        # Return legacy format (negative for southern hemisphere)
        return zone_with_hemisphere


__all__ = [
    # Coordinate functions (coordinates.py)
    "calculate_ground_distance",
    "calculate_pixel_area", 
    "calculate_transform",
    "calculate_utm_zone",  # CRITICAL FIX: Unified API function
    "calculate_utm_zone_coords",  # Keep for backward compatibility
    "create_crs_transformer",
    "get_crs_from_epsg",
    "reproject_geometry",
    "transform_bounds",
    "validate_crs_compatibility",
    # I/O functions
    "create_memory_efficient_reader",
    "create_virtual_raster",
    "detect_format",
    "load_with_gdal",
    "load_with_rasterio",
    "read_metadata",
    "setup_output_profile",
    "validate_raster_stack",
    # Quality functions
    "combine_masks",
    "compute_mask_statistics",
    "compute_quality_flags",
    "create_aerosol_mask",
    "create_cloud_mask",
    "create_quality_mask",
    "create_saturation_mask",
    "create_urban_mask",
    "create_viirs_quality_mask",
    "create_water_mask",
    "dilate_mask",
    # Scaling functions
    "apply_lunar_correction",
    "apply_viirs_floor_ceiling",
    "calculate_ntli",
    "detect_if_already_scaled",
    "normalize_to_range",
    "scale_landsat_bands",
    "scale_to_byte",
    # Spatial functions (spatial.py)
    "align_rasters",
    "align_to_common_grid",
    "bounds_to_utm",
    "calculate_utm_zone_spatial",  # Keep for backward compatibility
    "convert_to_wgs84",
    "crop_to_bounds",
    "mosaic_tiles",
    "reproject_image",
    "resample_image",
    "stack_arrays_with_padding",
    # Diagnostic functions
    "save_diagnostic",
    # Landsat functions
    "process_landsat_date",
    # Landsat QA functions
    "QALevel",
    "qa_mask",
    "composite_masked",
    "create_landsat_cloud_mask",
    "decode_qa_pixel",
    "print_qa_pixel_summary",
]