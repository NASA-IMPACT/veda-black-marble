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



__all__ = [
    # Coordinate functions (coordinates.py)
    "calculate_ground_distance",
    "calculate_pixel_area", 
    "calculate_transform",
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