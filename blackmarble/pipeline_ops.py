"""
Extracted pipeline operations that can be swapped between GDAL and rasterio.
"""

import os
from pathlib import Path
from osgeo import gdal


def mosaic_viirs_tiles_gdal(
    ntl_tiles: list[str],
    lunar_tiles: list[str],
    quality_tiles: list[str],
    temp_dir: str,
) -> tuple[str, str, str]:
    """
    Mosaic VIIRS tiles using GDAL.

    Returns paths to mosaiced NTL, lunar, and quality files.
    """
    print("Mosaicing VIIRS tiles using GDAL...")

    ntl_temp_tif = os.path.join(temp_dir, "ntl_raw_mosaic.tif")
    gdal.Warp(ntl_temp_tif, ntl_tiles, format="GTiff", resampleAlg="bilinear")

    lunar_temp_tif = os.path.join(temp_dir, "lunar_irradiance_mosaic.tif")
    gdal.Warp(lunar_temp_tif, lunar_tiles, format="GTiff", resampleAlg="bilinear")

    quality_temp_tif = os.path.join(temp_dir, "mandatory_quality_flag_mosaic.tif")
    gdal.Warp(quality_temp_tif, quality_tiles, format="GTiff", resampleAlg="nearest")

    return ntl_temp_tif, lunar_temp_tif, quality_temp_tif


def mosaic_viirs_tiles_rasterio(
    ntl_tiles: list[str],
    lunar_tiles: list[str],
    quality_tiles: list[str],
    temp_dir: str,
) -> tuple[str, str, str]:
    """
    Mosaic VIIRS tiles using rasterio.

    Returns paths to mosaiced NTL, lunar, and quality files.
    """
    from .mosaic_rasterio import mosaic_viirs_tiles

    print("Mosaicing VIIRS tiles using rasterio...")
    from typing import cast
    return mosaic_viirs_tiles(
        cast(list[str | Path], ntl_tiles),
        cast(list[str | Path], lunar_tiles), 
        cast(list[str | Path], quality_tiles),
        temp_dir
    )


def resample_to_resolution_gdal(
    input_path: str, output_path: str, resolution: tuple[float, float]
) -> bool:
    """
    Resample a raster to a specific resolution using GDAL.

    Returns True on success, False on failure.
    """
    warp_opts = gdal.WarpOptions(format="GTiff", xRes=resolution[0], yRes=resolution[1])
    result = gdal.Warp(output_path, input_path, options=warp_opts)
    return result is not None


def resample_to_resolution_rasterio(
    input_path: str, output_path: str, resolution: tuple[float, float]
) -> bool:
    """
    Resample a raster to a specific resolution using rasterio.

    Returns True on success, False on failure.
    """
    from .mosaic_rasterio import resample_to_resolution

    try:
        resample_to_resolution(input_path, output_path, resolution)
        return True
    except Exception as e:
        print(f"Rasterio resampling failed: {e}")
        return False


# Default implementations (GDAL)
mosaic_viirs_tiles = mosaic_viirs_tiles_gdal
resample_to_resolution = resample_to_resolution_gdal

# Try to load rasterio implementations
print("INFO: Attempting to load rasterio implementations for pipeline operations")
try:
    # Test imports
    from .mosaic_rasterio import mosaic_viirs_tiles as _test_mosaic
    from .mosaic_rasterio import resample_to_resolution as _test_resample

    # If imports succeed, use rasterio implementations
    mosaic_viirs_tiles = mosaic_viirs_tiles_rasterio
    resample_to_resolution = resample_to_resolution_rasterio

    print("INFO: Successfully loaded rasterio implementations for pipeline operations")
except ImportError as e:
    print(f"WARNING: Could not load rasterio implementations: {e}")
    print("INFO: Using GDAL implementations for pipeline operations")
