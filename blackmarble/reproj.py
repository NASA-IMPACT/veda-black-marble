"""
Reprojection & subsetting utilities:
  - subset_raster_normal: crop + resample via interim files
  - subset_raster: crop + warp directly to EPSG:4326
  - convert_to_wgs84: reproject to WGS84 and crop to lon/lat box
  - crop_to_bb_utm: crop a UTM-projected raster with margin
"""

# Imports
import os
from osgeo import gdal
from pyproj import Proj
from typing import Any
from .coords import wkt_to_json


def subset_raster_normal(
    input_raster: str,
    output_raster: str,
    minX: float,
    maxY: float,
    maxX: float,
    minY: float,
    scale: float,
    temp_dir: str,
) -> None:
    temp_file1 = os.path.join(temp_dir, "new1_subset.tif")
    if os.path.exists(temp_file1):
        os.remove(temp_file1)
    temp_file2 = os.path.join(temp_dir, "new2_subset.tif")
    if os.path.exists(temp_file2):
        os.remove(temp_file2)
    ds = gdal.Translate(
        temp_file1,
        input_raster,
        projWin=[minX, maxY, maxX, minY],
        resampleAlg="bilinear",
    )
    ds = None
    ds = gdal.Open(temp_file1)
    warp_opts = gdal.WarpOptions(xRes=scale, yRes=scale, resampleAlg="bilinear")
    ds = gdal.Warp(temp_file2, ds, options=warp_opts)
    ds = None
    ds = gdal.Open(temp_file2)
    ds = gdal.Translate(
        output_raster, ds, projWin=[minX, maxY, maxX, minY], resampleAlg="bilinear"
    )
    ds = None


def subset_raster(
    input_raster: str,
    output_raster: str,
    minX: float,
    maxY: float,
    maxX: float,
    minY: float,
    scale: float,
    temp_dir: str,
) -> None:
    """Crop and warp directly to EPSG:4326 at given scale, using temp_dir."""
    # Define intermediate file path within temp_dir
    temp_file1 = os.path.join(temp_dir, "new1_warp.tif")

    # Initial crop
    ds1 = gdal.Open(input_raster)
    ds2 = gdal.Translate(
        temp_file1, ds1, projWin=[minX, maxY, maxX, minY], resampleAlg="bilinear"
    )
    ds1 = None
    ds2 = None

    # Warp to target SRS and scale
    ds = gdal.Open(temp_file1)
    # Warp to EPSG:4326 with target resolution
    warp_opts = gdal.WarpOptions(
        xRes=scale,
        yRes=scale,
        resampleAlg="bilinear",
        warpMemoryLimit=4000,
        multithread=True,
        dstSRS="EPSG:4326",
    )
    ds = gdal.Warp(output_raster, ds, options=warp_opts)
    ds = None
    try:
        os.remove(temp_file1)
    except OSError as e:
        print(f"Warning: Could not remove temporary file: {e}")


def convert_to_wgs84(
    input_path: str,
    output_path: str,
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
    output_dir: str,
) -> None:
    """Reproject raster to WGS84 and crop to lon/lat bounds, using output_dir for temp."""
    # Open the input dataset
    ds = gdal.Open(input_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Failed to open input file: {input_path}")

    # Get the projection - it should be set by save_geotiff_single
    srs_wkt: str = ds.GetProjection()
    if not srs_wkt:
        print(f"Warning: No projection found in {input_path}")
        # Check if geotransform exists
        gt = ds.GetGeoTransform()
        print(f"Geotransform: {gt}")

    # Close the dataset before warping
    ds = None

    # Define intermediate file path within output_dir (or its temp subdir)
    temp_conv_path = os.path.join(
        output_dir, "conv.tiff"
    )  # Place in main output dir for simplicity
    if os.path.exists(temp_conv_path):
        os.remove(temp_conv_path)

    # Reproject to WGS84 using WarpOptions object
    # Let GDAL read the projection from the file itself
    warp_opts = gdal.WarpOptions(
        dstSRS="EPSG:4326",
        resampleAlg="bilinear",
        warpMemoryLimit=4000,
        multithread=True,
    )

    # Perform the warp
    ds_result = gdal.Warp(temp_conv_path, input_path, options=warp_opts)
    if ds_result is None:
        raise RuntimeError(f"Failed to reproject {input_path} to WGS84")
    ds_result = None

    # Crop to provided lon/lat box
    min_x, max_x = sorted([lon1, lon2])
    min_y, max_y = sorted([lat1, lat2])
    ds = gdal.Open(temp_conv_path)
    ds = gdal.Translate(
        output_path, ds, projWin=[min_x, max_y, max_x, min_y], resampleAlg="bilinear"
    )
    ds = None
    # Clean up intermediate file
    try:
        os.remove(temp_conv_path)
    except OSError as e:
        print(f"Warning: Could not remove temporary file: {e}")


def crop_to_bb_utm(
    input_path: str, output_path: str, x1: float, y1: float, x2: float, y2: float
) -> None:
    """Crop a UTM-projected raster to a bounding box plus margin (1000m)."""
    ds = gdal.Open(input_path)
    min_x: float
    max_x: float
    min_x, max_x = sorted([x1, x2])
    min_y: float
    max_y: float
    min_y, max_y = sorted([y1, y2])
    # add margin
    min_x -= 1000
    min_y -= 1000
    max_x += 1000
    max_y += 1000

    ds = gdal.Translate(
        output_path, ds, projWin=[min_x, max_y, max_x, min_y], resampleAlg="bilinear"
    )
    ds = None


# Use rasterio implementations
print("INFO: Switching to rasterio-based reprojection implementations")
try:
    from .reproj_rasterio import (
        subset_raster_normal as _rio_subset_raster_normal,
        subset_raster as _rio_subset_raster,
        convert_to_wgs84 as _rio_convert_to_wgs84,
        crop_to_bb_utm as _rio_crop_to_bb_utm,
    )

    # Replace GDAL implementations with rasterio ones
    # Note: Parameter names differ but are compatible at runtime
    convert_to_wgs84 = _rio_convert_to_wgs84  # type: ignore
    crop_to_bb_utm = _rio_crop_to_bb_utm  # type: ignore
    subset_raster_normal = _rio_subset_raster_normal
    subset_raster = _rio_subset_raster

    print("INFO: Successfully loaded rasterio reprojection implementations")
except ImportError as e:
    print(f"WARNING: Could not import rasterio reprojection implementations: {e}")
    print("INFO: Using GDAL implementations")
