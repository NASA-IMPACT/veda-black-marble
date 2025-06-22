"""
Rasterio-based I/O utilities - drop-in replacements for GDAL functions.
"""

import os
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS


def get_geo(filepath: str, band: int) -> tuple[np.ndarray, tuple[float, ...], str]:
    """
    Reads a specific band from a raster file using rasterio.

    Args:
        filepath: Path to the raster file
        band: Band number (1-based index)

    Returns:
        Tuple of (data array, geotransform tuple, projection WKT)
    """
    with rasterio.open(filepath) as src:
        # Read the specific band
        array = src.read(band)

        # Handle NoData
        if src.nodata is not None:
            # Check if this is likely a QA band
            is_qa_band = any(
                qa in os.path.basename(filepath).upper()
                for qa in ["QA", "PIXEL", "RADSAT", "AEROSOL"]
            )

            if is_qa_band and src.nodata == 0:
                # QA band logic - check if zeros are actually data
                zero_count = np.sum(array == 0)
                if zero_count > 0.9 * array.size:
                    array = array.astype(np.float32)
                    array = np.where(array == src.nodata, np.nan, array)
            else:
                # Regular band - replace nodata with NaN
                array = array.astype(np.float32)
                array = np.where(array == src.nodata, np.nan, array)

        # Get geotransform as GDAL-style tuple
        geotransform = src.transform.to_gdal()

        # Get projection as WKT
        if src.crs:
            projection = src.crs.to_wkt()
        else:
            projection = ""

    return array, geotransform, projection


def save_image(
    array: np.ndarray,
    geotransform: tuple[float, ...],
    projection: str,
    gdal_type: int,  # We'll map this to rasterio dtype
    out_path: str,
) -> None:
    """
    Saves a NumPy array as a single-band GeoTIFF using rasterio.
    """
    # Map GDAL types to rasterio
    from osgeo import gdal

    dtype_map = {
        gdal.GDT_Byte: rasterio.uint8,
        gdal.GDT_UInt16: rasterio.uint16,
        gdal.GDT_Int16: rasterio.int16,
        gdal.GDT_UInt32: rasterio.uint32,
        gdal.GDT_Int32: rasterio.int32,
        gdal.GDT_Float32: rasterio.float32,
        gdal.GDT_Float64: rasterio.float64,
    }

    dtype = dtype_map.get(gdal_type, rasterio.float32)

    # Convert array to appropriate type
    if dtype in [rasterio.uint8, rasterio.uint16, rasterio.uint32]:
        nodata = 0
        array_copy = array.astype(dtype)
    else:
        nodata = -9999.0 if np.issubdtype(dtype, np.floating) else -9999
        array_copy = array.copy()
        if np.issubdtype(array.dtype, np.floating):
            array_copy[np.isnan(array)] = nodata

    rows, cols = array.shape
    transform = Affine.from_gdal(*geotransform)

    # Handle CRS
    if projection:
        if projection.upper().startswith("EPSG:"):
            crs = CRS.from_string(projection)
        else:
            crs = CRS.from_wkt(projection)
    else:
        crs = None

    profile = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array_copy, 1)


def save_geotiff_rgb(
    out_path: str,
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    projection: str,
    geotransform: tuple[float, ...],
) -> None:
    """
    Saves RGB arrays as a 3-band GeoTIFF using rasterio.
    """
    rows, cols = r.shape
    transform = Affine.from_gdal(*geotransform)

    # Handle CRS
    if projection:
        if projection.upper().startswith("EPSG:"):
            crs = CRS.from_string(projection)
        else:
            crs = CRS.from_wkt(projection)
    else:
        crs = None

    profile = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": 3,
        "dtype": rasterio.uint8,
        "crs": crs,
        "transform": transform,
        "nodata": 0,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "photometric": "RGB",
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(r, 1)
        dst.write(g, 2)
        dst.write(b, 3)


def save_geotiff_single(
    out_path: str, array: np.ndarray, projection: str, geotransform: tuple[float, ...]
) -> None:
    """
    Saves a NumPy array as a single-band GeoTIFF file using rasterio.
    """
    rows, cols = array.shape
    transform = Affine.from_gdal(*geotransform)

    # Handle CRS
    if projection:
        if projection.upper().startswith("EPSG:"):
            crs = CRS.from_string(projection)
        else:
            crs = CRS.from_wkt(projection)
    else:
        raise ValueError(
            f"Cannot save GeoTIFF without projection information to {out_path}"
        )

    # Determine data type and handle NoData
    if array.dtype in [np.float32, np.float64]:
        dtype = rasterio.float32
        nodata = -9999.0
        array_copy = array.copy()
        array_copy[np.isnan(array)] = nodata
    else:
        dtype = rasterio.uint16
        nodata = 0
        array_copy = array.astype(np.uint16)

    profile = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array_copy, 1)


def scale_image(
    input_array: np.ndarray, min_value: int | float, max_value: int | float
) -> np.ndarray:
    """
    Clamps and scales array to uint8 range [0, 255].
    Note: This is a pure numpy operation, no rasterio needed.
    """
    # Handle NaN values first
    if np.any(np.isnan(input_array)):
        print(
            f"WARNING scale_image: Input contains {np.sum(np.isnan(input_array))} NaN values"
        )
        arr = np.where(input_array > max_value, max_value, input_array)
        arr = np.where(arr < min_value, min_value, arr)
        arr_min = np.nanmin(arr)
        arr_max = np.nanmax(arr)
    else:
        arr = np.where(input_array > max_value, max_value, input_array)
        arr = np.where(arr < min_value, min_value, arr)
        arr_min = arr.min()
        arr_max = arr.max()

    # Handle division by zero
    if arr_max == arr_min or np.isnan(arr_min) or np.isnan(arr_max):
        print(
            f"WARNING scale_image: arr_min={arr_min}, arr_max={arr_max}, returning zeros"
        )
        return np.zeros_like(arr, dtype="uint8")

    # Scale to 0-255
    scaled = (arr - arr_min) * (255.0 / (arr_max - arr_min))
    scaled = np.where(np.isnan(scaled), 0, scaled).astype("uint8")
    return scaled
