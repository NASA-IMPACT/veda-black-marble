"""
Rasterio-based reprojection & subsetting utilities.
"""

import os
import tempfile
import numpy as np
import rasterio
import rasterio.windows
import rasterio.errors
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds, Window
from rasterio.transform import Affine
from rasterio.crs import CRS
from rasterio.merge import merge
from pathlib import Path


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
    """
    Crop and resample raster using rasterio.
    Replaces the multi-step GDAL translate/warp/translate approach.
    """
    with rasterio.open(input_raster) as src:
        # Calculate window for cropping
        window = from_bounds(minX, minY, maxX, maxY, src.transform)

        # Read the cropped data
        data = src.read(window=window)

        # Calculate new transform for the window
        transform = src.window_transform(window)

        # Calculate output dimensions based on scale
        out_width = int((maxX - minX) / scale)
        out_height = int((maxY - minY) / scale)

        # Calculate transform for target resolution
        dst_transform = Affine(scale, 0, minX, 0, -scale, maxY)

        # Create output array
        out_data = np.empty((src.count, out_height, out_width), dtype=src.dtypes[0])

        # Reproject
        reproject(
            source=data,
            destination=out_data,
            src_transform=transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=src.crs,
            resampling=Resampling.bilinear,
        )

        # Write output
        profile = src.profile.copy()
        profile.update(
            {
                "width": out_width,
                "height": out_height,
                "transform": dst_transform,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
            }
        )

        with rasterio.open(output_raster, "w", **profile) as dst:
            dst.write(out_data)


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
    """
    Crop and warp to EPSG:4326 at given scale using rasterio.

    This function:
    1. Crops the input raster to the specified bounds
    2. Reprojects to WGS84 (EPSG:4326)
    3. Resamples to the specified resolution

    Args:
        input_raster: Path to input raster
        output_raster: Path to output raster
        minX, maxY, maxX, minY: Bounds in source CRS
        scale: Target resolution in source CRS units (typically meters)
        temp_dir: Directory for temporary files
    """
    with rasterio.open(input_raster) as src:
        # Define target CRS
        dst_crs = CRS.from_epsg(4326)

        # Handle dateline crossing
        if minX > maxX:  # Dateline crossing
            # For now, just swap to make it valid
            # A proper implementation would need to handle wrapping
            minX, maxX = maxX, minX

        # First crop to the bounds in source CRS
        window = from_bounds(minX, minY, maxX, maxY, src.transform)

        # Read the windowed data
        window_data = src.read(window=window)
        window_transform = src.window_transform(window)

        # Get the bounds of the window
        window_bounds = rasterio.windows.bounds(window, src.transform)

        # Convert scale from meters to degrees (approximate)
        # At equator: 1 degree ≈ 111,000 meters
        # This is a rough approximation that varies by latitude
        if src.crs.to_epsg() in [32601, 32660, 32630]:  # UTM zones
            # For UTM, scale is in meters
            # Convert to approximate degrees
            scale_degrees = scale / 111000.0
        else:
            # Assume scale is already in appropriate units
            scale_degrees = scale

        # Calculate transform for reprojection with target resolution
        transform, width, height = calculate_default_transform(
            src.crs,
            dst_crs,
            int(window.width),
            int(window.height),
            left=window_bounds[0],
            bottom=window_bounds[1],
            right=window_bounds[2],
            top=window_bounds[3],
            resolution=scale_degrees,
        )

        # Create output array (width and height should not be None from calculate_default_transform)
        assert width is not None and height is not None, "Transform calculation failed"
        out_data = np.empty((src.count, int(height), int(width)), dtype=src.dtypes[0])

        # Reproject the windowed data
        reproject(
            source=window_data,
            destination=out_data,
            src_transform=window_transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            num_threads=4,
        )

        # Update metadata
        profile = src.profile.copy()
        profile.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
            }
        )

        # Write output
        with rasterio.open(output_raster, "w", **profile) as dst:
            dst.write(out_data)


def convert_to_wgs84(
    inpath: str,
    outpath: str,
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
    temp_dir: str,
) -> None:
    """
    Convert raster to WGS84 and crop to lon/lat bounds using rasterio.

    Args:
        inpath: Input raster path
        outpath: Output raster path
        lon1, lat1, lon2, lat2: Output bounds in WGS84
        temp_dir: Directory for temporary files
    """
    # First, reproject entire raster to WGS84
    temp_wgs84 = os.path.join(temp_dir, "temp_wgs84.tif")

    with rasterio.open(inpath) as src:
        # Define target CRS
        dst_crs = CRS.from_epsg(4326)

        # Calculate transform for full reprojection to WGS84
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        # Update metadata
        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
            }
        )

        # Create temporary reprojected file
        with rasterio.open(temp_wgs84, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                    num_threads=4,
                )

    # Now crop to the specified bounds
    with rasterio.open(temp_wgs84) as src:
        # Calculate window for the bounds
        window = from_bounds(lon1, lat1, lon2, lat2, src.transform)

        # Check if window overlaps with image at all
        image_window = Window(0, 0, src.width, src.height)  # type: ignore

        # Try to get intersection - if no overlap, use whatever portion is available
        try:
            # If windows don't overlap, intersection will raise WindowError
            window = window.intersection(image_window)
        except rasterio.errors.WindowError:
            # No overlap - try to salvage what we can
            # This mimics GDAL behavior of returning partial/empty results

            # Calculate the overlap region manually
            col_min = max(0, window.col_off)
            col_max = min(src.width, window.col_off + window.width)
            row_min = max(0, window.row_off)
            row_max = min(src.height, window.row_off + window.height)

            if col_max > col_min and row_max > row_min:
                # There's some overlap
                window = Window(col_min, row_min, col_max - col_min, row_max - row_min)  # type: ignore
            else:
                # No overlap at all - create a minimal window
                # This will result in a very small output, matching GDAL behavior
                window = Window(0, 0, 1, 1)  # type: ignore

        # Read the windowed data
        data = src.read(window=window)

        # Get the transform for the window
        transform = src.window_transform(window)

        # Update profile for cropped output
        profile = src.profile.copy()
        profile.update(
            {
                "width": int(window.width),
                "height": int(window.height),
                "transform": transform,
            }
        )

        # Write final output
        with rasterio.open(outpath, "w", **profile) as dst:
            dst.write(data)

    # Clean up temp file
    if os.path.exists(temp_wgs84):
        os.remove(temp_wgs84)


def crop_to_bb_utm(
    inpath: str,
    outpath: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    margin: float = 1000,
) -> None:
    """
    Crop raster to bounding box with margin in UTM coordinates.

    Args:
        inpath: Input raster path
        outpath: Output raster path
        x1, y1, x2, y2: Bounding box corners in UTM coordinates
        margin: Buffer margin in meters (default: 1000)
    """
    # Sort coordinates to get min/max
    minX, maxX = sorted([x1, x2])
    minY, maxY = sorted([y1, y2])

    # Add margin to bounds
    minX_with_margin = minX - margin
    maxX_with_margin = maxX + margin
    minY_with_margin = minY - margin
    maxY_with_margin = maxY + margin

    with rasterio.open(inpath) as src:
        # Calculate window for cropping
        window = from_bounds(
            minX_with_margin,
            minY_with_margin,
            maxX_with_margin,
            maxY_with_margin,
            src.transform,
        )

        # Adjust window to be within raster bounds
        window = window.intersection(Window(0, 0, src.width, src.height))  # type: ignore

        # Read the windowed data
        data = src.read(window=window)

        # Get the transform for the window
        transform = src.window_transform(window)

        # Update profile
        profile = src.profile.copy()
        profile.update(
            {
                "width": int(window.width),
                "height": int(window.height),
                "transform": transform,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
            }
        )

        # Write output
        with rasterio.open(outpath, "w", **profile) as dst:
            dst.write(data)
