"""Landsat-specific data preparation functions.

This module provides functions for processing Landsat data in the pipeline,
including date-by-date processing with QA masking applied before temporal compositing.
"""

import os
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import rasterio
import rasterio.windows
from osgeo import gdal
from pyproj import Transformer

from .landsat_qa import create_landsat_cloud_mask, print_qa_pixel_summary

# Type aliases
ArrayLike = npt.NDArray[Any]
BBox = tuple[float, float, float, float]


def _read_windowed(file_path: str, bbox: BBox | None, transform_bbox: bool = True) -> tuple[ArrayLike, dict[str, Any]]:
    """Read only the window that intersects bbox from a raster file.
    
    Args:
        file_path: Path to the raster file (can be a VRT)
        bbox: Bounding box in lon/lat (min_lon, min_lat, max_lon, max_lat)
        transform_bbox: Whether to transform bbox from EPSG:4326 to file CRS
        
    Returns:
        Tuple of (data array, profile dict with updated transform/dimensions)
    """
    with rasterio.open(file_path) as src:
        if bbox is None:
            # No bbox specified, read entire file
            data = src.read(1)
            return data, src.profile.copy()
        
        # Transform bbox to file CRS if needed
        bbox_in_src_crs = bbox
        if transform_bbox and src.crs and src.crs.to_epsg() != 4326:
            try:
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                minx, miny = transformer.transform(bbox[0], bbox[1])
                maxx, maxy = transformer.transform(bbox[2], bbox[3])
                bbox_in_src_crs = (minx, miny, maxx, maxy)
            except Exception as e:
                print(f"  WARNING: Could not transform bbox: {e}")
                bbox_in_src_crs = bbox
        
        # Get the window for this bbox
        window = rasterio.windows.from_bounds(
            bbox_in_src_crs[0], bbox_in_src_crs[1], 
            bbox_in_src_crs[2], bbox_in_src_crs[3],
            transform=src.transform
        )
        
        # Log window details
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"  Windowed read from {os.path.basename(file_path)}:")
        logger.debug(f"    File shape: {src.shape}")
        logger.debug(f"    File bounds: {src.bounds}")
        logger.debug(f"    Window BEFORE intersection: col_off={window.col_off:.1f}, row_off={window.row_off:.1f}, "
                    f"width={window.width:.1f}, height={window.height:.1f}")
        logger.debug(f"    Requested bbox (src CRS): {bbox_in_src_crs}")
        
        # Don't clip to raster bounds - use boundless read for consistent shapes
        # This ensures all dates have the same array dimensions
        logger.debug(f"    Using boundless read with original window")
        
        # Convert window to integer bounds for the read
        window_int = window.round_lengths().round_offsets()
        
        # Read the windowed data with boundless=True to get consistent shapes
        # Areas outside the raster bounds will be filled with NaN
        data = src.read(1, window=window_int, boundless=True, fill_value=np.nan)
        
        # Create profile for the windowed data using the original requested window
        profile = src.profile.copy()
        profile.update({
            'height': int(window_int.height),
            'width': int(window_int.width),
            'transform': rasterio.windows.transform(window_int, src.transform)
        })
        
        logger.debug(f"    Output shape: {data.shape}")
        logger.debug(f"    Non-zero pixels: {np.count_nonzero(data)} ({np.count_nonzero(data) / data.size * 100:.1f}%)")
        
        return data, profile


def process_landsat_date(
    bands_data: dict[str, tuple[list[str], list[str], list[str], list[str]]],
    bbox: BBox | None = None,
    apply_qa_mask: bool = True,
    dilate_pixels: int = 3,
    cloud_strategy: Literal["conservative", "moderate", "permissive"] = "moderate",
) -> dict[str, tuple[ArrayLike, dict[str, Any]]]:
    """Process all Landsat bands for a single date with QA masking.

    This function loads and processes multiple Landsat bands from the same
    acquisition date, applying QA masks before the data is used for temporal
    compositing. This ensures temporal consistency in cloud masking.

    Args:
        bands_data: Dictionary from download_all_bands_for_date containing
                   band data for a single date
        bbox: Optional bounding box for cropping (min_lon, min_lat, max_lon, max_lat)
        apply_qa_mask: Whether to apply QA-based cloud masking
        dilate_pixels: Number of pixels to dilate cloud mask
        cloud_strategy: Cloud masking strategy ("conservative", "moderate", or "permissive")

    Returns:
        Dictionary mapping band names to tuples of (masked_data, profile)

    Example:
        >>> bands_data = download_all_bands_for_date(2023, 9, 15, bbox)
        >>> processed = process_landsat_date(bands_data, bbox)
        >>> b2_masked = processed['B2'][0]  # Masked Band 2 data
    """
    result = {}

    # First, process QA_PIXEL if available and masking is requested
    cloud_mask = None
    qa_profile = None

    if apply_qa_mask and "QA_PIXEL" in bands_data:
        qa_files, _, _, _ = bands_data["QA_PIXEL"]

        if qa_files:
            print(f"Processing QA_PIXEL from {len(qa_files)} tiles")

            # Create VRT mosaic if multiple tiles
            if len(qa_files) > 1:
                qa_vrt = "data/temp/qa_pixel_mosaic.vrt"
                os.makedirs("data/temp", exist_ok=True)
                # Allow projection differences for areas spanning UTM zones
                vrt_options = gdal.BuildVRTOptions(allowProjectionDifference=True)
                gdal.BuildVRT(qa_vrt, qa_files, options=vrt_options)
                
                # Use windowed read
                qa_data, qa_profile = _read_windowed(qa_vrt, bbox)
            else:
                # Use windowed read for single file too
                qa_data, qa_profile = _read_windowed(qa_files[0], bbox)

            # Debug: check data shape
            print(f"  QA data shape: {qa_data.shape}")
            print(f"  QA data dtype: {qa_data.dtype}")

            # Data is already cropped by windowed read, no need for additional cropping

            # Print QA pixel summary for debugging
            print_qa_pixel_summary(qa_data)

            # Check if QA data is empty
            if qa_data.size == 0:
                print("  WARNING: QA_PIXEL data is empty after loading/cropping!")
                cloud_mask = None
            else:
                # Create cloud mask using bit-based approach
                cloud_mask = create_landsat_cloud_mask(
                    qa_pixel=qa_data,
                    strategy=cloud_strategy,
                    dilate_pixels=dilate_pixels,
                )

                # Debug cloud mask
                mask_clear = cloud_mask.sum()
                mask_total = cloud_mask.size

                if mask_total > 0:
                    print(f"  Cloud mask coverage: {mask_clear / mask_total * 100:.1f}% clear")
                else:
                    print("  WARNING: Cloud mask is empty!")

    # Process each spectral band
    for band_name, (band_files, _, _, _) in bands_data.items():
        if band_name == "QA_PIXEL" or not band_files:
            continue

        print(f"Processing {band_name} from {len(band_files)} tiles")

        # Create VRT mosaic if multiple tiles
        if len(band_files) > 1:
            print(f"  Creating VRT mosaic from {len(band_files)} tiles...")
            
            # Log details about input files
            import logging
            logger = logging.getLogger(__name__)
            for i, file_path in enumerate(band_files):
                logger.debug(f"    Tile {i+1}: {os.path.basename(file_path)}")
            
            band_vrt = f"data/temp/{band_name.lower()}_mosaic.vrt"
            os.makedirs("data/temp", exist_ok=True)
            # Allow projection differences for areas spanning UTM zones
            vrt_options = gdal.BuildVRTOptions(allowProjectionDifference=True)
            vrt_ds = gdal.BuildVRT(band_vrt, band_files, options=vrt_options)
            
            if vrt_ds is not None:
                # Log VRT details
                vrt_bounds = [
                    vrt_ds.GetGeoTransform()[0],  # left
                    vrt_ds.GetGeoTransform()[3] + vrt_ds.GetGeoTransform()[5] * vrt_ds.RasterYSize,  # bottom
                    vrt_ds.GetGeoTransform()[0] + vrt_ds.GetGeoTransform()[1] * vrt_ds.RasterXSize,  # right
                    vrt_ds.GetGeoTransform()[3]  # top
                ]
                logger.info(f"    VRT bounds: {vrt_bounds}")
                logger.info(f"    VRT size: {vrt_ds.RasterXSize} x {vrt_ds.RasterYSize}")
                vrt_ds = None  # Close

            # Use windowed read
            band_data, band_profile = _read_windowed(band_vrt, bbox)
        else:
            # Use windowed read for single file too
            band_data, band_profile = _read_windowed(band_files[0], bbox)

        # Data is already cropped by windowed read, no need for additional cropping

        # Check if band data is empty
        if band_data.size == 0:
            print(f"  WARNING: {band_name} data is empty after loading/cropping!")
            masked_data = band_data
        elif cloud_mask is not None and apply_qa_mask:
            # Apply cloud mask if available
            # Ensure mask and data have same shape
            if cloud_mask.shape != band_data.shape:
                print(
                    f"  Warning: Shape mismatch - band: {band_data.shape}, mask: {cloud_mask.shape}"
                )
                # Crop to common size
                min_height = min(cloud_mask.shape[0], band_data.shape[0])
                min_width = min(cloud_mask.shape[1], band_data.shape[1])
                band_data = band_data[:min_height, :min_width]
                cloud_mask_cropped = cloud_mask[:min_height, :min_width]
            else:
                cloud_mask_cropped = cloud_mask

            # Apply mask: set cloudy pixels to NaN directly
            masked_data = band_data.astype(np.float32)
            masked_data[~cloud_mask_cropped] = np.nan
            clear_pixels = cloud_mask_cropped.sum()
            total_pixels = cloud_mask_cropped.size

            # Debug info
            if total_pixels == 0:
                print(f"  WARNING: {band_name} has zero pixels after cropping!")
                print(f"    Band shape: {band_data.shape}, Mask shape: {cloud_mask.shape}")
                print(f"    Cropped mask shape: {cloud_mask_cropped.shape}")
            else:
                print(
                    f"  {band_name} after masking: "
                    f"{clear_pixels / total_pixels * 100:.1f}% valid pixels"
                )
        else:
            masked_data = band_data

        result[band_name] = (masked_data, band_profile)

    # Log summary of processed data
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Processing summary:")
    for band_name, (data, profile) in result.items():
        if data is not None:
            valid_pixels = ~np.isnan(data) if np.issubdtype(data.dtype, np.floating) else data != 0
            coverage = np.sum(valid_pixels) / data.size * 100 if data.size > 0 else 0
            logger.info(f"  {band_name}: shape={data.shape}, dtype={data.dtype}, coverage={coverage:.1f}%")
            if coverage > 0:
                valid_data = data[valid_pixels]
                logger.info(f"    Range: [{np.min(valid_data):.3f}, {np.max(valid_data):.3f}]")
        else:
            logger.info(f"  {band_name}: None")

    return result
