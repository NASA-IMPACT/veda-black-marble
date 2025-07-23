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
        # Use the file's NoData value for fill, or 0 for Landsat spectral bands
        fill_value = src.nodata if src.nodata is not None else 0
        data = src.read(1, window=window_int, boundless=True, fill_value=fill_value)
        
        # Create profile for the windowed data using the original requested window
        profile = src.profile.copy()
        profile.update({
            'height': int(window_int.height),
            'width': int(window_int.width),
            'transform': rasterio.windows.transform(window_int, src.transform)
        })
        
        logger.debug(f"    Output shape: {data.shape}")
        if data.size > 0:
            logger.debug(f"    Non-zero pixels: {np.count_nonzero(data)} ({np.count_nonzero(data) / data.size * 100:.1f}%)")
        else:
            logger.debug(f"    Empty array returned")
        
        return data, profile


def _calculate_optimal_dimensions(files: list[str], bbox: tuple[float, float, float, float]) -> tuple[int, int]:
    """Calculate optimal dimensions based on actual file resolutions.
    
    This accounts for latitude-dependent distortion by calculating resolution
    at the center of the bounding box.
    
    Args:
        files: List of raster files
        bbox: Output bounds in EPSG:4326 (west, south, east, north)
        
    Returns:
        (width, height) dimensions for output grid
    """
    from pyproj import Transformer
    import math
    
    # Center of the bounding box
    center_lon = (bbox[0] + bbox[2]) / 2
    center_lat = (bbox[1] + bbox[3]) / 2
    
    # Target resolution in meters (from Landsat)
    target_resolution_m = 30.0
    
    # Calculate resolution in degrees at the center latitude
    # Account for latitude-dependent distortion
    # 1 degree of latitude is always ~111km
    res_deg_y = target_resolution_m / 111000.0
    
    # 1 degree of longitude varies by latitude: 111km * cos(latitude)
    lat_rad = math.radians(center_lat)
    meters_per_degree_lon = 111000.0 * math.cos(lat_rad)
    res_deg_x = target_resolution_m / meters_per_degree_lon
    
    # Verify with actual file resolutions if available
    if files:
        file_resolutions = []
        for file in files[:3]:  # Sample a few files
            try:
                with rasterio.open(file) as src:
                    # Get resolution in degrees at file center
                    if src.crs and src.crs.to_epsg() != 4326:
                        file_center_x = (src.bounds.left + src.bounds.right) / 2
                        file_center_y = (src.bounds.bottom + src.bounds.top) / 2
                        
                        transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                        
                        # Transform along each axis separately
                        x1, y1 = transformer.transform(file_center_x, file_center_y)
                        x2, y2 = transformer.transform(file_center_x + abs(src.res[0]), file_center_y)
                        x3, y3 = transformer.transform(file_center_x, file_center_y + abs(src.res[1]))
                        
                        file_res_x = abs(x2 - x1)
                        file_res_y = abs(y3 - y1)
                        file_resolutions.append((file_res_x, file_res_y))
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Could not read resolution from {file}: {e}")
        
        # If we got file resolutions, use the finest (but not if suspiciously different)
        if file_resolutions:
            min_file_res_x = min(r[0] for r in file_resolutions)
            min_file_res_y = min(r[1] for r in file_resolutions)
            
            # Sanity check - file resolution should be close to calculated
            if 0.5 < min_file_res_x / res_deg_x < 2.0:
                res_deg_x = min_file_res_x
            if 0.5 < min_file_res_y / res_deg_y < 2.0:
                res_deg_y = min_file_res_y
    
    # Calculate dimensions with ceiling to ensure full coverage
    width = int(np.ceil((bbox[2] - bbox[0]) / res_deg_x))
    height = int(np.ceil((bbox[3] - bbox[1]) / res_deg_y))
    
    # Debug logging
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Calculated optimal dimensions for mosaic:")
    logger.info(f"  BBox: {bbox}")
    logger.info(f"  Center: ({center_lon:.3f}°, {center_lat:.3f}°)")
    logger.info(f"  Resolution at center latitude:")
    logger.info(f"    X: {res_deg_x:.8f}° (~{res_deg_x * meters_per_degree_lon:.1f}m)")
    logger.info(f"    Y: {res_deg_y:.8f}° (~{res_deg_y * 111000:.1f}m)")
    logger.info(f"  Dimensions: {width} x {height} pixels")
    
    return width, height




def _load_and_mask_tile(
    qa_path: str,
    band_path: str,
    window_expanded: rasterio.windows.Window,
    bbox_src: BBox | None,
    src_crs,
    dilate_pixels: int,
    cloud_strategy: Literal["conservative", "moderate", "permissive"],
) -> tuple[ArrayLike, "rasterio.Affine"]:
    """
    Read matching QA_PIXEL and spectral-band windows from one tile, create a
    cloud mask, apply it immediately, and return the masked band data together
    with the transform valid for the expanded window.

    The caller is responsible for re-projecting the masked output to the
    common mosaic grid.
    """
    import rasterio
    import numpy as np

    # --- open QA and band files in parallel ---
    with rasterio.open(qa_path) as qsrc, rasterio.open(band_path) as bsrc:
        # Safety: ensure both tiles are in the same native CRS
        if qsrc.crs != bsrc.crs:
            raise ValueError(
                f"CRS mismatch for tile pair:\n  QA  : {qa_path}\n  BAND: {band_path}"
            )

        # Read the EXPANDED window (boundless so shape always matches)
        qa_arr = qsrc.read(
            1, window=window_expanded, boundless=True, fill_value=0
        ).astype(np.uint16)

        band_arr = bsrc.read(
            1, window=window_expanded, boundless=True, fill_value=0
        ).astype(np.float32)

        # Window-specific transform (identical for QA and band)
        win_transform = rasterio.windows.transform(window_expanded, bsrc.transform)

    # --- per-tile cloud mask ---
    mask = create_landsat_cloud_mask(
        qa_pixel=qa_arr,
        strategy=cloud_strategy,
        dilate_pixels=dilate_pixels,
    )

    # --- apply mask *in-place* ---
    band_arr[~mask] = np.nan
    return band_arr, win_transform


def process_landsat_date(
    bands_data: dict[str, tuple[list[str], list[str], list[str], list[str]]],
    bbox: BBox | None = None,
    apply_qa_mask: bool = True,
    dilate_pixels: int = 3,
    cloud_strategy: Literal["conservative", "moderate", "permissive"] = "moderate",
    save_diagnostics: bool = False,
    date_str: str | None = None,
    output_shape: tuple[int, int] | None = None,
    processing_crs: Any = None,
    reference_transform: Any = None,
    reference_shape: tuple[int, int] | None = None,
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
        save_diagnostics: Whether to save diagnostic outputs
        date_str: Date string for diagnostic filenames
        output_shape: Optional output shape (height, width) to use instead of auto-calculation

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
    qa_files = []

    if apply_qa_mask and "QA_PIXEL" in bands_data:
        qa_files, _, _, _ = bands_data["QA_PIXEL"]

        if qa_files:
            print(f"Processing QA_PIXEL from {len(qa_files)} tiles")

            # For multiple tiles, always use windowed mosaic approach
            if len(qa_files) > 1:
                print(f"    Creating windowed mosaic from {len(qa_files)} QA tiles...")
                
                # Use the standard processing grid
                from ..crs import create_processing_grid
                
                # Create the standard processing grid
                output_crs, transform, (height, width) = create_processing_grid(bbox)
                
                print(f"    Using locally-optimized Albers Equal Area projection")
                print(f"    Grid dimensions: {width} x {height} at 30m resolution")
                
                qa_data = np.zeros((height, width), dtype=np.uint16)
                qa_profile = {
                    'driver': 'GTiff',
                    'dtype': 'uint16',
                    'width': width,
                    'height': height,
                    'count': 1,
                    'crs': output_crs,
                    'transform': transform,
                    'nodata': None
                }
                
                # Process each file with windowed read and reproject
                for i, qa_file in enumerate(qa_files):
                    print(f"    Processing QA tile {i+1}/{len(qa_files)}: {os.path.basename(qa_file)}")
                    
                    # Read window from source file
                    with rasterio.open(qa_file) as src:
                        # Transform bbox to source CRS
                        if src.crs and src.crs.to_epsg() != 4326:
                            from pyproj import Transformer
                            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                            minx, miny = transformer.transform(bbox[0], bbox[1])
                            maxx, maxy = transformer.transform(bbox[2], bbox[3])
                            bbox_src = (minx, miny, maxx, maxy)
                        else:
                            bbox_src = bbox
                        
                        # Get window for bbox
                        window = rasterio.windows.from_bounds(
                            bbox_src[0], bbox_src[1], bbox_src[2], bbox_src[3],
                            transform=src.transform
                        )
                        
                        # Expand window by margin to avoid gaps at UTM zone boundaries
                        margin_pixels = 120  # ~3.6km at 30m resolution
                        window_expanded = rasterio.windows.Window(
                            col_off=window.col_off - margin_pixels,
                            row_off=window.row_off - margin_pixels,
                            width=window.width + 2 * margin_pixels,
                            height=window.height + 2 * margin_pixels
                        )
                        
                        # Read expanded window
                        window_data = src.read(1, window=window_expanded, boundless=True, fill_value=0)
                        
                        # Calculate the destination window in output grid (for EXPANDED window)
                        from rasterio.warp import reproject, Resampling, transform_bounds
                        
                        # Get bounds of the EXPANDED window in source CRS
                        src_window_bounds = rasterio.windows.bounds(window_expanded, src.transform)
                        
                        # Transform to output CRS
                        dst_bounds = transform_bounds(src.crs, qa_profile['crs'], *src_window_bounds)
                        dst_window = rasterio.windows.from_bounds(*dst_bounds, transform=qa_profile['transform'])
                        
                        # Round the window to integer pixels using correct rounding for north-up rasters
                        from .window_utils import round_window_offsets_correctly
                        dst_window_int = round_window_offsets_correctly(dst_window, qa_profile['transform'])
                        
                        # Reproject to a temporary array matching the destination window size
                        dst_height = dst_window_int.height
                        dst_width = dst_window_int.width
                        tile_data = np.zeros((dst_height, dst_width), dtype=qa_data.dtype)
                        
                        # Calculate transform for the expanded window
                        window_transform = rasterio.windows.transform(window_expanded, src.transform)
                        
                        reproject(
                            source=window_data,
                            destination=tile_data,
                            src_transform=window_transform,
                            src_crs=src.crs,
                            dst_transform=rasterio.windows.transform(dst_window, qa_profile['transform']),
                            dst_crs=qa_profile['crs'],
                            resampling=Resampling.nearest
                        )
                        
                        # Calculate where to place this tile in the full mosaic
                        # DEBUG: Log exact offsets before rounding
                        print(f"      DEBUG: Exact dst_window.row_off = {dst_window.row_off:.6f}")
                        print(f"      DEBUG: Exact dst_window.col_off = {dst_window.col_off:.6f}")
                        print(f"      DEBUG: Delta from rounding - row: {dst_window.row_off - dst_window_int.row_off:.6f}, col: {dst_window.col_off - dst_window_int.col_off:.6f}")
                        
                        # Use the already-rounded window offsets
                        dst_row_off = dst_window_int.row_off
                        dst_col_off = dst_window_int.col_off
                        
                        print(f"      DEBUG: Final dst_row_off = {dst_row_off}")
                        print(f"      DEBUG: Final dst_col_off = {dst_col_off}")
                        
                        # Extract the slice of the mosaic we'll update
                        # Handle negative offsets properly
                        if dst_row_off < 0:
                            # Skip negative rows from tile_data
                            tile_row_start = -dst_row_off
                            mosaic_row_start = 0
                        else:
                            tile_row_start = 0
                            mosaic_row_start = dst_row_off
                        
                        if dst_col_off < 0:
                            # Skip negative cols from tile_data
                            tile_col_start = -dst_col_off
                            mosaic_col_start = 0
                        else:
                            tile_col_start = 0
                            mosaic_col_start = dst_col_off
                        
                        # Calculate end positions
                        mosaic_row_end = min(mosaic_row_start + (dst_height - tile_row_start), qa_data.shape[0])
                        mosaic_col_end = min(mosaic_col_start + (dst_width - tile_col_start), qa_data.shape[1])
                        
                        # Extract slices
                        mosaic_slice = qa_data[
                            mosaic_row_start:mosaic_row_end,
                            mosaic_col_start:mosaic_col_end
                        ]
                        
                        tile_data_cropped = tile_data[
                            tile_row_start:tile_row_start + (mosaic_row_end - mosaic_row_start),
                            tile_col_start:tile_col_start + (mosaic_col_end - mosaic_col_start)
                        ]
                        
                        # Fill only where we don't have data
                        if tile_data_cropped.shape == mosaic_slice.shape:
                            # 0 or NaN in the mosaic means "needs data"
                            nodata_mask = (mosaic_slice == 0) | np.isnan(mosaic_slice)
                            # any finite (non-NaN) value from the tile is valid
                            valid_tile_mask = ~np.isnan(tile_data_cropped)
                            fill_mask = nodata_mask & valid_tile_mask
                            
                            if fill_mask.any():
                                mosaic_slice[fill_mask] = tile_data_cropped[fill_mask]
                                print(f"      Filled {fill_mask.sum():,} pixels")
                        else:
                            print(f"      WARNING: Shape mismatch after cropping - mosaic: {mosaic_slice.shape}, tile: {tile_data_cropped.shape}")
            else:
                # Single file case - still need to reproject to processing grid for consistency
                print(f"    Processing single QA_PIXEL tile...")
                
                # Use the standard processing grid
                from ..crs import create_processing_grid
                
                # Create the standard processing grid
                output_crs, transform, (height, width) = create_processing_grid(bbox)
                
                qa_profile = {
                    'driver': 'GTiff',
                    'dtype': 'uint16',
                    'width': width,
                    'height': height,
                    'count': 1,
                    'crs': output_crs,
                    'transform': transform,
                    'nodata': None
                }
                
                # Read and reproject the single file
                with rasterio.open(qa_files[0]) as src:
                    # Read the entire file
                    src_data = src.read(1)
                    
                    # Create output array
                    qa_data = np.zeros((height, width), dtype=np.uint16)
                    
                    # Reproject to processing grid
                    from rasterio.warp import reproject, Resampling
                    reproject(
                        source=src_data,
                        destination=qa_data,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=qa_profile['transform'],
                        dst_crs=qa_profile['crs'],
                        resampling=Resampling.nearest
                    )
                    
                    print(f"    Reprojected from {src.crs} to {qa_profile['crs']}")

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
                
                # Save cloud mask diagnostic if requested
                if save_diagnostics and date_str:
                    diag_dir = "data/diagnostics/cloud_masks"
                    os.makedirs(diag_dir, exist_ok=True)
                    
                    # Save as GeoTIFF
                    mask_path = f"{diag_dir}/cloud_mask_{date_str}.tif"
                    with rasterio.open(mask_path, 'w', **qa_profile) as dst:
                        dst.write(cloud_mask.astype(np.uint8), 1)
                    print(f"  Saved cloud mask to: {mask_path}")

    # Process each spectral band
    for band_name, (band_files, _, _, _) in bands_data.items():
        if band_name == "QA_PIXEL" or not band_files:
            continue

        print(f"Processing {band_name} from {len(band_files)} tiles")

        # For multiple tiles, always use windowed mosaic approach
        if len(band_files) > 1:
            print(f"  Creating windowed mosaic from {len(band_files)} tiles...")
            
            # Use the same grid as QA_PIXEL if available
            if qa_profile is not None:
                # Reuse the same grid
                band_profile = qa_profile.copy()
                band_profile['dtype'] = 'float32'
                band_profile['nodata'] = 0
                width = band_profile['width']
                height = band_profile['height']
            else:
                # Define common output grid (same logic as QA)
                from ..crs import create_processing_grid
                
                # Create the standard processing grid
                output_crs, transform, (height, width) = create_processing_grid(bbox)
                
                band_profile = {
                    'driver': 'GTiff',
                    'dtype': 'float32',
                    'width': width,
                    'height': height,
                    'count': 1,
                    'crs': output_crs,
                    'transform': transform,
                    'nodata': 0
                }
            
            band_data = np.zeros((height, width), dtype=np.float32)
            
            # Process each file with windowed read and reproject
            for i, band_file in enumerate(band_files):
                print(f"    Processing {band_name} tile {i+1}/{len(band_files)}: {os.path.basename(band_file)}")

                qa_file = qa_files[i] if i < len(qa_files) else None
                if qa_file is None:
                    print(f"      WARNING: No matching QA_PIXEL file for {band_file} – skipping tile")
                    continue

                with rasterio.open(band_file) as src:
                    # Transform bbox to source CRS
                    if src.crs and src.crs.to_epsg() != 4326:
                        from pyproj import Transformer
                        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                        minx, miny = transformer.transform(bbox[0], bbox[1])
                        maxx, maxy = transformer.transform(bbox[2], bbox[3])
                        bbox_src = (minx, miny, maxx, maxy)
                    else:
                        bbox_src = bbox

                    # Build ORIGINAL window then expand symmetrically (allow negatives)
                    window = rasterio.windows.from_bounds(
                        bbox_src[0], bbox_src[1], bbox_src[2], bbox_src[3],
                        transform=src.transform
                    )
                    margin_pixels = 120
                    window_expanded = rasterio.windows.Window(
                        col_off=window.col_off - margin_pixels,
                        row_off=window.row_off - margin_pixels,
                        width=window.width + 2 * margin_pixels,
                        height=window.height + 2 * margin_pixels
                    )

                # --- NEW: load + mask this tile BEFORE any reprojection ---
                tile_data, window_transform = _load_and_mask_tile(
                    qa_path=qa_file,
                    band_path=band_file,
                    window_expanded=window_expanded,
                    bbox_src=bbox_src,
                    src_crs=src.crs,
                    dilate_pixels=dilate_pixels,
                    cloud_strategy=cloud_strategy,
                )

                # Calculate the destination window in output grid (for EXPANDED window)
                from rasterio.warp import reproject, Resampling, transform_bounds
                dst_bounds = transform_bounds(src.crs, band_profile['crs'], *rasterio.windows.bounds(window_expanded, src.transform))
                dst_window = rasterio.windows.from_bounds(*dst_bounds, transform=band_profile['transform'])

                from .window_utils import round_window_offsets_correctly
                dst_window_int = round_window_offsets_correctly(dst_window, band_profile['transform'])

                dst_height = dst_window_int.height
                dst_width = dst_window_int.width
                tile_data_reprojected = np.zeros((dst_height, dst_width), dtype=band_data.dtype)
                
                reproject(
                    source=tile_data,
                    destination=tile_data_reprojected,
                    src_transform=window_transform,
                    src_crs=src.crs,
                    dst_transform=rasterio.windows.transform(dst_window, band_profile['transform']),
                    dst_crs=band_profile['crs'],
                    resampling=Resampling.nearest,
                    src_nodata=np.nan,
                    dst_nodata=0
                )
                
                # Calculate where to place this tile in the full mosaic
                # DEBUG: Log exact offsets before rounding
                print(f"      DEBUG: Exact dst_window.row_off = {dst_window.row_off:.6f}")
                print(f"      DEBUG: Exact dst_window.col_off = {dst_window.col_off:.6f}")
                print(f"      DEBUG: Delta from rounding - row: {dst_window.row_off - dst_window_int.row_off:.6f}, col: {dst_window.col_off - dst_window_int.col_off:.6f}")
                
                # Use the already-rounded window offsets
                dst_row_off = dst_window_int.row_off
                dst_col_off = dst_window_int.col_off
                
                print(f"      DEBUG: Final dst_row_off = {dst_row_off}")
                print(f"      DEBUG: Final dst_col_off = {dst_col_off}")
                
                # Extract the slice of the mosaic we'll update
                # Handle negative offsets properly
                if dst_row_off < 0:
                    # Skip negative rows from tile_data
                    tile_row_start = -dst_row_off
                    mosaic_row_start = 0
                else:
                    tile_row_start = 0
                    mosaic_row_start = dst_row_off
                
                if dst_col_off < 0:
                    # Skip negative cols from tile_data
                    tile_col_start = -dst_col_off
                    mosaic_col_start = 0
                else:
                    tile_col_start = 0
                    mosaic_col_start = dst_col_off
                
                # Calculate end positions
                mosaic_row_end = min(mosaic_row_start + (dst_height - tile_row_start), band_data.shape[0])
                mosaic_col_end = min(mosaic_col_start + (dst_width - tile_col_start), band_data.shape[1])
                
                # Extract slices
                mosaic_slice = band_data[
                    mosaic_row_start:mosaic_row_end,
                    mosaic_col_start:mosaic_col_end
                ]
                
                tile_data_cropped = tile_data_reprojected[
                    tile_row_start:tile_row_start + (mosaic_row_end - mosaic_row_start),
                    tile_col_start:tile_col_start + (mosaic_col_end - mosaic_col_start)
                ]
                
                # Fill only where we don't have data
                if tile_data_cropped.shape == mosaic_slice.shape:
                    # 0 or NaN in the mosaic means "needs data"
                    nodata_mask = (mosaic_slice == 0) | np.isnan(mosaic_slice)
                    # any finite (non-NaN) value from the tile is valid
                    valid_tile_mask = ~np.isnan(tile_data_cropped)
                    fill_mask = nodata_mask & valid_tile_mask
                    
                    if fill_mask.any():
                        mosaic_slice[fill_mask] = tile_data_cropped[fill_mask]
                        print(f"      Filled {fill_mask.sum():,} pixels")
                else:
                    print(f"      WARNING: Shape mismatch after cropping - mosaic: {mosaic_slice.shape}, tile: {tile_data_cropped.shape}")
            
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"    Final mosaic has {(band_data > 0).sum()} valid pixels")
            logger.info(f"    Final mosaic shape: {band_data.shape}")
        else:
            # Single file case - still need to reproject to processing grid for consistency
            print(f"  Processing single {band_name} tile...")
            
            # Use the same grid as QA_PIXEL if available
            if qa_profile is not None:
                # Reuse the same grid
                band_profile = qa_profile.copy()
                band_profile['dtype'] = 'float32'
                band_profile['nodata'] = 0
                width = band_profile['width']
                height = band_profile['height']
            else:
                # Define common output grid
                from ..crs import create_processing_grid
                
                # Create the standard processing grid
                output_crs, transform, (height, width) = create_processing_grid(bbox)
                
                band_profile = {
                    'driver': 'GTiff',
                    'dtype': 'float32',
                    'width': width,
                    'height': height,
                    'count': 1,
                    'crs': output_crs,
                    'transform': transform,
                    'nodata': 0
                }
            
            # Read and reproject the single file
            if qa_files and apply_qa_mask:
                # Apply cloud masking
                with rasterio.open(band_files[0]) as src_band, rasterio.open(qa_files[0]) as src_qa:
                    # Read entire file (no bbox window) and mask immediately
                    full_window = rasterio.windows.Window(0, 0, src_band.width, src_band.height)
                    tile_data, window_transform = _load_and_mask_tile(
                        qa_path=qa_files[0],
                        band_path=band_files[0],
                        window_expanded=full_window,
                        bbox_src=None,
                        src_crs=src_band.crs,
                        dilate_pixels=dilate_pixels,
                        cloud_strategy=cloud_strategy,
                    )

                # Create output array and reproject masked tile to processing grid
                band_data = np.zeros((height, width), dtype=np.float32)
                from rasterio.warp import reproject, Resampling
                reproject(
                    source=tile_data,
                    destination=band_data,
                    src_transform=window_transform,
                    src_crs=src_band.crs,
                    dst_transform=band_profile['transform'],
                    dst_crs=band_profile['crs'],
                    resampling=Resampling.nearest,
                    src_nodata=np.nan,
                    dst_nodata=0
                )
                
                print(f"    Reprojected from {src_band.crs} to {band_profile['crs']}")
            else:
                # No cloud masking - read directly
                with rasterio.open(band_files[0]) as src:
                    # Read the entire file
                    src_data = src.read(1)
                    
                    # Create output array
                    band_data = np.zeros((height, width), dtype=np.float32)
                    
                    # Reproject to processing grid
                    from rasterio.warp import reproject, Resampling
                    reproject(
                        source=src_data,
                        destination=band_data,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=band_profile['transform'],
                        dst_crs=band_profile['crs'],
                        resampling=Resampling.nearest,
                        src_nodata=0,
                        dst_nodata=0
                    )
                    
                    print(f"    Reprojected from {src.crs} to {band_profile['crs']}")

        # Data is already cropped by windowed read, no need for additional cropping

        # Check if band data is empty
        if band_data.size == 0:
            print(f"  WARNING: {band_name} data is empty after loading/cropping!")
            masked_data = band_data
        else:
            # Data were cloud-masked per tile before mosaicking
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
