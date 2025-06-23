"""Spatial data preparation and transformations.

This module handles reprojection, resampling, cropping, and alignment
of raster data to prepare it for analysis.
"""

from typing import Any, Literal
import numpy.typing as npt

# Type aliases
ArrayLike = npt.NDArray[Any]
Transform = Any  # rasterio.transform.Affine
CRS = Any  # rasterio.crs.CRS
BBox = tuple[float, float, float, float]  # (minx, miny, maxx, maxy)


def reproject_image(
    data: ArrayLike,
    src_transform: Transform,
    src_crs: CRS,
    dst_crs: CRS,
    dst_shape: tuple[int, int] | None = None,
    dst_transform: Transform | None = None,
    resampling: Literal["nearest", "bilinear", "cubic", "average"] = "nearest",
    nodata: float | None = None
) -> tuple[ArrayLike, Transform, BBox]:
    """Reproject raster data to different coordinate system.
    
    Args:
        data: Input raster array (2D or 3D)
        src_transform: Source affine transform
        src_crs: Source coordinate reference system
        dst_crs: Destination coordinate reference system
        dst_shape: Optional output shape (height, width)
        dst_transform: Optional destination transform
        resampling: Resampling method
        nodata: NoData value
        
    Returns:
        Tuple of (reprojected_data, new_transform, new_bounds)
    """
    # Implementation adapted from processing.py:117
    # WATCHPOINT: processing.py expects CRS as strings, we have CRS objects
    # WATCHPOINT: processing.py uses different parameter names
    from ..processing import reproject_image as _reproject_image
    
    # Convert CRS objects to strings if needed
    src_crs_str = str(src_crs) if not isinstance(src_crs, str) else src_crs
    dst_crs_str = str(dst_crs) if not isinstance(dst_crs, str) else dst_crs
    
    # Calculate source bounds if not provided
    if data.ndim == 2:
        height, width = data.shape
    else:
        _, height, width = data.shape
    
    west = src_transform.c
    north = src_transform.f
    east = west + src_transform.a * width
    south = north + src_transform.e * height
    source_bounds = (west, south, east, north)
    
    return _reproject_image(
        data,
        source_crs=src_crs_str,
        target_crs=dst_crs_str,
        source_transform=src_transform,
        source_bounds=source_bounds,
        method=resampling,
        nodata=nodata
    )


def resample_image(
    data: ArrayLike,
    src_transform: Transform,
    scale_factor: float | None = None,
    target_resolution: float | tuple[float, float] | None = None,
    target_shape: tuple[int, int] | None = None,
    resampling: Literal["nearest", "bilinear", "cubic", "average"] = "bilinear"
) -> tuple[ArrayLike, Transform]:
    """Resample raster to different resolution.
    
    Args:
        data: Input raster array
        src_transform: Source affine transform
        scale_factor: Scale factor for resolution (e.g., 0.5 for half resolution)
        target_resolution: Target resolution in CRS units
        target_shape: Target shape (height, width)
        resampling: Resampling method
        
    Returns:
        Tuple of (resampled_data, new_transform)
    """
    # Implementation adapted from processing.py:19
    # WATCHPOINT: processing.py expects source/target resolution, not scale_factor or target_shape
    from ..processing import resample_image as _resample_image
    
    # Extract current resolution from transform
    source_resolution = (abs(src_transform.a), abs(src_transform.e))
    
    # Determine target resolution
    if target_resolution is not None:
        # Use provided target resolution
        if isinstance(target_resolution, (int, float)):
            target_res = (target_resolution, target_resolution)
        else:
            target_res = target_resolution
    elif scale_factor is not None:
        # Calculate from scale factor
        target_res = (source_resolution[0] / scale_factor, source_resolution[1] / scale_factor)
    elif target_shape is not None:
        # Calculate from target shape
        if data.ndim == 2:
            height, width = data.shape
        else:
            _, height, width = data.shape
        target_res = (
            source_resolution[0] * width / target_shape[1],
            source_resolution[1] * height / target_shape[0]
        )
    else:
        # No change
        return data, src_transform
    
    return _resample_image(
        data,
        source_resolution=source_resolution,
        target_resolution=target_res,
        method=resampling,
        transform=src_transform,
        nodata=None
    )


def crop_to_bounds(
    data: ArrayLike,
    src_transform: Transform,
    bounds: BBox,
    buffer_distance: float | None = None,
    buffer_units: Literal["crs", "pixels"] = "crs"
) -> tuple[ArrayLike, Transform, BBox]:
    """Crop raster to specified bounds.
    
    Args:
        data: Input raster array
        src_transform: Source affine transform
        bounds: Target bounds (minx, miny, maxx, maxy)
        buffer_distance: Optional buffer to add around bounds
        buffer_units: Units for buffer ('crs' or 'pixels')
        
    Returns:
        Tuple of (cropped_data, new_transform, actual_bounds)
    """
    # Implementation from processing.py:206
    # WATCHPOINT: Parameter name mismatch - 'buffer_distance' vs 'buffer'
    from ..processing import crop_to_bounds as _crop_to_bounds
    
    buffer_val = buffer_distance if buffer_distance is not None else 0.0
    
    return _crop_to_bounds(
        data,
        transform=src_transform,
        bounds=bounds,
        buffer=buffer_val,
        buffer_units=buffer_units
    )


def align_rasters(
    rasters: list[ArrayLike],
    transforms: list[Transform],
    reference_idx: int = 0,
    resampling: Literal["nearest", "bilinear", "cubic"] = "nearest",
    target_crs: CRS | None = None
) -> list[tuple[ArrayLike, Transform]]:
    """Align multiple rasters to common grid.
    
    Args:
        rasters: List of raster arrays
        transforms: List of affine transforms
        reference_idx: Index of reference raster for alignment
        resampling: Resampling method
        target_crs: Optional target CRS (uses reference CRS if None)
        
    Returns:
        List of tuples (aligned_raster, transform)
    """
    # Implementation from processing.py:278
    # WATCHPOINT: processing.py expects crs_list parameter, not target_crs
    # WATCHPOINT: processing.py uses 'method' not 'resampling'
    from ..processing import align_rasters as _align_rasters
    
    # Create CRS list if target_crs provided
    crs_list = None
    if target_crs is not None:
        crs_str = str(target_crs) if not isinstance(target_crs, str) else target_crs
        crs_list = [crs_str] * len(rasters)
    
    return _align_rasters(
        rasters,
        transforms,
        crs_list=crs_list,
        reference_idx=reference_idx,
        method=resampling,
        nodata=None
    )


def stack_arrays_with_padding(
    arrays: list[ArrayLike]
) -> ArrayLike:
    """Stack arrays of different shapes by padding to maximum shape.
    
    Args:
        arrays: List of 2D arrays with potentially different shapes
        
    Returns:
        3D array of shape (n_arrays, max_height, max_width)
    """
    # Implementation from bm_algorithms.py:297
    from ..bm_algorithms import stack_arrays_with_padding as _stack_arrays_with_padding
    return _stack_arrays_with_padding(arrays)


def align_to_common_grid(
    arrays: list[ArrayLike],
    transforms: list[Transform],
    reference_idx: int = 0
) -> list[tuple[ArrayLike, Transform]]:
    """Align arrays to common spatial grid.
    
    Args:
        arrays: List of arrays to align
        transforms: List of affine transforms
        reference_idx: Index of reference array
        
    Returns:
        List of tuples (aligned_array, transform)
    """
    # Implementation from bm_algorithms.py:333
    # WATCHPOINT: bm_algorithms.py version is a placeholder that returns input unchanged
    from ..bm_algorithms import align_to_common_grid as _align_to_common_grid
    return _align_to_common_grid(arrays, transforms, reference_idx)


def mosaic_tiles(
    tiles: list[str],
    output_path: str | None = None,
    resampling: str = "bilinear",
    bounds: BBox | None = None,
    resolution: float | None = None
) -> str | tuple[ArrayLike, Transform, CRS]:
    """Mosaic multiple raster tiles into single raster.
    
    Args:
        tiles: List of tile file paths
        output_path: Optional output file path
        resampling: Resampling method
        bounds: Optional bounds for output mosaic
        resolution: Optional output resolution
        
    Returns:
        Output path if output_path provided, else (data, transform, crs)
    """
    # Implementation adapted from mosaic_rasterio.py:16
    # WATCHPOINT: mosaic_rasterio version always writes to file, doesn't return data
    # We need to handle the case where output_path is None
    
    if output_path is not None:
        # Use existing implementation that writes to file
        from ..mosaic_rasterio import mosaic_tiles as _mosaic_tiles
        _mosaic_tiles(
            input_tiles=tiles,
            output_path=output_path,
            resampling=resampling,
            nodata=None,
            dtype=None,
            bounds=bounds,
            res=resolution
        )
        return output_path
    else:
        # Need to mosaic in memory
        import numpy as np
        import rasterio
        from rasterio.merge import merge
        from rasterio.warp import Resampling
        
        # Map resampling string to enum
        resampling_map = {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
            "average": Resampling.average,
        }
        resampling_enum = resampling_map.get(resampling, Resampling.bilinear)
        
        # Open all datasets
        datasets = [rasterio.open(tile) for tile in tiles]
        
        try:
            # Merge tiles
            mosaic, out_transform = merge(
                datasets,
                bounds=bounds,
                res=resolution,
                resampling=resampling_enum
            )
            
            # Get CRS from first dataset
            out_crs = datasets[0].crs
            
            # Convert from (bands, height, width) to (height, width) if single band
            if mosaic.shape[0] == 1:
                mosaic = mosaic[0]
            
            return mosaic, out_transform, out_crs
        finally:
            # Close all datasets
            for ds in datasets:
                ds.close()


def convert_to_wgs84(
    data: ArrayLike,
    src_transform: Transform,
    src_crs: CRS,
    bounds: BBox,
    target_resolution: float | None = None
) -> tuple[ArrayLike, Transform]:
    """Convert raster to WGS84 (EPSG:4326) projection.
    
    Args:
        data: Input raster array
        src_transform: Source affine transform
        src_crs: Source CRS
        bounds: Geographic bounds for output
        target_resolution: Optional target resolution in degrees
        
    Returns:
        Tuple of (wgs84_data, wgs84_transform)
    """
    # Use reproject_image with WGS84 as target
    wgs84_data, wgs84_transform, _ = reproject_image(
        data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_crs="EPSG:4326",
        resampling="bilinear"
    )
    
    # If target resolution specified, resample
    if target_resolution is not None:
        wgs84_data, wgs84_transform = resample_image(
            wgs84_data,
            src_transform=wgs84_transform,
            target_resolution=target_resolution,
            resampling="bilinear"
        )
    
    return wgs84_data, wgs84_transform


def calculate_utm_zone(lat: float, lon: float) -> tuple[int, str]:
    """Calculate UTM zone for given coordinates.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Tuple of (zone_number, hemisphere)
    """
    # Implementation adapted from coords.py:22 get_utm_zone()
    # WATCHPOINT: coords.py returns (easting, northing, zone_number, zone_letter)
    # We need just (zone_number, hemisphere)
    from ..coords import get_utm_zone
    
    easting, northing, zone_number, zone_letter = get_utm_zone(lat, lon)
    
    # Determine hemisphere from zone letter or latitude
    if zone_letter and ord(zone_letter) >= ord('N'):
        hemisphere = 'N'
    else:
        hemisphere = 'S' if lat < 0 else 'N'
    
    return zone_number, hemisphere


def bounds_to_utm(
    bounds: BBox,
    center_lat: float,
    center_lon: float
) -> tuple[BBox, CRS]:
    """Convert geographic bounds to appropriate UTM projection.
    
    Args:
        bounds: Geographic bounds (minx, miny, maxx, maxy) in degrees
        center_lat: Center latitude for UTM zone selection
        center_lon: Center longitude for UTM zone selection
        
    Returns:
        Tuple of (utm_bounds, utm_crs)
    """
    # Calculate UTM zone
    zone_number, hemisphere = calculate_utm_zone(center_lat, center_lon)
    
    # Create UTM CRS
    import rasterio.crs
    utm_crs = rasterio.crs.CRS.from_epsg(32600 + zone_number if hemisphere == 'N' else 32700 + zone_number)
    
    # Transform bounds to UTM
    from rasterio.warp import transform_bounds
    utm_bounds = transform_bounds("EPSG:4326", utm_crs, *bounds)
    
    return utm_bounds, utm_crs