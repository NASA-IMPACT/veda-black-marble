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
    # Currently in processing.py
    ...


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
    # Currently in processing.py
    ...


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
    # Currently in processing.py
    ...


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
    # Currently in processing.py
    ...


def stack_arrays_with_padding(
    arrays: list[ArrayLike]
) -> ArrayLike:
    """Stack arrays of different shapes by padding to maximum shape.
    
    Args:
        arrays: List of 2D arrays with potentially different shapes
        
    Returns:
        3D array of shape (n_arrays, max_height, max_width)
    """
    # Currently in bm_algorithms.py
    ...


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
    # Currently in bm_algorithms.py (placeholder implementation)
    ...


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
    # Currently in reproj_rasterio.py mosaic_tiles()
    ...


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
    # Currently in reproj.py/reproj_rasterio.py
    ...


def calculate_utm_zone(lat: float, lon: float) -> tuple[int, str]:
    """Calculate UTM zone for given coordinates.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Tuple of (zone_number, hemisphere)
    """
    # Currently in coords.py get_utm_zone()
    ...


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
    # Logic from bm.py UTM conversion
    ...