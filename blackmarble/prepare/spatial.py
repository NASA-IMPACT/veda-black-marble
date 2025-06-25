"""Spatial data preparation and transformations.

This module handles reprojection, resampling, cropping, and alignment
of raster data to prepare it for analysis.
"""

from typing import Any, Literal
import numpy as np
import numpy.typing as npt
from osgeo import gdal
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import Affine

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
    """Reproject image data to target coordinate reference system.

    Args:
        data: Input image array (2D or 3D with bands as first dimension)
        src_crs: Source coordinate reference system (e.g., "EPSG:4326")
        dst_crs: Target coordinate reference system
        src_transform: Affine transform for source data
        dst_shape: Optional output shape (height, width)
        dst_transform: Optional destination transform
        resampling: Resampling method ('bilinear', 'cubic', 'nearest', 'average')
        nodata: Optional nodata value

    Returns:
        Tuple of (reprojected_data, new_transform, new_bounds)
    """
    # Map resampling methods
    resample_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
    }
    resample_alg = resample_map.get(resampling, Resampling.bilinear)

    # Get dimensions
    if data.ndim == 2:
        height, width = data.shape
        bands = 1
        data_3d = data[np.newaxis, ...]
    else:
        bands, height, width = data.shape
        data_3d = data

    # Calculate bounds from transform if not provided
    west = src_transform.c
    north = src_transform.f
    east = west + src_transform.a * width
    south = north + src_transform.e * height
    source_bounds = (west, south, east, north)

    # Convert CRS to string if needed
    source_crs = str(src_crs) if not isinstance(src_crs, str) else src_crs
    target_crs = str(dst_crs) if not isinstance(dst_crs, str) else dst_crs

    # Calculate transform for target CRS if not provided
    if dst_transform is None or dst_shape is None:
        dst_transform, dst_width, dst_height = calculate_default_transform(
            source_crs, target_crs, width, height, *source_bounds
        )
        if dst_shape is None:
            dst_shape = (dst_height, dst_width)

    # Create output array
    reprojected = np.empty((bands,) + dst_shape, dtype=data.dtype)

    # Reproject each band
    for i in range(bands):
        reproject(
            source=data_3d[i],
            destination=reprojected[i],
            src_transform=src_transform,
            src_crs=source_crs,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            resampling=resample_alg,
            src_nodata=nodata,
            dst_nodata=nodata,
        )

    # Calculate new bounds
    west = dst_transform.c
    north = dst_transform.f
    east = west + dst_transform.a * dst_shape[1]
    south = north + dst_transform.e * dst_shape[0]
    new_bounds = (west, south, east, north)

    # Return 2D array if input was 2D
    if data.ndim == 2:
        reprojected = reprojected[0]

    return reprojected, dst_transform, new_bounds


def resample_image(
    data: ArrayLike,
    src_transform: Transform,
    scale_factor: float | None = None,
    target_resolution: float | tuple[float, float] | None = None,
    target_shape: tuple[int, int] | None = None,
    resampling: Literal["nearest", "bilinear", "cubic", "average"] = "bilinear",
    nodata: float | None = None
) -> tuple[ArrayLike, Transform]:
    """Resample image data to target resolution.

    Args:
        data: Input image array (2D or 3D with bands as first dimension)
        src_transform: Source affine transform
        scale_factor: Scale factor for resolution (e.g., 0.5 for half resolution)
        target_resolution: Desired resolution in meters (single value or (x_res, y_res))
        target_shape: Target shape (height, width)
        resampling: Resampling method ('bilinear', 'cubic', 'nearest', 'average')
        nodata: Optional nodata value

    Returns:
        Tuple of (resampled_data, new_transform)
    """
    # Normalize resolutions to tuples
    source_resolution = (abs(src_transform.a), abs(src_transform.e))
    
    if target_resolution is not None:
        if isinstance(target_resolution, (int, float)):
            target_resolution = (target_resolution, target_resolution)
    
    # Map resampling methods
    resample_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
    }
    resample_alg = resample_map.get(resampling, Resampling.bilinear)

    # Get dimensions
    if data.ndim == 2:
        height, width = data.shape
        bands = 1
        data_3d = data[np.newaxis, ...]
    else:
        bands, height, width = data.shape
        data_3d = data

    # Calculate new dimensions based on input parameters
    if scale_factor is not None:
        # Scale factor provided
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        target_resolution = (
            source_resolution[0] / scale_factor,
            source_resolution[1] / scale_factor
        )
    elif target_resolution is not None:
        # Target resolution provided
        scale_x = source_resolution[0] / target_resolution[0]
        scale_y = source_resolution[1] / target_resolution[1]
        new_width = int(width * scale_x)
        new_height = int(height * scale_y)
    elif target_shape is not None:
        # Target shape provided
        new_height, new_width = target_shape
        target_resolution = (
            source_resolution[0] * width / new_width,
            source_resolution[1] * height / new_height
        )
    else:
        # No change needed
        return data, src_transform

    # Create new transform with target resolution
    new_transform = Affine(
        target_resolution[0] * (1 if src_transform.a > 0 else -1),
        src_transform.b,
        src_transform.c,
        src_transform.d,
        target_resolution[1] * (1 if src_transform.e > 0 else -1),
        src_transform.f,
    )

    # Create output array
    resampled = np.empty((bands, new_height, new_width), dtype=data.dtype)

    # Use MemoryFile for in-memory resampling
    from rasterio import MemoryFile
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": bands,
        "dtype": data.dtype,
        "transform": src_transform,
        "nodata": nodata,
    }

    with MemoryFile() as memfile:
        with memfile.open(**profile) as src:
            # Write all bands
            for i in range(bands):
                src.write(data_3d[i], i + 1)

            # Read resampled data
            out_shape = (bands, new_height, new_width)
            resampled = src.read(out_shape=out_shape, resampling=resample_alg)

    # Return 2D array if input was 2D
    if data.ndim == 2:
        resampled = resampled[0]

    return resampled, new_transform


def crop_to_bounds(
    data: ArrayLike,
    src_transform: Transform,
    bounds: BBox,
    buffer_distance: float | None = None,
    buffer_units: Literal["crs", "pixels"] = "crs"
) -> tuple[ArrayLike, Transform, BBox]:
    """Crop raster data to specified spatial bounds.

    Args:
        data: Input image array (2D or 3D with bands as first dimension)
        src_transform: Affine transform for the data
        bounds: Target bounds (min_x, min_y, max_x, max_y) in CRS units
        buffer_distance: Optional buffer distance to add around bounds
        buffer_units: Units for buffer ("crs" for coordinate units or "pixels")

    Returns:
        Tuple of (cropped_data, new_transform, actual_bounds)
    """
    try:
        # Use rasterio for exact cropping
        import rasterio
        from rasterio.windows import from_bounds
        
        # Apply buffer to bounds
        min_x, min_y, max_x, max_y = bounds
        if buffer_distance is not None and buffer_distance > 0:
            if buffer_units == "crs":
                min_x -= buffer_distance
                min_y -= buffer_distance
                max_x += buffer_distance
                max_y += buffer_distance
            else:  # pixels
                pixel_buffer_x = buffer_distance * abs(src_transform.a)
                pixel_buffer_y = buffer_distance * abs(src_transform.e)
                min_x -= pixel_buffer_x
                min_y -= pixel_buffer_y
                max_x += pixel_buffer_x
                max_y += pixel_buffer_y
        
        # Get window from bounds - rasterio handles fractional pixels
        window = from_bounds(min_x, min_y, max_x, max_y, src_transform)
        
        # Round to integer pixels to match GDAL's projWin behavior
        col_off = round(window.col_off)
        row_off = round(window.row_off)
        width = round(window.width)
        height = round(window.height)
        
        # Get data dimensions
        if data.ndim == 2:
            data_height, data_width = data.shape
        else:
            _, data_height, data_width = data.shape
        
        # Ensure we don't exceed data bounds
        col_off = max(0, col_off)
        row_off = max(0, row_off)
        col_end = min(data_width, col_off + width)
        row_end = min(data_height, row_off + height)
        
        # Crop the data
        if data.ndim == 2:
            cropped = data[row_off:row_end, col_off:col_end]
        else:
            cropped = data[:, row_off:row_end, col_off:col_end]
        
        # Calculate the new transform
        new_transform: Affine = src_transform * Affine.translation(col_off, row_off)
        
        # Calculate actual bounds of the cropped data
        actual_height, actual_width = cropped.shape[-2:]
        actual_bounds = (
            new_transform.c,  # west
            new_transform.f + new_transform.e * actual_height,  # south
            new_transform.c + new_transform.a * actual_width,  # east
            new_transform.f   # north
        )
        
        return cropped, new_transform, actual_bounds
        
    except ImportError:
        # Fallback if rasterio not available
        raise ImportError("Rasterio is required for crop_to_bounds")


def align_rasters(
    rasters: list[ArrayLike],
    transforms: list[Transform],
    crs_list: list[str] | None = None,
    reference_idx: int = 0,
    resampling: Literal["nearest", "bilinear", "cubic"] = "nearest",
    nodata: float | None = None
) -> list[tuple[ArrayLike, Transform]]:
    """Align multiple rasters to a common grid and extent.

    Args:
        rasters: List of raster arrays to align
        transforms: List of affine transforms for each raster
        crs_list: Optional list of CRS strings for each raster
        reference_idx: Index of reference raster to align others to
        resampling: Resampling method for alignment
        nodata: Optional nodata value

    Returns:
        List of tuples (aligned_data, transform) for each raster
    """
    if not rasters:
        raise ValueError("No rasters provided for alignment")

    if len(rasters) != len(transforms):
        raise ValueError("Number of rasters must match number of transforms")

    if crs_list and len(crs_list) != len(rasters):
        raise ValueError("Number of CRS strings must match number of rasters")

    # Use reference raster for alignment target
    ref_raster = rasters[reference_idx]
    ref_transform = transforms[reference_idx]
    ref_crs = crs_list[reference_idx] if crs_list else "EPSG:4326"

    # Get reference dimensions
    if ref_raster.ndim == 2:
        ref_height, ref_width = ref_raster.shape
    else:
        _, ref_height, ref_width = ref_raster.shape

    # Map resampling methods
    resample_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
    }
    resample_alg = resample_map.get(resampling, Resampling.nearest)

    # Align each raster
    results = []
    for i, (raster, transform) in enumerate(zip(rasters, transforms)):
        if i == reference_idx:
            # Reference raster stays unchanged
            results.append((raster, transform))
        else:
            # Get source CRS
            src_crs = crs_list[i] if crs_list else ref_crs

            # Handle 2D or 3D data
            if raster.ndim == 2:
                bands = 1
                height, width = raster.shape
                raster_3d = raster[np.newaxis, ...]
            else:
                bands, height, width = raster.shape
                raster_3d = raster

            # Create aligned array
            aligned = np.empty((bands, ref_height, ref_width), dtype=raster.dtype)

            # Reproject each band to reference grid
            for b in range(bands):
                reproject(
                    source=raster_3d[b],
                    destination=aligned[b],
                    src_transform=transform,
                    src_crs=src_crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=resample_alg,
                    src_nodata=nodata,
                    dst_nodata=nodata,
                )

            # Return 2D if input was 2D
            if raster.ndim == 2:
                aligned = aligned[0]

            results.append((aligned, ref_transform))

    return results


def stack_arrays_with_padding(
    arrays: list[ArrayLike]
) -> ArrayLike:
    """Stack arrays of different shapes by padding to maximum shape.
    
    Args:
        arrays: List of 2D arrays with potentially different shapes
        
    Returns:
        3D array of shape (n_arrays, max_height, max_width)
    """
    if not arrays:
        raise ValueError("No arrays provided")
    
    # Find maximum dimensions
    max_height = max(arr.shape[0] for arr in arrays)
    max_width = max(arr.shape[1] for arr in arrays)
    
    # Create output array filled with zeros
    stacked = np.zeros((len(arrays), max_height, max_width), dtype=arrays[0].dtype)
    
    # Copy each array into the output, top-left aligned
    for i, arr in enumerate(arrays):
        h, w = arr.shape
        stacked[i, :h, :w] = arr
    
    return stacked


def crop_and_resample_to_bounds(
    data: ArrayLike,
    transform: Transform,
    bounds: tuple[float, float, float, float],
    target_resolution: float,
    resampling: str = "bilinear",
    crs: str = "EPSG:4326",
) -> tuple[ArrayLike, Transform, tuple[float, float, float, float]]:
    """Crop and resample to exact bounds at specified resolution.
    
    This replicates bm.py's behavior of resampling to a target resolution
    which creates a pixel-aligned grid that can exactly match the bounds.
    
    Args:
        data: Input image array (2D or 3D)
        transform: Affine transform for the data
        bounds: Target bounds (min_x, min_y, max_x, max_y)
        target_resolution: Target pixel size in CRS units
        resampling: Resampling method (default: "bilinear")
        crs: Coordinate reference system (default: "EPSG:4326")
        
    Returns:
        Tuple of (resampled_data, new_transform, actual_bounds)
    """
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds
    
    min_x, min_y, max_x, max_y = bounds
    
    # Calculate output dimensions for exact bounds
    out_width = int(round((max_x - min_x) / target_resolution))
    out_height = int(round((max_y - min_y) / target_resolution))
    
    # Create transform for exact bounds
    out_transform = from_bounds(min_x, min_y, max_x, max_y, out_width, out_height)
    
    # Map resampling string to enum
    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
    }
    resampling_enum = resampling_map.get(resampling, Resampling.bilinear)
    
    # Create output array
    if data.ndim == 2:
        out_shape = (out_height, out_width)
        out_data = np.empty(out_shape, dtype=data.dtype)
        
        # Reproject to exact bounds
        reproject(
            source=data,
            destination=out_data,
            src_transform=transform,
            dst_transform=out_transform,
            src_crs=crs,
            dst_crs=crs,
            resampling=resampling_enum
        )
    else:
        # Handle multi-band
        n_bands = data.shape[0]
        out_shape = (n_bands, out_height, out_width)
        out_data = np.empty(out_shape, dtype=data.dtype)
        
        for i in range(n_bands):
            reproject(
                source=data[i],
                destination=out_data[i],
                src_transform=transform,
                dst_transform=out_transform,
                src_crs=crs,
                dst_crs=crs,
                resampling=resampling_enum
            )
    
    # The output bounds are exactly what was requested
    actual_bounds = (min_x, min_y, max_x, max_y)
    
    return out_data, out_transform, actual_bounds


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
    # This is a wrapper around align_rasters with default resampling
    return align_rasters(
        arrays,
        transforms,
        reference_idx=reference_idx,
        resampling="bilinear"
    )


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
        
        if output_path is not None:
            # Write to file
            profile = datasets[0].profile.copy()
            profile.update({
                'transform': out_transform,
                'width': mosaic.shape[2] if mosaic.ndim > 2 else mosaic.shape[1],
                'height': mosaic.shape[1] if mosaic.ndim > 2 else mosaic.shape[0],
                'count': mosaic.shape[0] if mosaic.ndim > 2 else 1
            })
            
            if bounds:
                profile['bounds'] = bounds
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                if mosaic.ndim == 2:
                    dst.write(mosaic, 1)
                else:
                    dst.write(mosaic)
            
            return output_path
        else:
            # Return in-memory mosaic
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