"""Spatial intersection-based VIIRS tile selection."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box

logger = logging.getLogger(__name__)


def get_tile_footprint_wgs84(tile_path: Path) -> tuple[float, float, float, float]:
    """Get the footprint of a VIIRS tile in WGS84.
    
    Args:
        tile_path: Path to VIIRS tile
        
    Returns:
        Bounds in WGS84 (min_lon, min_lat, max_lon, max_lat)
    """
    with rasterio.open(tile_path) as src:
        bounds = src.bounds
        
        # Transform to WGS84 if needed
        if src.crs and str(src.crs) != "EPSG:4326":
            bounds = transform_bounds(src.crs, "EPSG:4326", *bounds)
        
        return bounds


def find_viirs_tiles_by_intersection(
    bbox: tuple[float, float, float, float],
    viirs_base_dir: Path,
    date_filter: Optional[str] = None
) -> list[Path]:
    """Find VIIRS tiles that intersect with the given bbox using spatial overlap.
    
    Args:
        bbox: Target bounding box (min_lon, min_lat, max_lon, max_lat) in WGS84
        viirs_base_dir: Base directory containing VIIRS tiles
        date_filter: Optional date string to filter tiles (e.g., "20230124")
        
    Returns:
        List of paths to tiles that intersect the bbox
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    target_box = box(min_lon, min_lat, max_lon, max_lat)
    
    logger.info(f"Selecting VIIRS tiles by spatial intersection for bbox: {bbox}")
    
    # Find all VIIRS tiles
    pattern = f"**/*{date_filter}*.tif" if date_filter else "**/*.tif"
    viirs_files = list(viirs_base_dir.glob(pattern))
    
    if not viirs_files:
        logger.warning(f"No VIIRS files found in {viirs_base_dir} with pattern {pattern}")
        return []
    
    selected_tiles = []
    
    for tile_path in viirs_files:
        try:
            # Get tile footprint
            tile_bounds = get_tile_footprint_wgs84(tile_path)
            tile_box = box(*tile_bounds)
            
            # Check intersection
            if target_box.intersects(tile_box):
                intersection = target_box.intersection(tile_box)
                overlap_pct = (intersection.area / target_box.area) * 100
                logger.info(f"  ✓ {tile_path.name}: {overlap_pct:.1f}% overlap")
                selected_tiles.append(tile_path)
                
        except Exception as e:
            logger.warning(f"Error checking tile {tile_path.name}: {e}")
    
    logger.info(f"Selected {len(selected_tiles)} tiles by spatial intersection")
    return selected_tiles


def create_viirs_mosaic(tile_paths: list[Path], target_crs: str = "EPSG:4326") -> tuple[np.ndarray, dict]:
    """Create a mosaic from multiple VIIRS tiles.
    
    Args:
        tile_paths: List of paths to VIIRS tiles
        target_crs: Target CRS for the mosaic
        
    Returns:
        Tuple of (mosaic_array, profile_dict)
    """
    if not tile_paths:
        raise ValueError("No tiles provided for mosaic")
    
    if len(tile_paths) == 1:
        # Single tile, just read it
        with rasterio.open(tile_paths[0]) as src:
            data = src.read(1)
            profile = src.profile.copy()
            return data, profile
    
    # Multiple tiles - create mosaic using rasterio merge
    from rasterio.merge import merge
    
    logger.info(f"Creating mosaic from {len(tile_paths)} tiles")
    
    # Open all tiles
    src_files = [rasterio.open(p) for p in tile_paths]
    
    try:
        # Merge tiles
        mosaic, out_transform = merge(src_files, method='max')
        
        # Create profile from first file and update
        profile = src_files[0].profile.copy()
        profile.update({
            'transform': out_transform,
            'height': mosaic.shape[1],
            'width': mosaic.shape[2],
            'crs': target_crs
        })
        
        # Return 2D array
        return mosaic[0], profile
        
    finally:
        # Close all files
        for src in src_files:
            src.close()


def lonlat_to_modis_sinusoidal_tile(lon: float, lat: float) -> tuple[int, int]:
    """
    Convert lon/lat to MODIS sinusoidal tile indices (h, v).
    
    This is a corrected implementation based on the MODIS sinusoidal projection.
    
    Args:
        lon: Longitude in degrees
        lat: Latitude in degrees
        
    Returns:
        Tuple of (h, v) tile indices
    """
    # MODIS sinusoidal projection parameters
    R = 6371007.181  # Earth radius used by MODIS (meters)
    T = 1111950.5196666667  # Tile size at equator (meters)
    xmin = -20015109.354  # Western edge of projection
    ymax = 10007554.677   # Northern edge of projection
    
    # Convert to radians
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    
    # Sinusoidal projection
    x = R * lon_rad * np.cos(lat_rad)
    y = R * lat_rad
    
    # Calculate tile indices
    h = int((x - xmin) / T)
    v = int((ymax - y) / T)
    
    # Clamp to valid ranges
    h = max(0, min(35, h))
    v = max(0, min(17, v))
    
    return h, v