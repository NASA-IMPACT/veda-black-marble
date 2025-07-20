"""Fractional road rasterization using aligned sub-pixel grids.

This module provides fractional coverage values (0.0-1.0) for roads
by rasterizing at 6m resolution (5x5 sub-pixels per 30m Landsat pixel)
and averaging to get the fraction of sub-pixels covered.
"""

import logging
from typing import Any

import networkx as nx
import numpy as np
import numpy.typing as npt
import osmnx as ox
import rasterio
import rasterio.features
from rasterio.crs import CRS
from rasterio.transform import Affine
from shapely.geometry import mapping

# Create logger for spatial debugging
log = logging.getLogger("bm.spatial")

# Enable debug logging if environment variable is set
import os
if os.environ.get("BM_SPATIAL_DEBUG", "").lower() in ("1", "true", "yes"):
    log.setLevel(logging.DEBUG)

# Type aliases
ArrayLike = npt.NDArray[Any]
Transform = Affine
BBox = tuple[float, float, float, float]


def buffer_and_rasterize_roads_fractional(
    graph: nx.MultiDiGraph,
    bbox: BBox,
    transform: Transform,
    crs: str | CRS,
    shape: tuple[int, int],
    buffer_meters: float = 30.0,
    sub_pixels: int = 5,
    hierarchical: bool = False,
) -> ArrayLike:
    """Buffer and rasterize roads with fractional coverage values.

    Uses aligned sub-pixel rasterization: creates a 6m grid (30m/5) that
    perfectly aligns with the 30m Landsat grid. Each 30m pixel contains
    exactly 5x5 6m sub-pixels. The fractional coverage is the mean of
    these 25 sub-pixels.

    Args:
        graph: NetworkX graph of road network
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
        transform: Target transform for 30m output
        crs: Target CRS
        shape: Target shape (height, width) at 30m
        buffer_meters: Buffer distance in meters
        sub_pixels: Sub-pixels per dimension (5 = 5x5 = 25 total)
        hierarchical: If True, raises NotImplementedError

    Returns:
        Array with fractional coverage values 0.0-1.0
    """
    log.debug("Entering buffer_and_rasterize_roads_fractional")
    log.debug(f"  CRS parameter: {crs}")
    log.debug(f"  Transform: {transform}")
    log.debug(f"  Shape: {shape}")
    
    if hierarchical:
        raise NotImplementedError(
            "Hierarchical road values not yet implemented for fractional rasterization. "
            "Set hierarchical=False to use binary fractional coverage."
        )

    # Get edges from graph
    try:
        log.debug(f"Converting graph to GeoDataFrame...")
        
        # Check if graph has CRS info
        if not hasattr(graph, 'graph') or 'crs' not in graph.graph:
            log.warning(f"Graph missing CRS attribute, assuming EPSG:4326")
            # Add CRS to graph for osmnx compatibility
            if not hasattr(graph, 'graph'):
                graph.graph = {}
            graph.graph['crs'] = 'EPSG:4326'
        
        _, edges_gdf = ox.graph_to_gdfs(graph)
        log.debug(f"Got {len(edges_gdf)} edges with CRS: {edges_gdf.crs}")
    except (ValueError, KeyError) as e:
        # Empty graph or other error
        log.error(f"Failed to convert graph to GeoDataFrame: {e}")
        return np.zeros(shape, dtype=np.float32)

    if len(edges_gdf) == 0:
        log.error(f"No edges in GeoDataFrame")
        return np.zeros(shape, dtype=np.float32)
    
    # Log initial state
    log.debug(f"STEP 1: edges_gdf created from graph")
    log.debug(f"  - CRS: {edges_gdf.crs}")
    log.debug(f"  - Bounds: {edges_gdf.total_bounds}")
    log.debug(f"  - Num edges: {len(edges_gdf)}")
    if len(edges_gdf) > 0:
        centroid = edges_gdf.unary_union.centroid
        log.debug(f"  - Centroid: ({centroid.x:.6f}, {centroid.y:.6f})")

    # Buffer roads in meters (project to UTM for accuracy)
    utm_crs = edges_gdf.estimate_utm_crs()
    log.debug(f"Estimated UTM CRS for buffering: {utm_crs}")
    edges_utm = edges_gdf.to_crs(utm_crs)
    edges_utm["geometry"] = edges_utm.geometry.buffer(buffer_meters)
    log.debug(f"Buffered {len(edges_utm)} road segments")

    # Convert to target CRS for rasterization
    # Ensure CRS is properly formatted
    if isinstance(crs, str):
        target_crs = CRS.from_string(crs)
    else:
        target_crs = crs
    
    log.debug(f"Target CRS for rasterization: {target_crs}")
    
    edges_buffered = edges_utm.to_crs(target_crs)
    log.debug(f"After projection to target CRS:")
    log.debug(f"  Bounds: {edges_buffered.total_bounds}")
    target_centroid = edges_buffered.unary_union.centroid
    log.debug(f"  Centroid: ({target_centroid.x:.1f}, {target_centroid.y:.1f})")
    
    # Check the transform bounds and overlap
    raster_bounds = rasterio.transform.array_bounds(shape[0], shape[1], transform)
    
    log.debug(f"STEP 5: Checking overlap between roads and raster grid")
    log.debug(f"  - Raster grid info:")
    log.debug(f"    - Shape: {shape}")
    log.debug(f"    - Transform: {transform}")
    log.debug(f"    - Bounds: {raster_bounds}")
    
    # Calculate raster centroid
    raster_minx, raster_miny, raster_maxx, raster_maxy = raster_bounds
    raster_center_x = (raster_minx + raster_maxx) / 2
    raster_center_y = (raster_miny + raster_maxy) / 2
    log.debug(f"    - Centroid: ({raster_center_x:.1f}, {raster_center_y:.1f})")
    
    # Calculate distance between centroids
    if len(edges_buffered) > 0:
        distance = np.sqrt(
            (target_centroid.x - raster_center_x)**2 + 
            (target_centroid.y - raster_center_y)**2
        )
        log.debug(f"  - Distance between centroids: {distance:.1f}m ({distance/1000:.1f}km)")
    
    # Sanity check: ensure bounds overlap
    edges_minx, edges_miny, edges_maxx, edges_maxy = edges_buffered.total_bounds
    raster_minx, raster_miny, raster_maxx, raster_maxy = raster_bounds
    
    # Check for overlap
    x_overlap = not (edges_maxx < raster_minx or edges_minx > raster_maxx)
    y_overlap = not (edges_maxy < raster_miny or edges_miny > raster_maxy)
    
    log.debug(f"  - Overlap check:")
    log.debug(f"    - X-axis overlap: {x_overlap}")
    log.debug(f"      Roads X: [{edges_minx:.1f}, {edges_maxx:.1f}]")
    log.debug(f"      Raster X: [{raster_minx:.1f}, {raster_maxx:.1f}]")
    log.debug(f"      Gap: {max(0, raster_minx - edges_maxx):.1f}m (left), {max(0, edges_minx - raster_maxx):.1f}m (right)")
    log.debug(f"    - Y-axis overlap: {y_overlap}")
    log.debug(f"      Roads Y: [{edges_miny:.1f}, {edges_maxy:.1f}]")
    log.debug(f"      Raster Y: [{raster_miny:.1f}, {raster_maxy:.1f}]")
    log.debug(f"      Gap: {max(0, raster_miny - edges_maxy):.1f}m (bottom), {max(0, edges_miny - raster_maxy):.1f}m (top)")
    
    if not (x_overlap and y_overlap):
        log.warning(f"⚠️  WARNING: No overlap between roads and raster!")
        log.warning(f"  This will result in zero roads being rasterized")
        
        # This will result in zero roads being rasterized

    # Convert to shapes
    shapes = [mapping(geom) for geom in edges_buffered.geometry]
    log.debug(f"Processing {len(shapes):,} buffered road segments")

    # Create sub-pixel grid (6m for 30m/5)
    height_30m, width_30m = shape
    height_6m = height_30m * sub_pixels
    width_6m = width_30m * sub_pixels

    # Check memory requirements
    memory_gb = (height_6m * width_6m) / (1024**3)
    log.debug(f"Sub-pixel grid size: {height_6m:,} x {width_6m:,} = {height_6m * width_6m:,} pixels")
    log.debug(f"Estimated memory requirement: {memory_gb:.2f} GB")

    if memory_gb > 4.0:
        log.warning("Large memory requirement!")
        log.warning("Consider using binary rasterization (osm_fractional: false) for large areas")

    # For extremely large areas, process in chunks
    if memory_gb > 8.0 or len(shapes) > 1_000_000:
        log.error("Area too large for fractional rasterization")
        log.error("Please use binary rasterization instead (set osm_fractional: false)")
        return np.zeros(shape, dtype=np.float32)

    # Scale transform for 6m resolution
    transform_6m = transform * transform.scale(1 / sub_pixels, 1 / sub_pixels)

    # Rasterize at 6m resolution
    # Use all_touched=False for more precise sub-pixel coverage
    try:
        raster_6m = rasterio.features.rasterize(
            shapes,
            out_shape=(height_6m, width_6m),
            transform=transform_6m,
            fill=0,
            all_touched=False,
            dtype=np.uint8,
        )
    except MemoryError:
        log.error("Out of memory during rasterization!")
        log.error("Falling back to zeros array")
        return np.zeros(shape, dtype=np.float32)

    # Calculate fractional coverage for each 30m pixel
    # Reshape to group sub-pixels, then take mean
    raster_30m_fractional = (
        raster_6m.reshape(height_30m, sub_pixels, width_30m, sub_pixels)
        .mean(axis=(1, 3))
        .astype(np.float32)
    )

    # Log statistics
    pixels_with_roads = np.sum(raster_30m_fractional > 0)
    if pixels_with_roads > 0:
        min_frac = raster_30m_fractional[raster_30m_fractional > 0].min()
        max_frac = raster_30m_fractional.max()
        log.info(f"Fractional road rasterization: {pixels_with_roads} pixels with roads")
        log.info(f"Coverage range: {min_frac:.3f} to {max_frac:.3f}")
    else:
        log.warning("No roads rasterized at 30m resolution")
        log.warning(f"  6m raster had {np.sum(raster_6m > 0)} pixels with roads")
        log.warning("  Check projection and buffer size")

    return raster_30m_fractional
