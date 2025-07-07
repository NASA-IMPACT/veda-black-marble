"""Fractional road rasterization using aligned sub-pixel grids.

This module provides fractional coverage values (0.0-1.0) for roads
by rasterizing at 6m resolution (5x5 sub-pixels per 30m Landsat pixel)
and averaging to get the fraction of sub-pixels covered.
"""

from typing import Any, Literal
import numpy as np
import numpy.typing as npt
import rasterio
import rasterio.features
from rasterio.crs import CRS
from rasterio.transform import Affine
from shapely.geometry import mapping
import networkx as nx
import osmnx as ox

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
    if hierarchical:
        raise NotImplementedError(
            "Hierarchical road values not yet implemented for fractional rasterization. "
            "Set hierarchical=False to use binary fractional coverage."
        )

    # Get edges from graph
    try:
        _, edges_gdf = ox.graph_to_gdfs(graph)
    except (ValueError, KeyError):
        # Empty graph or missing CRS
        return np.zeros(shape, dtype=np.float32)

    if len(edges_gdf) == 0:
        return np.zeros(shape, dtype=np.float32)

    # Buffer roads in meters (project to UTM for accuracy)
    utm_crs = edges_gdf.estimate_utm_crs()
    edges_utm = edges_gdf.to_crs(utm_crs)
    edges_utm["geometry"] = edges_utm.geometry.buffer(buffer_meters)

    # Convert to target CRS for rasterization
    edges_buffered = edges_utm.to_crs(crs)

    # Convert to shapes
    shapes = [mapping(geom) for geom in edges_buffered.geometry]
    print(f"  Processing {len(shapes):,} buffered road segments")

    # Create sub-pixel grid (6m for 30m/5)
    height_30m, width_30m = shape
    height_6m = height_30m * sub_pixels
    width_6m = width_30m * sub_pixels

    # Check memory requirements
    memory_gb = (height_6m * width_6m) / (1024**3)
    print(
        f"  Sub-pixel grid size: {height_6m:,} x {width_6m:,} = {height_6m * width_6m:,} pixels"
    )
    print(f"  Estimated memory requirement: {memory_gb:.2f} GB")

    if memory_gb > 4.0:
        print(f"  ⚠️  WARNING: Large memory requirement!")
        print(
            f"  Consider using binary rasterization (osm_fractional: false) for large areas"
        )

    # For extremely large areas, process in chunks
    if memory_gb > 8.0 or len(shapes) > 1_000_000:
        print(f"  ❌ ERROR: Area too large for fractional rasterization")
        print(f"  Please use binary rasterization instead (set osm_fractional: false)")
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
        print("  ❌ ERROR: Out of memory during rasterization!")
        print("  Falling back to zeros array")
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
        print(f"Fractional road rasterization: {pixels_with_roads} pixels with roads")
        print(f"Coverage range: {min_frac:.3f} to {max_frac:.3f}")
    else:
        print(f"WARNING: No roads rasterized at 30m resolution")
        print(f"  6m raster had {np.sum(raster_6m > 0)} pixels with roads")
        print(f"  Check projection and buffer size")

    return raster_30m_fractional
