"""OpenStreetMap road network data acquisition.

This module handles fetching road network data from OpenStreetMap using OSMnx,
and converting it to raster format for integration with satellite imagery.
"""

import hashlib
import shutil
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pyproj
import rasterio
import rasterio.features
from rasterio.crs import CRS
from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape
from shapely.ops import transform

from blackmarble.typing import ArrayLike, BBox

from ..logging_utils import get_logger


logger = get_logger(__name__)

# Import default buffer sizes from fractional module

# Type aliases
Transform = Any  # rasterio.transform.Affine


def download_osm_roads(
    north: float,
    south: float,
    east: float,
    west: float,
    road_filter: str | None = None,
) -> "nx.MultiDiGraph[Any]":
    """Download OSM road network for a bounding box."""
    logger.info("Downloading roads for bbox: N=%s, S=%s, E=%s, W=%s", north, south, east, west)
    if road_filter:
        logger.info("Using filter: %s", road_filter)

    logger.info("Fetching OSM road network...")
    try:
        # osmnx expects bbox as (west, south, east, north)
        G = ox.graph_from_bbox(
            bbox=(west, south, east, north),
            retain_all=True,
            truncate_by_edge=True,
            simplify=True,
            network_type="all",
            custom_filter=road_filter if road_filter else None,
        )

        logger.info("Downloaded %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

        # Add CRS attribute to graph to avoid warnings
        G.graph["crs"] = "EPSG:4326"

        return G
    except Exception as e:
        logger.error("Error downloading roads: %s: %s", type(e).__name__, e)
        raise


def clear_osm_cache(cache_dir: str | Path = "data/osm_cache") -> None:
    """Clear all cached OSM data."""
    cache_dir_path = Path(cache_dir)
    if cache_dir_path.exists():
        shutil.rmtree(cache_dir_path)
        logger.info("Cleared OSM cache directory: %s", cache_dir_path)


def get_osm_cache_path(
    bbox: BBox,
    network_type: str,
    custom_filter: str | None,
    cache_dir: str | Path = "data/osm_cache",
) -> Path:
    """Generate a cache file path for OSM data based on parameters."""
    # Create a unique hash for the parameters
    params_str = f"{bbox}_{network_type}_{custom_filter}"
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:12]

    # Create cache directory
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    # Generate filename with bbox info for readability
    min_lon, min_lat, max_lon, max_lat = bbox
    # Use .gpkg extension for GeoPackage
    filename = f"osm_{min_lon:.2f}_{min_lat:.2f}_{max_lon:.2f}_{max_lat:.2f}_{params_hash}.gpkg"

    return cache_dir_path / filename


def fetch_road_network(
    bbox: BBox,
    network_type: str = "all",
    custom_filter: str | None = None,
    use_cache: bool = True,
    cache_dir: str | Path = "data/osm_cache",
) -> Any:  # Returns networkx graph
    """Fetch road network from OpenStreetMap with caching support.

    Args:
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
        network_type: Type of road network to fetch
        custom_filter: Custom OSM filter string
        use_cache: Whether to use cached data if available

    Returns:
        NetworkX graph of the road network
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    # Check cache first
    if use_cache:
        cache_path = get_osm_cache_path(bbox, network_type, custom_filter, cache_dir=cache_dir)

        if cache_path.exists():
            logger.info("Loading OSM data from cache: %s", cache_path.name)
            try:
                nodes_gdf = gpd.read_file(cache_path, layer="nodes")
                edges_gdf = gpd.read_file(cache_path, layer="edges")

                if "osmid" in nodes_gdf.columns:
                    nodes_gdf = nodes_gdf.set_index("osmid")
                if all(c in edges_gdf.columns for c in ("u", "v", "key")):
                    edges_gdf = edges_gdf.set_index(["u", "v", "key"])
                elif all(c in edges_gdf.columns for c in ("u", "v")):
                    edges_gdf["key"] = 0
                    edges_gdf = edges_gdf.set_index(["u", "v", "key"])

                graph = ox.graph_from_gdfs(nodes_gdf, edges_gdf)

                logger.info(
                    "Loaded %d nodes, %d edges from cache",
                    graph.number_of_nodes(),
                    graph.number_of_edges(),
                )

                # Ensure CRS attribute is present to avoid warnings
                graph.graph.setdefault("crs", "EPSG:4326")

                return graph
            except Exception as e:
                logger.warning("Failed to load cache: %s. Downloading fresh data...", e)

    # Download fresh data
    logger.info("Downloading OSM data from server...")
    graph = download_osm_roads(max_lat, min_lat, max_lon, min_lon, custom_filter)

    # Save to cache
    if use_cache and graph.number_of_edges() > 0:
        cache_path = get_osm_cache_path(bbox, network_type, custom_filter, cache_dir=cache_dir)
        try:
            # Convert graph to GeoDataFrames and cache both layers.
            # Storing nodes+edges preserves original node IDs and edge keys.
            nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph)

            # Save to GeoPackage - no size limitations!
            nodes_gdf.to_file(cache_path, layer="nodes", driver="GPKG")
            edges_gdf.to_file(cache_path, layer="edges", driver="GPKG")
            logger.info("Saved OSM data to cache: %s", cache_path.name)
            logger.info("  Cached %d road segments", len(edges_gdf))
        except Exception as e:
            logger.warning("Failed to save cache: %s", e)
            # Continue without caching - the download succeeded

    return graph


def calculate_utm_zone(lon: float, lat: float) -> tuple[int, Literal["north", "south"]]:
    """Calculate UTM zone and hemisphere for coordinates."""
    zone = min(60, int((lon + 180) / 6) + 1)
    hemisphere: Literal["north", "south"] = "north" if lat >= 0 else "south"
    return zone, hemisphere


def buffer_roads_in_utm(
    geometry: dict[str, Any], buffer_meters: float, lon_center: float, lat_center: float
) -> dict[str, Any]:
    """Buffer a road geometry in meters using UTM projection."""
    zone, hemisphere = calculate_utm_zone(lon_center, lat_center)
    epsg_code = 32600 + zone if hemisphere == "north" else 32700 + zone

    wgs84_to_utm = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)
    utm_to_wgs84 = pyproj.Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)

    geom = shape(geometry)
    geom_utm = transform(wgs84_to_utm.transform, geom)
    buffered_utm = geom_utm.buffer(buffer_meters)
    buffered_wgs84 = transform(utm_to_wgs84.transform, buffered_utm)

    return mapping(buffered_wgs84)


def rasterize_road_shapes_with_crs(
    shapes: list[dict[str, Any]],
    reference_transform: Transform,
    shape: tuple[int, int],
    src_crs: str | CRS,
    dst_crs: str | CRS,
    all_touched: bool = False,
) -> ArrayLike:
    """Rasterize road shapes with coordinate system transformation."""
    if isinstance(src_crs, str):
        src_crs = CRS.from_string(src_crs)
    if isinstance(dst_crs, str):
        dst_crs = CRS.from_string(dst_crs)

    should_transform = src_crs != dst_crs
    rasterize_shapes = []
    for shape_dict in shapes:
        geom = shape_dict["geometry"]
        value = shape_dict.get("value", 1)

        if should_transform:
            geom = transform_geom(src_crs, dst_crs, geom)

        rasterize_shapes.append((geom, value))

    height, width = shape
    return rasterio.features.rasterize(
        rasterize_shapes,
        out_shape=(height, width),
        transform=reference_transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    )
