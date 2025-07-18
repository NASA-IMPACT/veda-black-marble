"""OpenStreetMap road network data acquisition.

This module handles fetching road network data from OpenStreetMap using OSMnx,
and converting it to raster format for integration with satellite imagery.
"""

from typing import Any, Literal
import numpy.typing as npt
import numpy as np
import rasterio
import rasterio.features
from rasterio.crs import CRS
from rasterio.warp import transform_geom
from shapely.geometry import shape, mapping
from shapely.geometry import mapping as geom_to_dict
from shapely.ops import transform
import pyproj
import shapely
import osmnx as ox
import networkx as nx
import os
import hashlib
from pathlib import Path
import geopandas as gpd

# Type aliases
BBox = tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
ArrayLike = npt.NDArray[Any]
Transform = Any  # rasterio.transform.Affine


def download_osm_roads(
    north: float,
    south: float,
    east: float,
    west: float,
    road_filter: str | None = None,
) -> nx.MultiDiGraph:
    """Download OSM road network for a bounding box."""
    print(f"Downloading roads for bbox: N={north}, S={south}, E={east}, W={west}")
    if road_filter:
        print(f"Using filter: {road_filter}")

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
        print(f"Downloaded {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    except Exception as e:
        print(f"Error downloading roads: {type(e).__name__}: {e}")
        raise


def graph_to_shapes(
    G: nx.MultiDiGraph, buffer_distance: float = 0.00005
) -> list[dict[str, Any]]:
    """Convert road graph to buffered shapes."""
    if G.number_of_edges() == 0:
        return []

    _, edges_gdf = ox.graph_to_gdfs(G)
    shapes = []
    for _, edge in edges_gdf.iterrows():
        geom = (
            edge.geometry.buffer(buffer_distance)
            if buffer_distance > 0
            else edge.geometry
        )
        shapes.append(shapely.geometry.mapping(geom))

    print(f"Created {len(shapes)} buffered road shapes")
    return shapes


def rasterize_shapes(
    shapes: list[dict[str, Any]],
    height: int,
    width: int,
    transform: Any,
    value: int = 1,
    all_touched: bool = True,
) -> np.ndarray:
    """Rasterize polygon shapes to a numpy array."""
    if not shapes:
        print("No shapes to rasterize")
        return np.zeros((height, width), dtype=np.uint8)

    mask = rasterio.features.rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=all_touched,
        default_value=value,
        dtype=np.uint8,
    )

    pixel_count = np.sum(mask > 0)
    print(f"Rasterized {pixel_count} pixels ({100.0 * pixel_count / mask.size:.2f}%)")

    return mask


def get_osm_cache_dir() -> Path:
    """Get the OSM cache directory path."""
    return Path("data/osm_cache")


def clear_osm_cache() -> None:
    """Clear all cached OSM data."""
    cache_dir = get_osm_cache_dir()
    if cache_dir.exists():
        import shutil

        shutil.rmtree(cache_dir)
        print(f"Cleared OSM cache directory: {cache_dir}")


def get_osm_cache_path(
    bbox: BBox, network_type: str, custom_filter: str | None
) -> Path:
    """Generate a cache file path for OSM data based on parameters."""
    # Create a unique hash for the parameters
    params_str = f"{bbox}_{network_type}_{custom_filter}"
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:12]

    # Create cache directory
    cache_dir = get_osm_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with bbox info for readability
    min_lon, min_lat, max_lon, max_lat = bbox
    # Use .gpkg extension for GeoPackage
    filename = f"osm_{min_lon:.2f}_{min_lat:.2f}_{max_lon:.2f}_{max_lat:.2f}_{params_hash}.gpkg"

    return cache_dir / filename


def fetch_road_network(
    bbox: BBox,
    network_type: str = "all",
    custom_filter: str | None = None,
    use_cache: bool = True,
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
        cache_path = get_osm_cache_path(bbox, network_type, custom_filter)

        if cache_path.exists():
            print(f"Loading OSM data from cache: {cache_path.name}")
            try:
                # Load from GeoPackage
                edges_gdf = gpd.read_file(cache_path, layer="edges")

                # Convert back to NetworkX graph
                # Create empty graph
                graph = nx.MultiDiGraph()

                # Add edges from geodataframe
                for idx, row in edges_gdf.iterrows():
                    # Extract nodes from geometry
                    coords = list(row.geometry.coords)
                    if len(coords) >= 2:
                        # Get first and last coordinates
                        u_coord = coords[0]  # (lon, lat) tuple
                        v_coord = coords[-1]  # (lon, lat) tuple

                        # Create node IDs (use coordinate tuples as IDs)
                        u = u_coord
                        v = v_coord

                        # Add nodes with coordinates
                        graph.add_node(u, x=u_coord[0], y=u_coord[1])
                        graph.add_node(v, x=v_coord[0], y=v_coord[1])

                        # Add edge with attributes
                        edge_attrs = row.to_dict()
                        edge_attrs.pop(
                            "geometry", None
                        )  # Remove geometry from attributes
                        graph.add_edge(u, v, **edge_attrs)

                print(
                    f"Loaded {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges from cache"
                )
                return graph
            except Exception as e:
                print(f"Failed to load cache: {e}. Downloading fresh data...")

    # Download fresh data
    print("Downloading OSM data from server...")
    graph = download_osm_roads(max_lat, min_lat, max_lon, min_lon, custom_filter)

    # Save to cache
    if use_cache and graph.number_of_edges() > 0:
        cache_path = get_osm_cache_path(bbox, network_type, custom_filter)
        try:
            # Convert graph to GeoDataFrame
            # Note: ox.graph_to_gdfs returns just the edges_gdf when nodes=False
            edges_gdf = ox.graph_to_gdfs(graph, nodes=False, edges=True)

            # Save to GeoPackage - no size limitations!
            edges_gdf.to_file(cache_path, layer="edges", driver="GPKG")
            print(f"Saved OSM data to cache: {cache_path.name}")
            print(f"  Cached {len(edges_gdf):,} road segments")
        except Exception as e:
            print(f"Failed to save cache: {e}")
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

    wgs84_to_utm = pyproj.Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg_code}", always_xy=True
    )
    utm_to_wgs84 = pyproj.Transformer.from_crs(
        f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True
    )

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

    if src_crs == dst_crs:
        return rasterize_shapes(
            shapes, shape[0], shape[1], reference_transform, all_touched=all_touched
        )

    reprojected_shapes = []
    for shape_dict in shapes:
        if isinstance(shape_dict, dict) and "geometry" in shape_dict:
            geom = shape_dict["geometry"]
            value = shape_dict.get("value", 1)
        else:
            geom = shape_dict
            value = 1

        geom_transformed = transform_geom(src_crs, dst_crs, geom)
        reprojected_shapes.append({"geometry": geom_transformed, "value": value})

    height, width = shape
    return rasterio.features.rasterize(
        [(s["geometry"], s["value"]) for s in reprojected_shapes],
        out_shape=(height, width),
        transform=reference_transform,
        fill=0,
        all_touched=all_touched,
        dtype=np.uint8,
    )


def buffer_and_rasterize_roads(
    graph: Any,  # NetworkX graph
    bbox: BBox,
    transform: Transform,
    crs: str | CRS,
    shape: tuple[int, int],
    buffer_meters: float = 30.0,
    hierarchical: bool = True,
) -> ArrayLike:
    """Buffer and rasterize an existing road network graph."""
    _, edges_gdf = ox.graph_to_gdfs(graph)

    if len(edges_gdf) == 0:
        return np.zeros(shape, dtype=np.uint8)

    lon_center = (bbox[0] + bbox[2]) / 2
    lat_center = (bbox[1] + bbox[3]) / 2

    buffered_shapes = []

    if hierarchical:
        road_hierarchy = {
            "primary": 1,
            "primary_link": 1,
            "secondary": 1,
            "secondary_link": 1,
            "tertiary": 1,
            "tertiary_link": 1,
            "motorway": 2,
            "motorway_link": 2,
            "residential": 3,
            "trunk": 4,
            "trunk_link": 4,
            "service": 5,
            "unclassified": 5,
            "road": 5,
            "busway": 5,
        }

        for _, edge in edges_gdf.iterrows():
            highway_type = edge["highway"] if "highway" in edge else "unclassified"
            if isinstance(highway_type, list):
                highway_type = highway_type[0]

            value = road_hierarchy.get(str(highway_type), 6)
            geom_dict = geom_to_dict(edge.geometry)

            buffered_geom = buffer_roads_in_utm(
                geometry=geom_dict,
                buffer_meters=buffer_meters,
                lon_center=lon_center,
                lat_center=lat_center,
            )
            buffered_shapes.append({"geometry": buffered_geom, "value": value})
    else:
        for _, edge in edges_gdf.iterrows():
            geom_dict = geom_to_dict(edge.geometry)
            buffered_geom = buffer_roads_in_utm(
                geometry=geom_dict,
                buffer_meters=buffer_meters,
                lon_center=lon_center,
                lat_center=lat_center,
            )
            buffered_shapes.append({"geometry": buffered_geom, "value": 1})

    print(f"Buffered {len(buffered_shapes)} road segments")

    return rasterize_road_shapes_with_crs(
        shapes=buffered_shapes,
        reference_transform=transform,
        shape=shape,
        src_crs="EPSG:4326",
        dst_crs=crs,
        all_touched=False,
    )
