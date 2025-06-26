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
from rasterio.warp import transform_geom, transform_bounds
from shapely.geometry import shape, mapping
from shapely.geometry import mapping as geom_to_dict
from shapely.ops import transform
import pyproj
import shapely
import osmnx as ox
import networkx as nx

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


def graph_to_shapes(G: nx.MultiDiGraph, buffer_distance: float = 0.00005) -> list[dict[str, Any]]:
    """Convert road graph to buffered shapes."""
    if G.number_of_edges() == 0:
        return []
    
    _, edges_gdf = ox.graph_to_gdfs(G)
    shapes = []
    for _, edge in edges_gdf.iterrows():
        geom = edge.geometry.buffer(buffer_distance) if buffer_distance > 0 else edge.geometry
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


def fetch_road_network(
    bbox: BBox,
    network_type: str = "all",
    custom_filter: str | None = None
) -> Any:  # Returns networkx graph
    """Fetch road network from OpenStreetMap."""
    min_lon, min_lat, max_lon, max_lat = bbox
    return download_osm_roads(max_lat, min_lat, max_lon, min_lon, custom_filter)




def get_hierarchical_road_filters() -> dict[int, str]:
    """Get default OSM filters for hierarchical road classification."""
    return {
        1: '["highway"~"primary_link|primary|secondary|secondary_link|tertiary|tertiary_link"]',
        2: '["highway"~"motorway|motorway_link"]',
        3: '["highway"~"residential"]',
        4: '["highway"~"trunk|trunk_link"]',
        5: '["highway"~"service|unclassified|road|busway"]',
    }






def calculate_utm_zone(lon: float, lat: float) -> tuple[int, Literal["north", "south"]]:
    """Calculate UTM zone and hemisphere for coordinates."""
    zone = min(60, int((lon + 180) / 6) + 1)
    hemisphere: Literal["north", "south"] = "north" if lat >= 0 else "south"
    return zone, hemisphere


def buffer_roads_in_utm(
    geometry: dict[str, Any],
    buffer_meters: float,
    lon_center: float,
    lat_center: float
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
    all_touched: bool = False
) -> ArrayLike:
    """Rasterize road shapes with coordinate system transformation."""
    if isinstance(src_crs, str):
        src_crs = CRS.from_string(src_crs)
    if isinstance(dst_crs, str):
        dst_crs = CRS.from_string(dst_crs)
    
    if src_crs == dst_crs:
        return rasterize_shapes(shapes, shape[0], shape[1], reference_transform, all_touched=all_touched)
    
    reprojected_shapes = []
    for shape_dict in shapes:
        if isinstance(shape_dict, dict) and 'geometry' in shape_dict:
            geom = shape_dict['geometry']
            value = shape_dict.get('value', 1)
        else:
            geom = shape_dict
            value = 1
        
        geom_transformed = transform_geom(src_crs, dst_crs, geom)
        reprojected_shapes.append({'geometry': geom_transformed, 'value': value})
    
    height, width = shape
    return rasterio.features.rasterize(
        [(s['geometry'], s['value']) for s in reprojected_shapes],
        out_shape=(height, width),
        transform=reference_transform,
        fill=0,
        all_touched=all_touched,
        dtype=np.uint8
    )


def create_road_raster_from_transform(
    bbox: BBox,
    transform: Transform,
    crs: str | CRS,
    shape: tuple[int, int],
    buffer_meters: float = 30.0,
    road_type_values: dict[str, int] | None = None
) -> ArrayLike:
    """Create road raster from transform and CRS."""
    if road_type_values is None:
        road_type_values = {
            'motorway': 1, 'motorway_link': 1,
            'primary': 2, 'primary_link': 2,
            'secondary': 3, 'secondary_link': 3,
            'tertiary': 4, 'tertiary_link': 4,
            'residential': 5,
            'service': 6, 'unclassified': 6, 'road': 6
        }
    
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_center = (min_lon + max_lon) / 2
    lat_center = (min_lat + max_lat) / 2
    north, south, east, west = max_lat, min_lat, max_lon, min_lon
    
    height, width = shape
    road_raster = np.zeros((height, width), dtype=np.uint8)
    
    for highway_type, raster_value in sorted(road_type_values.items(), key=lambda x: x[1], reverse=True):
        road_filter = f'["highway"~"{highway_type}"]'
        
        try:
            print(f"Downloading {highway_type} roads...")
            graph = download_osm_roads(north, south, east, west, road_filter)
            
            if graph.number_of_edges() == 0:
                print(f"No {highway_type} roads found")
                continue
            
            line_shapes = graph_to_shapes(graph, buffer_distance=0.0)
            
            buffered_shapes = []
            for shape_dict in line_shapes:
                buffered_geom = buffer_roads_in_utm(
                    geometry=shape_dict,
                    buffer_meters=buffer_meters,
                    lon_center=lon_center,
                    lat_center=lat_center
                )
                buffered_shapes.append({'geometry': buffered_geom, 'value': raster_value})
            
            print(f"Created {len(buffered_shapes)} buffered {highway_type} shapes")
            
            road_type_raster = rasterize_road_shapes_with_crs(
                shapes=buffered_shapes,
                reference_transform=transform,
                shape=shape,
                src_crs="EPSG:4326",
                dst_crs=crs,
                all_touched=True
            )
            
            mask = road_type_raster > 0
            road_raster[mask] = road_type_raster[mask]
            
            pixels_added = np.sum(mask)
            print(f"Rasterized {pixels_added} pixels for {highway_type}")
            
        except Exception as e:
            print(f"Error processing {highway_type}: {e}")
            continue
    
    total_pixels = np.sum(road_raster > 0)
    coverage = (total_pixels / road_raster.size) * 100
    print(f"\nTotal road coverage: {total_pixels} pixels ({coverage:.1f}%)")
    
    return road_raster


def buffer_and_rasterize_roads(
    graph: Any,  # NetworkX graph
    bbox: BBox,
    transform: Transform,
    crs: str | CRS,
    shape: tuple[int, int],
    buffer_meters: float = 30.0,
    hierarchical: bool = True
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
            'primary': 1, 'primary_link': 1, 'secondary': 1, 'secondary_link': 1,
            'tertiary': 1, 'tertiary_link': 1,
            'motorway': 2, 'motorway_link': 2,
            'residential': 3,
            'trunk': 4, 'trunk_link': 4,
            'service': 5, 'unclassified': 5, 'road': 5, 'busway': 5
        }
        
        for _, edge in edges_gdf.iterrows():
            highway_type = edge['highway'] if 'highway' in edge else 'unclassified'
            if isinstance(highway_type, list):
                highway_type = highway_type[0]
            
            value = road_hierarchy.get(str(highway_type), 6)
            geom_dict = geom_to_dict(edge.geometry)
            
            buffered_geom = buffer_roads_in_utm(
                geometry=geom_dict,
                buffer_meters=buffer_meters,
                lon_center=lon_center,
                lat_center=lat_center
            )
            buffered_shapes.append({'geometry': buffered_geom, 'value': value})
    else:
        for _, edge in edges_gdf.iterrows():
            geom_dict = geom_to_dict(edge.geometry)
            buffered_geom = buffer_roads_in_utm(
                geometry=geom_dict,
                buffer_meters=buffer_meters,
                lon_center=lon_center,
                lat_center=lat_center
            )
            buffered_shapes.append({'geometry': buffered_geom, 'value': 1})
    
    print(f"Buffered {len(buffered_shapes)} road segments")
    
    return rasterize_road_shapes_with_crs(
        shapes=buffered_shapes,
        reference_transform=transform,
        shape=shape,
        src_crs="EPSG:4326",
        dst_crs=crs,
        all_touched=False
    )