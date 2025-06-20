"""Pure functions for OSM road data processing - testable components."""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import osmnx as ox
import networkx as nx
import shapely.geometry
import rasterio.features
from rasterio.transform import Affine


def download_osm_roads(
    north: float, south: float, east: float, west: float, 
    road_filter: Optional[str] = None
) -> nx.MultiDiGraph:
    """
    Download OSM road network for a bounding box.
    
    Args:
        north: Northern latitude boundary
        south: Southern latitude boundary  
        east: Eastern longitude boundary
        west: Western longitude boundary
        road_filter: Optional OSM filter string for road types
        
    Returns:
        NetworkX graph of roads
        
    Raises:
        Exception: If download fails
    """
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
            custom_filter=road_filter if road_filter else None
        )
        print(f"Downloaded {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    except Exception as e:
        print(f"Error downloading roads: {type(e).__name__}: {e}")
        raise


def graph_to_shapes(
    G: nx.MultiDiGraph, 
    buffer_distance: float = 0.00005
) -> List[Dict[str, Any]]:
    """
    Convert NetworkX road graph to buffered polygon shapes.
    
    Args:
        G: NetworkX graph of roads
        buffer_distance: Buffer distance in degrees
        
    Returns:
        List of GeoJSON-like shape dictionaries
    """
    if G.number_of_edges() == 0:
        return []
    
    # Get edges as GeoDataFrame
    _, edges_gdf = ox.graph_to_gdfs(G)
    
    shapes = []
    for _, edge in edges_gdf.iterrows():
        # Buffer the line geometry
        buffered = edge.geometry.buffer(buffer_distance)
        shapes.append(shapely.geometry.mapping(buffered))
    
    print(f"Created {len(shapes)} buffered road shapes")
    return shapes


def rasterize_shapes(
    shapes: List[Dict[str, Any]], 
    height: int, 
    width: int,
    transform: Affine,
    value: int = 1,
    all_touched: bool = True
) -> np.ndarray:
    """
    Rasterize polygon shapes to a numpy array.
    
    Args:
        shapes: List of GeoJSON-like shape dictionaries
        height: Output raster height in pixels
        width: Output raster width in pixels
        transform: Affine transform for georeferencing
        value: Value to burn into raster for shapes
        all_touched: Whether to rasterize all touched pixels
        
    Returns:
        Numpy array with rasterized shapes
    """
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


def create_multi_type_road_raster(
    north: float, south: float, east: float, west: float,
    height: int, width: int, transform: Affine,
    buffer_distance: float = 0.00005,
    road_filters: Optional[List[Tuple[str, int]]] = None
) -> np.ndarray:
    """
    Create a raster with different road types having different values.
    
    Args:
        north, south, east, west: Bounding box
        height, width: Output raster dimensions
        transform: Affine transform
        buffer_distance: Road buffer in degrees
        road_filters: List of (filter_string, raster_value) tuples
        
    Returns:
        Numpy array with different values for different road types
    """
    if road_filters is None:
        road_filters = [
            ('["highway"~"primary_link|primary|secondary|secondary_link|tertiary|tertiary_link"]', 1),
            ('["highway"~"motorway|motorway_link"]', 2),
            ('["highway"~"residential"]', 3),
            ('["highway"~"trunk|trunk_link"]', 4),
            ('["highway"~"service|unclassified|road|busway"]', 5),
        ]
    
    base = np.zeros((height, width), dtype=np.uint8)
    
    for filter_str, value in road_filters:
        try:
            print(f"\nProcessing road type {value}: {filter_str[:50]}...")
            
            # Download roads for this type
            G = download_osm_roads(north, south, east, west, filter_str)
            
            if G.number_of_edges() == 0:
                print("No roads found for this filter")
                continue
                
            # Convert to shapes
            shapes = graph_to_shapes(G, buffer_distance)
            
            # Rasterize
            mask = rasterize_shapes(shapes, height, width, transform, value)
            
            # Update base (higher values overwrite lower)
            pixels_updated = np.sum((mask > 0) & (mask > base))
            print(f"Updating {pixels_updated} pixels in base layer")
            base = np.maximum(base, mask)
            
        except Exception as e:
            print(f"Error processing filter {filter_str}: {e}")
            continue
    
    # Summary statistics
    total_pixels = np.sum(base > 0)
    print(f"\nFinal road raster: {total_pixels} pixels ({100.0 * total_pixels / base.size:.2f}%)")
    for val in range(1, 6):
        count = np.sum(base == val)
        if count > 0:
            print(f"  Type {val}: {count} pixels")
    
    return base


def bbox_area_km2(north: float, south: float, east: float, west: float) -> float:
    """
    Calculate approximate area of a lat/lon bounding box in km².
    
    Args:
        north, south, east, west: Bounding box coordinates
        
    Returns:
        Approximate area in square kilometers
    """
    from geopy.distance import distance
    
    # Calculate dimensions
    height_km = distance((south, west), (north, west)).km
    width_km = distance((south, west), (south, east)).km
    
    return height_km * width_km