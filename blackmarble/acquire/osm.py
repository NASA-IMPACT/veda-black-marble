"""OpenStreetMap road network data acquisition.

This module handles fetching road network data from OpenStreetMap using OSMnx,
and converting it to raster format for integration with satellite imagery.
"""

from typing import Any
import numpy.typing as npt
import numpy as np
import os

# Import existing implementations
from blackmarble.osm import OSMUtil
from blackmarble.osm_functions import (
    download_osm_roads,
    graph_to_shapes,
    rasterize_shapes,
    create_multi_type_road_raster
)

# Type aliases
BBox = tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
ArrayLike = npt.NDArray[Any]
Transform = Any  # rasterio.transform.Affine


def fetch_road_network(
    bbox: BBox,
    network_type: str = "all",
    custom_filter: str | None = None
) -> Any:  # Returns networkx graph
    """Fetch road network from OpenStreetMap.
    
    Args:
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
        network_type: OSMnx network type ('all', 'drive', 'walk', etc.)
        custom_filter: Optional custom OSM filter string
        
    Returns:
        NetworkX graph of road network
    """
    # Extract bbox components
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # Calculate north, south, east, west
    north = max_lat
    south = min_lat
    east = max_lon
    west = min_lon
    
    # Delegate to existing implementation
    # Note: download_osm_roads doesn't use network_type parameter,
    # it always uses 'all' internally
    return download_osm_roads(north, south, east, west, custom_filter)


def create_road_raster(
    bbox: BBox,
    reference_raster: str,
    output_path: str,
    buffer_distance: float = 0.00005,
    road_filters: list[str] | None = None
) -> str:
    """Create rasterized road network with hierarchical values.
    
    Args:
        bbox: Bounding box for road data
        reference_raster: Path to reference raster for resolution/alignment
        output_path: Path for output road raster
        buffer_distance: Buffer distance for roads in degrees
        road_filters: List of OSM filters for different road types
        
    Returns:
        Path to created road raster
    """
    # Extract bbox components
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # Delegate to existing OSMUtil implementation
    util = OSMUtil()
    util.create_road_raster(
        min_lat, min_lon,  # lat1, lon1
        max_lat, max_lon,  # lat2, lon2
        reference_raster,
        output_path,
        output_dir=os.path.dirname(output_path),
        path_thickness=buffer_distance
    )
    
    return output_path


def create_simple_road_mask(
    bbox: BBox,
    reference_raster: str,
    output_path: str,
    buffer_distance: float = 0.00005,
    road_value: int = 1
) -> str:
    """Create simple binary road mask.
    
    Args:
        bbox: Bounding box for road data
        reference_raster: Path to reference raster for resolution/alignment
        output_path: Path for output road mask
        buffer_distance: Buffer distance for roads in degrees
        road_value: Value to use for road pixels
        
    Returns:
        Path to created road mask
    """
    # Extract bbox components
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # Import os for path operations
    import os
    
    # Delegate to existing OSMUtil implementation
    # Note: The flattened version uses a fixed value of 5 internally,
    # but we can't control that from here
    util = OSMUtil()
    util.create_road_raster_flattened(
        min_lat, min_lon,  # lat1, lon1
        max_lat, max_lon,  # lat2, lon2
        reference_raster,
        output_path,
        output_dir=os.path.dirname(output_path),
        path_thickness=buffer_distance
    )
    
    return output_path


def get_hierarchical_road_filters() -> dict[int, str]:
    """Get default OSM filters for hierarchical road classification.
    
    Returns:
        Dictionary mapping road class values to OSM filter strings:
        1: Motorways and motorway links
        2: Primary roads
        3: Secondary roads
        4: Tertiary roads
        5: Residential roads
        6: Service roads
    """
    # Return the default filters used in OSMUtil.create_road_raster()
    return {
        1: '["highway"~"primary_link|primary|secondary|secondary_link|tertiary|tertiary_link"]',
        2: '["highway"~"motorway|motorway_link"]',
        3: '["highway"~"residential"]',
        4: '["highway"~"trunk|trunk_link"]',
        5: '["highway"~"service|unclassified|road|busway"]',
    }


def roads_graph_to_shapes(
    graph: Any,  # NetworkX graph
    buffer_distance: float = 0.00005
) -> list[dict[str, Any]]:
    """Convert road network graph to buffered polygon shapes.
    
    Args:
        graph: NetworkX graph of roads
        buffer_distance: Buffer distance in degrees
        
    Returns:
        List of shapely geometry dictionaries ready for rasterization
    """
    # Delegate to existing implementation
    return graph_to_shapes(graph, buffer_distance)


def rasterize_road_shapes(
    shapes: list[dict[str, Any]],
    reference_transform: Transform,
    shape: tuple[int, int],
    all_touched: bool = False
) -> ArrayLike:
    """Rasterize road shapes to array.
    
    Args:
        shapes: List of geometry dictionaries with values
        reference_transform: Affine transform for output raster
        shape: Output array shape (height, width)
        all_touched: Whether to rasterize all touched pixels
        
    Returns:
        Rasterized road array
    """
    # Extract height and width from shape tuple
    height, width = shape
    
    # Delegate to existing implementation
    # Default value is 1 if not specified in shapes
    return rasterize_shapes(
        shapes,
        height,
        width,
        reference_transform,
        value=1,  # Default value
        all_touched=all_touched
    )