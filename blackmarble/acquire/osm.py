"""OpenStreetMap road network data acquisition.

This module handles fetching road network data from OpenStreetMap using OSMnx,
and converting it to raster format for integration with satellite imagery.
"""

from typing import Any
import numpy.typing as npt
import numpy as np
import os
import rasterio

# Import pure functional implementations
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
    
    Generates a hierarchical road classification raster where different road types
    receive different pixel values (1-5). This is essential for the NDUI calculation
    which uses road proximity to enhance urban light detection accuracy.
    
    The hierarchical classification prioritizes major transportation corridors
    (motorways, primary roads) over local streets, enabling the pipeline to
    apply appropriate enhancement factors based on road importance.
    
    Args:
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat) for road data
        reference_raster: Path to reference raster for spatial alignment
        output_path: Path for output hierarchical road raster
        buffer_distance: Road buffer width in decimal degrees (default ~5.5m at equator)
        road_filters: Optional list of OSM filter strings for custom road types
        
    Returns:
        Path to created road raster with hierarchical values (1-5)
    """
    # Extract spatial parameters from reference raster
    with rasterio.open(reference_raster) as src:
        height, width = src.height, src.width
        transform = src.transform
        profile = src.profile.copy()
    
    # Convert bbox to standard north/south/east/west for OSM functions
    min_lon, min_lat, max_lon, max_lat = bbox
    north, south, east, west = max_lat, min_lat, max_lon, min_lon
    
    # Generate hierarchical road raster using pure functional approach
    road_array = create_multi_type_road_raster(
        north=north,
        south=south,
        east=east,
        west=west,
        height=height,
        width=width,
        transform=transform,
        buffer_distance=buffer_distance
    )
    
    # Configure output profile for road classification raster
    profile.update({
        'count': 1,
        'dtype': 'uint8',
        'compress': 'deflate',
        'nodata': 0
    })
    
    # Write hierarchical road raster to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(road_array, 1)
    
    return output_path


def create_simple_road_mask(
    bbox: BBox,
    reference_raster: str,
    output_path: str,
    buffer_distance: float = 0.00005,
    road_value: int = 1
) -> str:
    """Create simple binary road mask.
    
    Generates a binary road mask where all road types receive the same pixel value.
    This simplified representation is useful for basic road proximity analysis
    without the complexity of hierarchical road classification.
    
    Unlike the hierarchical version, this mask treats all roads equally,
    making it suitable for applications where road type distinction is not
    required, such as simple distance-to-road calculations.
    
    Args:
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat) for road data
        reference_raster: Path to reference raster for spatial alignment
        output_path: Path for output binary road mask
        buffer_distance: Road buffer width in decimal degrees (default ~5.5m at equator)
        road_value: Pixel value for all road types (default: 1)
        
    Returns:
        Path to created binary road mask
    """
    # Extract spatial parameters from reference raster
    with rasterio.open(reference_raster) as src:
        height, width = src.height, src.width
        transform = src.transform
        profile = src.profile.copy()
    
    # Convert bbox to standard north/south/east/west for OSM functions  
    min_lon, min_lat, max_lon, max_lat = bbox
    north, south, east, west = max_lat, min_lat, max_lon, min_lon
    
    # Download all road types without filtering
    road_graph = download_osm_roads(
        north=north,
        south=south, 
        east=east,
        west=west,
        road_filter=None  # Download all available roads
    )
    
    # Convert road network to buffered polygon shapes
    road_shapes = graph_to_shapes(
        G=road_graph,
        buffer_distance=buffer_distance
    )
    
    # Rasterize all roads with uniform value
    road_array = rasterize_shapes(
        shapes=road_shapes,
        height=height,
        width=width,
        transform=transform,
        value=road_value,
        all_touched=True
    )
    
    # Configure output profile for binary road mask
    profile.update({
        'count': 1,
        'dtype': 'uint8',
        'compress': 'deflate',
        'nodata': 0
    })
    
    # Write binary road mask to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(road_array, 1)
    
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