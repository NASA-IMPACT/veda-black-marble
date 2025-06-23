"""OpenStreetMap road network data acquisition.

This module handles fetching road network data from OpenStreetMap using OSMnx,
and converting it to raster format for integration with satellite imagery.
"""

from typing import Any, Literal
import numpy.typing as npt
import numpy as np
import os
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_geom, transform_bounds
from shapely.geometry import shape, mapping
from shapely.ops import transform
import pyproj

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
    road_filters: list[str] | None = None,
    buffer_meters: bool | None = None
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
        buffer_distance: Road buffer width. Units depend on buffer_meters parameter:
                        - If buffer_meters is True: distance in meters
                        - If buffer_meters is False/None: distance in degrees
                        Default: 0.00005 degrees (~5.5m at equator)
        road_filters: Optional list of OSM filter strings for custom road types
        buffer_meters: If True, interpret buffer_distance as meters and use UTM
                      for accurate buffering. If None, auto-detect based on CRS.
        
    Returns:
        Path to created road raster with hierarchical values (1-5)
    """
    # Extract spatial parameters from reference raster
    with rasterio.open(reference_raster) as src:
        height, width = src.height, src.width
        transform = src.transform
        profile = src.profile.copy()
        crs = src.crs
    
    # Auto-detect if we should use meter-based buffering
    if buffer_meters is None:
        # Use meters for projected CRS (UTM, Web Mercator, etc)
        # Use degrees for geographic CRS (WGS84, etc)
        buffer_meters = crs is not None and crs.is_projected
    
    # If using meter-based buffering, delegate to CRS-aware function
    if buffer_meters:
        road_array = create_road_raster_from_transform(
            bbox=bbox,
            transform=transform,
            crs=crs,
            shape=(height, width),
            buffer_meters=buffer_distance,  # Interpret as meters
            road_type_values=None  # Use default mapping
        )
    else:
        # Use traditional degree-based buffering
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
            buffer_distance=buffer_distance  # In degrees
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


def calculate_utm_zone(lon: float, lat: float) -> tuple[int, Literal["north", "south"]]:
    """Calculate the UTM zone and hemisphere for a given coordinate.
    
    Uses the standard UTM zone calculation formula where zones are 6 degrees
    wide, starting from -180 degrees longitude. Zone 1 spans from -180 to -174,
    zone 31 includes 0 degrees (Greenwich), and zone 60 wraps back to -180.
    
    Args:
        lon: Longitude in decimal degrees (-180 to 180)
        lat: Latitude in decimal degrees (-90 to 90)
        
    Returns:
        Tuple of (zone_number, hemisphere) where:
        - zone_number: UTM zone (1-60)
        - hemisphere: 'north' or 'south'
        
    Example:
        >>> calculate_utm_zone(12.453, 41.902)  # Vatican
        (33, 'north')
        >>> calculate_utm_zone(-73.935, 40.730)  # New York
        (18, 'north')
        >>> calculate_utm_zone(151.209, -33.865)  # Sydney
        (56, 'south')
    """
    # Calculate UTM zone from longitude
    # Formula: zone = floor((lon + 180) / 6) + 1
    # This gives us zones 1-60 where zone 1 starts at -180°
    zone = int((lon + 180) / 6) + 1
    
    # Handle edge case at exactly 180° longitude (should be zone 60, not 61)
    if zone > 60:
        zone = 60
    
    # Determine hemisphere based on latitude
    hemisphere: Literal["north", "south"] = "north" if lat >= 0 else "south"
    
    return zone, hemisphere


def buffer_roads_in_utm(
    geometry: dict[str, Any],
    buffer_meters: float,
    lon_center: float,
    lat_center: float
) -> dict[str, Any]:
    """Buffer a road geometry in meters using UTM projection.
    
    This function ensures accurate meter-based buffering by:
    1. Converting the geometry from WGS84 to the appropriate UTM zone
    2. Applying the buffer operation in UTM (where units are meters)
    3. Converting the buffered geometry back to WGS84
    
    This approach guarantees consistent road widths regardless of latitude,
    unlike buffering in geographic coordinates where the same angular distance
    represents different ground distances at different latitudes.
    
    Args:
        geometry: GeoJSON-like geometry dict in WGS84 (EPSG:4326)
                 e.g., {'type': 'LineString', 'coordinates': [[lon1, lat1], [lon2, lat2]]}
        buffer_meters: Buffer distance in meters to apply
        lon_center: Center longitude for UTM zone calculation
        lat_center: Center latitude for UTM zone calculation
        
    Returns:
        Buffered geometry as GeoJSON-like dict in WGS84
        
    Example:
        >>> road = {'type': 'LineString', 'coordinates': [[12.453, 41.902], [12.454, 41.903]]}
        >>> buffered = buffer_roads_in_utm(road, 30.0, 12.453, 41.902)
        >>> buffered['type']
        'Polygon'
    """
    # Calculate appropriate UTM zone for the geometry center
    zone, hemisphere = calculate_utm_zone(lon_center, lat_center)
    
    # Construct EPSG code for UTM zone
    # Northern hemisphere: 32600 + zone (e.g., 32633 for zone 33N)
    # Southern hemisphere: 32700 + zone (e.g., 32756 for zone 56S)
    epsg_code = 32600 + zone if hemisphere == "north" else 32700 + zone
    
    # Create coordinate transformation objects
    # WGS84 (lat/lon) to UTM projection
    wgs84_to_utm = pyproj.Transformer.from_crs(
        "EPSG:4326",  # WGS84
        f"EPSG:{epsg_code}",  # UTM zone
        always_xy=True  # Ensure (x,y) order instead of (y,x)
    )
    
    # UTM back to WGS84 for output
    utm_to_wgs84 = pyproj.Transformer.from_crs(
        f"EPSG:{epsg_code}",  # UTM zone
        "EPSG:4326",  # WGS84
        always_xy=True
    )
    
    # Convert GeoJSON dict to Shapely geometry
    geom = shape(geometry)
    
    # Project geometry to UTM using shapely.ops.transform
    # This applies the coordinate transformation to all vertices
    geom_utm = transform(wgs84_to_utm.transform, geom)
    
    # Apply buffer in UTM projection (meters)
    # Buffer operation creates a polygon around the input geometry
    buffered_utm = geom_utm.buffer(buffer_meters)
    
    # Project buffered geometry back to WGS84
    buffered_wgs84 = transform(utm_to_wgs84.transform, buffered_utm)
    
    # Convert back to GeoJSON-like dict
    return mapping(buffered_wgs84)


def rasterize_road_shapes_with_crs(
    shapes: list[dict[str, Any]],
    reference_transform: Transform,
    shape: tuple[int, int],
    src_crs: str | CRS,
    dst_crs: str | CRS,
    all_touched: bool = True
) -> ArrayLike:
    """Rasterize road shapes with coordinate system transformation.
    
    This enhanced version of rasterize_road_shapes handles reprojection
    from source CRS to destination CRS before rasterization. This is
    essential when road geometries are in a different CRS than the
    target raster (e.g., WGS84 roads to Web Mercator raster).
    
    Args:
        shapes: List of GeoJSON-like shape dictionaries in src_crs
               Each shape should have 'geometry' key with coordinates
        reference_transform: Affine transform for output raster in dst_crs
        shape: Output array shape (height, width)
        src_crs: Source CRS of input shapes (e.g., "EPSG:4326" for WGS84)
        dst_crs: Destination CRS for rasterization (e.g., "EPSG:3857")
        all_touched: If True, all pixels touched by geometry are rasterized
        
    Returns:
        Rasterized road array with shape (height, width)
        
    Example:
        >>> shapes_wgs84 = [{'geometry': {'type': 'Polygon', ...}, 'value': 1}]
        >>> raster = rasterize_road_shapes_with_crs(
        ...     shapes_wgs84, transform_3857, (100, 100),
        ...     "EPSG:4326", "EPSG:3857"
        ... )
    """
    # Convert CRS strings to CRS objects if needed
    if isinstance(src_crs, str):
        src_crs = CRS.from_string(src_crs)
    if isinstance(dst_crs, str):
        dst_crs = CRS.from_string(dst_crs)
    
    # If CRS are the same, skip reprojection for efficiency
    if src_crs == dst_crs:
        return rasterize_road_shapes(
            shapes, reference_transform, shape, all_touched
        )
    
    # Reproject each shape from source to destination CRS
    reprojected_shapes = []
    for shape_dict in shapes:
        # Extract geometry from shape dictionary
        if isinstance(shape_dict, dict) and 'geometry' in shape_dict:
            geom = shape_dict['geometry']
            value = shape_dict.get('value', 1)  # Default value if not specified
        else:
            geom = shape_dict
            value = 1
        
        # Transform geometry to destination CRS
        # transform_geom handles the CRS transformation for any geometry type
        geom_transformed = transform_geom(src_crs, dst_crs, geom)
        
        # Create new shape dict with transformed geometry
        reprojected_shapes.append({
            'geometry': geom_transformed,
            'value': value
        })
    
    # Now rasterize in the destination CRS
    height, width = shape
    
    # Use rasterio.features.rasterize directly for more control
    import rasterio.features
    raster = rasterio.features.rasterize(
        [(s['geometry'], s['value']) for s in reprojected_shapes],
        out_shape=(height, width),
        transform=reference_transform,
        fill=0,
        all_touched=all_touched,
        dtype=np.uint8
    )
    
    return raster


def create_road_raster_from_transform(
    bbox: BBox,
    transform: Transform,
    crs: str | CRS,
    shape: tuple[int, int],
    buffer_meters: float = 30.0,
    road_type_values: dict[str, int] | None = None
) -> ArrayLike:
    """Create road raster from transform and CRS without reference file.
    
    This pipeline-friendly function creates a road raster by:
    1. Downloading roads in WGS84 from OpenStreetMap
    2. Calculating the appropriate UTM zone for the area
    3. Buffering roads by the specified distance in meters (via UTM)
    4. Reprojecting to the target CRS
    5. Rasterizing to the provided grid
    
    This approach ensures consistent road widths across all latitudes
    and makes the function suitable for direct use in processing pipelines
    where transform and CRS are already known.
    
    Args:
        bbox: Bounding box in WGS84 (min_lon, min_lat, max_lon, max_lat)
        transform: Affine transform for output grid
        crs: Target CRS for output raster (e.g., "EPSG:3857")
        shape: Output shape (height, width)
        buffer_meters: Buffer distance in meters (applied in UTM)
        road_type_values: Optional dict mapping OSM highway types to raster values
                         Default: {'motorway': 1, 'primary': 2, 'secondary': 3, 
                                  'tertiary': 4, 'residential': 5, 'service': 6}
                         
    Returns:
        Road raster array with specified shape and road type values
        
    Example:
        >>> # Create road raster for Web Mercator grid
        >>> from rasterio.transform import from_origin
        >>> transform = from_origin(825000, 5430000, 30, 30)
        >>> roads = create_road_raster_from_transform(
        ...     bbox=(7.42, 43.73, 7.425, 43.735),  # Monaco
        ...     transform=transform,
        ...     crs="EPSG:3857",
        ...     shape=(200, 200),
        ...     buffer_meters=30.0
        ... )
    """
    # Set default road type mapping if not provided
    if road_type_values is None:
        road_type_values = {
            'motorway': 1,
            'motorway_link': 1,
            'primary': 2,
            'primary_link': 2,
            'secondary': 3,
            'secondary_link': 3,
            'tertiary': 4,
            'tertiary_link': 4,
            'residential': 5,
            'service': 6,
            'unclassified': 6,
            'road': 6
        }
    
    # Extract bbox components and calculate center
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_center = (min_lon + max_lon) / 2
    lat_center = (min_lat + max_lat) / 2
    
    # Convert to north/south/east/west for OSM functions
    north, south, east, west = max_lat, min_lat, max_lon, min_lon
    
    # Initialize output raster
    height, width = shape
    road_raster = np.zeros((height, width), dtype=np.uint8)
    
    # Process each road type separately to maintain hierarchy
    for highway_type, raster_value in sorted(road_type_values.items(), 
                                             key=lambda x: x[1], reverse=True):
        # Create OSM filter for this road type
        road_filter = f'["highway"~"{highway_type}"]'
        
        try:
            # Download roads of this specific type
            print(f"Downloading {highway_type} roads...")
            graph = download_osm_roads(north, south, east, west, road_filter)
            
            if graph.number_of_edges() == 0:
                print(f"No {highway_type} roads found")
                continue
            
            # Convert graph to shapes (lines)
            # Use very small buffer here since we'll buffer properly in UTM
            line_shapes = graph_to_shapes(graph, buffer_distance=0.0)
            
            # Buffer each shape in UTM for accurate meter-based widths
            buffered_shapes = []
            for shape_dict in line_shapes:
                # Buffer the geometry in UTM
                buffered_geom = buffer_roads_in_utm(
                    geometry=shape_dict,
                    buffer_meters=buffer_meters,
                    lon_center=lon_center,
                    lat_center=lat_center
                )
                
                # Create shape dict with buffered geometry and appropriate value
                buffered_shapes.append({
                    'geometry': buffered_geom,
                    'value': raster_value
                })
            
            print(f"Created {len(buffered_shapes)} buffered {highway_type} shapes")
            
            # Rasterize with CRS transformation from WGS84 to target CRS
            road_type_raster = rasterize_road_shapes_with_crs(
                shapes=buffered_shapes,
                reference_transform=transform,
                shape=shape,
                src_crs="EPSG:4326",  # OSM data is in WGS84
                dst_crs=crs,  # Target CRS
                all_touched=True
            )
            
            # Update road raster - higher values overwrite lower values
            # This maintains the road hierarchy
            mask = road_type_raster > 0
            road_raster[mask] = road_type_raster[mask]
            
            pixels_added = np.sum(mask)
            print(f"Rasterized {pixels_added} pixels for {highway_type}")
            
        except Exception as e:
            print(f"Error processing {highway_type}: {e}")
            continue
    
    # Summary statistics
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
    """Buffer and rasterize an existing road network graph with CRS support.
    
    This function takes an already-downloaded road network and:
    1. Extracts road geometries with hierarchical classification
    2. Buffers roads by specified distance in meters (via UTM)
    3. Reprojects to target CRS
    4. Rasterizes to the provided grid
    
    Args:
        graph: NetworkX graph of roads (from fetch_road_network)
        bbox: Bounding box in WGS84 (min_lon, min_lat, max_lon, max_lat)
        transform: Affine transform for output grid
        crs: Target CRS for output raster
        shape: Output shape (height, width)
        buffer_meters: Buffer distance in meters (applied in UTM)
        hierarchical: If True, use hierarchical road classification (1-6)
                     If False, all roads get value 1
        
    Returns:
        Road raster array with specified shape
        
    Example:
        >>> graph = fetch_road_network(bbox)
        >>> roads = buffer_and_rasterize_roads(
        ...     graph, bbox, transform, "EPSG:3857", (200, 200), 30.0
        ... )
    """
    import osmnx as ox
    from shapely.geometry import mapping as geom_to_dict
    
    # Get edges as GeoDataFrame for road type classification
    _, edges_gdf = ox.graph_to_gdfs(graph)
    
    if len(edges_gdf) == 0:
        return np.zeros(shape, dtype=np.uint8)
    
    # Calculate center for UTM zone determination
    lon_center = (bbox[0] + bbox[2]) / 2
    lat_center = (bbox[1] + bbox[3]) / 2
    
    # Process roads by type for hierarchical classification
    buffered_shapes = []
    
    if hierarchical:
        # Define road type hierarchy
        road_hierarchy = {
            'motorway': 1, 'motorway_link': 1, 'trunk': 1, 'trunk_link': 1,
            'primary': 2, 'primary_link': 2,
            'secondary': 3, 'secondary_link': 3,
            'tertiary': 4, 'tertiary_link': 4,
            'residential': 5,
            'service': 6, 'unclassified': 6, 'road': 6
        }
        
        # Group edges by highway type
        for _, edge in edges_gdf.iterrows():
            # Access highway attribute safely
            highway_type = edge['highway'] if 'highway' in edge else 'unclassified'
            
            # Handle list of highway types (some edges have multiple)
            if isinstance(highway_type, list):
                highway_type = highway_type[0]  # Use first type
                
            # Get value for this road type
            value = road_hierarchy.get(highway_type, 6)  # Default to service/other
            
            # Convert geometry to GeoJSON dict
            geom_dict = geom_to_dict(edge.geometry)
            
            # Buffer in UTM for accurate meters
            buffered_geom = buffer_roads_in_utm(
                geometry=geom_dict,
                buffer_meters=buffer_meters,
                lon_center=lon_center,
                lat_center=lat_center
            )
            
            buffered_shapes.append({
                'geometry': buffered_geom,
                'value': value
            })
    else:
        # Simple mode - all roads get value 1
        for _, edge in edges_gdf.iterrows():
            geom_dict = geom_to_dict(edge.geometry)
            
            buffered_geom = buffer_roads_in_utm(
                geometry=geom_dict,
                buffer_meters=buffer_meters,
                lon_center=lon_center,
                lat_center=lat_center
            )
            
            buffered_shapes.append({
                'geometry': buffered_geom,
                'value': 1
            })
    
    print(f"Buffered {len(buffered_shapes)} road segments")
    
    # Rasterize with CRS transformation
    return rasterize_road_shapes_with_crs(
        shapes=buffered_shapes,
        reference_transform=transform,
        shape=shape,
        src_crs="EPSG:4326",  # OSM data is in WGS84
        dst_crs=crs,
        all_touched=True
    )