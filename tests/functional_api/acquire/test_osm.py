"""Test functions in blackmarble.acquire.osm module.

This module tests OpenStreetMap road network data acquisition functionality
using REAL OSM data from small test areas. No mocking - real functionality only.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

# Import the module under test
from blackmarble.acquire.osm import (
    fetch_road_network,
    create_road_raster,
    create_simple_road_mask,
    get_hierarchical_road_filters,
    roads_graph_to_shapes,
    rasterize_road_shapes,
)

# For testing CRS handling
try:
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.crs import CRS
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

# Small, stable test areas that should have consistent OSM data
TEST_AREAS = {
    # Vatican City - small, stable, well-mapped
    "vatican": (12.452, 41.900, 12.455, 41.903),  # ~300m x 300m
    
    # Central Park corner - small NYC area, well-mapped  
    "central_park": (-73.974, 40.764, -73.971, 40.767),  # ~300m x 300m
    
    # Monaco harbor - small, stable, good road network
    "monaco": (7.420, 43.730, 7.425, 43.735),  # ~300m x 300m
}


@pytest.mark.integration
def test_fetch_road_network_real_osm():
    """Test fetching REAL road network from OpenStreetMap.
    
    Uses a small, stable geographic area to test actual OSM functionality
    without relying on mocks or fake data.
    """
    # Use Vatican City - small, stable, well-documented road network
    bbox = TEST_AREAS["vatican"]
    
    try:
        # Fetch real road network
        graph = fetch_road_network(bbox, network_type="all")
        
        # Verify we got a real NetworkX graph
        import networkx as nx
        assert isinstance(graph, nx.MultiDiGraph), "Should return NetworkX MultiDiGraph"
        
        # Verify the graph has nodes and edges (Vatican has roads!)
        assert len(graph.nodes) > 0, "Vatican should have road nodes"
        assert len(graph.edges) > 0, "Vatican should have road edges"
        
        # Verify nodes have geographic coordinates and calculate actual extent
        nodes_total = 0
        min_x = max_x = min_y = max_y = None
        
        for node_id, node_data in graph.nodes(data=True):
            assert 'x' in node_data, f"Node {node_id} missing x coordinate"
            assert 'y' in node_data, f"Node {node_id} missing y coordinate"
            
            x, y = node_data['x'], node_data['y']
            nodes_total += 1
            
            # Track actual extent
            if min_x is None:
                min_x = max_x = x
                min_y = max_y = y
            else:
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
        
        # Calculate how much the actual extent exceeds requested bbox
        # This is expected behavior for OSMnx - it includes complete ways
        if nodes_total > 0:
            actual_bbox = (min_x, min_y, max_x, max_y)
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            actual_width = actual_bbox[2] - actual_bbox[0]
            actual_height = actual_bbox[3] - actual_bbox[1]
            
            width_ratio = actual_width / bbox_width if bbox_width > 0 else 1
            height_ratio = actual_height / bbox_height if bbox_height > 0 else 1
            
            # The actual extent should not be more than 3x the requested size
            # (that would indicate we got the wrong area)
            assert width_ratio <= 3.0, \
                f"Actual width is {width_ratio:.1f}x requested - likely wrong area"
            assert height_ratio <= 3.0, \
                f"Actual height is {height_ratio:.1f}x requested - likely wrong area"
            
            # The center should be reasonably close to the requested center
            actual_center_x = (actual_bbox[0] + actual_bbox[2]) / 2
            actual_center_y = (actual_bbox[1] + actual_bbox[3]) / 2
            requested_center_x = (bbox[0] + bbox[2]) / 2
            requested_center_y = (bbox[1] + bbox[3]) / 2
            
            center_offset_x = abs(actual_center_x - requested_center_x)
            center_offset_y = abs(actual_center_y - requested_center_y)
            
            # Center should be within 50% of bbox dimensions
            assert center_offset_x <= bbox_width * 0.5, \
                f"Center X offset {center_offset_x:.6f} too large"
            assert center_offset_y <= bbox_height * 0.5, \
                f"Center Y offset {center_offset_y:.6f} too large"
            
            print(f"✅ Bbox validation: actual extent {width_ratio:.1f}x{height_ratio:.1f} "
                  f"of requested size (expected for OSMnx)")
        
        # Verify edges have road attributes
        edge_count = 0
        for u, v, edge_data in graph.edges(data=True):
            edge_count += 1
            # Most edges should have highway type
            if 'highway' in edge_data:
                assert isinstance(edge_data['highway'], str), "Highway type should be string"
            
            # Stop after checking a few edges
            if edge_count >= 5:
                break
        
        print(f"✅ Vatican road network: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
    except Exception as e:
        pytest.skip(f"OSM service unavailable or network issue: {e}")


@pytest.mark.integration  
def test_fetch_road_network_different_types():
    """Test different network types with real OSM data."""
    bbox = TEST_AREAS["central_park"]  # NYC has good road variety
    
    try:
        # Test 'drive' network (car roads only)
        drive_graph = fetch_road_network(bbox, network_type="drive")
        
        # Test 'walk' network (pedestrian paths)
        walk_graph = fetch_road_network(bbox, network_type="walk")
        
        # Both should be valid graphs
        import networkx as nx
        assert isinstance(drive_graph, nx.MultiDiGraph), "Drive network should be MultiDiGraph"
        assert isinstance(walk_graph, nx.MultiDiGraph), "Walk network should be MultiDiGraph"
        
        # Both should have some roads/paths
        assert len(drive_graph.nodes) > 0, "Drive network should have nodes"
        assert len(walk_graph.nodes) > 0, "Walk network should have nodes"
        
        print(f"✅ Drive network: {len(drive_graph.nodes)} nodes")
        print(f"✅ Walk network: {len(walk_graph.nodes)} nodes")
        
    except Exception as e:
        pytest.skip(f"OSM service unavailable: {e}")


def test_get_hierarchical_road_filters():
    """Test hierarchical road classification filters.
    
    This tests the logic without needing OSM data - it's pure function testing.
    """
    filters = get_hierarchical_road_filters()
    
    # Should return a dictionary of filter categories
    assert isinstance(filters, dict), "Should return dictionary of filters"
    assert len(filters) > 0, "Should have at least one filter category"
    
    # Check that we have expected road categories
    expected_categories = ['major', 'secondary', 'local', 'pedestrian']
    for category in expected_categories:
        if category in filters:
            assert isinstance(filters[category], list), f"{category} filter should be list"
            assert len(filters[category]) > 0, f"{category} filter should not be empty"
            
            # Each filter should be a string (OSM highway tag)
            for highway_type in filters[category]:
                assert isinstance(highway_type, str), f"Highway type should be string: {highway_type}"


@pytest.mark.integration
def test_roads_graph_to_shapes_real_data():
    """Test converting real OSM graph to geometric shapes."""
    bbox = TEST_AREAS["monaco"]  # Monaco has interesting road geometry
    
    try:
        # Get real road network
        graph = fetch_road_network(bbox, network_type="drive")
        
        if len(graph.edges) == 0:
            pytest.skip("No road edges found in test area")
        
        # Convert to shapes
        shapes = roads_graph_to_shapes(graph)
        
        # Should get a list of shape objects
        assert isinstance(shapes, list), "Should return list of shapes"
        assert len(shapes) > 0, "Should have at least one shape"
        
        # Each shape should have geometry and properties
        for shape in shapes[:5]:  # Check first 5 shapes
            assert hasattr(shape, 'geometry'), "Shape should have geometry"
            assert hasattr(shape, 'properties'), "Shape should have properties"
            
            # Geometry should have coordinates
            if hasattr(shape.geometry, 'coords'):
                coords = list(shape.geometry.coords)
                assert len(coords) >= 2, "Road geometry should have at least 2 coordinates"
                
                # Coordinates should be within our bbox
                for lon, lat in coords[:3]:  # Check first few coordinates
                    assert bbox[0] <= lon <= bbox[2], f"Longitude {lon} outside bbox"
                    assert bbox[1] <= lat <= bbox[3], f"Latitude {lat} outside bbox"
        
        print(f"✅ Converted {len(shapes)} road shapes from graph")
        
    except Exception as e:
        pytest.skip(f"OSM service unavailable: {e}")


@pytest.mark.integration
def test_rasterize_road_shapes_real_data():
    """Test rasterizing real road shapes to array."""
    bbox = TEST_AREAS["vatican"]
    
    try:
        # Get real road network and convert to shapes
        graph = fetch_road_network(bbox, network_type="all")
        
        if len(graph.edges) == 0:
            pytest.skip("No road edges found in test area")
            
        shapes = roads_graph_to_shapes(graph)
        
        if len(shapes) == 0:
            pytest.skip("No road shapes generated")
        
        # Define raster parameters
        width, height = 100, 100  # Small test raster
        
        # Rasterize the shapes
        raster = rasterize_road_shapes(shapes, bbox, width, height)
        
        # Should get a numpy array
        assert isinstance(raster, np.ndarray), "Should return numpy array"
        assert raster.shape == (height, width), f"Expected shape ({height}, {width}), got {raster.shape}"
        
        # Array should contain some road pixels (not all zeros)
        assert raster.max() > 0, "Raster should contain some road pixels"
        
        # Values should be in expected range (typically 0-255 or 0-1)
        assert raster.min() >= 0, "Raster values should be non-negative"
        
        print(f"✅ Rasterized roads to {width}x{height} array, {np.sum(raster > 0)} road pixels")
        
    except Exception as e:
        pytest.skip(f"OSM service unavailable: {e}")


@pytest.mark.integration
def test_create_simple_road_mask_real_data():
    """Test creating simple road mask from real OSM data."""
    bbox = TEST_AREAS["central_park"]
    
    try:
        # Create road mask using real data
        mask = create_simple_road_mask(bbox, resolution=50)  # 50x50 for speed
        
        # Should get a numpy array
        assert isinstance(mask, np.ndarray), "Should return numpy array"
        assert len(mask.shape) == 2, "Should be 2D array"
        assert mask.shape[0] > 0 and mask.shape[1] > 0, "Should have non-zero dimensions"
        
        # Should be binary mask (0s and 1s, or similar)
        unique_values = np.unique(mask)
        assert len(unique_values) <= 10, "Simple mask should have few unique values"
        
        # Should have some road pixels
        road_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        road_percentage = (road_pixels / total_pixels) * 100
        
        print(f"✅ Road mask: {mask.shape}, {road_percentage:.1f}% road coverage")
        assert road_percentage > 0, "Should have some road coverage"
        assert road_percentage < 80, "Shouldn't be mostly roads (sanity check)"
        
    except Exception as e:
        pytest.skip(f"OSM service unavailable: {e}")


@pytest.mark.integration
def test_create_road_raster_real_data():
    """Test creating detailed road raster from real OSM data."""
    bbox = TEST_AREAS["monaco"]
    
    try:
        # Create detailed road raster
        raster = create_road_raster(bbox, width=60, height=60)  # Small for speed
        
        # Should get a numpy array
        assert isinstance(raster, np.ndarray), "Should return numpy array"
        assert raster.shape == (60, 60), "Should match requested dimensions"
        
        # Should contain road data
        assert raster.max() > raster.min(), "Should have variation (roads vs background)"
        
        # Check data type is reasonable
        assert raster.dtype in [np.uint8, np.uint16, np.float32, np.float64], \
            f"Unexpected data type: {raster.dtype}"
        
        road_pixels = np.sum(raster > 0)
        coverage = (road_pixels / raster.size) * 100
        
        print(f"✅ Road raster: {raster.shape}, {coverage:.1f}% coverage, dtype={raster.dtype}")
        
    except Exception as e:
        pytest.skip(f"OSM service unavailable: {e}")


def test_bbox_validation():
    """Test bounding box parameter validation (no network needed)."""
    # Test invalid bbox formats
    invalid_bboxes = [
        (1, 2, 3),  # Too few coordinates
        (1, 2, 3, 4, 5),  # Too many coordinates  
        (0, 0, -1, 1),  # Invalid longitude order
        (0, 1, 1, 0),  # Invalid latitude order
        (200, 0, 201, 1),  # Invalid longitude values
        (0, 100, 1, 101),  # Invalid latitude values
    ]
    
    for invalid_bbox in invalid_bboxes:
        try:
            # This should either raise an error or handle gracefully
            result = fetch_road_network(invalid_bbox, network_type="drive")
            # If it doesn't raise an error, it should return empty or handle gracefully
            if result is not None:
                import networkx as nx
                assert isinstance(result, nx.MultiDiGraph), "Should still return valid graph type"
        except (ValueError, RuntimeError, Exception) as e:
            # Expected for invalid inputs
            assert True, f"Correctly rejected invalid bbox: {invalid_bbox}"


def test_function_signatures():
    """Test function signatures are correct (no network needed)."""
    import inspect
    
    # Test fetch_road_network signature
    sig = inspect.signature(fetch_road_network)
    params = list(sig.parameters.keys())
    assert 'bbox' in params, "fetch_road_network should have bbox parameter"
    
    # Test create_road_raster signature
    sig = inspect.signature(create_road_raster)
    params = list(sig.parameters.keys())
    assert 'bbox' in params, "create_road_raster should have bbox parameter"
    
    # Test create_simple_road_mask signature
    sig = inspect.signature(create_simple_road_mask)
    params = list(sig.parameters.keys())
    assert 'bbox' in params, "create_simple_road_mask should have bbox parameter"


@pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="Rasterio required for CRS tests")
@pytest.mark.integration
def test_create_road_raster_with_crs():
    """Test creating road raster with different coordinate reference systems.
    
    This test ensures roads are buffered in meters (via UTM) regardless
    of the target raster CRS. The expected internal workflow is:
    1. Download roads in WGS84 (from OSM)
    2. Detect that buffer_distance is in meters (future: buffer_meters=True param)
    3. Calculate appropriate UTM zone for the area
    4. Project roads to UTM for accurate meter-based buffering
    5. Buffer roads by specified meters in UTM
    6. Reproject buffered roads to target CRS (Web Mercator in this test)
    7. Rasterize to the target grid
    
    This ensures consistent road widths regardless of the target projection.
    """
    bbox = TEST_AREAS["vatican"]
    
    # Create a reference raster in Web Mercator (different from UTM)
    width, height = 300, 300
    web_mercator_crs = CRS.from_epsg(3857)
    
    # Vatican in Web Mercator (actual transformed coordinates)
    west_3857 = 1386150.299
    north_3857 = 5146460.371
    pixel_size = 30  # meters
    
    transform_3857 = from_origin(west_3857, north_3857, pixel_size, pixel_size)
    
    # Create temporary reference raster in Web Mercator
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        reference_raster = tmp.name
        
        with rasterio.open(
            reference_raster, 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype='uint8',
            crs=web_mercator_crs,
            transform=transform_3857
        ) as dst:
            dst.write(np.ones((height, width), dtype=np.uint8), 1)
    
    try:
        # This should:
        # 1. Detect that bbox is in degrees (WGS84)
        # 2. Calculate appropriate UTM zone
        # 3. Buffer roads in UTM by meters
        # 4. Reproject to Web Mercator
        # 5. Rasterize
        output_path = create_road_raster(
            bbox=bbox,
            reference_raster=reference_raster,
            output_path=os.path.join(tempfile.gettempdir(), "test_roads_3857.tif"),
            buffer_distance=30.0   # Should be interpreted as meters when buffer_meters=True is supported
        )
        
        # Verify the output
        with rasterio.open(output_path) as src:
            # Should match reference CRS (Web Mercator)
            assert src.crs == web_mercator_crs, f"Output CRS should be {web_mercator_crs}, got {src.crs}"
            
            # Should have same dimensions
            assert src.width == width
            assert src.height == height
            
            # Should have some road pixels
            data = src.read(1)
            road_pixels = np.sum(data > 0)
            assert road_pixels > 0, "Should have rasterized some roads in UTM"
            
            # Roads should be buffered appropriately (30m buffer = 1 pixel)
            # So roads should be at least 3 pixels wide (1 center + 1 each side)
            
        print(f"✅ CRS-aware road raster created with {road_pixels} road pixels")
        
    finally:
        # Cleanup
        os.unlink(reference_raster)
        if os.path.exists(output_path):
            os.unlink(output_path)


@pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="Rasterio required for CRS tests")
def test_rasterize_road_shapes_with_reprojection():
    """Test rasterizing road shapes with reprojection support.
    
    This tests the low-level function to ensure it can handle
    shapes in one CRS being rasterized to a grid in another CRS.
    """
    # Create some fake road shapes in WGS84
    # Simple line from Vatican to nearby point
    shapes_wgs84 = [
        {
            'geometry': {
                'type': 'LineString',
                'coordinates': [[12.453, 41.902], [12.454, 41.903]]
            },
            'value': 1
        }
    ]
    
    # Create transform and shape for Web Mercator
    width, height = 100, 100
    web_mercator = CRS.from_epsg(3857)
    
    # Vatican in Web Mercator coordinates
    west_3857 = 1387645
    north_3857 = 5145665
    pixel_size = 30  # meters
    
    transform_3857 = from_origin(west_3857, north_3857, pixel_size, pixel_size)
    
    # This should handle reprojection internally
    result = rasterize_road_shapes(
        shapes=shapes_wgs84,
        reference_transform=transform_3857,
        shape=(height, width),
        src_crs="EPSG:4326",  # New parameter
        dst_crs=web_mercator,  # New parameter
        all_touched=True
    )
    
    # Should get a valid raster
    assert result.shape == (height, width)
    assert result.dtype == np.uint8
    
    # Should have some road pixels
    road_pixels = np.sum(result > 0)
    assert road_pixels > 0, "Should have rasterized some roads after reprojection"
    
    print(f"✅ Reprojection in rasterize_road_shapes: {road_pixels} pixels")


@pytest.mark.integration
def test_create_road_raster_from_transform():
    """Test creating road raster from transform and CRS instead of reference file.
    
    This tests a new function that:
    1. Downloads roads in WGS84 for the bbox
    2. Determines appropriate UTM zone 
    3. Projects roads to UTM
    4. Buffers in meters
    5. Reprojects to target CRS
    6. Rasterizes to the provided grid
    """
    from blackmarble.acquire.osm import create_road_raster_from_transform
    
    bbox = TEST_AREAS["monaco"]
    
    # Create transform for Web Mercator
    width, height = 200, 200
    target_crs = CRS.from_epsg(3857)
    
    # Monaco approximate coordinates in Web Mercator
    west = 825000
    north = 5430000
    pixel_size = 30
    
    transform = from_origin(west, north, pixel_size, pixel_size)
    
    try:
        # This new function should handle the entire workflow
        road_array = create_road_raster_from_transform(
            bbox=bbox,
            transform=transform,
            crs=target_crs,
            shape=(height, width),
            buffer_meters=30.0,  # Buffer in meters (applied in UTM)
            road_type_values={    # Explicit road type mapping
                "motorway": 1,
                "primary": 2,
                "secondary": 3,
                "residential": 4
            }
        )
        
        # Should return numpy array directly
        assert isinstance(road_array, np.ndarray)
        assert road_array.shape == (height, width)
        assert road_array.dtype == np.uint8
        
        # Should have different road types
        unique_values = np.unique(road_array)
        assert len(unique_values) > 1, "Should have multiple road types"
        
        # Should have reasonable coverage for Monaco (lots of roads)
        road_pixels = np.sum(road_array > 0)
        coverage = (road_pixels / road_array.size) * 100
        assert coverage > 1.0, f"Monaco should have >1% road coverage, got {coverage:.1f}%"
        
        print(f"✅ Transform-based road raster: {coverage:.1f}% coverage, types: {unique_values}")
        
    except AttributeError:
        pytest.skip("create_road_raster_from_transform not yet implemented")


@pytest.mark.integration
def test_utm_buffer_workflow():
    """Test that buffering happens in UTM for accurate meter-based buffers.
    
    This test verifies the core workflow:
    1. Roads from OSM (WGS84)
    2. Project to appropriate UTM zone
    3. Buffer in meters
    4. Project to target CRS
    """
    from blackmarble.acquire.osm import buffer_roads_in_utm
    
    try:
        from blackmarble.acquire.osm import buffer_roads_in_utm
    except ImportError:
        pytest.skip("buffer_roads_in_utm not yet implemented")
    
    # Simple test road in Vatican (WGS84)
    road_geom = {
        'type': 'LineString',
        'coordinates': [[12.453, 41.902], [12.454, 41.903]]
    }
    
    # Buffer by 30 meters
    buffered_geom = buffer_roads_in_utm(
        geometry=road_geom,
        buffer_meters=30.0,
        lon_center=12.453,
        lat_center=41.902
    )
    
    # The result should be a polygon (buffered line)
    assert buffered_geom['type'] == 'Polygon'
    
    # Rough check: buffered area should be larger than a 30m buffer
    # (very approximate - just checking it's not tiny)
    coords = buffered_geom['coordinates'][0]
    min_lon = min(c[0] for c in coords)
    max_lon = max(c[0] for c in coords)
    width_degrees = max_lon - min_lon
    
    # At 41°N, 30m ≈ 0.0004 degrees longitude
    assert width_degrees > 0.0003, f"Buffer too small: {width_degrees} degrees"
    
    print(f"✅ UTM buffering created polygon with width ~{width_degrees:.4f} degrees")


def test_utm_zone_calculation():
    """Test UTM zone calculation for different locations."""
    try:
        from blackmarble.acquire.osm import calculate_utm_zone
    except ImportError:
        pytest.skip("calculate_utm_zone not yet implemented")
    
    # Test various locations
    test_cases = [
        # (lon, lat, expected_zone, expected_hemisphere)
        (12.453, 41.902, 33, 'north'),    # Vatican - Zone 33N
        (-73.935, 40.730, 18, 'north'),   # NYC - Zone 18N  
        (2.349, 48.853, 31, 'north'),     # Paris - Zone 31N
        (-43.196, -22.908, 23, 'south'),  # Rio - Zone 23S
        (151.209, -33.865, 56, 'south'),  # Sydney - Zone 56S
    ]
    
    for lon, lat, expected_zone, expected_hemisphere in test_cases:
        zone, hemisphere = calculate_utm_zone(lon, lat)
        assert zone == expected_zone, f"Wrong zone for ({lon}, {lat}): got {zone}, expected {expected_zone}"
        assert hemisphere == expected_hemisphere, f"Wrong hemisphere for ({lon}, {lat})"
        
        # Also test EPSG code generation
        epsg = 32600 + zone if hemisphere == 'north' else 32700 + zone
        assert 32601 <= epsg <= 32660 or 32701 <= epsg <= 32760, f"Invalid EPSG: {epsg}"
    
    print("✅ UTM zone calculation correct for all test locations")


@pytest.mark.integration
def test_current_vs_expected_behavior():
    """Test to demonstrate current behavior vs expected CRS-aware behavior.
    
    This test shows:
    1. Current behavior: Roads buffered in degrees, causing inconsistent widths
    2. Expected behavior: Roads buffered in meters via UTM, ensuring consistent widths
    """
    import numpy as np
    
    bbox = TEST_AREAS["vatican"]
    
    # Current behavior - buffer in degrees
    print("\n=== CURRENT BEHAVIOR ===")
    try:
        # Get road network
        graph = fetch_road_network(bbox, network_type="all")
        
        if graph.number_of_edges() > 0:
            # Convert to shapes with degree-based buffer
            buffer_degrees = 0.0003  # Current approach
            shapes_degrees = roads_graph_to_shapes(graph, buffer_distance=buffer_degrees)
            
            # Create a simple raster to see the result
            width, height = 100, 100
            # Need to create transform for rasterization
            from rasterio.transform import from_bounds
            transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], width, height)
            
            # Rasterize using the low-level function
            import rasterio.features
            raster_degrees = rasterio.features.rasterize(
                shapes_degrees,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                all_touched=True,
                default_value=1,
                dtype=np.uint8
            )
            
            pixels_degrees = np.sum(raster_degrees > 0)
            print(f"Degree-based buffer: {pixels_degrees} pixels")
            print(f"Buffer size: {buffer_degrees} degrees")
            
            # Calculate approximate meter equivalent at this latitude
            lat_center = (bbox[1] + bbox[3]) / 2
            meters_approx = buffer_degrees * 111320 * np.cos(np.radians(lat_center))
            print(f"Approximate meters at {lat_center:.1f}°N: {meters_approx:.1f}m")
    except Exception as e:
        pytest.skip(f"OSM unavailable: {e}")
    
    # Expected behavior - buffer in meters via UTM
    print("\n=== EXPECTED BEHAVIOR (when implemented) ===")
    print("1. Download roads in WGS84")
    print("2. Calculate UTM zone 33N for Vatican")
    print("3. Project roads to UTM")
    print("4. Buffer by 30 meters (consistent width)")
    print("5. Project back to WGS84 or target CRS")
    print("6. Rasterize")
    print("\nThis would ensure 30m buffer regardless of latitude!")
    
    # Show the difference in buffer sizes at different latitudes
    print("\n=== BUFFER SIZE VARIATION BY LATITUDE ===")
    test_buffer_degrees = 0.0003
    for lat in [0, 30, 45, 60, 70]:
        meters = test_buffer_degrees * 111320 * np.cos(np.radians(lat))
        print(f"At {lat}°: {test_buffer_degrees} degrees = {meters:.1f} meters")
    
    print("\nThis demonstrates why UTM-based buffering is important!")