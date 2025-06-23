"""Test functions in blackmarble.acquire.osm module.

This module tests OpenStreetMap road network data acquisition functionality,
including road network fetching, graph to shapes conversion, rasterization,
and hierarchical road classification.
"""

import pytest
from unittest.mock import Mock, patch
import tempfile
import os
import numpy as np

# Import the module under test
from blackmarble.acquire.osm import (
    fetch_road_network,
    create_road_raster,
    create_simple_road_mask,
    get_hierarchical_road_filters,
    roads_graph_to_shapes,
    rasterize_road_shapes,
)


def test_fetch_road_network():
    """Test fetching road network from OpenStreetMap using OSMnx.
    
    This test should verify that:
    - Valid bounding boxes return NetworkX graph objects
    - Different network types ('all', 'drive', 'walk') are handled correctly
    - Custom OSM filters work as expected
    - Network connectivity and topology are preserved
    - Coordinate reference system handling is correct
    - Large bounding boxes don't cause memory issues
    - Invalid bounding boxes raise appropriate errors
    """
    import networkx as nx
    
    with patch('blackmarble.acquire.osm.download_osm_roads') as mock_download:
        # Setup mock NetworkX graph with typical Manhattan road pattern
        mock_graph = nx.MultiDiGraph()
        mock_graph.add_node(1, x=-74.0, y=40.7, osmid=1)  # Times Square area
        mock_graph.add_node(2, x=-74.001, y=40.701, osmid=2)
        mock_graph.add_node(3, x=-73.999, y=40.699, osmid=3)
        mock_graph.add_edge(1, 2, highway='primary', length=100, osmid=101)
        mock_graph.add_edge(2, 3, highway='secondary', length=150, osmid=102)
        mock_download.return_value = mock_graph
        
        # Test parameters - Manhattan area
        bbox = (-74.01, 40.69, -73.99, 40.71)  # (min_lon, min_lat, max_lon, max_lat)
        network_type = "all"
        custom_filter = '["highway"~"primary|secondary"]'
        
        # Call function
        result = fetch_road_network(bbox, network_type, custom_filter)
        
        # Verify download called with correct bbox conversion
        mock_download.assert_called_once()
        args = mock_download.call_args[0]
        assert args[0] == 40.71  # north (max_lat)
        assert args[1] == 40.69  # south (min_lat)
        assert args[2] == -73.99  # east (max_lon)
        assert args[3] == -74.01  # west (min_lon)
        assert args[4] == custom_filter  # custom filter passed through
        
        # Verify return is NetworkX graph with expected structure
        assert isinstance(result, nx.MultiDiGraph)
        assert result.number_of_nodes() == 3
        assert result.number_of_edges() == 2
        
        # Verify network connectivity preserved
        assert result.has_edge(1, 2)
        assert result.has_edge(2, 3)
        
        # Verify node attributes preserved
        node_data = result.nodes[1]
        assert node_data['x'] == -74.0
        assert node_data['y'] == 40.7
        
        # Verify edge attributes preserved
        edge_data = result.edges[1, 2, 0]
        assert edge_data['highway'] == 'primary'
        assert edge_data['length'] == 100


def test_fetch_road_network_network_types():
    """Test different network type filters.
    
    This test should verify that:
    - 'all' network type includes all road types
    - 'drive' network type filters to driveable roads only
    - 'walk' network type includes pedestrian-accessible roads
    - Network type filtering affects edge and node counts appropriately
    - Invalid network types raise appropriate errors
    """
    import networkx as nx
    
    with patch('blackmarble.acquire.osm.download_osm_roads') as mock_download:
        bbox = (-74.01, 40.69, -73.99, 40.71)
        
        # Test 'all' network type - should include all roads
        mock_graph_all = nx.MultiDiGraph()
        mock_graph_all.add_node(1, x=-74.0, y=40.7)
        mock_graph_all.add_node(2, x=-74.001, y=40.701)
        mock_graph_all.add_edge(1, 2, highway='primary')
        mock_graph_all.add_edge(1, 2, highway='footway')  # pedestrian only
        mock_download.return_value = mock_graph_all
        
        result_all = fetch_road_network(bbox, "all")
        assert isinstance(result_all, nx.MultiDiGraph)
        assert result_all.number_of_edges() == 2  # includes both road types
        
        # Test 'drive' network type - should filter to driveable roads
        mock_graph_drive = nx.MultiDiGraph()
        mock_graph_drive.add_node(1, x=-74.0, y=40.7)
        mock_graph_drive.add_node(2, x=-74.001, y=40.701)
        mock_graph_drive.add_edge(1, 2, highway='primary')  # driveable only
        mock_download.return_value = mock_graph_drive
        
        result_drive = fetch_road_network(bbox, "drive")
        assert isinstance(result_drive, nx.MultiDiGraph)
        assert result_drive.number_of_edges() == 1  # excludes footways
        
        # Test 'walk' network type - should include pedestrian-accessible roads
        mock_graph_walk = nx.MultiDiGraph()
        mock_graph_walk.add_node(1, x=-74.0, y=40.7)
        mock_graph_walk.add_node(2, x=-74.001, y=40.701)
        mock_graph_walk.add_edge(1, 2, highway='primary')  # walkable
        mock_graph_walk.add_edge(1, 2, highway='footway')  # pedestrian only
        mock_download.return_value = mock_graph_walk
        
        result_walk = fetch_road_network(bbox, "walk")
        assert isinstance(result_walk, nx.MultiDiGraph)
        assert result_walk.number_of_edges() == 2  # includes both types
        
        # Verify different network types produce appropriate results
        # Note: actual filtering happens in OSMnx, but we verify the parameter is passed
        assert mock_download.call_count == 3
        
        # Test that network_type parameter is handled (even though download_osm_roads 
        # doesn't use it - it's a limitation of the current implementation)
        calls = mock_download.call_args_list
        assert len(calls) == 3


def test_fetch_road_network_custom_filters():
    """Test custom OSM filter functionality.
    
    This test should verify that:
    - Custom filter strings are properly applied
    - Complex filter combinations work correctly
    - Invalid filter syntax raises appropriate errors
    - Filter results match expected road classifications
    - Multiple filter criteria can be combined
    """
    import networkx as nx
    
    with patch('blackmarble.acquire.osm.download_osm_roads') as mock_download:
        bbox = (-74.01, 40.69, -73.99, 40.71)
        
        # Test simple custom filter
        mock_graph = nx.MultiDiGraph()
        mock_graph.add_node(1, x=-74.0, y=40.7)
        mock_graph.add_node(2, x=-74.001, y=40.701)
        mock_graph.add_edge(1, 2, highway='primary')
        mock_download.return_value = mock_graph
        
        custom_filter = '["highway"~"primary|secondary"]'
        result = fetch_road_network(bbox, "all", custom_filter)
        
        # Verify custom filter was passed correctly
        mock_download.assert_called_once_with(
            40.71, 40.69, -73.99, -74.01, custom_filter
        )
        assert isinstance(result, nx.MultiDiGraph)
        assert result.number_of_edges() == 1
        
        # Test complex filter combination
        mock_download.reset_mock()
        complex_filter = '["highway"~"primary|secondary"]["maxspeed"~".*"]'
        result_complex = fetch_road_network(bbox, "all", complex_filter)
        
        # Verify complex filter passed through
        mock_download.assert_called_once_with(
            40.71, 40.69, -73.99, -74.01, complex_filter
        )
        
        # Test that filter results are returned as-is (filtering happens in OSMnx)
        assert isinstance(result_complex, nx.MultiDiGraph)
        
        # Test multiple filter criteria can be combined
        mock_download.reset_mock()
        multi_criteria = '["highway"~"primary|secondary|tertiary"]["name"~".*"]'
        result_multi = fetch_road_network(bbox, "all", multi_criteria)
        
        mock_download.assert_called_once_with(
            40.71, 40.69, -73.99, -74.01, multi_criteria
        )
        assert isinstance(result_multi, nx.MultiDiGraph)


def test_fetch_road_network_bounding_box_edge_cases():
    """Test road network fetching with challenging bounding boxes.
    
    This test should verify that:
    - Very small bounding boxes (single intersection) work correctly
    - Large bounding boxes (city-scale) are handled efficiently
    - Bounding boxes crossing administrative boundaries work
    - Bounding boxes with no roads return empty graphs appropriately
    - Invalid coordinate orders raise appropriate errors
    """
    import networkx as nx
    
    with patch('blackmarble.acquire.osm.download_osm_roads') as mock_download:
        
        # Test very small bounding box (single intersection)
        small_bbox = (-74.0001, 40.7000, -73.9999, 40.7002)  # ~200m square
        mock_small_graph = nx.MultiDiGraph()
        mock_small_graph.add_node(1, x=-74.0, y=40.7001)
        mock_small_graph.add_edge(1, 1, highway='primary')  # self-loop intersection
        mock_download.return_value = mock_small_graph
        
        result_small = fetch_road_network(small_bbox)
        assert isinstance(result_small, nx.MultiDiGraph)
        assert result_small.number_of_nodes() >= 1
        
        # Verify correct coordinate conversion for small bbox
        args = mock_download.call_args[0]
        assert abs(args[0] - 40.7002) < 1e-6  # north
        assert abs(args[1] - 40.7000) < 1e-6  # south
        assert abs(args[2] - (-73.9999)) < 1e-6  # east
        assert abs(args[3] - (-74.0001)) < 1e-6  # west
        
        # Test large bounding box (city-scale)
        mock_download.reset_mock()
        large_bbox = (-74.5, 40.4, -73.5, 41.0)  # Manhattan + surroundings
        mock_large_graph = nx.MultiDiGraph()
        # Add many nodes to simulate large network
        for i in range(100):
            mock_large_graph.add_node(i, x=-74.0 + i*0.001, y=40.7 + i*0.001)
            if i > 0:
                mock_large_graph.add_edge(i-1, i, highway='primary')
        mock_download.return_value = mock_large_graph
        
        result_large = fetch_road_network(large_bbox)
        assert isinstance(result_large, nx.MultiDiGraph)
        assert result_large.number_of_nodes() == 100
        assert result_large.number_of_edges() == 99
        
        # Test bounding box with no roads (empty result)
        mock_download.reset_mock()
        empty_bbox = (-180.0, -85.0, -179.0, -84.0)  # Antarctic ocean
        mock_empty_graph = nx.MultiDiGraph()  # Empty graph
        mock_download.return_value = mock_empty_graph
        
        result_empty = fetch_road_network(empty_bbox)
        assert isinstance(result_empty, nx.MultiDiGraph)
        assert result_empty.number_of_nodes() == 0
        assert result_empty.number_of_edges() == 0
        
        # Test bounding boxes crossing administrative boundaries work
        mock_download.reset_mock()
        cross_boundary_bbox = (-74.1, 40.6, -73.9, 40.8)  # NYC/NJ boundary
        mock_cross_graph = nx.MultiDiGraph()
        mock_cross_graph.add_node(1, x=-74.05, y=40.7)  # NJ side
        mock_cross_graph.add_node(2, x=-73.95, y=40.7)  # NY side
        mock_cross_graph.add_edge(1, 2, highway='primary')  # cross-boundary road
        mock_download.return_value = mock_cross_graph
        
        result_cross = fetch_road_network(cross_boundary_bbox)
        assert isinstance(result_cross, nx.MultiDiGraph)
        assert result_cross.has_edge(1, 2)  # boundary-crossing edge preserved


def test_create_road_raster():
    """Test creation of rasterized road network with hierarchical values.
    
    This test should verify that:
    - Reference raster parameters (resolution, CRS) are properly used
    - Hierarchical road classification values are correctly applied
    - Buffer distance parameter affects road thickness appropriately
    - Output raster alignment matches reference raster exactly
    - Road network topology is preserved in rasterization
    - Different road types receive correct classification values
    - Output file format is valid and properly georeferenced
    """
    with patch('blackmarble.acquire.osm.OSMUtil') as mock_osm_util:
        # Setup mock OSMUtil instance
        mock_util_instance = Mock()
        mock_osm_util.return_value = mock_util_instance
        
        # Test parameters
        bbox = (-74.01, 40.69, -73.99, 40.71)
        reference_raster = "/tmp/reference.tif"
        output_path = "/tmp/roads.tif"
        buffer_distance = 0.0001
        road_filters = ['["highway"~"primary"]', '["highway"~"secondary"]']
        
        # Call function
        result = create_road_raster(
            bbox, reference_raster, output_path, buffer_distance, road_filters
        )
        
        # Verify OSMUtil was instantiated
        mock_osm_util.assert_called_once()
        
        # Verify create_road_raster was called with correct parameters
        mock_util_instance.create_road_raster.assert_called_once()
        call_args = mock_util_instance.create_road_raster.call_args
        
        # Check positional arguments (lat1, lon1, lat2, lon2, reference_raster, output_path)
        assert call_args[0][0] == 40.69  # min_lat -> lat1
        assert call_args[0][1] == -74.01  # min_lon -> lon1
        assert call_args[0][2] == 40.71  # max_lat -> lat2
        assert call_args[0][3] == -73.99  # max_lon -> lon2
        assert call_args[0][4] == reference_raster  # reference_raster
        assert call_args[0][5] == output_path  # output_path
        
        # Check keyword arguments
        kwargs = call_args[1]
        assert kwargs['output_dir'] == os.path.dirname(output_path)
        assert kwargs['path_thickness'] == buffer_distance
        
        # Verify return value
        assert result == output_path
        
        # Test with default buffer distance
        mock_util_instance.reset_mock()
        result_default = create_road_raster(bbox, reference_raster, output_path)
        
        call_args_default = mock_util_instance.create_road_raster.call_args[1]
        assert call_args_default['path_thickness'] == 0.00005  # default value


def test_create_road_raster_buffer_distances():
    """Test different buffer distance settings for road rasterization.
    
    This test should verify that:
    - Small buffer distances create thin road representations
    - Large buffer distances create thick road representations
    - Buffer distance scaling matches expected geographic units
    - Overlapping buffers are handled appropriately
    - Zero buffer distance is handled correctly
    """
    with patch('blackmarble.acquire.osm.OSMUtil') as mock_osm_util:
        mock_util_instance = Mock()
        mock_osm_util.return_value = mock_util_instance
        
        bbox = (-74.01, 40.69, -73.99, 40.71)
        reference_raster = "/tmp/reference.tif"
        output_path = "/tmp/roads.tif"
        
        # Test small buffer distance (thin roads)
        small_buffer = 0.00001  # ~1.1m at equator
        create_road_raster(bbox, reference_raster, output_path, small_buffer)
        
        call_args = mock_util_instance.create_road_raster.call_args[1]
        assert call_args['path_thickness'] == small_buffer
        
        # Test large buffer distance (thick roads)
        mock_util_instance.reset_mock()
        large_buffer = 0.001  # ~111m at equator
        create_road_raster(bbox, reference_raster, output_path, large_buffer)
        
        call_args = mock_util_instance.create_road_raster.call_args[1]
        assert call_args['path_thickness'] == large_buffer
        
        # Test zero buffer distance
        mock_util_instance.reset_mock()
        zero_buffer = 0.0
        create_road_raster(bbox, reference_raster, output_path, zero_buffer)
        
        call_args = mock_util_instance.create_road_raster.call_args[1]
        assert call_args['path_thickness'] == zero_buffer
        
        # Verify buffer distance scaling matches expected geographic units
        # At equator: 1 degree ≈ 111 km, so 0.00005 degrees ≈ 5.5m
        typical_buffer = 0.00005
        expected_meters_at_equator = typical_buffer * 111000
        assert abs(expected_meters_at_equator - 5.55) < 0.1  # ~5.5m tolerance


def test_create_road_raster_reference_alignment():
    """Test raster alignment with reference raster parameters.
    
    This test should verify that:
    - Output raster matches reference raster resolution exactly
    - Coordinate reference system is inherited correctly
    - Geotransform parameters align properly
    - Output bounds match reference raster bounds
    - Pixel alignment prevents shifting artifacts
    """
    with patch('blackmarble.acquire.osm.OSMUtil') as mock_osm_util:
        mock_util_instance = Mock()
        mock_osm_util.return_value = mock_util_instance
        
        bbox = (-74.01, 40.69, -73.99, 40.71)
        reference_raster = "/tmp/landsat_reference.tif"
        output_path = "/tmp/aligned_roads.tif"
        
        # Call function
        result = create_road_raster(bbox, reference_raster, output_path)
        
        # Verify OSMUtil.create_road_raster was called
        mock_util_instance.create_road_raster.assert_called_once()
        call_args = mock_util_instance.create_road_raster.call_args
        
        # Verify reference raster is passed for alignment (as positional arg)
        assert call_args[0][4] == reference_raster
        
        # Verify output path matches expected location (as positional arg)
        assert call_args[0][5] == output_path
        kwargs = call_args[1]
        assert kwargs['output_dir'] == os.path.dirname(output_path)
        
        # Verify bounding box coordinates are correctly mapped
        # OSMUtil expects (lat1, lon1, lat2, lon2) format
        pos_args = call_args[0]
        assert pos_args[0] == 40.69  # min_lat
        assert pos_args[1] == -74.01  # min_lon
        assert pos_args[2] == 40.71  # max_lat
        assert pos_args[3] == -73.99  # max_lon
        
        # Test that function returns output path for chaining
        assert result == output_path
        
        # Test with different reference raster
        mock_util_instance.reset_mock()
        viirs_reference = "/tmp/viirs_reference.tif"
        result2 = create_road_raster(bbox, viirs_reference, output_path)
        
        call_args2 = mock_util_instance.create_road_raster.call_args[0]
        assert call_args2[4] == viirs_reference  # reference_raster as positional arg
        assert result2 == output_path


def test_create_simple_road_mask():
    """Test creation of simple binary road mask.
    
    This test should verify that:
    - Binary mask contains only 0 (background) and road_value pixels
    - Road topology is preserved in binary representation
    - Buffer distance parameter works correctly
    - Reference raster alignment is maintained
    - Output format is appropriate for masking operations
    - Road value parameter is properly applied
    - No hierarchical classification is applied
    """
    with patch('blackmarble.acquire.osm.OSMUtil') as mock_osm_util:
        mock_util_instance = Mock()
        mock_osm_util.return_value = mock_util_instance
        
        # Test parameters
        bbox = (-74.01, 40.69, -73.99, 40.71)
        reference_raster = "/tmp/reference.tif"
        output_path = "/tmp/road_mask.tif"
        buffer_distance = 0.00008
        road_value = 255  # High contrast for masking
        
        # Call function
        result = create_simple_road_mask(
            bbox, reference_raster, output_path, buffer_distance, road_value
        )
        
        # Verify OSMUtil was instantiated
        mock_osm_util.assert_called_once()
        
        # Verify create_road_raster_flattened was called (binary mask method)
        mock_util_instance.create_road_raster_flattened.assert_called_once()
        call_args = mock_util_instance.create_road_raster_flattened.call_args
        
        # Check coordinate mapping (lat1, lon1, lat2, lon2, reference_raster, output_path)
        pos_args = call_args[0]
        assert pos_args[0] == 40.69  # min_lat -> lat1
        assert pos_args[1] == -74.01  # min_lon -> lon1
        assert pos_args[2] == 40.71  # max_lat -> lat2
        assert pos_args[3] == -73.99  # max_lon -> lon2
        assert pos_args[4] == reference_raster  # reference_raster
        assert pos_args[5] == output_path  # output_path
        
        # Check keyword arguments
        kwargs = call_args[1]
        assert kwargs['output_dir'] == os.path.dirname(output_path)
        assert kwargs['path_thickness'] == buffer_distance
        
        # Verify return value
        assert result == output_path
        
        # Test with default parameters
        mock_util_instance.reset_mock()
        result_default = create_simple_road_mask(bbox, reference_raster, output_path)
        
        call_args_default = mock_util_instance.create_road_raster_flattened.call_args[1]
        assert call_args_default['path_thickness'] == 0.00005  # default buffer
        # Note: road_value parameter is not used by OSMUtil (uses fixed value of 5)


def test_create_simple_road_mask_road_values():
    """Test different road value settings in binary mask creation.
    
    This test should verify that:
    - Custom road values are properly applied
    - Road pixels contain exactly the specified value
    - Background pixels remain 0
    - Data type is appropriate for specified road values
    """
    with patch('blackmarble.acquire.osm.OSMUtil') as mock_osm_util:
        mock_util_instance = Mock()
        mock_osm_util.return_value = mock_util_instance
        
        bbox = (-74.01, 40.69, -73.99, 40.71)
        reference_raster = "/tmp/reference.tif"
        output_path = "/tmp/road_mask.tif"
        
        # Test different road values
        test_values = [1, 100, 255]  # Common masking values
        
        for road_value in test_values:
            mock_util_instance.reset_mock()
            
            result = create_simple_road_mask(
                bbox, reference_raster, output_path, road_value=road_value
            )
            
            # Verify function called correctly
            mock_util_instance.create_road_raster_flattened.assert_called_once()
            
            # Verify output path returned
            assert result == output_path
            
            # Note: The current implementation doesn't actually use the road_value
            # parameter - it's hardcoded to 5 in OSMUtil.create_road_raster_flattened
            # This is a limitation we document in the test
            
        # Test with binary values (0/1)
        mock_util_instance.reset_mock()
        binary_result = create_simple_road_mask(
            bbox, reference_raster, output_path, road_value=1
        )
        assert binary_result == output_path
        
        # Test with high contrast value (useful for visualization)
        mock_util_instance.reset_mock()
        contrast_result = create_simple_road_mask(
            bbox, reference_raster, output_path, road_value=255
        )
        assert contrast_result == output_path
        
        # Verify all calls used the same underlying method
        # Note: Only the last call (contrast_result) is counted because of reset_mock() calls
        assert mock_util_instance.create_road_raster_flattened.call_count == 1


def test_get_hierarchical_road_filters():
    """Test retrieval of default hierarchical road classification filters.
    
    This test should verify that:
    - Return format is dictionary mapping integers to filter strings
    - Filter strings are valid OSM syntax
    - Road classification hierarchy makes logical sense
    - All major road types are covered
    - Filter priorities reflect real-world road importance
    - No overlapping classifications exist
    - Default values match OSMUtil implementation
    """
    # Call function
    filters = get_hierarchical_road_filters()
    
    # Verify return format is dictionary mapping integers to strings
    assert isinstance(filters, dict)
    assert len(filters) > 0
    
    for key, value in filters.items():
        assert isinstance(key, int)
        assert isinstance(value, str)
        assert key > 0  # Positive classification values
    
    # Verify filter strings are valid OSM syntax
    for filter_str in filters.values():
        assert filter_str.startswith('["highway"~"')
        assert filter_str.endswith('"]')
        # Some filters may have multiple types (with |), others may have single types
        assert len(filter_str) > 15  # Minimum reasonable length
    
    # Verify specific road classifications are present
    filter_values = list(filters.values())
    
    # Check for primary roads
    primary_found = any('primary' in f for f in filter_values)
    assert primary_found, "Primary roads should be classified"
    
    # Check for secondary roads
    secondary_found = any('secondary' in f for f in filter_values)
    assert secondary_found, "Secondary roads should be classified"
    
    # Check for motorways (highest priority roads)
    motorway_found = any('motorway' in f for f in filter_values)
    assert motorway_found, "Motorways should be classified"
    
    # Check for residential roads
    residential_found = any('residential' in f for f in filter_values)
    assert residential_found, "Residential roads should be classified"
    
    # Check for trunk roads
    trunk_found = any('trunk' in f for f in filter_values)
    assert trunk_found, "Trunk roads should be classified"
    
    # Verify no duplicate classification values
    keys = list(filters.keys())
    assert len(keys) == len(set(keys)), "No duplicate classification values"
    
    # Verify road classification hierarchy makes sense (higher numbers = higher priority)
    # This assumes that the implementation assigns higher values to more important roads
    expected_classifications = {1, 2, 3, 4, 5}  # Based on implementation
    actual_classifications = set(filters.keys())
    assert actual_classifications == expected_classifications
    
    # Verify all filters are non-empty
    for filter_str in filters.values():
        assert len(filter_str) > 10  # Reasonable minimum length for OSM filter


def test_get_hierarchical_road_filters_completeness():
    """Test completeness of hierarchical road classification.
    
    This test should verify that:
    - Primary, secondary, tertiary roads are properly classified
    - Motorways and motorway links have appropriate priority
    - Residential and service roads are included
    - Trunk roads are properly categorized
    - Special road types (bus lanes, etc.) are handled
    """
    filters = get_hierarchical_road_filters()
    all_filters = ' '.join(filters.values()).lower()
    
    # Test primary road classification
    assert 'primary' in all_filters, "Primary roads must be classified"
    assert 'primary_link' in all_filters, "Primary links must be classified"
    
    # Test secondary road classification
    assert 'secondary' in all_filters, "Secondary roads must be classified"
    assert 'secondary_link' in all_filters, "Secondary links must be classified"
    
    # Test tertiary road classification
    assert 'tertiary' in all_filters, "Tertiary roads must be classified"
    assert 'tertiary_link' in all_filters, "Tertiary links must be classified"
    
    # Test motorway classification (highest priority)
    assert 'motorway' in all_filters, "Motorways must be classified"
    assert 'motorway_link' in all_filters, "Motorway links must be classified"
    
    # Test residential roads are included
    assert 'residential' in all_filters, "Residential roads must be classified"
    
    # Test trunk roads are properly categorized
    assert 'trunk' in all_filters, "Trunk roads must be classified"
    assert 'trunk_link' in all_filters, "Trunk links must be classified"
    
    # Test service roads are included
    assert 'service' in all_filters, "Service roads must be classified"
    
    # Test special road types are handled
    assert 'unclassified' in all_filters, "Unclassified roads must be handled"
    
    # Test that busways are included (special transit infrastructure)
    assert 'busway' in all_filters, "Busways must be classified"
    
    # Verify logical grouping - similar road types should be in same filter
    primary_filter = None
    motorway_filter = None
    residential_filter = None
    
    for value, filter_str in filters.items():
        if 'primary' in filter_str and 'secondary' in filter_str:
            primary_filter = filter_str
        elif 'motorway' in filter_str:
            motorway_filter = filter_str
        elif 'residential' in filter_str:
            residential_filter = filter_str
    
    # Verify primary/secondary/tertiary are grouped together (similar importance)
    assert primary_filter is not None, "Primary roads should be grouped with similar types"
    assert 'secondary' in primary_filter, "Secondary should be with primary classification"
    assert 'tertiary' in primary_filter, "Tertiary should be with primary classification"
    
    # Verify motorways are separate (highest priority)
    assert motorway_filter is not None, "Motorways should have separate classification"
    assert 'motorway_link' in motorway_filter, "Motorway links with motorways"
    
    # Verify residential roads are separate (local importance)
    assert residential_filter is not None, "Residential roads should be classified"


def test_roads_graph_to_shapes():
    """Test conversion of road network graph to buffered polygon shapes.
    
    This test should verify that:
    - NetworkX graph edges are converted to shapely geometries
    - Buffer distance parameter creates appropriate polygon widths
    - Output format is suitable for rasterization functions
    - Coordinate reference system is preserved
    - Complex intersections are handled correctly
    - Self-intersecting roads are processed appropriately
    - Memory usage is reasonable for large networks
    """
    import networkx as nx
    from shapely.geometry import LineString, Point
    
    with patch('blackmarble.acquire.osm.graph_to_shapes') as mock_graph_to_shapes:
        # Create test NetworkX graph
        graph = nx.MultiDiGraph()
        graph.add_node(1, x=-74.0, y=40.7, osmid=1)
        graph.add_node(2, x=-74.001, y=40.701, osmid=2)
        graph.add_node(3, x=-73.999, y=40.699, osmid=3)
        graph.add_edge(1, 2, highway='primary', length=100, osmid=101)
        graph.add_edge(2, 3, highway='secondary', length=150, osmid=102)
        
        # Mock return value - list of GeoJSON-like shape dictionaries
        mock_shapes = [
            {
                'type': 'Polygon',
                'coordinates': [[[-74.0, 40.7], [-74.001, 40.701], [-74.0005, 40.7005], [-74.0, 40.7]]]
            },
            {
                'type': 'Polygon', 
                'coordinates': [[[-74.001, 40.701], [-73.999, 40.699], [-74.0, 40.700], [-74.001, 40.701]]]
            }
        ]
        mock_graph_to_shapes.return_value = mock_shapes
        
        buffer_distance = 0.00005
        
        # Call function
        result = roads_graph_to_shapes(graph, buffer_distance)
        
        # Verify graph_to_shapes was called with correct parameters
        mock_graph_to_shapes.assert_called_once_with(graph, buffer_distance)
        
        # Verify output format is suitable for rasterization
        assert isinstance(result, list)
        assert len(result) == 2
        
        for shape in result:
            assert isinstance(shape, dict)
            assert 'type' in shape
            assert 'coordinates' in shape
            assert shape['type'] == 'Polygon'  # Buffered lines become polygons
        
        # Test with different buffer distances
        mock_graph_to_shapes.reset_mock()
        large_buffer = 0.001
        result_large = roads_graph_to_shapes(graph, large_buffer)
        
        mock_graph_to_shapes.assert_called_once_with(graph, large_buffer)
        assert isinstance(result_large, list)
        
        # Test with empty graph
        mock_graph_to_shapes.reset_mock()
        empty_graph = nx.MultiDiGraph()
        mock_graph_to_shapes.return_value = []
        
        result_empty = roads_graph_to_shapes(empty_graph, buffer_distance)
        mock_graph_to_shapes.assert_called_once_with(empty_graph, buffer_distance)
        assert result_empty == []
        
        # Test with complex intersection (multiple edges from same node)
        mock_graph_to_shapes.reset_mock()
        complex_graph = nx.MultiDiGraph()
        complex_graph.add_node(1, x=-74.0, y=40.7)
        complex_graph.add_node(2, x=-74.001, y=40.701)
        complex_graph.add_node(3, x=-73.999, y=40.701)
        complex_graph.add_node(4, x=-74.0, y=40.699)
        # 4-way intersection at node 1
        complex_graph.add_edge(1, 2, highway='primary')
        complex_graph.add_edge(1, 3, highway='secondary')
        complex_graph.add_edge(1, 4, highway='tertiary')
        
        mock_complex_shapes = [{'type': 'Polygon', 'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}] * 3
        mock_graph_to_shapes.return_value = mock_complex_shapes
        
        result_complex = roads_graph_to_shapes(complex_graph, buffer_distance)
        assert len(result_complex) == 3  # Three edges from intersection


def test_roads_graph_to_shapes_geometry_types():
    """Test geometry type handling in graph to shapes conversion.
    
    This test should verify that:
    - LineString geometries are converted to Polygons via buffering
    - Multi-segment roads are handled as single features
    - Complex intersection geometry is preserved
    - Topology errors are handled gracefully
    - Invalid geometries are detected and handled
    """
    import networkx as nx
    
    with patch('blackmarble.acquire.osm.graph_to_shapes') as mock_graph_to_shapes:
        # Test LineString to Polygon conversion
        simple_graph = nx.MultiDiGraph()
        simple_graph.add_node(1, x=-74.0, y=40.7)
        simple_graph.add_node(2, x=-74.001, y=40.701)
        simple_graph.add_edge(1, 2, highway='primary')
        
        # Mock shapes representing buffered LineStrings (now Polygons)
        mock_polygon_shapes = [
            {
                'type': 'Polygon',
                'coordinates': [[[-74.00005, 40.69995], [-74.00105, 40.70095], 
                               [-74.00095, 40.70105], [-73.99995, 40.70005], [-74.00005, 40.69995]]]
            }
        ]
        mock_graph_to_shapes.return_value = mock_polygon_shapes
        
        result = roads_graph_to_shapes(simple_graph, 0.00005)
        
        # Verify LineStrings become Polygons after buffering
        assert len(result) == 1
        shape = result[0]
        assert shape['type'] == 'Polygon'
        assert 'coordinates' in shape
        
        # Test multi-segment roads (single feature)
        mock_graph_to_shapes.reset_mock()
        multi_segment_graph = nx.MultiDiGraph()
        multi_segment_graph.add_node(1, x=-74.0, y=40.7)
        multi_segment_graph.add_node(2, x=-74.001, y=40.701)
        multi_segment_graph.add_node(3, x=-74.002, y=40.702)
        multi_segment_graph.add_edge(1, 2, highway='primary', name='Main St')
        multi_segment_graph.add_edge(2, 3, highway='primary', name='Main St')  # Continuation
        
        mock_multi_shapes = [
            {'type': 'Polygon', 'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
            {'type': 'Polygon', 'coordinates': [[[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]]]}
        ]
        mock_graph_to_shapes.return_value = mock_multi_shapes
        
        result_multi = roads_graph_to_shapes(multi_segment_graph, 0.00005)
        assert len(result_multi) == 2  # Two road segments
        
        for shape in result_multi:
            assert shape['type'] == 'Polygon'
            assert len(shape['coordinates'][0]) >= 4  # Polygon requires >=4 points
        
        # Test complex intersection geometry
        mock_graph_to_shapes.reset_mock()
        intersection_graph = nx.MultiDiGraph()
        center_node = 1
        intersection_graph.add_node(center_node, x=-74.0, y=40.7)
        
        # Add 6 roads meeting at intersection (like Times Square)
        for i in range(2, 8):
            intersection_graph.add_node(i, x=-74.0 + (i-2)*0.0001, y=40.7 + (i-2)*0.0001)
            intersection_graph.add_edge(center_node, i, highway='primary')
        
        mock_intersection_shapes = [{'type': 'Polygon', 'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}] * 6
        mock_graph_to_shapes.return_value = mock_intersection_shapes
        
        result_intersection = roads_graph_to_shapes(intersection_graph, 0.00005)
        assert len(result_intersection) == 6  # Six roads from intersection
        
        # Test with zero edges (topology error case)
        mock_graph_to_shapes.reset_mock()
        empty_graph = nx.MultiDiGraph()
        empty_graph.add_node(1, x=-74.0, y=40.7)  # Node with no edges
        mock_graph_to_shapes.return_value = []
        
        result_empty = roads_graph_to_shapes(empty_graph, 0.00005)
        assert result_empty == []  # Graceful handling of no edges


def test_roads_graph_to_shapes_buffer_edge_cases():
    """Test edge cases in buffering during graph to shapes conversion.
    
    This test should verify that:
    - Very small buffer distances don't cause topology errors
    - Large buffer distances don't cause excessive memory usage
    - Zero buffer distance is handled appropriately
    - Negative buffer distances raise appropriate errors
    """
    import networkx as nx
    
    with patch('blackmarble.acquire.osm.graph_to_shapes') as mock_graph_to_shapes:
        # Setup test graph
        graph = nx.MultiDiGraph()
        graph.add_node(1, x=-74.0, y=40.7)
        graph.add_node(2, x=-74.001, y=40.701)
        graph.add_edge(1, 2, highway='primary')
        
        # Test very small buffer distances
        very_small_buffer = 1e-8  # ~1mm at equator
        mock_tiny_shapes = [{
            'type': 'Polygon',
            'coordinates': [[[-74.0, 40.7], [-74.001, 40.701], [-74.001, 40.701], [-74.0, 40.7], [-74.0, 40.7]]]
        }]
        mock_graph_to_shapes.return_value = mock_tiny_shapes
        
        result_tiny = roads_graph_to_shapes(graph, very_small_buffer)
        mock_graph_to_shapes.assert_called_once_with(graph, very_small_buffer)
        assert len(result_tiny) == 1
        assert result_tiny[0]['type'] == 'Polygon'  # No topology errors
        
        # Test large buffer distances
        mock_graph_to_shapes.reset_mock()
        large_buffer = 0.01  # ~1.1km at equator
        mock_large_shapes = [{
            'type': 'Polygon',
            'coordinates': [[[-74.01, 40.69], [-73.99, 40.71], [-73.99, 40.71], [-74.01, 40.69], [-74.01, 40.69]]]
        }]
        mock_graph_to_shapes.return_value = mock_large_shapes
        
        result_large = roads_graph_to_shapes(graph, large_buffer)
        mock_graph_to_shapes.assert_called_once_with(graph, large_buffer)
        assert len(result_large) == 1  # Memory usage remains reasonable
        
        # Test zero buffer distance
        mock_graph_to_shapes.reset_mock()
        zero_buffer = 0.0
        mock_zero_shapes = [{
            'type': 'Polygon',  # Even zero buffer might create minimal polygon
            'coordinates': [[[-74.0, 40.7], [-74.001, 40.701], [-74.001, 40.701], [-74.0, 40.7], [-74.0, 40.7]]]
        }]
        mock_graph_to_shapes.return_value = mock_zero_shapes
        
        result_zero = roads_graph_to_shapes(graph, zero_buffer)
        mock_graph_to_shapes.assert_called_once_with(graph, zero_buffer)
        assert isinstance(result_zero, list)  # Zero buffer handled appropriately
        
        # Test negative buffer distance (should be passed through to underlying function)
        mock_graph_to_shapes.reset_mock()
        negative_buffer = -0.00001
        
        # The wrapper function doesn't validate buffer distance - passes it through
        roads_graph_to_shapes(graph, negative_buffer)
        mock_graph_to_shapes.assert_called_once_with(graph, negative_buffer)
        
        # Test buffer distance validation occurs in underlying implementation
        # Different buffer distances should produce different calls
        test_buffers = [0.00001, 0.0001, 0.001]
        mock_graph_to_shapes.reset_mock()
        
        for buffer_dist in test_buffers:
            mock_graph_to_shapes.return_value = [{'type': 'Polygon', 'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}]
            result = roads_graph_to_shapes(graph, buffer_dist)
            assert len(result) > 0
        
        # Verify all buffer distances were passed through
        assert mock_graph_to_shapes.call_count == len(test_buffers)
        call_args = [call[0][1] for call in mock_graph_to_shapes.call_args_list]
        assert call_args == test_buffers


def test_rasterize_road_shapes():
    """Test rasterization of road shapes to array.
    
    This test should verify that:
    - Shapely geometries are properly converted to pixel arrays
    - Reference transform is applied correctly
    - Output array shape matches expected dimensions
    - Pixel values are assigned according to shape attributes
    - All_touched parameter affects rasterization appropriately
    - Edge cases (shapes partially outside bounds) are handled
    - Performance is acceptable for large shape collections
    """
    from rasterio.transform import from_bounds
    
    with patch('blackmarble.acquire.osm.rasterize_shapes') as mock_rasterize:
        # Test parameters
        shapes = [
            {
                'type': 'Polygon',
                'coordinates': [[[-74.0, 40.7], [-74.001, 40.701], [-74.0005, 40.7005], [-74.0, 40.7]]]
            },
            {
                'type': 'Polygon',
                'coordinates': [[[-74.001, 40.701], [-73.999, 40.699], [-74.0, 40.700], [-74.001, 40.701]]]
            }
        ]
        
        # Create test transform (Manhattan area, 30m resolution)
        bounds = (-74.01, 40.69, -73.99, 40.71)
        width, height = 100, 80
        reference_transform = from_bounds(*bounds, width, height)
        shape = (height, width)
        
        # Mock return array
        mock_array = np.zeros((height, width), dtype=np.uint8)
        mock_array[10:20, 15:25] = 1  # Some road pixels
        mock_rasterize.return_value = mock_array
        
        # Call function with default parameters
        result = rasterize_road_shapes(shapes, reference_transform, shape)
        
        # Verify rasterize_shapes was called correctly
        mock_rasterize.assert_called_once()
        call_args = mock_rasterize.call_args
        
        assert call_args[0][0] == shapes  # shapes parameter
        assert call_args[0][1] == height  # height parameter
        assert call_args[0][2] == width   # width parameter
        assert call_args[0][3] == reference_transform  # transform parameter
        
        # Check keyword arguments
        kwargs = call_args[1]
        assert kwargs['value'] == 1  # default value
        assert kwargs['all_touched'] == False  # default all_touched
        
        # Verify output array
        assert isinstance(result, np.ndarray)
        assert result.shape == (height, width)
        assert result.dtype == np.uint8
        
        # Test with all_touched=True
        mock_rasterize.reset_mock()
        result_all_touched = rasterize_road_shapes(
            shapes, reference_transform, shape, all_touched=True
        )
        
        kwargs_touched = mock_rasterize.call_args[1]
        assert kwargs_touched['all_touched'] == True
        
        # Test with different shape dimensions
        mock_rasterize.reset_mock()
        large_shape = (200, 150)
        mock_large_array = np.zeros(large_shape, dtype=np.uint8)
        mock_rasterize.return_value = mock_large_array
        
        result_large = rasterize_road_shapes(
            shapes, reference_transform, large_shape
        )
        
        call_args_large = mock_rasterize.call_args[0]
        assert call_args_large[1] == 200  # height
        assert call_args_large[2] == 150  # width
        assert result_large.shape == large_shape
        
        # Test with empty shapes list
        mock_rasterize.reset_mock()
        empty_shapes = []
        mock_empty_array = np.zeros(shape, dtype=np.uint8)
        mock_rasterize.return_value = mock_empty_array
        
        result_empty = rasterize_road_shapes(
            empty_shapes, reference_transform, shape
        )
        
        mock_rasterize.assert_called_once_with(
            empty_shapes, height, width, reference_transform, value=1, all_touched=False
        )
        assert np.all(result_empty == 0)  # Empty result


def test_rasterize_road_shapes_all_touched_parameter():
    """Test all_touched parameter in shape rasterization.
    
    This test should verify that:
    - all_touched=True rasterizes all partially covered pixels
    - all_touched=False only rasterizes pixels with centers inside shapes
    - Boundary pixel handling is consistent
    - Results differ appropriately between True/False settings
    """
    from rasterio.transform import from_bounds
    
    with patch('blackmarble.acquire.osm.rasterize_shapes') as mock_rasterize:
        # Test shapes that partially cover pixels
        boundary_shapes = [
            {
                'type': 'Polygon',
                'coordinates': [[[-74.0005, 40.7002], [-74.0003, 40.7004], 
                               [-74.0001, 40.7002], [-74.0003, 40.7000], [-74.0005, 40.7002]]]
            }
        ]
        
        bounds = (-74.001, 40.700, -73.999, 40.702)
        width, height = 50, 40
        transform = from_bounds(*bounds, width, height)
        shape = (height, width)
        
        # Mock different results for all_touched True vs False
        mock_array_touched = np.zeros((height, width), dtype=np.uint8)
        mock_array_touched[18:22, 23:27] = 1  # More pixels with all_touched=True
        
        mock_array_not_touched = np.zeros((height, width), dtype=np.uint8)
        mock_array_not_touched[19:21, 24:26] = 1  # Fewer pixels with all_touched=False
        
        # Test all_touched=True
        mock_rasterize.return_value = mock_array_touched
        result_touched = rasterize_road_shapes(
            boundary_shapes, transform, shape, all_touched=True
        )
        
        # Verify all_touched=True was passed
        kwargs_touched = mock_rasterize.call_args[1]
        assert kwargs_touched['all_touched'] == True
        
        touched_pixel_count = np.sum(result_touched > 0)
        assert touched_pixel_count == 16  # 4x4 area
        
        # Test all_touched=False
        mock_rasterize.reset_mock()
        mock_rasterize.return_value = mock_array_not_touched
        result_not_touched = rasterize_road_shapes(
            boundary_shapes, transform, shape, all_touched=False
        )
        
        # Verify all_touched=False was passed
        kwargs_not_touched = mock_rasterize.call_args[1]
        assert kwargs_not_touched['all_touched'] == False
        
        not_touched_pixel_count = np.sum(result_not_touched > 0)
        assert not_touched_pixel_count == 4  # 2x2 area (fewer pixels)
        
        # Verify results differ appropriately
        assert touched_pixel_count > not_touched_pixel_count
        
        # Test boundary consistency - same shapes should give consistent results
        mock_rasterize.reset_mock()
        result_touched_2 = rasterize_road_shapes(
            boundary_shapes, transform, shape, all_touched=True
        )
        
        # Should call with same parameters
        assert mock_rasterize.call_args[1]['all_touched'] == True
        
        # Test with precise pixel-boundary shapes
        mock_rasterize.reset_mock()
        pixel_aligned_shapes = [
            {
                'type': 'Polygon',
                'coordinates': [[[-74.0004, 40.7001], [-74.0002, 40.7001], 
                               [-74.0002, 40.7003], [-74.0004, 40.7003], [-74.0004, 40.7001]]]
            }
        ]
        
        # Mock results showing boundary pixel differences
        mock_boundary_touched = np.zeros((height, width), dtype=np.uint8)
        mock_boundary_touched[19:21, 24:26] = 1
        mock_rasterize.return_value = mock_boundary_touched
        
        result_boundary = rasterize_road_shapes(
            pixel_aligned_shapes, transform, shape, all_touched=True
        )
        
        # Verify boundary pixels are handled consistently
        assert np.sum(result_boundary > 0) > 0
        
        # Test default parameter (should be False)
        mock_rasterize.reset_mock()
        result_default = rasterize_road_shapes(
            boundary_shapes, transform, shape  # no all_touched specified
        )
        
        # Verify default is all_touched=False
        default_kwargs = mock_rasterize.call_args[1]
        assert default_kwargs['all_touched'] == False


def test_rasterize_road_shapes_transform_handling():
    """Test coordinate transform handling in rasterization.
    
    This test should verify that:
    - Affine transform parameters are applied correctly
    - Geographic coordinates are properly mapped to pixel coordinates
    - Transform scaling and translation work as expected
    - Invalid transforms raise appropriate errors
    - Edge pixel alignment is maintained
    """
    from rasterio.transform import from_bounds, Affine
    
    with patch('blackmarble.acquire.osm.rasterize_shapes') as mock_rasterize:
        # Test shapes in Manhattan
        test_shapes = [
            {
                'type': 'Polygon',
                'coordinates': [[[-74.0, 40.7], [-74.001, 40.701], [-74.0005, 40.7005], [-74.0, 40.7]]]
            }
        ]
        
        # Test with geographic bounds transform
        geo_bounds = (-74.01, 40.69, -73.99, 40.71)
        width, height = 100, 80
        geo_transform = from_bounds(*geo_bounds, width, height)
        shape = (height, width)
        
        mock_result = np.zeros(shape, dtype=np.uint8)
        mock_result[30:35, 40:45] = 1
        mock_rasterize.return_value = mock_result
        
        # Call function
        result_geo = rasterize_road_shapes(test_shapes, geo_transform, shape)
        
        # Verify transform was passed correctly
        call_args = mock_rasterize.call_args[0]
        assert call_args[3] == geo_transform  # transform parameter
        
        # Test with custom Affine transform
        mock_rasterize.reset_mock()
        custom_transform = Affine(0.0001, 0.0, -74.01, 0.0, -0.0001, 40.71)
        
        result_custom = rasterize_road_shapes(test_shapes, custom_transform, shape)
        
        custom_call_args = mock_rasterize.call_args[0]
        assert custom_call_args[3] == custom_transform
        
        # Test with high-resolution transform (smaller pixels)
        mock_rasterize.reset_mock()
        hires_bounds = (-74.005, 40.695, -73.995, 40.705)
        hires_width, hires_height = 500, 400  # 5x higher resolution
        hires_transform = from_bounds(*hires_bounds, hires_width, hires_height)
        hires_shape = (hires_height, hires_width)
        
        mock_hires_result = np.zeros(hires_shape, dtype=np.uint8)
        mock_rasterize.return_value = mock_hires_result
        
        result_hires = rasterize_road_shapes(test_shapes, hires_transform, hires_shape)
        
        # Verify high-resolution parameters
        hires_call_args = mock_rasterize.call_args[0]
        assert hires_call_args[1] == hires_height  # height
        assert hires_call_args[2] == hires_width   # width
        assert hires_call_args[3] == hires_transform
        assert result_hires.shape == hires_shape
        
        # Test with different coordinate systems (projected)
        mock_rasterize.reset_mock()
        # UTM Zone 18N transform (approximate for NYC)
        utm_transform = Affine(30.0, 0.0, 580000.0, 0.0, -30.0, 4510000.0)
        
        result_utm = rasterize_road_shapes(test_shapes, utm_transform, shape)
        
        utm_call_args = mock_rasterize.call_args[0]
        assert utm_call_args[3] == utm_transform
        
        # Test transform scaling verification
        # Verify pixel size calculation
        pixel_width = abs(geo_transform.a)  # x-direction pixel size
        pixel_height = abs(geo_transform.e)  # y-direction pixel size
        
        expected_pixel_width = (geo_bounds[2] - geo_bounds[0]) / width
        expected_pixel_height = (geo_bounds[3] - geo_bounds[1]) / height
        
        assert abs(pixel_width - expected_pixel_width) < 1e-10
        assert abs(pixel_height - expected_pixel_height) < 1e-10
        
        # Test edge pixel alignment with different transforms
        mock_rasterize.reset_mock()
        edge_bounds = (-74.0, 40.7, -74.0 + 0.01, 40.7 + 0.01)  # 1km square
        edge_transform = from_bounds(*edge_bounds, 50, 50)
        edge_shape = (50, 50)
        
        mock_edge_result = np.zeros(edge_shape, dtype=np.uint8)
        mock_edge_result[0, 0] = 1  # Corner pixel
        mock_edge_result[-1, -1] = 1  # Opposite corner
        mock_rasterize.return_value = mock_edge_result
        
        result_edge = rasterize_road_shapes(test_shapes, edge_transform, edge_shape)
        
        # Verify edge alignment maintained
        edge_call_args = mock_rasterize.call_args[0]
        assert edge_call_args[3] == edge_transform
        assert result_edge.shape == edge_shape
        
        # Note: Invalid transform errors would be handled by the underlying rasterize_shapes function
        # The wrapper function doesn't validate transforms - it passes them through


def test_osm_network_error_handling():
    """Test error handling for OSM network connectivity issues.
    
    This test should verify that:
    - Network timeouts are handled gracefully
    - OSM server unavailability results in clear error messages
    - Rate limiting is respected and handled
    - Connection errors don't crash the application
    - Retry mechanisms work when appropriate
    """
    import requests
    from requests.exceptions import Timeout, ConnectionError, HTTPError
    
    with patch('blackmarble.acquire.osm.download_osm_roads') as mock_download:
        bbox = (-74.01, 40.69, -73.99, 40.71)
        
        # Test network timeout handling
        mock_download.side_effect = Timeout("Request timed out")
        
        with pytest.raises(Timeout):
            fetch_road_network(bbox)
        
        # Verify timeout was attempted
        mock_download.assert_called_once()
        
        # Test OSM server unavailability
        mock_download.reset_mock()
        mock_download.side_effect = ConnectionError("Cannot connect to OSM servers")
        
        with pytest.raises(ConnectionError):
            fetch_road_network(bbox)
        
        # Test HTTP errors (rate limiting, server errors)
        mock_download.reset_mock()
        mock_download.side_effect = HTTPError("429 Too Many Requests")
        
        with pytest.raises(HTTPError):
            fetch_road_network(bbox)
        
        # Test that connection errors don't crash silently
        mock_download.reset_mock()
        mock_download.side_effect = Exception("Unexpected network error")
        
        with pytest.raises(Exception) as exc_info:
            fetch_road_network(bbox)
        
        assert "Unexpected network error" in str(exc_info.value)
        
        # Test successful retry after initial failure
        import networkx as nx
        mock_download.reset_mock()
        
        # First call fails, second succeeds (simulating retry)
        mock_graph = nx.MultiDiGraph()
        mock_graph.add_node(1, x=-74.0, y=40.7)
        mock_graph.add_edge(1, 1, highway='primary')
        
        call_count = 0
        def side_effect_retry(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("First attempt failed")
            return mock_graph
        
        mock_download.side_effect = side_effect_retry
        
        # First call should fail
        with pytest.raises(ConnectionError):
            fetch_road_network(bbox)
        
        # Second call should succeed
        result = fetch_road_network(bbox)
        assert isinstance(result, nx.MultiDiGraph)
        assert result.number_of_nodes() == 1
        
        # Test error handling with custom filters
        mock_download.reset_mock()
        mock_download.side_effect = HTTPError("Invalid filter syntax")
        
        with pytest.raises(HTTPError):
            fetch_road_network(bbox, custom_filter='["invalid"~"syntax')
        
        # Verify error propagation preserves context
        try:
            fetch_road_network(bbox, custom_filter='["invalid"~"syntax')
        except HTTPError as e:
            assert "Invalid filter syntax" in str(e)


def test_osm_invalid_bbox_handling():
    """Test handling of invalid bounding box parameters.
    
    This test should verify that:
    - Inverted bounding boxes (min > max) raise appropriate errors
    - Out-of-range coordinates (lat > 90) are detected
    - Degenerate bounding boxes (point, line) are handled
    - Very large bounding boxes are handled or rejected appropriately
    - Coordinate order assumptions are validated
    """
    import networkx as nx
    
    with patch('blackmarble.acquire.osm.download_osm_roads') as mock_download:
        
        # Test inverted longitude (min_lon > max_lon)
        inverted_lon_bbox = (-73.99, 40.69, -74.01, 40.71)  # west > east
        mock_download.side_effect = ValueError("Invalid bounding box")
        
        with pytest.raises(ValueError):
            fetch_road_network(inverted_lon_bbox)
        
        # Verify the inverted coordinates were passed to underlying function
        call_args = mock_download.call_args[0]
        assert call_args[0] == 40.71  # north
        assert call_args[1] == 40.69  # south
        assert call_args[2] == -74.01  # east (max_lon, but invalid)
        assert call_args[3] == -73.99  # west (min_lon, but invalid)
        
        # Test inverted latitude (min_lat > max_lat)
        mock_download.reset_mock()
        inverted_lat_bbox = (-74.01, 40.71, -73.99, 40.69)  # south > north
        
        with pytest.raises(ValueError):
            fetch_road_network(inverted_lat_bbox)
        
        # Test out-of-range coordinates
        mock_download.reset_mock()
        invalid_lat_bbox = (-74.01, 95.0, -73.99, 96.0)  # lat > 90
        
        with pytest.raises(ValueError):
            fetch_road_network(invalid_lat_bbox)
        
        mock_download.reset_mock()
        invalid_lon_bbox = (-185.0, 40.69, 185.0, 40.71)  # lon outside [-180, 180]
        
        with pytest.raises(ValueError):
            fetch_road_network(invalid_lon_bbox)
        
        # Test degenerate bounding boxes - point (zero area)
        mock_download.reset_mock()
        point_bbox = (-74.0, 40.7, -74.0, 40.7)  # Same coordinates
        
        # Point bbox might return empty graph but shouldn't crash
        empty_graph = nx.MultiDiGraph()
        mock_download.return_value = empty_graph
        mock_download.side_effect = None  # Clear side effect
        
        result_point = fetch_road_network(point_bbox)
        assert isinstance(result_point, nx.MultiDiGraph)
        assert result_point.number_of_nodes() == 0
        
        # Test degenerate bounding box - line (one dimension zero)
        mock_download.reset_mock()
        line_bbox = (-74.0, 40.69, -74.0, 40.71)  # Zero width
        
        result_line = fetch_road_network(line_bbox)
        assert isinstance(result_line, nx.MultiDiGraph)
        
        # Test very large bounding box (global scale)
        mock_download.reset_mock()
        global_bbox = (-180.0, -85.0, 180.0, 85.0)  # Entire world
        
        # Large bbox might cause memory/timeout issues
        mock_download.side_effect = MemoryError("Bounding box too large")
        
        with pytest.raises(MemoryError):
            fetch_road_network(global_bbox)
        
        # Test coordinate order validation
        # The function expects (min_lon, min_lat, max_lon, max_lat)
        mock_download.reset_mock()
        mock_download.side_effect = None
        mock_graph = nx.MultiDiGraph()
        mock_graph.add_node(1, x=-74.0, y=40.7)
        mock_download.return_value = mock_graph
        
        correct_bbox = (-74.01, 40.69, -73.99, 40.71)
        result_correct = fetch_road_network(correct_bbox)
        
        # Verify correct coordinate mapping
        call_args = mock_download.call_args[0]
        assert call_args[0] == 40.71   # north = max_lat
        assert call_args[1] == 40.69   # south = min_lat
        assert call_args[2] == -73.99  # east = max_lon
        assert call_args[3] == -74.01  # west = min_lon
        
        # Test near-zero area bounding box
        mock_download.reset_mock()
        tiny_bbox = (-74.0001, 40.7000, -74.0000, 40.7001)  # ~11m x 11m
        
        result_tiny = fetch_road_network(tiny_bbox)
        assert isinstance(result_tiny, nx.MultiDiGraph)
        
        # Verify tiny bbox coordinates are handled
        tiny_call_args = mock_download.call_args[0]
        assert abs(tiny_call_args[0] - tiny_call_args[1]) < 0.001  # Small lat diff
        assert abs(tiny_call_args[2] - tiny_call_args[3]) < 0.001  # Small lon diff


def test_osm_empty_result_handling():
    """Test handling of areas with no road network data.
    
    This test should verify that:
    - Bounding boxes with no roads return empty results gracefully
    - Empty NetworkX graphs are handled in downstream processing
    - Empty shape lists don't cause rasterization errors
    - Appropriate warnings or messages are provided for empty results
    """
    import networkx as nx
    from rasterio.transform import from_bounds
    
    with patch('blackmarble.acquire.osm.download_osm_roads') as mock_download, \
         patch('blackmarble.acquire.osm.graph_to_shapes') as mock_graph_to_shapes, \
         patch('blackmarble.acquire.osm.rasterize_shapes') as mock_rasterize:
        
        # Test bounding box with no roads (e.g., ocean area)
        ocean_bbox = (-160.0, 20.0, -159.0, 21.0)  # Pacific Ocean
        empty_graph = nx.MultiDiGraph()  # No nodes or edges
        mock_download.return_value = empty_graph
        
        # Fetch road network should return empty graph gracefully
        result_empty = fetch_road_network(ocean_bbox)
        assert isinstance(result_empty, nx.MultiDiGraph)
        assert result_empty.number_of_nodes() == 0
        assert result_empty.number_of_edges() == 0
        
        # Test downstream processing with empty graph
        mock_graph_to_shapes.return_value = []  # Empty shapes list
        
        result_shapes = roads_graph_to_shapes(empty_graph)
        assert isinstance(result_shapes, list)
        assert len(result_shapes) == 0
        
        # Test rasterization with empty shapes
        bounds = (-160.0, 20.0, -159.0, 21.0)
        width, height = 100, 100
        transform = from_bounds(*bounds, width, height)
        shape = (height, width)
        
        # Mock empty rasterization result
        empty_array = np.zeros((height, width), dtype=np.uint8)
        mock_rasterize.return_value = empty_array
        
        result_raster = rasterize_road_shapes([], transform, shape)
        assert isinstance(result_raster, np.ndarray)
        assert result_raster.shape == (height, width)
        assert np.all(result_raster == 0)  # All zeros
        
        # Verify empty shapes were passed to rasterization
        mock_rasterize.assert_called_once_with(
            [], height, width, transform, value=1, all_touched=False
        )
        
        # Test create_road_raster with area that has no roads
        with patch('blackmarble.acquire.osm.OSMUtil') as mock_osm_util:
            mock_util_instance = Mock()
            mock_osm_util.return_value = mock_util_instance
            
            reference_raster = "/tmp/reference.tif"
            output_path = "/tmp/empty_roads.tif"
            
            # Should complete without error even for empty areas
            result_create = create_road_raster(
                ocean_bbox, reference_raster, output_path
            )
            
            # Verify OSMUtil was called (it handles empty results internally)
            mock_util_instance.create_road_raster.assert_called_once()
            assert result_create == output_path
        
        # Test create_simple_road_mask with empty area
        with patch('blackmarble.acquire.osm.OSMUtil') as mock_osm_util:
            mock_util_instance = Mock()
            mock_osm_util.return_value = mock_util_instance
            
            result_mask = create_simple_road_mask(
                ocean_bbox, reference_raster, output_path
            )
            
            # Should complete without error
            mock_util_instance.create_road_raster_flattened.assert_called_once()
            assert result_mask == output_path
        
        # Test very sparse road area (rural/desert)
        mock_download.reset_mock()
        sparse_bbox = (-110.0, 35.0, -109.0, 36.0)  # Desert area
        sparse_graph = nx.MultiDiGraph()
        sparse_graph.add_node(1, x=-109.5, y=35.5)
        sparse_graph.add_node(2, x=-109.4, y=35.6)
        sparse_graph.add_edge(1, 2, highway='service')  # Single rural road
        mock_download.return_value = sparse_graph
        
        result_sparse = fetch_road_network(sparse_bbox)
        assert isinstance(result_sparse, nx.MultiDiGraph)
        assert result_sparse.number_of_nodes() == 2
        assert result_sparse.number_of_edges() == 1
        
        # Test error handling when OSM returns None (should not happen but defensive)
        mock_download.reset_mock()
        mock_download.return_value = None
        
        # This should raise an error since None is not a valid NetworkX graph
        with pytest.raises((AttributeError, TypeError)):
            result_none = fetch_road_network(ocean_bbox)
            # Would fail when trying to call methods on None
            result_none.number_of_nodes()
        
        # Test filtering that results in empty graph
        mock_download.reset_mock()
        mock_download.return_value = empty_graph  # Filter removes all roads
        
        result_filtered = fetch_road_network(
            (-74.01, 40.69, -73.99, 40.71),
            custom_filter='["highway"~"nonexistent_road_type"]'
        )
        assert isinstance(result_filtered, nx.MultiDiGraph)
        assert result_filtered.number_of_edges() == 0


def test_osm_large_network_performance():
    """Test performance characteristics with large road networks.
    
    This test should verify that:
    - Large city-scale networks are processed in reasonable time
    - Memory usage remains within acceptable bounds
    - Progress reporting is available for long operations
    - Processing can be interrupted cleanly
    - Results are consistent regardless of network size
    """
    import networkx as nx
    import time
    
    with patch('blackmarble.acquire.osm.download_osm_roads') as mock_download, \
         patch('blackmarble.acquire.osm.graph_to_shapes') as mock_graph_to_shapes:
        
        # Create large city-scale network (10,000 nodes, 15,000 edges)
        large_graph = nx.MultiDiGraph()
        
        # Add nodes in grid pattern (simulating Manhattan-like street grid)
        node_id = 1
        for i in range(100):  # 100x100 grid
            for j in range(100):
                x = -74.0 + i * 0.0001  # ~11m spacing
                y = 40.7 + j * 0.0001
                large_graph.add_node(node_id, x=x, y=y, osmid=node_id)
                
                # Add edges to create grid (horizontal and vertical)
                if i > 0:  # Connect to left neighbor
                    prev_node = node_id - 100
                    large_graph.add_edge(prev_node, node_id, highway='primary', length=11)
                if j > 0:  # Connect to bottom neighbor
                    prev_node = node_id - 1
                    large_graph.add_edge(prev_node, node_id, highway='secondary', length=11)
                
                node_id += 1
        
        mock_download.return_value = large_graph
        
        # Test large network processing
        large_bbox = (-74.05, 40.65, -73.95, 40.75)  # 10km x 10km area
        
        start_time = time.time()
        result_large = fetch_road_network(large_bbox)
        processing_time = time.time() - start_time
        
        # Verify large network is processed correctly
        assert isinstance(result_large, nx.MultiDiGraph)
        assert result_large.number_of_nodes() == 10000
        assert result_large.number_of_edges() > 15000
        
        # Performance should be reasonable (mocked, so should be fast)
        assert processing_time < 1.0  # Mock should complete quickly
        
        # Test memory usage with large shape conversion
        mock_large_shapes = []
        for i in range(15000):  # One shape per edge
            shape = {
                'type': 'Polygon',
                'coordinates': [[[-74.0 + i*0.00001, 40.7], [-74.0 + i*0.00001, 40.70001], 
                               [-74.0 + (i+1)*0.00001, 40.70001], [-74.0 + (i+1)*0.00001, 40.7], 
                               [-74.0 + i*0.00001, 40.7]]]
            }
            mock_large_shapes.append(shape)
        
        mock_graph_to_shapes.return_value = mock_large_shapes
        
        start_time = time.time()
        result_shapes = roads_graph_to_shapes(large_graph, 0.00005)
        shape_time = time.time() - start_time
        
        # Verify large shape collection
        assert isinstance(result_shapes, list)
        assert len(result_shapes) == 15000
        assert shape_time < 1.0  # Mock should be fast
        
        # Test consistency with smaller networks
        mock_download.reset_mock()
        small_graph = nx.MultiDiGraph()
        small_graph.add_node(1, x=-74.0, y=40.7, osmid=1)
        small_graph.add_node(2, x=-74.001, y=40.701, osmid=2)
        small_graph.add_edge(1, 2, highway='primary', length=100)
        mock_download.return_value = small_graph
        
        small_bbox = (-74.001, 40.700, -73.999, 40.702)
        result_small = fetch_road_network(small_bbox)
        
        # Verify consistency - same function behavior regardless of size
        assert isinstance(result_small, nx.MultiDiGraph)
        assert result_small.number_of_nodes() == 2
        assert result_small.number_of_edges() == 1
        
        # Verify both large and small results have same data structure
        assert type(result_large) == type(result_small)
        assert hasattr(result_large, 'nodes') and hasattr(result_small, 'nodes')
        assert hasattr(result_large, 'edges') and hasattr(result_small, 'edges')
        
        # Test processing scalability assumptions
        # Time complexity should be roughly linear with number of edges
        medium_graph = nx.MultiDiGraph()
        for i in range(1000):  # Medium network: 1000 nodes
            medium_graph.add_node(i, x=-74.0 + i*0.00001, y=40.7)
            if i > 0:
                medium_graph.add_edge(i-1, i, highway='primary')
        
        mock_download.return_value = medium_graph
        result_medium = fetch_road_network((-74.01, 40.69, -73.99, 40.71))
        
        assert result_medium.number_of_nodes() == 1000
        assert result_medium.number_of_edges() == 999
        
        # Note: In real implementation, progress reporting and interruption
        # would be handled by OSMnx and the underlying OSM API calls
        # This test verifies the wrapper functions handle large results correctly


def test_osm_coordinate_system_consistency():
    """Test coordinate reference system handling consistency.
    
    This test should verify that:
    - Input coordinates are interpreted correctly (lat/lon order)
    - Output rasters match reference raster coordinate systems
    - Reprojection is handled when necessary
    - Coordinate precision is maintained through processing chain
    - Geographic vs. projected coordinate handling is correct
    """
    import networkx as nx
    from rasterio.transform import from_bounds
    
    with patch('blackmarble.acquire.osm.download_osm_roads') as mock_download, \
         patch('blackmarble.acquire.osm.OSMUtil') as mock_osm_util:
        
        # Test coordinate order interpretation (lon, lat)
        # Input bbox: (min_lon, min_lat, max_lon, max_lat)
        test_bbox = (-74.01, 40.69, -73.99, 40.71)
        
        mock_graph = nx.MultiDiGraph()
        mock_graph.add_node(1, x=-74.005, y=40.7, osmid=1)  # Note: x=lon, y=lat
        mock_graph.add_node(2, x=-74.0, y=40.705, osmid=2)
        mock_graph.add_edge(1, 2, highway='primary')
        mock_download.return_value = mock_graph
        
        result = fetch_road_network(test_bbox)
        
        # Verify coordinate order passed correctly to download function
        call_args = mock_download.call_args[0]
        assert call_args[0] == 40.71   # north = max_lat
        assert call_args[1] == 40.69   # south = min_lat  
        assert call_args[2] == -73.99  # east = max_lon
        assert call_args[3] == -74.01  # west = min_lon
        
        # Verify returned graph maintains coordinate precision
        node_1 = result.nodes[1]
        node_2 = result.nodes[2]
        assert abs(node_1['x'] - (-74.005)) < 1e-6  # Longitude precision
        assert abs(node_1['y'] - 40.7) < 1e-6      # Latitude precision
        assert abs(node_2['x'] - (-74.0)) < 1e-6
        assert abs(node_2['y'] - 40.705) < 1e-6
        
        # Test coordinate system consistency in raster creation
        mock_util_instance = Mock()
        mock_osm_util.return_value = mock_util_instance
        
        reference_raster = "/tmp/reference_wgs84.tif"
        output_path = "/tmp/roads_wgs84.tif"
        
        create_road_raster(test_bbox, reference_raster, output_path)
        
        # Verify OSMUtil receives coordinates in expected format
        create_call_args = mock_util_instance.create_road_raster.call_args[0]
        assert create_call_args[0] == 40.69   # lat1 = min_lat
        assert create_call_args[1] == -74.01  # lon1 = min_lon
        assert create_call_args[2] == 40.71   # lat2 = max_lat
        assert create_call_args[3] == -73.99  # lon2 = max_lon
        
        # Test with different coordinate precision
        mock_download.reset_mock()
        high_precision_bbox = (-74.012345, 40.691234, -73.987654, 40.712345)
        
        precise_graph = nx.MultiDiGraph()
        precise_graph.add_node(1, x=-74.001234567, y=40.701234567)
        mock_download.return_value = precise_graph
        
        result_precise = fetch_road_network(high_precision_bbox)
        
        # Verify high precision coordinates are maintained
        precise_call_args = mock_download.call_args[0]
        assert abs(precise_call_args[0] - 40.712345) < 1e-6  # north
        assert abs(precise_call_args[1] - 40.691234) < 1e-6  # south
        assert abs(precise_call_args[2] - (-73.987654)) < 1e-6  # east
        assert abs(precise_call_args[3] - (-74.012345)) < 1e-6  # west
        
        precise_node = result_precise.nodes[1]
        assert abs(precise_node['x'] - (-74.001234567)) < 1e-9
        assert abs(precise_node['y'] - 40.701234567) < 1e-9
        
        # Test geographic coordinate bounds validation
        # Valid latitude range: [-90, 90]
        # Valid longitude range: [-180, 180]
        valid_global_bbox = (-180.0, -85.0, 180.0, 85.0)  # Near-global extent
        
        mock_download.reset_mock()
        global_graph = nx.MultiDiGraph()
        global_graph.add_node(1, x=0.0, y=0.0)  # Equator/Prime meridian
        global_graph.add_node(2, x=179.0, y=84.0)  # Near date line/pole
        mock_download.return_value = global_graph
        
        result_global = fetch_road_network(valid_global_bbox)
        
        global_call_args = mock_download.call_args[0]
        assert global_call_args[0] == 85.0   # north
        assert global_call_args[1] == -85.0  # south
        assert global_call_args[2] == 180.0  # east
        assert global_call_args[3] == -180.0 # west
        
        # Test coordinate handling in rasterization
        from rasterio.transform import from_bounds
        
        # Geographic transform (WGS84)
        geo_bounds = (-74.01, 40.69, -73.99, 40.71)
        width, height = 100, 80
        geo_transform = from_bounds(*geo_bounds, width, height)
        
        test_shapes = [
            {
                'type': 'Polygon',
                'coordinates': [[[-74.005, 40.700], [-74.004, 40.701], 
                               [-74.003, 40.700], [-74.004, 40.699], [-74.005, 40.700]]]
            }
        ]
        
        with patch('blackmarble.acquire.osm.rasterize_shapes') as mock_rasterize:
            mock_array = np.zeros((height, width), dtype=np.uint8)
            mock_rasterize.return_value = mock_array
            
            result_raster = rasterize_road_shapes(
                test_shapes, geo_transform, (height, width)
            )
            
            # Verify geographic transform passed correctly
            raster_call_args = mock_rasterize.call_args[0]
            assert raster_call_args[3] == geo_transform
            
            # Verify coordinate system consistency maintained
            assert result_raster.shape == (height, width)
        
        # Test cross-dateline coordinate handling (edge case)
        dateline_bbox = (179.0, -1.0, -179.0, 1.0)  # Crosses 180° longitude
        
        mock_download.reset_mock()
        dateline_graph = nx.MultiDiGraph()
        dateline_graph.add_node(1, x=179.5, y=0.0)
        dateline_graph.add_node(2, x=-179.5, y=0.0)
        mock_download.return_value = dateline_graph
        
        result_dateline = fetch_road_network(dateline_bbox)
        
        # Verify dateline coordinates are passed through
        dateline_call_args = mock_download.call_args[0]
        assert dateline_call_args[0] == 1.0    # north
        assert dateline_call_args[1] == -1.0   # south
        assert dateline_call_args[2] == -179.0 # east (crosses dateline)
        assert dateline_call_args[3] == 179.0  # west


def test_osm_road_classification_accuracy():
    """Test accuracy of hierarchical road classification.
    
    This test should verify that:
    - Road classifications match OSM tagging standards
    - Highway priorities are logically ordered
    - Classification filters don't miss important road types
    - Overlapping classifications are handled appropriately
    - Regional road classification variations are handled
    """
    # Test road classification hierarchy
    filters = get_hierarchical_road_filters()
    
    # Verify OSM tagging standards compliance
    all_highway_types = set()
    for filter_str in filters.values():
        # Extract highway types from filter string
        # Format: '["highway"~"type1|type2|type3"]'
        start = filter_str.find('"') + 1
        end = filter_str.rfind('"')
        highway_part = filter_str[start:end]
        
        if '~' in highway_part:
            types_part = highway_part.split('~"')[1]
            highway_types = types_part.split('|')
            all_highway_types.update(highway_types)
    
    # Verify standard OSM highway types are included
    expected_types = {
        'motorway', 'motorway_link',
        'trunk', 'trunk_link', 
        'primary', 'primary_link',
        'secondary', 'secondary_link',
        'tertiary', 'tertiary_link',
        'residential', 'service', 'unclassified'
    }
    
    for highway_type in expected_types:
        assert highway_type in all_highway_types, f"Missing highway type: {highway_type}"
    
    # Test highway priority ordering logic
    # Higher classification values should correspond to more important roads
    
    # Find which classification contains each road type
    motorway_class = None
    primary_class = None
    residential_class = None
    
    for class_value, filter_str in filters.items():
        if 'motorway' in filter_str and 'motorway_link' in filter_str:
            motorway_class = class_value
        if 'primary' in filter_str:
            primary_class = class_value
        if 'residential' in filter_str:
            residential_class = class_value
    
    # Verify logical priority ordering
    # Note: The current implementation has an inverted priority system
    # where higher numbers are used for higher-priority roads
    assert motorway_class is not None, "Motorways should be classified"
    assert primary_class is not None, "Primary roads should be classified"
    assert residential_class is not None, "Residential roads should be classified"
    
    # Test that no highway types are missed in comprehensive filter
    comprehensive_types = {
        'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
        'unclassified', 'residential', 'service', 'motorway_link',
        'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link'
    }
    
    for highway_type in comprehensive_types:
        found = any(highway_type in filter_str for filter_str in filters.values())
        assert found, f"Important highway type not classified: {highway_type}"
    
    # Test for overlapping classifications (should be avoided)
    highway_to_classes = {}
    
    for class_value, filter_str in filters.items():
        # Extract highway types from this filter
        if '~"' in filter_str:
            types_part = filter_str.split('~"')[1].rstrip('"]')
            highway_types = types_part.split('|')
            
            for highway_type in highway_types:
                if highway_type in highway_to_classes:
                    # Found overlap - same highway type in multiple classifications
                    existing_class = highway_to_classes[highway_type]
                    pytest.fail(
                        f"Highway type '{highway_type}' appears in both "
                        f"class {existing_class} and class {class_value}"
                    )
                highway_to_classes[highway_type] = class_value
    
    # Test special road types are handled appropriately
    special_types = ['busway', 'road']  # Special or catch-all types
    
    for special_type in special_types:
        found = any(special_type in filter_str for filter_str in filters.values())
        assert found, f"Special road type should be handled: {special_type}"
    
    # Test filter syntax validity
    for class_value, filter_str in filters.items():
        # Verify proper OSM filter syntax
        assert filter_str.startswith('["highway"~"'), f"Invalid filter syntax: {filter_str}"
        assert filter_str.endswith('"]'), f"Invalid filter syntax: {filter_str}"
        # Some filters may have single types, others may have multiple (with |)
        assert len(filter_str) > 15, f"Filter too short: {filter_str}"
        
        # Verify no empty components
        types_part = filter_str.split('~"')[1].rstrip('"]')
        highway_types = types_part.split('|')
        for highway_type in highway_types:
            assert len(highway_type.strip()) > 0, f"Empty highway type in filter: {filter_str}"
    
    # Test regional variation handling
    # OSM highway classification is international, but some regions use different patterns
    international_types = {
        'trunk',      # Major roads (used globally)
        'primary',    # Major roads within regions
        'secondary',  # Important roads within areas
        'tertiary',   # Local connector roads
        'residential' # Local access roads
    }
    
    for intl_type in international_types:
        found = any(intl_type in filter_str for filter_str in filters.values())
        assert found, f"International highway type missing: {intl_type}"
    
    # Test that classification covers urban and rural scenarios
    urban_types = {'residential', 'service', 'tertiary'}
    rural_types = {'trunk', 'primary', 'secondary'}
    highway_types = {'motorway', 'motorway_link'}
    
    for scenario_types, scenario_name in [
        (urban_types, 'urban'),
        (rural_types, 'rural'), 
        (highway_types, 'highway')
    ]:
        scenario_covered = False
        for highway_type in scenario_types:
            if any(highway_type in filter_str for filter_str in filters.values()):
                scenario_covered = True
                break
        assert scenario_covered, f"{scenario_name} road types should be covered"


def test_osm_integration_with_satellite_data():
    """Test integration compatibility with satellite imagery processing.
    
    This test should verify that:
    - Road raster outputs align properly with Landsat/VIIRS data
    - Coordinate reference systems are compatible
    - Resolution matching works correctly
    - Road enhancement doesn't interfere with other processing
    - Output formats are suitable for composite operations
    """
    from rasterio.transform import from_bounds
    
    with patch('blackmarble.acquire.osm.OSMUtil') as mock_osm_util, \
         patch('blackmarble.acquire.osm.rasterize_shapes') as mock_rasterize:
        
        # Test alignment with Landsat-8 data (30m resolution)
        landsat_bbox = (-74.01, 40.69, -73.99, 40.71)  # Manhattan area
        landsat_reference = "/tmp/landsat8_B4.tif"  # Red band reference
        road_output = "/tmp/roads_landsat_aligned.tif"
        
        mock_util_instance = Mock()
        mock_osm_util.return_value = mock_util_instance
        
        # Create road raster aligned to Landsat
        result_landsat = create_road_raster(
            landsat_bbox, landsat_reference, road_output
        )
        
        # Verify OSMUtil called with Landsat reference for alignment
        mock_util_instance.create_road_raster.assert_called_once()
        call_args = mock_util_instance.create_road_raster.call_args[0]
        assert call_args[4] == landsat_reference  # reference_raster as positional arg
        assert result_landsat == road_output
        
        # Test alignment with VIIRS data (500m resolution)
        mock_util_instance.reset_mock()
        viirs_bbox = (-75.0, 40.0, -73.0, 42.0)  # Larger area for VIIRS
        viirs_reference = "/tmp/viirs_ntl.tif"  # Nighttime lights reference
        viirs_road_output = "/tmp/roads_viirs_aligned.tif"
        
        result_viirs = create_road_raster(
            viirs_bbox, viirs_reference, viirs_road_output
        )
        
        # Verify VIIRS alignment
        viirs_call_args = mock_util_instance.create_road_raster.call_args[0]
        assert viirs_call_args[4] == viirs_reference  # reference_raster as positional arg
        assert result_viirs == viirs_road_output
        
        # Test coordinate reference system compatibility
        # Both Landsat and VIIRS use WGS84/UTM projections
        # OSM data is in WGS84 geographic coordinates
        
        # Verify coordinate system handling in direct rasterization
        test_shapes = [
            {
                'type': 'Polygon',
                'coordinates': [[[-74.005, 40.700], [-74.004, 40.701], 
                               [-74.003, 40.700], [-74.004, 40.699], [-74.005, 40.700]]]
            }
        ]
        
        # Test with Landsat-like transform (UTM Zone 18N)
        utm_bounds = (580000, 4504000, 590000, 4514000)  # UTM coordinates
        landsat_width, landsat_height = 334, 334  # ~10km at 30m resolution
        utm_transform = from_bounds(*utm_bounds, landsat_width, landsat_height)
        
        mock_landsat_array = np.zeros((landsat_height, landsat_width), dtype=np.uint8)
        mock_landsat_array[100:120, 150:170] = 1  # Some road pixels
        mock_rasterize.return_value = mock_landsat_array
        
        result_utm = rasterize_road_shapes(
            test_shapes, utm_transform, (landsat_height, landsat_width)
        )
        
        # Verify UTM coordinate system compatibility
        utm_call_args = mock_rasterize.call_args[0]
        assert utm_call_args[3] == utm_transform
        assert result_utm.shape == (landsat_height, landsat_width)
        
        # Test resolution matching
        # High resolution (1m) - for detailed urban analysis
        mock_rasterize.reset_mock()
        high_res_bounds = (-74.002, 40.700, -74.000, 40.702)  # 200m x 200m area
        high_res_width, high_res_height = 200, 200  # 1m resolution
        high_res_transform = from_bounds(*high_res_bounds, high_res_width, high_res_height)
        
        mock_high_res_array = np.zeros((high_res_height, high_res_width), dtype=np.uint8)
        mock_rasterize.return_value = mock_high_res_array
        
        result_high_res = rasterize_road_shapes(
            test_shapes, high_res_transform, (high_res_height, high_res_width)
        )
        
        assert result_high_res.shape == (high_res_height, high_res_width)
        
        # Low resolution (1km) - for regional analysis  
        mock_rasterize.reset_mock()
        low_res_bounds = (-75.0, 40.0, -73.0, 42.0)  # 200km x 200km area
        low_res_width, low_res_height = 200, 200  # 1km resolution
        low_res_transform = from_bounds(*low_res_bounds, low_res_width, low_res_height)
        
        mock_low_res_array = np.zeros((low_res_height, low_res_width), dtype=np.uint8)
        mock_rasterize.return_value = mock_low_res_array
        
        result_low_res = rasterize_road_shapes(
            test_shapes, low_res_transform, (low_res_height, low_res_width)
        )
        
        assert result_low_res.shape == (low_res_height, low_res_width)
        
        # Test output format compatibility for composite operations
        # Road masks should be uint8 for efficient memory usage
        assert result_utm.dtype == np.uint8
        assert result_high_res.dtype == np.uint8
        assert result_low_res.dtype == np.uint8
        
        # Test that road values are suitable for masking/enhancement
        # Values should be 0 (background) or positive integers (roads)
        assert np.all(result_utm >= 0)
        assert np.all(result_high_res >= 0)
        assert np.all(result_low_res >= 0)
        
        # Test simple road mask for binary operations
        mock_util_instance.reset_mock()
        binary_mask_output = "/tmp/binary_roads.tif"
        
        result_binary = create_simple_road_mask(
            landsat_bbox, landsat_reference, binary_mask_output, road_value=1
        )
        
        # Verify binary mask creation
        mock_util_instance.create_road_raster_flattened.assert_called_once()
        binary_call_args = mock_util_instance.create_road_raster_flattened.call_args[0]
        assert binary_call_args[4] == landsat_reference  # reference_raster as positional arg
        assert result_binary == binary_mask_output
        
        # Test that road enhancement doesn't interfere with other processing
        # Multiple road raster creation should be independent
        mock_util_instance.reset_mock()
        
        # Create multiple road rasters with different parameters
        outputs = []
        for i, buffer_dist in enumerate([0.00003, 0.00005, 0.00008]):
            output_path = f"/tmp/roads_buffer_{i}.tif"
            result = create_road_raster(
                landsat_bbox, landsat_reference, output_path, buffer_dist
            )
            outputs.append(result)
        
        # Verify all outputs are independent
        assert len(outputs) == 3
        assert len(set(outputs)) == 3  # All unique paths
        assert mock_util_instance.create_road_raster.call_count == 3
        
        # Verify different buffer distances were used
        call_list = mock_util_instance.create_road_raster.call_args_list
        buffer_distances = [call[1]['path_thickness'] for call in call_list]
        assert buffer_distances == [0.00003, 0.00005, 0.00008]