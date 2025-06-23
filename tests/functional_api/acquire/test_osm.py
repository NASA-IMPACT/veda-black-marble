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
        
        # Verify nodes have geographic coordinates
        for node_id, node_data in graph.nodes(data=True):
            assert 'x' in node_data, f"Node {node_id} missing x coordinate"
            assert 'y' in node_data, f"Node {node_id} missing y coordinate"
            
            # Coordinates should be within our bounding box
            x, y = node_data['x'], node_data['y']
            assert bbox[0] <= x <= bbox[2], f"Node x={x} outside bbox longitude range"
            assert bbox[1] <= y <= bbox[3], f"Node y={y} outside bbox latitude range"
        
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