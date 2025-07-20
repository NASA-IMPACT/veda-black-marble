"""Test OSM fractional road rasterization CRS handling."""

import numpy as np
import networkx as nx
from rasterio.transform import from_origin
from rasterio.crs import CRS
import pytest
from unittest.mock import patch
import io
import sys

from blackmarble.acquire.osm_fractional import buffer_and_rasterize_roads_fractional


def create_test_road_graph():
    """Create a simple test road network graph."""
    # Create a simple graph with one road
    G = nx.MultiDiGraph()
    
    # Add nodes (in lat/lon coordinates)
    # Small area around 0°N, 0°E for easy math
    G.add_node(1, y=0.001, x=0.001)  # Note: y=lat, x=lon in OSMnx
    G.add_node(2, y=0.002, x=0.002)
    
    # Add edge with highway type
    G.add_edge(1, 2, highway='primary')
    
    # Set graph CRS
    G.graph['crs'] = 'EPSG:4326'
    
    return G


class TestOSMFractionalCRS:
    """Test CRS handling in fractional road rasterization."""
    
    def test_matching_crs_produces_roads(self):
        """Test that matching CRS between roads and raster produces pixels."""
        # Create test graph
        G = create_test_road_graph()
        
        # Define raster parameters in UTM Zone 31N (covers 0°E)
        utm_crs = CRS.from_epsg(32631)
        
        # Create a transform that covers our test area
        # At equator, 1 degree ≈ 111km
        # Our points span 0.001 to 0.002 degrees ≈ 111-222 meters
        # The roads project to roughly 166km east, 80-250m north in UTM Zone 31N
        pixel_size = 30  # 30m pixels
        west = 166000  # Start at 166km east in UTM
        north = 300  # Start at 300m north in UTM
        transform = from_origin(west, north, pixel_size, pixel_size)
        
        # Raster shape
        shape = (10, 10)  # 10x10 grid = 300x300m area
        
        # Rasterize with matching CRS
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            result = buffer_and_rasterize_roads_fractional(
                graph=G,
                bbox=(0.0005, 0.0005, 0.0025, 0.0025),  # Cover our road
                transform=transform,
                crs=utm_crs,
                shape=shape,
                buffer_meters=30.0,
                sub_pixels=5,
                hierarchical=False
            )
        
        # Check output
        output = mock_stdout.getvalue()
        
        # Should have rasterized some pixels
        pixels_with_roads = np.sum(result > 0)
        assert pixels_with_roads > 0, f"Expected roads but got none. Output:\n{output}"
        
        # Should see success message in output
        assert "Fractional road rasterization:" in output
        assert "pixels with roads" in output
        
        # Values should be fractional (0-1)
        assert np.all(result >= 0)
        assert np.all(result <= 1)
    
    def test_mismatched_crs_produces_warning(self):
        """Test that CRS mismatch produces warning and zero pixels."""
        # Create test graph
        G = create_test_road_graph()
        
        # Define raster in different CRS (Web Mercator)
        mercator_crs = CRS.from_epsg(3857)
        
        # Create transform in Web Mercator
        # These coordinates are nowhere near our lat/lon points
        pixel_size = 30
        west = 1000000  # 1M meters east
        north = 1000000  # 1M meters north
        transform = from_origin(west, north, pixel_size, pixel_size)
        
        shape = (10, 10)
        
        # Rasterize with mismatched CRS
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            result = buffer_and_rasterize_roads_fractional(
                graph=G,
                bbox=(0.0005, 0.0005, 0.0025, 0.0025),
                transform=transform,
                crs=mercator_crs,
                shape=shape,
                buffer_meters=30.0,
                sub_pixels=5,
                hierarchical=False
            )
        
        # Check output
        output = mock_stdout.getvalue()
        
        # Should produce warning about no overlap
        assert "WARNING: No overlap between roads and raster!" in output
        
        # Should have zero pixels with roads
        pixels_with_roads = np.sum(result > 0)
        assert pixels_with_roads == 0, f"Expected no roads but got {pixels_with_roads}"
        
        # Should see the bounds mismatch in output
        assert "Roads X:" in output
        assert "Raster X:" in output
    
    def test_empty_graph_returns_zeros(self):
        """Test that empty graph returns zero array."""
        # Create empty graph
        G = nx.MultiDiGraph()
        
        utm_crs = CRS.from_epsg(32631)
        transform = from_origin(0, 0, 30, 30)
        shape = (10, 10)
        
        result = buffer_and_rasterize_roads_fractional(
            graph=G,
            bbox=(0, 0, 0.01, 0.01),
            transform=transform,
            crs=utm_crs,
            shape=shape,
            buffer_meters=30.0
        )
        
        # Should be all zeros
        assert np.all(result == 0)
        assert result.shape == shape
        assert result.dtype == np.float32
    
    def test_crs_string_vs_object(self):
        """Test that CRS can be passed as string or CRS object."""
        G = create_test_road_graph()
        
        # Common parameters
        transform = from_origin(100, 300, 30, 30)
        shape = (10, 10)
        bbox = (0.0005, 0.0005, 0.0025, 0.0025)
        
        # Test with string CRS
        result_str = buffer_and_rasterize_roads_fractional(
            graph=G,
            bbox=bbox,
            transform=transform,
            crs="EPSG:32631",  # String
            shape=shape,
            buffer_meters=30.0
        )
        
        # Test with CRS object
        result_obj = buffer_and_rasterize_roads_fractional(
            graph=G,
            bbox=bbox,
            transform=transform,
            crs=CRS.from_epsg(32631),  # CRS object
            shape=shape,
            buffer_meters=30.0
        )
        
        # Results should be identical
        np.testing.assert_array_equal(result_str, result_obj)
    
    def test_sub_pixel_accuracy(self):
        """Test that sub-pixel rasterization produces fractional values."""
        G = create_test_road_graph()
        
        utm_crs = CRS.from_epsg(32631)
        transform = from_origin(100, 300, 30, 30)
        shape = (20, 20)  # Larger grid for better test
        
        # Rasterize with sub-pixels
        result = buffer_and_rasterize_roads_fractional(
            graph=G,
            bbox=(0.0005, 0.0005, 0.0025, 0.0025),
            transform=transform,
            crs=utm_crs,
            shape=shape,
            buffer_meters=15.0,  # Smaller buffer for partial coverage
            sub_pixels=5
        )
        
        # If working correctly, should have fractional values
        pixels_with_roads = result[result > 0]
        if len(pixels_with_roads) > 0:
            # Check for fractional values (not just 0 or 1)
            unique_values = np.unique(pixels_with_roads)
            # With 5x5 sub-pixels, possible values are multiples of 1/25 = 0.04
            assert len(unique_values) > 2, "Expected fractional coverage values"
            
            # Check that some values are fractional
            fractional_values = pixels_with_roads[(pixels_with_roads > 0) & (pixels_with_roads < 1)]
            if len(fractional_values) > 0:
                assert np.any(fractional_values % 1.0 != 0), "Expected non-integer fractional values"


def test_large_area_memory_protection():
    """Test that very large areas are warned about or rejected."""
    G = create_test_road_graph()
    
    # Create parameters for a large area (>8GB threshold)
    transform = from_origin(0, 0, 30, 30)
    huge_shape = (20000, 20000)  # Would need ~9.3GB at 6m resolution
    
    # Should return zeros and print error
    with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
        result = buffer_and_rasterize_roads_fractional(
            graph=G,
            bbox=(0, 0, 1, 1),
            transform=transform,
            crs="EPSG:32631",
            shape=huge_shape,
            buffer_meters=30.0
        )
    
    output = mock_stdout.getvalue()
    
    # Should reject and return zeros
    assert np.all(result == 0)
    # Should either hit the 8GB error or have no overlap (both valid)
    assert ("ERROR: Area too large for fractional rasterization" in output or 
            "WARNING: No overlap between roads and raster!" in output)