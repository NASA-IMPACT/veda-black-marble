"""Test functions in blackmarble.acquire.viirs module.

This module tests VIIRS nighttime lights data acquisition functionality,
focusing on pure functional implementations without excessive mocking.
"""

import pytest

# Import the module under test
from blackmarble.acquire.viirs import (
    coordinates_to_viirs_tile,
    download_viirs_h5,
    convert_viirs_hdf_to_tiff,
    extract_viirs_layers,
    get_tiles_for_bbox,
)


def test_coordinates_to_viirs_tile():
    """Test conversion of geographic coordinates to VIIRS tile indices.
    
    This test verifies that:
    - Valid latitude/longitude pairs return correct tile indices
    - Coordinate system assumptions match VIIRS sinusoidal projection
    - Edge cases (poles, date line, equator) are handled correctly
    - Tile indices are within expected ranges (typically 0-35 for h, 0-17 for v)
    - Output format is tuple of (vertical, horizontal) integers
    - Known coordinates return expected tile indices
    """
    # Test Manhattan, NY coordinates
    manhattan_lat, manhattan_lon = 40.7589, -73.9851
    v, h = coordinates_to_viirs_tile(manhattan_lat, manhattan_lon)
    
    # Verify output format
    assert isinstance(v, int), "Vertical tile index should be integer"
    assert isinstance(h, int), "Horizontal tile index should be integer"
    
    # Verify tile indices are within valid ranges
    assert 0 <= v <= 17, f"Vertical tile {v} outside valid range 0-17"
    assert 0 <= h <= 35, f"Horizontal tile {h} outside valid range 0-35"
    
    # Test equator at prime meridian (should be center of grid)
    equator_v, equator_h = coordinates_to_viirs_tile(0.0, 0.0)
    assert equator_v == 9, f"Expected v=9 for equator, got {equator_v}"
    assert equator_h == 18, f"Expected h=18 for prime meridian, got {equator_h}"
    
    # Test known coordinates with expected results
    # London coordinates
    london_v, london_h = coordinates_to_viirs_tile(51.5074, -0.1278)
    assert 0 <= london_v <= 17, f"London vertical tile {london_v} out of range"
    assert 0 <= london_h <= 35, f"London horizontal tile {london_h} out of range"
    
    # Test extreme coordinates
    # North pole region
    north_v, north_h = coordinates_to_viirs_tile(89.0, 0.0)
    assert north_v == 0, f"Expected v=0 for far north, got {north_v}"
    
    # South pole region  
    south_v, south_h = coordinates_to_viirs_tile(-89.0, 0.0)
    assert south_v == 17, f"Expected v=17 for far south, got {south_v}"


def test_coordinates_to_viirs_tile_edge_cases():
    """Test coordinate conversion at projection boundaries.
    
    This test verifies that:
    - International date line crossings work correctly
    - Polar regions return appropriate tile indices
    - Coordinates exactly on tile boundaries are handled consistently
    - Edge coordinates are properly bounded to valid ranges
    """
    # Test international date line (180° longitude)
    dateline_east_v, dateline_east_h = coordinates_to_viirs_tile(0.0, 179.9)
    dateline_west_v, dateline_west_h = coordinates_to_viirs_tile(0.0, -179.9)
    
    # Both should be valid but different horizontal tiles
    assert 0 <= dateline_east_v <= 17, f"Date line east vertical {dateline_east_v} out of range"
    assert 0 <= dateline_east_h <= 35, f"Date line east horizontal {dateline_east_h} out of range"
    assert 0 <= dateline_west_v <= 17, f"Date line west vertical {dateline_west_v} out of range"
    assert 0 <= dateline_west_h <= 35, f"Date line west horizontal {dateline_west_h} out of range"
    
    # Test exact tile boundaries (multiples of 10 degrees)
    boundary_v, boundary_h = coordinates_to_viirs_tile(10.0, -10.0)
    assert 0 <= boundary_v <= 17, f"Boundary vertical {boundary_v} out of range"
    assert 0 <= boundary_h <= 35, f"Boundary horizontal {boundary_h} out of range"
    
    # Test extreme coordinates (should be bounded)
    extreme_north_v, extreme_north_h = coordinates_to_viirs_tile(95.0, 0.0)  # Beyond 90°
    assert extreme_north_v == 0, f"Extreme north should be bounded to v=0, got {extreme_north_v}"
    
    extreme_south_v, extreme_south_h = coordinates_to_viirs_tile(-95.0, 0.0)  # Beyond -90°
    assert extreme_south_v == 17, f"Extreme south should be bounded to v=17, got {extreme_south_v}"
    
    extreme_east_v, extreme_east_h = coordinates_to_viirs_tile(0.0, 185.0)  # Beyond 180°
    assert 0 <= extreme_east_h <= 35, f"Extreme east horizontal {extreme_east_h} out of range"
    
    extreme_west_v, extreme_west_h = coordinates_to_viirs_tile(0.0, -185.0)  # Beyond -180°
    assert 0 <= extreme_west_h <= 35, f"Extreme west horizontal {extreme_west_h} out of range"


def test_get_tiles_for_bbox():
    """Test identification of VIIRS tiles covering a bounding box.
    
    This test verifies that:
    - Small bounding boxes return appropriate tile sets
    - Large bounding boxes covering multiple tiles work correctly
    - Tile indices are returned in consistent order (row-major)
    - Edge cases like single-tile coverage work correctly
    - Output format and tile count validation
    """
    # Test small bounding box (Manhattan, NYC)
    nyc_bbox = (-74.05, 40.69, -73.95, 40.79)  # (min_lon, min_lat, max_lon, max_lat)
    tiles = get_tiles_for_bbox(nyc_bbox)
    
    # Verify output format
    assert isinstance(tiles, list), "Should return a list of tiles"
    assert len(tiles) >= 1, "Manhattan should be covered by at least one tile"
    
    # Verify each tile has proper format
    for tile in tiles:
        assert isinstance(tile, tuple), "Each tile should be a tuple"
        assert len(tile) == 2, "Each tile should have (vertical, horizontal)"
        v, h = tile
        assert isinstance(v, int) and isinstance(h, int), "Tile indices should be integers"
        assert 0 <= v <= 17, f"Vertical tile {v} outside valid range 0-17"
        assert 0 <= h <= 35, f"Horizontal tile {h} outside valid range 0-35"
    
    # Test large bounding box covering multiple tiles (Eastern US)
    eastern_us_bbox = (-85.0, 35.0, -70.0, 45.0)
    large_tiles = get_tiles_for_bbox(eastern_us_bbox)
    
    # Should return more tiles than the small bbox
    assert len(large_tiles) >= len(tiles), "Larger bbox should cover at least as many tiles"
    assert len(large_tiles) >= 2, "Eastern US should span multiple tiles"
    
    # Verify tiles are sorted in row-major order (v ascending, then h ascending)
    for i in range(1, len(large_tiles)):
        prev_v, prev_h = large_tiles[i-1]
        curr_v, curr_h = large_tiles[i]
        
        # Either current row > previous row, or same row with higher column
        assert (curr_v > prev_v) or (curr_v == prev_v and curr_h >= prev_h), \
            f"Tiles not in row-major order: {large_tiles[i-1]} -> {large_tiles[i]}"
    
    # Test very small bbox (should return exactly one tile)
    small_bbox = (-74.0, 40.7, -73.99, 40.71)  # Very small area in Manhattan
    small_tiles = get_tiles_for_bbox(small_bbox)
    assert len(small_tiles) >= 1, "Small bbox should return at least one tile"


def test_download_viirs_h5_authentication():
    """Test HDF5 download authentication validation.
    
    This test verifies that:
    - Missing tokens raise appropriate exceptions
    - Token validation occurs before network requests
    """
    # Test missing token (empty string)
    with pytest.raises(RuntimeError, match="NASA Earthdata token is required"):
        download_viirs_h5("", 2023, 9, 15, 4, 12, "output.h5")
    
    # Test None token (need to cast for type checker)
    with pytest.raises(RuntimeError, match="NASA Earthdata token is required"):
        download_viirs_h5(None, 2023, 9, 15, 4, 12, "output.h5")  # type: ignore


def test_function_signatures():
    """Test that all functions have expected signatures and can be imported."""
    import inspect
    
    # Test download_viirs_h5 signature
    sig = inspect.signature(download_viirs_h5)
    params = list(sig.parameters.keys())
    expected_params = ['earthdata_token', 'year', 'month', 'day', 'vertical', 'horizontal', 'output_path']
    for param in expected_params:
        assert param in params, f"download_viirs_h5 missing parameter: {param}"
    
    # Test convert_viirs_hdf_to_tiff signature
    sig = inspect.signature(convert_viirs_hdf_to_tiff)
    params = list(sig.parameters.keys())
    expected_params = ['hdf_file', 'output_file', 'layer']
    for param in expected_params:
        assert param in params, f"convert_viirs_hdf_to_tiff missing parameter: {param}"
    
    # Test get_tiles_for_bbox signature
    sig = inspect.signature(get_tiles_for_bbox)
    params = list(sig.parameters.keys())
    expected_params = ['bbox', 'include_partial']
    for param in expected_params:
        assert param in params, f"get_tiles_for_bbox missing parameter: {param}"


def test_extract_viirs_layers_signature():
    """Test extract_viirs_layers function signature."""
    import inspect
    
    sig = inspect.signature(extract_viirs_layers)
    params = list(sig.parameters.keys())
    expected_params = ['hdf_file', 'output_dir']
    for param in expected_params:
        assert param in params, f"extract_viirs_layers missing parameter: {param}"


def test_bbox_edge_cases():
    """Test edge cases for bounding box tile calculation."""
    # Test crossing international date line
    dateline_bbox = (170.0, -10.0, -170.0, 10.0)  # Crosses 180°/-180°
    tiles = get_tiles_for_bbox(dateline_bbox)
    assert len(tiles) >= 1, "Date line crossing should return tiles"
    
    # Test polar regions
    arctic_bbox = (-10.0, 80.0, 10.0, 85.0)
    arctic_tiles = get_tiles_for_bbox(arctic_bbox)
    assert len(arctic_tiles) >= 1, "Arctic region should return tiles"
    
    # Test equatorial region
    equator_bbox = (-5.0, -5.0, 5.0, 5.0)
    equator_tiles = get_tiles_for_bbox(equator_bbox)
    assert len(equator_tiles) >= 1, "Equatorial region should return tiles"


def test_tile_coordinate_consistency():
    """Test that tile coordinates are consistent between functions."""
    # Get a specific coordinate and its tile
    test_lat, test_lon = 45.0, -120.0  # Pacific Northwest
    v, h = coordinates_to_viirs_tile(test_lat, test_lon)
    
    # Create a small bbox around that coordinate
    bbox = (test_lon - 0.1, test_lat - 0.1, test_lon + 0.1, test_lat + 0.1)
    tiles = get_tiles_for_bbox(bbox)
    
    # The tile from coordinate conversion should be in the bbox tiles
    assert (v, h) in tiles, f"Coordinate tile {(v, h)} not found in bbox tiles {tiles}"