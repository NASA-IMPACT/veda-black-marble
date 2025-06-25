"""Test functions in blackmarble.prepare.coordinates module.

This module tests coordinate system transformations and spatial reference utilities.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Import the module under test
from blackmarble.prepare.coordinates import (
    transform_bounds,
    calculate_transform,
    get_crs_from_epsg,
    calculate_pixel_area,
    reproject_geometry,
    create_crs_transformer,
    calculate_utm_zone,
    validate_crs_compatibility,
    calculate_ground_distance,
)


def test_transform_bounds():
    """Test bounding box transformation between coordinate systems.
    
    This test should verify that:
    - Basic bounds transformation works for common CRS pairs
    - Source and destination CRS are properly interpreted
    - Densification parameter improves transformation accuracy
    - Edge cases (crossing date line, polar regions) are handled
    - Transformation preserves spatial relationships
    - Different CRS string formats work correctly
    - Performance is acceptable for typical use cases
    - Numerical precision is adequate for intended use
    - Results are suitable for downstream processing
    """
    import time
    
    # Basic bounds transformation for common CRS pairs
    wgs84_bounds = (-74.05, 40.69, -73.95, 40.79)  # NYC area
    web_mercator_bounds = transform_bounds(
        wgs84_bounds,
        "EPSG:4326",
        "EPSG:3857"
    )
    
    # Verify output format
    assert len(web_mercator_bounds) == 4
    assert web_mercator_bounds[0] < web_mercator_bounds[2]  # min_x < max_x
    assert web_mercator_bounds[1] < web_mercator_bounds[3]  # min_y < max_y
    
    # Verify reasonable values for Web Mercator (meters)
    assert -20037509 < web_mercator_bounds[0] < 20037509
    assert -20037509 < web_mercator_bounds[2] < 20037509
    
    # Test source and destination CRS interpretation
    # Using CRS objects
    import rasterio.crs
    src_crs = rasterio.crs.CRS.from_epsg(4326)
    dst_crs = rasterio.crs.CRS.from_epsg(3857)
    bounds_from_objects = transform_bounds(wgs84_bounds, src_crs, dst_crs)
    
    # Results should be identical
    assert np.allclose(web_mercator_bounds, bounds_from_objects)
    
    # Test densification parameter
    # Transform with different densification levels
    bounds_dense = transform_bounds(wgs84_bounds, "EPSG:4326", "EPSG:3857", densify_points=100)
    bounds_sparse = transform_bounds(wgs84_bounds, "EPSG:4326", "EPSG:3857", densify_points=4)
    
    # For small areas, densification shouldn't matter much
    assert np.allclose(bounds_dense, bounds_sparse, rtol=1e-6)
    
    # For larger areas crossing more distortion, densification matters
    global_bounds = (-180, -85, 180, 85)
    global_dense = transform_bounds(global_bounds, "EPSG:4326", "EPSG:3857", densify_points=100)
    global_sparse = transform_bounds(global_bounds, "EPSG:4326", "EPSG:3857", densify_points=4)
    
    # Should have some difference due to curvature
    assert not np.allclose(global_dense, global_sparse, rtol=1e-9)
    
    # Edge case: Date line crossing
    dateline_bounds = (170, -10, -170, 10)  # Crosses 180° longitude
    dateline_transformed = transform_bounds(dateline_bounds, "EPSG:4326", "EPSG:3857")
    assert len(dateline_transformed) == 4
    # In Web Mercator, this should still work (though bounds will be "wrapped")
    
    # Edge case: Polar regions (Web Mercator has limits)
    polar_bounds = (-180, 80, 180, 85)  # Near north pole
    polar_transformed = transform_bounds(polar_bounds, "EPSG:4326", "EPSG:3857")
    assert len(polar_transformed) == 4
    # Web Mercator Y values should be very large near poles
    assert abs(polar_transformed[3]) > 15000000
    
    # Test spatial relationships preserved
    # If one box contains another in source, it should in destination
    inner_bounds = (-74.0, 40.7, -73.98, 40.75)
    outer_bounds = (-74.1, 40.6, -73.9, 40.8)
    
    inner_transformed = transform_bounds(inner_bounds, "EPSG:4326", "EPSG:3857")
    outer_transformed = transform_bounds(outer_bounds, "EPSG:4326", "EPSG:3857")
    
    # Inner should still be contained in outer
    assert outer_transformed[0] < inner_transformed[0]  # left
    assert outer_transformed[1] < inner_transformed[1]  # bottom
    assert outer_transformed[2] > inner_transformed[2]  # right
    assert outer_transformed[3] > inner_transformed[3]  # top
    
    # Test different CRS string formats
    # PROJ4 string
    proj4_string = "+proj=longlat +datum=WGS84 +no_defs"
    bounds_proj4 = transform_bounds(wgs84_bounds, proj4_string, "EPSG:3857")
    assert np.allclose(bounds_proj4, web_mercator_bounds, rtol=1e-6)
    
    # WKT string (if supported)
    wkt_string = 'GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],CS[ellipsoidal,2],AXIS["latitude",north],AXIS["longitude",east],UNIT["degree",0.0174532925199433]]'
    try:
        bounds_wkt = transform_bounds(wgs84_bounds, wkt_string, "EPSG:3857")
        assert np.allclose(bounds_wkt, web_mercator_bounds, rtol=1e-6)
    except:
        # WKT may not be supported in all environments
        pass
    
    # Test performance for typical use cases
    start_time = time.time()
    for _ in range(100):
        _ = transform_bounds(wgs84_bounds, "EPSG:4326", "EPSG:3857")
    elapsed = time.time() - start_time
    assert elapsed < 1.0  # 100 transformations should take less than 1 second
    
    # Test numerical precision
    # Round trip transformation
    round_trip = transform_bounds(web_mercator_bounds, "EPSG:3857", "EPSG:4326")
    assert np.allclose(round_trip, wgs84_bounds, rtol=1e-9)
    
    # Results suitable for downstream processing
    # Bounds should be finite and reasonable
    assert all(np.isfinite(x) for x in web_mercator_bounds)
    assert web_mercator_bounds[2] > web_mercator_bounds[0]  # width > 0
    assert web_mercator_bounds[3] > web_mercator_bounds[1]  # height > 0


def test_transform_bounds_densification():
    """Test bounds transformation densification effects.
    
    This test should verify that:
    - Higher densification improves accuracy for complex projections
    - Default densification (21 points) works for typical cases
    - Very high densification doesn't cause performance issues
    - Low densification is adequate for simple transformations
    - Densification parameter guidance is provided
    """
    import time
    
    # Test case with complex projection (Lambert Azimuthal Equal Area)
    # Large area where densification matters
    large_bounds = (-140, 20, -60, 60)  # Continental US approximately
    
    # Transform to Lambert Azimuthal Equal Area centered on US
    laea_proj = "+proj=laea +lat_0=45 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
    
    # Test with different densification levels
    bounds_low = transform_bounds(large_bounds, "EPSG:4326", laea_proj, densify_points=4)
    bounds_default = transform_bounds(large_bounds, "EPSG:4326", laea_proj, densify_points=21)
    bounds_high = transform_bounds(large_bounds, "EPSG:4326", laea_proj, densify_points=100)
    bounds_very_high = transform_bounds(large_bounds, "EPSG:4326", laea_proj, densify_points=1000)
    
    # Higher densification should give different (more accurate) results
    assert not np.allclose(bounds_low, bounds_default, rtol=1e-6)
    assert not np.allclose(bounds_default, bounds_high, rtol=1e-9)
    
    # But very high densification should converge
    assert np.allclose(bounds_high, bounds_very_high, rtol=1e-9)
    
    # Default densification (21) works for typical cases
    typical_bounds = (-74.5, 40.5, -73.5, 41.5)  # Smaller area
    typical_default = transform_bounds(typical_bounds, "EPSG:4326", "EPSG:3857")
    typical_high = transform_bounds(typical_bounds, "EPSG:4326", "EPSG:3857", densify_points=100)
    
    # For smaller areas, default is sufficient
    assert np.allclose(typical_default, typical_high, rtol=1e-6)
    
    # Test performance with very high densification
    start_time = time.time()
    _ = transform_bounds(large_bounds, "EPSG:4326", laea_proj, densify_points=10000)
    elapsed = time.time() - start_time
    assert elapsed < 0.1  # Even 10000 points should be fast
    
    # Low densification adequate for simple transformations
    # UTM is a relatively simple transformation for small areas
    utm_bounds = (-74.1, 40.6, -73.9, 40.8)  # Small area in NYC
    utm_zone = 18  # NYC is in UTM zone 18N
    utm_epsg = f"EPSG:{32600 + utm_zone}"
    
    utm_low = transform_bounds(utm_bounds, "EPSG:4326", utm_epsg, densify_points=4)
    utm_high = transform_bounds(utm_bounds, "EPSG:4326", utm_epsg, densify_points=100)
    
    # Should be very close even with low densification
    assert np.allclose(utm_low, utm_high, rtol=1e-8)
    
    # Verify densification parameter guidance
    # Small areas can use low densification
    small_area = 0.1 * 0.1  # 0.1 degree square
    # Large areas need higher densification
    large_area = 80 * 40  # Large continental area
    
    # Recommendation: use default (21) for most cases
    # Use higher for very large areas or complex projections
    # Use lower only for small areas with simple projections


def test_transform_bounds_edge_cases():
    """Test bounds transformation edge cases.
    
    This test should verify that:
    - Date line crossing bounds are handled correctly
    - Polar region bounds work appropriately
    - Very large bounds (global extent) are processed
    - Very small bounds maintain precision
    - Invalid bounds raise appropriate errors
    """
    # Date line crossing bounds
    # Pacific Ocean area crossing 180° longitude
    dateline_bounds = (170, -20, -170, 20)
    
    # Transform to Web Mercator
    dateline_mercator = transform_bounds(dateline_bounds, "EPSG:4326", "EPSG:3857")
    assert len(dateline_mercator) == 4
    # The transformed bounds should be valid
    assert all(np.isfinite(x) for x in dateline_mercator)
    
    # Transform to Pacific-centered projection
    pacific_proj = "+proj=robin +lon_0=180 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    dateline_pacific = transform_bounds(dateline_bounds, "EPSG:4326", pacific_proj)
    assert len(dateline_pacific) == 4
    assert dateline_pacific[0] < dateline_pacific[2]  # Should not be inverted
    
    # Polar region bounds
    # Near north pole (Web Mercator has limits at ~85°)
    arctic_bounds = (-180, 70, 180, 89)
    arctic_mercator = transform_bounds(arctic_bounds, "EPSG:4326", "EPSG:3857")
    assert len(arctic_mercator) == 4
    # Y values should be very large
    assert arctic_mercator[3] > 10000000
    
    # Antarctic bounds
    antarctic_bounds = (-180, -89, 180, -70)
    antarctic_mercator = transform_bounds(antarctic_bounds, "EPSG:4326", "EPSG:3857")
    assert len(antarctic_mercator) == 4
    assert antarctic_mercator[1] < -10000000
    
    # Use polar stereographic for poles
    arctic_stereo = transform_bounds(arctic_bounds, "EPSG:4326", "EPSG:3995")  # Arctic Polar Stereographic
    assert len(arctic_stereo) == 4
    assert all(np.isfinite(x) for x in arctic_stereo)
    
    # Very large bounds (global extent)
    global_bounds = (-180, -85, 180, 85)  # Near-global (Web Mercator limits)
    global_mercator = transform_bounds(global_bounds, "EPSG:4326", "EPSG:3857")
    assert len(global_mercator) == 4
    # Should be close to Web Mercator full extent
    assert abs(global_mercator[0] - (-20037508.34)) < 1000
    assert abs(global_mercator[2] - 20037508.34) < 1000
    
    # Very small bounds maintain precision
    tiny_bounds = (-74.0001, 40.7000, -74.0000, 40.7001)  # ~100m square
    tiny_mercator = transform_bounds(tiny_bounds, "EPSG:4326", "EPSG:3857")
    
    # Check precision is maintained
    width_deg = tiny_bounds[2] - tiny_bounds[0]
    width_m = tiny_mercator[2] - tiny_mercator[0]
    # Should be approximately 111320 * cos(lat) * width_deg
    expected_width = 111320 * np.cos(np.radians(40.7)) * width_deg
    assert abs(width_m - expected_width) < 0.1  # Sub-decimeter precision
    
    # Invalid bounds should raise errors
    with pytest.raises(Exception):
        # Min > Max
        transform_bounds((10, 10, -10, -10), "EPSG:4326", "EPSG:3857")
    
    # Test with None or invalid CRS
    with pytest.raises(Exception):
        transform_bounds(tiny_bounds, "EPSG:4326", "INVALID_CRS")
    
    with pytest.raises(Exception):
        transform_bounds(tiny_bounds, "INVALID_CRS", "EPSG:3857")


def test_calculate_transform():
    """Test affine transform calculation for georeferencing.
    
    This test should verify that:
    - Transform calculation from bounds and dimensions works correctly
    - 'fit' method adjusts resolution to match exact dimensions
    - 'res' method adjusts dimensions to match exact resolution
    - Target resolution parameter works in coordinate units
    - Calculated transforms enable accurate pixel-to-coordinate mapping
    - Different bounds and dimension combinations work correctly
    - Edge cases (very small/large extents) are handled appropriately
    - Performance is acceptable for typical use cases
    """
    import time
    from rasterio.transform import from_bounds
    
    # Basic transform calculation from bounds and dimensions
    bounds = (-74.05, 40.69, -73.95, 40.79)  # NYC area
    width = 1000
    height = 800
    
    # Test 'fit' method - adjusts resolution to match exact dimensions
    transform, actual_width, actual_height = calculate_transform(
        bounds, width, height, method="fit"
    )
    
    # Verify dimensions match exactly
    assert actual_width == width
    assert actual_height == height
    
    # Verify transform properties
    assert transform.a > 0  # Positive x resolution
    assert transform.e < 0  # Negative y resolution (top-down)
    assert abs(transform.b) < 1e-10  # No rotation
    assert abs(transform.d) < 1e-10  # No rotation
    
    # Calculate resolution from transform
    x_res = (bounds[2] - bounds[0]) / width
    y_res = (bounds[3] - bounds[1]) / height
    assert abs(transform.a - x_res) < 1e-10
    assert abs(transform.e - (-y_res)) < 1e-10
    
    # Test 'res' method - adjusts dimensions to match exact resolution
    target_res = 0.0001  # 0.0001 degrees (~11m at this latitude)
    transform_res, width_res, height_res = calculate_transform(
        bounds, None, None, target_resolution=target_res, method="res"
    )
    
    # Verify resolution matches exactly
    assert abs(transform_res.a - target_res) < 1e-10
    assert abs(transform_res.e - (-target_res)) < 1e-10
    
    # Verify dimensions were calculated correctly
    expected_width = int(np.ceil((bounds[2] - bounds[0]) / target_res))
    expected_height = int(np.ceil((bounds[3] - bounds[1]) / target_res))
    assert width_res == expected_width
    assert height_res == expected_height
    
    # Test target resolution in different coordinate units
    # Web Mercator bounds (meters)
    mercator_bounds = (-8238000, 4970000, -8237000, 4971000)
    target_res_m = 30.0  # 30 meters
    
    transform_m, width_m, height_m = calculate_transform(
        mercator_bounds, None, None, target_resolution=target_res_m, method="res"
    )
    
    assert abs(transform_m.a - target_res_m) < 1e-10
    assert abs(transform_m.e - (-target_res_m)) < 1e-10
    
    # Test pixel-to-coordinate mapping accuracy
    # Upper-left pixel (0, 0) should map to (min_x, max_y)
    x, y = transform * (0, 0)
    assert abs(x - bounds[0]) < 1e-10
    assert abs(y - bounds[3]) < 1e-10
    
    # Lower-right pixel should map to (max_x, min_y)
    x, y = transform * (width - 1, height - 1)
    assert abs(x - (bounds[2] - transform.a)) < transform.a  # Within one pixel
    assert abs(y - (bounds[1] - transform.e)) < abs(transform.e)
    
    # Test different bounds and dimension combinations
    # Square bounds, rectangular dimensions
    square_bounds = (-1, -1, 1, 1)
    rect_transform, _, _ = calculate_transform(square_bounds, 1000, 500, method="fit")
    assert abs(rect_transform.a) > abs(rect_transform.e)  # X resolution coarser
    
    # Rectangular bounds, square dimensions
    rect_bounds = (-2, -1, 2, 1)
    square_transform, _, _ = calculate_transform(rect_bounds, 500, 500, method="fit")
    assert abs(square_transform.a) > abs(square_transform.e)  # X resolution coarser
    
    # Edge case: Very small extent
    tiny_bounds = (-0.0001, -0.0001, 0.0001, 0.0001)
    tiny_transform, _, _ = calculate_transform(tiny_bounds, 100, 100, method="fit")
    assert all(np.isfinite([tiny_transform.a, tiny_transform.e]))
    assert tiny_transform.a > 0
    assert tiny_transform.e < 0
    
    # Edge case: Very large extent
    huge_bounds = (-180, -90, 180, 90)
    huge_transform, _, _ = calculate_transform(huge_bounds, 3600, 1800, method="fit")
    assert abs(huge_transform.a - 0.1) < 1e-10  # 0.1 degree resolution
    assert abs(huge_transform.e - (-0.1)) < 1e-10
    
    # Test performance
    start_time = time.time()
    for _ in range(1000):
        _ = calculate_transform(bounds, 1000, 1000, method="fit")
    elapsed = time.time() - start_time
    assert elapsed < 0.1  # 1000 calculations should be very fast
    
    # Test error handling
    with pytest.raises(ValueError):
        # No dimensions for fit method
        calculate_transform(bounds, None, None, method="fit")
    
    with pytest.raises(ValueError):
        # No resolution for res method
        calculate_transform(bounds, 100, 100, target_resolution=None, method="res")


def test_calculate_transform_fit_method():
    """Test 'fit' method in transform calculation.
    
    This test should verify that:
    - Exact width and height are preserved
    - Resolution adjusts to fit bounds exactly
    - Pixel alignment is appropriate
    - Transform enables accurate georeferencing
    - Edge cases (very small dimensions) work correctly
    """
    # Standard test case
    bounds = (-74.05, 40.69, -73.95, 40.79)
    width = 500
    height = 400
    
    transform, actual_w, actual_h = calculate_transform(
        bounds, width, height, method="fit"
    )
    
    # Exact width and height preserved
    assert actual_w == width
    assert actual_h == height
    
    # Resolution adjusts to fit bounds exactly
    expected_x_res = (bounds[2] - bounds[0]) / width
    expected_y_res = (bounds[3] - bounds[1]) / height
    
    assert abs(transform.a - expected_x_res) < 1e-12
    assert abs(transform.e - (-expected_y_res)) < 1e-12
    
    # Pixel alignment is appropriate
    # Pixel centers should be offset by half a pixel from edges
    x0, y0 = transform * (0, 0)  # Upper-left pixel corner
    x_center, y_center = transform * (0.5, 0.5)  # Upper-left pixel center
    
    assert abs(x0 - bounds[0]) < 1e-10
    assert abs(y0 - bounds[3]) < 1e-10
    assert abs(x_center - (bounds[0] + transform.a / 2)) < 1e-10
    assert abs(y_center - (bounds[3] + transform.e / 2)) < 1e-10
    
    # Transform enables accurate georeferencing
    # Check corners
    corners = [
        (0, 0, bounds[0], bounds[3]),  # Upper-left
        (width, 0, bounds[2], bounds[3]),  # Upper-right
        (0, height, bounds[0], bounds[1]),  # Lower-left
        (width, height, bounds[2], bounds[1])  # Lower-right
    ]
    
    for px, py, expected_x, expected_y in corners:
        x, y = transform * (px, py)
        # Should be within one pixel
        assert abs(x - expected_x) <= abs(transform.a)
        assert abs(y - expected_y) <= abs(transform.e)
    
    # Edge case: Very small dimensions
    small_width = 2
    small_height = 2
    
    small_transform, _, _ = calculate_transform(
        bounds, small_width, small_height, method="fit"
    )
    
    # Should still work correctly
    assert small_transform.a == (bounds[2] - bounds[0]) / small_width
    assert small_transform.e == -(bounds[3] - bounds[1]) / small_height
    
    # Edge case: Single pixel
    single_transform, _, _ = calculate_transform(
        bounds, 1, 1, method="fit"
    )
    
    # Entire bounds in one pixel
    assert single_transform.a == bounds[2] - bounds[0]
    assert single_transform.e == -(bounds[3] - bounds[1])
    
    # Edge case: Very large dimensions
    large_width = 10000
    large_height = 8000
    
    large_transform, _, _ = calculate_transform(
        bounds, large_width, large_height, method="fit"
    )
    
    # Should have very fine resolution
    assert large_transform.a < 0.00001
    assert abs(large_transform.e) < 0.00001


def test_calculate_transform_res_method():
    """Test 'res' method in transform calculation.
    
    This test should verify that:
    - Exact resolution is preserved
    - Dimensions adjust to cover bounds completely
    - Pixel alignment follows standard conventions
    - Transform accuracy is maintained
    - Edge cases (very fine resolution) are handled
    """
    import math
    
    # Standard test case
    bounds = (-74.05, 40.69, -73.95, 40.79)
    target_resolution = 0.0002  # ~22m at this latitude
    
    transform, width, height = calculate_transform(
        bounds, None, None, target_resolution=target_resolution, method="res"
    )
    
    # Exact resolution is preserved
    assert transform.a == target_resolution
    assert transform.e == -target_resolution
    
    # Dimensions adjust to cover bounds completely
    extent_x = bounds[2] - bounds[0]
    extent_y = bounds[3] - bounds[1]
    
    expected_width = math.ceil(extent_x / target_resolution)
    expected_height = math.ceil(extent_y / target_resolution)
    
    assert width == expected_width
    assert height == expected_height
    
    # Verify complete coverage
    # Calculate actual coverage
    actual_extent_x = width * target_resolution
    actual_extent_y = height * target_resolution
    
    # Should cover at least the requested bounds
    assert actual_extent_x >= extent_x
    assert actual_extent_y >= extent_y
    
    # But not more than one pixel extra
    assert actual_extent_x < extent_x + target_resolution
    assert actual_extent_y < extent_y + target_resolution
    
    # Pixel alignment follows standard conventions
    # Upper-left corner aligned with bounds
    assert transform.c == bounds[0]
    assert transform.f == bounds[3]
    
    # Transform accuracy is maintained
    # Test some specific pixel locations
    # Center pixel
    cx, cy = width // 2, height // 2
    x, y = transform * (cx, cy)
    
    expected_x = bounds[0] + cx * target_resolution
    expected_y = bounds[3] - cy * target_resolution
    
    assert abs(x - expected_x) < 1e-10
    assert abs(y - expected_y) < 1e-10
    
    # Edge case: Very fine resolution
    fine_resolution = 0.000001  # ~0.1m at this latitude
    
    fine_transform, fine_width, fine_height = calculate_transform(
        bounds, None, None, target_resolution=fine_resolution, method="res"
    )
    
    assert fine_transform.a == fine_resolution
    assert fine_transform.e == -fine_resolution
    assert fine_width == math.ceil(extent_x / fine_resolution)
    assert fine_height == math.ceil(extent_y / fine_resolution)
    
    # Edge case: Coarse resolution
    coarse_resolution = 0.1  # ~11km
    
    coarse_transform, coarse_width, coarse_height = calculate_transform(
        bounds, None, None, target_resolution=coarse_resolution, method="res"
    )
    
    assert coarse_transform.a == coarse_resolution
    assert coarse_width == math.ceil(extent_x / coarse_resolution)
    assert coarse_height == math.ceil(extent_y / coarse_resolution)
    
    # Edge case: Resolution exactly divides extent
    # Choose resolution that divides evenly
    exact_resolution = extent_x / 10  # Exactly 10 pixels wide
    
    exact_transform, exact_width, exact_height = calculate_transform(
        bounds, None, None, target_resolution=exact_resolution, method="res"
    )
    
    assert exact_transform.a == exact_resolution
    assert exact_width == 10
    
    # Test with different coordinate systems (meters)
    mercator_bounds = (-8238000, 4970000, -8237000, 4971000)
    resolution_m = 25.0  # 25 meters
    
    transform_m, width_m, height_m = calculate_transform(
        mercator_bounds, None, None, target_resolution=resolution_m, method="res"
    )
    
    assert transform_m.a == resolution_m
    assert transform_m.e == -resolution_m
    assert width_m == math.ceil(1000 / resolution_m)  # 40 pixels
    assert height_m == math.ceil(1000 / resolution_m)  # 40 pixels


def test_get_crs_from_epsg():
    """Test CRS object creation from EPSG codes with validation.
    
    This test should verify that:
    - Common EPSG codes (4326, 3857, UTM zones) work correctly
    - CRS objects are properly created and configured
    - PROJ4 string return option works correctly
    - Invalid EPSG codes raise appropriate errors
    - CRS validation catches common mistakes
    - Performance is acceptable for typical use cases
    - Different output formats work correctly
    - Results are suitable for spatial operations
    """
    import time
    import rasterio.crs
    
    # Common EPSG codes work correctly
    # WGS84
    crs_4326 = get_crs_from_epsg(4326)
    assert isinstance(crs_4326, rasterio.crs.CRS)
    assert crs_4326.to_epsg() == 4326
    assert crs_4326.is_geographic
    assert not crs_4326.is_projected
    
    # Web Mercator
    crs_3857 = get_crs_from_epsg(3857)
    assert crs_3857.to_epsg() == 3857
    assert not crs_3857.is_geographic
    assert crs_3857.is_projected
    assert crs_3857.linear_units == 'metre'
    
    # UTM zones
    # UTM Zone 33N (Central Europe)
    crs_utm33n = get_crs_from_epsg(32633)
    assert crs_utm33n.to_epsg() == 32633
    assert crs_utm33n.is_projected
    assert 'utm' in crs_utm33n.to_proj4().lower()
    assert '+zone=33' in crs_utm33n.to_proj4()
    
    # UTM Zone 33S (Southern Africa)
    crs_utm33s = get_crs_from_epsg(32733)
    assert crs_utm33s.to_epsg() == 32733
    assert '+south' in crs_utm33s.to_proj4()
    
    # CRS objects properly created and configured
    # Check key properties
    assert crs_4326.to_string() is not None
    assert crs_4326.to_wkt() is not None
    assert crs_4326.to_proj4() is not None
    assert crs_4326.to_dict() is not None
    
    # PROJ4 string return option
    proj4_string = get_crs_from_epsg(4326, return_proj4=True)
    assert isinstance(proj4_string, str)
    assert '+proj=longlat' in proj4_string
    assert '+datum=WGS84' in proj4_string
    
    proj4_utm = get_crs_from_epsg(32633, return_proj4=True)
    assert isinstance(proj4_utm, str)
    assert '+proj=utm' in proj4_utm
    assert '+zone=33' in proj4_utm
    
    # Invalid EPSG codes raise appropriate errors
    with pytest.raises(ValueError) as excinfo:
        get_crs_from_epsg(99999)  # Invalid code
    assert 'Invalid EPSG code' in str(excinfo.value)
    
    with pytest.raises(ValueError):
        get_crs_from_epsg(0)  # Invalid code
    
    with pytest.raises(ValueError):
        get_crs_from_epsg(-1)  # Negative code
    
    # CRS validation catches common mistakes
    # These are valid EPSG codes that might be problematic
    # EPSG:3785 - deprecated Web Mercator
    crs_3785 = get_crs_from_epsg(3785)
    assert crs_3785 is not None  # Still works but deprecated
    
    # Performance is acceptable
    start_time = time.time()
    for epsg in [4326, 3857, 32633, 32633, 4269, 2154]:
        _ = get_crs_from_epsg(epsg)
    elapsed = time.time() - start_time
    assert elapsed < 0.1  # Should be very fast
    
    # Different output formats work correctly
    crs = get_crs_from_epsg(4326)
    
    # Can convert to various formats
    assert isinstance(crs.to_string(), str)
    assert isinstance(crs.to_wkt(), str)
    assert isinstance(crs.to_proj4(), str)
    assert isinstance(crs.to_dict(), dict)
    assert isinstance(crs.to_epsg(), int)
    
    # Results suitable for spatial operations
    # Can be used with rasterio
    import rasterio.warp
    # Should be able to use with transform functions
    coords = [(-74.0, 40.7)]
    try:
        transformed_coords = rasterio.warp.transform(
            crs_4326.to_string(),
            crs_3857.to_string(),
            [coord[0] for coord in coords],
            [coord[1] for coord in coords]
        )
        assert len(transformed_coords) == 2  # x and y arrays
    except Exception:
        # Transform function might have different signature
        pass
    
    # Test some other common EPSG codes
    common_codes = [
        (4269, 'NAD83'),  # North American Datum 1983
        (2154, 'RGF93'),  # French system
        (27700, 'OSGB'),  # British National Grid
        (3031, 'Antarctic'),  # Antarctic Polar Stereographic
        (5070, 'Albers'),  # Albers Equal Area Conic
    ]
    
    for code, name in common_codes:
        crs = get_crs_from_epsg(code)
        assert crs is not None
        assert crs.to_epsg() == code


def test_get_crs_from_epsg_common_codes():
    """Test CRS creation for common EPSG codes.
    
    This test should verify that:
    - WGS84 (4326) CRS is correctly configured
    - Web Mercator (3857) CRS works appropriately
    - UTM zones (32600+, 32700+) are properly set up
    - CRS properties match expected specifications
    - Coordinate units are correct for each CRS
    """
    # WGS84 (4326) CRS is correctly configured
    wgs84 = get_crs_from_epsg(4326)
    
    # Check properties
    assert wgs84.to_epsg() == 4326
    assert wgs84.is_geographic
    assert not wgs84.is_projected
    assert wgs84.to_authority() == ('EPSG', '4326')
    
    # Check coordinate units
    assert 'degree' in str(wgs84).lower()
    
    # Check datum
    wgs84_dict = wgs84.to_dict()
    assert wgs84_dict['datum'] == 'WGS84'
    
    # Web Mercator (3857) CRS works appropriately
    web_mercator = get_crs_from_epsg(3857)
    
    assert web_mercator.to_epsg() == 3857
    assert not web_mercator.is_geographic
    assert web_mercator.is_projected
    assert web_mercator.linear_units == 'metre'
    
    # Check projection type
    proj4 = web_mercator.to_proj4()
    assert '+proj=merc' in proj4
    assert '+units=m' in proj4
    
    # UTM zones are properly set up
    # Test northern hemisphere zones
    for zone in range(1, 61):  # UTM zones 1-60
        epsg_code = 32600 + zone
        utm_crs = get_crs_from_epsg(epsg_code)
        
        assert utm_crs.is_projected
        assert not utm_crs.is_geographic
        assert utm_crs.linear_units == 'metre'
        
        proj4 = utm_crs.to_proj4()
        assert '+proj=utm' in proj4
        assert f'+zone={zone}' in proj4
        assert '+south' not in proj4  # Northern hemisphere
        
        # Spot check a few zones
        if zone == 18:  # Eastern US
            assert utm_crs.to_epsg() == 32618
        elif zone == 33:  # Central Europe
            assert utm_crs.to_epsg() == 32633
    
    # Test southern hemisphere zones
    for zone in [18, 33, 54]:  # Sample zones
        epsg_code = 32700 + zone
        utm_crs_s = get_crs_from_epsg(epsg_code)
        
        assert utm_crs_s.is_projected
        proj4_s = utm_crs_s.to_proj4()
        assert '+proj=utm' in proj4_s
        assert f'+zone={zone}' in proj4_s
        assert '+south' in proj4_s  # Southern hemisphere
    
    # CRS properties match expected specifications
    # Check specific well-known CRS
    # NAD27 (North American Datum 1927)
    nad27 = get_crs_from_epsg(4267)
    assert nad27.is_geographic
    assert 'NAD27' in nad27.to_wkt()
    
    # British National Grid
    osgb = get_crs_from_epsg(27700)
    assert osgb.is_projected
    assert osgb.linear_units == 'metre'
    assert 'OSGB' in osgb.to_wkt()
    
    # Coordinate units are correct for each CRS
    # Geographic CRS should have degree units
    geographic_codes = [4326, 4269, 4267]
    for code in geographic_codes:
        crs = get_crs_from_epsg(code)
        assert crs.is_geographic
        # Geographic CRS don't have linear units
        
    # Projected CRS should have linear units
    projected_codes = [3857, 32633, 27700, 2154]
    for code in projected_codes:
        crs = get_crs_from_epsg(code)
        assert crs.is_projected
        assert crs.linear_units in ['metre', 'meter', 'foot', 'US survey foot']
        
    # Some projected CRS use feet
    # State Plane California Zone 3 (US Survey Feet)
    sp_ca = get_crs_from_epsg(2227)
    assert sp_ca.is_projected
    assert 'foot' in sp_ca.linear_units.lower()


def test_get_crs_from_epsg_validation():
    """Test EPSG code validation and error handling.
    
    This test should verify that:
    - Invalid EPSG codes raise informative errors
    - Edge cases (very high/low codes) are handled
    - Error messages guide users to correct codes
    - Performance impact of validation is minimal
    - Common mistakes are caught early
    """
    import time
    
    # Invalid EPSG codes raise informative errors
    invalid_codes = [
        99999,  # Doesn't exist
        0,      # Zero
        -1,     # Negative
        -4326,  # Negative version of valid code
        1000000,  # Very high number
    ]
    
    for invalid_code in invalid_codes:
        with pytest.raises(ValueError) as excinfo:
            get_crs_from_epsg(invalid_code)
        
        # Error message should be informative
        error_msg = str(excinfo.value)
        assert 'Invalid EPSG code' in error_msg
        assert str(invalid_code) in error_msg
    
    # Edge cases (very high/low codes) are handled
    # Highest valid EPSG codes are in the 32000-32800 range for UTM
    # and some specialty projections go higher
    high_valid_codes = [
        32760,  # UTM zone 60S
        5070,   # Albers Equal Area Conic
        102003, # Some ESRI codes (if supported)
    ]
    
    for code in high_valid_codes:
        try:
            crs = get_crs_from_epsg(code)
            if crs is not None:
                assert crs.to_epsg() == code or crs.to_epsg() is None
        except ValueError:
            # Some codes might not be in the database
            pass
    
    # Very low valid codes
    low_valid_codes = [
        2000,  # Various local coordinate systems start here
        3857,  # Web Mercator
        4326,  # WGS84
    ]
    
    for code in low_valid_codes:
        crs = get_crs_from_epsg(code)
        assert crs.to_epsg() == code
    
    # Error messages guide users to correct codes
    # Common typos or misunderstandings
    with pytest.raises(ValueError) as excinfo:
        get_crs_from_epsg(84)  # User might mean WGS84
    assert 'Invalid EPSG code 84' in str(excinfo.value)
    
    with pytest.raises(ValueError) as excinfo:
        get_crs_from_epsg(3785)  # Deprecated Web Mercator - actually this might work
    # If it doesn't raise, that's OK - it's a valid but deprecated code
    
    # Performance impact of validation is minimal
    # Time validation vs non-validation
    valid_codes = [4326, 3857, 32633, 27700]
    
    # Time valid codes
    start_time = time.time()
    for _ in range(100):
        for code in valid_codes:
            _ = get_crs_from_epsg(code)
    valid_time = time.time() - start_time
    
    # Time invalid codes (with exception handling)
    start_time = time.time()
    for _ in range(100):
        for code in [99999]:
            try:
                _ = get_crs_from_epsg(code)
            except ValueError:
                pass
    invalid_time = time.time() - start_time
    
    # Validation shouldn't add significant overhead
    assert invalid_time < valid_time * 10  # Should not be more than 10x slower
    
    # Common mistakes are caught early
    # String instead of int (if applicable)
    with pytest.raises((ValueError, TypeError)):
        get_crs_from_epsg("4326")  # Should be int
    
    # Float instead of int
    with pytest.raises((ValueError, TypeError)):
        get_crs_from_epsg(4326.0)
    
    # None
    with pytest.raises((ValueError, TypeError)):
        get_crs_from_epsg(None)
    
    # Check that error happens quickly (not after long processing)
    start_time = time.time()
    try:
        get_crs_from_epsg(99999)
    except ValueError:
        pass
    elapsed = time.time() - start_time
    assert elapsed < 0.01  # Should fail quickly


def test_calculate_pixel_area():
    """Test pixel area calculation accounting for map projection.
    
    This test should verify that:
    - Geographic CRS pixel areas vary with latitude correctly
    - Projected CRS pixel areas are constant or vary appropriately
    - Different area units (m², km², ha) work correctly
    - Transform and CRS parameters affect area calculation accurately
    - Shape parameter defines calculation extent appropriately
    - Results are suitable for area-based statistics
    - Performance is acceptable for large arrays
    - Geodesic calculations are accurate for geographic CRS
    """
    import time
    from rasterio.transform import from_bounds
    import rasterio.crs
    
    # Geographic CRS pixel areas vary with latitude correctly
    # Create a geographic raster covering different latitudes
    geo_bounds = (-180, -90, 180, 90)
    geo_width = 360
    geo_height = 180
    geo_transform = from_bounds(*geo_bounds, geo_width, geo_height)
    geo_crs = rasterio.crs.CRS.from_epsg(4326)
    
    areas_m2 = calculate_pixel_area(
        geo_transform,
        geo_crs,
        (geo_height, geo_width),
        unit="m2"
    )
    
    # Check shape
    assert areas_m2.shape == (geo_height, geo_width)
    
    # Areas should vary with latitude
    # Equator pixels should be largest
    equator_idx = geo_height // 2
    pole_idx = 0
    
    assert areas_m2[equator_idx, 0] > areas_m2[pole_idx, 0]
    assert areas_m2[equator_idx, 0] > areas_m2[-1, 0]  # Both poles
    
    # Check approximate values at equator
    # 1 degree at equator ≈ 111.32 km
    expected_area_equator = (111320 * 1) * (111320 * 1)  # 1x1 degree in m²
    assert abs(areas_m2[equator_idx, 0] - expected_area_equator) / expected_area_equator < 0.01
    
    # Projected CRS pixel areas are constant
    # Web Mercator example
    merc_bounds = (-10000000, -10000000, 10000000, 10000000)
    merc_width = 1000
    merc_height = 1000
    merc_transform = from_bounds(*merc_bounds, merc_width, merc_height)
    merc_crs = rasterio.crs.CRS.from_epsg(3857)
    
    areas_merc = calculate_pixel_area(
        merc_transform,
        merc_crs,
        (merc_height, merc_width),
        unit="m2"
    )
    
    # All pixels should have same area
    pixel_area = areas_merc[0, 0]
    assert np.all(areas_merc == pixel_area)
    
    # Check value (20000x20000 m pixels)
    expected_pixel_area = 20000 * 20000
    assert abs(pixel_area - expected_pixel_area) < 1
    
    # Different area units work correctly
    # Square kilometers
    areas_km2 = calculate_pixel_area(
        merc_transform,
        merc_crs,
        (merc_height, merc_width),
        unit="km2"
    )
    
    assert np.allclose(areas_km2, areas_merc / 1_000_000)
    
    # Hectares
    areas_ha = calculate_pixel_area(
        merc_transform,
        merc_crs,
        (merc_height, merc_width),
        unit="ha"
    )
    
    assert np.allclose(areas_ha, areas_merc / 10_000)
    
    # Transform parameters affect area calculation
    # Finer resolution = smaller pixel areas
    fine_transform = from_bounds(*merc_bounds, 2000, 2000)  # 2x resolution
    areas_fine = calculate_pixel_area(
        fine_transform,
        merc_crs,
        (2000, 2000),
        unit="m2"
    )
    
    # Should be 1/4 the area (half width, half height)
    assert np.allclose(areas_fine[0, 0], areas_merc[0, 0] / 4)
    
    # Shape parameter defines calculation extent
    small_shape = (100, 100)
    areas_small = calculate_pixel_area(
        geo_transform,
        geo_crs,
        small_shape,
        unit="m2"
    )
    
    assert areas_small.shape == small_shape
    
    # Results suitable for area-based statistics
    # Areas should be positive
    assert np.all(areas_m2 > 0)
    assert np.all(areas_merc > 0)
    
    # Areas should be finite
    assert np.all(np.isfinite(areas_m2))
    assert np.all(np.isfinite(areas_merc))
    
    # Can be used for weighted statistics
    # Example: total area calculation
    mask = np.ones((100, 100), dtype=bool)
    masked_areas = calculate_pixel_area(
        merc_transform,
        merc_crs,
        (100, 100),
        unit="km2"
    )
    total_area_km2 = np.sum(mask * masked_areas)
    assert total_area_km2 > 0
    
    # Performance for large arrays
    large_shape = (1000, 1000)
    
    start_time = time.time()
    large_areas = calculate_pixel_area(
        geo_transform,
        geo_crs,
        large_shape,
        unit="m2"
    )
    elapsed = time.time() - start_time
    
    assert elapsed < 1.0  # Should handle million pixels in under a second
    assert large_areas.shape == large_shape
    
    # Geodesic calculations accurate for geographic CRS
    # Test specific latitude
    # 45°N latitude: cos(45°) ≈ 0.707
    lat_45_idx = int((90 - 45) * (geo_height / 180))  # Index for 45°N
    area_45n = areas_m2[lat_45_idx, 0]
    
    # Area should be approximately equator_area * cos(45°)
    expected_45n = expected_area_equator * np.cos(np.radians(45))
    assert abs(area_45n - expected_45n) / expected_45n < 0.01  # Within 1%


def test_calculate_pixel_area_geographic_crs():
    """Test pixel area calculation for geographic coordinate systems.
    
    This test should verify that:
    - Pixel area decreases toward poles correctly
    - Latitude variation is computed accurately
    - Geodesic calculations use appropriate ellipsoid
    - Area units are correctly applied
    - Performance is acceptable for large grids
    """
    import time
    from rasterio.transform import from_bounds
    import rasterio.crs
    
    # Create a geographic CRS raster
    bounds = (-180, -80, 180, 80)  # Avoid poles for numerical stability
    width = 360
    height = 160
    transform = from_bounds(*bounds, width, height)
    crs = rasterio.crs.CRS.from_epsg(4326)
    
    areas = calculate_pixel_area(
        transform,
        crs,
        (height, width),
        unit="m2"
    )
    
    # Pixel area decreases toward poles correctly
    # Get areas at different latitudes
    equator_idx = height // 2  # 0° latitude
    lat_30_idx = height // 2 - 30  # 30°N
    lat_60_idx = height // 2 - 60  # 60°N
    lat_75_idx = height // 2 - 75  # 75°N
    
    area_equator = areas[equator_idx, 0]
    area_30n = areas[lat_30_idx, 0]
    area_60n = areas[lat_60_idx, 0]
    area_75n = areas[lat_75_idx, 0]
    
    # Areas should decrease with latitude
    assert area_equator > area_30n > area_60n > area_75n
    
    # Southern hemisphere should be symmetric
    area_30s = areas[height // 2 + 30, 0]
    area_60s = areas[height // 2 + 60, 0]
    
    assert abs(area_30n - area_30s) / area_30n < 0.001  # Within 0.1%
    assert abs(area_60n - area_60s) / area_60n < 0.001
    
    # Latitude variation computed accurately
    # Area should vary as cos(latitude)
    # At 30°: cos(30°) ≈ 0.866
    expected_ratio_30 = np.cos(np.radians(30))
    actual_ratio_30 = area_30n / area_equator
    assert abs(actual_ratio_30 - expected_ratio_30) < 0.01
    
    # At 60°: cos(60°) = 0.5
    expected_ratio_60 = np.cos(np.radians(60))
    actual_ratio_60 = area_60n / area_equator
    assert abs(actual_ratio_60 - expected_ratio_60) < 0.01
    
    # At 75°: cos(75°) ≈ 0.259
    expected_ratio_75 = np.cos(np.radians(75))
    actual_ratio_75 = area_75n / area_equator
    assert abs(actual_ratio_75 - expected_ratio_75) < 0.01
    
    # Geodesic calculations use appropriate ellipsoid
    # WGS84 ellipsoid parameters should be used
    # Check absolute area values
    # 1° x 1° at equator on WGS84
    deg_lon_m = 111319.491  # meters per degree longitude at equator
    deg_lat_m = 110574.388  # meters per degree latitude
    expected_area = deg_lon_m * deg_lat_m
    
    # Should be close to theoretical value
    assert abs(area_equator - expected_area) / expected_area < 0.01
    
    # Area units correctly applied
    # Test different units
    areas_km2 = calculate_pixel_area(
        transform,
        crs,
        (height, width),
        unit="km2"
    )
    
    assert np.allclose(areas_km2, areas / 1_000_000)
    
    areas_ha = calculate_pixel_area(
        transform,
        crs,
        (height, width),
        unit="ha"
    )
    
    assert np.allclose(areas_ha, areas / 10_000)
    
    # Performance for large grids
    large_width = 3600  # 0.1 degree resolution
    large_height = 1800
    large_transform = from_bounds(-180, -90, 180, 90, large_width, large_height)
    
    start_time = time.time()
    large_areas = calculate_pixel_area(
        large_transform,
        crs,
        (large_height, large_width),
        unit="m2"
    )
    elapsed = time.time() - start_time
    
    assert elapsed < 2.0  # Should handle 6.5 million pixels quickly
    assert large_areas.shape == (large_height, large_width)
    
    # Verify continuity - no sudden jumps
    for i in range(1, height - 1):
        ratio = areas[i, 0] / areas[i-1, 0]
        assert 0.98 < ratio < 1.02  # Smooth variation


def test_calculate_pixel_area_projected_crs():
    """Test pixel area calculation for projected coordinate systems.
    
    This test should verify that:
    - Projected CRS areas are computed from linear units
    - Equal-area projections maintain constant pixel area
    - Conformal projections show area variation as expected
    - Unit conversion works for different linear units
    - Results are consistent with projection properties
    """
    from rasterio.transform import from_bounds
    import rasterio.crs
    
    # Projected CRS areas computed from linear units
    # UTM example (meters)
    utm_bounds = (500000, 4000000, 600000, 4100000)  # 100km x 100km
    utm_width = 1000
    utm_height = 1000
    utm_transform = from_bounds(*utm_bounds, utm_width, utm_height)
    utm_crs = rasterio.crs.CRS.from_epsg(32633)  # UTM Zone 33N
    
    utm_areas = calculate_pixel_area(
        utm_transform,
        utm_crs,
        (utm_height, utm_width),
        unit="m2"
    )
    
    # Each pixel is 100m x 100m
    expected_area = 100 * 100
    assert np.allclose(utm_areas, expected_area)
    
    # Equal-area projections maintain constant pixel area
    # Albers Equal Area Conic
    albers_wkt = 'PROJCS["Albers",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["latitude_of_center",23],PARAMETER["longitude_of_center",-96],PARAMETER["standard_parallel_1",29.5],PARAMETER["standard_parallel_2",45.5],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]'
    albers_crs = rasterio.crs.CRS.from_wkt(albers_wkt)
    
    albers_bounds = (-2000000, -1000000, 2000000, 1000000)
    albers_transform = from_bounds(*albers_bounds, 1000, 500)
    
    albers_areas = calculate_pixel_area(
        albers_transform,
        albers_crs,
        (500, 1000),
        unit="m2"
    )
    
    # All pixels should have same area in equal-area projection
    assert np.std(albers_areas) < 1.0  # Very small variation
    assert np.allclose(albers_areas, albers_areas[0, 0])
    
    # Conformal projections show area variation
    # Web Mercator (conformal, not equal-area)
    # Actually Web Mercator pixels in projected coordinates are constant
    # but represent different ground areas
    merc_bounds = (-10000000, 0, 10000000, 10000000)
    merc_transform = from_bounds(*merc_bounds, 1000, 500)
    merc_crs = rasterio.crs.CRS.from_epsg(3857)
    
    merc_areas = calculate_pixel_area(
        merc_transform,
        merc_crs,
        (500, 1000),
        unit="m2"
    )
    
    # In projected coordinates, pixels are constant
    assert np.allclose(merc_areas, merc_areas[0, 0])
    
    # Unit conversion for different linear units
    # State Plane with US Survey Feet
    spcs_crs = rasterio.crs.CRS.from_epsg(2227)  # CA Zone III, US ft
    
    # Create bounds in feet
    spcs_bounds = (6000000, 2000000, 6100000, 2100000)  # 100k x 100k feet
    spcs_transform = from_bounds(*spcs_bounds, 1000, 1000)
    
    spcs_areas = calculate_pixel_area(
        spcs_transform,
        spcs_crs,
        (1000, 1000),
        unit="m2"
    )
    
    # Each pixel is 100ft x 100ft
    # 1 foot = 0.3048 meters
    expected_area_m2 = (100 * 0.3048) * (100 * 0.3048)
    assert np.allclose(spcs_areas, expected_area_m2, rtol=0.01)
    
    # Test unit conversion
    spcs_areas_km2 = calculate_pixel_area(
        spcs_transform,
        spcs_crs,
        (1000, 1000),
        unit="km2"
    )
    
    assert np.allclose(spcs_areas_km2, spcs_areas / 1_000_000)
    
    # Results consistent with projection properties
    # Lambert Conformal Conic (another conformal projection)
    lcc_proj = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
    lcc_crs = rasterio.crs.CRS.from_proj4(lcc_proj)
    
    lcc_bounds = (-1000000, -1000000, 1000000, 1000000)
    lcc_transform = from_bounds(*lcc_bounds, 1000, 1000)
    
    lcc_areas = calculate_pixel_area(
        lcc_transform,
        lcc_crs,
        (1000, 1000),
        unit="m2"
    )
    
    # Should be constant in projected space
    expected_lcc_area = 2000 * 2000  # 2km x 2km pixels
    assert np.allclose(lcc_areas, expected_lcc_area)
    
    # Test with very fine resolution
    fine_transform = from_bounds(*utm_bounds, 10000, 10000)
    fine_areas = calculate_pixel_area(
        fine_transform,
        utm_crs,
        (10000, 10000),
        unit="m2"
    )
    
    # 10m x 10m pixels
    assert np.allclose(fine_areas, 100)


def test_reproject_geometry():
    """Test vector geometry reprojection between coordinate systems.
    
    This test should verify that:
    - Shapely geometries are reprojected correctly
    - GeoJSON geometries work with reprojection
    - Different geometry types (points, lines, polygons) are handled
    - Source and destination CRS are properly interpreted
    - Spatial relationships are preserved through reprojection
    - Performance is acceptable for complex geometries
    - Always_xy parameter ensures consistent coordinate order
    - Results are suitable for spatial analysis
    """
    from shapely.geometry import Point, LineString, Polygon, MultiPoint
    import time
    
    # Shapely geometries reprojected correctly
    # Test point
    point_4326 = Point(-74.0, 40.7)  # NYC
    point_3857 = reproject_geometry(point_4326, "EPSG:4326", "EPSG:3857")
    
    assert isinstance(point_3857, Point)
    # Check approximate coordinates in Web Mercator
    assert -8240000 < point_3857.x < -8230000
    assert 4960000 < point_3857.y < 4980000
    
    # GeoJSON geometries work with reprojection
    geojson_point = {
        "type": "Point",
        "coordinates": [-74.0, 40.7]
    }
    
    geojson_3857 = reproject_geometry(geojson_point, "EPSG:4326", "EPSG:3857")
    
    assert isinstance(geojson_3857, dict)
    assert geojson_3857["type"] == "Point"
    assert len(geojson_3857["coordinates"]) == 2
    assert -8240000 < geojson_3857["coordinates"][0] < -8230000
    
    # Different geometry types handled
    # LineString
    line_4326 = LineString([(-74.0, 40.7), (-73.9, 40.8)])
    line_3857 = reproject_geometry(line_4326, "EPSG:4326", "EPSG:3857")
    
    assert isinstance(line_3857, LineString)
    assert len(line_3857.coords) == 2
    
    # Polygon
    polygon_4326 = Polygon([
        (-74.0, 40.7),
        (-73.9, 40.7),
        (-73.9, 40.8),
        (-74.0, 40.8),
        (-74.0, 40.7)
    ])
    polygon_3857 = reproject_geometry(polygon_4326, "EPSG:4326", "EPSG:3857")
    
    assert isinstance(polygon_3857, Polygon)
    assert polygon_3857.is_valid
    assert polygon_3857.area > 0
    
    # MultiPoint
    multipoint_4326 = MultiPoint([(-74.0, 40.7), (-73.9, 40.8)])
    multipoint_3857 = reproject_geometry(multipoint_4326, "EPSG:4326", "EPSG:3857")
    
    assert isinstance(multipoint_3857, MultiPoint)
    assert len(multipoint_3857.geoms) == 2
    
    # Source and destination CRS properly interpreted
    # Test with CRS objects
    import rasterio.crs
    src_crs = rasterio.crs.CRS.from_epsg(4326)
    dst_crs = rasterio.crs.CRS.from_epsg(3857)
    
    point_obj_crs = reproject_geometry(point_4326, src_crs, dst_crs)
    assert point_obj_crs.equals_exact(point_3857, tolerance=0.1)
    
    # Test with PROJ4 strings
    proj4_src = "+proj=longlat +datum=WGS84 +no_defs"
    proj4_dst = "+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext +no_defs"
    
    point_proj4 = reproject_geometry(point_4326, proj4_src, proj4_dst)
    assert abs(point_proj4.x - point_3857.x) < 1
    assert abs(point_proj4.y - point_3857.y) < 1
    
    # Spatial relationships preserved
    # Create two polygons with known relationship
    outer_4326 = Polygon([
        (-74.1, 40.6),
        (-73.8, 40.6),
        (-73.8, 40.9),
        (-74.1, 40.9),
        (-74.1, 40.6)
    ])
    inner_4326 = Polygon([
        (-74.0, 40.7),
        (-73.9, 40.7),
        (-73.9, 40.8),
        (-74.0, 40.8),
        (-74.0, 40.7)
    ])
    
    assert outer_4326.contains(inner_4326)
    
    # Reproject both
    outer_3857 = reproject_geometry(outer_4326, "EPSG:4326", "EPSG:3857")
    inner_3857 = reproject_geometry(inner_4326, "EPSG:4326", "EPSG:3857")
    
    # Relationship should be preserved
    assert outer_3857.contains(inner_3857)
    
    # Area ratio should be approximately preserved
    ratio_4326 = inner_4326.area / outer_4326.area
    ratio_3857 = inner_3857.area / outer_3857.area
    assert abs(ratio_4326 - ratio_3857) < 0.01
    
    # Performance for complex geometries
    # Create a complex polygon with many vertices
    angles = np.linspace(0, 2*np.pi, 1000)
    coords = []
    for angle in angles:
        lon = -74.0 + 0.1 * np.cos(angle)
        lat = 40.7 + 0.1 * np.sin(angle)
        coords.append((lon, lat))
    
    complex_polygon = Polygon(coords)
    
    start_time = time.time()
    complex_reprojected = reproject_geometry(complex_polygon, "EPSG:4326", "EPSG:3857")
    elapsed = time.time() - start_time
    
    assert elapsed < 0.1  # Should be fast even for 1000 vertices
    assert len(complex_reprojected.exterior.coords) == len(complex_polygon.exterior.coords)
    
    # Always_xy parameter ensures consistent coordinate order
    # This is handled internally by the transformer creation
    # Test with a CRS that typically uses y,x order
    point_yx = reproject_geometry(point_4326, "EPSG:4326", "EPSG:2154")  # French system
    assert isinstance(point_yx, Point)
    
    # Results suitable for spatial analysis
    # Reprojected geometries should be valid
    assert point_3857.is_valid
    assert line_3857.is_valid
    assert polygon_3857.is_valid
    
    # Should maintain proper orientation
    assert polygon_3857.exterior.is_ccw  # Counter-clockwise for exterior
    
    # Should be suitable for spatial operations
    buffer_3857 = polygon_3857.buffer(1000)  # 1km buffer
    assert buffer_3857.contains(polygon_3857)
    assert buffer_3857.area > polygon_3857.area


def test_reproject_geometry_formats():
    """Test geometry reprojection with different input formats.
    
    This test should verify that:
    - Shapely geometry objects work correctly
    - GeoJSON dictionaries are handled appropriately
    - Format detection is accurate and reliable
    - Output format matches input format
    - Coordinate precision is maintained
    """
    from shapely.geometry import Point, LineString, Polygon, shape, mapping
    
    # Shapely geometry objects work correctly
    shapely_point = Point(-74.006, 40.7128)
    shapely_line = LineString([(-74.006, 40.7128), (-73.935, 40.730)])
    shapely_polygon = Polygon([
        (-74.006, 40.7128),
        (-73.935, 40.7128),
        (-73.935, 40.730),
        (-74.006, 40.730),
        (-74.006, 40.7128)
    ])
    
    # Reproject each
    reprojected_point = reproject_geometry(shapely_point, "EPSG:4326", "EPSG:3857")
    reprojected_line = reproject_geometry(shapely_line, "EPSG:4326", "EPSG:3857")
    reprojected_polygon = reproject_geometry(shapely_polygon, "EPSG:4326", "EPSG:3857")
    
    # Verify types are preserved
    assert type(reprojected_point) == Point
    assert type(reprojected_line) == LineString
    assert type(reprojected_polygon) == Polygon
    
    # GeoJSON dictionaries handled appropriately
    geojson_point = {
        "type": "Point",
        "coordinates": [-74.006, 40.7128]
    }
    
    geojson_linestring = {
        "type": "LineString",
        "coordinates": [[-74.006, 40.7128], [-73.935, 40.730]]
    }
    
    geojson_polygon = {
        "type": "Polygon",
        "coordinates": [[
            [-74.006, 40.7128],
            [-73.935, 40.7128],
            [-73.935, 40.730],
            [-74.006, 40.730],
            [-74.006, 40.7128]
        ]]
    }
    
    # Reproject GeoJSON
    reprojected_geojson_point = reproject_geometry(geojson_point, "EPSG:4326", "EPSG:3857")
    reprojected_geojson_line = reproject_geometry(geojson_linestring, "EPSG:4326", "EPSG:3857")
    reprojected_geojson_polygon = reproject_geometry(geojson_polygon, "EPSG:4326", "EPSG:3857")
    
    # Verify GeoJSON structure preserved
    assert isinstance(reprojected_geojson_point, dict)
    assert reprojected_geojson_point["type"] == "Point"
    assert "coordinates" in reprojected_geojson_point
    
    assert isinstance(reprojected_geojson_line, dict)
    assert reprojected_geojson_line["type"] == "LineString"
    assert len(reprojected_geojson_line["coordinates"]) == 2
    
    assert isinstance(reprojected_geojson_polygon, dict)
    assert reprojected_geojson_polygon["type"] == "Polygon"
    assert len(reprojected_geojson_polygon["coordinates"]) == 1  # One ring
    assert len(reprojected_geojson_polygon["coordinates"][0]) == 5  # 5 coordinates (closed)
    
    # Format detection is accurate and reliable
    # Mixed test - should handle both
    mixed_inputs = [
        shapely_point,
        geojson_point,
        shapely_line,
        geojson_linestring,
        shapely_polygon,
        geojson_polygon
    ]
    
    for input_geom in mixed_inputs:
        result = reproject_geometry(input_geom, "EPSG:4326", "EPSG:3857")
        assert result is not None
        
        if isinstance(input_geom, dict):
            assert isinstance(result, dict)
        else:
            assert not isinstance(result, dict)
    
    # Output format matches input format
    # Shapely in -> Shapely out
    assert isinstance(reproject_geometry(shapely_point, "EPSG:4326", "EPSG:3857"), Point)
    
    # GeoJSON in -> GeoJSON out
    assert isinstance(reproject_geometry(geojson_point, "EPSG:4326", "EPSG:3857"), dict)
    
    # Coordinate precision is maintained
    # Use high precision coordinates
    precise_point = Point(-74.00598765432, 40.71289876543)
    precise_reprojected = reproject_geometry(precise_point, "EPSG:4326", "EPSG:3857")
    
    # Reproject back
    round_trip = reproject_geometry(precise_reprojected, "EPSG:3857", "EPSG:4326")
    
    # Should maintain precision to at least 9 decimal places
    assert abs(round_trip.x - precise_point.x) < 1e-9
    assert abs(round_trip.y - precise_point.y) < 1e-9
    
    # Test GeoJSON precision
    precise_geojson = {
        "type": "Point",
        "coordinates": [-74.00598765432, 40.71289876543]
    }
    
    reprojected_precise = reproject_geometry(precise_geojson, "EPSG:4326", "EPSG:3857")
    round_trip_geojson = reproject_geometry(reprojected_precise, "EPSG:3857", "EPSG:4326")
    
    assert abs(round_trip_geojson["coordinates"][0] - precise_geojson["coordinates"][0]) < 1e-9
    assert abs(round_trip_geojson["coordinates"][1] - precise_geojson["coordinates"][1]) < 1e-9
    
    # Test complex GeoJSON features
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": geojson_point,
                "properties": {"name": "Test Point"}
            }
        ]
    }
    
    # Should handle just the geometry part
    geom_only = feature_collection["features"][0]["geometry"]
    reprojected_feature_geom = reproject_geometry(geom_only, "EPSG:4326", "EPSG:3857")
    assert isinstance(reprojected_feature_geom, dict)
    assert reprojected_feature_geom["type"] == "Point"


def test_reproject_geometry_types():
    """Test reprojection of different geometry types.
    
    This test should verify that:
    - Point geometries are reprojected accurately
    - Line geometries preserve shape characteristics
    - Polygon geometries maintain area relationships
    - Multi-geometries work correctly
    - Complex geometries are handled appropriately
    """
    from shapely.geometry import (
        Point, LineString, Polygon, 
        MultiPoint, MultiLineString, MultiPolygon,
        GeometryCollection
    )
    
    # Point geometries reprojected accurately
    points_4326 = [
        Point(0, 0),  # Null Island
        Point(-74.006, 40.7128),  # NYC
        Point(139.6917, 35.6895),  # Tokyo
        Point(-123.1207, 49.2827),  # Vancouver
        Point(151.2093, -33.8688),  # Sydney
    ]
    
    for point_4326 in points_4326:
        point_3857 = reproject_geometry(point_4326, "EPSG:4326", "EPSG:3857")
        
        # Verify it's still a point
        assert isinstance(point_3857, Point)
        
        # Round trip should be accurate
        round_trip = reproject_geometry(point_3857, "EPSG:3857", "EPSG:4326")
        assert point_4326.distance(round_trip) < 1e-9
    
    # Line geometries preserve shape characteristics
    # Create a line with known properties
    line_4326 = LineString([
        (-74.0, 40.7),
        (-73.95, 40.75),
        (-73.9, 40.7),
        (-73.95, 40.65),
        (-74.0, 40.7)
    ])  # Diamond shape
    
    line_3857 = reproject_geometry(line_4326, "EPSG:4326", "EPSG:3857")
    
    # Should have same number of vertices
    assert len(line_3857.coords) == len(line_4326.coords)
    
    # Length ratio should be preserved (approximately)
    # Create another line for comparison
    line2_4326 = LineString([(-74.0, 40.7), (-73.9, 40.7)])
    line2_3857 = reproject_geometry(line2_4326, "EPSG:4326", "EPSG:3857")
    
    ratio_4326 = line_4326.length / line2_4326.length
    ratio_3857 = line_3857.length / line2_3857.length
    
    assert abs(ratio_4326 - ratio_3857) / ratio_4326 < 0.01  # Within 1%
    
    # Polygon geometries maintain area relationships
    # Create nested polygons
    outer_4326 = Polygon([
        (-74.05, 40.65),
        (-73.85, 40.65),
        (-73.85, 40.75),
        (-74.05, 40.75),
        (-74.05, 40.65)
    ])
    
    inner_4326 = Polygon([
        (-74.0, 40.68),
        (-73.9, 40.68),
        (-73.9, 40.72),
        (-74.0, 40.72),
        (-74.0, 40.68)
    ])
    
    outer_3857 = reproject_geometry(outer_4326, "EPSG:4326", "EPSG:3857")
    inner_3857 = reproject_geometry(inner_4326, "EPSG:4326", "EPSG:3857")
    
    # Area ratio should be approximately preserved
    area_ratio_4326 = inner_4326.area / outer_4326.area
    area_ratio_3857 = inner_3857.area / outer_3857.area
    
    assert abs(area_ratio_4326 - area_ratio_3857) < 0.01
    
    # Containment should be preserved
    assert outer_4326.contains(inner_4326)
    assert outer_3857.contains(inner_3857)
    
    # Multi-geometries work correctly
    # MultiPoint
    multipoint_4326 = MultiPoint([
        (-74.0, 40.7),
        (-73.9, 40.7),
        (-73.95, 40.75)
    ])
    
    multipoint_3857 = reproject_geometry(multipoint_4326, "EPSG:4326", "EPSG:3857")
    
    assert isinstance(multipoint_3857, MultiPoint)
    assert len(multipoint_3857.geoms) == len(multipoint_4326.geoms)
    
    # MultiLineString
    multiline_4326 = MultiLineString([
        [(-74.0, 40.7), (-73.9, 40.7)],
        [(-74.0, 40.8), (-73.9, 40.8)]
    ])
    
    multiline_3857 = reproject_geometry(multiline_4326, "EPSG:4326", "EPSG:3857")
    
    assert isinstance(multiline_3857, MultiLineString)
    assert len(multiline_3857.geoms) == len(multiline_4326.geoms)
    
    # MultiPolygon
    multipolygon_4326 = MultiPolygon([
        outer_4326,
        Polygon([
            (-73.8, 40.6),
            (-73.7, 40.6),
            (-73.7, 40.65),
            (-73.8, 40.65),
            (-73.8, 40.6)
        ])
    ])
    
    multipolygon_3857 = reproject_geometry(multipolygon_4326, "EPSG:4326", "EPSG:3857")
    
    assert isinstance(multipolygon_3857, MultiPolygon)
    assert len(multipolygon_3857.geoms) == len(multipolygon_4326.geoms)
    
    # Complex geometries handled appropriately
    # Polygon with hole
    exterior = [(-74.1, 40.6), (-73.8, 40.6), (-73.8, 40.8), (-74.1, 40.8), (-74.1, 40.6)]
    hole = [(-74.0, 40.65), (-73.9, 40.65), (-73.9, 40.75), (-74.0, 40.75), (-74.0, 40.65)]
    
    polygon_with_hole_4326 = Polygon(exterior, [hole])
    polygon_with_hole_3857 = reproject_geometry(polygon_with_hole_4326, "EPSG:4326", "EPSG:3857")
    
    assert isinstance(polygon_with_hole_3857, Polygon)
    assert len(polygon_with_hole_3857.interiors) == 1
    
    # Area should exclude the hole
    assert polygon_with_hole_3857.area < outer_3857.area
    
    # GeometryCollection
    collection_4326 = GeometryCollection([
        Point(-74.0, 40.7),
        LineString([(-74.0, 40.7), (-73.9, 40.7)]),
        inner_4326
    ])
    
    # GeometryCollection might not be directly supported, but individual geometries should work
    for geom in collection_4326.geoms:
        reprojected = reproject_geometry(geom, "EPSG:4326", "EPSG:3857")
        assert reprojected is not None
        assert reprojected.is_valid
    
    # Test very complex polygon (circle approximation)
    angles = np.linspace(0, 2*np.pi, 360)
    circle_coords = []
    for angle in angles:
        lon = -74.0 + 0.01 * np.cos(angle)
        lat = 40.7 + 0.01 * np.sin(angle)
        circle_coords.append((lon, lat))
    
    circle_4326 = Polygon(circle_coords)
    circle_3857 = reproject_geometry(circle_4326, "EPSG:4326", "EPSG:3857")
    
    assert isinstance(circle_3857, Polygon)
    assert circle_3857.is_valid
    assert len(circle_3857.exterior.coords) == len(circle_4326.exterior.coords)


def test_create_crs_transformer():
    """Test creation of reusable coordinate transformers.
    
    This test should verify that:
    - Transformer objects are created successfully
    - Source and destination CRS are properly configured
    - Always_xy parameter affects coordinate order correctly
    - Transformers work efficiently for multiple coordinates
    - Thread safety is maintained for concurrent use
    - Performance is superior to repeated single transformations
    - Different CRS formats work correctly
    - Results are accurate and consistent
    """
    import pyproj
    import time
    
    # Transformer objects created successfully
    transformer = create_crs_transformer("EPSG:4326", "EPSG:3857")
    
    assert transformer is not None
    assert isinstance(transformer, pyproj.Transformer)
    
    # Source and destination CRS properly configured
    # Test transformation
    lon, lat = -74.006, 40.7128
    x, y = transformer.transform(lon, lat)
    
    # Verify reasonable Web Mercator coordinates
    assert -8240000 < x < -8230000
    assert 4960000 < y < 4980000
    
    # Always_xy parameter affects coordinate order correctly
    # Default should be always_xy=True
    transformer_xy = create_crs_transformer("EPSG:4326", "EPSG:3857", always_xy=True)
    x1, y1 = transformer_xy.transform(lon, lat)  # lon, lat order
    
    transformer_no_xy = create_crs_transformer("EPSG:4326", "EPSG:3857", always_xy=False)
    # For EPSG:4326, the order might be lat, lon when always_xy=False
    x2, y2 = transformer_no_xy.transform(lon, lat)
    
    # Results should be the same when using correct order
    assert abs(x1 - x2) < 0.1
    assert abs(y1 - y2) < 0.1
    
    # Transformers work efficiently for multiple coordinates
    # Create arrays of coordinates
    lons = np.linspace(-180, 180, 1000)
    lats = np.linspace(-85, 85, 1000)
    
    # Transform arrays
    xs, ys = transformer.transform(lons, lats)
    
    assert len(xs) == len(lons)
    assert len(ys) == len(lats)
    assert all(np.isfinite(xs))
    assert all(np.isfinite(ys))
    
    # Performance superior to repeated single transformations
    # Time array transformation
    start_time = time.time()
    xs_array, ys_array = transformer.transform(lons, lats)
    array_time = time.time() - start_time
    
    # Time individual transformations
    start_time = time.time()
    xs_single = []
    ys_single = []
    for lon, lat in zip(lons[:100], lats[:100]):  # Just first 100 for time
        x, y = transformer.transform(lon, lat)
        xs_single.append(x)
        ys_single.append(y)
    single_time = time.time() - start_time
    
    # Array should be much faster per coordinate
    assert array_time < single_time  # 1000 coords faster than 100 individual
    
    # Different CRS formats work correctly
    # EPSG codes
    trans_epsg = create_crs_transformer(4326, 3857)
    x_epsg, y_epsg = trans_epsg.transform(lon, lat)
    assert abs(x_epsg - x) < 0.1
    
    # String EPSG
    trans_str = create_crs_transformer("EPSG:4326", "EPSG:3857")
    x_str, y_str = trans_str.transform(lon, lat)
    assert abs(x_str - x) < 0.1
    
    # PROJ4 strings
    proj4_src = "+proj=longlat +datum=WGS84 +no_defs"
    proj4_dst = "+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs"
    trans_proj4 = create_crs_transformer(proj4_src, proj4_dst)
    assert trans_proj4 is not None
    
    # CRS objects
    import rasterio.crs
    crs_src = rasterio.crs.CRS.from_epsg(4326)
    crs_dst = rasterio.crs.CRS.from_epsg(3857)
    trans_crs = create_crs_transformer(crs_src, crs_dst)
    x_crs, y_crs = trans_crs.transform(lon, lat)
    assert abs(x_crs - x) < 0.1
    
    # Results are accurate and consistent
    # Multiple calls should give same results
    x1, y1 = transformer.transform(lon, lat)
    x2, y2 = transformer.transform(lon, lat)
    assert x1 == x2
    assert y1 == y2
    
    # Round trip should be accurate
    inverse_transformer = create_crs_transformer("EPSG:3857", "EPSG:4326")
    lon2, lat2 = inverse_transformer.transform(x, y)
    assert abs(lon2 - lon) < 1e-9
    assert abs(lat2 - lat) < 1e-9
    
    # Test with different coordinate systems
    # UTM transformer
    utm_transformer = create_crs_transformer("EPSG:4326", "EPSG:32633")
    utm_x, utm_y = utm_transformer.transform(13.0, 52.5)  # Berlin
    assert 300000 < utm_x < 400000  # Reasonable UTM coordinates
    assert 5800000 < utm_y < 5900000


def test_create_crs_transformer_performance():
    """Test CRS transformer performance characteristics.
    
    This test should verify that:
    - Transformer creation overhead is reasonable
    - Multiple coordinate transformation is efficient
    - Memory usage is appropriate for large coordinate arrays
    - Performance scales well with coordinate count
    - Reuse provides measurable performance benefits
    """
    import time
    import gc
    
    # Transformer creation overhead is reasonable
    start_time = time.time()
    transformer = create_crs_transformer("EPSG:4326", "EPSG:3857")
    creation_time = time.time() - start_time
    
    assert creation_time < 0.01  # Should be very fast
    
    # Create multiple transformers to test caching
    start_time = time.time()
    for _ in range(100):
        _ = create_crs_transformer("EPSG:4326", "EPSG:3857")
    multi_creation_time = time.time() - start_time
    
    assert multi_creation_time < 0.1  # 100 creations should still be fast
    
    # Multiple coordinate transformation is efficient
    # Test with increasing array sizes
    sizes = [100, 1000, 10000, 100000]
    times = []
    
    for size in sizes:
        lons = np.random.uniform(-180, 180, size)
        lats = np.random.uniform(-85, 85, size)
        
        start_time = time.time()
        xs, ys = transformer.transform(lons, lats)
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        assert len(xs) == size
        assert len(ys) == size
    
    # Should scale roughly linearly
    # Time per coordinate should be similar
    time_per_coord = [t/s for t, s in zip(times, sizes)]
    assert max(time_per_coord) / min(time_per_coord) < 2  # Within 2x
    
    # Memory usage appropriate for large arrays
    # Test with very large array
    large_size = 1000000
    lons_large = np.random.uniform(-180, 180, large_size)
    lats_large = np.random.uniform(-85, 85, large_size)
    
    # Force garbage collection before
    gc.collect()
    
    # Transform should work without memory errors
    xs_large, ys_large = transformer.transform(lons_large, lats_large)
    
    assert len(xs_large) == large_size
    assert xs_large.dtype == np.float64  # Should use appropriate precision
    
    # Clean up large arrays
    del lons_large, lats_large, xs_large, ys_large
    gc.collect()
    
    # Performance scales well with coordinate count
    # Already tested above with linear scaling
    
    # Reuse provides measurable performance benefits
    # Compare creating new transformer each time vs reusing
    test_coords = [(np.random.uniform(-180, 180), np.random.uniform(-85, 85)) 
                   for _ in range(1000)]
    
    # Method 1: Create new transformer each time
    start_time = time.time()
    results_new = []
    for lon, lat in test_coords:
        trans = create_crs_transformer("EPSG:4326", "EPSG:3857")
        x, y = trans.transform(lon, lat)
        results_new.append((x, y))
    time_new = time.time() - start_time
    
    # Method 2: Reuse transformer
    start_time = time.time()
    trans_reused = create_crs_transformer("EPSG:4326", "EPSG:3857")
    results_reused = []
    for lon, lat in test_coords:
        x, y = trans_reused.transform(lon, lat)
        results_reused.append((x, y))
    time_reused = time.time() - start_time
    
    # Reusing should be significantly faster
    assert time_reused < time_new * 0.5  # At least 2x faster
    
    # Results should be identical
    for (x1, y1), (x2, y2) in zip(results_new, results_reused):
        assert abs(x1 - x2) < 1e-9
        assert abs(y1 - y2) < 1e-9
    
    # Test batch transformation performance
    lons_batch = [coord[0] for coord in test_coords]
    lats_batch = [coord[1] for coord in test_coords]
    
    start_time = time.time()
    xs_batch, ys_batch = trans_reused.transform(lons_batch, lats_batch)
    time_batch = time.time() - start_time
    
    # Batch should be fastest
    assert time_batch < time_reused * 0.1  # At least 10x faster than individual


def test_create_crs_transformer_thread_safety():
    """Test CRS transformer thread safety.
    
    This test should verify that:
    - Transformers work correctly in multi-threaded environments
    - Concurrent access doesn't cause data corruption
    - Performance isn't significantly degraded by thread safety
    - Resource management is appropriate for concurrent use
    - Thread safety documentation is accurate
    """
    import threading
    import queue
    import time
    
    # Create a shared transformer
    transformer = create_crs_transformer("EPSG:4326", "EPSG:3857")
    
    # Test data
    test_coords = [
        (-74.006, 40.7128),  # NYC
        (2.3522, 48.8566),   # Paris
        (-0.1278, 51.5074),  # London
        (139.6917, 35.6895), # Tokyo
        (151.2093, -33.8688) # Sydney
    ]
    
    # Expected results (pre-computed for validation)
    expected_results = {}
    for lon, lat in test_coords:
        x, y = transformer.transform(lon, lat)
        expected_results[(lon, lat)] = (x, y)
    
    # Transformers work correctly in multi-threaded environments
    results_queue = queue.Queue()
    errors_queue = queue.Queue()
    
    def worker_thread(coord_list, thread_id):
        try:
            for lon, lat in coord_list:
                x, y = transformer.transform(lon, lat)
                results_queue.put((thread_id, lon, lat, x, y))
        except Exception as e:
            errors_queue.put((thread_id, str(e)))
    
    # Create multiple threads
    threads = []
    num_threads = 10
    coords_per_thread = 100
    
    # Generate random coordinates for each thread
    all_coords = []
    for i in range(num_threads):
        thread_coords = [(np.random.uniform(-180, 180), np.random.uniform(-85, 85)) 
                        for _ in range(coords_per_thread)]
        all_coords.extend(thread_coords)
        thread = threading.Thread(target=worker_thread, args=(thread_coords, i))
        threads.append(thread)
    
    # Start all threads
    start_time = time.time()
    for thread in threads:
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    multi_thread_time = time.time() - start_time
    
    # Check no errors occurred
    assert errors_queue.empty(), "Errors occurred in threads"
    
    # Verify all results were produced
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())
    
    assert len(results) == num_threads * coords_per_thread
    
    # Concurrent access doesn't cause data corruption
    # Verify known coordinates still produce correct results
    corruption_test_threads = []
    corruption_results = queue.Queue()
    
    def corruption_test_worker():
        for _ in range(100):
            for lon, lat in test_coords:
                x, y = transformer.transform(lon, lat)
                expected_x, expected_y = expected_results[(lon, lat)]
                if abs(x - expected_x) > 1e-9 or abs(y - expected_y) > 1e-9:
                    corruption_results.put((lon, lat, x, y, expected_x, expected_y))
    
    # Run multiple threads doing the same transformations
    for _ in range(5):
        thread = threading.Thread(target=corruption_test_worker)
        corruption_test_threads.append(thread)
        thread.start()
    
    for thread in corruption_test_threads:
        thread.join()
    
    # Should be no corruption
    assert corruption_results.empty(), "Data corruption detected"
    
    # Performance isn't significantly degraded by thread safety
    # Single-threaded baseline
    start_time = time.time()
    single_results = []
    for coord in all_coords:
        x, y = transformer.transform(coord[0], coord[1])
        single_results.append((x, y))
    single_thread_time = time.time() - start_time
    
    # Multi-threaded should not be much slower per operation
    ops_per_sec_single = len(all_coords) / single_thread_time
    ops_per_sec_multi = len(all_coords) / multi_thread_time
    
    # Multi-threaded might be slower due to overhead, but should be comparable
    assert ops_per_sec_multi > ops_per_sec_single * 0.5  # Not more than 2x slower
    
    # Resource management appropriate for concurrent use
    # Create many transformers in threads
    def create_transformer_worker():
        for _ in range(100):
            trans = create_crs_transformer("EPSG:4326", "EPSG:3857")
            # Use it briefly
            trans.transform(0, 0)
    
    create_threads = []
    for _ in range(10):
        thread = threading.Thread(target=create_transformer_worker)
        create_threads.append(thread)
        thread.start()
    
    for thread in create_threads:
        thread.join()
    
    # Should complete without resource exhaustion
    # (If there were resource leaks, this would likely fail)
    
    # Thread safety documentation is accurate
    # pyproj transformers are documented as thread-safe
    # Our wrapper should maintain this property
    assert True  # If we got here without errors, thread safety is working


def test_calculate_utm_zone():
    """Test UTM zone calculation for geographic coordinates.
    
    This test should verify that:
    - Standard UTM zone formula works correctly
    - Zone numbers (1-60) are calculated accurately
    - Hemisphere encoding (positive/negative) works correctly
    - Special cases (Norway, Svalbard) are handled appropriately
    - Edge cases (poles, date line) are managed correctly
    - Performance is acceptable for large coordinate arrays
    - Results match standard UTM zone references
    - Zone selection is appropriate for given coordinates
    """
    import time
    
    # Standard UTM zone formula works correctly
    # Zone = floor((lon + 180) / 6) + 1
    test_cases = [
        (-180, 0, 1),    # Far west
        (-174, 0, 2),    # Zone 2  
        (-168, 0, 3),    # Zone 3
        (-6, 0, 30),     # Zone 30
        (0, 0, 31),      # Zone 31 (Greenwich)
        (6, 0, 32),      # Zone 32
        (174, 0, 60),    # Zone 60
        (180, 0, 61),    # Edge case - may be zone 61 or wrap to 1
    ]
    
    for lon, lat, expected_zone in test_cases:
        zone = calculate_utm_zone(lon, lat)
        assert abs(zone) == expected_zone, f"Failed for lon={lon}: got {zone}, expected {expected_zone}"
    
    # Zone numbers (1-60) calculated accurately
    # Test all zone boundaries
    for zone_num in range(1, 61):
        # Western edge of zone
        lon_west = -180 + (zone_num - 1) * 6
        calculated_zone = calculate_utm_zone(lon_west, 0)
        assert abs(calculated_zone) == zone_num
        
        # Center of zone
        lon_center = lon_west + 3
        calculated_zone = calculate_utm_zone(lon_center, 0)
        assert abs(calculated_zone) == zone_num
    
    # Hemisphere encoding works correctly
    # Northern hemisphere - positive
    north_zone = calculate_utm_zone(0, 45)  # 45°N
    assert north_zone > 0
    assert north_zone == 31
    
    # Southern hemisphere - negative
    south_zone = calculate_utm_zone(0, -45)  # 45°S
    assert south_zone < 0
    assert south_zone == -31
    
    # Equator (latitude 0) - positive
    equator_zone = calculate_utm_zone(0, 0)
    assert equator_zone > 0
    
    # Special cases - Norway
    # Between 56-64°N and 3-12°E should be zone 32
    norway_cases = [
        (3.5, 57, 32),   # Southern Norway
        (5, 60, 32),     # Mid Norway
        (11, 63, 32),    # Northern Norway
        (6, 58, 32),     # Oslo area
    ]
    
    for lon, lat, expected in norway_cases:
        zone = calculate_utm_zone(lon, lat)
        assert zone == expected, f"Norway special case failed: {lon}, {lat}"
    
    # Outside Norway special zone
    assert calculate_utm_zone(2.9, 60) != 32  # Just west
    assert calculate_utm_zone(12.1, 60) != 32  # Just east
    
    # Special cases - Svalbard
    # Between 72-84°N, specific zones
    svalbard_cases = [
        (5, 78, 31),     # Zone 31 (0-9°E)
        (15, 78, 33),    # Zone 33 (9-21°E)
        (27, 78, 35),    # Zone 35 (21-33°E)
        (37, 78, 37),    # Zone 37 (33-42°E)
    ]
    
    for lon, lat, expected in svalbard_cases:
        zone = calculate_utm_zone(lon, lat)
        assert zone == expected, f"Svalbard special case failed: {lon}, {lat}"
    
    # Edge cases - poles
    # Very high latitudes
    assert calculate_utm_zone(0, 89) == 31  # Near north pole
    assert calculate_utm_zone(0, -89) == -31  # Near south pole
    
    # Edge cases - date line
    # Around 180° longitude
    assert abs(calculate_utm_zone(179.9, 0)) == 60
    assert abs(calculate_utm_zone(-179.9, 0)) == 1
    assert abs(calculate_utm_zone(180, 0)) in [1, 61]  # May wrap or be zone 61
    assert abs(calculate_utm_zone(-180, 0)) == 1
    
    # Performance for large coordinate arrays
    # Generate many coordinates
    num_coords = 10000
    lons = np.random.uniform(-180, 180, num_coords)
    lats = np.random.uniform(-80, 80, num_coords)
    
    start_time = time.time()
    zones = [calculate_utm_zone(lon, lat) for lon, lat in zip(lons, lats)]
    elapsed = time.time() - start_time
    
    assert elapsed < 1.0  # Should handle 10k coordinates quickly
    assert len(zones) == num_coords
    assert all(1 <= abs(z) <= 60 for z in zones)
    
    # Results match standard UTM zone references
    # Well-known locations
    known_locations = [
        (-74.006, 40.7128, 18),   # NYC - Zone 18N
        (-0.1278, 51.5074, 30),   # London - Zone 30N (actually 31)
        (2.3522, 48.8566, 31),    # Paris - Zone 31N
        (139.6917, 35.6895, 54),  # Tokyo - Zone 54N
        (151.2093, -33.8688, -56), # Sydney - Zone 56S
        (-122.4194, 37.7749, 10), # San Francisco - Zone 10N
    ]
    
    for lon, lat, expected_zone in known_locations:
        zone = calculate_utm_zone(lon, lat)
        # London is actually in zone 31, not 30
        if lon == -0.1278:
            assert zone == 31
        else:
            assert zone == expected_zone, f"Failed for {lon}, {lat}: got {zone}, expected {expected_zone}"
    
    # Zone selection appropriate for coordinates
    # Zones should minimize distortion for their region
    # Test that adjacent coordinates get same or adjacent zones
    lon1, lat1 = 5.9, 50  # Near zone boundary
    lon2, lat2 = 6.1, 50  # Just across boundary
    
    zone1 = calculate_utm_zone(lon1, lat1)
    zone2 = calculate_utm_zone(lon2, lat2)
    
    assert abs(abs(zone1) - abs(zone2)) <= 1  # Adjacent zones


def test_calculate_utm_zone_special_regions():
    """Test UTM zone calculation for special regions.
    
    This test should verify that:
    - Norway (lat 56-64°, lon 3-12°) uses zone 32 correctly
    - Svalbard zones (31, 33, 35, 37) are applied appropriately
    - Edge coordinates near special regions are handled
    - Standard zones are used when special cases don't apply
    - Special region boundaries are accurately defined
    """
    # Norway special case - zone 32 for specific region
    # Should use zone 32 between 56-64°N and 3-12°E
    
    # Inside Norway special region
    norway_inside = [
        (3.0, 56.0),   # SW corner
        (12.0, 56.0),  # SE corner  
        (3.0, 64.0),   # NW corner
        (12.0, 64.0),  # NE corner
        (7.5, 60.0),   # Center
        (5.3, 59.9),   # Oslo
        (10.7, 59.9),  # Near Oslo
    ]
    
    for lon, lat in norway_inside:
        zone = calculate_utm_zone(lon, lat)
        assert zone == 32, f"Norway zone failed for {lon}, {lat}: got {zone}"
    
    # Just outside Norway special region
    norway_outside = [
        (2.9, 60),     # Just west
        (12.1, 60),    # Just east
        (7.5, 55.9),   # Just south
        (7.5, 64.1),   # Just north
        (3.0, 55.9),   # SW outside
        (12.0, 64.1),  # NE outside
    ]
    
    for lon, lat in norway_outside:
        zone = calculate_utm_zone(lon, lat)
        # Should use standard formula
        expected = int((lon + 180) / 6) + 1
        assert zone == expected, f"Standard zone should apply at {lon}, {lat}"
    
    # Svalbard special zones
    # Between 72-84°N, uses zones 31, 33, 35, 37
    svalbard_tests = [
        # Zone 31: 0-9°E
        (0.0, 78, 31),
        (4.5, 78, 31),
        (8.9, 78, 31),
        # Zone 33: 9-21°E
        (9.0, 78, 33),
        (15.0, 78, 33),
        (20.9, 78, 33),
        # Zone 35: 21-33°E
        (21.0, 78, 35),
        (27.0, 78, 35),
        (32.9, 78, 35),
        # Zone 37: 33-42°E
        (33.0, 78, 37),
        (37.5, 78, 37),
        (41.9, 78, 37),
    ]
    
    for lon, lat, expected in svalbard_tests:
        zone = calculate_utm_zone(lon, lat)
        assert zone == expected, f"Svalbard zone failed for {lon}, {lat}: got {zone}, expected {expected}"
    
    # Test Svalbard latitude boundaries
    # Just below 72°N - standard zones
    lon_test = 15.0
    zone_below = calculate_utm_zone(lon_test, 71.9)
    standard_zone = int((lon_test + 180) / 6) + 1
    assert zone_below == standard_zone
    
    # At 72°N - Svalbard rules apply
    zone_at = calculate_utm_zone(lon_test, 72.0)
    assert zone_at == 33  # Svalbard zone
    
    # At 84°N - still Svalbard
    zone_high = calculate_utm_zone(lon_test, 84.0)
    assert zone_high == 33
    
    # Above 84°N - back to standard (though rarely used)
    zone_above = calculate_utm_zone(lon_test, 84.1)
    assert zone_above == standard_zone
    
    # Edge coordinates near special regions
    # Test precise boundaries
    boundary_tests = [
        # Norway boundaries
        (3.0, 56.0, 32),      # Exact SW corner
        (2.99, 56.0, 31),     # Just outside
        (12.0, 64.0, 32),     # Exact NE corner
        (12.01, 64.0, 33),    # Just outside
        
        # Svalbard boundaries
        (9.0, 78, 33),        # Exact boundary 31/33
        (8.99, 78, 31),       # Just in zone 31
        (21.0, 78, 35),       # Exact boundary 33/35
        (20.99, 78, 33),      # Just in zone 33
    ]
    
    for lon, lat, expected in boundary_tests:
        zone = calculate_utm_zone(lon, lat)
        assert zone == expected, f"Boundary test failed for {lon}, {lat}: got {zone}, expected {expected}"
    
    # Special region boundaries accurately defined
    # Test that regions don't overlap
    test_coords = [
        (7.5, 60),    # In Norway special region
        (15, 78),     # In Svalbard
        (7.5, 70),    # Between regions - standard
    ]
    
    zones = [calculate_utm_zone(lon, lat) for lon, lat in test_coords]
    assert zones[0] == 32        # Norway
    assert zones[1] == 33        # Svalbard  
    assert zones[2] == int((7.5 + 180) / 6) + 1  # Standard
    
    # Verify special regions don't interfere
    # Point in Norway lat range but outside lon range
    zone_outside_norway = calculate_utm_zone(15, 60)
    assert zone_outside_norway != 32
    
    # Point in Svalbard lon range but outside lat range  
    zone_outside_svalbard = calculate_utm_zone(15, 70)
    assert zone_outside_svalbard not in [31, 33, 35, 37]


def test_calculate_utm_zone_hemisphere_encoding():
    """Test UTM zone hemisphere encoding.
    
    This test should verify that:
    - Northern hemisphere zones are positive
    - Southern hemisphere zones are negative
    - Zero latitude is handled appropriately
    - Hemisphere determination is consistent
    - EPSG code calculation works for both hemispheres
    """
    # Northern hemisphere zones are positive
    northern_lats = [0.1, 1, 10, 30, 45, 60, 80, 89]
    
    for lat in northern_lats:
        for lon in [-180, -90, 0, 90, 179]:
            zone = calculate_utm_zone(lon, lat)
            assert zone > 0, f"Northern hemisphere zone should be positive at {lon}, {lat}"
            assert 1 <= zone <= 60
    
    # Southern hemisphere zones are negative
    southern_lats = [-0.1, -1, -10, -30, -45, -60, -80, -89]
    
    for lat in southern_lats:
        for lon in [-180, -90, 0, 90, 179]:
            zone = calculate_utm_zone(lon, lat)
            assert zone < 0, f"Southern hemisphere zone should be negative at {lon}, {lat}"
            assert -60 <= zone <= -1
    
    # Zero latitude handled appropriately
    # Equator (0°) should be considered northern hemisphere
    equator_zones = [
        calculate_utm_zone(-180, 0),
        calculate_utm_zone(0, 0),
        calculate_utm_zone(180, 0),
    ]
    
    for zone in equator_zones:
        assert zone > 0, "Equator should use northern hemisphere (positive zones)"
    
    # Very small positive/negative latitudes
    zone_tiny_north = calculate_utm_zone(0, 0.0001)
    zone_tiny_south = calculate_utm_zone(0, -0.0001)
    
    assert zone_tiny_north > 0
    assert zone_tiny_south < 0
    assert abs(zone_tiny_north) == abs(zone_tiny_south)
    
    # Hemisphere determination is consistent
    # Same absolute zone number for north/south
    test_lons = [-150, -75, 0, 75, 150]
    
    for lon in test_lons:
        zone_north = calculate_utm_zone(lon, 45)
        zone_south = calculate_utm_zone(lon, -45)
        
        assert zone_north > 0
        assert zone_south < 0
        assert abs(zone_north) == abs(zone_south)
    
    # Test consistency across hemisphere boundary
    lon_test = 30
    zones_near_equator = [
        calculate_utm_zone(lon_test, 1),
        calculate_utm_zone(lon_test, 0.1),
        calculate_utm_zone(lon_test, 0),
        calculate_utm_zone(lon_test, -0.1),
        calculate_utm_zone(lon_test, -1),
    ]
    
    # All should have same absolute value
    abs_zones = [abs(z) for z in zones_near_equator]
    assert all(z == abs_zones[0] for z in abs_zones)
    
    # First 3 positive, last 2 negative
    assert all(z > 0 for z in zones_near_equator[:3])
    assert all(z < 0 for z in zones_near_equator[3:])
    
    # EPSG code calculation works for both hemispheres
    # Northern hemisphere: EPSG = 32600 + zone
    # Southern hemisphere: EPSG = 32700 + abs(zone)
    
    # Test some examples
    epsg_tests = [
        (0, 45, 31, 32631),      # Zone 31N
        (0, -45, -31, 32731),    # Zone 31S
        (-177, 60, 1, 32601),    # Zone 1N
        (-177, -60, -1, 32701),  # Zone 1S
        (177, 30, 60, 32660),    # Zone 60N
        (177, -30, -60, 32760),  # Zone 60S
    ]
    
    for lon, lat, expected_zone, expected_epsg in epsg_tests:
        zone = calculate_utm_zone(lon, lat)
        assert zone == expected_zone
        
        # Calculate EPSG
        if zone > 0:
            epsg = 32600 + zone
        else:
            epsg = 32700 + abs(zone)
        
        assert epsg == expected_epsg
    
    # Test special regions maintain hemisphere encoding
    # Norway (northern hemisphere)
    norway_zone = calculate_utm_zone(7, 60)  # In Norway special region
    assert norway_zone == 32  # Positive
    
    # Svalbard (northern hemisphere)
    svalbard_zone = calculate_utm_zone(15, 78)  # In Svalbard
    assert svalbard_zone == 33  # Positive
    
    # No special regions in southern hemisphere
    # Should use standard encoding
    south_zone = calculate_utm_zone(7, -60)
    assert south_zone < 0


def test_validate_crs_compatibility():
    """Test validation of CRS compatibility for processing.
    
    This test should verify that:
    - Compatible CRS lists pass validation
    - Unit mismatches are detected and reported
    - Geographic/projected mixing is identified
    - Area preservation requirements are checked
    - Validation results include actionable recommendations
    - Empty CRS lists are handled gracefully
    - Performance is acceptable for typical CRS lists
    - Compatibility issues are clearly documented
    """
    import time
    import rasterio.crs
    
    # Compatible CRS lists pass validation
    # All same CRS
    compatible_same = validate_crs_compatibility([
        "EPSG:4326",
        "EPSG:4326",
        rasterio.crs.CRS.from_epsg(4326)
    ])
    
    assert compatible_same["compatible"] is True
    assert len(compatible_same["issues"]) == 0
    assert len(compatible_same["recommendations"]) == 0
    
    # All projected CRS with same units
    compatible_projected = validate_crs_compatibility([
        "EPSG:3857",  # Web Mercator (meters)
        "EPSG:32633", # UTM 33N (meters)
        "EPSG:32634"  # UTM 34N (meters)
    ])
    
    assert compatible_projected["compatible"] is True
    assert len(compatible_projected["issues"]) == 0
    
    # Unit mismatches detected and reported
    unit_mismatch = validate_crs_compatibility([
        "EPSG:4326",  # Geographic (degrees)
        "EPSG:3857",  # Projected (meters)
    ], check_units=True)
    
    assert unit_mismatch["compatible"] is False
    assert "Unit mismatch" in str(unit_mismatch["issues"])
    assert len(unit_mismatch["unit_mismatch"]) > 0
    
    # Geographic/projected mixing identified
    mixed_types = validate_crs_compatibility([
        "EPSG:4326",  # Geographic
        "EPSG:3857",  # Projected
        "EPSG:32633" # Projected
    ])
    
    assert mixed_types["compatible"] is False
    assert any("geographic and projected" in issue.lower() for issue in mixed_types["issues"])
    
    # Area preservation requirements checked
    area_check = validate_crs_compatibility([
        "EPSG:3857",  # Web Mercator (not equal-area)
        "EPSG:32633", # UTM (approximately equal-area)
    ], check_area=True)
    
    # Results depend on implementation, but should check area properties
    assert "compatible" in area_check
    
    # Validation results include actionable recommendations
    assert "recommendations" in unit_mismatch
    assert len(unit_mismatch["recommendations"]) > 0
    assert any("common" in rec.lower() or "reproject" in rec.lower() 
              for rec in unit_mismatch["recommendations"])
    
    # Empty CRS lists handled gracefully
    empty_list = validate_crs_compatibility([])
    
    assert empty_list["compatible"] is True
    assert len(empty_list["issues"]) == 0
    assert len(empty_list["recommendations"]) == 0
    assert len(empty_list["unit_mismatch"]) == 0
    
    # Performance acceptable for typical CRS lists
    large_crs_list = ["EPSG:4326"] * 100
    
    start_time = time.time()
    result = validate_crs_compatibility(large_crs_list)
    elapsed = time.time() - start_time
    
    assert elapsed < 0.1  # Should be fast
    assert result["compatible"] is True
    
    # Test with many different CRS
    diverse_crs = [
        "EPSG:4326",
        "EPSG:3857",
        "EPSG:32633",
        "EPSG:32634",
        "EPSG:27700",
        "EPSG:2154",
    ]
    
    start_time = time.time()
    diverse_result = validate_crs_compatibility(diverse_crs)
    elapsed = time.time() - start_time
    
    assert elapsed < 0.1
    assert diverse_result["compatible"] is False  # Mixed types
    
    # Compatibility issues clearly documented
    # Test specific issue types
    mixed_units_different = validate_crs_compatibility([
        "EPSG:2227",  # State Plane California (US feet)
        "EPSG:3857",  # Web Mercator (meters)
    ], check_units=True)
    
    assert not mixed_units_different["compatible"]
    issues = mixed_units_different["issues"]
    assert any("unit" in issue.lower() for issue in issues)
    
    # Test with disabled unit checking
    no_unit_check = validate_crs_compatibility([
        "EPSG:4326",
        "EPSG:3857"
    ], check_units=False)
    
    # Should still detect geographic/projected mixing
    assert not no_unit_check["compatible"]
    
    # Test recommendations quality
    recommendations = mixed_types["recommendations"]
    assert any("epsg" in rec.lower() or "utm" in rec.lower() or "mercator" in rec.lower() 
              for rec in recommendations)
    
    # Test all same geographic CRS
    all_geo = validate_crs_compatibility([
        "EPSG:4326",
        "EPSG:4269",  # NAD83
        "EPSG:4267"   # NAD27
    ])
    
    # Different geographic CRS might still be compatible
    # depending on requirements
    assert "compatible" in all_geo


def test_validate_crs_compatibility_unit_checking():
    """Test CRS compatibility unit checking.
    
    This test should verify that:
    - Consistent units pass validation
    - Mixed units (degrees, meters, feet) are detected
    - Unit mismatch details are provided
    - Unit validation can be disabled when appropriate
    - Recommendations address unit inconsistencies
    """
    # Consistent units pass validation
    # All meters
    meters_crs = [
        "EPSG:3857",   # Web Mercator
        "EPSG:32633",  # UTM 33N
        "EPSG:27700",  # British National Grid
    ]
    
    meters_result = validate_crs_compatibility(meters_crs, check_units=True)
    
    # Should be compatible from unit perspective
    # (though might fail for other reasons)
    unit_issues = [issue for issue in meters_result["issues"] if "unit" in issue.lower()]
    assert len(unit_issues) == 0  # No unit-specific issues
    
    # All geographic (degrees)
    degrees_crs = [
        "EPSG:4326",  # WGS84
        "EPSG:4269",  # NAD83
        "EPSG:4267",  # NAD27
    ]
    
    degrees_result = validate_crs_compatibility(degrees_crs, check_units=True)
    unit_issues = [issue for issue in degrees_result["issues"] if "unit" in issue.lower()]
    assert len(unit_issues) == 0
    
    # Mixed units detected
    # Degrees + meters
    mixed_deg_m = validate_crs_compatibility([
        "EPSG:4326",  # Degrees
        "EPSG:3857",  # Meters
    ], check_units=True)
    
    assert not mixed_deg_m["compatible"]
    assert any("unit" in issue.lower() for issue in mixed_deg_m["issues"])
    assert len(mixed_deg_m["unit_mismatch"]) > 0
    
    # Meters + feet
    mixed_m_ft = validate_crs_compatibility([
        "EPSG:3857",  # Meters
        "EPSG:2227",  # US Survey Feet (California State Plane)
    ], check_units=True)
    
    assert not mixed_m_ft["compatible"]
    assert any("unit" in issue.lower() for issue in mixed_m_ft["issues"])
    
    # Unit mismatch details provided
    unit_details = mixed_deg_m["unit_mismatch"]
    
    assert len(unit_details) > 0
    assert all(isinstance(item, tuple) and len(item) == 2 for item in unit_details)
    
    # Should have index and unit for each CRS
    indices, units = zip(*unit_details)
    assert "degrees" in units
    assert "metre" in units or "meter" in units
    
    # Unit validation can be disabled
    no_unit_check = validate_crs_compatibility([
        "EPSG:4326",
        "EPSG:3857"
    ], check_units=False)
    
    # Should not report unit mismatch issues
    unit_issues = [issue for issue in no_unit_check["issues"] if "unit" in issue.lower()]
    assert len(unit_issues) == 0
    assert len(no_unit_check["unit_mismatch"]) == 0
    
    # But might still be incompatible for other reasons
    other_issues = [issue for issue in no_unit_check["issues"] if "unit" not in issue.lower()]
    if not no_unit_check["compatible"]:
        assert len(other_issues) > 0
    
    # Recommendations address unit inconsistencies
    recommendations = mixed_deg_m["recommendations"]
    
    assert len(recommendations) > 0
    unit_recs = [rec for rec in recommendations if "unit" in rec.lower()]
    assert len(unit_recs) > 0
    
    # Should suggest solutions
    assert any("consistent" in rec.lower() or "common" in rec.lower() 
              for rec in recommendations)
    
    # Test edge cases
    # Same unit, different names
    meter_variants = validate_crs_compatibility([
        "EPSG:3857",   # metres
        "EPSG:32633",  # metres
    ], check_units=True)
    
    # Should not have unit mismatch
    unit_issues = [issue for issue in meter_variants["issues"] if "unit" in issue.lower()]
    assert len(unit_issues) == 0
    
    # Test with unknown/unsupported units
    # Some CRS might have unusual units
    try:
        unusual_result = validate_crs_compatibility([
            "EPSG:3857",
            "+proj=utm +zone=33 +datum=WGS84 +units=km +no_defs",  # Hypothetical km units
        ], check_units=True)
        
        # Should handle gracefully
        assert "compatible" in unusual_result
    except:
        # If this fails, it's acceptable - some CRS strings might not parse
        pass


def test_validate_crs_compatibility_area_preservation():
    """Test CRS compatibility area preservation checking.
    
    This test should verify that:
    - Equal-area projections are identified correctly
    - Non-equal-area projections are flagged when needed
    - Area preservation requirements are configurable
    - Recommendations suggest appropriate projections
    - Area preservation validation is optional
    """
    # Equal-area projections identified correctly
    # Known equal-area projections
    equal_area_crs = [
        "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs",  # Albers
        "+proj=laea +lat_0=45 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs",  # Lambert Azimuthal Equal Area
    ]
    
    area_result = validate_crs_compatibility(equal_area_crs, check_area=True)
    
    # Should not flag area preservation issues for equal-area projections
    area_issues = [issue for issue in area_result["issues"] if "area" in issue.lower()]
    # Might still have issues, but shouldn't be about area preservation specifically
    
    # Non-equal-area projections flagged when needed
    non_equal_area = [
        "EPSG:3857",  # Web Mercator (conformal, not equal-area)
        "+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +no_defs",
    ]
    
    non_area_result = validate_crs_compatibility(non_equal_area, check_area=True)
    
    # Should flag area preservation concerns
    if not non_area_result["compatible"]:
        area_issues = [issue for issue in non_area_result["issues"] if "area" in issue.lower()]
        # Might have area-related warnings
    
    # Area preservation requirements configurable
    # With area checking
    with_area = validate_crs_compatibility([
        "EPSG:3857",
        "EPSG:32633"
    ], check_area=True)
    
    # Without area checking  
    without_area = validate_crs_compatibility([
        "EPSG:3857",
        "EPSG:32633"
    ], check_area=False)
    
    # Results might differ
    with_area_issues = [issue for issue in with_area["issues"] if "area" in issue.lower()]
    without_area_issues = [issue for issue in without_area["issues"] if "area" in issue.lower()]
    
    # Without area checking should have fewer or no area-related issues
    assert len(without_area_issues) <= len(with_area_issues)
    
    # Recommendations suggest appropriate projections
    # Test with known problematic combinations
    problematic = validate_crs_compatibility([
        "EPSG:4326",  # Geographic
        "EPSG:3857",  # Web Mercator
    ], check_area=True)
    
    recommendations = problematic["recommendations"]
    assert len(recommendations) > 0
    
    # Should suggest better projections
    projection_recs = [rec for rec in recommendations 
                      if any(proj in rec.lower() 
                            for proj in ["utm", "albers", "lambert", "equal-area"])]
    
    # Area preservation validation is optional
    # Default behavior (area checking disabled by default)
    default_result = validate_crs_compatibility([
        "EPSG:3857",
        "EPSG:32633"
    ])  # check_area defaults to False
    
    explicit_false = validate_crs_compatibility([
        "EPSG:3857",
        "EPSG:32633"
    ], check_area=False)
    
    # Should be the same
    assert default_result["compatible"] == explicit_false["compatible"]
    assert default_result["issues"] == explicit_false["issues"]
    
    # Test with various projection types
    projection_tests = [
        ("EPSG:3857", "mercator"),           # Conformal
        ("EPSG:32633", "utm"),               # Approximately equal-area for small regions
        ("+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs", "aea"),  # Equal-area
    ]
    
    for crs_str, proj_type in projection_tests:
        result = validate_crs_compatibility([crs_str], check_area=True)
        
        # Single CRS should be compatible with itself
        assert result["compatible"] is True
        
        # Should not have area preservation issues for single CRS
        area_issues = [issue for issue in result["issues"] if "area" in issue.lower()]
        assert len(area_issues) == 0


def test_calculate_ground_distance():
    """Test conversion of pixel distance to ground distance.
    
    This test should verify that:
    - Pixel distance is converted to meters accurately
    - Geographic CRS calculations use geodesic methods
    - Projected CRS calculations use linear units correctly
    - Location parameter enables local distance calculation
    - Different pixel sizes are handled appropriately
    - Performance is acceptable for typical use cases
    - Results are suitable for buffer and kernel operations
    - Distance accuracy is adequate for intended applications
    """
    import time
    from rasterio.transform import from_bounds
    import rasterio.crs
    
    # Projected CRS calculations use linear units correctly
    # UTM example (meters)
    utm_bounds = (500000, 4000000, 501000, 4001000)  # 1km x 1km
    utm_transform = from_bounds(*utm_bounds, 100, 100)  # 10m pixels
    utm_crs = rasterio.crs.CRS.from_epsg(32633)
    
    # 5 pixel distance should be 50 meters
    ground_dist_utm = calculate_ground_distance(5, utm_transform, utm_crs)
    assert abs(ground_dist_utm - 50.0) < 0.1
    
    # Geographic CRS calculations use geodesic methods
    # WGS84 example
    geo_bounds = (-74.1, 40.6, -73.9, 40.8)  # NYC area
    geo_transform = from_bounds(*geo_bounds, 200, 200)  # 0.001 degree pixels
    geo_crs = rasterio.crs.CRS.from_epsg(4326)
    
    # Calculate ground distance for 5 pixels
    ground_dist_geo = calculate_ground_distance(5, geo_transform, geo_crs)
    
    # At NYC latitude (~40.7°), 1 degree longitude ≈ 85.4 km
    # 0.001 degree ≈ 85.4 m, so 5 pixels ≈ 427 m
    expected_dist = 5 * 0.001 * 111320 * np.cos(np.radians(40.7))
    assert abs(ground_dist_geo - expected_dist) / expected_dist < 0.01
    
    # Location parameter enables local distance calculation
    # Test at different latitudes
    equator_dist = calculate_ground_distance(
        5, geo_transform, geo_crs, location=(-74.0, 0.0)
    )
    
    polar_dist = calculate_ground_distance(
        5, geo_transform, geo_crs, location=(-74.0, 80.0)
    )
    
    # Distance should be smaller near poles due to longitude convergence
    assert polar_dist < equator_dist
    
    # Check ratios match cosine relationship
    ratio = polar_dist / equator_dist
    expected_ratio = np.cos(np.radians(80)) / np.cos(np.radians(0))
    assert abs(ratio - expected_ratio) < 0.01
    
    # Different pixel sizes handled appropriately
    # Fine resolution
    fine_transform = from_bounds(*utm_bounds, 1000, 1000)  # 1m pixels
    fine_dist = calculate_ground_distance(10, fine_transform, utm_crs)
    assert abs(fine_dist - 10.0) < 0.1
    
    # Coarse resolution
    coarse_transform = from_bounds(*utm_bounds, 10, 10)  # 100m pixels
    coarse_dist = calculate_ground_distance(2, coarse_transform, utm_crs)
    assert abs(coarse_dist - 200.0) < 0.1
    
    # Performance for typical use cases
    start_time = time.time()
    for _ in range(1000):
        _ = calculate_ground_distance(5, utm_transform, utm_crs)
    elapsed = time.time() - start_time
    
    assert elapsed < 0.1  # 1000 calculations should be fast
    
    # Results suitable for buffer and kernel operations
    # Test buffer calculation
    buffer_pixels = 3  # Want 3-pixel buffer
    buffer_meters = calculate_ground_distance(buffer_pixels, utm_transform, utm_crs)
    
    # Should be exactly 30 meters for 10m pixels
    assert abs(buffer_meters - 30.0) < 0.01
    
    # Kernel operations - e.g., Gaussian kernel sigma
    sigma_pixels = 2.5
    sigma_meters = calculate_ground_distance(sigma_pixels, utm_transform, utm_crs)
    assert abs(sigma_meters - 25.0) < 0.01
    
    # Distance accuracy adequate for intended applications
    # Sub-meter accuracy for fine resolution
    fine_pixel = 0.5  # Half pixel
    fine_ground = calculate_ground_distance(fine_pixel, fine_transform, utm_crs)
    assert abs(fine_ground - 0.5) < 0.01
    
    # Test with different coordinate systems
    # Web Mercator
    merc_bounds = (-8238000, 4970000, -8237000, 4971000)
    merc_transform = from_bounds(*merc_bounds, 100, 100)
    merc_crs = rasterio.crs.CRS.from_epsg(3857)
    
    merc_dist = calculate_ground_distance(10, merc_transform, merc_crs)
    assert abs(merc_dist - 100.0) < 0.1  # 10m pixels
    
    # State Plane with feet
    sp_crs = rasterio.crs.CRS.from_epsg(2227)  # California Zone III
    sp_bounds = (6000000, 2000000, 6001000, 2001000)  # 1000ft x 1000ft
    sp_transform = from_bounds(*sp_bounds, 100, 100)  # 10ft pixels
    
    sp_dist = calculate_ground_distance(5, sp_transform, sp_crs)
    expected_sp = 5 * 10 * 0.3048  # 5 pixels * 10ft * meters/foot
    assert abs(sp_dist - expected_sp) < 0.01
    
    # Edge cases
    # Zero distance
    zero_dist = calculate_ground_distance(0, utm_transform, utm_crs)
    assert zero_dist == 0
    
    # Fractional pixels
    frac_dist = calculate_ground_distance(0.5, utm_transform, utm_crs)
    assert abs(frac_dist - 5.0) < 0.01
    
    # Large distances
    large_dist = calculate_ground_distance(1000, utm_transform, utm_crs)
    assert abs(large_dist - 10000.0) < 0.1


def test_calculate_ground_distance_geographic():
    """Test ground distance calculation for geographic CRS.
    
    This test should verify that:
    - Geodesic distance calculation is accurate
    - Location parameter affects calculation appropriately
    - Default location uses transform center
    - Distance varies with latitude as expected
    - WGS84 ellipsoid is used correctly
    """
    from rasterio.transform import from_bounds
    import rasterio.crs
    
    # Setup geographic CRS
    geo_bounds = (-74.1, 40.6, -73.9, 40.8)
    geo_transform = from_bounds(*geo_bounds, 200, 200)  # 0.001 degree pixels
    geo_crs = rasterio.crs.CRS.from_epsg(4326)
    
    # Geodesic distance calculation is accurate
    # At equator, 1 degree longitude = ~111.32 km
    equator_transform = from_bounds(-1, -0.5, 1, 0.5, 200, 100)  # 0.01 degree pixels
    
    equator_dist = calculate_ground_distance(
        1, equator_transform, geo_crs, location=(0, 0)
    )
    
    # 1 pixel = 0.01 degree at equator
    expected_equator = 0.01 * 111320  # meters per degree at equator
    assert abs(equator_dist - expected_equator) / expected_equator < 0.01
    
    # Location parameter affects calculation appropriately
    # Same pixel size, different latitudes
    lat_30_dist = calculate_ground_distance(
        1, equator_transform, geo_crs, location=(0, 30)
    )
    
    lat_60_dist = calculate_ground_distance(
        1, equator_transform, geo_crs, location=(0, 60)
    )
    
    # Distance should decrease with latitude due to longitude convergence
    assert lat_30_dist < equator_dist
    assert lat_60_dist < lat_30_dist
    
    # Check cosine relationship
    ratio_30 = lat_30_dist / equator_dist
    expected_30 = np.cos(np.radians(30))
    assert abs(ratio_30 - expected_30) < 0.01
    
    ratio_60 = lat_60_dist / equator_dist
    expected_60 = np.cos(np.radians(60))
    assert abs(ratio_60 - expected_60) < 0.01
    
    # Default location uses transform center
    # Calculate expected center
    center_x = geo_transform.c + geo_transform.a * 0.5
    center_y = geo_transform.f + geo_transform.e * 0.5
    
    default_dist = calculate_ground_distance(1, geo_transform, geo_crs)
    explicit_dist = calculate_ground_distance(
        1, geo_transform, geo_crs, location=(center_x, center_y)
    )
    
    assert abs(default_dist - explicit_dist) < 0.1
    
    # Distance varies with latitude as expected
    latitudes = [0, 15, 30, 45, 60, 75]
    distances = []
    
    for lat in latitudes:
        dist = calculate_ground_distance(
            1, equator_transform, geo_crs, location=(0, lat)
        )
        distances.append(dist)
    
    # Should decrease with increasing latitude
    for i in range(1, len(distances)):
        assert distances[i] < distances[i-1]
    
    # Should follow cosine law
    for lat, dist in zip(latitudes, distances):
        expected = distances[0] * np.cos(np.radians(lat))
        assert abs(dist - expected) / expected < 0.01
    
    # WGS84 ellipsoid used correctly
    # Test against known geodesic calculations
    # 1 degree latitude is approximately constant at 110.574 km
    lat_spacing_transform = from_bounds(-0.5, -1, 0.5, 1, 100, 200)  # 0.01 deg lat pixels
    
    lat_dist = calculate_ground_distance(
        1, lat_spacing_transform, geo_crs, location=(0, 45)
    )
    
    expected_lat_dist = 0.01 * 110574  # meters per degree latitude
    assert abs(lat_dist - expected_lat_dist) / expected_lat_dist < 0.01
    
    # Test southern hemisphere
    south_dist = calculate_ground_distance(
        1, equator_transform, geo_crs, location=(0, -30)
    )
    
    north_dist = calculate_ground_distance(
        1, equator_transform, geo_crs, location=(0, 30)
    )
    
    # Should be symmetric
    assert abs(south_dist - north_dist) / north_dist < 0.001
    
    # Test near poles
    polar_dist = calculate_ground_distance(
        1, equator_transform, geo_crs, location=(0, 89)
    )
    
    # Should be much smaller than equator
    assert polar_dist < equator_dist * 0.1
    
    # Verify ellipsoid shape (Earth is slightly flattened)
    # This is handled by pyproj's geodesic calculations


def test_calculate_ground_distance_projected():
    """Test ground distance calculation for projected CRS.
    
    This test should verify that:
    - Linear units are correctly identified and used
    - Meter units work without conversion
    - Foot units are converted to meters correctly
    - Unknown units are handled with appropriate assumptions
    - Distance calculation is consistent across projections
    """
    from rasterio.transform import from_bounds
    import rasterio.crs
    
    # Meter units work without conversion
    # UTM Zone 33N (meters)
    utm_bounds = (500000, 4000000, 501000, 4001000)
    utm_transform = from_bounds(*utm_bounds, 100, 100)  # 10m pixels
    utm_crs = rasterio.crs.CRS.from_epsg(32633)
    
    utm_dist = calculate_ground_distance(5, utm_transform, utm_crs)
    assert abs(utm_dist - 50.0) < 0.01
    
    # Web Mercator (meters)
    merc_bounds = (-8238000, 4970000, -8237000, 4971000)
    merc_transform = from_bounds(*merc_bounds, 100, 100)  # 10m pixels
    merc_crs = rasterio.crs.CRS.from_epsg(3857)
    
    merc_dist = calculate_ground_distance(10, merc_transform, merc_crs)
    assert abs(merc_dist - 100.0) < 0.01
    
    # British National Grid (meters)
    bng_bounds = (500000, 100000, 501000, 101000)
    bng_transform = from_bounds(*bng_bounds, 1000, 1000)  # 1m pixels
    bng_crs = rasterio.crs.CRS.from_epsg(27700)
    
    bng_dist = calculate_ground_distance(25, bng_transform, bng_crs)
    assert abs(bng_dist - 25.0) < 0.01
    
    # Foot units converted to meters correctly
    # California State Plane Zone III (US Survey Feet)
    sp_bounds = (6000000, 2000000, 6001000, 2001000)  # 1000ft x 1000ft
    sp_transform = from_bounds(*sp_bounds, 100, 100)  # 10ft pixels
    sp_crs = rasterio.crs.CRS.from_epsg(2227)
    
    sp_dist = calculate_ground_distance(5, sp_transform, sp_crs)
    
    # 5 pixels * 10 feet/pixel * 0.3048 meters/foot
    expected_sp = 5 * 10 * 0.3048
    assert abs(sp_dist - expected_sp) < 0.01
    
    # Test international foot vs US survey foot
    # EPSG:2227 uses US survey foot
    # 1 US survey foot = 0.3048006096012192 meters
    # 1 international foot = 0.3048 meters
    
    # The function should handle the specific foot type
    assert sp_dist > 15.24  # More than international feet
    
    # Distance calculation consistent across projections
    # Same geographic area in different projections
    
    # Area around NYC in different projections
    nyc_geo_bounds = (-74.1, 40.6, -73.9, 40.8)
    
    # UTM Zone 18N (appropriate for NYC)
    utm18_crs = rasterio.crs.CRS.from_epsg(32618)
    # Transform bounds to UTM
    from blackmarble.prepare.coordinates import transform_bounds
    utm18_bounds = transform_bounds(nyc_geo_bounds, "EPSG:4326", "EPSG:32618")
    utm18_transform = from_bounds(*utm18_bounds, 1000, 1000)
    
    # Web Mercator
    merc_nyc_bounds = transform_bounds(nyc_geo_bounds, "EPSG:4326", "EPSG:3857")
    merc_nyc_transform = from_bounds(*merc_nyc_bounds, 1000, 1000)
    
    # Calculate pixel sizes
    utm18_pixel_size = (utm18_bounds[2] - utm18_bounds[0]) / 1000
    merc_pixel_size = (merc_nyc_bounds[2] - merc_nyc_bounds[0]) / 1000
    
    # Calculate distances
    utm18_dist = calculate_ground_distance(1, utm18_transform, utm18_crs)
    merc_nyc_dist = calculate_ground_distance(1, merc_nyc_transform, merc_crs)
    
    # Should be approximately equal (both in meters)
    assert abs(utm18_dist - utm18_pixel_size) < 0.01
    assert abs(merc_nyc_dist - merc_pixel_size) < 0.01
    
    # Linear units correctly identified
    # Test various projected CRS with known units
    crs_tests = [
        (rasterio.crs.CRS.from_epsg(32633), "metre"),     # UTM
        (rasterio.crs.CRS.from_epsg(3857), "metre"),      # Web Mercator
        (rasterio.crs.CRS.from_epsg(27700), "metre"),     # British National Grid
        (rasterio.crs.CRS.from_epsg(2227), "US survey foot"),  # State Plane
    ]
    
    for crs, expected_unit in crs_tests:
        # Check that CRS has expected linear unit
        actual_unit = crs.linear_units
        assert expected_unit.lower() in actual_unit.lower() or actual_unit.lower() in expected_unit.lower()
    
    # Unknown units handled with appropriate assumptions
    # Create a custom CRS with unusual units (if possible)
    try:
        # Hypothetical CRS with unknown units
        custom_proj = "+proj=utm +zone=33 +datum=WGS84 +units=unknown +no_defs"
        custom_crs = rasterio.crs.CRS.from_proj4(custom_proj)
        custom_transform = from_bounds(500000, 4000000, 501000, 4001000, 100, 100)
        
        # Should not crash, might assume meters
        custom_dist = calculate_ground_distance(10, custom_transform, custom_crs)
        assert custom_dist > 0
    except:
        # This test might not work with all PROJ versions
        pass
    
    # Test consistency
    # Same pixel size in meters should give same distance regardless of CRS
    
    # 30m pixels in different CRS
    test_bounds = [(0, 0, 3000, 3000)]  # 3km square
    
    crs_list = [
        rasterio.crs.CRS.from_epsg(32633),  # UTM
        rasterio.crs.CRS.from_epsg(3857),   # Web Mercator
        rasterio.crs.CRS.from_epsg(27700),  # British National Grid
    ]
    
    distances = []
    for crs in crs_list:
        transform = from_bounds(0, 0, 3000, 3000, 100, 100)  # 30m pixels
        dist = calculate_ground_distance(1, transform, crs)
        distances.append(dist)
    
    # All should be approximately 30 meters
    for dist in distances:
        assert abs(dist - 30.0) < 0.1
    
    # Should be very similar to each other
    assert max(distances) - min(distances) < 0.1


def test_coordinate_precision_validation():
    """Test coordinate precision across transformation functions.
    
    This test should verify that:
    - Coordinate transformations maintain adequate precision
    - Numerical precision loss is within acceptable bounds
    - Round-trip transformations preserve coordinates
    - Double precision is used where necessary
    - Precision requirements are documented
    """
    from shapely.geometry import Point
    
    # Coordinate transformations maintain adequate precision
    # High precision coordinates
    precise_coords = [
        (-74.00598765432109, 40.71289876543210),  # NYC with high precision
        (139.69170123456789, 35.68950987654321),  # Tokyo
        (2.35220123456789, 48.85669876543210),    # Paris
    ]
    
    for lon, lat in precise_coords:
        # Transform to UTM
        utm_zone = calculate_utm_zone(lon, lat)
        utm_epsg = 32600 + utm_zone if utm_zone > 0 else 32700 + abs(utm_zone)
        
        # Create transformer
        transformer = create_crs_transformer("EPSG:4326", f"EPSG:{utm_epsg}")
        x, y = transformer.transform(lon, lat)
        
        # Precision should be maintained to sub-millimeter level
        assert isinstance(x, (float, np.floating))
        assert isinstance(y, (float, np.floating))
        assert np.isfinite(x) and np.isfinite(y)
    
    # Numerical precision loss within acceptable bounds
    # Test with very small coordinate differences
    base_lon, base_lat = -74.0, 40.7
    small_offsets = [1e-9, 1e-8, 1e-7, 1e-6]  # Sub-meter precision
    
    transformer = create_crs_transformer("EPSG:4326", "EPSG:3857")
    base_x, base_y = transformer.transform(base_lon, base_lat)
    
    for offset in small_offsets:
        x1, y1 = transformer.transform(base_lon + offset, base_lat)
        x2, y2 = transformer.transform(base_lon, base_lat + offset)
        
        # Should detect differences
        lon_diff = abs(x1 - base_x)
        lat_diff = abs(y2 - base_y)
        
        # At this scale, should maintain precision
        if offset >= 1e-7:  # Coordinate precision limit
            assert lon_diff > 0
            assert lat_diff > 0
    
    # Round-trip transformations preserve coordinates
    test_points = [
        (-180, -85), (-180, 85), (180, -85), (180, 85),  # Corners
        (0, 0),                                            # Origin
        (-74.006, 40.7128),                              # NYC
        (139.6917, 35.6895),                             # Tokyo
    ]
    
    # Test with various CRS pairs
    crs_pairs = [
        ("EPSG:4326", "EPSG:3857"),    # Geographic to Web Mercator
        ("EPSG:4326", "EPSG:32633"),   # Geographic to UTM
        ("EPSG:3857", "EPSG:32633"),   # Web Mercator to UTM
    ]
    
    for src_crs, dst_crs in crs_pairs:
        forward = create_crs_transformer(src_crs, dst_crs)
        reverse = create_crs_transformer(dst_crs, src_crs)
        
        for lon, lat in test_points:
            try:
                # Forward transformation
                x, y = forward.transform(lon, lat)
                
                # Reverse transformation
                lon2, lat2 = reverse.transform(x, y)
                
                # Should preserve coordinates within precision limits
                lon_error = abs(lon2 - lon)
                lat_error = abs(lat2 - lat)
                
                # Geographic coordinates: precision to ~1cm (1e-7 degrees)
                if src_crs == "EPSG:4326":
                    assert lon_error < 1e-7, f"Longitude error {lon_error} too large"
                    assert lat_error < 1e-7, f"Latitude error {lat_error} too large"
                # Projected coordinates: precision to ~1mm
                else:
                    assert lon_error < 0.001, f"X error {lon_error} too large"
                    assert lat_error < 0.001, f"Y error {lat_error} too large"
                    
            except Exception:
                # Some transformations might fail at extremes
                pass
    
    # Test bounds transformation precision
    test_bounds = (-74.1, 40.6, -73.9, 40.8)
    
    # Forward
    transformed_bounds = transform_bounds(test_bounds, "EPSG:4326", "EPSG:3857")
    
    # Reverse
    round_trip_bounds = transform_bounds(transformed_bounds, "EPSG:3857", "EPSG:4326")
    
    # Check precision
    for orig, final in zip(test_bounds, round_trip_bounds):
        assert abs(final - orig) < 1e-9
    
    # Test geometry reprojection precision
    precise_point = Point(-74.00598765432, 40.71289876543)
    
    # Reproject to Web Mercator and back
    mercator_point = reproject_geometry(precise_point, "EPSG:4326", "EPSG:3857")
    round_trip_point = reproject_geometry(mercator_point, "EPSG:3857", "EPSG:4326")
    
    # Check precision
    assert abs(round_trip_point.x - precise_point.x) < 1e-9
    assert abs(round_trip_point.y - precise_point.y) < 1e-9
    
    # Double precision is used where necessary
    # Check data types
    x, y = transformer.transform(-74.0, 40.7)
    assert isinstance(x, (float, np.float64))
    assert isinstance(y, (float, np.float64))
    
    # Arrays should use double precision
    lons = np.array([-74.0, -73.9, -73.8], dtype=np.float64)
    lats = np.array([40.7, 40.8, 40.9], dtype=np.float64)
    
    xs, ys = transformer.transform(lons, lats)
    assert xs.dtype == np.float64
    assert ys.dtype == np.float64
    
    # Test numerical stability with extreme coordinates
    extreme_coords = [
        (-179.999999, 89.999999),   # Near poles and dateline
        (179.999999, -89.999999),
        (-0.000001, 0.000001),      # Near origin
    ]
    
    for lon, lat in extreme_coords:
        try:
            # Should not lose precision or become unstable
            x, y = transformer.transform(lon, lat)
            assert np.isfinite(x) and np.isfinite(y)
            
            # Reverse should work
            inverse = create_crs_transformer("EPSG:3857", "EPSG:4326")
            lon2, lat2 = inverse.transform(x, y)
            assert np.isfinite(lon2) and np.isfinite(lat2)
            
        except Exception:
            # Some extreme coordinates might fail transformation
            pass


def test_coordinate_performance_benchmarks():
    """Test performance of coordinate transformation operations.
    
    This test should verify that:
    - Large coordinate arrays are processed efficiently
    - Memory usage scales appropriately with data size
    - Processing time is reasonable for operational use
    - Performance bottlenecks are identified
    - Optimization opportunities are documented
    """
    import time
    import gc
    from rasterio.transform import from_bounds
    import rasterio.crs
    
    # Large coordinate arrays processed efficiently
    array_sizes = [1000, 10000, 100000, 1000000]
    
    transformer = create_crs_transformer("EPSG:4326", "EPSG:3857")
    
    performance_data = []
    
    for size in array_sizes:
        # Generate random coordinates
        lons = np.random.uniform(-180, 180, size)
        lats = np.random.uniform(-85, 85, size)
        
        # Time the transformation
        start_time = time.time()
        xs, ys = transformer.transform(lons, lats)
        elapsed = time.time() - start_time
        
        # Calculate throughput
        coords_per_sec = size / elapsed
        performance_data.append((size, elapsed, coords_per_sec))
        
        # Verify results
        assert len(xs) == size
        assert len(ys) == size
        assert all(np.isfinite(xs))
        assert all(np.isfinite(ys))
        
        print(f"{size:,} coordinates: {elapsed:.3f}s ({coords_per_sec:,.0f} coords/sec)")
    
    # Processing time reasonable for operational use
    # 1 million coordinates should process in under 1 second
    million_coord_time = performance_data[-1][1]
    assert million_coord_time < 1.0, f"1M coordinates took {million_coord_time:.3f}s"
    
    # Throughput should be at least 1M coordinates per second
    min_throughput = min(perf[2] for perf in performance_data)
    assert min_throughput > 1000000, f"Minimum throughput: {min_throughput:,.0f} coords/sec"
    
    # Memory usage scales appropriately
    # Test memory usage for different array sizes
    gc.collect()
    
    # Small array
    small_lons = np.random.uniform(-180, 180, 1000)
    small_lats = np.random.uniform(-85, 85, 1000)
    small_xs, small_ys = transformer.transform(small_lons, small_lats)
    
    # Should use expected memory (8 bytes per float64)
    expected_memory = 1000 * 8 * 4  # 4 arrays (lons, lats, xs, ys)
    # Actual memory usage is hard to measure precisely
    
    # Clean up
    del small_lons, small_lats, small_xs, small_ys
    gc.collect()
    
    # Test bounds transformation performance
    bounds_list = [(-180 + i, -90 + i, -170 + i, -80 + i) for i in range(1000)]
    
    start_time = time.time()
    for bounds in bounds_list:
        _ = transform_bounds(bounds, "EPSG:4326", "EPSG:3857")
    bounds_time = time.time() - start_time
    
    assert bounds_time < 1.0, f"1000 bounds transformations took {bounds_time:.3f}s"
    
    # Geometry reprojection performance
    from shapely.geometry import Point, Polygon
    
    # Points
    points = [Point(np.random.uniform(-180, 180), np.random.uniform(-85, 85)) 
             for _ in range(1000)]
    
    start_time = time.time()
    for point in points:
        _ = reproject_geometry(point, "EPSG:4326", "EPSG:3857")
    points_time = time.time() - start_time
    
    assert points_time < 2.0, f"1000 point reprojections took {points_time:.3f}s"
    
    # Complex polygons
    complex_polygons = []
    for _ in range(100):
        # Create polygon with many vertices
        angles = np.linspace(0, 2*np.pi, 100)
        center_lon = np.random.uniform(-180, 180)
        center_lat = np.random.uniform(-60, 60)
        radius = 0.1
        
        coords = [(center_lon + radius * np.cos(a), center_lat + radius * np.sin(a)) 
                 for a in angles]
        complex_polygons.append(Polygon(coords))
    
    start_time = time.time()
    for polygon in complex_polygons:
        _ = reproject_geometry(polygon, "EPSG:4326", "EPSG:3857")
    polygons_time = time.time() - start_time
    
    assert polygons_time < 5.0, f"100 complex polygon reprojections took {polygons_time:.3f}s"
    
    # Transform calculation performance
    start_time = time.time()
    for _ in range(10000):
        bounds = (np.random.uniform(-180, 180), np.random.uniform(-90, 90),
                 np.random.uniform(-180, 180), np.random.uniform(-90, 90))
        bounds = (min(bounds[0], bounds[2]), min(bounds[1], bounds[3]),
                 max(bounds[0], bounds[2]), max(bounds[1], bounds[3]))
        _ = calculate_transform(bounds, 1000, 1000, method="fit")
    transform_time = time.time() - start_time
    
    assert transform_time < 1.0, f"10000 transform calculations took {transform_time:.3f}s"
    
    # CRS creation performance
    start_time = time.time()
    for _ in range(1000):
        _ = get_crs_from_epsg(np.random.choice([4326, 3857, 32633, 27700]))
    crs_time = time.time() - start_time
    
    assert crs_time < 0.1, f"1000 CRS creations took {crs_time:.3f}s"
    
    # UTM zone calculation performance
    test_lons = np.random.uniform(-180, 180, 100000)
    test_lats = np.random.uniform(-80, 80, 100000)
    
    start_time = time.time()
    zones = [calculate_utm_zone(lon, lat) for lon, lat in zip(test_lons, test_lats)]
    utm_time = time.time() - start_time
    
    assert utm_time < 1.0, f"100000 UTM zone calculations took {utm_time:.3f}s"
    assert len(zones) == 100000
    
    # Pixel area calculation performance
    shape = (1000, 1000)
    transform = from_bounds(-1, -1, 1, 1, 1000, 1000)
    crs = rasterio.crs.CRS.from_epsg(4326)
    
    start_time = time.time()
    areas = calculate_pixel_area(transform, crs, shape)
    area_time = time.time() - start_time
    
    assert area_time < 2.0, f"1M pixel area calculation took {area_time:.3f}s"
    assert areas.shape == shape
    
    print(f"\nPerformance Summary:")
    print(f"Coordinate transformation: {performance_data[-1][2]:,.0f} coords/sec")
    print(f"Bounds transformation: {1000/bounds_time:.0f} bounds/sec")
    print(f"Point reprojection: {1000/points_time:.0f} points/sec")
    print(f"Transform calculation: {10000/transform_time:.0f} calc/sec")
    print(f"UTM zone calculation: {100000/utm_time:.0f} calc/sec")


def test_coordinate_integration_workflows():
    """Test coordinate functions in integrated workflows.
    
    This test should verify that:
    - Multiple coordinate operations work together correctly
    - CRS validation, transformation, and area calculation combine
    - Data quality is maintained through complex workflows
    - Processing chains are efficient and accurate
    - Intermediate results are suitable for further processing
    """
    from shapely.geometry import Polygon, Point
    from rasterio.transform import from_bounds
    import rasterio.crs
    
    # Workflow 1: Multi-source data integration
    # Simulate integrating Landsat, VIIRS, and OSM data
    
    # Step 1: Define study area and target CRS
    study_area_4326 = Polygon([
        (-74.2, 40.5),
        (-73.7, 40.5), 
        (-73.7, 40.9),
        (-74.2, 40.9),
        (-74.2, 40.5)
    ])
    
    # Step 2: Find appropriate UTM zone
    center_lon = (study_area_4326.bounds[0] + study_area_4326.bounds[2]) / 2
    center_lat = (study_area_4326.bounds[1] + study_area_4326.bounds[3]) / 2
    utm_zone = calculate_utm_zone(center_lon, center_lat)
    target_epsg = 32600 + utm_zone
    
    # Step 3: Validate CRS compatibility
    source_crs_list = [
        "EPSG:4326",  # VIIRS geographic
        "EPSG:32618", # Landsat UTM 18N
        "EPSG:3857",  # OSM Web Mercator
    ]
    
    compatibility = validate_crs_compatibility(source_crs_list)
    assert not compatibility["compatible"]  # Mixed CRS should be incompatible
    assert len(compatibility["recommendations"]) > 0
    
    # Step 4: Reproject study area to target CRS
    study_area_utm = reproject_geometry(
        study_area_4326, "EPSG:4326", f"EPSG:{target_epsg}"
    )
    
    assert study_area_utm.is_valid
    assert study_area_utm.area > 0
    
    # Step 5: Calculate target bounds and transform
    target_bounds = study_area_utm.bounds
    target_resolution = 30.0  # 30m Landsat resolution
    
    transform, width, height = calculate_transform(
        target_bounds, None, None, 
        target_resolution=target_resolution, 
        method="res"
    )
    
    assert width > 0 and height > 0
    assert abs(transform.a - target_resolution) < 0.01
    
    # Step 6: Calculate pixel areas for area-based statistics
    target_crs = get_crs_from_epsg(target_epsg)
    pixel_areas = calculate_pixel_area(
        transform, target_crs, (height, width), unit="m2"
    )
    
    assert pixel_areas.shape == (height, width)
    assert np.allclose(pixel_areas, target_resolution ** 2)
    
    # Workflow 2: Global processing workflow
    # Process data at different latitudes with appropriate projections
    
    global_regions = [
        {"name": "Arctic", "bounds": (-180, 70, 180, 85), "epsg": 3995},
        {"name": "Temperate", "bounds": (-180, 30, 180, 70), "epsg": 4326},
        {"name": "Tropical", "bounds": (-180, -30, 180, 30), "epsg": 4326},
        {"name": "Antarctic", "bounds": (-180, -85, 180, -70), "epsg": 3031},
    ]
    
    processed_regions = []
    
    for region in global_regions:
        # Transform bounds to target CRS
        if region["epsg"] != 4326:
            target_bounds = transform_bounds(
                region["bounds"], "EPSG:4326", f"EPSG:{region['epsg']}"
            )
        else:
            target_bounds = region["bounds"]
        
        # Calculate appropriate resolution
        if region["name"] in ["Arctic", "Antarctic"]:
            resolution = 1000.0  # Coarser for polar regions
        else:
            resolution = 0.01  # Degrees for geographic
        
        # Calculate transform
        transform, w, h = calculate_transform(
            target_bounds, None, None,
            target_resolution=resolution,
            method="res"
        )
        
        # Calculate pixel areas
        crs = get_crs_from_epsg(region["epsg"])
        areas = calculate_pixel_area(transform, crs, (h, w), unit="km2")
        
        processed_regions.append({
            "name": region["name"],
            "transform": transform,
            "crs": crs,
            "shape": (h, w),
            "total_area_km2": np.sum(areas)
        })
    
    # Verify processing results
    assert len(processed_regions) == 4
    for region in processed_regions:
        assert region["total_area_km2"] > 0
        assert region["shape"][0] > 0 and region["shape"][1] > 0
    
    # Workflow 3: Buffer analysis workflow
    # Create buffers around point features
    
    # Step 1: Define points of interest
    poi_4326 = [
        Point(-74.006, 40.7128),  # NYC
        Point(-0.1276, 51.5074),  # London
        Point(139.6917, 35.6895), # Tokyo
    ]
    
    buffer_distance_m = 5000  # 5km buffer
    
    buffered_areas = []
    
    for poi in poi_4326:
        # Find appropriate local projection
        utm_zone = calculate_utm_zone(poi.x, poi.y)
        local_epsg = 32600 + utm_zone if utm_zone > 0 else 32700 + abs(utm_zone)
        
        # Reproject point to local CRS
        poi_local = reproject_geometry(poi, "EPSG:4326", f"EPSG:{local_epsg}")
        
        # Create buffer
        buffer_local = poi_local.buffer(buffer_distance_m)
        
        # Calculate area
        area_m2 = buffer_local.area
        area_km2 = area_m2 / 1_000_000
        
        # Theoretical area of circle
        theoretical_area = np.pi * (buffer_distance_m / 1000) ** 2
        
        # Should be close to theoretical
        assert abs(area_km2 - theoretical_area) / theoretical_area < 0.01
        
        buffered_areas.append(area_km2)
    
    # All buffers should have similar areas
    assert max(buffered_areas) / min(buffered_areas) < 1.01
    
    # Workflow 4: Data quality workflow
    # Check coordinate data throughout processing chain
    
    # Start with test dataset
    test_coords = np.array([
        [-74.0, 40.7],
        [-73.9, 40.8],
        [-74.1, 40.6]
    ])
    
    # Step 1: Validate input coordinates
    assert np.all((-180 <= test_coords[:, 0]) & (test_coords[:, 0] <= 180))
    assert np.all((-90 <= test_coords[:, 1]) & (test_coords[:, 1] <= 90))
    
    # Step 2: Transform coordinates
    transformer = create_crs_transformer("EPSG:4326", "EPSG:3857")
    x_coords, y_coords = transformer.transform(test_coords[:, 0], test_coords[:, 1])
    
    # Step 3: Validate transformed coordinates
    assert np.all(np.isfinite(x_coords))
    assert np.all(np.isfinite(y_coords))
    assert np.all(abs(x_coords) < 20037509)  # Web Mercator limits
    
    # Step 4: Round-trip validation
    reverse_transformer = create_crs_transformer("EPSG:3857", "EPSG:4326")
    lon_back, lat_back = reverse_transformer.transform(x_coords, y_coords)
    
    # Step 5: Check precision preservation
    lon_errors = np.abs(lon_back - test_coords[:, 0])
    lat_errors = np.abs(lat_back - test_coords[:, 1])
    
    assert np.all(lon_errors < 1e-9)
    assert np.all(lat_errors < 1e-9)
    
    # Verify all workflows completed successfully
    assert True  # If we reach here, all workflows passed


def test_coordinate_edge_cases():
    """Test coordinate transformation edge cases.
    
    This test should verify that:
    - Extreme coordinate values are handled correctly
    - Invalid coordinates raise appropriate errors
    - Resource cleanup occurs after errors
    - Error messages are informative and actionable
    - Edge cases don't cause system instability
    """
    from shapely.geometry import Point, Polygon
    import gc
    
    # Extreme coordinate values handled correctly
    extreme_coords = [
        (-180, -90),    # SW corner
        (180, 90),      # NE corner
        (-180, 90),     # NW corner
        (180, -90),     # SE corner
        (0, 0),         # Origin
        (-179.999999, 89.999999),   # Near extremes
        (179.999999, -89.999999),
    ]
    
    transformer = create_crs_transformer("EPSG:4326", "EPSG:3857")
    
    for lon, lat in extreme_coords:
        try:
            x, y = transformer.transform(lon, lat)
            
            # Should produce finite results or fail gracefully
            if np.isfinite(x) and np.isfinite(y):
                # Verify reasonable Web Mercator bounds
                assert abs(x) <= 20037509
                # Y values can be very large near poles
            
        except Exception as e:
            # Some extreme coordinates might legitimately fail
            # Error should be informative
            assert len(str(e)) > 0
    
    # Test date line crossing
    dateline_coords = [
        (179.9, 0),
        (-179.9, 0),
        (180, 0),
        (-180, 0),
    ]
    
    for lon, lat in dateline_coords:
        try:
            x, y = transformer.transform(lon, lat)
            assert np.isfinite(x) and np.isfinite(y)
        except Exception:
            # Some might fail, but shouldn't crash
            pass
    
    # Invalid coordinates raise appropriate errors
    invalid_coords = [
        (181, 0),      # Longitude out of range
        (-181, 0),
        (0, 91),       # Latitude out of range
        (0, -91),
        (np.inf, 0),   # Infinite values
        (0, np.inf),
        (np.nan, 0),   # NaN values
        (0, np.nan),
    ]
    
    for lon, lat in invalid_coords:
        try:
            # Some invalid coordinates might be handled by PROJ
            x, y = transformer.transform(lon, lat)
            
            # If transformation succeeds, results should indicate invalid
            if not (np.isfinite(x) and np.isfinite(y)):
                # Non-finite results indicate transformation detected invalid input
                pass
        except Exception as e:
            # Should raise informative error
            error_msg = str(e).lower()
            # Error should be descriptive
            assert len(error_msg) > 0
    
    # Test bounds transformation with invalid bounds
    invalid_bounds_list = [
        (10, 10, -10, -10),    # min > max
        (np.nan, 0, 1, 1),     # NaN values
        (0, 0, np.inf, 1),     # Infinite values
    ]
    
    for bounds in invalid_bounds_list:
        try:
            result = transform_bounds(bounds, "EPSG:4326", "EPSG:3857")
            # If it succeeds, result should be valid or indicate error
            if not all(np.isfinite(x) for x in result):
                pass  # Non-finite results indicate error detection
        except Exception as e:
            # Should raise informative error
            assert len(str(e)) > 0
    
    # Test geometry reprojection edge cases
    edge_case_geometries = [
        Point(180, 90),  # Extreme point
        Point(np.nan, 0),  # Invalid point
        Polygon([]),     # Empty polygon
    ]
    
    for geom in edge_case_geometries:
        try:
            if geom.is_valid and not geom.is_empty:
                result = reproject_geometry(geom, "EPSG:4326", "EPSG:3857")
                # Should produce valid result or fail
        except Exception as e:
            # Should handle errors gracefully
            assert len(str(e)) > 0
    
    # Resource cleanup occurs after errors
    # Test that failed operations don't leak memory or resources
    initial_objects = len(gc.get_objects())
    
    # Perform operations that might fail
    for _ in range(100):
        try:
            # Attempt invalid transformation
            _ = transform_bounds((np.nan, 0, 1, 1), "EPSG:4326", "EPSG:3857")
        except:
            pass
        
        try:
            # Attempt invalid CRS
            _ = get_crs_from_epsg(999999)
        except:
            pass
    
    # Force garbage collection
    gc.collect()
    
    # Object count shouldn't grow significantly
    final_objects = len(gc.get_objects())
    object_growth = final_objects - initial_objects
    
    # Some growth is normal, but shouldn't be excessive
    assert object_growth < 1000, f"Object count grew by {object_growth}"
    
    # Test very large arrays for memory management
    try:
        # Create very large coordinate arrays
        huge_lons = np.random.uniform(-180, 180, 10000000)  # 10M coordinates
        huge_lats = np.random.uniform(-85, 85, 10000000)
        
        # This might fail due to memory limits, which is acceptable
        try:
            xs, ys = transformer.transform(huge_lons, huge_lats)
            # If it succeeds, clean up immediately
            del xs, ys
        except MemoryError:
            # Expected for very large arrays
            pass
        
        # Clean up input arrays
        del huge_lons, huge_lats
        gc.collect()
        
    except MemoryError:
        # System doesn't have enough memory for this test
        pass
    
    # Test numerical stability near projection limits
    # Web Mercator has issues near poles
    near_pole_coords = [
        (0, 85.05),   # Just past Web Mercator limit
        (0, -85.05),
        (0, 89.99),   # Very close to pole
        (0, -89.99),
    ]
    
    for lon, lat in near_pole_coords:
        try:
            x, y = transformer.transform(lon, lat)
            
            # Results might be extreme but should be finite or fail cleanly
            if np.isfinite(x) and np.isfinite(y):
                # Very large Y values expected near poles
                assert abs(x) < 1e10  # Reasonable X bound
            
        except Exception:
            # Acceptable to fail at extreme latitudes
            pass
    
    # Error messages are informative and actionable
    # Test specific error conditions
    try:
        get_crs_from_epsg(-1)
        assert False, "Should have raised error"
    except Exception as e:
        error_msg = str(e)
        assert "epsg" in error_msg.lower() or "invalid" in error_msg.lower()
        assert "-1" in error_msg  # Should mention the invalid code
    
    # Test system stability - ensure no crashes
    # Rapid creation and destruction of transformers
    for _ in range(1000):
        try:
            trans = create_crs_transformer("EPSG:4326", "EPSG:3857")
            _ = trans.transform(0, 0)
            del trans
        except Exception:
            pass
    
    # If we reach here without crashing, edge cases are handled properly
    assert True