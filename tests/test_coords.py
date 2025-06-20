import pytest
from pyproj.exceptions import CRSError
import utm

from blackmarble import coords

# --- Test wkt_to_json ---


# Removed test_wkt_to_json_valid - only tests that pyproj library parsing works


def test_wkt_to_json_invalid():
    """Test wkt_to_json with invalid WKT strings."""
    with pytest.raises(CRSError):
        coords.wkt_to_json("INVALID WKT STRING")
    with pytest.raises(CRSError):
        coords.wkt_to_json("")  # Empty string is also invalid


# --- Test get_utm_zone ---


def test_get_utm_zone_basic():
    """Test get_utm_zone with known coordinates."""
    # Example: Somewhere in UTM Zone 10N (Vancouver, BC)
    lat, lon = 49.2827, -123.1207
    easting, northing, zone_num, zone_letter = coords.get_utm_zone(lat, lon)
    assert zone_num == 10
    assert zone_letter == "U"
    # Check if easting/northing are approximately correct (using utm library directly for comparison)
    expected_easting, expected_northing, _, _ = utm.from_latlon(lat, lon)
    assert easting == pytest.approx(expected_easting)
    assert northing == pytest.approx(expected_northing)


def test_get_utm_zone_equator():
    """Test get_utm_zone near the equator."""
    lat, lon = 0.0, 10.0  # Zone 32N
    easting, northing, zone_num, zone_letter = coords.get_utm_zone(lat, lon)
    assert zone_num == 32
    assert zone_letter == "N"
    assert northing == pytest.approx(0.0)


def test_get_utm_zone_antimeridian():
    """Test get_utm_zone near the antimeridian."""
    # Crossing from West to East (e.g., -179.9 -> 179.9) should change zone
    lat = 50.0
    _, _, zone_num_west, _ = coords.get_utm_zone(lat, -179.9)  # Zone 1
    _, _, zone_num_east, _ = coords.get_utm_zone(lat, 179.9)  # Zone 60
    assert zone_num_west == 1
    assert zone_num_east == 60

    # Test wrap-around (library might handle this automatically)
    # utm library handles > 180 or < -180 by wrapping


def test_get_utm_zone_poles():
    """Test get_utm_zone near the poles (utm library has limitations)."""
    # Note: The 'utm' library might not be accurate very close to poles.
    # Test behavior within its typical operating range.
    lat_north = 83.9  # Close to North Pole limit (84N)
    lon = 0.0
    _, _, zone_num_n, zone_letter_n = coords.get_utm_zone(lat_north, lon)
    # The utm library uses letters C–X; 72°–84° N is letter 'X'.
    assert zone_letter_n == "X"

    lat_south = -79.9  # Close to South Pole limit (80S)
    _, _, zone_num_s, zone_letter_s = coords.get_utm_zone(lat_south, lon)
    assert zone_letter_s == "C"  # Special polar zone handling might occur


# --- Test calculate_bounding_box_area ---


def test_calculate_bounding_box_area_small():
    """Test calculate_bounding_box_area with a small, known area."""
    # Approx 1km square near equator (lon diff approx 1/111 km, lat diff approx 1/111 km)
    min_lat, min_lon = 0.0, 0.0
    max_lat, max_lon = 1.0 / 111.0, 1.0 / 111.0  # Approx 1km north and east
    area = coords.calculate_bounding_box_area(min_lon, min_lat, max_lon, max_lat)
    # Should be roughly 1 km * 1 km = 1 km^2
    assert area == pytest.approx(
        1.0, abs=0.1
    )  # Allow some tolerance due to projection/geodesic


# Removed test_calculate_bounding_box_area_zero - only tests basic edge case that area of zero-size box is zero


def test_calculate_bounding_box_area_invalid_order():
    """Test calculate_bounding_box_area with potentially swapped min/max."""
    # Current implementation calculates width/height based on corners,
    # so order might not strictly matter, but let's verify consistency.
    min_lat, min_lon = 10.0, 20.0
    max_lat, max_lon = 11.0, 21.0
    area1 = coords.calculate_bounding_box_area(min_lon, min_lat, max_lon, max_lat)
    # Swapped lat
    area2 = coords.calculate_bounding_box_area(min_lon, max_lat, max_lon, min_lat)
    # Swapped lon
    area3 = coords.calculate_bounding_box_area(max_lon, min_lat, min_lon, max_lat)
    # Swapped both
    area4 = coords.calculate_bounding_box_area(max_lon, max_lat, min_lon, min_lat)

    # Due to how width/height are calculated using fixed corners, these might differ.
    # The key is that the result should be non-negative.
    assert area1 >= 0
    assert (
        area2 >= 0
    )  # This might be zero if height becomes negative in geodesic calc? Check impl.
    assert area3 >= 0  # This might be zero if width becomes negative? Check impl.
    assert area4 >= 0

    # Let's re-check the implementation: it uses geodesic distance between specific pairs.
    # geodesic((min_lat, min_lon), (min_lat, max_lon)) # width at min_lat
    # geodesic((min_lat, min_lon), (max_lat, min_lon)) # height at min_lon
    # This calculation is an approximation (area of rectangle on projected plane).
    # Swapping inputs will change which points are used for width/height.
    # The test should primarily ensure non-negativity.
    # A more robust area calculation might use shapely or pyproj.
    assert area1 > 0  # Expect positive area for non-zero box
    # The other areas might be different, but should be non-negative.


# --- Test Example Bounding Boxes ---


# Removed hardcoded coordinate tests - only verify constant tuples return constant tuples:
# - test_get_test_coord: checks (33.034682, 21.279194, 32.465949, 22.919832)
# - test_get_peurto_coord: checks (18.518435, -67.244568, 17.918391, -65.788879)
# - test_get_fort_myers: checks (26.718905, -82.103920, 26.520302, -81.833752)
# - test_turkey: checks (38.256313, 35.385847, 36.448697, 36.448697)
# These are configuration constants, not business logic to test.
