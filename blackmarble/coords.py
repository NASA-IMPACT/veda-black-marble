"""
Coordinate & projection helpers:
  - wkt_to_json: parse a WKT string to JSON dict
  - get_utm_zone: convert lat/lon to UTM coordinates
  - calculate_bounding_box_area: geodesic area in km²
  - get_test_coord, get_peurto_coord, get_fort_myers, turkey: example bounding boxes
"""
# Imports
import utm
from pyproj import Proj
from typing import Any
import pyproj

def wkt_to_json(wkt_string: str) -> dict[str, Any]:
    """Parse a WKT projection string into a JSON dict."""
    proj = Proj(wkt_string)
    return proj.crs.to_json_dict()

def get_utm_zone(latitude: float, longitude: float) -> tuple[float, float, int, str]:
    """Return (easting, northing, zone_number, zone_letter) for given lat/lon."""
    return utm.from_latlon(latitude, longitude)

def calculate_bounding_box_area(min_lon: float, min_lat: float,
                                max_lon: float, max_lat: float) -> float:
    """
    Calculate the geodesic area (km²) of an axis-aligned bounding box
    defined by its corner coordinates.
    """
    geod = pyproj.Geod(ellps='WGS84')

    # Vertices of the rectangle, closing the polygon by repeating the first
    lons = [min_lon, max_lon, max_lon, min_lon, min_lon]
    lats = [min_lat, min_lat, max_lat, max_lat, min_lat]

    # polygon_area_perimeter returns area in square metres (may be negative
    # depending on vertex order) and perimeter in metres.
    area_m2, _ = geod.polygon_area_perimeter(lons, lats)
    return abs(area_m2) / 1_000_000  # convert to km²

def get_test_coord() -> tuple[float, float, float, float]:
    """Example bbox (test)."""
    return 33.034682, 21.279194, 32.465949, 22.919832

def get_peurto_coord() -> tuple[float, float, float, float]:
    """Example Puerto Rico bbox."""
    return 18.518435, -67.244568, 17.918391, -65.788879

def get_fort_myers() -> tuple[float, float, float, float]:
    """Example Fort Myers bbox."""
    return 26.718905, -82.103920, 26.520302, -81.833752

def turkey() -> tuple[float, float, float, float]:
    """Hard-coded bounding box covering most of Türkiye."""
    return 38.256313, 35.385847, 36.448697, 36.448697