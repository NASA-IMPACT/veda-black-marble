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
from geopy.distance import geodesic
from typing import Dict, Any, Tuple
import pyproj

def wkt_to_json(wkt_string: str) -> Dict[str, Any]:
    """Parse a WKT projection string into a JSON dict."""
    proj = Proj(wkt_string)
    return proj.crs.to_json_dict()

def get_utm_zone(latitude: float, longitude: float) -> Tuple[float, float, int, str]:
    """Return (easting, northing, zone_number, zone_letter) for given lat/lon."""
    return utm.from_latlon(latitude, longitude)

def calculate_bounding_box_area(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> float:
    geodesic = pyproj.Geod(ellps='WGS84')
    polygon = [(min_lon, min_lat), (max_lon, min_lat), (max_lon, max_lat), (min_lon, max_lat), (min_lon, min_lat)]
    area_km2, _ = geodesic.polygon_area_perimeter(polygon)
    return abs(area_km2) / 1e6

def get_test_coord() -> Tuple[float, float, float, float]:
    """Example bbox (test)."""
    return 33.034682, 21.279194, 32.465949, 22.919832

def get_peurto_coord() -> Tuple[float, float, float, float]:
    """Example Puerto Rico bbox."""
    return 18.518435, -67.244568, 17.918391, -65.788879

def get_fort_myers() -> Tuple[float, float, float, float]:
    """Example Fort Myers bbox."""
    return 26.718905, -82.103920, 26.520302, -81.833752

def turkey() -> Tuple[float, float, float, float]:
    return 38.256313, 35.385847, 42.448697, 36.448697