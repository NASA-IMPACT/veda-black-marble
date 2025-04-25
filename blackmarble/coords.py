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

def wkt_to_json(wkt_string):
    """Parse a WKT projection string into a JSON dict."""
    proj = Proj(wkt_string)
    return proj.crs.to_json_dict()

def get_utm_zone(latitude, longitude):
    """Return (easting, northing, zone_number, zone_letter) for given lat/lon."""
    return utm.from_latlon(latitude, longitude)

def calculate_bounding_box_area(min_lon, min_lat, max_lon, max_lat):
    """Compute bounding box area in square kilometers via geodesic distances."""
    sw = (min_lat, min_lon)
    ne = (max_lat, max_lon)
    # width: south-west to south-east
    width = geodesic((min_lat, min_lon), (min_lat, max_lon)).meters
    # height: south-west to north-west
    height = geodesic((min_lat, min_lon), (max_lat, min_lon)).meters
    area_m2 = width * height
    return area_m2 / 1e6

def get_test_coord():
    """Example bbox (test)."""
    return 33.034682, 21.279194, 32.465949, 22.919832

def get_peurto_coord():
    """Example Puerto Rico bbox."""
    return 18.518435, -67.244568, 17.918391, -65.788879

def get_fort_myers():
    """Example Fort Myers bbox."""
    return 26.718905, -82.103920, 26.520302, -81.833752

def turkey():
    """Example Turkey bbox."""
    return 38.256313, 35.385847, 36.448697, 36.448697