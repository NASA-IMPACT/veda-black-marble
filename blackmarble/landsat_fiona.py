"""
Fiona-based Landsat utilities - replacement for OGR operations.
"""

from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple
import shapely.geometry
from shapely.geometry import shape, Point
import fiona

# Get the default WRS shapefile path
from .landsat import DEFAULT_WRS_SHAPEFILE


class LandsatUtil:
    """
    Landsat utility for WRS-2 path/row lookups using Fiona.
    
    This is a drop-in replacement for the OGR-based implementation.
    """
    
    polygons: List[Tuple[shapely.geometry.Polygon, int, int]]
    
    def __init__(self, shapefile: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize with WRS-2 shapefile.
        
        Args:
            shapefile: Path to WRS-2 shapefile. If None, uses bundled default.
        """
        # Use bundled default WRS-2 shapefile if none provided
        if shapefile is None:
            shapefile = DEFAULT_WRS_SHAPEFILE
            
        self.shapefile_path = Path(shapefile)
        self.polygons = []
        
        # Read all features from the shapefile
        with fiona.open(self.shapefile_path, 'r') as collection:
            for feature in collection:
                # Get path and row from properties
                properties = feature['properties']
                path = int(properties['PATH'])
                row = int(properties['ROW'])
                
                # Convert geometry to shapely
                geom = shape(feature['geometry'])
                
                # Only store polygons (skip any other geometry types)
                if isinstance(geom, shapely.geometry.Polygon):
                    self.polygons.append((geom, path, row))
    
    def get_wrs(self, lat: float, lon: float) -> List[Dict[str, int]]:
        """
        Return list of {'path': path, 'row': row} for given lat/lon.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            List of dictionaries with 'path' and 'row' keys
        """
        point = Point(lon, lat)
        
        wrs_list = []
        for polygon, path, row in self.polygons:
            if polygon.contains(point):
                wrs_list.append({'path': path, 'row': row})
                
        return wrs_list
    
    def get_wrs_list(self, lat1: float, lon1: float, lat2: float, lon2: float) -> List[Dict[str, int]]:
        """
        Return all WRS path/rows within a bounding box.
        
        Args:
            lat1, lon1: First corner of bounding box
            lat2, lon2: Second corner of bounding box
            
        Returns:
            List of dictionaries with 'path' and 'row' keys
        """
        # Create bounding box
        min_lat = min(lat1, lat2)
        max_lat = max(lat1, lat2)
        min_lon = min(lon1, lon2)
        max_lon = max(lon1, lon2)
        
        bbox = shapely.geometry.box(min_lon, min_lat, max_lon, max_lat)
        
        wrs_list = []
        seen = set()  # Track unique path/row combinations
        
        for polygon, path, row in self.polygons:
            if polygon.intersects(bbox):
                key = (path, row)
                if key not in seen:
                    seen.add(key)
                    wrs_list.append({'path': path, 'row': row})
                    
        return wrs_list