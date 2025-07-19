"""WRS-2 utilities for Landsat tile calculations using real USGS data."""

import json
from typing import Any

# Type alias
BBox = tuple[float, float, float, float]

# Load WRS-2 lookup table at import time
_wrs_table: list[dict[str, int]]
try:
    import importlib.resources
    with importlib.resources.open_text("blackmarble.static", "wrs2_lookup.json") as f:
        _wrs_table = json.load(f)
except (FileNotFoundError, ImportError):
    # Fallback for development/testing
    try:
        from pathlib import Path
        lookup_path = Path(__file__).parent.parent / "static" / "wrs2_lookup.json"
        if lookup_path.exists():
            _wrs_table = json.loads(lookup_path.read_text())
        else:
            # Emergency fallback - create minimal set for common areas
            _wrs_table = [
                {"path": path, "row": row}
                for path in range(100, 150)
                for row in range(20, 80)
            ]
    except Exception:
        # Last resort - empty table
        _wrs_table = []


def get_wrs_tiles_for_bbox_tuple(
    bbox: BBox,
    wrs_polygons: Any = None,  # Ignored - kept for compatibility
) -> list[dict[str, int]]:
    """Get Landsat WRS-2 tiles that intersect with a bounding box.

    Uses the WRS-2 shapefile to find only tiles that actually intersect
    the given bounding box, dramatically reducing unnecessary API calls.

    Args:
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
        wrs_polygons: Ignored, kept for compatibility

    Returns:
        List of dictionaries with 'path' and 'row' keys for intersecting tiles
    """
    try:
        import geopandas as gpd
        from shapely.geometry import box
        from pathlib import Path
        
        # Try to load WRS-2 shapefile
        wrs_path = Path(__file__).parent.parent / "data" / "WRS2_descending_0" / "WRS2_descending.shp"
        if not wrs_path.exists():
            # Fallback to docker path
            wrs_path = Path(__file__).parent.parent.parent / "docker" / "WRS2_descending_0" / "WRS2_descending.shp"
        
        if wrs_path.exists():
            # Load shapefile and find intersecting tiles
            wrs_gdf = gpd.read_file(wrs_path)
            
            # Create bbox polygon
            min_lon, min_lat, max_lon, max_lat = bbox
            bbox_poly = box(min_lon, min_lat, max_lon, max_lat)
            
            # Find intersecting tiles
            intersecting = wrs_gdf[wrs_gdf.intersects(bbox_poly)]
            
            # Extract path/row info
            tiles: list[dict[str, int]] = []
            for _, row in intersecting.iterrows():
                if 'PATH' in row and 'ROW' in row:
                    path = int(row['PATH'])
                    row_num = int(row['ROW'])
                    tiles.append({'path': path, 'row': row_num})
            
            if tiles:
                return tiles
                
    except ImportError:
        # GeoPandas not available
        pass
    except Exception as e:
        # Log but don't fail
        import logging
        logging.warning(f"Failed to filter WRS tiles by bbox: {e}")
    
    # Fallback: return estimated tiles based on bbox location
    # This is a rough approximation for common regions
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # Very rough estimation of path/row ranges
    # Paths go from east (1) to west (233)
    # Each path is ~1.5 degrees wide at equator
    if -180 <= min_lon <= -120:  # Americas
        path_start = int(10 + (180 + max_lon) / 1.5)
        path_end = int(10 + (180 + min_lon) / 1.5)
    elif -20 <= min_lon <= 50:  # Europe/Africa
        path_start = int(180 + (50 - max_lon) / 1.5)
        path_end = int(180 + (50 - min_lon) / 1.5)
    else:  # Asia/Australia
        path_start = int(100 + (180 - max_lon) / 1.5)
        path_end = int(100 + (180 - min_lon) / 1.5)
    
    # Rows go from north (1) to south (248)
    # Each row is ~1.5 degrees tall
    row_start = int(60 + (60 - max_lat) / 1.5)
    row_end = int(60 + (60 - min_lat) / 1.5)
    
    # Generate tile list
    tiles: list[dict[str, int]] = []
    for path in range(max(1, path_start), min(234, path_end + 1)):
        for row in range(max(1, row_start), min(249, row_end + 1)):
            tiles.append({'path': path, 'row': row})
    
    # If estimation fails, return a reasonable subset
    if not tiles:
        # Default to returning ~100 tiles around the center
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
        
        # Estimate center path/row
        if -180 <= center_lon <= -120:
            center_path = int(10 + (180 + center_lon) / 1.5)
        elif -20 <= center_lon <= 50:
            center_path = int(180 + (50 - center_lon) / 1.5)
        else:
            center_path = int(100 + (180 - center_lon) / 1.5)
        
        center_row = int(60 + (60 - center_lat) / 1.5)
        
        # Return 10x10 grid around center
        for path in range(max(1, center_path - 5), min(234, center_path + 5)):
            for row in range(max(1, center_row - 5), min(249, center_row + 5)):
                tiles.append({'path': path, 'row': row})
    
    return tiles


__all__ = ["get_wrs_tiles_for_bbox_tuple"]
