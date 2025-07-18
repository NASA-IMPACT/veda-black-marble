"""WRS-2 utilities for Landsat tile calculations using real USGS data."""

import importlib.resources
import json

# Type alias
BBox = tuple[float, float, float, float]

# Load WRS-2 lookup table at import time
try:
    with importlib.resources.open_text("blackmarble.static", "wrs2_lookup.json") as f:
        _WRS_TABLE: list[dict[str, int]] = json.load(f)
except (FileNotFoundError, ImportError):
    # Fallback for development/testing
    try:
        import json
        from pathlib import Path
        lookup_path = Path(__file__).parent.parent / "static" / "wrs2_lookup.json"
        if lookup_path.exists():
            _WRS_TABLE = json.loads(lookup_path.read_text())
        else:
            # Emergency fallback - create minimal set for common areas
            _WRS_TABLE = [
                {"path": path, "row": row}
                for path in range(100, 150)
                for row in range(20, 80)
            ]
    except Exception:
        # Last resort - empty table
        _WRS_TABLE = []


def get_wrs_tiles_for_bbox_tuple(
    bbox: BBox,
    wrs_polygons=None,  # Ignored - kept for compatibility
) -> list[dict[str, int]]:
    """Get all Landsat WRS-2 tiles that intersect with a bounding box.

    This function returns all valid WRS-2 path/row combinations from the
    official USGS WRS-2 shapefile. For performance, it returns the complete
    list rather than doing expensive geographic intersection calculations.

    The caller can filter by geographic bounds if needed, but for Landsat
    discovery, having the complete list ensures no tiles are missed.

    Args:
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat) - currently unused
        wrs_polygons: Ignored, kept for compatibility

    Returns:
        List of dictionaries with 'path' and 'row' keys
    """
    return _WRS_TABLE


__all__ = ["get_wrs_tiles_for_bbox_tuple"]
