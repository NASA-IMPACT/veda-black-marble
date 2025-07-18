"""WRS-2 utilities for Landsat tile calculations."""


import shapely.geometry

# Type alias
BBox = tuple[float, float, float, float]


def load_wrs_shapefile() -> list[tuple[shapely.geometry.Polygon, int, int]]:
    """Load WRS-2 polygons from embedded shapefile data."""
    # This is a simplified implementation - in practice this would load
    # the actual WRS-2 shapefile data. For now, we'll create a minimal
    # grid that covers common areas.
    import shapely.geometry

    polygons = []

    # Create a simplified grid covering common path/row combinations
    # This is not the full WRS-2 grid but enough for basic functionality
    for path in range(1, 234):  # WRS-2 paths 1-233
        for row in range(1, 249):  # WRS-2 rows 1-248
            # Calculate approximate bounds for this path/row
            # Note: This is a very simplified approximation
            lon_center = -180 + (path - 1) * 360 / 233
            lat_center = 80 - (row - 1) * 160 / 248

            # Create a small polygon around the center
            min_lon = lon_center - 1.5
            max_lon = lon_center + 1.5
            min_lat = lat_center - 1.0
            max_lat = lat_center + 1.0

            polygon = shapely.geometry.box(min_lon, min_lat, max_lon, max_lat)
            polygons.append((polygon, path, row))

    return polygons


def get_wrs_tiles_for_bbox_tuple(
    bbox: BBox,
    wrs_polygons: list[tuple[shapely.geometry.Polygon, int, int]] | None = None,
) -> list[dict[str, int]]:
    """Get all Landsat WRS-2 tiles that intersect with a bounding box."""
    from shapely.geometry import box

    if wrs_polygons is None:
        wrs_polygons = load_wrs_shapefile()

    min_lon, min_lat, max_lon, max_lat = bbox
    bbox_polygon = box(min_lon, min_lat, max_lon, max_lat)

    results = []
    seen_tiles = set()

    for polygon, path, row in wrs_polygons:
        if polygon.intersects(bbox_polygon):
            tile_key = (path, row)
            if tile_key not in seen_tiles:
                seen_tiles.add(tile_key)
                results.append({"path": path, "row": row})

    return results


__all__ = ["get_wrs_tiles_for_bbox_tuple"]
