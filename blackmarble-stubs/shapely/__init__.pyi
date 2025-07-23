"""Type stubs for shapely."""

from .geometry import (
    Point, 
    LineString,
    Polygon,
    MultiPoint,
    MultiLineString,
    MultiPolygon,
    GeometryCollection,
    box
)
from .ops import unary_union, cascaded_union

__all__ = [
    "Point",
    "LineString", 
    "Polygon",
    "MultiPoint",
    "MultiLineString",
    "MultiPolygon",
    "GeometryCollection",
    "box",
    "unary_union",
    "cascaded_union"
]