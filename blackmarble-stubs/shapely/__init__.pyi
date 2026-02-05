"""Type stubs for shapely."""

from .geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from .ops import cascaded_union, unary_union

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
    "cascaded_union",
]
