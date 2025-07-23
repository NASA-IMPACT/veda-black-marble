"""Type stubs for shapely.ops."""

from typing import Sequence
from .geometry import BaseGeometry

def unary_union(geoms: Sequence[BaseGeometry]) -> BaseGeometry:
    """Returns a geometry containing the union of all geometries in the input."""
    ...

def cascaded_union(geoms: Sequence[BaseGeometry]) -> BaseGeometry:
    """Deprecated: use unary_union instead."""
    ...

__all__ = ["unary_union", "cascaded_union"]