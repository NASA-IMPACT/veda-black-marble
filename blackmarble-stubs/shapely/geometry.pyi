"""Type stubs for shapely.geometry."""

from typing import Any, Sequence, Union, overload
import numpy.typing as npt

CoordinateSequence = Union[
    Sequence[tuple[float, float]],
    Sequence[tuple[float, float, float]],
    npt.NDArray[Any]
]

class BaseGeometry:
    """Base class for all geometries."""
    
    @property
    def bounds(self) -> tuple[float, float, float, float]: ...
    
    @property
    def area(self) -> float: ...
    
    @property
    def length(self) -> float: ...
    
    @property
    def is_empty(self) -> bool: ...
    
    @property
    def is_valid(self) -> bool: ...
    
    @property
    def is_simple(self) -> bool: ...
    
    @property
    def is_ring(self) -> bool: ...
    
    @property
    def wkt(self) -> str: ...
    
    @property
    def __geo_interface__(self) -> dict[str, Any]: ...
    
    def buffer(self, distance: float, resolution: int = 16) -> BaseGeometry: ...
    
    def contains(self, other: BaseGeometry) -> bool: ...
    
    def crosses(self, other: BaseGeometry) -> bool: ...
    
    def disjoint(self, other: BaseGeometry) -> bool: ...
    
    def equals(self, other: BaseGeometry) -> bool: ...
    
    def intersects(self, other: BaseGeometry) -> bool: ...
    
    def overlaps(self, other: BaseGeometry) -> bool: ...
    
    def touches(self, other: BaseGeometry) -> bool: ...
    
    def within(self, other: BaseGeometry) -> bool: ...
    
    def intersection(self, other: BaseGeometry) -> BaseGeometry: ...
    
    def union(self, other: BaseGeometry) -> BaseGeometry: ...
    
    def difference(self, other: BaseGeometry) -> BaseGeometry: ...
    
    def symmetric_difference(self, other: BaseGeometry) -> BaseGeometry: ...

class Point(BaseGeometry):
    """Point geometry."""
    
    @overload
    def __init__(self, x: float, y: float) -> None: ...
    
    @overload
    def __init__(self, coords: tuple[float, float] | tuple[float, float, float]) -> None: ...
    
    @property
    def x(self) -> float: ...
    
    @property
    def y(self) -> float: ...
    
    @property
    def z(self) -> float | None: ...
    
    @property
    def coords(self) -> CoordinateSequence: ...

class LineString(BaseGeometry):
    """LineString geometry."""
    
    def __init__(self, coordinates: CoordinateSequence | None = None) -> None: ...
    
    @property
    def coords(self) -> CoordinateSequence: ...

class Polygon(BaseGeometry):
    """Polygon geometry."""
    
    def __init__(
        self,
        shell: CoordinateSequence | LinearRing | None = None,
        holes: Sequence[CoordinateSequence | LinearRing] | None = None
    ) -> None: ...
    
    @property
    def exterior(self) -> LinearRing: ...
    
    @property
    def interiors(self) -> list[LinearRing]: ...

class LinearRing(LineString):
    """Linear ring (closed LineString)."""
    pass

class MultiPoint(BaseGeometry):
    """Collection of Points."""
    
    def __init__(self, points: Sequence[Point | tuple[float, float]] | None = None) -> None: ...
    
    @property
    def geoms(self) -> list[Point]: ...

class MultiLineString(BaseGeometry):
    """Collection of LineStrings."""
    
    def __init__(self, lines: Sequence[LineString | CoordinateSequence] | None = None) -> None: ...
    
    @property
    def geoms(self) -> list[LineString]: ...

class MultiPolygon(BaseGeometry):
    """Collection of Polygons."""
    
    def __init__(self, polygons: Sequence[Polygon] | None = None) -> None: ...
    
    @property
    def geoms(self) -> list[Polygon]: ...

class GeometryCollection(BaseGeometry):
    """Collection of heterogeneous geometries."""
    
    def __init__(self, geoms: Sequence[BaseGeometry] | None = None) -> None: ...
    
    @property
    def geoms(self) -> list[BaseGeometry]: ...

def box(minx: float, miny: float, maxx: float, maxy: float) -> Polygon:
    """Create a rectangular polygon."""
    ...

__all__ = [
    "BaseGeometry",
    "Point",
    "LineString",
    "LinearRing",
    "Polygon",
    "MultiPoint",
    "MultiLineString",
    "MultiPolygon",
    "GeometryCollection",
    "box"
]