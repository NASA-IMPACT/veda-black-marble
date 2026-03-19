"""Type stubs for rasterio.transform."""

from collections.abc import Sequence
from typing import Any

class Affine:
    """2D affine transformation matrix."""

    a: float
    b: float
    c: float
    d: float
    e: float
    f: float

    def __init__(
        self,
        a: float = 1.0,
        b: float = 0.0,
        c: float = 0.0,
        d: float = 0.0,
        e: float = 1.0,
        f: float = 0.0,
    ) -> None: ...
    @classmethod
    def identity(cls) -> Affine: ...
    @classmethod
    def translation(cls, xoff: float, yoff: float) -> Affine: ...
    @classmethod
    def scale(cls, sx: float, sy: float | None = None) -> Affine: ...
    @classmethod
    def rotation(cls, angle: float) -> Affine: ...
    def __mul__(self, other: Affine) -> Affine: ...
    def __rmul__(self, other: Affine) -> Affine: ...
    def __invert__(self) -> Affine: ...
    def __getitem__(self, index: int) -> float: ...
    @property
    def determinant(self) -> float: ...
    def to_shapely(self) -> tuple[float, float, float, float, float, float]: ...
    def __iter__(self) -> Any: ...
    def xy(self, row: float, col: float) -> tuple[float, float]: ...
    def rowcol(self, x: float, y: float) -> tuple[float, float]: ...

def from_bounds(
    west: float, south: float, east: float, north: float, width: int, height: int
) -> Affine: ...
def from_origin(west: float, north: float, xsize: float, ysize: float) -> Affine: ...
def array_bounds(
    height: int, width: int, transform: Affine
) -> tuple[float, float, float, float]: ...
def rowcol(
    transform: Affine,
    xs: Sequence[float] | float,
    ys: Sequence[float] | float,
    op: Any = ...,
    precision: float | None = None,
) -> tuple[Any, Any]: ...
def xy(
    transform: Affine,
    rows: Sequence[float] | float,
    cols: Sequence[float] | float,
    offset: str = "center",
) -> tuple[Any, Any]: ...

__all__ = [
    "Affine",
    "from_bounds",
    "from_origin",
    "array_bounds",
    "rowcol",
    "xy",
]
