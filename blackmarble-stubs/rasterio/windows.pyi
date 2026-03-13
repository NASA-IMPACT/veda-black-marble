"""Type stubs for rasterio.windows."""

from .transform import Affine

class Window:
    """A window of a raster."""

    col_off: float
    row_off: float
    width: float
    height: float

    def __init__(self, col_off: float, row_off: float, width: float, height: float) -> None: ...
    @classmethod
    def from_slices(
        cls,
        rows: tuple[int, int] | slice,
        cols: tuple[int, int] | slice,
        boundless: bool = False,
        width: int | None = None,
        height: int | None = None,
    ) -> Window: ...
    def intersection(self, other: Window) -> Window: ...
    def union(self, other: Window) -> Window: ...
    def round_lengths(self, op: str = "floor", pixel_precision: int = 0) -> Window: ...
    def round_offsets(self, op: str = "floor", pixel_precision: int = 0) -> Window: ...
    def toranges(self) -> tuple[tuple[float, float], tuple[float, float]]: ...
    def flatten(self) -> list[float]: ...
    @property
    def width(self) -> float: ...
    @property
    def height(self) -> float: ...

def from_bounds(
    left: float,
    bottom: float,
    right: float,
    top: float,
    transform: Affine,
    width: int | None = None,
    height: int | None = None,
    boundless: bool = False,
    precision: float | None = None,
) -> Window: ...
def transform(window: Window, transform: Affine) -> Affine: ...
def bounds(window: Window, transform: Affine) -> tuple[float, float, float, float]: ...

__all__ = ["Window", "from_bounds", "transform", "bounds"]
