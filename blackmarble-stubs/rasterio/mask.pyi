"""Type stubs for rasterio.mask."""

from collections.abc import Sequence
from typing import Any

import numpy.typing as npt

from .transform import Affine

def mask(
    dataset: Any,
    shapes: Sequence[dict[str, Any]],
    all_touched: bool = False,
    invert: bool = False,
    nodata: float | None = None,
    filled: bool = True,
    crop: bool = True,
    pad: bool = False,
    pad_width: float = 0.5,
    indexes: int | Sequence[int] | None = None,
) -> tuple[npt.NDArray[Any], Affine]: ...
def geometry_mask(
    dataset: Any,
    shapes: Sequence[dict[str, Any]],
    all_touched: bool = False,
    invert: bool = False,
    burn_value: float = 1.0,
) -> npt.NDArray[Any]: ...

__all__ = ["mask", "geometry_mask"]
