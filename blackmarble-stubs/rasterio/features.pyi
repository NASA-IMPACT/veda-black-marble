"""Type stubs for rasterio.features."""

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from .transform import Affine

def geometry_mask(
    geometries: Sequence[dict[str, Any]] | Sequence[tuple[dict[str, Any], Any]],
    out_shape: tuple[int, int],
    transform: Affine,
    all_touched: bool = False,
    invert: bool = False,
) -> npt.NDArray[np.bool_]: ...
def rasterize(
    shapes: Sequence[tuple[dict[str, Any], Any]] | Sequence[dict[str, Any]],
    out_shape: tuple[int, int] | None = None,
    out: npt.NDArray[Any] | None = None,
    transform: Affine | None = None,
    all_touched: bool = False,
    fill: float = 0,
    merge_alg: Any = ...,
    default_value: float = 1,
    dtype: type[np.dtype[Any]] | np.dtype[Any] | str | None = None,
) -> npt.NDArray[Any]: ...
def shapes(
    image: npt.NDArray[Any],
    mask: npt.NDArray[np.bool_] | None = None,
    connectivity: int = 4,
    transform: Affine | None = None,
) -> Iterator[tuple[dict[str, Any], Any]]: ...

__all__ = ["geometry_mask", "rasterize", "shapes"]
