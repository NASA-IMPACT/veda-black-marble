"""Type stubs for rasterio.warp."""

from enum import IntEnum
from typing import Any

import numpy.typing as npt

from .crs import CRS
from .transform import Affine

class Resampling(IntEnum):
    """Available resampling methods."""

    nearest = 0
    bilinear = 1
    cubic = 2
    cubic_spline = 3
    lanczos = 4
    average = 5
    mode = 6
    gauss = 7
    max = 8
    min = 9
    med = 10
    q1 = 11
    q3 = 12
    sum = 13
    rms = 14

def reproject(
    source: npt.NDArray[Any],
    destination: npt.NDArray[Any] | None = None,
    src_transform: Affine | None = None,
    gcps: Any = None,
    rpcs: Any = None,
    src_crs: CRS | str | dict[str, Any] | None = None,
    src_nodata: float | None = None,
    dst_transform: Affine | None = None,
    dst_crs: CRS | str | dict[str, Any] | None = None,
    dst_nodata: float | None = None,
    dst_resolution: float | tuple[float, float] | None = None,
    src_alpha: int = 0,
    dst_alpha: int = 0,
    init_dest_nodata: bool = True,
    warp_mem_limit: int = 0,
    resampling: Resampling = Resampling.nearest,
    **kwargs: Any,
) -> tuple[npt.NDArray[Any], Affine]: ...
def calculate_default_transform(
    src_crs: CRS | str | dict[str, Any],
    dst_crs: CRS | str | dict[str, Any],
    width: int,
    height: int,
    left: float | None = None,
    bottom: float | None = None,
    right: float | None = None,
    top: float | None = None,
    gcps: Any = None,
    rpcs: Any = None,
    resolution: float | tuple[float, float] | None = None,
    dst_width: int | None = None,
    dst_height: int | None = None,
    **kwargs: Any,
) -> tuple[Affine, int, int]: ...
def transform_bounds(
    src_crs: CRS | str | dict[str, Any],
    dst_crs: CRS | str | dict[str, Any],
    left: float,
    bottom: float,
    right: float,
    top: float,
    densify_pts: int = 21,
) -> tuple[float, float, float, float]: ...
def transform_geom(
    src_crs: CRS | str | dict[str, Any],
    dst_crs: CRS | str | dict[str, Any],
    geom: dict[str, Any],
    antimeridian_cutting: bool = False,
    antimeridian_offset: float = 10.0,
    precision: float = -1,
) -> dict[str, Any]: ...

__all__ = [
    "Resampling",
    "reproject",
    "calculate_default_transform",
    "transform_bounds",
    "transform_geom",
]
