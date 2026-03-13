"""Type stubs for rasterio."""

from typing import Any, Literal, TypedDict, overload

import numpy as np
import numpy.typing as npt

from .crs import CRS
from .transform import Affine
from .warp import Resampling

class Profile(TypedDict, total=False):
    """Raster dataset profile."""

    driver: str
    dtype: str | np.dtype[Any]
    width: int
    height: int
    count: int
    crs: CRS | str | dict[str, Any] | None
    transform: Affine
    nodata: float | int | None
    compress: str
    predictor: int
    tiled: bool
    blockxsize: int
    blockysize: int
    BIGTIFF: str

class DatasetReader:
    """Read-only dataset."""

    profile: Profile
    transform: Affine
    crs: CRS | None
    width: int
    height: int
    count: int
    dtypes: list[str]
    bounds: tuple[float, float, float, float]
    subdatasets: list[str]
    def read(
        self,
        indexes: int | list[int] | None = ...,
        window: Any = ...,
        masked: bool = ...,
        boundless: bool = ...,
        resampling: Resampling = ...,
        fill_value: float | None = ...,
        out_shape: tuple[int, ...] | None = ...,
    ) -> npt.NDArray[Any]: ...
    def __enter__(self) -> DatasetReader: ...
    def __exit__(self, *args: Any) -> None: ...
    def close(self) -> None: ...

class DatasetWriter(DatasetReader):
    """Read-write dataset."""
    def __enter__(self) -> DatasetWriter: ...
    def write(
        self, arr: npt.NDArray[Any], indexes: int | list[int] | None = ..., window: Any = ...
    ) -> None: ...
    def build_overviews(self, factors: list[int], resampling: Resampling = ...) -> None: ...
    def update_tags(self, ns: str | None = ..., **kwargs: Any) -> None: ...
    def set_band_description(self, bidx: int, value: str) -> None: ...
    def set_band_unit(self, bidx: int, value: str) -> None: ...

class MemoryFile:
    """In-memory raster file."""
    def __init__(self, data: bytes | None = ...) -> None: ...
    @overload
    def open(self, mode: Literal["r"] = ..., **kwargs: Any) -> DatasetReader: ...
    @overload
    def open(self, mode: Literal["w", "w+", "r+"], **kwargs: Any) -> DatasetWriter: ...
    def __enter__(self) -> MemoryFile: ...
    def __exit__(self, *args: Any) -> None: ...
    def close(self) -> None: ...

@overload
def open(fp: str | Any, mode: Literal["r"] = ..., **kwargs: Any) -> DatasetReader: ...
@overload
def open(fp: str | Any, mode: Literal["w", "w+", "r+"], **kwargs: Any) -> DatasetWriter: ...

class Env:
    """Rasterio environment."""
    def __init__(self, **options: Any) -> None: ...
    def __enter__(self) -> Env: ...
    def __exit__(self, *args: Any) -> None: ...

__all__ = [
    "open",
    "Env",
    "DatasetReader",
    "DatasetWriter",
    "MemoryFile",
    "Profile",
    "Affine",
    "CRS",
    "Resampling",
]
