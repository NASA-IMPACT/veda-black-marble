"""Type stubs for rasterio."""

from typing import Any, Literal, overload, ContextManager
import numpy as np
import numpy.typing as npt
from .transform import Affine
from .warp import Resampling
from .crs import CRS
from . import crs, enums, features, io, mask, transform, warp, windows

class Profile(dict[str, Any]):
    """Raster dataset profile."""
    ...

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
    
    def read(
        self, 
        indexes: int | list[int] | None = None,
        window: Any = None,
        masked: bool = False,
        boundless: bool = False,
        resampling: Resampling = ...,
        fill_value: float | None = None,
        out_shape: tuple[int, ...] | None = None
    ) -> npt.NDArray[Any]: ...
    
    def __enter__(self) -> DatasetReader: ...
    def __exit__(self, *args: Any) -> None: ...
    def close(self) -> None: ...

class DatasetWriter(DatasetReader):
    """Read-write dataset."""
    def write(
        self,
        arr: npt.NDArray[Any],
        indexes: int | list[int] | None = None,
        window: Any = None
    ) -> None: ...
    
    def build_overviews(
        self,
        factors: list[int],
        resampling: Resampling = ...
    ) -> None: ...
    
    def update_tags(self, ns: str | None = None, **kwargs: Any) -> None: ...

class MemoryFile:
    """In-memory raster file."""
    def __init__(self, data: bytes | None = None) -> None: ...
    def open(self, **kwargs: Any) -> DatasetReader | DatasetWriter: ...
    def __enter__(self) -> MemoryFile: ...
    def __exit__(self, *args: Any) -> None: ...
    def close(self) -> None: ...

@overload
def open(
    fp: str | Any,
    mode: Literal["r"] = "r",
    **kwargs: Any
) -> DatasetReader: ...

@overload
def open(
    fp: str | Any,
    mode: Literal["w", "w+", "r+"],
    **kwargs: Any
) -> DatasetWriter: ...

def open(
    fp: str | Any,
    mode: str = "r",
    driver: str | None = None,
    width: int | None = None,
    height: int | None = None,
    count: int | None = None,
    crs: CRS | str | dict[str, Any] | None = None,
    transform: Affine | None = None,
    dtype: str | np.dtype[Any] | None = None,
    nodata: float | None = None,
    **kwargs: Any
) -> DatasetReader | DatasetWriter: ...

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