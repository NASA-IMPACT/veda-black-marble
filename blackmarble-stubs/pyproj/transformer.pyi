"""Type stubs for pyproj.transformer."""

from collections.abc import Sequence
from typing import Any, overload

import numpy.typing as npt

from .crs import CRS

class Transformer:
    """Coordinate transformer between two CRS."""

    @classmethod
    def from_crs(
        cls,
        crs_from: CRS | str | dict[str, Any] | int,
        crs_to: CRS | str | dict[str, Any] | int,
        always_xy: bool = False,
        area_of_interest: tuple[float, float, float, float] | None = None,
        authority: str | None = None,
        accuracy: float | None = None,
        allow_ballpark: bool = True,
    ) -> Transformer: ...
    @overload
    def transform(
        self,
        x: float,
        y: float,
        z: float | None = None,
        t: float | None = None,
        radians: bool = False,
        errcheck: bool = False,
        direction: str = "FORWARD",
    ) -> tuple[float, float]: ...
    @overload
    def transform(
        self,
        x: float,
        y: float,
        z: float,
        t: float | None = None,
        radians: bool = False,
        errcheck: bool = False,
        direction: str = "FORWARD",
    ) -> tuple[float, float, float]: ...
    @overload
    def transform(
        self,
        x: Sequence[float] | npt.NDArray[Any],
        y: Sequence[float] | npt.NDArray[Any],
        z: Sequence[float] | npt.NDArray[Any] | None = None,
        t: Sequence[float] | npt.NDArray[Any] | None = None,
        radians: bool = False,
        errcheck: bool = False,
        direction: str = "FORWARD",
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]: ...
    @overload
    def transform(
        self,
        x: Sequence[float] | npt.NDArray[Any],
        y: Sequence[float] | npt.NDArray[Any],
        z: Sequence[float] | npt.NDArray[Any],
        t: Sequence[float] | npt.NDArray[Any] | None = None,
        radians: bool = False,
        errcheck: bool = False,
        direction: str = "FORWARD",
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]: ...
    def transform_bounds(
        self,
        left: float,
        bottom: float,
        right: float,
        top: float,
        densify_pts: int = 21,
        radians: bool = False,
        errcheck: bool = False,
        direction: str = "FORWARD",
    ) -> tuple[float, float, float, float]: ...
    @property
    def source_crs(self) -> CRS | None: ...
    @property
    def target_crs(self) -> CRS | None: ...
    @property
    def is_network_enabled(self) -> bool: ...

__all__ = ["Transformer"]
