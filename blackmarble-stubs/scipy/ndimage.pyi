"""Type stubs for scipy.ndimage."""

from collections.abc import Sequence
from typing import Any

import numpy.typing as npt

def gaussian_filter(
    input: npt.NDArray[Any],
    sigma: float | Sequence[float],
    order: int = 0,
    output: npt.NDArray[Any] | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    truncate: float = 4.0,
    *,
    radius: int | Sequence[int] | None = None,
    axes: tuple[int, ...] | None = None,
) -> npt.NDArray[Any]: ...
def binary_dilation(
    input: npt.NDArray[Any],
    structure: npt.NDArray[Any] | None = None,
    iterations: int = 1,
    mask: npt.NDArray[Any] | None = None,
    output: npt.NDArray[Any] | None = None,
    border_value: int = 0,
    origin: int | Sequence[int] = 0,
    brute_force: bool = False,
) -> npt.NDArray[Any]: ...
def label(
    input: npt.NDArray[Any],
    structure: npt.NDArray[Any] | None = None,
    output: npt.NDArray[Any] | None = None,
) -> tuple[npt.NDArray[Any], int]: ...
def generate_binary_structure(rank: int, connectivity: int) -> npt.NDArray[Any]: ...

__all__ = ["gaussian_filter", "binary_dilation", "label", "generate_binary_structure"]
