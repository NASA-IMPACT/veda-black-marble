"""Feature enhancement functions."""

from typing import Any

import numpy as np
import numpy.typing as npt

ArrayLike = npt.NDArray[Any]


def enhance_roads(
    data: ArrayLike,
    road_mask: ArrayLike,
    major_factor: float = 1.3,
    minor_factor: float = 1.1,
    custom_factors: dict[int, float] | None = None,
) -> ArrayLike:
    """Enhance roads in nighttime lights data.

    Args:
        data: Input nighttime lights data
        road_mask: Road mask with different values for road types
        major_factor: Enhancement factor for major roads
        minor_factor: Enhancement factor for minor roads
        custom_factors: Optional custom factors for specific road types

    Returns:
        Enhanced nighttime lights data
    """
    enhanced = data.copy()

    # Apply enhancement based on road type values in mask
    if custom_factors:
        # Use custom factors if provided
        for road_value, factor in custom_factors.items():
            mask = road_mask == road_value
            enhanced[mask] = np.clip(data[mask] * factor, 0, 1)
    else:
        # Default behavior: major roads (value 1-2) vs minor roads (value 3+)
        major_mask = (road_mask > 0) & (road_mask <= 2)
        minor_mask = road_mask > 2

        enhanced[major_mask] = np.clip(data[major_mask] * major_factor, 0, 1)
        enhanced[minor_mask] = np.clip(data[minor_mask] * minor_factor, 0, 1)

    return enhanced


