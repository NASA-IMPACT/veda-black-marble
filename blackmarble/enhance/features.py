"""Feature enhancement - ONLY WHAT'S USED."""

from typing import Any
import numpy as np
import numpy.typing as npt

ArrayLike = npt.NDArray[Any]


def enhance_roads(
    data: ArrayLike,
    road_mask: ArrayLike,
    major_factor: float = 1.3,
    minor_factor: float = 1.1,
    custom_factors: dict[int, float] | None = None
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


def enhance_urban_features(
    ntl_data: ArrayLike,
    urban_mask: ArrayLike,
    road_mask: ArrayLike | None = None,
    enhancement_factor: float = 1.2,
    road_factor: float = 1.1
) -> ArrayLike:
    """Enhance urban features in nighttime lights.
    
    Args:
        ntl_data: Input nighttime lights data
        urban_mask: Urban area mask
        road_mask: Optional road mask for additional enhancement
        enhancement_factor: Factor for enhancing urban areas
        road_factor: Factor for enhancing roads within urban areas
        
    Returns:
        Enhanced nighttime lights data
    """
    enhanced = ntl_data.copy()
    
    # Enhance urban areas
    urban_pixels = urban_mask > 0.5
    enhanced[urban_pixels] = np.clip(ntl_data[urban_pixels] * enhancement_factor, 0, 1)
    
    # Additional enhancement for roads within urban areas
    if road_mask is not None:
        urban_road_pixels = urban_pixels & (road_mask > 0)
        enhanced[urban_road_pixels] = np.clip(enhanced[urban_road_pixels] * road_factor, 0, 1)
    
    return enhanced