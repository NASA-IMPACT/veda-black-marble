"""Feature enhancement for urban and infrastructure elements.

This module provides functions for enhancing specific features like roads,
urban areas, and other infrastructure in the processed imagery.
"""

from typing import Any, Literal
import numpy as np
import numpy.typing as npt
from scipy.ndimage import distance_transform_edt, binary_dilation, label

# Type aliases
ArrayLike = npt.NDArray[Any]


def enhance_roads(
    data: ArrayLike,
    road_mask: ArrayLike,
    major_factor: float = 1.3,
    minor_factor: float = 1.1,
    custom_factors: dict[int, float] | None = None
) -> ArrayLike:
    """Enhance values along road networks.
    
    Args:
        data: Base data to enhance (e.g., nighttime lights)
        road_mask: Road classification array (0=no road, 1-6=road types)
        major_factor: Enhancement factor for major roads (motorways, primary)
        minor_factor: Enhancement factor for minor roads
        custom_factors: Optional dict mapping road class to enhancement factor
        
    Returns:
        Enhanced data array with boosted road values
    """
    # Pure functional implementation
    enhanced = data.copy()
    
    # Default road type factors if not custom provided
    if custom_factors is None:
        # Road types: 1=motorway, 2=primary, 3=secondary, 4=tertiary, 5=residential, 6=other
        factors = {
            1: major_factor,  # motorway
            2: major_factor,  # primary
            3: minor_factor,  # secondary
            4: minor_factor,  # tertiary
            5: minor_factor,  # residential
            6: minor_factor,  # other
        }
    else:
        factors = custom_factors
    
    # Apply enhancement for each road type
    for road_type, factor in factors.items():
        mask = road_mask == road_type
        enhanced[mask] = data[mask] * factor
    
    return enhanced


def enhance_urban_features(
    ntl_data: ArrayLike,
    urban_mask: ArrayLike,
    road_mask: ArrayLike | None = None,
    enhancement_factor: float = 1.2,
    road_factor: float = 1.1,
    preserve_range: bool = True
) -> ArrayLike:
    """Enhance urban features in nighttime lights.
    
    Args:
        ntl_data: Nighttime lights data
        urban_mask: Urban area mask  
        road_mask: Optional road mask for additional enhancement
        enhancement_factor: Factor for urban area enhancement
        road_factor: Additional factor for roads in urban areas
        preserve_range: Whether to preserve original data range
        
    Returns:
        Enhanced nighttime lights emphasizing urban features
    """
    # Pure functional implementation
    enhanced = ntl_data.copy()
    
    # Apply urban enhancement
    urban_areas = urban_mask > 0.5 if urban_mask.dtype == bool else urban_mask > 0
    enhanced[urban_areas] = ntl_data[urban_areas] * enhancement_factor
    
    # Apply additional road enhancement if provided
    if road_mask is not None:
        road_areas = road_mask > 0
        # Combine urban and road areas
        urban_roads = urban_areas & road_areas
        enhanced[urban_roads] = ntl_data[urban_roads] * enhancement_factor * road_factor
    
    if preserve_range:
        # Scale back to original range
        orig_min, orig_max = np.nanmin(ntl_data), np.nanmax(ntl_data)
        if orig_max > orig_min:
            enh_min, enh_max = np.nanmin(enhanced), np.nanmax(enhanced)
            if enh_max > enh_min:
                enhanced = (enhanced - enh_min) / (enh_max - enh_min)
                enhanced = enhanced * (orig_max - orig_min) + orig_min
    
    return enhanced


def detect_urban_corridors(
    ntl_data: ArrayLike,
    road_mask: ArrayLike,
    threshold: float = 0.5,
    min_length: int = 10
) -> ArrayLike:
    """Detect and enhance urban corridors along transportation routes.
    
    Args:
        ntl_data: Nighttime lights data
        road_mask: Road network mask
        threshold: NTL threshold for corridor detection
        min_length: Minimum corridor length in pixels
        
    Returns:
        Urban corridor mask
    """
    # Identify lit transportation corridors
    # Combine roads with nighttime lights above threshold
    lit_areas = ntl_data > threshold
    road_areas = road_mask > 0
    
    # Find intersection of lit areas and roads
    corridors = lit_areas & road_areas
    
    # Label connected components
    labeled_array, num_features = label(corridors)
    num_features = int(num_features)  # Ensure it's an int
    
    # Filter by minimum length
    corridor_mask = np.zeros_like(corridors, dtype=bool)
    for i in range(1, num_features + 1):
        component = labeled_array == i
        # Approximate length as sqrt of area
        component_area = np.sum(component)
        if np.sqrt(float(component_area)) >= min_length:
            corridor_mask |= component
    
    return corridor_mask.astype(np.float32)


def enhance_urban_boundaries(
    urban_mask: ArrayLike,
    ntl_data: ArrayLike,
    method: Literal["gradient", "morphological"] = "gradient",
    width: int = 3
) -> ArrayLike:
    """Enhance urban area boundaries for better delineation.
    
    Args:
        urban_mask: Binary urban mask
        ntl_data: Nighttime lights data
        method: Enhancement method
        width: Width of boundary enhancement in pixels
        
    Returns:
        Enhanced urban mask with clearer boundaries
    """
    # Convert to binary if needed
    binary_mask = urban_mask > 0.5
    
    if method == "gradient":
        # Use distance transform to create gradient
        dist_inside = distance_transform_edt(binary_mask)
        dist_outside = distance_transform_edt(~binary_mask)
        
        # Create boundary region
        boundary_region = (dist_inside <= width) | (dist_outside <= width)
        
        # Enhance using NTL data in boundary region
        enhanced = urban_mask.copy()
        enhanced[boundary_region] = ntl_data[boundary_region]
        
    else:  # morphological
        # Dilate then subtract original to get boundary
        dilated = binary_dilation(binary_mask, iterations=width)
        boundary = dilated & ~binary_mask
        
        # Enhance boundary pixels based on NTL
        enhanced = urban_mask.copy()
        enhanced[boundary] = np.maximum(enhanced[boundary], ntl_data[boundary] * 0.5)
    
    return enhanced


def suppress_blooming(
    ntl_data: ArrayLike,
    threshold: float = 0.8,
    kernel_size: int = 5
) -> ArrayLike:
    """Suppress blooming effects in bright urban cores.
    
    Args:
        ntl_data: Nighttime lights data
        threshold: Threshold for identifying blooming
        kernel_size: Size of suppression kernel
        
    Returns:
        NTL data with reduced blooming artifacts
    """
    # Identify potential blooming areas (very bright pixels)
    bloom_mask = ntl_data > threshold
    
    # Create distance map from bloom centers
    dist = distance_transform_edt(~bloom_mask)
    
    # Apply suppression based on distance
    suppression_factor = np.ones_like(ntl_data)
    affected_region = dist < kernel_size
    
    # Gradual suppression based on distance
    suppression_factor[affected_region] = 0.5 + 0.5 * (dist[affected_region] / kernel_size)
    
    # Apply suppression to bright areas only
    result = ntl_data.copy()
    result[bloom_mask] = ntl_data[bloom_mask] * suppression_factor[bloom_mask]
    
    return result


def enhance_infrastructure(
    data: ArrayLike,
    infrastructure_types: dict[str, ArrayLike],
    enhancement_factors: dict[str, float]
) -> ArrayLike:
    """Enhance multiple infrastructure types.
    
    Args:
        data: Base data array
        infrastructure_types: Dict of infrastructure masks
        enhancement_factors: Enhancement factor for each type
        
    Returns:
        Data with enhanced infrastructure features
    """
    # Apply enhancement for each infrastructure type
    enhanced = data.copy()
    
    for infra_type, mask in infrastructure_types.items():
        if infra_type in enhancement_factors:
            factor = enhancement_factors[infra_type]
            # Apply enhancement where infrastructure exists
            enhanced = np.where(mask > 0, enhanced * factor, enhanced)
    
    return enhanced


def create_urban_gradient(
    urban_mask: ArrayLike,
    distance_decay: float = 0.1,
    max_distance: int = 50
) -> ArrayLike:
    """Create gradient from urban centers.
    
    Args:
        urban_mask: Binary urban mask
        distance_decay: Decay rate with distance
        max_distance: Maximum distance for gradient
        
    Returns:
        Urban gradient array
    """
    # Convert to binary
    binary_mask = urban_mask > 0.5
    
    # Calculate distance from urban areas
    distance = distance_transform_edt(~binary_mask)
    
    # Clip to maximum distance
    distance = np.minimum(distance, float(max_distance))
    
    # Apply exponential decay
    gradient = np.exp(-distance_decay * distance)
    
    # Ensure urban areas are 1.0
    gradient[binary_mask] = 1.0
    
    return gradient


def enhance_small_settlements(
    ntl_data: ArrayLike,
    min_size: int = 5,
    enhancement_factor: float = 1.5
) -> ArrayLike:
    """Enhance small settlements that might be underrepresented.
    
    Args:
        ntl_data: Nighttime lights data
        min_size: Minimum settlement size in pixels
        enhancement_factor: Enhancement multiplier
        
    Returns:
        Enhanced data with boosted small settlements
    """
    # Threshold to find lit areas
    threshold = np.percentile(ntl_data[ntl_data > 0], 25)
    lit_areas = ntl_data > threshold
    
    # Label connected components  
    labeled_array, num_features = label(lit_areas)
    num_features = int(num_features)  # Ensure it's an int
    
    # Enhance small settlements
    enhanced = ntl_data.copy()
    
    for i in range(1, num_features + 1):
        component = labeled_array == i
        size = np.sum(component)
        
        # Enhance if it's a small settlement
        component_size = int(size)
        if min_size <= component_size <= min_size * 10:
            enhanced[component] *= enhancement_factor
    
    return enhanced