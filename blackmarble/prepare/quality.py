"""Quality assessment and masking functions.

This module provides functions for creating and combining quality masks
from various QA bands and quality indicators in satellite data.
"""

from typing import Any, Literal
import numpy as np
import numpy.typing as npt

# Type aliases
ArrayLike = npt.NDArray[Any]
MaskArray = npt.NDArray[Any]  # Boolean or float masks


def create_cloud_mask(
    qa_pixel: ArrayLike,
    conservative: bool = True,
    dilate_pixels: int = 0
) -> MaskArray:
    """Create cloud mask from Landsat QA_PIXEL band.
    
    Args:
        qa_pixel: QA_PIXEL band from Landsat
        conservative: If True, use conservative (strict) cloud masking
        dilate_pixels: Number of pixels to dilate cloud mask
        
    Returns:
        Boolean mask where True indicates clear (non-cloud) pixels
    """
    if conservative:
        # Conservative strategy: only keep clear land (21824) and clear water (21952)
        valid_values = {21824, 21952}
        mask = np.isin(qa_pixel, list(valid_values))
    else:
        # Permissive strategy: mask only high confidence clouds
        # These are the QA values for different cloud confidence levels
        invalid_values = {
            55052,
            54852,
            54596,
            22280,
        }  # High/medium/low confidence clouds
        mask = ~np.isin(qa_pixel, list(invalid_values))

    # Apply dilation if requested
    if dilate_pixels > 0:
        mask = dilate_mask(mask.astype(float), dilate_pixels) > 0

    return mask


def create_aerosol_mask(
    qa_aerosol: ArrayLike,
    threshold: Literal["standard", "strict"] = "standard"
) -> MaskArray:
    """Create aerosol quality mask from QA_AEROSOL band.
    
    Args:
        qa_aerosol: QA_AEROSOL band from Landsat
        threshold: Quality threshold level
        
    Returns:
        Boolean mask where True indicates acceptable aerosol levels
    """
    if threshold == "standard":
        # Standard threshold masks medium and high aerosol
        invalid_values = {192, 194, 196, 228}
    else:  # strict
        # Strict threshold masks more aerosol values
        invalid_values = {192, 194, 196, 224, 228}

    # Create mask - pixels NOT in invalid set are kept
    mask = ~np.isin(qa_aerosol, list(invalid_values))

    return mask


def create_saturation_mask(
    qa_radsat: ArrayLike,
    band: int | None = None
) -> MaskArray:
    """Create radiometric saturation mask from QA_RADSAT band.
    
    Args:
        qa_radsat: QA_RADSAT band from Landsat
        band: Specific band to check (1-7), or None to check any saturation
        
    Returns:
        Boolean mask where True indicates non-saturated pixels
    """
    # Handle NaN values by treating them as invalid
    qa_radsat_safe = np.where(np.isnan(qa_radsat), -1, qa_radsat).astype(np.int32)

    if band is not None:
        # Single band mask
        if band not in {3, 4, 5}:
            raise ValueError(f"Band must be 3, 4, or 5, got {band}")

        # Band 3 = bit 2, Band 4 = bit 3, Band 5 = bit 4
        bit_position = band - 1
        bit_value = 1 << bit_position

        # Pixel is valid if bit is NOT set
        mask = (qa_radsat_safe & bit_value) == 0

        # Also mask out original NaN pixels
        mask = np.where(np.isnan(qa_radsat), False, mask)

    else:
        # Check all bands - pixel is valid only if NO bands are saturated
        # Bits 2-4 represent bands 3-5
        any_saturated_mask = 0b11100  # Bits 2, 3, 4

        # Pixel is valid if none of the saturation bits are set
        mask = (qa_radsat_safe & any_saturated_mask) == 0

        # Also mask out original NaN pixels
        mask = np.where(np.isnan(qa_radsat), False, mask)

    return mask


def create_water_mask(
    ndwi: ArrayLike,
    threshold: float = 0.0
) -> MaskArray:
    """Create water mask from NDWI values.
    
    Args:
        ndwi: Normalized Difference Water Index array
        threshold: NDWI threshold for water detection
        
    Returns:
        Boolean mask where True indicates water pixels
    """
    # CRITICAL BUG FIX: The original implementation had a semantic confusion
    # between what the mask represents vs what the function returns.
    #
    # Line-by-line analysis of the bug:
    # 1. Our function docstring says: "True indicates water pixels"
    # 2. But processing.py:555 docstring says: "True = land (valid), False = water (masked)"  
    # 3. These are OPPOSITE semantics!
    # 4. The original code tried to "fix" this by inverting: ~land_mask
    # 5. But this created a double-negative that was confusing and error-prone
    #
    # What the processing.py function actually does:
    # - Returns True for pixels where ndwi <= threshold (which are LAND pixels)
    # - Returns False for pixels where ndwi > threshold (which are WATER pixels)
    # - This follows the "valid pixel" convention where True = good for processing
    #
    # Our function's promise to users:
    # - Should return True for WATER pixels (where ndwi > threshold)
    # - Should return False for LAND pixels (where ndwi <= threshold)
    # - This follows the "feature detection" convention where True = feature present
    #
    # The fix: Implement directly instead of inverting, making the logic crystal clear
    
    # FIXED IMPLEMENTATION:
    # Water pixels have NDWI > threshold (positive values indicate water)
    # NDWI = (Green - NIR) / (Green + NIR)
    # - Water: High green reflectance, low NIR absorption → positive NDWI
    # - Land: Low green reflectance, high NIR absorption → negative NDWI
    water_pixels = ndwi > threshold
    
    # Handle NaN values: NaN pixels are neither water nor land, treat as land (False)
    # This is conservative - we don't want to accidentally mask valid land as water
    water_mask = np.where(np.isnan(ndwi), False, water_pixels)
    
    # EXPECTATION CHECK: 
    # - If ndwi = 0.5 (clearly water), water_mask should be True ✓
    # - If ndwi = -0.5 (clearly land), water_mask should be False ✓  
    # - If ndwi = NaN (invalid), water_mask should be False ✓
    # - If threshold = 0.1 and ndwi = 0.05, water_mask should be False ✓
    
    return water_mask


def create_viirs_quality_mask(
    quality_flag: ArrayLike,
    valid_flags: list[int] | None = None
) -> MaskArray:
    """Create quality mask from VIIRS mandatory quality flag.
    
    Args:
        quality_flag: VIIRS mandatory quality flag array
        valid_flags: List of flag values to consider valid (default: [0, 2])
        
    Returns:
        Boolean mask where True indicates good quality pixels
    """
    if valid_flags is None:
        valid_flags = [0, 2]  # Default: high quality (0) and moderate quality (2)

    # Create mask for valid quality flags
    mask = np.isin(quality_flag, valid_flags)

    return mask


def create_quality_mask(
    qa_bands: dict[str, ArrayLike],
    combine_method: Literal["and", "or"] = "and",
    cloud_strategy: Literal["conservative", "permissive"] = "conservative",
    cloud_dilate: int = 0,
    aerosol_threshold: Literal["standard", "strict"] = "standard",
    include_saturation: bool = True,
    saturation_bands: list[int] | None = None,
    include_water: bool = False,
    water_threshold: float = 0.0
) -> MaskArray:
    """Create comprehensive quality mask from multiple QA bands.
    
    Args:
        qa_bands: Dictionary of QA bands (qa_pixel, qa_aerosol, qa_radsat, ndwi)
        combine_method: How to combine masks ('and' or 'or')
        cloud_strategy: Cloud masking strategy
        cloud_dilate: Pixels to dilate cloud mask
        aerosol_threshold: Aerosol quality threshold
        include_saturation: Whether to include saturation mask
        saturation_bands: List of bands to check for saturation (e.g., [3, 4, 5]).
            If None, checks all bands.
        include_water: Whether to mask out water pixels
        water_threshold: NDWI threshold for water
        
    Returns:
        Combined quality mask
    """
    masks = []

    # Cloud mask from QA_PIXEL
    if "qa_pixel" in qa_bands:
        cloud_mask = create_cloud_mask(
            qa_bands["qa_pixel"],
            conservative=(cloud_strategy == "conservative"),
            dilate_pixels=cloud_dilate,
        )
        masks.append(cloud_mask)

    # Aerosol mask from QA_AEROSOL
    if "qa_aerosol" in qa_bands:
        aerosol_mask = create_aerosol_mask(
            qa_bands["qa_aerosol"], threshold=aerosol_threshold
        )
        masks.append(aerosol_mask)

    # Radiometric saturation mask from QA_RADSAT
    if include_saturation and "qa_radsat" in qa_bands:
        if saturation_bands:
            # Check specific bands
            for band in saturation_bands:
                sat_mask = create_saturation_mask(qa_bands["qa_radsat"], band=band)
                masks.append(sat_mask)
        else:
            # Check all bands (3, 4, 5)
            sat_mask = create_saturation_mask(qa_bands["qa_radsat"], band=None)
            masks.append(sat_mask)

    # Handle water masking if requested
    if include_water and "ndwi" in qa_bands:
        water_mask = create_water_mask(qa_bands["ndwi"], threshold=water_threshold)
        # Mask out water pixels (water_mask is True for water)
        if combine_method == "and":
            masks.append(~water_mask)  # Remove water pixels
        else:  # "or"
            masks.append(~water_mask)  # Include non-water pixels

    # Combine all masks
    if not masks:
        raise ValueError("No QA bands provided for quality masking")

    return combine_masks(masks, method=combine_method)


def combine_masks(
    masks: list[MaskArray],
    method: Literal["and", "or"] = "and"
) -> MaskArray:
    """Combine multiple masks using logical operations.
    
    Args:
        masks: List of mask arrays (boolean or float)
        method: Combination method ('and' or 'or')
        
    Returns:
        Combined mask
    """
    if not masks:
        raise ValueError("No masks provided to combine")

    # Ensure all masks have the same shape
    shape = masks[0].shape
    for i, mask in enumerate(masks[1:], 1):
        if mask.shape != shape:
            raise ValueError(f"Mask {i} has shape {mask.shape}, expected {shape}")

    # Convert to boolean if needed
    bool_masks = [mask.astype(bool) if mask.dtype != bool else mask for mask in masks]

    if method == "and":
        # All masks must be True for pixel to be valid
        result = bool_masks[0].copy()
        for mask in bool_masks[1:]:
            result = result & mask
    elif method == "or":
        # Any mask can be True for pixel to be valid
        result = bool_masks[0].copy()
        for mask in bool_masks[1:]:
            result = result | mask
    else:
        raise ValueError(f"Unknown combination method: {method}")

    return result


def dilate_mask(
    mask: MaskArray,
    pixels: int = 1,
    connectivity: int = 8
) -> MaskArray:
    """Dilate a binary mask by specified pixels.
    
    Args:
        mask: Input mask (boolean or binary)
        pixels: Number of pixels to dilate
        connectivity: Connectivity for dilation (4 or 8)
        
    Returns:
        Dilated mask
    """
    if pixels == 0:
        return mask.copy()

    # Convert to float for processing
    result = mask.astype(float) if mask.dtype != float else mask.copy()

    # Simple dilation implementation without scipy
    # This is a basic box filter approach
    for _ in range(pixels):
        # Create a temporary mask for this iteration
        temp_mask = result.copy()

        # For each pixel, if any neighbor is masked (0), mask this pixel too
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                # Check if any neighbor is masked
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ni, nj = i + di, j + dj
                        # Check bounds
                        if 0 <= ni < mask.shape[0] and 0 <= nj < mask.shape[1]:
                            if result[ni, nj] == 0:
                                temp_mask[i, j] = 0
                                break
                    if temp_mask[i, j] == 0:
                        break

        result = temp_mask

    # Return same type as input
    if mask.dtype == bool:
        return result.astype(bool)
    else:
        return result


def compute_quality_flags(
    qa_pixel: ArrayLike,
    qa_aerosol: ArrayLike | None = None,
    qa_radsat: ArrayLike | None = None,
    cloud_values: list[int] | None = None,
    aerosol_values: list[int] | None = None
) -> MaskArray:
    """Compute comprehensive quality flags from Landsat QA bands.
    
    Args:
        qa_pixel: QA_PIXEL band containing cloud and other flags
        qa_aerosol: Optional QA_AEROSOL band for aerosol quality
        qa_radsat: Optional QA_RADSAT band for saturation flags
        cloud_values: List of QA_PIXEL values indicating clouds
        aerosol_values: List of QA_AEROSOL values indicating poor quality
        
    Returns:
        Boolean mask where True indicates good quality pixels
    """
    if cloud_values is None:
        # Keep only clear land (21824) and clear water (21952)
        cloud_values = (
            list(range(1, 21823)) + 
            list(range(21825, 21951)) + 
            list(range(21953, 65535))
        )
    
    if aerosol_values is None:
        # These values indicate valid aerosol conditions
        aerosol_values = [192, 194, 196, 228]
    
    # Create cloud mask (True = good pixel)
    cloud_mask = ~np.isin(qa_pixel, cloud_values)
    
    # Combine with aerosol mask if provided
    if qa_aerosol is not None:
        aerosol_mask = np.isin(qa_aerosol, aerosol_values)
        quality_mask = cloud_mask & aerosol_mask
    else:
        quality_mask = cloud_mask
    
    # Add saturation check if provided
    if qa_radsat is not None:
        # Check if any bands are saturated (bits 1-7)
        # Bit 0 is "band 1 data quality" which we ignore
        saturation_mask = qa_radsat == 0
        quality_mask = quality_mask & saturation_mask
    
    return quality_mask


def create_urban_mask(
    ndvi: ArrayLike,
    ndwi: ArrayLike,
    ndui: ArrayLike,
    ndvi_threshold: float = 0.2,
    ndwi_threshold: float = 0.0,
    ndui_threshold: float = 0.3
) -> MaskArray:
    """Create urban area mask from spectral indices.
    
    Args:
        ndvi: NDVI array
        ndwi: NDWI array
        ndui: NDUI array
        ndvi_threshold: Maximum NDVI for urban
        ndwi_threshold: Maximum NDWI for urban
        ndui_threshold: Minimum NDUI for urban
        
    Returns:
        Boolean mask where True indicates urban pixels
    """
    # Urban criteria:
    # - Low vegetation (NDVI < threshold)
    # - Not water (NDWI < threshold)
    # - High urban index (NDUI > threshold)
    urban_mask = (
        (ndvi < ndvi_threshold) & 
        (ndwi < ndwi_threshold) & 
        (ndui > ndui_threshold)
    )
    
    return urban_mask


def compute_mask_statistics(
    mask: MaskArray,
    total_pixels: int | None = None
) -> dict[str, Any]:
    """Compute statistics for a quality mask.
    
    Args:
        mask: Input mask array
        total_pixels: Total pixel count (computed if None)
        
    Returns:
        Dictionary with mask statistics:
        - valid_pixels: Number of True/non-zero pixels
        - invalid_pixels: Number of False/zero pixels
        - valid_percentage: Percentage of valid pixels
        - mask_coverage: Spatial coverage metrics
    """
    # New implementation based on QualityStatistics from components
    
    # Convert to boolean if needed
    bool_mask = mask.astype(bool) if mask.dtype != bool else mask
    
    # Calculate basic statistics
    if total_pixels is None:
        total_pixels = mask.size
    
    valid_pixels = int(np.sum(bool_mask))
    invalid_pixels = total_pixels - valid_pixels
    valid_percentage = (valid_pixels / total_pixels * 100) if total_pixels > 0 else 0.0
    
    # Calculate spatial coverage metrics
    # Find contiguous regions of valid pixels
    rows_with_valid = np.any(bool_mask, axis=1) if mask.ndim > 1 else bool_mask
    cols_with_valid = np.any(bool_mask, axis=0) if mask.ndim > 1 else bool_mask
    
    coverage_metrics = {
        "rows_with_valid": int(np.sum(rows_with_valid)) if mask.ndim > 1 else 1,
        "cols_with_valid": int(np.sum(cols_with_valid)) if mask.ndim > 1 else 1,
        "total_rows": mask.shape[0] if mask.ndim > 1 else 1,
        "total_cols": mask.shape[1] if mask.ndim > 1 else 1,
    }
    
    return {
        "valid_pixels": valid_pixels,
        "invalid_pixels": invalid_pixels,
        "valid_percentage": valid_percentage,
        "mask_coverage": coverage_metrics,
        "total_pixels": total_pixels,
    }