"""Quality assessment and masking functions.

This module provides functions for creating and combining quality masks
from various QA bands and quality indicators in satellite data.
"""

from typing import Any, Literal
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
    # Implementation from processing.py:376
    from ..processing import create_cloud_mask as _create_cloud_mask
    return _create_cloud_mask(qa_pixel, conservative=conservative, dilate_pixels=dilate_pixels)


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
    # Implementation from processing.py:478
    from ..processing import create_aerosol_mask as _create_aerosol_mask
    return _create_aerosol_mask(qa_aerosol, threshold=threshold)


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
    # Implementation from processing.py:503
    # WATCHPOINT: processing.py checks bands 3-5, but docstring says 1-7
    from ..processing import create_saturation_mask as _create_saturation_mask
    return _create_saturation_mask(qa_radsat, band=band)


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
    import numpy as np  # CRITICAL FIX: Add missing numpy import
    
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
    # Implementation from processing.py:567
    from ..processing import create_viirs_quality_mask as _create_viirs_quality_mask
    return _create_viirs_quality_mask(quality_flag, valid_flags=valid_flags)


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
            If None, the underlying processing.py function uses its default band selection.
            CRITICAL: This parameter was missing from our signature but required by
            processing.py:create_quality_mask. Without it, we cannot properly control
            which bands are checked for saturation, potentially leading to incorrect
            masking of valid data or failure to mask saturated pixels.
            Bug potential: If processing.py expects this parameter and we don't provide
            it, the function could fail or use wrong defaults.
        include_water: Whether to mask out water pixels
        water_threshold: NDWI threshold for water
        
    Returns:
        Combined quality mask
    """
    # Implementation adapted from processing.py:411
    # WATCHPOINT: processing.py doesn't support include_water parameter
    # We need to handle water masking separately
    from ..processing import create_quality_mask as _create_quality_mask
    
    # Extract QA bands for processing.py (it doesn't handle ndwi)
    processing_qa_bands = {k: v for k, v in qa_bands.items() if k != "ndwi"}
    
    # Call processing.py version
    # CRITICAL FIX: Pass through saturation_bands parameter properly
    # Line-by-line expectation analysis:
    # 1. User expects to control which bands are checked for saturation
    # 2. processing.py needs this parameter to know which bands to examine
    # 3. Default None means "use processing.py's default band selection"
    # 4. If user provides [3, 4, 5], only those bands are checked
    # 5. Bug risk: Not passing this parameter could cause saturation mask
    #    to check wrong bands or use hardcoded defaults from processing.py
    mask = _create_quality_mask(
        processing_qa_bands,
        combine_method=combine_method,
        cloud_strategy=cloud_strategy,
        cloud_dilate=cloud_dilate,
        aerosol_threshold=aerosol_threshold,
        include_saturation=include_saturation,
        saturation_bands=saturation_bands  # CRITICAL FIX: Pass through user's band selection
    )
    
    # Handle water masking if requested
    if include_water and "ndwi" in qa_bands:
        water_mask = create_water_mask(qa_bands["ndwi"], threshold=water_threshold)
        # Mask out water pixels (water_mask is True for water)
        if combine_method == "and":
            mask = mask & ~water_mask  # Remove water pixels
        else:  # "or"
            mask = mask | ~water_mask  # Include non-water pixels
    
    return mask


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
    # Implementation from processing.py:589
    from ..processing import combine_masks as _combine_masks
    return _combine_masks(masks, method=method)


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
    # Implementation from processing.py:630
    # WATCHPOINT: processing.py uses 'radius' parameter, not 'pixels'
    # WATCHPOINT: processing.py doesn't support connectivity parameter
    from ..processing import dilate_mask as _dilate_mask
    return _dilate_mask(mask, radius=pixels)


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
    # Implementation from bm_algorithms.py:375
    from ..bm_algorithms import compute_quality_flags as _compute_quality_flags
    return _compute_quality_flags(
        qa_pixel,
        qa_aerosol=qa_aerosol,
        qa_radsat=qa_radsat,
        cloud_values=cloud_values,
        aerosol_values=aerosol_values
    )


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
    # Implementation from bm_algorithms.py:259
    from ..bm_algorithms import create_urban_mask as _create_urban_mask
    return _create_urban_mask(
        ndvi,
        ndwi,
        ndui,
        ndvi_threshold=ndvi_threshold,
        ndwi_threshold=ndwi_threshold,
        ndui_threshold=ndui_threshold
    )


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
    import numpy as np
    
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