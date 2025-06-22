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
    # Currently in processing.py
    ...


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
    # Currently in processing.py
    ...


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
    # Currently in processing.py
    ...


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
    # Currently in processing.py
    ...


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
    # Currently in processing.py
    ...


def create_quality_mask(
    qa_bands: dict[str, ArrayLike],
    combine_method: Literal["and", "or"] = "and",
    cloud_strategy: Literal["conservative", "permissive"] = "conservative",
    cloud_dilate: int = 0,
    aerosol_threshold: Literal["standard", "strict"] = "standard",
    include_saturation: bool = True,
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
        include_water: Whether to mask out water pixels
        water_threshold: NDWI threshold for water
        
    Returns:
        Combined quality mask
    """
    # Currently in processing.py
    ...


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
    # Currently in processing.py
    ...


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
    # Currently in processing.py
    ...


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
    # Currently in bm_algorithms.py
    ...


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
    # Currently in bm_algorithms.py
    ...


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
    # Extract from component MaskStatistics functionality
    ...