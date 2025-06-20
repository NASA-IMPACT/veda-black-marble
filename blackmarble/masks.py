"""
Quality‐masking utilities:
  - check_values: build mask where pixel values not in a set
  - mark_cloud / mark_cloud2: cloud QA masks
  - mark_aerosol / mark_aerosol2: aerosol QA masks
  - mark_radsat: radiance saturation QA mask for B3/B4/B5
"""

# Quality-masking utilities
import numpy as np


def check_values(values_to_check, array):
    """Return mask (float) where 1.0 = pixel should be kept, 0.0 = pixel should be masked.

    Args:
        values_to_check: Values that should be MASKED OUT (return 0.0)
        array: The array to check

    Returns:
        Mask where pixels NOT in values_to_check get 1.0 (keep), pixels IN values_to_check get 0.0 (mask)
    """
    return (~np.isin(array, values_to_check)).astype(float)


def mark_cloud2(p_px):
    """Mask cloud pixels using a small QA set."""
    cloud_px_values = [55052, 54852, 54596, 22280]
    return check_values(cloud_px_values, p_px)


def mark_cloud(p_px):
    """Mask cloud pixels (keep only clear land/water codes).

    Keeps pixels with values 21824 (clear land) and 21952 (clear water).
    All other values are masked out (clouds, shadows, etc.).
    """
    # Values to mask out (everything except 21824 and 21952)
    cloud_values = (
        list(range(1, 21824)) + list(range(21825, 21952)) + list(range(21953, 65535))
    )
    return check_values(cloud_values, p_px)


def mark_aerosol2(a_px):
    """Mask aerosol pixels using extended QA set."""
    aerosol_px_values = [192, 194, 196, 224, 228]
    return check_values(aerosol_px_values, a_px)


def mark_aerosol(a_px):
    """Mask aerosol pixels (QA)."""
    aerosol_px_values = [192, 194, 196, 228]
    return check_values(aerosol_px_values, a_px)


def mark_radsat(r_px):
    """Mask radiance saturation for bands 3, 4, 5 from QA flags.
    
    Returns masks where 1.0 = pixel is good (not saturated), 0.0 = pixel is saturated.
    If input contains NaN (from NoData), those pixels are marked as 0.0 (masked).
    """
    # Handle NaN values - convert to a value that won't match any bit flags
    # This ensures NaN pixels are masked out (return 0.0)
    r_px_safe = np.where(np.isnan(r_px), -1, r_px).astype(np.int32)
    
    # Check bit flags for saturation
    # Bit 2 (value 4) = Band 3 saturated
    # Bit 3 (value 8) = Band 4 saturated  
    # Bit 4 (value 16) = Band 5 saturated
    b3 = check_values([4], r_px_safe & 4)
    b4 = check_values([8], r_px_safe & 8)
    b5 = check_values([16], r_px_safe & 16)
    
    # Also mask out original NaN pixels
    b3 = np.where(np.isnan(r_px), 0.0, b3)
    b4 = np.where(np.isnan(r_px), 0.0, b4)
    b5 = np.where(np.isnan(r_px), 0.0, b5)
    
    return b3, b4, b5
