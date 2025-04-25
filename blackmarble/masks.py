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
    """Return mask (float) of pixels not matching any of values_to_check."""
    return (~np.isin(array, values_to_check)).astype(float)

def mark_cloud2(p_px):
    """Mask cloud pixels using a small QA set."""
    cloud_px_values = [55052, 54852, 54596, 22280]
    return check_values(cloud_px_values, p_px)

def mark_cloud(p_px):
    """Mask cloud pixels (keep only clear land/water codes)."""
    # keep only values 21824 and 21952
    valid = list(range(1, 21823)) + list(range(21825, 21951)) + list(range(21953, 65535))
    return check_values(valid, p_px)

def mark_aerosol2(a_px):
    """Mask aerosol pixels using extended QA set."""
    aerosol_px_values = [192, 194, 196, 224, 228]
    return check_values(aerosol_px_values, a_px)

def mark_aerosol(a_px):
    """Mask aerosol pixels (QA)."""
    aerosol_px_values = [192, 194, 196, 228]
    return check_values(aerosol_px_values, a_px)

def mark_radsat(r_px):
    """Mask radiance saturation for bands 3, 4, 5 from QA flags."""
    b3 = check_values([4], r_px & 4)
    b4 = check_values([8], r_px & 8)
    b5 = check_values([16], r_px & 16)
    return b3, b4, b5