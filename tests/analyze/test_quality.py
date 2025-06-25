"""Test quality flag interpretation for bm.py parity.

This module tests only the critical quality flag functionality needed for bm.py parity.
"""

import pytest
import numpy as np

from blackmarble.analyze.quality import (
    assess_data_quality,
    create_composite_quality_mask,
)


def test_create_composite_quality_mask():
    """Test creation of composite quality mask - CRITICAL for bm.py parity.
    
    This is essential for proper VIIRS quality flag interpretation.
    """
    # Create test data
    shape = (10, 10)
    cloud_mask = np.zeros(shape, dtype=bool)
    cloud_mask[0:3, 0:3] = True  # Cloud in top-left (True = bad pixel)
    
    quality_mask = np.zeros(shape, dtype=bool) 
    quality_mask[5:8, 5:8] = True  # Bad quality in center
    
    # Note: For masks dict, True typically means VALID pixel (not masked)
    # So we need to invert our test masks
    masks = {
        'cloud': ~cloud_mask,  # Invert: False where clouds exist
        'quality': ~quality_mask  # Invert: False where bad quality
    }
    
    # Test basic masking without dilation (using 'and' combination)
    composite_mask, metadata = create_composite_quality_mask(
        masks,
        combination_method='and',
        dilation_radius=0
    )
    
    # composite_mask: True = valid pixel, False = masked
    # Verify cloud area is masked (False)
    assert not np.any(composite_mask[0:3, 0:3])
    
    # Verify quality flag area is masked (False)
    assert not np.any(composite_mask[5:8, 5:8])
    
    # Verify other areas are not masked (True)
    assert composite_mask[9, 9]
    
    # Test with only cloud mask
    cloud_only_masks = {'cloud': ~cloud_mask}
    cloud_only_mask, _ = create_composite_quality_mask(
        cloud_only_masks,
        dilation_radius=0
    )
    
    assert not np.any(cloud_only_mask[0:3, 0:3])  # Cloud area masked
    assert np.all(cloud_only_mask[5:8, 5:8])  # Quality area not masked


def test_assess_data_quality_basic():
    """Test basic data quality assessment - needed for pipeline validation."""
    # Create test data with known quality issues
    data = np.array([[1.0, 2.0, 100.0],  # 100.0 is outlier
                     [1.5, 2.5, 3.0],
                     [2.0, 3.0, 4.0]])
    
    # Test basic assessment
    result = assess_data_quality(
        data,
        valid_range=(0.0, 10.0),
        check_outliers=True,
        outlier_threshold=3.0
    )
    
    # Verify valid range checking
    assert result['valid_pixel_count'] == 8  # 100.0 is excluded
    assert result['min'] == 1.0
    assert result['max'] == 4.0  # Max of valid data only (100.0 excluded)
    
    # Verify quality score exists
    assert 0 <= result['quality_score'] <= 100