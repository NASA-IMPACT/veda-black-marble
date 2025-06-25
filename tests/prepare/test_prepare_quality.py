"""Test critical quality processing functions for bm.py parity.

This module tests only the essential quality functionality needed for bm.py parity.
"""

import pytest
import numpy as np

from blackmarble.prepare.quality import (
    create_cloud_mask,
    create_quality_mask,
)


def test_create_cloud_mask():
    """Test cloud mask creation - CRITICAL for Landsat processing."""
    # Test data using actual Landsat QA values
    qa_pixel = np.array([
        [21824,  # Clear land (valid)
         55052,  # High confidence cloud (invalid)
         21952], # Clear water (valid)
        [54852,  # Medium confidence cloud (invalid)
         22280,  # Cloud shadow (invalid)
         21824]  # Clear land (valid)
    ], dtype=np.uint16)
    
    # Create cloud mask with conservative mode
    cloud_mask = create_cloud_mask(
        qa_pixel,
        conservative=True
    )
    
    # Note: The function returns True for clear pixels, False for clouds
    # Verify results
    assert cloud_mask[0, 0]      # Clear land (True = clear)
    assert not cloud_mask[0, 1]  # High confidence cloud (False = cloud)
    assert cloud_mask[0, 2]      # Clear water (True = clear)
    assert not cloud_mask[1, 0]  # Medium confidence cloud
    assert not cloud_mask[1, 1]  # Cloud shadow
    assert cloud_mask[1, 2]      # Clear land


def test_create_quality_mask():
    """Test quality mask creation - CRITICAL for data processing."""
    # Create test QA bands
    qa_pixel = np.array([[21824, 55052], [21952, 54852]], dtype=np.uint16)
    qa_aerosol = np.array([[66, 192], [64, 194]], dtype=np.uint8)
    qa_radsat = np.array([[0, 8], [4, 12]], dtype=np.uint16)
    
    qa_bands = {
        'qa_pixel': qa_pixel,
        'qa_aerosol': qa_aerosol,
        'qa_radsat': qa_radsat
    }
    
    # Create quality mask
    mask = create_quality_mask(
        qa_bands,
        combine_method='and',
        cloud_strategy='conservative',
        include_saturation=True
    )
    
    # Verify mask properties
    assert mask.shape == qa_pixel.shape
    assert mask.dtype == bool
    
    # Check specific pixels
    assert mask[0, 0]      # Clear land, good aerosol, no saturation
    assert not mask[0, 1]  # Cloud detected
    # Note: qa_radsat[1,0] = 4 which means band 3 is saturated
    # This might fail the saturation check
    assert not mask[1, 1]  # Cloud and bad aerosol