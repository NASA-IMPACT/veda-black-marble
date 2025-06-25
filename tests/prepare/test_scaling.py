"""Test critical scaling functions for bm.py parity.

This module tests only the essential scaling functionality needed for bm.py parity.
"""

import pytest
import numpy as np

from blackmarble.prepare.scaling import (
    scale_landsat_bands,
    detect_if_already_scaled,
    calculate_ntli,
    apply_viirs_floor_ceiling,
)


def test_scale_landsat_bands():
    """Test Landsat band scaling - CRITICAL for bm.py parity.
    
    This ensures Landsat Collection 2 DN values are properly converted to reflectance.
    """
    # Create test data matching Landsat Collection 2 DN values
    bands = {
        'blue': np.array([[10000, 20000, 30000]], dtype=np.uint16),
        'green': np.array([[15000, 25000, 35000]], dtype=np.uint16),
        'red': np.array([[12000, 22000, 32000]], dtype=np.uint16),
        'nir': np.array([[18000, 28000, 38000]], dtype=np.uint16),
        'swir1': np.array([[14000, 24000, 34000]], dtype=np.uint16),
        'swir2': np.array([[11000, 21000, 31000]], dtype=np.uint16),
    }
    
    # Apply scaling to each band individually
    scaled = {}
    for band_name, band_data in bands.items():
        scaled[band_name] = scale_landsat_bands(band_data)
    
    # Verify scaling was applied
    for band_name in bands:
        # Should be converted to reflectance (0-1 range typical)
        assert scaled[band_name].dtype == np.float32 or scaled[band_name].dtype == np.float64
        # Some values might be NaN due to masking
        valid_mask = ~np.isnan(scaled[band_name])
        if np.any(valid_mask):
            assert np.all(scaled[band_name][valid_mask] < 10)  # Reflectance shouldn't be huge
        
    # Test specific values (approximate due to scaling)
    # DN 20000 should scale to ~0.35 reflectance (0.0000275 * 20000 - 0.2)
    assert 0.3 < scaled['blue'][0, 1] < 0.4


def test_detect_if_already_scaled():
    """Test detection of pre-scaled data - prevents double scaling."""
    # Create reflectance data (already scaled)
    reflectance_data = np.array([[0.1, 0.2, 0.3],
                                 [0.15, 0.25, 0.35],
                                 [0.2, 0.3, 0.4]])
    
    # Create DN data (not scaled)
    dn_data = np.array([[10000, 20000, 30000],
                        [15000, 25000, 35000],
                        [12000, 22000, 32000]], dtype=np.uint16)
    
    # Test detection
    assert detect_if_already_scaled(reflectance_data) == True
    assert detect_if_already_scaled(dn_data) == False


def test_calculate_ntli():
    """Test NTLI calculation - CRITICAL for bm.py parity.
    
    NTLI = Nighttime Light Index, the VIIRS-based component of NDUI.
    """
    # Create test VIIRS data
    viirs_data = np.array([[0.0, 0.5, 1.0],
                           [1.5, 2.0, 2.5],
                           [3.0, 5.0, 65531.0]])  # Include fill value
    
    # Create quality flag array
    quality_flag = np.array([[0, 2, 0],
                            [2, 0, 1],
                            [0, 2, 3]])  # 1 and 3 are invalid
    
    # Calculate NTLI with quality masking
    ntli = calculate_ntli(viirs_data, quality_flag=quality_flag)
    
    # Verify NTLI properties
    assert ntli.shape == viirs_data.shape
    assert np.all(ntli >= 0)
    
    # Check specific values
    # Zero light should give NTLI = 0
    assert ntli[0, 0] == 0.0
    
    # Invalid quality flag should give 0
    assert ntli[1, 2] == 0.0  # quality flag = 1
    assert ntli[2, 2] == 0.0  # fill value
    
    # Valid values should pass through
    assert ntli[0, 1] == 0.5  # quality flag = 2
    assert ntli[1, 1] == 2.0  # quality flag = 0


def test_apply_viirs_floor_ceiling():
    """Test VIIRS floor/ceiling application - ensures valid data range."""
    # Create test data with outliers
    viirs_data = np.array([[-0.5, 0.0, 0.5],
                           [1.0, 10.0, 100.0],
                           [1000.0, 0.3, 0.7]])
    
    # Apply floor/ceiling
    clipped = apply_viirs_floor_ceiling(
        viirs_data,
        floor_value=0.0,
        ceiling_value=10.0
    )
    
    # Verify clipping
    assert np.all(clipped >= 0.0)
    assert np.all(clipped <= 10.0)
    
    # Check specific values
    assert clipped[0, 0] == 0.0  # Floor applied
    assert clipped[1, 2] == 10.0  # Ceiling applied
    assert clipped[2, 1] == 0.3  # Within range, unchanged