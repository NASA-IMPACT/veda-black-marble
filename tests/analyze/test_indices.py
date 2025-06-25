"""Test critical spectral indices for bm.py parity.

This module tests only the essential indices functionality needed for bm.py parity.
"""

import pytest
import numpy as np

from blackmarble.analyze.indices import (
    calculate_ndvi,
    calculate_ndwi,
    calculate_ndui,
)


def test_calculate_ndvi():
    """Test NDVI calculation - CRITICAL for vegetation masking in NDUI."""
    # Create test data
    nir_band = np.array([[0.5, 0.7], [0.3, 0.9]])
    red_band = np.array([[0.2, 0.1], [0.4, 0.2]])
    
    # Calculate NDVI
    result = calculate_ndvi(nir_band, red_band)
    
    # Verify NDVI formula (NIR - Red) / (NIR + Red)
    expected = (nir_band - red_band) / (nir_band + red_band + 0.0001)
    np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    # Verify output shape
    assert result.shape == nir_band.shape
    
    # Test division by zero protection
    zero_nir = np.array([[0.0, 0.0]])
    zero_red = np.array([[0.0, 0.0]])
    result_zeros = calculate_ndvi(zero_nir, zero_red)
    
    # Should not produce NaN or inf
    assert not np.any(np.isnan(result_zeros))
    assert not np.any(np.isinf(result_zeros))


def test_calculate_ndwi():
    """Test NDWI calculation - CRITICAL for water masking in NDUI."""
    # Create test data
    green_band = np.array([[0.3, 0.5], [0.2, 0.6]])
    nir_band = np.array([[0.5, 0.2], [0.8, 0.1]])
    
    # Calculate NDWI
    result = calculate_ndwi(green_band, nir_band)
    
    # Verify NDWI formula (Green - NIR) / (Green + NIR)
    expected = (green_band - nir_band) / (green_band + nir_band + 0.0001)
    np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    # Water should have positive NDWI (green > NIR)
    # Test case where green > NIR (water)
    assert result[0, 1] > 0  # 0.5 > 0.2
    
    # Land should have negative NDWI (NIR > green)
    assert result[1, 0] < 0  # 0.2 < 0.8


def test_calculate_ndui():
    """Test NDUI calculation - CRITICAL for bm.py parity.
    
    NDUI = Normalized Difference Urban Index
    This is the core index that combines Landsat and VIIRS data.
    """
    # Create test data with realistic NTL values (scaled nW/cm²/sr)
    # Using scaled values in range 0-50 nW/cm²/sr
    ntl = np.array([[0.1, 2.0, 8.0],     # Low, medium, high lights
                    [0.5, 5.0, 15.0],
                    [1.0, 10.0, 25.0]])
    
    ndvi = np.array([[0.8, 0.5, -0.1],   # Vegetation index  
                     [0.6, 0.3, -0.2],
                     [0.4, 0.1, -0.3]])
    
    ndwi = np.array([[-0.2, -0.1, -0.4], # Water index (negative = no water)
                     [-0.3, -0.2, -0.5],
                     [-0.4, 0.8, 0.9]])   # Last row has water pixels
    
    # Calculate NDUI
    result = calculate_ndui(ntl, ndvi, ndwi)
    
    # Verify NDUI properties
    assert result.shape == ntl.shape
    
    # Urban areas: high NTL, low NDVI, negative NDWI (no water)
    # Rural/vegetation: low NTL, high NDVI
    # Water: positive NDWI results in NDUI=0
    
    # Check urban pixel (high lights, no vegetation, no water)
    urban_idx = (1, 2)  # NTL=15.0, NDVI=-0.2, NDWI=-0.5 (no water)
    assert result[urban_idx] > 0.5  # Should have high NDUI
    
    # Check vegetation pixel (low lights, high vegetation)
    veg_idx = (0, 0)  # NTL=0.1, NDVI=0.8, NDWI=-0.2
    assert result[veg_idx] < 0.1  # Should have very low NDUI
    
    # Check water pixels (positive NDWI)
    water_idx1 = (2, 1)  # NDWI=0.8 (water)
    water_idx2 = (2, 2)  # NDWI=0.9 (water)
    assert result[water_idx1] == 0.0  # Water pixels get NDUI=0
    assert result[water_idx2] == 0.0  # Water pixels get NDUI=0