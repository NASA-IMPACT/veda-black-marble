"""Test NDVI calculation edge cases, especially near-zero denominators."""

import numpy as np
import pytest

from blackmarble.analyze.indices import calculate_ndvi, calculate_ndwi


class TestNDVIEdgeCases:
    """Test NDVI calculation with problematic edge cases."""
    
    def test_normal_ndvi_values(self):
        """Test NDVI with normal reflectance values."""
        nir = np.array([0.4, 0.5, 0.6])
        red = np.array([0.1, 0.2, 0.3])
        ndvi = calculate_ndvi(nir, red)
        
        # Expected values calculated manually
        expected = np.array([0.6, 0.428571, 0.333333])
        np.testing.assert_allclose(ndvi, expected, rtol=1e-5)
        
        # Ensure all values are in valid range
        assert np.all(ndvi >= -1)
        assert np.all(ndvi <= 1)
    
    def test_near_zero_denominator_positive(self):
        """Test when denominator is very small but positive."""
        # NIR + Red = 0.0001, which is less than epsilon=0.001
        nir = np.array([0.00005])
        red = np.array([0.00005])
        ndvi = calculate_ndvi(nir, red)
        
        # Should be 0 since NIR - Red = 0
        assert np.abs(ndvi[0]) < 0.01
        assert -1 <= ndvi[0] <= 1
    
    def test_near_zero_denominator_negative(self):
        """Test when denominator is very small but negative."""
        # With Landsat offset: NIR=-0.1999, Red=-0.2001
        # NIR + Red = -0.4, NIR - Red = 0.0002
        nir = np.array([-0.1999])
        red = np.array([-0.2001])
        ndvi = calculate_ndvi(nir, red)
        
        # Should be a small positive value
        assert 0 < ndvi[0] < 0.01
        assert -1 <= ndvi[0] <= 1
    
    def test_exact_zero_denominator(self):
        """Test when denominator is exactly zero."""
        # NIR + Red = 0
        nir = np.array([0.5])
        red = np.array([-0.5])
        ndvi = calculate_ndvi(nir, red)
        
        # Should handle gracefully - result should be 1.0 (NIR - Red = 1.0, denom = epsilon)
        assert ndvi[0] == 1.0  # Maximum NDVI value
    
    def test_landsat_offset_values(self):
        """Test with typical Landsat offset values."""
        # Both bands at the -0.2 offset
        nir = np.array([-0.2, -0.2, -0.2])
        red = np.array([-0.2, -0.2, -0.2])
        ndvi = calculate_ndvi(nir, red)
        
        # Should all be 0 since NIR = Red
        np.testing.assert_allclose(ndvi, [0, 0, 0], atol=1e-10)
    
    def test_critical_denominator_just_below_epsilon(self):
        """Test the critical case where denominator is just below epsilon."""
        # This was causing the 137.0 error
        # NIR + Red = 0.0009, which is less than epsilon=0.001
        nir = np.array([0.0005])
        red = np.array([0.0004])
        ndvi = calculate_ndvi(nir, red)
        
        # (0.0005 - 0.0004) / 0.001 = 0.1
        expected = 0.1
        assert np.abs(ndvi[0] - expected) < 0.01
        assert -1 <= ndvi[0] <= 1
    
    def test_very_small_negative_denominator(self):
        """Test when denominator is -0.000999 (just above -epsilon)."""
        # This case was producing extreme values
        nir = np.array([-0.0004995])
        red = np.array([-0.0004995])
        ndvi = calculate_ndvi(nir, red)
        
        # Should be 0 since NIR = Red
        assert np.abs(ndvi[0]) < 0.01
        assert -1 <= ndvi[0] <= 1
    
    def test_ndvi_with_various_edge_cases(self):
        """Test a variety of edge cases in one array."""
        nir = np.array([
            0.6,      # Normal case
            -0.2,     # Offset value
            0.0005,   # Very small positive
            -0.0005,  # Very small negative
            0.5,      # NIR + Red = 0 case
        ])
        red = np.array([
            0.2,      # Normal case
            -0.2,     # Offset value
            0.0004,   # Very small positive
            -0.0004,  # Very small negative
            -0.5,     # NIR + Red = 0 case
        ])
        
        ndvi = calculate_ndvi(nir, red)
        
        # All values should be in valid range
        assert np.all(ndvi >= -1)
        assert np.all(ndvi <= 1)
        assert not np.any(np.isnan(ndvi))
        assert not np.any(np.isinf(ndvi))
    
    def test_ndvi_sanity_check_catches_extreme_values(self):
        """Test that the sanity check would catch extreme values if they occurred."""
        # This test verifies our safety rail works
        # We'll mock a situation where the calculation produces extreme values
        nir = np.array([10.0])  # Unrealistic value
        red = np.array([0.001])
        
        # The sanity check should catch values > 5.0
        with pytest.raises(ValueError, match="NDVI calculation error.*exceeds reasonable bounds"):
            calculate_ndvi(nir, red)


class TestNDWIEdgeCases:
    """Test NDWI calculation with similar edge cases."""
    
    def test_ndwi_near_zero_denominator(self):
        """Test NDWI with near-zero denominator."""
        green = np.array([0.0005])
        nir = np.array([0.0004])
        ndwi = calculate_ndwi(green, nir)
        
        # Should be a small positive value
        assert 0 < ndwi[0] < 0.2
        assert -1 <= ndwi[0] <= 1
    
    def test_ndwi_exact_zero_denominator(self):
        """Test NDWI when denominator is exactly zero."""
        green = np.array([0.3])
        nir = np.array([-0.3])
        ndwi = calculate_ndwi(green, nir)
        
        # (0.3 - (-0.3)) / epsilon = 0.6 / 0.001 = 600, but capped at 1
        assert ndwi[0] == 1.0
    
    def test_ndwi_landsat_offset_values(self):
        """Test NDWI with Landsat offset values."""
        green = np.array([-0.2, -0.2])
        nir = np.array([-0.2, -0.2])
        ndwi = calculate_ndwi(green, nir)
        
        # Should be 0 since green = nir
        np.testing.assert_allclose(ndwi, [0, 0], atol=1e-10)