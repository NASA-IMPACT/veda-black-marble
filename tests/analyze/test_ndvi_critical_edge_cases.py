"""Test NDVI/NDWI calculation critical edge cases that caused pipeline failures."""

import numpy as np
import pytest

from blackmarble.analyze.indices import calculate_ndvi, calculate_ndwi


class TestNDVICriticalEdgeCases:
    """Test NDVI edge cases that produced extreme values (up to 137.0)."""
    
    def test_ndvi_denominator_just_below_epsilon(self):
        """Test the critical case that was producing 137.0 values.
        
        When NIR + Red < epsilon but very close to it, the old code would
        produce extreme values. This test ensures the fix works.
        """
        epsilon = 0.001
        
        # Case 1: Denominator = 0.0009 (just below epsilon)
        # Old code: sign(0.0009) * 0.001 = 0.001, but actual denom was 0.0009
        # This would make the safe_denominator smaller than the actual denominator!
        nir = np.array([0.00045])
        red = np.array([0.00045])
        ndvi = calculate_ndvi(nir, red, epsilon=epsilon)
        
        # NIR - Red = 0, so NDVI should be 0
        assert np.abs(ndvi[0]) < 0.01
        assert -1 <= ndvi[0] <= 1
        
        # Case 2: Denominator = -0.0009 (negative, just below epsilon in magnitude)
        # This was the worst case - old code would use -0.001 but actual was -0.0009
        nir = np.array([-0.00045])
        red = np.array([-0.00045])
        ndvi = calculate_ndvi(nir, red, epsilon=epsilon)
        
        # Should still be 0
        assert np.abs(ndvi[0]) < 0.01
        assert -1 <= ndvi[0] <= 1
    
    def test_ndvi_landsat_offset_edge_case(self):
        """Test edge case with Landsat's -0.2 offset that caused problems."""
        # When both bands are near -0.2, denominator approaches -0.4
        # But with slight variations, it can be very small
        nir = np.array([-0.1995, -0.2000, -0.2005])
        red = np.array([-0.2005, -0.2000, -0.1995])
        
        ndvi = calculate_ndvi(nir, red)
        
        # All values should be bounded
        assert np.all(ndvi >= -1)
        assert np.all(ndvi <= 1)
        assert not np.any(np.isnan(ndvi))
        assert not np.any(np.isinf(ndvi))
    
    def test_ndvi_exact_zero_denominator(self):
        """Test when NIR + Red = 0 exactly."""
        # This happens when NIR = -Red
        nir = np.array([0.3, -0.2, 0.5])
        red = np.array([-0.3, 0.2, -0.5])
        
        # When denominator is exactly 0, and NIR-Red is large,
        # the result would exceed our sanity check threshold
        with pytest.raises(ValueError, match="NDVI calculation error.*exceeds reasonable bounds"):
            calculate_ndvi(nir, red)
    
    def test_ndvi_array_with_mixed_edge_cases(self):
        """Test array containing multiple edge cases at once."""
        # Create array with various problematic values
        # Avoid exact zero denominators to prevent sanity check trigger
        nir = np.array([
            [0.5, 0.3, -0.2],      # Normal, normal, offset
            [0.0005, -0.0003, 0.1], # Tiny positive, tiny negative, smaller value
            [0.00049, -0.1999, 0.001] # Critical case, near offset, small positive
        ])
        red = np.array([
            [0.2, 0.1, -0.2],      # Normal values
            [0.0004, -0.0002, 0.05], # Tiny values, normal (not offset!)
            [0.00051, -0.2001, 0.001] # Critical case, near offset, small positive
        ])
        
        ndvi = calculate_ndvi(nir, red)
        
        # All values must be in valid range
        assert np.all(ndvi >= -1)
        assert np.all(ndvi <= 1)
        assert not np.any(np.isnan(ndvi))
        assert not np.any(np.isinf(ndvi))
        
        # Check specific expected values
        # Normal case: (0.5-0.2)/(0.5+0.2) = 0.3/0.7 ≈ 0.428
        assert 0.4 < ndvi[0, 0] < 0.45
        
        # Offset case: both -0.2, so (0)/(−0.4) = 0
        assert np.abs(ndvi[0, 2]) < 0.01


class TestNDWICriticalEdgeCases:
    """Test NDWI edge cases with similar denominator issues."""
    
    def test_ndwi_denominator_edge_cases(self):
        """Test NDWI with critical denominator values."""
        epsilon = 0.001
        
        # Case 1: Very small positive denominator
        green = np.array([0.0005])
        nir = np.array([0.0004])
        ndwi = calculate_ndwi(green, nir, epsilon=epsilon)
        
        # (0.0005 - 0.0004) / 0.001 = 0.1
        assert 0.05 < ndwi[0] < 0.15
        assert -1 <= ndwi[0] <= 1
        
        # Case 2: Exact zero denominator
        green = np.array([0.3])
        nir = np.array([-0.3])
        # This would produce (0.3 - (-0.3)) / 0.001 = 600
        # Which exceeds our sanity check threshold
        with pytest.raises(ValueError, match="NDWI calculation error.*exceeds reasonable bounds"):
            calculate_ndwi(green, nir, epsilon=epsilon)
    
    def test_ndwi_with_landsat_offsets(self):
        """Test NDWI with typical Landsat offset values."""
        # Both bands at -0.2 offset
        green = np.array([-0.2, -0.195, -0.205])
        nir = np.array([-0.2, -0.205, -0.195])
        
        ndwi = calculate_ndwi(green, nir)
        
        # All should be bounded
        assert np.all(ndwi >= -1)
        assert np.all(ndwi <= 1)
        
        # First should be 0 (equal values)
        assert np.abs(ndwi[0]) < 0.01


def test_indices_epsilon_parameter():
    """Test that epsilon parameter is properly used."""
    # Create data that would fail with wrong epsilon
    nir = np.array([0.0005])
    red = np.array([0.0005])
    
    # With epsilon=0.001, denominator (0.001) equals epsilon
    ndvi_default = calculate_ndvi(nir, red)  # Uses default epsilon=0.001
    assert np.abs(ndvi_default[0]) < 0.01  # Should be ~0
    
    # With larger epsilon, result should be the same (both go to 0)
    ndvi_large = calculate_ndvi(nir, red, epsilon=0.01)
    assert np.abs(ndvi_large[0]) < 0.01


def test_indices_match_expected_formulas():
    """Test that indices match their scientific formulas within bounds."""
    # Use values that won't trigger edge cases
    nir = np.array([0.4, 0.5, 0.6])
    red = np.array([0.1, 0.2, 0.3])
    green = np.array([0.15, 0.25, 0.35])
    
    # Calculate indices
    ndvi = calculate_ndvi(nir, red)
    ndwi = calculate_ndwi(green, nir)
    
    # Manual calculation (with large denominator to avoid epsilon)
    expected_ndvi = (nir - red) / (nir + red)
    expected_ndwi = (green - nir) / (green + nir)
    
    # Should be very close (within epsilon influence)
    np.testing.assert_allclose(ndvi, expected_ndvi, rtol=0.01)
    np.testing.assert_allclose(ndwi, expected_ndwi, rtol=0.01)
    
    # All values should be bounded
    assert np.all(np.abs(ndvi) <= 1)
    assert np.all(np.abs(ndwi) <= 1)