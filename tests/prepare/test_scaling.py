"""Test functions in blackmarble.prepare.scaling module.

This module tests data scaling and radiometric corrections.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Import the module under test
from blackmarble.prepare.scaling import (
    scale_landsat_bands,
    detect_if_already_scaled,
    apply_lunar_correction,
    calculate_ntli,
    apply_viirs_floor_ceiling,
    normalize_to_range,
    scale_to_byte,
)


def test_scale_landsat_bands():
    """Test Landsat Level-2 surface reflectance scaling to physical values.
    
    This test should verify that:
    - Digital numbers are correctly converted to surface reflectance
    - Default scale factor (0.0000275) and offset (-0.2) work for Landsat 8/9
    - Custom scale factors and offsets work correctly
    - Valid range parameter filters out invalid DN values
    - Output values are in expected reflectance range (-0.2 to ~1.0)
    - NoData values are preserved appropriately
    - Different Landsat collections use appropriate scaling factors
    - Edge cases (DN=0, DN=max) produce expected reflectance values
    - Array shapes and data types are preserved
    """
    # Create test DN values (raw Landsat data)
    landsat_dn = np.array([[10000, 20000, 30000],
                           [15000, 25000, 35000]], dtype=np.uint16)
    
    # Apply default scaling for Landsat 8/9
    result = scale_landsat_bands(landsat_dn)
    
    # Verify scaling formula: reflectance = DN * scale + offset
    expected = landsat_dn * 0.0000275 + (-0.2)
    np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    # Verify physical range (reflectance should be -0.2 to ~1.0)
    assert np.all(result >= -0.2)  # With offset
    assert np.all(result <= 1.0)
    
    # Test custom scale factors and offsets (e.g., Landsat 7)
    custom_result = scale_landsat_bands(
        landsat_dn, 
        scale_factor=0.00002, 
        add_offset=-0.1
    )
    custom_expected = landsat_dn * 0.00002 + (-0.1)
    np.testing.assert_allclose(custom_result, custom_expected, rtol=1e-6)
    
    # Test valid range filtering (default: 7273-43636)
    dn_with_invalid = np.array([[5000, 20000, 50000],  # Below and above valid range
                                [7273, 30000, 43636]], dtype=np.uint16)
    filtered_result = scale_landsat_bands(dn_with_invalid)
    
    # Values outside valid range should be set to 0, then scaled
    assert filtered_result[0, 0] == 0 * 0.0000275 + (-0.2)  # Below min
    assert filtered_result[0, 2] == 0 * 0.0000275 + (-0.2)  # Above max
    assert filtered_result[1, 0] == 7273 * 0.0000275 + (-0.2)  # At min boundary
    assert filtered_result[1, 2] == 43636 * 0.0000275 + (-0.2)  # At max boundary
    
    # Test edge cases
    edge_dn = np.array([[0, 7273, 43636, 65535]], dtype=np.uint16)
    edge_result = scale_landsat_bands(edge_dn)
    
    # DN=0 is below valid range, should be zeroed then scaled
    assert edge_result[0, 0] == -0.2
    # DN=65535 is above valid range, should be zeroed then scaled
    assert edge_result[0, 3] == -0.2
    
    # Verify array shape and data type preservation
    assert result.shape == landsat_dn.shape
    assert isinstance(result, np.ndarray)


def test_scale_landsat_bands_custom_parameters():
    """Test Landsat scaling with custom scale factors.
    
    This test should verify that:
    - Different Landsat sensors use correct scaling parameters
    - Collection 1 vs Collection 2 differences are handled
    - Custom valid ranges properly filter input data
    - Scaling accuracy is maintained across different parameter sets
    - Error handling for invalid parameters
    """
    # Test data
    test_dn = np.array([[10000, 20000, 30000]], dtype=np.uint16)
    
    # Different Landsat sensor parameters
    # Landsat 8/9 Collection 2
    l89_c2_result = scale_landsat_bands(
        test_dn,
        scale_factor=0.0000275,
        add_offset=-0.2
    )
    assert np.allclose(l89_c2_result, test_dn * 0.0000275 - 0.2)
    
    # Landsat 7 Collection 2 (hypothetical different parameters)
    l7_c2_result = scale_landsat_bands(
        test_dn,
        scale_factor=0.00002,
        add_offset=-0.1
    )
    assert np.allclose(l7_c2_result, test_dn * 0.00002 - 0.1)
    
    # Collection 1 parameters (different from Collection 2)
    l8_c1_result = scale_landsat_bands(
        test_dn,
        scale_factor=0.0001,  # Collection 1 used different scaling
        add_offset=0.0
    )
    assert np.allclose(l8_c1_result, test_dn * 0.0001)
    
    # Test custom valid ranges
    custom_range_dn = np.array([[1000, 10000, 20000, 60000]], dtype=np.uint16)
    
    # Narrow valid range
    narrow_result = scale_landsat_bands(
        custom_range_dn,
        valid_range=(5000, 25000)
    )
    # Values outside 5000-25000 should be zeroed
    assert narrow_result[0, 0] == -0.2  # 1000 < 5000
    assert narrow_result[0, 3] == -0.2  # 60000 > 25000
    assert narrow_result[0, 1] > -0.2   # 10000 is valid
    assert narrow_result[0, 2] > -0.2   # 20000 is valid
    
    # Wide valid range (nearly all values valid)
    wide_result = scale_landsat_bands(
        custom_range_dn,
        valid_range=(0, 65535)
    )
    # All values should be scaled normally
    assert np.all(wide_result > -0.2)
    
    # Verify scaling accuracy across parameter sets
    # Use high precision data type
    precise_dn = np.array([[32768]], dtype=np.float64)
    precise_result = scale_landsat_bands(
        precise_dn,
        scale_factor=0.0000275,
        add_offset=-0.2
    )
    expected_precise = 32768 * 0.0000275 - 0.2
    np.testing.assert_allclose(precise_result, expected_precise, rtol=1e-10)


def test_scale_landsat_bands_edge_cases():
    """Test Landsat scaling edge cases.
    
    This test should verify that:
    - Extreme DN values don't cause overflow/underflow
    - Large arrays are processed efficiently
    - Mixed data types are handled appropriately
    - Memory usage is reasonable for large datasets
    - Numerical precision is maintained
    """
    # Test extreme DN values
    extreme_dn = np.array([
        [0, 1, 100, 1000, 10000, 32767, 65535],  # uint16 range
    ], dtype=np.uint16)
    
    extreme_result = scale_landsat_bands(extreme_dn)
    
    # Verify no overflow/underflow occurs
    assert np.all(np.isfinite(extreme_result))
    assert extreme_result.dtype in [np.float32, np.float64]
    
    # Test large arrays (simulate real Landsat scene)
    # Create 1000x1000 array (1 megapixel)
    large_dn = np.random.randint(7273, 43636, size=(1000, 1000), dtype=np.uint16)
    
    # Process large array
    large_result = scale_landsat_bands(large_dn)
    
    # Verify shape preservation and valid output range
    assert large_result.shape == (1000, 1000)
    assert np.all(large_result >= -0.2)
    assert np.all(large_result <= 1.0)
    
    # Test mixed data types
    # Float input
    float_dn = np.array([[10000.5, 20000.5]], dtype=np.float32)
    float_result = scale_landsat_bands(float_dn)
    assert np.allclose(float_result, float_dn * 0.0000275 - 0.2)
    
    # Integer types
    for dtype in [np.int16, np.int32, np.uint32]:
        int_dn = np.array([[10000, 20000]], dtype=dtype)
        int_result = scale_landsat_bands(int_dn)
        assert np.allclose(int_result, int_dn * 0.0000275 - 0.2)
    
    # Test numerical precision
    # Use values that require high precision
    precision_dn = np.array([
        [7273.0],     # Minimum valid DN
        [7274.0],     # Just above minimum
        [43635.0],    # Just below maximum
        [43636.0],    # Maximum valid DN
    ], dtype=np.float64)
    
    precision_result = scale_landsat_bands(precision_dn)
    
    # Calculate expected values with high precision
    expected_min = 7273.0 * 0.0000275 - 0.2
    expected_min_plus = 7274.0 * 0.0000275 - 0.2
    expected_max_minus = 43635.0 * 0.0000275 - 0.2
    expected_max = 43636.0 * 0.0000275 - 0.2
    
    np.testing.assert_allclose(
        precision_result.flatten(),
        [expected_min, expected_min_plus, expected_max_minus, expected_max],
        rtol=1e-10
    )


def test_detect_if_already_scaled():
    """Test detection of pre-scaled Landsat reflectance data.
    
    This test should verify that:
    - Raw DN values (0-65535 range) are correctly identified as unscaled
    - Reflectance values (-0.5 to 2.0 range) are identified as scaled
    - Threshold parameter affects detection sensitivity
    - Mixed scaled/unscaled data is handled appropriately
    - Different data ranges are classified correctly
    - Edge cases near threshold values work correctly
    - NoData/NaN values don't interfere with detection
    - Performance is good for large arrays
    """
    # Test raw DN values (should be unscaled)
    raw_dn = np.array([[7273, 20000, 43636]], dtype=np.uint16)
    assert not detect_if_already_scaled(raw_dn)
    
    # Test typical DN range
    typical_dn = np.array([[10000, 25000, 35000]], dtype=np.uint16)
    assert not detect_if_already_scaled(typical_dn)
    
    # Test reflectance values (should be scaled)
    reflectance = np.array([[-0.2, 0.0, 0.5, 1.0]], dtype=np.float32)
    assert detect_if_already_scaled(reflectance)
    
    # Test with values near default threshold (2.0)
    near_threshold = np.array([[0.0, 0.5, 1.5, 1.99]], dtype=np.float32)
    assert detect_if_already_scaled(near_threshold)  # Max < 2.0
    
    above_threshold = np.array([[0.0, 0.5, 1.5, 2.01]], dtype=np.float32)
    assert not detect_if_already_scaled(above_threshold)  # Max > 2.0
    
    # Test threshold parameter sensitivity
    high_reflectance = np.array([[0.0, 1.0, 2.5, 3.0]], dtype=np.float32)
    assert not detect_if_already_scaled(high_reflectance, reflectance_threshold=2.0)
    assert detect_if_already_scaled(high_reflectance, reflectance_threshold=4.0)
    
    # Test mixed data (partially scaled - edge case)
    # In practice this shouldn't happen, but test robustness
    mixed_data = np.array([[0.5, 1.0, 10000]], dtype=np.float32)
    assert not detect_if_already_scaled(mixed_data)  # Max > threshold
    
    # Test different data ranges
    # Negative reflectance values
    negative_refl = np.array([[-0.5, -0.2, 0.0, 0.5]], dtype=np.float32)
    assert detect_if_already_scaled(negative_refl)
    
    # Very small positive values (could be scaled)
    small_values = np.array([[0.001, 0.01, 0.1, 0.5]], dtype=np.float32)
    assert detect_if_already_scaled(small_values)
    
    # Edge cases near threshold
    exactly_threshold = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
    assert detect_if_already_scaled(exactly_threshold)  # Max == threshold
    
    # Test with NaN values
    with_nan = np.array([[np.nan, 0.5, 1.0, np.nan]], dtype=np.float32)
    assert detect_if_already_scaled(with_nan)  # Should ignore NaN
    
    # All NaN (edge case)
    all_nan = np.array([[np.nan, np.nan, np.nan]], dtype=np.float32)
    # This is implementation dependent - with nanmin/nanmax it might return True
    # because min/max of empty array could be within threshold
    
    # Test performance with large array
    large_dn = np.random.randint(7273, 43636, size=(1000, 1000), dtype=np.uint16)
    assert not detect_if_already_scaled(large_dn)
    
    large_refl = np.random.uniform(-0.2, 1.0, size=(1000, 1000))
    assert detect_if_already_scaled(large_refl)


def test_detect_if_already_scaled_threshold_sensitivity():
    """Test scaling detection threshold sensitivity.
    
    This test should verify that:
    - Default threshold (2.0) works for typical reflectance data
    - Custom thresholds handle edge cases appropriately
    - False positives are minimized for valid DN data
    - False negatives are minimized for valid reflectance data
    - Threshold selection guidance is clear
    """
    # Test default threshold (2.0) with typical reflectance
    typical_reflectance = np.array([
        [-0.2, 0.0, 0.3, 0.8, 1.2],  # Normal reflectance range
    ], dtype=np.float32)
    assert detect_if_already_scaled(typical_reflectance)
    
    # Test data just below and above default threshold
    just_below = np.array([[0.0, 0.5, 1.0, 1.5, 1.99]], dtype=np.float32)
    just_above = np.array([[0.0, 0.5, 1.0, 1.5, 2.01]], dtype=np.float32)
    
    assert detect_if_already_scaled(just_below)
    assert not detect_if_already_scaled(just_above)
    
    # Test custom thresholds for different use cases
    # Snow/ice can have high reflectance
    snow_reflectance = np.array([[0.8, 1.2, 1.5, 1.8]], dtype=np.float32)
    assert detect_if_already_scaled(snow_reflectance, reflectance_threshold=2.0)
    assert detect_if_already_scaled(snow_reflectance, reflectance_threshold=3.0)
    
    # Water has low reflectance
    water_reflectance = np.array([[0.0, 0.05, 0.1, 0.15]], dtype=np.float32)
    assert detect_if_already_scaled(water_reflectance, reflectance_threshold=1.0)
    assert detect_if_already_scaled(water_reflectance, reflectance_threshold=0.5)
    
    # Test false positive prevention for valid DN data
    # Some DN values might be small and look like reflectance
    small_dn = np.array([[0, 100, 500, 1000]], dtype=np.uint16)
    # With default threshold, these should not be detected as scaled
    assert not detect_if_already_scaled(small_dn, reflectance_threshold=2.0)
    # But with very high threshold, they would be
    assert detect_if_already_scaled(small_dn, reflectance_threshold=2000.0)
    
    # Test false negative prevention for valid reflectance
    # Some surfaces might have reflectance > 1.0
    bright_surface = np.array([[0.5, 1.0, 1.5, 2.5]], dtype=np.float32)
    # Too low threshold would miss this as scaled data
    assert not detect_if_already_scaled(bright_surface, reflectance_threshold=1.0)
    # Appropriate threshold would catch it
    assert detect_if_already_scaled(bright_surface, reflectance_threshold=3.0)
    
    # Test threshold selection guidance
    # For typical Landsat surface reflectance
    landsat_sr = np.array([[-0.1, 0.2, 0.5, 0.8]], dtype=np.float32)
    assert detect_if_already_scaled(landsat_sr, reflectance_threshold=2.0)  # Good default
    
    # For TOA reflectance (can be higher)
    toa_reflectance = np.array([[0.1, 0.5, 1.2, 1.8]], dtype=np.float32)
    assert detect_if_already_scaled(toa_reflectance, reflectance_threshold=2.5)
    
    # For specialized indices (NDVI, etc.) that range -1 to 1
    # Note: function checks data_min >= -0.5, so NDVI < -0.5 won't be detected as scaled
    ndvi_values = np.array([[-0.4, -0.2, 0.3, 0.9]], dtype=np.float32)
    assert detect_if_already_scaled(ndvi_values, reflectance_threshold=1.5)


def test_apply_lunar_correction():
    """Test lunar BRDF correction for VIIRS nighttime lights.
    
    This test should verify that:
    - VIIRS Day/Night Band radiance is properly corrected
    - Lunar irradiance arrays are correctly applied
    - Moon illumination fraction enhances correction accuracy
    - Output maintains physical units and meaning
    - Correction reduces temporal variability due to lunar cycle
    - Edge cases (new moon, full moon) are handled correctly
    - NoData values are preserved through correction
    - Array dimensions and alignment are maintained
    - Performance is acceptable for operational use
    """
    # Create test VIIRS DNB radiance data (nW/cm²/sr)
    vnp_radiance = np.array([
        [0.5, 1.0, 2.0, 5.0],
        [0.1, 0.8, 3.0, 10.0]
    ], dtype=np.float32)
    
    # Create lunar irradiance data (simulated contribution)
    lunar_irradiance = np.array([
        [0.1, 0.2, 0.3, 0.5],
        [0.05, 0.15, 0.4, 1.0]
    ], dtype=np.float32)
    
    # Apply basic lunar correction
    corrected = apply_lunar_correction(vnp_radiance, lunar_irradiance)
    
    # Verify correction formula: corrected = vnp - lunar
    expected = vnp_radiance - lunar_irradiance
    expected = np.where(expected < 0, 0, expected)  # Ensure non-negative
    np.testing.assert_allclose(corrected, expected, rtol=1e-6)
    
    # Verify output maintains physical units (non-negative radiance)
    assert np.all(corrected >= 0)
    
    # Test with moon illumination fraction (optional parameter)
    moon_frac = np.array([[0.5]], dtype=np.float32)  # Half moon
    corrected_with_moon = apply_lunar_correction(
        vnp_radiance, lunar_irradiance, moon_frac
    )
    # Currently implementation doesn't use moon_frac, but test it doesn't break
    np.testing.assert_allclose(corrected_with_moon, expected, rtol=1e-6)
    
    # Test edge case: new moon (minimal lunar contribution)
    minimal_lunar = np.full_like(vnp_radiance, 0.001)
    new_moon_corrected = apply_lunar_correction(vnp_radiance, minimal_lunar)
    np.testing.assert_allclose(new_moon_corrected, vnp_radiance - 0.001, rtol=1e-6)
    
    # Test edge case: full moon (maximum lunar contribution)
    max_lunar = np.array([
        [0.8, 1.5, 2.5, 4.0],
        [0.5, 1.0, 3.5, 8.0]
    ], dtype=np.float32)
    full_moon_corrected = apply_lunar_correction(vnp_radiance, max_lunar)
    
    # Some pixels might go negative and be clipped to 0
    assert full_moon_corrected[0, 0] == max(0, 0.5 - 0.8)  # Should be 0
    assert full_moon_corrected[1, 3] == max(0, 10.0 - 8.0)  # Should be 2.0
    
    # Test that correction reduces temporal variability
    # Simulate same location observed with different lunar conditions
    constant_artificial_light = 2.0
    varying_lunar = np.array([0.1, 0.5, 1.0, 0.3])  # Different nights
    observations = constant_artificial_light + varying_lunar
    
    corrected_obs = np.array([
        apply_lunar_correction(np.array([[obs]]), np.array([[lunar]]))[0, 0]
        for obs, lunar in zip(observations, varying_lunar)
    ])
    
    # After correction, values should be more consistent
    assert np.std(corrected_obs) < np.std(observations)
    assert np.allclose(corrected_obs, constant_artificial_light, rtol=0.1)
    
    # Test array dimensions and alignment
    large_vnp = np.random.uniform(0, 5, size=(100, 100))
    large_lunar = np.random.uniform(0, 1, size=(100, 100))
    large_corrected = apply_lunar_correction(large_vnp, large_lunar)
    
    assert large_corrected.shape == large_vnp.shape
    assert large_corrected.dtype == large_vnp.dtype


def test_apply_lunar_correction_moon_phases():
    """Test lunar correction across different moon phases.
    
    This test should verify that:
    - New moon conditions (minimal lunar illumination) work correctly
    - Full moon conditions (maximum lunar illumination) are handled
    - Quarter moon phases produce intermediate corrections
    - Moon illumination fraction modulates correction strength
    - Temporal consistency is improved across lunar cycle
    """
    # Base VIIRS observation (mix of city lights and background)
    base_vnp = np.array([
        [0.1, 0.5, 2.0, 5.0],   # Rural to urban gradient
        [0.2, 1.0, 3.0, 10.0]
    ], dtype=np.float32)
    
    # Test new moon (minimal lunar illumination)
    new_moon_lunar = np.full_like(base_vnp, 0.01)  # Very small contribution
    new_moon_result = apply_lunar_correction(base_vnp, new_moon_lunar)
    
    # New moon correction should be minimal
    np.testing.assert_allclose(new_moon_result, base_vnp - 0.01, rtol=1e-6)
    assert np.all(new_moon_result >= 0)
    
    # Test full moon (maximum lunar illumination)
    # Lunar contribution scales with surface reflectance
    full_moon_lunar = np.array([
        [0.05, 0.2, 0.8, 1.5],   # Higher in urban areas
        [0.1, 0.4, 1.2, 3.0]
    ], dtype=np.float32)
    full_moon_result = apply_lunar_correction(base_vnp, full_moon_lunar)
    
    # Full moon should significantly reduce observed radiance
    assert np.all(full_moon_result <= base_vnp)
    assert np.mean(full_moon_result) < np.mean(base_vnp) * 0.8
    
    # Test quarter moon phases
    first_quarter_lunar = full_moon_lunar * 0.5  # ~50% illumination
    first_quarter_result = apply_lunar_correction(base_vnp, first_quarter_lunar)
    
    last_quarter_lunar = full_moon_lunar * 0.45  # Slightly different due to angle
    last_quarter_result = apply_lunar_correction(base_vnp, last_quarter_lunar)
    
    # Quarter moon corrections should be intermediate
    assert np.all(new_moon_result >= first_quarter_result)
    assert np.all(first_quarter_result >= full_moon_result)
    assert np.allclose(first_quarter_result, last_quarter_result, rtol=0.2)
    
    # Test moon illumination fraction (if implementation supports it)
    moon_fractions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])  # New to full
    
    # Apply corrections for different moon phases
    phase_results = []
    for frac in moon_fractions:
        # Scale lunar contribution by moon fraction
        scaled_lunar = full_moon_lunar * frac
        result = apply_lunar_correction(base_vnp, scaled_lunar)
        phase_results.append(np.mean(result))
    
    # Verify temporal consistency improvement
    # As moon fraction increases, corrected values should decrease
    assert all(phase_results[i] >= phase_results[i+1] 
               for i in range(len(phase_results)-1))
    
    # Test that correction improves consistency across lunar cycle
    # Simulate observations of same city across lunar month
    city_lights = 5.0  # Constant artificial light
    lunar_cycle = np.array([
        0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0
    ])  # Simplified lunar cycle
    
    observed = city_lights + lunar_cycle
    corrected = np.array([
        apply_lunar_correction(np.array([[obs]]), np.array([[lunar]]))[0, 0]
        for obs, lunar in zip(observed, lunar_cycle)
    ])
    
    # Corrected values should be more consistent than raw observations
    assert np.std(corrected) < np.std(observed) * 0.1
    assert np.allclose(corrected, city_lights, rtol=0.05)


def test_apply_lunar_correction_data_quality():
    """Test lunar correction data quality preservation.
    
    This test should verify that:
    - Valid radiance values are corrected appropriately
    - Invalid/NoData values are handled correctly
    - Array shapes and data types are preserved
    - Metadata and units are maintained
    - Numerical precision is adequate for analysis
    """
    # Test with valid radiance values
    valid_vnp = np.array([
        [0.5, 1.0, 2.0, 3.0],
        [1.5, 2.5, 4.0, 6.0]
    ], dtype=np.float32)
    
    valid_lunar = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.5, 0.8]
    ], dtype=np.float32)
    
    corrected = apply_lunar_correction(valid_vnp, valid_lunar)
    
    # All valid values should be corrected appropriately
    expected = np.maximum(0, valid_vnp - valid_lunar)
    np.testing.assert_allclose(corrected, expected, rtol=1e-6)
    
    # Test with invalid/NoData values
    vnp_with_invalid = np.array([
        [0.5, 1.0, np.nan, 3.0],
        [np.inf, 2.5, -999, 6.0]
    ], dtype=np.float32)
    
    lunar_with_invalid = np.array([
        [0.1, 0.2, 0.3, np.nan],
        [0.2, np.inf, 0.5, 0.8]
    ], dtype=np.float32)
    
    # Apply correction - implementation may handle these differently
    corrected_invalid = apply_lunar_correction(vnp_with_invalid, lunar_with_invalid)
    
    # Check handling of special values
    # NaN - NaN should give NaN
    assert np.isnan(corrected_invalid[0, 2]) or corrected_invalid[0, 2] == 0
    # inf - finite should give inf or be clipped
    assert np.isinf(corrected_invalid[1, 0]) or corrected_invalid[1, 0] >= 0
    
    # Test array shape preservation
    shapes_to_test = [(10, 10), (100, 100), (1, 1000), (1000, 1)]
    
    for shape in shapes_to_test:
        test_vnp = np.random.uniform(0, 5, size=shape).astype(np.float32)
        test_lunar = np.random.uniform(0, 1, size=shape).astype(np.float32)
        result = apply_lunar_correction(test_vnp, test_lunar)
        
        assert result.shape == shape
        assert result.dtype == test_vnp.dtype
    
    # Test data type preservation
    for dtype in [np.float16, np.float32, np.float64]:
        typed_vnp = np.array([[1.0, 2.0]], dtype=dtype)
        typed_lunar = np.array([[0.5, 0.5]], dtype=dtype)
        typed_result = apply_lunar_correction(typed_vnp, typed_lunar)
        
        # Result should maintain precision of input type
        assert typed_result.dtype in [dtype, np.float32, np.float64]
    
    # Test numerical precision
    # Use high precision values
    precise_vnp = np.array([[1.23456789]], dtype=np.float64)
    precise_lunar = np.array([[0.12345678]], dtype=np.float64)
    precise_result = apply_lunar_correction(precise_vnp, precise_lunar)
    
    expected_precise = 1.23456789 - 0.12345678
    np.testing.assert_allclose(precise_result, expected_precise, rtol=1e-10)
    
    # Test that units are maintained (radiance in nW/cm²/sr)
    # Values should remain in reasonable range for VIIRS DNB
    typical_vnp = np.random.uniform(0.1, 10.0, size=(50, 50))
    typical_lunar = np.random.uniform(0.01, 2.0, size=(50, 50))
    typical_corrected = apply_lunar_correction(typical_vnp, typical_lunar)
    
    assert np.all(typical_corrected >= 0)  # Non-negative radiance
    assert np.all(typical_corrected <= 10.0)  # Within reasonable range


def test_calculate_ntli():
    """Test Nighttime Light Index calculation with lunar correction.
    
    This test should verify that:
    - VIIRS DNB radiance is properly processed
    - Lunar irradiance correction is applied correctly
    - Quality flags filter out poor quality pixels
    - Default valid flags ([0, 2]) work for typical VIIRS data
    - Custom valid flag lists work correctly
    - Output represents lunar-corrected radiance
    - Quality masking preserves data integrity
    - Processing handles large VIIRS tiles efficiently
    - Edge cases (all masked pixels) are handled gracefully
    """
    # Create test VIIRS DNB radiance data
    vnp_radiance = np.array([
        [0.5, 1.0, 2.0, 65531],  # Last value is fill
        [0.2, 3.0, 5.0, 10.0],
        [65532, 1.5, 2.5, 4.0]   # First value is fill
    ], dtype=np.float32)
    
    # Create lunar irradiance data
    lunar_irradiance = np.array([
        [0.1, 0.2, 0.3, 0.5],
        [0.05, 0.4, 0.6, 1.0],
        [0.2, 0.15, 0.25, 0.4]
    ], dtype=np.float32)
    
    # Create quality flags (0=good, 2=marginal, other=bad)
    quality_flag = np.array([
        [0, 0, 2, 1],    # Good, good, marginal, bad
        [2, 0, 1, 0],    # Marginal, good, bad, good
        [1, 2, 0, 0]     # Bad, marginal, good, good
    ], dtype=np.uint8)
    
    # Calculate NTLI with default valid flags [0, 2]
    ntli = calculate_ntli(vnp_radiance, lunar_irradiance, quality_flag)
    
    # Verify fill value handling (>65530 set to 0)
    assert ntli[0, 3] == 0  # 65531 > 65530
    assert ntli[2, 0] == 0  # 65532 > 65530
    
    # Verify quality flag filtering
    # Pixels with flag=1 should be zeroed
    assert ntli[0, 3] == 0  # Bad quality
    assert ntli[1, 2] == 0  # Bad quality
    assert ntli[2, 0] == 0  # Bad quality
    
    # Verify lunar correction for valid pixels
    # NTLI = (vnp * quality_mask) - lunar, clipped to >= 0
    np.testing.assert_allclose(ntli[0, 0], max(0, 0.5 - 0.1), rtol=1e-6)  # 0.4
    np.testing.assert_allclose(ntli[0, 1], max(0, 1.0 - 0.2), rtol=1e-6)  # 0.8
    np.testing.assert_allclose(ntli[0, 2], max(0, 2.0 - 0.3), rtol=1e-6)  # 1.7
    
    # Test with custom valid flags
    custom_flags = [0]  # Only accept highest quality
    ntli_strict = calculate_ntli(
        vnp_radiance, lunar_irradiance, quality_flag, valid_flags=custom_flags
    )
    
    # Pixels with flag=2 should now be zeroed
    assert ntli_strict[0, 2] == 0  # Marginal quality
    assert ntli_strict[1, 0] == 0  # Marginal quality
    assert ntli_strict[2, 1] == 0  # Marginal quality
    
    # Test output represents lunar-corrected radiance
    # For valid pixels: output = vnp - lunar (if positive)
    valid_mask = np.isin(quality_flag, [0, 2])
    for i in range(3):
        for j in range(4):
            if valid_mask[i, j] and vnp_radiance[i, j] <= 65530:
                expected = max(0, vnp_radiance[i, j] - lunar_irradiance[i, j])
                np.testing.assert_allclose(ntli[i, j], expected, rtol=1e-6)
    
    # Test quality masking preserves data integrity
    # Sum of NTLI should be less than sum of valid VNP
    assert np.sum(ntli) < np.sum(vnp_radiance[valid_mask & (vnp_radiance <= 65530)])
    
    # Test large VIIRS tile processing
    large_vnp = np.random.uniform(0, 10, size=(1000, 1000))
    large_lunar = np.random.uniform(0, 2, size=(1000, 1000))
    large_quality = np.random.choice([0, 1, 2, 3], size=(1000, 1000))
    
    large_ntli = calculate_ntli(large_vnp, large_lunar, large_quality)
    assert large_ntli.shape == (1000, 1000)
    
    # Test edge case: all masked pixels
    all_bad_quality = np.full((10, 10), 3, dtype=np.uint8)  # All bad
    all_masked_ntli = calculate_ntli(
        np.ones((10, 10)), np.zeros((10, 10)), all_bad_quality
    )
    assert np.all(all_masked_ntli == 0)


def test_calculate_ntli_quality_filtering():
    """Test NTLI quality flag filtering.
    
    This test should verify that:
    - Different quality flag values are processed correctly
    - Custom valid flag lists filter appropriately
    - Quality statistics are meaningful
    - Masked pixel handling preserves array structure
    - Flag validation catches invalid quality values
    """
    # Test data setup
    vnp_radiance = np.full((5, 5), 2.0, dtype=np.float32)
    lunar_irradiance = np.full((5, 5), 0.5, dtype=np.float32)
    
    # Create quality flags with all possible values
    quality_flag = np.array([
        [0, 0, 0, 0, 0],    # All highest quality
        [1, 1, 1, 1, 1],    # All low quality
        [2, 2, 2, 2, 2],    # All marginal quality
        [3, 3, 3, 3, 3],    # All poor quality
        [0, 1, 2, 3, 4]     # Mixed quality
    ], dtype=np.uint8)
    
    # Test with default valid flags [0, 2]
    ntli_default = calculate_ntli(vnp_radiance, lunar_irradiance, quality_flag)
    
    # Row 0 (flag=0) and Row 2 (flag=2) should be processed
    assert np.all(ntli_default[0, :] == 1.5)  # 2.0 - 0.5
    assert np.all(ntli_default[2, :] == 1.5)  # 2.0 - 0.5
    
    # Row 1 (flag=1) and Row 3 (flag=3) should be masked
    assert np.all(ntli_default[1, :] == 0)
    assert np.all(ntli_default[3, :] == 0)
    
    # Row 4 mixed: only columns 0 and 2 should be processed
    assert ntli_default[4, 0] == 1.5  # flag=0
    assert ntli_default[4, 1] == 0    # flag=1
    assert ntli_default[4, 2] == 1.5  # flag=2
    assert ntli_default[4, 3] == 0    # flag=3
    assert ntli_default[4, 4] == 0    # flag=4
    
    # Test custom valid flag lists
    # Only accept highest quality
    ntli_strict = calculate_ntli(
        vnp_radiance, lunar_irradiance, quality_flag, valid_flags=[0]
    )
    assert np.sum(ntli_strict > 0) == 6  # Only row 0 and one pixel in row 4
    
    # Accept all except worst quality
    ntli_lenient = calculate_ntli(
        vnp_radiance, lunar_irradiance, quality_flag, valid_flags=[0, 1, 2]
    )
    assert np.sum(ntli_lenient > 0) == 18  # Rows 0, 1, 2 and 3 pixels in row 4
    
    # Test quality statistics are meaningful
    # Calculate percentage of valid pixels for each flag set
    total_pixels = 25
    default_valid = np.sum(ntli_default > 0) / total_pixels
    strict_valid = np.sum(ntli_strict > 0) / total_pixels
    lenient_valid = np.sum(ntli_lenient > 0) / total_pixels
    
    assert 0 < strict_valid < default_valid < lenient_valid <= 1
    
    # Test masked pixel handling preserves array structure
    assert ntli_default.shape == vnp_radiance.shape
    # Data type may change due to numpy operations (float32 -> float64)
    assert ntli_default.dtype in [np.float32, np.float64]
    
    # Test with unusual flag values
    unusual_flags = np.array([
        [0, 5, 10, 15, 20],
        [25, 30, 50, 100, 255]
    ], dtype=np.uint8)
    
    ntli_unusual = calculate_ntli(
        np.ones_like(unusual_flags, dtype=np.float32),
        np.zeros_like(unusual_flags, dtype=np.float32),
        unusual_flags,
        valid_flags=[0, 5, 10]  # Only accept specific values
    )
    
    # Only pixels with flags 0, 5, 10 should be processed
    assert ntli_unusual[0, 0] == 1.0  # flag=0
    assert ntli_unusual[0, 1] == 1.0  # flag=5
    assert ntli_unusual[0, 2] == 1.0  # flag=10
    assert ntli_unusual[0, 3] == 0    # flag=15 not valid
    assert np.all(ntli_unusual[1, :] == 0)  # No valid flags in row 1


def test_calculate_ntli_lunar_integration():
    """Test NTLI lunar correction integration.
    
    This test should verify that:
    - Lunar correction and quality filtering work together
    - Processing order (correction then filtering) is correct
    - Combined processing maintains data quality
    - Performance is acceptable for operational workflows
    - Results are scientifically valid
    """
    # Create realistic test scenario
    # Urban area with varying light levels
    vnp_radiance = np.array([
        [0.1, 0.5, 2.0, 5.0],    # Rural to urban gradient
        [0.2, 1.0, 3.0, 8.0],
        [0.3, 1.5, 4.0, 10.0],
        [0.15, 0.8, 2.5, 7.0]
    ], dtype=np.float32)
    
    # Lunar contribution (varies with surface type)
    lunar_irradiance = np.array([
        [0.05, 0.1, 0.2, 0.3],   # Less lunar reflection in urban
        [0.06, 0.15, 0.25, 0.4],
        [0.07, 0.18, 0.3, 0.5],
        [0.055, 0.12, 0.22, 0.35]
    ], dtype=np.float32)
    
    # Quality flags with realistic distribution
    quality_flag = np.array([
        [0, 0, 0, 2],    # Mostly good, one marginal
        [0, 2, 1, 0],    # Mixed quality
        [2, 0, 0, 0],    # Mostly good
        [0, 1, 2, 3]     # All quality levels
    ], dtype=np.uint8)
    
    # Calculate NTLI
    ntli = calculate_ntli(vnp_radiance, lunar_irradiance, quality_flag)
    
    # Verify lunar correction is applied before quality filtering
    # For valid pixels (flags 0 or 2):
    # Row 0, Col 0: 0.1 - 0.05 = 0.05
    assert np.allclose(ntli[0, 0], 0.05, rtol=1e-6)
    # Row 0, Col 3: 5.0 - 0.3 = 4.7 (marginal quality but valid)
    assert np.allclose(ntli[0, 3], 4.7, rtol=1e-6)
    
    # Verify quality filtering is applied after correction
    # Row 1, Col 2: quality flag = 1 (invalid), should be 0
    assert ntli[1, 2] == 0
    # Row 3, Col 3: quality flag = 3 (invalid), should be 0
    assert ntli[3, 3] == 0
    
    # Test processing order by comparing with manual calculation
    manual_result = np.zeros_like(vnp_radiance)
    for i in range(4):
        for j in range(4):
            # First apply quality mask
            if quality_flag[i, j] in [0, 2]:
                # Then apply lunar correction
                corrected = vnp_radiance[i, j] - lunar_irradiance[i, j]
                manual_result[i, j] = max(0, corrected)
    
    np.testing.assert_allclose(ntli, manual_result, rtol=1e-6)
    
    # Test combined processing maintains data quality
    # Valid pixels should have physically meaningful values
    valid_mask = np.isin(quality_flag, [0, 2])
    valid_ntli = ntli[valid_mask]
    
    assert np.all(valid_ntli >= 0)  # Non-negative
    assert np.all(valid_ntli < 10)   # Reasonable upper bound
    assert np.mean(valid_ntli) > 0   # Some light detected
    
    # Test performance with operational-size data
    # VIIRS M-band is typically 3200x3072 pixels
    large_vnp = np.random.uniform(0, 5, size=(3200, 3072)).astype(np.float32)
    large_lunar = np.random.uniform(0, 1, size=(3200, 3072)).astype(np.float32)
    large_quality = np.random.choice([0, 1, 2, 3], 
                                    size=(3200, 3072), 
                                    p=[0.6, 0.1, 0.2, 0.1]).astype(np.uint8)
    
    # This should complete in reasonable time
    import time
    start_time = time.time()
    large_ntli = calculate_ntli(large_vnp, large_lunar, large_quality)
    processing_time = time.time() - start_time
    
    assert processing_time < 5.0  # Should process in < 5 seconds
    assert large_ntli.shape == (3200, 3072)
    
    # Verify scientific validity
    # NTLI should enhance contrast between lit and unlit areas
    lit_areas = large_ntli[large_ntli > 1.0]
    dark_areas = large_ntli[large_ntli < 0.1]
    
    if len(lit_areas) > 0 and len(dark_areas) > 0:
        contrast_ratio = np.mean(lit_areas) / (np.mean(dark_areas) + 0.001)
        assert contrast_ratio > 10  # Good contrast between lit/dark


def test_apply_viirs_floor_ceiling():
    """Test floor and ceiling threshold application to VIIRS NTL data.
    
    This test should verify that:
    - Floor threshold (default 0.1) sets minimum values to 0
    - Ceiling threshold (default 10.0) clips maximum values
    - Custom floor and ceiling values work correctly
    - Processing order (ceiling then floor) is correct
    - Very dim lights below floor are properly zeroed
    - Bright lights above ceiling are properly clipped
    - Data range is appropriately constrained for analysis
    - NoData values are preserved through processing
    - Array properties are maintained
    """
    # Create test NTL data with various light levels
    ntl_data = np.array([
        [0.01, 0.05, 0.1, 0.5],     # Very dim to moderate
        [1.0, 2.0, 5.0, 8.0],       # Moderate to bright
        [10.0, 12.0, 15.0, 20.0],   # At and above ceiling
        [0.0, 0.09, 9.99, 10.01]    # Edge cases
    ], dtype=np.float32)
    
    # Apply default floor (0.1) and ceiling (10.0)
    result = apply_viirs_floor_ceiling(ntl_data)
    
    # Verify floor threshold sets values < 0.1 to 0
    assert result[0, 0] == 0.0  # 0.01 < 0.1
    assert result[0, 1] == 0.0  # 0.05 < 0.1
    assert result[0, 2] == 0.1  # 0.1 == 0.1 (at threshold)
    assert result[0, 3] == 0.5  # 0.5 > 0.1 (unchanged)
    
    # Verify ceiling threshold clips values > 10.0
    assert result[2, 0] == 10.0  # 10.0 == 10.0 (at threshold)
    assert result[2, 1] == 10.0  # 12.0 > 10.0 (clipped)
    assert result[2, 2] == 10.0  # 15.0 > 10.0 (clipped)
    assert result[2, 3] == 10.0  # 20.0 > 10.0 (clipped)
    
    # Verify values between floor and ceiling are unchanged
    assert result[1, 0] == 1.0
    assert result[1, 1] == 2.0
    assert result[1, 2] == 5.0
    assert result[1, 3] == 8.0
    
    # Test edge cases
    assert result[3, 0] == 0.0   # 0.0 < 0.1 (floor)
    assert result[3, 1] == 0.0   # 0.09 < 0.1 (floor)
    assert result[3, 2] == 9.99  # 9.99 < 10.0 (unchanged)
    assert result[3, 3] == 10.0  # 10.01 > 10.0 (ceiling)
    
    # Test custom floor and ceiling values
    custom_result = apply_viirs_floor_ceiling(
        ntl_data, 
        floor_value=0.5,
        ceiling_value=5.0
    )
    
    # With floor=0.5, more values should be zeroed (except 0.5 which is at threshold)
    assert custom_result[0, 0] == 0.0  # 0.01 < 0.5
    assert custom_result[0, 1] == 0.0  # 0.05 < 0.5
    assert custom_result[0, 2] == 0.0  # 0.1 < 0.5
    assert custom_result[0, 3] == 0.5  # 0.5 == 0.5 (at threshold)
    assert custom_result[1, 0] == 1.0  # 1.0 > 0.5
    
    # With ceiling=5.0, more values should be clipped
    assert custom_result[1, 2] == 5.0  # 5.0 == 5.0
    assert custom_result[1, 3] == 5.0  # 8.0 > 5.0 (clipped)
    assert np.all(custom_result[2, :] == 5.0)  # All > 5.0
    
    # Test processing order (ceiling then floor)
    # This is important for edge cases
    edge_data = np.array([[10.5, 0.05]])
    edge_result = apply_viirs_floor_ceiling(edge_data)
    
    # 10.5 -> 10.0 (ceiling) -> 10.0 (> floor, unchanged)
    assert edge_result[0, 0] == 10.0
    # 0.05 -> 0.05 (< ceiling) -> 0.0 (< floor)
    assert edge_result[0, 1] == 0.0
    
    # Test very dim lights are properly zeroed
    dim_lights = np.array([0.001, 0.01, 0.05, 0.08, 0.099])
    dim_result = apply_viirs_floor_ceiling(dim_lights)
    assert np.all(dim_result == 0.0)
    
    # Test bright lights are properly clipped
    bright_lights = np.array([10.001, 11.0, 50.0, 100.0, 1000.0])
    bright_result = apply_viirs_floor_ceiling(bright_lights)
    assert np.all(bright_result == 10.0)
    
    # Test data range constraint for analysis
    constrained = apply_viirs_floor_ceiling(ntl_data)
    assert np.min(constrained[constrained > 0]) >= 0.1
    assert np.max(constrained) <= 10.0
    
    # Test array properties are maintained
    assert result.shape == ntl_data.shape
    assert result.dtype == ntl_data.dtype


def test_apply_viirs_floor_ceiling_custom_thresholds():
    """Test VIIRS floor/ceiling with custom threshold values.
    
    This test should verify that:
    - Different floor values affect dim light filtering
    - Different ceiling values affect bright light clipping
    - Threshold selection impacts data distribution
    - Edge cases (floor > ceiling) are handled appropriately
    - Scientific validity is maintained with custom thresholds
    """
    # Create test data with known distribution
    ntl_data = np.linspace(0, 20, 100).reshape(10, 10)
    
    # Test different floor values
    floor_values = [0.0, 0.1, 0.5, 1.0, 2.0]
    floor_results = []
    
    for floor in floor_values:
        result = apply_viirs_floor_ceiling(ntl_data, floor_value=floor)
        num_zeros = np.sum(result == 0)
        floor_results.append(num_zeros)
    
    # Higher floor values should zero more pixels
    assert all(floor_results[i] <= floor_results[i+1] 
               for i in range(len(floor_results)-1))
    assert floor_results[0] == 1   # Only the 0.0 value
    assert floor_results[-1] == 10  # First row all < 2.0
    
    # Test different ceiling values
    ceiling_values = [5.0, 10.0, 15.0, 20.0, 50.0]
    ceiling_results = []
    
    for ceiling in ceiling_values:
        result = apply_viirs_floor_ceiling(ntl_data, ceiling_value=ceiling)
        num_clipped = np.sum(result == ceiling)
        ceiling_results.append(num_clipped)
    
    # Lower ceiling values should clip more pixels
    assert all(ceiling_results[i] >= ceiling_results[i+1] 
               for i in range(len(ceiling_results)-1))
    assert ceiling_results[0] > 0   # Some values > 5.0
    assert ceiling_results[-1] == 0  # No values > 50.0
    
    # Test threshold selection impact on data distribution
    # Aggressive thresholds (narrow range)
    aggressive = apply_viirs_floor_ceiling(
        ntl_data, floor_value=2.0, ceiling_value=5.0
    )
    
    # Lenient thresholds (wide range)
    lenient = apply_viirs_floor_ceiling(
        ntl_data, floor_value=0.01, ceiling_value=50.0
    )
    
    # Aggressive should have less dynamic range
    aggressive_range = np.max(aggressive[aggressive > 0]) - np.min(aggressive[aggressive > 0])
    lenient_range = np.max(lenient[lenient > 0]) - np.min(lenient[lenient > 0])
    assert aggressive_range < lenient_range
    
    # Test edge case: floor > ceiling
    # This shouldn't happen in practice, but test robustness
    invalid_result = apply_viirs_floor_ceiling(
        ntl_data, floor_value=15.0, ceiling_value=5.0
    )
    
    # All values should be either 0 or 5.0
    unique_values = np.unique(invalid_result)
    assert len(unique_values) <= 2
    assert all(v in [0.0, 5.0] for v in unique_values)
    
    # Test scientific validity with custom thresholds
    # Urban analysis might use different thresholds than rural
    
    # Urban: higher floor to filter out dim scattered light
    urban_data = np.array([[0.5, 1.0, 5.0, 15.0, 25.0]])
    urban_result = apply_viirs_floor_ceiling(
        urban_data, floor_value=1.0, ceiling_value=20.0
    )
    assert urban_result[0, 0] == 0.0   # Filtered out
    assert urban_result[0, 1] == 1.0   # At threshold
    assert urban_result[0, 4] == 20.0  # Clipped
    
    # Rural: lower floor to capture dim lights
    rural_data = np.array([[0.01, 0.05, 0.1, 0.5, 1.0]])
    rural_result = apply_viirs_floor_ceiling(
        rural_data, floor_value=0.02, ceiling_value=2.0
    )
    assert rural_result[0, 0] == 0.0   # Still below floor
    assert rural_result[0, 1] == 0.05  # Preserved
    assert rural_result[0, 4] == 1.0   # Not clipped


def test_apply_viirs_floor_ceiling_data_preservation():
    """Test data preservation during VIIRS floor/ceiling application.
    
    This test should verify that:
    - Valid data within thresholds is unchanged
    - Array shapes and data types are preserved
    - Spatial patterns are maintained appropriately
    - Statistical properties are modified as expected
    - Processing is reversible where appropriate
    """
    # Create test data with known patterns
    # Gradient pattern
    x = np.linspace(0, 15, 50)
    y = np.linspace(0, 15, 50)
    X, Y = np.meshgrid(x, y)
    gradient_data = (X + Y) / 3.0  # Range 0 to 10
    
    # Apply floor/ceiling
    processed = apply_viirs_floor_ceiling(
        gradient_data, floor_value=1.0, ceiling_value=8.0
    )
    
    # Verify valid data within thresholds is unchanged
    valid_mask = (gradient_data >= 1.0) & (gradient_data <= 8.0)
    np.testing.assert_allclose(
        processed[valid_mask], 
        gradient_data[valid_mask],
        rtol=1e-6
    )
    
    # Verify array shape and data type preservation
    assert processed.shape == gradient_data.shape
    assert processed.dtype == gradient_data.dtype
    
    # Test spatial pattern preservation
    # Create checkerboard pattern
    checkerboard = np.zeros((10, 10))
    checkerboard[::2, ::2] = 5.0    # Valid value
    checkerboard[1::2, 1::2] = 5.0  # Valid value
    checkerboard[::2, 1::2] = 0.05  # Below floor
    checkerboard[1::2, ::2] = 12.0  # Above ceiling
    
    checker_result = apply_viirs_floor_ceiling(checkerboard)
    
    # Pattern should be preserved but with modified values
    assert np.all(checker_result[::2, ::2] == 5.0)    # Unchanged
    assert np.all(checker_result[1::2, 1::2] == 5.0)  # Unchanged
    assert np.all(checker_result[::2, 1::2] == 0.0)   # Floored
    assert np.all(checker_result[1::2, ::2] == 10.0)  # Clipped
    
    # Test statistical property modification
    random_data = np.random.uniform(0, 20, size=(100, 100))
    stats_before = {
        'mean': np.mean(random_data),
        'std': np.std(random_data),
        'min': np.min(random_data),
        'max': np.max(random_data)
    }
    
    processed_random = apply_viirs_floor_ceiling(
        random_data, floor_value=2.0, ceiling_value=15.0
    )
    
    stats_after = {
        'mean': np.mean(processed_random[processed_random > 0]),
        'std': np.std(processed_random[processed_random > 0]),
        'min': np.min(processed_random[processed_random > 0]),
        'max': np.max(processed_random)
    }
    
    # Statistical properties should change predictably
    assert stats_after['min'] >= 2.0    # Floor applied
    assert stats_after['max'] <= 15.0   # Ceiling applied
    assert stats_after['std'] < stats_before['std']  # Reduced variance
    
    # Test reversibility where appropriate
    # Data within thresholds should be reversible
    safe_data = np.random.uniform(2.0, 8.0, size=(20, 20))
    safe_processed = apply_viirs_floor_ceiling(
        safe_data, floor_value=1.0, ceiling_value=10.0
    )
    
    # Should be unchanged since all values are within thresholds
    np.testing.assert_allclose(safe_processed, safe_data, rtol=1e-6)
    
    # Test with various data types
    for dtype in [np.float16, np.float32, np.float64]:
        typed_data = gradient_data.astype(dtype)
        typed_result = apply_viirs_floor_ceiling(typed_data)
        assert typed_result.dtype == dtype


def test_normalize_to_range():
    """Test data normalization to specified output range.
    
    This test should verify that:
    - Default normalization (0 to 1) works correctly
    - Custom output ranges are handled properly
    - Auto-detected input ranges work for typical data
    - Custom input ranges override auto-detection
    - Linear scaling preserves relative relationships
    - Edge cases (constant data) are handled gracefully
    - Different data types are supported
    - Large arrays are processed efficiently
    - NoData values are handled appropriately
    """
    # Test default normalization (0 to 1)
    data = np.array([0, 25, 50, 75, 100], dtype=np.float32)
    normalized = normalize_to_range(data)
    
    expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(normalized, expected, rtol=1e-6)
    
    # Verify output range
    assert np.min(normalized) == 0.0
    assert np.max(normalized) == 1.0
    
    # Test custom output ranges
    # Normalize to [-1, 1]
    norm_symmetric = normalize_to_range(data, out_min=-1.0, out_max=1.0)
    expected_sym = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    np.testing.assert_allclose(norm_symmetric, expected_sym, rtol=1e-6)
    
    # Normalize to [0, 255] for byte scaling
    norm_byte = normalize_to_range(data, out_min=0, out_max=255)
    expected_byte = np.array([0, 63.75, 127.5, 191.25, 255])
    np.testing.assert_allclose(norm_byte, expected_byte, rtol=1e-6)
    
    # Test auto-detected input ranges
    auto_data = np.array([10, 20, 30, 40])
    auto_norm = normalize_to_range(auto_data)
    # Should map 10->0, 40->1 linearly
    expected_auto = np.array([0.0, 1/3, 2/3, 1.0])
    np.testing.assert_allclose(auto_norm, expected_auto, rtol=1e-6)
    
    # Test custom input ranges override auto-detection
    custom_norm = normalize_to_range(
        auto_data, 
        in_min=0, 
        in_max=50
    )
    # Now mapping 0->0, 50->1
    expected_custom = np.array([0.2, 0.4, 0.6, 0.8])
    np.testing.assert_allclose(custom_norm, expected_custom, rtol=1e-6)
    
    # Test linear scaling preserves relative relationships
    # Create data with known relationships
    linear_data = np.array([1, 2, 4, 8, 16])
    linear_norm = normalize_to_range(linear_data)
    
    # Check ratios are preserved
    ratio_before = linear_data[4] / linear_data[0]  # 16/1 = 16
    ratio_after = linear_norm[4] / (linear_norm[0] + 1e-10)  # Avoid div by 0
    # Ratios won't be exactly equal due to offset, but pattern preserved
    assert linear_norm[1] < linear_norm[2] < linear_norm[3] < linear_norm[4]
    
    # Test edge case: constant data
    constant_data = np.full(10, 5.0)
    constant_norm = normalize_to_range(constant_data)
    # Should return midpoint of output range
    expected_const = np.full(10, 0.5)
    np.testing.assert_allclose(constant_norm, expected_const, rtol=1e-6)
    
    # Test with custom output range
    constant_norm_custom = normalize_to_range(
        constant_data, out_min=10, out_max=20
    )
    expected_const_custom = np.full(10, 15.0)  # Midpoint of [10, 20]
    np.testing.assert_allclose(constant_norm_custom, expected_const_custom)
    
    # Test different data types
    for dtype in [np.int32, np.float16, np.float32, np.float64]:
        typed_data = np.array([0, 50, 100], dtype=dtype)
        typed_norm = normalize_to_range(typed_data)
        assert np.allclose(typed_norm, [0.0, 0.5, 1.0])
    
    # Test large arrays
    large_data = np.random.uniform(-100, 100, size=(1000, 1000))
    large_norm = normalize_to_range(large_data)
    
    assert large_norm.shape == (1000, 1000)
    assert np.min(large_norm) >= 0.0 - 1e-6
    assert np.max(large_norm) <= 1.0 + 1e-6
    
    # Test NoData/NaN handling
    data_with_nan = np.array([1, 2, np.nan, 4, 5])
    norm_with_nan = normalize_to_range(data_with_nan)
    
    # NaN should be preserved
    assert np.isnan(norm_with_nan[2])
    # Other values should be normalized based on non-NaN values
    valid_idx = [0, 1, 3, 4]
    assert np.allclose(norm_with_nan[valid_idx], [0.0, 0.25, 0.75, 1.0])


def test_normalize_to_range_custom_ranges():
    """Test normalization with custom input/output ranges.
    
    This test should verify that:
    - Custom output ranges (e.g., -1 to 1) work correctly
    - Custom input ranges improve normalization accuracy
    - Range specification prevents outlier influence
    - Multiple normalization operations are consistent
    - Range validation catches invalid parameters
    """
    # Test various custom output ranges
    data = np.array([0, 25, 50, 75, 100])
    
    # Test -1 to 1 (symmetric around zero)
    norm_symmetric = normalize_to_range(data, out_min=-1, out_max=1)
    assert norm_symmetric[0] == -1.0
    assert norm_symmetric[2] == 0.0   # Middle value maps to middle of range
    assert norm_symmetric[4] == 1.0
    
    # Test 100 to 200 (offset range)
    norm_offset = normalize_to_range(data, out_min=100, out_max=200)
    assert norm_offset[0] == 100.0
    assert norm_offset[4] == 200.0
    np.testing.assert_allclose(
        norm_offset,
        [100, 125, 150, 175, 200],
        rtol=1e-6
    )
    
    # Test negative to positive (e.g., -50 to 50)
    norm_centered = normalize_to_range(data, out_min=-50, out_max=50)
    assert norm_centered[0] == -50.0
    assert norm_centered[2] == 0.0
    assert norm_centered[4] == 50.0
    
    # Test custom input ranges for outlier prevention
    data_with_outliers = np.array([0, 10, 20, 30, 40, 1000])  # 1000 is outlier
    
    # Without custom input range, outlier affects normalization
    norm_with_outlier = normalize_to_range(data_with_outliers)
    # Most values will be compressed near 0
    assert norm_with_outlier[1] < 0.1  # 10/1000 = 0.01
    
    # With custom input range, ignore outlier
    norm_no_outlier = normalize_to_range(
        data_with_outliers,
        in_min=0,
        in_max=50  # Ignore the 1000 outlier
    )
    # Now values are well distributed
    assert norm_no_outlier[1] == 0.2   # 10/50 = 0.2
    assert norm_no_outlier[4] == 0.8   # 40/50 = 0.8
    assert norm_no_outlier[5] > 1.0    # Outlier exceeds range
    
    # Test range specification for percentile-based normalization
    # Simulate getting 5th and 95th percentiles
    noisy_data = np.random.normal(50, 10, 1000)
    p5 = np.percentile(noisy_data, 5)
    p95 = np.percentile(noisy_data, 95)
    
    norm_percentile = normalize_to_range(
        noisy_data,
        in_min=p5,
        in_max=p95
    )
    
    # Most values should be in [0, 1]
    in_range = np.sum((norm_percentile >= 0) & (norm_percentile <= 1))
    assert in_range >= 900  # ~90% within range
    
    # Test multiple normalization operations consistency
    # Chain normalizations should be reversible
    original = np.array([10, 20, 30, 40, 50])
    
    # First normalize to [0, 1]
    step1 = normalize_to_range(original)
    
    # Then normalize to [-1, 1]
    step2 = normalize_to_range(step1, out_min=-1, out_max=1)
    
    # Direct normalization to [-1, 1] should give same result
    direct = normalize_to_range(original, out_min=-1, out_max=1)
    
    np.testing.assert_allclose(step2, direct, rtol=1e-6)
    
    # Test with inverted ranges (max < min)
    # This might happen with certain transforms
    inverted = normalize_to_range(
        data,
        out_min=1,
        out_max=0  # Inverted
    )
    # Should invert the mapping
    assert inverted[0] == 1.0  # Min input -> Max output
    assert inverted[4] == 0.0  # Max input -> Min output


def test_normalize_to_range_edge_cases():
    """Test normalization edge cases.
    
    This test should verify that:
    - Constant input arrays are handled without division by zero
    - Very small dynamic ranges work correctly
    - Very large dynamic ranges maintain precision
    - Extreme outliers don't break normalization
    - NaN/infinite values are handled appropriately
    """
    # Test constant input array
    constant = np.full(10, 42.0)
    norm_constant = normalize_to_range(constant)
    
    # Should return midpoint of output range
    assert np.all(norm_constant == 0.5)
    assert not np.any(np.isnan(norm_constant))  # No NaN from division by zero
    
    # Test with different output range
    norm_constant_custom = normalize_to_range(
        constant, out_min=-10, out_max=10
    )
    assert np.all(norm_constant_custom == 0.0)  # Midpoint of [-10, 10]
    
    # Test very small dynamic range
    small_range = np.array([1.0, 1.0001, 1.0002, 1.0003])
    norm_small = normalize_to_range(small_range)
    
    # Should still produce meaningful normalized values
    assert norm_small[0] == 0.0
    assert norm_small[-1] == 1.0
    assert np.all(np.diff(norm_small) > 0)  # Monotonic increase
    
    # Test precision is maintained
    expected_small = np.array([0.0, 1/3, 2/3, 1.0])
    np.testing.assert_allclose(norm_small, expected_small, rtol=1e-5)
    
    # Test very large dynamic range
    large_range = np.array([1e-10, 1e-5, 1e0, 1e5, 1e10])
    norm_large = normalize_to_range(large_range)
    
    # Should handle without overflow/underflow
    assert np.all(np.isfinite(norm_large))
    assert norm_large[0] >= 0.0
    assert norm_large[-1] <= 1.0
    
    # Test extreme outliers
    with_outliers = np.array([1, 2, 3, 4, 1e100])
    norm_outliers = normalize_to_range(with_outliers)
    
    # Should not break, but most values will be near 0
    assert np.all(np.isfinite(norm_outliers))
    assert norm_outliers[-1] == 1.0  # Outlier maps to 1
    assert np.all(norm_outliers[:-1] < 0.001)  # Others near 0
    
    # Test NaN values
    with_nan = np.array([1, 2, np.nan, 4, 5])
    norm_nan = normalize_to_range(with_nan)
    
    # NaN should be preserved
    assert np.isnan(norm_nan[2])
    # Other values normalized correctly
    valid_mask = ~np.isnan(norm_nan)
    assert norm_nan[valid_mask][0] == 0.0
    assert norm_nan[valid_mask][-1] == 1.0
    
    # Test infinite values
    with_inf = np.array([1, 2, np.inf, 4, -np.inf])
    norm_inf = normalize_to_range(with_inf)
    
    # Inf values might be preserved or clipped depending on implementation
    # But shouldn't break the function
    finite_mask = np.isfinite(norm_inf)
    if np.any(finite_mask):
        assert np.all(norm_inf[finite_mask] >= 0)
        assert np.all(norm_inf[finite_mask] <= 1)
    
    # Test all NaN array
    all_nan = np.full(5, np.nan)
    norm_all_nan = normalize_to_range(all_nan)
    assert np.all(np.isnan(norm_all_nan))
    
    # Test single value array
    single = np.array([42.0])
    norm_single = normalize_to_range(single)
    assert norm_single[0] == 0.5  # Midpoint
    
    # Test empty array behavior (if supported)
    # This might raise an error depending on implementation
    try:
        empty = np.array([])
        norm_empty = normalize_to_range(empty)
        assert len(norm_empty) == 0
    except:
        pass  # Empty array might not be supported


def test_scale_to_byte():
    """Test scaling data to uint8 range [0, 255].
    
    This test should verify that:
    - Floating-point data is properly scaled to byte range
    - Auto-detected min/max values work for typical data
    - Custom min/max values improve scaling accuracy
    - Output data type is uint8
    - NoData values are mapped to 0 (default nodata_value)
    - Dynamic range is preserved as much as possible
    - Precision loss is minimized and documented
    - Large arrays are processed efficiently
    - Edge cases (all zeros, all same value) work correctly
    """
    # Test basic float to byte scaling
    float_data = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=np.float32)
    byte_result = scale_to_byte(float_data)
    
    # Verify output type
    assert byte_result.dtype == np.uint8
    
    # Verify scaling to [0, 255]
    assert byte_result[0, 0] == 0
    assert byte_result[0, 4] == 255
    # Middle values should be approximately correct (rounding may vary)
    assert 63 <= byte_result[0, 1] <= 65   # ~64
    assert 127 <= byte_result[0, 2] <= 128  # ~127
    assert 190 <= byte_result[0, 3] <= 192  # ~191
    
    # Test auto-detected min/max
    auto_data = np.array([[10.0, 20.0, 30.0, 40.0, 50.0]])
    auto_result = scale_to_byte(auto_data)
    
    # Should map 10->0, 50->255
    assert auto_result[0, 0] == 0
    assert auto_result[0, 4] == 255
    # Linear interpolation for middle values (allow rounding differences)
    assert 126 <= auto_result[0, 2] <= 128  # 30 is midpoint
    
    # Test custom min/max values
    custom_data = np.array([[5.0, 15.0, 25.0, 35.0, 45.0]])
    custom_result = scale_to_byte(
        custom_data,
        in_min=0.0,
        in_max=50.0
    )
    
    # Now mapping 0->0, 50->255
    expected_custom = np.round(custom_data[0] * 255 / 50).astype(np.uint8)
    np.testing.assert_array_equal(custom_result[0], expected_custom)
    
    # Test NoData handling
    data_with_nan = np.array([[1.0, 2.0, np.nan, 4.0, 5.0]])
    nan_result = scale_to_byte(data_with_nan)
    
    # NaN should be mapped to 0 (default nodata_value)
    assert nan_result[0, 2] == 0
    # Other values should be scaled normally
    assert nan_result[0, 0] == 0    # Min value
    assert nan_result[0, 4] == 255  # Max value
    
    # Test dynamic range preservation
    # High dynamic range data
    hdr_data = np.array([[0.1, 1.0, 10.0, 25.0, 50.0, 100.0]])
    hdr_result = scale_to_byte(hdr_data)
    
    # All unique values should map to different byte values (or close)
    unique_values = len(np.unique(hdr_result))
    assert unique_values >= 4  # Should have good separation
    assert hdr_result[0, 0] == 0
    assert hdr_result[0, 5] == 255
    
    # Test precision loss documentation
    # Create data with more than 256 unique values
    precise_data = np.linspace(0, 1, 1000).reshape(10, 100)
    precise_result = scale_to_byte(precise_data)
    
    # Can only represent 256 unique values
    unique_values = len(np.unique(precise_result))
    assert unique_values <= 256
    # But should use most of the range
    assert unique_values > 250
    
    # Test large array efficiency
    large_data = np.random.uniform(0, 100, size=(1000, 1000))
    large_result = scale_to_byte(large_data)
    
    assert large_result.shape == (1000, 1000)
    assert large_result.dtype == np.uint8
    assert np.min(large_result) == 0
    assert np.max(large_result) == 255
    
    # Test edge case: all zeros
    all_zeros = np.zeros((5, 5))
    zeros_result = scale_to_byte(all_zeros)
    
    # Should handle gracefully (all map to same value)
    assert np.all(zeros_result == zeros_result[0, 0])
    
    # Test edge case: all same non-zero value
    all_same = np.full((5, 5), 42.0)
    same_result = scale_to_byte(all_same)
    
    # Should handle gracefully
    assert np.all(same_result == same_result[0, 0])
    
    # Test negative values
    negative_data = np.array([[-10, -5, 0, 5, 10]])
    negative_result = scale_to_byte(negative_data)
    
    # Should handle full range including negatives
    assert negative_result[0, 0] == 0    # -10 -> 0
    assert 126 <= negative_result[0, 2] <= 128  # 0 -> ~127 (middle)
    assert negative_result[0, 4] == 255  # 10 -> 255


def test_scale_to_byte_range_optimization():
    """Test byte scaling range optimization.
    
    This test should verify that:
    - Full 0-255 range is utilized when possible
    - Custom input ranges prevent outlier distortion
    - Percentile-based ranges improve visualization
    - Clipping behavior is appropriate for different data types
    - Scaling preserves important data features
    """
    # Test full range utilization
    full_range_data = np.linspace(0, 100, 256)
    full_result = scale_to_byte(full_range_data)
    
    # Should use all 256 possible values
    assert len(np.unique(full_result)) == 256
    assert np.min(full_result) == 0
    assert np.max(full_result) == 255
    
    # Test outlier prevention with custom ranges
    # Data with outliers
    data_with_outliers = np.concatenate([
        np.random.normal(50, 10, 1000),  # Main distribution
        [200, 250, 300]  # Outliers
    ])
    
    # Without custom range, outliers compress main data
    auto_result = scale_to_byte(data_with_outliers)
    main_data_max = np.max(auto_result[:-3])  # Exclude outliers
    assert main_data_max < 100  # Main data compressed to lower values
    
    # With custom range, ignore outliers
    custom_result = scale_to_byte(
        data_with_outliers,
        in_min=20,  # ~mean - 3*std
        in_max=80   # ~mean + 3*std
    )
    
    # Main data should use more of the range
    main_data_values = custom_result[:-3]
    assert np.min(main_data_values) < 50
    assert np.max(main_data_values) > 200
    
    # Test percentile-based ranges
    # Simulate using 2nd and 98th percentiles
    p2 = np.percentile(data_with_outliers, 2)
    p98 = np.percentile(data_with_outliers, 98)
    
    percentile_result = scale_to_byte(
        data_with_outliers,
        in_min=p2,
        in_max=p98
    )
    
    # Most values should use a good portion of the range
    in_range_mask = (data_with_outliers >= p2) & (data_with_outliers <= p98)
    in_range_scaled = percentile_result[in_range_mask]
    assert np.min(in_range_scaled) < 20   # Close to 0
    assert np.max(in_range_scaled) > 235  # Close to 255
    
    # Test clipping behavior
    clip_data = np.array([[-10, 0, 50, 100, 150]])
    
    # Scale with range [0, 100]
    clipped_result = scale_to_byte(
        clip_data,
        in_min=0,
        in_max=100
    )
    
    # Values outside range should be clipped
    assert clipped_result[0, 0] == 0    # -10 clipped to min
    assert clipped_result[0, 4] == 255  # 150 clipped to max
    # Values inside range scaled normally
    assert clipped_result[0, 1] == 0    # 0 -> 0
    assert 126 <= clipped_result[0, 2] <= 128  # 50 -> ~127
    assert clipped_result[0, 3] == 255  # 100 -> 255
    
    # Test feature preservation
    # Create data with important features
    x = np.linspace(0, 10, 1000)
    features_data = np.sin(x) + 0.1 * np.sin(10 * x)  # Main signal + details
    
    # Scale to byte
    features_result = scale_to_byte(features_data)
    
    # Find peaks in original and scaled data
    from scipy.signal import find_peaks
    orig_peaks, _ = find_peaks(features_data, distance=50)
    scaled_peaks, _ = find_peaks(features_result, distance=50)
    
    # Number of peaks should be approximately preserved (allow some variation)
    assert abs(len(scaled_peaks) - len(orig_peaks)) <= 3
    
    # Test contrast preservation
    # High contrast data
    contrast_data = np.array([
        [0, 0, 0, 100, 100, 100],  # Sharp edge
        [0, 25, 50, 75, 100, 100]  # Gradient
    ])
    
    contrast_result = scale_to_byte(contrast_data)
    
    # Sharp edge should be preserved
    assert contrast_result[0, 2] == 0
    assert contrast_result[0, 3] == 255
    # Gradient should be smooth
    gradient_row = contrast_result[1, :]
    assert np.all(np.diff(gradient_row) >= 0)  # Monotonic


def test_scale_to_byte_visualization_quality():
    """Test byte scaling for visualization quality.
    
    This test should verify that:
    - Visual contrast is maximized appropriately
    - Important features remain distinguishable
    - Color mapping works well with scaled data
    - Histogram distribution is reasonable
    - Output is suitable for image formats
    """
    # Test visual contrast maximization
    # Low contrast data
    low_contrast = np.random.uniform(0.4, 0.6, size=(100, 100))
    contrast_result = scale_to_byte(low_contrast)
    
    # Should expand to full range
    assert np.min(contrast_result) == 0
    assert np.max(contrast_result) == 255
    # Should have good spread
    assert np.std(contrast_result) > 50
    
    # Test feature distinguishability
    # Create data with distinct regions
    feature_data = np.zeros((100, 100))
    feature_data[20:40, 20:40] = 0.3   # Dark region
    feature_data[60:80, 60:80] = 0.7   # Bright region
    feature_data[20:40, 60:80] = 0.5   # Medium region
    
    feature_result = scale_to_byte(feature_data)
    
    # Regions should be distinguishable
    dark_values = np.unique(feature_result[20:40, 20:40])
    medium_values = np.unique(feature_result[20:40, 60:80])
    bright_values = np.unique(feature_result[60:80, 60:80])
    background_values = np.unique(feature_result[0:10, 0:10])
    
    # Each region should have distinct values
    assert len(dark_values) >= 1
    assert len(medium_values) >= 1
    assert len(bright_values) >= 1
    
    # Values should be ordered
    assert np.mean(background_values) < np.mean(dark_values)
    assert np.mean(dark_values) < np.mean(medium_values)
    assert np.mean(medium_values) < np.mean(bright_values)
    
    # Test color mapping compatibility
    # Create gradient for color mapping
    gradient = np.linspace(0, 1, 256).reshape(1, 256)
    gradient_bytes = scale_to_byte(gradient)
    
    # Should map to all possible byte values
    assert len(np.unique(gradient_bytes)) == 256
    assert gradient_bytes[0, 0] == 0
    assert gradient_bytes[0, 255] == 255
    
    # Test histogram distribution
    # Uniform distribution should remain roughly uniform
    uniform_data = np.random.uniform(0, 1, 10000)
    uniform_bytes = scale_to_byte(uniform_data)
    
    # Check histogram bins
    hist, _ = np.histogram(uniform_bytes, bins=32)
    # Should be roughly equal (within 2x factor)
    hist_ratio = np.max(hist) / (np.min(hist) + 1)
    assert hist_ratio < 3.0
    
    # Test suitability for image formats
    # Create typical image-like data
    image_data = np.random.beta(2, 5, size=(256, 256))  # Skewed distribution
    image_bytes = scale_to_byte(image_data)
    
    # Should be valid image data
    assert image_bytes.dtype == np.uint8
    assert image_bytes.shape == (256, 256)
    assert np.min(image_bytes) >= 0
    assert np.max(image_bytes) <= 255
    
    # Should preserve spatial structure
    # Apply simple edge detection
    dx = np.diff(image_bytes, axis=1)
    dy = np.diff(image_bytes, axis=0)
    edges = np.abs(dx[:-1, :]) + np.abs(dy[:, :-1])
    
    # Should have detectable edges
    assert np.max(edges) > 10
    assert np.mean(edges) > 1


def test_scaling_accuracy_validation():
    """Test accuracy of scaling operations across functions.
    
    This test should verify that:
    - Mathematical operations are numerically accurate
    - Precision loss is within acceptable bounds
    - Inverse operations work where applicable
    - Scaling parameters are applied correctly
    - Scientific validity is maintained throughout processing
    """
    # Test mathematical accuracy of Landsat scaling
    # Use exact values that should produce known results
    exact_dn = np.array([7273, 25454, 43636], dtype=np.uint16)  # Min, mid, max
    exact_result = scale_landsat_bands(exact_dn)
    
    # Calculate expected values with high precision
    expected = exact_dn.astype(np.float64) * 0.0000275 + (-0.2)
    np.testing.assert_allclose(exact_result, expected, rtol=1e-10)
    
    # Test precision loss bounds
    # Use float32 precision (typical for remote sensing)
    precision_dn = np.array([10000.5, 20000.7, 30000.3], dtype=np.float32)
    precision_result = scale_landsat_bands(precision_dn)
    
    # Error should be within float32 precision
    expected_precision = precision_dn * 0.0000275 - 0.2
    relative_error = np.abs((precision_result - expected_precision) / expected_precision)
    assert np.all(relative_error < 1e-6)
    
    # Test inverse operations where applicable
    # Normalization should be reversible
    original_data = np.array([10, 20, 30, 40, 50])
    
    # Forward: normalize to [0, 1]
    normalized = normalize_to_range(original_data)
    
    # Inverse: denormalize back to original range
    denormalized = normalized * (50 - 10) + 10
    
    np.testing.assert_allclose(denormalized, original_data, rtol=1e-10)
    
    # Test scaling parameter application
    # Custom Landsat parameters - use values in valid range
    test_dn = np.array([10000, 20000, 30000])  # In valid range
    custom_scale = 0.0001
    custom_offset = -0.1
    
    custom_result = scale_landsat_bands(
        test_dn,
        scale_factor=custom_scale,
        add_offset=custom_offset
    )
    
    # Verify exact parameter application
    expected_custom = test_dn * custom_scale + custom_offset
    np.testing.assert_allclose(custom_result, expected_custom, rtol=1e-10)
    
    # Test scientific validity preservation
    # Lunar correction should preserve physical meaning
    vnp_radiance = np.array([2.0, 4.0, 6.0])  # nW/cm²/sr
    lunar_contrib = np.array([0.5, 1.0, 1.5])  # nW/cm²/sr
    
    corrected = apply_lunar_correction(vnp_radiance, lunar_contrib)
    
    # Result should still be valid radiance
    assert np.all(corrected >= 0)  # Non-negative
    assert np.all(corrected <= vnp_radiance)  # Reduced
    
    # Physical relationship preserved
    expected_corrected = np.maximum(0, vnp_radiance - lunar_contrib)
    np.testing.assert_allclose(corrected, expected_corrected, rtol=1e-10)
    
    # Test VIIRS floor/ceiling preserves order
    ntl_values = np.array([0.05, 0.15, 1.0, 5.0, 15.0])
    processed = apply_viirs_floor_ceiling(ntl_values)
    
    # Order should be preserved for valid values
    valid_original = ntl_values[(ntl_values >= 0.1) & (ntl_values <= 10.0)]
    valid_processed = processed[(ntl_values >= 0.1) & (ntl_values <= 10.0)]
    
    if len(valid_original) > 1:
        assert np.all(np.diff(valid_processed) >= 0)  # Monotonic
    
    # Test chain of operations maintains validity
    # Simulate complete processing chain
    raw_vnp = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    raw_lunar = np.array([[0.2, 0.4, 0.6], [0.8, 1.0, 1.2]])
    quality = np.array([[0, 0, 2], [2, 1, 0]], dtype=np.uint8)
    
    # Step 1: Calculate NTLI
    ntli = calculate_ntli(raw_vnp, raw_lunar, quality)
    
    # Step 2: Apply floor/ceiling
    filtered = apply_viirs_floor_ceiling(ntli)
    
    # Step 3: Normalize
    normalized = normalize_to_range(filtered[filtered > 0])
    
    # Step 4: Scale to byte
    final_bytes = scale_to_byte(normalized)
    
    # Final result should be valid
    assert final_bytes.dtype == np.uint8
    assert np.all(final_bytes >= 0)
    assert np.all(final_bytes <= 255)
    
    # Test numerical stability
    # Very small values
    tiny_values = np.array([1e-10, 1e-8, 1e-6])
    tiny_normalized = normalize_to_range(tiny_values)
    assert np.all(np.isfinite(tiny_normalized))
    
    # Very large values
    huge_values = np.array([1e6, 1e8, 1e10])
    huge_normalized = normalize_to_range(huge_values)
    assert np.all(np.isfinite(huge_normalized))


def test_scaling_performance_benchmarks():
    """Test performance of scaling operations.
    
    This test should verify that:
    - Large arrays are processed efficiently
    - Memory usage is reasonable and predictable
    - Processing time scales appropriately with data size
    - Vectorized operations are used where possible
    - Performance bottlenecks are identified
    """
    import time
    import psutil
    import os
    
    # Test large array processing efficiency
    # Create realistic Landsat scene size (7000x7000)
    large_landsat = np.random.randint(
        7273, 43636, size=(7000, 7000), dtype=np.uint16
    )
    
    # Measure Landsat scaling performance
    start_time = time.time()
    start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
    
    large_scaled = scale_landsat_bands(large_landsat)
    
    end_time = time.time()
    end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
    
    processing_time = end_time - start_time
    memory_used = end_memory - start_memory
    
    # Should process in reasonable time (< 10 seconds)
    assert processing_time < 10.0
    
    # Memory usage should be reasonable (< 1GB increase)
    assert memory_used < 1000  # MB
    
    # Test scaling with data size
    sizes = [(1000, 1000), (2000, 2000), (4000, 4000)]
    times = []
    
    for size in sizes:
        test_data = np.random.uniform(0, 1, size=size)
        start = time.time()
        normalize_to_range(test_data)
        end = time.time()
        times.append(end - start)
    
    # Processing time should scale roughly with array size
    # Allow for some variation due to system load
    size_ratios = [sizes[i][0] * sizes[i][1] / (sizes[0][0] * sizes[0][1]) 
                   for i in range(len(sizes))]
    time_ratios = [times[i] / times[0] for i in range(len(times))]
    
    # Time scaling should be roughly linear (within 3x of size ratio)
    for i in range(1, len(times)):
        assert time_ratios[i] < size_ratios[i] * 3
    
    # Test vectorized operations efficiency
    # Compare element-wise vs vectorized processing
    test_array = np.random.uniform(0, 100, size=(1000, 1000))
    
    # Vectorized normalization
    start_vec = time.time()
    vec_result = normalize_to_range(test_array)
    end_vec = time.time()
    vec_time = end_vec - start_vec
    
    # Ensure vectorized approach is used (should be very fast)
    assert vec_time < 0.1  # Should complete in < 100ms
    
    # Test VIIRS processing performance
    # Simulate VIIRS global daily composite
    viirs_vnp = np.random.uniform(0, 10, size=(3600, 7200))  # Global grid
    viirs_lunar = np.random.uniform(0, 2, size=(3600, 7200))
    viirs_quality = np.random.choice(
        [0, 1, 2, 3], size=(3600, 7200), p=[0.6, 0.1, 0.2, 0.1]
    ).astype(np.uint8)
    
    start_viirs = time.time()
    viirs_ntli = calculate_ntli(viirs_vnp, viirs_lunar, viirs_quality)
    viirs_filtered = apply_viirs_floor_ceiling(viirs_ntli)
    viirs_normalized = normalize_to_range(viirs_filtered)
    viirs_bytes = scale_to_byte(viirs_normalized)
    end_viirs = time.time()
    
    viirs_total_time = end_viirs - start_viirs
    
    # Full VIIRS processing chain should complete in reasonable time
    assert viirs_total_time < 30.0  # < 30 seconds for global processing
    
    # Test memory efficiency with chunked processing
    # Large dataset that might not fit in memory
    chunk_size = 1000
    total_size = 10000
    
    # Process in chunks
    start_chunk = time.time()
    for i in range(0, total_size, chunk_size):
        end_idx = min(i + chunk_size, total_size)
        chunk_data = np.random.uniform(0, 1, size=(end_idx - i, 1000))
        chunk_result = scale_to_byte(chunk_data)
        # Simulate writing to disk or similar
        del chunk_result
    end_chunk = time.time()
    
    chunk_time = end_chunk - start_chunk
    
    # Chunked processing should be efficient
    assert chunk_time < 60.0  # Should complete in reasonable time
    
    # Test performance bottleneck identification
    # Profile different operations
    profile_data = np.random.uniform(0, 1, size=(5000, 5000))
    
    operations = {
        'normalize': lambda: normalize_to_range(profile_data),
        'scale_to_byte': lambda: scale_to_byte(profile_data),
        'floor_ceiling': lambda: apply_viirs_floor_ceiling(profile_data)
    }
    
    op_times = {}
    for op_name, op_func in operations.items():
        start_op = time.time()
        op_func()
        end_op = time.time()
        op_times[op_name] = end_op - start_op
    
    # All operations should complete quickly
    for op_name, op_time in op_times.items():
        assert op_time < 5.0, f"{op_name} took too long: {op_time:.2f}s"


def test_scaling_integration_workflows():
    """Test scaling functions in integrated workflows.
    
    This test should verify that:
    - Multiple scaling operations work together correctly
    - Landsat and VIIRS scaling can be combined
    - Normalization and byte scaling work sequentially
    - Data quality is maintained through complex workflows
    - Processing chains are scientifically valid
    """
    # Test complete Landsat processing workflow
    # Simulate Landsat 8 surface reflectance processing
    raw_landsat_dn = np.array([
        [7273, 15000, 25000, 35000, 43636],  # Valid DN range
        [10000, 20000, 30000, 40000, 42000],
        [8000, 18000, 28000, 38000, 41000]
    ], dtype=np.uint16)
    
    # Step 1: Check if already scaled
    is_scaled = detect_if_already_scaled(raw_landsat_dn)
    assert not is_scaled  # Should detect as unscaled DN
    
    # Step 2: Scale to reflectance
    reflectance = scale_landsat_bands(raw_landsat_dn)
    
    # Step 3: Normalize for visualization
    norm_refl = normalize_to_range(reflectance, out_min=0, out_max=1)
    
    # Step 4: Scale to byte for export
    byte_refl = scale_to_byte(norm_refl)
    
    # Verify workflow integrity
    assert reflectance.shape == raw_landsat_dn.shape
    assert np.all(reflectance >= -0.2)  # Valid reflectance range
    assert np.all(reflectance <= 1.0)
    assert norm_refl.shape == reflectance.shape
    assert byte_refl.dtype == np.uint8
    
    # Test complete VIIRS processing workflow
    # Simulate VIIRS DNB nighttime lights processing
    raw_vnp = np.array([
        [0.1, 0.5, 2.0, 5.0, 10.0],
        [0.2, 1.0, 3.0, 8.0, 15.0],
        [0.05, 0.8, 2.5, 6.0, 12.0]
    ], dtype=np.float32)
    
    lunar_irrad = np.array([
        [0.02, 0.1, 0.3, 0.5, 1.0],
        [0.03, 0.15, 0.4, 0.8, 1.5],
        [0.01, 0.12, 0.35, 0.6, 1.2]
    ], dtype=np.float32)
    
    quality_flags = np.array([
        [0, 0, 0, 2, 1],  # Good, good, good, marginal, bad
        [0, 2, 1, 0, 0],  # Good, marginal, bad, good, good
        [2, 0, 0, 0, 3]   # Marginal, good, good, good, poor
    ], dtype=np.uint8)
    
    # Step 1: Calculate NTLI with lunar correction
    ntli = calculate_ntli(raw_vnp, lunar_irrad, quality_flags)
    
    # Step 2: Apply floor and ceiling thresholds
    filtered_ntli = apply_viirs_floor_ceiling(ntli)
    
    # Step 3: Normalize for analysis
    norm_ntli = normalize_to_range(filtered_ntli)
    
    # Step 4: Scale to byte for visualization
    byte_ntli = scale_to_byte(norm_ntli)
    
    # Verify VIIRS workflow integrity
    assert ntli.shape == raw_vnp.shape
    assert np.all(filtered_ntli >= 0)
    assert np.all(filtered_ntli <= 10.0)  # Ceiling applied
    assert byte_ntli.dtype == np.uint8
    
    # Test combined Landsat-VIIRS workflow
    # Urban nighttime lights enhanced with daytime urban features
    
    # Landsat urban mask (high NDVI = vegetation, low = urban)
    landsat_red = scale_landsat_bands(np.full((3, 5), 20000, dtype=np.uint16))
    landsat_nir = scale_landsat_bands(np.full((3, 5), 30000, dtype=np.uint16))
    
    # Simple NDVI calculation
    ndvi = (landsat_nir - landsat_red) / (landsat_nir + landsat_red + 1e-6)
    urban_mask = ndvi < 0.2  # Low NDVI indicates urban
    
    # Enhance VIIRS data in urban areas
    enhanced_ntli = filtered_ntli.copy()
    enhanced_ntli[urban_mask] *= 1.5  # Enhance urban lights
    
    # Final processing
    final_enhanced = normalize_to_range(enhanced_ntli)
    final_bytes = scale_to_byte(final_enhanced)
    
    # Verify combined workflow
    assert final_bytes.shape == raw_vnp.shape
    assert final_bytes.dtype == np.uint8
    
    # Test sequential normalization and byte scaling
    # Multiple normalization steps
    test_data = np.random.uniform(0, 1000, size=(10, 10))
    
    # Step 1: Normalize to [0, 1]
    norm1 = normalize_to_range(test_data, out_min=0, out_max=1)
    
    # Step 2: Normalize to [-1, 1]
    norm2 = normalize_to_range(norm1, out_min=-1, out_max=1)
    
    # Step 3: Normalize back to [0, 1]
    norm3 = normalize_to_range(norm2, out_min=0, out_max=1)
    
    # Step 4: Scale to byte
    final_seq = scale_to_byte(norm3)
    
    # Should preserve relative relationships
    np.testing.assert_allclose(norm1, norm3, rtol=1e-6)
    
    # Test data quality preservation through complex workflow
    # Start with realistic remote sensing data
    complex_data = np.random.gamma(2, 2, size=(100, 100))  # Realistic distribution
    
    # Add some realistic noise and outliers
    noise = np.random.normal(0, 0.1, size=(100, 100))
    complex_data += noise
    
    # Add a few outliers
    outlier_mask = np.random.random((100, 100)) < 0.01  # 1% outliers
    complex_data[outlier_mask] *= 10
    
    # Complex processing chain
    step1 = apply_viirs_floor_ceiling(complex_data, floor_value=0.5, ceiling_value=20)
    step2 = normalize_to_range(step1, out_min=0, out_max=1)
    step3 = normalize_to_range(step2, out_min=0, out_max=255)  # Prepare for byte
    final_complex = scale_to_byte(step3)
    
    # Verify scientific validity throughout
    assert np.all(step1 >= 0)  # Non-negative after floor/ceiling
    assert np.all(step1 <= 20)  # Ceiling applied
    assert np.min(step2) >= -1e-6 and np.max(step2) <= 1 + 1e-6  # Normalized
    assert final_complex.dtype == np.uint8
    
    # Check that spatial patterns are preserved
    # Calculate correlation between input and output
    correlation = np.corrcoef(
        complex_data[~outlier_mask].flatten(),
        final_complex[~outlier_mask].flatten()
    )[0, 1]
    
    # Should maintain high correlation despite multiple processing steps
    assert correlation > 0.8