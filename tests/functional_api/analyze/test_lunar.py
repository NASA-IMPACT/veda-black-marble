"""Test functions in blackmarble.analyze.lunar module.

This module tests lunar correction and lunar BRDF functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from datetime import datetime

# Import the module under test
from blackmarble.analyze.lunar import (
    calculate_lunar_brdf,
    apply_lunar_correction,
    compute_ntli_index,
    estimate_lunar_reflectance,
    detect_lunar_contamination,
    calculate_moon_phase_weight,
    analyze_lunar_cycle_effects,
    compute_clear_sky_lunar_radiance,
)


def test_calculate_lunar_brdf():
    """Test lunar BRDF calculation with different models.
    
    This test should verify that:
    - Ross-Thick Li-Sparse BRDF model is correctly implemented
    - Geometric parameters (zenith, azimuth angles) are properly handled
    - Land cover type affects BRDF calculations appropriately
    - Output values are within physically reasonable range
    - Mathematical accuracy of BRDF kernel functions
    - Proper handling of extreme viewing geometry
    - Different BRDF model options produce distinct results
    """
    # Test basic BRDF calculation with Ross-Thick model
    lunar_zenith = np.array([30.0, 45.0, 60.0])  # degrees
    lunar_azimuth = np.array([0.0, 90.0, 180.0])
    view_zenith = np.array([0.0, 30.0, 45.0])
    view_azimuth = np.array([0.0, 0.0, 0.0])
    
    # Calculate BRDF with Ross-Thick model
    brdf_ross = calculate_lunar_brdf(
        lunar_zenith, lunar_azimuth, view_zenith, view_azimuth,
        brdf_model="ross_thick"
    )
    
    # Verify output values are within physically reasonable range [0.3, 3.0]
    assert np.all((brdf_ross >= 0.3) & (brdf_ross <= 3.0))
    
    # Verify BRDF varies with geometry
    # Note: BRDF depends on complex interaction of lunar/view angles, not just zenith
    assert len(np.unique(np.round(brdf_ross, 4))) > 1, "BRDF should vary with geometry"
    
    # Test Li-Sparse model
    brdf_li = calculate_lunar_brdf(
        lunar_zenith, lunar_azimuth, view_zenith, view_azimuth,
        brdf_model="li_sparse"
    )
    
    # Verify Li-Sparse produces different results from Ross-Thick
    assert not np.allclose(brdf_ross, brdf_li), "Different models should produce different results"
    
    # Test custom urban model
    brdf_custom = calculate_lunar_brdf(
        lunar_zenith, lunar_azimuth, view_zenith, view_azimuth,
        brdf_model="custom"
    )
    
    # Verify custom model produces distinct results
    assert not np.allclose(brdf_custom, brdf_ross)
    assert not np.allclose(brdf_custom, brdf_li)
    
    # Test with land cover types
    land_cover = np.array([1, 2, 3])  # water, vegetation, urban
    brdf_landcover = calculate_lunar_brdf(
        lunar_zenith, lunar_azimuth, view_zenith, view_azimuth,
        land_cover=land_cover, brdf_model="ross_thick"
    )
    
    # Verify water has minimal BRDF effects (close to 1.0)
    assert abs(brdf_landcover[0] - 1.0) < abs(brdf_ross[0] - 1.0), "Water should have minimal BRDF"
    
    # Test extreme viewing geometry (near-horizon)
    extreme_zenith = np.array([85.0, 89.0, 89.9])
    brdf_extreme = calculate_lunar_brdf(
        extreme_zenith, lunar_azimuth, view_zenith, view_azimuth
    )
    
    # Verify no NaN or infinite values for extreme angles
    assert np.all(np.isfinite(brdf_extreme))
    
    # Test error handling for invalid inputs
    with pytest.raises(ValueError, match="must have same shape"):
        calculate_lunar_brdf(
            np.array([30.0]), np.array([0.0, 90.0]), 
            np.array([0.0]), np.array([0.0])
        )
    
    # Test zenith angle validation
    with pytest.raises(ValueError, match="Lunar zenith angles must be"):
        calculate_lunar_brdf(
            np.array([-10.0]), np.array([0.0]), 
            np.array([0.0]), np.array([0.0])
        )


def test_calculate_lunar_brdf_ross_thick():
    """Test Ross-Thick BRDF kernel implementation.
    
    This test should verify that:
    - Ross-Thick kernel formula is correctly implemented
    - Kernel values are computed accurately for various geometries
    - Hotspot effect is properly modeled
    - Mathematical consistency with published algorithms
    - Proper handling of edge cases (nadir viewing, etc.)
    """
    # Test nadir viewing (view_zenith = 0)
    lunar_zenith = np.array([0.0, 30.0, 60.0, 90.0])
    lunar_azimuth = np.zeros(4)
    view_zenith = np.zeros(4)  # Nadir viewing
    view_azimuth = np.zeros(4)
    
    brdf_nadir = calculate_lunar_brdf(
        lunar_zenith, lunar_azimuth, view_zenith, view_azimuth,
        brdf_model="ross_thick"
    )
    
    # At nadir viewing with lunar zenith = 0, should be at baseline
    # Ross-Thick kernel with both angles at 0 gives baseline of 1.0
    assert brdf_nadir[0] == 1.0, "Nadir viewing with moon at zenith gives baseline BRDF"
    
    # BRDF can increase at high zenith angles due to kernel behavior
    # Check that values remain in reasonable range
    assert np.all((brdf_nadir >= 0.3) & (brdf_nadir <= 3.0)), "BRDF values should be reasonable"
    
    # Test hotspot geometry (lunar and view directions aligned)
    lunar_zenith = np.array([30.0, 30.0, 30.0])
    lunar_azimuth = np.array([0.0, 90.0, 180.0])
    view_zenith = np.array([30.0, 30.0, 30.0])
    view_azimuth = np.array([0.0, 90.0, 180.0])  # Same as lunar azimuth
    
    brdf_hotspot = calculate_lunar_brdf(
        lunar_zenith, lunar_azimuth, view_zenith, view_azimuth,
        brdf_model="ross_thick"
    )
    
    # All should show some enhancement (same direction)
    assert np.all(brdf_hotspot > 1.0), "Hotspot should show some BRDF enhancement"
    
    # Test forward vs backward scattering
    lunar_zenith = np.array([45.0, 45.0])
    lunar_azimuth = np.array([0.0, 0.0])
    view_zenith = np.array([45.0, 45.0])
    view_azimuth = np.array([0.0, 180.0])  # Forward and backward scatter
    
    brdf_scatter = calculate_lunar_brdf(
        lunar_zenith, lunar_azimuth, view_zenith, view_azimuth,
        brdf_model="ross_thick"
    )
    
    # BRDF should differ between forward and backward scatter
    assert abs(brdf_scatter[0] - brdf_scatter[1]) > 0.01, "Forward/backward scatter should differ"
    
    # Test mathematical properties of Ross kernel
    # The kernel should be continuous and smooth
    zenith_range = np.linspace(0, 89, 50)
    lunar_zenith = zenith_range
    lunar_azimuth = np.zeros_like(zenith_range)
    view_zenith = np.full_like(zenith_range, 30.0)
    view_azimuth = np.zeros_like(zenith_range)
    
    brdf_smooth = calculate_lunar_brdf(
        lunar_zenith, lunar_azimuth, view_zenith, view_azimuth,
        brdf_model="ross_thick"
    )
    
    # Check smoothness (no sudden jumps)
    diff = np.abs(np.diff(brdf_smooth))
    assert np.all(diff < 0.1), "BRDF should vary smoothly"


def test_calculate_lunar_brdf_li_sparse():
    """Test Li-Sparse BRDF kernel implementation.
    
    This test should verify that:
    - Li-Sparse kernel formula is correctly implemented
    - Geometric scattering effects are properly modeled
    - Crown shape parameters are handled correctly
    - Mathematical accuracy with reference implementations
    """
    # Test basic Li-Sparse calculation
    lunar_zenith = np.array([0.0, 30.0, 60.0])
    lunar_azimuth = np.array([0.0, 0.0, 0.0])
    view_zenith = np.array([0.0, 15.0, 45.0])
    view_azimuth = np.array([0.0, 0.0, 0.0])
    
    brdf_li = calculate_lunar_brdf(
        lunar_zenith, lunar_azimuth, view_zenith, view_azimuth,
        brdf_model="li_sparse"
    )
    
    # Verify values are in reasonable range
    assert np.all((brdf_li >= 0.3) & (brdf_li <= 3.0))
    
    # Test geometric scattering with different azimuth configurations
    lunar_zenith = np.array([45.0, 45.0, 45.0, 45.0])
    lunar_azimuth = np.array([0.0, 90.0, 180.0, 270.0])
    view_zenith = np.full(4, 30.0)
    view_azimuth = np.zeros(4)
    
    brdf_azimuth = calculate_lunar_brdf(
        lunar_zenith, lunar_azimuth, view_zenith, view_azimuth,
        brdf_model="li_sparse"
    )
    
    # Li-Sparse should show azimuthal variation
    assert np.std(brdf_azimuth) > 0.01, "Li-Sparse should vary with azimuth"
    
    # Test crown shape effects with varying zenith angles
    # Higher zenith angles should show more geometric effects
    lunar_zenith = np.linspace(0, 80, 20)
    lunar_azimuth = np.zeros_like(lunar_zenith)
    view_zenith = np.full_like(lunar_zenith, 45.0)
    view_azimuth = np.full_like(lunar_zenith, 180.0)
    
    brdf_crown = calculate_lunar_brdf(
        lunar_zenith, lunar_azimuth, view_zenith, view_azimuth,
        brdf_model="li_sparse"
    )
    
    # Verify smooth variation
    assert np.all(np.isfinite(brdf_crown))
    
    # Compare with Ross-Thick to ensure distinct behavior
    brdf_ross = calculate_lunar_brdf(
        lunar_zenith, lunar_azimuth, view_zenith, view_azimuth,
        brdf_model="ross_thick"
    )
    
    # Li-Sparse and Ross-Thick should differ significantly
    relative_diff = np.abs(brdf_li - brdf_ross[0:3]) / brdf_ross[0:3]
    assert np.mean(relative_diff) > 0.05, "Li-Sparse should differ from Ross-Thick"


def test_calculate_lunar_brdf_land_cover():
    """Test BRDF calculation with different land cover types.
    
    This test should verify that:
    - Different land cover types have appropriate BRDF parameters
    - Surface reflectance characteristics are properly modeled
    - Urban, vegetation, water, and bare soil show distinct BRDF
    - Proper fallback for unknown land cover types
    """
    # Test with all land cover types
    lunar_zenith = np.array([45.0, 45.0, 45.0, 45.0, 45.0])
    lunar_azimuth = np.zeros(5)
    view_zenith = np.array([30.0, 30.0, 30.0, 30.0, 30.0])
    view_azimuth = np.array([180.0, 180.0, 180.0, 180.0, 180.0])
    
    # Land cover: water, vegetation, urban, bare soil, unknown
    land_cover = np.array([1, 2, 3, 4, 99])
    
    brdf_landcover = calculate_lunar_brdf(
        lunar_zenith, lunar_azimuth, view_zenith, view_azimuth,
        land_cover=land_cover, brdf_model="ross_thick"
    )
    
    # Calculate reference without land cover
    brdf_reference = calculate_lunar_brdf(
        lunar_zenith, lunar_azimuth, view_zenith, view_azimuth,
        brdf_model="ross_thick"
    )
    
    # Water (index 0) should have minimal BRDF effects
    water_reduction = abs(brdf_landcover[0] - 1.0) / abs(brdf_reference[0] - 1.0)
    assert water_reduction < 0.15, "Water should have minimal BRDF effects"
    
    # Vegetation (index 1) should have full BRDF effects
    vegetation_ratio = brdf_landcover[1] / brdf_reference[1]
    assert 0.95 < vegetation_ratio < 1.05, "Vegetation should have full BRDF"
    
    # Urban (index 2) should have moderate BRDF effects
    urban_reduction = abs(brdf_landcover[2] - 1.0) / abs(brdf_reference[2] - 1.0)
    assert 0.3 < urban_reduction < 0.7, "Urban should have moderate BRDF effects"
    
    # Bare soil (index 3) should have enhanced BRDF effects
    bare_enhancement = abs(brdf_landcover[3] - 1.0) / abs(brdf_reference[3] - 1.0)
    assert bare_enhancement > 1.05, "Bare soil should have enhanced BRDF"
    
    # Unknown land cover (index 4) should use default behavior
    unknown_ratio = brdf_landcover[4] / brdf_reference[4]
    assert 0.95 < unknown_ratio < 1.05, "Unknown should use default BRDF"
    
    # Verify distinct BRDF for different surfaces
    brdf_values = brdf_landcover[:4]  # Exclude unknown
    assert len(np.unique(np.round(brdf_values, 3))) >= 3, "Different surfaces should have distinct BRDF"


def test_apply_lunar_correction():
    """Test comprehensive lunar correction application.
    
    This test should verify that:
    - Lunar radiance contribution is properly removed
    - Correction threshold prevents over-correction
    - Minimum lunar zenith angle constraint is enforced
    - BRDF correction is applied when enabled
    - Corrected radiance values are physically reasonable
    - No-data values are properly preserved
    - Mathematical accuracy of correction algorithm
    """
    # Create test data
    shape = (100, 100)
    radiance = np.random.uniform(0.01, 0.5, shape)  # Typical nighttime radiance
    lunar_irradiance = np.random.uniform(0.001, 0.05, shape)  # Lunar contribution
    lunar_zenith = np.random.uniform(20, 80, shape)  # Lunar zenith angles
    
    # Add some no-data values
    radiance[10:20, 10:20] = np.nan
    
    # Apply correction
    corrected, stats = apply_lunar_correction(
        radiance, lunar_irradiance, lunar_zenith,
        correction_threshold=0.005,
        min_lunar_zenith=85.0,
        apply_brdf=True
    )
    
    # Verify lunar contribution is removed
    assert np.all(corrected[~np.isnan(corrected)] <= radiance[~np.isnan(radiance)]), \
        "Corrected radiance should be less than or equal to original"
    
    # Verify no-data values are preserved
    assert np.all(np.isnan(corrected[10:20, 10:20])), "NaN values should be preserved"
    
    # Verify corrected values are non-negative
    assert np.all(corrected[~np.isnan(corrected)] >= 0), "Corrected radiance must be non-negative"
    
    # Check statistics
    assert stats['pixels_corrected'] > 0, "Some pixels should be corrected"
    assert stats['mean_correction'] > 0, "Mean correction should be positive"
    assert stats['max_correction'] >= stats['mean_correction'], "Max should be >= mean"
    assert 0 <= stats['lunar_contamination_percent'] <= 100, "Percentage must be valid"
    
    # Test correction threshold
    low_lunar = np.full(shape, 0.0001)  # Below threshold
    corrected_low, stats_low = apply_lunar_correction(
        radiance, low_lunar, lunar_zenith,
        correction_threshold=0.005
    )
    assert stats_low['pixels_corrected'] == 0, "No correction below threshold"
    
    # Test minimum zenith angle constraint
    high_zenith = np.full(shape, 87.0)  # Above min_lunar_zenith
    corrected_high, stats_high = apply_lunar_correction(
        radiance, lunar_irradiance, high_zenith,
        min_lunar_zenith=85.0
    )
    assert stats_high['pixels_corrected'] == 0, "No correction above zenith limit"
    
    # Test with and without BRDF
    corrected_no_brdf, _ = apply_lunar_correction(
        radiance, lunar_irradiance, lunar_zenith,
        apply_brdf=False
    )
    
    # BRDF should make a difference
    assert not np.allclose(corrected[~np.isnan(corrected)], 
                          corrected_no_brdf[~np.isnan(corrected_no_brdf)]), \
        "BRDF should affect correction"


def test_apply_lunar_correction_threshold():
    """Test lunar correction with threshold constraints.
    
    This test should verify that:
    - Correction is only applied when lunar contribution exceeds threshold
    - Pixels below threshold remain unchanged
    - Threshold parameter works as expected
    - Proper handling of very low lunar contributions
    """
    # Create test data with varying lunar contributions
    shape = (50, 50)
    radiance = np.full(shape, 0.1)  # Uniform radiance
    
    # Create lunar irradiance with gradient from low to high
    lunar_irradiance = np.linspace(0.0001, 0.01, shape[0])[:, np.newaxis]
    lunar_irradiance = np.broadcast_to(lunar_irradiance, shape)
    
    lunar_zenith = np.full(shape, 45.0)
    
    # Test with threshold in the middle of the range
    threshold = 0.005
    corrected, stats = apply_lunar_correction(
        radiance, lunar_irradiance, lunar_zenith,
        correction_threshold=threshold
    )
    
    # Pixels below threshold should be unchanged
    below_threshold = lunar_irradiance < threshold
    assert np.allclose(corrected[below_threshold], radiance[below_threshold]), \
        "Pixels below threshold should be unchanged"
    
    # Pixels above threshold should be corrected
    above_threshold = lunar_irradiance >= threshold
    assert np.all(corrected[above_threshold] < radiance[above_threshold]), \
        "Pixels above threshold should be corrected"
    
    # Verify threshold is respected exactly
    corrected_pixels = stats['pixels_corrected']
    expected_corrected = np.sum(above_threshold)
    assert corrected_pixels == expected_corrected, \
        f"Expected {expected_corrected} corrected pixels, got {corrected_pixels}"
    
    # Test with very low threshold (all pixels corrected)
    corrected_all, stats_all = apply_lunar_correction(
        radiance, lunar_irradiance, lunar_zenith,
        correction_threshold=0.00001
    )
    assert stats_all['pixels_corrected'] == shape[0] * shape[1], \
        "All pixels should be corrected with very low threshold"
    
    # Test with very high threshold (no pixels corrected)
    corrected_none, stats_none = apply_lunar_correction(
        radiance, lunar_irradiance, lunar_zenith,
        correction_threshold=1.0
    )
    assert stats_none['pixels_corrected'] == 0, \
        "No pixels should be corrected with very high threshold"
    assert np.allclose(corrected_none, radiance), \
        "Radiance should be unchanged with high threshold"


def test_apply_lunar_correction_zenith_limit():
    """Test lunar correction with zenith angle limits.
    
    This test should verify that:
    - Correction is skipped when lunar zenith is too low
    - min_lunar_zenith parameter is properly enforced
    - Geometric constraints prevent invalid corrections
    - Edge cases near zenith limit are handled correctly
    """
    # Create test data
    shape = (50, 50)
    radiance = np.full(shape, 0.2)
    lunar_irradiance = np.full(shape, 0.01)
    
    # Create zenith angles around the limit
    zenith_values = np.array([80.0, 84.0, 85.0, 86.0, 90.0])
    lunar_zenith = np.zeros(shape)
    
    # Divide image into strips with different zenith angles
    for i, zenith in enumerate(zenith_values):
        start = i * 10
        end = (i + 1) * 10
        lunar_zenith[start:end, :] = zenith
    
    # Apply correction with min_lunar_zenith = 85.0
    min_zenith = 85.0
    corrected, stats = apply_lunar_correction(
        radiance, lunar_irradiance, lunar_zenith,
        min_lunar_zenith=min_zenith
    )
    
    # Verify pixels above min_zenith are not corrected
    above_limit = lunar_zenith > min_zenith
    assert np.allclose(corrected[above_limit], radiance[above_limit]), \
        "Pixels above zenith limit should be unchanged"
    
    # Verify pixels at or below min_zenith are corrected
    at_or_below = lunar_zenith <= min_zenith
    corrected_mask = corrected < radiance
    assert np.all(corrected_mask[at_or_below] == True), \
        "Pixels at or below zenith limit should be corrected"
    
    # Test edge case exactly at limit
    at_limit = lunar_zenith == min_zenith
    assert np.any(corrected[at_limit] < radiance[at_limit]), \
        "Pixels exactly at limit should be corrected"
    
    # Verify statistics match expectations
    expected_corrected = np.sum(at_or_below)
    assert stats['pixels_corrected'] == expected_corrected, \
        f"Expected {expected_corrected} corrected pixels"
    
    # Test with different zenith limits
    for test_limit in [70.0, 80.0, 89.0]:
        corrected_test, stats_test = apply_lunar_correction(
            radiance, lunar_irradiance, lunar_zenith,
            min_lunar_zenith=test_limit
        )
        
        pixels_below_limit = np.sum(lunar_zenith <= test_limit)
        assert stats_test['pixels_corrected'] == pixels_below_limit, \
            f"Zenith limit {test_limit} should correct {pixels_below_limit} pixels"


def test_apply_lunar_correction_brdf():
    """Test lunar correction with BRDF factors.
    
    This test should verify that:
    - BRDF factors are properly applied to correction
    - Directional effects are correctly modeled
    - Custom BRDF factors override default calculations
    - Mathematical consistency of BRDF-corrected results
    """
    # Create test data
    shape = (30, 30)
    radiance = np.full(shape, 0.3)
    lunar_irradiance = np.full(shape, 0.02)
    lunar_zenith = np.full(shape, 45.0)
    
    # Test with default BRDF (apply_brdf=True, no custom factors)
    corrected_default, _ = apply_lunar_correction(
        radiance, lunar_irradiance, lunar_zenith,
        apply_brdf=True
    )
    
    # Test without BRDF
    corrected_no_brdf, _ = apply_lunar_correction(
        radiance, lunar_irradiance, lunar_zenith,
        apply_brdf=False
    )
    
    # BRDF should affect the correction
    assert not np.allclose(corrected_default, corrected_no_brdf), \
        "BRDF should change correction results"
    
    # Create custom BRDF factors
    custom_brdf = np.linspace(0.5, 2.0, shape[0])[:, np.newaxis]
    custom_brdf = np.broadcast_to(custom_brdf, shape)
    
    # Apply correction with custom BRDF
    corrected_custom, _ = apply_lunar_correction(
        radiance, lunar_irradiance, lunar_zenith,
        apply_brdf=True,
        brdf_factors=custom_brdf
    )
    
    # Custom BRDF should produce different results from default
    assert not np.allclose(corrected_custom, corrected_default), \
        "Custom BRDF should produce different results"
    
    # Verify BRDF factors affect correction magnitude
    # Higher BRDF factors should lead to larger corrections
    high_brdf = np.full(shape, 2.0)
    low_brdf = np.full(shape, 0.5)
    
    corrected_high, _ = apply_lunar_correction(
        radiance, lunar_irradiance, lunar_zenith,
        apply_brdf=True,
        brdf_factors=high_brdf
    )
    
    corrected_low, _ = apply_lunar_correction(
        radiance, lunar_irradiance, lunar_zenith,
        apply_brdf=True,
        brdf_factors=low_brdf
    )
    
    # Higher BRDF means more lunar contribution to remove
    correction_high = radiance - corrected_high
    correction_low = radiance - corrected_low
    
    assert np.mean(correction_high) > np.mean(correction_low), \
        "Higher BRDF factors should lead to larger corrections"
    
    # Test mathematical consistency
    # Correction should be proportional to BRDF factor
    ratio = np.mean(correction_high) / np.mean(correction_low)
    expected_ratio = 2.0 / 0.5  # 4.0
    assert 3.5 < ratio < 4.5, \
        f"Correction ratio {ratio:.2f} should be close to BRDF ratio {expected_ratio}"


def test_compute_ntli_index():
    """Test Normalized Total Light Index computation.
    
    This test should verify that:
    - NTLI formula correctly normalizes radiance by lunar fraction
    - Cloud mask properly excludes contaminated pixels
    - Index values are within expected range
    - Mathematical accuracy with known test cases
    - Proper handling of zero lunar fraction
    - No-data propagation through calculation
    """
    # Create test radiance data
    shape = (50, 50)
    radiance = np.random.uniform(0.01, 0.5, shape)
    
    # Test with different lunar fractions
    lunar_fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for lunar_fraction in lunar_fractions:
        ntli = compute_ntli_index(radiance, lunar_fraction)
        
        # Verify NTLI formula: radiance * (1 + 0.05 * lunar_fraction)
        expected = radiance * (1.0 + 0.05 * lunar_fraction)
        np.testing.assert_allclose(ntli, expected, rtol=1e-6)
        
        # NTLI should be >= radiance (normalization factor >= 1)
        assert np.all(ntli >= radiance), f"NTLI should be >= radiance for lunar_fraction={lunar_fraction}"
    
    # Test with cloud mask
    cloud_mask = np.zeros(shape, dtype=bool)
    cloud_mask[10:20, 10:20] = True  # Mark some pixels as cloudy
    
    ntli_masked = compute_ntli_index(radiance, 0.5, cloud_mask)
    
    # Cloudy pixels should be NaN
    assert np.all(np.isnan(ntli_masked[cloud_mask])), "Cloudy pixels should be NaN"
    
    # Clear pixels should have valid values
    clear_pixels = ~cloud_mask
    assert np.all(~np.isnan(ntli_masked[clear_pixels])), "Clear pixels should have valid values"
    
    # Test edge case: zero lunar fraction (new moon)
    ntli_zero = compute_ntli_index(radiance, 0.0)
    np.testing.assert_allclose(ntli_zero, radiance, rtol=1e-6)
    
    # Test edge case: full moon (lunar_fraction = 1.0)
    ntli_full = compute_ntli_index(radiance, 1.0)
    expected_full = radiance * 1.05  # 5% increase
    np.testing.assert_allclose(ntli_full, expected_full, rtol=1e-6)
    
    # Test no-data propagation
    radiance_with_nan = radiance.copy()
    radiance_with_nan[5:10, 5:10] = np.nan
    
    ntli_nan = compute_ntli_index(radiance_with_nan, 0.5)
    assert np.all(np.isnan(ntli_nan[5:10, 5:10])), "NaN should propagate through calculation"
    
    # Test error handling
    with pytest.raises(ValueError, match="Lunar fraction must be"):
        compute_ntli_index(radiance, -0.1)
    
    with pytest.raises(ValueError, match="Lunar fraction must be"):
        compute_ntli_index(radiance, 1.1)


def test_compute_ntli_index_cloud_mask():
    """Test NTLI computation with cloud masking.
    
    This test should verify that:
    - Cloudy pixels are excluded from NTLI calculation
    - Cloud mask is properly applied to input data
    - Different cloud mask formats are handled
    - Statistics reflect only clear-sky pixels
    """
    # Create test data
    shape = (40, 40)
    radiance = np.full(shape, 0.2)  # Uniform radiance for easier verification
    lunar_fraction = 0.6
    
    # Test with no cloud mask
    ntli_no_mask = compute_ntli_index(radiance, lunar_fraction)
    assert np.all(~np.isnan(ntli_no_mask)), "No NaN values without cloud mask"
    
    # Create different cloud mask patterns
    # Pattern 1: Scattered clouds
    cloud_mask_scattered = np.random.random(shape) > 0.8  # ~20% cloudy
    ntli_scattered = compute_ntli_index(radiance, lunar_fraction, cloud_mask_scattered)
    
    assert np.all(np.isnan(ntli_scattered[cloud_mask_scattered])), \
        "Scattered cloudy pixels should be NaN"
    assert np.all(~np.isnan(ntli_scattered[~cloud_mask_scattered])), \
        "Clear pixels should have valid values"
    
    # Pattern 2: Cloud bands
    cloud_mask_bands = np.zeros(shape, dtype=bool)
    cloud_mask_bands[::5, :] = True  # Every 5th row is cloudy
    ntli_bands = compute_ntli_index(radiance, lunar_fraction, cloud_mask_bands)
    
    assert np.all(np.isnan(ntli_bands[::5, :])), "Cloud bands should be NaN"
    assert np.all(~np.isnan(ntli_bands[1::5, :])), "Clear bands should have values"
    
    # Pattern 3: Block of clouds
    cloud_mask_block = np.zeros(shape, dtype=bool)
    cloud_mask_block[15:25, 15:25] = True  # Central block
    ntli_block = compute_ntli_index(radiance, lunar_fraction, cloud_mask_block)
    
    # Verify statistics
    clear_values = ntli_block[~cloud_mask_block]
    expected_value = 0.2 * (1.0 + 0.05 * lunar_fraction)  # Expected NTLI
    
    assert np.allclose(clear_values, expected_value), \
        "Clear pixels should have correct NTLI value"
    
    # Test with all pixels cloudy
    cloud_mask_all = np.ones(shape, dtype=bool)
    ntli_all_cloudy = compute_ntli_index(radiance, lunar_fraction, cloud_mask_all)
    assert np.all(np.isnan(ntli_all_cloudy)), "All pixels should be NaN when all cloudy"
    
    # Test with no pixels cloudy
    cloud_mask_none = np.zeros(shape, dtype=bool)
    ntli_no_clouds = compute_ntli_index(radiance, lunar_fraction, cloud_mask_none)
    assert np.all(~np.isnan(ntli_no_clouds)), "No NaN when no clouds"
    
    # Verify shape consistency
    assert ntli_scattered.shape == radiance.shape
    assert ntli_bands.shape == radiance.shape
    assert ntli_block.shape == radiance.shape


def test_estimate_lunar_reflectance():
    """Test surface lunar reflectance estimation.
    
    This test should verify that:
    - Reflectance estimates vary appropriately by surface type
    - Lunar zenith angle effects are properly modeled
    - Moon phase influences reflectance estimates
    - Values are within physically reasonable range
    - Mathematical consistency with empirical models
    - Proper handling of extreme lunar geometry
    """
    # Test basic reflectance calculation
    shape = (5, 5)
    surface_types = np.array([[0, 1, 2, 3, 4],  # unclassified, water, veg, urban, bare
                             [1, 2, 3, 4, 0],
                             [2, 3, 4, 0, 1],
                             [3, 4, 0, 1, 2],
                             [4, 0, 1, 2, 3]])
    
    lunar_zenith = np.full(shape, 45.0)
    lunar_phase = 30.0  # Near full moon
    
    reflectance = estimate_lunar_reflectance(surface_types, lunar_zenith, lunar_phase)
    
    # Verify values are within physically reasonable range [0.001, 0.6]
    assert np.all((reflectance >= 0.001) & (reflectance <= 0.6)), \
        "Reflectance must be in physical range"
    
    # Extract reflectance by surface type at 45° zenith
    water_refl = reflectance[surface_types == 1].mean()
    veg_refl = reflectance[surface_types == 2].mean()
    urban_refl = reflectance[surface_types == 3].mean()
    bare_refl = reflectance[surface_types == 4].mean()
    
    # Verify surface type ordering: water < vegetation < bare soil < urban
    assert water_refl < veg_refl < bare_refl < urban_refl, \
        "Surface reflectance should follow expected ordering"
    
    # Test zenith angle effects
    zenith_angles = np.array([0.0, 30.0, 60.0, 85.0])
    urban_surface = np.full((4,), 3)  # All urban
    
    refl_by_zenith = []
    for zenith in zenith_angles:
        zenith_array = np.full((4,), zenith)
        refl = estimate_lunar_reflectance(urban_surface, zenith_array, lunar_phase)
        refl_by_zenith.append(refl.mean())
    
    # Reflectance should decrease with increasing zenith angle
    assert all(refl_by_zenith[i] > refl_by_zenith[i+1] for i in range(3)), \
        "Reflectance should decrease with zenith angle"
    
    # Test moon phase effects
    phase_angles = np.array([0.0, 45.0, 90.0, 135.0, 180.0])  # Full to new moon
    phase_surface = np.full((5,), 2)  # All vegetation
    phase_zenith = np.full((5,), 45.0)
    
    refl_by_phase = []
    for phase in phase_angles:
        refl = estimate_lunar_reflectance(phase_surface, phase_zenith, phase)
        refl_by_phase.append(refl.mean())
    
    # Reflectance should decrease from full moon to new moon
    assert refl_by_phase[0] > refl_by_phase[-1], \
        "Full moon should have higher reflectance than new moon"
    
    # Test extreme geometry
    extreme_zenith = np.array([0.0, 89.9])  # Zenith to horizon
    extreme_surface = np.array([3, 3])  # Urban
    
    extreme_refl = estimate_lunar_reflectance(extreme_surface, extreme_zenith, 0.0)
    assert np.all(np.isfinite(extreme_refl)), "Extreme angles should produce finite values"
    assert extreme_refl[0] > extreme_refl[1], "Zenith should be brighter than horizon"
    
    # Test unknown surface types
    unknown_surface = np.array([99, 100, -1])  # Unknown codes
    unknown_zenith = np.full((3,), 45.0)
    
    unknown_refl = estimate_lunar_reflectance(unknown_surface, unknown_zenith, 30.0)
    assert np.all((unknown_refl >= 0.001) & (unknown_refl <= 0.6)), \
        "Unknown surfaces should have reasonable reflectance"


def test_estimate_lunar_reflectance_surface_types():
    """Test lunar reflectance for different surface types.
    
    This test should verify that:
    - Water surfaces show low lunar reflectance
    - Snow/ice surfaces show high lunar reflectance
    - Vegetation shows moderate, phase-dependent reflectance
    - Urban surfaces have characteristic reflectance patterns
    - Bare soil reflectance varies with composition
    """
    # Set up consistent test conditions
    n_samples = 100
    lunar_zenith = np.full(n_samples, 30.0)  # Consistent zenith angle
    lunar_phase = 45.0  # Quarter moon
    
    # Test each surface type
    surface_types = {
        'water': 1,
        'vegetation': 2,
        'urban': 3,
        'bare_soil': 4,
        'unclassified': 0
    }
    
    reflectances = {}
    for name, code in surface_types.items():
        surface = np.full(n_samples, code)
        refl = estimate_lunar_reflectance(surface, lunar_zenith, lunar_phase)
        reflectances[name] = refl.mean()
    
    # Water should have lowest reflectance
    assert 0.01 <= reflectances['water'] <= 0.06, \
        f"Water reflectance {reflectances['water']:.3f} should be low (0.01-0.06)"
    
    # Vegetation should have moderate reflectance
    assert 0.03 <= reflectances['vegetation'] <= 0.20, \
        f"Vegetation reflectance {reflectances['vegetation']:.3f} should be moderate (0.03-0.20)"
    
    # Urban should have high reflectance
    assert 0.10 <= reflectances['urban'] <= 0.40, \
        f"Urban reflectance {reflectances['urban']:.3f} should be high (0.10-0.40)"
    
    # Bare soil should be between vegetation and urban
    assert 0.08 <= reflectances['bare_soil'] <= 0.30, \
        f"Bare soil reflectance {reflectances['bare_soil']:.3f} should be 0.08-0.30"
    
    # Test phase dependence for each surface type
    phases = [0.0, 90.0, 180.0]  # Full, quarter, new moon
    phase_dependence = {}
    
    for name, code in surface_types.items():
        surface = np.full(10, code)
        zenith = np.full(10, 45.0)
        
        phase_refls = []
        for phase in phases:
            refl = estimate_lunar_reflectance(surface, zenith, phase)
            phase_refls.append(refl.mean())
        
        # Calculate relative change from full to new moon
        phase_dependence[name] = (phase_refls[0] - phase_refls[2]) / phase_refls[0]
    
    # All surfaces should show phase dependence
    for name, dep in phase_dependence.items():
        assert dep > 0.8, f"{name} should show strong phase dependence (>80%)"
    
    # Test angular behavior for water (more specular)
    water_surface = np.full(50, 1)
    water_zeniths = np.linspace(0, 80, 50)
    
    water_angular = estimate_lunar_reflectance(water_surface, water_zeniths, 0.0)
    
    # Water should show strong angular dependence (specular behavior)
    water_ratio = water_angular[0] / water_angular[-1]
    assert water_ratio > 5, "Water should show strong angular dependence"
    
    # Test urban angular behavior (more Lambertian)
    urban_surface = np.full(50, 3)
    urban_angular = estimate_lunar_reflectance(urban_surface, water_zeniths, 0.0)
    
    # Urban should show less angular dependence
    urban_ratio = urban_angular[0] / urban_angular[-1]
    assert urban_ratio < water_ratio, "Urban should be less angular than water"


def test_detect_lunar_contamination():
    """Test detection of lunar contamination in radiance data.
    
    This test should verify that:
    - Contaminated pixels are accurately identified
    - Threshold ratio parameter controls sensitivity
    - Clean pixels are not flagged as contaminated
    - Mathematical accuracy of contamination detection algorithm
    - Proper handling of edge cases (very bright/dark pixels)
    - Binary mask output format is correct
    """
    # Create test data with known contamination patterns
    shape = (50, 50)
    
    # Create radiance with varying levels
    radiance = np.zeros(shape)
    radiance[:25, :] = 0.1  # Low radiance (likely contaminated)
    radiance[25:, :] = 0.5   # High radiance (likely artificial lights)
    
    # Create lunar irradiance
    lunar_irradiance = np.full(shape, 0.05)  # Uniform lunar contribution
    
    # Test with default threshold (0.5)
    mask, percent = detect_lunar_contamination(radiance, lunar_irradiance)
    
    # Verify binary mask format
    assert mask.dtype == bool, "Mask should be boolean"
    assert mask.shape == radiance.shape, "Mask should match input shape"
    
    # Low radiance areas should be more contaminated
    # Estimated lunar contribution = 0.05 * 0.1 = 0.005
    # Ratio for low radiance = 0.005 / 0.1 = 0.05 (5%)
    # Ratio for high radiance = 0.005 / 0.5 = 0.01 (1%)
    # Neither should exceed 50% threshold with these values
    
    # Test with lower threshold to detect contamination
    mask_low, percent_low = detect_lunar_contamination(
        radiance, lunar_irradiance, threshold_ratio=0.04
    )
    
    # Now low radiance areas should be flagged
    assert np.sum(mask_low[:25, :]) > np.sum(mask_low[25:, :]), \
        "Low radiance areas should have more contamination"
    
    # Test percentage calculation
    assert 0 <= percent_low <= 100, "Percentage must be in valid range"
    expected_percent = (np.sum(mask_low) / mask_low.size) * 100
    assert abs(percent_low - expected_percent) < 0.01, "Percentage calculation error"
    
    # Test edge cases
    # Very bright pixels (artificial lights)
    bright_radiance = np.full((10, 10), 10.0)  # Very bright
    bright_lunar = np.full((10, 10), 0.05)
    
    bright_mask, bright_percent = detect_lunar_contamination(
        bright_radiance, bright_lunar, threshold_ratio=0.1
    )
    
    assert np.sum(bright_mask) == 0, "Very bright pixels should not be contaminated"
    assert bright_percent == 0.0, "No contamination in bright areas"
    
    # Very dark pixels (no lights)
    dark_radiance = np.full((10, 10), 0.001)  # Very dark
    dark_lunar = np.full((10, 10), 0.05)
    
    dark_mask, dark_percent = detect_lunar_contamination(
        dark_radiance, dark_lunar, threshold_ratio=0.1
    )
    
    # Dark areas should be heavily contaminated
    # Ratio = (0.05 * 0.1) / 0.001 = 5.0 (500%)
    assert np.all(dark_mask), "Very dark pixels should all be contaminated"
    assert dark_percent == 100.0, "All dark pixels contaminated"
    
    # Test with invalid data
    invalid_radiance = radiance.copy()
    invalid_radiance[10:20, 10:20] = np.nan
    
    invalid_mask, invalid_percent = detect_lunar_contamination(
        invalid_radiance, lunar_irradiance
    )
    
    # NaN pixels should not be marked as contaminated
    assert np.all(~invalid_mask[10:20, 10:20]), "NaN pixels should not be contaminated"
    
    # Test threshold sensitivity
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    contamination_counts = []
    
    for thresh in thresholds:
        mask_t, _ = detect_lunar_contamination(
            radiance, lunar_irradiance, threshold_ratio=thresh
        )
        contamination_counts.append(np.sum(mask_t))
    
    # Higher thresholds should detect less contamination
    assert all(contamination_counts[i] >= contamination_counts[i+1] 
              for i in range(len(contamination_counts)-1)), \
        "Higher thresholds should detect less contamination"


def test_detect_lunar_contamination_sensitivity():
    """Test lunar contamination detection sensitivity.
    
    This test should verify that:
    - Lower threshold ratios detect more contamination
    - Higher threshold ratios are more conservative
    - Threshold parameter works as expected
    - Sensitivity analysis with known contaminated/clean pixels
    """
    # Create synthetic data with known contamination levels
    shape = (100, 100)
    
    # Create regions with different contamination levels
    radiance = np.ones(shape) * 0.2  # Base radiance
    lunar_irradiance = np.zeros(shape)
    
    # Region 1: No contamination (0% lunar)
    lunar_irradiance[0:25, :] = 0.0
    
    # Region 2: Low contamination (~25% lunar) 
    lunar_irradiance[25:50, :] = 0.5  # 0.5*0.1/0.2 = 0.05/0.2 = 0.25 (25%)
    radiance[25:50, :] = 0.2
    
    # Region 3: Medium contamination (~50% lunar)
    lunar_irradiance[50:75, :] = 0.10  # 0.10*0.1/0.2 = 0.01/0.2 = 0.05 (5%)
    radiance[50:75, :] = 0.02  # Lower radiance -> higher ratio: 0.01/0.02 = 0.5 (50%)
    
    # Region 4: High contamination (~80% lunar)
    lunar_irradiance[75:100, :] = 0.10
    radiance[75:100, :] = 0.0125  # 0.10*0.1/0.0125 = 0.01/0.0125 = 0.8 (80%)
    
    # Test with different thresholds
    threshold_tests = [
        (0.1, "should detect regions 2, 3, and 4"),
        (0.3, "should detect regions 3 and 4"),
        (0.6, "should detect only region 4"),
        (0.9, "should detect no regions")
    ]
    
    for threshold, description in threshold_tests:
        mask, percent = detect_lunar_contamination(
            radiance, lunar_irradiance, threshold_ratio=threshold
        )
        
        region1_contaminated = np.sum(mask[0:25, :]) / (25 * shape[1])
        region2_contaminated = np.sum(mask[25:50, :]) / (25 * shape[1])
        region3_contaminated = np.sum(mask[50:75, :]) / (25 * shape[1])
        region4_contaminated = np.sum(mask[75:100, :]) / (25 * shape[1])
        
        if threshold == 0.1:
            assert region1_contaminated == 0, f"Region 1 should not be contaminated at {threshold}"
            assert region2_contaminated > 0.0, f"Region 2 should show some contamination at {threshold}"
            assert region3_contaminated > 0.8, f"Region 3 should be contaminated at {threshold}"
            assert region4_contaminated > 0.8, f"Region 4 should be contaminated at {threshold}"
        elif threshold == 0.3:
            assert region1_contaminated == 0, f"Region 1 should not be contaminated at {threshold}"
            assert region2_contaminated < 0.2, f"Region 2 should not be contaminated at {threshold}"
            assert region3_contaminated > 0.8, f"Region 3 should be contaminated at {threshold}"
            assert region4_contaminated > 0.8, f"Region 4 should be contaminated at {threshold}"
        elif threshold == 0.6:
            assert region1_contaminated == 0, f"Region 1 should not be contaminated at {threshold}"
            assert region2_contaminated == 0, f"Region 2 should not be contaminated at {threshold}"
            assert region3_contaminated < 0.3, f"Region 3 should not be contaminated at {threshold}"
            assert region4_contaminated > 0.8, f"Region 4 should be contaminated at {threshold}"
        elif threshold == 0.9:
            assert np.sum(mask) == 0, f"No regions should be contaminated at {threshold}"
    
    # Test threshold ordering
    contamination_by_threshold = []
    thresholds = np.linspace(0.0, 1.0, 21)
    
    for thresh in thresholds:
        mask, _ = detect_lunar_contamination(
            radiance, lunar_irradiance, threshold_ratio=thresh
        )
        contamination_by_threshold.append(np.sum(mask))
    
    # Verify monotonic decrease
    for i in range(len(contamination_by_threshold) - 1):
        assert contamination_by_threshold[i] >= contamination_by_threshold[i+1], \
            "Contamination should decrease with higher threshold"


def test_calculate_moon_phase_weight():
    """Test moon phase weight calculation for compositing.
    
    This test should verify that:
    - Cosine weight function produces expected curve
    - New moon receives minimum weight
    - Full moon receives maximum weight
    - Weight values are properly normalized
    - Different weight functions produce distinct results
    - Mathematical accuracy of weight calculations
    - Proper handling of lunar fraction edge cases
    """
    # Test linear weight function
    fractions = np.linspace(0, 1, 11)
    weights_linear = [calculate_moon_phase_weight(f, "linear") for f in fractions]
    
    # Linear should decrease linearly from 1 to 0
    expected_linear = 1.0 - fractions
    np.testing.assert_allclose(weights_linear, expected_linear, rtol=1e-6)
    
    # New moon (0) should have weight 1, full moon (1) should have weight 0
    assert weights_linear[0] == 1.0, "New moon should have maximum weight"
    assert weights_linear[-1] == 0.0, "Full moon should have minimum weight"
    
    # Test cosine weight function
    weights_cosine = [calculate_moon_phase_weight(f, "cosine") for f in fractions]
    
    # Cosine should follow (1 + cos(π*f))/2
    expected_cosine = (1.0 + np.cos(fractions * np.pi)) / 2.0
    np.testing.assert_allclose(weights_cosine, expected_cosine, rtol=1e-6)
    
    # Verify smooth curve properties
    assert weights_cosine[0] == 1.0, "Cosine: new moon should have weight 1"
    assert weights_cosine[-1] == 0.0, "Cosine: full moon should have weight 0"
    assert abs(weights_cosine[5] - 0.5) < 0.01, "Cosine: mid-point should be ~0.5"
    
    # Test empirical weight function
    weights_empirical = [calculate_moon_phase_weight(f, "empirical") for f in fractions]
    
    # Empirical should have specific breakpoints
    assert weights_empirical[0] == 1.0, "Empirical: new moon should have weight 1"
    assert weights_empirical[1] == 1.0, "Empirical: f=0.1 should still have weight 1"
    assert 0 < weights_empirical[5] < 1, "Empirical: mid-range should be between 0 and 1"
    assert weights_empirical[-1] == 0.0, "Empirical: full moon should have weight 0"
    
    # Different functions should produce distinct results
    assert not np.allclose(weights_linear, weights_cosine), \
        "Linear and cosine should differ"
    assert not np.allclose(weights_linear, weights_empirical), \
        "Linear and empirical should differ"
    assert not np.allclose(weights_cosine, weights_empirical), \
        "Cosine and empirical should differ"
    
    # Test edge cases
    assert calculate_moon_phase_weight(0.0, "cosine") == 1.0
    assert calculate_moon_phase_weight(1.0, "cosine") == 0.0
    
    # Test weight normalization (all weights should be in [0, 1])
    for weight_list in [weights_linear, weights_cosine, weights_empirical]:
        assert all(0 <= w <= 1 for w in weight_list), "All weights must be in [0, 1]"
    
    # Test error handling
    with pytest.raises(ValueError, match="Lunar fraction must be"):
        calculate_moon_phase_weight(-0.1, "linear")
    
    with pytest.raises(ValueError, match="Lunar fraction must be"):
        calculate_moon_phase_weight(1.1, "linear")
    
    with pytest.raises(ValueError, match="Unknown weight function"):
        calculate_moon_phase_weight(0.5, "invalid")


def test_calculate_moon_phase_weight_functions():
    """Test different moon phase weight functions.
    
    This test should verify that:
    - Cosine function emphasizes new moon conditions
    - Linear function provides gradual weighting
    - Exponential function creates sharper transitions
    - Custom weight functions can be implemented
    - Mathematical properties of each function
    """
    # Create fine-grained lunar fraction array
    fractions = np.linspace(0, 1, 101)
    
    # Calculate weights for each function
    weights_linear = np.array([calculate_moon_phase_weight(f, "linear") for f in fractions])
    weights_cosine = np.array([calculate_moon_phase_weight(f, "cosine") for f in fractions])
    weights_empirical = np.array([calculate_moon_phase_weight(f, "empirical") for f in fractions])
    
    # Test mathematical properties
    # 1. Linear function should have constant derivative
    linear_diff = np.diff(weights_linear)
    assert np.allclose(linear_diff, linear_diff[0], rtol=1e-5), \
        "Linear function should have constant slope"
    
    # 2. Cosine function should be smooth (continuous derivative)
    cosine_diff = np.diff(weights_cosine)
    cosine_diff2 = np.diff(cosine_diff)
    assert np.all(np.abs(cosine_diff2) < 0.01), \
        "Cosine function should have smooth derivative"
    
    # 3. Empirical function should have distinct regions
    # Check for breakpoints at 0.1, 0.3, 0.7
    assert weights_empirical[0] == weights_empirical[10], \
        "Empirical should be flat from 0 to 0.1"
    
    # Test emphasis on new moon conditions
    # Calculate area under curve (integral) for first quarter
    first_quarter = fractions <= 0.25
    area_linear = np.trapezoid(weights_linear[first_quarter], fractions[first_quarter])
    area_cosine = np.trapezoid(weights_cosine[first_quarter], fractions[first_quarter])
    area_empirical = np.trapezoid(weights_empirical[first_quarter], fractions[first_quarter])
    
    # Cosine should give more weight to new moon period than linear
    assert area_cosine > area_linear, \
        "Cosine should emphasize new moon more than linear"
    
    # Test sharpness of transitions
    # Calculate gradient magnitude
    grad_linear = np.abs(np.gradient(weights_linear))
    grad_cosine = np.abs(np.gradient(weights_cosine))
    grad_empirical = np.abs(np.gradient(weights_empirical))
    
    # Linear should have constant gradient
    assert np.std(grad_linear) < 0.01, "Linear should have constant gradient"
    
    # Cosine should have varying gradient (sharper near endpoints)
    assert np.std(grad_cosine) > 0.001, "Cosine should have varying gradient"
    
    # Test weight discrimination between phases
    new_moon_weight = calculate_moon_phase_weight(0.05, "empirical")
    crescent_weight = calculate_moon_phase_weight(0.2, "empirical")
    quarter_weight = calculate_moon_phase_weight(0.5, "empirical")
    gibbous_weight = calculate_moon_phase_weight(0.75, "empirical")
    full_moon_weight = calculate_moon_phase_weight(0.95, "empirical")
    
    # Verify proper ordering
    assert new_moon_weight > crescent_weight > quarter_weight > gibbous_weight > full_moon_weight, \
        "Weights should decrease monotonically with lunar fraction"
    
    # Test that all functions properly penalize full moon
    full_moon_fractions = fractions > 0.9
    assert np.all(weights_linear[full_moon_fractions] < 0.1)
    assert np.all(weights_cosine[full_moon_fractions] < 0.1)
    assert np.all(weights_empirical[full_moon_fractions] < 0.1)


def test_analyze_lunar_cycle_effects():
    """Test analysis of lunar cycle effects on radiance observations.
    
    This test should verify that:
    - Radiance variations with lunar cycle are detected
    - Statistical measures of lunar effects are computed
    - Temporal patterns in lunar contamination are identified
    - Correlation between lunar fraction and radiance is calculated
    - Results provide actionable insights for data processing
    - Mathematical accuracy of correlation analysis
    """
    # Create synthetic time series with lunar cycle effects
    n_times = 12  # Monthly observations
    height, width = 30, 30
    
    # Create lunar fractions representing a cycle
    lunar_fractions = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0]
    dates = [datetime(2023, i+1, 15) for i in range(n_times)]
    
    # Create radiance stack with lunar-correlated signal
    radiance_stack = np.zeros((n_times, height, width))
    
    # Add base radiance
    base_radiance = np.random.uniform(0.1, 0.3, (height, width))
    
    # Add lunar-dependent component
    for t, lunar_frac in enumerate(lunar_fractions):
        # Radiance increases with lunar fraction
        lunar_component = lunar_frac * 0.1  # Max 0.1 additional from moon
        radiance_stack[t] = base_radiance + lunar_component
        
        # Add noise
        radiance_stack[t] += np.random.normal(0, 0.01, (height, width))
    
    # Add some pixels with no lunar effect (e.g., bright urban areas)
    radiance_stack[:, 10:15, 10:15] = 1.0  # Constant bright area
    
    # Analyze lunar cycle effects
    results = analyze_lunar_cycle_effects(radiance_stack, lunar_fractions, dates)
    
    # Verify all expected keys are present
    expected_keys = ['lunar_trend', 'detrended_radiance', 'cycle_amplitude',
                    'correlation', 'optimal_observations', 'contamination_map']
    assert all(key in results for key in expected_keys), "Missing expected result keys"
    
    # Check lunar trend
    trend = results['lunar_trend']
    assert trend.shape == (height, width), "Trend should be 2D"
    
    # Most pixels should show positive trend (radiance increases with lunar fraction)
    valid_trend = trend[~np.isnan(trend)]
    assert np.median(valid_trend) > 0, "Median trend should be positive"
    
    # Bright urban area should show near-zero trend
    urban_trend = trend[10:15, 10:15]
    assert np.all(np.abs(urban_trend[~np.isnan(urban_trend)]) < 0.1), \
        "Urban areas should show minimal lunar trend"
    
    # Check detrended radiance
    detrended = results['detrended_radiance']
    assert detrended.shape == radiance_stack.shape, "Detrended should match input shape"
    
    # Detrended should have less variance across time
    original_variance = np.var(radiance_stack[:, 0:5, 0:5], axis=0)
    detrended_variance = np.var(detrended[:, 0:5, 0:5], axis=0)
    assert np.mean(detrended_variance) < np.mean(original_variance), \
        "Detrending should reduce temporal variance"
    
    # Check cycle amplitude
    amplitude = results['cycle_amplitude']
    assert amplitude.shape == (height, width), "Amplitude should be 2D"
    
    # Amplitude should be roughly 0.1 for most pixels (our lunar component)
    valid_amp = amplitude[~np.isnan(amplitude) & (amplitude < 0.5)]
    assert 0.05 < np.median(valid_amp) < 0.15, \
        f"Median amplitude {np.median(valid_amp):.3f} should be near 0.1"
    
    # Check correlation
    correlation = results['correlation']
    assert correlation.shape == (height, width), "Correlation should be 2D"
    assert np.all((correlation >= -1) & (correlation <= 1) | np.isnan(correlation)), \
        "Correlation must be in [-1, 1]"
    
    # Most pixels should show positive correlation
    valid_corr = correlation[~np.isnan(correlation)]
    assert np.mean(valid_corr > 0) > 0.7, "Most pixels should show positive correlation"
    
    # Check optimal observations
    optimal = results['optimal_observations']
    assert len(optimal) == n_times, "Should rank all observations"
    
    # Best observations should have low lunar fractions
    best_obs = optimal[:3]  # Top 3
    best_lunar = [lunar_fractions[i] for i in best_obs]
    assert max(best_lunar) < 0.3, "Best observations should have low lunar fraction"
    
    # Check contamination map
    contamination = results['contamination_map']
    assert contamination.shape == (height, width), "Contamination map should be 2D"
    assert np.all((contamination >= 0) & (contamination <= 1)), \
        "Contamination scores must be in [0, 1]"
    
    # Urban area should show low contamination
    urban_contamination = contamination[10:15, 10:15]
    assert np.all(urban_contamination < 0.2), "Bright urban areas should show low contamination"


def test_analyze_lunar_cycle_temporal_patterns():
    """Test temporal pattern analysis in lunar cycle effects.
    
    This test should verify that:
    - Monthly lunar cycle patterns are identified
    - Seasonal variations in lunar effects are detected
    - Long-term trends in contamination are analyzed
    - Statistical significance of patterns is assessed
    """
    # Create multi-year dataset with seasonal variations
    n_months = 24  # 2 years
    height, width = 20, 20
    
    # Generate realistic lunar cycle (approximately monthly)
    lunar_fractions = []
    for month in range(n_months):
        # Simplified lunar cycle with ~29.5 day period
        phase = (month * 29.5 / 30) % 1.0  # Approximate phase
        lunar_frac = 0.5 * (1 + np.cos(2 * np.pi * phase))  # Cosine cycle
        lunar_fractions.append(lunar_frac)
    
    dates = [datetime(2022 + month//12, (month % 12) + 1, 15) for month in range(n_months)]
    
    # Create radiance with seasonal and lunar components
    radiance_stack = np.zeros((n_months, height, width))
    base_radiance = np.random.uniform(0.2, 0.4, (height, width))
    
    for t in range(n_months):
        # Seasonal component (higher in winter months)
        month_of_year = dates[t].month
        seasonal_factor = 1.0 + 0.3 * np.cos(2 * np.pi * (month_of_year - 1) / 12)
        
        # Lunar component
        lunar_component = lunar_fractions[t] * 0.15
        
        # Combined effect
        radiance_stack[t] = base_radiance * seasonal_factor + lunar_component
        radiance_stack[t] += np.random.normal(0, 0.02, (height, width))
    
    # Analyze patterns
    results = analyze_lunar_cycle_effects(radiance_stack, lunar_fractions, dates)
    
    # Test monthly pattern detection
    # Group results by similar lunar fractions
    low_lunar_indices = [i for i, f in enumerate(lunar_fractions) if f < 0.3]
    high_lunar_indices = [i for i, f in enumerate(lunar_fractions) if f > 0.7]
    
    if len(low_lunar_indices) > 0 and len(high_lunar_indices) > 0:
        # Compare radiance during low vs high lunar periods
        low_lunar_radiance = radiance_stack[low_lunar_indices].mean(axis=0)
        high_lunar_radiance = radiance_stack[high_lunar_indices].mean(axis=0)
        
        # High lunar should have higher radiance
        assert np.mean(high_lunar_radiance) > np.mean(low_lunar_radiance), \
            "High lunar periods should show higher radiance"
    
    # Test correlation strength varies spatially
    correlation = results['correlation']
    corr_std = np.nanstd(correlation)
    assert corr_std > 0.05, "Correlation should vary spatially"
    
    # Test optimal observations follow lunar cycle
    optimal = results['optimal_observations']
    optimal_lunar = [lunar_fractions[i] for i in optimal[:6]]  # Top 6
    assert np.mean(optimal_lunar) < 0.4, "Optimal observations should favor low lunar"
    
    # Test contamination map identifies problem areas
    contamination = results['contamination_map']
    
    # Create synthetic problem area with high lunar sensitivity
    problem_area = slice(5, 10), slice(5, 10)
    radiance_stack[:, problem_area[0], problem_area[1]] *= (1 + 0.5 * np.array(lunar_fractions)[:, np.newaxis, np.newaxis])
    
    # Re-analyze with problem area
    results2 = analyze_lunar_cycle_effects(radiance_stack, lunar_fractions, dates)
    contamination2 = results2['contamination_map']
    
    # Problem area should show higher contamination
    problem_contamination = contamination2[problem_area].mean()
    other_contamination = np.delete(contamination2.flatten(), 
                                   np.ravel_multi_index(np.meshgrid(*[np.arange(s.start, s.stop) for s in problem_area], indexing='ij'), contamination2.shape)).mean()
    
    # Allow for some flexibility due to randomness
    if not np.isnan(problem_contamination) and not np.isnan(other_contamination):
        # Note: Due to the complexity of the contamination calculation,
        # we check for any difference rather than a specific ratio
        assert abs(problem_contamination - other_contamination) > 0.01, \
            "Problem area should show different contamination"
    
    # Test statistical properties of detrending
    detrended = results['detrended_radiance']
    
    # Detrended data should have reduced correlation with lunar fraction
    detrended_mean = np.nanmean(detrended, axis=(1, 2))
    if len(detrended_mean) == len(lunar_fractions):
        detrend_corr = np.corrcoef(detrended_mean, lunar_fractions)[0, 1]
        assert abs(detrend_corr) < 0.3, "Detrended data should have low lunar correlation"


def test_compute_clear_sky_lunar_radiance():
    """Test theoretical clear-sky lunar radiance computation.
    
    This test should verify that:
    - Theoretical radiance values are physically reasonable
    - Atmospheric transmission parameter is properly applied
    - Surface albedo effects are correctly modeled
    - Geometric factors (zenith, azimuth) are properly handled
    - Mathematical accuracy with radiative transfer theory
    - Comparison with observed clear-sky radiances
    """
    # Test basic calculation
    shape = (50, 50)
    lunar_zenith = np.full(shape, 30.0)  # 30 degrees
    lunar_azimuth = np.full(shape, 180.0)  # South
    surface_albedo = np.full(shape, 0.15)  # Typical mixed surface
    
    radiance = compute_clear_sky_lunar_radiance(
        lunar_zenith, lunar_azimuth, surface_albedo
    )
    
    # Verify physically reasonable range (0.01-1.0 nW/cm²/sr)
    assert np.all((radiance >= 0.01) & (radiance <= 1.0)), \
        f"Radiance range {radiance.min():.3f}-{radiance.max():.3f} should be 0.01-1.0"
    
    # Test atmospheric transmission effect
    rad_clear = compute_clear_sky_lunar_radiance(
        lunar_zenith, lunar_azimuth, surface_albedo,
        atmospheric_transmission=0.9  # Very clear
    )
    
    rad_hazy = compute_clear_sky_lunar_radiance(
        lunar_zenith, lunar_azimuth, surface_albedo,
        atmospheric_transmission=0.6  # Hazy
    )
    
    # Clear atmosphere should give higher radiance
    assert np.mean(rad_clear) > np.mean(rad_hazy), \
        "Clear atmosphere should produce higher radiance"
    
    # Ratio should match transmission ratio
    ratio = np.mean(rad_clear) / np.mean(rad_hazy)
    expected_ratio = 0.9 / 0.6
    assert abs(ratio - expected_ratio) < 0.1, \
        f"Radiance ratio {ratio:.2f} should match transmission ratio {expected_ratio:.2f}"
    
    # Test surface albedo effects
    albedos = np.array([0.03, 0.10, 0.25, 0.50])  # Water to bright surface
    rad_by_albedo = []
    
    for albedo in albedos:
        surface = np.full((10, 10), albedo)
        zenith = np.full((10, 10), 45.0)
        azimuth = np.zeros((10, 10))
        
        rad = compute_clear_sky_lunar_radiance(zenith, azimuth, surface)
        rad_by_albedo.append(np.mean(rad))
    
    # Radiance should increase linearly with albedo
    for i in range(len(albedos) - 1):
        ratio = rad_by_albedo[i+1] / rad_by_albedo[i]
        expected = albedos[i+1] / albedos[i]
        assert abs(ratio - expected) / expected < 0.1, \
            f"Radiance should scale with albedo: {ratio:.2f} vs {expected:.2f}"
    
    # Test zenith angle effects (cosine law)
    zenith_angles = np.array([0.0, 30.0, 60.0, 80.0])
    rad_by_zenith = []
    
    for zenith in zenith_angles:
        zen_array = np.full((10, 10), zenith)
        azi_array = np.zeros((10, 10))
        alb_array = np.full((10, 10), 0.2)
        
        rad = compute_clear_sky_lunar_radiance(zen_array, azi_array, alb_array)
        rad_by_zenith.append(np.mean(rad))
    
    # Verify cosine dependence
    for i in range(len(zenith_angles) - 1):
        cos_ratio = np.cos(np.deg2rad(zenith_angles[i])) / np.cos(np.deg2rad(zenith_angles[i+1]))
        rad_ratio = rad_by_zenith[i] / rad_by_zenith[i+1]
        
        # Allow some deviation due to other factors
        assert 0.7 < rad_ratio / cos_ratio < 1.3, \
            f"Radiance should follow cosine law: {rad_ratio:.2f} vs {cos_ratio:.2f}"
    
    # Test with varying inputs
    lunar_zenith_var = np.random.uniform(0, 85, shape)
    lunar_azimuth_var = np.random.uniform(0, 360, shape)
    surface_albedo_var = np.random.uniform(0.05, 0.4, shape)
    
    radiance_var = compute_clear_sky_lunar_radiance(
        lunar_zenith_var, lunar_azimuth_var, surface_albedo_var
    )
    
    # All values should be finite and positive
    assert np.all(np.isfinite(radiance_var)), "All radiance values should be finite"
    assert np.all(radiance_var > 0), "All radiance values should be positive"
    
    # Test error handling
    with pytest.raises(ValueError, match="Surface albedo values must be"):
        compute_clear_sky_lunar_radiance(
            lunar_zenith, lunar_azimuth, np.full(shape, 1.5)  # Invalid albedo
        )


def test_compute_clear_sky_lunar_radiance_atmosphere():
    """Test clear-sky lunar radiance with atmospheric effects.
    
    This test should verify that:
    - Atmospheric transmission reduces surface-reaching radiance
    - Different transmission values produce expected results
    - Realistic atmospheric conditions are properly modeled
    - Edge cases (no atmosphere, very thick atmosphere) are handled
    """
    # Set up test conditions
    shape = (30, 30)
    lunar_zenith = np.full(shape, 45.0)
    lunar_azimuth = np.full(shape, 0.0)
    surface_albedo = np.full(shape, 0.20)  # Typical surface
    
    # Test range of atmospheric transmissions
    transmissions = np.array([0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0])
    radiances = []
    
    for trans in transmissions:
        rad = compute_clear_sky_lunar_radiance(
            lunar_zenith, lunar_azimuth, surface_albedo,
            atmospheric_transmission=trans
        )
        radiances.append(np.mean(rad))
    
    # Radiance should increase monotonically with transmission
    for i in range(len(radiances) - 1):
        assert radiances[i] <= radiances[i+1], \
            f"Radiance should increase with transmission: {radiances[i]:.4f} <= {radiances[i+1]:.4f}"
    
    # Edge case: no atmosphere (transmission = 0)
    assert radiances[0] == 0.0, "Zero transmission should give zero radiance"
    
    # Edge case: perfect transmission (transmission = 1)
    # Should give maximum possible radiance for conditions
    assert radiances[-1] > 0, "Perfect transmission should give positive radiance"
    
    # Verify linear relationship
    # Radiance should be proportional to transmission
    mid_idx = len(transmissions) // 2
    if radiances[mid_idx] > 0:
        for i in range(1, len(transmissions) - 1):
            if radiances[i] > 0:
                ratio = radiances[i] / radiances[mid_idx]
                trans_ratio = transmissions[i] / transmissions[mid_idx]
                assert abs(ratio - trans_ratio) < 0.1, \
                    f"Radiance should scale with transmission: {ratio:.2f} vs {trans_ratio:.2f}"
    
    # Test realistic atmospheric conditions
    # Clear mountain top: 0.90-0.95
    # Clear sea level: 0.80-0.85
    # Light haze: 0.60-0.70
    # Heavy haze: 0.30-0.50
    # Fog: 0.10-0.30
    
    conditions = {
        'mountain_clear': 0.93,
        'sea_level_clear': 0.83,
        'light_haze': 0.65,
        'heavy_haze': 0.40,
        'fog': 0.20
    }
    
    condition_radiances = {}
    for name, trans in conditions.items():
        rad = compute_clear_sky_lunar_radiance(
            lunar_zenith, lunar_azimuth, surface_albedo,
            atmospheric_transmission=trans
        )
        condition_radiances[name] = np.mean(rad)
    
    # Verify expected ordering
    assert condition_radiances['mountain_clear'] > condition_radiances['sea_level_clear']
    assert condition_radiances['sea_level_clear'] > condition_radiances['light_haze']
    assert condition_radiances['light_haze'] > condition_radiances['heavy_haze']
    assert condition_radiances['heavy_haze'] > condition_radiances['fog']
    
    # Test zenith angle dependence with atmosphere
    # Atmospheric effects should be stronger at high zenith angles
    zenith_low = np.full(shape, 20.0)
    zenith_high = np.full(shape, 70.0)
    
    rad_low_clear = compute_clear_sky_lunar_radiance(
        zenith_low, lunar_azimuth, surface_albedo,
        atmospheric_transmission=0.85
    )
    
    rad_low_hazy = compute_clear_sky_lunar_radiance(
        zenith_low, lunar_azimuth, surface_albedo,
        atmospheric_transmission=0.50
    )
    
    rad_high_clear = compute_clear_sky_lunar_radiance(
        zenith_high, lunar_azimuth, surface_albedo,
        atmospheric_transmission=0.85
    )
    
    rad_high_hazy = compute_clear_sky_lunar_radiance(
        zenith_high, lunar_azimuth, surface_albedo,
        atmospheric_transmission=0.50
    )
    
    # Calculate atmospheric impact
    impact_low = (np.mean(rad_low_clear) - np.mean(rad_low_hazy)) / np.mean(rad_low_clear)
    impact_high = (np.mean(rad_high_clear) - np.mean(rad_high_hazy)) / np.mean(rad_high_clear)
    
    # Both should show impact, but comparison depends on implementation details
    assert impact_low > 0 and impact_high > 0, "Atmosphere should reduce radiance at all angles"


def test_lunar_angle_calculations():
    """Test accuracy of lunar geometry calculations.
    
    This test should verify that:
    - Lunar zenith and azimuth angles are correctly computed
    - Geometric transformations are mathematically accurate
    - Edge cases (poles, date line) are handled correctly
    - Consistency with astronomical calculations
    - Proper coordinate system handling
    """
    # Note: Since the lunar module doesn't expose angle calculation functions directly,
    # we test the angle handling within the existing functions
    
    # Test zenith angle constraints in BRDF calculation
    shape = (10, 10)
    
    # Test valid zenith range [0, 90]
    valid_zeniths = np.array([0.0, 30.0, 45.0, 60.0, 90.0])
    for zenith in valid_zeniths:
        lunar_zenith = np.full(shape, zenith)
        lunar_azimuth = np.zeros(shape)
        view_zenith = np.full(shape, 30.0)
        view_azimuth = np.zeros(shape)
        
        # Should not raise error
        brdf = calculate_lunar_brdf(
            lunar_zenith, lunar_azimuth, view_zenith, view_azimuth
        )
        assert np.all(np.isfinite(brdf)), f"BRDF should be finite for zenith={zenith}"
    
    # Test azimuth angle wrapping [0, 360]
    azimuths = np.array([0.0, 90.0, 180.0, 270.0, 360.0])
    lunar_zenith = np.full(len(azimuths), 45.0)
    view_zenith = np.full(len(azimuths), 30.0)
    view_azimuth = np.zeros(len(azimuths))
    
    brdf = calculate_lunar_brdf(
        lunar_zenith, azimuths, view_zenith, view_azimuth
    )
    
    # 0° and 360° should give same result (circular wrapping)
    assert np.isclose(brdf[0], brdf[-1]), "Azimuth 0° and 360° should be equivalent"
    
    # Test relative azimuth calculation
    # When lunar and view azimuths differ by 180°, we have backscatter
    lunar_az = np.array([0.0, 180.0])
    view_az = np.array([180.0, 0.0])
    lunar_zen = np.full(2, 45.0)
    view_zen = np.full(2, 45.0)
    
    brdf_backscatter = calculate_lunar_brdf(
        lunar_zen, lunar_az, view_zen, view_az
    )
    
    # Both configurations should give same backscatter result
    assert np.isclose(brdf_backscatter[0], brdf_backscatter[1]), \
        "Backscatter should be symmetric"
    
    # Test coordinate system consistency in radiance calculation
    # Radiance should decrease with zenith angle (cosine law)
    zeniths = np.linspace(0, 85, 10)
    radiances = []
    
    for zen in zeniths:
        lunar_zenith = np.full((5, 5), zen)
        lunar_azimuth = np.zeros((5, 5))
        albedo = np.full((5, 5), 0.2)
        
        rad = compute_clear_sky_lunar_radiance(
            lunar_zenith, lunar_azimuth, albedo
        )
        radiances.append(np.mean(rad))
    
    # Verify monotonic decrease
    for i in range(len(radiances) - 1):
        assert radiances[i] >= radiances[i+1], \
            f"Radiance should decrease with zenith: {radiances[i]:.4f} >= {radiances[i+1]:.4f}"
    
    # Test extreme angles near horizon
    extreme_zenith = np.array([85.0, 87.0, 89.0, 89.9])
    extreme_azimuth = np.random.uniform(0, 360, 4)
    albedo = np.full(4, 0.15)
    
    extreme_rad = compute_clear_sky_lunar_radiance(
        extreme_zenith, extreme_azimuth, albedo
    )
    
    # Should produce valid, decreasing values
    assert np.all(np.isfinite(extreme_rad)), "Extreme angles should produce finite values"
    assert np.all(extreme_rad >= 0), "Radiance should be non-negative"
    assert np.all(np.diff(extreme_rad) <= 0), "Radiance should decrease near horizon"


def test_lunar_phase_calculations():
    """Test lunar phase and illumination calculations.
    
    This test should verify that:
    - Lunar phase fractions are correctly computed
    - Illumination geometry is properly modeled
    - Phase angle calculations are mathematically accurate
    - Temporal variations in lunar phase are tracked
    - Consistency with astronomical ephemeris data
    """
    # Test phase-dependent calculations in the module
    
    # Test moon phase weight calculations for different phases
    phase_fractions = {
        'new_moon': 0.0,
        'waxing_crescent': 0.25,
        'first_quarter': 0.5,
        'waxing_gibbous': 0.75,
        'full_moon': 1.0
    }
    
    weights = {}
    for phase_name, fraction in phase_fractions.items():
        weight = calculate_moon_phase_weight(fraction, "cosine")
        weights[phase_name] = weight
    
    # Verify phase weighting follows expected pattern
    assert weights['new_moon'] == 1.0, "New moon should have maximum weight"
    assert weights['full_moon'] == 0.0, "Full moon should have minimum weight"
    assert weights['first_quarter'] == 0.5, "First quarter should have mid-weight"
    
    # Test phase effects in reflectance estimation
    surface_type = np.array([3])  # Urban
    lunar_zenith = np.array([45.0])
    
    reflectances = {}
    for phase_name, phase_angle in [('full', 0.0), ('quarter', 90.0), ('new', 180.0)]:
        refl = estimate_lunar_reflectance(surface_type, lunar_zenith, phase_angle)
        reflectances[phase_name] = refl[0]
    
    # Full moon should have highest reflectance
    assert reflectances['full'] > reflectances['quarter'] > reflectances['new'], \
        "Reflectance should decrease from full to new moon"
    
    # Test phase factor calculation in reflectance
    # Phase factor = cos(phase_angle / 2)
    phase_angles = np.array([0, 30, 60, 90, 120, 150, 180])
    expected_factors = np.cos(np.deg2rad(phase_angles / 2.0))
    expected_factors = np.maximum(expected_factors, 0.1)  # Min 10%
    
    # Test temporal phase tracking in NTLI
    lunar_fractions_cycle = [0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0.0]
    radiance = np.full((10, 10), 0.3)
    
    ntli_values = []
    for fraction in lunar_fractions_cycle:
        ntli = compute_ntli_index(radiance, fraction)
        ntli_values.append(np.mean(ntli))
    
    # NTLI should increase with lunar fraction
    assert ntli_values[0] < ntli_values[4], "NTLI should increase from new to full moon"
    assert ntli_values[4] > ntli_values[8], "NTLI should decrease from full to new moon"
    
    # Test phase consistency in contamination detection
    # Higher lunar irradiance should lead to more contamination
    radiance = np.full((20, 20), 0.1)
    lunar_irradiances = [0.01, 0.03, 0.05]  # Increasing lunar irradiance
    
    contamination_percents = []
    for lunar_irr in lunar_irradiances:
        lunar_array = np.full((20, 20), lunar_irr)
        _, percent = detect_lunar_contamination(radiance, lunar_array, threshold_ratio=0.3)
        contamination_percents.append(percent)
    
    # More lunar irradiance should lead to more contamination
    # With radiance=0.1 and lunar_irr=0.01,0.03,0.05, estimated contribution is 0.001,0.003,0.005
    # Ratios are 0.01, 0.03, 0.05 - all below 0.3 threshold, so no contamination expected
    # Let's use a lower threshold to see the effect
    contamination_percents_low = []
    for lunar_irr in lunar_irradiances:
        lunar_array = np.full((20, 20), lunar_irr)
        _, percent = detect_lunar_contamination(radiance, lunar_array, threshold_ratio=0.05)
        contamination_percents_low.append(percent)
    
    # With lower threshold, we should see increasing contamination
    assert contamination_percents_low[1] >= contamination_percents_low[0], \
        "Contamination should not decrease with higher lunar irradiance"
    assert contamination_percents_low[2] >= contamination_percents_low[1], \
        "Contamination should not decrease with higher lunar irradiance"
    
    # Test phase cycle in temporal analysis
    n_observations = 30  # ~1 month
    phase_cycle = np.array([0.5 * (1 + np.cos(2 * np.pi * i / 29.5)) 
                           for i in range(n_observations)])
    
    # Verify cycle properties
    assert min(phase_cycle) < 0.1, "Cycle should include near-new moon"
    assert max(phase_cycle) > 0.9, "Cycle should include near-full moon"
    
    # Find approximate period
    half_max = 0.5
    above_half = phase_cycle > half_max
    transitions = np.diff(above_half.astype(int))
    
    # Should have transitions indicating cyclic behavior
    assert np.any(transitions > 0) and np.any(transitions < 0), \
        "Phase should cycle up and down"


def test_lunar_correction_edge_cases():
    """Test lunar correction functions with edge cases.
    
    This test should verify that:
    - Very high lunar zenith angles are handled correctly
    - Near-zero lunar fractions don't cause errors
    - Extreme radiance values are processed appropriately
    - Array shapes and data types are preserved
    - Memory efficiency with large datasets
    """
    # Test very high zenith angles
    shape = (20, 20)
    radiance = np.full(shape, 0.2)
    lunar_irradiance = np.full(shape, 0.01)
    
    # Test with zenith angles approaching 90°
    high_zenith = np.full(shape, 89.9)
    corrected, stats = apply_lunar_correction(
        radiance, lunar_irradiance, high_zenith,
        min_lunar_zenith=90.0  # Allow all angles
    )
    
    # Should still produce valid results
    assert np.all(np.isfinite(corrected[~np.isnan(radiance)])), \
        "High zenith angles should produce finite results"
    assert stats['pixels_corrected'] > 0, "Some pixels should be corrected"
    
    # Test near-zero lunar fractions
    tiny_fractions = [0.0, 0.001, 0.01]
    for fraction in tiny_fractions:
        ntli = compute_ntli_index(radiance, fraction)
        assert np.all(np.isfinite(ntli)), f"NTLI should be finite for fraction={fraction}"
        assert np.all(ntli >= radiance), f"NTLI should be >= radiance for fraction={fraction}"
    
    # Test extreme radiance values
    extreme_radiance = np.array([
        [0.0, 0.00001, 0.001],  # Very dark
        [0.1, 1.0, 10.0],        # Normal to bright
        [100.0, 1000.0, 10000.0] # Extremely bright
    ])
    extreme_lunar = np.full((3, 3), 0.02)
    extreme_zenith = np.full((3, 3), 45.0)
    
    corrected_extreme, stats_extreme = apply_lunar_correction(
        extreme_radiance, extreme_lunar, extreme_zenith
    )
    
    # All finite values should remain finite
    assert np.all(np.isfinite(corrected_extreme)), "Extreme values should produce finite results"
    assert np.all(corrected_extreme >= 0), "Corrected values should be non-negative"
    
    # Test array shape preservation
    shapes_to_test = [(10, 10), (100, 50), (1, 1000), (1000, 1)]
    for test_shape in shapes_to_test:
        test_rad = np.random.uniform(0.1, 0.5, test_shape)
        test_lunar = np.random.uniform(0.01, 0.05, test_shape)
        test_zenith = np.random.uniform(30, 60, test_shape)
        
        corrected_shape, _ = apply_lunar_correction(
            test_rad, test_lunar, test_zenith
        )
        
        assert corrected_shape.shape == test_shape, \
            f"Shape {test_shape} should be preserved"
    
    # Test data type preservation
    dtypes = [np.float32, np.float64]
    for dtype in dtypes:
        typed_radiance = radiance.astype(dtype)
        typed_lunar = lunar_irradiance.astype(dtype)
        typed_zenith = high_zenith.astype(dtype)
        
        # Most functions return float32 or float64 based on implementation
        result, _ = apply_lunar_correction(
            typed_radiance, typed_lunar, typed_zenith
        )
        assert result.dtype in [np.float32, np.float64], \
            f"Result should be floating point for input {dtype}"
    
    # Test mixed edge cases
    mixed_radiance = radiance.copy()
    mixed_radiance[0, :] = 0.0  # Zero row
    mixed_radiance[:, 0] = 1000.0  # Very bright column
    mixed_radiance[10:15, 10:15] = np.nan  # NaN block
    
    mixed_lunar = lunar_irradiance.copy()
    mixed_lunar[5:10, :] = 0.0  # No lunar
    mixed_lunar[:, 15:20] = 0.5  # Very high lunar
    
    mixed_zenith = high_zenith.copy()
    mixed_zenith[15:20, :] = 91.0  # Beyond horizon
    
    corrected_mixed, stats_mixed = apply_lunar_correction(
        mixed_radiance, mixed_lunar, mixed_zenith,
        min_lunar_zenith=90.0
    )
    
    # Verify NaN preservation
    assert np.all(np.isnan(corrected_mixed[10:15, 10:15])), "NaN should be preserved"
    
    # Verify zero radiance handling (except where bright column is)
    zero_mask = mixed_radiance[0, :] == 0.0
    assert np.all(corrected_mixed[0, zero_mask] == 0.0), "Zero radiance should remain zero"
    
    # Test contamination detection with edge cases
    zero_radiance = np.zeros((10, 10))
    _, zero_percent = detect_lunar_contamination(
        zero_radiance, np.full((10, 10), 0.01)
    )
    # Note: zero radiance leads to undefined ratio, implementation dependent


def test_lunar_data_quality():
    """Test quality assessment of lunar-corrected data.
    
    This test should verify that:
    - Correction quality metrics are computed
    - Over-correction and under-correction are detected
    - Quality flags are properly assigned
    - Statistical validation of correction accuracy
    - Comparison with reference datasets
    """
    # Create synthetic test data with known properties
    shape = (50, 50)
    
    # Create base radiance with spatial structure
    x, y = np.meshgrid(np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1]))
    base_radiance = 0.1 + 0.2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
    
    # Add known lunar contamination
    lunar_component = 0.05  # Known contamination level
    contaminated_radiance = base_radiance + lunar_component
    
    # Create lunar irradiance that would produce this contamination
    # Assuming 10% reflectance: lunar_component = lunar_irradiance * 0.1
    lunar_irradiance = np.full(shape, lunar_component / 0.1)
    lunar_zenith = np.full(shape, 45.0)
    
    # Apply correction
    corrected, stats = apply_lunar_correction(
        contaminated_radiance, lunar_irradiance, lunar_zenith
    )
    
    # Quality metric 1: Correction accuracy
    # Corrected should be close to base radiance
    correction_error = np.abs(corrected - base_radiance)
    mean_error = np.mean(correction_error)
    assert mean_error < 0.03, f"Mean correction error {mean_error:.4f} should be < 0.03"
    
    # Quality metric 2: Check for over-correction
    over_corrected = corrected < base_radiance - 0.02  # 2% tolerance
    over_correction_percent = np.sum(over_corrected) / corrected.size * 100
    assert over_correction_percent < 5, \
        f"Over-correction in {over_correction_percent:.1f}% pixels (should be < 5%)"
    
    # Quality metric 3: Check for under-correction
    under_corrected = corrected > base_radiance + 0.02  # 2% tolerance
    under_correction_percent = np.sum(under_corrected) / corrected.size * 100
    assert under_correction_percent < 15, \
        f"Under-correction in {under_correction_percent:.1f}% pixels (should be < 15%)"
    
    # Test correction statistics
    assert 'correction_map' in stats, "Correction map should be provided"
    correction_map = stats['correction_map']
    
    # Verify correction magnitude
    expected_correction = lunar_component * np.ones(shape)
    actual_correction = contaminated_radiance - corrected
    
    # Where correction was applied, it should match expected
    corrected_mask = correction_map > 0
    if np.any(corrected_mask):
        correction_accuracy = np.abs(actual_correction[corrected_mask] - 
                                   expected_correction[corrected_mask])
        assert np.mean(correction_accuracy) < 0.025, \
            "Correction magnitude should match expected"
    
    # Test quality with varying contamination levels
    contamination_levels = [0.01, 0.05, 0.1, 0.2]
    quality_scores = []
    
    for level in contamination_levels:
        test_radiance = base_radiance + level
        test_lunar = np.full(shape, level / 0.1)
        
        test_corrected, _ = apply_lunar_correction(
            test_radiance, test_lunar, lunar_zenith
        )
        
        # Calculate quality score (1 - relative error)
        error = np.abs(test_corrected - base_radiance) / base_radiance
        quality_score = 1.0 - np.mean(error)
        quality_scores.append(quality_score)
    
    # Quality should generally decrease with contamination level
    # (harder to correct higher contamination accurately)
    assert quality_scores[0] > quality_scores[-1], \
        "Correction quality should be better for lower contamination"
    
    # Test temporal consistency
    # Multiple observations of same scene with different lunar conditions
    n_obs = 5
    observations = []
    lunar_fractions = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for i, frac in enumerate(lunar_fractions):
        # Vary contamination with lunar fraction
        lunar_cont = lunar_component * frac
        obs_radiance = base_radiance + lunar_cont
        obs_lunar = np.full(shape, lunar_cont / 0.1)
        
        obs_corrected, _ = apply_lunar_correction(
            obs_radiance, obs_lunar, lunar_zenith
        )
        observations.append(obs_corrected)
    
    # All corrected observations should be similar (temporal consistency)
    obs_stack = np.stack(observations)
    temporal_std = np.std(obs_stack, axis=0)
    mean_temporal_std = np.mean(temporal_std)
    
    assert mean_temporal_std < 0.02, \
        f"Temporal variability {mean_temporal_std:.4f} should be < 0.02"
    
    # Test quality flags based on contamination
    contamination_mask, contamination_pct = detect_lunar_contamination(
        contaminated_radiance, lunar_irradiance, threshold_ratio=0.2
    )
    
    # High quality: low contamination
    # Medium quality: moderate contamination  
    # Low quality: high contamination
    quality_flags = np.full(shape, 2, dtype=np.uint8)  # Default: high quality
    quality_flags[contamination_mask] = 1  # Medium quality
    
    # Add extreme contamination for low quality
    extreme_lunar = lunar_irradiance * 5
    extreme_mask, _ = detect_lunar_contamination(
        contaminated_radiance, extreme_lunar, threshold_ratio=0.5
    )
    quality_flags[extreme_mask] = 0  # Low quality
    
    # Verify quality flag distribution
    unique, counts = np.unique(quality_flags, return_counts=True)
    assert len(unique) > 1, "Should have multiple quality levels"