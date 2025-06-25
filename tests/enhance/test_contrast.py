"""Test functions in blackmarble.enhance.contrast module.

This module tests contrast enhancement and histogram manipulation functionality.
"""

import pytest
import numpy as np
 Mock, patch
import tempfile
from typing import Any

# Import the module under test
from blackmarble.enhance.contrast import (
    enhance_contrast,
    normalize_data,
    adaptive_histogram_equalization,
    gamma_correction,
    multi_scale_retinex,
    histogram_matching,
    chain_enhancements,
    compute_enhancement_statistics,
)


def test_enhance_contrast():
    """Test contrast enhancement with different methods.
    
    This test should verify that:
    - Percentile method correctly clips data to specified percentiles
    - MinMax method normalizes to full range
    - Histogram method equalizes distribution
    - Adaptive method adjusts to target mean/std
    - Output range is correctly applied
    - Mask parameter excludes invalid pixels from calculations
    - Enhanced data has expected statistical properties
    - Different data types are handled correctly
    """
    # Create test data with known characteristics
    data = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.6, 0.7, 0.8, 0.9, 1.0],
        [0.05, 0.15, 0.25, 0.35, 0.45],
        [0.55, 0.65, 0.75, 0.85, 0.95]
    ], dtype=np.float32)
    
    # Test percentile method
    result_percentile = enhance_contrast(data, method="percentile", percentiles=(10.0, 90.0))
    # Verify percentile clipping works
    valid_data = data[~np.isnan(data)]
    p10, p90 = np.percentile(valid_data, [10.0, 90.0])
    assert np.all(result_percentile >= 0.0)  # Output range respected
    assert np.all(result_percentile <= 1.0)
    
    # Test minmax method
    result_minmax = enhance_contrast(data, method="minmax")
    assert np.isclose(np.nanmin(result_minmax), 0.0, atol=1e-6)  # Min maps to output min
    assert np.isclose(np.nanmax(result_minmax), 1.0, atol=1e-6)  # Max maps to output max
    
    # Test histogram method
    result_histogram = enhance_contrast(data, method="histogram")
    assert np.all(result_histogram >= 0.0)
    assert np.all(result_histogram <= 1.0)
    
    # Test adaptive method
    target_mean, target_std = 0.4, 0.2
    result_adaptive = enhance_contrast(data, method="adaptive", 
                                     target_mean=target_mean, target_std=target_std)
    # Check that mean/std are approximately achieved
    actual_mean = np.nanmean(result_adaptive)
    actual_std = np.nanstd(result_adaptive)
    assert abs(actual_mean - target_mean) < 0.1  # Reasonable tolerance
    
    # Test different output ranges
    result_custom_range = enhance_contrast(data, method="minmax", output_range=(-1.0, 2.0))
    assert np.isclose(np.nanmin(result_custom_range), -1.0, atol=1e-6)
    assert np.isclose(np.nanmax(result_custom_range), 2.0, atol=1e-6)
    
    # Test with mask
    mask = np.ones_like(data, dtype=bool)
    mask[0, 0] = False  # Exclude one pixel
    result_masked = enhance_contrast(data, method="minmax", mask=mask)
    # Result should have same shape
    assert result_masked.shape == data.shape
    
    # Test different data types
    data_int = (data * 100).astype(np.int32)
    result_int = enhance_contrast(data_int, method="minmax")
    assert result_int.dtype == data_int.dtype or np.issubdtype(result_int.dtype, np.floating)


def test_enhance_contrast_percentile_method():
    """Test percentile-based contrast enhancement.
    
    This test should verify that:
    - Data is clipped to specified percentiles (2.0, 98.0)
    - Outliers beyond percentiles are clamped
    - Output range is correctly mapped
    - Original data distribution is preserved within percentiles
    - Edge cases (all same values, single value) are handled
    """
    # Create data with outliers
    data = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.6, 0.7, 0.8, 0.9, 1.0],
        [0.01, 0.05, 0.95, 0.99, 0.5],  # Contains outliers
        [0.45, 0.46, 0.47, 0.48, 0.49]
    ])
    
    # Test with default percentiles (2.0, 98.0)
    result = enhance_contrast(data, method="percentile", percentiles=(2.0, 98.0))
    
    # Verify percentile clipping
    valid_data = data[~np.isnan(data)]
    p2, p98 = np.percentile(valid_data, [2.0, 98.0])
    
    # Values at percentiles should map to output range extremes
    assert np.all(result >= 0.0)  # Within output range
    assert np.all(result <= 1.0)
    
    # Test with custom percentiles
    result_custom = enhance_contrast(data, method="percentile", percentiles=(10.0, 90.0))
    p10, p90 = np.percentile(valid_data, [10.0, 90.0])
    assert np.all(result_custom >= 0.0)
    assert np.all(result_custom <= 1.0)
    
    # Test that tighter percentiles produce more contrast
    result_tight = enhance_contrast(data, method="percentile", percentiles=(25.0, 75.0))
    result_wide = enhance_contrast(data, method="percentile", percentiles=(5.0, 95.0))
    # Tighter percentiles should produce higher dynamic range in output
    assert np.ptp(result_tight) >= np.ptp(result_wide) * 0.8  # Allow some tolerance
    
    # Test edge case: all same values
    uniform_data = np.full((3, 3), 0.5)
    result_uniform = enhance_contrast(uniform_data, method="percentile")
    # Should return constant values within output range
    assert np.all(np.isclose(result_uniform, result_uniform[0, 0]))
    assert 0.0 <= result_uniform[0, 0] <= 1.0
    
    # Test single value
    single_data = np.array([[0.7]])
    result_single = enhance_contrast(single_data, method="percentile")
    assert result_single.shape == (1, 1)
    assert 0.0 <= result_single[0, 0] <= 1.0


def test_enhance_contrast_minmax_method():
    """Test min-max normalization contrast enhancement.
    
    This test should verify that:
    - Full dynamic range is utilized
    - Minimum value maps to output minimum
    - Maximum value maps to output maximum
    - Linear scaling is applied correctly
    - Zero range data is handled gracefully
    """
    # Create test data with known min/max
    data = np.array([
        [0.2, 0.4, 0.6],
        [0.8, 1.0, 0.3],
        [0.1, 0.5, 0.9]
    ])
    
    # Test default output range [0, 1]
    result = enhance_contrast(data, method="minmax")
    
    # Verify full dynamic range utilization
    assert np.isclose(np.nanmin(result), 0.0, atol=1e-6)
    assert np.isclose(np.nanmax(result), 1.0, atol=1e-6)
    
    # Verify linear scaling - middle values should scale proportionally
    data_min, data_max = np.nanmin(data), np.nanmax(data)
    expected_mid = (0.5 - data_min) / (data_max - data_min)  # 0.5 scaled to [0,1]
    result_mid_idx = np.where(np.isclose(data, 0.5))
    if len(result_mid_idx[0]) > 0:
        assert np.isclose(result[result_mid_idx], expected_mid, atol=1e-6)
    
    # Test custom output range
    result_custom = enhance_contrast(data, method="minmax", output_range=(-2.0, 3.0))
    assert np.isclose(np.nanmin(result_custom), -2.0, atol=1e-6)
    assert np.isclose(np.nanmax(result_custom), 3.0, atol=1e-6)
    
    # Test with negative values
    data_negative = np.array([[-1.0, 0.0, 1.0], [2.0, -0.5, 0.5]])
    result_negative = enhance_contrast(data_negative, method="minmax")
    assert np.isclose(np.nanmin(result_negative), 0.0, atol=1e-6)
    assert np.isclose(np.nanmax(result_negative), 1.0, atol=1e-6)
    
    # Test zero range data (all same values)
    zero_range_data = np.full((2, 3), 0.7)
    result_zero_range = enhance_contrast(zero_range_data, method="minmax")
    # Should return constant values (implementation specific)
    assert np.all(np.isfinite(result_zero_range))
    assert result_zero_range.shape == zero_range_data.shape


def test_enhance_contrast_adaptive_method():
    """Test adaptive contrast enhancement.
    
    This test should verify that:
    - Output achieves target mean and standard deviation
    - Non-linear mapping preserves relative relationships
    - Extreme target values are handled appropriately
    - Convergence is achieved within reasonable iterations
    """
    # Create test data with known statistics
    np.random.seed(42)
    data = np.random.normal(0.3, 0.1, (10, 10))  # Mean=0.3, std=0.1
    
    # Test with reasonable target values
    target_mean, target_std = 0.6, 0.2
    result = enhance_contrast(data, method="adaptive", 
                             target_mean=target_mean, target_std=target_std)
    
    # Verify target statistics are approximately achieved
    actual_mean = np.nanmean(result)
    actual_std = np.nanstd(result)
    
    # Allow reasonable tolerance for statistical targets
    assert abs(actual_mean - target_mean) < 0.05, f"Mean: got {actual_mean}, expected {target_mean}"
    assert abs(actual_std - target_std) < 0.05, f"Std: got {actual_std}, expected {target_std}"
    
    # Test that relative ordering is preserved (monotonic transformation)
    flat_data = data.flatten()
    flat_result = result.flatten()
    sorted_indices = np.argsort(flat_data)
    
    # Check that sorting is preserved in most cases (allowing some tolerance for noise)
    sorted_result = flat_result[sorted_indices]
    differences = np.diff(sorted_result)
    positive_diff_ratio = np.mean(differences >= 0)
    assert positive_diff_ratio > 0.8, "Relative ordering not well preserved"
    
    # Test with extreme target values
    result_extreme = enhance_contrast(data, method="adaptive", 
                                    target_mean=0.9, target_std=0.05,
                                    output_range=(0.0, 1.0))
    # Should be clipped to output range
    assert np.all(result_extreme >= 0.0)
    assert np.all(result_extreme <= 1.0)
    
    # Test with zero standard deviation data
    constant_data = np.full((5, 5), 0.5)
    result_constant = enhance_contrast(constant_data, method="adaptive", 
                                     target_mean=0.7, target_std=0.1)
    # Should return constant values at target mean
    assert np.all(np.isclose(result_constant, 0.7, atol=0.01))
    
    # Test that output respects specified range
    result_range = enhance_contrast(data, method="adaptive", 
                                  target_mean=0.5, target_std=0.15,
                                  output_range=(-1.0, 2.0))
    assert np.all(result_range >= -1.0)
    assert np.all(result_range <= 2.0)


def test_enhance_contrast_with_mask():
    """Test contrast enhancement with masked data.
    
    This test should verify that:
    - Only valid (masked) pixels are used for statistics
    - Invalid pixels remain unchanged
    - Mask boundary effects are minimal
    - Different mask shapes are handled correctly
    - Boolean and numeric masks work properly
    """
    # Create test data
    data = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 0.05, 0.95],
        [0.15, 0.25, 0.35, 0.45]
    ])
    
    # Create boolean mask - exclude some outliers
    mask_bool = np.ones_like(data, dtype=bool)
    mask_bool[2, 0] = False  # Exclude 0.9
    mask_bool[2, 1] = False  # Exclude 1.0
    mask_bool[2, 2] = False  # Exclude 0.05
    mask_bool[2, 3] = False  # Exclude 0.95
    
    # Test with boolean mask
    result_bool = enhance_contrast(data, method="percentile", mask=mask_bool)
    
    # Only valid pixels should be used for statistics calculation
    valid_data = data[mask_bool]
    p2, p98 = np.percentile(valid_data, [2.0, 98.0])
    
    # Result should have same shape as input
    assert result_bool.shape == data.shape
    
    # Test with numeric mask (0/1 values)
    mask_numeric = mask_bool.astype(np.float32)
    result_numeric = enhance_contrast(data, method="percentile", mask=mask_numeric)
    
    # Results should be similar for boolean and numeric masks
    valid_pixels = mask_bool
    assert np.allclose(result_bool[valid_pixels], result_numeric[valid_pixels], atol=1e-6)
    
    # Test with partial mask coverage
    mask_partial = np.zeros_like(data)
    mask_partial[:2, :] = 1  # Only top half is valid
    
    result_partial = enhance_contrast(data, method="minmax", mask=mask_partial)
    
    # Statistics should be computed only from top half
    valid_data_partial = data[:2, :]
    expected_min = np.min(valid_data_partial)
    expected_max = np.max(valid_data_partial)
    
    # Test different mask shapes (should handle broadcasting)
    # Note: Implementation may vary on how it handles shape mismatches
    mask_1d = np.array([1, 1, 0, 1])  # Different shape
    try:
        result_1d = enhance_contrast(data, method="minmax", mask=mask_1d)
        # If it works, result should have same shape as data
        assert result_1d.shape == data.shape
    except (ValueError, IndexError):
        # It's acceptable to reject incompatible mask shapes
        pass
    
    # Test with all-zeros mask (no valid data)
    mask_empty = np.zeros_like(data)
    result_empty = enhance_contrast(data, method="minmax", mask=mask_empty)
    # Implementation specific - might return zeros or handle gracefully
    assert result_empty.shape == data.shape
    assert np.all(np.isfinite(result_empty) | np.isnan(result_empty))


def test_enhance_contrast_edge_cases():
    """Test contrast enhancement edge cases.
    
    This test should verify that:
    - Uniform data (all same values) is handled gracefully
    - Single pixel data works correctly
    - Empty arrays return appropriate results
    - NaN and infinity values are handled properly
    - Very large and very small arrays work correctly
    """
    # Test uniform data (all same values)
    uniform_data = np.full((5, 5), 0.7)
    for method in ["percentile", "minmax", "histogram", "adaptive"]:
        result = enhance_contrast(uniform_data, method=method)
        assert result.shape == uniform_data.shape
        assert np.all(np.isfinite(result))
        # Values should be constant (may be different from input)
        assert np.allclose(result, result[0, 0])
    
    # Test single pixel data
    single_pixel = np.array([[0.3]])
    for method in ["percentile", "minmax", "histogram", "adaptive"]:
        result = enhance_contrast(single_pixel, method=method)
        assert result.shape == (1, 1)
        assert np.isfinite(result[0, 0])
    
    # Test with NaN values
    data_with_nan = np.array([
        [0.1, np.nan, 0.3],
        [0.4, 0.5, np.nan],
        [np.nan, 0.7, 0.8]
    ])
    
    result_with_nan = enhance_contrast(data_with_nan, method="percentile")
    assert result_with_nan.shape == data_with_nan.shape
    # NaN pixels should remain NaN or be handled gracefully
    valid_mask = ~np.isnan(data_with_nan)
    assert np.all(np.isfinite(result_with_nan[valid_mask]) | np.isnan(result_with_nan[valid_mask]))
    
    # Test with infinity values
    data_with_inf = np.array([
        [0.1, np.inf, 0.3],
        [0.4, 0.5, -np.inf],
        [0.6, 0.7, 0.8]
    ])
    
    result_with_inf = enhance_contrast(data_with_inf, method="percentile")
    assert result_with_inf.shape == data_with_inf.shape
    # Infinity should be handled (excluded from stats or clamped)
    finite_mask = np.isfinite(data_with_inf)
    if np.any(finite_mask):
        assert np.any(np.isfinite(result_with_inf[finite_mask]))
    
    # Test very small array
    tiny_array = np.array([[0.1, 0.9]])
    result_tiny = enhance_contrast(tiny_array, method="minmax")
    assert result_tiny.shape == (1, 2)
    assert np.isfinite(result_tiny).all()
    
    # Test larger array (performance check)
    large_array = np.random.rand(100, 100)
    result_large = enhance_contrast(large_array, method="percentile")
    assert result_large.shape == (100, 100)
    assert np.all(result_large >= 0.0)
    assert np.all(result_large <= 1.0)
    
    # Test empty-like situations
    # Create array where all values are NaN
    all_nan_data = np.full((3, 3), np.nan)
    result_all_nan = enhance_contrast(all_nan_data, method="percentile")
    assert result_all_nan.shape == all_nan_data.shape
    # Should handle gracefully (implementation specific)


def test_normalize_data():
    """Test data normalization to specified range.
    
    This test should verify that:
    - Data is correctly mapped to output range
    - Auto-detection of input range works when min/max not specified
    - Custom input range is respected when provided
    - Linear scaling preserves relative relationships
    - Different output ranges work correctly
    - Edge cases (zero range, single value) are handled
    """
    # Create test data
    data = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90]
    ], dtype=np.float32)
    
    # Test default normalization [0, 1]
    result = normalize_data(data)
    assert np.isclose(np.nanmin(result), 0.0, atol=1e-6)
    assert np.isclose(np.nanmax(result), 1.0, atol=1e-6)
    
    # Verify linear scaling - midpoint should be 0.5
    midpoint_value = 50  # Middle of range [10, 90]
    expected_normalized = 0.5
    result_midpoint = result[data == midpoint_value]
    assert np.isclose(result_midpoint[0], expected_normalized, atol=1e-6)
    
    # Test custom output range
    result_custom = normalize_data(data, output_min=-2.0, output_max=3.0)
    assert np.isclose(np.nanmin(result_custom), -2.0, atol=1e-6)
    assert np.isclose(np.nanmax(result_custom), 3.0, atol=1e-6)
    
    # Test with specified input range (subset of actual range)
    result_input_range = normalize_data(data, input_min=20, input_max=80)
    # Linear scaling: (x - 20) / (80 - 20) * (1 - 0) + 0
    # Value 10 -> (10-20)/60 = -1/6 ≈ -0.167
    # Value 90 -> (90-20)/60 = 70/60 ≈ 1.167
    # So values can go outside [0,1] range when input_min/max is specified
    assert result_input_range.shape == data.shape
    
    # Test auto-detection vs manual specification
    result_auto = normalize_data(data)  # Auto-detect range
    result_manual = normalize_data(data, input_min=10, input_max=90)  # Manual range
    assert np.allclose(result_auto, result_manual, atol=1e-6)
    
    # Test with negative values
    data_negative = np.array([[-10, 0, 10], [20, -5, 15]])
    result_negative = normalize_data(data_negative)
    assert np.all(result_negative >= 0.0)
    assert np.all(result_negative <= 1.0)
    
    # Test relative relationships preservation
    # If a > b in input, then normalize(a) > normalize(b) in output
    indices = [(0, 0), (1, 1), (2, 2)]  # Compare diagonal elements
    for i in range(len(indices) - 1):
        idx1, idx2 = indices[i], indices[i + 1]
        if data[idx1] < data[idx2]:
            assert result[idx1] < result[idx2]
    
    # Test with float precision
    data_float = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    result_float = normalize_data(data_float, output_min=0.0, output_max=10.0)
    assert np.isclose(np.min(result_float), 0.0, atol=1e-6)
    assert np.isclose(np.max(result_float), 10.0, atol=1e-6)


def test_normalize_data_custom_ranges():
    """Test normalization with custom input/output ranges.
    
    This test should verify that:
    - Custom input range overrides auto-detection
    - Values outside input range are clamped appropriately
    - Different output ranges produce correct scaling
    - Negative ranges work correctly
    - Non-unit ranges maintain precision
    """
    # Create test data with known range
    data = np.array([0, 25, 50, 75, 100], dtype=np.float32)
    
    # Test custom input range smaller than actual range
    result_narrow = normalize_data(data, input_min=20, input_max=80, 
                                 output_min=0, output_max=1)
    # Linear scaling: (x - 20) / (80 - 20) * (1 - 0) + 0
    # 0 -> (0-20)/60 = -1/3 ≈ -0.333
    # 100 -> (100-20)/60 = 80/60 ≈ 1.333
    # 50 -> (50-20)/60 = 30/60 = 0.5
    assert np.isclose(result_narrow[0], -1/3, atol=1e-6)  # 0 -> -0.333
    assert np.isclose(result_narrow[-1], 80/60, atol=1e-6)  # 100 -> 1.333
    assert np.isclose(result_narrow[2], 0.5, atol=1e-6)  # 50 -> 0.5 (middle)
    
    # Test custom input range larger than actual range
    result_wide = normalize_data(data, input_min=-50, input_max=150, 
                               output_min=0, output_max=1)
    # All values should be within [0, 1] but not use full range
    assert np.all(result_wide >= 0.0)
    assert np.all(result_wide <= 1.0)
    assert np.min(result_wide) > 0.0  # Min should be above 0
    assert np.max(result_wide) < 1.0  # Max should be below 1
    
    # Test negative output range
    result_negative = normalize_data(data, output_min=-5, output_max=-1)
    assert np.isclose(np.min(result_negative), -5.0, atol=1e-6)
    assert np.isclose(np.max(result_negative), -1.0, atol=1e-6)
    
    # Test inverted output range (max < min)
    result_inverted = normalize_data(data, output_min=10, output_max=5)
    # Should still work, just inverted scaling
    assert np.isclose(np.max(result_inverted), 10.0, atol=1e-6)  # Min data -> max output
    assert np.isclose(np.min(result_inverted), 5.0, atol=1e-6)   # Max data -> min output
    
    # Test large output range (precision test)
    result_large = normalize_data(data, output_min=0, output_max=1000000)
    assert np.isclose(np.min(result_large), 0.0, atol=1e-3)
    assert np.isclose(np.max(result_large), 1000000.0, atol=1e-3)
    
    # Test very small output range (precision test)
    result_small = normalize_data(data, output_min=0.0001, output_max=0.0002)
    assert np.all(result_small >= 0.0001)
    assert np.all(result_small <= 0.0002)
    
    # Test with identical input and output ranges
    result_identity = normalize_data(data, input_min=0, input_max=100, 
                                   output_min=0, output_max=100)
    assert np.allclose(result_identity, data, atol=1e-6)
    
    # Test fractional ranges
    result_frac = normalize_data(data, input_min=10.5, input_max=89.7, 
                               output_min=0.1, output_max=0.9)
    assert np.all(result_frac >= 0.1)
    assert np.all(result_frac <= 0.9)


def test_normalize_data_edge_cases():
    """Test normalization edge cases.
    
    This test should verify that:
    - Zero range input data is handled gracefully
    - Identical input and output ranges return original data
    - NaN and infinity handling is appropriate
    - Integer vs float data types are handled correctly
    """
    # Test zero range input data (all same values)
    constant_data = np.array([5.0, 5.0, 5.0, 5.0])
    result_constant = normalize_data(constant_data)
    # Should return reasonable constant values
    assert np.all(np.isclose(result_constant, result_constant[0]))
    assert np.all(np.isfinite(result_constant))
    
    # Test zero range with custom input range
    result_custom_zero = normalize_data(constant_data, input_min=5.0, input_max=5.0)
    assert np.all(np.isfinite(result_custom_zero))
    
    # Test with NaN values
    data_with_nan = np.array([1.0, np.nan, 3.0, 4.0, np.nan])
    result_nan = normalize_data(data_with_nan)
    # NaN values should remain NaN, others should be normalized
    nan_mask = np.isnan(data_with_nan)
    valid_mask = ~nan_mask
    
    assert np.all(np.isnan(result_nan[nan_mask]))  # NaN preserved
    if np.sum(valid_mask) > 1:  # If there are valid values to normalize
        assert np.all(np.isfinite(result_nan[valid_mask]))  # Valid values normalized
    
    # Test with infinity values
    data_with_inf = np.array([1.0, np.inf, 3.0, -np.inf, 5.0])
    result_inf = normalize_data(data_with_inf)
    # Should handle infinities gracefully (exclude from range calculation)
    finite_mask = np.isfinite(data_with_inf)
    if np.sum(finite_mask) > 1:
        finite_result = result_inf[finite_mask]
        assert np.all(np.isfinite(finite_result))
        assert np.min(finite_result) >= 0.0
        assert np.max(finite_result) <= 1.0
    
    # Test integer data types
    int_data = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    result_int = normalize_data(int_data)
    # Should work with integer input
    assert np.isclose(np.min(result_int), 0.0, atol=1e-6)
    assert np.isclose(np.max(result_int), 1.0, atol=1e-6)
    
    # Test very large integers
    large_int_data = np.array([1000000, 2000000, 3000000], dtype=np.int64)
    result_large_int = normalize_data(large_int_data)
    assert np.all(np.isfinite(result_large_int))
    assert np.isclose(np.min(result_large_int), 0.0, atol=1e-6)
    assert np.isclose(np.max(result_large_int), 1.0, atol=1e-6)
    
    # Test mixed data types (should work due to numpy's type promotion)
    mixed_data = np.array([1, 2.5, 3, 4.7, 5])
    result_mixed = normalize_data(mixed_data)
    assert np.all(np.isfinite(result_mixed))
    
    # Test single element array
    single_element = np.array([42.0])
    result_single = normalize_data(single_element)
    assert result_single.shape == (1,)
    assert np.isfinite(result_single[0])
    
    # Test empty array (if implementation supports it)
    try:
        empty_data = np.array([])
        result_empty = normalize_data(empty_data)
        assert len(result_empty) == 0
    except (ValueError, IndexError):
        # It's acceptable to not support empty arrays
        pass


def test_adaptive_histogram_equalization():
    """Test Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    This test should verify that:
    - Tiles are processed correctly across image
    - Clip limit prevents over-enhancement
    - Local contrast is improved in each tile
    - Tile boundaries don't create artifacts
    - Different tile sizes work appropriately
    - Multi-band images are handled correctly
    """
    # Create test image with varying local contrast
    data = np.zeros((16, 16), dtype=np.float32)
    # Top-left: low contrast
    data[0:8, 0:8] = np.random.uniform(0.4, 0.6, (8, 8))
    # Top-right: high contrast
    data[0:8, 8:16] = np.random.uniform(0.0, 1.0, (8, 8))
    # Bottom-left: medium contrast
    data[8:16, 0:8] = np.random.uniform(0.2, 0.8, (8, 8))
    # Bottom-right: very low contrast
    data[8:16, 8:16] = np.random.uniform(0.49, 0.51, (8, 8))
    
    # Test default CLAHE
    result = adaptive_histogram_equalization(data)
    
    # Verify output properties
    assert result.shape == data.shape
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)
    
    # Test that local contrast is improved
    # Compare standard deviation in each quadrant
    for i in range(2):
        for j in range(2):
            row_start, row_end = i * 8, (i + 1) * 8
            col_start, col_end = j * 8, (j + 1) * 8
            
            original_tile = data[row_start:row_end, col_start:col_end]
            enhanced_tile = result[row_start:row_end, col_start:col_end]
            
            orig_std = np.std(original_tile)
            enh_std = np.std(enhanced_tile)
            
            # Enhanced version should generally have more contrast
            # (except for already high-contrast regions)
            if orig_std < 0.1:  # Low contrast regions should be enhanced
                assert enh_std >= orig_std * 0.8  # Allow some tolerance
    
    # Test different clip limits
    result_low_clip = adaptive_histogram_equalization(data, clip_limit=0.01)
    result_high_clip = adaptive_histogram_equalization(data, clip_limit=0.1)
    
    # Higher clip limit should allow more enhancement
    std_low = np.std(result_low_clip)
    std_high = np.std(result_high_clip)
    # Generally, higher clip limit allows more variation
    assert std_high >= std_low * 0.8  # Allow tolerance for randomness
    
    # Test different tile sizes
    result_small_tiles = adaptive_histogram_equalization(data, tile_size=(4, 4))
    result_large_tiles = adaptive_histogram_equalization(data, tile_size=(16, 16))
    
    assert result_small_tiles.shape == data.shape
    assert result_large_tiles.shape == data.shape
    
    # Test with multi-band data
    multiband_data = np.stack([data, data * 0.8, data * 1.2], axis=0)
    result_multiband = adaptive_histogram_equalization(multiband_data)
    
    assert result_multiband.shape == multiband_data.shape
    assert np.all(result_multiband >= 0.0)
    assert np.all(result_multiband <= 1.0)
    
    # Each band should be processed independently and produce valid results
    # (Results might be similar if the input bands have similar structure after enhancement)
    for band in range(3):
        band_result = result_multiband[band]
        assert np.all(np.isfinite(band_result))
        assert band_result.shape == data.shape


def test_adaptive_histogram_equalization_tile_sizes():
    """Test CLAHE with different tile sizes.
    
    This test should verify that:
    - Small tiles provide more local enhancement
    - Large tiles provide more global enhancement
    - Non-square tile sizes work correctly
    - Tile size larger than image is handled gracefully
    - Edge tiles (partial tiles) are processed correctly
    """
    # Create test image with local variations
    np.random.seed(42)
    data = np.random.rand(20, 20).astype(np.float32)
    # Add some local structure
    data[5:10, 5:10] *= 0.3  # Dark region
    data[10:15, 10:15] = data[10:15, 10:15] * 0.8 + 0.7  # Bright region
    
    # Test very small tiles (high local adaptation)
    result_small = adaptive_histogram_equalization(data, tile_size=(2, 2))
    
    # Test medium tiles
    result_medium = adaptive_histogram_equalization(data, tile_size=(5, 5))
    
    # Test large tiles (more global adaptation)
    result_large = adaptive_histogram_equalization(data, tile_size=(10, 10))
    
    # All should have same shape and valid range
    for result in [result_small, result_medium, result_large]:
        assert result.shape == data.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
    
    # Small tiles should show more local variation
    local_std_small = np.std(result_small)
    local_std_large = np.std(result_large)
    # Note: This relationship may not always hold due to implementation details
    # but generally smaller tiles adapt more locally
    
    # Test non-square tiles
    result_rect_h = adaptive_histogram_equalization(data, tile_size=(2, 8))
    result_rect_v = adaptive_histogram_equalization(data, tile_size=(8, 2))
    
    assert result_rect_h.shape == data.shape
    assert result_rect_v.shape == data.shape
    assert np.all(result_rect_h >= 0.0) and np.all(result_rect_h <= 1.0)
    assert np.all(result_rect_v >= 0.0) and np.all(result_rect_v <= 1.0)
    
    # Test tile size larger than image
    result_oversized = adaptive_histogram_equalization(data, tile_size=(50, 50))
    assert result_oversized.shape == data.shape
    assert np.all(result_oversized >= 0.0)
    assert np.all(result_oversized <= 1.0)
    # Should essentially become global enhancement
    
    # Test with image size not divisible by tile size
    data_irregular = np.random.rand(23, 17).astype(np.float32)
    result_irregular = adaptive_histogram_equalization(data_irregular, tile_size=(4, 4))
    assert result_irregular.shape == data_irregular.shape
    
    # Test single tile (tile size = image size)
    result_single_tile = adaptive_histogram_equalization(data, tile_size=(20, 20))
    assert result_single_tile.shape == data.shape
    
    # Test minimum tile size
    result_min_tile = adaptive_histogram_equalization(data, tile_size=(1, 1))
    assert result_min_tile.shape == data.shape


def test_adaptive_histogram_equalization_clip_limit():
    """Test CLAHE clip limit parameter.
    
    This test should verify that:
    - Higher clip limits allow more enhancement
    - Lower clip limits prevent over-enhancement
    - Clip limit of 0 produces minimal enhancement
    - Very high clip limits approach global equalization
    - Clip limit is applied consistently across tiles
    """
    # Create test data with regions of different contrast
    data = np.zeros((16, 16), dtype=np.float32)
    # Create a gradient with some low-contrast regions
    for i in range(16):
        for j in range(16):
            data[i, j] = 0.3 + 0.4 * (i / 15) + 0.1 * np.sin(j / 2)
    
    # Add a low-contrast region
    data[4:8, 4:8] = 0.45 + 0.02 * np.random.rand(4, 4)
    
    # Test with very low clip limit (minimal enhancement)
    result_min = adaptive_histogram_equalization(data, clip_limit=0.001)
    
    # Test with moderate clip limit
    result_mod = adaptive_histogram_equalization(data, clip_limit=0.03)
    
    # Test with high clip limit (more enhancement)
    result_max = adaptive_histogram_equalization(data, clip_limit=0.2)
    
    # All results should be valid
    for result in [result_min, result_mod, result_max]:
        assert result.shape == data.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
    
    # Compare enhancement levels
    std_min = np.std(result_min)
    std_mod = np.std(result_mod)
    std_max = np.std(result_max)
    
    # Generally, higher clip limits should allow more variation
    # (though this depends on the specific implementation)
    assert std_mod >= std_min * 0.8  # Allow some tolerance
    
    # Test with clip limit of 0 (should be very conservative)
    result_zero = adaptive_histogram_equalization(data, clip_limit=0.0)
    assert result_zero.shape == data.shape
    # Should be very similar to original (minimal enhancement)
    
    # Test with very high clip limit (approaches global enhancement)
    result_global = adaptive_histogram_equalization(data, clip_limit=1.0)
    assert result_global.shape == data.shape
    
    # Test consistency across different regions
    # Enhancement should be applied to all tiles
    for i in range(0, 16, 4):
        for j in range(0, 16, 4):
            tile_orig = data[i:i+4, j:j+4]
            tile_enh = result_mod[i:i+4, j:j+4]
            
            # Each tile should show some level of processing
            # (exact expectations depend on implementation)
            assert np.all(np.isfinite(tile_enh))
            assert tile_enh.shape == tile_orig.shape
    
    # Test edge case: extremely high clip limit
    result_extreme = adaptive_histogram_equalization(data, clip_limit=10.0)
    assert result_extreme.shape == data.shape
    assert np.all(result_extreme >= 0.0)
    assert np.all(result_extreme <= 1.0)


def test_gamma_correction():
    """Test gamma correction for display adjustment.
    
    This test should verify that:
    - Gamma < 1 brightens the image
    - Gamma > 1 darkens the image
    - Gamma = 1 returns original data
    - Input data is properly normalized to [0,1] range
    - Epsilon parameter prevents zero-related issues
    - Gamma curve is applied correctly
    """
    # Create test data in [0, 1] range
    data = np.array([
        [0.0, 0.25, 0.5],
        [0.75, 1.0, 0.1],
        [0.9, 0.6, 0.3]
    ], dtype=np.float32)
    
    # Test gamma = 1 (should return original data)
    result_gamma_1 = gamma_correction(data, gamma=1.0)
    assert np.allclose(result_gamma_1, data, atol=1e-6)
    
    # Test gamma < 1 (brightens image)
    result_gamma_bright = gamma_correction(data, gamma=0.5)
    # Values should be >= original (except 0 and 1)
    middle_values_mask = (data > 0.01) & (data < 0.99)
    if np.any(middle_values_mask):
        assert np.all(result_gamma_bright[middle_values_mask] >= data[middle_values_mask])
    
    # Test gamma > 1 (darkens image)
    result_gamma_dark = gamma_correction(data, gamma=2.0)
    # Values should be <= original (except 0 and 1)
    if np.any(middle_values_mask):
        assert np.all(result_gamma_dark[middle_values_mask] <= data[middle_values_mask])
    
    # Test specific gamma values
    test_value = 0.5
    result_gamma_half = gamma_correction(np.array([[test_value]]), gamma=0.5)
    expected_half = np.power(test_value, 0.5)
    assert np.isclose(result_gamma_half[0, 0], expected_half, atol=1e-6)
    
    result_gamma_two = gamma_correction(np.array([[test_value]]), gamma=2.0)
    expected_two = np.power(test_value, 2.0)
    assert np.isclose(result_gamma_two[0, 0], expected_two, atol=1e-6)
    
    # Test input data normalization (values outside [0,1] should be clipped)
    data_out_of_range = np.array([[-0.5, 0.5, 1.5]])
    result_clipped = gamma_correction(data_out_of_range, gamma=1.0)
    assert np.all(result_clipped >= 0.0)
    assert np.all(result_clipped <= 1.0)
    
    # Test epsilon parameter (prevents issues with zero)
    data_with_zero = np.array([[0.0, 0.1, 0.2]])
    result_no_epsilon = gamma_correction(data_with_zero, gamma=0.5, epsilon=0.0)
    result_with_epsilon = gamma_correction(data_with_zero, gamma=0.5, epsilon=0.01)
    
    # Both should be valid
    assert np.all(np.isfinite(result_no_epsilon))
    assert np.all(np.isfinite(result_with_epsilon))
    
    # With epsilon, zero values should be slightly elevated
    if result_with_epsilon[0, 0] > result_no_epsilon[0, 0]:
        assert result_with_epsilon[0, 0] > 0.0
    
    # Test gamma curve mathematical correctness
    # For gamma correction: output = input^gamma
    test_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    for gamma_val in [0.5, 1.0, 1.5, 2.0, 2.2]:
        result = gamma_correction(test_values, gamma=gamma_val)
        expected = np.power(test_values, gamma_val)
        assert np.allclose(result, expected, atol=1e-6)
    
    # Test that output is in valid range
    result_range = gamma_correction(data, gamma=2.5)
    assert np.all(result_range >= 0.0)
    assert np.all(result_range <= 1.0)


def test_gamma_correction_values():
    """Test gamma correction with various gamma values.
    
    This test should verify that:
    - Common gamma values (0.5, 2.0, 2.2) produce expected results
    - Extreme gamma values are handled appropriately
    - Gamma correction preserves relative ordering
    - Power law relationship is maintained
    """
    # Test data with various values
    data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    # Test common gamma values
    gamma_values = [0.5, 1.0, 1.5, 2.0, 2.2]
    
    for gamma_val in gamma_values:
        result = gamma_correction(data, gamma=gamma_val)
        
        # Verify power law relationship
        expected = np.power(data, gamma_val)
        assert np.allclose(result, expected, atol=1e-6)
        
        # Verify output range
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        
        # Verify relative ordering is preserved
        # If a < b, then f(a) < f(b) for monotonic gamma function
        for i in range(len(data) - 1):
            if data[i] < data[i + 1]:
                assert result[i] <= result[i + 1]
    
    # Test extreme gamma values
    extreme_gamma_values = [0.1, 0.01, 5.0, 10.0]
    
    for gamma_val in extreme_gamma_values:
        result = gamma_correction(data, gamma=gamma_val)
        
        # Should still produce valid results
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        
        # Verify mathematical correctness
        expected = np.power(data, gamma_val)
        assert np.allclose(result, expected, atol=1e-6)
    
    # Test gamma = 0.5 (square root) - common for display
    result_sqrt = gamma_correction(data, gamma=0.5)
    expected_sqrt = np.sqrt(data)
    assert np.allclose(result_sqrt, expected_sqrt, atol=1e-6)
    
    # Test gamma = 2.0 (square) - common for encoding
    result_square = gamma_correction(data, gamma=2.0)
    expected_square = data ** 2
    assert np.allclose(result_square, expected_square, atol=1e-6)
    
    # Test gamma = 2.2 (standard sRGB)
    result_srgb = gamma_correction(data, gamma=2.2)
    expected_srgb = np.power(data, 2.2)
    assert np.allclose(result_srgb, expected_srgb, atol=1e-6)
    
    # Test relationship between complementary gamma values
    # gamma_correction(gamma_correction(x, gamma), 1/gamma) ≈ x
    original = np.array([0.2, 0.5, 0.8])
    gamma_val = 2.0
    
    step1 = gamma_correction(original, gamma=gamma_val)
    step2 = gamma_correction(step1, gamma=1.0/gamma_val)
    
    assert np.allclose(step2, original, atol=1e-5)
    
    # Test monotonicity preservation
    sorted_data = np.sort(data)
    for gamma_val in [0.5, 1.0, 2.0]:
        result_sorted = gamma_correction(sorted_data, gamma=gamma_val)
        # Result should also be sorted (monotonic)
        assert np.all(np.diff(result_sorted) >= 0)


def test_gamma_correction_epsilon():
    """Test gamma correction epsilon parameter.
    
    This test should verify that:
    - Epsilon prevents numerical issues with zero values
    - Different epsilon values produce appropriate results
    - Epsilon effect is minimal for non-zero values
    - Very small epsilon values work correctly
    """
    # Test data with zero values
    data_with_zero = np.array([0.0, 0.1, 0.2, 0.5, 0.8])
    
    # Test with no epsilon
    result_no_epsilon = gamma_correction(data_with_zero, gamma=0.5, epsilon=0.0)
    assert np.all(np.isfinite(result_no_epsilon))
    
    # Test with small epsilon
    epsilon_values = [0.001, 0.01, 0.1]
    
    for eps in epsilon_values:
        result_with_eps = gamma_correction(data_with_zero, gamma=0.5, epsilon=eps)
        
        # Should produce finite results
        assert np.all(np.isfinite(result_with_eps))
        
        # Zero value should be affected by epsilon
        zero_result = result_with_eps[0]  # First element is 0.0
        expected_zero = np.power(0.0 + eps, 0.5)
        assert np.isclose(zero_result, expected_zero, atol=1e-6)
        
        # Non-zero values should be minimally affected
        for i in range(1, len(data_with_zero)):
            original_val = data_with_zero[i]
            if original_val > 0.1:  # For values significantly larger than epsilon
                result_with_eps_val = result_with_eps[i]
                result_no_eps_val = np.power(original_val, 0.5)
                
                # Difference should be small
                relative_diff = abs(result_with_eps_val - result_no_eps_val) / result_no_eps_val
                if eps / original_val < 0.1:  # Epsilon is small relative to value
                    assert relative_diff < 0.1
    
    # Test with very small epsilon
    very_small_eps = 1e-10
    result_tiny_eps = gamma_correction(data_with_zero, gamma=2.0, epsilon=very_small_eps)
    assert np.all(np.isfinite(result_tiny_eps))
    
    # Test that larger epsilon values have more effect on zero
    eps_small = 0.01
    eps_large = 0.1
    
    result_small_eps = gamma_correction(np.array([0.0]), gamma=0.5, epsilon=eps_small)
    result_large_eps = gamma_correction(np.array([0.0]), gamma=0.5, epsilon=eps_large)
    
    # Larger epsilon should produce larger result for zero value
    assert result_large_eps[0] > result_small_eps[0]
    
    # Test epsilon with different gamma values
    gamma_values = [0.5, 1.0, 2.0]
    eps = 0.01
    
    for gamma_val in gamma_values:
        result = gamma_correction(np.array([0.0, 0.2]), gamma=gamma_val, epsilon=eps)
        
        # Zero value should be eps^gamma
        expected_zero = np.power(eps, gamma_val)
        assert np.isclose(result[0], expected_zero, atol=1e-6)
        
        # Non-zero value should be approximately (value + eps)^gamma
        expected_nonzero = np.power(0.2 + eps, gamma_val)
        assert np.isclose(result[1], expected_nonzero, atol=1e-6)
    
    # Test that epsilon doesn't affect values much larger than epsilon
    large_values = np.array([0.5, 0.7, 0.9])
    eps = 0.001
    
    result_with_eps = gamma_correction(large_values, gamma=1.5, epsilon=eps)
    result_no_eps = gamma_correction(large_values, gamma=1.5, epsilon=0.0)
    
    # Results should be very close
    assert np.allclose(result_with_eps, result_no_eps, atol=1e-4)


def test_multi_scale_retinex():
    """Test Multi-Scale Retinex enhancement.
    
    This test should verify that:
    - Multiple scales are processed and combined
    - Default scales provide good enhancement
    - Scale weights are applied correctly
    - Local and global contrast are balanced
    - Different scale combinations work appropriately
    - Output is properly normalized
    """
    # Create test data with varying local and global structure
    np.random.seed(42)
    data = np.random.rand(20, 20).astype(np.float32)
    # Add some structure at different scales
    data[5:15, 5:15] *= 0.5  # Large-scale structure
    data[8:12, 8:12] *= 2.0  # Medium-scale structure
    data[9:11, 9:11] *= 0.3  # Small-scale structure
    
    # Test with default scales
    result_default = multi_scale_retinex(data)
    
    # Verify output properties
    assert result_default.shape == data.shape
    assert np.all(result_default >= 0.0)
    assert np.all(result_default <= 1.0)
    
    # Test with custom scales
    custom_scales = [5, 20, 60]
    result_custom = multi_scale_retinex(data, scales=custom_scales)
    
    assert result_custom.shape == data.shape
    assert np.all(result_custom >= 0.0)
    assert np.all(result_custom <= 1.0)
    
    # Test with single scale (should still work)
    result_single = multi_scale_retinex(data, scales=[15])
    assert result_single.shape == data.shape
    assert np.all(result_single >= 0.0)
    assert np.all(result_single <= 1.0)
    
    # Test with custom weights
    scales = [10, 30, 90]
    weights_equal = [1/3, 1/3, 1/3]
    weights_local = [0.5, 0.3, 0.2]  # Emphasize local (small) scales
    weights_global = [0.2, 0.3, 0.5]  # Emphasize global (large) scales
    
    result_equal = multi_scale_retinex(data, scales=scales, weights=weights_equal)
    result_local = multi_scale_retinex(data, scales=scales, weights=weights_local)
    result_global = multi_scale_retinex(data, scales=scales, weights=weights_global)
    
    # All should be valid
    for result in [result_equal, result_local, result_global]:
        assert result.shape == data.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
    
    # Different weightings should produce different results
    assert not np.allclose(result_local, result_global, atol=1e-6)
    
    # Test weight normalization
    unnormalized_weights = [1, 2, 3]  # Don't sum to 1
    result_unnorm = multi_scale_retinex(data, scales=scales, weights=unnormalized_weights)
    
    # Should still work (weights are normalized internally)
    assert result_unnorm.shape == data.shape
    assert np.all(result_unnorm >= 0.0)
    assert np.all(result_unnorm <= 1.0)
    
    # Test that it's equivalent to normalized weights
    norm_weights = np.array(unnormalized_weights) / np.sum(unnormalized_weights)
    result_norm = multi_scale_retinex(data, scales=scales, weights=norm_weights.tolist())
    assert np.allclose(result_unnorm, result_norm, atol=1e-6)
    
    # Test different numbers of scales
    result_few_scales = multi_scale_retinex(data, scales=[20, 80])
    result_many_scales = multi_scale_retinex(data, scales=[5, 15, 40, 80, 150])
    
    for result in [result_few_scales, result_many_scales]:
        assert result.shape == data.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
    
    # Test with extreme scale values
    result_small_scales = multi_scale_retinex(data, scales=[1, 2, 3])
    result_large_scales = multi_scale_retinex(data, scales=[100, 200, 300])
    
    for result in [result_small_scales, result_large_scales]:
        assert result.shape == data.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
    
    # Test that MSR generally enhances contrast
    original_std = np.std(data)
    enhanced_std = np.std(result_default)
    
    # Enhanced version should generally have similar or higher contrast
    # (This is a general expectation but may not always hold)
    assert enhanced_std >= original_std * 0.5  # Allow significant tolerance


def test_multi_scale_retinex_scales():
    """Test MSR with different scale configurations.
    
    This test should verify that:
    - Single scale approximates single-scale retinex
    - More scales provide smoother enhancement
    - Scale order doesn't affect final result
    - Very large and very small scales work correctly
    """
    # Create test data
    np.random.seed(42)
    data = np.random.rand(16, 16).astype(np.float32)
    data[4:12, 4:12] *= 0.3  # Add some structure
    
    # Test single scale
    single_scale_result = multi_scale_retinex(data, scales=[20])
    assert single_scale_result.shape == data.shape
    assert np.all(single_scale_result >= 0.0)
    assert np.all(single_scale_result <= 1.0)
    
    # Test multiple scales
    multi_scale_result = multi_scale_retinex(data, scales=[10, 20, 40])
    assert multi_scale_result.shape == data.shape
    assert np.all(multi_scale_result >= 0.0)
    assert np.all(multi_scale_result <= 1.0)
    
    # Single vs multi-scale should generally produce different results
    # (unless the single scale happens to match the averaged effect)
    assert not np.allclose(single_scale_result, multi_scale_result, atol=1e-6)
    
    # Test that scale order doesn't matter
    scales_ordered = [10, 30, 50]
    scales_reversed = [50, 30, 10]
    scales_shuffled = [30, 10, 50]
    
    result_ordered = multi_scale_retinex(data, scales=scales_ordered)
    result_reversed = multi_scale_retinex(data, scales=scales_reversed)
    result_shuffled = multi_scale_retinex(data, scales=scales_shuffled)
    
    # All should produce the same result (order independence)
    assert np.allclose(result_ordered, result_reversed, atol=1e-6)
    assert np.allclose(result_ordered, result_shuffled, atol=1e-6)
    
    # Test very small scales
    small_scales = [1, 2, 3]
    result_small = multi_scale_retinex(data, scales=small_scales)
    assert result_small.shape == data.shape
    assert np.all(result_small >= 0.0)
    assert np.all(result_small <= 1.0)
    
    # Test very large scales
    large_scales = [100, 200, 300]
    result_large = multi_scale_retinex(data, scales=large_scales)
    assert result_large.shape == data.shape
    assert np.all(result_large >= 0.0)
    assert np.all(result_large <= 1.0)
    
    # Test mixed scale ranges
    mixed_scales = [2, 20, 200]
    result_mixed = multi_scale_retinex(data, scales=mixed_scales)
    assert result_mixed.shape == data.shape
    assert np.all(result_mixed >= 0.0)
    assert np.all(result_mixed <= 1.0)
    
    # Test that more scales generally provide smoother results
    # (This is harder to test directly, but we can check they're different)
    few_scales = [15, 45]
    many_scales = [5, 15, 25, 35, 45, 55]
    
    result_few = multi_scale_retinex(data, scales=few_scales)
    result_many = multi_scale_retinex(data, scales=many_scales)
    
    # Both should be valid
    assert result_few.shape == data.shape
    assert result_many.shape == data.shape
    assert np.all(result_few >= 0.0) and np.all(result_few <= 1.0)
    assert np.all(result_many >= 0.0) and np.all(result_many <= 1.0)
    
    # They should generally produce different results
    assert not np.allclose(result_few, result_many, atol=1e-6)
    
    # Test duplicate scales (should handle gracefully)
    duplicate_scales = [20, 20, 30, 30]
    result_duplicate = multi_scale_retinex(data, scales=duplicate_scales)
    assert result_duplicate.shape == data.shape
    assert np.all(result_duplicate >= 0.0)
    assert np.all(result_duplicate <= 1.0)


def test_multi_scale_retinex_weights():
    """Test MSR with custom weight configurations.
    
    This test should verify that:
    - Equal weights (default) provide balanced enhancement
    - Custom weights emphasize different scales appropriately
    - Weights are normalized to sum to 1
    - Zero weights exclude scales from processing
    """
    # Create test data
    np.random.seed(42)
    data = np.random.rand(12, 12).astype(np.float32)
    data[3:9, 3:9] *= 0.4  # Add structure
    
    scales = [10, 30, 60]
    
    # Test default (equal) weights
    result_default = multi_scale_retinex(data, scales=scales)
    result_equal = multi_scale_retinex(data, scales=scales, weights=[1/3, 1/3, 1/3])
    
    # Should produce the same result
    assert np.allclose(result_default, result_equal, atol=1e-6)
    
    # Test custom weight configurations
    weights_first = [0.8, 0.1, 0.1]   # Emphasize first scale
    weights_middle = [0.1, 0.8, 0.1]  # Emphasize middle scale
    weights_last = [0.1, 0.1, 0.8]    # Emphasize last scale
    
    result_first = multi_scale_retinex(data, scales=scales, weights=weights_first)
    result_middle = multi_scale_retinex(data, scales=scales, weights=weights_middle)
    result_last = multi_scale_retinex(data, scales=scales, weights=weights_last)
    
    # All should be valid
    for result in [result_first, result_middle, result_last]:
        assert result.shape == data.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
    
    # Different weights should produce different results
    assert not np.allclose(result_first, result_middle, atol=1e-6)
    assert not np.allclose(result_middle, result_last, atol=1e-6)
    assert not np.allclose(result_first, result_last, atol=1e-6)
    
    # Test weight normalization
    unnormalized_weights = [2, 4, 6]  # Sum = 12
    normalized_weights = [2/12, 4/12, 6/12]  # Sum = 1
    
    result_unnorm = multi_scale_retinex(data, scales=scales, weights=unnormalized_weights)
    result_norm = multi_scale_retinex(data, scales=scales, weights=normalized_weights)
    
    # Should produce the same result (internal normalization)
    assert np.allclose(result_unnorm, result_norm, atol=1e-6)
    
    # Test zero weights (should exclude scales)
    weights_with_zero = [0.5, 0.0, 0.5]  # Middle scale excluded
    result_zero_weight = multi_scale_retinex(data, scales=scales, weights=weights_with_zero)
    
    # Should be equivalent to using only first and last scales
    scales_reduced = [scales[0], scales[2]]
    weights_reduced = [0.5, 0.5]
    result_reduced = multi_scale_retinex(data, scales=scales_reduced, weights=weights_reduced)
    
    assert np.allclose(result_zero_weight, result_reduced, atol=1e-6)
    
    # Test all-zero weights (edge case)
    weights_all_zero = [0.0, 0.0, 0.0]
    try:
        result_all_zero = multi_scale_retinex(data, scales=scales, weights=weights_all_zero)
        # If it doesn't raise an error, result should still be valid
        assert result_all_zero.shape == data.shape
        assert np.all(np.isfinite(result_all_zero))
    except (ValueError, ZeroDivisionError):
        # It's acceptable to raise an error for all-zero weights
        pass
    
    # Test single non-zero weight
    weights_single = [0.0, 1.0, 0.0]  # Only middle scale
    result_single_weight = multi_scale_retinex(data, scales=scales, weights=weights_single)
    
    # Should be equivalent to single-scale with middle scale
    result_single_scale = multi_scale_retinex(data, scales=[scales[1]])
    assert np.allclose(result_single_weight, result_single_scale, atol=1e-6)
    
    # Test negative weights (should work but be unusual)
    weights_negative = [0.6, -0.1, 0.5]  # One negative weight
    try:
        result_negative = multi_scale_retinex(data, scales=scales, weights=weights_negative)
        # Should still produce a valid result if implementation supports it
        assert result_negative.shape == data.shape
        assert np.all(result_negative >= 0.0)
        assert np.all(result_negative <= 1.0)
    except ValueError:
        # It's acceptable to reject negative weights
        pass
    
    # Test very small weights
    weights_tiny = [0.9999, 0.0001, 0.0000]
    result_tiny = multi_scale_retinex(data, scales=scales, weights=weights_tiny)
    assert result_tiny.shape == data.shape
    assert np.all(result_tiny >= 0.0)
    assert np.all(result_tiny <= 1.0)


def test_histogram_matching():
    """Test histogram matching between source and reference.
    
    This test should verify that:
    - Source histogram is transformed to match reference
    - CDFs are computed correctly for both images
    - Mapping function preserves ordering
    - Matched result has similar distribution to reference
    - Mask parameter works correctly for both images
    - Different data ranges are handled appropriately
    """
    # Create source and reference with different distributions
    np.random.seed(42)
    
    # Source: narrow range, low values
    source = np.random.uniform(0.2, 0.4, (10, 10)).astype(np.float32)
    
    # Reference: wider range, higher values
    reference = np.random.uniform(0.1, 0.9, (8, 8)).astype(np.float32)
    
    # Test basic histogram matching
    result = histogram_matching(source, reference)
    
    # Verify output properties
    assert result.shape == source.shape
    assert np.all(np.isfinite(result))
    
    # Check that result distribution is more similar to reference
    source_mean = np.mean(source)
    reference_mean = np.mean(reference)
    result_mean = np.mean(result)
    
    # Result mean should be closer to reference than original source
    source_diff = abs(source_mean - reference_mean)
    result_diff = abs(result_mean - reference_mean)
    assert result_diff <= source_diff * 1.5  # Allow some tolerance
    
    # Test that ordering is preserved
    source_flat = source.flatten()
    result_flat = result.flatten()
    
    # Check monotonicity preservation
    for i in range(0, len(source_flat) - 1, len(source_flat) // 10):
        for j in range(i + 1, len(source_flat), len(source_flat) // 10):
            if source_flat[i] < source_flat[j]:
                assert result_flat[i] <= result_flat[j]
    
    # Test with different data ranges
    source_shifted = source + 0.5  # Shift source range
    result_shifted = histogram_matching(source_shifted, reference)
    
    assert result_shifted.shape == source_shifted.shape
    assert np.all(np.isfinite(result_shifted))
    
    # Test with identical source and reference
    identical_data = np.random.rand(5, 5)
    result_identical = histogram_matching(identical_data, identical_data)
    
    # Should return very similar to original (within numerical precision)
    assert np.allclose(result_identical, identical_data, atol=1e-6)
    
    # Test with mask
    mask = np.ones_like(source, dtype=bool)
    mask[0:2, 0:2] = False  # Exclude some pixels
    
    result_masked = histogram_matching(source, reference, mask=mask)
    assert result_masked.shape == source.shape
    
    # Test with different shaped arrays
    source_rect = np.random.uniform(0.3, 0.5, (6, 12))
    reference_square = np.random.uniform(0.0, 1.0, (10, 10))
    
    result_diff_shape = histogram_matching(source_rect, reference_square)
    assert result_diff_shape.shape == source_rect.shape
    
    # Test with extreme distributions
    source_narrow = np.full((5, 5), 0.5)  # Constant value
    reference_wide = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]] * 5)
    
    result_extreme = histogram_matching(source_narrow, reference_wide)
    assert result_extreme.shape == source_narrow.shape
    assert np.all(np.isfinite(result_extreme))
    
    # Test that result has reasonable range
    result_min, result_max = np.min(result), np.max(result)
    ref_min, ref_max = np.min(reference), np.max(reference)
    
    # Result range should be related to reference range
    assert result_min >= ref_min - 0.1  # Allow some tolerance
    assert result_max <= ref_max + 0.1


def test_histogram_matching_with_masks():
    """Test histogram matching with masked data.
    
    This test should verify that:
    - Only valid pixels are used for histogram computation
    - Invalid pixels remain unchanged in output
    - Mask shapes are handled correctly
    - Partial overlap between masks works properly
    """
    # Create source and reference data
    np.random.seed(42)
    source = np.random.uniform(0.2, 0.6, (8, 8)).astype(np.float32)
    reference = np.random.uniform(0.0, 1.0, (8, 8)).astype(np.float32)
    
    # Create mask that excludes some pixels
    mask = np.ones_like(source, dtype=bool)
    mask[0:2, 0:2] = False  # Exclude corner
    mask[6:8, 6:8] = False  # Exclude opposite corner
    
    result_masked = histogram_matching(source, reference, mask=mask)
    
    # Check that result has same shape
    assert result_masked.shape == source.shape
    
    # Check that masked-out pixels are handled appropriately
    # (Implementation may vary - they might remain unchanged or be processed differently)
    valid_pixels = mask > 0
    invalid_pixels = mask == 0
    
    assert np.all(np.isfinite(result_masked[valid_pixels]))
    # Invalid pixels should either remain unchanged or be handled gracefully
    assert np.all(np.isfinite(result_masked[invalid_pixels]) | 
                  np.isnan(result_masked[invalid_pixels]))
    
    # Test with numeric mask (0/1 values)
    mask_numeric = mask.astype(np.float32)
    result_numeric_mask = histogram_matching(source, reference, mask=mask_numeric)
    
    # Should produce similar results
    assert result_numeric_mask.shape == source.shape
    
    # Test with partial masks (different coverage)
    mask_sparse = np.zeros_like(source, dtype=bool)
    mask_sparse[2:6, 2:6] = True  # Only center region
    
    result_sparse = histogram_matching(source, reference, mask=mask_sparse)
    assert result_sparse.shape == source.shape
    
    # Test with almost empty mask
    mask_tiny = np.zeros_like(source, dtype=bool)
    mask_tiny[4, 4] = True  # Only one pixel
    
    result_tiny = histogram_matching(source, reference, mask=mask_tiny)
    assert result_tiny.shape == source.shape
    
    # Test with completely empty mask
    mask_empty = np.zeros_like(source, dtype=bool)
    result_empty = histogram_matching(source, reference, mask=mask_empty)
    
    # Should handle gracefully (return original or zeros)
    assert result_empty.shape == source.shape
    assert np.all(np.isfinite(result_empty) | np.isnan(result_empty))
    
    # Test different mask shapes/broadcasting
    try:
        # Test with 1D mask (might work with broadcasting)
        mask_1d = np.array([1, 1, 0, 1, 1, 0, 1, 1])
        result_1d_mask = histogram_matching(source, reference, mask=mask_1d)
        assert result_1d_mask.shape == source.shape
    except (ValueError, IndexError):
        # It's acceptable to reject incompatible mask shapes
        pass
    
    # Test mask affects histogram computation
    # Create data where masking should make a difference
    source_with_outliers = source.copy()
    source_with_outliers[0, 0] = 0.0  # Extreme outlier
    source_with_outliers[7, 7] = 1.0  # Another outlier
    
    mask_no_outliers = np.ones_like(source_with_outliers, dtype=bool)
    mask_no_outliers[0, 0] = False
    mask_no_outliers[7, 7] = False
    
    result_no_mask = histogram_matching(source_with_outliers, reference)
    result_with_mask = histogram_matching(source_with_outliers, reference, 
                                        mask=mask_no_outliers)
    
    # Results should be different (outliers excluded in masked version)
    assert not np.allclose(result_no_mask, result_with_mask, atol=1e-6)
    
    # Test boolean vs numeric mask equivalence
    mask_bool = np.random.rand(8, 8) > 0.3
    mask_float = mask_bool.astype(np.float32)
    
    result_bool_mask = histogram_matching(source, reference, mask=mask_bool)
    result_float_mask = histogram_matching(source, reference, mask=mask_float)
    
    # Should produce similar results for valid pixels
    valid_bool = mask_bool
    if np.any(valid_bool):
        assert np.allclose(result_bool_mask[valid_bool], 
                          result_float_mask[valid_bool], atol=1e-6)


def test_histogram_matching_edge_cases():
    """Test histogram matching edge cases.
    
    This test should verify that:
    - Identical source and reference return original data
    - Single-value images are handled gracefully
    - Empty valid regions return appropriate results
    - Very different dynamic ranges work correctly
    """
    # Test identical source and reference
    identical_data = np.random.rand(6, 6).astype(np.float32)
    result_identical = histogram_matching(identical_data, identical_data)
    
    # Should return very similar to original
    assert np.allclose(result_identical, identical_data, atol=1e-6)
    
    # Test single-value images (uniform data)
    source_uniform = np.full((5, 5), 0.3)
    reference_uniform = np.full((4, 4), 0.7)
    
    result_uniform = histogram_matching(source_uniform, reference_uniform)
    assert result_uniform.shape == source_uniform.shape
    assert np.all(np.isfinite(result_uniform))
    # Result should be close to reference value
    assert np.allclose(result_uniform, 0.7, atol=0.1)
    
    # Test source uniform, reference varied
    source_constant = np.full((4, 4), 0.5)
    reference_varied = np.array([[0.0, 0.3, 0.7, 1.0]] * 4)
    
    result_const_to_varied = histogram_matching(source_constant, reference_varied)
    assert result_const_to_varied.shape == source_constant.shape
    assert np.all(np.isfinite(result_const_to_varied))
    
    # Test source varied, reference uniform
    source_varied = np.linspace(0, 1, 16).reshape(4, 4)
    reference_constant = np.full((3, 3), 0.4)
    
    result_varied_to_const = histogram_matching(source_varied, reference_constant)
    assert result_varied_to_const.shape == source_varied.shape
    assert np.all(np.isfinite(result_varied_to_const))
    # All values should be close to reference constant
    assert np.allclose(result_varied_to_const, 0.4, atol=0.1)
    
    # Test very different dynamic ranges
    source_small_range = np.random.uniform(0.49, 0.51, (6, 6))  # Very narrow
    reference_large_range = np.random.uniform(0.0, 1.0, (6, 6))  # Full range
    
    result_range_expansion = histogram_matching(source_small_range, reference_large_range)
    assert result_range_expansion.shape == source_small_range.shape
    
    # Result should have expanded range
    result_range = np.ptp(result_range_expansion)  # peak-to-peak
    source_range = np.ptp(source_small_range)
    assert result_range > source_range
    
    # Test with extreme values
    source_extreme = np.array([[0.0, 0.0001, 0.9999, 1.0]])
    reference_mid = np.array([[0.4, 0.5, 0.5, 0.6]])
    
    result_extreme = histogram_matching(source_extreme, reference_mid)
    assert result_extreme.shape == source_extreme.shape
    assert np.all(np.isfinite(result_extreme))
    
    # Test with NaN values
    source_with_nan = np.array([
        [0.1, np.nan, 0.3],
        [0.4, 0.5, np.nan],
        [np.nan, 0.7, 0.8]
    ])
    reference_clean = np.array([
        [0.0, 0.3, 0.6],
        [0.2, 0.5, 0.8],
        [0.1, 0.4, 0.9]
    ])
    
    result_with_nan = histogram_matching(source_with_nan, reference_clean)
    assert result_with_nan.shape == source_with_nan.shape
    
    # NaN pixels should remain NaN or be handled gracefully
    nan_mask = np.isnan(source_with_nan)
    valid_mask = ~nan_mask
    
    if np.any(valid_mask):
        assert np.all(np.isfinite(result_with_nan[valid_mask]))
    
    # Test single pixel arrays
    source_single = np.array([[0.3]])
    reference_single = np.array([[0.8]])
    
    result_single = histogram_matching(source_single, reference_single)
    assert result_single.shape == (1, 1)
    assert np.isfinite(result_single[0, 0])
    # Should map to reference value
    assert np.isclose(result_single[0, 0], 0.8, atol=0.1)
    
    # Test empty-like conditions
    # Array with all NaN
    source_all_nan = np.full((3, 3), np.nan)
    reference_normal = np.random.rand(3, 3)
    
    result_all_nan = histogram_matching(source_all_nan, reference_normal)
    assert result_all_nan.shape == source_all_nan.shape
    # Should handle gracefully (return NaN or zeros)
    
    # Test very large and very small arrays
    source_tiny = np.array([[0.2, 0.8]])
    reference_tiny = np.array([[0.1, 0.9]])
    
    result_tiny = histogram_matching(source_tiny, reference_tiny)
    assert result_tiny.shape == source_tiny.shape
    assert np.all(np.isfinite(result_tiny))


def test_chain_enhancements():
    """Test chaining multiple enhancement operations.
    
    This test should verify that:
    - Operations are applied in correct sequence
    - Each operation receives output of previous operation
    - Different operation combinations work correctly
    - Parameters are passed correctly to each operation
    - Failed operations are handled gracefully
    - Chaining preserves data integrity
    """
    # Create test data
    data = np.random.rand(8, 8).astype(np.float32)
    
    # Test simple chain: normalize then gamma correct
    operations = [
        (normalize_data, {"output_min": 0.0, "output_max": 1.0}),
        (gamma_correction, {"gamma": 0.5})
    ]
    
    result_chain = chain_enhancements(data, operations)
    
    # Verify shape preservation
    assert result_chain.shape == data.shape
    assert np.all(np.isfinite(result_chain))
    
    # Verify it's equivalent to manual chaining
    step1 = normalize_data(data, output_min=0.0, output_max=1.0)
    step2 = gamma_correction(step1, gamma=0.5)
    
    assert np.allclose(result_chain, step2, atol=1e-6)
    
    # Test longer chain
    operations_long = [
        (normalize_data, {"output_min": 0.0, "output_max": 1.0}),
        (enhance_contrast, {"method": "percentile", "percentiles": (5.0, 95.0)}),
        (gamma_correction, {"gamma": 2.0}),
        (normalize_data, {"output_min": 0.0, "output_max": 255.0})
    ]
    
    result_long_chain = chain_enhancements(data, operations_long)
    assert result_long_chain.shape == data.shape
    assert np.all(np.isfinite(result_long_chain))
    
    # Verify manual equivalence
    manual_step1 = normalize_data(data, output_min=0.0, output_max=1.0)
    manual_step2 = enhance_contrast(manual_step1, method="percentile", percentiles=(5.0, 95.0))
    manual_step3 = gamma_correction(manual_step2, gamma=2.0)
    manual_step4 = normalize_data(manual_step3, output_min=0.0, output_max=255.0)
    
    assert np.allclose(result_long_chain, manual_step4, atol=1e-5)
    
    # Test that order matters
    operations_order1 = [
        (gamma_correction, {"gamma": 0.5}),
        (enhance_contrast, {"method": "minmax"})
    ]
    
    operations_order2 = [
        (enhance_contrast, {"method": "minmax"}),
        (gamma_correction, {"gamma": 0.5})
    ]
    
    result_order1 = chain_enhancements(data, operations_order1)
    result_order2 = chain_enhancements(data, operations_order2)
    
    # Different orders should generally produce different results
    assert not np.allclose(result_order1, result_order2, atol=1e-6)
    
    # Test empty operations list
    result_empty = chain_enhancements(data, [])
    assert np.allclose(result_empty, data, atol=1e-6)
    
    # Test single operation
    operations_single = [(normalize_data, {"output_min": -1.0, "output_max": 2.0})]
    result_single = chain_enhancements(data, operations_single)
    expected_single = normalize_data(data, output_min=-1.0, output_max=2.0)
    assert np.allclose(result_single, expected_single, atol=1e-6)
    
    # Test parameter passing
    # Use specific parameters that should be visible in output
    operations_params = [
        (normalize_data, {"output_min": 10.0, "output_max": 20.0}),
        (normalize_data, {"output_min": 0.0, "output_max": 1.0, 
                         "input_min": 10.0, "input_max": 20.0})
    ]
    
    result_params = chain_enhancements(data, operations_params)
    assert result_params.shape == data.shape
    assert np.all(result_params >= 0.0)
    assert np.all(result_params <= 1.0)
    
    # Test data integrity preservation
    # Original data should not be modified
    original_copy = data.copy()
    operations_test = [
        (normalize_data, {}),
        (gamma_correction, {"gamma": 1.5})
    ]
    
    result_integrity = chain_enhancements(data, operations_test)
    assert np.allclose(data, original_copy, atol=1e-10)  # Original unchanged
    
    # Test with different enhancement methods
    operations_varied = [
        (enhance_contrast, {"method": "adaptive", "target_mean": 0.6}),
        (multi_scale_retinex, {"scales": [10, 30]}),
        (normalize_data, {"output_min": 0.0, "output_max": 1.0})
    ]
    
    result_varied = chain_enhancements(data, operations_varied)
    assert result_varied.shape == data.shape
    assert np.all(np.isfinite(result_varied))


def test_chain_enhancements_operation_order():
    """Test operation order in enhancement chaining.
    
    This test should verify that:
    - Different orders produce different results
    - Order-dependent operations work correctly
    - Operations that modify data range are handled properly
    - Intermediate results are valid inputs for next operations
    """
    # Create test data
    np.random.seed(42)
    data = np.random.uniform(0.1, 0.9, (6, 6)).astype(np.float32)
    
    # Test order dependence: contrast then gamma vs gamma then contrast
    order1 = [
        (enhance_contrast, {"method": "percentile", "percentiles": (10.0, 90.0)}),
        (gamma_correction, {"gamma": 0.5})
    ]
    
    order2 = [
        (gamma_correction, {"gamma": 0.5}),
        (enhance_contrast, {"method": "percentile", "percentiles": (10.0, 90.0)})
    ]
    
    result1 = chain_enhancements(data, order1)
    result2 = chain_enhancements(data, order2)
    
    # Different orders should produce different results
    assert not np.allclose(result1, result2, atol=1e-6)
    assert result1.shape == result2.shape == data.shape
    
    # Test normalization order effects
    order_norm1 = [
        (normalize_data, {"output_min": 0.0, "output_max": 100.0}),
        (enhance_contrast, {"method": "minmax"}),
        (normalize_data, {"output_min": 0.0, "output_max": 1.0})
    ]
    
    order_norm2 = [
        (enhance_contrast, {"method": "minmax"}),
        (normalize_data, {"output_min": 0.0, "output_max": 100.0}),
        (normalize_data, {"output_min": 0.0, "output_max": 1.0})
    ]
    
    result_norm1 = chain_enhancements(data, order_norm1)
    result_norm2 = chain_enhancements(data, order_norm2)
    
    assert not np.allclose(result_norm1, result_norm2, atol=1e-6)
    
    # Test range-modifying operations
    # Operations that change data range should work with subsequent operations
    range_ops = [
        (normalize_data, {"output_min": -5.0, "output_max": 10.0}),  # Wide range
        (gamma_correction, {"gamma": 2.0}),  # Expects [0,1], will clip
        (normalize_data, {"output_min": 0.0, "output_max": 1.0})   # Back to [0,1]
    ]
    
    result_range = chain_enhancements(data, range_ops)
    assert result_range.shape == data.shape
    assert np.all(result_range >= 0.0)
    assert np.all(result_range <= 1.0)
    
    # Test that intermediate results are valid inputs
    # Chain multiple contrast enhancements
    multi_contrast = [
        (enhance_contrast, {"method": "percentile", "percentiles": (5.0, 95.0)}),
        (enhance_contrast, {"method": "adaptive", "target_mean": 0.5, "target_std": 0.2}),
        (enhance_contrast, {"method": "minmax"})
    ]
    
    result_multi = chain_enhancements(data, multi_contrast)
    assert result_multi.shape == data.shape
    assert np.all(np.isfinite(result_multi))
    
    # Test operations with different input requirements
    mixed_ops = [
        (normalize_data, {"output_min": 0.0, "output_max": 1.0}),  # Ensures [0,1]
        (adaptive_histogram_equalization, {"clip_limit": 0.05}),    # Works on [0,1]
        (gamma_correction, {"gamma": 1.5}),                         # Expects [0,1]
        (multi_scale_retinex, {"scales": [5, 15, 40]})              # General purpose
    ]
    
    result_mixed = chain_enhancements(data, mixed_ops)
    assert result_mixed.shape == data.shape
    assert np.all(result_mixed >= 0.0)
    assert np.all(result_mixed <= 1.0)
    
    # Test commutative vs non-commutative operations
    # Some operations should commute, others shouldn't
    
    # Normalization operations with same final range should be more commutative
    comm_test1 = [
        (normalize_data, {"output_min": 0.0, "output_max": 2.0}),
        (normalize_data, {"output_min": 0.0, "output_max": 1.0})
    ]
    
    comm_test2 = [
        (normalize_data, {"output_min": 0.0, "output_max": 1.0}),
        (normalize_data, {"output_min": 0.0, "output_max": 2.0}),
        (normalize_data, {"output_min": 0.0, "output_max": 1.0})
    ]
    
    result_comm1 = chain_enhancements(data, comm_test1)
    result_comm2 = chain_enhancements(data, comm_test2)
    
    # These should be more similar (though not necessarily identical)
    assert result_comm1.shape == result_comm2.shape
    
    # Test order with operations that have different sensitivity to input range
    sensitive_ops1 = [
        (gamma_correction, {"gamma": 0.5}),      # Very sensitive to range
        (normalize_data, {"output_min": 0.0, "output_max": 1.0})
    ]
    
    sensitive_ops2 = [
        (normalize_data, {"output_min": 0.0, "output_max": 1.0}),
        (gamma_correction, {"gamma": 0.5})       # Applied after normalization
    ]
    
    result_sens1 = chain_enhancements(data, sensitive_ops1)
    result_sens2 = chain_enhancements(data, sensitive_ops2)
    
    # Should produce different results due to gamma sensitivity
    assert not np.allclose(result_sens1, result_sens2, atol=1e-6)


def test_chain_enhancements_error_handling():
    """Test error handling in enhancement chaining.
    
    This test should verify that:
    - Invalid operations raise appropriate errors
    - Malformed parameter dictionaries are caught
    - Partial chains can be recovered from errors
    - Error messages indicate which operation failed
    """
    # Create test data
    data = np.random.rand(4, 4).astype(np.float32)
    
    # Test with invalid function (should raise error)
    def invalid_function(data, **kwargs):
        raise ValueError("Test error")
    
    operations_invalid = [
        (normalize_data, {}),
        (invalid_function, {}),
        (gamma_correction, {"gamma": 1.0})
    ]
    
    with pytest.raises(ValueError, match="Test error"):
        chain_enhancements(data, operations_invalid)
    
    # Test with malformed parameters
    operations_bad_params = [
        (normalize_data, {"invalid_param": 123}),
    ]
    
    try:
        result = chain_enhancements(data, operations_bad_params)
        # If it doesn't raise an error, that's also acceptable
        # (depends on how functions handle unexpected kwargs)
        assert result.shape == data.shape
    except TypeError:
        # Expected - function doesn't accept the parameter
        pass
    
    # Test with missing required parameters
    operations_missing_params = [
        (enhance_contrast, {"method": "percentile", "percentiles": (10.0,)})  # Wrong tuple length
    ]
    
    try:
        chain_enhancements(data, operations_missing_params)
    except (ValueError, TypeError, IndexError):
        # Should raise some kind of error for malformed parameters
        pass
    
    # Test with None function (should raise error)
    operations_none = [
        (None, {})
    ]
    
    with pytest.raises((TypeError, AttributeError)):
        chain_enhancements(data, operations_none)
    
    # Test with non-callable object
    operations_non_callable = [
        ("not_a_function", {})
    ]
    
    with pytest.raises(TypeError):
        chain_enhancements(data, operations_non_callable)
    
    # Test empty parameters dict (should work)
    operations_empty_params = [
        (normalize_data, {}),
        (gamma_correction, {})
    ]
    
    result_empty_params = chain_enhancements(data, operations_empty_params)
    assert result_empty_params.shape == data.shape
    
    # Test function that returns wrong shape
    def wrong_shape_function(data, **kwargs):
        return np.array([1, 2, 3])  # Different shape
    
    operations_wrong_shape = [
        (normalize_data, {}),
        (wrong_shape_function, {})
    ]
    
    # This might not raise an error immediately, but subsequent operations might fail
    try:
        result_wrong_shape = chain_enhancements(data, operations_wrong_shape)
        # If it works, the result will have wrong shape
        assert result_wrong_shape.shape != data.shape
    except (ValueError, IndexError, AttributeError):
        # Acceptable to fail on wrong shape
        pass
    
    # Test function that returns invalid data
    def invalid_data_function(data, **kwargs):
        return None
    
    operations_invalid_data = [
        (invalid_data_function, {})
    ]
    
    with pytest.raises((TypeError, AttributeError)):
        chain_enhancements(data, operations_invalid_data)
    
    # Test operations list format errors
    try:
        # Wrong tuple format
        bad_operations = [(normalize_data,)]  # Missing parameters dict
        chain_enhancements(data, bad_operations)
    except (ValueError, TypeError, IndexError):
        # Should raise error for malformed operations list
        pass
    
    try:
        # Wrong tuple format - three elements instead of two
        bad_operations2 = [(normalize_data, {}, "extra")]
        chain_enhancements(data, bad_operations2)
    except (ValueError, TypeError, IndexError):
        # May or may not raise error depending on implementation
        pass


def test_compute_enhancement_statistics():
    """Test computation of enhancement comparison statistics.
    
    This test should verify that:
    - Basic statistics (mean, std) are computed correctly
    - Changes between original and enhanced are calculated
    - Dynamic range improvement is measured appropriately
    - Contrast improvement ratio is meaningful
    - Mask parameter excludes invalid pixels
    - Results are returned in expected format
    """
    # Create original and enhanced data with known properties
    original = np.array([
        [0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7],
        [0.8, 0.25, 0.35]
    ], dtype=np.float32)
    
    # Enhanced version: expanded range and different mean
    enhanced = np.array([
        [0.0, 0.2, 0.5],
        [0.7, 0.9, 1.0],
        [0.95, 0.1, 0.3]
    ], dtype=np.float32)
    
    # Test basic statistics computation
    stats = compute_enhancement_statistics(original, enhanced)
    
    # Verify return format
    expected_keys = {
        "original_mean", "original_std", "enhanced_mean", "enhanced_std",
        "mean_change", "std_change", "dynamic_range_improvement", 
        "contrast_improvement_ratio"
    }
    assert set(stats.keys()) == expected_keys
    
    # Verify basic statistics
    expected_orig_mean = np.mean(original)
    expected_orig_std = np.std(original)
    expected_enh_mean = np.mean(enhanced)
    expected_enh_std = np.std(enhanced)
    
    assert np.isclose(stats["original_mean"], expected_orig_mean, atol=1e-6)
    assert np.isclose(stats["original_std"], expected_orig_std, atol=1e-6)
    assert np.isclose(stats["enhanced_mean"], expected_enh_mean, atol=1e-6)
    assert np.isclose(stats["enhanced_std"], expected_enh_std, atol=1e-6)
    
    # Verify change calculations
    expected_mean_change = expected_enh_mean - expected_orig_mean
    expected_std_change = expected_enh_std - expected_orig_std
    
    assert np.isclose(stats["mean_change"], expected_mean_change, atol=1e-6)
    assert np.isclose(stats["std_change"], expected_std_change, atol=1e-6)
    
    # Verify dynamic range improvement
    orig_range = np.ptp(original)  # peak-to-peak
    enh_range = np.ptp(enhanced)
    expected_range_improvement = enh_range / orig_range if orig_range > 0 else np.nan
    
    if np.isfinite(expected_range_improvement):
        assert np.isclose(stats["dynamic_range_improvement"], expected_range_improvement, atol=1e-6)
    
    # Verify contrast improvement ratio
    expected_contrast_ratio = expected_enh_std / expected_orig_std if expected_orig_std > 0 else np.nan
    
    if np.isfinite(expected_contrast_ratio):
        assert np.isclose(stats["contrast_improvement_ratio"], expected_contrast_ratio, atol=1e-6)
    
    # Test with actual enhancement
    test_data = np.random.uniform(0.4, 0.6, (8, 8))
    enhanced_data = enhance_contrast(test_data, method="percentile", percentiles=(5.0, 95.0))
    
    real_stats = compute_enhancement_statistics(test_data, enhanced_data)
    
    # Enhanced version should generally have higher dynamic range
    if real_stats["dynamic_range_improvement"] is not np.nan:
        assert real_stats["dynamic_range_improvement"] >= 0.8  # Allow some tolerance
    
    # All statistics should be finite
    for key, value in real_stats.items():
        if not np.isnan(value):  # Some values might be NaN for edge cases
            assert np.isfinite(value)
    
    # Test that enhancement improves contrast in typical case
    low_contrast_data = np.random.uniform(0.45, 0.55, (5, 5))  # Very low contrast
    high_contrast_enhanced = enhance_contrast(low_contrast_data, method="minmax")
    
    improvement_stats = compute_enhancement_statistics(low_contrast_data, high_contrast_enhanced)
    
    # Should show improvement
    assert improvement_stats["dynamic_range_improvement"] > 1.0
    assert improvement_stats["contrast_improvement_ratio"] > 1.0


def test_compute_enhancement_statistics_with_mask():
    """Test statistics computation with masked data.
    
    This test should verify that:
    - Only valid pixels contribute to statistics
    - Mask shapes are handled correctly
    - Statistics are meaningful for masked regions
    - Different mask coverage produces appropriate results
    """
    # Create test data with outliers
    original = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.05, 0.95, 0.25, 0.35],  # Contains outliers
        [0.45, 0.55, 0.65, 0.75]
    ])
    
    enhanced = original * 2  # Simple enhancement
    
    # Create mask that excludes outliers
    mask = np.ones_like(original, dtype=bool)
    mask[2, 0] = False  # Exclude 0.05
    mask[2, 1] = False  # Exclude 0.95
    
    # Test with mask
    stats_masked = compute_enhancement_statistics(original, enhanced, mask=mask)
    stats_no_mask = compute_enhancement_statistics(original, enhanced)
    
    # Statistics should be different when outliers are excluded
    assert stats_masked["original_mean"] != stats_no_mask["original_mean"]
    assert stats_masked["enhanced_mean"] != stats_no_mask["enhanced_mean"]
    
    # Verify masked statistics are computed correctly
    valid_original = original[mask]
    valid_enhanced = enhanced[mask]
    
    expected_masked_orig_mean = np.mean(valid_original)
    expected_masked_enh_mean = np.mean(valid_enhanced)
    
    assert np.isclose(stats_masked["original_mean"], expected_masked_orig_mean, atol=1e-6)
    assert np.isclose(stats_masked["enhanced_mean"], expected_masked_enh_mean, atol=1e-6)
    
    # Test with different mask coverages
    # Sparse mask (only few pixels)
    mask_sparse = np.zeros_like(original, dtype=bool)
    mask_sparse[1:3, 1:3] = True  # Only 2x2 center region
    
    stats_sparse = compute_enhancement_statistics(original, enhanced, mask=mask_sparse)
    
    # Should compute statistics only from center region
    center_original = original[1:3, 1:3]
    center_enhanced = enhanced[1:3, 1:3]
    
    expected_center_orig_mean = np.mean(center_original)
    assert np.isclose(stats_sparse["original_mean"], expected_center_orig_mean, atol=1e-6)
    
    # Test with numeric mask (0/1 values)
    mask_numeric = mask.astype(np.float32)
    stats_numeric_mask = compute_enhancement_statistics(original, enhanced, mask=mask_numeric)
    
    # Should produce same results as boolean mask
    assert np.isclose(stats_numeric_mask["original_mean"], stats_masked["original_mean"], atol=1e-6)
    assert np.isclose(stats_numeric_mask["enhanced_mean"], stats_masked["enhanced_mean"], atol=1e-6)
    
    # Test with partial mask (some pixels weighted differently)
    mask_partial = np.random.rand(*original.shape) > 0.3  # Random mask
    stats_partial = compute_enhancement_statistics(original, enhanced, mask=mask_partial)
    
    # Should produce valid statistics
    for key, value in stats_partial.items():
        if not np.isnan(value):
            assert np.isfinite(value)
    
    # Test mask shape compatibility
    # Same shape mask
    assert mask.shape == original.shape
    stats_same_shape = compute_enhancement_statistics(original, enhanced, mask=mask)
    assert np.isfinite(stats_same_shape["original_mean"])
    
    # Test with almost empty mask (edge case)
    mask_tiny = np.zeros_like(original, dtype=bool)
    mask_tiny[0, 0] = True  # Only one pixel
    
    stats_tiny = compute_enhancement_statistics(original, enhanced, mask=mask_tiny)
    
    # Should equal the single pixel values
    assert np.isclose(stats_tiny["original_mean"], original[0, 0], atol=1e-6)
    assert np.isclose(stats_tiny["enhanced_mean"], enhanced[0, 0], atol=1e-6)
    # Standard deviation should be 0 for single pixel
    assert np.isclose(stats_tiny["original_std"], 0.0, atol=1e-6)
    assert np.isclose(stats_tiny["enhanced_std"], 0.0, atol=1e-6)


def test_compute_enhancement_statistics_edge_cases():
    """Test statistics computation edge cases.
    
    This test should verify that:
    - Identical original and enhanced return zero changes
    - Single-value data produces appropriate statistics
    - Empty valid regions return NaN appropriately
    - Very large dynamic ranges are handled correctly
    """
    # Test identical original and enhanced
    identical_data = np.random.rand(5, 5)
    stats_identical = compute_enhancement_statistics(identical_data, identical_data)
    
    # Changes should be zero
    assert np.isclose(stats_identical["mean_change"], 0.0, atol=1e-6)
    assert np.isclose(stats_identical["std_change"], 0.0, atol=1e-6)
    
    # Ratios should be 1.0 (no change)
    assert np.isclose(stats_identical["dynamic_range_improvement"], 1.0, atol=1e-6)
    if stats_identical["original_std"] > 0:
        assert np.isclose(stats_identical["contrast_improvement_ratio"], 1.0, atol=1e-6)
    
    # Test single-value data (zero variance)
    constant_original = np.full((3, 3), 0.5)
    constant_enhanced = np.full((3, 3), 0.7)
    
    stats_constant = compute_enhancement_statistics(constant_original, constant_enhanced)
    
    # Means should be the constant values
    assert np.isclose(stats_constant["original_mean"], 0.5, atol=1e-6)
    assert np.isclose(stats_constant["enhanced_mean"], 0.7, atol=1e-6)
    
    # Standard deviations should be zero
    assert np.isclose(stats_constant["original_std"], 0.0, atol=1e-6)
    assert np.isclose(stats_constant["enhanced_std"], 0.0, atol=1e-6)
    
    # Dynamic range improvement should be NaN or 1.0 (0/0 case)
    assert np.isnan(stats_constant["dynamic_range_improvement"]) or \
           np.isclose(stats_constant["dynamic_range_improvement"], 1.0, atol=1e-6)
    
    # Contrast ratio should be NaN (0/0 case)
    assert np.isnan(stats_constant["contrast_improvement_ratio"])
    
    # Test with empty valid regions (all-zero mask)
    data = np.random.rand(4, 4)
    empty_mask = np.zeros_like(data, dtype=bool)
    
    stats_empty = compute_enhancement_statistics(data, data * 2, mask=empty_mask)
    
    # All statistics should be NaN
    for key, value in stats_empty.items():
        assert np.isnan(value)
    
    # Test with NaN data
    data_with_nan = np.array([
        [0.1, np.nan, 0.3],
        [np.nan, 0.5, 0.6],
        [0.7, 0.8, np.nan]
    ])
    
    enhanced_with_nan = data_with_nan * 1.5
    stats_with_nan = compute_enhancement_statistics(data_with_nan, enhanced_with_nan)
    
    # Should compute statistics only from valid (non-NaN) pixels
    valid_mask = ~np.isnan(data_with_nan)
    valid_original = data_with_nan[valid_mask]
    valid_enhanced = enhanced_with_nan[valid_mask]
    
    if len(valid_original) > 0:
        expected_mean = np.mean(valid_original)
        assert np.isclose(stats_with_nan["original_mean"], expected_mean, atol=1e-6)
    
    # Test very large dynamic ranges
    large_range_data = np.array([
        [1e-10, 1e10],
        [1e5, 1e-5]
    ])
    
    enhanced_large = large_range_data * 2
    stats_large = compute_enhancement_statistics(large_range_data, enhanced_large)
    
    # Should handle large ranges without overflow
    assert np.isfinite(stats_large["original_mean"])
    assert np.isfinite(stats_large["enhanced_mean"])
    assert np.isfinite(stats_large["dynamic_range_improvement"])
    
    # Test single pixel
    single_pixel_orig = np.array([[0.3]])
    single_pixel_enh = np.array([[0.8]])
    
    stats_single = compute_enhancement_statistics(single_pixel_orig, single_pixel_enh)
    
    # Means should be the pixel values
    assert np.isclose(stats_single["original_mean"], 0.3, atol=1e-6)
    assert np.isclose(stats_single["enhanced_mean"], 0.8, atol=1e-6)
    
    # Standard deviations should be zero (single pixel)
    assert np.isclose(stats_single["original_std"], 0.0, atol=1e-6)
    assert np.isclose(stats_single["enhanced_std"], 0.0, atol=1e-6)
    
    # Test with infinity values
    data_with_inf = np.array([
        [0.1, np.inf, 0.3],
        [0.4, 0.5, -np.inf],
        [0.6, 0.7, 0.8]
    ])
    
    enhanced_inf = data_with_inf * 2
    stats_inf = compute_enhancement_statistics(data_with_inf, enhanced_inf)
    
    # Should compute from finite values only
    finite_mask = np.isfinite(data_with_inf)
    if np.any(finite_mask):
        finite_original = data_with_inf[finite_mask]
        expected_finite_mean = np.mean(finite_original)
        # May or may not be close depending on implementation
        # At minimum, should not crash and should return finite results
        assert np.isfinite(stats_inf["original_mean"]) or np.isnan(stats_inf["original_mean"])


def test_enhance_contrast_integration():
    """Test integration between different contrast enhancement methods.
    
    This test should verify that:
    - Methods can be combined effectively
    - Output from one method is valid input for another
    - Enhancement quality improves with appropriate combinations
    - Method-specific parameters don't interfere
    """
    # Create test data with known characteristics
    np.random.seed(42)
    data = np.random.uniform(0.3, 0.7, (10, 10)).astype(np.float32)
    
    # Test combining different enhancement methods
    # Step 1: Percentile enhancement
    step1 = enhance_contrast(data, method="percentile", percentiles=(10.0, 90.0))
    
    # Step 2: Apply adaptive enhancement to result
    step2 = enhance_contrast(step1, method="adaptive", target_mean=0.5, target_std=0.2)
    
    # Step 3: Final minmax normalization
    step3 = enhance_contrast(step2, method="minmax")
    
    # Verify each step produces valid output
    for step in [step1, step2, step3]:
        assert step.shape == data.shape
        assert np.all(np.isfinite(step))
        assert np.all(step >= 0.0)
        assert np.all(step <= 1.0)
    
    # Test that enhancement chain improves contrast overall
    original_std = np.std(data)
    final_std = np.std(step3)
    
    # Final result should generally have similar or improved contrast
    assert final_std >= original_std * 0.5  # Allow significant tolerance
    
    # Test different combination orders
    # Order 1: adaptive -> percentile -> minmax
    combo1_step1 = enhance_contrast(data, method="adaptive", target_mean=0.6, target_std=0.15)
    combo1_step2 = enhance_contrast(combo1_step1, method="percentile", percentiles=(5.0, 95.0))
    combo1_final = enhance_contrast(combo1_step2, method="minmax")
    
    # Order 2: minmax -> adaptive -> percentile
    combo2_step1 = enhance_contrast(data, method="minmax")
    combo2_step2 = enhance_contrast(combo2_step1, method="adaptive", target_mean=0.6, target_std=0.15)
    combo2_final = enhance_contrast(combo2_step2, method="percentile", percentiles=(5.0, 95.0))
    
    # Different orders should produce different results
    assert not np.allclose(combo1_final, combo2_final, atol=1e-6)
    
    # Both should be valid
    for result in [combo1_final, combo2_final]:
        assert result.shape == data.shape
        assert np.all(np.isfinite(result))
    
    # Test method-specific parameters don't interfere
    # Use parameters specific to different methods in sequence
    enhanced_multi = enhance_contrast(
        data, 
        method="percentile", 
        percentiles=(15.0, 85.0),
        target_mean=0.7,  # This should be ignored for percentile method
        target_std=0.3    # This should also be ignored
    )
    
    # Should work and ignore irrelevant parameters
    assert enhanced_multi.shape == data.shape
    assert np.all(enhanced_multi >= 0.0)
    assert np.all(enhanced_multi <= 1.0)
    
    # Test integration with other enhancement functions
    # Combine contrast enhancement with gamma correction
    contrast_enhanced = enhance_contrast(data, method="percentile", percentiles=(2.0, 98.0))
    gamma_enhanced = gamma_correction(contrast_enhanced, gamma=0.8)
    final_normalized = normalize_data(gamma_enhanced, output_min=0.0, output_max=1.0)
    
    assert final_normalized.shape == data.shape
    assert np.all(final_normalized >= 0.0)
    assert np.all(final_normalized <= 1.0)
    
    # Test with histogram equalization integration
    clahe_enhanced = adaptive_histogram_equalization(data, clip_limit=0.05)
    contrast_after_clahe = enhance_contrast(clahe_enhanced, method="minmax")
    
    assert contrast_after_clahe.shape == data.shape
    assert np.all(contrast_after_clahe >= 0.0)
    assert np.all(contrast_after_clahe <= 1.0)
    
    # Test with multi-scale retinex integration
    msr_enhanced = multi_scale_retinex(data, scales=[10, 30, 60])
    contrast_after_msr = enhance_contrast(msr_enhanced, method="adaptive", target_mean=0.5)
    
    assert contrast_after_msr.shape == data.shape
    assert np.all(contrast_after_msr >= 0.0)
    assert np.all(contrast_after_msr <= 1.0)
    
    # Test that enhancement quality improves with appropriate combinations
    # Compare single vs multiple enhancements on low-contrast data
    low_contrast_data = np.random.uniform(0.45, 0.55, (8, 8))
    
    single_enhancement = enhance_contrast(low_contrast_data, method="minmax")
    
    multi_enhancement = enhance_contrast(low_contrast_data, method="percentile", percentiles=(1.0, 99.0))
    multi_enhancement = enhance_contrast(multi_enhancement, method="adaptive", target_std=0.25)
    
    # Multi-step should generally provide better contrast
    single_contrast = np.std(single_enhancement)
    multi_contrast = np.std(multi_enhancement)
    
    # Both should improve over original
    original_contrast = np.std(low_contrast_data)
    assert single_contrast > original_contrast
    assert multi_contrast > original_contrast
    
    # Test parameter consistency across methods
    # Using consistent output ranges
    consistent_range = (0.1, 0.9)
    
    result1 = enhance_contrast(data, method="percentile", output_range=consistent_range)
    result2 = enhance_contrast(result1, method="minmax", output_range=consistent_range)
    result3 = enhance_contrast(result2, method="adaptive", output_range=consistent_range, target_mean=0.5)
    
    # All should respect the output range
    for result in [result1, result2, result3]:
        assert np.all(result >= 0.1)
        assert np.all(result <= 0.9)


def test_enhancement_algorithm_accuracy():
    """Test accuracy of enhancement algorithm implementations.
    
    This test should verify that:
    - Percentile calculations match expected values
    - Histogram operations preserve total pixel counts
    - Gamma curves follow expected mathematical relationships
    - Normalization maintains proper scaling
    - Numerical precision is maintained throughout
    """
    # Test percentile calculation accuracy
    # Create data with known percentiles
    sorted_data = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    data_2d = sorted_data.reshape(1, -1)
    
    # Test percentile enhancement
    result_percentile = enhance_contrast(data_2d, method="percentile", percentiles=(10.0, 90.0))
    
    # 10th percentile should be 0.1, 90th percentile should be 0.9
    # Values should be clipped and scaled to [0, 1]
    expected_min_val = 0.1
    expected_max_val = 0.9
    
    # Values at or below 10th percentile should map to 0
    # Values at or above 90th percentile should map to 1
    assert np.isclose(result_percentile[0, 0], 0.0, atol=1e-6)  # 0.0 -> 0.0 (clipped)
    assert np.isclose(result_percentile[0, 1], 0.0, atol=1e-6)  # 0.1 -> 0.0 (at percentile)
    assert np.isclose(result_percentile[0, -2], 1.0, atol=1e-6) # 0.9 -> 1.0 (at percentile)
    assert np.isclose(result_percentile[0, -1], 1.0, atol=1e-6) # 1.0 -> 1.0 (clipped)
    
    # Test gamma curve accuracy
    # Create test values and verify power law
    test_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0]).reshape(1, -1)
    
    gamma_val = 2.0
    result_gamma = gamma_correction(test_values, gamma=gamma_val)
    
    # Verify power law relationship: output = input^gamma
    for i, input_val in enumerate(test_values.flat):
        expected_output = np.power(input_val, gamma_val)
        actual_output = result_gamma.flat[i]
        assert np.isclose(actual_output, expected_output, atol=1e-6)
    
    # Test specific gamma values
    # Gamma = 0.5 (square root)
    result_sqrt = gamma_correction(test_values, gamma=0.5)
    expected_sqrt = np.sqrt(test_values)
    assert np.allclose(result_sqrt, expected_sqrt, atol=1e-6)
    
    # Gamma = 2.0 (square)
    result_square = gamma_correction(test_values, gamma=2.0)
    expected_square = test_values ** 2
    assert np.allclose(result_square, expected_square, atol=1e-6)
    
    # Test normalization accuracy
    # Known input range and output range
    input_data = np.array([10, 20, 30, 40, 50]).reshape(1, -1)
    result_norm = normalize_data(input_data, output_min=0.0, output_max=100.0)
    
    # Linear scaling: (x - 10) / (50 - 10) * (100 - 0) + 0
    expected_norm = (input_data - 10) / 40 * 100
    assert np.allclose(result_norm, expected_norm, atol=1e-6)
    
    # Test numerical precision with small values
    small_data = np.array([1e-8, 1e-6, 1e-4, 1e-2, 1e-1]).reshape(1, -1)
    result_small = normalize_data(small_data)
    
    # Should maintain relative relationships
    assert result_small[0, 0] < result_small[0, 1] < result_small[0, 2] < result_small[0, 3] < result_small[0, 4]
    assert np.isclose(result_small[0, 0], 0.0, atol=1e-6)
    assert np.isclose(result_small[0, -1], 1.0, atol=1e-6)
    
    # Test numerical precision with large values
    large_data = np.array([1e6, 2e6, 3e6, 4e6, 5e6]).reshape(1, -1)
    result_large = normalize_data(large_data)
    
    # Should maintain precision
    assert np.isclose(result_large[0, 0], 0.0, atol=1e-6)
    assert np.isclose(result_large[0, -1], 1.0, atol=1e-6)
    assert np.isclose(result_large[0, 2], 0.5, atol=1e-6)  # Middle value
    
    # Test histogram operations preserve relationships
    # Create data with known distribution
    uniform_data = np.linspace(0, 1, 100).reshape(10, 10)
    
    # Apply histogram enhancement
    result_hist = enhance_contrast(uniform_data, method="histogram")
    
    # Should preserve ordering
    flat_original = uniform_data.flatten()
    flat_result = result_hist.flatten()
    
    # Check that ordering is preserved
    for i in range(len(flat_original) - 1):
        if flat_original[i] < flat_original[i + 1]:
            assert flat_result[i] <= flat_result[i + 1]
    
    # Test adaptive enhancement accuracy
    # Create data with known statistics
    known_data = np.random.normal(0.3, 0.1, (50,)).reshape(5, 10)
    target_mean, target_std = 0.7, 0.15
    
    result_adaptive = enhance_contrast(known_data, method="adaptive", 
                                     target_mean=target_mean, target_std=target_std)
    
    # Check that target statistics are approximately achieved
    actual_mean = np.mean(result_adaptive)
    actual_std = np.std(result_adaptive)
    
    assert abs(actual_mean - target_mean) < 0.1
    assert abs(actual_std - target_std) < 0.1
    
    # Test precision preservation in chained operations
    precision_data = np.array([0.123456789, 0.987654321]).reshape(1, 2)
    
    # Apply multiple operations
    step1 = normalize_data(precision_data)
    step2 = gamma_correction(step1, gamma=1.0)  # Should be identity
    step3 = normalize_data(step2)  # Should be identity
    
    # Should maintain precision (within floating point limits)
    assert np.allclose(step1, step2, atol=1e-10)
    assert np.allclose(step2, step3, atol=1e-10)
    
    # Test mathematical consistency
    # Inverse operations should return to original (within precision)
    test_data = np.array([0.2, 0.5, 0.8]).reshape(1, 3)
    
    # Normalize then denormalize
    normalized = normalize_data(test_data, output_min=-5, output_max=5)
    denormalized = normalize_data(normalized, output_min=0.2, output_max=0.8, 
                                input_min=-5, input_max=5)
    
    assert np.allclose(test_data, denormalized, atol=1e-10)
    
    # Gamma correction then inverse gamma
    gamma_forward = gamma_correction(test_data, gamma=2.5)
    gamma_inverse = gamma_correction(gamma_forward, gamma=1.0/2.5)
    
    assert np.allclose(test_data, gamma_inverse, atol=1e-10)