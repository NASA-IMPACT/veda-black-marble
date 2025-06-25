"""Test functions in blackmarble.analyze.quality module.

This module tests data quality assessment and validation functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from datetime import datetime

# Import the module under test
from blackmarble.analyze.quality import (
    assess_data_quality,
    compute_quality_metrics,
    create_composite_quality_mask,
    flag_suspicious_values,
    compute_confidence_scores,
    identify_processing_artifacts,
    generate_quality_report,
    calculate_quality_weighted_composite,
)


def test_assess_data_quality():
    """Test comprehensive data quality assessment.
    
    This test should verify that:
    - Quality flags are properly interpreted and applied
    - Valid range constraints are enforced correctly
    - No-data values are identified and handled
    - Outlier detection works with configurable thresholds
    - Statistical quality metrics are computed accurately
    - Assessment results include all relevant quality indicators
    - Performance with large datasets is acceptable
    """
    # Create test data with known quality issues
    data = np.array([[1.0, 2.0, 100.0],  # 100.0 is outlier
                     [1.5, 2.5, 3.0],
                     [2.0, 3.0, 4.0]])
    
    # Test basic assessment without quality flags
    result = assess_data_quality(
        data,
        valid_range=(0.0, 10.0),
        check_outliers=True,
        outlier_threshold=3.0
    )
    
    # Verify outlier detection (100.0 is outside valid range)
    # Since 100.0 is excluded by valid range, it won't be in outlier detection
    assert result['outlier_count'] == 0  # No outliers within valid range
    assert len(result['outlier_locations']) == 0
    
    # Verify valid range checking
    assert result['valid_pixel_count'] == 8  # 100.0 is excluded
    assert result['valid_pixel_percent'] == pytest.approx(88.89, rel=0.01)
    
    # Verify statistical metrics are computed accurately
    assert 1.0 <= result['mean'] <= 4.0  # Reasonable mean for this data
    assert result['std'] > 0  # Should have non-zero standard deviation
    assert result['min'] == 1.0
    assert result['max'] == 100.0
    
    # Verify quality score
    assert 0 <= result['quality_score'] <= 100
    assert result['quality_score'] < 100  # Should be penalized for outlier
    
    # Verify warnings and recommendations
    assert isinstance(result['warnings'], list)
    assert isinstance(result['recommendations'], list)
    assert any('Outliers detected' in w for w in result['warnings'])
    
    # Test with quality flags
    quality_flags = np.array([[0, 0, 1],  # Flag at (0, 2)
                              [0, 0, 0],
                              [0, 0, 0]])
    
    result_with_flags = assess_data_quality(
        data,
        quality_flags=quality_flags,
        valid_range=(0.0, 10.0),
        check_outliers=True
    )
    
    # Quality flags should reduce valid pixel count
    assert result_with_flags['valid_pixel_count'] == 8
    assert result_with_flags['valid_pixel_percent'] < 100.0
    
    # Test with no-data values
    data_with_nodata = np.array([[1.0, 2.0, np.nan],
                                 [1.5, 2.5, 3.0],
                                 [2.0, np.nan, 4.0]])
    
    result_nodata = assess_data_quality(
        data_with_nodata,
        nodata_value=np.nan,
        check_outliers=True
    )
    
    assert result_nodata['valid_pixel_count'] == 7  # 2 NaN values
    assert result_nodata['valid_pixel_percent'] == pytest.approx(77.78, rel=0.01)
    
    # Test edge case: no valid data
    empty_data = np.full((3, 3), np.nan)
    result_empty = assess_data_quality(empty_data)
    
    assert result_empty['valid_pixel_count'] == 0
    assert result_empty['quality_score'] < 20  # Very low quality
    assert 'No valid pixels found' in result_empty['warnings']
    
    # Test performance with larger dataset
    large_data = np.random.randn(1000, 1000)
    import time
    start_time = time.time()
    result_large = assess_data_quality(large_data, check_outliers=True)
    elapsed_time = time.time() - start_time
    
    assert elapsed_time < 5.0  # Should complete within 5 seconds
    assert result_large['valid_pixel_count'] == 1000000


def test_assess_data_quality_flags():
    """Test quality assessment with quality flags.
    
    This test should verify that:
    - Different quality flag formats are handled correctly
    - Binary and multi-bit flags are interpreted properly
    - Flag combinations are processed accurately
    - Invalid or missing flags are handled gracefully
    - Quality flag weights are applied appropriately
    """
    # Create test data
    data = np.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0],
                     [7.0, 8.0, 9.0]])
    
    # Test with binary flags (0 = good, 1 = bad)
    binary_flags = np.array([[0, 0, 1],
                             [0, 1, 0],
                             [1, 0, 0]])
    
    result = assess_data_quality(data, quality_flags=binary_flags)
    
    # Should have 6 valid pixels (3 flagged as bad)
    assert result['valid_pixel_count'] == 6
    assert result['valid_pixel_percent'] == pytest.approx(66.67, rel=0.01)
    
    # Test with multi-bit flags (bitwise flags)
    # Bit 0: cloud, Bit 1: shadow, Bit 2: snow
    multibit_flags = np.array([[0, 1, 2],   # 0=clear, 1=cloud, 2=shadow
                               [4, 0, 3],   # 4=snow, 0=clear, 3=cloud+shadow
                               [0, 5, 0]])  # 0=clear, 5=cloud+snow, 0=clear
    
    result_multibit = assess_data_quality(data, quality_flags=multibit_flags)
    
    # Non-zero flags are considered bad
    assert result_multibit['valid_pixel_count'] == 4  # Only zeros are valid
    assert result_multibit['valid_pixel_percent'] == pytest.approx(44.44, rel=0.01)
    
    # Test with mismatched flag shape (should raise error)
    wrong_shape_flags = np.array([[0, 0], [0, 0]])  # 2x2 instead of 3x3
    
    with pytest.raises(ValueError, match="Quality flags shape.*must match data shape"):
        assess_data_quality(data, quality_flags=wrong_shape_flags)
    
    # Test with None flags (should work fine)
    result_no_flags = assess_data_quality(data, quality_flags=None)
    assert result_no_flags['valid_pixel_count'] == 9  # All valid
    
    # Test flag combinations with valid range
    flags_and_range = np.array([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]])
    
    result_combined = assess_data_quality(
        data,
        quality_flags=flags_and_range,
        valid_range=(2.0, 7.0)  # This excludes 1.0, 8.0, 9.0
    )
    
    # Should exclude: (0,0)=1.0, (1,1)=5.0(flagged), (2,1)=8.0, (2,2)=9.0
    assert result_combined['valid_pixel_count'] == 5
    
    # Test with float quality flags (should still work)
    float_flags = np.array([[0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0],
                            [2.0, 0.0, 0.0]])
    
    result_float_flags = assess_data_quality(data, quality_flags=float_flags)
    assert result_float_flags['valid_pixel_count'] == 7  # Non-zero are invalid


def test_assess_data_quality_valid_range():
    """Test quality assessment with valid range constraints.
    
    This test should verify that:
    - Values outside valid range are flagged appropriately
    - Range constraints work with different data types
    - Boundary conditions are handled correctly
    - Custom valid ranges override defaults
    - Range validation preserves valid data
    """
    # Test with floating point data
    float_data = np.array([[-1.0, 0.0, 0.5],
                           [1.0, 1.5, 2.0],
                           [2.5, 3.0, 3.5]])
    
    # Apply valid range [0, 2]
    result = assess_data_quality(float_data, valid_range=(0.0, 2.0))
    
    # Should exclude: -1.0, 2.5, 3.0, 3.5 (4 values)
    assert result['valid_pixel_count'] == 5
    assert result['min'] == 0.0
    assert result['max'] == 2.0
    
    # Test boundary conditions (inclusive)
    boundary_data = np.array([[0.0, 1.0, 2.0],
                              [2.0, 1.0, 0.0],
                              [0.0, 2.0, 1.0]])
    
    result_boundary = assess_data_quality(boundary_data, valid_range=(0.0, 2.0))
    assert result_boundary['valid_pixel_count'] == 9  # All should be valid
    
    # Test with integer data
    int_data = np.array([[0, 50, 100],
                         [150, 200, 250],
                         [255, 260, 300]], dtype=np.int16)
    
    result_int = assess_data_quality(int_data, valid_range=(0, 255))
    assert result_int['valid_pixel_count'] == 7  # 260 and 300 are out of range
    
    # Test with no valid range (all data should be valid)
    result_no_range = assess_data_quality(float_data, valid_range=None)
    assert result_no_range['valid_pixel_count'] == 9
    assert result_no_range['min'] == -1.0
    assert result_no_range['max'] == 3.5
    
    # Test with very narrow range
    result_narrow = assess_data_quality(float_data, valid_range=(0.9, 1.1))
    assert result_narrow['valid_pixel_count'] == 1  # Only 1.0 is valid
    
    # Test with invalid range (min >= max)
    with pytest.raises(ValueError, match="Invalid range: min.*must be less than max"):
        assess_data_quality(float_data, valid_range=(2.0, 2.0))
    
    with pytest.raises(ValueError, match="Invalid range: min.*must be less than max"):
        assess_data_quality(float_data, valid_range=(3.0, 1.0))
    
    # Test range validation preserves valid data statistics
    constrained_result = assess_data_quality(
        float_data,
        valid_range=(0.0, 2.0),
        check_outliers=False
    )
    
    # Stats should only include valid range data
    assert 0.0 <= constrained_result['mean'] <= 2.0
    assert constrained_result['std'] > 0  # Should have variation
    
    # Test with all data outside range
    out_of_range_data = np.array([[10.0, 20.0, 30.0],
                                   [40.0, 50.0, 60.0],
                                   [70.0, 80.0, 90.0]])
    
    result_all_out = assess_data_quality(out_of_range_data, valid_range=(0.0, 5.0))
    assert result_all_out['valid_pixel_count'] == 0
    assert result_all_out['quality_score'] < 20  # Very poor quality
    assert 'No valid pixels found' in result_all_out['warnings']


def test_assess_data_quality_outliers():
    """Test outlier detection in quality assessment.
    
    This test should verify that:
    - Statistical outliers are identified correctly
    - Outlier threshold parameter controls sensitivity
    - Different outlier detection methods work
    - Spatial and temporal outliers are distinguished
    - Outlier flags don't affect valid extreme values
    """
    # Create data with known outliers
    normal_data = np.random.normal(50, 10, size=(10, 10))  # Mean=50, std=10
    normal_data[0, 0] = 200  # Clear outlier
    normal_data[5, 5] = -100  # Another outlier
    normal_data[9, 9] = 150  # Third outlier
    
    # Test with default threshold (3.0 standard deviations)
    result_default = assess_data_quality(
        normal_data,
        check_outliers=True,
        outlier_threshold=3.0
    )
    
    assert result_default['outlier_count'] >= 2  # At least the extreme outliers
    assert len(result_default['outlier_locations']) == result_default['outlier_count']
    
    # Verify outlier locations contain our known outliers
    outlier_locs = result_default['outlier_locations']
    assert (0, 0) in outlier_locs  # 200 should be detected
    assert (5, 5) in outlier_locs  # -100 should be detected
    
    # Test with more sensitive threshold (2.0 standard deviations)
    result_sensitive = assess_data_quality(
        normal_data,
        check_outliers=True,
        outlier_threshold=2.0
    )
    
    assert result_sensitive['outlier_count'] > result_default['outlier_count']
    
    # Test with less sensitive threshold (4.0 standard deviations)
    result_insensitive = assess_data_quality(
        normal_data,
        check_outliers=True,
        outlier_threshold=4.0
    )
    
    assert result_insensitive['outlier_count'] < result_default['outlier_count']
    
    # Test with outliers disabled
    result_no_outliers = assess_data_quality(
        normal_data,
        check_outliers=False
    )
    
    assert result_no_outliers['outlier_count'] == 0
    assert result_no_outliers['outlier_locations'] == []
    
    # Test with uniform data (no outliers)
    uniform_data = np.ones((5, 5)) * 10.0
    result_uniform = assess_data_quality(
        uniform_data,
        check_outliers=True,
        outlier_threshold=3.0
    )
    
    assert result_uniform['outlier_count'] == 0
    assert result_uniform['std'] == 0.0  # No variation
    
    # Test outlier detection with valid range
    # Outliers within valid range should still be detected
    constrained_result = assess_data_quality(
        normal_data,
        valid_range=(-200, 300),  # Wide range includes all values
        check_outliers=True,
        outlier_threshold=3.0
    )
    
    assert constrained_result['outlier_count'] >= 2  # Should still detect statistical outliers
    
    # Test with 3D data (flattened for outlier detection)
    data_3d = np.random.normal(0, 1, size=(3, 5, 5))
    data_3d[0, 0, 0] = 10  # Outlier in first slice
    
    result_3d = assess_data_quality(
        data_3d,
        check_outliers=True,
        outlier_threshold=3.0
    )
    
    assert result_3d['outlier_count'] >= 1
    
    # Test with invalid threshold
    with pytest.raises(ValueError, match="Outlier threshold must be positive"):
        assess_data_quality(normal_data, outlier_threshold=0)
    
    with pytest.raises(ValueError, match="Outlier threshold must be positive"):
        assess_data_quality(normal_data, outlier_threshold=-1)
    
    # Test edge case: too few points for meaningful statistics
    tiny_data = np.array([[1.0, 2.0]])
    result_tiny = assess_data_quality(tiny_data, check_outliers=True)
    assert result_tiny['outlier_count'] == 0  # Not enough data for outlier detection


def test_compute_quality_metrics():
    """Test multi-dimensional quality metrics computation.
    
    This test should verify that:
    - Temporal stability metrics are calculated accurately
    - Spatial coherence measures detect inconsistencies
    - Multi-temporal data stacks are processed correctly
    - Date information enhances temporal analysis
    - Quality metrics are statistically meaningful
    - Performance with large data stacks is acceptable
    - Edge cases (single image, missing dates) are handled
    """
    # Create test data stack with known patterns
    n_times, height, width = 5, 10, 10
    
    # Create stable region and variable region
    data_stack = np.zeros((n_times, height, width))
    
    # Stable region (left half) - low temporal variation
    for t in range(n_times):
        data_stack[t, :, :5] = 50 + np.random.normal(0, 1, size=(height, 5))
    
    # Variable region (right half) - high temporal variation
    for t in range(n_times):
        data_stack[t, :, 5:] = 50 + t * 10 + np.random.normal(0, 5, size=(height, 5))
    
    # Add some NaN values to test handling
    data_stack[0, 0, 0] = np.nan
    data_stack[2, 5, 5] = np.nan
    
    # Test with dates
    dates = [datetime(2023, 1, i+1) for i in range(n_times)]
    
    metrics = compute_quality_metrics(
        data_stack,
        dates=dates,
        compute_temporal_stability=True,
        compute_spatial_coherence=True
    )
    
    # Verify all expected metrics are present
    expected_keys = {
        'observation_count', 'data_availability', 'temporal_std',
        'temporal_range', 'temporal_cv', 'spatial_gradient',
        'edge_density', 'composite_quality', 'confidence'
    }
    assert set(metrics.keys()) == expected_keys
    
    # Verify observation count
    assert metrics['observation_count'].shape == (height, width)
    assert metrics['observation_count'][0, 0] == 4  # One NaN
    assert metrics['observation_count'][1, 1] == 5  # No NaN
    
    # Verify data availability
    assert np.all(metrics['data_availability'] >= 0)
    assert np.all(metrics['data_availability'] <= 1)
    assert metrics['data_availability'][0, 0] == 0.8  # 4/5
    
    # Verify temporal stability metrics
    # Stable region should have lower std than variable region
    stable_std = np.nanmean(metrics['temporal_std'][:, :5])
    variable_std = np.nanmean(metrics['temporal_std'][:, 5:])
    assert stable_std < variable_std
    
    # Verify temporal range
    stable_range = np.nanmean(metrics['temporal_range'][:, :5])
    variable_range = np.nanmean(metrics['temporal_range'][:, 5:])
    assert stable_range < variable_range
    
    # Verify spatial gradient (should detect edge between regions)
    gradient_at_boundary = metrics['spatial_gradient'][:, 4:6]
    assert np.nanmean(gradient_at_boundary) > np.nanmean(metrics['spatial_gradient'][:, :4])
    
    # Verify composite quality scores
    assert np.all(metrics['composite_quality'] >= 0)
    assert np.all(metrics['composite_quality'] <= 100)
    
    # Verify confidence scores
    assert np.all(metrics['confidence'] >= 0)
    assert np.all(metrics['confidence'] <= 1)
    
    # Test without dates
    metrics_no_dates = compute_quality_metrics(
        data_stack,
        dates=None,
        compute_temporal_stability=True,
        compute_spatial_coherence=True
    )
    
    assert 'observation_count' in metrics_no_dates
    assert metrics_no_dates['observation_count'].shape == (height, width)
    
    # Test edge case: insufficient time steps
    with pytest.raises(ValueError, match="Need at least 2 time steps"):
        compute_quality_metrics(data_stack[0:1, :, :])  # Only 1 time step
    
    # Test with wrong date count
    with pytest.raises(ValueError, match="Dates length.*must match time dimension"):
        compute_quality_metrics(data_stack, dates=[datetime.now()])
    
    # Test performance with larger stack
    large_stack = np.random.randn(10, 100, 100)
    import time
    start_time = time.time()
    large_metrics = compute_quality_metrics(large_stack)
    elapsed_time = time.time() - start_time
    
    assert elapsed_time < 10.0  # Should complete within 10 seconds


def test_compute_quality_metrics_temporal():
    """Test temporal stability quality metrics.
    
    This test should verify that:
    - Temporal variance measures detect instability
    - Trend analysis identifies systematic changes
    - Anomaly detection flags unusual temporal behavior
    - Seasonal patterns are properly accounted for
    - Missing data doesn't bias temporal metrics
    """
    n_times, height, width = 12, 5, 5  # Monthly data for one year
    
    # Create data with different temporal patterns
    data_stack = np.zeros((n_times, height, width))
    
    # Stable pixel (0, 0) - minimal variation
    data_stack[:, 0, 0] = 50 + np.random.normal(0, 0.5, n_times)
    
    # Linear trend pixel (1, 1)
    data_stack[:, 1, 1] = np.linspace(30, 60, n_times) + np.random.normal(0, 1, n_times)
    
    # High variance pixel (2, 2)
    data_stack[:, 2, 2] = 50 + np.random.normal(0, 10, n_times)
    
    # Seasonal pattern pixel (3, 3)
    t = np.arange(n_times)
    data_stack[:, 3, 3] = 50 + 10 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 1, n_times)
    
    # Anomaly pixel (4, 4) - mostly stable with one spike
    data_stack[:, 4, 4] = 50 + np.random.normal(0, 1, n_times)
    data_stack[6, 4, 4] = 100  # Anomaly in month 7
    
    # Add missing data to test robustness
    data_stack[2, 0, 0] = np.nan
    data_stack[5, 1, 1] = np.nan
    data_stack[8, 2, 2] = np.nan
    
    metrics = compute_quality_metrics(
        data_stack,
        compute_temporal_stability=True,
        compute_spatial_coherence=False  # Focus on temporal
    )
    
    # Verify temporal variance detection
    # Note: With missing data, exact values may vary
    assert metrics['temporal_std'][0, 0] < 2.0  # Stable pixel has low std
    assert metrics['temporal_std'][2, 2] > 3.0  # High variance pixel has high std
    
    # Verify temporal range
    assert metrics['temporal_range'][1, 1] > 20  # Trend pixel has large range
    assert metrics['temporal_range'][0, 0] < 5   # Stable pixel has small range
    
    # Verify coefficient of variation
    # CV should be low for stable pixel
    assert metrics['temporal_cv'][0, 0] < 0.1
    # CV should be higher for variable pixel
    assert metrics['temporal_cv'][2, 2] > 0.1
    
    # Verify handling of pixels with all NaN
    all_nan_stack = np.full((5, 3, 3), np.nan)
    metrics_nan = compute_quality_metrics(
        all_nan_stack,
        compute_temporal_stability=True
    )
    
    assert np.all(np.isnan(metrics_nan['temporal_std']))
    assert np.all(np.isnan(metrics_nan['temporal_cv']))
    
    # Verify metrics with insufficient observations
    sparse_stack = np.full((10, 3, 3), np.nan)
    sparse_stack[0, 0, 0] = 10  # Only one valid observation
    sparse_stack[5, 0, 0] = 20  # Second valid observation
    
    metrics_sparse = compute_quality_metrics(
        sparse_stack,
        compute_temporal_stability=True
    )
    
    # Should compute metrics with 2 observations
    assert not np.isnan(metrics_sparse['temporal_std'][0, 0])
    assert metrics_sparse['observation_count'][0, 0] == 2
    
    # Verify anomaly affects quality score
    normal_quality = metrics['composite_quality'][0, 0]  # Stable pixel
    anomaly_quality = metrics['composite_quality'][4, 4]  # Anomaly pixel
    
    # Anomaly pixel should have lower quality due to high temporal variance
    assert anomaly_quality < normal_quality


def test_compute_quality_metrics_spatial():
    """Test spatial coherence quality metrics.
    
    This test should verify that:
    - Spatial autocorrelation measures are computed correctly
    - Edge detection identifies processing artifacts
    - Smoothness metrics assess spatial continuity
    - Local variance measures detect noise
    - Spatial patterns are statistically validated
    """
    n_times, height, width = 3, 20, 20
    
    # Create data with different spatial patterns
    data_stack = np.zeros((n_times, height, width))
    
    for t in range(n_times):
        # Smooth region (top-left quadrant)
        smooth_region = np.ones((10, 10)) * 50 + np.random.normal(0, 0.5, (10, 10))
        data_stack[t, :10, :10] = smooth_region
        
        # Edge/discontinuity (vertical line in center)
        data_stack[t, :, 9:11] = 100  # Bright vertical stripe
        
        # Noisy region (bottom-right quadrant)
        noisy_region = 50 + np.random.normal(0, 10, (10, 10))
        data_stack[t, 10:, 10:] = noisy_region
        
        # Checkerboard pattern (top-right) - processing artifact
        for i in range(10):
            for j in range(10, 20):
                if (i + j) % 2 == 0:
                    data_stack[t, i, j] = 30
                else:
                    data_stack[t, i, j] = 70
    
    metrics = compute_quality_metrics(
        data_stack,
        compute_temporal_stability=False,  # Focus on spatial
        compute_spatial_coherence=True
    )
    
    # Verify spatial gradient detection
    # Edge region should have high gradient
    edge_gradient = metrics['spatial_gradient'][:, 9:11]
    smooth_gradient = metrics['spatial_gradient'][:5, :5]
    assert np.nanmean(edge_gradient) > np.nanmean(smooth_gradient) * 2
    
    # Verify edge density
    # Noisy region should have high edge density
    noisy_edge_density = metrics['edge_density'][10:, 10:]
    smooth_edge_density = metrics['edge_density'][:5, :5]
    assert np.nanmean(noisy_edge_density) > np.nanmean(smooth_edge_density)
    
    # Checkerboard region should also have high edge density
    checker_edge_density = metrics['edge_density'][:10, 10:]
    assert np.nanmean(checker_edge_density) > np.nanmean(smooth_edge_density)
    
    # Verify composite quality reflects spatial coherence
    # Smooth region should have higher quality
    smooth_quality = np.nanmean(metrics['composite_quality'][:5, :5])
    noisy_quality = np.nanmean(metrics['composite_quality'][15:, 15:])
    assert smooth_quality > noisy_quality
    
    # Test with uniform data (no spatial variation)
    uniform_stack = np.ones((3, 10, 10)) * 42
    metrics_uniform = compute_quality_metrics(
        uniform_stack,
        compute_spatial_coherence=True
    )
    
    # Uniform data should have zero gradient and edge density
    assert np.all(metrics_uniform['spatial_gradient'] < 0.1)
    assert np.all(metrics_uniform['edge_density'] < 0.1)
    
    # Test with extreme gradients
    gradient_stack = np.zeros((3, 10, 10))
    for t in range(3):
        # Create sharp horizontal gradient
        gradient_stack[t, :5, :] = 0
        gradient_stack[t, 5:, :] = 100
    
    metrics_gradient = compute_quality_metrics(
        gradient_stack,
        compute_spatial_coherence=True
    )
    
    # Should detect high gradient at boundary
    boundary_gradient = metrics_gradient['spatial_gradient'][4:6, :]
    assert np.all(boundary_gradient > 10)
    
    # Test edge detection with isolated pixels (salt and pepper noise)
    salt_pepper_stack = np.ones((3, 10, 10)) * 50
    # Add isolated bright/dark pixels
    for t in range(3):
        salt_pepper_stack[t, 2, 2] = 100  # Salt
        salt_pepper_stack[t, 7, 7] = 0    # Pepper
    
    metrics_salt_pepper = compute_quality_metrics(
        salt_pepper_stack,
        compute_spatial_coherence=True
    )
    
    # Should detect high edge density around isolated pixels
    assert metrics_salt_pepper['edge_density'][2, 2] > 5
    assert metrics_salt_pepper['edge_density'][7, 7] > 5


def test_create_composite_quality_mask():
    """Test creation of composite quality masks.
    
    This test should verify that:
    - Multiple input masks are combined correctly
    - Weight parameters influence mask combination
    - Different combination methods (AND, OR, weighted) work
    - Dilation radius expands mask regions appropriately
    - Output mask format is consistent and usable
    - Memory efficiency with multiple large masks
    """
    # Create test masks with known patterns
    shape = (10, 10)
    
    # Cloud mask - top-left quadrant
    cloud_mask = np.ones(shape, dtype=bool)
    cloud_mask[:5, :5] = False  # Clouds in top-left
    
    # Shadow mask - bottom-right quadrant
    shadow_mask = np.ones(shape, dtype=bool)
    shadow_mask[5:, 5:] = False  # Shadows in bottom-right
    
    # Saturation mask - diagonal stripe
    saturation_mask = np.ones(shape, dtype=bool)
    for i in range(10):
        saturation_mask[i, i] = False  # Saturated pixels on diagonal
    
    masks = {
        'cloud': cloud_mask,
        'shadow': shadow_mask,
        'saturated': saturation_mask
    }
    
    # Test AND combination (most restrictive)
    composite_and, stats_and = create_composite_quality_mask(
        masks,
        combination_method='and'
    )
    
    # AND should only be True where ALL masks are True
    expected_valid_and = np.sum(cloud_mask & shadow_mask & saturation_mask)
    assert np.sum(composite_and) == expected_valid_and
    assert stats_and['composite_coverage'] < 50  # Should be restrictive
    
    # Test OR combination (most permissive)
    composite_or, stats_or = create_composite_quality_mask(
        masks,
        combination_method='or'
    )
    
    # OR should be True where ANY mask is True
    expected_valid_or = np.sum(cloud_mask | shadow_mask | saturation_mask)
    assert np.sum(composite_or) == expected_valid_or
    assert stats_or['composite_coverage'] > stats_and['composite_coverage']
    
    # Test weighted combination
    weights = {'cloud': 0.5, 'shadow': 0.3, 'saturated': 0.2}
    composite_weighted, stats_weighted = create_composite_quality_mask(
        masks,
        weights=weights,
        combination_method='weighted'
    )
    
    # Weighted should be between AND and OR
    assert stats_and['composite_coverage'] <= stats_weighted['composite_coverage']
    assert stats_weighted['composite_coverage'] <= stats_or['composite_coverage']
    assert 'normalized_weights' in stats_weighted
    
    # Test dilation
    composite_dilated, stats_dilated = create_composite_quality_mask(
        masks,
        combination_method='and',
        dilation_radius=1
    )
    
    # Dilation should reduce valid pixels (expands invalid regions)
    assert np.sum(composite_dilated) < np.sum(composite_and)
    assert stats_dilated['composite_coverage'] < stats_and['composite_coverage']
    
    # Verify statistics
    assert 'mask_coverage' in stats_and
    assert 'overlap_matrix' in stats_and
    assert 'combination_efficiency' in stats_and
    assert stats_and['total_pixels'] == 100
    
    # Test overlap matrix
    overlap = stats_and['overlap_matrix']
    assert overlap['cloud']['cloud'] == 100.0  # Self-overlap
    assert 0 <= overlap['cloud']['shadow'] <= 100.0
    
    # Test with empty masks dictionary
    with pytest.raises(ValueError, match="At least one mask must be provided"):
        create_composite_quality_mask({})
    
    # Test with mismatched shapes
    bad_masks = {
        'mask1': np.ones((10, 10), dtype=bool),
        'mask2': np.ones((5, 5), dtype=bool)
    }
    with pytest.raises(ValueError, match="All masks must have same shape"):
        create_composite_quality_mask(bad_masks)
    
    # Test with invalid weights
    with pytest.raises(ValueError, match="Weight keys.*must match mask keys"):
        create_composite_quality_mask(masks, weights={'wrong_key': 1.0})
    
    # Test with negative weights
    with pytest.raises(ValueError, match="All weights must be positive"):
        create_composite_quality_mask(masks, weights={'cloud': -1.0, 'shadow': 1.0, 'saturated': 1.0})
    
    # Test memory efficiency with large masks
    large_masks = {
        'mask1': np.random.random((1000, 1000)) > 0.5,
        'mask2': np.random.random((1000, 1000)) > 0.5
    }
    
    import time
    start_time = time.time()
    composite_large, stats_large = create_composite_quality_mask(large_masks)
    elapsed_time = time.time() - start_time
    
    assert elapsed_time < 5.0  # Should complete quickly
    assert composite_large.shape == (1000, 1000)


def test_create_composite_quality_mask_methods():
    """Test different quality mask combination methods.
    
    This test should verify that:
    - AND method requires all masks to be valid
    - OR method accepts any valid mask
    - Weighted method uses quality scores appropriately
    - Custom combination functions can be implemented
    - Mathematical consistency of combination results
    """
    # Create specific test patterns
    shape = (5, 5)
    
    # All True mask
    all_true = np.ones(shape, dtype=bool)
    
    # All False mask
    all_false = np.zeros(shape, dtype=bool)
    
    # Half True mask (left half)
    half_true = np.ones(shape, dtype=bool)
    half_true[:, :3] = False
    
    # Checkerboard mask
    checkerboard = np.ones(shape, dtype=bool)
    for i in range(5):
        for j in range(5):
            if (i + j) % 2 == 0:
                checkerboard[i, j] = False
    
    # Test AND method with all combinations
    masks_and = {
        'all_true': all_true,
        'half_true': half_true
    }
    
    composite_and, _ = create_composite_quality_mask(
        masks_and,
        combination_method='and'
    )
    
    # AND result should match the most restrictive mask
    assert np.array_equal(composite_and, half_true)
    
    # Test AND with one all-false mask
    masks_and_false = {
        'all_true': all_true,
        'all_false': all_false
    }
    
    composite_and_false, stats = create_composite_quality_mask(
        masks_and_false,
        combination_method='and'
    )
    
    assert np.all(composite_and_false == False)  # Should be all False
    assert stats['composite_coverage'] == 0.0
    
    # Test OR method
    masks_or = {
        'all_false': all_false,
        'half_true': half_true
    }
    
    composite_or, _ = create_composite_quality_mask(
        masks_or,
        combination_method='or'
    )
    
    # OR result should match the most permissive mask
    assert np.array_equal(composite_or, half_true)
    
    # Test weighted method with different weight scenarios
    masks_weighted = {
        'mask1': all_true,
        'mask2': all_false,
        'mask3': checkerboard
    }
    
    # Equal weights
    composite_equal, stats_equal = create_composite_quality_mask(
        masks_weighted,
        weights={'mask1': 1.0, 'mask2': 1.0, 'mask3': 1.0},
        combination_method='weighted'
    )
    
    # With equal weights, need >50% to be valid
    # mask1=1, mask2=0, mask3=0.5 avg -> weighted avg = 0.5
    # Checkerboard pixels: (1+0+0)/3 = 0.33 < 0.5 -> False
    # Non-checkerboard: (1+0+1)/3 = 0.67 > 0.5 -> True
    expected_coverage = (np.sum(checkerboard) / 25) * 100
    assert abs(stats_equal['composite_coverage'] - expected_coverage) < 1
    
    # Heavy weight on all_true mask
    composite_heavy, stats_heavy = create_composite_quality_mask(
        masks_weighted,
        weights={'mask1': 10.0, 'mask2': 1.0, 'mask3': 1.0},
        combination_method='weighted'
    )
    
    # Should be mostly True due to heavy weight on all_true
    assert stats_heavy['composite_coverage'] > 90
    
    # Heavy weight on all_false mask
    composite_false_heavy, stats_false_heavy = create_composite_quality_mask(
        masks_weighted,
        weights={'mask1': 1.0, 'mask2': 10.0, 'mask3': 1.0},
        combination_method='weighted'
    )
    
    # Should be mostly False due to heavy weight on all_false
    assert stats_false_heavy['composite_coverage'] < 20
    
    # Test mathematical consistency
    # Verify normalized weights sum to 1.0
    assert abs(sum(stats_equal['normalized_weights'].values()) - 1.0) < 1e-6
    
    # Test invalid combination method
    with pytest.raises(ValueError, match="Unknown combination method"):
        create_composite_quality_mask(masks_and, combination_method='invalid')
    
    # Test combination efficiency metric
    # For AND method, efficiency should be low when masks don't overlap much
    masks_no_overlap = {
        'top_left': all_true.copy(),
        'bottom_right': all_true.copy()
    }
    masks_no_overlap['top_left'][3:, :] = False
    masks_no_overlap['bottom_right'][:2, :] = False
    
    _, stats_no_overlap = create_composite_quality_mask(
        masks_no_overlap,
        combination_method='and'
    )
    
    assert stats_no_overlap['combination_efficiency'] < 50  # Low efficiency due to poor overlap


def test_create_composite_quality_mask_dilation():
    """Test mask dilation in composite quality mask creation.
    
    This test should verify that:
    - Dilation radius expands invalid regions correctly
    - Different dilation patterns (circular, square) work
    - Edge effects are handled properly
    - Dilation doesn't create invalid artifacts
    - Performance with large dilation radii
    """
    # Create mask with single invalid pixel in center
    shape = (11, 11)
    mask_single_pixel = np.ones(shape, dtype=bool)
    mask_single_pixel[5, 5] = False  # Invalid pixel in center
    
    masks = {'single_pixel': mask_single_pixel}
    
    # Test dilation radius 1
    composite_r1, stats_r1 = create_composite_quality_mask(
        masks,
        dilation_radius=1
    )
    
    # Should expand to 3x3 region around invalid pixel
    assert composite_r1[5, 5] == False  # Center still invalid
    assert composite_r1[4, 5] == False  # Top
    assert composite_r1[6, 5] == False  # Bottom
    assert composite_r1[5, 4] == False  # Left
    assert composite_r1[5, 6] == False  # Right
    assert composite_r1[4, 4] == False  # Corners
    assert composite_r1[6, 6] == False
    
    # Count invalid pixels after dilation
    invalid_r1 = np.sum(~composite_r1)
    assert invalid_r1 == 9  # 3x3 region
    
    # Test dilation radius 2
    composite_r2, stats_r2 = create_composite_quality_mask(
        masks,
        dilation_radius=2
    )
    
    invalid_r2 = np.sum(~composite_r2)
    assert invalid_r2 > invalid_r1  # More dilation
    assert invalid_r2 == 25  # 5x5 region
    
    # Test edge effects
    mask_edge = np.ones(shape, dtype=bool)
    mask_edge[0, 0] = False  # Invalid pixel at corner
    
    masks_edge = {'edge': mask_edge}
    composite_edge, _ = create_composite_quality_mask(
        masks_edge,
        dilation_radius=2
    )
    
    # Should handle edge properly without going out of bounds
    assert composite_edge[0, 0] == False
    assert composite_edge[0, 1] == False
    assert composite_edge[1, 0] == False
    assert composite_edge.shape == shape  # No size change
    
    # Test with multiple masks and dilation
    mask1 = np.ones(shape, dtype=bool)
    mask1[3, 3] = False
    
    mask2 = np.ones(shape, dtype=bool)
    mask2[7, 7] = False
    
    masks_multiple = {'mask1': mask1, 'mask2': mask2}
    
    composite_multi, stats_multi = create_composite_quality_mask(
        masks_multiple,
        combination_method='or',
        dilation_radius=1
    )
    
    # Both invalid regions should be dilated
    assert composite_multi[3, 3] == False
    assert composite_multi[2, 3] == False  # Dilated from first
    assert composite_multi[7, 7] == False
    assert composite_multi[6, 7] == False  # Dilated from second
    
    # Test no dilation (radius=0)
    composite_r0, stats_r0 = create_composite_quality_mask(
        masks,
        dilation_radius=0
    )
    
    # Should be identical to input
    assert np.array_equal(composite_r0, mask_single_pixel)
    
    # Test performance with large dilation radius
    large_mask = np.ones((100, 100), dtype=bool)
    # Add scattered invalid pixels
    for i in range(10, 90, 20):
        for j in range(10, 90, 20):
            large_mask[i, j] = False
    
    masks_large = {'large': large_mask}
    
    import time
    start_time = time.time()
    composite_large, _ = create_composite_quality_mask(
        masks_large,
        dilation_radius=5  # Large radius
    )
    elapsed_time = time.time() - start_time
    
    assert elapsed_time < 2.0  # Should still be fast
    assert np.sum(~composite_large) > np.sum(~large_mask)  # Dilation occurred
    
    # Verify dilation creates connected regions
    # Two nearby invalid pixels should connect with sufficient dilation
    mask_nearby = np.ones((10, 10), dtype=bool)
    mask_nearby[3, 3] = False
    mask_nearby[3, 6] = False  # 3 pixels apart
    
    masks_nearby = {'nearby': mask_nearby}
    
    composite_connect, _ = create_composite_quality_mask(
        masks_nearby,
        dilation_radius=2  # Should connect the pixels
    )
    
    # Check if path is connected
    assert composite_connect[3, 4] == False  # Should be invalid due to dilation
    assert composite_connect[3, 5] == False  # Creating connected region


def test_flag_suspicious_values():
    """Test flagging of suspicious or impossible values.
    
    This test should verify that:
    - Impossible values for each index type are flagged
    - Unlikely but possible values are optionally flagged
    - Context data improves suspicious value detection
    - Different index types have appropriate validation rules
    - Physical constraints are properly enforced
    - Statistical outliers are distinguished from errors
    """
    # Test NDVI validation
    ndvi_data = np.array([[-1.5, -0.5, 0.0],  # -1.5 is impossible
                          [0.5, 0.8, 1.0],
                          [1.2, 0.95, 0.3]])  # 1.2 is impossible
    
    flags_ndvi, stats_ndvi = flag_suspicious_values(
        ndvi_data,
        index_type='ndvi',
        flag_impossible=True,
        flag_unlikely=True
    )
    
    # Check impossible values are flagged
    assert flags_ndvi[0, 0] == True  # -1.5 outside [-1, 1]
    assert flags_ndvi[2, 0] == True  # 1.2 outside [-1, 1]
    assert stats_ndvi['impossible_count'] >= 2
    
    # Check unlikely values
    assert flags_ndvi[1, 2] == True  # 1.0 is unlikely (exact boundary)
    assert flags_ndvi[2, 1] == True  # 0.95 is very high
    
    # Test NDUI validation
    ndui_data = np.array([[-0.1, 0.0, 0.3],  # -0.1 is impossible
                          [0.5, 0.8, 0.9],   # 0.9 is unlikely
                          [1.0, 1.1, 0.4]])  # 1.1 is impossible
    
    flags_ndui, stats_ndui = flag_suspicious_values(
        ndui_data,
        index_type='ndui',
        flag_impossible=True,
        flag_unlikely=True
    )
    
    assert flags_ndui[0, 0] == True  # Negative NDUI
    assert flags_ndui[2, 1] == True  # > 1.0
    assert flags_ndui[1, 2] == True  # 0.9 is unlikely for NDUI
    
    # Test radiance validation
    radiance_data = np.array([[-5.0, 0.0, 10.0],   # -5.0 is impossible
                              [50.0, 100.0, 500.0],
                              [1000.0, 2000.0, 5.0]])  # 2000.0 is impossible
    
    flags_rad, stats_rad = flag_suspicious_values(
        radiance_data,
        index_type='radiance',
        flag_impossible=True,
        flag_unlikely=False  # Only flag impossible
    )
    
    assert flags_rad[0, 0] == True   # Negative radiance
    assert flags_rad[2, 1] == True   # > 1000 (sensor saturation)
    assert stats_rad['unlikely_count'] == 0  # Didn't check unlikely
    
    # Test with context data
    urban_mask = np.array([[True, False, False],
                           [True, True, False],
                           [False, False, False]])
    
    water_mask = np.array([[False, False, True],
                           [False, False, True],
                           [False, False, True]])
    
    context = {'urban_mask': urban_mask, 'water_mask': water_mask}
    
    # High NDVI in urban areas should be flagged
    ndvi_urban = np.array([[0.7, 0.5, 0.3],
                           [0.8, 0.7, 0.4],
                           [0.3, 0.4, 0.5]])
    
    flags_context, stats_context = flag_suspicious_values(
        ndvi_urban,
        index_type='ndvi',
        context_data=context,
        flag_unlikely=True
    )
    
    # Urban pixels with high NDVI should be flagged
    assert flags_context[0, 0] == True  # Urban with NDVI 0.7
    assert flags_context[1, 0] == True  # Urban with NDVI 0.8
    assert flags_context[1, 1] == True  # Urban with NDVI 0.7
    
    # Test severity scores
    assert np.all(stats_context['severity_scores'] >= 0)
    assert np.all(stats_context['severity_scores'] <= 3)
    
    # Test cluster detection
    # Create data with large suspicious cluster
    cluster_data = np.ones((50, 50)) * 0.5
    cluster_data[10:30, 10:30] = 1.5  # Large impossible region for NDVI
    
    flags_cluster, stats_cluster = flag_suspicious_values(
        cluster_data,
        index_type='ndvi',
        flag_impossible=True
    )
    
    # Should detect the large cluster
    assert stats_cluster['flagged_regions'] >= 1
    assert np.any(stats_cluster['severity_scores'] == 3)  # Extreme cluster
    
    # Test unsupported index type
    with pytest.raises(ValueError, match="Unsupported index type"):
        flag_suspicious_values(ndvi_data, index_type='invalid_type')
    
    # Test context data shape mismatch
    wrong_context = {'urban_mask': np.ones((5, 5))}
    with pytest.raises(ValueError, match="Context data.*shape.*must match data shape"):
        flag_suspicious_values(ndvi_data, index_type='ndvi', context_data=wrong_context)


def test_flag_suspicious_values_ndvi():
    """Test suspicious value flagging for NDVI data.
    
    This test should verify that:
    - NDVI values outside [-1, 1] range are flagged
    - Impossible combinations with other indices are detected
    - Context data (land cover, season) informs validation
    - Vegetation vs non-vegetation patterns are considered
    """
    # Create NDVI test data with various edge cases
    ndvi_data = np.array([[-1.0, -0.8, -0.5],   # Water-like values
                          [0.0, 0.3, 0.6],       # Bare soil to vegetation
                          [0.8, 0.95, 1.0]])     # Dense vegetation
    
    # Add impossible values
    ndvi_impossible = ndvi_data.copy()
    ndvi_impossible[0, 0] = -1.1  # Below minimum
    ndvi_impossible[2, 2] = 1.1   # Above maximum
    
    flags, stats = flag_suspicious_values(
        ndvi_impossible,
        index_type='ndvi',
        flag_impossible=True,
        flag_unlikely=True
    )
    
    # Verify impossible values are flagged
    assert flags[0, 0] == True  # -1.1
    assert flags[2, 2] == True  # 1.1
    assert stats['impossible_count'] >= 2
    
    # Verify exact boundaries are flagged as unlikely
    ndvi_boundary = np.array([[-1.0, 0.5, 1.0]])
    flags_boundary, _ = flag_suspicious_values(
        ndvi_boundary,
        index_type='ndvi',
        flag_unlikely=True
    )
    
    assert flags_boundary[0, 0] == True  # -1.0 is unlikely
    assert flags_boundary[0, 2] == True  # 1.0 is unlikely
    
    # Test with urban context
    urban_mask = np.array([[True, True, False],
                           [True, False, False],
                           [False, False, False]])
    
    ndvi_urban = np.array([[0.7, 0.8, 0.8],   # High NDVI
                           [0.65, 0.3, 0.4],
                           [0.5, 0.6, 0.7]])
    
    flags_urban, stats_urban = flag_suspicious_values(
        ndvi_urban,
        index_type='ndvi',
        context_data={'urban_mask': urban_mask},
        flag_unlikely=True
    )
    
    # High NDVI in urban areas should be flagged
    assert flags_urban[0, 0] == True  # Urban with 0.7 NDVI
    assert flags_urban[0, 1] == True  # Urban with 0.8 NDVI
    assert flags_urban[1, 0] == True  # Urban with 0.65 NDVI
    assert flags_urban[0, 2] == False # Non-urban with 0.8 NDVI is OK
    
    # Test with water context
    water_mask = np.array([[True, True, False],
                           [False, False, False],
                           [False, False, True]])
    
    ndvi_water = np.array([[-0.3, -0.2, 0.6],
                           [0.3, 0.4, 0.5],
                           [0.6, 0.7, 0.2]])
    
    flags_water, _ = flag_suspicious_values(
        ndvi_water,
        index_type='ndvi',
        context_data={'water_mask': water_mask},
        flag_unlikely=True
    )
    
    # Positive NDVI over water should be flagged
    assert flags_water[2, 2] == True  # Water with 0.2 NDVI
    assert flags_water[0, 0] == False # Water with negative NDVI is normal
    
    # Test extremely high NDVI values
    ndvi_extreme = np.array([[0.96, 0.97, 0.98],
                             [0.99, 0.95, 0.94],
                             [0.93, 0.92, 0.91]])
    
    flags_extreme, _ = flag_suspicious_values(
        ndvi_extreme,
        index_type='ndvi',
        flag_unlikely=True
    )
    
    # All values > 0.95 should be flagged as unlikely
    assert np.sum(flags_extreme) >= 4  # At least the > 0.95 values
    
    # Test very negative NDVI values
    ndvi_negative = np.array([[-0.6, -0.7, -0.8],
                              [-0.9, -0.5, -0.4],
                              [-0.3, -0.2, -0.1]])
    
    flags_negative, _ = flag_suspicious_values(
        ndvi_negative,
        index_type='ndvi',
        flag_unlikely=True
    )
    
    # Values < -0.5 should be flagged as unlikely
    assert flags_negative[0, 0] == True  # -0.6
    assert flags_negative[0, 1] == True  # -0.7
    assert flags_negative[0, 2] == True  # -0.8
    assert flags_negative[1, 0] == True  # -0.9


def test_flag_suspicious_values_ntl():
    """Test suspicious value flagging for nighttime lights data.
    
    This test should verify that:
    - Negative radiance values are flagged as impossible
    - Extremely high values are flagged as suspicious
    - Spatial context helps identify processing errors
    - Temporal consistency checks detect anomalies
    """
    # Create nighttime lights test data
    ntl_data = np.array([[0.0, 5.0, 10.0],      # Rural/suburban
                         [50.0, 100.0, 200.0],   # Urban
                         [500.0, 800.0, 1200.0]]) # Bright urban/impossible
    
    # Add negative values (impossible)
    ntl_impossible = ntl_data.copy()
    ntl_impossible[0, 0] = -1.0  # Negative radiance
    ntl_impossible[0, 1] = -0.1  # Small negative
    
    flags, stats = flag_suspicious_values(
        ntl_impossible,
        index_type='radiance',
        flag_impossible=True,
        flag_unlikely=True
    )
    
    # Verify negative values are flagged as impossible
    assert flags[0, 0] == True  # -1.0
    assert flags[0, 1] == True  # -0.1
    assert flags[2, 2] == True  # 1200.0 > 1000 (saturation)
    assert stats['impossible_count'] >= 3
    
    # Test with water context - bright lights over water are suspicious
    water_mask = np.array([[False, False, True],
                           [False, False, True],
                           [True, True, False]])
    
    ntl_water = np.array([[5.0, 10.0, 50.0],    # 50 over water
                          [20.0, 30.0, 100.0],   # 100 over water
                          [15.0, 25.0, 200.0]])
    
    flags_water, _ = flag_suspicious_values(
        ntl_water,
        index_type='radiance',
        context_data={'water_mask': water_mask},
        flag_unlikely=True
    )
    
    # Bright lights over water should be flagged
    assert flags_water[0, 2] == True  # 50 over water
    assert flags_water[1, 2] == True  # 100 over water
    assert flags_water[2, 0] == True  # 15 over water (still suspicious)
    assert flags_water[2, 1] == True  # 25 over water
    
    # Test statistical outlier detection
    # Create data with clear outliers
    ntl_outliers = np.random.exponential(10, size=(20, 20))  # Typical distribution
    ntl_outliers[5, 5] = 500   # Clear outlier
    ntl_outliers[10, 10] = 600 # Another outlier
    ntl_outliers[15, 15] = 700 # Third outlier
    
    flags_outliers, stats_outliers = flag_suspicious_values(
        ntl_outliers,
        index_type='radiance',
        flag_unlikely=True
    )
    
    # Statistical outliers should be flagged
    assert flags_outliers[5, 5] == True
    assert flags_outliers[10, 10] == True
    assert flags_outliers[15, 15] == True
    assert stats_outliers['unlikely_count'] >= 3
    
    # Test sensor saturation values
    ntl_saturated = np.array([[900.0, 950.0, 999.0],
                              [1000.0, 1050.0, 1100.0],
                              [1500.0, 2000.0, 3000.0]])
    
    flags_sat, stats_sat = flag_suspicious_values(
        ntl_saturated,
        index_type='radiance',
        flag_impossible=True
    )
    
    # Values > 1000 should be flagged as impossible (sensor saturation)
    assert np.sum(flags_sat[1:, :]) == 6  # All values in rows 1-2
    
    # Test zero radiance in urban context
    urban_mask = np.ones((3, 3), dtype=bool)  # All urban
    ntl_urban_zero = np.array([[0.0, 0.0, 50.0],
                               [0.0, 100.0, 150.0],
                               [200.0, 0.0, 250.0]])
    
    # Note: Current implementation doesn't specifically check for zero in urban
    # but this test structure is here for potential future enhancement
    flags_urban_zero, _ = flag_suspicious_values(
        ntl_urban_zero,
        index_type='radiance',
        context_data={'urban_mask': urban_mask},
        flag_unlikely=True
    )
    
    # Verify basic functionality still works with urban context
    assert isinstance(flags_urban_zero, np.ndarray)
    assert flags_urban_zero.shape == ntl_urban_zero.shape


def test_compute_confidence_scores():
    """Test per-pixel confidence score computation.
    
    This test should verify that:
    - Quality metrics are properly weighted in confidence calculation
    - Observation count influences confidence appropriately
    - Temporal stability affects confidence scores
    - Confidence scores are normalized to [0, 1] range
    - Mathematical accuracy of confidence algorithms
    - Edge cases (single observation, perfect stability) are handled
    """
    # Create test data
    shape = (10, 10)
    data = np.random.randn(*shape) * 10 + 50
    
    # Create quality metrics with known patterns
    quality_metrics = {
        'composite_quality': np.full(shape, 80.0),  # Good quality
        'data_availability': np.full(shape, 0.8),    # 80% availability
        'temporal_cv': np.full(shape, 0.2),          # Low variation
        'spatial_gradient': np.full(shape, 1.0)
    }
    
    # Test basic confidence computation
    confidence = compute_confidence_scores(
        data,
        quality_metrics
    )
    
    # Verify output shape and range
    assert confidence.shape == shape
    assert np.all(confidence >= 0.0)
    assert np.all(confidence <= 1.0)
    
    # Test with varying quality scores
    quality_metrics_varied = {
        'composite_quality': np.linspace(0, 100, 100).reshape(shape),
        'data_availability': np.linspace(0, 1, 100).reshape(shape)
    }
    
    confidence_varied = compute_confidence_scores(
        data,
        quality_metrics_varied
    )
    
    # Higher quality should yield higher confidence
    low_quality_conf = confidence_varied[0, 0]  # Low quality corner
    high_quality_conf = confidence_varied[-1, -1]  # High quality corner
    assert high_quality_conf > low_quality_conf
    
    # Test with observation count
    observation_count = np.full(shape, 10)
    observation_count[0, 0] = 1  # Single observation
    observation_count[5, 5] = 100  # Many observations
    
    confidence_obs = compute_confidence_scores(
        data,
        quality_metrics,
        observation_count=observation_count
    )
    
    # More observations should increase confidence
    assert confidence_obs[5, 5] > confidence_obs[0, 0]
    
    # Test with temporal stability
    temporal_stability = np.full(shape, 0.1)  # Low values = stable
    temporal_stability[3, 3] = 0.9  # High value = unstable
    
    confidence_stability = compute_confidence_scores(
        data,
        quality_metrics,
        temporal_stability=temporal_stability
    )
    
    # Stable pixels should have higher confidence
    assert confidence_stability[0, 0] > confidence_stability[3, 3]
    
    # Test edge case: very low quality
    low_quality_metrics = {
        'composite_quality': np.full(shape, 10.0),  # Very low quality
        'data_availability': np.full(shape, 0.1)    # Low availability
    }
    
    confidence_low = compute_confidence_scores(
        data,
        low_quality_metrics
    )
    
    # Should have low confidence but still valid
    assert np.all(confidence_low < 0.5)  # All low confidence
    assert np.all(confidence_low >= 0.05)  # But not zero (floor applied)
    
    # Test with NaN data
    data_with_nan = data.copy()
    data_with_nan[2, 2] = np.nan
    
    confidence_nan = compute_confidence_scores(
        data_with_nan,
        quality_metrics
    )
    
    # NaN pixels should have zero confidence
    assert confidence_nan[2, 2] == 0.0
    assert confidence_nan[0, 0] > 0  # Other pixels unaffected
    
    # Test perfect conditions
    perfect_metrics = {
        'composite_quality': np.full(shape, 100.0),  # Perfect quality
        'data_availability': np.full(shape, 1.0),    # 100% availability
        'temporal_cv': np.full(shape, 0.0)          # No variation
    }
    
    confidence_perfect = compute_confidence_scores(
        data,
        perfect_metrics,
        observation_count=np.full(shape, 50),  # Many observations
        temporal_stability=np.full(shape, 0.0)  # Perfect stability
    )
    
    # Should have very high confidence
    assert np.all(confidence_perfect > 0.9)
    
    # Test with missing required metrics
    incomplete_metrics = {'composite_quality': np.full(shape, 50.0)}
    
    with pytest.raises(ValueError, match="Quality metrics missing required keys"):
        compute_confidence_scores(data, incomplete_metrics)
    
    # Test shape mismatch
    wrong_shape_metrics = {
        'composite_quality': np.full((5, 5), 80.0),
        'data_availability': np.full((5, 5), 0.8)
    }
    
    with pytest.raises(ValueError, match="Quality metric.*shape.*must match data shape"):
        compute_confidence_scores(data, wrong_shape_metrics)


def test_compute_confidence_scores_weighting():
    """Test confidence score weighting schemes.
    
    This test should verify that:
    - Different quality metrics are weighted appropriately
    - Observation count scaling works correctly
    - Temporal stability contributes meaningfully
    - Custom weighting schemes can be implemented
    - Confidence scores reflect actual data reliability
    """
    shape = (5, 5)
    data = np.ones(shape) * 50
    
    # Create baseline metrics
    base_metrics = {
        'composite_quality': np.full(shape, 50.0),  # Medium quality
        'data_availability': np.full(shape, 0.5)    # 50% availability
    }
    
    # Test individual factor contributions
    
    # 1. Test data availability weighting (30% weight)
    high_avail_metrics = base_metrics.copy()
    high_avail_metrics['data_availability'] = np.full(shape, 1.0)  # 100%
    
    low_avail_metrics = base_metrics.copy()
    low_avail_metrics['data_availability'] = np.full(shape, 0.1)  # 10%
    
    conf_high_avail = compute_confidence_scores(data, high_avail_metrics)
    conf_low_avail = compute_confidence_scores(data, low_avail_metrics)
    
    # Availability should significantly affect confidence
    assert np.mean(conf_high_avail) > np.mean(conf_low_avail)
    availability_effect = np.mean(conf_high_avail) - np.mean(conf_low_avail)
    assert 0.1 < availability_effect < 0.4  # Reasonable range for 30% weight
    
    # 2. Test observation count scaling (25% weight)
    obs_count_low = np.full(shape, 1)
    obs_count_high = np.full(shape, 50)
    
    conf_low_obs = compute_confidence_scores(
        data, base_metrics, observation_count=obs_count_low
    )
    conf_high_obs = compute_confidence_scores(
        data, base_metrics, observation_count=obs_count_high
    )
    
    # More observations should increase confidence
    assert np.mean(conf_high_obs) > np.mean(conf_low_obs)
    obs_effect = np.mean(conf_high_obs) - np.mean(conf_low_obs)
    assert 0.05 < obs_effect < 0.3  # Reasonable range for 25% weight
    
    # 3. Test temporal stability contribution (20% weight)
    stability_good = np.full(shape, 0.1)  # Low = stable
    stability_bad = np.full(shape, 0.9)   # High = unstable
    
    conf_stable = compute_confidence_scores(
        data, base_metrics, temporal_stability=stability_good
    )
    conf_unstable = compute_confidence_scores(
        data, base_metrics, temporal_stability=stability_bad
    )
    
    # Stable data should have higher confidence
    assert np.mean(conf_stable) > np.mean(conf_unstable)
    stability_effect = np.mean(conf_stable) - np.mean(conf_unstable)
    assert 0.05 < stability_effect < 0.25  # Reasonable range for 20% weight
    
    # 4. Test diminishing returns on observation count
    obs_counts = [1, 5, 10, 20, 50, 100]
    mean_confidences = []
    
    for obs in obs_counts:
        conf = compute_confidence_scores(
            data, base_metrics, observation_count=np.full(shape, obs)
        )
        mean_confidences.append(np.mean(conf))
    
    # Verify diminishing returns (decreasing marginal benefit)
    differences = np.diff(mean_confidences)
    assert all(d > 0 for d in differences)  # Always increasing
    assert all(differences[i] >= differences[i+1] for i in range(len(differences)-1))  # Decreasing rate
    
    # 5. Test interaction between factors
    # Good everything
    good_all_metrics = {
        'composite_quality': np.full(shape, 90.0),
        'data_availability': np.full(shape, 0.9),
        'temporal_cv': np.full(shape, 0.1)
    }
    
    conf_good_all = compute_confidence_scores(
        data,
        good_all_metrics,
        observation_count=np.full(shape, 30),
        temporal_stability=np.full(shape, 0.1)
    )
    
    # Bad everything
    bad_all_metrics = {
        'composite_quality': np.full(shape, 20.0),
        'data_availability': np.full(shape, 0.2),
        'temporal_cv': np.full(shape, 2.0)
    }
    
    conf_bad_all = compute_confidence_scores(
        data,
        bad_all_metrics,
        observation_count=np.full(shape, 2),
        temporal_stability=np.full(shape, 0.8)
    )
    
    # Combined effect should be substantial
    assert np.mean(conf_good_all) > 0.8  # High confidence
    assert np.mean(conf_bad_all) < 0.3   # Low confidence
    
    # 6. Test CV-based temporal stability (when temporal_stability not provided)
    cv_metrics = base_metrics.copy()
    cv_metrics['temporal_cv'] = np.full(shape, 0.1)  # Low CV = stable
    
    conf_with_cv = compute_confidence_scores(data, cv_metrics)
    
    # Should use CV for stability assessment
    assert np.all(conf_with_cv > 0)  # Valid confidence scores
    
    # 7. Test spatial gradient contribution (10% weight)
    gradient_metrics = base_metrics.copy()
    gradient_metrics['spatial_gradient'] = np.random.rand(*shape) * 5
    
    conf_with_gradient = compute_confidence_scores(data, gradient_metrics)
    
    # Should produce valid confidence incorporating spatial info
    assert np.all(conf_with_gradient >= 0)
    assert np.all(conf_with_gradient <= 1)


def test_identify_processing_artifacts():
    """Test identification of common processing artifacts.
    
    This test should verify that:
    - Striping artifacts are detected accurately
    - Blocky patterns from resampling are identified
    - Edge effects from reprojection are flagged
    - Systematic biases are detected
    - Window size parameter affects artifact detection
    - Different artifact types are distinguished
    - False positive rate is acceptably low
    """
    # Create data with known artifacts
    shape = (50, 50)
    
    # 1. Create striping artifact (horizontal stripes)
    striped_data = np.ones(shape) * 50
    for i in range(0, shape[0], 5):
        striped_data[i, :] = 100  # Bright stripes every 5 rows
    
    artifacts_stripe = identify_processing_artifacts(
        striped_data,
        artifact_types=['striping'],
        window_size=5
    )
    
    # Should detect striping on the bright rows
    assert 'striping' in artifacts_stripe
    assert np.any(artifacts_stripe['striping'][::5, :])  # Stripes detected
    
    # 2. Create edge artifacts
    edge_data = np.ones(shape) * 50
    # Add bright edges
    edge_data[:3, :] = 100   # Top edge
    edge_data[-3:, :] = 100  # Bottom edge
    edge_data[:, :3] = 100   # Left edge
    edge_data[:, -3:] = 100  # Right edge
    
    artifacts_edge = identify_processing_artifacts(
        edge_data,
        artifact_types=['edges'],
        window_size=5
    )
    
    # Should detect edge artifacts
    assert 'edges' in artifacts_edge
    assert np.any(artifacts_edge['edges'][:3, :])   # Top edge
    assert np.any(artifacts_edge['edges'][-3:, :])  # Bottom edge
    
    # 3. Create tile boundary artifacts
    tile_data = np.ones(shape) * 50
    # Add discontinuities at tile boundaries (every 16 pixels)
    for i in range(16, shape[0], 16):
        tile_data[i-1:i+1, :] = 100  # Horizontal boundaries
    for j in range(16, shape[1], 16):
        tile_data[:, j-1:j+1] = 100  # Vertical boundaries
    
    # Note: Common tile sizes include 256, 512, etc., but for small test we'll detect patterns
    artifacts_tile = identify_processing_artifacts(
        tile_data,
        artifact_types=['tiles'],
        window_size=5
    )
    
    assert 'tiles' in artifacts_tile
    # Should detect some tile artifacts (implementation looks for standard sizes)
    
    # 4. Create resampling artifacts (checkerboard pattern)
    resampling_data = np.ones(shape) * 50
    for i in range(shape[0]):
        for j in range(shape[1]):
            if (i + j) % 2 == 0:
                resampling_data[i, j] = 70  # Checkerboard
    
    artifacts_resample = identify_processing_artifacts(
        resampling_data,
        artifact_types=['resampling'],
        window_size=3  # Small window for fine patterns
    )
    
    assert 'resampling' in artifacts_resample
    assert np.sum(artifacts_resample['resampling']) > 0  # Some artifacts detected
    
    # 5. Test with all artifact types
    # Create complex data with multiple artifacts
    complex_data = np.ones(shape) * 50
    # Add striping
    complex_data[10, :] = 100
    complex_data[20, :] = 100
    # Add edge effects
    complex_data[:2, :] = 90
    # Add some noise
    complex_data += np.random.normal(0, 2, shape)
    
    all_artifacts = identify_processing_artifacts(
        complex_data,
        artifact_types=None,  # Check all types
        window_size=5
    )
    
    # Should return all artifact types
    assert len(all_artifacts) == 4
    assert all(key in all_artifacts for key in ['striping', 'edges', 'tiles', 'resampling'])
    
    # 6. Test window size effect
    artifacts_small_window = identify_processing_artifacts(
        striped_data,
        artifact_types=['striping'],
        window_size=3
    )
    
    artifacts_large_window = identify_processing_artifacts(
        striped_data,
        artifact_types=['striping'],
        window_size=7
    )
    
    # Different window sizes may detect different amounts
    assert artifacts_small_window['striping'].shape == shape
    assert artifacts_large_window['striping'].shape == shape
    
    # 7. Test with clean data (low false positive rate)
    clean_data = np.random.normal(50, 5, shape)  # Natural variation
    
    artifacts_clean = identify_processing_artifacts(
        clean_data,
        window_size=5
    )
    
    # Should have very few detections on clean data
    total_detections = sum(np.sum(mask) for mask in artifacts_clean.values())
    false_positive_rate = total_detections / (shape[0] * shape[1])
    assert false_positive_rate < 0.1  # Less than 10% false positives
    
    # 8. Test with 3D data
    data_3d = np.ones((3, 20, 20)) * 50
    # Add striping to each slice
    for t in range(3):
        data_3d[t, 5, :] = 100
        data_3d[t, 10, :] = 100
    
    artifacts_3d = identify_processing_artifacts(
        data_3d,
        artifact_types=['striping']
    )
    
    assert artifacts_3d['striping'].shape == (3, 20, 20)
    assert np.any(artifacts_3d['striping'][:, 5, :])  # Striping detected in all slices
    
    # 9. Test invalid parameters
    with pytest.raises(ValueError, match="Data must be 2D or 3D"):
        identify_processing_artifacts(np.ones(10))  # 1D data
    
    with pytest.raises(ValueError, match="Window size must be odd integer >= 3"):
        identify_processing_artifacts(clean_data, window_size=2)
    
    with pytest.raises(ValueError, match="Unsupported artifact types"):
        identify_processing_artifacts(clean_data, artifact_types=['invalid_type'])


def test_identify_processing_artifacts_types():
    """Test identification of specific artifact types.
    
    This test should verify that:
    - Striping (horizontal/vertical) is detected
    - Checkerboard patterns are identified
    - Systematic shifts are flagged
    - Noise patterns are distinguished from signals
    - Sensor-specific artifacts are recognized
    """
    shape = (30, 30)
    
    # 1. Test horizontal striping
    h_stripe_data = np.ones(shape) * 50
    # Create systematic row differences
    h_stripe_data[5, :] = 80   # Single bright stripe
    h_stripe_data[10, :] = 20  # Single dark stripe
    h_stripe_data[15:17, :] = 90  # Double stripe
    
    artifacts_h = identify_processing_artifacts(
        h_stripe_data,
        artifact_types=['striping']
    )
    
    # Should detect horizontal stripes
    assert np.any(artifacts_h['striping'][5, :])
    assert np.any(artifacts_h['striping'][10, :])
    assert np.any(artifacts_h['striping'][15:17, :])
    
    # 2. Test vertical striping
    v_stripe_data = np.ones(shape) * 50
    # Create systematic column differences
    v_stripe_data[:, 8] = 85
    v_stripe_data[:, 16] = 15
    v_stripe_data[:, 24] = 75
    
    artifacts_v = identify_processing_artifacts(
        v_stripe_data,
        artifact_types=['striping']
    )
    
    # Should detect vertical stripes
    assert np.any(artifacts_v['striping'][:, 8])
    assert np.any(artifacts_v['striping'][:, 16])
    assert np.any(artifacts_v['striping'][:, 24])
    
    # 3. Test fine checkerboard pattern (resampling artifact)
    checker_fine = np.ones(shape) * 50
    for i in range(shape[0]):
        for j in range(shape[1]):
            if (i + j) % 2 == 0:
                checker_fine[i, j] = 60
    
    artifacts_checker_fine = identify_processing_artifacts(
        checker_fine,
        artifact_types=['resampling'],
        window_size=3  # Small window for fine pattern
    )
    
    # Should detect checkerboard as resampling artifact
    assert np.sum(artifacts_checker_fine['resampling']) > shape[0] * shape[1] * 0.1
    
    # 4. Test coarse checkerboard pattern
    checker_coarse = np.ones(shape) * 50
    block_size = 4
    for i in range(0, shape[0], block_size * 2):
        for j in range(0, shape[1], block_size * 2):
            checker_coarse[i:i+block_size, j:j+block_size] = 70
            if i+block_size < shape[0] and j+block_size < shape[1]:
                checker_coarse[i+block_size:i+2*block_size, j+block_size:j+2*block_size] = 70
    
    artifacts_checker_coarse = identify_processing_artifacts(
        checker_coarse,
        artifact_types=['resampling'],
        window_size=5
    )
    
    # Should detect block patterns
    assert np.any(artifacts_checker_coarse['resampling'])
    
    # 5. Test systematic shift (sensor calibration issue)
    shift_data = np.ones(shape) * 50
    # Top half systematically brighter
    shift_data[:shape[0]//2, :] += 20
    # Add gradient to make it more realistic
    for i in range(shape[0]//2 - 2, shape[0]//2 + 2):
        if 0 <= i < shape[0]:
            shift_data[i, :] = 50 + 10 * (i - shape[0]//2 + 2)
    
    artifacts_shift = identify_processing_artifacts(
        shift_data,
        artifact_types=['edges'],
        window_size=5
    )
    
    # Should detect discontinuity at boundary
    boundary_region = artifacts_shift['edges'][shape[0]//2-3:shape[0]//2+3, :]
    assert np.any(boundary_region)
    
    # 6. Test noise vs signal discrimination
    # Create data with structured signal + noise
    signal_data = np.ones(shape) * 50
    # Add legitimate gradient (not an artifact)
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    signal_data += (x + y) / 10  # Smooth gradient
    # Add random noise
    noise = np.random.normal(0, 5, shape)
    noisy_signal = signal_data + noise
    
    artifacts_noise = identify_processing_artifacts(
        noisy_signal,
        window_size=5
    )
    
    # Should not flag smooth gradient as artifact
    # But might detect some noise spikes
    total_artifacts = sum(np.sum(mask) for mask in artifacts_noise.values())
    artifact_rate = total_artifacts / (shape[0] * shape[1])
    assert artifact_rate < 0.2  # Most pixels should not be flagged
    
    # 7. Test banding artifact (sensor-specific)
    banding_data = np.ones(shape) * 50
    # Create periodic banding every 8 rows (sensor artifact)
    for i in range(0, shape[0], 8):
        if i < shape[0]:
            banding_data[i, :] *= 1.2  # 20% brighter
    
    artifacts_banding = identify_processing_artifacts(
        banding_data,
        artifact_types=['striping']
    )
    
    # Should detect periodic banding
    banding_rows = [i for i in range(0, shape[0], 8) if i < shape[0]]
    detected_banding = sum(1 for i in banding_rows if np.any(artifacts_banding['striping'][i, :]))
    assert detected_banding >= len(banding_rows) * 0.5  # At least half detected
    
    # 8. Test mixed artifacts
    mixed_data = np.ones(shape) * 50
    # Add multiple artifact types
    mixed_data[10, :] = 80  # Horizontal stripe
    mixed_data[:, 15] = 20  # Vertical stripe
    # Add small checkerboard region
    for i in range(20, 25):
        for j in range(20, 25):
            if (i + j) % 2 == 0:
                mixed_data[i, j] = 70
    
    artifacts_mixed = identify_processing_artifacts(
        mixed_data,
        window_size=3
    )
    
    # Should detect multiple artifact types
    assert np.any(artifacts_mixed['striping'][10, :])  # Horizontal
    assert np.any(artifacts_mixed['striping'][:, 15])  # Vertical
    assert np.any(artifacts_mixed['resampling'][20:25, 20:25])  # Checkerboard region


def test_generate_quality_report():
    """Test generation of comprehensive quality reports.
    
    This test should verify that:
    - Summary format provides concise quality overview
    - Detailed format includes comprehensive diagnostics
    - Statistical summaries are accurate and meaningful
    - Visual elements (histograms, plots) are informative
    - Report format is consistent and readable
    - Multiple assessment results are properly aggregated
    """
    # Create multiple quality assessments
    assessments = []
    
    # Good quality assessment
    good_assessment = {
        'valid_pixel_percent': 95.0,
        'quality_score': 85.0,
        'warnings': ['Minor outliers detected'],
        'recommendations': ['Suitable for analysis'],
        'valid_pixel_count': 9500,
        'mean': 50.0,
        'std': 5.0,
        'outlier_count': 10
    }
    assessments.append(good_assessment)
    
    # Medium quality assessment
    medium_assessment = {
        'valid_pixel_percent': 70.0,
        'quality_score': 60.0,
        'warnings': ['Moderate data coverage (70.0%)', 'Outliers detected (50 pixels)'],
        'recommendations': ['Monitor data quality in final products', 'Consider outlier removal for sensitive analyses'],
        'valid_pixel_count': 7000,
        'mean': 45.0,
        'std': 8.0,
        'outlier_count': 50
    }
    assessments.append(medium_assessment)
    
    # Poor quality assessment
    poor_assessment = {
        'valid_pixel_percent': 30.0,
        'quality_score': 25.0,
        'warnings': ['Low data coverage (30.0%)', 'Very poor overall quality'],
        'recommendations': ['Consider relaxing quality filters or using gap filling', 'Do not use for analysis - investigate data issues'],
        'valid_pixel_count': 3000,
        'mean': 40.0,
        'std': 15.0,
        'outlier_count': 200
    }
    assessments.append(poor_assessment)
    
    # Test summary format
    summary_report = generate_quality_report(assessments, output_format='summary')
    
    assert isinstance(summary_report, str)
    assert 'Quality Assessment Summary' in summary_report
    assert 'Overall Quality:' in summary_report
    assert 'Key Statistics:' in summary_report
    assert 'Mean quality score:' in summary_report
    assert 'Most Common Issues:' in summary_report
    
    # Verify statistics in summary
    assert '3' in summary_report  # Total assessments
    assert '56.7' in summary_report  # Mean quality score (85+60+25)/3
    assert '65.0%' in summary_report  # Mean coverage (95+70+30)/3
    
    # Test detailed format
    detailed_report = generate_quality_report(assessments, output_format='detailed')
    
    assert isinstance(detailed_report, str)
    assert 'Comprehensive Quality Assessment Report' in detailed_report
    assert 'EXECUTIVE SUMMARY' in detailed_report
    assert 'AGGREGATE STATISTICS' in detailed_report
    assert 'DETAILED ASSESSMENT RESULTS' in detailed_report
    assert 'ISSUE ANALYSIS' in detailed_report
    assert 'ACTIONABLE INSIGHTS' in detailed_report
    
    # Verify individual assessments are included
    assert 'Assessment 1:' in detailed_report
    assert 'Assessment 2:' in detailed_report
    assert 'Assessment 3:' in detailed_report
    assert '85.0/100' in detailed_report  # Good quality score
    assert '25.0/100' in detailed_report  # Poor quality score
    
    # Test JSON format
    json_report = generate_quality_report(assessments, output_format='json')
    
    assert isinstance(json_report, dict)
    assert 'timestamp' in json_report
    assert 'overall_statistics' in json_report
    assert 'individual_assessments' in json_report
    assert 'warning_summary' in json_report
    assert 'recommendation_summary' in json_report
    assert 'quality_classification' in json_report
    
    # Verify JSON statistics
    stats = json_report['overall_statistics']
    assert stats['total_assessments'] == 3
    assert abs(stats['mean_quality_score'] - 56.67) < 0.1
    assert stats['assessments_with_poor_quality'] == 2  # Score < 50
    assert stats['assessments_with_excellent_quality'] == 1  # Score >= 80
    
    # Verify quality classification
    classification = json_report['quality_classification']
    assert 'overall_quality' in classification
    assert 'confidence_level' in classification
    assert 'coverage_quality' in classification
    assert 'recommendation' in classification
    
    # Test warning and recommendation aggregation
    warning_summary = json_report['warning_summary']
    assert warning_summary['total_warnings'] > 0
    assert warning_summary['unique_warnings'] > 0
    assert len(warning_summary['most_common']) > 0
    
    # Test with empty assessments
    with pytest.raises(ValueError, match="Quality assessments list cannot be empty"):
        generate_quality_report([], output_format='summary')
    
    # Test with missing required keys
    bad_assessment = {
        'valid_pixel_percent': 50.0,
        'warnings': []  # Missing quality_score and recommendations
    }
    
    with pytest.raises(ValueError, match="Assessment.*missing required keys"):
        generate_quality_report([bad_assessment])
    
    # Test with many assessments
    many_assessments = []
    for i in range(20):
        assessment = {
            'valid_pixel_percent': 50 + i * 2,
            'quality_score': 40 + i * 3,
            'warnings': [f'Warning {i}'] if i % 2 == 0 else [],
            'recommendations': [f'Recommendation {i}'] if i % 3 == 0 else []
        }
        many_assessments.append(assessment)
    
    large_report = generate_quality_report(many_assessments, output_format='json')
    assert large_report['overall_statistics']['total_assessments'] == 20
    
    # Test invalid output format
    with pytest.raises(ValueError, match="Unsupported output format"):
        generate_quality_report(assessments, output_format='invalid')


def test_generate_quality_report_formats():
    """Test different quality report output formats.
    
    This test should verify that:
    - Summary format highlights key quality issues
    - Detailed format provides comprehensive analysis
    - JSON format enables programmatic access
    - Visual format includes charts and graphics
    - Custom formats can be implemented
    """
    # Create diverse assessments to test format handling
    assessments = [
        {
            'valid_pixel_percent': 90.0,
            'quality_score': 82.0,
            'warnings': ['Outliers detected (5 pixels)'],
            'recommendations': ['Acceptable for analysis with quality considerations'],
            'mean': 100.0,
            'std': 10.0,
            'min': 80.0,
            'max': 120.0,
            'outlier_count': 5
        },
        {
            'valid_pixel_percent': 45.0,
            'quality_score': 35.0,
            'warnings': ['Low data coverage (45.0%)', 'High outlier rate (15.0%)'],
            'recommendations': ['Use with caution - apply additional quality filters', 'Review processing parameters and input data quality'],
            'mean': 50.0,
            'std': 25.0,
            'outlier_count': 150
        }
    ]
    
    # Test summary format structure and content
    summary = generate_quality_report(assessments, output_format='summary')
    
    # Check summary is concise
    lines = summary.strip().split('\n')
    assert len(lines) < 20  # Summary should be brief
    
    # Verify key elements
    assert any('Fair' in line or 'Poor' in line or 'Good' in line for line in lines)  # Quality rating
    assert any('confidence' in line.lower() for line in lines)  # Confidence level
    assert any('%' in line for line in lines)  # Coverage percentages
    
    # Check for most common issues
    assert 'Low data coverage' in summary or 'Outliers detected' in summary
    
    # Test detailed format structure
    detailed = generate_quality_report(assessments, output_format='detailed')
    
    # Check detailed is comprehensive
    lines = detailed.strip().split('\n')
    assert len(lines) > 40  # Detailed should be extensive
    
    # Verify all sections present
    sections = [
        'EXECUTIVE SUMMARY',
        'AGGREGATE STATISTICS',
        'DETAILED ASSESSMENT RESULTS',
        'ISSUE ANALYSIS',
        'ACTIONABLE INSIGHTS'
    ]
    for section in sections:
        assert section in detailed
    
    # Verify individual assessment details
    assert 'Quality Score: 82.0/100' in detailed
    assert 'Quality Score: 35.0/100' in detailed
    assert 'Mean Value:' in detailed
    assert 'Std Dev:' in detailed
    
    # Test JSON format structure
    json_report = generate_quality_report(assessments, output_format='json')
    
    # Verify JSON is properly structured for programmatic access
    assert isinstance(json_report, dict)
    
    # Check nested structure
    assert isinstance(json_report['overall_statistics'], dict)
    assert isinstance(json_report['individual_assessments'], list)
    assert isinstance(json_report['warning_summary'], dict)
    assert isinstance(json_report['recommendation_summary'], dict)
    assert isinstance(json_report['quality_classification'], dict)
    
    # Verify JSON can be serialized
    import json
    json_str = json.dumps(json_report)
    parsed = json.loads(json_str)
    assert parsed['overall_statistics']['total_assessments'] == 2
    
    # Test format-specific features
    
    # 1. Summary focuses on actionable info
    summary_lower = summary.lower()
    assert 'recommendation' in summary_lower
    assert any(word in summary_lower for word in ['poor', 'fair', 'good', 'excellent'])
    
    # 2. Detailed includes percentage breakdowns
    assert '(50.0%)' in detailed  # Percentage notation
    assert 'assessments' in detailed
    
    # 3. JSON includes raw data for further processing
    assert all(isinstance(assessment, dict) for assessment in json_report['individual_assessments'])
    assert len(json_report['individual_assessments']) == 2
    
    # Test edge cases in formatting
    
    # All excellent quality
    excellent_assessments = [{
        'valid_pixel_percent': 98.0,
        'quality_score': 92.0,
        'warnings': [],
        'recommendations': []
    } for _ in range(3)]
    
    excellent_summary = generate_quality_report(excellent_assessments, output_format='summary')
    assert 'Excellent' in excellent_summary
    
    # All poor quality
    poor_assessments = [{
        'valid_pixel_percent': 20.0,
        'quality_score': 15.0,
        'warnings': ['Very poor overall quality', 'Low data coverage (20.0%)'],
        'recommendations': ['Do not use for analysis - investigate data issues']
    } for _ in range(3)]
    
    poor_detailed = generate_quality_report(poor_assessments, output_format='detailed')
    assert '100.0% of assessments have poor quality' in poor_detailed or 'investigate processing issues' in poor_detailed
    
    # Mixed quality for classification testing
    mixed_assessments = [
        {'valid_pixel_percent': 95.0, 'quality_score': 90.0, 'warnings': [], 'recommendations': []},
        {'valid_pixel_percent': 10.0, 'quality_score': 5.0, 'warnings': ['Critical issues'], 'recommendations': ['Do not use']}
    ]
    
    mixed_json = generate_quality_report(mixed_assessments, output_format='json')
    classification = mixed_json['quality_classification']
    
    # Should recognize the wide quality range
    assert classification['overall_quality'] in ['Fair', 'Poor']  # Average is pulled down
    
    # Test warning/recommendation aggregation in different formats
    common_warnings_assessments = [
        {
            'valid_pixel_percent': 60.0,
            'quality_score': 55.0,
            'warnings': ['Common warning', 'Unique warning 1'],
            'recommendations': ['Common recommendation']
        },
        {
            'valid_pixel_percent': 65.0,
            'quality_score': 58.0,
            'warnings': ['Common warning', 'Unique warning 2'],
            'recommendations': ['Common recommendation']
        }
    ]
    
    # Summary should highlight most common
    common_summary = generate_quality_report(common_warnings_assessments, output_format='summary')
    assert 'Common warning' in common_summary
    
    # JSON should provide counts
    common_json = generate_quality_report(common_warnings_assessments, output_format='json')
    warning_counts = {w[0]: w[1] for w in common_json['warning_summary']['most_common']}
    assert warning_counts.get('Common warning', 0) == 2


def test_calculate_quality_weighted_composite():
    """Test quality-weighted composite creation.
    
    This test should verify that:
    - Quality scores properly weight pixel contributions
    - Weighted mean method produces reasonable composites
    - Quality threshold excludes low-quality pixels
    - Different weighting methods produce distinct results
    - Mathematical accuracy of weighted averaging
    - No-data handling in weighted composites
    - Performance with large data stacks
    """
    # Create test data stack
    n_times, height, width = 5, 10, 10
    
    # Create data with varying quality
    data_stack = np.zeros((n_times, height, width))
    quality_scores = np.zeros((n_times, height, width))
    
    # Time 0: Low quality, value=10
    data_stack[0, :, :] = 10
    quality_scores[0, :, :] = 0.2  # 20% quality
    
    # Time 1: Medium quality, value=20
    data_stack[1, :, :] = 20
    quality_scores[1, :, :] = 0.5  # 50% quality
    
    # Time 2: High quality, value=30
    data_stack[2, :, :] = 30
    quality_scores[2, :, :] = 0.8  # 80% quality
    
    # Time 3: Very high quality, value=40
    data_stack[3, :, :] = 40
    quality_scores[3, :, :] = 0.9  # 90% quality
    
    # Time 4: Medium quality with some NaN, value=50
    data_stack[4, :, :] = 50
    data_stack[4, 0:2, 0:2] = np.nan  # Add some missing data
    quality_scores[4, :, :] = 0.6  # 60% quality
    
    # Test weighted mean method
    composite_wmean, quality_wmean = calculate_quality_weighted_composite(
        data_stack,
        quality_scores,
        method='weighted_mean'
    )
    
    # Verify output shapes
    assert composite_wmean.shape == (height, width)
    assert quality_wmean.shape == (height, width)
    
    # Verify weighted mean calculation
    # For pixels without NaN, weighted mean should favor high-quality observations
    # Expected: (10*0.2 + 20*0.5 + 30*0.8 + 40*0.9 + 50*0.6) / (0.2+0.5+0.8+0.9+0.6)
    # = (2 + 10 + 24 + 36 + 30) / 3.0 = 102 / 3.0 = 34.0
    expected_wmean = 34.0
    assert abs(composite_wmean[5, 5] - expected_wmean) < 0.1
    
    # Verify quality is also weighted
    expected_quality = (0.2*0.2 + 0.5*0.5 + 0.8*0.8 + 0.9*0.9 + 0.6*0.6) / 3.0
    assert abs(quality_wmean[5, 5] - expected_quality) < 0.1
    
    # Verify NaN handling
    # Pixels with NaN in time 4 should exclude that time
    expected_wmean_nan = (10*0.2 + 20*0.5 + 30*0.8 + 40*0.9) / (0.2+0.5+0.8+0.9)
    assert abs(composite_wmean[0, 0] - expected_wmean_nan) < 0.1
    
    # Test best quality method
    composite_best, quality_best = calculate_quality_weighted_composite(
        data_stack,
        quality_scores,
        method='best_quality'
    )
    
    # Should select time 3 (highest quality=0.9, value=40)
    assert np.all(composite_best[2:, 2:] == 40)  # Away from NaN region
    assert np.all(quality_best[2:, 2:] == 0.9)
    
    # Test threshold method
    composite_thresh, quality_thresh = calculate_quality_weighted_composite(
        data_stack,
        quality_scores,
        method='threshold',
        threshold=0.7  # Only use quality >= 0.7
    )
    
    # Should only use times 2 and 3 (quality 0.8 and 0.9)
    # Mean of 30 and 40 = 35
    assert abs(composite_thresh[5, 5] - 35.0) < 0.1
    # Mean quality of 0.8 and 0.9 = 0.85
    assert abs(quality_thresh[5, 5] - 0.85) < 0.1
    
    # Test with quality scores in 0-100 range
    quality_scores_100 = quality_scores * 100  # Convert to 0-100
    
    composite_100, quality_100 = calculate_quality_weighted_composite(
        data_stack,
        quality_scores_100,
        method='weighted_mean'
    )
    
    # Results should be identical to 0-1 range
    assert np.allclose(composite_100, composite_wmean, rtol=1e-5)
    assert np.allclose(quality_100, quality_wmean, rtol=1e-5)
    
    # Test with all NaN at a pixel
    data_all_nan = data_stack.copy()
    data_all_nan[:, 7, 7] = np.nan
    
    composite_nan, quality_nan = calculate_quality_weighted_composite(
        data_all_nan,
        quality_scores,
        method='weighted_mean'
    )
    
    assert np.isnan(composite_nan[7, 7])
    assert np.isnan(quality_nan[7, 7])
    
    # Test edge case: single time step
    single_data = data_stack[0:1, :, :]
    single_quality = quality_scores[0:1, :, :]
    
    composite_single, quality_single = calculate_quality_weighted_composite(
        single_data,
        single_quality,
        method='weighted_mean'
    )
    
    assert np.all(composite_single == 10)  # Only value available
    assert np.all(quality_single == 0.2)   # Only quality available
    
    # Test performance with larger stack
    large_stack = np.random.randn(20, 100, 100) * 10 + 50
    large_quality = np.random.rand(20, 100, 100)
    
    import time
    start_time = time.time()
    composite_large, quality_large = calculate_quality_weighted_composite(
        large_stack,
        large_quality,
        method='weighted_mean'
    )
    elapsed_time = time.time() - start_time
    
    assert elapsed_time < 5.0  # Should complete within 5 seconds
    assert composite_large.shape == (100, 100)
    
    # Test invalid method
    with pytest.raises(ValueError, match="Unsupported compositing method"):
        calculate_quality_weighted_composite(
            data_stack,
            quality_scores,
            method='invalid_method'
        )
    
    # Test shape mismatch
    wrong_quality = np.ones((3, 5, 5))  # Wrong shape
    with pytest.raises(ValueError, match="Quality scores shape.*must match data stack shape"):
        calculate_quality_weighted_composite(data_stack, wrong_quality)
    
    # Test invalid dimensions
    with pytest.raises(ValueError, match="Data stack must be 3D"):
        calculate_quality_weighted_composite(data_stack[0], quality_scores[0])


def test_calculate_quality_weighted_composite_methods():
    """Test different quality-weighted compositing methods.
    
    This test should verify that:
    - Weighted mean emphasizes high-quality pixels
    - Weighted median is robust to outliers
    - Best pixel selection chooses highest quality
    - Custom weighting functions can be implemented
    - Mathematical properties of each method
    """
    # Create test scenario with clear quality differences
    n_times, height, width = 5, 5, 5
    data_stack = np.zeros((n_times, height, width))
    quality_scores = np.zeros((n_times, height, width))
    
    # Set up specific test case at pixel (2, 2)
    # Values: [10, 20, 100, 40, 50] (100 is outlier)
    # Quality: [0.3, 0.7, 0.2, 0.9, 0.8]
    data_stack[:, 2, 2] = [10, 20, 100, 40, 50]
    quality_scores[:, 2, 2] = [0.3, 0.7, 0.2, 0.9, 0.8]
    
    # Fill rest with normal values
    for i in range(height):
        for j in range(width):
            if i != 2 or j != 2:
                data_stack[:, i, j] = np.random.normal(30, 5, n_times)
                quality_scores[:, i, j] = np.random.uniform(0.5, 0.9, n_times)
    
    # Test weighted mean method
    composite_wmean, quality_wmean = calculate_quality_weighted_composite(
        data_stack,
        quality_scores,
        method='weighted_mean'
    )
    
    # Calculate expected weighted mean for pixel (2, 2)
    values = data_stack[:, 2, 2]
    weights = quality_scores[:, 2, 2]
    expected_wmean = np.sum(values * weights) / np.sum(weights)
    
    assert abs(composite_wmean[2, 2] - expected_wmean) < 0.01
    
    # Weighted mean should de-emphasize the outlier (100) due to low quality (0.2)
    assert composite_wmean[2, 2] < 50  # Should be pulled down from simple mean
    
    # Test best quality method
    composite_best, quality_best = calculate_quality_weighted_composite(
        data_stack,
        quality_scores,
        method='best_quality'
    )
    
    # Should select value with highest quality (0.9) -> value 40
    assert composite_best[2, 2] == 40
    assert quality_best[2, 2] == 0.9
    
    # Test threshold method with different thresholds
    # Threshold 0.6: includes times 1, 3, 4 (qualities 0.7, 0.9, 0.8)
    composite_t6, quality_t6 = calculate_quality_weighted_composite(
        data_stack,
        quality_scores,
        method='threshold',
        threshold=0.6
    )
    
    # Mean of [20, 40, 50] = 36.67
    expected_t6 = np.mean([20, 40, 50])
    assert abs(composite_t6[2, 2] - expected_t6) < 0.01
    
    # Threshold 0.85: includes only time 3 (quality 0.9)
    composite_t85, quality_t85 = calculate_quality_weighted_composite(
        data_stack,
        quality_scores,
        method='threshold',
        threshold=0.85
    )
    
    assert composite_t85[2, 2] == 40  # Only one value passes threshold
    assert quality_t85[2, 2] == 0.9
    
    # Test mathematical properties
    
    # 1. Weighted mean should be between min and max
    all_values = data_stack.reshape(-1)
    valid_values = all_values[~np.isnan(all_values)]
    assert np.all(composite_wmean >= np.min(valid_values) - 0.01)
    assert np.all(composite_wmean <= np.max(valid_values) + 0.01)
    
    # 2. Best quality method should have highest mean quality
    mean_quality_wmean = np.nanmean(quality_wmean)
    mean_quality_best = np.nanmean(quality_best)
    mean_quality_t6 = np.nanmean(quality_t6)
    
    assert mean_quality_best >= mean_quality_wmean
    assert mean_quality_best >= mean_quality_t6
    
    # 3. Higher threshold should result in higher quality
    composite_t3, quality_t3 = calculate_quality_weighted_composite(
        data_stack,
        quality_scores,
        method='threshold',
        threshold=0.3
    )
    
    assert np.nanmean(quality_t6) > np.nanmean(quality_t3)
    
    # Test robustness to outliers
    # Create data with extreme outlier
    outlier_data = data_stack.copy()
    outlier_data[2, 3, 3] = 1000  # Extreme outlier
    quality_scores[2, 3, 3] = 0.1  # Very low quality
    
    composite_outlier, _ = calculate_quality_weighted_composite(
        outlier_data,
        quality_scores,
        method='weighted_mean'
    )
    
    # Weighted mean should not be heavily influenced by low-quality outlier
    assert composite_outlier[3, 3] < 100  # Much less than outlier value
    
    # Test edge case: all equal quality
    equal_quality = np.ones((n_times, height, width)) * 0.5
    
    composite_equal, quality_equal = calculate_quality_weighted_composite(
        data_stack,
        equal_quality,
        method='weighted_mean'
    )
    
    # Should be equivalent to simple mean when all qualities are equal
    expected_mean = np.nanmean(data_stack, axis=0)
    assert np.allclose(composite_equal, expected_mean, rtol=1e-5)
    
    # Test no observations pass threshold
    composite_no_pass, quality_no_pass = calculate_quality_weighted_composite(
        data_stack,
        quality_scores * 0.1,  # All qualities < 0.1
        method='threshold',
        threshold=0.5
    )
    
    # Should be all NaN
    assert np.all(np.isnan(composite_no_pass))
    assert np.all(np.isnan(quality_no_pass))


def test_calculate_quality_weighted_composite_threshold():
    """Test quality thresholding in weighted composites.
    
    This test should verify that:
    - Pixels below quality threshold are excluded
    - Threshold parameter controls inclusion criteria
    - Adaptive thresholding works with local quality
    - Edge cases (all pixels below threshold) are handled
    """
    n_times, height, width = 6, 5, 5
    
    # Create data with varying quality levels
    data_stack = np.zeros((n_times, height, width))
    quality_scores = np.zeros((n_times, height, width))
    
    # Set up quality gradient across time
    for t in range(n_times):
        data_stack[t, :, :] = (t + 1) * 10  # Values: 10, 20, 30, 40, 50, 60
        quality_scores[t, :, :] = (t + 1) / (n_times + 1)  # Quality: ~0.14, 0.29, 0.43, 0.57, 0.71, 0.86
    
    # Test different threshold levels
    thresholds = [0.2, 0.4, 0.6, 0.8]
    results = {}
    
    for thresh in thresholds:
        composite, quality = calculate_quality_weighted_composite(
            data_stack,
            quality_scores,
            method='threshold',
            threshold=thresh
        )
        results[thresh] = (composite, quality)
    
    # Verify threshold filtering
    # Threshold 0.2: should include times 1-5 (qualities >= 0.29)
    assert abs(results[0.2][0][0, 0] - np.mean([20, 30, 40, 50, 60])) < 0.1
    
    # Threshold 0.4: should include times 2-5 (qualities >= 0.43)
    assert abs(results[0.4][0][0, 0] - np.mean([30, 40, 50, 60])) < 0.1
    
    # Threshold 0.6: should include times 4-5 (qualities >= 0.71)
    assert abs(results[0.6][0][0, 0] - np.mean([50, 60])) < 0.1
    
    # Threshold 0.8: should include only time 5 (quality 0.86)
    assert results[0.8][0][0, 0] == 60
    
    # Verify quality scores reflect filtered observations
    assert results[0.2][1][0, 0] < results[0.4][1][0, 0]
    assert results[0.4][1][0, 0] < results[0.6][1][0, 0]
    assert results[0.6][1][0, 0] < results[0.8][1][0, 0]
    
    # Test threshold with mixed quality across pixels
    mixed_quality = quality_scores.copy()
    # Make some pixels have consistently low quality
    mixed_quality[:, 0, 0] = 0.1  # All times low quality at (0,0)
    mixed_quality[:, 1, 1] = 0.9  # All times high quality at (1,1)
    
    composite_mixed, quality_mixed = calculate_quality_weighted_composite(
        data_stack,
        mixed_quality,
        method='threshold',
        threshold=0.5
    )
    
    # Low quality pixel should be NaN (no observations pass)
    assert np.isnan(composite_mixed[0, 0])
    assert np.isnan(quality_mixed[0, 0])
    
    # High quality pixel should have all observations
    assert abs(composite_mixed[1, 1] - np.mean(data_stack[:, 1, 1])) < 0.1
    assert quality_mixed[1, 1] == 0.9
    
    # Test threshold with quality in 0-100 range
    quality_100 = quality_scores * 100
    
    composite_100, quality_100_out = calculate_quality_weighted_composite(
        data_stack,
        quality_100,
        method='threshold',
        threshold=60  # 60 in 0-100 range = 0.6 in 0-1 range
    )
    
    # Should match threshold=0.6 result
    assert np.allclose(composite_100, results[0.6][0], rtol=1e-5)
    
    # Test default threshold
    composite_default, _ = calculate_quality_weighted_composite(
        data_stack,
        quality_scores,
        method='threshold'
        # No threshold specified
    )
    
    # Default should be 0.5
    expected_default = calculate_quality_weighted_composite(
        data_stack,
        quality_scores,
        method='threshold',
        threshold=0.5
    )[0]
    
    assert np.allclose(composite_default, expected_default, rtol=1e-5)
    
    # Test edge case: threshold too high
    composite_high_thresh, quality_high_thresh = calculate_quality_weighted_composite(
        data_stack,
        quality_scores,
        method='threshold',
        threshold=0.95  # Higher than all qualities
    )
    
    # All pixels should be NaN
    assert np.all(np.isnan(composite_high_thresh))
    assert np.all(np.isnan(quality_high_thresh))
    
    # Test edge case: threshold too low (0)
    composite_low_thresh, _ = calculate_quality_weighted_composite(
        data_stack,
        quality_scores,
        method='threshold',
        threshold=0.0  # All observations pass
    )
    
    # Should include all observations
    expected_all = np.mean(data_stack, axis=0)
    assert np.allclose(composite_low_thresh, expected_all, rtol=1e-5)
    
    # Test invalid threshold
    with pytest.raises(ValueError, match="Threshold must be in.*range"):
        calculate_quality_weighted_composite(
            data_stack,
            quality_scores,
            method='threshold',
            threshold=1.5  # Out of range
        )
    
    with pytest.raises(ValueError, match="Threshold must be in.*range"):
        calculate_quality_weighted_composite(
            data_stack,
            quality_scores,
            method='threshold',
            threshold=-0.1  # Negative
        )


def test_quality_assessment_edge_cases():
    """Test quality assessment functions with edge cases.
    
    This test should verify that:
    - Single-pixel arrays are handled correctly
    - Arrays with all invalid data don't cause errors
    - Very large arrays are processed efficiently
    - Different array shapes and data types work
    - Memory usage is reasonable for large datasets
    """
    # Test single-pixel array
    single_pixel = np.array([[42.0]])
    
    result_single = assess_data_quality(single_pixel)
    assert result_single['valid_pixel_count'] == 1
    assert result_single['mean'] == 42.0
    assert result_single['std'] == 0.0  # No variation
    
    # Test with quality metrics computation
    single_3d = single_pixel.reshape(1, 1, 1)
    with pytest.raises(ValueError, match="Need at least 2 time steps"):
        compute_quality_metrics(single_3d)  # Need temporal dimension
    
    # Test all invalid data (all NaN)
    all_nan = np.full((10, 10), np.nan)
    
    result_nan = assess_data_quality(all_nan)
    assert result_nan['valid_pixel_count'] == 0
    assert result_nan['quality_score'] < 20
    assert 'No valid pixels found' in result_nan['warnings']
    
    # Test all invalid with quality metrics
    all_nan_3d = np.full((5, 10, 10), np.nan)
    metrics_nan = compute_quality_metrics(all_nan_3d)
    assert np.all(metrics_nan['observation_count'] == 0)
    assert np.all(np.isnan(metrics_nan['temporal_std']))
    
    # Test very large array performance
    large_2d = np.random.randn(5000, 5000)
    
    import time
    start_time = time.time()
    result_large = assess_data_quality(large_2d)
    elapsed_2d = time.time() - start_time
    
    assert elapsed_2d < 10.0  # Should complete within 10 seconds
    assert result_large['valid_pixel_count'] == 25_000_000
    
    # Test different shapes
    shapes_to_test = [
        (1, 100),    # Single row
        (100, 1),    # Single column
        (1000, 10),  # Tall and narrow
        (10, 1000),  # Short and wide
    ]
    
    for shape in shapes_to_test:
        data = np.random.randn(*shape)
        result = assess_data_quality(data)
        assert result['valid_pixel_count'] == shape[0] * shape[1]
    
    # Test different data types
    # Integer data
    int_data = np.random.randint(0, 100, size=(50, 50), dtype=np.int32)
    result_int = assess_data_quality(int_data)
    assert result_int['valid_pixel_count'] == 2500
    assert isinstance(result_int['mean'], float)  # Converted to float
    
    # Float32 data
    float32_data = np.random.randn(50, 50).astype(np.float32)
    result_float32 = assess_data_quality(float32_data)
    assert result_float32['valid_pixel_count'] == 2500
    
    # Test with extreme values
    extreme_data = np.array([[1e-10, 1e10],
                             [0.0, np.inf]])
    
    result_extreme = assess_data_quality(extreme_data)
    assert not np.isnan(result_extreme['mean'])  # Should handle inf
    
    # Test memory usage with quality metrics on large 3D array
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    large_3d = np.random.randn(10, 1000, 1000)
    metrics_large = compute_quality_metrics(large_3d)
    
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = memory_after - memory_before
    
    # Should not use excessive memory (less than 1GB increase)
    assert memory_increase < 1000
    
    # Test with constant data
    constant_data = np.ones((100, 100)) * 42
    
    result_constant = assess_data_quality(constant_data)
    assert result_constant['std'] == 0.0
    assert 'No data variation' in result_constant['warnings']
    
    # Test composite mask with single mask
    single_mask = {'only_mask': np.ones((10, 10), dtype=bool)}
    composite, stats = create_composite_quality_mask(single_mask)
    assert np.all(composite)
    assert stats['composite_coverage'] == 100.0
    
    # Test suspicious values with empty data
    empty_data = np.array([])
    with pytest.raises(ValueError):
        flag_suspicious_values(empty_data.reshape(0, 0), index_type='ndvi')
    
    # Test confidence scores with all zero quality
    zero_quality_metrics = {
        'composite_quality': np.zeros((10, 10)),
        'data_availability': np.zeros((10, 10))
    }
    
    confidence_zero = compute_confidence_scores(
        np.ones((10, 10)),
        zero_quality_metrics
    )
    
    assert np.all(confidence_zero >= 0.05)  # Floor applied
    assert np.all(confidence_zero < 0.2)    # Very low confidence
    
    # Test artifact detection with tiny array
    tiny_data = np.array([[1, 2], [3, 4]])
    
    with pytest.raises(ValueError, match="Window size must be odd integer >= 3"):
        identify_processing_artifacts(tiny_data, window_size=5)  # Window too large
    
    artifacts_tiny = identify_processing_artifacts(tiny_data, window_size=3)
    assert all(mask.shape == (2, 2) for mask in artifacts_tiny.values())
    
    # Test quality report with single assessment
    single_assessment = [{
        'valid_pixel_percent': 100.0,
        'quality_score': 95.0,
        'warnings': [],
        'recommendations': []
    }]
    
    report_single = generate_quality_report(single_assessment)
    assert 'Excellent' in report_single


def test_quality_metrics_validation():
    """Test validation of quality metrics results.
    
    This test should verify that:
    - Quality metrics are within expected ranges
    - Metrics are statistically meaningful
    - Correlation between different metrics makes sense
    - Extreme cases produce reasonable metric values
    - Metrics are consistent across different data types
    """
    # Create test data with known properties
    n_times, height, width = 10, 20, 20
    
    # 1. Perfect data (no variation, all valid)
    perfect_data = np.ones((n_times, height, width)) * 50
    metrics_perfect = compute_quality_metrics(perfect_data)
    
    # Verify ranges
    assert np.all(metrics_perfect['observation_count'] == n_times)
    assert np.all(metrics_perfect['data_availability'] == 1.0)
    assert np.all(metrics_perfect['temporal_std'] == 0.0)
    assert np.all(metrics_perfect['temporal_cv'] == np.inf)  # Zero mean edge case
    assert np.all(metrics_perfect['composite_quality'] >= 80)  # High quality
    assert np.all(metrics_perfect['confidence'] >= 0.8)  # High confidence
    
    # 2. Noisy data
    noisy_data = np.random.normal(50, 10, (n_times, height, width))
    metrics_noisy = compute_quality_metrics(noisy_data)
    
    # Temporal std should reflect noise level
    assert 5 < np.mean(metrics_noisy['temporal_std']) < 15
    # CV should be reasonable
    assert 0.1 < np.mean(metrics_noisy['temporal_cv']) < 0.5
    # Quality should be moderate
    assert 40 < np.mean(metrics_noisy['composite_quality']) < 80
    
    # 3. Data with missing observations
    missing_data = noisy_data.copy()
    # Add systematic missing data
    missing_data[:5, :10, :10] = np.nan  # Half time, quarter space
    metrics_missing = compute_quality_metrics(missing_data)
    
    # Verify observation counts
    assert np.all(metrics_missing['observation_count'][:10, :10] == 5)
    assert np.all(metrics_missing['observation_count'][10:, :] == 10)
    # Data availability should reflect missing pattern
    assert np.all(metrics_missing['data_availability'][:10, :10] == 0.5)
    
    # 4. Test metric correlations
    # High observation count should correlate with high confidence
    obs_flat = metrics_missing['observation_count'].flatten()
    conf_flat = metrics_missing['confidence'].flatten()
    correlation = np.corrcoef(obs_flat, conf_flat)[0, 1]
    assert correlation > 0.5  # Positive correlation
    
    # High temporal std should correlate with low quality
    std_flat = metrics_noisy['temporal_std'].flatten()
    quality_flat = metrics_noisy['composite_quality'].flatten()
    # Remove NaN values
    valid_idx = ~(np.isnan(std_flat) | np.isnan(quality_flat))
    correlation_std_quality = np.corrcoef(
        std_flat[valid_idx], 
        quality_flat[valid_idx]
    )[0, 1]
    assert correlation_std_quality < -0.3  # Negative correlation
    
    # 5. Test extreme temporal patterns
    # Linear trend
    trend_data = np.zeros((n_times, height, width))
    for t in range(n_times):
        trend_data[t] = 10 + t * 5  # Strong linear trend
    
    metrics_trend = compute_quality_metrics(trend_data)
    # Should have high temporal range
    assert np.all(metrics_trend['temporal_range'] >= 45)  # 5*(10-1)
    # But moderate CV (relative variation)
    assert np.mean(metrics_trend['temporal_cv']) < 1.0
    
    # 6. Test spatial patterns
    # Create data with sharp edge
    edge_data = np.ones((5, height, width)) * 50
    edge_data[:, :, width//2:] = 100  # Right half different
    
    metrics_edge = compute_quality_metrics(edge_data)
    # Should detect high gradient at boundary
    boundary_gradient = metrics_edge['spatial_gradient'][:, width//2-2:width//2+2]
    interior_gradient = metrics_edge['spatial_gradient'][:, :5]
    assert np.mean(boundary_gradient) > 2 * np.mean(interior_gradient)
    
    # 7. Test different data types consistency
    # Float64
    data_f64 = np.random.randn(5, 10, 10).astype(np.float64)
    metrics_f64 = compute_quality_metrics(data_f64)
    
    # Float32
    data_f32 = data_f64.astype(np.float32)
    metrics_f32 = compute_quality_metrics(data_f32)
    
    # Results should be very similar
    assert np.allclose(
        metrics_f64['temporal_std'], 
        metrics_f32['temporal_std'], 
        rtol=1e-5
    )
    
    # 8. Test metric bounds
    # All metrics should be in valid ranges
    all_data = [perfect_data, noisy_data, missing_data, trend_data, edge_data]
    
    for data in all_data:
        metrics = compute_quality_metrics(data)
        
        # Observation count: [0, n_times]
        assert np.all(metrics['observation_count'] >= 0)
        assert np.all(metrics['observation_count'] <= n_times)
        
        # Data availability: [0, 1]
        assert np.all(metrics['data_availability'] >= 0)
        assert np.all(metrics['data_availability'] <= 1)
        
        # Composite quality: [0, 100]
        assert np.all(metrics['composite_quality'] >= 0)
        assert np.all(metrics['composite_quality'] <= 100)
        
        # Confidence: [0, 1]
        assert np.all(metrics['confidence'] >= 0)
        assert np.all(metrics['confidence'] <= 1)
        
        # Temporal metrics should be non-negative (except CV can be inf)
        valid_std = metrics['temporal_std'][~np.isnan(metrics['temporal_std'])]
        assert np.all(valid_std >= 0)
        
        valid_range = metrics['temporal_range'][~np.isnan(metrics['temporal_range'])]
        assert np.all(valid_range >= 0)
    
    # 9. Test meaningfulness of composite quality
    # Create deliberately bad data
    bad_data = np.random.randn(3, 10, 10) * 100  # Few obs, high variance
    bad_data[0, :5, :5] = np.nan  # Missing data
    
    metrics_bad = compute_quality_metrics(bad_data)
    
    # Should have low quality in problem areas
    assert np.mean(metrics_bad['composite_quality'][:5, :5]) < 50
    # Should have higher quality in complete areas  
    assert np.mean(metrics_bad['composite_quality'][5:, 5:]) > np.mean(metrics_bad['composite_quality'][:5, :5])


def test_quality_assessment_integration():
    """Test integration between different quality assessment functions.
    
    This test should verify that:
    - Quality assessments feed into composite creation
    - Confidence scores align with quality metrics
    - Artifact detection complements quality assessment
    - Report generation integrates all assessment results
    - Workflow efficiency with chained operations
    """
    # Create realistic test scenario
    n_times, height, width = 8, 30, 30
    
    # Generate multi-temporal data with various quality issues
    data_stack = np.zeros((n_times, height, width))
    
    for t in range(n_times):
        # Base data
        data_stack[t] = np.random.normal(50, 5, (height, width))
        
        # Add time-specific issues
        if t == 0:
            # Add striping artifact
            data_stack[t, 10, :] = 80
            data_stack[t, 20, :] = 20
        elif t == 2:
            # Add missing data
            data_stack[t, :10, :10] = np.nan
        elif t == 4:
            # Add noise
            data_stack[t] += np.random.normal(0, 15, (height, width))
    
    # Step 1: Compute quality metrics
    quality_metrics = compute_quality_metrics(
        data_stack,
        compute_temporal_stability=True,
        compute_spatial_coherence=True
    )
    
    # Step 2: Assess individual time slices
    assessments = []
    for t in range(n_times):
        assessment = assess_data_quality(
            data_stack[t],
            valid_range=(0, 100),
            check_outliers=True
        )
        assessments.append(assessment)
    
    # Step 3: Detect artifacts
    artifacts_by_time = []
    for t in range(n_times):
        artifacts = identify_processing_artifacts(
            data_stack[t],
            window_size=5
        )
        artifacts_by_time.append(artifacts)
    
    # Step 4: Create quality-based composite
    # Use composite quality as weights
    quality_weights = np.repeat(
        quality_metrics['composite_quality'][np.newaxis, :, :],
        n_times,
        axis=0
    )
    
    # Add time-specific adjustments based on artifacts
    for t in range(n_times):
        # Reduce quality where artifacts detected
        for artifact_type, artifact_mask in artifacts_by_time[t].items():
            quality_weights[t][artifact_mask] *= 0.5
    
    composite, composite_quality = calculate_quality_weighted_composite(
        data_stack,
        quality_weights,
        method='weighted_mean'
    )
    
    # Step 5: Compute confidence scores
    confidence = compute_confidence_scores(
        composite,
        quality_metrics,
        observation_count=quality_metrics['observation_count']
    )
    
    # Step 6: Generate quality report
    quality_report = generate_quality_report(
        assessments,
        output_format='json'
    )
    
    # Verify integration results
    
    # 1. Quality metrics should reflect known issues
    # Time 0 with striping should have lower quality
    assert assessments[0]['quality_score'] < assessments[1]['quality_score']
    # Time 2 with missing data should have lower coverage
    assert assessments[2]['valid_pixel_percent'] < 100
    # Time 4 with extra noise should have higher variation
    # Note: Random noise doesn't always create statistical outliers
    assert assessments[4]['std'] > assessments[1]['std']
    
    # 2. Artifacts should be detected
    assert np.any(artifacts_by_time[0]['striping'])  # Striping in time 0
    
    # 3. Composite should de-emphasize problematic areas
    # Areas with consistent data should have higher confidence
    stable_area_confidence = np.mean(confidence[15:25, 15:25])
    problem_area_confidence = np.mean(confidence[:10, :10])  # Where we had missing data
    assert stable_area_confidence > problem_area_confidence
    
    # 4. Quality report should summarize issues
    assert quality_report['overall_statistics']['total_assessments'] == n_times
    assert quality_report['warning_summary']['total_warnings'] > 0
    
    # 5. Test workflow efficiency
    import time
    
    # Time the full workflow
    start_time = time.time()
    
    # Repeat workflow
    metrics = compute_quality_metrics(data_stack)
    assessments = [assess_data_quality(data_stack[t]) for t in range(n_times)]
    artifacts = [identify_processing_artifacts(data_stack[t]) for t in range(n_times)]
    composite, _ = calculate_quality_weighted_composite(
        data_stack, 
        quality_metrics['confidence'][:, np.newaxis, np.newaxis].repeat(height, axis=1).repeat(width, axis=2)
    )
    confidence = compute_confidence_scores(composite, metrics)
    report = generate_quality_report(assessments)
    
    elapsed_time = time.time() - start_time
    
    # Should complete reasonably quickly
    assert elapsed_time < 10.0  # Full workflow in under 10 seconds
    
    # 6. Test consistency between components
    # Areas with low temporal stability should have lower confidence
    high_temporal_std = quality_metrics['temporal_std'] > np.nanpercentile(quality_metrics['temporal_std'], 75)
    mean_conf_unstable = np.mean(confidence[high_temporal_std])
    mean_conf_stable = np.mean(confidence[~high_temporal_std & ~np.isnan(quality_metrics['temporal_std'])])
    assert mean_conf_stable > mean_conf_unstable
    
    # 7. Test with quality-based masking
    # Create masks from quality assessment
    quality_masks = {}
    
    # Mask based on confidence
    quality_masks['low_confidence'] = confidence < 0.5
    
    # Mask based on artifacts
    artifact_mask = np.zeros((height, width), dtype=bool)
    for t in range(n_times):
        for artifact_type, mask in artifacts_by_time[t].items():
            artifact_mask |= mask
    quality_masks['artifacts'] = artifact_mask
    
    # Mask based on quality metrics
    quality_masks['high_cv'] = quality_metrics['temporal_cv'] > 1.0
    
    # Combine masks
    combined_mask, mask_stats = create_composite_quality_mask(
        quality_masks,
        combination_method='or',
        dilation_radius=1
    )
    
    # Final masked composite
    final_composite = np.where(combined_mask, composite, np.nan)
    
    # Verify final product quality
    valid_pixels = ~np.isnan(final_composite)
    assert np.sum(valid_pixels) > 0  # Some valid data remains
    assert np.sum(valid_pixels) < height * width  # Some masking occurred
    
    # 8. Test error propagation
    # Introduce an error in one component
    bad_metrics = quality_metrics.copy()
    bad_metrics['composite_quality'] = np.full((height, width), -10)  # Invalid
    
    # Should still produce valid confidence (clamped to valid range)
    confidence_bad = compute_confidence_scores(composite, bad_metrics)
    assert np.all(confidence_bad >= 0)
    assert np.all(confidence_bad <= 1)