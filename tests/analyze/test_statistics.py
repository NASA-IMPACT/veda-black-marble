"""Test functions in blackmarble.analyze.statistics module.

This module tests statistical analysis and metrics computation functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from datetime import datetime
from scipy import stats as scipy_stats

# Import the module under test
from blackmarble.analyze.statistics import (
    calculate_coverage_statistics,
    compute_zonal_statistics,
    generate_histogram_analysis,
    calculate_temporal_statistics,
    compute_comparison_metrics,
    analyze_spatial_autocorrelation,
    generate_quality_metrics_summary,
    calculate_uncertainty_metrics,
)


def test_calculate_coverage_statistics():
    """Test comprehensive coverage statistics calculation.
    
    This test should verify that:
    - Data coverage percentages are calculated correctly
    - Mask-based coverage statistics are accurate
    - Regional coverage analysis works with region labels
    - Valid pixel counts and ratios are computed correctly
    - No-data handling doesn't bias coverage calculations
    - Different data types produce consistent results
    - Performance with large datasets is acceptable
    """
    # Test basic coverage calculation
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    result = calculate_coverage_statistics(data)
    
    # Verify data coverage percentages are calculated correctly
    assert result['total_pixels'] == 9
    assert result['valid_pixels'] == 9
    assert result['valid_percent'] == 100.0
    assert result['nodata_pixels'] == 0
    assert result['nodata_percent'] == 0.0
    
    # Test with mask-based coverage
    mask = np.array([[True, True, False], [True, False, True], [False, True, True]])
    result_masked = calculate_coverage_statistics(data, mask=mask)
    
    # Verify mask-based coverage statistics are accurate
    assert result_masked['valid_pixels'] == 6
    assert result_masked['valid_percent'] == pytest.approx(66.67, rel=0.01)
    assert np.array_equal(result_masked['coverage_map'], mask.astype(np.uint8))
    
    # Test regional coverage analysis
    region_labels = np.array([[1, 1, 2], [1, 1, 2], [3, 3, 2]])
    result_regional = calculate_coverage_statistics(data, mask=mask, region_labels=region_labels)
    
    # Verify regional coverage analysis works with region labels
    assert 'regional_stats' in result_regional
    assert len(result_regional['regional_stats']) == 3
    
    # Check region 1 statistics (4 pixels total, 3 valid)
    assert result_regional['regional_stats'][1]['total_pixels'] == 4
    assert result_regional['regional_stats'][1]['valid_pixels'] == 3
    assert result_regional['regional_stats'][1]['valid_percent'] == 75.0
    
    # Check region 2 statistics (3 pixels total, 2 valid)
    assert result_regional['regional_stats'][2]['total_pixels'] == 3
    assert result_regional['regional_stats'][2]['valid_pixels'] == 2
    assert result_regional['regional_stats'][2]['valid_percent'] == pytest.approx(66.67, rel=0.01)
    
    # Check region 3 statistics (2 pixels total, 1 valid)
    assert result_regional['regional_stats'][3]['total_pixels'] == 2
    assert result_regional['regional_stats'][3]['valid_pixels'] == 1
    assert result_regional['regional_stats'][3]['valid_percent'] == 50.0
    
    # Test no-data handling
    data_with_nan = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]], dtype=np.float32)
    result_nan = calculate_coverage_statistics(data_with_nan)
    
    # Verify no-data handling doesn't bias coverage calculations
    assert result_nan['valid_pixels'] == 7
    assert result_nan['nodata_pixels'] == 2
    assert result_nan['valid_percent'] == pytest.approx(77.78, rel=0.01)
    
    # Test different data types
    data_int = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    result_int = calculate_coverage_statistics(data_int)
    
    # Verify different data types produce consistent results
    assert result_int['valid_pixels'] == 9
    assert result_int['valid_percent'] == 100.0
    
    # Test quality flag assignment
    assert result['quality_flag'] == 'excellent'  # 100% coverage
    assert result_masked['quality_flag'] == 'good'  # ~67% coverage
    
    # Test large dataset performance (simplified)
    large_data = np.random.rand(1000, 1000)
    start_time = datetime.now()
    result_large = calculate_coverage_statistics(large_data, compute_regional=False)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Verify performance with large datasets is acceptable (< 1 second)
    assert elapsed < 1.0
    assert result_large['total_pixels'] == 1_000_000


def test_calculate_coverage_statistics_regional():
    """Test regional coverage statistics calculation.
    
    This test should verify that:
    - Region labels are properly processed
    - Coverage statistics are computed for each region
    - Regional summaries are accurate and complete
    - Overlapping regions are handled correctly
    - Edge effects don't bias regional statistics
    """
    # Create test data with known regional patterns
    data = np.ones((10, 10), dtype=np.float32)
    
    # Create region labels with various configurations
    region_labels = np.zeros((10, 10), dtype=int)
    region_labels[:5, :5] = 1  # Region 1: 5x5 = 25 pixels
    region_labels[:5, 5:] = 2   # Region 2: 5x5 = 25 pixels
    region_labels[5:, :5] = 3   # Region 3: 5x5 = 25 pixels
    region_labels[5:, 5:] = 4   # Region 4: 5x5 = 25 pixels
    
    # Add mask with different coverage per region
    mask = np.ones((10, 10), dtype=bool)
    mask[:3, :3] = False  # 9 pixels invalid in region 1
    mask[7:, 7:] = False  # 9 pixels invalid in region 4
    
    result = calculate_coverage_statistics(data, mask=mask, region_labels=region_labels)
    
    # Verify region labels are properly processed
    assert 'regional_stats' in result
    assert len(result['regional_stats']) == 4
    
    # Verify coverage statistics are computed for each region
    assert result['regional_stats'][1]['total_pixels'] == 25
    assert result['regional_stats'][1]['valid_pixels'] == 16  # 25 - 9
    assert result['regional_stats'][1]['valid_percent'] == 64.0
    
    assert result['regional_stats'][2]['total_pixels'] == 25
    assert result['regional_stats'][2]['valid_pixels'] == 25
    assert result['regional_stats'][2]['valid_percent'] == 100.0
    
    assert result['regional_stats'][3]['total_pixels'] == 25
    assert result['regional_stats'][3]['valid_pixels'] == 25
    assert result['regional_stats'][3]['valid_percent'] == 100.0
    
    assert result['regional_stats'][4]['total_pixels'] == 25
    assert result['regional_stats'][4]['valid_pixels'] == 16  # 25 - 9
    assert result['regional_stats'][4]['valid_percent'] == 64.0
    
    # Test with background pixels (label 0)
    region_labels_with_bg = region_labels.copy()
    region_labels_with_bg[4:6, 4:6] = 0  # 4 pixels as background
    
    result_bg = calculate_coverage_statistics(data, region_labels=region_labels_with_bg)
    
    # Verify background (0) is excluded from regional stats
    assert 0 not in result_bg['regional_stats']
    
    # Test edge effects with partial regions
    edge_labels = np.zeros((5, 5), dtype=int)
    edge_labels[0, :] = 1    # Top edge
    edge_labels[-1, :] = 2   # Bottom edge
    edge_labels[:, 0] = 3    # Left edge
    edge_labels[:, -1] = 4   # Right edge
    edge_labels[2, 2] = 5    # Center pixel
    
    result_edge = calculate_coverage_statistics(np.ones((5, 5)), region_labels=edge_labels)
    
    # Verify edge effects don't bias regional statistics
    assert result_edge['regional_stats'][1]['total_pixels'] == 5
    assert result_edge['regional_stats'][2]['total_pixels'] == 5
    assert result_edge['regional_stats'][3]['total_pixels'] == 5
    assert result_edge['regional_stats'][4]['total_pixels'] == 5
    assert result_edge['regional_stats'][5]['total_pixels'] == 1


def test_calculate_coverage_statistics_masks():
    """Test coverage statistics with different mask types.
    
    This test should verify that:
    - Boolean masks work correctly
    - Integer masks are properly interpreted
    - Multiple masks can be processed simultaneously
    - Mask combinations produce expected coverage results
    - Invalid mask values are handled gracefully
    """
    data = np.arange(25).reshape(5, 5).astype(np.float32)
    
    # Test boolean masks work correctly
    bool_mask = np.array([
        [True, True, False, True, False],
        [True, False, True, False, True],
        [False, True, True, True, False],
        [True, False, True, False, True],
        [False, True, False, True, True]
    ])
    
    result_bool = calculate_coverage_statistics(data, mask=bool_mask)
    assert result_bool['valid_pixels'] == np.sum(bool_mask)
    assert result_bool['valid_percent'] == np.sum(bool_mask) / 25 * 100
    
    # Test integer masks are properly interpreted (converted to boolean)
    int_mask = np.array([
        [1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 1, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 1]
    ], dtype=np.int32)
    
    result_int = calculate_coverage_statistics(data, mask=int_mask)
    assert result_int['valid_pixels'] == np.sum(int_mask > 0)
    assert result_int['valid_percent'] == result_bool['valid_percent']  # Should match boolean mask
    
    # Test mask combinations (intersection of multiple conditions)
    quality_mask = data < 20  # Quality threshold
    spatial_mask = bool_mask   # Spatial validity
    combined_mask = quality_mask & spatial_mask
    
    result_combined = calculate_coverage_statistics(data, mask=combined_mask)
    expected_valid = np.sum(combined_mask)
    assert result_combined['valid_pixels'] == expected_valid
    
    # Test with float mask values (should be converted to boolean)
    float_mask = np.random.rand(5, 5)
    float_mask_bool = float_mask > 0.5
    
    result_float = calculate_coverage_statistics(data, mask=float_mask)
    # Should interpret any non-zero as True
    assert result_float['valid_pixels'] == np.sum(float_mask.astype(bool))
    
    # Test invalid mask shapes
    with pytest.raises(ValueError, match="Mask shape.*doesn't match data shape"):
        calculate_coverage_statistics(data, mask=np.ones((3, 3)))
    
    # Test mask with NaN values
    nan_data = data.astype(float)
    nan_data[1, 1] = np.nan
    nan_data[3, 3] = np.nan
    
    result_nan = calculate_coverage_statistics(nan_data, mask=bool_mask)
    # NaN values should be excluded even if mask says valid
    assert result_nan['valid_pixels'] == np.sum(bool_mask) - 2
    assert result_nan['nodata_pixels'] == 2


def test_compute_zonal_statistics():
    """Test zonal statistics computation for labeled regions.
    
    This test should verify that:
    - Statistics are computed correctly for each zone
    - Default statistics (mean, std, min, max, count) are accurate
    - Custom statistics lists work as expected
    - Percentiles are calculated correctly for each zone
    - Weighted statistics use weights appropriately
    - Zone boundaries are handled correctly
    - Mathematical accuracy of all statistical measures
    """
    # Create test data with known values per zone
    data = np.array([
        [1, 2, 3, 10, 11],
        [4, 5, 6, 12, 13],
        [7, 8, 9, 14, 15],
        [20, 21, 30, 31, 32],
        [22, 23, 33, 34, 35]
    ], dtype=np.float32)
    
    zones = np.array([
        [1, 1, 1, 2, 2],
        [1, 1, 1, 2, 2],
        [1, 1, 1, 2, 2],
        [3, 3, 4, 4, 4],
        [3, 3, 4, 4, 4]
    ])
    
    # Test default statistics
    result = compute_zonal_statistics(data, zones)
    
    # Verify statistics are computed correctly for each zone
    assert len(result) == 4  # 4 zones (1, 2, 3, 4)
    
    # Zone 1: values 1-9
    zone1_values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert result[1]['mean'] == np.mean(zone1_values)
    assert result[1]['std'] == np.std(zone1_values)
    assert result[1]['min'] == 1
    assert result[1]['max'] == 9
    assert result[1]['sum'] == sum(zone1_values)
    assert result[1]['count'] == 9
    assert result[1]['median'] == 5
    
    # Zone 2: values 10-15
    zone2_values = [10, 11, 12, 13, 14, 15]
    assert result[2]['mean'] == np.mean(zone2_values)
    assert result[2]['std'] == np.std(zone2_values)
    assert result[2]['min'] == 10
    assert result[2]['max'] == 15
    assert result[2]['count'] == 6
    
    # Test custom statistics list
    custom_stats = ['mean', 'sum', 'count']
    result_custom = compute_zonal_statistics(data, zones, stats_list=custom_stats)
    
    # Verify custom statistics lists work as expected
    assert set(result_custom[1].keys()) == set(custom_stats)
    assert 'std' not in result_custom[1]
    assert 'median' not in result_custom[1]
    
    # Test percentiles
    percentiles = [25, 50, 75, 90]
    result_perc = compute_zonal_statistics(data, zones, percentiles=percentiles)
    
    # Verify percentiles are calculated correctly for each zone
    assert result_perc[1]['p25'] == np.percentile(zone1_values, 25)
    assert result_perc[1]['p50'] == np.percentile(zone1_values, 50)
    assert result_perc[1]['p75'] == np.percentile(zone1_values, 75)
    assert result_perc[1]['p90'] == np.percentile(zone1_values, 90)
    
    # Test zone boundaries
    # Zone 0 should be excluded
    zones_with_background = zones.copy()
    zones_with_background[0, 0] = 0
    result_bg = compute_zonal_statistics(data, zones_with_background)
    assert 0 not in result_bg
    
    # Test with NaN values
    data_nan = data.copy()
    data_nan[0, 0] = np.nan  # In zone 1
    data_nan[0, 3] = np.nan  # In zone 2
    
    result_nan = compute_zonal_statistics(data_nan, zones)
    
    # Verify NaN handling
    assert result_nan[1]['count'] == 8  # 9 - 1 NaN
    assert result_nan[2]['count'] == 5  # 6 - 1 NaN
    
    # Verify mathematical accuracy
    assert np.isclose(result_nan[1]['mean'], np.nanmean(data_nan[zones == 1]))


def test_compute_zonal_statistics_weighted():
    """Test weighted zonal statistics computation.
    
    This test should verify that:
    - Weight arrays are properly applied to statistics
    - Weighted means and variances are mathematically accurate
    - Zero weights effectively exclude pixels
    - Weight normalization works correctly
    - Edge cases (all zero weights) are handled
    """
    # Create simple test data
    data = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.float32)
    
    zones = np.array([
        [1, 1, 2],
        [1, 1, 2],
        [3, 3, 2]
    ])
    
    # Create weights that emphasize certain pixels
    weights = np.array([
        [1.0, 2.0, 1.0],
        [2.0, 3.0, 0.5],
        [1.0, 1.0, 2.0]
    ])
    
    result = compute_zonal_statistics(data, zones, weighted=True, weights=weights)
    
    # Verify weight arrays are properly applied to statistics
    # Zone 1: values [1,2,4,5] with weights [1,2,2,3]
    zone1_values = np.array([1, 2, 4, 5])
    zone1_weights = np.array([1, 2, 2, 3])
    expected_mean = np.average(zone1_values, weights=zone1_weights)
    
    assert np.isclose(result[1]['mean'], expected_mean)
    assert np.isclose(result[1]['sum'], np.sum(zone1_values * zone1_weights))
    
    # Test zero weights effectively exclude pixels
    weights_zero = weights.copy()
    weights_zero[0, 0] = 0  # Exclude value 1 from zone 1
    
    result_zero = compute_zonal_statistics(data, zones, weighted=True, weights=weights_zero)
    
    # Zone 1 without first pixel
    zone1_values_exc = np.array([2, 4, 5])
    zone1_weights_exc = np.array([2, 2, 3])
    expected_mean_exc = np.average(zone1_values_exc, weights=zone1_weights_exc)
    
    assert np.isclose(result_zero[1]['mean'], expected_mean_exc)
    
    # Test weight normalization
    # Scaling weights shouldn't change weighted mean
    weights_scaled = weights * 10
    result_scaled = compute_zonal_statistics(data, zones, weighted=True, weights=weights_scaled)
    
    assert np.isclose(result_scaled[1]['mean'], result[1]['mean'])
    
    # Test edge case with all zero weights for a zone
    weights_all_zero = np.ones_like(weights)
    weights_all_zero[zones == 3] = 0  # All weights zero for zone 3
    
    result_all_zero = compute_zonal_statistics(data, zones, weighted=True, weights=weights_all_zero)
    
    # Zone 3 should have valid unweighted statistics when all weights are zero
    assert 3 in result_all_zero
    assert result_all_zero[3]['count'] == 2  # Still has 2 pixels
    
    # Test weight shape mismatch
    with pytest.raises(ValueError, match="Weights shape.*doesn't match data shape"):
        compute_zonal_statistics(data, zones, weighted=True, weights=np.ones((2, 2)))


def test_compute_zonal_statistics_percentiles():
    """Test percentile calculation in zonal statistics.
    
    This test should verify that:
    - Standard percentiles (25th, 50th, 75th) are accurate
    - Custom percentile lists work correctly
    - Percentiles are calculated per zone
    - Edge cases (single pixel zones) are handled
    - Mathematical accuracy with reference implementations
    """
    # Create data with known distributions
    data = np.arange(100).reshape(10, 10).astype(np.float32)
    
    # Create zones with different sizes
    zones = np.zeros((10, 10), dtype=int)
    zones[:5, :5] = 1    # 25 pixels
    zones[:5, 5:] = 2    # 25 pixels
    zones[5:, :5] = 3    # 25 pixels
    zones[5:, 5:] = 4    # 25 pixels
    zones[9, 9] = 5      # Single pixel zone
    
    # Test standard percentiles
    standard_percentiles = [25, 50, 75]
    result = compute_zonal_statistics(data, zones, percentiles=standard_percentiles)
    
    # Verify standard percentiles are accurate for zone 1
    zone1_values = data[:5, :5].ravel()
    assert result[1]['p25'] == np.percentile(zone1_values, 25)
    assert result[1]['p50'] == np.percentile(zone1_values, 50)
    assert result[1]['p75'] == np.percentile(zone1_values, 75)
    
    # Test custom percentile lists
    custom_percentiles = [10, 20, 30, 40, 60, 70, 80, 90]
    result_custom = compute_zonal_statistics(data, zones, percentiles=custom_percentiles)
    
    # Verify custom percentile lists work correctly
    for p in custom_percentiles:
        assert f'p{int(p)}' in result_custom[1]
        assert result_custom[1][f'p{int(p)}'] == np.percentile(zone1_values, p)
    
    # Verify percentiles are calculated per zone
    zone2_values = data[:5, 5:].ravel()
    zone3_values = data[5:, :5].ravel()
    zone4_values = data[5:, 5:9].ravel()  # Exclude the single pixel zone
    
    assert result[2]['p50'] == np.percentile(zone2_values, 50)
    assert result[3]['p50'] == np.percentile(zone3_values, 50)
    assert result[4]['p50'] == np.percentile(zone4_values, 50)
    
    # Test edge case: single pixel zone
    # Zone 5 has only one pixel (value 99)
    assert result[5]['p25'] == 99
    assert result[5]['p50'] == 99
    assert result[5]['p75'] == 99
    assert result[5]['min'] == 99
    assert result[5]['max'] == 99
    
    # Test mathematical accuracy with scipy
    from scipy import stats as scipy_stats
    
    # Compare with scipy's percentile implementation
    for zone_id in [1, 2, 3, 4]:
        zone_data = data[zones == zone_id]
        for p in standard_percentiles:
            numpy_result = result[zone_id][f'p{int(p)}']
            scipy_result = float(scipy_stats.scoreatpercentile(zone_data, p))
            assert np.isclose(numpy_result, scipy_result, rtol=1e-7)
    
    # Test extreme percentiles
    extreme_percentiles = [0, 1, 99, 100]
    result_extreme = compute_zonal_statistics(data, zones, percentiles=extreme_percentiles)
    
    assert result_extreme[1]['p0'] == np.min(zone1_values)
    assert result_extreme[1]['p100'] == np.max(zone1_values)


def test_generate_histogram_analysis():
    """Test comprehensive histogram analysis of data distribution.
    
    This test should verify that:
    - Histogram bins are calculated correctly
    - Custom bin counts and ranges work as expected
    - Mask parameter excludes invalid pixels
    - Distribution metrics (skewness, kurtosis) are accurate
    - Histogram statistics are mathematically correct
    - Memory efficiency with large datasets
    - Edge cases (uniform data, extreme ranges) are handled
    """
    # Create test data with known distribution
    np.random.seed(42)
    data = np.random.normal(loc=50, scale=10, size=(100, 100))
    
    # Test basic histogram analysis
    result = generate_histogram_analysis(data, bins=50)
    
    # Verify histogram bins are calculated correctly
    assert len(result['counts']) == 50
    assert len(result['edges']) == 51  # n_bins + 1
    assert len(result['centers']) == 50
    assert np.sum(result['counts']) == data.size
    
    # Verify cumulative distribution
    assert result['cumulative'][0] >= 0
    assert result['cumulative'][-1] == 1.0
    assert np.all(np.diff(result['cumulative']) >= 0)  # Monotonic
    
    # Test custom bin counts and ranges
    result_custom = generate_histogram_analysis(data, bins=100, range=(30, 70))
    
    # Verify custom bin counts and ranges work as expected
    assert len(result_custom['counts']) == 100
    assert result_custom['edges'][0] == 30
    assert result_custom['edges'][-1] == 70
    
    # Test mask parameter excludes invalid pixels
    mask = data > 40  # Only include values > 40
    result_masked = generate_histogram_analysis(data, bins=50, mask=mask)
    
    masked_data = data[mask]
    assert np.sum(result_masked['counts']) == len(masked_data)
    assert result_masked['data_range'][0] >= 40
    
    # Verify distribution metrics are accurate
    from scipy import stats as scipy_stats
    
    assert np.isclose(result['skewness'], scipy_stats.skew(data.ravel()), rtol=1e-5)
    assert np.isclose(result['kurtosis'], scipy_stats.kurtosis(data.ravel()), rtol=1e-5)
    
    # Verify histogram statistics are mathematically correct
    assert np.isclose(result['mean'], np.mean(data), rtol=1e-5)
    assert np.isclose(result['std'], np.std(data), rtol=1e-5)
    
    # Test percentiles
    for p_key, p_val in result['percentiles'].items():
        p_num = int(p_key[1:])  # Extract number from 'p1', 'p5', etc.
        assert np.isclose(p_val, np.percentile(data, p_num), rtol=1e-5)
    
    # Test edge case: uniform data
    uniform_data = np.full((50, 50), 42.0)
    result_uniform = generate_histogram_analysis(uniform_data, bins=10)
    
    assert result_uniform['std'] == 0.0
    assert result_uniform['skewness'] == 0.0  # Perfectly symmetric
    # All data in one bin
    assert np.max(result_uniform['counts']) == uniform_data.size
    
    # Test edge case: extreme ranges
    extreme_data = np.array([1e-10, 1e10])
    result_extreme = generate_histogram_analysis(extreme_data, bins=10)
    
    assert len(result_extreme['counts']) == 10
    assert result_extreme['data_range'][0] == 1e-10
    assert result_extreme['data_range'][1] == 1e10
    
    # Test visualization range suggestion
    # Should clip outliers for better visualization
    assert result['visualization_range'][0] > result['data_range'][0]
    assert result['visualization_range'][1] < result['data_range'][1]
    
    # Test empty data
    with pytest.raises(ValueError, match="No valid data points"):
        generate_histogram_analysis(np.array([np.nan, np.nan]))


def test_generate_histogram_analysis_bins():
    """Test histogram analysis with different binning strategies.
    
    This test should verify that:
    - Fixed bin counts produce expected results
    - Custom bin ranges work correctly
    - Automatic binning strategies are reasonable
    - Edge effects in binning are minimal
    - Bin boundaries are properly defined
    """
    # Create test data
    np.random.seed(123)
    data = np.random.exponential(scale=2.0, size=(50, 50))
    
    # Test fixed bin counts produce expected results
    for n_bins in [10, 50, 100, 256]:
        result = generate_histogram_analysis(data, bins=n_bins)
        assert len(result['counts']) == n_bins
        assert len(result['edges']) == n_bins + 1
        assert len(result['centers']) == n_bins
    
    # Test custom bin ranges work correctly
    data_range = (0, 10)
    result_range = generate_histogram_analysis(data, bins=20, range=data_range)
    
    assert result_range['edges'][0] == data_range[0]
    assert result_range['edges'][-1] == data_range[1]
    
    # Verify bins are evenly spaced
    bin_widths = np.diff(result_range['edges'])
    assert np.allclose(bin_widths, bin_widths[0])
    
    # Test bin edges array
    custom_edges = np.linspace(0, 20, 11)  # 10 bins with specific edges
    result_edges = generate_histogram_analysis(data, bins=custom_edges)
    
    assert np.array_equal(result_edges['edges'], custom_edges)
    assert len(result_edges['counts']) == len(custom_edges) - 1
    
    # Test edge effects in binning
    # Create data with values exactly at bin edges
    edge_data = np.array([0, 1, 2, 3, 4, 5, 5, 5])
    result_edge = generate_histogram_analysis(edge_data, bins=6, range=(0, 6))
    
    # Verify edge values are included correctly
    assert np.sum(result_edge['counts']) == len(edge_data)
    
    # Test bin boundaries are properly defined
    # Each bin should be [left, right)
    test_value = 2.5
    bin_idx = np.searchsorted(result_edge['edges'][1:], test_value)
    assert result_edge['edges'][bin_idx] <= test_value < result_edge['edges'][bin_idx + 1]
    
    # Test with integer data
    int_data = np.random.randint(0, 100, size=(30, 30))
    result_int = generate_histogram_analysis(int_data, bins=20)
    
    # Verify all integers are counted
    assert np.sum(result_int['counts']) == int_data.size
    
    # Test automatic binning with very skewed data
    skewed_data = np.concatenate([
        np.random.normal(0, 1, 1000),
        np.random.normal(100, 1, 10)  # Outliers
    ])
    
    result_skewed = generate_histogram_analysis(skewed_data, bins=50)
    
    # Most bins should cover the main distribution, not the outliers
    main_range_bins = np.sum((result_skewed['edges'][:-1] >= -5) & 
                            (result_skewed['edges'][1:] <= 5))
    assert main_range_bins > 10  # Reasonable coverage of main distribution


def test_generate_histogram_analysis_metrics():
    """Test distribution metrics in histogram analysis.
    
    This test should verify that:
    - Basic statistics (mean, std) match histogram data
    - Skewness measures distribution asymmetry correctly
    - Kurtosis measures distribution tail behavior
    - Entropy measures provide information content
    - Mathematical accuracy of all distribution metrics
    """
    from scipy import stats as scipy_stats
    
    # Test with normal distribution
    np.random.seed(456)
    normal_data = np.random.normal(loc=0, scale=1, size=10000)
    result_normal = generate_histogram_analysis(normal_data, bins=100)
    
    # Verify basic statistics match histogram data
    assert np.isclose(result_normal['mean'], np.mean(normal_data), rtol=1e-6)
    assert np.isclose(result_normal['std'], np.std(normal_data), rtol=1e-6)
    
    # Normal distribution should have skewness near 0
    assert np.abs(result_normal['skewness']) < 0.1
    
    # Normal distribution should have kurtosis near 0 (excess kurtosis)
    assert np.abs(result_normal['kurtosis']) < 0.2
    
    # Test with skewed distribution
    skewed_data = np.random.exponential(scale=1, size=10000)
    result_skewed = generate_histogram_analysis(skewed_data, bins=100)
    
    # Verify skewness measures distribution asymmetry correctly
    # Exponential distribution is right-skewed (positive skewness)
    assert result_skewed['skewness'] > 1.5
    assert np.isclose(result_skewed['skewness'], 
                     scipy_stats.skew(skewed_data), rtol=1e-5)
    
    # Test with heavy-tailed distribution
    heavy_tail_data = np.random.standard_t(df=3, size=10000)  # t-distribution
    result_heavy = generate_histogram_analysis(heavy_tail_data, bins=100)
    
    # Verify kurtosis measures distribution tail behavior
    # t-distribution with df=3 has heavy tails (positive excess kurtosis)
    assert result_heavy['kurtosis'] > 2.0
    assert np.isclose(result_heavy['kurtosis'], 
                     scipy_stats.kurtosis(heavy_tail_data), rtol=1e-5)
    
    # Test entropy calculation
    # Uniform distribution has maximum entropy
    uniform_data = np.random.uniform(0, 1, size=10000)
    result_uniform = generate_histogram_analysis(uniform_data, bins=50)
    
    # Peaked distribution has lower entropy
    peaked_data = np.random.normal(0, 0.1, size=10000)
    result_peaked = generate_histogram_analysis(peaked_data, bins=50)
    
    # Verify entropy measures provide information content
    assert result_uniform['entropy'] > result_peaked['entropy']
    
    # Test entropy calculation manually
    counts = result_uniform['counts']
    probs = counts[counts > 0] / np.sum(counts)
    manual_entropy = -np.sum(probs * np.log(probs))
    assert np.isclose(result_uniform['entropy'], manual_entropy, rtol=1e-6)
    
    # Test with bimodal distribution
    bimodal_data = np.concatenate([
        np.random.normal(-2, 0.5, 5000),
        np.random.normal(2, 0.5, 5000)
    ])
    result_bimodal = generate_histogram_analysis(bimodal_data, bins=100)
    
    # Bimodal distribution should have multiple peaks in mode
    assert len(result_bimodal['mode']) >= 1
    
    # Test mathematical accuracy with reference values
    # Create data with known statistics
    test_data = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5])
    result_test = generate_histogram_analysis(test_data, bins=5)
    
    expected_mean = np.mean(test_data)
    expected_std = np.std(test_data)
    expected_skew = scipy_stats.skew(test_data)
    expected_kurt = scipy_stats.kurtosis(test_data)
    
    assert np.isclose(result_test['mean'], expected_mean)
    assert np.isclose(result_test['std'], expected_std)
    assert np.isclose(result_test['skewness'], expected_skew)
    assert np.isclose(result_test['kurtosis'], expected_kurt)


def test_calculate_temporal_statistics():
    """Test temporal statistics calculation across time series.
    
    This test should verify that:
    - Temporal means and variances are calculated correctly
    - Trend analysis detects systematic changes over time
    - Detrending removes systematic trends appropriately
    - Date information enhances temporal analysis
    - Seasonal patterns are properly handled
    - Missing data doesn't bias temporal statistics
    - Mathematical accuracy of trend calculations
    """
    # Create synthetic time series with known properties
    n_times = 12
    height, width = 5, 5
    
    # Create data with linear trend + seasonal pattern + noise
    time_indices = np.arange(n_times)
    trend_slope = 2.0
    seasonal_amplitude = 5.0
    base_value = 100.0
    
    data_stack = np.zeros((n_times, height, width))
    for t in range(n_times):
        # Linear trend + seasonal (cosine) + noise
        trend = base_value + trend_slope * t
        seasonal = seasonal_amplitude * np.cos(2 * np.pi * t / 12)
        noise = np.random.normal(0, 1, (height, width))
        data_stack[t] = trend + seasonal + noise
    
    # Test basic temporal statistics
    result = calculate_temporal_statistics(data_stack, compute_trend=False)
    
    # Verify temporal means and variances are calculated correctly
    expected_mean = np.mean(data_stack, axis=0)
    expected_std = np.std(data_stack, axis=0)
    
    assert np.allclose(result['temporal_mean'], expected_mean)
    assert np.allclose(result['temporal_std'], expected_std)
    assert np.allclose(result['temporal_min'], np.min(data_stack, axis=0))
    assert np.allclose(result['temporal_max'], np.max(data_stack, axis=0))
    assert np.allclose(result['temporal_range'], 
                      np.max(data_stack, axis=0) - np.min(data_stack, axis=0))
    
    # Test coefficient of variation
    expected_cv = expected_std / np.abs(expected_mean)
    assert np.allclose(result['coefficient_variation'], expected_cv, 
                      atol=1e-10, equal_nan=True)
    
    # Test trend analysis
    result_trend = calculate_temporal_statistics(data_stack, compute_trend=True)
    
    # Verify trend analysis detects systematic changes
    # Most pixels should show positive trend close to trend_slope
    mean_slope = np.nanmean(result_trend['trend_slope'])
    assert 1.5 < mean_slope < 2.5  # Close to true slope of 2.0
    
    # R² should be reasonable for linear trend with noise
    mean_r2 = np.nanmean(result_trend['trend_r2'])
    assert mean_r2 > 0.7  # Good fit despite seasonal and noise
    
    # Test with dates
    dates = [datetime(2023, i+1, 1) for i in range(n_times)]
    result_dates = calculate_temporal_statistics(data_stack, dates=dates, 
                                               compute_trend=True)
    
    # Verify date information enhances temporal analysis
    # Slopes should be in units per day now
    assert 'trend_slope' in result_dates
    
    # Test detrending
    result_detrend = calculate_temporal_statistics(data_stack, 
                                                  compute_trend=True, 
                                                  detrend=True)
    
    # Verify detrending removes systematic trends
    # Detrended std should be lower than original
    assert np.nanmean(result_detrend['detrended_std']) < 
           np.nanmean(result_detrend['temporal_std'])
    
    # Test with missing data
    data_missing = data_stack.copy()
    # Add NaN values at different time points
    data_missing[0, 0, 0] = np.nan
    data_missing[5, 2, 2] = np.nan
    data_missing[10, 4, 4] = np.nan
    
    result_missing = calculate_temporal_statistics(data_missing, compute_trend=True)
    
    # Verify missing data doesn't bias temporal statistics
    # Stats should still be computed for pixels with sufficient data
    assert not np.isnan(result_missing['temporal_mean'][1, 1])
    assert not np.isnan(result_missing['trend_slope'][1, 1])
    
    # Test mathematical accuracy of trend calculations
    # Create perfect linear trend for one pixel
    perfect_trend = np.arange(n_times, dtype=float) * 3.0 + 50.0
    data_perfect = np.zeros((n_times, 2, 2))
    data_perfect[:, 0, 0] = perfect_trend
    
    result_perfect = calculate_temporal_statistics(data_perfect, compute_trend=True)
    
    # Should get exact slope and R²=1 for perfect linear data
    assert np.isclose(result_perfect['trend_slope'][0, 0], 3.0, rtol=1e-6)
    assert np.isclose(result_perfect['trend_r2'][0, 0], 1.0, rtol=1e-6)


def test_calculate_temporal_statistics_trends():
    """Test trend analysis in temporal statistics.
    
    This test should verify that:
    - Linear trends are detected and quantified accurately
    - Trend significance is properly assessed
    - Seasonal trends are distinguished from long-term trends
    - Multiple trend models can be compared
    - Trend confidence intervals are calculated
    """
    from scipy import stats as scipy_stats
    
    # Create data with different trend patterns
    n_times = 24  # 2 years monthly
    height, width = 4, 4
    time_indices = np.arange(n_times)
    
    # Pixel 1: Strong positive linear trend
    strong_trend = 10 + 2.0 * time_indices + np.random.normal(0, 0.5, n_times)
    
    # Pixel 2: Weak negative trend
    weak_trend = 50 - 0.1 * time_indices + np.random.normal(0, 2, n_times)
    
    # Pixel 3: No trend (pure noise)
    no_trend = 30 + np.random.normal(0, 3, n_times)
    
    # Pixel 4: Seasonal only (no linear trend)
    seasonal_only = 40 + 10 * np.sin(2 * np.pi * time_indices / 12) + \
                   np.random.normal(0, 1, n_times)
    
    # Assemble test data
    data_stack = np.zeros((n_times, height, width))
    data_stack[:, 0, 0] = strong_trend
    data_stack[:, 0, 1] = weak_trend
    data_stack[:, 0, 2] = no_trend
    data_stack[:, 0, 3] = seasonal_only
    
    # Fill rest with random data
    data_stack[:, 1:, :] = np.random.normal(25, 5, (n_times, 3, width))
    
    result = calculate_temporal_statistics(data_stack, compute_trend=True)
    
    # Verify linear trends are detected and quantified accurately
    assert 1.8 < result['trend_slope'][0, 0] < 2.2  # Strong positive ~2.0
    assert -0.3 < result['trend_slope'][0, 1] < 0.1  # Weak negative ~-0.1
    assert -0.5 < result['trend_slope'][0, 2] < 0.5  # No trend ~0.0
    
    # Verify trend significance via R²
    assert result['trend_r2'][0, 0] > 0.95  # Strong trend has high R²
    assert result['trend_r2'][0, 1] < 0.3   # Weak trend has low R²
    assert result['trend_r2'][0, 2] < 0.1   # No trend has very low R²
    
    # Seasonal pattern should have moderate R² (captures some variance)
    assert 0.3 < result['trend_r2'][0, 3] < 0.7
    
    # Test with known perfect trends
    perfect_data = np.zeros((10, 2, 2))
    x = np.arange(10)
    
    # Different slopes
    slopes = [0.5, 1.0, -0.5, -1.0]
    for i, slope in enumerate(slopes):
        row = i // 2
        col = i % 2
        perfect_data[:, row, col] = 20 + slope * x
    
    result_perfect = calculate_temporal_statistics(perfect_data, compute_trend=True)
    
    # Verify perfect detection of known slopes
    assert np.isclose(result_perfect['trend_slope'][0, 0], 0.5)
    assert np.isclose(result_perfect['trend_slope'][0, 1], 1.0)
    assert np.isclose(result_perfect['trend_slope'][1, 0], -0.5)
    assert np.isclose(result_perfect['trend_slope'][1, 1], -1.0)
    
    # All perfect trends should have R² = 1
    assert np.allclose(result_perfect['trend_r2'], 1.0)
    
    # Test seasonal vs long-term trend distinction
    # Create data with both seasonal and linear components
    combined = 50 + 0.5 * time_indices + \
              8 * np.cos(2 * np.pi * time_indices / 12) + \
              np.random.normal(0, 1, n_times)
    
    data_combined = np.zeros((n_times, 1, 1))
    data_combined[:, 0, 0] = combined
    
    result_combined = calculate_temporal_statistics(data_combined, 
                                                  compute_trend=True, 
                                                  detrend=True)
    
    # Should detect the linear trend component
    assert 0.4 < result_combined['trend_slope'][0, 0] < 0.6
    
    # Detrended data should still show seasonal pattern in std
    assert result_combined['detrended_std'][0, 0] > 5  # Seasonal amplitude


def test_calculate_temporal_statistics_detrending():
    """Test detrending in temporal statistics calculation.
    
    This test should verify that:
    - Linear detrending removes systematic trends
    - Seasonal detrending removes periodic patterns
    - Detrended data preserves residual patterns
    - Detrending doesn't introduce artifacts
    - Mathematical consistency of detrending algorithms
    """
    # Create test data with known components
    n_times = 36  # 3 years monthly
    height, width = 3, 3
    time_indices = np.arange(n_times)
    
    # Component 1: Linear trend
    linear_component = 1.5 * time_indices
    
    # Component 2: Seasonal pattern
    seasonal_component = 10 * np.sin(2 * np.pi * time_indices / 12)
    
    # Component 3: Random noise
    np.random.seed(789)
    noise_component = np.random.normal(0, 2, (n_times, height, width))
    
    # Combine components
    data_stack = np.zeros((n_times, height, width))
    for i in range(height):
        for j in range(width):
            data_stack[:, i, j] = 100 + linear_component + \
                                 seasonal_component + noise_component[:, i, j]
    
    # Test linear detrending
    result = calculate_temporal_statistics(data_stack, 
                                         compute_trend=True, 
                                         detrend=True)
    
    # Verify linear detrending removes systematic trends
    # Original data should have high temporal std due to trend
    original_std = result['temporal_std']
    detrended_std = result['detrended_std']
    
    # Detrended std should be significantly lower
    assert np.all(detrended_std < original_std)
    mean_reduction = np.mean(original_std - detrended_std)
    assert mean_reduction > 5  # Significant reduction
    
    # Verify detrended data preserves residual patterns
    # Detrended std should still capture seasonal + noise
    expected_residual_std = np.sqrt(10**2 / 2 + 2**2)  # RMS of seasonal + noise
    mean_detrended_std = np.mean(detrended_std)
    assert 0.7 * expected_residual_std < mean_detrended_std < \
           1.3 * expected_residual_std
    
    # Test that detrending doesn't introduce artifacts
    # Create data with no trend
    no_trend_data = np.random.normal(50, 5, (n_times, 2, 2))
    result_no_trend = calculate_temporal_statistics(no_trend_data, 
                                                  compute_trend=True, 
                                                  detrend=True)
    
    # Detrended std should be very close to original std
    assert np.allclose(result_no_trend['temporal_std'], 
                      result_no_trend['detrended_std'], 
                      rtol=0.1)
    
    # Test mathematical consistency
    # Manually detrend one pixel and compare
    pixel_data = data_stack[:, 0, 0]
    x = time_indices.astype(float)
    
    # Fit linear trend
    from scipy import stats as scipy_stats
    slope, intercept, _, _, _ = scipy_stats.linregress(x, pixel_data)
    
    # Remove trend
    trend_line = slope * x + intercept
    manual_detrended = pixel_data - (trend_line - np.mean(pixel_data))
    manual_std = np.std(manual_detrended)
    
    # Compare with function result
    assert np.isclose(result['detrended_std'][0, 0], manual_std, rtol=1e-5)
    
    # Test edge case: single time point
    single_time = np.random.rand(1, 3, 3)
    result_single = calculate_temporal_statistics(single_time, 
                                                compute_trend=True, 
                                                detrend=True)
    
    # Should handle gracefully
    assert np.all(np.isnan(result_single['trend_slope']))
    assert np.all(result_single['temporal_std'] == 0)
    
    # Test with sparse valid data
    sparse_data = data_stack.copy()
    # Make most time points NaN for one pixel
    sparse_data[3:, 1, 1] = np.nan
    
    result_sparse = calculate_temporal_statistics(sparse_data, 
                                                compute_trend=True, 
                                                detrend=True)
    
    # Should still compute for pixels with enough data
    assert not np.isnan(result_sparse['trend_slope'][0, 0])
    assert not np.isnan(result_sparse['detrended_std'][0, 0])
    # But not for sparse pixel
    assert np.isnan(result_sparse['trend_slope'][1, 1])


def test_compute_comparison_metrics():
    """Test comparison metrics between reference and comparison data.
    
    This test should verify that:
    - Correlation coefficients are calculated correctly
    - Mean absolute error (MAE) is computed accurately
    - Root mean square error (RMSE) is mathematically correct
    - Bias calculations detect systematic differences
    - Custom metrics can be added to the analysis
    - Mask parameter excludes invalid comparisons
    - Statistical significance of differences is assessed
    """
    # Create test data with known relationships
    size = (50, 50)
    reference = np.random.normal(50, 10, size)
    
    # Perfect match
    comparison_perfect = reference.copy()
    
    # Systematic bias
    comparison_bias = reference + 5  # Constant positive bias
    
    # Random noise
    comparison_noise = reference + np.random.normal(0, 3, size)
    
    # Scaled version
    comparison_scaled = reference * 1.1 + 2
    
    # Test perfect match
    metrics_perfect = compute_comparison_metrics(reference, comparison_perfect)
    
    assert metrics_perfect['rmse'] == 0.0
    assert metrics_perfect['mae'] == 0.0
    assert metrics_perfect['bias'] == 0.0
    assert metrics_perfect['correlation'] == 1.0
    assert metrics_perfect['r2'] == 1.0
    
    # Test systematic bias detection
    metrics_bias = compute_comparison_metrics(reference, comparison_bias)
    
    assert np.isclose(metrics_bias['bias'], 5.0, rtol=1e-6)
    assert np.isclose(metrics_bias['mae'], 5.0, rtol=1e-6)
    assert np.isclose(metrics_bias['rmse'], 5.0, rtol=1e-6)
    assert np.isclose(metrics_bias['correlation'], 1.0, rtol=1e-6)  # Perfect correlation despite bias
    
    # Test with noise
    metrics_noise = compute_comparison_metrics(reference, comparison_noise)
    
    # Verify MAE and RMSE are computed accurately
    errors = comparison_noise - reference
    expected_mae = np.mean(np.abs(errors))
    expected_rmse = np.sqrt(np.mean(errors**2))
    
    assert np.isclose(metrics_noise['mae'], expected_mae, rtol=1e-6)
    assert np.isclose(metrics_noise['rmse'], expected_rmse, rtol=1e-6)
    assert np.abs(metrics_noise['bias']) < 0.5  # Should be near zero
    assert metrics_noise['correlation'] > 0.95  # High correlation despite noise
    
    # Test MAPE (Mean Absolute Percentage Error)
    metrics_scaled = compute_comparison_metrics(reference, comparison_scaled, 
                                              metrics=['mape', 'correlation'])
    
    # MAPE should detect the 10% scaling
    assert 8 < metrics_scaled['mape'] < 12  # Around 10%
    
    # Test with mask
    mask = reference > 50  # Only compare values > 50
    metrics_masked = compute_comparison_metrics(reference, comparison_noise, mask=mask)
    
    # Verify mask parameter excludes invalid comparisons
    assert metrics_masked['n_samples'] == np.sum(mask)
    assert metrics_masked['n_samples'] < reference.size
    
    # Test custom metrics list
    custom_metrics = ['rmse', 'correlation', 'max_error']
    metrics_custom = compute_comparison_metrics(reference, comparison_noise, 
                                              metrics=custom_metrics)
    
    assert set(metrics_custom.keys()) == {'rmse', 'correlation', 'max_error', 
                                         'percentile_errors', 'n_samples'}
    assert 'mae' not in metrics_custom
    assert 'bias' not in metrics_custom
    
    # Test percentile errors
    assert metrics_custom['percentile_errors']['p50'] <= \
           metrics_custom['percentile_errors']['p75']
    assert metrics_custom['percentile_errors']['p75'] <= \
           metrics_custom['percentile_errors']['p90']
    
    # Test with NaN values
    ref_nan = reference.copy()
    comp_nan = comparison_noise.copy()
    ref_nan[10:15, 10:15] = np.nan
    comp_nan[20:25, 20:25] = np.nan
    
    metrics_nan = compute_comparison_metrics(ref_nan, comp_nan)
    
    # Should exclude all NaN pairs
    expected_valid = reference.size - (5*5 + 5*5)  # Two 5x5 NaN regions
    assert metrics_nan['n_samples'] < expected_valid  # Some overlap
    
    # Test error with mismatched shapes
    with pytest.raises(ValueError, match="shape.*doesn't match"):
        compute_comparison_metrics(reference, reference[:40, :40])
    
    # Test mathematical correctness of R²
    # R² = 1 - (SS_res / SS_tot)
    ss_res = np.sum((comparison_noise - reference)**2)
    ss_tot = np.sum((reference - np.mean(reference))**2)
    expected_r2 = 1 - (ss_res / ss_tot)
    
    assert np.isclose(metrics_noise['r2'], expected_r2, rtol=1e-6)


def test_compute_comparison_metrics_correlation():
    """Test correlation analysis in comparison metrics.
    
    This test should verify that:
    - Pearson correlation coefficients are accurate
    - Spearman rank correlations handle non-linear relationships
    - Correlation significance is properly tested
    - Partial correlations can be computed
    - Edge cases (constant arrays) are handled
    """
    from scipy import stats as scipy_stats
    
    # Test Pearson correlation accuracy
    # Create perfectly correlated data
    x = np.linspace(0, 10, 100)
    y_perfect = 2 * x + 5
    
    metrics = compute_comparison_metrics(x, y_perfect, metrics=['correlation'])
    assert np.isclose(metrics['correlation'], 1.0, atol=1e-10)
    
    # Test negative correlation
    y_negative = -3 * x + 20
    metrics_neg = compute_comparison_metrics(x, y_negative, metrics=['correlation'])
    assert np.isclose(metrics_neg['correlation'], -1.0, atol=1e-10)
    
    # Test with non-linear relationship (for Spearman)
    # Note: Current implementation uses Pearson, but test the concept
    y_nonlinear = x ** 2
    metrics_nonlinear = compute_comparison_metrics(x, y_nonlinear, metrics=['correlation'])
    
    # Pearson correlation for quadratic relationship
    expected_pearson = np.corrcoef(x, y_nonlinear)[0, 1]
    assert np.isclose(metrics_nonlinear['correlation'], expected_pearson, rtol=1e-6)
    
    # Manual Spearman correlation for comparison
    spearman_r = scipy_stats.spearmanr(x, y_nonlinear)[0]
    assert spearman_r > 0.99  # Perfect monotonic relationship
    
    # Test with noisy data
    np.random.seed(321)
    noise_level = 2.0
    y_noisy = 1.5 * x + 3 + np.random.normal(0, noise_level, len(x))
    
    metrics_noisy = compute_comparison_metrics(x, y_noisy, metrics=['correlation', 'r2'])
    
    # Correlation should be high but not perfect
    assert 0.8 < metrics_noisy['correlation'] < 0.95
    
    # R² should be correlation squared for linear relationships
    assert np.isclose(metrics_noisy['r2'], metrics_noisy['correlation']**2, rtol=0.1)
    
    # Test edge case: constant arrays
    const_ref = np.full((50, 50), 42.0)
    const_comp = np.full((50, 50), 37.0)
    
    metrics_const = compute_comparison_metrics(const_ref, const_comp, metrics=['correlation', 'r2'])
    
    # Correlation is undefined for constant arrays (will be NaN)
    assert np.isnan(metrics_const['correlation'])
    assert metrics_const['r2'] == 0.0  # No variance explained
    
    # Test with nearly constant arrays (small variance)
    nearly_const = const_ref.copy()
    nearly_const[0, 0] = 42.001  # Tiny variation
    
    metrics_nearly = compute_comparison_metrics(nearly_const, const_comp, metrics=['correlation'])
    # Should handle gracefully
    assert not np.isnan(metrics_nearly['correlation']) or \
           np.abs(metrics_nearly['correlation']) < 0.1
    
    # Test correlation with outliers
    x_outlier = x.copy()
    y_outlier = y_noisy.copy()
    # Add extreme outliers
    x_outlier[-5:] = 100
    y_outlier[-5:] = -100
    
    metrics_outlier = compute_comparison_metrics(x_outlier, y_outlier, metrics=['correlation'])
    
    # Correlation should be affected by outliers
    assert metrics_outlier['correlation'] < metrics_noisy['correlation']
    
    # Test with 2D arrays
    ref_2d = np.random.rand(30, 40)
    # Create correlated 2D array
    comp_2d = 0.8 * ref_2d + 0.2 * np.random.rand(30, 40)
    
    metrics_2d = compute_comparison_metrics(ref_2d, comp_2d, metrics=['correlation'])
    
    # Should flatten and compute correlation
    expected_corr_2d = np.corrcoef(ref_2d.ravel(), comp_2d.ravel())[0, 1]
    assert np.isclose(metrics_2d['correlation'], expected_corr_2d, rtol=1e-6)
    
    # Verify correlation bounds
    assert -1.0 <= metrics_2d['correlation'] <= 1.0


def test_compute_comparison_metrics_error():
    """Test error metrics in comparison analysis.
    
    This test should verify that:
    - MAE provides robust error measurement
    - RMSE emphasizes larger errors appropriately
    - Relative errors are scaled correctly
    - Error distributions are characterized
    - Outlier-resistant error metrics are available
    """
    # Create test data with known errors
    reference = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    
    # Test 1: Constant error
    constant_error = reference + 5
    metrics_const = compute_comparison_metrics(reference, constant_error, 
                                             metrics=['mae', 'rmse', 'bias'])
    
    assert metrics_const['mae'] == 5.0
    assert metrics_const['rmse'] == 5.0
    assert metrics_const['bias'] == 5.0
    
    # Test 2: Variable errors to show RMSE > MAE
    variable_errors = reference + np.array([1, -1, 2, -2, 3, -3, 4, -4, 5, -5])
    metrics_var = compute_comparison_metrics(reference, variable_errors, 
                                           metrics=['mae', 'rmse', 'bias'])
    
    # Verify MAE provides robust error measurement
    expected_mae = np.mean([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    assert np.isclose(metrics_var['mae'], expected_mae)
    
    # Verify RMSE emphasizes larger errors appropriately
    expected_rmse = np.sqrt(np.mean([1, 1, 4, 4, 9, 9, 16, 16, 25, 25]))
    assert np.isclose(metrics_var['rmse'], expected_rmse)
    assert metrics_var['rmse'] > metrics_var['mae']  # RMSE weights large errors more
    
    # Bias should be near zero for symmetric errors
    assert np.abs(metrics_var['bias']) < 0.1
    
    # Test 3: Outlier impact
    with_outlier = reference.copy()
    with_outlier[-1] = 200  # Large outlier
    
    metrics_outlier = compute_comparison_metrics(reference, with_outlier, 
                                               metrics=['mae', 'rmse', 'max_error'])
    
    # RMSE should be much more affected by outlier than MAE
    mae_ratio = metrics_outlier['mae'] / metrics_const['mae']
    rmse_ratio = metrics_outlier['rmse'] / metrics_const['rmse']
    assert rmse_ratio > mae_ratio
    
    # Max error captures the outlier
    assert metrics_outlier['max_error'] == 100
    
    # Test 4: Relative errors (MAPE)
    # Small values for meaningful percentages
    ref_small = np.array([1, 2, 3, 4, 5])
    comp_small = np.array([1.1, 2.2, 3.3, 4.4, 5.5])  # 10% error
    
    metrics_mape = compute_comparison_metrics(ref_small, comp_small, metrics=['mape'])
    
    # Verify relative errors are scaled correctly
    assert np.isclose(metrics_mape['mape'], 10.0, rtol=0.01)
    
    # Test MAPE with zero values
    ref_zero = np.array([0, 1, 2, 3, 4])
    comp_zero = np.array([0.1, 1.1, 2.1, 3.1, 4.1])
    
    metrics_zero = compute_comparison_metrics(ref_zero, comp_zero, metrics=['mape'])
    
    # MAPE should exclude zero reference values
    # Only compute for [1,2,3,4] with 10% error
    assert np.isclose(metrics_zero['mape'], 10.0, rtol=0.1)
    
    # Test 5: Error distribution characterization
    np.random.seed(654)
    ref_large = np.random.normal(100, 20, 1000)
    errors = np.concatenate([
        np.random.normal(0, 5, 950),   # Most errors small
        np.random.normal(0, 50, 50)    # Some large errors
    ])
    comp_large = ref_large + errors
    
    metrics_dist = compute_comparison_metrics(ref_large, comp_large)
    
    # Verify error distributions are characterized
    assert 'percentile_errors' in metrics_dist
    percs = metrics_dist['percentile_errors']
    
    # Percentiles should increase monotonically
    assert percs['p25'] < percs['p50'] < percs['p75'] < percs['p90'] < percs['p95']
    
    # Median error should be relatively small
    assert percs['p50'] < 10
    
    # High percentiles capture the outliers
    assert percs['p95'] > 30
    
    # Test 6: 2D arrays
    ref_2d = np.random.rand(20, 30) * 100
    comp_2d = ref_2d * 1.05 + np.random.normal(0, 2, (20, 30))
    
    metrics_2d = compute_comparison_metrics(ref_2d, comp_2d, 
                                          metrics=['mae', 'rmse', 'mape'])
    
    # Should work with 2D arrays
    assert metrics_2d['mae'] > 0
    assert metrics_2d['rmse'] > metrics_2d['mae']
    assert 3 < metrics_2d['mape'] < 7  # Roughly 5% error


def test_analyze_spatial_autocorrelation():
    """Test spatial autocorrelation analysis in raster data.
    
    This test should verify that:
    - Moran's I statistic is calculated correctly
    - Geary's C provides alternative autocorrelation measure
    - Lag distances define spatial neighborhood appropriately
    - Autocorrelation significance is properly tested
    - Spatial patterns are correctly characterized
    - Edge effects are handled appropriately
    - Mathematical accuracy with reference implementations
    """
    # Test 1: Perfectly clustered pattern (checkerboard)
    size = 10
    checkerboard = np.zeros((size, size))
    checkerboard[::2, ::2] = 1
    checkerboard[1::2, 1::2] = 1
    
    result_moran = analyze_spatial_autocorrelation(checkerboard, method="moran")
    
    # Moran's I should be negative for checkerboard (perfect dispersion)
    assert result_moran['global_statistic'] < -0.3
    assert 'local_statistics' in result_moran
    assert result_moran['local_statistics'].shape == checkerboard.shape
    
    # Test 2: Smooth gradient (high positive autocorrelation)
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    gradient = xx + yy
    
    result_gradient = analyze_spatial_autocorrelation(gradient, method="moran")
    
    # Moran's I should be high positive for smooth gradient
    assert result_gradient['global_statistic'] > 0.8
    
    # Test 3: Random pattern (no autocorrelation)
    np.random.seed(987)
    random_data = np.random.rand(size, size)
    
    result_random = analyze_spatial_autocorrelation(random_data, method="moran")
    
    # Moran's I should be near expected value for random data
    expected = result_random['expected_value']
    assert np.abs(result_random['global_statistic'] - expected) < 0.1
    
    # Test cluster identification
    assert 'clusters' in result_gradient
    clusters = result_gradient['clusters']
    assert clusters.shape == gradient.shape
    assert np.unique(clusters).size > 1  # Multiple cluster types
    
    # Test 4: Geary's C
    result_geary = analyze_spatial_autocorrelation(gradient, method="geary")
    
    # Geary's C should be low for clustered data (opposite of Moran's I)
    assert result_geary['global_statistic'] < 0.5
    assert result_geary['expected_value'] == 1.0
    
    # Random data should have Geary's C near 1
    result_geary_random = analyze_spatial_autocorrelation(random_data, method="geary")
    assert 0.8 < result_geary_random['global_statistic'] < 1.2
    
    # Test 5: Variogram analysis
    lag_distances = [1, 2, 3, 5, 7, 10]
    result_variogram = analyze_spatial_autocorrelation(gradient, 
                                                     lag_distances=lag_distances,
                                                     method="variogram")
    
    # Verify lag distances are used
    assert np.array_equal(result_variogram['lag_distances'], lag_distances)
    assert len(result_variogram['lag_semivariance']) == len(lag_distances)
    assert len(result_variogram['lag_counts']) == len(lag_distances)
    
    # Semivariance should increase with distance for gradient
    semivar = result_variogram['lag_semivariance']
    # Check monotonic increase (allowing for some noise)
    differences = np.diff([s for s in semivar if not np.isnan(s)])
    assert np.sum(differences > 0) >= len(differences) * 0.7
    
    # Test range estimation
    assert 'range' in result_variogram
    assert result_variogram['range'] > 0
    
    # Test 6: Edge effects with small array
    small_data = np.array([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])
    
    result_small = analyze_spatial_autocorrelation(small_data, method="moran")
    
    # Should still compute despite small size
    assert not np.isnan(result_small['global_statistic'])
    
    # Test 7: Data with NaN values
    data_nan = gradient.copy()
    data_nan[2:4, 2:4] = np.nan
    
    result_nan = analyze_spatial_autocorrelation(data_nan, method="moran")
    
    # Should handle NaN values gracefully
    assert not np.isnan(result_nan['global_statistic'])
    
    # Test 8: Specific patterns
    # Create data with known Moran's I
    # Horizontal stripes - high autocorrelation in one direction
    stripes = np.zeros((10, 10))
    stripes[::2, :] = 1
    
    result_stripes = analyze_spatial_autocorrelation(stripes, method="moran")
    
    # Should have moderate positive autocorrelation
    assert 0.2 < result_stripes['global_statistic'] < 0.6


def test_analyze_spatial_autocorrelation_moran():
    """Test Moran's I spatial autocorrelation calculation.
    
    This test should verify that:
    - Moran's I formula is correctly implemented
    - Spatial weights matrices are properly constructed
    - Statistical significance is accurately assessed
    - Different lag distances produce expected results
    - Edge corrections are applied appropriately
    """
    # Test 1: Known pattern with calculable Moran's I
    # Create a simple 4x4 pattern
    data = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1]
    ], dtype=float)
    
    result = analyze_spatial_autocorrelation(data, method="moran")
    
    # This pattern has strong positive autocorrelation
    assert result['global_statistic'] > 0.5
    
    # Verify expected value calculation
    n = data.size
    expected = -1.0 / (n - 1)
    assert np.isclose(result['expected_value'], expected, rtol=1e-6)
    
    # Test 2: Local Moran's I statistics
    local_i = result['local_statistics']
    
    # Center pixels of same-value regions should have positive local I
    assert local_i[0, 0] > 0  # Center of top-left 1s
    assert local_i[3, 3] > 0  # Center of bottom-right 1s
    
    # Edge pixels between regions should have negative local I
    assert local_i[1, 2] < 0  # Border between regions
    
    # Test 3: Cluster classification
    clusters = result['clusters']
    
    # High-High clusters (1s surrounded by 1s)
    assert clusters[0, 0] == 1
    assert clusters[1, 1] == 1
    
    # Low-Low clusters (0s surrounded by 0s)  
    assert clusters[0, 2] == 2
    assert clusters[0, 3] == 2
    
    # Test 4: Perfect segregation pattern
    segregated = np.zeros((6, 6))
    segregated[:3, :] = 1  # Top half = 1
    segregated[3:, :] = 0  # Bottom half = 0
    
    result_seg = analyze_spatial_autocorrelation(segregated, method="moran")
    
    # Should have very high positive autocorrelation
    assert result_seg['global_statistic'] > 0.7
    
    # Test 5: Weights matrix effect
    # For queen contiguity (8 neighbors), the pattern matters
    # Create a pattern where queen vs rook contiguity would differ
    diagonal_pattern = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ], dtype=float)
    
    result_diag = analyze_spatial_autocorrelation(diagonal_pattern, method="moran")
    
    # With queen contiguity, diagonal neighbors matter
    # This pattern should show some negative autocorrelation
    assert result_diag['global_statistic'] < 0
    
    # Test 6: Edge corrections
    # Create data where edge effects matter
    edge_data = np.ones((5, 5))
    edge_data[2, 2] = 0  # Single different value in center
    
    result_edge = analyze_spatial_autocorrelation(edge_data, method="moran")
    
    # Despite one outlier, should have high positive autocorrelation
    assert result_edge['global_statistic'] > 0.5
    
    # Local I at the outlier should be negative
    assert result_edge['local_statistics'][2, 2] < 0
    
    # Test 7: Constant data
    constant = np.full((5, 5), 42.0)
    result_const = analyze_spatial_autocorrelation(constant, method="moran")
    
    # Undefined autocorrelation for constant data
    assert np.isnan(result_const['global_statistic'])
    
    # Test 8: Mathematical verification
    # Manually calculate Moran's I for small example
    small = np.array([[1, 2], [3, 4]], dtype=float)
    result_small = analyze_spatial_autocorrelation(small, method="moran")
    
    # Manual calculation would be complex, but verify it's in valid range
    assert -1 <= result_small['global_statistic'] <= 1


def test_analyze_spatial_autocorrelation_geary():
    """Test Geary's C spatial autocorrelation calculation.
    
    This test should verify that:
    - Geary's C formula is correctly implemented
    - Results complement Moran's I analysis
    - Local autocorrelation patterns are detected
    - Statistical properties are properly calculated
    - Interpretation aligns with theoretical expectations
    """
    # Test 1: Clustered pattern (low Geary's C)
    clustered = np.array([
        [1, 1, 1, 2, 2],
        [1, 1, 1, 2, 2],
        [1, 1, 1, 2, 2],
        [3, 3, 3, 4, 4],
        [3, 3, 3, 4, 4]
    ], dtype=float)
    
    result_clustered = analyze_spatial_autocorrelation(clustered, method="geary")
    
    # Geary's C should be low (< 1) for clustered data
    assert result_clustered['global_statistic'] < 0.5
    assert result_clustered['expected_value'] == 1.0
    
    # Test 2: Random pattern (Geary's C near 1)
    np.random.seed(111)
    random_data = np.random.rand(10, 10)
    
    result_random = analyze_spatial_autocorrelation(random_data, method="geary")
    
    # Should be close to expected value of 1
    assert 0.8 < result_random['global_statistic'] < 1.2
    
    # Test 3: Dispersed pattern (high Geary's C)
    dispersed = np.array([
        [1, 2, 1, 2],
        [2, 1, 2, 1],
        [1, 2, 1, 2],
        [2, 1, 2, 1]
    ], dtype=float)
    
    result_dispersed = analyze_spatial_autocorrelation(dispersed, method="geary")
    
    # Geary's C should be high (> 1) for dispersed data
    assert result_dispersed['global_statistic'] > 1.3
    
    # Test 4: Complement to Moran's I
    # Same data should show opposite patterns
    gradient = np.outer(np.linspace(0, 1, 8), np.linspace(0, 1, 8))
    
    moran_result = analyze_spatial_autocorrelation(gradient, method="moran")
    geary_result = analyze_spatial_autocorrelation(gradient, method="geary")
    
    # High Moran's I should correspond to low Geary's C
    assert moran_result['global_statistic'] > 0.7
    assert geary_result['global_statistic'] < 0.3
    
    # Test 5: Edge effects
    # Small array to test edge handling
    small = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    
    result_small = analyze_spatial_autocorrelation(small, method="geary")
    
    # Should handle edges properly
    assert not np.isnan(result_small['global_statistic'])
    assert result_small['global_statistic'] > 0  # Valid positive value
    
    # Test 6: Constant values
    constant = np.full((5, 5), 10.0)
    result_const = analyze_spatial_autocorrelation(constant, method="geary")
    
    # Undefined for constant data
    assert np.isnan(result_const['global_statistic'])
    
    # Test 7: Interpretation test
    # Create specific patterns and verify interpretation
    
    # Perfect gradient (very smooth transitions)
    smooth_gradient = np.outer(np.arange(10), np.arange(10)) / 100.0
    result_smooth = analyze_spatial_autocorrelation(smooth_gradient, method="geary")
    
    # Should have very low Geary's C (high similarity between neighbors)
    assert result_smooth['global_statistic'] < 0.1
    
    # Noisy data (high variation between neighbors)
    noisy = smooth_gradient + np.random.normal(0, 0.5, (10, 10))
    result_noisy = analyze_spatial_autocorrelation(noisy, method="geary")
    
    # Should have higher Geary's C than smooth version
    assert result_noisy['global_statistic'] > result_smooth['global_statistic']
    
    # Test 8: NaN handling
    data_nan = clustered.copy()
    data_nan[2, 2] = np.nan
    
    result_nan = analyze_spatial_autocorrelation(data_nan, method="geary")
    
    # Should handle NaN gracefully
    assert not np.isnan(result_nan['global_statistic'])
    assert result_nan['global_statistic'] < 1.0  # Still clustered despite NaN


def test_generate_quality_metrics_summary():
    """Test comprehensive quality metrics summary generation.
    
    This test should verify that:
    - Multiple quality dimensions are assessed
    - Reference data comparisons are included when available
    - Comprehensive flag provides detailed analysis
    - Summary format is informative and actionable
    - Statistical significance is properly reported
    - Quality thresholds are appropriately applied
    - Results guide data processing decisions
    """
    # Create test data with known properties
    np.random.seed(222)
    
    # Good quality data
    good_data = np.random.normal(50, 10, (100, 100))
    
    # Poor quality data (lots of missing values)
    poor_data = good_data.copy()
    poor_data[np.random.rand(100, 100) > 0.3] = np.nan  # 70% missing
    
    # Test basic summary
    summary_good = generate_quality_metrics_summary(good_data, comprehensive=False)
    
    # Verify multiple quality dimensions are assessed
    assert 'basic_stats' in summary_good
    assert 'quality_scores' in summary_good
    assert 'recommendations' in summary_good
    
    # Check basic stats
    stats = summary_good['basic_stats']
    assert np.isclose(stats['mean'], 50, rtol=0.1)
    assert np.isclose(stats['std'], 10, rtol=0.1)
    assert stats['valid_percent'] == 100.0
    
    # Check quality scores
    scores = summary_good['quality_scores']
    assert scores['completeness'] == 1.0
    assert 'overall' in scores
    
    # Test comprehensive analysis
    summary_comprehensive = generate_quality_metrics_summary(good_data, comprehensive=True)
    
    # Verify comprehensive flag provides detailed analysis
    assert 'distribution_metrics' in summary_comprehensive
    assert 'spatial_metrics' in summary_comprehensive
    
    # Check distribution metrics
    dist_metrics = summary_comprehensive['distribution_metrics']
    assert 'skewness' in dist_metrics
    assert 'kurtosis' in dist_metrics
    assert 'entropy' in dist_metrics
    assert 'percentiles' in dist_metrics
    
    # Normal distribution should have low skewness/kurtosis
    assert np.abs(dist_metrics['skewness']) < 0.5
    assert np.abs(dist_metrics['kurtosis']) < 1.0
    
    # Test with reference data
    reference = good_data + np.random.normal(0, 2, (100, 100))
    summary_ref = generate_quality_metrics_summary(good_data, reference=reference, 
                                                 comprehensive=True)
    
    # Verify reference data comparisons are included
    assert 'comparison_metrics' in summary_ref
    comp_metrics = summary_ref['comparison_metrics']
    assert 'correlation' in comp_metrics
    assert comp_metrics['correlation'] > 0.95  # High correlation
    
    # Test with poor quality data
    summary_poor = generate_quality_metrics_summary(poor_data, comprehensive=True)
    
    # Check quality assessment
    assert summary_poor['quality_scores']['completeness'] < 0.35
    assert summary_poor['quality_scores']['overall'] < 0.5
    
    # Verify recommendations are actionable
    recommendations = summary_poor['recommendations']
    assert len(recommendations) > 0
    assert any('completeness' in r.lower() for r in recommendations)
    
    # Test with mask
    mask = good_data > 40
    summary_masked = generate_quality_metrics_summary(good_data, mask=mask)
    
    # Stats should reflect masked data
    assert summary_masked['basic_stats']['valid_percent'] < 100.0
    assert summary_masked['basic_stats']['mean'] > 50  # Only values > 40
    
    # Test quality thresholds
    # Create skewed data
    skewed_data = np.random.exponential(scale=10, size=(50, 50))
    summary_skewed = generate_quality_metrics_summary(skewed_data, comprehensive=True)
    
    # Should detect high skewness
    assert summary_skewed['distribution_metrics']['skewness'] > 1.5
    assert any('skewed' in r.lower() for r in summary_skewed['recommendations'])
    
    # Test spatial quality assessment
    # Create data with spatial artifacts
    artifacts_data = np.random.rand(50, 50)
    artifacts_data[::2, ::2] = 10  # Checkerboard artifact
    
    summary_artifacts = generate_quality_metrics_summary(artifacts_data, 
                                                       comprehensive=True)
    
    # Should detect negative spatial autocorrelation
    if 'spatial_metrics' in summary_artifacts:
        assert summary_artifacts['spatial_metrics']['spatial_pattern'] == 'dispersed'
        assert any('artifact' in r.lower() for r in 
                  summary_artifacts['recommendations'])
    
    # Test empty data
    empty_data = np.full((10, 10), np.nan)
    summary_empty = generate_quality_metrics_summary(empty_data)
    
    assert summary_empty['basic_stats']['n_valid'] == 0
    assert summary_empty['basic_stats']['valid_percent'] == 0.0
    assert summary_empty['quality_scores']['completeness'] == 0.0


def test_generate_quality_metrics_summary_comprehensive():
    """Test comprehensive quality metrics analysis.
    
    This test should verify that:
    - All relevant quality dimensions are covered
    - Detailed diagnostics are provided
    - Quality scores are properly weighted
    - Recommendations are actionable
    - Performance doesn't degrade with comprehensive analysis
    """
    # Create complex test data with multiple quality issues
    np.random.seed(333)
    
    # Base data
    base = np.random.normal(100, 20, (200, 200))
    
    # Add various quality issues
    # 1. Missing data patches
    base[50:70, 50:70] = np.nan
    base[150:160, 150:160] = np.nan
    
    # 2. Outliers
    base[np.random.rand(200, 200) < 0.01] = 1000  # 1% extreme outliers
    
    # 3. Systematic bias in one region
    base[100:150, :50] += 50
    
    # Test comprehensive analysis
    start_time = datetime.now()
    summary = generate_quality_metrics_summary(base, comprehensive=True)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Verify performance doesn't degrade with comprehensive analysis
    assert elapsed < 5.0  # Should complete within 5 seconds
    
    # Verify all relevant quality dimensions are covered
    required_sections = ['basic_stats', 'distribution_metrics', 
                        'spatial_metrics', 'quality_scores', 'recommendations']
    for section in required_sections:
        assert section in summary
    
    # Check detailed diagnostics are provided
    # Distribution metrics
    dist = summary['distribution_metrics']
    assert 'skewness' in dist
    assert 'kurtosis' in dist
    assert 'entropy' in dist
    assert 'percentiles' in dist
    assert 'visualization_range' in dist
    
    # Percentiles should be comprehensive
    percentiles = dist['percentiles']
    expected_percentiles = ['p1', 'p5', 'p25', 'p50', 'p75', 'p95', 'p99']
    for p in expected_percentiles:
        assert p in percentiles
    
    # Spatial metrics
    spatial = summary['spatial_metrics']
    assert 'moran_i' in spatial
    assert 'expected_i' in spatial
    assert 'spatial_pattern' in spatial
    
    # Verify quality scores are properly weighted
    scores = summary['quality_scores']
    assert 'completeness' in scores
    assert 'distribution' in scores
    assert 'spatial_coherence' in scores
    assert 'overall' in scores
    
    # Overall score should be weighted average
    # With missing data, outliers, etc., scores should be moderate
    assert 0.3 < scores['overall'] < 0.8
    
    # Completeness should reflect missing data
    missing_fraction = (20*20 + 10*10) / (200*200)  # Two missing patches
    expected_completeness = 1 - missing_fraction
    assert np.abs(scores['completeness'] - expected_completeness) < 0.05
    
    # Distribution score should be affected by outliers
    assert scores['distribution'] < 0.8  # Penalized for outliers
    
    # Verify recommendations are actionable
    recommendations = summary['recommendations']
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    
    # Should recommend addressing outliers due to extreme values
    assert any('skewed' in r.lower() or 'transform' in r.lower() 
              for r in recommendations)
    
    # Test with reference data for additional metrics
    reference = base + np.random.normal(0, 5, (200, 200))
    summary_ref = generate_quality_metrics_summary(base, reference=reference, 
                                                 comprehensive=True)
    
    assert 'comparison_metrics' in summary_ref
    comp = summary_ref['comparison_metrics']
    assert 'rmse' in comp
    assert 'correlation' in comp
    assert 'percentile_errors' in comp
    
    # Test edge case: uniform data
    uniform = np.full((100, 100), 42.0)
    summary_uniform = generate_quality_metrics_summary(uniform, comprehensive=True)
    
    # Should handle gracefully
    assert summary_uniform['distribution_metrics']['skewness'] == 0.0
    assert summary_uniform['distribution_metrics']['kurtosis'] == 0.0
    
    # Test with very small data
    small = np.random.rand(5, 5)
    summary_small = generate_quality_metrics_summary(small, comprehensive=True)
    
    # Should still provide all metrics
    assert 'distribution_metrics' in summary_small
    assert 'spatial_metrics' in summary_small


def test_calculate_uncertainty_metrics():
    """Test uncertainty metrics calculation for processed data.
    
    This test should verify that:
    - Observation count influences uncertainty estimates
    - Quality scores affect uncertainty calculations
    - Statistical uncertainty measures are accurate
    - Spatial uncertainty patterns are identified
    - Temporal uncertainty variations are characterized
    - Mathematical accuracy of uncertainty propagation
    - Results inform confidence in derived products
    """
    # Create test data
    data = np.random.normal(100, 10, (50, 50))
    
    # Test basic uncertainty calculation
    result = calculate_uncertainty_metrics(data)
    
    # Should return all components
    assert 'uncertainty_estimate' in result
    assert 'confidence_interval_lower' in result
    assert 'confidence_interval_upper' in result
    assert 'uncertainty_sources' in result
    assert 'relative_contributions' in result
    
    # Basic uncertainty should be reasonable
    uncertainty = result['uncertainty_estimate']
    assert uncertainty.shape == data.shape
    assert np.all(uncertainty > 0)
    assert np.all(~np.isnan(uncertainty[~np.isnan(data)]))
    
    # Test observation count influence
    obs_count = np.random.randint(1, 20, size=(50, 50))
    result_obs = calculate_uncertainty_metrics(data, observation_count=obs_count)
    
    # Verify observation count influences uncertainty estimates
    # More observations should reduce uncertainty
    high_obs_mask = obs_count > 15
    low_obs_mask = obs_count < 5
    
    mean_uncertainty_high = np.mean(result_obs['uncertainty_estimate'][high_obs_mask])
    mean_uncertainty_low = np.mean(result_obs['uncertainty_estimate'][low_obs_mask])
    
    assert mean_uncertainty_high < mean_uncertainty_low
    
    # Test quality scores influence
    quality_scores = np.random.uniform(0.3, 1.0, size=(50, 50))
    result_quality = calculate_uncertainty_metrics(data, quality_scores=quality_scores)
    
    # Verify quality scores affect uncertainty calculations
    # Lower quality should increase uncertainty
    high_quality_mask = quality_scores > 0.9
    low_quality_mask = quality_scores < 0.4
    
    mean_unc_high_qual = np.mean(result_quality['uncertainty_estimate'][high_quality_mask])
    mean_unc_low_qual = np.mean(result_quality['uncertainty_estimate'][low_quality_mask])
    
    assert mean_unc_high_qual < mean_unc_low_qual
    
    # Test uncertainty sources breakdown
    sources = result_quality['uncertainty_sources']
    assert 'measurement' in sources
    assert 'sampling' in sources
    assert 'processing' in sources
    
    # All sources should have same shape as data
    for source_unc in sources.values():
        assert source_unc.shape == data.shape
    
    # Test relative contributions
    contributions = result_quality['relative_contributions']
    assert 'measurement_percent' in contributions
    assert 'sampling_percent' in contributions
    assert 'processing_percent' in contributions
    
    # Contributions should sum to ~100% for each pixel
    total_contrib = (contributions['measurement_percent'] + 
                    contributions['sampling_percent'] + 
                    contributions['processing_percent'])
    
    # Allow for numerical errors
    assert np.all(np.abs(total_contrib[~np.isnan(total_contrib)] - 100) < 1)
    
    # Test confidence intervals
    ci_lower = result['confidence_interval_lower']
    ci_upper = result['confidence_interval_upper']
    
    # Confidence intervals should bracket the data
    assert np.all(ci_lower <= data)
    assert np.all(ci_upper >= data)
    
    # 95% CI should be ~1.96 * uncertainty
    expected_width = 2 * 1.96 * result['uncertainty_estimate']
    actual_width = ci_upper - ci_lower
    assert np.allclose(actual_width, expected_width, rtol=1e-6)
    
    # Test mathematical accuracy of uncertainty propagation
    # Create data with known uncertainty components
    known_data = np.full((10, 10), 100.0)
    known_obs_count = np.full((10, 10), 16)  # sqrt(16) = 4
    known_quality = np.full((10, 10), 0.5)  # 50% quality
    
    result_known = calculate_uncertainty_metrics(known_data, 
                                               observation_count=known_obs_count,
                                               quality_scores=known_quality)
    
    # Verify propagation follows root sum of squares
    meas_unc = result_known['uncertainty_sources']['measurement']
    samp_unc = result_known['uncertainty_sources']['sampling']
    proc_unc = result_known['uncertainty_sources']['processing']
    
    expected_total = np.sqrt(meas_unc**2 + samp_unc**2 + proc_unc**2)
    assert np.allclose(result_known['uncertainty_estimate'], expected_total, rtol=1e-6)
    
    # Test with spatial patterns
    # Create data with spatial gradient
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    xx, yy = np.meshgrid(x, y)
    spatial_data = 100 + 50 * (xx + yy)
    
    # Observation count decreases with distance from center
    center_x, center_y = 25, 25
    dist_from_center = np.sqrt((xx * 50 - center_x)**2 + (yy * 50 - center_y)**2)
    spatial_obs_count = np.maximum(1, 20 - dist_from_center / 2).astype(int)
    
    result_spatial = calculate_uncertainty_metrics(spatial_data, 
                                                 observation_count=spatial_obs_count)
    
    # Uncertainty should be lower at center (more observations)
    center_unc = result_spatial['uncertainty_estimate'][20:30, 20:30].mean()
    edge_unc = result_spatial['uncertainty_estimate'][:10, :10].mean()
    assert center_unc < edge_unc


def test_calculate_uncertainty_metrics_propagation():
    """Test uncertainty propagation in metrics calculation.
    
    This test should verify that:
    - Input uncertainties are properly propagated
    - Non-linear transformations handle uncertainty correctly
    - Correlation effects are accounted for
    - Monte Carlo methods validate analytical results
    - Mathematical consistency of propagation rules
    """
    # Test 1: Linear propagation
    # For linear operations, uncertainty propagates simply
    data = np.full((20, 20), 100.0)
    
    # Known observation counts for predictable uncertainty
    obs_count = np.full((20, 20), 25)  # std_error = sigma/sqrt(25) = sigma/5
    
    result = calculate_uncertainty_metrics(data, observation_count=obs_count)
    
    # Verify input uncertainties are properly propagated
    # Components should combine as root sum of squares
    sources = result['uncertainty_sources']
    total_unc = result['uncertainty_estimate']
    
    # Manual calculation
    expected_total = np.sqrt(sources['measurement']**2 + 
                           sources['sampling']**2 + 
                           sources['processing']**2)
    
    assert np.allclose(total_unc, expected_total, rtol=1e-10)
    
    # Test 2: Scaling effects
    # If data is scaled, uncertainty should scale proportionally
    data_scaled = data * 2
    result_scaled = calculate_uncertainty_metrics(data_scaled, 
                                                observation_count=obs_count)
    
    # Measurement uncertainty (5% of value) should double
    assert np.allclose(result_scaled['uncertainty_sources']['measurement'],
                      result['uncertainty_sources']['measurement'] * 2,
                      rtol=1e-6)
    
    # Test 3: Quality score effects
    # Test full range of quality scores
    quality_range = np.linspace(0, 1, 20)
    quality_grid = np.repeat(quality_range[:, np.newaxis], 20, axis=1)
    
    result_quality = calculate_uncertainty_metrics(data, quality_scores=quality_grid)
    
    # Processing uncertainty should vary with quality
    proc_unc = result_quality['uncertainty_sources']['processing']
    
    # Higher quality (right side) should have lower uncertainty
    assert proc_unc[0, 0] > proc_unc[0, -1]  # First row, left vs right
    
    # Test 4: Zero observation edge case
    obs_zero = obs_count.copy()
    obs_zero[5:7, 5:7] = 0
    
    result_zero = calculate_uncertainty_metrics(data, observation_count=obs_zero)
    
    # Should handle gracefully with NaN
    assert np.all(np.isnan(result_zero['uncertainty_estimate'][5:7, 5:7]))
    assert np.all(~np.isnan(result_zero['uncertainty_estimate'][0, 0]))
    
    # Test 5: Mathematical consistency
    # Test specific uncertainty values
    test_value = 50.0
    test_data = np.full((5, 5), test_value)
    test_obs = np.array([[1, 4, 9, 16, 25],
                        [1, 4, 9, 16, 25],
                        [1, 4, 9, 16, 25],
                        [1, 4, 9, 16, 25],
                        [1, 4, 9, 16, 25]])
    
    result_test = calculate_uncertainty_metrics(test_data, observation_count=test_obs)
    
    # Sampling uncertainty should scale as 1/sqrt(n)
    samp_unc = result_test['uncertainty_sources']['sampling']
    base_sigma = np.std(test_data) if np.std(test_data) > 0 else 1.0
    
    # For n=1, uncertainty is high
    # For n=25, uncertainty is base_sigma/5
    ratio = samp_unc[0, 0] / samp_unc[0, 4]
    expected_ratio = np.sqrt(25) / np.sqrt(1)  # 5
    assert np.abs(ratio - expected_ratio) < 0.1
    
    # Test 6: Confidence interval validation
    # For normal distribution, 95% CI = mean ± 1.96*sigma
    ci_lower = result_test['confidence_interval_lower']
    ci_upper = result_test['confidence_interval_upper']
    total_unc = result_test['uncertainty_estimate']
    
    expected_lower = test_data - 1.96 * total_unc
    expected_upper = test_data + 1.96 * total_unc
    
    assert np.allclose(ci_lower, expected_lower, rtol=1e-10)
    assert np.allclose(ci_upper, expected_upper, rtol=1e-10)
    
    # Test 7: Relative contributions consistency
    contributions = result_test['relative_contributions']
    
    # Each contribution should be between 0 and 100%
    for key in ['measurement_percent', 'sampling_percent', 'processing_percent']:
        contrib = contributions[key]
        assert np.all((contrib >= 0) | np.isnan(contrib))
        assert np.all((contrib <= 100) | np.isnan(contrib))
    
    # Test confidence level
    assert result_test['confidence_level'] == 0.95


def test_calculate_uncertainty_metrics_observation_count():
    """Test uncertainty scaling with observation count.
    
    This test should verify that:
    - More observations reduce uncertainty appropriately
    - Statistical uncertainty follows expected scaling laws
    - Minimum observation thresholds are enforced
    - Edge cases (single observation) are handled
    - Uncertainty confidence intervals are calculated
    """
    # Create consistent test data
    np.random.seed(444)
    base_value = 100.0
    base_std = 20.0
    
    # Test array with varying observation counts
    data = np.full((10, 10), base_value)
    
    # Create observation count array with specific values
    obs_counts = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        [1, 4, 9, 16, 25, 36, 49, 64, 81, 100],
        [1, 4, 9, 16, 25, 36, 49, 64, 81, 100],
        [2, 2, 2, 2, 2, 8, 8, 8, 8, 8],
        [5, 5, 5, 5, 5, 20, 20, 20, 20, 20]
    ])
    
    result = calculate_uncertainty_metrics(data, observation_count=obs_counts)
    
    # Test 1: More observations reduce uncertainty appropriately
    sampling_unc = result['uncertainty_sources']['sampling']
    
    # Compare pixels with different observation counts
    # Row 0: increasing counts 1-10
    assert sampling_unc[0, 0] > sampling_unc[0, 9]  # 1 obs vs 10 obs
    assert sampling_unc[0, 1] > sampling_unc[0, 4]  # 2 obs vs 5 obs
    
    # Test 2: Statistical uncertainty follows expected scaling laws
    # Uncertainty should scale as 1/sqrt(n)
    # Row 6 has perfect squares: 1, 4, 9, 16, 25, ...
    
    # Compare ratios
    ratio_4_to_1 = sampling_unc[6, 0] / sampling_unc[6, 1]  # n=1 vs n=4
    assert np.isclose(ratio_4_to_1, 2.0, rtol=0.1)  # sqrt(4)/sqrt(1) = 2
    
    ratio_25_to_1 = sampling_unc[6, 0] / sampling_unc[6, 4]  # n=1 vs n=25
    assert np.isclose(ratio_25_to_1, 5.0, rtol=0.1)  # sqrt(25)/sqrt(1) = 5
    
    ratio_100_to_25 = sampling_unc[6, 4] / sampling_unc[6, 9]  # n=25 vs n=100
    assert np.isclose(ratio_100_to_25, 2.0, rtol=0.1)  # sqrt(100)/sqrt(25) = 2
    
    # Test 3: Edge case - single observation
    single_obs_unc = sampling_unc[4, :]  # Row with all n=1
    
    # Should have high but consistent uncertainty
    assert np.all(single_obs_unc > 0)
    assert np.allclose(single_obs_unc, single_obs_unc[0])  # All same
    
    # Test 4: High observation counts
    high_obs_unc = sampling_unc[5, :]  # Row with all n=100
    
    # Should have low uncertainty
    assert np.all(high_obs_unc < single_obs_unc)
    assert np.all(high_obs_unc < sampling_unc[0, 0])  # Less than n=1
    
    # Test 5: Uncertainty confidence intervals
    ci_lower = result['confidence_interval_lower']
    ci_upper = result['confidence_interval_upper']
    total_unc = result['uncertainty_estimate']
    
    # Width should be proportional to observation count
    width = ci_upper - ci_lower
    
    # Pixels with more observations should have narrower CIs
    assert width[0, 0] > width[0, 9]  # n=1 vs n=10
    assert width[2, 0] > width[2, 9]  # n=10 vs n=100
    
    # CI width should be 2 * 1.96 * uncertainty
    expected_width = 2 * 1.96 * total_unc
    assert np.allclose(width, expected_width, rtol=1e-6)
    
    # Test 6: Zero observations
    obs_with_zero = obs_counts.copy()
    obs_with_zero[7, 5:7] = 0
    
    result_zero = calculate_uncertainty_metrics(data, observation_count=obs_with_zero)
    
    # Should produce NaN for zero observations
    assert np.all(np.isnan(result_zero['uncertainty_estimate'][7, 5:7]))
    assert np.all(~np.isnan(result_zero['uncertainty_estimate'][7, 0:5]))
    
    # Test 7: Minimum threshold behavior
    # Even with 1 observation, should provide uncertainty estimate
    one_obs_result = calculate_uncertainty_metrics(data, 
                                                 observation_count=np.ones((5, 5)))
    
    assert np.all(one_obs_result['uncertainty_estimate'] > 0)
    assert np.all(~np.isnan(one_obs_result['uncertainty_estimate']))
    
    # Test 8: Relative contribution changes with observation count
    contributions = result['relative_contributions']
    
    # With few observations, sampling uncertainty dominates
    few_obs_mask = obs_counts < 5
    many_obs_mask = obs_counts > 50
    
    mean_sampling_few = np.nanmean(contributions['sampling_percent'][few_obs_mask])
    mean_sampling_many = np.nanmean(contributions['sampling_percent'][many_obs_mask])
    
    assert mean_sampling_few > mean_sampling_many


def test_statistics_array_shapes():
    """Test statistical functions with various array shapes.
    
    This test should verify that:
    - 2D arrays (single images) are handled correctly
    - 3D arrays (time series, multi-band) work properly
    - Different array dimensions produce consistent results
    - Broadcasting rules are followed correctly
    - Memory efficiency with large arrays
    """
    # Test 1: 2D arrays (single images)
    data_2d = np.random.rand(50, 60)
    
    # Coverage statistics
    cov_2d = calculate_coverage_statistics(data_2d)
    assert cov_2d['total_pixels'] == 50 * 60
    
    # Histogram analysis
    hist_2d = generate_histogram_analysis(data_2d)
    assert np.sum(hist_2d['counts']) == data_2d.size
    
    # Quality metrics
    quality_2d = generate_quality_metrics_summary(data_2d)
    assert quality_2d['basic_stats']['n_total'] == data_2d.size
    
    # Test 2: 3D arrays (time series)
    data_3d = np.random.rand(10, 50, 60)  # 10 time steps
    
    # Temporal statistics
    temp_stats = calculate_temporal_statistics(data_3d)
    assert temp_stats['temporal_mean'].shape == (50, 60)
    assert temp_stats['temporal_std'].shape == (50, 60)
    
    # Test 3: Different shapes produce consistent results
    # Same data, different shapes
    data_flat = data_2d.ravel()
    data_reshaped = data_2d.reshape(25, 120)
    
    # Histogram should be identical regardless of shape
    hist_flat = generate_histogram_analysis(data_flat, bins=50)
    hist_reshaped = generate_histogram_analysis(data_reshaped, bins=50)
    
    assert np.array_equal(hist_flat['counts'], hist_reshaped['counts'])
    assert np.isclose(hist_flat['mean'], hist_reshaped['mean'])
    
    # Test 4: Broadcasting with masks
    mask_2d = data_2d > 0.5
    zones_2d = np.random.randint(1, 5, size=(50, 60))
    
    # Coverage with mask
    cov_masked = calculate_coverage_statistics(data_2d, mask=mask_2d)
    assert cov_masked['valid_pixels'] == np.sum(mask_2d)
    
    # Zonal statistics
    zonal_2d = compute_zonal_statistics(data_2d, zones_2d)
    for zone_id in np.unique(zones_2d):
        if zone_id != 0:
            assert zone_id in zonal_2d
            assert zonal_2d[zone_id]['count'] == np.sum(zones_2d == zone_id)
    
    # Test 5: Large arrays (memory efficiency)
    # Create large array but test with smaller sample
    large_size = (1000, 1000)
    
    # Test coverage statistics with large array
    large_data = np.random.rand(*large_size)
    start_time = datetime.now()
    cov_large = calculate_coverage_statistics(large_data, compute_regional=False)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    assert elapsed < 2.0  # Should be fast
    assert cov_large['total_pixels'] == np.prod(large_size)
    
    # Test 6: Edge cases - single pixel
    single_pixel = np.array([[42.0]])
    
    cov_single = calculate_coverage_statistics(single_pixel)
    assert cov_single['total_pixels'] == 1
    assert cov_single['valid_pixels'] == 1
    
    hist_single = generate_histogram_analysis(single_pixel)
    assert hist_single['mean'] == 42.0
    assert hist_single['std'] == 0.0
    
    # Test 7: 1D arrays
    data_1d = np.random.rand(100)
    
    # Should work with 1D arrays
    hist_1d = generate_histogram_analysis(data_1d)
    assert np.sum(hist_1d['counts']) == 100
    
    # Test 8: Shape preservation in outputs
    # Uncertainty metrics
    unc_2d = calculate_uncertainty_metrics(data_2d)
    assert unc_2d['uncertainty_estimate'].shape == data_2d.shape
    
    # Spatial autocorrelation with 2D only
    spatial_2d = analyze_spatial_autocorrelation(data_2d)
    if 'local_statistics' in spatial_2d:
        assert spatial_2d['local_statistics'].shape == data_2d.shape
    
    # Test 9: Mismatched shapes
    mask_wrong = np.ones((30, 40), dtype=bool)
    with pytest.raises(ValueError, match="shape.*doesn't match"):
        calculate_coverage_statistics(data_2d, mask=mask_wrong)
    
    zones_wrong = np.ones((30, 40), dtype=int)
    with pytest.raises(ValueError, match="shape.*doesn't match"):
        compute_zonal_statistics(data_2d, zones_wrong)


def test_statistics_data_types():
    """Test statistical calculations with different data types.
    
    This test should verify that:
    - Integer data preserves precision where appropriate
    - Float data maintains numerical accuracy
    - Mixed data types are handled correctly
    - Data type conversions are performed when needed
    - Numerical stability with extreme values
    """
    # Test 1: Integer data types
    int_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    
    # Coverage statistics with integers
    cov_int = calculate_coverage_statistics(int_data)
    assert cov_int['valid_pixels'] == 9
    assert isinstance(cov_int['valid_pixels'], (int, np.integer))
    
    # Zonal statistics with integers
    zones_int = np.array([[1, 1, 2], [1, 1, 2], [3, 3, 2]], dtype=np.int16)
    zonal_int = compute_zonal_statistics(int_data, zones_int)
    
    # Check precision is maintained
    assert zonal_int[1]['sum'] == 12.0  # 1+2+4+5
    assert zonal_int[1]['mean'] == 3.0
    
    # Test 2: Float data types
    float32_data = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=np.float32)
    float64_data = float32_data.astype(np.float64)
    
    # Compare results between float32 and float64
    hist32 = generate_histogram_analysis(float32_data, bins=10)
    hist64 = generate_histogram_analysis(float64_data, bins=10)
    
    # Results should be very close
    assert np.isclose(hist32['mean'], hist64['mean'], rtol=1e-6)
    assert np.isclose(hist32['std'], hist64['std'], rtol=1e-6)
    
    # Test 3: Mixed precision in operations
    mixed_ref = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    mixed_comp = np.array([1.1, 2.1, 3.1, 4.1, 5.1], dtype=np.float64)
    
    metrics_mixed = compute_comparison_metrics(mixed_ref, mixed_comp)
    
    # Should handle type promotion correctly
    assert np.isclose(metrics_mixed['mae'], 0.1, rtol=1e-6)
    assert np.isclose(metrics_mixed['bias'], 0.1, rtol=1e-6)
    
    # Test 4: Boolean data
    bool_data = np.array([[True, False, True], [False, True, False]], dtype=bool)
    
    # Should convert to numeric for statistics
    cov_bool = calculate_coverage_statistics(bool_data)
    assert cov_bool['valid_pixels'] == 6  # All non-masked pixels
    
    # Test 5: Extreme values
    # Very large values
    large_data = np.array([1e10, 2e10, 3e10, 4e10], dtype=np.float64)
    hist_large = generate_histogram_analysis(large_data)
    
    assert not np.isnan(hist_large['mean'])
    assert not np.isnan(hist_large['std'])
    assert hist_large['mean'] == 2.5e10
    
    # Very small values
    small_data = np.array([1e-10, 2e-10, 3e-10, 4e-10], dtype=np.float64)
    hist_small = generate_histogram_analysis(small_data)
    
    assert not np.isnan(hist_small['mean'])
    assert hist_small['mean'] == 2.5e-10
    
    # Test 6: Unsigned integers
    uint8_data = np.array([[0, 128, 255], [64, 192, 128]], dtype=np.uint8)
    
    hist_uint8 = generate_histogram_analysis(uint8_data, bins=256)
    assert hist_uint8['data_range'][0] >= 0
    assert hist_uint8['data_range'][1] <= 255
    
    # Test 7: Type preservation in outputs
    # Integer zones should stay integer
    zones_preserve = np.array([[1, 2], [3, 4]], dtype=np.int64)
    data_preserve = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    zonal_preserve = compute_zonal_statistics(data_preserve, zones_preserve)
    
    # Zone IDs should be integers in output
    for zone_id in zonal_preserve.keys():
        assert isinstance(zone_id, (int, np.integer))
    
    # Test 8: NaN handling with different types
    # Float with NaN
    float_nan = np.array([1.0, 2.0, np.nan, 4.0])
    hist_nan = generate_histogram_analysis(float_nan)
    
    assert hist_nan['mean'] == pytest.approx(7/3)  # (1+2+4)/3
    assert not np.isnan(hist_nan['mean'])
    
    # Test 9: Complex operations with type mixing
    # Temporal statistics with integer time series
    time_series_int = np.random.randint(0, 100, size=(10, 20, 20), dtype=np.int16)
    temp_int = calculate_temporal_statistics(time_series_int)
    
    # Should produce float outputs even from integer inputs
    assert temp_int['temporal_mean'].dtype == np.float64
    assert temp_int['temporal_std'].dtype == np.float64
    
    # Test 10: Overflow protection
    # Large integers that might overflow
    large_int = np.array([2**30, 2**30, 2**30], dtype=np.int32)
    
    # Should handle without overflow
    stats_overflow = compute_zonal_statistics(large_int, 
                                            np.ones_like(large_int, dtype=int))
    
    assert stats_overflow[1]['sum'] == 3 * (2**30)
    assert not np.isinf(stats_overflow[1]['sum'])


def test_statistics_edge_cases():
    """Test statistical functions with edge cases.
    
    This test should verify that:
    - Empty arrays are handled gracefully
    - Single-value arrays produce appropriate statistics
    - Arrays with all identical values don't cause errors
    - Very large arrays are processed efficiently
    - Extreme value ranges are handled correctly
    """
    # Test 1: Empty arrays (all NaN)
    empty_data = np.full((10, 10), np.nan)
    
    # Coverage statistics
    cov_empty = calculate_coverage_statistics(empty_data)
    assert cov_empty['valid_pixels'] == 0
    assert cov_empty['nodata_pixels'] == 100
    assert cov_empty['quality_flag'] == 'insufficient'
    
    # Histogram analysis should raise error
    with pytest.raises(ValueError, match="No valid data points"):
        generate_histogram_analysis(empty_data)
    
    # Test 2: Single-value arrays
    single_value = np.array([[42.0]])
    
    # Coverage
    cov_single = calculate_coverage_statistics(single_value)
    assert cov_single['valid_pixels'] == 1
    assert cov_single['valid_percent'] == 100.0
    
    # Histogram
    hist_single = generate_histogram_analysis(single_value, bins=10)
    assert hist_single['mean'] == 42.0
    assert hist_single['std'] == 0.0
    assert hist_single['skewness'] == 0.0  # Undefined but returns 0
    
    # Zonal statistics
    zones_single = np.array([[1]])
    zonal_single = compute_zonal_statistics(single_value, zones_single)
    assert zonal_single[1]['mean'] == 42.0
    assert zonal_single[1]['std'] == 0.0
    assert zonal_single[1]['count'] == 1
    
    # Test 3: Arrays with all identical values
    constant_data = np.full((50, 50), 7.5)
    
    # Histogram
    hist_const = generate_histogram_analysis(constant_data, bins=20)
    assert hist_const['mean'] == 7.5
    assert hist_const['std'] == 0.0
    assert hist_const['min'] == 7.5
    assert hist_const['max'] == 7.5
    
    # All data in one bin
    assert np.max(hist_const['counts']) == constant_data.size
    assert np.sum(hist_const['counts'] > 0) == 1  # Only one non-zero bin
    
    # Temporal statistics with constant time series
    const_stack = np.full((10, 20, 20), 100.0)
    temp_const = calculate_temporal_statistics(const_stack)
    
    assert np.all(temp_const['temporal_std'] == 0.0)
    assert np.all(temp_const['coefficient_variation'] == 0.0)
    assert np.all(temp_const['temporal_range'] == 0.0)
    
    # Test 4: Very large arrays
    # Test memory efficiency without creating huge arrays
    # Simulate with smaller array but test algorithm complexity
    large_sim = np.random.rand(500, 500)
    
    start_time = datetime.now()
    # Run multiple statistics
    cov_large = calculate_coverage_statistics(large_sim)
    hist_large = generate_histogram_analysis(large_sim, bins=100)
    quality_large = generate_quality_metrics_summary(large_sim, comprehensive=False)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Should complete quickly
    assert elapsed < 3.0
    
    # Test 5: Extreme value ranges
    # Mix of very small and very large values
    extreme_data = np.array([1e-100, 1e-50, 0, 1e50, 1e100])
    
    # Should handle without overflow/underflow
    hist_extreme = generate_histogram_analysis(extreme_data, bins=50)
    
    assert not np.isnan(hist_extreme['mean'])
    assert not np.isinf(hist_extreme['mean'])
    assert len(hist_extreme['counts']) == 50
    
    # Test 6: Single pixel with multiple time steps
    single_pixel_time = np.array([[[1]], [[2]], [[3]], [[4]], [[5]]])
    
    temp_single_pixel = calculate_temporal_statistics(single_pixel_time)
    assert temp_single_pixel['temporal_mean'].shape == (1, 1)
    assert temp_single_pixel['temporal_mean'][0, 0] == 3.0
    assert temp_single_pixel['temporal_std'][0, 0] == np.std([1, 2, 3, 4, 5])
    
    # Test 7: Zero-sized arrays (edge case)
    # Some functions might encounter this internally
    zero_size = np.array([])
    
    # Most functions should handle gracefully or raise appropriate errors
    with pytest.raises(ValueError):
        generate_histogram_analysis(zero_size)
    
    # Test 8: Arrays with single non-NaN value
    mostly_nan = np.full((5, 5), np.nan)
    mostly_nan[2, 2] = 42.0
    
    cov_mostly_nan = calculate_coverage_statistics(mostly_nan)
    assert cov_mostly_nan['valid_pixels'] == 1
    assert cov_mostly_nan['valid_percent'] == 4.0  # 1/25
    assert cov_mostly_nan['quality_flag'] == 'insufficient'
    
    # Test 9: Infinity values
    inf_data = np.array([1, 2, np.inf, 4, 5])
    
    # Should handle infinity appropriately
    # Most statistical functions will propagate inf
    hist_inf = generate_histogram_analysis(inf_data)
    assert np.isinf(hist_inf['max'])
    assert np.isinf(hist_inf['mean'])
    
    # Test 10: Negative infinity
    neginf_data = np.array([1, 2, -np.inf, 4, 5])
    
    hist_neginf = generate_histogram_analysis(neginf_data)
    assert np.isinf(hist_neginf['min'])
    assert hist_neginf['min'] < 0


def test_statistics_performance():
    """Test performance of statistical functions with large datasets.
    
    This test should verify that:
    - Large arrays are processed efficiently
    - Memory usage is reasonable
    - Computation time scales appropriately
    - Parallel processing is utilized when available
    - Algorithmic complexity matches expectations
    """
    import time
    
    # Test 1: Coverage statistics performance
    sizes = [(100, 100), (500, 500), (1000, 1000)]
    times_coverage = []
    
    for size in sizes:
        data = np.random.rand(*size)
        
        start = time.time()
        result = calculate_coverage_statistics(data, compute_regional=False)
        elapsed = time.time() - start
        
        times_coverage.append(elapsed)
        
        # Should complete quickly even for large arrays
        if size == (1000, 1000):
            assert elapsed < 1.0  # 1M pixels in under 1 second
    
    # Verify scaling is reasonable (not quadratic)
    # Time should roughly scale with number of pixels
    time_ratio = times_coverage[2] / times_coverage[0]
    pixel_ratio = (sizes[2][0] * sizes[2][1]) / (sizes[0][0] * sizes[0][1])
    
    # Should be roughly linear (allow 2x for overhead)
    assert time_ratio < pixel_ratio * 2
    
    # Test 2: Histogram analysis performance
    large_data = np.random.normal(50, 10, (2000, 2000))
    
    start = time.time()
    hist_result = generate_histogram_analysis(large_data, bins=256)
    hist_time = time.time() - start
    
    assert hist_time < 2.0  # 4M pixels in under 2 seconds
    assert len(hist_result['counts']) == 256
    
    # Test 3: Zonal statistics with many zones
    data_zones = np.random.rand(500, 500)
    # Create many small zones
    zones = np.random.randint(1, 101, size=(500, 500))  # 100 zones
    
    start = time.time()
    zonal_result = compute_zonal_statistics(data_zones, zones)
    zonal_time = time.time() - start
    
    assert zonal_time < 3.0  # Should handle 100 zones efficiently
    assert len(zonal_result) <= 100
    
    # Test 4: Temporal statistics with large time series
    # 50 time steps of 200x200 images
    time_series = np.random.rand(50, 200, 200)
    
    start = time.time()
    temp_result = calculate_temporal_statistics(time_series, compute_trend=True)
    temp_time = time.time() - start
    
    assert temp_time < 5.0  # Trend analysis for 40k pixels
    assert temp_result['temporal_mean'].shape == (200, 200)
    
    # Test 5: Comparison metrics performance
    ref_large = np.random.rand(1000, 1000)
    comp_large = ref_large + np.random.normal(0, 0.1, (1000, 1000))
    
    start = time.time()
    comp_result = compute_comparison_metrics(ref_large, comp_large)
    comp_time = time.time() - start
    
    assert comp_time < 1.0  # 1M pixel comparisons
    
    # Test 6: Spatial autocorrelation (most computationally intensive)
    spatial_data = np.random.rand(100, 100)
    
    # Test Moran's I
    start = time.time()
    moran_result = analyze_spatial_autocorrelation(spatial_data, method="moran")
    moran_time = time.time() - start
    
    assert moran_time < 2.0  # Should use efficient algorithms
    
    # Test variogram (sampling-based)
    start = time.time()
    variogram_result = analyze_spatial_autocorrelation(spatial_data, 
                                                     method="variogram",
                                                     lag_distances=[1, 2, 5, 10])
    variogram_time = time.time() - start
    
    assert variogram_time < 3.0  # Uses sampling for efficiency
    
    # Test 7: Memory efficiency with quality metrics
    # Should not create unnecessary copies
    quality_data = np.random.rand(800, 800)
    
    start = time.time()
    quality_result = generate_quality_metrics_summary(quality_data, 
                                                    comprehensive=True)
    quality_time = time.time() - start
    
    assert quality_time < 5.0  # Comprehensive analysis
    
    # Test 8: Algorithmic complexity verification
    # Test that doubling data size doesn't quadruple time
    data_small = np.random.rand(200, 200)
    data_medium = np.random.rand(400, 400)
    
    # Coverage statistics should be O(n)
    start = time.time()
    cov_small = calculate_coverage_statistics(data_small)
    time_small = time.time() - start
    
    start = time.time()
    cov_medium = calculate_coverage_statistics(data_medium)
    time_medium = time.time() - start
    
    # Time should roughly quadruple (4x pixels), not more
    assert time_medium < time_small * 6  # Allow overhead
    
    # Test 9: Batch processing efficiency
    # Multiple operations on same data should reuse computations
    batch_data = np.random.rand(300, 300)
    
    start = time.time()
    # Run multiple analyses
    cov = calculate_coverage_statistics(batch_data)
    hist = generate_histogram_analysis(batch_data)
    quality = generate_quality_metrics_summary(batch_data)
    batch_time = time.time() - start
    
    # Should be efficient when run together
    assert batch_time < 3.0