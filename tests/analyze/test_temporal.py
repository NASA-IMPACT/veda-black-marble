"""Test functions in blackmarble.analyze.temporal module.

This module tests temporal compositing and time series analysis functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from datetime import datetime

# Import the module under test
from blackmarble.analyze.temporal import (
    create_temporal_composite,
    merge_scenes,
    select_by_date_range,
    composite_with_quality_weighting,
    compute_temporal_statistics,
    detect_change,
    fill_temporal_gaps,
    compute_phenology_metrics,
)


def test_create_temporal_composite():
    """Test temporal composite creation from time series data.
    
    This test should verify that:
    - Median compositing produces robust results
    - Different compositing methods (mean, percentile) work correctly
    - Minimum valid observations threshold is enforced
    - Quality masks exclude contaminated pixels
    - Weight arrays influence composite calculation
    - No-data values are properly handled and propagated
    - SciPy gap filling improves composite quality
    - Mathematical accuracy of compositing algorithms
    """
    # Create test data stack (5 time steps, 3x3 spatial)
    data_stack = np.array([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],      # t=0
        [[2, 3, 4], [5, 6, 7], [8, 9, 10]],     # t=1
        [[3, 4, 5], [6, 7, 8], [9, 10, 11]],    # t=2
        [[np.nan, 5, 6], [7, 8, 9], [10, 11, 12]], # t=3 with gap
        [[5, 6, 7], [8, 9, 10], [11, 12, 13]]   # t=4
    ])
    
    # Test dates
    dates = [
        datetime(2023, 1, 1),
        datetime(2023, 1, 15),
        datetime(2023, 2, 1),
        datetime(2023, 2, 15),
        datetime(2023, 3, 1)
    ]
    
    # Test median compositing (default)
    composite, valid_count, stats = create_temporal_composite(
        data_stack=data_stack,
        dates=dates,
        method="median"
    )
    
    # Verify median calculation accuracy
    # For pixel [0,0]: [1,2,3,nan,5] -> median of [1,2,3,5] = 2.5
    expected_median_00 = np.median([1, 2, 3, 5])
    assert np.isclose(composite[0, 0], expected_median_00), "Median composite incorrect"
    
    # Verify shape preservation
    assert composite.shape == (3, 3), "Composite shape should match spatial dimensions"
    
    # Verify valid count is correct
    assert valid_count[0, 0] == 4, "Valid count should exclude NaN values"
    assert valid_count[1, 1] == 5, "Valid count should be 5 for pixels without NaN"
    
    # Test minimum valid observations threshold
    composite_min3, _, _ = create_temporal_composite(
        data_stack=data_stack,
        min_valid_observations=3
    )
    assert not np.isnan(composite_min3[0, 0]), "Pixel with 4 valid obs should be included"
    
    # Test with higher threshold
    composite_min5, _, _ = create_temporal_composite(
        data_stack=data_stack,
        min_valid_observations=5
    )
    assert np.isnan(composite_min5[0, 0]), "Pixel with 4 valid obs should be excluded when min=5"
    
    # Test quality masks
    quality_masks = np.array([
        [[True, True, True], [True, True, True], [True, True, True]],
        [[True, True, True], [True, True, True], [True, True, True]],
        [[False, False, False], [False, False, False], [False, False, False]],  # Mask out t=2
        [[True, True, True], [True, True, True], [True, True, True]],
        [[True, True, True], [True, True, True], [True, True, True]]
    ])
    
    composite_masked, valid_count_masked, _ = create_temporal_composite(
        data_stack=data_stack,
        quality_masks=quality_masks
    )
    
    # Verify quality masks exclude contaminated pixels
    # For pixel [0,0]: [1,2,masked,nan,5] -> median of [1,2,5] = 2
    assert composite_masked[0, 0] == 2.0, "Quality mask should exclude masked observations"
    assert valid_count_masked[0, 0] == 3, "Valid count should reflect quality masking"
    
    # Test mean method
    composite_mean, _, _ = create_temporal_composite(
        data_stack=data_stack,
        method="mean"
    )
    expected_mean_11 = np.mean([5, 6, 7, 8, 9])  # No NaN at [1,1]
    assert np.isclose(composite_mean[1, 1], expected_mean_11), "Mean composite incorrect"
    
    # Test percentile method
    composite_p75, _, _ = create_temporal_composite(
        data_stack=data_stack,
        method="percentile",
        percentile=75
    )
    expected_p75_11 = np.percentile([5, 6, 7, 8, 9], 75)
    assert np.isclose(composite_p75[1, 1], expected_p75_11), "75th percentile incorrect"
    
    # Test no-data handling
    nodata_value = -9999
    data_with_nodata = data_stack.copy()
    data_with_nodata[3, 0, 0] = nodata_value  # Replace NaN with custom nodata
    
    composite_nodata, _, _ = create_temporal_composite(
        data_stack=data_with_nodata,
        nodata_value=nodata_value
    )
    # Should still compute median correctly excluding nodata
    assert np.isclose(composite_nodata[0, 0], expected_median_00), "Custom nodata handling failed"
    
    # Test SciPy gap filling
    # Create data with more gaps
    gappy_data = np.full((5, 3, 3), np.nan)
    gappy_data[:, 1, 1] = [1, 2, 3, 4, 5]  # Only center pixel has data
    
    composite_filled, _, stats_filled = create_temporal_composite(
        data_stack=gappy_data,
        use_scipy_fill_gaps=True
    )
    
    # Gap filling should propagate center value to neighbors
    assert not np.isnan(composite_filled[0, 1]), "Gap filling should fill adjacent pixels"
    assert not np.isnan(composite_filled[1, 0]), "Gap filling should fill adjacent pixels"
    
    # Verify coverage statistics
    assert stats['total_pixels'] == 9, "Total pixels should be 3x3 = 9"
    assert stats['valid_pixels'] == 9, "All pixels should be valid with default threshold"
    assert stats['coverage_percent'] == 100.0, "Coverage should be 100% with threshold=1"
    assert stats['temporal_range'] == (dates[0], dates[-1]), "Temporal range should match input dates"


def test_create_temporal_composite_methods():
    """Test different temporal compositing methods.
    
    This test should verify that:
    - Median method is robust to outliers
    - Mean method provides smooth composites
    - Percentile method allows custom quantile selection
    - Min/max methods capture extreme values
    - Custom compositing functions can be implemented
    """
    # Create test data with outliers
    data_stack = np.array([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[2, 3, 4], [5, 6, 7], [8, 9, 10]],
        [[100, 3, 4], [5, 6, 7], [8, 9, 10]],  # Outlier at [0,0]
        [[2, 4, 5], [6, 7, 8], [9, 10, 11]],
        [[3, 4, 5], [6, 7, 8], [9, 10, 11]]
    ])
    
    # Test median robustness to outliers
    composite_median, _, _ = create_temporal_composite(
        data_stack=data_stack,
        method="median"
    )
    # Median of [1, 2, 100, 2, 3] = 2
    assert composite_median[0, 0] == 2.0, "Median should be robust to outliers"
    
    # Test mean sensitivity to outliers
    composite_mean, _, _ = create_temporal_composite(
        data_stack=data_stack,
        method="mean"
    )
    # Mean of [1, 2, 100, 2, 3] = 21.6
    expected_mean = np.mean([1, 2, 100, 2, 3])
    assert np.isclose(composite_mean[0, 0], expected_mean), "Mean should include outliers"
    assert composite_mean[0, 0] > composite_median[0, 0], "Mean should be affected by outliers"
    
    # Test min method
    composite_min, _, _ = create_temporal_composite(
        data_stack=data_stack,
        method="min"
    )
    assert composite_min[0, 0] == 1.0, "Min should capture lowest value"
    assert composite_min[1, 1] == 5.0, "Min method incorrect"
    
    # Test max method
    composite_max, _, _ = create_temporal_composite(
        data_stack=data_stack,
        method="max"
    )
    assert composite_max[0, 0] == 100.0, "Max should capture highest value (outlier)"
    assert composite_max[1, 1] == 7.0, "Max method incorrect"
    
    # Test percentile methods
    composite_p25, _, _ = create_temporal_composite(
        data_stack=data_stack,
        method="percentile",
        percentile=25
    )
    composite_p50, _, _ = create_temporal_composite(
        data_stack=data_stack,
        method="percentile",
        percentile=50
    )
    composite_p90, _, _ = create_temporal_composite(
        data_stack=data_stack,
        method="percentile",
        percentile=90
    )
    
    # Verify percentile ordering
    assert composite_p25[1, 1] <= composite_p50[1, 1], "25th percentile should be <= 50th"
    assert composite_p50[1, 1] <= composite_p90[1, 1], "50th percentile should be <= 90th"
    assert composite_p50[1, 1] == composite_median[1, 1], "50th percentile should equal median"
    
    # Test quality-weighted method
    weights = np.array([
        [[0.8, 0.9, 1.0], [0.7, 0.8, 0.9], [0.6, 0.7, 0.8]],
        [[0.9, 1.0, 0.9], [0.8, 0.9, 1.0], [0.7, 0.8, 0.9]],
        [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],  # Low weight for outlier time
        [[1.0, 0.9, 0.8], [0.9, 0.8, 0.7], [0.8, 0.7, 0.6]],
        [[0.9, 0.8, 0.7], [0.8, 0.7, 0.6], [0.7, 0.6, 0.5]]
    ])
    
    composite_weighted, _, _ = create_temporal_composite(
        data_stack=data_stack,
        method="quality_weighted",
        weights=weights
    )
    
    # Weighted mean should downweight the outlier
    assert composite_weighted[0, 0] < composite_mean[0, 0], "Quality weighting should reduce outlier influence"
    assert composite_weighted[0, 0] < 10, "Quality-weighted result should be much less than mean with outlier"


def test_create_temporal_composite_quality_masks():
    """Test temporal compositing with quality masks.
    
    This test should verify that:
    - Quality masks exclude contaminated observations
    - Multiple quality masks are combined appropriately
    - Mask application doesn't bias composite results
    - Edge cases (all pixels masked) are handled
    - Quality-based pixel selection improves results
    """
    # Create test data with known patterns
    data_stack = np.array([
        [[10, 20, 30], [40, 50, 60], [70, 80, 90]],    # Good quality
        [[11, 21, 31], [41, 51, 61], [71, 81, 91]],    # Good quality
        [[100, 200, 300], [400, 500, 600], [700, 800, 900]], # Contaminated (clouds)
        [[12, 22, 32], [42, 52, 62], [72, 82, 92]],    # Good quality
        [[13, 23, 33], [43, 53, 63], [73, 83, 93]]     # Good quality
    ], dtype=np.float32)
    
    # Create quality mask that excludes contaminated observation
    quality_masks = np.array([
        [[True, True, True], [True, True, True], [True, True, True]],
        [[True, True, True], [True, True, True], [True, True, True]],
        [[False, False, False], [False, False, False], [False, False, False]], # Cloud contamination
        [[True, True, True], [True, True, True], [True, True, True]],
        [[True, True, True], [True, True, True], [True, True, True]]
    ])
    
    # Test quality mask exclusion
    composite_masked, valid_count, _ = create_temporal_composite(
        data_stack=data_stack,
        quality_masks=quality_masks,
        method="median"
    )
    
    # Verify contaminated observations are excluded
    # Median of [10, 11, 12, 13] (excluding 100) = 11.5
    expected_median = np.median([10, 11, 12, 13])
    assert np.isclose(composite_masked[0, 0], expected_median), "Quality mask should exclude contaminated data"
    assert valid_count[0, 0] == 4, "Valid count should exclude masked observation"
    
    # Test without quality masks for comparison
    composite_unmasked, _, _ = create_temporal_composite(
        data_stack=data_stack,
        method="median"
    )
    # Median of [10, 11, 100, 12, 13] = 12
    assert composite_unmasked[0, 0] != composite_masked[0, 0], "Quality mask should change results"
    assert composite_unmasked[0, 0] > composite_masked[0, 0], "Contaminated data should bias unmasked result"
    
    # Test partial masking (some pixels masked, others not)
    partial_masks = np.array([
        [[True, True, True], [True, True, True], [True, True, True]],
        [[True, False, True], [False, True, False], [True, False, True]], # Checkerboard pattern
        [[True, True, True], [True, True, True], [True, True, True]],
        [[False, True, False], [True, False, True], [False, True, False]], # Inverse checkerboard
        [[True, True, True], [True, True, True], [True, True, True]]
    ])
    
    composite_partial, valid_partial, _ = create_temporal_composite(
        data_stack=data_stack,
        quality_masks=partial_masks,
        method="mean"
    )
    
    # Check different valid counts for different pixels
    # Time 0: all True (all pixels valid)
    # Time 1: checkerboard - [0,0]=True, [0,1]=False, [1,1]=True
    # Time 2: all True
    # Time 3: inverse checkerboard - [0,0]=False, [0,1]=True, [1,1]=False
    # Time 4: all True
    # So [0,0] has times 0,1,2,4 valid = 4 observations
    # [0,1] has times 0,2,3,4 valid = 4 observations
    # [1,1] has times 0,1,2,4 valid = 4 observations
    assert valid_partial[0, 0] == 4, "Corner pixel should have 4 valid observations"
    assert valid_partial[0, 1] == 4, "Edge pixel should have 4 valid observations"
    assert valid_partial[1, 1] == 4, "Center pixel should have 4 valid observations"
    
    # Test edge case: all pixels masked for one location
    all_masked = np.ones((5, 3, 3), dtype=bool)
    all_masked[:, 2, 2] = False  # Mask all observations for bottom-right pixel
    
    composite_edge, valid_edge, stats_edge = create_temporal_composite(
        data_stack=data_stack,
        quality_masks=all_masked,
        min_valid_observations=1
    )
    
    # Verify pixel with all masked observations is NaN
    assert np.isnan(composite_edge[2, 2]), "Pixel with all masked obs should be NaN"
    assert valid_edge[2, 2] == 0, "Valid count should be 0 for fully masked pixel"
    assert stats_edge['valid_pixels'] == 8, "Only 8 pixels should be valid (9 - 1 masked)"
    
    # Test quality improvement by masking outliers
    outlier_data = data_stack.copy()
    outlier_data[2, :, :] = 1000  # Add extreme outliers
    
    composite_with_outliers, _, _ = create_temporal_composite(
        data_stack=outlier_data,
        method="mean"
    )
    
    composite_outliers_masked, _, _ = create_temporal_composite(
        data_stack=outlier_data,
        quality_masks=quality_masks,  # Masks exclude the outlier time
        method="mean"
    )
    
    # Verify quality masking improves results
    assert np.all(composite_outliers_masked < composite_with_outliers), "Masking outliers should reduce mean values"
    assert np.all(composite_outliers_masked < 100), "Masked composite should have reasonable values"


def test_create_temporal_composite_scipy_gaps():
    """Test SciPy-based gap filling in temporal composites.
    
    This test should verify that:
    - Gap filling improves composite completeness
    - Interpolation preserves spatial patterns
    - Filling doesn't introduce unrealistic values
    - Performance impact is acceptable
    - Gap-filled regions are properly flagged
    """
    # Create data with systematic gaps
    data_stack = np.array([
        [[1, np.nan, 3], [np.nan, 5, np.nan], [7, np.nan, 9]],
        [[2, np.nan, 4], [np.nan, 6, np.nan], [8, np.nan, 10]],
        [[3, np.nan, 5], [np.nan, 7, np.nan], [9, np.nan, 11]],
        [[4, np.nan, 6], [np.nan, 8, np.nan], [10, np.nan, 12]],
        [[5, np.nan, 7], [np.nan, 9, np.nan], [11, np.nan, 13]]
    ])
    
    # Test without gap filling
    composite_no_fill, _, stats_no_fill = create_temporal_composite(
        data_stack=data_stack,
        use_scipy_fill_gaps=False
    )
    
    # Test with gap filling
    composite_filled, _, stats_filled = create_temporal_composite(
        data_stack=data_stack,
        use_scipy_fill_gaps=True
    )
    
    # Verify gap filling improves completeness
    nan_count_no_fill = np.sum(np.isnan(composite_no_fill))
    nan_count_filled = np.sum(np.isnan(composite_filled))
    assert nan_count_filled < nan_count_no_fill, "Gap filling should reduce NaN count"
    
    # Verify filled values are reasonable (should be similar to neighbors)
    # Check center pixel which had all NaN values
    assert not np.isnan(composite_filled[1, 1]), "Center pixel should be filled"
    # Filled value should be between neighboring values
    neighbor_values = [
        composite_filled[0, 1], composite_filled[2, 1],
        composite_filled[1, 0], composite_filled[1, 2]
    ]
    valid_neighbors = [v for v in neighbor_values if not np.isnan(v)]
    if valid_neighbors:
        assert (min(valid_neighbors) <= composite_filled[1, 1] <= max(valid_neighbors)), \
            "Filled values should be within neighbor range"
    
    # Test gap filling with partial data
    partial_data = np.full((5, 5, 5), np.nan)
    # Add data in corners only
    partial_data[:, 0, 0] = [1, 2, 3, 4, 5]
    partial_data[:, 0, -1] = [2, 3, 4, 5, 6]
    partial_data[:, -1, 0] = [3, 4, 5, 6, 7]
    partial_data[:, -1, -1] = [4, 5, 6, 7, 8]
    
    composite_partial_filled, _, _ = create_temporal_composite(
        data_stack=partial_data,
        use_scipy_fill_gaps=True
    )
    
    # Verify spatial patterns are preserved
    # Filled values should show gradients between corners
    assert not np.isnan(composite_partial_filled[2, 2]), "Center should be filled"
    assert not np.isnan(composite_partial_filled[1, 1]), "Interior points should be filled"
    
    # Test that completely empty data isn't filled unrealistically
    empty_data = np.full((3, 3, 3), np.nan)
    composite_empty, _, _ = create_temporal_composite(
        data_stack=empty_data,
        use_scipy_fill_gaps=True
    )
    assert np.all(np.isnan(composite_empty)), "Completely empty data should remain NaN"
    
    # Test multi-band gap filling
    multiband_data = np.zeros((5, 2, 3, 3))  # 2 bands
    multiband_data[:, 0, :, :] = data_stack  # First band has gaps
    multiband_data[:, 1, :, :] = data_stack * 2  # Second band has same gap pattern
    
    composite_multiband, _, _ = create_temporal_composite(
        data_stack=multiband_data,
        use_scipy_fill_gaps=True
    )
    
    # Verify both bands are filled
    assert composite_multiband.shape == (2, 3, 3), "Multi-band shape preserved"
    assert not np.isnan(composite_multiband[0, 1, 1]), "First band center filled"
    assert not np.isnan(composite_multiband[1, 1, 1]), "Second band center filled"
    # Second band should be ~2x first band
    assert np.isclose(composite_multiband[1, 0, 0], composite_multiband[0, 0, 0] * 2, rtol=0.1), \
        "Band relationship preserved"


def test_merge_scenes():
    """Test merging of multiple satellite scenes into seamless mosaics.
    
    This test should verify that:
    - Scene boundaries are properly handled
    - Transform information is used correctly
    - Bounds parameter crops output appropriately
    - Feather blending creates smooth transitions
    - Priority order influences overlap resolution
    - No-data values are handled consistently
    - SciPy feathering produces professional results
    """
    from rasterio.transform import Affine
    
    # Create test scenes with overlap
    scene1 = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    
    scene2 = np.array([
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18]
    ])
    
    # Define transforms (pixel size = 1, overlapping by 1 pixel)
    transform1 = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 3.0)  # Upper left at (0, 3)
    transform2 = Affine(1.0, 0.0, 2.0, 0.0, -1.0, 3.0)  # Upper left at (2, 3), 1 pixel overlap
    
    # Define bounds (xmin, ymin, xmax, ymax)
    bounds1 = (0.0, 0.0, 3.0, 3.0)
    bounds2 = (2.0, 0.0, 5.0, 3.0)
    
    # Test basic merging
    merged, merged_transform, merged_bounds = merge_scenes(
        scenes=[scene1, scene2],
        transforms=[transform1, transform2],
        bounds=[bounds1, bounds2],
        blend_method="max"
    )
    
    # Verify merged dimensions
    assert merged.shape == (3, 5), "Merged scene should be 3x5"
    assert merged_bounds == (0.0, 0.0, 5.0, 3.0), "Merged bounds incorrect"
    
    # Verify transform
    assert merged_transform.a == 1.0, "Pixel width should be preserved"
    assert merged_transform.e == -1.0, "Pixel height should be preserved"
    assert merged_transform.c == 0.0, "Transform origin X incorrect"
    assert merged_transform.f == 3.0, "Transform origin Y incorrect"
    
    # Verify max blending in overlap region
    # Overlap is at column index 2 in merged array
    assert merged[0, 2] == max(scene1[0, 2], scene2[0, 0]), "Max blending failed"
    assert merged[1, 2] == max(scene1[1, 2], scene2[1, 0]), "Max blending failed"
    
    # Test feather blending
    merged_feather, _, _ = merge_scenes(
        scenes=[scene1, scene2],
        transforms=[transform1, transform2],
        bounds=[bounds1, bounds2],
        blend_method="feather",
        feather_distance=2
    )
    
    # Verify feather creates smooth transition
    # Values in overlap should be between scene values
    overlap_col = merged_feather[:, 2]
    for i in range(3):
        assert scene1[i, 2] <= overlap_col[i] <= scene2[i, 0] or \
               scene2[i, 0] <= overlap_col[i] <= scene1[i, 2], \
               "Feather blend should create intermediate values"
    
    # Test priority order
    merged_priority, _, _ = merge_scenes(
        scenes=[scene1, scene2],
        transforms=[transform1, transform2],
        bounds=[bounds1, bounds2],
        blend_method="feather",
        priority_order=[1, 0]  # Scene2 has higher priority
    )
    
    # With priority, scene2 should dominate in overlap
    # Priority affects the order of processing, which can affect feathering weights
    # The result may be similar but not identical due to feathering algorithm
    # Check that the priority-ordered merge is closer to scene2 values
    diff_to_scene2 = np.abs(merged_priority[:, 2] - scene2[:, 0])
    diff_to_scene1 = np.abs(merged_priority[:, 2] - scene1[:, 2])
    # With scene2 having higher priority, it should generally be closer to scene2
    assert np.mean(diff_to_scene2) <= np.mean(diff_to_scene1), "Priority should favor scene2"
    
    # Test nodata handling
    scene1_nodata = scene1.copy().astype(float)
    scene1_nodata[1, 1] = np.nan
    
    merged_nodata, _, _ = merge_scenes(
        scenes=[scene1_nodata, scene2],
        transforms=[transform1, transform2],
        bounds=[bounds1, bounds2],
        blend_method="mean",
        nodata=np.nan
    )
    
    # Verify nodata is preserved
    assert np.isnan(merged_nodata[1, 1]), "NoData should be preserved"
    
    # Test scipy feathering
    merged_scipy, _, _ = merge_scenes(
        scenes=[scene1, scene2],
        transforms=[transform1, transform2],
        bounds=[bounds1, bounds2],
        blend_method="feather",
        feather_distance=3,
        use_scipy_feather=True
    )
    
    # Scipy feathering should produce smoother results
    # Check that transition is gradual
    transition_region = merged_scipy[:, 1:4]  # Columns around overlap
    for row in transition_region:
        diffs = np.diff(row)
        assert np.all(np.abs(diffs) < 10), "Scipy feather should create smooth gradients"
    
    # Test multi-band scenes
    scene1_3band = np.stack([scene1, scene1 * 2, scene1 * 3])
    scene2_3band = np.stack([scene2, scene2 * 2, scene2 * 3])
    
    merged_multiband, _, _ = merge_scenes(
        scenes=[scene1_3band, scene2_3band],
        transforms=[transform1, transform2],
        bounds=[bounds1, bounds2],
        blend_method="mean"
    )
    
    assert merged_multiband.shape == (3, 3, 5), "Multi-band merge shape incorrect"
    # Verify band relationships preserved
    assert np.allclose(merged_multiband[1], merged_multiband[0] * 2, rtol=0.01), "Band 2 relationship"
    assert np.allclose(merged_multiband[2], merged_multiband[0] * 3, rtol=0.01), "Band 3 relationship"


def test_merge_scenes_feather_blending():
    """Test feather blending in scene merging.
    
    This test should verify that:
    - Feather distance controls transition zone width
    - Blending eliminates visible seams
    - Edge effects are minimized
    - Radiometric consistency is maintained
    - Performance scales with feather distance
    """
    from rasterio.transform import Affine
    
    # Create scenes with sharp contrast for testing blending
    scene1 = np.full((10, 10), 100.0)  # Bright scene
    scene2 = np.full((10, 10), 20.0)   # Dark scene
    
    # Create overlapping configuration (5 pixel overlap)
    transform1 = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0)
    transform2 = Affine(1.0, 0.0, 5.0, 0.0, -1.0, 10.0)
    bounds1 = (0.0, 0.0, 10.0, 10.0)
    bounds2 = (5.0, 0.0, 15.0, 10.0)
    
    # Test different feather distances
    feather_distances = [1, 3, 5]
    transition_widths = []
    
    for distance in feather_distances:
        merged, _, _ = merge_scenes(
            scenes=[scene1, scene2],
            transforms=[transform1, transform2],
            bounds=[bounds1, bounds2],
            blend_method="feather",
            feather_distance=distance
        )
        
        # Measure transition zone width
        # Look at middle row to analyze transition
        middle_row = merged[5, :]
        
        # Find where values start transitioning
        transition_start = None
        transition_end = None
        
        for i in range(len(middle_row) - 1):
            if transition_start is None and middle_row[i] != middle_row[i+1]:
                transition_start = i
            if transition_start is not None and middle_row[i] == middle_row[i+1]:
                transition_end = i
                break
        
        if transition_start is not None and transition_end is not None:
            width = transition_end - transition_start
            transition_widths.append(width)
            
            # Verify smooth transition (no jumps)
            transition_zone = middle_row[transition_start:transition_end]
            diffs = np.diff(transition_zone)
            assert np.all(diffs <= 0), f"Transition should be monotonic for distance={distance}"
    
    # Verify feather distance affects transition width
    assert transition_widths[0] < transition_widths[1] < transition_widths[2], \
        "Larger feather distance should create wider transitions"
    
    # Test seam elimination
    # Without feathering (using max blend)
    merged_no_feather, _, _ = merge_scenes(
        scenes=[scene1, scene2],
        transforms=[transform1, transform2],
        bounds=[bounds1, bounds2],
        blend_method="max"
    )
    
    # With feathering
    merged_feather, _, _ = merge_scenes(
        scenes=[scene1, scene2],
        transforms=[transform1, transform2],
        bounds=[bounds1, bounds2],
        blend_method="feather",
        feather_distance=5
    )
    
    # Calculate gradient magnitude to detect seams
    gradient_no_feather = np.abs(np.diff(merged_no_feather[5, :]))
    gradient_feather = np.abs(np.diff(merged_feather[5, :]))
    
    # Feathering should reduce maximum gradient (eliminate sharp seam)
    assert np.max(gradient_feather) < np.max(gradient_no_feather), \
        "Feathering should reduce gradient magnitude at seams"
    
    # Test radiometric consistency
    # Total energy should be preserved (approximately)
    total_energy_scenes = np.sum(scene1) + np.sum(scene2)
    total_energy_merged = np.sum(merged_feather)
    
    # Account for overlap region being counted twice
    overlap_area = 5 * 10  # 5 columns × 10 rows
    expected_total = total_energy_scenes - overlap_area * np.mean([100, 20])
    
    assert np.abs(total_energy_merged - expected_total) / expected_total < 0.1, \
        "Radiometric consistency should be maintained"


def test_merge_scenes_scipy_feather():
    """Test SciPy-based feather blending implementation.
    
    This test should verify that:
    - SciPy feathering produces superior results
    - Distance transform creates smooth weight gradients
    - Edge artifacts are eliminated
    - Mathematical accuracy of blending algorithms
    - Performance comparison with standard feathering
    """
    from rasterio.transform import Affine
    import time
    
    # Create test pattern with features to preserve
    x = np.linspace(0, 2*np.pi, 20)
    y = np.linspace(0, 2*np.pi, 20)
    X, Y = np.meshgrid(x, y)
    scene1 = 50 + 30 * np.sin(X) * np.cos(Y)  # Sinusoidal pattern
    scene2 = 50 + 30 * np.cos(X) * np.sin(Y)  # Different sinusoidal pattern
    
    # Overlapping configuration
    transform1 = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 20.0)
    transform2 = Affine(1.0, 0.0, 10.0, 0.0, -1.0, 20.0)
    bounds1 = (0.0, 0.0, 20.0, 20.0)
    bounds2 = (10.0, 0.0, 30.0, 20.0)
    
    # Test standard feathering
    start_time = time.time()
    merged_standard, _, _ = merge_scenes(
        scenes=[scene1, scene2],
        transforms=[transform1, transform2],
        bounds=[bounds1, bounds2],
        blend_method="feather",
        feather_distance=5,
        use_scipy_feather=False
    )
    standard_time = time.time() - start_time
    
    # Test scipy feathering
    start_time = time.time()
    merged_scipy, _, _ = merge_scenes(
        scenes=[scene1, scene2],
        transforms=[transform1, transform2],
        bounds=[bounds1, bounds2],
        blend_method="feather",
        feather_distance=5,
        use_scipy_feather=True
    )
    scipy_time = time.time() - start_time
    
    # Verify scipy produces smoother results
    # Calculate second derivative (curvature) to measure smoothness
    def calculate_smoothness(array):
        # Calculate Laplacian (second derivative)
        dy, dx = np.gradient(array)
        dyy, dyx = np.gradient(dy)
        dxy, dxx = np.gradient(dx)
        laplacian = np.abs(dxx) + np.abs(dyy)
        return np.mean(laplacian)
    
    smoothness_standard = calculate_smoothness(merged_standard)
    smoothness_scipy = calculate_smoothness(merged_scipy)
    
    # SciPy should produce smoother results (lower curvature)
    assert smoothness_scipy < smoothness_standard, \
        "SciPy feathering should produce smoother blending"
    
    # Test weight gradient quality
    # Extract transition region
    transition_region_scipy = merged_scipy[:, 8:12]
    transition_region_standard = merged_standard[:, 8:12]
    
    # Calculate gradient consistency
    grad_scipy = np.diff(transition_region_scipy, axis=1)
    grad_standard = np.diff(transition_region_standard, axis=1)
    
    # Variance of gradients (lower = more consistent)
    grad_var_scipy = np.var(grad_scipy)
    grad_var_standard = np.var(grad_standard)
    
    assert grad_var_scipy < grad_var_standard, \
        "SciPy should produce more consistent gradients"
    
    # Test edge artifact elimination
    # Check edges of overlap region
    edge_diff_scipy = np.abs(merged_scipy[:, 10] - merged_scipy[:, 9])
    edge_diff_standard = np.abs(merged_standard[:, 10] - merged_standard[:, 9])
    
    assert np.max(edge_diff_scipy) < np.max(edge_diff_standard), \
        "SciPy should reduce edge artifacts"
    
    # Verify mathematical accuracy
    # In overlap region, should be weighted average
    overlap_scipy = merged_scipy[:, 10:20]
    overlap_standard = merged_standard[:, 10:20]
    
    # Both should be bounded by input values
    min_val = min(np.min(scene1), np.min(scene2))
    max_val = max(np.max(scene1), np.max(scene2))
    
    assert np.all((overlap_scipy >= min_val) & (overlap_scipy <= max_val)), \
        "SciPy blend should stay within input range"
    assert np.all((overlap_standard >= min_val) & (overlap_standard <= max_val)), \
        "Standard blend should stay within input range"
    
    # Performance should be reasonable (scipy might be slightly slower but not excessive)
    assert scipy_time < standard_time * 10, \
        "SciPy performance should be within reasonable bounds"


def test_merge_scenes_priority():
    """Test priority-based scene merging.
    
    This test should verify that:
    - Priority order determines overlap resolution
    - Higher priority scenes override lower priority
    - Priority weighting produces expected results
    - Edge cases (equal priority) are handled
    """
    pass


def test_select_by_date_range():
    """Test scene selection based on date and quality criteria.
    
    This test should verify that:
    - Date range filtering works correctly
    - Target date proximity is properly calculated
    - Season filtering selects appropriate time periods
    - Cloud threshold excludes poor quality scenes
    - Min/max scene constraints are enforced
    - Quality scores influence scene selection
    - Selection criteria can be combined effectively
    """
    # Create test scene dates throughout the year
    scene_dates = [
        datetime(2023, 1, 15),   # Winter
        datetime(2023, 3, 20),   # Spring  
        datetime(2023, 5, 10),   # Spring
        datetime(2023, 7, 4),    # Summer
        datetime(2023, 8, 15),   # Summer
        datetime(2023, 10, 1),   # Fall
        datetime(2023, 11, 25),  # Fall
        datetime(2023, 12, 20)   # Winter
    ]
    
    # Create metadata with cloud cover
    scene_metadata = [
        {'cloud_cover': 10.0, 'solar_zenith': 45.0},
        {'cloud_cover': 25.0, 'solar_zenith': 40.0},
        {'cloud_cover': 5.0, 'solar_zenith': 35.0},
        {'cloud_cover': 60.0, 'solar_zenith': 30.0},  # High cloud
        {'cloud_cover': 15.0, 'solar_zenith': 32.0},
        {'cloud_cover': 20.0, 'solar_zenith': 50.0},
        {'cloud_cover': 80.0, 'solar_zenith': 55.0},  # High cloud
        {'cloud_cover': 12.0, 'solar_zenith': 60.0}
    ]
    
    # Create quality scores
    quality_scores = [0.8, 0.6, 0.9, 0.3, 0.7, 0.75, 0.2, 0.85]
    
    # Test date range filtering
    selected = select_by_date_range(
        scene_dates=scene_dates,
        date_range=(datetime(2023, 3, 1), datetime(2023, 8, 31))
    )
    
    # March 1 - Aug 31 should include: Mar 20, May 10, Jul 4, Aug 15 (indices 1, 2, 3, 4)
    assert len(selected) == 4, "Should select 4 scenes in date range"
    assert set(selected) == {1, 2, 3, 4}, "Should select spring/summer scenes"
    assert 0 not in selected and 7 not in selected, "Winter scenes should be excluded"
    
    # Test target date proximity
    target = datetime(2023, 7, 15)
    selected_proximity = select_by_date_range(
        scene_dates=scene_dates,
        target_date=target,
        max_days_from_target=30
    )
    
    # Should select scenes 3 (July 4) and 4 (Aug 15)
    assert set(selected_proximity) == {3, 4}, "Should select scenes within 30 days of target"
    
    # Test seasonal filtering
    selected_summer = select_by_date_range(
        scene_dates=scene_dates,
        seasons=["summer"]
    )
    
    assert set(selected_summer) == {3, 4}, "Should select only summer scenes"
    
    # Test cloud threshold
    selected_clear = select_by_date_range(
        scene_dates=scene_dates,
        scene_metadata=scene_metadata,
        cloud_threshold=30.0
    )
    
    # Should exclude scenes 3 (60%) and 6 (80%) due to clouds
    assert 3 not in selected_clear and 6 not in selected_clear, "High cloud scenes excluded"
    assert len(selected_clear) == 6, "Should have 6 scenes below cloud threshold"
    
    # Test min/max scene constraints
    selected_limited = select_by_date_range(
        scene_dates=scene_dates,
        scene_metadata=scene_metadata,
        quality_scores=quality_scores,
        max_scenes=3
    )
    
    assert len(selected_limited) == 3, "Should limit to 3 scenes"
    # Should select highest quality scenes: indices 2 (0.9), 7 (0.85), 0 (0.8)
    assert selected_limited[0] == 2, "Highest quality scene should be first"
    
    # Test minimum scenes requirement
    with pytest.raises(ValueError, match="Only.*scenes pass filters"):
        select_by_date_range(
            scene_dates=scene_dates,
            seasons=["spring"],
            cloud_threshold=10.0,
            scene_metadata=scene_metadata,
            min_scenes=5  # Only 1 spring scene has <10% clouds
        )
    
    # Test combined criteria
    selected_combined = select_by_date_range(
        scene_dates=scene_dates,
        scene_metadata=scene_metadata,
        quality_scores=quality_scores,
        target_date=datetime(2023, 6, 1),
        max_days_from_target=60,
        cloud_threshold=30.0,
        max_scenes=2
    )
    
    # Should select best scenes near June 1 with low clouds
    assert len(selected_combined) == 2, "Should select 2 scenes"
    assert 2 in selected_combined, "High quality spring scene should be selected"
    
    # Verify quality ordering
    if len(selected_combined) == 2:
        idx0, idx1 = selected_combined
        # First selected should have higher composite score
        assert quality_scores[idx0] >= quality_scores[idx1] or \
               scene_metadata[idx0]['cloud_cover'] < scene_metadata[idx1]['cloud_cover'], \
               "Scenes should be ordered by quality"


def test_select_by_date_range_seasons():
    """Test seasonal scene selection.
    
    This test should verify that:
    - Season definitions work across year boundaries
    - Multi-year seasonal selection is consistent
    - Different hemisphere seasons are handled
    - Custom season definitions can be used
    """
    pass


def test_select_by_date_range_quality():
    """Test quality-based scene selection.
    
    This test should verify that:
    - Quality scores influence selection decisions
    - Cloud threshold filtering works correctly
    - Quality weighting balances date and quality criteria
    - Selection produces optimal scene combinations
    """
    pass


def test_composite_with_quality_weighting():
    """Test quality-weighted compositing for optimal pixel selection.
    
    This test should verify that:
    - Quality scores properly weight pixel contributions
    - Weighted mean produces quality-optimized composites
    - Quality thresholds exclude poor pixels
    - Different weighting methods produce distinct results
    - Mathematical accuracy of quality weighting
    - No-data handling preserves quality information
    """
    # Create test data
    data_stack = np.array([
        [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
        [[15, 25, 35], [45, 55, 65], [75, 85, 95]],
        [[12, 22, 32], [42, 52, 62], [72, 82, 92]],
        [[18, 28, 38], [48, 58, 68], [78, 88, 98]],
        [[14, 24, 34], [44, 54, 64], [74, 84, 94]]
    ], dtype=np.float32)
    
    # Scene-level quality scores
    scene_quality = np.array([0.9, 0.3, 0.8, 0.6, 0.7])
    
    # Test weighted mean with scene-level quality
    composite_weighted, quality_map = composite_with_quality_weighting(
        data_stack=data_stack,
        quality_scores=scene_quality,
        method="weighted_mean"
    )
    
    # Verify shape
    assert composite_weighted.shape == (3, 3), "Composite shape should match spatial dimensions"
    assert quality_map.shape == (3, 3), "Quality map shape should match spatial dimensions"
    
    # Verify weighted mean calculation
    # For pixel [0,0]: values=[10,15,12,18,14], weights=[0.9,0.3,0.8,0.6,0.7]
    expected_00 = np.average([10, 15, 12, 18, 14], weights=scene_quality)
    assert np.isclose(composite_weighted[0, 0], expected_00), "Weighted mean incorrect"
    
    # Verify quality weighting affects results
    # High quality observation (index 0) should have more influence
    unweighted_mean = np.mean([10, 15, 12, 18, 14])
    assert abs(composite_weighted[0, 0] - 10) < abs(unweighted_mean - 10), \
        "High quality value should have more influence"
    
    # Test pixel-level quality scores
    pixel_quality = np.zeros((5, 3, 3))
    # Create varying quality across pixels
    for t in range(5):
        pixel_quality[t] = scene_quality[t] * np.random.uniform(0.5, 1.0, (3, 3))
    
    composite_pixel_qual, quality_map_pixel = composite_with_quality_weighting(
        data_stack=data_stack,
        quality_scores=pixel_quality,
        method="weighted_mean"
    )
    
    # Results should differ from scene-level weighting
    assert not np.allclose(composite_pixel_qual, composite_weighted), \
        "Pixel-level quality should produce different results"
    
    # Test best quality method
    composite_best, quality_best = composite_with_quality_weighting(
        data_stack=data_stack,
        quality_scores=scene_quality,
        method="best_quality"
    )
    
    # Should select from highest quality scene (index 0, quality=0.9)
    assert np.all(composite_best == data_stack[0]), "Best quality should select scene 0"
    assert np.all(quality_best == 0.9), "Quality map should show best quality value"
    
    # Test threshold mean
    composite_threshold, quality_threshold = composite_with_quality_weighting(
        data_stack=data_stack,
        quality_scores=scene_quality,
        method="threshold_mean",
        threshold=0.7
    )
    
    # Should only average scenes 0, 2, 4 (quality >= 0.7)
    expected_threshold = np.mean([data_stack[0], data_stack[2], data_stack[4]], axis=0)
    assert np.allclose(composite_threshold, expected_threshold), "Threshold mean incorrect"
    
    # Quality should reflect fraction above threshold (3/5 = 0.6)
    assert np.allclose(quality_threshold, 0.6), "Quality fraction incorrect"
    
    # Test with NaN values
    data_with_nan = data_stack.copy()
    data_with_nan[0, 1, 1] = np.nan
    data_with_nan[2, 0, 0] = np.nan
    
    composite_nan, quality_nan = composite_with_quality_weighting(
        data_stack=data_with_nan,
        quality_scores=scene_quality,
        method="weighted_mean"
    )
    
    # NaN pixels should be excluded from weighting
    assert not np.isnan(composite_nan[1, 1]), "Should compute valid result excluding NaN"
    assert not np.isnan(composite_nan[0, 0]), "Should compute valid result excluding NaN"
    
    # Test multi-band data
    multiband_data = np.stack([data_stack, data_stack * 2])  # 2 bands
    multiband_data = np.transpose(multiband_data, (1, 0, 2, 3))  # (time, bands, h, w)
    
    composite_multiband, quality_multiband = composite_with_quality_weighting(
        data_stack=multiband_data,
        quality_scores=scene_quality,
        method="weighted_mean"
    )
    
    assert composite_multiband.shape == (2, 3, 3), "Multi-band shape preserved"
    assert np.allclose(composite_multiband[1], composite_multiband[0] * 2, rtol=0.01), \
        "Band relationships preserved"


def test_composite_with_quality_weighting_methods():
    """Test different quality weighting methods.
    
    This test should verify that:
    - Weighted mean emphasizes high-quality pixels
    - Best pixel selection chooses highest quality
    - Quality-scaled percentiles work correctly
    - Custom weighting functions can be implemented
    """
    pass


def test_compute_temporal_statistics():
    """Test comprehensive temporal statistics computation.
    
    This test should verify that:
    - Basic temporal statistics (mean, variance) are accurate
    - Trend analysis detects systematic changes
    - Seasonal decomposition works correctly
    - Date information enhances temporal analysis
    - Missing data doesn't bias statistics
    - Statistical significance is properly assessed
    - Performance with large time series is acceptable
    """
    # Create test data with known statistics
    # Linear trend + noise
    n_times = 12
    height, width = 5, 5
    
    # Create systematic patterns
    trend = np.linspace(0, 10, n_times)
    noise = np.random.normal(0, 0.5, (n_times, height, width))
    
    data_stack = np.zeros((n_times, height, width))
    for t in range(n_times):
        data_stack[t] = 50 + trend[t] + noise[t]
    
    # Add some missing data
    data_stack[5, 2, 2] = np.nan
    data_stack[7, 3, 3] = np.nan
    
    # Create dates
    dates = [datetime(2023, i+1, 15) for i in range(n_times)]
    
    # Compute statistics without trend
    stats = compute_temporal_statistics(
        data_stack=data_stack,
        dates=dates,
        compute_trend=False
    )
    
    # Verify basic statistics
    assert stats['mean'].shape == (height, width), "Mean shape incorrect"
    assert stats['std'].shape == (height, width), "Std shape incorrect"
    
    # Check mean calculation for center pixel (no NaN)
    expected_mean = np.mean(data_stack[:, 2, 3])
    assert np.isclose(stats['mean'][2, 3], expected_mean), "Mean calculation incorrect"
    
    # Check std calculation
    expected_std = np.std(data_stack[:, 2, 3])
    assert np.isclose(stats['std'][2, 3], expected_std), "Std calculation incorrect"
    
    # Verify min/max
    assert np.isclose(stats['min'][2, 3], np.min(data_stack[:, 2, 3])), "Min incorrect"
    assert np.isclose(stats['max'][2, 3], np.max(data_stack[:, 2, 3])), "Max incorrect"
    
    # Verify range
    assert np.allclose(stats['range'], stats['max'] - stats['min']), "Range incorrect"
    
    # Verify count handles NaN
    assert stats['count'][2, 2] == n_times - 1, "Count should exclude NaN"
    assert stats['count'][3, 3] == n_times - 1, "Count should exclude NaN"
    assert stats['count'][0, 0] == n_times, "Count should be full for no NaN"
    
    # Verify percentiles
    assert np.all(stats['p25'] <= stats['median']), "25th percentile should be <= median"
    assert np.all(stats['median'] <= stats['p75']), "Median should be <= 75th percentile"
    
    # Verify coefficient of variation
    expected_cv = expected_std / abs(expected_mean)
    assert np.isclose(stats['cv'][2, 3], expected_cv), "CV calculation incorrect"
    
    # Test with trend analysis
    stats_trend = compute_temporal_statistics(
        data_stack=data_stack,
        dates=dates,
        compute_trend=True
    )
    
    # Verify trend detection
    assert 'trend' in stats_trend, "Trend should be computed"
    assert 'trend_pvalue' in stats_trend, "Trend p-value should be computed"
    
    # Most pixels should show positive trend
    positive_trends = np.sum(stats_trend['trend'] > 0)
    total_valid = np.sum(~np.isnan(stats_trend['trend']))
    assert positive_trends / total_valid > 0.8, "Most pixels should show positive trend"
    
    # Pixels with clear trend should have low p-values
    significant_pixels = np.sum(stats_trend['trend_pvalue'] < 0.05)
    assert significant_pixels > 0, "Some pixels should show significant trends"
    
    # Test multi-band statistics
    multiband_data = np.stack([data_stack, data_stack * 0.5, data_stack * 2.0])
    multiband_data = np.transpose(multiband_data, (1, 0, 2, 3))  # (time, bands, h, w)
    
    stats_multiband = compute_temporal_statistics(
        data_stack=multiband_data,
        compute_trend=False
    )
    
    # Verify multi-band shape
    assert stats_multiband['mean'].shape == (3, height, width), "Multi-band mean shape"
    
    # Verify band relationships preserved
    assert np.allclose(stats_multiband['mean'][1], stats_multiband['mean'][0] * 0.5, rtol=0.1), \
        "Band 2 relationship"
    assert np.allclose(stats_multiband['mean'][2], stats_multiband['mean'][0] * 2.0, rtol=0.1), \
        "Band 3 relationship"


def test_compute_temporal_statistics_trends():
    """Test trend analysis in temporal statistics.
    
    This test should verify that:
    - Linear trends are detected and quantified
    - Non-linear trends can be identified
    - Trend significance testing works correctly
    - Seasonal trends are separated from long-term trends
    - Confidence intervals are calculated appropriately
    """
    pass


def test_detect_change():
    """Test change detection between temporal composites.
    
    This test should verify that:
    - Significant changes are accurately detected
    - Change threshold controls detection sensitivity
    - Different index types have appropriate thresholds
    - Change magnitude and direction are quantified
    - Statistical significance of changes is assessed
    - False positive rates are acceptably low
    """
    # Create before/after composites with known changes
    size = (10, 10)
    
    # NDVI test case - vegetation change
    ndvi_before = np.full(size, 0.6)  # Healthy vegetation
    ndvi_after = ndvi_before.copy()
    
    # Add deforestation patch (vegetation loss)
    ndvi_after[2:5, 2:5] = 0.1  # Low NDVI (bare soil)
    
    # Add reforestation patch (vegetation gain)  
    ndvi_after[6:9, 6:9] = 0.8  # Higher NDVI
    
    # Test NDVI change detection
    change_mask = detect_change(
        before_composite=ndvi_before,
        after_composite=ndvi_after,
        threshold=0.1,
        index_type="ndvi"
    )
    
    # Verify change detection
    assert np.all(change_mask[2:5, 2:5] == -1), "Deforestation should be detected as negative change"
    assert np.all(change_mask[6:9, 6:9] == 1), "Reforestation should be detected as positive change"
    assert np.all(change_mask[0:2, 0:2] == 0), "No change areas should be 0"
    
    # Test threshold sensitivity
    change_strict = detect_change(
        before_composite=ndvi_before,
        after_composite=ndvi_after,
        threshold=0.3,  # Stricter threshold
        index_type="ndvi"
    )
    
    # Fewer pixels should be detected with stricter threshold
    assert np.sum(np.abs(change_strict) > 0) < np.sum(np.abs(change_mask) > 0), \
        "Stricter threshold should detect fewer changes"
    
    # Test with noise to assess false positives
    ndvi_noise = ndvi_before + np.random.normal(0, 0.05, size)
    
    change_noise = detect_change(
        before_composite=ndvi_before,
        after_composite=ndvi_noise,
        threshold=0.1,
        index_type="ndvi"
    )
    
    # Most pixels should show no change (low false positive rate)
    false_positive_rate = np.sum(np.abs(change_noise) > 0) / size[0] / size[1]
    assert false_positive_rate < 0.1, "False positive rate should be low"
    
    # Test radiance change detection
    radiance_before = np.full(size, 2.0)  # 2 nW/cm²/sr
    radiance_after = radiance_before.copy()
    
    # Add new development (lights)
    radiance_after[3:6, 3:6] = 10.0  # Bright lights
    
    # Add power outage area
    radiance_after[7:9, 1:3] = 0.1  # Lights out
    
    change_radiance = detect_change(
        before_composite=radiance_before,
        after_composite=radiance_after,
        index_type="radiance"
    )
    
    # Verify radiance changes detected
    assert np.all(change_radiance[3:6, 3:6] == 1), "New lights should be positive change"
    assert np.all(change_radiance[7:9, 1:3] == -1), "Power outage should be negative change"
    
    # Test default thresholds for different indices
    indices = ["ndvi", "ndwi", "ndui", "radiance", "generic"]
    
    for index in indices:
        # Create small change
        before = np.full((5, 5), 0.5)
        after = np.full((5, 5), 0.52)  # 0.02 change
        
        change = detect_change(
            before_composite=before,
            after_composite=after,
            threshold=None,  # Use default
            index_type=index
        )
        
        # Verify default thresholds are reasonable
        if index == "ndui":
            # NDUI has stricter threshold (0.05)
            assert np.all(change == 0), f"Small change should not be detected for {index}"
        else:
            # Others have more relaxed thresholds
            pass  # May or may not detect depending on exact threshold
    
    # Test NaN handling
    before_nan = ndvi_before.copy()
    after_nan = ndvi_after.copy()
    before_nan[0, 0] = np.nan
    after_nan[9, 9] = np.nan
    
    change_nan = detect_change(
        before_composite=before_nan,
        after_composite=after_nan,
        threshold=0.1,
        index_type="ndvi"
    )
    
    # NaN pixels should show no change
    assert change_nan[0, 0] == 0, "NaN in before should show no change"
    assert change_nan[9, 9] == 0, "NaN in after should show no change"


def test_detect_change_thresholds():
    """Test change detection with different threshold strategies.
    
    This test should verify that:
    - Fixed thresholds work for known change types
    - Adaptive thresholds adjust to local conditions
    - Statistical thresholds use data-driven cutoffs
    - Threshold selection affects detection performance
    """
    pass


def test_detect_change_indices():
    """Test change detection for different spectral indices.
    
    This test should verify that:
    - NDVI change detection works for vegetation monitoring
    - NDUI change detection captures urban development
    - NTL change detection identifies lighting changes
    - Index-specific thresholds are appropriately set
    """
    pass


def test_fill_temporal_gaps():
    """Test intelligent gap filling in temporal data.
    
    This test should verify that:
    - Linear interpolation fills short gaps effectively
    - Spline interpolation preserves temporal patterns
    - Maximum gap length prevents unrealistic filling
    - Interpolation doesn't introduce artifacts
    - Edge effects are handled appropriately
    - Gap-filled values are properly flagged
    """
    # Create time series with known patterns and gaps
    n_times = 10
    
    # Linear trend with gaps
    linear_data = np.zeros((n_times, 3, 3))
    for t in range(n_times):
        linear_data[t] = t * 10  # Linear increase
    
    # Add gaps
    linear_data[3, :, :] = np.nan  # Single gap
    linear_data[6:8, :, :] = np.nan  # Double gap
    
    # Test linear interpolation
    filled_linear = fill_temporal_gaps(
        data_stack=linear_data,
        method="linear",
        max_gap=3
    )
    
    # Verify single gap is filled correctly
    # Linear interpolation between t=2 (20) and t=4 (40) -> t=3 should be 30
    assert np.allclose(filled_linear[3, :, :], 30), "Linear interpolation of single gap failed"
    
    # Verify double gap is filled
    # t=6 should be 60, t=7 should be 70
    assert np.allclose(filled_linear[6, :, :], 60), "Linear interpolation of double gap failed"
    assert np.allclose(filled_linear[7, :, :], 70), "Linear interpolation of double gap failed"
    
    # Test max gap constraint
    large_gap_data = linear_data.copy()
    large_gap_data[2:7, 1, 1] = np.nan  # 5-time gap at one pixel
    
    filled_constrained = fill_temporal_gaps(
        data_stack=large_gap_data,
        method="linear",
        max_gap=3
    )
    
    # Large gap should not be filled
    assert np.isnan(filled_constrained[4, 1, 1]), "Gap larger than max_gap should not be filled"
    
    # But smaller gaps should still be filled
    assert not np.isnan(filled_constrained[3, 0, 0]), "Small gaps should still be filled"
    
    # Test spline interpolation with curved data
    curved_data = np.zeros((n_times, 3, 3))
    t = np.linspace(0, 2*np.pi, n_times)
    for i in range(n_times):
        curved_data[i] = 50 + 30 * np.sin(t[i])  # Sinusoidal pattern
    
    # Add gaps
    curved_data[2, :, :] = np.nan
    curved_data[5, :, :] = np.nan
    curved_data[8, :, :] = np.nan
    
    filled_spline = fill_temporal_gaps(
        data_stack=curved_data,
        method="spline",
        max_gap=2
    )
    
    # Spline should preserve smooth curves
    # Check that filled values are reasonable (within range of neighboring values)
    for gap_idx in [2, 5, 8]:
        filled_val = filled_spline[gap_idx, 0, 0]
        
        # Get neighbors
        before_idx = max(0, gap_idx - 1)
        after_idx = min(n_times - 1, gap_idx + 1)
        
        if before_idx != gap_idx and after_idx != gap_idx:
            neighbor_min = min(curved_data[before_idx, 0, 0], curved_data[after_idx, 0, 0])
            neighbor_max = max(curved_data[before_idx, 0, 0], curved_data[after_idx, 0, 0])
            
            # Spline can overshoot slightly but should be reasonable
            assert neighbor_min - 10 < filled_val < neighbor_max + 10, \
                f"Spline interpolation produced unrealistic value at t={gap_idx}"
    
    # Test nearest neighbor interpolation
    filled_nearest = fill_temporal_gaps(
        data_stack=linear_data,
        method="nearest",
        max_gap=3
    )
    
    # Nearest should pick closest valid value
    # For t=3, should be either 20 (t=2) or 40 (t=4)
    assert filled_nearest[3, 0, 0] in [20, 40], "Nearest neighbor should pick adjacent value"
    
    # Test edge effects
    edge_data = np.arange(n_times * 3 * 3).reshape(n_times, 3, 3).astype(float)
    edge_data[0, :, :] = np.nan  # Gap at start
    edge_data[-1, :, :] = np.nan  # Gap at end
    
    filled_edges = fill_temporal_gaps(
        data_stack=edge_data,
        method="nearest",
        max_gap=1
    )
    
    # Edge gaps should be filled with nearest valid values
    assert np.allclose(filled_edges[0, :, :], edge_data[1, :, :]), "Start gap should use first valid"
    assert np.allclose(filled_edges[-1, :, :], edge_data[-2, :, :]), "End gap should use last valid"
    
    # Test multi-band gap filling
    multiband = np.zeros((n_times, 2, 3, 3))
    multiband[:, 0, :, :] = linear_data
    multiband[:, 1, :, :] = linear_data * 2
    
    filled_multiband = fill_temporal_gaps(
        data_stack=multiband,
        method="linear",
        max_gap=3
    )
    
    # Both bands should be filled
    assert not np.isnan(filled_multiband[3, 0, 0, 0]), "Band 1 should be filled"
    assert not np.isnan(filled_multiband[3, 1, 0, 0]), "Band 2 should be filled"
    
    # Band relationship should be preserved
    assert np.allclose(filled_multiband[3, 1, :, :], filled_multiband[3, 0, :, :] * 2), \
        "Band relationships should be preserved in gap filling"


def test_fill_temporal_gaps_methods():
    """Test different gap filling interpolation methods.
    
    This test should verify that:
    - Linear interpolation is simple and robust
    - Spline interpolation preserves smooth curves
    - Polynomial interpolation handles complex patterns
    - Seasonal interpolation uses periodic patterns
    - Method selection affects filling quality
    """
    pass


def test_fill_temporal_gaps_constraints():
    """Test gap filling with various constraints.
    
    This test should verify that:
    - Maximum gap length prevents over-interpolation
    - Minimum data points ensure reliable interpolation
    - Edge gaps are handled differently from interior gaps
    - Quality constraints influence interpolation decisions
    """
    pass


def test_compute_phenology_metrics():
    """Test vegetation phenology metrics computation from NDVI time series.
    
    This test should verify that:
    - Growing season start/end dates are identified correctly
    - Peak greenness timing and magnitude are accurate
    - Growing season length calculations are correct
    - Rate of green-up and senescence are quantified
    - Phenology metrics are ecologically meaningful
    - Missing data doesn't bias phenology calculations
    - Multi-year phenology patterns are characterized
    """
    # Create synthetic NDVI time series with realistic phenology
    n_times = 12  # Monthly data
    height, width = 5, 5
    
    # Create dates for each month
    dates = [datetime(2023, i+1, 15) for i in range(n_times)]
    
    # Create seasonal NDVI pattern (Northern Hemisphere deciduous)
    # Low in winter, peak in summer
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    
    # Generate realistic NDVI curves
    ndvi_stack = np.zeros((n_times, height, width))
    
    for i in range(height):
        for j in range(width):
            # Base seasonal pattern
            # Use cosine curve centered on day 180 (end of June)
            seasonal = 0.3 + 0.4 * np.cos(2 * np.pi * (day_of_year - 180) / 365)
            
            # Add some spatial variation
            amplitude_var = 0.3 + 0.2 * (i + j) / (height + width)
            seasonal = seasonal * amplitude_var
            
            # Add small noise
            noise = np.random.normal(0, 0.02, n_times)
            
            ndvi_stack[:, i, j] = seasonal + noise
    
    # Add some missing data
    ndvi_stack[5, 2, 2] = np.nan
    ndvi_stack[8, 3, 3] = np.nan
    
    # Compute phenology metrics
    metrics = compute_phenology_metrics(
        ndvi_stack=ndvi_stack,
        dates=dates
    )
    
    # Verify all metrics are computed
    expected_metrics = [
        'greenup_date', 'peak_date', 'senescence_date', 'dormancy_date',
        'growing_season_length', 'peak_ndvi', 'min_ndvi', 'ndvi_amplitude',
        'integrated_ndvi'
    ]
    
    for metric in expected_metrics:
        assert metric in metrics, f"Missing metric: {metric}"
        assert metrics[metric].shape == (height, width), f"{metric} shape incorrect"
    
    # Check phenology timing makes ecological sense
    # For most pixels (excluding edges and NaN):
    valid_pixels = []
    for i in range(1, height-1):
        for j in range(1, width-1):
            if not np.isnan(metrics['greenup_date'][i, j]):
                valid_pixels.append((i, j))
    
    for i, j in valid_pixels:
        greenup = metrics['greenup_date'][i, j]
        peak = metrics['peak_date'][i, j]
        senescence = metrics['senescence_date'][i, j]
        dormancy = metrics['dormancy_date'][i, j]
        
        # Verify temporal ordering
        if not np.isnan(greenup) and not np.isnan(peak):
            assert greenup < peak, f"Green-up should precede peak at ({i},{j})"
        
        if not np.isnan(peak) and not np.isnan(senescence):
            assert peak < senescence, f"Peak should precede senescence at ({i},{j})"
        
        if not np.isnan(senescence) and not np.isnan(dormancy):
            assert senescence < dormancy, f"Senescence should precede dormancy at ({i},{j})"
        
        # Peak should be in summer (approximately June-August, days 152-243)
        if not np.isnan(peak):
            assert 120 < peak < 270, f"Peak should be in growing season at ({i},{j})"
        
        # Growing season length should be reasonable (60-300 days for temperate)
        gsl = metrics['growing_season_length'][i, j]
        if not np.isnan(gsl):
            assert 30 < gsl < 300, f"Growing season length unrealistic at ({i},{j}): {gsl}"
    
    # Verify peak NDVI > minimum NDVI
    valid_mask = ~np.isnan(metrics['peak_ndvi']) & ~np.isnan(metrics['min_ndvi'])
    assert np.all(metrics['peak_ndvi'][valid_mask] > metrics['min_ndvi'][valid_mask]), \
        "Peak NDVI should exceed minimum NDVI"
    
    # Verify amplitude calculation
    expected_amplitude = metrics['peak_ndvi'] - metrics['min_ndvi']
    assert np.allclose(metrics['ndvi_amplitude'], expected_amplitude, equal_nan=True), \
        "NDVI amplitude calculation incorrect"
    
    # Check integrated NDVI (productivity proxy)
    # Should be positive and reasonable
    valid_integrated = metrics['integrated_ndvi'][~np.isnan(metrics['integrated_ndvi'])]
    assert np.all(valid_integrated > 0), "Integrated NDVI should be positive"
    assert np.all(valid_integrated < 365), "Integrated NDVI unrealistically high"
    
    # Test with flat NDVI (non-vegetated area)
    flat_ndvi = np.full((n_times, 3, 3), 0.05)  # Low, constant NDVI
    flat_ndvi += np.random.normal(0, 0.01, flat_ndvi.shape)  # Small noise
    
    metrics_flat = compute_phenology_metrics(
        ndvi_stack=flat_ndvi,
        dates=dates
    )
    
    # Most pixels should have NaN phenology (no clear season)
    nan_count = np.sum(np.isnan(metrics_flat['greenup_date']))
    total_pixels = 9
    assert nan_count > total_pixels * 0.5, "Flat NDVI should not show clear phenology"
    
    # Test with double peak (e.g., double cropping)
    double_peak_ndvi = np.zeros((n_times, 2, 2))
    
    # Create two peaks - early summer and early fall
    for t in range(n_times):
        doy = day_of_year[t]
        # First peak around day 150, second around day 250
        val = 0.2 + 0.3 * np.exp(-((doy - 150)**2) / 1000) + \
              0.25 * np.exp(-((doy - 250)**2) / 1000)
        double_peak_ndvi[t, :, :] = val
    
    metrics_double = compute_phenology_metrics(
        ndvi_stack=double_peak_ndvi,
        dates=dates
    )
    
    # Should detect the higher peak
    peak_doy = metrics_double['peak_date'][0, 0]
    if not np.isnan(peak_doy):
        # Peak should be near one of the two peaks
        assert (130 < peak_doy < 170) or (230 < peak_doy < 270), \
            "Should detect one of the two peaks"
    
    # Test error handling with insufficient data
    short_ndvi = np.random.rand(3, 5, 5)  # Only 3 time points
    short_dates = dates[:3]
    
    with pytest.raises(ValueError, match="at least 6 observations"):
        compute_phenology_metrics(
            ndvi_stack=short_ndvi,
            dates=short_dates
        )


def test_compute_phenology_metrics_seasonality():
    """Test phenology metrics for different seasonal patterns.
    
    This test should verify that:
    - Single-peak seasonal patterns are handled correctly
    - Multi-peak patterns (e.g., double cropping) are detected
    - Evergreen vegetation shows appropriate metrics
    - Desert vegetation phenology is characterized
    - Seasonal timing variations are captured
    """
    pass


def test_compute_phenology_metrics_thresholds():
    """Test phenology threshold detection methods.
    
    This test should verify that:
    - Fixed NDVI thresholds work for standard vegetation
    - Relative thresholds adapt to vegetation types
    - Derivative-based methods detect inflection points
    - Threshold selection affects phenology timing
    """
    pass


def test_temporal_compositing_edge_cases():
    """Test temporal functions with edge cases.
    
    This test should verify that:
    - Single-image time series are handled correctly
    - Time series with large gaps don't cause errors
    - Very short time series produce reasonable results
    - Different temporal sampling frequencies work
    - Memory efficiency with large time series
    """
    pass


def test_temporal_data_quality():
    """Test quality assessment of temporal processing results.
    
    This test should verify that:
    - Composite quality metrics are computed
    - Temporal consistency is assessed
    - Uncertainty estimates are provided
    - Quality flags indicate processing reliability
    - Results guide interpretation of temporal patterns
    """
    pass


def test_temporal_performance():
    """Test performance of temporal processing functions.
    
    This test should verify that:
    - Large time series are processed efficiently
    - Memory usage scales appropriately
    - Parallel processing is utilized when available
    - Algorithmic complexity matches expectations
    - Processing time is reasonable for operational use
    """
    pass


def test_temporal_integration():
    """Test integration between temporal processing functions.
    
    This test should verify that:
    - Scene selection feeds into compositing
    - Quality weighting improves composite results
    - Gap filling enhances time series analysis
    - Phenology metrics use composite inputs
    - Workflow efficiency with chained operations
    """
    pass