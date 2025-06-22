"""Tests for temporal and composite processing functions.

These tests verify the temporal compositing and scene merging functions
that replace the old component-based system.
"""

import numpy as np
import pytest
from datetime import datetime, timedelta
from rasterio.transform import Affine

from blackmarble.processing import (
    create_temporal_composite,
    merge_scenes,
    select_by_date_range,
    composite_with_quality,
    compute_statistics,
)


class TestTemporalComposite:
    """Test create_temporal_composite function."""

    def test_median_composite_simple(self):
        """Test median composite with simple data."""
        # Stack of 3 time steps
        data = np.array(
            [[[1, 2], [3, 4]], [[2, 3], [4, 5]], [[3, 4], [5, 6]]], dtype=np.float32
        )

        composite = create_temporal_composite(data, method="median")

        # Median of [1,2,3] is 2, [2,3,4] is 3, etc.
        expected = np.array([[2, 3], [4, 5]], dtype=np.float32)
        np.testing.assert_array_equal(composite, expected)

    def test_mean_composite(self):
        """Test mean composite."""
        data = np.array(
            [[[1, 2], [3, 4]], [[2, 3], [4, 5]], [[3, 4], [5, 6]]], dtype=np.float32
        )

        composite = create_temporal_composite(data, method="mean")

        # Mean of [1,2,3] is 2, [2,3,4] is 3, etc.
        expected = np.array([[2, 3], [4, 5]], dtype=np.float32)
        np.testing.assert_array_equal(composite, expected)

    def test_max_composite(self):
        """Test maximum composite."""
        data = np.array(
            [[[1, 2], [3, 4]], [[5, 3], [2, 1]], [[3, 4], [5, 6]]], dtype=np.float32
        )

        composite = create_temporal_composite(data, method="max")

        expected = np.array([[5, 4], [5, 6]], dtype=np.float32)
        np.testing.assert_array_equal(composite, expected)

    def test_percentile_composite(self):
        """Test percentile composite."""
        data = np.array(
            [[[1, 2], [3, 4]], [[2, 4], [6, 8]], [[3, 6], [9, 12]], [[4, 8], [12, 16]]],
            dtype=np.float32,
        )

        composite = create_temporal_composite(data, method="percentile", percentile=75)

        # 75th percentile of [1,2,3,4] is 3.25
        assert composite[0, 0] == pytest.approx(3.25)
        assert composite[0, 1] == pytest.approx(6.5)
        assert composite[1, 0] == pytest.approx(9.75)
        assert composite[1, 1] == pytest.approx(13.0)

    def test_composite_with_nan(self):
        """Test compositing with NaN values."""
        data = np.array(
            [[[1, np.nan], [3, 4]], [[2, 3], [np.nan, 5]], [[3, 4], [5, np.nan]]],
            dtype=np.float32,
        )

        composite = create_temporal_composite(data, method="median")

        # Median ignoring NaN
        # Column 0: [1,2,3] -> median 2
        # Column 1: [nan,3,4] -> median 3.5
        # For [1,0]: [3,nan,5] -> median 4
        # For [1,1]: [4,5,nan] -> median 4.5
        expected = np.array([[2, 3.5], [4, 4.5]], dtype=np.float32)
        np.testing.assert_array_almost_equal(composite, expected)

    def test_min_valid_observations(self):
        """Test minimum valid observations requirement."""
        data = np.array(
            [[[1, np.nan], [3, 4]], [[2, np.nan], [4, 5]], [[3, np.nan], [5, 6]]],
            dtype=np.float32,
        )

        composite = create_temporal_composite(
            data, method="median", min_valid_observations=2
        )

        # Second column has only 0 valid observations, should be NaN
        assert not np.isnan(composite[0, 0])
        assert np.isnan(composite[0, 1])
        assert not np.isnan(composite[1, 0])
        assert not np.isnan(composite[1, 1])

    def test_quality_weighted_composite(self):
        """Test quality-weighted composite."""
        data = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32
        )

        # Higher weights for later dates
        weights = np.array(
            [
                [[0.1, 0.1], [0.1, 0.1]],
                [[0.3, 0.3], [0.3, 0.3]],
                [[0.6, 0.6], [0.6, 0.6]],
            ]
        )

        composite = create_temporal_composite(
            data, method="quality_weighted", weights=weights
        )

        # Weighted average: (1*0.1 + 5*0.3 + 9*0.6) / (0.1+0.3+0.6) = 7.0
        expected = np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(composite, expected)

    def test_multi_band_composite(self):
        """Test compositing multi-band data."""
        # 3 time steps, 2 bands, 2x2 pixels
        data = np.array(
            [
                [[[1, 2], [3, 4]], [[10, 20], [30, 40]]],  # Time 1
                [[[2, 3], [4, 5]], [[20, 30], [40, 50]]],  # Time 2
                [[[3, 4], [5, 6]], [[30, 40], [50, 60]]],  # Time 3
            ],
            dtype=np.float32,
        )

        composite = create_temporal_composite(data, method="median")

        # Should return 2 bands
        assert composite.shape == (2, 2, 2)

        # Band 1 median
        expected_b1 = np.array([[2, 3], [4, 5]], dtype=np.float32)
        np.testing.assert_array_equal(composite[0], expected_b1)

        # Band 2 median
        expected_b2 = np.array([[20, 30], [40, 50]], dtype=np.float32)
        np.testing.assert_array_equal(composite[1], expected_b2)


class TestMergeScenes:
    """Test merge_scenes function."""

    def test_merge_non_overlapping(self):
        """Test merging non-overlapping scenes."""
        scene1 = np.ones((10, 10), dtype=np.float32) * 1.0
        scene2 = np.ones((10, 10), dtype=np.float32) * 2.0

        # Non-overlapping transforms
        transform1 = Affine.translation(0, 0)
        transform2 = Affine.translation(10, 0)

        bounds1 = (0, 0, 10, 10)
        bounds2 = (10, 0, 20, 10)

        merged, _, merged_bounds = merge_scenes(
            [scene1, scene2],
            [transform1, transform2],
            [bounds1, bounds2],
            method="first",
        )

        assert merged.shape == (10, 20)
        assert merged_bounds == (0, 0, 20, 10)

        # Check values
        np.testing.assert_array_equal(merged[:, :10], 1.0)
        np.testing.assert_array_equal(merged[:, 10:], 2.0)

    def test_merge_overlapping_max(self):
        """Test merging overlapping scenes with max method."""
        scene1 = np.ones((10, 10), dtype=np.float32) * 1.0
        scene2 = np.ones((10, 10), dtype=np.float32) * 2.0

        # Overlapping transforms
        transform1 = Affine.translation(0, 0)
        transform2 = Affine.translation(5, 0)

        bounds1 = (0, 0, 10, 10)
        bounds2 = (5, 0, 15, 10)

        merged, _, _ = merge_scenes(
            [scene1, scene2], [transform1, transform2], [bounds1, bounds2], method="max"
        )

        # In overlap region (5-10), should have max value (2.0)
        assert merged[5, 7] == 2.0

        # Non-overlap regions
        assert merged[5, 2] == 1.0  # Only in scene1
        assert merged[5, 12] == 2.0  # Only in scene2

    def test_merge_mean(self):
        """Test mean merging."""
        scene1 = np.ones((10, 10), dtype=np.float32) * 2.0
        scene2 = np.ones((10, 10), dtype=np.float32) * 4.0

        # Fully overlapping
        transform = Affine.translation(0, 0)
        bounds = (0, 0, 10, 10)

        merged, _, _ = merge_scenes(
            [scene1, scene2], [transform, transform], [bounds, bounds], method="mean"
        )

        # Mean of 2.0 and 4.0 is 3.0
        np.testing.assert_array_equal(merged, 3.0)

    def test_merge_with_nodata(self):
        """Test merging with nodata values."""
        scene1 = np.ones((10, 10), dtype=np.float32) * 1.0
        scene1[5:, :] = np.nan  # Bottom half is nodata

        scene2 = np.ones((10, 10), dtype=np.float32) * 2.0
        scene2[:5, :] = np.nan  # Top half is nodata

        transform = Affine.translation(0, 0)
        bounds = (0, 0, 10, 10)

        merged, _, _ = merge_scenes(
            [scene1, scene2],
            [transform, transform],
            [bounds, bounds],
            method="first",
            nodata=np.nan,
        )

        # Top half should be from scene1, bottom from scene2
        np.testing.assert_array_equal(merged[:5, :], 1.0)
        np.testing.assert_array_equal(merged[5:, :], 2.0)


class TestSelectByDateRange:
    """Test select_by_date_range function."""

    def test_date_range_filter(self):
        """Test filtering by date range."""
        scenes = [np.ones((10, 10)) for _ in range(10)]
        base_date = datetime(2023, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(10)]

        # Select middle 5 days
        start = datetime(2023, 1, 3)
        end = datetime(2023, 1, 7)

        selected, sel_dates, indices = select_by_date_range(
            scenes, dates, date_range=(start, end)
        )

        assert len(selected) == 5
        assert indices == [2, 3, 4, 5, 6]
        assert sel_dates[0] == datetime(2023, 1, 3)
        assert sel_dates[-1] == datetime(2023, 1, 7)

    def test_target_date_filter(self):
        """Test filtering by proximity to target date."""
        scenes = [np.ones((10, 10)) for _ in range(10)]
        base_date = datetime(2023, 1, 1)
        dates = [base_date + timedelta(days=i * 2) for i in range(10)]

        target = datetime(2023, 1, 10)

        selected, sel_dates, indices = select_by_date_range(
            scenes, dates, target_date=target, max_days_from_target=3
        )

        # Dates are: Jan 1, 3, 5, 7, 9, 11, 13, 15, 17, 19
        # Within 3 days of Jan 10 are: Jan 7 (idx 3), Jan 9 (idx 4), Jan 11 (idx 5), Jan 13 (idx 6)
        assert len(selected) == 4
        assert 3 in indices
        assert 4 in indices
        assert 5 in indices
        assert 6 in indices

    def test_cloud_filter(self):
        """Test filtering by cloud cover."""
        scenes = [np.ones((10, 10)) for _ in range(5)]
        dates = [datetime(2023, 1, i + 1) for i in range(5)]
        metadata = [
            {"cloud_cover": 10},
            {"cloud_cover": 30},
            {"cloud_cover": 5},
            {"cloud_cover": 50},
            {"cloud_cover": 15},
        ]

        selected, _, indices = select_by_date_range(
            scenes, dates, metadata=metadata, cloud_threshold=20
        )

        # Should select scenes with cloud cover <= 20%
        assert len(selected) == 3
        assert indices == [0, 2, 4]

    def test_max_scenes_limit(self):
        """Test limiting maximum number of scenes."""
        scenes = [np.ones((10, 10)) for _ in range(10)]
        dates = [datetime(2023, 1, i + 1) for i in range(10)]

        selected, _, indices = select_by_date_range(scenes, dates, max_scenes=3)

        assert len(selected) == 3
        assert indices == [0, 1, 2]  # First 3 by date order


class TestCompositeWithQuality:
    """Test composite_with_quality function."""

    def test_weighted_mean(self):
        """Test quality-weighted mean composite."""
        data = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32
        )

        quality = np.array(
            [
                [[0.1, 0.2], [0.3, 0.4]],
                [[0.5, 0.6], [0.7, 0.8]],
                [[0.9, 1.0], [0.9, 0.8]],
            ]
        )

        composite = composite_with_quality(data, quality, method="weighted_mean")

        # Weighted averages
        # Position [0,0]: (1*0.1 + 5*0.5 + 9*0.9) / (0.1+0.5+0.9) = 10.7/1.5 = 7.133...
        assert composite[0, 0] == pytest.approx(7.133333, rel=1e-5)

    def test_best_quality(self):
        """Test best quality selection."""
        data = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32
        )

        quality = np.array(
            [
                [[0.9, 0.2], [0.3, 0.4]],
                [[0.5, 0.6], [0.7, 0.8]],
                [[0.1, 1.0], [0.9, 0.2]],
            ]
        )

        composite = composite_with_quality(data, quality, method="best_quality")

        # Should select from highest quality
        assert composite[0, 0] == 1  # Quality 0.9 from first date
        assert composite[0, 1] == 10  # Quality 1.0 from third date
        assert composite[1, 0] == 11  # Quality 0.9 from third date
        assert composite[1, 1] == 8  # Quality 0.8 from second date

    def test_threshold_mean(self):
        """Test threshold-based mean composite."""
        data = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32
        )

        quality = np.array(
            [
                [[0.1, 0.8], [0.3, 0.9]],
                [[0.7, 0.2], [0.8, 0.1]],
                [[0.9, 0.6], [0.7, 0.5]],
            ]
        )

        composite = composite_with_quality(
            data, quality, method="threshold_mean", quality_threshold=0.5
        )

        # Position [0,0]: mean of values with quality >= 0.5: (5*0.7 + 9*0.9)/(0.7+0.9)
        assert composite[0, 0] == pytest.approx(7.0, rel=1e-5)
        # Position [0,1]: mean of [2, 10] (quality 0.8, 0.6)
        assert composite[0, 1] == pytest.approx(6.0, rel=1e-5)


class TestComputeStatistics:
    """Test compute_statistics function."""

    def test_basic_statistics(self):
        """Test basic statistics computation."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

        stats = compute_statistics(data)

        assert stats["count"] == 9
        assert stats["min"] == 1.0
        assert stats["max"] == 9.0
        assert stats["mean"] == 5.0
        assert stats["median"] == 5.0
        assert stats["valid_fraction"] == 1.0

    def test_statistics_with_nan(self):
        """Test statistics with NaN values."""
        data = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]], dtype=np.float32)

        stats = compute_statistics(data)

        assert stats["count"] == 7  # 9 - 2 NaN values
        assert stats["min"] == 1.0
        assert stats["max"] == 9.0
        assert stats["valid_fraction"] == pytest.approx(7 / 9)

    def test_statistics_with_mask(self):
        """Test statistics with boolean mask."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        mask = np.array([[True, True, False], [True, False, True], [False, True, True]])

        stats = compute_statistics(data, mask=mask)

        # Values where mask is True: 1, 2, 4, 6, 8, 9
        assert stats["count"] == 6
        assert stats["mean"] == pytest.approx((1 + 2 + 4 + 6 + 8 + 9) / 6)

    def test_statistics_with_percentiles(self):
        """Test statistics with custom percentiles."""
        data = np.arange(100, dtype=np.float32).reshape(10, 10)

        stats = compute_statistics(data, percentiles=[25, 50, 75, 95])

        assert "p25" in stats
        assert "p50" in stats
        assert "p75" in stats
        assert "p95" in stats

        assert stats["p50"] == pytest.approx(49.5)  # Median of 0-99
