"""Behavioral tests for temporal processing components.

These tests use predictable artificial data to verify exact behavior
of temporal components without external dependencies.
"""

import numpy as np
import pytest
from datetime import datetime, timedelta

from blackmarble.components.temporal import (
    TemporalCompositor,
    CompositeMethod,
    SceneMerger,
    DateRangeSelector,
)


class TestTemporalCompositor:
    """Test TemporalCompositor median/mean calculations."""
    
    def test_median_composite_simple(self):
        """Test median composite with simple data."""
        compositor = TemporalCompositor(method=CompositeMethod.MEDIAN)
        
        # Create 3 time steps of 2x2 data
        data = np.array([
            [[1, 2], [3, 4]],
            [[2, 3], [4, 5]],
            [[3, 4], [5, 6]]
        ], dtype=np.float32)
        
        result = compositor.process(data)
        composite = result.metadata["composite"]
        
        # Median should be middle values
        expected = np.array([[2, 3], [4, 5]], dtype=np.float32)
        assert np.array_equal(composite, expected)
        assert result.mask.shape == (2, 2)
        assert np.all(result.mask == 1.0)
        
    def test_mean_composite_simple(self):
        """Test mean composite with simple data."""
        compositor = TemporalCompositor(method=CompositeMethod.MEAN)
        
        # Create 3 time steps
        data = np.array([
            [[0, 3], [6, 9]],
            [[3, 6], [9, 12]],
            [[6, 9], [12, 15]]
        ], dtype=np.float32)
        
        result = compositor.process(data)
        composite = result.metadata["composite"]
        
        # Mean should be average
        expected = np.array([[3, 6], [9, 12]], dtype=np.float32)
        assert np.allclose(composite, expected)
        
    def test_max_composite(self):
        """Test maximum value composite."""
        compositor = TemporalCompositor(method=CompositeMethod.MAX)
        
        data = np.array([
            [[1, 5], [2, 3]],
            [[4, 2], [6, 1]],
            [[3, 3], [1, 7]]
        ], dtype=np.float32)
        
        result = compositor.process(data)
        composite = result.metadata["composite"]
        
        expected = np.array([[4, 5], [6, 7]], dtype=np.float32)
        assert np.array_equal(composite, expected)
        
    def test_min_composite(self):
        """Test minimum value composite."""
        compositor = TemporalCompositor(method=CompositeMethod.MIN)
        
        data = np.array([
            [[1, 5], [2, 3]],
            [[4, 2], [6, 1]],
            [[3, 3], [1, 7]]
        ], dtype=np.float32)
        
        result = compositor.process(data)
        composite = result.metadata["composite"]
        
        expected = np.array([[1, 2], [1, 1]], dtype=np.float32)
        assert np.array_equal(composite, expected)
        
    def test_percentile_composite(self):
        """Test percentile composite."""
        compositor = TemporalCompositor(
            method=CompositeMethod.PERCENTILE,
            percentile=75
        )
        
        # Create data where 75th percentile is clear
        data = np.array([
            [[1, 2], [3, 4]],
            [[2, 4], [6, 8]],
            [[3, 6], [9, 12]],
            [[4, 8], [12, 16]]
        ], dtype=np.float32)
        
        result = compositor.process(data)
        composite = result.metadata["composite"]
        
        # 75th percentile of [1,2,3,4] is 3.25
        assert composite[0, 0] == pytest.approx(3.25)
        assert composite[0, 1] == pytest.approx(6.5)
        assert composite[1, 0] == pytest.approx(9.75)
        assert composite[1, 1] == pytest.approx(13.0)
        
    def test_quality_weighted_composite(self):
        """Test quality-weighted composite."""
        compositor = TemporalCompositor(method=CompositeMethod.QUALITY_WEIGHTED)
        
        data = np.array([
            [[10, 20], [30, 40]],
            [[15, 25], [35, 45]],
            [[20, 30], [40, 50]]
        ], dtype=np.float32)
        
        # Weights: high for middle time step
        weights = np.array([0.2, 0.6, 0.2])
        
        result = compositor.process(data, weights=weights)
        composite = result.metadata["composite"]
        
        # Weighted average
        expected = np.array([
            [15, 25],  # 10*0.2 + 15*0.6 + 20*0.2 = 15
            [35, 45]   # 30*0.2 + 35*0.6 + 40*0.2 = 35
        ], dtype=np.float32)
        assert np.allclose(composite, expected)
        
    def test_composite_with_nodata(self):
        """Test compositing with nodata values."""
        compositor = TemporalCompositor(method=CompositeMethod.MEDIAN)
        
        # Create data with NaN values
        data = np.array([
            [[1, np.nan], [3, 4]],
            [[2, 3], [np.nan, 5]],
            [[3, 4], [5, 6]]
        ], dtype=np.float32)
        
        result = compositor.process(data)
        composite = result.metadata["composite"]
        
        # Should ignore NaN in median calculation
        assert composite[0, 0] == 2  # median of [1, 2, 3]
        assert composite[0, 1] == pytest.approx(3.5)  # median of [3, 4]
        assert composite[1, 0] == 4  # median of [3, 5]
        assert composite[1, 1] == 5  # median of [4, 5, 6]
        
    def test_composite_with_quality_masks(self):
        """Test compositing with quality masks."""
        compositor = TemporalCompositor(method=CompositeMethod.MEAN)
        
        data = np.array([
            [[10, 20], [30, 40]],
            [[100, 200], [300, 400]],  # Bad quality
            [[15, 25], [35, 45]]
        ], dtype=np.float32)
        
        # Quality mask: 0 = bad, 1 = good
        quality_masks = np.array([
            [[1, 1], [1, 1]],
            [[0, 0], [0, 0]],  # All bad
            [[1, 1], [1, 1]]
        ], dtype=np.float32)
        
        result = compositor.process(data, quality_masks=quality_masks)
        composite = result.metadata["composite"]
        
        # Should only use good quality data
        expected = np.array([[12.5, 22.5], [32.5, 42.5]], dtype=np.float32)
        assert np.allclose(composite, expected)
        
    def test_min_valid_observations(self):
        """Test minimum valid observations requirement."""
        compositor = TemporalCompositor(
            method=CompositeMethod.MEDIAN,
            min_valid_observations=3
        )
        
        # Create data where some pixels have insufficient observations
        data = np.array([
            [[1, np.nan], [3, 4]],
            [[2, np.nan], [4, 5]],
            [[3, np.nan], [5, 6]]
        ], dtype=np.float32)
        
        result = compositor.process(data)
        
        # Pixel [0,1] has 0 valid obs, should be masked out
        assert result.mask[0, 1] == 0.0
        # Other pixels have 3 valid obs
        assert result.mask[0, 0] == 1.0
        assert result.mask[1, 0] == 1.0
        assert result.mask[1, 1] == 1.0
        
    def test_multiband_composite(self):
        """Test compositing multi-band data."""
        compositor = TemporalCompositor(method=CompositeMethod.MEDIAN)
        
        # Create 3 time steps, 2 bands, 2x2 pixels
        data = np.array([
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            [[[2, 3], [4, 5]], [[6, 7], [8, 9]]],
            [[[3, 4], [5, 6]], [[7, 8], [9, 10]]]
        ], dtype=np.float32)
        
        result = compositor.process(data)
        composite = result.metadata["composite"]
        
        assert composite.shape == (2, 2, 2)  # 2 bands, 2x2
        # Check medians
        assert np.array_equal(composite[0], [[2, 3], [4, 5]])
        assert np.array_equal(composite[1], [[6, 7], [8, 9]])
        
    def test_temporal_coverage_stats(self):
        """Test temporal coverage statistics calculation."""
        compositor = TemporalCompositor(method=CompositeMethod.MEAN)
        
        # Create data spanning 10 days
        data = np.ones((5, 2, 2), dtype=np.float32)
        dates = [
            datetime(2023, 1, 1),
            datetime(2023, 1, 3),
            datetime(2023, 1, 5),
            datetime(2023, 1, 7),
            datetime(2023, 1, 10),
        ]
        
        result = compositor.process(data, dates=dates)
        coverage = result.metadata["coverage"]
        
        assert coverage["temporal_range_days"] == 9
        assert coverage["observation_frequency"] == pytest.approx(5/9)
        assert coverage["coverage_percentage"] == 100.0
        
    def test_fill_nodata_gaps(self):
        """Test gap filling functionality."""
        compositor = TemporalCompositor(
            method=CompositeMethod.MEDIAN,
            min_valid_observations=1,
            fill_nodata=True
        )
        
        # Create sparse data
        data = np.array([
            [[1, np.nan, np.nan], [np.nan, 5, np.nan], [np.nan, np.nan, 9]],
            [[np.nan, np.nan, 3], [4, np.nan, np.nan], [np.nan, 8, np.nan]]
        ], dtype=np.float32)
        
        result = compositor.process(data)
        composite = result.metadata["composite"]
        
        # Check that gaps are filled
        assert not np.any(np.isnan(composite))
        # Check specific filled values
        assert composite[0, 0] == 1  # Original value
        assert composite[1, 1] == 5  # Original value
        assert composite[2, 2] == 9  # Original value


class TestSceneMerger:
    """Test SceneMerger blending methods."""
    
    def test_simple_mosaic_no_overlap(self):
        """Test merging scenes with no overlap."""
        merger = SceneMerger(blend_method="max")
        
        # Create two non-overlapping scenes
        scene1 = np.ones((2, 2), dtype=np.float32)
        scene2 = np.full((2, 2), 2.0, dtype=np.float32)
        
        from rasterio.transform import from_bounds
        transform1 = from_bounds(0, 0, 2, 2, 2, 2)
        transform2 = from_bounds(2, 0, 4, 2, 2, 2)
        
        bounds1 = (0, 0, 2, 2)
        bounds2 = (2, 0, 4, 2)
        
        merged, merged_transform, merged_bounds = merger.merge_scenes(
            [scene1, scene2],
            [transform1, transform2],
            [bounds1, bounds2],
            nodata=np.nan
        )
        
        # Should create 2x4 mosaic
        assert merged.shape == (2, 4)
        assert np.all(merged[:, :2] == 1)
        assert np.all(merged[:, 2:] == 2)
        assert merged_bounds == (0, 0, 4, 2)
        
    def test_max_merge_with_overlap(self):
        """Test max merging with overlapping scenes."""
        merger = SceneMerger(blend_method="max")
        
        # Create overlapping scenes
        scene1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
        scene2 = np.array([[5, 6], [7, 8]], dtype=np.float32)
        
        from rasterio.transform import from_bounds
        transform1 = from_bounds(0, 0, 2, 2, 2, 2)
        transform2 = from_bounds(1, 0, 3, 2, 2, 2)  # Overlaps by 1 unit
        
        bounds1 = (0, 0, 2, 2)
        bounds2 = (1, 0, 3, 2)
        
        merged, _, _ = merger.merge_scenes(
            [scene1, scene2],
            [transform1, transform2],
            [bounds1, bounds2],
            nodata=np.nan
        )
        
        # In overlap region, should take max values
        assert merged.shape == (2, 3)
        assert merged[0, 0] == 1  # From scene1
        assert merged[0, 1] == 5  # Max of 2 and 5
        assert merged[0, 2] == 6  # From scene2
        
    def test_min_merge(self):
        """Test min merging method."""
        merger = SceneMerger(blend_method="min")
        
        scene1 = np.array([[10, 20], [30, 40]], dtype=np.float32)
        scene2 = np.array([[5, 15], [25, 35]], dtype=np.float32)
        
        from rasterio.transform import from_bounds
        transform = from_bounds(0, 0, 2, 2, 2, 2)
        bounds = (0, 0, 2, 2)
        
        merged, _, _ = merger.merge_scenes(
            [scene1, scene2],
            [transform, transform],
            [bounds, bounds],
            nodata=np.nan
        )
        
        # Should take minimum values
        expected = np.array([[5, 15], [25, 35]], dtype=np.float32)
        assert np.array_equal(merged, expected)
        
    def test_mean_merge(self):
        """Test mean merging method."""
        merger = SceneMerger(blend_method="mean")
        
        scene1 = np.array([[10, 20], [30, 40]], dtype=np.float32)
        scene2 = np.array([[20, 30], [40, 50]], dtype=np.float32)
        scene3 = np.array([[30, 40], [50, 60]], dtype=np.float32)
        
        from rasterio.transform import from_bounds
        transform = from_bounds(0, 0, 2, 2, 2, 2)
        bounds = (0, 0, 2, 2)
        
        merged, _, _ = merger.merge_scenes(
            [scene1, scene2, scene3],
            [transform, transform, transform],
            [bounds, bounds, bounds],
            nodata=np.nan
        )
        
        # Should average all scenes
        expected = np.array([[20, 30], [40, 50]], dtype=np.float32)
        assert np.allclose(merged, expected)
        
    def test_feather_blend(self):
        """Test feather blending method."""
        merger = SceneMerger(
            blend_method="feather",
            feather_distance=2
        )
        
        # Create scenes with clear edge
        scene1 = np.ones((4, 4), dtype=np.float32) * 100
        scene2 = np.ones((4, 4), dtype=np.float32) * 200
        
        from rasterio.transform import from_bounds
        transform1 = from_bounds(0, 0, 4, 4, 4, 4)
        transform2 = from_bounds(2, 0, 6, 4, 4, 4)  # 2-unit overlap
        
        bounds1 = (0, 0, 4, 4)
        bounds2 = (2, 0, 6, 4)
        
        merged, _, _ = merger.merge_scenes(
            [scene1, scene2],
            [transform1, transform2],
            [bounds1, bounds2],
            nodata=np.nan
        )
        
        # Check feathering in overlap region
        assert merged.shape == (4, 6)
        # Non-overlap regions should have original values
        assert np.all(merged[:, 0] == 100)
        assert np.all(merged[:, -1] == 200)
        # Overlap region should have blended values
        assert np.any((merged[:, 2:4] > 100) & (merged[:, 2:4] < 200))
        
    def test_priority_order(self):
        """Test scene priority ordering."""
        merger = SceneMerger(
            blend_method="max",
            priority_order=[1, 0]  # Process scene 1 first
        )
        
        scene1 = np.ones((2, 2), dtype=np.float32) * 100
        scene2 = np.ones((2, 2), dtype=np.float32) * 50
        
        from rasterio.transform import from_bounds
        transform = from_bounds(0, 0, 2, 2, 2, 2)
        bounds = (0, 0, 2, 2)
        
        merged, _, _ = merger.merge_scenes(
            [scene1, scene2],
            [transform, transform],
            [bounds, bounds],
            nodata=np.nan
        )
        
        # With max merge, should still get 100 (max value)
        assert np.all(merged == 100)
        
    def test_multiband_merge(self):
        """Test merging multi-band scenes."""
        merger = SceneMerger(blend_method="max")
        
        # Create 2-band scenes
        scene1 = np.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ], dtype=np.float32)
        
        scene2 = np.array([
            [[2, 3], [4, 5]],
            [[6, 7], [8, 9]]
        ], dtype=np.float32)
        
        from rasterio.transform import from_bounds
        transform = from_bounds(0, 0, 2, 2, 2, 2)
        bounds = (0, 0, 2, 2)
        
        merged, _, _ = merger.merge_scenes(
            [scene1, scene2],
            [transform, transform],
            [bounds, bounds],
            nodata=np.nan
        )
        
        assert merged.shape == (2, 2, 2)  # 2 bands
        # Should take max per band
        assert np.array_equal(merged[0], [[2, 3], [4, 5]])
        assert np.array_equal(merged[1], [[6, 7], [8, 9]])
        
    def test_nodata_handling_in_merge(self):
        """Test proper handling of nodata in merging."""
        merger = SceneMerger(blend_method="mean")
        
        # Create scenes with nodata
        scene1 = np.array([[1, -999], [3, 4]], dtype=np.float32)
        scene2 = np.array([[5, 6], [-999, 8]], dtype=np.float32)
        
        from rasterio.transform import from_bounds
        transform = from_bounds(0, 0, 2, 2, 2, 2)
        bounds = (0, 0, 2, 2)
        
        merged, _, _ = merger.merge_scenes(
            [scene1, scene2],
            [transform, transform],
            [bounds, bounds],
            nodata=-999
        )
        
        # Mean should ignore nodata
        assert merged[0, 0] == 3  # mean of 1 and 5
        assert merged[0, 1] == 6  # only scene2 has valid data
        assert merged[1, 0] == 3  # only scene1 has valid data
        assert merged[1, 1] == 6  # mean of 4 and 8


class TestDateRangeSelector:
    """Test DateRangeSelector filtering logic."""
    
    def test_date_range_filtering(self):
        """Test filtering by date range."""
        selector = DateRangeSelector(
            date_range=(datetime(2023, 1, 5), datetime(2023, 1, 15))
        )
        
        scene_dates = [
            datetime(2023, 1, 1),   # Before range
            datetime(2023, 1, 5),   # Start of range
            datetime(2023, 1, 10),  # In range
            datetime(2023, 1, 15),  # End of range
            datetime(2023, 1, 20),  # After range
        ]
        
        selected = selector.select_scenes(scene_dates)
        
        assert selected == [1, 2, 3]  # Indices of scenes in range
        
    def test_target_date_filtering(self):
        """Test filtering by distance from target date."""
        selector = DateRangeSelector(
            target_date=datetime(2023, 1, 10),
            max_days_from_target=5
        )
        
        scene_dates = [
            datetime(2023, 1, 1),   # 9 days before
            datetime(2023, 1, 5),   # 5 days before
            datetime(2023, 1, 10),  # Target date
            datetime(2023, 1, 15),  # 5 days after
            datetime(2023, 1, 20),  # 10 days after
        ]
        
        selected = selector.select_scenes(scene_dates)
        
        # Should be sorted by distance from target: 0 days, then 5 days each
        assert selected == [2, 1, 3]  # Sorted by distance from target
        
    def test_season_filtering(self):
        """Test filtering by season."""
        selector = DateRangeSelector(season_filter=["summer", "winter"])
        
        scene_dates = [
            datetime(2023, 1, 15),   # Winter
            datetime(2023, 4, 15),   # Spring
            datetime(2023, 7, 15),   # Summer
            datetime(2023, 10, 15),  # Fall
            datetime(2023, 12, 15),  # Winter
        ]
        
        selected = selector.select_scenes(scene_dates)
        
        assert selected == [0, 2, 4]  # Winter and summer scenes
        
    def test_cloud_cover_filtering(self):
        """Test filtering by cloud cover."""
        selector = DateRangeSelector(cloud_threshold=30.0)
        
        scene_dates = [datetime(2023, 1, i) for i in range(1, 6)]
        scene_metadata = [
            {"cloud_cover": 10},
            {"cloud_cover": 25},
            {"cloud_cover": 35},
            {"cloud_cover": 20},
            {"cloud_cover": 50},
        ]
        
        selected = selector.select_scenes(scene_dates, scene_metadata)
        
        assert selected == [0, 1, 3]  # Scenes with cloud cover <= 30
        
    def test_quality_score_sorting(self):
        """Test sorting by quality scores."""
        selector = DateRangeSelector(max_scenes=3)
        
        scene_dates = [datetime(2023, 1, i) for i in range(1, 6)]
        quality_scores = [0.7, 0.9, 0.5, 0.8, 0.6]
        
        selected = selector.select_scenes(
            scene_dates,
            quality_scores=quality_scores
        )
        
        # Should select top 3 by quality
        assert selected == [1, 3, 0]  # Indices sorted by quality
        
    def test_target_date_sorting(self):
        """Test sorting by distance from target date."""
        selector = DateRangeSelector(
            target_date=datetime(2023, 1, 10),
            max_scenes=3
        )
        
        scene_dates = [
            datetime(2023, 1, 1),   # 9 days
            datetime(2023, 1, 8),   # 2 days
            datetime(2023, 1, 10),  # 0 days
            datetime(2023, 1, 13),  # 3 days
            datetime(2023, 1, 20),  # 10 days
        ]
        
        selected = selector.select_scenes(scene_dates)
        
        # Should select closest 3 to target
        assert selected == [2, 1, 3]  # Sorted by distance
        
    def test_min_scenes_requirement(self):
        """Test relaxing constraints for minimum scenes."""
        selector = DateRangeSelector(
            date_range=(datetime(2023, 1, 1), datetime(2023, 1, 5)),
            cloud_threshold=10.0,
            min_scenes=3
        )
        
        scene_dates = [
            datetime(2023, 1, 1),
            datetime(2023, 1, 3),
            datetime(2023, 1, 10),  # Outside range
            datetime(2023, 1, 15),  # Outside range
        ]
        
        scene_metadata = [
            {"cloud_cover": 5},
            {"cloud_cover": 20},  # Above threshold
            {"cloud_cover": 5},
            {"cloud_cover": 5},
        ]
        
        selected = selector.select_scenes(scene_dates, scene_metadata)
        
        # Should relax cloud constraint to meet min_scenes
        assert len(selected) >= 2  # At least scenes in date range
        
    def test_combined_filters(self):
        """Test combining multiple filter criteria."""
        selector = DateRangeSelector(
            target_date=datetime(2023, 7, 15),
            max_days_from_target=30,
            season_filter=["summer"],
            cloud_threshold=20.0,
            max_scenes=2
        )
        
        scene_dates = [
            datetime(2023, 6, 20),  # Summer, 25 days
            datetime(2023, 7, 10),  # Summer, 5 days
            datetime(2023, 7, 20),  # Summer, 5 days
            datetime(2023, 8, 10),  # Summer, 26 days
            datetime(2023, 9, 15),  # Fall, 62 days
        ]
        
        scene_metadata = [
            {"cloud_cover": 15},
            {"cloud_cover": 10},
            {"cloud_cover": 25},  # Above threshold
            {"cloud_cover": 5},
            {"cloud_cover": 10},
        ]
        
        selected = selector.select_scenes(scene_dates, scene_metadata)
        
        # Should select the 2 closest valid scenes (within filters)
        # Scene 1 (5 days) and Scene 0 (25 days) - both summer, low cloud
        assert selected == [1, 0]
        assert len(selected) == 2