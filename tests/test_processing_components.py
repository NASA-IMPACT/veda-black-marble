"""Tests for processing module functional API."""

import numpy as np
import pytest
from datetime import datetime
from rasterio.transform import Affine
from tests.qa_constants import (
    QA_CLEAR_LAND,
    QA_CLEAR_WATER,
    QA_LOW_CLOUD,
    QA_MEDIUM_CLOUD,
    QA_MEDIUM_CLOUD_ALT,
    QA_HIGH_CLOUD,
    QA_DILATED_CLOUD,
    QA_AEROSOL_VALID,
    QA_AEROSOL_MEDIUM,
    QA_AEROSOL_HIGH,
    QA_AEROSOL_192,
    QA_AEROSOL_194,
    QA_AEROSOL_196,
    QA_AEROSOL_224,
    QA_AEROSOL_228,
)
from blackmarble.processing import (
    create_cloud_mask,
    create_aerosol_mask,
    create_saturation_mask,
    create_viirs_quality_mask,
    create_quality_mask,
    create_water_mask,
    combine_masks,
    dilate_mask,
    enhance_contrast,
    normalize_data,
    scale_to_byte,
    apply_colormap,
    enhance_roads,
    enhance_urban_features,
    create_temporal_composite,
    merge_scenes,
    select_by_date_range,
    composite_with_quality,
    compute_statistics,
    resample_image,
    reproject_image,
    crop_to_bounds,
    align_rasters,
)


class TestCloudMask:
    """Test cloud masking functions."""

    def test_conservative_strategy(self):
        """Test conservative cloud masking strategy."""
        # Create test data with known QA values
        qa_pixel = np.array(
            [
                [QA_CLEAR_LAND, QA_CLEAR_LAND, QA_HIGH_CLOUD],
                [QA_CLEAR_WATER, QA_LOW_CLOUD, QA_DILATED_CLOUD],
                [QA_CLEAR_LAND, QA_MEDIUM_CLOUD, 99999],
            ]
        )

        mask = create_cloud_mask(qa_pixel, conservative=True)

        # Expected mask: only clear land and clear water are kept
        expected = np.array(
            [
                [True, True, False],
                [True, False, False],
                [True, False, False],
            ]
        )

        np.testing.assert_array_equal(mask, expected)
        assert np.sum(mask) == 4  # 4 valid pixels

    def test_permissive_strategy(self):
        """Test permissive cloud masking strategy."""
        qa_pixel = np.array(
            [
                [QA_CLEAR_LAND, QA_CLEAR_LAND, QA_HIGH_CLOUD],
                [QA_CLEAR_WATER, QA_LOW_CLOUD, QA_DILATED_CLOUD],
                [QA_CLEAR_LAND, QA_MEDIUM_CLOUD_ALT, 12345],
            ]
        )

        mask = create_cloud_mask(qa_pixel, conservative=False)

        # Permissive keeps everything except specific cloud values
        expected = np.array(
            [
                [True, True, False],
                [True, False, False],
                [True, False, True],  # 12345 is kept
            ]
        )

        np.testing.assert_array_equal(mask, expected)
        assert np.sum(mask) == 5

    def test_cloud_dilation(self):
        """Test cloud mask dilation."""
        qa_pixel = np.array(
            [
                [QA_CLEAR_LAND, QA_CLEAR_LAND, QA_CLEAR_LAND, QA_CLEAR_LAND],
                [QA_CLEAR_LAND, QA_HIGH_CLOUD, QA_HIGH_CLOUD, QA_CLEAR_LAND],
                [QA_CLEAR_LAND, QA_HIGH_CLOUD, QA_HIGH_CLOUD, QA_CLEAR_LAND],
                [QA_CLEAR_LAND, QA_CLEAR_LAND, QA_CLEAR_LAND, QA_CLEAR_LAND],
            ]
        )

        mask = create_cloud_mask(qa_pixel, conservative=True, dilate_pixels=1)

        # With dilation, pixels adjacent to clouds should also be masked
        assert np.sum(mask) < 12  # Less than without dilation


class TestAerosolMask:
    """Test aerosol masking functions."""

    def test_standard_threshold(self):
        """Test standard aerosol masking."""
        qa_aerosol = np.array(
            [
                [QA_AEROSOL_VALID, QA_AEROSOL_VALID, QA_AEROSOL_192],
                [QA_AEROSOL_194, QA_AEROSOL_196, QA_AEROSOL_228],
                [100, 150, 200],
            ]
        )

        mask = create_aerosol_mask(qa_aerosol, threshold="standard")

        # Standard threshold masks: 192, 194, 196, 228
        expected = np.array(
            [
                [True, True, False],
                [False, False, False],
                [True, True, True],
            ]
        )

        np.testing.assert_array_equal(mask, expected)
        assert np.sum(mask) == 5

    def test_strict_threshold(self):
        """Test strict aerosol masking."""
        qa_aerosol = np.array([[QA_AEROSOL_224, QA_AEROSOL_228, QA_AEROSOL_VALID]])

        mask = create_aerosol_mask(qa_aerosol, threshold="strict")

        expected = np.array([[False, False, True]])
        np.testing.assert_array_equal(mask, expected)


class TestRadianceSaturationMask:
    """Test radiometric saturation masking functions."""

    def test_single_band_masking(self):
        """Test masking for a single band."""
        qa_radsat = np.array(
            [
                [0, 4, 8],  # none, B3 saturated, B4 saturated
                [12, 16, 20],  # B3+B4, B5 saturated, B3+B5
                [28, 0, 4],  # B3+B4+B5, none, B3 saturated
            ]
        )

        # Test Band 3
        mask_b3 = create_saturation_mask(qa_radsat, band=3)
        expected_b3 = np.array(
            [
                [True, False, True],
                [False, True, False],
                [False, True, False],
            ]
        )
        np.testing.assert_array_equal(mask_b3, expected_b3)

        # Test Band 4
        mask_b4 = create_saturation_mask(qa_radsat, band=4)
        expected_b4 = np.array(
            [
                [True, True, False],
                [False, True, True],
                [False, True, True],
            ]
        )
        np.testing.assert_array_equal(mask_b4, expected_b4)

    def test_all_bands_masking(self):
        """Test masking for all bands at once."""
        qa_radsat = np.array([[0, 4, 8, 12, 16, 20, 24, 28]])

        mask = create_saturation_mask(qa_radsat, band=None)

        # Pixel is valid only if NO bands are saturated
        expected = np.array([[True, False, False, False, False, False, False, False]])
        np.testing.assert_array_equal(mask, expected)

    def test_nan_handling(self):
        """Test handling of NaN values in radsat data."""
        qa_radsat = np.array(
            [
                [0, 4, np.nan],
                [np.nan, 16, 20],
            ]
        )

        mask = create_saturation_mask(qa_radsat, band=3)

        # NaN pixels should be masked (False)
        expected = np.array(
            [
                [True, False, False],
                [False, True, False],
            ]
        )
        np.testing.assert_array_equal(mask, expected)


class TestVIIRSQualityMask:
    """Test VIIRS quality masking functions."""

    def test_quality_flag_masking(self):
        """Test VIIRS quality flag masking."""
        quality_flags = np.array(
            [
                [0, 0, 1],  # high quality, high quality, poor
                [2, 3, 4],  # moderate, poor, poor
                [0, 2, 255],  # high, moderate, poor
            ]
        )

        mask = create_viirs_quality_mask(quality_flags)

        # Default valid flags are [0, 2]
        expected = np.array(
            [
                [True, True, False],
                [True, False, False],
                [True, True, False],
            ]
        )

        np.testing.assert_array_equal(mask, expected)
        assert np.sum(mask) == 5

    def test_custom_valid_flags(self):
        """Test with custom valid flags."""
        quality_flags = np.array([[0, 1, 2, 3]])

        mask = create_viirs_quality_mask(quality_flags, valid_flags=[0, 1, 2])

        expected = np.array([[True, True, True, False]])
        np.testing.assert_array_equal(mask, expected)


class TestQualityMask:
    """Test combined quality masking functions."""

    def test_and_combination(self):
        """Test AND combination of masks."""
        qa_bands = {
            "qa_pixel": np.array(
                [[QA_CLEAR_LAND, QA_HIGH_CLOUD], [QA_CLEAR_WATER, QA_CLEAR_LAND]]
            ),
            "qa_aerosol": np.array(
                [
                    [QA_AEROSOL_VALID, QA_AEROSOL_192],
                    [QA_AEROSOL_VALID, QA_AEROSOL_VALID],
                ]
            ),
        }

        mask = create_quality_mask(qa_bands, combine_method="and")

        # Expected: only pixels that pass BOTH cloud and aerosol
        expected = np.array(
            [
                [True, False],  # clear land + low aerosol, cloud + medium aerosol
                [True, True],  # clear water + low aerosol, clear land + low aerosol
            ]
        )

        np.testing.assert_array_equal(mask, expected)
        assert np.sum(mask) == 3

    def test_or_combination(self):
        """Test OR combination of masks."""
        # Create masks that will have different results
        mask1 = np.array([[True, False], [False, True]])
        mask2 = np.array([[False, True], [True, False]])

        combined = combine_masks([mask1, mask2], method="or")

        # OR combination: pixel is kept if ANY mask says it's good
        expected = np.array([[True, True], [True, True]])

        np.testing.assert_array_equal(combined, expected)

    def test_multiple_qa_bands(self):
        """Test combining multiple QA band types."""
        qa_bands = {
            "qa_pixel": np.array([[QA_CLEAR_LAND, QA_CLEAR_LAND]]),
            "qa_aerosol": np.array([[QA_AEROSOL_VALID, QA_AEROSOL_192]]),
            "qa_radsat": np.array([[0, 4]]),  # no saturation, band 3 saturated
        }

        mask = create_quality_mask(
            qa_bands,
            combine_method="and",
            cloud_strategy="conservative",
            aerosol_threshold="standard",
            include_saturation=True,
            saturation_bands=[3],
        )

        # First pixel: clear + low aerosol + no saturation = True
        # Second pixel: clear + high aerosol + saturation = False
        expected = np.array([[True, False]])
        np.testing.assert_array_equal(mask, expected)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_combine_masks_and(self):
        """Test AND combination of masks."""
        mask1 = np.array([[True, True], [False, True]])
        mask2 = np.array([[True, False], [True, True]])

        result = combine_masks([mask1, mask2], method="and")
        expected = np.array([[True, False], [False, True]])

        np.testing.assert_array_equal(result, expected)

    def test_combine_masks_or(self):
        """Test OR combination of masks."""
        mask1 = np.array([[True, False], [False, False]])
        mask2 = np.array([[False, True], [False, False]])

        result = combine_masks([mask1, mask2], method="or")
        expected = np.array([[True, True], [False, False]])

        np.testing.assert_array_equal(result, expected)

    def test_combine_masks_errors(self):
        """Test error handling in combine_masks."""
        # No masks
        with pytest.raises(ValueError, match="No masks provided"):
            combine_masks([])

        # Different shapes
        mask1 = np.ones((2, 2), dtype=bool)
        mask2 = np.ones((3, 3), dtype=bool)
        with pytest.raises(ValueError, match="shape"):
            combine_masks([mask1, mask2])

        # Invalid method
        with pytest.raises(ValueError, match="Unknown combination method"):
            combine_masks([mask1], method="invalid")

    def test_dilate_mask(self):
        """Test mask dilation."""
        # Simple mask with one bad pixel in center
        mask = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )

        # Dilate with radius 1
        result = dilate_mask(mask, radius=1)

        # Center and adjacent pixels should now be masked
        assert pytest.approx(result[1, 1], abs=1e-8) == 0.0  # Center stays bad
        assert np.sum(result == 0.0) > 1  # More pixels are bad after dilation

        # Test radius 0 (no dilation)
        result_no_dilate = dilate_mask(mask, radius=0)
        np.testing.assert_array_equal(result_no_dilate, mask)


class TestEnhancementFunctions:
    """Test data enhancement functions."""

    def test_enhance_contrast(self):
        """Test contrast enhancement."""
        data = np.array([[0.1, 0.2, 0.3], [0.4, 0.8, 0.9]])

        # Percentile enhancement
        enhanced = enhance_contrast(
            data, method="percentile", clip_percentiles=(10, 90)
        )

        # Should be scaled to 0-1 range
        assert enhanced.min() >= 0
        assert enhanced.max() <= 1
        assert enhanced.shape == data.shape

    def test_normalize_data(self):
        """Test data normalization."""
        data = np.array([0, 50, 100])
        normalized = normalize_data(data, 0, 1)

        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(normalized, expected)

    def test_scale_to_byte(self):
        """Test scaling to byte values."""
        data = np.array([[0.0, 0.5], [1.0, np.nan]])
        byte_data = scale_to_byte(data)

        assert byte_data.dtype == np.uint8
        assert byte_data[0, 0] == 0
        assert byte_data[0, 1] == 128  # 0.5 * 255 = 127.5, rounds to 128
        assert byte_data[1, 0] == 255
        assert byte_data[1, 1] == 0  # NaN becomes nodata_value

    def test_apply_colormap(self):
        """Test colormap application."""
        data = np.array([[0.0, 0.5], [0.7, 1.0]])
        rgb = apply_colormap(data, colormap="viridis")

        assert rgb.shape == (3, 2, 2)  # RGB channels
        assert rgb.dtype == np.uint8
        assert np.all(rgb >= 0) and np.all(rgb <= 255)

    def test_enhance_roads(self):
        """Test road enhancement."""
        data = np.array([[0.5, 0.6], [0.4, 0.7]])
        roads = np.array([[1, 0], [2, 0]])  # motorway, no road, primary, no road

        enhanced = enhance_roads(data, roads)

        # Major road (motorway) should be enhanced more than minor road (primary)
        assert enhanced[0, 0] > data[0, 0]  # Motorway enhanced
        assert enhanced[1, 0] > data[1, 0]  # Primary enhanced
        assert enhanced[0, 1] == data[0, 1]  # No road, no change
        assert enhanced[0, 0] > enhanced[1, 0]  # Motorway > Primary

    def test_enhance_urban_features(self):
        """Test urban feature enhancement."""
        ndui = np.array([[0.3, 0.5], [0.6, 0.8]])
        roads = np.array([[1, 0], [0, 2]])

        enhanced = enhance_urban_features(ndui, roads, enhancement_strength=0.5)

        # Areas with roads should be enhanced
        assert enhanced[0, 0] > ndui[0, 0]  # Motorway area enhanced
        assert enhanced[1, 1] > ndui[1, 1]  # Primary road area enhanced
        assert np.all(enhanced >= 0) and np.all(enhanced <= 1)


class TestTemporalFunctions:
    """Test temporal processing functions."""

    def test_create_temporal_composite(self):
        """Test temporal composite creation."""
        # Create 5 dates with some NaN values
        data = np.array(
            [
                [[1, 2], [3, 4]],  # Date 1
                [[2, 3], [4, 5]],  # Date 2
                [[3, np.nan], [5, 6]],  # Date 3 with NaN
                [[4, 5], [6, 7]],  # Date 4
                [[5, 6], [7, 8]],  # Date 5
            ]
        )

        # Test median composite
        composite = create_temporal_composite(data, method="median")
        assert composite.shape == (2, 2)
        assert composite[0, 0] == 3  # Median of [1,2,3,4,5]
        assert composite[0, 1] == 4  # Median of [2,3,5,6] (NaN excluded)

        # Test mean composite
        composite = create_temporal_composite(data, method="mean")
        assert composite[0, 0] == 3  # Mean of [1,2,3,4,5]

        # Test percentile composite
        composite = create_temporal_composite(data, method="percentile", percentile=75)
        assert composite[0, 0] == 4  # 75th percentile of [1,2,3,4,5]

    def test_merge_scenes(self):
        """Test scene merging."""
        # Create two overlapping scenes
        scene1 = np.ones((100, 100)) * 0.5
        scene2 = np.ones((100, 100)) * 0.7

        transform1 = Affine.identity()
        transform2 = Affine.translation(50, 0)  # Offset by 50 pixels

        bounds1 = (0, 0, 100, 100)
        bounds2 = (50, 0, 150, 100)

        merged, transform, bounds = merge_scenes(
            [scene1, scene2],
            [transform1, transform2],
            [bounds1, bounds2],
            method="mean",
        )

        # Check output dimensions
        assert merged.shape == (100, 150)  # Width should be 150
        assert bounds == (0, 0, 150, 100)

        # Check values
        assert merged[50, 25] == 0.5  # Only scene1
        assert merged[50, 125] == 0.7  # Only scene2
        # Overlap region should be average
        assert pytest.approx(merged[50, 75], rel=0.01) == 0.6

    def test_select_by_date_range(self):
        """Test scene selection by date."""
        scenes = [np.ones((10, 10)) for _ in range(10)]
        dates = [datetime(2023, 1, i) for i in range(1, 11)]
        target = datetime(2023, 1, 5)

        # Select within 2 days of target
        selected, sel_dates, indices = select_by_date_range(
            scenes, dates, target_date=target, max_days_from_target=2
        )

        assert len(selected) == 5  # Days 3,4,5,6,7
        assert sel_dates[0] == datetime(2023, 1, 5)  # Closest first
        assert indices[0] == 4  # 0-based index for day 5

    def test_composite_with_quality(self):
        """Test quality-weighted compositing."""
        data = np.array(
            [
                [[0.5, 0.6], [0.7, 0.8]],  # Date 1
                [[0.4, 0.5], [0.6, 0.7]],  # Date 2
                [[0.6, 0.7], [0.8, 0.9]],  # Date 3
            ]
        )

        quality = np.array(
            [
                [[0.3, 0.8], [0.5, 0.2]],  # Date 1 quality
                [[0.9, 0.2], [0.3, 0.7]],  # Date 2 quality
                [[0.5, 0.6], [0.8, 0.4]],  # Date 3 quality
            ]
        )

        # Test best quality selection
        composite = composite_with_quality(data, quality, method="best_quality")

        # Each pixel should come from the date with highest quality
        assert composite[0, 0] == 0.4  # Date 2 has quality 0.9
        assert composite[0, 1] == 0.6  # Date 1 has quality 0.8
        assert composite[1, 0] == 0.8  # Date 3 has quality 0.8
        assert composite[1, 1] == 0.7  # Date 2 has quality 0.7


class TestStatisticsFunctions:
    """Test statistics computation functions."""

    def test_compute_statistics(self):
        """Test statistical computation."""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, np.nan], [6.0, 7.0, 8.0]])

        stats = compute_statistics(data, percentiles=[25, 50, 75])

        assert stats["count"] == 8  # 9 - 1 NaN
        assert stats["mean"] == 4.5
        assert stats["min"] == 1.0
        assert stats["max"] == 8.0
        assert stats["median"] == 4.5
        assert stats["p25"] == 2.75
        assert stats["p75"] == 6.25

    def test_compute_statistics_with_mask(self):
        """Test statistics with mask."""
        data = np.array([[1, 2], [3, 4]])
        mask = np.array([[True, False], [True, True]])

        stats = compute_statistics(data, mask=mask)

        assert stats["count"] == 3  # Only masked values
        assert stats["mean"] == pytest.approx(8 / 3)  # (1+3+4)/3


class TestSpatialTransformations:
    """Test spatial transformation functions."""

    def test_resample_image(self):
        """Test image resampling."""
        # Create test data
        data = np.ones((100, 100)) * 0.5
        data[40:60, 40:60] = 1.0  # Bright square in center

        # Resample from 30m to 60m resolution
        resampled, new_transform = resample_image(
            data, source_resolution=30, target_resolution=60, method="bilinear"
        )

        # Check dimensions (should be half)
        assert resampled.shape == (50, 50)

        # Check that bright area is preserved (roughly)
        assert np.max(resampled) > 0.7

    def test_crop_to_bounds(self):
        """Test cropping to bounds."""
        data = np.ones((100, 100))
        transform = Affine(1, 0, 0, 0, -1, 100)  # 1m pixels, origin at (0,100)

        # Crop to center region
        bounds = (25, 25, 75, 75)
        cropped, new_transform, actual_bounds = crop_to_bounds(data, transform, bounds)

        assert cropped.shape == (50, 50)
        assert actual_bounds[0] >= bounds[0]
        assert actual_bounds[2] <= bounds[2]

    def test_align_rasters(self):
        """Test raster alignment."""
        # Create two rasters with different transforms
        raster1 = np.ones((100, 100)) * 0.5
        raster2 = np.ones((80, 80)) * 0.7

        transform1 = Affine.identity()
        transform2 = Affine(
            1.25, 0, 10, 0, -1.25, 90
        )  # Different resolution and offset

        aligned = align_rasters(
            [raster1, raster2], [transform1, transform2], reference_idx=0
        )

        # Both should now have same shape as reference
        assert aligned[0][0].shape == raster1.shape
        assert aligned[1][0].shape == raster1.shape

        # Transforms should match
        assert aligned[0][1] == aligned[1][1]


class TestWaterMask:
    """Test water masking functions."""

    def test_create_water_mask(self):
        """Test NDWI-based water masking."""
        # Simulate NDWI values
        ndwi = np.array(
            [
                [-0.2, 0.1, 0.3],  # land, water, water
                [0.0, -0.1, 0.2],  # threshold, land, water
                [np.nan, -0.3, 0.4],  # invalid, land, water
            ]
        )

        mask = create_water_mask(ndwi, threshold=0.0)

        # True = land (valid), False = water (masked)
        expected = np.array(
            [
                [True, False, False],
                [True, True, False],
                [False, True, False],  # NaN is masked
            ]
        )

        np.testing.assert_array_equal(mask, expected)


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_landsat_qa_pipeline(self):
        """Test creating and using a standard Landsat QA pipeline."""
        # Prepare test data
        qa_bands = {
            "qa_pixel": np.array(
                [
                    [21824, 21824, 54852],
                    [QA_CLEAR_WATER, QA_LOW_CLOUD, QA_DILATED_CLOUD],
                    [21824, 54596, 21824],
                ]
            ),
            "qa_aerosol": np.array(
                [
                    [QA_AEROSOL_VALID, QA_AEROSOL_VALID, QA_AEROSOL_192],
                    [QA_AEROSOL_194, QA_AEROSOL_196, QA_AEROSOL_228],
                    [100, 150, 64],
                ]
            ),
        }

        # Create combined quality mask
        mask = create_quality_mask(
            qa_bands,
            combine_method="and",
            cloud_strategy="conservative",
            cloud_dilate=1,
            aerosol_threshold="standard",
        )

        # Should have combined mask
        assert mask.shape == (3, 3)
        assert np.sum(mask) < 9  # Some pixels masked

    def test_temporal_processing_workflow(self):
        """Test complete temporal processing workflow."""
        # Create 5 synthetic scenes
        scenes = []
        dates = []
        quality_scores = []

        for i in range(5):
            scene = np.random.rand(100, 100) * (0.5 + i * 0.1)
            scenes.append(scene)
            dates.append(datetime(2023, 1, i + 1))

            # Quality decreases with cloud cover
            quality = np.ones((100, 100)) * (1.0 - i * 0.15)
            quality_scores.append(quality)

        # Select best scenes
        target_date = datetime(2023, 1, 3)
        selected, sel_dates, indices = select_by_date_range(
            scenes, dates, target_date=target_date, max_days_from_target=1
        )

        assert len(selected) == 3  # Days 2, 3, 4

        # Create quality-weighted composite
        selected_quality = [quality_scores[i] for i in indices]
        data_stack = np.array(selected)
        quality_stack = np.array(selected_quality)

        composite = composite_with_quality(
            data_stack, quality_stack, method="weighted_mean"
        )

        assert composite.shape == (100, 100)
        assert np.all(~np.isnan(composite))

        # Compute statistics
        stats = compute_statistics(composite, percentiles=[10, 90])

        assert "mean" in stats
        assert "p10" in stats
        assert "p90" in stats
