"""Tests for processing module functions."""

import numpy as np
import pytest

from blackmarble.processing import (
    create_cloud_mask,
    create_quality_mask,
    create_aerosol_mask,
    create_saturation_mask,
    create_water_mask,
    create_viirs_quality_mask,
    combine_masks,
    dilate_mask,
)


class TestCloudMask:
    """Test cloud mask generation."""

    def test_conservative_strategy(self):
        """Test conservative cloud masking strategy."""
        # Create test data with clear land, clear water, and cloud pixels
        qa_pixel = np.array(
            [
                [21824, 21824, 21952],  # Clear land, clear land, clear water
                [22280, 54596, 21824],  # Low conf cloud, medium conf cloud, clear land
                [54852, 55052, 21952],  # High conf cloud, high conf cloud, clear water
            ]
        )

        mask = create_cloud_mask(qa_pixel, conservative=True)

        # Expected: only clear land (21824) and clear water (21952) are True
        expected = np.array(
            [
                [True, True, True],
                [False, False, True],
                [False, False, True],
            ]
        )

        np.testing.assert_array_equal(mask, expected)

    def test_permissive_strategy(self):
        """Test permissive cloud masking strategy."""
        # Same test data
        qa_pixel = np.array(
            [
                [21824, 21824, 21952],
                [22280, 54596, 21824],
                [54852, 55052, 21952],
            ]
        )

        mask = create_cloud_mask(qa_pixel, conservative=False)

        # Expected: only high/medium confidence clouds are masked
        expected = np.array(
            [
                [True, True, True],
                [
                    False,
                    False,
                    True,
                ],  # Low conf cloud (22280) is masked in permissive too
                [False, False, True],
            ]
        )

        np.testing.assert_array_equal(mask, expected)

    def test_cloud_mask_dilation(self):
        """Test cloud mask with dilation."""
        # Create a small cloud in the center
        qa_pixel = np.array(
            [
                [21824, 21824, 21824],
                [21824, 55052, 21824],  # Cloud in center
                [21824, 21824, 21824],
            ]
        )

        mask = create_cloud_mask(qa_pixel, conservative=True, dilate_pixels=1)

        # After dilation, all pixels around the cloud should be masked
        expected = np.array(
            [
                [False, False, False],
                [False, False, False],
                [False, False, False],
            ]
        )

        np.testing.assert_array_equal(mask, expected)


class TestAerosolMask:
    """Test aerosol mask generation."""

    def test_standard_threshold(self):
        """Test standard aerosol threshold."""
        qa_aerosol = np.array(
            [
                [0, 64, 128],  # Low aerosol values (valid)
                [192, 194, 196],  # Medium/high aerosol (invalid)
                [224, 228, 0],  # High aerosol and one valid
            ]
        )

        mask = create_aerosol_mask(qa_aerosol, threshold="standard")

        expected = np.array(
            [
                [True, True, True],
                [False, False, False],
                [True, False, True],  # 224 is valid in standard threshold
            ]
        )

        np.testing.assert_array_equal(mask, expected)

    def test_strict_threshold(self):
        """Test strict aerosol threshold."""
        qa_aerosol = np.array(
            [
                [0, 64, 128],
                [192, 194, 196],
                [224, 228, 0],
            ]
        )

        mask = create_aerosol_mask(qa_aerosol, threshold="strict")

        expected = np.array(
            [
                [True, True, True],
                [False, False, False],
                [False, False, True],  # 224 is invalid in strict threshold
            ]
        )

        np.testing.assert_array_equal(mask, expected)


class TestSaturationMask:
    """Test radiometric saturation mask generation."""

    def test_single_band_saturation(self):
        """Test saturation mask for a single band."""
        # QA_RADSAT with bit patterns
        # Band 3 = bit 2, Band 4 = bit 3, Band 5 = bit 4
        qa_radsat = np.array(
            [
                [0b00000, 0b00100, 0b01000],  # No sat, band 3 sat, band 4 sat
                [0b10000, 0b11100, 0b00010],  # Band 5 sat, all sat, other bit
            ]
        )

        # Test band 3 (bit 2)
        mask = create_saturation_mask(qa_radsat, band=3)
        expected = np.array(
            [
                [True, False, True],
                [True, False, True],
            ]
        )
        np.testing.assert_array_equal(mask, expected)

        # Test band 4 (bit 3)
        mask = create_saturation_mask(qa_radsat, band=4)
        expected = np.array(
            [
                [True, True, False],
                [True, False, True],
            ]
        )
        np.testing.assert_array_equal(mask, expected)

    def test_all_bands_saturation(self):
        """Test saturation mask for all bands."""
        qa_radsat = np.array(
            [
                [0b00000, 0b00100, 0b01000],
                [0b10000, 0b11100, 0b00010],
            ]
        )

        mask = create_saturation_mask(qa_radsat, band=None)

        # Pixel is valid only if NO bands are saturated
        expected = np.array(
            [
                [True, False, False],
                [False, False, True],  # 0b00010 has no saturation bits set
            ]
        )

        np.testing.assert_array_equal(mask, expected)

    def test_nan_handling(self):
        """Test handling of NaN values in saturation mask."""
        qa_radsat = np.array(
            [
                [0, np.nan, 4],
                [8, 16, np.nan],
            ]
        )

        mask = create_saturation_mask(qa_radsat, band=3)

        # NaN pixels should be masked (False)
        assert mask[0, 1] == False
        assert mask[1, 2] == False


class TestWaterMask:
    """Test water mask generation."""

    def test_water_detection(self):
        """Test basic water detection from NDWI."""
        # NDWI values: positive = water, negative = land
        ndwi = np.array(
            [
                [-0.5, -0.2, 0.0],
                [0.1, 0.3, -0.1],
                [0.5, -0.3, np.nan],
            ]
        )

        mask = create_water_mask(ndwi, threshold=0.0)

        # True = land (NDWI <= 0), False = water (NDWI > 0)
        expected = np.array(
            [
                [True, True, True],
                [False, False, True],
                [False, True, False],  # NaN is masked
            ]
        )

        np.testing.assert_array_equal(mask, expected)

    def test_custom_threshold(self):
        """Test water detection with custom threshold."""
        ndwi = np.array(
            [
                [-0.5, -0.2, 0.0],
                [0.1, 0.3, -0.1],
            ]
        )

        # Higher threshold means more pixels classified as land
        mask = create_water_mask(ndwi, threshold=0.2)

        expected = np.array(
            [
                [True, True, True],
                [True, False, True],  # 0.1 is now land with threshold 0.2
            ]
        )

        np.testing.assert_array_equal(mask, expected)


class TestVIIRSQualityMask:
    """Test VIIRS quality mask generation."""

    def test_default_valid_flags(self):
        """Test VIIRS quality mask with default flags."""
        quality_flag = np.array(
            [
                [0, 1, 2],  # High, unknown, moderate
                [3, 4, 0],  # Poor quality values
                [2, 5, 6],
            ]
        )

        mask = create_viirs_quality_mask(quality_flag)

        # Default valid flags are [0, 2]
        expected = np.array(
            [
                [True, False, True],
                [False, False, True],
                [True, False, False],
            ]
        )

        np.testing.assert_array_equal(mask, expected)

    def test_custom_valid_flags(self):
        """Test VIIRS quality mask with custom flags."""
        quality_flag = np.array(
            [
                [0, 1, 2],
                [3, 4, 0],
            ]
        )

        # Only accept highest quality
        mask = create_viirs_quality_mask(quality_flag, valid_flags=[0])

        expected = np.array(
            [
                [True, False, False],
                [False, False, True],
            ]
        )

        np.testing.assert_array_equal(mask, expected)


class TestCombineMasks:
    """Test mask combination functionality."""

    def test_and_combination(self):
        """Test AND combination of masks."""
        mask1 = np.array([[True, True], [False, True]])
        mask2 = np.array([[True, False], [True, True]])

        result = combine_masks([mask1, mask2], method="and")

        expected = np.array([[True, False], [False, True]])
        np.testing.assert_array_equal(result, expected)

    def test_or_combination(self):
        """Test OR combination of masks."""
        mask1 = np.array([[True, True], [False, False]])
        mask2 = np.array([[False, False], [True, False]])

        result = combine_masks([mask1, mask2], method="or")

        expected = np.array([[True, True], [True, False]])
        np.testing.assert_array_equal(result, expected)

    def test_float_masks(self):
        """Test combination of float masks."""
        mask1 = np.array([[1.0, 0.0], [1.0, 0.0]])
        mask2 = np.array([[1.0, 1.0], [0.0, 0.0]])

        result = combine_masks([mask1, mask2], method="and")

        expected = np.array([[True, False], [False, False]])
        np.testing.assert_array_equal(result, expected)

    def test_shape_mismatch_error(self):
        """Test error on shape mismatch."""
        mask1 = np.array([[True, True]])
        mask2 = np.array([[True], [True]])

        with pytest.raises(ValueError, match="shape"):
            combine_masks([mask1, mask2])

    def test_no_masks_error(self):
        """Test error when no masks provided."""
        with pytest.raises(ValueError, match="No masks"):
            combine_masks([])

    def test_invalid_method_error(self):
        """Test error on invalid combination method."""
        mask1 = np.array([[True]])

        with pytest.raises(ValueError, match="Unknown combination method"):
            combine_masks([mask1], method="invalid")


class TestDilateMask:
    """Test mask dilation functionality."""

    def test_no_dilation(self):
        """Test that radius=0 returns unchanged mask."""
        mask = np.array([[True, False], [True, True]])
        result = dilate_mask(mask, radius=0)

        np.testing.assert_array_equal(result, mask)
        assert result is not mask  # Should be a copy

    def test_simple_dilation(self):
        """Test basic dilation with radius=1."""
        # Single False pixel in center
        mask = np.array(
            [
                [True, True, True],
                [True, False, True],
                [True, True, True],
            ]
        )

        result = dilate_mask(mask, radius=1)

        # All neighbors should become False
        expected = np.array(
            [
                [False, False, False],
                [False, False, False],
                [False, False, False],
            ]
        )

        np.testing.assert_array_equal(result, expected)

    def test_float_mask_dilation(self):
        """Test dilation with float mask."""
        mask = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=float,
        )

        result = dilate_mask(mask, radius=1)

        expected = np.zeros((3, 3), dtype=float)

        np.testing.assert_array_equal(result, expected)
        assert result.dtype == float


class TestQualityMaskIntegration:
    """Test the integrated quality mask function."""

    def test_full_quality_pipeline(self):
        """Test complete quality mask with all QA bands."""
        qa_bands = {
            "qa_pixel": np.array(
                [
                    [21824, 55052, 21824],
                    [21824, 21824, 54596],
                ]
            ),
            "qa_aerosol": np.array(
                [
                    [0, 192, 0],
                    [0, 0, 194],
                ]
            ),
            "qa_radsat": np.array(
                [
                    [0, 4, 0],
                    [0, 0, 8],
                ]
            ),
        }

        mask = create_quality_mask(
            qa_bands,
            combine_method="and",
            cloud_strategy="conservative",
            aerosol_threshold="standard",
            include_saturation=True,
            saturation_bands=[3, 4],
        )

        # Pixel is valid only if ALL conditions are met
        # Row 0: [clear & low aerosol & no sat, cloud, clear & low aerosol & band 3 sat]
        # Row 1: [clear & low aerosol & no sat, clear & low aerosol & no sat, cloud]
        # Note: When saturation_bands=[3,4], each band creates a separate mask that gets ANDed
        # Position [0,2]: clear (True) & low aerosol (True) & band 3 not sat (True) & band 4 not sat (True) = True
        expected = np.array(
            [
                [True, False, True],  # [0,2] passes all checks
                [True, True, False],
            ]
        )

        np.testing.assert_array_equal(mask, expected)

    def test_partial_qa_bands(self):
        """Test quality mask with only some QA bands."""
        qa_bands = {
            "qa_pixel": np.array(
                [
                    [21824, 55052],
                    [21952, 21824],
                ]
            ),
        }

        mask = create_quality_mask(
            qa_bands,
            cloud_strategy="conservative",
            include_saturation=False,
        )

        expected = np.array(
            [
                [True, False],
                [True, True],
            ]
        )

        np.testing.assert_array_equal(mask, expected)

    def test_no_qa_bands_error(self):
        """Test error when no QA bands provided."""
        with pytest.raises(ValueError, match="No QA bands"):
            create_quality_mask({})
