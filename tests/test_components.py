"""Tests for quality control components."""

import numpy as np
import pytest
from blackmarble.components import (
    CloudMasker,
    QualityCompositor,
    MaskStatistics,
    MaskResult,
    QualityStatistics,
)
from blackmarble.components.quality import (
    AerosolMasker,
    RadianceSaturationMasker,
    VIIRSQualityMasker,
    create_landsat_qa_pipeline,
)
from blackmarble.components.base import combine_masks, dilate_mask, apply_mask
from blackmarble.config import LandsatConfig, VIIRSConfig


class TestCloudMasker:
    """Test CloudMasker component.
    
    Landsat Collection 2 QA_PIXEL bit definitions:
    - Bits 0-1: Fill (0=no, 1=yes)
    - Bit 2: Clear (0=no, 1=yes)  
    - Bit 3: Water (0=no, 1=yes)
    - Bits 4-5: Cloud shadow confidence (0=none, 1=low, 2=medium, 3=high)
    - Bits 6-7: Snow/ice confidence
    - Bits 8-9: Cloud confidence (0=none, 1=low, 2=medium, 3=high)
    - Bits 10-11: Cirrus confidence
    
    Common QA values used in tests:
    - 21824 (0101010101000000): Clear land (clear=1, water=0, no clouds/shadows)
    - 21952 (0101010111000000): Clear water (clear=1, water=1, no clouds/shadows)
    - 22280 (0101011100001000): Low confidence cloud
    - 54596 (1101010101000100): Medium confidence cloud  
    - 54852 (1101011001000100): High confidence cloud
    - 55052 (1101011110001100): High confidence cloud + shadow
    """

    def test_conservative_strategy(self):
        """Test conservative cloud masking strategy."""
        # Create test data with known QA values
        qa_pixel = np.array(
            [
                [21824, 21824, 54852],  # clear land, clear land, cloud high conf
                [21952, 22280, 55052],  # clear water, cloud low conf, cloud high conf
                [21824, 54596, 99999],  # clear land, cloud med conf, other
            ]
        )

        masker = CloudMasker(strategy="conservative")
        result = masker.process(qa_pixel)

        # Expected mask: only clear land (21824) and clear water (21952) are kept
        expected = np.array(
            [
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )

        np.testing.assert_array_equal(result.mask, expected)
        assert result.statistics["valid_pixels"] == 4
        assert result.statistics["invalid_pixels"] == 5
        assert result.statistics["valid_percentage"] == pytest.approx(44.44, rel=0.01)

        # Check categories
        assert result.statistics["categories"]["clear_land"] == 3
        assert result.statistics["categories"]["clear_water"] == 1
        assert result.statistics["categories"]["cloud_high_confidence"] == 2

    def test_permissive_strategy(self):
        """Test permissive cloud masking strategy."""
        qa_pixel = np.array(
            [
                [21824, 21824, 54852],
                [21952, 22280, 55052],
                [21824, 54596, 12345],  # 12345 is not in invalid list
            ]
        )

        masker = CloudMasker(strategy="permissive")
        result = masker.process(qa_pixel)

        # Permissive keeps everything except specific cloud values
        expected = np.array(
            [
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],  # 12345 is kept
            ]
        )

        np.testing.assert_array_equal(result.mask, expected)
        assert result.statistics["valid_pixels"] == 5

    def test_cloud_dilation(self):
        """Test cloud mask dilation."""
        qa_pixel = np.array(
            [
                [21824, 21824, 21824, 21824],
                [21824, 54852, 54852, 21824],  # clouds in middle
                [21824, 54852, 54852, 21824],
                [21824, 21824, 21824, 21824],
            ]
        )

        masker = CloudMasker(strategy="conservative", dilate_radius=1)
        result = masker.process(qa_pixel)

        # With dilation, pixels adjacent to clouds should also be masked
        # The exact result depends on the dilation implementation
        assert result.statistics["valid_pixels"] < 12  # Less than without dilation
        assert result.metadata["dilate_radius"] == 1


class TestAerosolMasker:
    """Test AerosolMasker component."""

    def test_standard_threshold(self):
        """Test standard aerosol masking."""
        qa_aerosol = np.array(
            [
                [64, 64, 192],  # low, low, medium aerosol
                [194, 196, 228],  # medium, high, high aerosol
                [100, 150, 200],  # various non-flagged values
            ]
        )

        masker = AerosolMasker(threshold="standard")
        result = masker.process(qa_aerosol)

        # Standard threshold masks: 192, 194, 196, 228
        expected = np.array(
            [
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )

        np.testing.assert_array_equal(result.mask, expected)
        assert result.statistics["valid_pixels"] == 5
        # 196 and 228 are high aerosol (not 200, which is not in invalid list)
        assert result.statistics["categories"]["high_aerosol"] == 2

    def test_strict_threshold(self):
        """Test strict aerosol masking."""
        qa_aerosol = np.array([[224, 228, 64]])  # 224 is only in strict

        masker = AerosolMasker(threshold="strict")
        result = masker.process(qa_aerosol)

        expected = np.array([[0.0, 0.0, 1.0]])
        np.testing.assert_array_equal(result.mask, expected)


class TestRadianceSaturationMasker:
    """Test RadianceSaturationMasker component."""

    def test_single_band_masking(self):
        """Test masking for a single band."""
        # QA_RADSAT bit definitions for Landsat 8-9:
        # Bit 0 (value 1) = Band 1 saturated
        # Bit 1 (value 2) = Band 2 saturated  
        # Bit 2 (value 4) = Band 3 saturated
        # Bit 3 (value 8) = Band 4 saturated
        # Bit 4 (value 16) = Band 5 saturated
        # Bit 5 (value 32) = Band 6 saturated
        # Bit 6 (value 64) = Band 7 saturated
        # Multiple bits can be set (values add)
        qa_radsat = np.array(
            [
                [0, 4, 8],      # none, B3 saturated, B4 saturated
                [12, 16, 20],   # B3+B4 (4+8), B5 saturated, B3+B5 (4+16)
                [28, 0, 4],     # B3+B4+B5 (4+8+16), none, B3 saturated
            ]
        )

        masker = RadianceSaturationMasker()

        # Test Band 3
        result_b3 = masker.process(qa_radsat, band=3)
        expected_b3 = np.array(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(result_b3.mask, expected_b3)
        assert result_b3.metadata["band"] == 3
        assert result_b3.metadata["saturated_pixels"] == 5

        # Test Band 4
        result_b4 = masker.process(qa_radsat, band=4)
        expected_b4 = np.array(
            [
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        )
        np.testing.assert_array_equal(result_b4.mask, expected_b4)
        assert result_b4.metadata["band"] == 4

    def test_all_bands_masking(self):
        """Test masking for all bands at once."""
        # Values: 0, 4, 8, 12, 16, 20, 24, 28
        # Binary: 0000, 0100, 1000, 1100, 10000, 10100, 11000, 11100
        # Bit 2 (val 4): Band 3 - set in 4, 12, 20, 28 = 4 pixels
        # Bit 3 (val 8): Band 4 - set in 8, 12, 24, 28 = 4 pixels
        # Bit 4 (val 16): Band 5 - set in 16, 20, 24, 28 = 4 pixels
        qa_radsat = np.array([[0, 4, 8, 12, 16, 20, 24, 28]])

        masker = RadianceSaturationMasker()
        result = masker.process(qa_radsat)

        # Should return 3D array with masks for each band
        assert result.mask.shape == (3, 1, 8)
        assert result.statistics["band_statistics"]["band_3"]["saturated_pixels"] == 4
        assert result.statistics["band_statistics"]["band_4"]["saturated_pixels"] == 4
        assert result.statistics["band_statistics"]["band_5"]["saturated_pixels"] == 4

    def test_nan_handling(self):
        """Test handling of NaN values in radsat data."""
        qa_radsat = np.array(
            [
                [0, 4, np.nan],
                [np.nan, 16, 20],
            ]
        )

        masker = RadianceSaturationMasker()
        result = masker.process(qa_radsat, band=3)

        # NaN pixels should be masked (0.0)
        expected = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(result.mask, expected)


class TestVIIRSQualityMasker:
    """Test VIIRSQualityMasker component."""

    def test_quality_flag_masking(self):
        """Test VIIRS quality flag masking."""
        quality_flags = np.array(
            [
                [0, 0, 1],  # high quality, high quality, poor
                [2, 3, 4],  # moderate, poor, poor
                [0, 2, 255],  # high, moderate, poor
            ]
        )

        masker = VIIRSQualityMasker()
        result = masker.process(quality_flags)

        # Default valid flags are [0, 2]
        expected = np.array(
            [
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )

        np.testing.assert_array_equal(result.mask, expected)
        assert result.statistics["valid_pixels"] == 5
        assert result.statistics["categories"]["high_quality"] == 3
        assert result.statistics["categories"]["moderate_quality"] == 2

    def test_custom_valid_flags(self):
        """Test with custom valid flags."""
        config = VIIRSConfig(quality_valid_flags=[0, 1, 2])
        quality_flags = np.array([[0, 1, 2, 3]])

        masker = VIIRSQualityMasker(config=config)
        result = masker.process(quality_flags)

        expected = np.array([[1.0, 1.0, 1.0, 0.0]])
        np.testing.assert_array_equal(result.mask, expected)


class TestQualityCompositor:
    """Test QualityCompositor component."""

    def test_and_combination(self):
        """Test AND combination of masks."""
        cloud_masker = CloudMasker(name="clouds", strategy="conservative")
        aerosol_masker = AerosolMasker(name="aerosol")

        compositor = QualityCompositor(
            components=[cloud_masker, aerosol_masker], combination_method="and"
        )

        data = {
            "clouds": np.array([[21824, 54852], [21952, 21824]]),
            "aerosol": np.array([[64, 192], [64, 64]]),
        }

        result = compositor.process(data)

        # Expected: only pixels that pass BOTH cloud and aerosol
        expected = np.array(
            [
                [1.0, 0.0],  # clear land + low aerosol, cloud + medium aerosol
                [1.0, 1.0],  # clear water + low aerosol, clear land + low aerosol
            ]
        )

        np.testing.assert_array_equal(result.mask, expected)
        assert result.statistics["valid_pixels"] == 3

    def test_or_combination(self):
        """Test OR combination of masks."""
        # Create two maskers that will have different results
        masker1 = CloudMasker(name="mask1", strategy="conservative")
        masker2 = CloudMasker(name="mask2", strategy="permissive")

        compositor = QualityCompositor(
            components=[masker1, masker2], combination_method="or"
        )

        data = {
            "mask1": np.array([[21824, 12345]]),  # clear, other
            "mask2": np.array([[21824, 12345]]),  # same data
        }

        result = compositor.process(data)

        # OR combination: pixel is kept if ANY mask says it's good
        # Conservative keeps only 21824, permissive keeps both
        expected = np.array([[1.0, 1.0]])

        np.testing.assert_array_equal(result.mask, expected)

    def test_add_component(self):
        """Test adding components dynamically."""
        compositor = QualityCompositor()
        assert len(compositor.components) == 0

        masker = CloudMasker(name="test")
        compositor.add_component(masker)

        assert len(compositor.components) == 1
        assert "test" in compositor.get_config()["components"]


class TestMaskStatistics:
    """Test MaskStatistics component."""

    def test_tracking(self):
        """Test mask statistics tracking."""
        tracker = MaskStatistics()

        # Track first mask
        mask1 = np.array([[1.0, 1.0], [0.0, 1.0]])
        result1 = tracker.process(mask1, mask_name="cloud_mask")

        assert result1.statistics["valid_pixels"] == 3
        assert len(tracker.history) == 1

        # Track second mask
        mask2 = np.array([[1.0, 0.0], [0.0, 0.0]])
        result2 = tracker.process(mask2, mask_name="aerosol_mask")

        assert len(tracker.history) == 2

        # Get summary
        summary = tracker.get_summary()
        assert summary["total_masks"] == 2
        assert summary["total_pixels_processed"] == 8
        assert summary["total_valid_pixels"] == 4
        assert summary["overall_valid_percentage"] == 50.0
        assert "cloud_mask" in summary["mask_names"]
        assert "aerosol_mask" in summary["mask_names"]

    def test_clear_history(self):
        """Test clearing history."""
        tracker = MaskStatistics()
        mask = np.ones((2, 2))

        tracker.process(mask, mask_name="test")
        assert len(tracker.history) == 1

        tracker.clear_history()
        assert len(tracker.history) == 0

        summary = tracker.get_summary()
        assert summary["total_masks"] == 0


class TestUtilityFunctions:
    """Test utility functions in base module."""

    def test_combine_masks_and(self):
        """Test AND combination of masks."""
        mask1 = np.array([[1.0, 1.0], [0.0, 1.0]])
        mask2 = np.array([[1.0, 0.0], [1.0, 1.0]])

        result = combine_masks([mask1, mask2], method="and")
        expected = np.array([[1.0, 0.0], [0.0, 1.0]])

        np.testing.assert_array_equal(result, expected)

    def test_combine_masks_or(self):
        """Test OR combination of masks."""
        mask1 = np.array([[1.0, 0.0], [0.0, 0.0]])
        mask2 = np.array([[0.0, 1.0], [0.0, 0.0]])

        result = combine_masks([mask1, mask2], method="or")
        expected = np.array([[1.0, 1.0], [0.0, 0.0]])

        np.testing.assert_array_equal(result, expected)

    def test_combine_masks_errors(self):
        """Test error handling in combine_masks."""
        # No masks
        with pytest.raises(ValueError, match="No masks provided"):
            combine_masks([])

        # Different shapes
        mask1 = np.ones((2, 2))
        mask2 = np.ones((3, 3))
        with pytest.raises(ValueError, match="shape"):
            combine_masks([mask1, mask2])

        # Invalid method
        with pytest.raises(ValueError, match="Unknown combination method"):
            combine_masks([mask1], method="invalid")

    def test_apply_mask(self):
        """Test applying mask to data."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[1.0, 0.0], [1.0, 1.0]])

        result = apply_mask(data, mask, fill_value=-999)
        expected = np.array([[1.0, -999], [3.0, 4.0]])

        np.testing.assert_array_equal(result, expected)

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


class TestIntegration:
    """Integration tests for complete pipelines."""

    def test_landsat_qa_pipeline(self):
        """Test creating and using a standard Landsat QA pipeline."""
        pipeline = create_landsat_qa_pipeline(strategy="conservative", dilate_clouds=1)

        # Prepare test data
        data = {
            "qa_pixel": np.array(
                [
                    [21824, 21824, 54852],
                    [21952, 22280, 55052],
                    [21824, 54596, 21824],
                ]
            ),
            "qa_aerosol": np.array(
                [
                    [64, 64, 192],
                    [194, 196, 228],
                    [100, 150, 64],
                ]
            ),
        }

        result = pipeline.process(data)

        # Should have combined mask
        assert result.mask.shape == (3, 3)
        assert result.statistics["valid_pixels"] < 9  # Some pixels masked
        assert "qa_pixel" in result.metadata["component_statistics"]
        assert "qa_aerosol" in result.metadata["component_statistics"]

    def test_quality_statistics_creation(self):
        """Test QualityStatistics creation from mask."""
        mask = np.array(
            [
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ]
        )

        stats = QualityStatistics.from_mask(mask)

        assert stats.total_pixels == 6
        assert stats.valid_pixels == 4
        assert stats.invalid_pixels == 2
        assert stats.valid_percentage == pytest.approx(66.67, rel=0.01)
