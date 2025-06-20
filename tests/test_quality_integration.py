"""
Integration tests for quality components with the adapter pattern.

These tests demonstrate how the quality components integrate with
the Phase 2 adapter architecture for better testability.
"""

import pytest
import numpy as np
from pathlib import Path
from tests.qa_constants import (
    QA_CLEAR_LAND,
    QA_CLEAR_WATER,
    QA_LOW_CLOUD,
    QA_MEDIUM_CLOUD,
    QA_HIGH_CLOUD,
    QA_DILATED_CLOUD,
    QA_AEROSOL_MEDIUM,
    QA_AEROSOL_HIGH,
    QA_AEROSOL_192,
    QA_AEROSOL_194,
    QA_AEROSOL_196,
    QA_AEROSOL_228,
)

from blackmarble.components.quality import (
    CloudMasker,
    AerosolMasker,
    RadianceSaturationMasker,
    VIIRSQualityMasker,
    QualityCompositor,
    MaskStatistics,
    create_landsat_qa_pipeline,
)
from blackmarble.components.base import apply_mask, combine_masks


# Removed MockQualityRasterAdapter class and related mock tests - they only
# tested mock implementations rather than actual quality component logic.


def test_quality_component_composition():
    """Test composing multiple quality components."""

    # Create sample data
    shape = (50, 50)
    qa_pixel = np.full(shape, QA_CLEAR_LAND, dtype=np.uint16)  # All clear land
    qa_pixel[10:20, 10:20] = 54596  # Add cloud patch

    qa_aerosol = np.zeros(shape, dtype=np.uint16)  # All clear
    qa_aerosol[15:25, 15:25] = QA_AEROSOL_196  # Add high aerosol patch

    # Create individual components
    cloud_masker = CloudMasker(strategy="conservative")
    aerosol_masker = AerosolMasker(threshold="standard")

    # Test AND composition
    and_compositor = QualityCompositor(
        components=[cloud_masker, aerosol_masker], combination_method="and"
    )

    and_result = and_compositor.process(
        {"cloud_masker": qa_pixel, "aerosol_masker": qa_aerosol}
    )

    # Test OR composition
    or_compositor = QualityCompositor(
        components=[cloud_masker, aerosol_masker], combination_method="or"
    )

    or_result = or_compositor.process(
        {"cloud_masker": qa_pixel, "aerosol_masker": qa_aerosol}
    )

    # AND should be more restrictive than OR
    assert and_result.statistics["valid_pixels"] <= or_result.statistics["valid_pixels"]

    # Check that cloud and aerosol patches were detected
    cloud_stats = and_result.metadata["component_statistics"]["cloud_masker"]
    aerosol_stats = and_result.metadata["component_statistics"]["aerosol_masker"]

    assert cloud_stats["valid_pixels"] < shape[0] * shape[1]  # Some clouds masked
    assert aerosol_stats["valid_pixels"] < shape[0] * shape[1]  # Some aerosol masked


def test_mask_statistics_tracking():
    """Test mask statistics tracking across multiple operations."""

    tracker = MaskStatistics()

    # Create series of masks with decreasing valid pixels
    shape = (100, 100)
    masks = [
        np.ones(shape),  # All valid
        np.random.rand(*shape) > 0.2,  # ~80% valid
        np.random.rand(*shape) > 0.5,  # ~50% valid
        np.random.rand(*shape) > 0.8,  # ~20% valid
    ]

    # Track each mask
    for i, mask in enumerate(masks):
        tracker.process(mask, mask_name=f"mask_{i + 1}")

    # Get summary
    summary = tracker.get_summary()

    assert summary["total_masks"] == 4
    assert summary["total_pixels_processed"] == 4 * shape[0] * shape[1]
    assert len(summary["mask_names"]) == 4
    assert all(f"mask_{i + 1}" in summary["individual_stats"] for i in range(4))

    # Verify decreasing valid percentages
    stats = summary["individual_stats"]
    assert stats["mask_1"]["valid_percentage"] > stats["mask_2"]["valid_percentage"]
    assert stats["mask_2"]["valid_percentage"] > stats["mask_3"]["valid_percentage"]
    assert stats["mask_3"]["valid_percentage"] > stats["mask_4"]["valid_percentage"]


def test_viirs_quality_masking():
    """Test VIIRS quality masking component."""

    # Create sample VIIRS quality flags
    shape = (100, 100)
    quality_flags = np.random.choice(
        [0, 2, 1, 3],  # Good, moderate, poor quality, bad quality
        size=shape,
        p=[0.6, 0.25, 0.1, 0.05],
    ).astype(np.uint8)

    # Create VIIRS quality masker
    masker = VIIRSQualityMasker()

    # Process quality flags
    result = masker.process(quality_flags)

    # Check statistics
    stats = result.statistics
    assert stats["total_pixels"] == shape[0] * shape[1]
    assert stats["valid_pixels"] == np.sum(np.isin(quality_flags, [0, 2]))

    # Check metadata
    metadata = result.metadata
    assert "qa_value_distribution" in metadata
    assert metadata["valid_flags"] == [0, 2]

    # Verify value distribution exists
    qa_dist = metadata["qa_value_distribution"]
    assert 0 in qa_dist  # High quality value
    assert 2 in qa_dist  # Moderate quality value


def test_radiance_saturation_masking():
    """Test radiance saturation masking for individual bands."""

    # Create sample QA_RADSAT data with known bit patterns
    shape = (50, 50)
    qa_radsat = np.zeros(shape, dtype=np.uint16)

    # Set specific saturation patterns
    qa_radsat[10:20, :] = 4  # Band 3 saturated (bit 2)
    qa_radsat[20:30, :] = 8  # Band 4 saturated (bit 3)
    qa_radsat[30:40, :] = 16  # Band 5 saturated (bit 4)
    qa_radsat[40:50, :] = 28  # All bands saturated (bits 2, 3, 4)

    masker = RadianceSaturationMasker()

    # Test individual band masking
    for band_num in [3, 4, 5]:
        result = masker.process(qa_radsat, band=band_num)

        stats = result.statistics
        bit_position = band_num - 1

        # Calculate expected saturated pixels
        if band_num == 3:
            expected_saturated = 20 * 50  # rows 10-20 and 40-50
        elif band_num == 4:
            expected_saturated = 20 * 50  # rows 20-30 and 40-50
        else:  # band 5
            expected_saturated = 20 * 50  # rows 30-40 and 40-50

        assert result.metadata["saturated_pixels"] == expected_saturated
        assert stats["valid_pixels"] == shape[0] * shape[1] - expected_saturated

    # Test multi-band masking
    multi_result = masker.process(qa_radsat)

    assert multi_result.mask.shape == (3, *shape)  # 3 bands
    assert "band_statistics" in multi_result.statistics

    # Verify band statistics
    band_stats = multi_result.statistics["band_statistics"]
    assert all(f"band_{b}" in band_stats for b in [3, 4, 5])


if __name__ == "__main__":
    # Run specific test
    import sys

    adapter = MockQualityRasterAdapter()
    test_quality_components_with_mock_data(adapter)
    print("✓ Quality components with mock data test passed")

    test_quality_component_composition()
    print("✓ Quality component composition test passed")

    test_mask_statistics_tracking()
    print("✓ Mask statistics tracking test passed")

    test_viirs_quality_masking()
    print("✓ VIIRS quality masking test passed")

    test_radiance_saturation_masking()
    print("✓ Radiance saturation masking test passed")

    print("\n✅ All integration tests passed!")
