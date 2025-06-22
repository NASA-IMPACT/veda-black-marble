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

from blackmarble.processing import (
    create_cloud_mask,
    create_aerosol_mask,
    create_saturation_mask,
    create_viirs_quality_mask,
    create_quality_mask,
    combine_masks,
    compute_statistics,
)


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

    # Create individual masks
    cloud_mask = create_cloud_mask(qa_pixel, conservative=True)
    aerosol_mask = create_aerosol_mask(qa_aerosol, threshold="standard")

    # Test AND composition
    and_mask = combine_masks([cloud_mask, aerosol_mask], method="and")

    # Test OR composition
    or_mask = combine_masks([cloud_mask, aerosol_mask], method="or")

    # AND should be more restrictive than OR
    and_valid = np.sum(and_mask)
    or_valid = np.sum(or_mask)
    assert and_valid <= or_valid

    # Check that cloud and aerosol patches were detected
    cloud_stats = compute_statistics(qa_pixel, mask=cloud_mask)
    aerosol_stats = compute_statistics(qa_aerosol, mask=aerosol_mask)
    combined_stats = compute_statistics(qa_pixel, mask=and_mask)

    assert cloud_stats["count"] < shape[0] * shape[1]  # Some clouds masked
    assert aerosol_stats["count"] < shape[0] * shape[1]  # Some aerosol masked


def test_mask_statistics_tracking():
    """Test mask statistics tracking across multiple operations."""

    # Create series of masks with decreasing valid pixels
    shape = (100, 100)
    masks = [
        np.ones(shape, dtype=bool),  # All valid
        np.random.rand(*shape) > 0.2,  # ~80% valid
        np.random.rand(*shape) > 0.5,  # ~50% valid
        np.random.rand(*shape) > 0.8,  # ~20% valid
    ]

    # Track each mask
    mask_stats = []
    for i, mask in enumerate(masks):
        stats = compute_statistics(np.ones(shape), mask=mask)
        stats["mask_name"] = f"mask_{i + 1}"
        mask_stats.append(stats)

    # Create summary
    total_masks = len(mask_stats)
    total_pixels = sum(s["count"] for s in mask_stats)

    # Verify statistics
    assert total_masks == 4
    assert all(s["mask_name"] == f"mask_{i + 1}" for i, s in enumerate(mask_stats))

    # Verify decreasing valid percentages
    valid_fractions = [s["valid_fraction"] for s in mask_stats]
    assert valid_fractions[0] > valid_fractions[1]
    assert valid_fractions[1] > valid_fractions[2]
    assert valid_fractions[2] > valid_fractions[3]


def test_viirs_quality_masking():
    """Test VIIRS quality masking function."""

    # Create sample VIIRS quality flags
    shape = (100, 100)
    quality_flags = np.random.choice(
        [0, 2, 1, 3],  # Good, moderate, poor quality, bad quality
        size=shape,
        p=[0.6, 0.25, 0.1, 0.05],
    ).astype(np.uint8)

    # Process quality flags
    mask = create_viirs_quality_mask(quality_flags, valid_flags=[0, 2])

    # Check statistics
    stats = compute_statistics(quality_flags, mask=mask)
    assert stats["count"] == np.sum(np.isin(quality_flags, [0, 2]))
    assert stats["valid_fraction"] == np.sum(np.isin(quality_flags, [0, 2])) / (
        shape[0] * shape[1]
    )

    # Verify mask matches expected values
    expected_mask = np.isin(quality_flags, [0, 2])
    np.testing.assert_array_equal(mask, expected_mask)


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

    # Test individual band masking
    for band_num in [3, 4, 5]:
        mask = create_saturation_mask(qa_radsat, band=band_num)
        stats = compute_statistics(qa_radsat, mask=mask)

        # Calculate expected saturated pixels
        if band_num == 3:
            expected_saturated = 20 * 50  # rows 10-20 and 40-50
        elif band_num == 4:
            expected_saturated = 20 * 50  # rows 20-30 and 40-50
        else:  # band 5
            expected_saturated = 20 * 50  # rows 30-40 and 40-50

        assert stats["count"] == shape[0] * shape[1] - expected_saturated
        assert (
            np.sum(~mask) == expected_saturated
        )  # Inverted because mask=True means not saturated

    # Test multi-band masking (checks all bands)
    multi_mask = create_saturation_mask(qa_radsat, band=None)
    multi_stats = compute_statistics(qa_radsat, mask=multi_mask)

    # With band=None, pixel is valid only if NO bands are saturated
    # So only rows 0-10 should be valid (500 pixels)
    assert multi_stats["count"] <= 10 * 50


if __name__ == "__main__":
    # Run specific test
    import sys

    test_quality_component_composition()
    print("✓ Quality component composition test passed")

    test_mask_statistics_tracking()
    print("✓ Mask statistics tracking test passed")

    test_viirs_quality_masking()
    print("✓ VIIRS quality masking test passed")

    test_radiance_saturation_masking()
    print("✓ Radiance saturation masking test passed")

    print("\n✅ All integration tests passed!")
