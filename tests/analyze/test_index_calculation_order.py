"""Test that index calculation order produces mathematically correct results."""

import numpy as np

from blackmarble.analyze.indices import calculate_ndvi
from blackmarble.analyze.temporal import create_index_temporal_composite


def test_median_of_ratios_vs_ratio_of_medians():
    """Verify that median(ratios) ≠ ratio(medians) for spectral indices."""

    # Create synthetic data representing 3 dates with different vegetation states
    # Date 1: Dense vegetation (summer)
    nir_date1 = np.array([[0.45, 0.50], [0.48, 0.52]])
    red_date1 = np.array([[0.05, 0.06], [0.04, 0.07]])

    # Date 2: Sparse vegetation (winter)
    nir_date2 = np.array([[0.25, 0.20], [0.22, 0.18]])
    red_date2 = np.array([[0.15, 0.12], [0.14, 0.10]])

    # Date 3: Medium vegetation (spring)
    nir_date3 = np.array([[0.35, 0.38], [0.36, 0.40]])
    red_date3 = np.array([[0.10, 0.12], [0.11, 0.13]])

    # Stack the bands
    nir_stack = np.stack([nir_date1, nir_date2, nir_date3])
    red_stack = np.stack([red_date1, red_date2, red_date3])

    # Method 1: Calculate median of bands first, then NDVI (WRONG)
    nir_composite = np.median(nir_stack, axis=0)
    red_composite = np.median(red_stack, axis=0)
    ndvi_from_medians = calculate_ndvi(nir_composite, red_composite)

    # Method 2: Calculate NDVI per date, then median (CORRECT)
    ndvi_stack = []
    for i in range(3):
        ndvi = calculate_ndvi(nir_stack[i], red_stack[i])
        ndvi_stack.append(ndvi)
    ndvi_median_correct = create_index_temporal_composite(ndvi_stack, method="median")

    # Verify they are different
    assert not np.allclose(ndvi_from_medians, ndvi_median_correct), (
        "median(NDVI) should differ from NDVI(median(NIR), median(RED))"
    )

    # Calculate the difference
    diff = np.abs(ndvi_median_correct - ndvi_from_medians)
    max_diff = np.max(diff)

    # The difference should be significant (>1%)
    assert max_diff > 0.01, f"Maximum difference {max_diff:.4f} is too small"

    # Print results for inspection
    print(f"\nNDVI from median bands (incorrect):\n{ndvi_from_medians}")
    print(f"\nMedian of NDVI values (correct):\n{ndvi_median_correct}")
    print(f"\nMaximum difference: {max_diff:.4f}")
    print(f"Mean difference: {np.mean(diff):.4f}")


def test_low_denominator_nan_behavior():
    """Test that mask_low_denom returns NaN for low denominators."""

    # Create data with some very low denominators
    nir = np.array([[0.005, 0.30], [0.001, 0.40]])
    red = np.array([[0.005, 0.10], [0.0005, 0.15]])

    # Calculate with NaN masking
    ndvi_masked = calculate_ndvi(nir, red, epsilon=0.02, mask_low_denom=True)

    # First two pixels should be NaN (denominator < 0.02)
    assert np.isnan(ndvi_masked[0, 0]), "Low denominator pixel should be NaN"
    assert np.isnan(ndvi_masked[1, 0]), "Low denominator pixel should be NaN"

    # Other pixels should be valid
    assert not np.isnan(ndvi_masked[0, 1]), "Normal pixel should not be NaN"
    assert not np.isnan(ndvi_masked[1, 1]), "Normal pixel should not be NaN"

    # Calculate without NaN masking (legacy behavior)
    ndvi_legacy = calculate_ndvi(nir, red, epsilon=0.02, mask_low_denom=False)

    # No pixels should be NaN
    assert not np.any(np.isnan(ndvi_legacy)), "Legacy mode should not produce NaN"


def test_percentile_composite():
    """Test that percentile compositing works correctly."""

    # Create synthetic NDVI time series with seasonal variation
    # 10 dates across a year
    dates = 10
    rows, cols = 5, 5

    ndvi_stack = []
    for i in range(dates):
        # Simulate seasonal variation (higher in summer)
        seasonal_factor = 0.3 + 0.4 * np.sin(2 * np.pi * i / dates)

        # Add some spatial variation
        ndvi = np.random.uniform(seasonal_factor - 0.1, seasonal_factor + 0.1, (rows, cols))
        ndvi_stack.append(ndvi)

    # Calculate different composites
    median_composite = create_index_temporal_composite(ndvi_stack, method="median")
    mean_composite = create_index_temporal_composite(ndvi_stack, method="mean")
    p85_composite = create_index_temporal_composite(
        ndvi_stack, method="percentile", percentile=85.0
    )

    # 85th percentile should be higher than median
    assert np.all(p85_composite >= median_composite - 1e-6), "85th percentile should be >= median"

    # Mean should be between median and 85th percentile (usually)
    mean_val = np.mean(mean_composite)
    median_val = np.mean(median_composite)
    p85_val = np.mean(p85_composite)

    print("\nComposite statistics:")
    print(f"  Median: {median_val:.3f}")
    print(f"  Mean: {mean_val:.3f}")
    print(f"  85th percentile: {p85_val:.3f}")

    # Verify ordering (approximately)
    assert median_val <= p85_val, "Median should be <= 85th percentile"


def test_agricultural_area_ndvi_difference():
    """Test impact on agricultural areas with strong seasonal variation."""

    # Simulate agricultural area with strong seasonal NDVI variation
    # Winter: low NDVI (0.1-0.2)
    # Summer: high NDVI (0.7-0.8)

    # Create 12 monthly observations
    nir_stack = []
    red_stack = []

    for month in range(12):
        # Seasonal pattern (peak in June-July)
        # np.exp(-((month - 6) ** 2) / 8)  # Gaussian centered on month 6

        # Summer vegetation
        if 4 <= month <= 8:
            nir = np.random.uniform(0.4, 0.5, (10, 10))
            red = np.random.uniform(0.05, 0.10, (10, 10))
        # Winter bare soil
        else:
            nir = np.random.uniform(0.15, 0.25, (10, 10))
            red = np.random.uniform(0.10, 0.15, (10, 10))

        nir_stack.append(nir)
        red_stack.append(red)

    # Method 1: Band-first (using numpy median directly)
    nir_median = np.median(np.stack(nir_stack), axis=0)
    red_median = np.median(np.stack(red_stack), axis=0)
    ndvi_legacy = calculate_ndvi(nir_median, red_median)

    # Method 2: Index-first with median
    ndvi_stack = []
    for nir, red in zip(nir_stack, red_stack, strict=True):
        ndvi = calculate_ndvi(nir, red, mask_low_denom=True)
        ndvi_stack.append(ndvi)
    ndvi_median = create_index_temporal_composite(ndvi_stack, method="median")

    # Method 3: Index-first with 85th percentile
    ndvi_p85 = create_index_temporal_composite(ndvi_stack, method="percentile", percentile=85.0)

    # Compare results
    print("\nAgricultural area NDVI comparison:")
    print(f"  Band-first mean: {np.nanmean(ndvi_legacy):.3f}")
    print(f"  Index-first median mean: {np.nanmean(ndvi_median):.3f}")
    print(f"  Index-first 85th percentile mean: {np.nanmean(ndvi_p85):.3f}")

    # The 85th percentile should capture peak vegetation better
    assert np.nanmean(ndvi_p85) > np.nanmean(ndvi_median), (
        "85th percentile should show higher vegetation than median"
    )

    # Both new methods should differ from legacy
    assert not np.allclose(ndvi_legacy, ndvi_median, rtol=0.01), (
        "Index-first should differ from band-first"
    )


if __name__ == "__main__":
    # Run tests
    test_median_of_ratios_vs_ratio_of_medians()
    test_low_denominator_nan_behavior()
    test_percentile_composite()
    test_agricultural_area_ndvi_difference()
    print("\n✅ All tests passed!")
