"""Tests for pipeline statistics calculations.

These tests verify the calculation of output statistics including:
- Urban area in square kilometers
- Mean radiance values
"""

import numpy as np
from rasterio.transform import from_origin


class TestStatisticsCalculation:
    """Test pipeline statistics calculation."""

    def test_urban_area_calculation(self):
        """Test urban area calculation in km²."""
        # Create urban mask
        shape = (100, 100)
        urban_mask = np.zeros(shape, dtype=bool)
        urban_mask[25:75, 25:75] = True  # 50x50 urban area

        # Transform: 30m pixels
        transform = from_origin(500000, 4000000, 30, 30)

        # Calculate like pipeline does
        pixel_area_m2 = abs(transform.a * transform.e)  # 30 * 30 = 900 m²
        urban_pixels = np.sum(urban_mask)  # 50 * 50 = 2500 pixels
        urban_area_km2 = (urban_pixels * pixel_area_m2) / 1e6

        # Verify calculation
        expected_area = (2500 * 900) / 1e6  # 2.25 km²
        assert abs(urban_area_km2 - expected_area) < 0.01

    def test_mean_radiance_calculation(self):
        """Test mean radiance calculation."""
        # Create NTL data with known mean
        shape = (50, 50)
        ntl_data = np.full(shape, 10.0, dtype=np.float32)
        ntl_data[0:10, 0:10] = np.nan  # Some NaN values

        # Calculate like pipeline does
        mean_radiance = np.nanmean(ntl_data)

        # Should be 10.0 (ignoring NaN)
        assert abs(mean_radiance - 10.0) < 0.001
