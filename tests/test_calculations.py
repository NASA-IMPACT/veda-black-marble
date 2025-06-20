"""Tests for calculation functions."""

import numpy as np
import pytest
from tests.test_utils import assert_array_close, assert_close
from blackmarble.calculations import (
    calculate_ndvi,
    calculate_ndwi,
    calculate_ndui,
    apply_lunar_correction,
    scale_landsat_to_reflectance,
    enhance_with_roads,
    normalize_to_range,
)
from blackmarble.config import NDUIConfig, VIIRSConfig, LandsatConfig


class TestNDVICalculation:
    """Test NDVI calculation function."""

    def test_ndvi_vegetation(self):
        """Test NDVI for typical vegetation signatures."""
        # Healthy vegetation: high NIR, low red
        red = np.array([[0.1, 0.15], [0.12, 0.08]])
        nir = np.array([[0.5, 0.6], [0.55, 0.65]])

        ndvi = calculate_ndvi(red, nir)

        # All values should be positive for vegetation
        assert np.all(ndvi > 0)
        assert np.all(ndvi < 1)

        # Check specific value
        # NDVI = (0.5 - 0.1) / (0.5 + 0.1 + 1e-6) = 0.4 / 0.600001
        expected = (0.5 - 0.1) / (0.5 + 0.1 + 1e-6)
        assert pytest.approx(ndvi[0, 0], 0.001) == expected

    def test_ndvi_water(self):
        """Test NDVI for water bodies."""
        # Water: low NIR, higher red
        red = np.array([[0.3]])
        nir = np.array([[0.1]])

        ndvi = calculate_ndvi(red, nir)

        # Should be negative for water
        assert ndvi[0, 0] < 0
        # (0.1 - 0.3) / (0.1 + 0.3 + 1e-6) = -0.2 / 0.400001
        expected = (0.1 - 0.3) / (0.1 + 0.3 + 1e-6)
        assert pytest.approx(ndvi[0, 0], 0.001) == expected

    def test_ndvi_bare_soil(self):
        """Test NDVI for bare soil."""
        # Bare soil: similar red and NIR
        red = np.array([[0.2, 0.25]])
        nir = np.array([[0.22, 0.24]])

        ndvi = calculate_ndvi(red, nir)

        # Should be close to zero for bare soil
        assert np.all(np.abs(ndvi) < 0.2)

    def test_ndvi_clipping(self):
        """Test NDVI clipping to [-1, 1]."""
        # Create extreme values that would exceed [-1, 1]
        red = np.array([[0.0, 1.0]])
        nir = np.array([[1.0, 0.0]])

        ndvi = calculate_ndvi(red, nir)

        assert np.all(ndvi >= -1.0)
        assert np.all(ndvi <= 1.0)

    def test_ndvi_epsilon(self):
        """Test NDVI handles division by zero."""
        # Both bands zero
        red = np.array([[0.0]])
        nir = np.array([[0.0]])

        ndvi = calculate_ndvi(red, nir)

        # Should not produce inf or nan
        assert np.isfinite(ndvi[0, 0])


class TestNDWICalculation:
    """Test NDWI calculation function."""

    def test_ndwi_water(self):
        """Test NDWI for water bodies."""
        # Water: high green, low NIR
        green = np.array([[0.3, 0.4]])
        nir = np.array([[0.1, 0.15]])

        ndwi = calculate_ndwi(green, nir)

        # Should be positive for water
        assert np.all(ndwi > 0)

        # Check specific value
        # NDWI = (0.3 - 0.1) / (0.3 + 0.1) = 0.2 / 0.4 = 0.5
        assert pytest.approx(ndwi[0, 0], 0.001) == 0.5

    def test_ndwi_vegetation(self):
        """Test NDWI for vegetation."""
        # Vegetation: low green, high NIR
        green = np.array([[0.1, 0.15]])
        nir = np.array([[0.5, 0.6]])

        ndwi = calculate_ndwi(green, nir)

        # Should be negative for vegetation
        assert np.all(ndwi < 0)


class TestNDUICalculation:
    """Test NDUI calculation function."""

    def test_ndui_basic(self):
        """Test basic NDUI calculation without roads."""
        config = NDUIConfig()

        # Urban area: high NTL, low NDVI
        ntl = np.array([[0.8, 0.6]])
        ndvi = np.array([[0.1, 0.2]])
        ndwi = np.array([[-0.3, -0.2]])  # Not water
        roads = np.zeros((1, 2))

        ndui = calculate_ndui(ntl, ndvi, ndwi, roads, config)

        # Should be positive for urban areas
        assert np.all(ndui > 0)
        assert np.all(ndui <= 1)

    def test_ndui_water_masking(self):
        """Test NDUI water masking behavior."""
        config = NDUIConfig()

        ntl = np.array([[0.5, 0.5]])
        ndvi = np.array([[0.3, 0.3]])
        ndwi = np.array([[-0.2, 0.1]])  # Second pixel is water
        roads = np.zeros((1, 2))

        ndui = calculate_ndui(ntl, ndvi, ndwi, roads, config)

        # First pixel should have positive NDUI
        assert ndui[0, 0] > 0
        # Second pixel should be zero (water masked)
        assert pytest.approx(ndui[0, 1], abs=1e-8) == 0.0

    def test_ndui_floor_threshold(self):
        """Test NDUI floor threshold."""
        config = NDUIConfig(floor=0.5)  # High threshold

        # Create values where NDVI > NTL (negative NDUI)
        # This ensures transformed NDUI will be < 0.5
        ntl = np.array([[0.2, 0.1]])
        ndvi = np.array([[0.4, 0.3]])  # Higher than NTL
        ndwi = np.array([[-0.1, -0.1]])
        roads = np.zeros((1, 2))

        ndui = calculate_ndui(ntl, ndvi, ndwi, roads, config)

        # Should be zero due to floor threshold
        assert np.allclose(ndui, 0.0, atol=1e-8)

    def test_ndui_road_enhancement(self):
        """Test NDUI road enhancement."""
        config = NDUIConfig()

        # Same NTL/NDVI values
        ntl = np.array([[0.5, 0.5, 0.5]])
        ndvi = np.array([[0.2, 0.2, 0.2]])
        ndwi = np.array([[-0.1, -0.1, -0.1]])
        roads = np.array([[0, 1, 2]])  # No road, major road, minor road

        ndui = calculate_ndui(ntl, ndvi, ndwi, roads, config)

        # Road pixels should be enhanced
        assert ndui[0, 1] > ndui[0, 0]  # Major road > no road
        assert ndui[0, 2] > ndui[0, 0]  # Minor road > no road
        assert ndui[0, 1] > ndui[0, 2]  # Major road > minor road

        # Check enhancement factors
        base_value = ndui[0, 0]
        assert pytest.approx(ndui[0, 1], 0.001) == base_value * 1.3
        assert pytest.approx(ndui[0, 2], 0.001) == base_value * 1.1


class TestLunarCorrection:
    """Test lunar irradiance correction function."""

    def test_lunar_correction_basic(self):
        """Test basic lunar correction."""
        config = VIIRSConfig()

        ntl = np.array([[100.0, 200.0], [150.0, 300.0]])
        lunar = np.array([[10.0, 20.0], [15.0, 30.0]])
        quality = np.array([[0, 0], [2, 2]])  # All good quality

        corrected = apply_lunar_correction(ntl, lunar, quality, config)

        # All values should be positive and <= 1
        assert np.all(corrected >= 0)
        assert np.all(corrected <= 1)

        # Check specific calculation
        # (100 - 10) / 10 = 9.0, then 9.0 / 10 (ceiling) = 0.9
        assert pytest.approx(corrected[0, 0], 0.01) == 0.9

    def test_lunar_quality_masking(self):
        """Test quality flag masking."""
        config = VIIRSConfig()

        ntl = np.array([[100.0, 100.0], [100.0, 100.0]])
        lunar = np.array([[10.0, 10.0], [10.0, 10.0]])
        quality = np.array([[0, 1], [2, 3]])  # 0,2 good; 1,3 bad

        corrected = apply_lunar_correction(ntl, lunar, quality, config)

        # Good quality pixels should have values
        assert corrected[0, 0] > 0  # quality = 0
        assert corrected[1, 0] > 0  # quality = 2

        # Bad quality pixels should be zero
        assert pytest.approx(corrected[0, 1], abs=1e-8) == 0  # quality = 1
        assert pytest.approx(corrected[1, 1], abs=1e-8) == 0  # quality = 3

    def test_lunar_fill_values(self):
        """Test handling of fill values."""
        config = VIIRSConfig()

        ntl = np.array([[100.0, 65535.0]])  # Second is fill value
        lunar = np.array([[10.0, 65535.0]])
        quality = np.array([[0, 0]])

        corrected = apply_lunar_correction(ntl, lunar, quality, config)

        # Fill values should become zero
        assert corrected[0, 0] > 0  # Valid pixel
        assert pytest.approx(corrected[0, 1], abs=1e-8) == 0  # Fill value

    def test_lunar_floor_ceiling(self):
        """Test floor and ceiling thresholds."""
        config = VIIRSConfig(floor=0.5, ceiling=5.0)

        ntl = np.array([[4.0, 1000.0, 30.0]])  # Below floor, above ceiling, normal
        lunar = np.array([[0.0, 0.0, 0.0]])
        quality = np.array([[0, 0, 0]])

        corrected = apply_lunar_correction(ntl, lunar, quality, config)

        # Below floor -> 0
        assert pytest.approx(corrected[0, 0], abs=1e-8) == 0.0

        # Above ceiling -> normalized ceiling
        assert pytest.approx(corrected[0, 1], abs=1e-8) == 1.0

        # Normal value
        assert 0 < corrected[0, 2] < 1


class TestLandsatScaling:
    """Test Landsat DN to reflectance scaling."""

    def test_scale_dn_values(self):
        """Test scaling of DN values to reflectance."""
        config = LandsatConfig()

        # Valid DN values
        dn = np.array([[10000, 20000], [30000, 40000]])

        scaled = scale_landsat_to_reflectance(dn, config)

        # All values should be positive
        assert np.all(scaled > 0)

        # Check specific calculation
        # 10000 * 0.0000275 - 0.2 = 0.275 - 0.2 = 0.075
        assert pytest.approx(scaled[0, 0], 0.001) == 0.075

    def test_already_reflectance(self):
        """Test that already-scaled reflectance is not rescaled."""
        config = LandsatConfig()

        # Already reflectance values
        reflectance = np.array([[0.1, 0.3], [0.5, 0.8]])

        scaled = scale_landsat_to_reflectance(reflectance, config)

        # Should be unchanged
        np.testing.assert_array_equal(scaled, reflectance)

    def test_invalid_dn_range(self):
        """Test handling of out-of-range DN values."""
        config = LandsatConfig()

        # Values outside valid range
        dn = np.array([[100, 7272], [43637, 50000]])

        scaled = scale_landsat_to_reflectance(dn, config)

        # Out of range values should be NaN
        assert np.isnan(scaled[0, 0])  # Too low
        assert np.isnan(scaled[0, 1])  # Just below valid
        assert np.isnan(scaled[1, 0])  # Just above valid
        assert np.isnan(scaled[1, 1])  # Too high

    def test_nodata_handling(self):
        """Test NoData value handling."""
        config = LandsatConfig()

        # Include NoData value
        dn = np.array([[10000, 20000], [30000, -9999]])

        scaled = scale_landsat_to_reflectance(dn, config, nodata_value=-9999)

        # NoData should become NaN
        assert np.isfinite(scaled[0, 0])
        assert np.isfinite(scaled[0, 1])
        assert np.isfinite(scaled[1, 0])
        assert np.isnan(scaled[1, 1])

    def test_negative_reflectance(self):
        """Test removal of negative reflectance values."""
        config = LandsatConfig()

        # DN value that produces negative reflectance
        # 7000 * 0.0000275 - 0.2 = 0.1925 - 0.2 = -0.0075 (negative)
        dn = np.array([[7000]])

        scaled = scale_landsat_to_reflectance(dn, config)

        # Should be NaN (negative reflectance invalid)
        assert np.isnan(scaled[0, 0])


class TestEnhanceWithRoads:
    """Test road enhancement function."""

    def test_road_enhancement(self):
        """Test basic road enhancement."""
        data = np.array([[0.5, 0.6], [0.4, 0.7]])
        roads = np.array([[1, 0], [2, 0]])  # Major road, minor road

        enhanced = enhance_with_roads(data, roads)

        # Check major road enhancement
        assert enhanced[0, 0] == pytest.approx(0.5 * 1.3, 0.001)

        # Check minor road enhancement
        assert enhanced[1, 0] == pytest.approx(0.4 * 1.1, 0.001)

        # Check non-road pixels unchanged
        assert enhanced[0, 1] == 0.6
        assert enhanced[1, 1] == 0.7

    def test_custom_factors(self):
        """Test road enhancement with custom factors."""
        data = np.array([[1.0]])
        roads = np.array([[1]])

        enhanced = enhance_with_roads(data, roads, major_factor=2.0, minor_factor=1.5)

        assert enhanced[0, 0] == 2.0


class TestNormalizeToRange:
    """Test normalization function."""

    def test_normalize_basic(self):
        """Test basic normalization to [0, 1]."""
        data = np.array([0, 50, 100])

        normalized = normalize_to_range(data, 0, 1)

        assert normalized[0] == 0.0
        assert normalized[1] == 0.5
        assert normalized[2] == 1.0

    def test_normalize_custom_range(self):
        """Test normalization to custom range."""
        data = np.array([10, 20, 30])

        normalized = normalize_to_range(data, -1, 1, input_min=10, input_max=30)

        assert normalized[0] == -1.0
        assert normalized[1] == 0.0
        assert normalized[2] == 1.0

    def test_normalize_constant_values(self):
        """Test normalization when all values are the same."""
        data = np.array([5.0, 5.0, 5.0])

        normalized = normalize_to_range(data, 0, 1)

        # Should return midpoint of output range
        assert np.all(normalized == 0.5)
