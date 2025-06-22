"""Test enhancement functions in processing module."""

import numpy as np
import pytest

from blackmarble.processing import (
    enhance_contrast,
    normalize_data,
    scale_to_byte,
    apply_colormap,
    enhance_roads,
    enhance_urban_features,
)


class TestEnhanceContrast:
    """Test contrast enhancement function."""

    def test_percentile_method(self):
        """Test percentile-based contrast enhancement."""
        # Create test data with known distribution
        data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.9]])

        # Apply enhancement
        enhanced = enhance_contrast(
            data, method="percentile", clip_percentiles=(0, 100)
        )

        # Check range is expanded
        assert enhanced.min() == pytest.approx(0.0)
        assert enhanced.max() == pytest.approx(1.0)

    def test_minmax_method(self):
        """Test min-max contrast enhancement."""
        data = np.array([[0.2, 0.4], [0.6, 0.8]])

        enhanced = enhance_contrast(data, method="minmax")

        # Should scale to full 0-1 range
        assert enhanced.min() == pytest.approx(0.0)
        assert enhanced.max() == pytest.approx(1.0)

    def test_manual_values(self):
        """Test manual min/max override."""
        data = np.array([[0.0, 0.5], [1.0, 1.5]])

        enhanced = enhance_contrast(data, method="minmax", min_value=0.5, max_value=1.0)

        # Values below min should be 0, above max should be 1
        assert enhanced[0, 0] == pytest.approx(0.0)  # 0.0 < 0.5
        assert enhanced[0, 1] == pytest.approx(0.0)  # 0.5 exactly at min
        assert enhanced[1, 0] == pytest.approx(1.0)  # 1.0 at max
        assert enhanced[1, 1] == pytest.approx(1.0)  # 1.5 > 1.0

    def test_nan_handling(self):
        """Test NaN preservation."""
        data = np.array([[0.5, np.nan], [0.7, 0.9]])

        enhanced = enhance_contrast(data)

        # NaN should be preserved
        assert np.isnan(enhanced[0, 1])
        # Other values should be enhanced
        assert not np.isnan(enhanced[0, 0])

    def test_uniform_data(self):
        """Test handling of uniform data."""
        data = np.full((3, 3), 0.5)

        enhanced = enhance_contrast(data)

        # Should return middle value
        assert np.all(enhanced == 0.5)


class TestNormalizeData:
    """Test data normalization function."""

    def test_basic_normalization(self):
        """Test basic 0-1 normalization."""
        data = np.array([0, 50, 100])

        normalized = normalize_data(data, 0, 1)

        assert normalized[0] == pytest.approx(0.0)
        assert normalized[1] == pytest.approx(0.5)
        assert normalized[2] == pytest.approx(1.0)

    def test_custom_range(self):
        """Test normalization to custom range."""
        data = np.array([10, 20, 30])

        normalized = normalize_data(data, -1, 1)

        assert normalized[0] == pytest.approx(-1.0)
        assert normalized[1] == pytest.approx(0.0)
        assert normalized[2] == pytest.approx(1.0)

    def test_manual_input_range(self):
        """Test with manual input range."""
        data = np.array([5, 15, 25])

        normalized = normalize_data(data, 0, 1, input_min=0, input_max=50)

        assert normalized[0] == pytest.approx(0.1)  # 5/50
        assert normalized[1] == pytest.approx(0.3)  # 15/50
        assert normalized[2] == pytest.approx(0.5)  # 25/50


class TestScaleToByte:
    """Test byte scaling function."""

    def test_basic_scaling(self):
        """Test basic float to byte conversion."""
        data = np.array([[0.0, 0.5], [1.0, 0.25]])

        byte_data = scale_to_byte(data)

        assert byte_data.dtype == np.uint8
        assert byte_data[0, 0] == 0
        assert byte_data[0, 1] == 127 or byte_data[0, 1] == 128  # Rounding
        assert byte_data[1, 0] == 255
        assert byte_data[1, 1] == 63 or byte_data[1, 1] == 64  # Rounding

    def test_nan_handling(self):
        """Test NaN to nodata conversion."""
        data = np.array([[0.5, np.nan], [1.0, 0.0]])

        byte_data = scale_to_byte(data, nodata_value=255)

        assert byte_data[0, 1] == 255  # NaN -> nodata
        assert byte_data[0, 0] != 255  # Valid data

    def test_custom_range(self):
        """Test scaling with custom input range."""
        data = np.array([10, 20, 30])

        byte_data = scale_to_byte(data, min_value=0, max_value=100)

        # 10/100 * 255 = 25.5 -> 26
        # 20/100 * 255 = 51
        # 30/100 * 255 = 76.5 -> 77
        assert byte_data[0] == 26 or byte_data[0] == 25
        assert byte_data[1] == 51
        assert byte_data[2] == 77 or byte_data[2] == 76


class TestApplyColormap:
    """Test colormap application function."""

    def test_grayscale_colormap(self):
        """Test grayscale colormap."""
        data = np.array([[0.0, 0.5], [1.0, 0.25]])

        rgb = apply_colormap(data, colormap="grayscale")

        assert rgb.shape == (3, 2, 2)
        assert rgb.dtype == np.uint8
        # Grayscale should have R=G=B
        assert np.array_equal(rgb[0], rgb[1])
        assert np.array_equal(rgb[1], rgb[2])

    def test_viridis_colormap(self):
        """Test viridis colormap."""
        data = np.array([[0.0, 1.0]])

        rgb = apply_colormap(data, colormap="viridis")

        assert rgb.shape == (3, 1, 2)
        # Viridis: dark purple (low) to yellow-green (high)
        # Check that channels are different
        assert not np.array_equal(rgb[0], rgb[1])

    def test_with_alpha(self):
        """Test colormap with alpha channel."""
        data = np.array([[0.5, 0.7]])
        alpha = np.array([[1.0, 0.5]])

        rgba = apply_colormap(data, alpha=alpha)

        assert rgba.shape == (4, 1, 2)
        assert rgba[3, 0, 0] == 255  # Full opacity
        assert rgba[3, 0, 1] == 127 or rgba[3, 0, 1] == 128  # Half opacity

    def test_custom_scaling(self):
        """Test colormap with custom value range."""
        data = np.array([[0.2, 0.4, 0.6, 0.8]])

        rgb = apply_colormap(data, vmin=0.3, vmax=0.7)

        # 0.2 < 0.3 -> should be 0
        # 0.8 > 0.7 -> should be 255
        assert rgb[0, 0, 0] == 0  # All channels at minimum
        assert rgb[0, 0, 3] == 255  # All channels at maximum


class TestEnhanceRoads:
    """Test road enhancement function."""

    def test_basic_enhancement(self):
        """Test basic road enhancement."""
        data = np.array([[0.5, 0.5], [0.5, 0.5]])
        roads = np.array([[1, 0], [2, 0]])  # Motorway and primary road

        enhanced = enhance_roads(data, roads)

        # Motorway (major) should be enhanced more
        assert enhanced[0, 0] == pytest.approx(0.5 * 1.3)
        # Primary (minor) should be enhanced less
        assert enhanced[1, 0] == pytest.approx(0.5 * 1.1)
        # No road should be unchanged
        assert enhanced[0, 1] == 0.5
        assert enhanced[1, 1] == 0.5

    def test_custom_factors(self):
        """Test with custom enhancement factors."""
        data = np.array([[1.0, 1.0]])
        roads = np.array([[1, 2]])

        enhanced = enhance_roads(data, roads, custom_factors={1: 2.0, 2: 1.5})

        assert enhanced[0, 0] == 2.0
        assert enhanced[0, 1] == 1.5

    def test_all_road_types(self):
        """Test all road type classifications."""
        data = np.ones((2, 3))
        roads = np.array([[1, 2, 3], [4, 5, 0]])

        enhanced = enhance_roads(data, roads, major_factor=1.5, minor_factor=1.2)

        # Major roads (1, 4)
        assert enhanced[0, 0] == pytest.approx(1.5)
        assert enhanced[1, 0] == pytest.approx(1.5)
        # Minor roads (2, 3, 5)
        assert enhanced[0, 1] == pytest.approx(1.2)
        assert enhanced[0, 2] == pytest.approx(1.2)
        assert enhanced[1, 1] == pytest.approx(1.2)
        # No road
        assert enhanced[1, 2] == pytest.approx(1.0)


class TestEnhanceUrbanFeatures:
    """Test urban feature enhancement function."""

    def test_basic_urban_enhancement(self):
        """Test basic urban feature enhancement."""
        ndui = np.array([[0.5, 0.6], [0.7, 0.8]])
        roads = np.array([[1, 0], [0, 2]])

        enhanced = enhance_urban_features(ndui, roads, enhancement_strength=0.5)

        # Road pixels should be enhanced
        assert enhanced[0, 0] > ndui[0, 0]  # Motorway
        assert enhanced[1, 1] > ndui[1, 1]  # Primary road
        # Non-road pixels might be unchanged
        assert enhanced[0, 1] == ndui[0, 1]
        assert enhanced[1, 0] == ndui[1, 0]

    def test_with_nightlights(self):
        """Test enhancement with nighttime lights."""
        ndui = np.full((3, 3), 0.5)
        roads = np.zeros((3, 3))
        ntl = np.array([[0.1, 0.5, 0.9], [0.2, 0.6, 0.8], [0.0, 0.4, 0.7]])

        enhanced = enhance_urban_features(
            ndui, roads, ntl=ntl, enhancement_strength=1.0
        )

        # Bright areas should be enhanced more
        bright_indices = np.where(ntl > np.percentile(ntl[ntl > 0], 75))
        for i, j in zip(bright_indices[0], bright_indices[1]):
            assert enhanced[i, j] > ndui[i, j]

    def test_clipping(self):
        """Test that values are clipped to valid range."""
        ndui = np.array([[0.9, 0.95]])
        roads = np.array([[1, 4]])  # Major roads

        enhanced = enhance_urban_features(ndui, roads, enhancement_strength=1.0)

        # Should not exceed 1.0
        assert np.all(enhanced <= 1.0)
        assert np.all(enhanced >= 0.0)
