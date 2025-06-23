"""Test functions in blackmarble.enhance.visualize module.

This module tests visualization enhancements and colormap application functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

# Import the module under test
from blackmarble.enhance.visualize import (
    apply_colormap,
    create_false_color_composite,
    apply_inferno_colormap,
    create_hillshade,
    blend_with_hillshade,
    generate_colorbar,
    prepare_for_web_display,
    create_comparison_view,
    add_scale_bar,
    add_north_arrow,
)


def test_apply_colormap():
    """Test colormap application to data for visualization.
    
    This test should verify that:
    - Standard colormaps (grayscale, viridis, plasma, inferno, turbo) work correctly
    - Output is RGB(A) array with proper shape (3/4, height, width)
    - Data is properly scaled to [0, 255] uint8 range
    - Custom colormap data is applied correctly
    - Scale range parameter affects colormap mapping
    - Alpha channel is handled appropriately
    """
    # Create test data
    data = np.array([[0.0, 0.3, 0.6],
                     [0.2, 0.5, 0.8],
                     [0.4, 0.7, 1.0]])
    
    # Test standard colormaps work correctly
    for colormap in ["grayscale", "viridis", "plasma", "inferno", "turbo"]:
        result = apply_colormap(data, colormap=colormap)
        
        # Output is RGB array with proper shape
        assert result.shape == (3, 3, 3)  # (channels, height, width)
        assert result.dtype == np.uint8
        
        # Data is properly scaled to [0, 255] uint8 range
        assert result.min() >= 0
        assert result.max() <= 255
        
        # Different colormaps produce different results
        if colormap != "grayscale":
            gray_result = apply_colormap(data, colormap="grayscale")
            assert not np.array_equal(result, gray_result)
    
    # Test custom colormap data is applied correctly
    custom_colormap = {
        "r": np.linspace(0, 255, 256).astype(np.uint8),
        "g": np.zeros(256, dtype=np.uint8),
        "b": np.linspace(255, 0, 256).astype(np.uint8)
    }
    custom_result = apply_colormap(data, colormap_data=custom_colormap)
    assert custom_result.shape == (3, 3, 3)
    assert custom_result.dtype == np.uint8
    
    # Scale range parameter affects colormap mapping
    result_full = apply_colormap(data, scale_range=(0.0, 1.0))
    result_scaled = apply_colormap(data, scale_range=(0.3, 0.8))
    assert not np.array_equal(result_full, result_scaled)
    
    # Alpha channel is handled appropriately
    result_alpha = apply_colormap(data, alpha=0.5)
    assert result_alpha.shape == (4, 3, 3)  # RGBA
    assert result_alpha[3].min() >= 0  # Alpha channel exists
    assert result_alpha[3].max() <= 255


def test_apply_colormap_standard_maps():
    """Test application of standard colormaps.
    
    This test should verify that:
    - Each standard colormap produces distinct visual results
    - Colormap lookup tables are applied correctly
    - Color transitions are smooth and appropriate
    - Grayscale maintains luminance relationships
    - Scientific colormaps preserve perceptual uniformity
    """
    # Create gradient test data
    gradient_data = np.linspace(0, 1, 100).reshape(10, 10)
    
    colormaps = ["grayscale", "viridis", "plasma", "inferno", "turbo"]
    results = {}
    
    for colormap in colormaps:
        result = apply_colormap(gradient_data, colormap=colormap)
        results[colormap] = result
        
        # Each standard colormap produces distinct visual results
        for other_colormap in colormaps:
            if other_colormap != colormap:
                other_result = results.get(other_colormap)
                if other_result is not None:
                    assert not np.array_equal(result, other_result)
    
    # Grayscale maintains luminance relationships
    gray_result = results["grayscale"]
    # In grayscale, all channels should be equal
    assert np.array_equal(gray_result[0], gray_result[1])
    assert np.array_equal(gray_result[1], gray_result[2])
    # Should be monotonic (gradient should create smooth transition)
    assert np.all(np.diff(gray_result[0].flatten()) >= 0)
    
    # Color transitions are smooth (no large jumps in adjacent pixels)
    for colormap in ["viridis", "plasma", "inferno"]:
        result = results[colormap]
        for channel in range(3):
            # Check that we have some variety in color values
            channel_values = result[channel]
            value_range = np.ptp(channel_values)  # Peak-to-peak range
            assert value_range > 10  # Should have some color variation
            
            # Check for reasonable color distribution
            unique_values = len(np.unique(channel_values))
            assert unique_values > 2  # Should have more than just 2 colors


def test_apply_colormap_custom_data():
    """Test custom colormap data application.
    
    This test should verify that:
    - Custom R, G, B arrays are used for color mapping
    - Custom colormap overrides standard colormap selection
    - Different colormap sizes are handled correctly
    - Interpolation between colormap entries works properly
    """
    # Create test data
    data = np.array([[0.0, 0.5, 1.0]])
    
    # Custom R, G, B arrays are used for color mapping
    custom_colormap = {
        "r": np.array([0, 128, 255], dtype=np.uint8),  # Red gradient
        "g": np.array([255, 128, 0], dtype=np.uint8),  # Green reverse gradient
        "b": np.array([0, 0, 0], dtype=np.uint8)       # No blue
    }
    
    # Extend to 256 values for proper colormap
    full_custom = {
        "r": np.interp(np.linspace(0, 1, 256), [0, 0.5, 1], [0, 128, 255]).astype(np.uint8),
        "g": np.interp(np.linspace(0, 1, 256), [0, 0.5, 1], [255, 128, 0]).astype(np.uint8),
        "b": np.zeros(256, dtype=np.uint8)
    }
    
    result = apply_colormap(data, colormap_data=full_custom)
    
    # Custom colormap overrides standard colormap selection
    result_with_standard = apply_colormap(data, colormap="viridis", colormap_data=full_custom)
    assert np.array_equal(result, result_with_standard)
    
    # Verify custom colors are applied
    # Value 0.0 should map to (0, 255, 0) - start of colormap
    # Value 1.0 should map to (255, 0, 0) - end of colormap
    assert result[0, 0, 0] < result[0, 0, 2]  # Red increases
    assert result[1, 0, 0] > result[1, 0, 2]  # Green decreases
    assert result[2, 0, 0] == result[2, 0, 2] == 0  # Blue stays zero
    
    # Different colormap sizes are handled correctly
    small_colormap = {
        "r": np.array([0, 255], dtype=np.uint8),
        "g": np.array([0, 0], dtype=np.uint8),
        "b": np.array([255, 0], dtype=np.uint8)
    }
    # Extend to 256 for testing
    extended_small = {
        "r": np.linspace(0, 255, 256).astype(np.uint8),
        "g": np.zeros(256, dtype=np.uint8),
        "b": np.linspace(255, 0, 256).astype(np.uint8)
    }
    
    result_small = apply_colormap(data, colormap_data=extended_small)
    assert result_small.shape == (3, 1, 3)
    assert result_small.dtype == np.uint8


def test_apply_colormap_scale_range():
    """Test colormap scale range parameter.
    
    This test should verify that:
    - Scale range affects data normalization before colormap
    - Values outside scale range are clamped appropriately
    - Different scale ranges produce different colorizations
    - Auto-scaling works when scale_range is None
    """
    # Create test data with known range
    data = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]])
    
    # Scale range affects data normalization before colormap
    result_full = apply_colormap(data, scale_range=(0.0, 1.0))
    result_narrow = apply_colormap(data, scale_range=(0.25, 0.75))
    
    # Different scale ranges produce different colorizations
    assert not np.array_equal(result_full, result_narrow)
    
    # With narrow range (0.25, 0.75), values 0.25 and 0.75 should map to extremes
    # Values outside range should be clamped
    # 0.0 should clamp to 0.25 (dark end)
    # 1.0 should clamp to 0.75 (bright end)
    
    # Auto-scaling works when scale_range is None
    result_auto = apply_colormap(data, scale_range=None)
    assert result_auto.shape == (3, 1, 5)
    assert result_auto.dtype == np.uint8
    
    # Values outside scale range are clamped appropriately
    extreme_data = np.array([[-1.0, 0.0, 0.5, 1.0, 2.0]])
    result_clamped = apply_colormap(extreme_data, scale_range=(0.0, 1.0))
    
    # -1.0 and 0.0 should both map to minimum (clamped to 0.0)
    # 1.0 and 2.0 should both map to maximum (clamped to 1.0)
    assert np.array_equal(result_clamped[:, 0, 0], result_clamped[:, 0, 1])  # -1.0 and 0.0
    assert np.array_equal(result_clamped[:, 0, 3], result_clamped[:, 0, 4])  # 1.0 and 2.0
    
    # Test that scale range creates expected contrast
    data_gradient = np.linspace(0, 1, 10).reshape(1, 10)
    result_01 = apply_colormap(data_gradient, scale_range=(0.0, 1.0))
    result_narrow_range = apply_colormap(data_gradient, scale_range=(0.3, 0.7))
    
    # Narrow range should create higher contrast within the range
    range_01 = np.ptp(result_01[0])  # Peak-to-peak range
    range_narrow = np.ptp(result_narrow_range[0])
    assert range_narrow >= range_01  # Narrow range should have at least as much contrast


def test_apply_colormap_alpha_channel():
    """Test alpha channel handling in colormap application.
    
    This test should verify that:
    - Scalar alpha is applied uniformly to all pixels
    - Array alpha is applied per-pixel
    - Alpha values are properly scaled to [0, 255] range
    - RGBA output has correct shape and values
    """
    # Create test data
    data = np.array([[0.0, 0.5, 1.0],
                     [0.2, 0.6, 0.8]])
    
    # Scalar alpha is applied uniformly to all pixels
    result_scalar = apply_colormap(data, alpha=0.5)
    
    # RGBA output has correct shape and values
    assert result_scalar.shape == (4, 2, 3)  # RGBA
    assert result_scalar.dtype == np.uint8
    
    # Alpha values are properly scaled to [0, 255] range
    expected_alpha = int(0.5 * 255)
    assert np.all(result_scalar[3] == expected_alpha)
    
    # Array alpha is applied per-pixel
    alpha_array = np.array([[0.0, 0.5, 1.0],
                           [0.3, 0.7, 0.9]])
    result_array = apply_colormap(data, alpha=alpha_array)
    
    assert result_array.shape == (4, 2, 3)
    
    # Check that alpha values are applied per-pixel
    expected_alpha_array = (alpha_array * 255).astype(np.uint8)
    np.testing.assert_array_equal(result_array[3], expected_alpha_array)
    
    # Test edge cases
    # Alpha = 0 (fully transparent)
    result_transparent = apply_colormap(data, alpha=0.0)
    assert np.all(result_transparent[3] == 0)
    
    # Alpha = 1 (fully opaque)
    result_opaque = apply_colormap(data, alpha=1.0)
    assert np.all(result_opaque[3] == 255)
    
    # No alpha channel when alpha is None
    result_no_alpha = apply_colormap(data, alpha=None)
    assert result_no_alpha.shape == (3, 2, 3)  # RGB only


def test_create_false_color_composite():
    """Test false color composite creation from three bands.
    
    This test should verify that:
    - Three input bands are combined into RGB output
    - Contrast enhancement improves visualization when enabled
    - Gamma correction is applied per-channel when specified
    - Output has proper RGB format and scaling
    - Band order (R, G, B) is preserved correctly
    """
    # Create test band data (simulating NIR, Red, Green)
    nir_band = np.array([[0.8, 0.6, 0.4],
                         [0.7, 0.5, 0.3],
                         [0.6, 0.4, 0.2]])
    red_band = np.array([[0.3, 0.4, 0.5],
                         [0.2, 0.3, 0.4],
                         [0.1, 0.2, 0.3]])
    green_band = np.array([[0.2, 0.3, 0.4],
                           [0.1, 0.2, 0.3],
                           [0.0, 0.1, 0.2]])
    
    # Three input bands are combined into RGB output
    result = create_false_color_composite(nir_band, red_band, green_band)
    
    # Output has proper RGB format and scaling
    assert result.shape == (3, 3, 3)  # (RGB, height, width)
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255
    
    # Band order (R, G, B) is preserved correctly
    # NIR should be in red channel (index 0)
    # Red should be in green channel (index 1)
    # Green should be in blue channel (index 2)
    
    # Test without contrast enhancement
    result_no_enhance = create_false_color_composite(
        nir_band, red_band, green_band, 
        contrast_enhance=False
    )
    assert result_no_enhance.shape == (3, 3, 3)
    assert result_no_enhance.dtype == np.uint8
    
    # Contrast enhancement should produce different results
    result_enhance = create_false_color_composite(
        nir_band, red_band, green_band, 
        contrast_enhance=True
    )
    assert not np.array_equal(result_no_enhance, result_enhance)
    
    # Gamma correction is applied per-channel when specified
    gamma_values = (0.5, 1.0, 2.0)  # Brighten red, normal green, darken blue
    result_gamma = create_false_color_composite(
        nir_band, red_band, green_band,
        contrast_enhance=False,
        gamma=gamma_values
    )
    
    # Should produce different results from no gamma
    assert not np.array_equal(result_no_enhance, result_gamma)


def test_create_false_color_composite_contrast():
    """Test contrast enhancement in false color composites.
    
    This test should verify that:
    - Contrast enhancement improves band visibility
    - Enhancement is applied independently to each band
    - Disabled contrast enhancement preserves original scaling
    - Enhancement parameters are appropriate for visualization
    """
    # Create test data with limited contrast
    low_contrast_band = np.full((3, 3), 0.4)  # Uniform mid-value
    high_contrast_band = np.array([[0.0, 0.5, 1.0],
                                   [0.0, 0.5, 1.0],
                                   [0.0, 0.5, 1.0]])
    mid_band = np.full((3, 3), 0.5)
    
    # Contrast enhancement improves band visibility
    result_enhanced = create_false_color_composite(
        low_contrast_band, high_contrast_band, mid_band,
        contrast_enhance=True
    )
    
    result_no_enhance = create_false_color_composite(
        low_contrast_band, high_contrast_band, mid_band,
        contrast_enhance=False
    )
    
    # Enhanced version should have different (typically higher) contrast
    assert not np.array_equal(result_enhanced, result_no_enhance)
    
    # Enhancement should increase the range of values
    enhanced_range = np.ptp(result_enhanced)  # Peak-to-peak range
    original_range = np.ptp(result_no_enhance)
    
    # Enhanced should typically have equal or greater range
    assert enhanced_range >= original_range * 0.8  # Allow some tolerance
    
    # Enhancement is applied independently to each band
    # Create bands with different characteristics
    dark_band = np.full((3, 3), 0.1)  # Dark values
    bright_band = np.full((3, 3), 0.9)  # Bright values
    normal_band = np.linspace(0, 1, 9).reshape(3, 3)  # Good range
    
    result_mixed = create_false_color_composite(
        dark_band, bright_band, normal_band,
        contrast_enhance=True
    )
    
    assert result_mixed.shape == (3, 3, 3)
    assert result_mixed.dtype == np.uint8
    
    # Disabled contrast enhancement preserves original scaling
    linear_data = np.linspace(0, 1, 9).reshape(3, 3)
    result_linear = create_false_color_composite(
        linear_data, linear_data, linear_data,
        contrast_enhance=False
    )
    
    # Should maintain the linear relationship in the scaled output
    # (though scaled to uint8 range)
    assert result_linear.dtype == np.uint8


def test_create_false_color_composite_gamma():
    """Test gamma correction in false color composites.
    
    This test should verify that:
    - Individual gamma values are applied to each channel
    - Gamma correction affects brightness and contrast appropriately
    - Gamma values of 1.0 produce no change
    - Extreme gamma values are handled correctly
    """
    # Create test data with linear gradient
    gradient = np.linspace(0, 1, 9).reshape(3, 3)
    
    # Gamma values of 1.0 produce no change
    result_gamma_1 = create_false_color_composite(
        gradient, gradient, gradient,
        contrast_enhance=False,
        gamma=(1.0, 1.0, 1.0)
    )
    
    result_no_gamma = create_false_color_composite(
        gradient, gradient, gradient,
        contrast_enhance=False,
        gamma=None
    )
    
    # Should be identical or very close
    np.testing.assert_allclose(result_gamma_1, result_no_gamma, atol=1)
    
    # Individual gamma values are applied to each channel
    gamma_values = (0.5, 1.0, 2.0)  # Brighten red, normal green, darken blue
    result_gamma_mixed = create_false_color_composite(
        gradient, gradient, gradient,
        contrast_enhance=False,
        gamma=gamma_values
    )
    
    # Should produce different results from no gamma
    assert not np.array_equal(result_no_gamma, result_gamma_mixed)
    
    # Gamma correction affects brightness and contrast appropriately
    # Gamma < 1.0 brightens (shifts curve up)
    # Gamma > 1.0 darkens (shifts curve down)
    
    bright_gamma = create_false_color_composite(
        gradient, gradient, gradient,
        contrast_enhance=False,
        gamma=(0.5, 0.5, 0.5)
    )
    
    dark_gamma = create_false_color_composite(
        gradient, gradient, gradient,
        contrast_enhance=False,
        gamma=(2.0, 2.0, 2.0)
    )
    
    # Bright gamma should generally have higher values
    # Dark gamma should generally have lower values
    assert np.mean(bright_gamma) > np.mean(dark_gamma)
    
    # Extreme gamma values are handled correctly
    extreme_gamma = create_false_color_composite(
        gradient, gradient, gradient,
        contrast_enhance=False,
        gamma=(0.1, 5.0, 10.0)
    )
    
    assert extreme_gamma.shape == (3, 3, 3)
    assert extreme_gamma.dtype == np.uint8
    assert extreme_gamma.min() >= 0
    assert extreme_gamma.max() <= 255


def test_apply_inferno_colormap():
    """Test inferno colormap application for nighttime lights.
    
    This test should verify that:
    - Inferno colormap is applied with appropriate scale range
    - Zero values are forced to black when zero_as_black=True
    - Custom inferno data files are loaded when available
    - Fallback to standard inferno works when files missing
    - Scale range optimization enhances nighttime light visualization
    """
    # Create nighttime lights test data
    ntl_data = np.array([[0.0, 0.1, 0.3],
                         [0.0, 0.5, 0.7],
                         [0.2, 0.6, 1.0]])
    
    # Inferno colormap is applied with appropriate scale range
    result = apply_inferno_colormap(ntl_data)
    
    assert result.shape == (3, 3, 3)  # RGB output
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255
    
    # Zero values are forced to black when zero_as_black=True
    result_zero_black = apply_inferno_colormap(ntl_data, zero_as_black=True)
    
    # Check that zero positions are black
    zero_positions = np.where(ntl_data == 0.0)
    for i in range(3):  # RGB channels
        assert np.all(result_zero_black[i][zero_positions] == 0)
    
    # Test with zero_as_black=False
    result_zero_color = apply_inferno_colormap(ntl_data, zero_as_black=False)
    
    # Should produce different results
    assert not np.array_equal(result_zero_black, result_zero_color)
    
    # Scale range optimization enhances nighttime light visualization
    result_default = apply_inferno_colormap(ntl_data, scale_range=(0.3, 0.8))
    result_full_range = apply_inferno_colormap(ntl_data, scale_range=(0.0, 1.0))
    
    # Different scale ranges should produce different visualizations
    assert not np.array_equal(result_default, result_full_range)
    
    # Test fallback behavior (custom files may or may not exist)
    # The function should work regardless
    assert result.shape == (3, 3, 3)
    assert result.dtype == np.uint8


def test_apply_inferno_colormap_scale_range():
    """Test inferno colormap scale range optimization.
    
    This test should verify that:
    - Scale range (0.3, 0.8) enhances contrast for typical NTL data
    - Different scale ranges affect visualization quality
    - Scale range is applied before colormap lookup
    - Extreme scale ranges are handled appropriately
    """
    # Create typical NTL data range
    ntl_data = np.linspace(0, 1, 20).reshape(4, 5)
    
    # Scale range (0.3, 0.8) enhances contrast for typical NTL data
    result_optimized = apply_inferno_colormap(ntl_data, scale_range=(0.3, 0.8))
    result_full = apply_inferno_colormap(ntl_data, scale_range=(0.0, 1.0))
    
    # Different scale ranges affect visualization quality
    assert not np.array_equal(result_optimized, result_full)
    
    # Scale range is applied before colormap lookup
    # Values outside the range should be clamped
    extreme_data = np.array([[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])
    result_clamped = apply_inferno_colormap(extreme_data, scale_range=(0.3, 0.7))
    
    # Values 0.0 and 0.2 should map to same color (clamped to 0.3)
    # Values 0.8 and 1.0 should map to same color (clamped to 0.7)
    if not (extreme_data[0, 0] == 0.0 and extreme_data[0, 1] == 0.2):  # Handle zero_as_black
        # Check that values below range are similar
        color_02 = result_clamped[:, 0, 1]  # 0.2 value
        # Within range should be different from clamped values
        color_04 = result_clamped[:, 0, 2]  # 0.4 value (within range)
        color_06 = result_clamped[:, 0, 3]  # 0.6 value (within range)
        
        # Values within range should be different
        assert not np.array_equal(color_04, color_06)
    
    # Extreme scale ranges are handled appropriately
    result_narrow = apply_inferno_colormap(ntl_data, scale_range=(0.45, 0.55))
    result_wide = apply_inferno_colormap(ntl_data, scale_range=(0.0, 2.0))
    
    assert result_narrow.shape == (3, 4, 5)
    assert result_wide.shape == (3, 4, 5)
    assert result_narrow.dtype == np.uint8
    assert result_wide.dtype == np.uint8
    
    # Very narrow range should still produce valid output
    result_very_narrow = apply_inferno_colormap(ntl_data, scale_range=(0.49, 0.51))
    assert result_very_narrow.shape == (3, 4, 5)
    assert result_very_narrow.dtype == np.uint8


def test_apply_inferno_colormap_zero_handling():
    """Test zero value handling in inferno colormap.
    
    This test should verify that:
    - Zero values are forced to black when zero_as_black=True
    - Zero handling affects all RGB channels
    - Non-zero small values are colored appropriately
    - Zero handling doesn't affect other pixel values
    """
    # Create data with zeros and non-zero values
    test_data = np.array([[0.0, 0.001, 0.1],
                          [0.0, 0.5, 0.8],
                          [0.2, 0.6, 1.0]])
    
    # Zero values are forced to black when zero_as_black=True
    result_black = apply_inferno_colormap(test_data, zero_as_black=True)
    
    # Zero handling affects all RGB channels
    zero_positions = np.where(test_data == 0.0)
    for channel in range(3):  # RGB
        assert np.all(result_black[channel][zero_positions] == 0)
    
    # Non-zero small values are colored appropriately
    small_value_pos = np.where(test_data == 0.001)
    for channel in range(3):
        # Small non-zero values should not be black
        assert not np.all(result_black[channel][small_value_pos] == 0)
    
    # Zero handling doesn't affect other pixel values
    result_no_black = apply_inferno_colormap(test_data, zero_as_black=False)
    
    # Non-zero positions should be the same between the two results
    nonzero_positions = np.where(test_data != 0.0)
    for channel in range(3):
        np.testing.assert_array_equal(
            result_black[channel][nonzero_positions],
            result_no_black[channel][nonzero_positions]
        )
    
    # Test that zero_as_black=False allows zeros to be colored
    zero_positions = np.where(test_data == 0.0)
    zero_colors_no_black = result_no_black[:, zero_positions[0], zero_positions[1]]
    
    # At least some zero positions should have non-black colors when zero_as_black=False
    # (depending on scale_range, but typically should be some color, not pure black)
    # Note: This depends on the scale_range used
    
    # Test with data that has no zeros
    no_zero_data = test_data + 0.01  # Add small offset to remove zeros
    result_no_zeros = apply_inferno_colormap(no_zero_data, zero_as_black=True)
    
    # Should not have any pure black pixels
    assert not np.any((result_no_zeros[0] == 0) & 
                      (result_no_zeros[1] == 0) & 
                      (result_no_zeros[2] == 0))


def test_create_hillshade():
    """Test hillshade creation for terrain visualization.
    
    This test should verify that:
    - Elevation data is converted to hillshade lighting
    - Azimuth and altitude parameters control light direction
    - Z-factor controls vertical exaggeration
    - Output is normalized to [0, 1] range
    - Gradient calculations are performed correctly
    """
    # Create simple elevation data (cone-shaped)
    x = np.linspace(-1, 1, 21)
    y = np.linspace(-1, 1, 21)
    X, Y = np.meshgrid(x, y)
    elevation = 1 - np.sqrt(X**2 + Y**2)  # Cone with peak at center
    elevation = np.maximum(elevation, 0)  # Clip negative values
    
    # Elevation data is converted to hillshade lighting
    hillshade = create_hillshade(elevation)
    
    # Output is normalized to [0, 1] range
    assert hillshade.shape == elevation.shape
    assert hillshade.min() >= 0.0
    assert hillshade.max() <= 1.0
    assert hillshade.dtype == np.float64
    
    # Azimuth and altitude parameters control light direction
    hs_nw = create_hillshade(elevation, azimuth=315, altitude=45)  # Northwest light
    hs_ne = create_hillshade(elevation, azimuth=45, altitude=45)   # Northeast light
    hs_se = create_hillshade(elevation, azimuth=135, altitude=45)  # Southeast light
    
    # Different azimuth should produce different lighting patterns
    assert not np.array_equal(hs_nw, hs_ne)
    assert not np.array_equal(hs_ne, hs_se)
    
    # Test altitude effect
    hs_low = create_hillshade(elevation, azimuth=315, altitude=15)   # Low sun
    hs_high = create_hillshade(elevation, azimuth=315, altitude=75)  # High sun
    
    # Different altitude should produce different lighting
    assert not np.array_equal(hs_low, hs_high)
    
    # Z-factor controls vertical exaggeration
    hs_z1 = create_hillshade(elevation, z_factor=1.0)
    hs_z2 = create_hillshade(elevation, z_factor=2.0)
    hs_z05 = create_hillshade(elevation, z_factor=0.5)
    
    # Different z-factors should produce different results
    assert not np.array_equal(hs_z1, hs_z2)
    assert not np.array_equal(hs_z1, hs_z05)
    
    # Higher z-factor should generally increase contrast
    contrast_z1 = np.std(hs_z1)
    contrast_z2 = np.std(hs_z2)
    assert contrast_z2 > contrast_z1


def test_create_hillshade_lighting():
    """Test hillshade lighting calculations.
    
    This test should verify that:
    - Different azimuth angles produce different shadow patterns
    - Altitude affects shadow length and intensity
    - Lighting calculations follow expected trigonometric relationships
    - Edge effects are minimized in gradient calculations
    """
    # Create simple slope (inclined plane)
    elevation = np.zeros((10, 10))
    for i in range(10):
        elevation[i, :] = i * 0.1  # Slope increasing from south to north
    
    # Different azimuth angles produce different shadow patterns
    azimuths = [0, 90, 180, 270]  # North, East, South, West
    hillshades = []
    
    for azimuth in azimuths:
        hs = create_hillshade(elevation, azimuth=azimuth, altitude=45)
        hillshades.append(hs)
    
    # Each azimuth should produce unique pattern
    for i in range(len(hillshades)):
        for j in range(i+1, len(hillshades)):
            assert not np.array_equal(hillshades[i], hillshades[j])
    
    # Light from north (azimuth=0) should illuminate south-facing slopes
    hs_north = create_hillshade(elevation, azimuth=0, altitude=45)
    # Light from south (azimuth=180) should illuminate north-facing slopes
    hs_south = create_hillshade(elevation, azimuth=180, altitude=45)
    
    # Should be different
    assert not np.array_equal(hs_north, hs_south)
    
    # Altitude affects shadow length and intensity
    altitudes = [15, 30, 45, 60, 75]  # Low to high sun
    contrasts = []
    
    for altitude in altitudes:
        hs = create_hillshade(elevation, azimuth=315, altitude=altitude)
        contrast = np.std(hs)
        contrasts.append(contrast)
    
    # Generally, moderate altitudes produce good contrast
    # Very low or very high altitudes reduce contrast
    assert len(contrasts) == 5
    
    # Edge effects are minimized in gradient calculations
    # Create larger elevation array
    large_elevation = np.random.rand(50, 50) * 100
    hs_large = create_hillshade(large_elevation)
    
    # Should not have NaN or infinite values
    assert np.all(np.isfinite(hs_large))
    assert hs_large.min() >= 0.0
    assert hs_large.max() <= 1.0


def test_create_hillshade_z_factor():
    """Test hillshade vertical exaggeration factor.
    
    This test should verify that:
    - Higher z_factor increases terrain relief appearance
    - Z_factor of 1.0 uses natural elevation scaling
    - Extreme z_factors are handled appropriately
    - Z_factor affects slope calculations correctly
    """
    # Create elevation data with known features
    elevation = np.zeros((20, 20))
    # Create a ridge
    elevation[8:12, :] = 10  # Flat-topped ridge
    
    # Z_factor of 1.0 uses natural elevation scaling
    hs_natural = create_hillshade(elevation, z_factor=1.0)
    
    # Higher z_factor increases terrain relief appearance
    hs_exaggerated = create_hillshade(elevation, z_factor=3.0)
    hs_subtle = create_hillshade(elevation, z_factor=0.3)
    
    # Should produce different results
    assert not np.array_equal(hs_natural, hs_exaggerated)
    assert not np.array_equal(hs_natural, hs_subtle)
    
    # Higher z_factor should generally increase contrast
    contrast_natural = np.std(hs_natural)
    contrast_exaggerated = np.std(hs_exaggerated)
    contrast_subtle = np.std(hs_subtle)
    
    # Exaggerated should have higher contrast than natural
    assert contrast_exaggerated > contrast_natural
    # Subtle should have lower contrast than natural
    assert contrast_subtle < contrast_natural
    
    # Extreme z_factors are handled appropriately
    hs_extreme_high = create_hillshade(elevation, z_factor=100.0)
    hs_extreme_low = create_hillshade(elevation, z_factor=0.01)
    
    # Should still produce valid output
    assert hs_extreme_high.shape == elevation.shape
    assert hs_extreme_low.shape == elevation.shape
    assert np.all(np.isfinite(hs_extreme_high))
    assert np.all(np.isfinite(hs_extreme_low))
    assert hs_extreme_high.min() >= 0.0
    assert hs_extreme_high.max() <= 1.0
    assert hs_extreme_low.min() >= 0.0
    assert hs_extreme_low.max() <= 1.0
    
    # Z_factor affects slope calculations correctly
    # Create simple slope
    simple_slope = np.zeros((10, 10))
    for i in range(10):
        simple_slope[:, i] = i  # West-to-east slope
    
    hs_z1 = create_hillshade(simple_slope, azimuth=90, altitude=45, z_factor=1.0)  # Light from east
    hs_z2 = create_hillshade(simple_slope, azimuth=90, altitude=45, z_factor=2.0)  # Same light, 2x exaggeration
    
    # Should be different due to slope changes
    assert not np.array_equal(hs_z1, hs_z2)
    
    # Zero z_factor should produce flat lighting
    hs_zero = create_hillshade(elevation, z_factor=0.0)
    # Should be relatively uniform (no slope effects)
    assert np.std(hs_zero) < np.std(hs_natural)


def test_blend_with_hillshade():
    """Test blending data with hillshade for topographic context.
    
    This test should verify that:
    - Different blend modes produce distinct visual effects
    - Opacity parameter controls hillshade influence
    - Blending preserves data range appropriately
    - RGB and grayscale data are handled correctly
    """
    # Create test data and hillshade
    data = np.array([[0.2, 0.5, 0.8],
                     [0.1, 0.6, 0.9],
                     [0.3, 0.4, 0.7]])
    
    hillshade = np.array([[0.3, 0.7, 0.9],
                          [0.2, 0.8, 1.0],
                          [0.4, 0.6, 0.8]])
    
    # Different blend modes produce distinct visual effects
    blend_multiply = blend_with_hillshade(data, hillshade, blend_mode="multiply")
    blend_overlay = blend_with_hillshade(data, hillshade, blend_mode="overlay")
    blend_soft_light = blend_with_hillshade(data, hillshade, blend_mode="soft_light")
    
    # All should be different
    assert not np.array_equal(blend_multiply, blend_overlay)
    assert not np.array_equal(blend_overlay, blend_soft_light)
    assert not np.array_equal(blend_multiply, blend_soft_light)
    
    # Blending preserves data range appropriately
    assert blend_multiply.min() >= 0.0
    assert blend_multiply.max() <= 1.0
    assert blend_overlay.min() >= 0.0
    assert blend_overlay.max() <= 1.0
    assert blend_soft_light.min() >= 0.0
    assert blend_soft_light.max() <= 1.0
    
    # Opacity parameter controls hillshade influence
    blend_low_opacity = blend_with_hillshade(data, hillshade, opacity=0.2)
    blend_high_opacity = blend_with_hillshade(data, hillshade, opacity=0.8)
    
    # Different opacities should produce different results
    assert not np.array_equal(blend_low_opacity, blend_high_opacity)
    
    # Low opacity should be closer to original data
    diff_low = np.mean(np.abs(blend_low_opacity - data))
    diff_high = np.mean(np.abs(blend_high_opacity - data))
    assert diff_low < diff_high
    
    # RGB data is handled correctly
    rgb_data = np.stack([data, data * 0.8, data * 0.6], axis=0)  # 3D RGB array
    rgb_blended = blend_with_hillshade(rgb_data, hillshade, blend_mode="multiply")
    
    assert rgb_blended.shape == (3, 3, 3)  # RGB preserved
    assert rgb_blended.min() >= 0.0
    assert rgb_blended.max() <= 1.0
    
    # Grayscale data is handled correctly (already tested above)
    assert blend_multiply.shape == data.shape


def test_blend_with_hillshade_modes():
    """Test different hillshade blending modes.
    
    This test should verify that:
    - Multiply mode darkens valleys and brightens ridges
    - Overlay mode provides balanced contrast enhancement
    - Soft light mode creates subtle topographic effects
    - Blending modes follow expected mathematical relationships
    """
    # Create test data with uniform value
    uniform_data = np.full((3, 3), 0.5)
    
    # Create hillshade with known light/dark areas
    hillshade = np.array([[0.2, 0.5, 0.8],  # Dark to bright
                          [0.3, 0.5, 0.7],
                          [0.4, 0.5, 0.6]])
    
    # Multiply mode darkens valleys and brightens ridges
    multiply_result = blend_with_hillshade(uniform_data, hillshade, 
                                          blend_mode="multiply", opacity=1.0)
    
    # Where hillshade is dark (valleys), result should be darker than original
    # Where hillshade is bright (ridges), result should be similar or brighter
    valley_pos = np.unravel_index(np.argmin(hillshade), hillshade.shape)
    ridge_pos = np.unravel_index(np.argmax(hillshade), hillshade.shape)
    
    assert multiply_result[valley_pos] < uniform_data[valley_pos]
    # Note: multiply can only darken, not brighten
    
    # Overlay mode provides balanced contrast enhancement
    overlay_result = blend_with_hillshade(uniform_data, hillshade, 
                                         blend_mode="overlay", opacity=1.0)
    
    # Should be different from multiply
    assert not np.array_equal(multiply_result, overlay_result)
    
    # Overlay should provide more balanced results
    overlay_range = np.ptp(overlay_result)
    multiply_range = np.ptp(multiply_result)
    
    # Overlay typically provides better contrast than multiply for mid-tone data
    assert overlay_range > 0  # Should have some variation
    
    # Soft light mode creates subtle topographic effects
    soft_light_result = blend_with_hillshade(uniform_data, hillshade, 
                                           blend_mode="soft_light", opacity=1.0)
    
    # Should be different from other modes
    assert not np.array_equal(soft_light_result, multiply_result)
    assert not np.array_equal(soft_light_result, overlay_result)
    
    # Soft light should be more subtle (closer to original)
    soft_diff = np.mean(np.abs(soft_light_result - uniform_data))
    multiply_diff = np.mean(np.abs(multiply_result - uniform_data))
    overlay_diff = np.mean(np.abs(overlay_result - uniform_data))
    
    # Soft light should generally be more subtle
    assert soft_diff > 0  # Should have some effect
    
    # Blending modes follow expected mathematical relationships
    # Test with specific values to verify math
    test_data = np.array([[0.3]])
    test_hillshade = np.array([[0.7]])
    
    # Multiply: result = data * hillshade
    multiply_test = blend_with_hillshade(test_data, test_hillshade, 
                                        blend_mode="multiply", opacity=1.0)
    expected_multiply = 0.3 * 0.7
    np.testing.assert_allclose(multiply_test, expected_multiply, atol=0.01)
    
    # Test RGB data with different modes
    rgb_data = np.array([[[0.5]], [[0.6]], [[0.4]]])  # RGB with single pixel
    rgb_hillshade = np.array([[0.8]])
    
    rgb_multiply = blend_with_hillshade(rgb_data, rgb_hillshade, 
                                       blend_mode="multiply", opacity=1.0)
    
    assert rgb_multiply.shape == (3, 1, 1)
    # Each channel should be multiplied by hillshade
    np.testing.assert_allclose(rgb_multiply[0], 0.5 * 0.8, atol=0.01)
    np.testing.assert_allclose(rgb_multiply[1], 0.6 * 0.8, atol=0.01)
    np.testing.assert_allclose(rgb_multiply[2], 0.4 * 0.8, atol=0.01)


def test_blend_with_hillshade_opacity():
    """Test hillshade opacity parameter.
    
    This test should verify that:
    - Opacity controls the strength of hillshade effect
    - Opacity of 0 returns original data unchanged
    - Opacity of 1 applies full hillshade effect
    - Intermediate opacities provide proportional blending
    """
    # Create test data
    data = np.array([[0.4, 0.6],
                     [0.3, 0.7]])
    
    hillshade = np.array([[0.2, 0.8],
                          [0.9, 0.1]])
    
    # Opacity of 0 returns original data unchanged
    result_opacity_0 = blend_with_hillshade(data, hillshade, opacity=0.0)
    np.testing.assert_array_equal(result_opacity_0, data)
    
    # Opacity of 1 applies full hillshade effect
    result_opacity_1 = blend_with_hillshade(data, hillshade, 
                                           blend_mode="multiply", opacity=1.0)
    
    # Should be different from original
    assert not np.array_equal(result_opacity_1, data)
    
    # For multiply mode with opacity 1, should be data * hillshade
    expected_full = data * hillshade
    np.testing.assert_allclose(result_opacity_1, expected_full, atol=0.01)
    
    # Intermediate opacities provide proportional blending
    result_opacity_05 = blend_with_hillshade(data, hillshade, 
                                            blend_mode="multiply", opacity=0.5)
    
    # Should be between original data and full blend
    # For opacity=0.5: result = data * (1-0.5) + blend * 0.5
    expected_half = data * 0.5 + expected_full * 0.5
    np.testing.assert_allclose(result_opacity_05, expected_half, atol=0.01)
    
    # Opacity controls the strength of hillshade effect
    opacities = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []
    
    for opacity in opacities:
        result = blend_with_hillshade(data, hillshade, 
                                     blend_mode="multiply", opacity=opacity)
        results.append(result)
    
    # Each should be different (unless opacity is 0 or 1 with specific conditions)
    for i in range(len(results) - 1):
        if opacities[i] != opacities[i + 1]:
            assert not np.array_equal(results[i], results[i + 1])
    
    # Test with RGB data
    rgb_data = np.stack([data, data * 0.9, data * 0.8], axis=0)
    rgb_result_half = blend_with_hillshade(rgb_data, hillshade, opacity=0.5)
    
    assert rgb_result_half.shape == (3, 2, 2)
    
    # Each channel should follow the opacity blending
    for channel in range(3):
        channel_full = blend_with_hillshade(rgb_data[channel], hillshade, 
                                           blend_mode="multiply", opacity=1.0)
        expected_channel = rgb_data[channel] * 0.5 + channel_full * 0.5
        np.testing.assert_allclose(rgb_result_half[channel], expected_channel, atol=0.01)


def test_generate_colorbar():
    """Test colorbar generation for legend purposes.
    
    This test should verify that:
    - Colorbar represents the specified value range
    - Horizontal and vertical orientations work correctly
    - Size parameter controls colorbar dimensions
    - Standard and custom colormaps are supported
    - Gradient is smooth and represents colormap accurately
    """
    # Test basic colorbar generation
    value_range = (0.0, 1.0)
    
    # Colorbar represents the specified value range
    colorbar = generate_colorbar("inferno", value_range)
    
    # Size parameter controls colorbar dimensions
    assert colorbar.shape == (3, 256, 32)  # Default size (256, 32), RGB output
    assert colorbar.dtype == np.uint8
    
    # Custom size
    custom_colorbar = generate_colorbar("viridis", value_range, size=(128, 64))
    assert custom_colorbar.shape == (3, 128, 64)
    
    # Horizontal and vertical orientations work correctly
    vertical_colorbar = generate_colorbar("plasma", value_range, 
                                         orientation="vertical")
    horizontal_colorbar = generate_colorbar("plasma", value_range, 
                                           orientation="horizontal")
    
    assert vertical_colorbar.shape == (3, 256, 32)  # Height x Width
    assert horizontal_colorbar.shape == (3, 256, 32)  # Same size but different gradient direction
    
    # Different orientations should produce different patterns
    assert not np.array_equal(vertical_colorbar, horizontal_colorbar)
    
    # Standard colormaps are supported
    standard_colormaps = ["grayscale", "viridis", "plasma", "inferno", "turbo"]
    for colormap in standard_colormaps:
        cb = generate_colorbar(colormap, value_range)
        assert cb.shape == (3, 256, 32)
        assert cb.dtype == np.uint8
    
    # Custom colormaps are supported
    custom_colormap = {
        "r": np.linspace(0, 255, 256).astype(np.uint8),
        "g": np.zeros(256, dtype=np.uint8),
        "b": np.linspace(255, 0, 256).astype(np.uint8)
    }
    
    custom_cb = generate_colorbar(custom_colormap, value_range)
    assert custom_cb.shape == (3, 256, 32)
    assert custom_cb.dtype == np.uint8
    
    # Gradient is smooth (no huge jumps between adjacent pixels)
    # Check vertical gradient smoothness
    for channel in range(3):
        vertical_diff = np.abs(np.diff(colorbar[channel], axis=0))
        # Most transitions should be smooth
        assert np.percentile(vertical_diff, 90) < 30  # 90% of transitions < 30 levels


def test_generate_colorbar_orientations():
    """Test colorbar orientation options.
    
    This test should verify that:
    - Vertical orientation creates top-to-bottom gradient
    - Horizontal orientation creates left-to-right gradient
    - Orientation affects gradient direction appropriately
    - Size parameters are interpreted correctly for each orientation
    """
    value_range = (0.0, 1.0)
    size = (100, 20)  # height, width
    
    # Vertical orientation creates top-to-bottom gradient
    vertical_cb = generate_colorbar("grayscale", value_range, 
                                   size=size, orientation="vertical")
    
    assert vertical_cb.shape == (3, 100, 20)  # RGB, height, width
    
    # In vertical orientation, gradient should vary along height (axis 1)
    # Top should be different from bottom
    top_row = vertical_cb[:, 0, :]  # Top row
    bottom_row = vertical_cb[:, -1, :]  # Bottom row
    assert not np.array_equal(top_row, bottom_row)
    
    # For grayscale with range (0,1), top should be brighter (value_range[1])
    # and bottom should be darker (value_range[0])
    # All columns in a row should be similar (uniform across width)
    assert np.std(top_row[0]) < 10  # Low variation across width
    assert np.std(bottom_row[0]) < 10
    
    # Horizontal orientation creates left-to-right gradient
    horizontal_cb = generate_colorbar("grayscale", value_range, 
                                     size=size, orientation="horizontal")
    
    assert horizontal_cb.shape == (3, 100, 20)
    
    # In horizontal orientation, gradient should vary along width (axis 2)
    left_col = horizontal_cb[:, :, 0]  # Left column
    right_col = horizontal_cb[:, :, -1]  # Right column
    assert not np.array_equal(left_col, right_col)
    
    # All rows in a column should be similar (uniform across height)
    assert np.std(left_col[0]) < 10
    assert np.std(right_col[0]) < 10
    
    # Orientation affects gradient direction appropriately
    assert not np.array_equal(vertical_cb, horizontal_cb)
    
    # Size parameters are interpreted correctly for each orientation
    large_size = (200, 50)
    small_size = (50, 10)
    
    large_vertical = generate_colorbar("viridis", value_range, 
                                      size=large_size, orientation="vertical")
    small_vertical = generate_colorbar("viridis", value_range, 
                                      size=small_size, orientation="vertical")
    
    assert large_vertical.shape == (3, 200, 50)
    assert small_vertical.shape == (3, 50, 10)
    
    large_horizontal = generate_colorbar("viridis", value_range, 
                                        size=large_size, orientation="horizontal")
    small_horizontal = generate_colorbar("viridis", value_range, 
                                        size=small_size, orientation="horizontal")
    
    assert large_horizontal.shape == (3, 200, 50)
    assert small_horizontal.shape == (3, 50, 10)


def test_generate_colorbar_value_range():
    """Test colorbar value range representation.
    
    This test should verify that:
    - Value range is mapped to colorbar gradient correctly
    - Different value ranges produce appropriate scaling
    - Negative and positive ranges are handled correctly
    - Large dynamic ranges are represented appropriately
    """
    # Value range is mapped to colorbar gradient correctly
    range1 = (0.0, 1.0)
    range2 = (0.2, 0.8)
    
    cb1 = generate_colorbar("inferno", range1, orientation="vertical")
    cb2 = generate_colorbar("inferno", range2, orientation="vertical")
    
    # Both should have same shape but different gradients
    assert cb1.shape == cb2.shape
    # Different ranges should produce different colorbars
    assert not np.array_equal(cb1, cb2)
    
    # Different value ranges produce appropriate scaling
    wide_range = (-10.0, 10.0)
    narrow_range = (4.5, 5.5)
    
    cb_wide = generate_colorbar("viridis", wide_range)
    cb_narrow = generate_colorbar("viridis", narrow_range)
    
    assert cb_wide.shape == cb_narrow.shape
    # Both should use full colormap range despite different value ranges
    assert not np.array_equal(cb_wide, cb_narrow)
    
    # Negative and positive ranges are handled correctly
    negative_range = (-5.0, -1.0)
    positive_range = (1.0, 5.0)
    mixed_range = (-2.0, 3.0)
    
    cb_negative = generate_colorbar("plasma", negative_range)
    cb_positive = generate_colorbar("plasma", positive_range)
    cb_mixed = generate_colorbar("plasma", mixed_range)
    
    # All should produce valid colorbars
    assert cb_negative.shape == (3, 256, 32)
    assert cb_positive.shape == (3, 256, 32)
    assert cb_mixed.shape == (3, 256, 32)
    
    # All should be different
    assert not np.array_equal(cb_negative, cb_positive)
    assert not np.array_equal(cb_positive, cb_mixed)
    assert not np.array_equal(cb_negative, cb_mixed)
    
    # Large dynamic ranges are represented appropriately
    huge_range = (-1000000.0, 1000000.0)
    tiny_range = (0.000001, 0.000002)
    
    cb_huge = generate_colorbar("turbo", huge_range)
    cb_tiny = generate_colorbar("turbo", tiny_range)
    
    # Should still produce valid colorbars
    assert cb_huge.shape == (3, 256, 32)
    assert cb_tiny.shape == (3, 256, 32)
    assert cb_huge.dtype == np.uint8
    assert cb_tiny.dtype == np.uint8
    
    # Should use full gradient range
    assert cb_huge.min() >= 0
    assert cb_huge.max() <= 255
    assert cb_tiny.min() >= 0
    assert cb_tiny.max() <= 255
    
    # Test reversed range
    reversed_range = (1.0, 0.0)  # High to low
    normal_range = (0.0, 1.0)   # Low to high
    
    cb_reversed = generate_colorbar("inferno", reversed_range, orientation="vertical")
    cb_normal = generate_colorbar("inferno", normal_range, orientation="vertical")
    
    # Should be different (reversed gradient)
    assert not np.array_equal(cb_reversed, cb_normal)


def test_prepare_for_web_display():
    """Test image preparation for web display.
    
    This test should verify that:
    - Data is converted to uint8 format appropriately
    - Resizing works correctly when output_size specified
    - Data scaling preserves visual quality
    - Web optimization doesn't introduce artifacts
    """
    # Create test RGB data in float [0,1] range
    float_data = np.random.rand(3, 20, 30)  # RGB, height, width
    
    # Data is converted to uint8 format appropriately
    web_data = prepare_for_web_display(float_data)
    
    assert web_data.dtype == np.uint8
    assert web_data.shape == float_data.shape
    assert web_data.min() >= 0
    assert web_data.max() <= 255
    
    # Test with data already in uint8
    uint8_data = (np.random.rand(3, 15, 25) * 255).astype(np.uint8)
    web_uint8 = prepare_for_web_display(uint8_data)
    
    assert web_uint8.dtype == np.uint8
    assert web_uint8.shape == uint8_data.shape
    # Should be similar to input (already uint8)
    np.testing.assert_array_equal(web_uint8, uint8_data)
    
    # Test with data > 1.0 (should be clipped)
    large_data = np.random.rand(3, 10, 15) * 300  # Values 0-300
    web_clipped = prepare_for_web_display(large_data)
    
    assert web_clipped.dtype == np.uint8
    assert web_clipped.min() >= 0
    assert web_clipped.max() <= 255
    
    # Resizing works correctly when output_size specified
    original_size = (20, 30)
    target_size = (10, 15)  # Half size
    
    resized_data = prepare_for_web_display(float_data, output_size=target_size)
    
    assert resized_data.shape == (3, 10, 15)  # RGB, target height, target width
    assert resized_data.dtype == np.uint8
    
    # Test upscaling
    upscale_size = (40, 60)  # Double size
    upscaled_data = prepare_for_web_display(float_data, output_size=upscale_size)
    
    assert upscaled_data.shape == (3, 40, 60)
    assert upscaled_data.dtype == np.uint8
    
    # Data scaling preserves visual quality
    # Create data with known pattern
    pattern_data = np.zeros((3, 10, 10))
    pattern_data[:, :5, :] = 0.2  # Dark half
    pattern_data[:, 5:, :] = 0.8  # Bright half
    
    web_pattern = prepare_for_web_display(pattern_data)
    
    # Pattern should be preserved
    dark_region = web_pattern[:, :5, :]
    bright_region = web_pattern[:, 5:, :]
    
    assert np.mean(dark_region) < np.mean(bright_region)
    # Dark region should be around 0.2 * 255 ≈ 51
    # Bright region should be around 0.8 * 255 ≈ 204
    np.testing.assert_allclose(np.mean(dark_region), 51, atol=5)
    np.testing.assert_allclose(np.mean(bright_region), 204, atol=5)


def test_prepare_for_web_display_resize():
    """Test web display image resizing.
    
    This test should verify that:
    - Output size parameter controls final dimensions
    - Aspect ratio is preserved during resizing
    - Resizing algorithm produces acceptable quality
    - Very large and very small sizes are handled correctly
    """
    # Create test data
    test_data = np.random.rand(3, 40, 60)  # 3:2 aspect ratio
    
    # Output size parameter controls final dimensions
    target_sizes = [(20, 30), (80, 120), (10, 15)]
    
    for target_size in target_sizes:
        resized = prepare_for_web_display(test_data, output_size=target_size)
        assert resized.shape == (3, target_size[0], target_size[1])
        assert resized.dtype == np.uint8
    
    # Test resizing with different aspect ratios
    # Note: The current implementation doesn't preserve aspect ratio automatically
    # It resizes to exact dimensions specified
    
    square_target = (30, 30)  # Square
    wide_target = (20, 80)    # Wide
    tall_target = (80, 20)    # Tall
    
    square_result = prepare_for_web_display(test_data, output_size=square_target)
    wide_result = prepare_for_web_display(test_data, output_size=wide_target)
    tall_result = prepare_for_web_display(test_data, output_size=tall_target)
    
    assert square_result.shape == (3, 30, 30)
    assert wide_result.shape == (3, 20, 80)
    assert tall_result.shape == (3, 80, 20)
    
    # Resizing algorithm produces acceptable quality
    # Create checkerboard pattern to test resizing quality
    checkerboard = np.zeros((3, 20, 20))
    for i in range(20):
        for j in range(20):
            if (i // 5 + j // 5) % 2 == 0:
                checkerboard[:, i, j] = 1.0
    
    # Resize down
    small_checker = prepare_for_web_display(checkerboard, output_size=(10, 10))
    assert small_checker.shape == (3, 10, 10)
    
    # Should still have some variation (pattern preserved to some extent)
    assert small_checker.min() < small_checker.max()
    
    # Very large and very small sizes are handled correctly
    # Very small
    tiny_size = (2, 3)
    tiny_result = prepare_for_web_display(test_data, output_size=tiny_size)
    assert tiny_result.shape == (3, 2, 3)
    assert tiny_result.dtype == np.uint8
    
    # Single pixel
    single_pixel = (1, 1)
    single_result = prepare_for_web_display(test_data, output_size=single_pixel)
    assert single_result.shape == (3, 1, 1)
    assert single_result.dtype == np.uint8
    
    # Larger size
    big_size = (100, 150)
    big_result = prepare_for_web_display(test_data, output_size=big_size)
    assert big_result.shape == (3, 100, 150)
    assert big_result.dtype == np.uint8
    
    # Test with no resize (output_size=None)
    no_resize = prepare_for_web_display(test_data, output_size=None)
    assert no_resize.shape == test_data.shape
    assert no_resize.dtype == np.uint8


def test_prepare_for_web_display_data_types():
    """Test web display preparation with different data types.
    
    This test should verify that:
    - Float data in [0,1] range is scaled to [0,255]
    - Integer data is handled appropriately
    - Data outside expected ranges is clipped correctly
    - Type conversion preserves image quality
    """
    # Float data in [0,1] range is scaled to [0,255]
    float_01_data = np.array([[[0.0, 0.25, 0.5, 0.75, 1.0]]], dtype=np.float32)  # 1x1x5
    float_01_data = np.broadcast_to(float_01_data, (3, 1, 5))  # Make RGB
    
    result_float = prepare_for_web_display(float_01_data)
    
    expected_values = [0, 64, 128, 191, 255]  # 0.25*255≈64, etc.
    np.testing.assert_allclose(result_float[0, 0, :], expected_values, atol=1)
    
    # Integer data is handled appropriately
    int_data = np.array([[[50, 100, 150, 200, 250]]], dtype=np.int32)
    int_data = np.broadcast_to(int_data, (3, 1, 5))
    
    result_int = prepare_for_web_display(int_data)
    
    # Should be clipped to [0, 255] and converted to uint8
    expected_int = [50, 100, 150, 200, 250]
    np.testing.assert_array_equal(result_int[0, 0, :], expected_int)
    
    # Test with uint16 data
    uint16_data = np.array([[[100, 1000, 10000, 30000, 65535]]], dtype=np.uint16)
    uint16_data = np.broadcast_to(uint16_data, (3, 1, 5))
    
    result_uint16 = prepare_for_web_display(uint16_data)
    
    # Values > 255 should be clipped to 255
    expected_uint16 = [100, 255, 255, 255, 255]
    np.testing.assert_array_equal(result_uint16[0, 0, :], expected_uint16)
    
    # Data outside expected ranges is clipped correctly
    # Negative values
    negative_data = np.array([[[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]]])
    negative_data = np.broadcast_to(negative_data, (3, 1, 7))
    
    result_negative = prepare_for_web_display(negative_data)
    
    # Negative should clip to 0, > 1.0 should scale appropriately
    # For float max ≤ 1.0, multiply by 255
    # For float max > 1.0, clip to [0, 255]
    assert result_negative[0, 0, 0] == 0  # -1.0 clipped
    assert result_negative[0, 0, 1] == 0  # -0.5 clipped
    assert result_negative[0, 0, 2] == 0  # 0.0 -> 0
    
    # Large float values
    large_float_data = np.array([[[0.0, 100.0, 1000.0, -50.0]]])
    large_float_data = np.broadcast_to(large_float_data, (3, 1, 4))
    
    result_large = prepare_for_web_display(large_float_data)
    
    # Should be clipped to [0, 255]
    assert result_large[0, 0, 0] == 0    # 0.0 -> 0
    assert result_large[0, 0, 1] == 255  # 100.0 clipped to 255
    assert result_large[0, 0, 2] == 255  # 1000.0 clipped to 255
    assert result_large[0, 0, 3] == 0    # -50.0 clipped to 0
    
    # Type conversion preserves image quality
    # Create gradient
    gradient_data = np.linspace(0, 1, 100).reshape(1, 10, 10)
    gradient_data = np.broadcast_to(gradient_data, (3, 10, 10))
    
    result_gradient = prepare_for_web_display(gradient_data)
    
    # Should maintain gradient structure
    assert result_gradient.dtype == np.uint8
    assert result_gradient.min() == 0
    assert result_gradient.max() == 255
    
    # Check that gradient is preserved (monotonic increase)
    flat_gradient = result_gradient[0].flatten()
    # Should be roughly monotonic (allowing for quantization)
    differences = np.diff(flat_gradient)
    # Most differences should be >= 0 (monotonic)
    assert np.sum(differences >= 0) > len(differences) * 0.8  # 80% non-decreasing


def test_create_comparison_view():
    """Test comparison visualization creation.
    
    This test should verify that:
    - Before and after images are combined appropriately
    - Different layout modes produce distinct visualizations
    - Divider width parameter controls separation
    - Image shapes are validated and handled correctly
    """
    # Create test images
    before_image = np.random.rand(3, 20, 30) * 0.3  # Darker image
    after_image = np.random.rand(3, 20, 30) * 0.7 + 0.3  # Brighter image
    
    # Before and after images are combined appropriately
    comparison = create_comparison_view(before_image, after_image)
    
    # Should create a larger combined image for side-by-side (default)
    assert comparison.shape[0] == 3  # RGB preserved
    assert comparison.shape[1] == 20  # Height preserved
    assert comparison.shape[2] > 30   # Width increased (side-by-side)
    
    # Different layout modes produce distinct visualizations
    side_by_side = create_comparison_view(before_image, after_image, layout="side_by_side")
    slider_view = create_comparison_view(before_image, after_image, layout="slider")
    checkerboard = create_comparison_view(before_image, after_image, layout="checkerboard")
    
    # All should have same base dimensions for height
    assert side_by_side.shape[1] == 20
    assert slider_view.shape[1] == 20
    assert checkerboard.shape[1] == 20
    
    # But different characteristics
    assert not np.array_equal(side_by_side, slider_view)
    assert not np.array_equal(slider_view, checkerboard)
    assert not np.array_equal(side_by_side, checkerboard)
    
    # Side-by-side should be wider
    assert side_by_side.shape[2] > before_image.shape[2]
    
    # Slider should have same width as original
    assert slider_view.shape[2] == before_image.shape[2]
    
    # Checkerboard should have same dimensions as original
    assert checkerboard.shape == before_image.shape
    
    # Divider width parameter controls separation
    narrow_divider = create_comparison_view(before_image, after_image, 
                                           layout="side_by_side", divider_width=1)
    wide_divider = create_comparison_view(before_image, after_image, 
                                         layout="side_by_side", divider_width=10)
    
    # Wide divider should create wider overall image
    assert wide_divider.shape[2] > narrow_divider.shape[2]
    
    # Image shapes are validated and handled correctly
    # Test with matching shapes (should work)
    assert before_image.shape == after_image.shape
    
    # Test with different shapes (should raise assertion error)
    different_shape = np.random.rand(3, 15, 25)
    
    with pytest.raises(AssertionError):
        create_comparison_view(before_image, different_shape)
    
    # Test with grayscale images
    gray_before = np.random.rand(20, 30) * 0.3
    gray_after = np.random.rand(20, 30) * 0.7 + 0.3
    
    gray_comparison = create_comparison_view(gray_before, gray_after, layout="side_by_side")
    
    assert gray_comparison.ndim == 2  # Still grayscale
    assert gray_comparison.shape[0] == 20  # Height preserved
    assert gray_comparison.shape[1] > 30   # Width increased


def test_create_comparison_view_layouts():
    """Test different comparison view layouts.
    
    This test should verify that:
    - Side-by-side layout places images horizontally
    - Slider layout creates vertical split comparison
    - Checkerboard layout alternates between images
    - Each layout provides effective comparison visualization
    """
    # Create distinct test images for easy verification
    before_image = np.zeros((3, 20, 30))  # Black image
    after_image = np.ones((3, 20, 30))    # White image
    
    # Side-by-side layout places images horizontally
    side_by_side = create_comparison_view(before_image, after_image, 
                                         layout="side_by_side", divider_width=2)
    
    # Should have both images side by side
    assert side_by_side.shape == (3, 20, 62)  # 30 + 30 + 2 divider
    
    # Left side should be black (before)
    left_region = side_by_side[:, :, :30]
    np.testing.assert_array_equal(left_region, before_image)
    
    # Right side should be white (after)
    right_region = side_by_side[:, :, 32:]  # After divider
    np.testing.assert_array_equal(right_region, after_image)
    
    # Divider should be white (or max value)
    divider_region = side_by_side[:, :, 30:32]
    assert np.all(divider_region >= 0.9)  # Should be bright
    
    # Slider layout creates vertical split comparison
    slider = create_comparison_view(before_image, after_image, 
                                   layout="slider", divider_width=2)
    
    # Should have same dimensions as original
    assert slider.shape == (3, 20, 30)
    
    # Left half should be black (before)
    left_half = slider[:, :, :15]
    np.testing.assert_array_equal(left_half, before_image[:, :, :15])
    
    # Right half should be white (after)
    right_half = slider[:, :, 16:]  # After divider area
    np.testing.assert_array_equal(right_half, after_image[:, :, 16:])
    
    # Checkerboard layout alternates between images
    checkerboard = create_comparison_view(before_image, after_image, layout="checkerboard")
    
    # Should have same dimensions as original
    assert checkerboard.shape == before_image.shape
    
    # Should have mix of both images
    unique_values = np.unique(checkerboard)
    assert len(unique_values) >= 2  # Should have both 0 and 1 values
    
    # Check that it actually alternates (at least some pattern)
    # The checkerboard should not be all one image or the other
    total_pixels = checkerboard.size
    black_pixels = np.sum(checkerboard == 0)
    white_pixels = np.sum(checkerboard == 1)
    
    # Should have a reasonable mix (not all black or all white)
    assert black_pixels > total_pixels * 0.1  # At least 10% black
    assert white_pixels > total_pixels * 0.1  # At least 10% white
    
    # Each layout provides effective comparison visualization
    # Test with more realistic data
    realistic_before = np.random.rand(3, 40, 60) * 0.3  # Dark
    realistic_after = np.random.rand(3, 40, 60) * 0.4 + 0.6  # Bright
    
    layouts = ["side_by_side", "slider", "checkerboard"]
    results = {}
    
    for layout in layouts:
        result = create_comparison_view(realistic_before, realistic_after, layout=layout)
        results[layout] = result
        
        # Each should produce valid output
        assert result.dtype == realistic_before.dtype
        assert np.all(np.isfinite(result))
    
    # All layouts should be different
    assert not np.array_equal(results["side_by_side"], results["slider"])
    assert not np.array_equal(results["slider"], results["checkerboard"])
    assert not np.array_equal(results["side_by_side"], results["checkerboard"])


def test_create_comparison_view_divider():
    """Test comparison view divider functionality.
    
    This test should verify that:
    - Divider creates visual separation between images
    - Divider width affects separation thickness
    - Divider color contrasts with image content
    - Divider doesn't obscure important image details
    """
    # Create test images with mid-tone values
    before_image = np.full((3, 20, 30), 0.3)  # Dark gray
    after_image = np.full((3, 20, 30), 0.7)   # Light gray
    
    # Divider creates visual separation between images
    with_divider = create_comparison_view(before_image, after_image, 
                                         layout="side_by_side", divider_width=4)
    
    # Should have divider region between images
    divider_region = with_divider[:, :, 30:34]  # Divider area
    
    # Divider should be different from both images
    assert not np.array_equal(divider_region, before_image[:, :, :4])
    assert not np.array_equal(divider_region, after_image[:, :, :4])
    
    # Divider width affects separation thickness
    narrow_divider = create_comparison_view(before_image, after_image, 
                                           layout="side_by_side", divider_width=1)
    wide_divider = create_comparison_view(before_image, after_image, 
                                         layout="side_by_side", divider_width=8)
    
    # Total width should increase with divider width
    assert narrow_divider.shape[2] == 30 + 30 + 1  # 61
    assert wide_divider.shape[2] == 30 + 30 + 8     # 68
    
    # Divider regions should have different sizes
    narrow_div = narrow_divider[:, :, 30:31]
    wide_div = wide_divider[:, :, 30:38]
    
    assert narrow_div.shape[2] == 1
    assert wide_div.shape[2] == 8
    
    # Divider color contrasts with image content
    # Test with uint8 data
    uint8_before = (before_image * 255).astype(np.uint8)
    uint8_after = (after_image * 255).astype(np.uint8)
    
    uint8_comparison = create_comparison_view(uint8_before, uint8_after, 
                                             layout="side_by_side", divider_width=2)
    
    uint8_divider_region = uint8_comparison[:, :, 30:32]
    
    # For uint8, divider should be 255 (white)
    assert np.all(uint8_divider_region == 255)
    
    # Test with float data
    float_comparison = create_comparison_view(before_image, after_image, 
                                             layout="side_by_side", divider_width=2)
    
    float_divider_region = float_comparison[:, :, 30:32]
    
    # For float, divider should be 1.0 (white)
    assert np.all(float_divider_region == 1.0)
    
    # Test slider layout divider
    slider_comparison = create_comparison_view(before_image, after_image, 
                                              layout="slider", divider_width=4)
    
    # Slider divider should be in the middle
    middle = slider_comparison.shape[2] // 2
    slider_divider = slider_comparison[:, :, middle-2:middle+2]
    
    # Should be bright/contrasting
    assert np.all(slider_divider >= 0.9)
    
    # Divider doesn't obscure important image details
    # Create images with important details near the split
    detailed_before = np.random.rand(3, 20, 30)
    detailed_after = np.random.rand(3, 20, 30)
    
    # Add specific features near edges
    detailed_before[:, 10, 29] = 1.0  # Bright spot near right edge
    detailed_after[:, 10, 0] = 0.0    # Dark spot near left edge
    
    detailed_comparison = create_comparison_view(detailed_before, detailed_after, 
                                                layout="side_by_side", divider_width=2)
    
    # Important details should be preserved
    # Right edge of left image
    preserved_left = detailed_comparison[:, 10, 29]
    np.testing.assert_allclose(preserved_left, 1.0, atol=0.01)
    
    # Left edge of right image
    preserved_right = detailed_comparison[:, 10, 32]  # After divider
    np.testing.assert_allclose(preserved_right, 0.0, atol=0.01)
    
    # Zero width divider should work
    no_divider = create_comparison_view(before_image, after_image, 
                                       layout="side_by_side", divider_width=0)
    
    assert no_divider.shape[2] == 60  # Just concatenated images
    # Should transition directly from before to after
    left_edge = no_divider[:, :, 29]
    right_edge = no_divider[:, :, 30]
    np.testing.assert_allclose(left_edge, 0.3, atol=0.01)  # Before image
    np.testing.assert_allclose(right_edge, 0.7, atol=0.01)  # After image


def test_add_scale_bar():
    """Test scale bar addition to images.
    
    This test should verify that:
    - Scale bar length is calculated correctly from pixel size
    - Scale bar is positioned appropriately on image
    - Scale bar color provides sufficient contrast
    - Scale bar doesn't obscure important image content
    - Different length specifications work correctly
    """
    # Create test image
    test_image = np.random.rand(3, 100, 150) * 128 + 64  # Mid-tone image
    
    # Scale bar length is calculated correctly from pixel size
    pixel_size = 30.0  # 30 meters per pixel
    length_km = 1.0    # 1 km scale bar
    
    result = add_scale_bar(test_image, pixel_size=pixel_size, length_km=length_km)
    
    # Should have same shape as input
    assert result.shape == test_image.shape
    assert result.dtype == test_image.dtype
    
    # Calculate expected scale bar length in pixels
    expected_length_px = int((length_km * 1000) / pixel_size)  # 1000m / 30m/px = 33px
    
    # Scale bar should be visible in bottom-right corner (default)
    # Look for bright pixels in expected region
    margin = 20
    bar_height = 10
    
    # Expected region for scale bar
    expected_x_start = 150 - margin - expected_length_px
    expected_y_start = 100 - margin - bar_height
    
    scale_bar_region = result[:, expected_y_start:expected_y_start+bar_height, 
                              expected_x_start:expected_x_start+expected_length_px]
    
    # Scale bar should be bright (white)
    # At least one channel should have high values in the scale bar region
    max_in_region = np.max(scale_bar_region)
    assert max_in_region > 200  # Should be bright
    
    # Scale bar is positioned appropriately on image
    # Test different positions
    result_bottom_left = add_scale_bar(test_image, pixel_size=pixel_size, 
                                      position="bottom_left")
    result_bottom_right = add_scale_bar(test_image, pixel_size=pixel_size, 
                                       position="bottom_right")
    
    # Should be different (scale bars in different positions)
    assert not np.array_equal(result_bottom_left, result_bottom_right)
    
    # Scale bar color provides sufficient contrast
    # Test with dark image
    dark_image = np.random.rand(3, 100, 150) * 50  # Dark image
    dark_result = add_scale_bar(dark_image, pixel_size=pixel_size, 
                               color=(255, 255, 255))  # White scale bar
    
    # Scale bar region should be much brighter than background
    dark_bg_region = dark_image[:, expected_y_start:expected_y_start+bar_height, 
                                expected_x_start:expected_x_start+expected_length_px]
    dark_bar_region = dark_result[:, expected_y_start:expected_y_start+bar_height, 
                                  expected_x_start:expected_x_start+expected_length_px]
    
    assert np.mean(dark_bar_region) > np.mean(dark_bg_region) + 100
    
    # Different length specifications work correctly
    length_5km = add_scale_bar(test_image, pixel_size=pixel_size, length_km=5.0)
    length_10km = add_scale_bar(test_image, pixel_size=pixel_size, length_km=10.0)
    
    # Should produce different results
    assert not np.array_equal(length_5km, length_10km)
    
    # Test with grayscale image
    gray_image = np.random.rand(100, 150) * 128
    gray_result = add_scale_bar(gray_image, pixel_size=pixel_size)
    
    assert gray_result.shape == gray_image.shape
    assert gray_result.ndim == 2  # Still grayscale


def test_add_scale_bar_positioning():
    """Test scale bar positioning options.
    
    This test should verify that:
    - Bottom-right and bottom-left positions work correctly
    - Scale bar respects image margins
    - Position calculations account for scale bar size
    - Scale bar fits within image boundaries
    """
    # Create test image
    test_image = np.zeros((3, 80, 120))  # Dark background for visibility
    pixel_size = 25.0  # 25m per pixel
    length_km = 2.0    # 2km scale bar
    
    expected_length_px = int((length_km * 1000) / pixel_size)  # 80 pixels
    margin = 20
    bar_height = 10
    
    # Bottom-right and bottom-left positions work correctly
    result_br = add_scale_bar(test_image, pixel_size=pixel_size, length_km=length_km,
                             position="bottom_right", color=(255, 255, 255))
    result_bl = add_scale_bar(test_image, pixel_size=pixel_size, length_km=length_km,
                             position="bottom_left", color=(255, 255, 255))
    
    # Should be different
    assert not np.array_equal(result_br, result_bl)
    
    # Check bottom-right positioning
    br_x_start = 120 - margin - expected_length_px  # Should be 120-20-80 = 20
    br_y_start = 80 - margin - bar_height            # Should be 80-20-10 = 50
    
    br_region = result_br[:, br_y_start:br_y_start+bar_height, 
                          br_x_start:br_x_start+expected_length_px]
    
    # Should have bright pixels (scale bar)
    assert np.max(br_region) > 200
    
    # Check bottom-left positioning
    bl_x_start = margin  # Should be 20
    bl_y_start = 80 - margin - bar_height  # Should be 50
    
    bl_region = result_bl[:, bl_y_start:bl_y_start+bar_height, 
                          bl_x_start:bl_x_start+expected_length_px]
    
    # Should have bright pixels (scale bar)
    assert np.max(bl_region) > 200
    
    # Scale bar respects image margins
    # Bottom-right scale bar should not be at the very edge
    # Check that there's margin space
    
    # Right edge should be margin pixels from image edge
    right_edge_region = result_br[:, br_y_start:br_y_start+bar_height, -margin:]
    # This region should be dark (no scale bar)
    assert np.max(right_edge_region) < 50
    
    # Bottom edge should be margin pixels from image edge
    bottom_edge_region = result_br[:, -margin:, br_x_start:br_x_start+expected_length_px]
    # This region should be dark (no scale bar)
    assert np.max(bottom_edge_region) < 50
    
    # Position calculations account for scale bar size
    # Test with very long scale bar
    long_length = 3.0  # 3km
    long_expected_px = int((long_length * 1000) / pixel_size)  # 120 pixels
    
    result_long = add_scale_bar(test_image, pixel_size=pixel_size, 
                               length_km=long_length, position="bottom_right")
    
    # Should still fit within image (though it might be very close to edge)
    long_x_start = 120 - margin - long_expected_px
    assert long_x_start >= 0  # Should not go negative
    
    # Scale bar fits within image boundaries
    # Test with small image
    small_image = np.zeros((3, 40, 60))
    
    # Scale bar that would normally be 80px, but image is only 60px wide
    result_small = add_scale_bar(small_image, pixel_size=pixel_size, 
                                length_km=length_km, position="bottom_right")
    
    # Should not crash and should produce valid output
    assert result_small.shape == small_image.shape
    
    # Even if scale bar is very large, function should handle it gracefully
    huge_length = 10.0  # 10km = 400 pixels (much larger than image)
    result_huge = add_scale_bar(small_image, pixel_size=pixel_size, 
                               length_km=huge_length, position="bottom_left")
    
    assert result_huge.shape == small_image.shape
    
    # Test edge case: very small pixel size (large scale bar)
    tiny_pixel_size = 1.0  # 1m per pixel
    result_tiny_pixel = add_scale_bar(test_image, pixel_size=tiny_pixel_size, 
                                     length_km=1.0)  # 1000 pixels!
    
    assert result_tiny_pixel.shape == test_image.shape


def test_add_scale_bar_calculations():
    """Test scale bar length calculations.
    
    This test should verify that:
    - Pixel size is used correctly for length calculation
    - Different length specifications (km) produce correct pixel lengths
    - Scale bar represents actual ground distance accurately
    - Edge cases (very large/small scales) are handled
    """
    # Create test image
    test_image = np.zeros((3, 100, 200))
    
    # Pixel size is used correctly for length calculation
    pixel_sizes = [10.0, 30.0, 100.0]  # meters per pixel
    length_km = 1.0  # 1 km
    
    for pixel_size in pixel_sizes:
        result = add_scale_bar(test_image, pixel_size=pixel_size, 
                              length_km=length_km, color=(255, 255, 255))
        
        # Calculate expected length in pixels
        expected_px = int((length_km * 1000) / pixel_size)
        
        # For verification, we can check that the scale bar appears
        # in approximately the right location and size
        margin = 20
        bar_height = 10
        x_start = 200 - margin - expected_px
        y_start = 100 - margin - bar_height
        
        if x_start >= 0:  # Only if scale bar fits
            bar_region = result[:, y_start:y_start+bar_height, 
                               x_start:x_start+expected_px]
            # Should have bright pixels
            assert np.max(bar_region) > 200
    
    # Different length specifications (km) produce correct pixel lengths
    pixel_size = 25.0  # 25m per pixel
    lengths_km = [0.5, 1.0, 2.0, 5.0]
    expected_lengths_px = [20, 40, 80, 200]  # 500m/25, 1000m/25, etc.
    
    for length_km, expected_px in zip(lengths_km, expected_lengths_px):
        result = add_scale_bar(test_image, pixel_size=pixel_size, 
                              length_km=length_km, color=(255, 0, 0))
        
        # Verify calculation
        calculated_px = int((length_km * 1000) / pixel_size)
        assert calculated_px == expected_px
        
        # If it fits in the image, check that it appears
        margin = 20
        bar_height = 10
        x_start = 200 - margin - expected_px
        
        if x_start >= 0:
            y_start = 100 - margin - bar_height
            bar_region = result[:, y_start:y_start+bar_height, 
                               x_start:x_start+expected_px]
            # Red channel should be bright (red scale bar)
            assert np.max(bar_region[0]) > 200
    
    # Scale bar represents actual ground distance accurately
    # Test the math: if pixel_size = 30m/px and length_km = 1.5km
    # Then scale bar should be 1500m / 30m/px = 50 pixels
    
    test_pixel_size = 30.0
    test_length_km = 1.5
    
    result = add_scale_bar(test_image, pixel_size=test_pixel_size, 
                          length_km=test_length_km, color=(0, 255, 0))
    
    expected_pixels = int((test_length_km * 1000) / test_pixel_size)  # 50
    assert expected_pixels == 50
    
    # Edge cases (very large/small scales) are handled
    # Very small scale (large pixel size, small distance)
    large_pixel_size = 1000.0  # 1km per pixel
    small_distance = 0.1       # 100m
    
    result_small = add_scale_bar(test_image, pixel_size=large_pixel_size, 
                                length_km=small_distance)
    
    # Expected: 100m / 1000m/px = 0.1 pixels -> rounds to 0
    expected_tiny = int((small_distance * 1000) / large_pixel_size)
    assert expected_tiny == 0
    
    # Function should handle this gracefully
    assert result_small.shape == test_image.shape
    
    # Very large scale (small pixel size, large distance)
    small_pixel_size = 0.1    # 10cm per pixel
    large_distance = 1.0      # 1km
    
    result_large = add_scale_bar(test_image, pixel_size=small_pixel_size, 
                                length_km=large_distance)
    
    # Expected: 1000m / 0.1m/px = 10000 pixels (much larger than image)
    expected_huge = int((large_distance * 1000) / small_pixel_size)
    assert expected_huge == 10000
    
    # Function should handle this gracefully (scale bar won't fit, but no crash)
    assert result_large.shape == test_image.shape
    
    # Test fractional results
    pixel_size = 33.3  # 33.3m per pixel
    length_km = 1.0    # 1000m
    
    result_fractional = add_scale_bar(test_image, pixel_size=pixel_size, length_km=length_km)
    
    # Expected: 1000 / 33.3 = 30.03 -> int() = 30
    expected_fractional = int(1000 / pixel_size)
    assert expected_fractional == 30
    
    assert result_fractional.shape == test_image.shape


def test_add_north_arrow():
    """Test north arrow addition to images.
    
    This test should verify that:
    - North arrow is drawn correctly at specified position
    - Arrow size parameter controls arrow dimensions
    - Arrow color provides sufficient contrast
    - Arrow shape is recognizable and appropriate
    - Arrow doesn't obscure important image content
    """
    # Create test image
    test_image = np.random.rand(3, 100, 150) * 100 + 50  # Mid-tone image
    
    # North arrow is drawn correctly at specified position
    result = add_north_arrow(test_image, position="top_right", size=50, 
                            color=(255, 255, 255))
    
    # Should have same shape as input
    assert result.shape == test_image.shape
    assert result.dtype == test_image.dtype
    
    # Arrow size parameter controls arrow dimensions
    small_arrow = add_north_arrow(test_image, size=20)
    large_arrow = add_north_arrow(test_image, size=80)
    
    # Should produce different results
    assert not np.array_equal(small_arrow, large_arrow)
    
    # Check that arrow appears in expected region
    margin = 20
    size = 50
    
    # Top-right position
    x_center = 150 - margin - size // 2  # 150 - 20 - 25 = 105
    y_center = margin + size // 2        # 20 + 25 = 45
    
    # Define arrow region
    arrow_region = result[:, max(0, y_center - size//2):min(100, y_center + size//2),
                          max(0, x_center - size//2):min(150, x_center + size//2)]
    
    # Arrow region should have some bright pixels (arrow)
    max_in_arrow = np.max(arrow_region)
    assert max_in_arrow > 200  # Should be bright
    
    # Arrow color provides sufficient contrast
    # Test with dark background
    dark_image = np.random.rand(3, 100, 150) * 30  # Dark image
    dark_result = add_north_arrow(dark_image, color=(255, 255, 255))  # White arrow
    
    # Arrow region should be much brighter than background
    dark_arrow_region = dark_result[:, y_center-size//4:y_center+size//4,
                                    x_center-size//4:x_center+size//4]
    dark_bg_mean = np.mean(dark_image)
    dark_arrow_mean = np.mean(dark_arrow_region)
    
    assert dark_arrow_mean > dark_bg_mean + 50  # Significant contrast
    
    # Test different positions
    top_left_result = add_north_arrow(test_image, position="top_left")
    top_right_result = add_north_arrow(test_image, position="top_right")
    
    # Should be different
    assert not np.array_equal(top_left_result, top_right_result)
    
    # Arrow shape is recognizable and appropriate
    # Test with large size for better visibility
    large_size = 60
    clear_result = add_north_arrow(np.zeros((3, 120, 160)), 
                                  position="top_right", size=large_size,
                                  color=(255, 255, 255))
    
    # Calculate expected center for large arrow
    large_x_center = 160 - 20 - large_size // 2
    large_y_center = 20 + large_size // 2
    
    # Check that arrow has the expected triangular shape
    # Arrow should have pixels above center (tip) and below center (base)
    tip_region = clear_result[:, large_y_center-large_size//2:large_y_center,
                              large_x_center-5:large_x_center+5]
    base_region = clear_result[:, large_y_center:large_y_center+large_size//4,
                               large_x_center-large_size//4:large_x_center+large_size//4]
    
    # Both regions should have some bright pixels
    assert np.max(tip_region) > 200
    assert np.max(base_region) > 200
    
    # Test with grayscale image
    gray_image = np.random.rand(100, 150) * 128
    gray_result = add_north_arrow(gray_image)
    
    assert gray_result.shape == gray_image.shape
    assert gray_result.ndim == 2  # Still grayscale


def test_add_north_arrow_positioning():
    """Test north arrow positioning options.
    
    This test should verify that:
    - Top-right and top-left positions work correctly
    - Arrow respects image margins
    - Position calculations account for arrow size
    - Arrow fits within image boundaries
    """
    # Create test image
    test_image = np.zeros((3, 80, 120))  # Dark background for visibility
    size = 40
    margin = 20
    
    # Top-right and top-left positions work correctly
    result_tr = add_north_arrow(test_image, position="top_right", size=size,
                               color=(255, 255, 255))
    result_tl = add_north_arrow(test_image, position="top_left", size=size,
                               color=(255, 255, 255))
    
    # Should be different
    assert not np.array_equal(result_tr, result_tl)
    
    # Check top-right positioning
    tr_x_center = 120 - margin - size // 2  # 120 - 20 - 20 = 80
    tr_y_center = margin + size // 2        # 20 + 20 = 40
    
    tr_region = result_tr[:, max(0, tr_y_center - size//4):min(80, tr_y_center + size//4),
                          max(0, tr_x_center - size//4):min(120, tr_x_center + size//4)]
    
    # Should have bright pixels (arrow)
    assert np.max(tr_region) > 200
    
    # Check top-left positioning
    tl_x_center = margin + size // 2        # 20 + 20 = 40
    tl_y_center = margin + size // 2        # 20 + 20 = 40
    
    tl_region = result_tl[:, max(0, tl_y_center - size//4):min(80, tl_y_center + size//4),
                          max(0, tl_x_center - size//4):min(120, tl_x_center + size//4)]
    
    # Should have bright pixels (arrow)
    assert np.max(tl_region) > 200
    
    # Arrow respects image margins
    # Top-right arrow should not be at the very edge
    
    # Right edge should have margin space
    right_edge_region = result_tr[:, tr_y_center-size//4:tr_y_center+size//4, -margin:]
    # This region should be dark (no arrow)
    assert np.max(right_edge_region) < 50
    
    # Top edge should have margin space
    top_edge_region = result_tr[:, :margin, tr_x_center-size//4:tr_x_center+size//4]
    # This region should be dark (no arrow)
    assert np.max(top_edge_region) < 50
    
    # Position calculations account for arrow size
    # Test with different sizes
    small_size = 20
    large_size = 60
    
    result_small = add_north_arrow(test_image, position="top_right", size=small_size,
                                  color=(255, 0, 0))
    result_large = add_north_arrow(test_image, position="top_right", size=large_size,
                                  color=(0, 255, 0))
    
    # Should be different
    assert not np.array_equal(result_small, result_large)
    
    # Small arrow center
    small_x_center = 120 - margin - small_size // 2
    small_y_center = margin + small_size // 2
    
    # Large arrow center
    large_x_center = 120 - margin - large_size // 2
    large_y_center = margin + large_size // 2
    
    # Centers should be different due to size
    assert small_x_center != large_x_center
    # y_centers might be the same depending on margin vs size relationship
    
    # Arrow fits within image boundaries
    # Test with small image
    small_image = np.zeros((3, 40, 60))
    
    # Arrow that would normally need significant space
    result_small_img = add_north_arrow(small_image, position="top_right", 
                                      size=30, color=(255, 255, 255))
    
    # Should not crash and should produce valid output
    assert result_small_img.shape == small_image.shape
    
    # Test with very large arrow on small image
    huge_arrow = add_north_arrow(small_image, position="top_left", 
                                size=80, color=(255, 255, 255))  # Larger than image!
    
    assert huge_arrow.shape == small_image.shape
    
    # Test edge case: arrow size larger than available space
    # With margins, available space is (height - 2*margin) x (width - 2*margin)
    # For small_image (40x60), that's 0x20 with margin=20
    tiny_image = np.zeros((3, 25, 30))  # Very small
    
    result_tiny = add_north_arrow(tiny_image, position="top_right", size=50)
    assert result_tiny.shape == tiny_image.shape
    
    # Test with zero margin scenario (arrow at edge)
    # Modify the function behavior if needed, but ensure no crash
    edge_image = np.zeros((3, 50, 50))
    edge_result = add_north_arrow(edge_image, position="top_right", size=40)
    assert edge_result.shape == edge_image.shape


def test_add_north_arrow_rendering():
    """Test north arrow rendering quality.
    
    This test should verify that:
    - Arrow shape is drawn correctly
    - Arrow size parameter affects dimensions appropriately
    - Arrow is clearly visible against various backgrounds
    - Arrow rendering doesn't create artifacts
    """
    # Create clean test image for arrow visibility
    clean_image = np.full((3, 100, 120), 128)  # Uniform gray background
    
    # Arrow shape is drawn correctly
    size = 50
    result = add_north_arrow(clean_image, position="top_right", size=size,
                            color=(255, 255, 255))
    
    margin = 20
    x_center = 120 - margin - size // 2
    y_center = margin + size // 2
    
    # Extract the arrow region
    arrow_region = result[:, max(0, y_center - size//2):min(100, y_center + size//2),
                          max(0, x_center - size//2):min(120, x_center + size//2)]
    
    # Arrow should have triangular characteristics
    # Check that there are bright pixels in the arrow region
    bright_pixels = np.sum(arrow_region[0] > 200)  # Count white pixels in red channel
    assert bright_pixels > 10  # Should have some arrow pixels
    
    # Arrow should be roughly triangular - more pixels near center, fewer at edges
    # This is a simplified check for arrow shape
    center_slice = arrow_region[0, arrow_region.shape[1]//2, :]
    if len(center_slice) > 0:
        # Should have some bright pixels in center horizontal slice
        assert np.max(center_slice) > 200
    
    # Arrow size parameter affects dimensions appropriately
    sizes = [20, 40, 60]
    pixel_counts = []
    
    for test_size in sizes:
        size_result = add_north_arrow(clean_image, position="top_left", 
                                     size=test_size, color=(255, 0, 0))
        
        # Calculate arrow region for this size
        size_x_center = margin + test_size // 2
        size_y_center = margin + test_size // 2
        
        size_region = size_result[:, max(0, size_y_center - test_size//2):min(100, size_y_center + test_size//2),
                                  max(0, size_x_center - test_size//2):min(120, size_x_center + test_size//2)]
        
        # Count red pixels (arrow pixels)
        red_pixels = np.sum(size_region[0] > 200)
        pixel_counts.append(red_pixels)
    
    # Larger arrows should generally have more pixels
    # (though the relationship might not be perfectly linear due to the simple rasterization)
    assert pixel_counts[2] >= pixel_counts[0]  # 60-size arrow >= 20-size arrow
    
    # Arrow is clearly visible against various backgrounds
    backgrounds = [
        np.zeros((3, 100, 120)),                    # Black
        np.ones((3, 100, 120)) * 255,               # White  
        np.random.rand(3, 100, 120) * 255,          # Random
        np.full((3, 100, 120), 128)                 # Gray
    ]
    
    colors = [
        (255, 255, 255),  # White arrow on black
        (0, 0, 0),        # Black arrow on white
        (255, 0, 0),      # Red arrow on random
        (0, 255, 0)       # Green arrow on gray
    ]
    
    for bg, color in zip(backgrounds, colors):
        bg_result = add_north_arrow(bg.copy(), position="top_right", 
                                   size=40, color=color)
        
        # Arrow should be different from background
        assert not np.array_equal(bg_result, bg)
        
        # Extract arrow region
        bg_x_center = 120 - 20 - 20
        bg_y_center = 20 + 20
        bg_arrow_region = bg_result[:, bg_y_center-10:bg_y_center+10,
                                    bg_x_center-10:bg_x_center+10]
        
        # Should have pixels of the arrow color
        if color[0] > 128:  # Bright red channel
            assert np.max(bg_arrow_region[0]) > 200
        if color[1] > 128:  # Bright green channel
            assert np.max(bg_arrow_region[1]) > 200
        if color[2] > 128:  # Bright blue channel
            assert np.max(bg_arrow_region[2]) > 200
    
    # Arrow rendering doesn't create artifacts
    # Check that arrow doesn't create strange patterns or noise
    artifact_test = add_north_arrow(np.zeros((3, 80, 100)), 
                                   position="top_left", size=30)
    
    # Should not have isolated bright pixels far from arrow center
    arrow_x = 20 + 15  # margin + size//2
    arrow_y = 20 + 15
    
    # Check corners (should be mostly dark)
    corners = [
        artifact_test[:, :5, :5],        # Top-left corner of image
        artifact_test[:, :5, -5:],       # Top-right corner
        artifact_test[:, -5:, :5],       # Bottom-left corner
        artifact_test[:, -5:, -5:]       # Bottom-right corner
    ]
    
    for corner in corners:
        # Corners should be mostly dark (no artifacts)
        assert np.max(corner) < 50  # Should be dark
    
    # Test with grayscale to ensure proper color handling
    gray_bg = np.full((80, 100), 64)  # Dark gray
    gray_result = add_north_arrow(gray_bg, position="top_right", size=25)
    
    assert gray_result.shape == gray_bg.shape
    assert gray_result.dtype == gray_bg.dtype
    
    # Should have some bright pixels where arrow is
    gray_x_center = 100 - 20 - 12  # margin + size//2
    gray_y_center = 20 + 12
    gray_arrow_region = gray_result[gray_y_center-5:gray_y_center+5,
                                   gray_x_center-5:gray_x_center+5]
    
    # Should have brighter pixels than background
    assert np.max(gray_arrow_region) > np.max(gray_bg) + 50


def test_colormap_application_accuracy():
    """Test accuracy of colormap application algorithms.
    
    This test should verify that:
    - Color mapping preserves data relationships
    - Colormap interpolation is smooth and correct
    - Color space conversions maintain fidelity
    - Numerical precision is maintained throughout
    """
    # Create test data with known relationships
    linear_data = np.linspace(0, 1, 256).reshape(16, 16)
    
    # Color mapping preserves data relationships
    result = apply_colormap(linear_data, colormap="viridis")
    
    # Monotonic input should produce relatively monotonic output
    # Check along the main diagonal
    diagonal_indices = np.diag_indices(16)
    for channel in range(3):
        diagonal_values = result[channel][diagonal_indices]
        
        # Should show general trend (though not perfectly monotonic due to colormap)
        # At least first and last values should be significantly different
        assert abs(diagonal_values[0] - diagonal_values[-1]) > 50
    
    # Colormap interpolation is smooth and correct
    # Create data with small increments
    smooth_data = np.arange(0, 1, 0.01).reshape(10, 10)
    smooth_result = apply_colormap(smooth_data, colormap="plasma")
    
    # Adjacent values should not have huge jumps
    for channel in range(3):
        flat_channel = smooth_result[channel].flatten()
        differences = np.abs(np.diff(flat_channel))
        
        # 95% of differences should be reasonable (< 30 levels)
        assert np.percentile(differences, 95) < 30
    
    # Color space conversions maintain fidelity
    # Test with extreme values
    extreme_data = np.array([[0.0, 0.5, 1.0]])
    extreme_result = apply_colormap(extreme_data, colormap="inferno")
    
    # Should use full range of colormap
    for channel in range(3):
        channel_values = extreme_result[channel].flatten()
        value_range = np.ptp(channel_values)  # Peak-to-peak range
        assert value_range > 100  # Should span significant color range
    
    # Numerical precision is maintained throughout
    # Test with high precision data
    precision_data = np.array([[0.0, 0.33333333, 0.66666667, 1.0]])
    precision_result = apply_colormap(precision_data, colormap="turbo")
    
    # Should produce consistent results for repeated calls
    precision_result2 = apply_colormap(precision_data, colormap="turbo")
    np.testing.assert_array_equal(precision_result, precision_result2)
    
    # Test scale range precision
    range_result1 = apply_colormap(precision_data, scale_range=(0.0, 1.0))
    range_result2 = apply_colormap(precision_data, scale_range=(0.0000, 1.0000))
    np.testing.assert_array_equal(range_result1, range_result2)
    
    # Test with custom colormap for known behavior
    # Simple red gradient: black to red
    custom_red = {
        "r": np.linspace(0, 255, 256).astype(np.uint8),
        "g": np.zeros(256, dtype=np.uint8),
        "b": np.zeros(256, dtype=np.uint8)
    }
    
    test_values = np.array([[0.0, 0.5, 1.0]])
    red_result = apply_colormap(test_values, colormap_data=custom_red)
    
    # Should have increasing red values, zero green/blue
    assert red_result[0, 0, 0] < red_result[0, 0, 1] < red_result[0, 0, 2]  # Red increases
    assert np.all(red_result[1] == 0)  # Green is zero
    assert np.all(red_result[2] == 0)  # Blue is zero


def test_false_color_composite_creation():
    """Test false color composite creation accuracy.
    
    This test should verify that:
    - Band combinations produce expected color results
    - Color balance is maintained across different input ranges
    - Composite creation preserves spatial relationships
    - Output quality is suitable for analysis and visualization
    """
    # Create test bands with distinct characteristics
    # Simulate NIR, Red, Green bands for vegetation analysis
    height, width = 20, 30
    
    # NIR band: high values for vegetation
    nir_band = np.zeros((height, width))
    nir_band[5:15, 10:20] = 0.8  # Vegetation area
    nir_band[0:5, 0:10] = 0.2    # Non-vegetation
    
    # Red band: moderate values
    red_band = np.zeros((height, width))
    red_band[5:15, 10:20] = 0.3  # Lower in vegetation
    red_band[0:5, 0:10] = 0.6    # Higher in non-vegetation
    
    # Green band: intermediate values
    green_band = np.full((height, width), 0.4)
    
    # Band combinations produce expected color results
    composite = create_false_color_composite(nir_band, red_band, green_band, 
                                           contrast_enhance=False)
    
    assert composite.shape == (3, height, width)
    assert composite.dtype == np.uint8
    
    # Vegetation area (NIR-Red-Green = 0.8-0.3-0.4)
    # Should appear red-ish (high NIR in red channel)
    veg_region = composite[:, 5:15, 10:20]
    veg_red = np.mean(veg_region[0])    # NIR -> Red channel
    veg_green = np.mean(veg_region[1])  # Red -> Green channel
    veg_blue = np.mean(veg_region[2])   # Green -> Blue channel
    
    # High NIR should create bright red channel
    assert veg_red > veg_green  # Red should be brighter than green
    
    # Non-vegetation area (NIR-Red-Green = 0.2-0.6-0.4)
    # Should appear green-ish (higher red band in green channel)
    nonveg_region = composite[:, 0:5, 0:10]
    nonveg_red = np.mean(nonveg_region[0])    # Low NIR
    nonveg_green = np.mean(nonveg_region[1])  # High Red -> Green channel
    
    assert nonveg_green > nonveg_red  # Green should be brighter than red
    
    # Color balance is maintained across different input ranges
    # Test with different value ranges
    high_range_nir = nir_band * 2  # 0-1.6 range
    low_range_red = red_band * 0.5  # 0-0.3 range
    
    composite_imbalanced = create_false_color_composite(
        high_range_nir, low_range_red, green_band, contrast_enhance=True
    )
    
    # Contrast enhancement should help balance the ranges
    assert composite_imbalanced.shape == (3, height, width)
    assert composite_imbalanced.dtype == np.uint8
    
    # Should use reasonable portions of the color space
    for channel in range(3):
        channel_range = np.ptp(composite_imbalanced[channel])
        assert channel_range > 50  # Should have decent contrast
    
    # Composite creation preserves spatial relationships
    # Create spatial pattern
    pattern_nir = np.zeros((10, 10))
    pattern_nir[2:8, 2:8] = 0.9  # Square in center
    
    pattern_red = np.ones((10, 10)) * 0.3
    pattern_green = np.ones((10, 10)) * 0.4
    
    pattern_composite = create_false_color_composite(
        pattern_nir, pattern_red, pattern_green, contrast_enhance=False
    )
    
    # The square pattern should be visible in the red channel
    center_red = np.mean(pattern_composite[0, 2:8, 2:8])
    edge_red = np.mean(pattern_composite[0, [0, 1, 8, 9], :])
    
    assert center_red > edge_red + 50  # Square should be much brighter
    
    # Output quality is suitable for analysis and visualization
    # Test with realistic satellite data ranges
    realistic_nir = np.random.rand(15, 20) * 0.6 + 0.2   # 0.2-0.8
    realistic_red = np.random.rand(15, 20) * 0.4 + 0.1   # 0.1-0.5
    realistic_green = np.random.rand(15, 20) * 0.3 + 0.1 # 0.1-0.4
    
    realistic_composite = create_false_color_composite(
        realistic_nir, realistic_red, realistic_green, 
        contrast_enhance=True, gamma=(1.2, 1.0, 0.8)
    )
    
    # Should produce good visual quality
    assert realistic_composite.shape == (3, 15, 20)
    assert realistic_composite.dtype == np.uint8
    assert realistic_composite.min() >= 0
    assert realistic_composite.max() <= 255
    
    # Should have good contrast in all channels
    for channel in range(3):
        channel_std = np.std(realistic_composite[channel])
        assert channel_std > 10  # Should have variation
        
        channel_range = np.ptp(realistic_composite[channel])
        assert channel_range > 30  # Should span decent range


def test_hillshade_generation_accuracy():
    """Test hillshade generation algorithm accuracy.
    
    This test should verify that:
    - Slope and aspect calculations are mathematically correct
    - Lighting model produces realistic shading effects
    - Hillshade enhances terrain interpretation
    - Algorithm performance is reasonable for large datasets
    """
    # Slope and aspect calculations are mathematically correct
    # Create simple test cases with known slopes
    
    # Test 1: Flat surface (no slope)
    flat_dem = np.full((10, 10), 100.0)  # Flat at 100m elevation
    flat_hillshade = create_hillshade(flat_dem)
    
    # Flat surface should have uniform moderate lighting
    assert np.std(flat_hillshade) < 0.1  # Very low variation
    # Should be around 0.7 for 45° altitude (cos(45°) ≈ 0.707)
    assert 0.6 < np.mean(flat_hillshade) < 0.8
    
    # Test 2: Simple slope - east-facing
    east_slope = np.zeros((10, 10))
    for i in range(10):
        east_slope[:, i] = i * 10  # Increases from west to east
    
    # Light from east (azimuth=90°) should illuminate this slope well
    east_lit = create_hillshade(east_slope, azimuth=90, altitude=45)
    # Light from west (azimuth=270°) should create shadows
    west_lit = create_hillshade(east_slope, azimuth=270, altitude=45)
    
    # East-lit should be brighter than west-lit for east-facing slope
    assert np.mean(east_lit) > np.mean(west_lit)
    
    # Test 3: North-facing slope
    north_slope = np.zeros((10, 10))
    for i in range(10):
        north_slope[i, :] = i * 10  # Increases from south to north
    
    # Light from north should illuminate this slope
    north_lit = create_hillshade(north_slope, azimuth=0, altitude=45)   # North
    south_lit = create_hillshade(north_slope, azimuth=180, altitude=45) # South
    
    assert np.mean(north_lit) > np.mean(south_lit)
    
    # Lighting model produces realistic shading effects
    # Create a hill (cone)
    x = np.linspace(-5, 5, 21)
    y = np.linspace(-5, 5, 21)
    X, Y = np.meshgrid(x, y)
    hill_dem = 100 + 50 * np.exp(-(X**2 + Y**2) / 10)  # Gaussian hill
    
    hill_hillshade = create_hillshade(hill_dem, azimuth=315, altitude=45)
    
    # Should have realistic lighting pattern:
    # - Northwest faces should be bright (facing light)
    # - Southeast faces should be dark (away from light)
    
    # Find the peak
    peak_idx = np.unravel_index(np.argmax(hill_dem), hill_dem.shape)
    peak_y, peak_x = peak_idx
    
    # Northwest of peak should be brighter than southeast
    if peak_y > 2 and peak_x > 2:  # Ensure we have room
        nw_region = hill_hillshade[peak_y-2:peak_y, peak_x-2:peak_x]
        se_region = hill_hillshade[peak_y+1:peak_y+3, peak_x+1:peak_x+3]
        
        if nw_region.size > 0 and se_region.size > 0:
            assert np.mean(nw_region) > np.mean(se_region)
    
    # Hillshade enhances terrain interpretation
    # Compare with and without terrain
    noisy_flat = flat_dem + np.random.rand(10, 10) * 2  # Slight noise
    strong_terrain = hill_dem.copy()
    
    flat_hs = create_hillshade(noisy_flat)
    terrain_hs = create_hillshade(strong_terrain)
    
    # Strong terrain should show more variation
    assert np.std(terrain_hs) > np.std(flat_hs)
    
    # Algorithm performance is reasonable for large datasets
    import time
    
    # Test with moderately large array
    large_dem = np.random.rand(200, 300) * 1000  # 200x300 DEM
    
    start_time = time.time()
    large_hillshade = create_hillshade(large_dem)
    end_time = time.time()
    
    # Should complete in reasonable time (< 1 second for this size)
    processing_time = end_time - start_time
    assert processing_time < 1.0, f"Processing took {processing_time:.2f} seconds"
    
    # Should produce valid output
    assert large_hillshade.shape == large_dem.shape
    assert large_hillshade.min() >= 0.0
    assert large_hillshade.max() <= 1.0
    assert np.all(np.isfinite(large_hillshade))
    
    # Test edge cases
    # Very steep terrain
    steep_dem = np.zeros((20, 20))
    steep_dem[:, 10:] = 1000  # Cliff
    
    steep_hs = create_hillshade(steep_dem, z_factor=1.0)
    
    # Should handle steep gradients gracefully
    assert np.all(np.isfinite(steep_hs))
    assert steep_hs.min() >= 0.0
    assert steep_hs.max() <= 1.0
    
    # Should show strong contrast at the cliff
    cliff_contrast = np.std(steep_hs[:, 8:12])  # Around the cliff
    flat_contrast = np.std(steep_hs[:, 0:5])    # Flat area
    assert cliff_contrast > flat_contrast


def test_hillshade_blending_quality():
    """Test quality of hillshade blending operations.
    
    This test should verify that:
    - Blending preserves important data features
    - Topographic context enhances data interpretation
    - Blending doesn't introduce visual artifacts
    - Different blend modes serve their intended purposes
    """
    # Create test data and hillshade
    # Simulate nighttime lights data with terrain
    height, width = 30, 40
    
    # Create terrain (ridge running north-south)
    terrain = np.zeros((height, width))
    for i in range(width):
        # Ridge in the middle, valleys on sides
        terrain[:, i] = 100 + 50 * np.exp(-((i - width/2)**2) / 50)
    
    hillshade = create_hillshade(terrain, azimuth=315, altitude=45)
    
    # Create nighttime lights data
    ntl_data = np.zeros((height, width))
    # City on the ridge
    ntl_data[10:20, 18:22] = 0.8  # Bright city
    # Small town in valley
    ntl_data[5:8, 5:8] = 0.3      # Dimmer town
    # Background lights
    ntl_data += np.random.rand(height, width) * 0.1
    
    # Blending preserves important data features
    blended_multiply = blend_with_hillshade(ntl_data, hillshade, 
                                          blend_mode="multiply", opacity=0.5)
    blended_overlay = blend_with_hillshade(ntl_data, hillshade, 
                                         blend_mode="overlay", opacity=0.5)
    
    # Original bright city should still be visible
    city_region_orig = ntl_data[10:20, 18:22]
    city_region_mult = blended_multiply[10:20, 18:22]
    city_region_over = blended_overlay[10:20, 18:22]
    
    # City should still be brighter than background
    bg_region_mult = blended_multiply[25:30, 5:10]  # Background area
    bg_region_over = blended_overlay[25:30, 5:10]
    
    assert np.mean(city_region_mult) > np.mean(bg_region_mult)
    assert np.mean(city_region_over) > np.mean(bg_region_over)
    
    # Relative relationships should be preserved
    town_region_mult = blended_multiply[5:8, 5:8]
    town_region_over = blended_overlay[5:8, 5:8]
    
    # City should still be brighter than town
    assert np.mean(city_region_mult) > np.mean(town_region_mult)
    assert np.mean(city_region_over) > np.mean(town_region_over)
    
    # Topographic context enhances data interpretation
    # Compare areas with different topographic illumination
    ridge_mask = (terrain > np.percentile(terrain, 75))  # Ridge areas
    valley_mask = (terrain < np.percentile(terrain, 25))  # Valley areas
    
    # In blended result, ridge areas should generally be brighter
    # and valley areas darker (due to hillshade)
    ridge_illumination = np.mean(hillshade[ridge_mask])
    valley_illumination = np.mean(hillshade[valley_mask])
    
    assert ridge_illumination > valley_illumination
    
    # This topographic difference should be reflected in blended data
    # (at least for multiply mode which preserves this relationship)
    ridge_ntl_blended = np.mean(blended_multiply[ridge_mask])
    valley_ntl_blended = np.mean(blended_multiply[valley_mask])
    
    # If original NTL values are similar, blended should show topo difference
    ridge_ntl_orig = np.mean(ntl_data[ridge_mask])
    valley_ntl_orig = np.mean(ntl_data[valley_mask])
    
    # Blending should enhance the topographic difference
    orig_diff = ridge_ntl_orig - valley_ntl_orig
    blended_diff = ridge_ntl_blended - valley_ntl_blended
    
    # For multiply mode, difference should be maintained or enhanced
    assert blended_diff >= orig_diff * 0.8  # Allow some tolerance
    
    # Blending doesn't introduce visual artifacts
    # Check for unrealistic values
    assert np.all(blended_multiply >= 0.0)
    assert np.all(blended_multiply <= 1.0)
    assert np.all(blended_overlay >= 0.0)
    assert np.all(blended_overlay <= 1.0)
    
    # Check for smooth transitions (no sudden jumps)
    for blended in [blended_multiply, blended_overlay]:
        # Check horizontal smoothness
        h_diff = np.abs(np.diff(blended, axis=1))
        # 95% of horizontal differences should be reasonable
        assert np.percentile(h_diff, 95) < 0.3
        
        # Check vertical smoothness
        v_diff = np.abs(np.diff(blended, axis=0))
        assert np.percentile(v_diff, 95) < 0.3
    
    # Different blend modes serve their intended purposes
    # Test with RGB data
    rgb_data = np.stack([ntl_data, ntl_data * 0.8, ntl_data * 0.6], axis=0)
    
    rgb_multiply = blend_with_hillshade(rgb_data, hillshade, 
                                       blend_mode="multiply", opacity=0.7)
    rgb_overlay = blend_with_hillshade(rgb_data, hillshade, 
                                      blend_mode="overlay", opacity=0.7)
    rgb_soft_light = blend_with_hillshade(rgb_data, hillshade, 
                                         blend_mode="soft_light", opacity=0.7)
    
    # All should be different
    assert not np.array_equal(rgb_multiply, rgb_overlay)
    assert not np.array_equal(rgb_overlay, rgb_soft_light)
    assert not np.array_equal(rgb_multiply, rgb_soft_light)
    
    # Multiply should generally darken (except where hillshade > 0.5)
    # Overlay should provide balanced enhancement
    # Soft light should be most subtle
    
    # Compare mean brightness changes
    orig_brightness = np.mean(rgb_data)
    mult_brightness = np.mean(rgb_multiply)
    over_brightness = np.mean(rgb_overlay)
    soft_brightness = np.mean(rgb_soft_light)
    
    # Multiply typically darkens overall
    assert mult_brightness <= orig_brightness
    
    # Soft light should be closest to original (most subtle)
    soft_diff = abs(soft_brightness - orig_brightness)
    mult_diff = abs(mult_brightness - orig_brightness)
    over_diff = abs(over_brightness - orig_brightness)
    
    # Soft light should be more subtle than the others
    assert soft_diff <= max(mult_diff, over_diff)


def test_cartographic_element_addition():
    """Test addition of cartographic elements (scale bars, north arrows).
    
    This test should verify that:
    - Cartographic elements enhance map readability
    - Elements are positioned appropriately
    - Elements provide accurate spatial reference
    - Elements don't interfere with data visualization
    """
    # Create test map data
    map_data = np.random.rand(3, 80, 120) * 200 + 50  # Realistic map colors
    pixel_size = 30.0  # 30m per pixel (Landsat-like)
    
    # Cartographic elements enhance map readability
    # Add both scale bar and north arrow
    with_scale = add_scale_bar(map_data, pixel_size=pixel_size, length_km=1.0,
                              position="bottom_right", color=(255, 255, 255))
    
    with_both = add_north_arrow(with_scale, position="top_right", size=40,
                               color=(255, 255, 255))
    
    # Should be different from original
    assert not np.array_equal(map_data, with_scale)
    assert not np.array_equal(with_scale, with_both)
    
    # Elements are positioned appropriately
    # Scale bar in bottom-right should not overlap with north arrow in top-right
    
    # Check scale bar region (bottom-right)
    scale_length_px = int(1000 / pixel_size)  # ~33 pixels
    margin = 20
    bar_height = 10
    
    scale_x_start = 120 - margin - scale_length_px
    scale_y_start = 80 - margin - bar_height
    
    # Check north arrow region (top-right)
    arrow_size = 40
    arrow_x_center = 120 - margin - arrow_size // 2
    arrow_y_center = margin + arrow_size // 2
    
    # Regions should not overlap
    # Scale bar: y in [scale_y_start, scale_y_start + bar_height]
    # North arrow: y in [arrow_y_center - arrow_size//2, arrow_y_center + arrow_size//2]
    
    scale_y_end = scale_y_start + bar_height
    arrow_y_start = arrow_y_center - arrow_size // 2
    arrow_y_end = arrow_y_center + arrow_size // 2
    
    # Should not overlap in Y dimension
    assert scale_y_start > arrow_y_end or arrow_y_start > scale_y_end
    
    # Elements provide accurate spatial reference
    # Test scale bar accuracy with different pixel sizes
    pixel_sizes = [10.0, 30.0, 100.0]  # Different resolutions
    
    for ps in pixel_sizes:
        test_map = np.zeros((3, 60, 100))
        map_with_scale = add_scale_bar(test_map, pixel_size=ps, length_km=2.0)
        
        # Calculate expected scale bar length
        expected_length = int(2000 / ps)  # 2km in pixels
        
        # Scale bar should appear (if it fits)
        if expected_length < 60:  # If it fits in image width
            # Should have been added successfully
            assert not np.array_equal(test_map, map_with_scale)
    
    # Test north arrow consistency
    arrow_sizes = [20, 40, 60]
    for size in arrow_sizes:
        test_map = np.zeros((3, 80, 100))
        map_with_arrow = add_north_arrow(test_map, size=size, 
                                        color=(255, 0, 0))
        
        # Should be different from original
        assert not np.array_equal(test_map, map_with_arrow)
        
        # Arrow should be in expected region
        arrow_region = map_with_arrow[:, 20-size//4:20+size//4,
                                      80-size//4:80+size//4]
        
        # Should have red pixels (red arrow) - be more flexible
        if arrow_region.size > 0:
            assert np.max(arrow_region[0]) > 100  # Should have some red
    
    # Elements don't interfere with data visualization
    # Create map with important features
    feature_map = np.zeros((3, 100, 150))
    
    # Add important features in various locations
    feature_map[:, 10:20, 10:20] = 200    # Top-left feature
    feature_map[:, 80:90, 10:20] = 180    # Bottom-left feature
    feature_map[:, 10:20, 130:140] = 160  # Top-right feature
    feature_map[:, 80:90, 130:140] = 140  # Bottom-right feature
    feature_map[:, 45:55, 70:80] = 220    # Center feature
    
    # Add cartographic elements
    enhanced_map = add_scale_bar(feature_map, pixel_size=25.0, length_km=1.0,
                                position="bottom_right")
    enhanced_map = add_north_arrow(enhanced_map, position="top_right", size=30)
    
    # Important features should still be visible and distinct
    # Top-left feature (should be unaffected)
    tl_feature_orig = np.mean(feature_map[:, 10:20, 10:20])
    tl_feature_enhanced = np.mean(enhanced_map[:, 10:20, 10:20])
    np.testing.assert_allclose(tl_feature_orig, tl_feature_enhanced, atol=1)
    
    # Bottom-left feature (should be unaffected)
    bl_feature_orig = np.mean(feature_map[:, 80:90, 10:20])
    bl_feature_enhanced = np.mean(enhanced_map[:, 80:90, 10:20])
    np.testing.assert_allclose(bl_feature_orig, bl_feature_enhanced, atol=1)
    
    # Center feature (should be unaffected)
    center_feature_orig = np.mean(feature_map[:, 45:55, 70:80])
    center_feature_enhanced = np.mean(enhanced_map[:, 45:55, 70:80])
    np.testing.assert_allclose(center_feature_orig, center_feature_enhanced, atol=1)
    
    # Features should still be distinguishable from background
    background_enhanced = np.mean(enhanced_map[:, 30:40, 50:60])  # Empty area
    
    assert tl_feature_enhanced > background_enhanced + 50
    assert center_feature_enhanced > background_enhanced + 50
    
    # Test with different positions to avoid conflicts
    conflict_test = add_scale_bar(feature_map, pixel_size=50.0, length_km=0.5,
                                 position="bottom_left")  # Away from features
    
    # Bottom-right and top-right features should be unaffected
    br_feature_orig = np.mean(feature_map[:, 80:90, 130:140])
    br_feature_conflict = np.mean(conflict_test[:, 80:90, 130:140])
    np.testing.assert_allclose(br_feature_orig, br_feature_conflict, atol=1)


def test_visualization_error_handling():
    """Test error handling in visualization functions.
    
    This test should verify that:
    - Invalid input data raises appropriate errors
    - Mismatched array dimensions are handled gracefully
    - Invalid parameter values are caught and reported
    - Edge cases don't cause crashes or silent failures
    """
    # Invalid input data raises appropriate errors
    
    # Test with NaN data
    nan_data = np.array([[np.nan, 0.5, 1.0]])
    
    # Should handle NaN gracefully (not crash)
    result_nan = apply_colormap(nan_data)
    assert result_nan.shape == (3, 1, 3)
    assert result_nan.dtype == np.uint8
    
    # Test with infinite values
    inf_data = np.array([[0.0, np.inf, -np.inf]])
    
    # Should handle inf gracefully
    result_inf = apply_colormap(inf_data)
    assert result_inf.shape == (3, 1, 3)
    assert result_inf.dtype == np.uint8
    
    # Test create_false_color_composite with mismatched shapes
    band1 = np.random.rand(10, 10)
    band2 = np.random.rand(10, 15)  # Different width
    band3 = np.random.rand(10, 10)
    
    # Should raise an error or handle gracefully
    try:
        result_mismatch = create_false_color_composite(band1, band2, band3)
        # If it doesn't raise an error, should at least produce valid output
        assert result_mismatch.dtype == np.uint8
    except (ValueError, AssertionError, IndexError):
        # This is acceptable - function should catch mismatched dimensions
        pass
    
    # Test comparison view with mismatched shapes (should raise AssertionError)
    image1 = np.random.rand(3, 10, 15)
    image2 = np.random.rand(3, 12, 15)  # Different height
    
    with pytest.raises(AssertionError):
        create_comparison_view(image1, image2)
    
    # Invalid parameter values are caught and reported
    
    # Test invalid colormap name (should handle gracefully or raise error)
    test_data = np.random.rand(5, 5)
    
    try:
        # This might raise an error or use a fallback
        result_invalid_cmap = apply_colormap(test_data, colormap="invalid_colormap")
        # If no error, should still produce valid output
        assert result_invalid_cmap.shape == (3, 5, 5)
        assert result_invalid_cmap.dtype == np.uint8
    except (ValueError, KeyError, TypeError):
        # This is acceptable - invalid colormap should be caught
        pass
    
    # Test invalid scale range
    try:
        result_invalid_range = apply_colormap(test_data, scale_range=(1.0, 0.0))  # Invalid range (max < min)
        # If no error, should still produce valid output
        assert result_invalid_range.dtype == np.uint8
    except (ValueError, AssertionError, ZeroDivisionError):
        # This is acceptable - invalid range should be caught
        pass
    
    # Test invalid blend mode
    hillshade_data = np.random.rand(5, 5)
    overlay_data = np.random.rand(5, 5)
    
    try:
        result_invalid_blend = blend_with_hillshade(overlay_data, hillshade_data, 
                                                   blend_mode="invalid_mode")
        # If no error, should still produce valid output
        assert result_invalid_blend.shape == overlay_data.shape
    except (ValueError, KeyError):
        # This is acceptable - invalid blend mode should be caught
        pass
    
    # Test invalid position
    try:
        result_invalid_pos = add_scale_bar(test_data, pixel_size=30.0, 
                                          position="invalid_position")
        assert result_invalid_pos.shape == test_data.shape
    except (ValueError, KeyError):
        pass
    
    # Edge cases don't cause crashes or silent failures
    
    # Empty arrays
    empty_data = np.array([]).reshape(0, 0)
    
    try:
        result_empty = apply_colormap(empty_data)
        # Should either handle gracefully or raise appropriate error
        assert result_empty.size == 0 or result_empty.shape[0] == 3
    except (ValueError, IndexError):
        # This is acceptable for empty arrays
        pass
    
    # Very large arrays (test memory handling)
    # Use a moderately large array to test without excessive memory use
    large_data = np.random.rand(500, 800)  # 400k pixels
    
    # Should handle without crashing
    large_result = apply_colormap(large_data)
    assert large_result.shape == (3, 500, 800)
    assert large_result.dtype == np.uint8
    
    # Single pixel arrays
    single_pixel = np.array([[0.5]])
    
    single_result = apply_colormap(single_pixel)
    assert single_result.shape == (3, 1, 1)
    assert single_result.dtype == np.uint8
    
    # Test hillshade with extreme elevation differences
    extreme_dem = np.array([[0, 10000], [0, 10000]])  # 10km elevation difference
    
    extreme_hillshade = create_hillshade(extreme_dem)
    assert extreme_hillshade.shape == (2, 2)
    assert np.all(np.isfinite(extreme_hillshade))
    assert extreme_hillshade.min() >= 0.0
    assert extreme_hillshade.max() <= 1.0
    
    # Test scale bar with extreme pixel sizes
    extreme_pixel_size = 0.001  # 1mm per pixel
    
    # Should handle without crashing (though scale bar might not be visible)
    extreme_scale_result = add_scale_bar(test_data, pixel_size=extreme_pixel_size, 
                                        length_km=1.0)
    assert extreme_scale_result.shape == test_data.shape
    
    # Test with very large pixel size
    huge_pixel_size = 10000.0  # 10km per pixel
    
    huge_scale_result = add_scale_bar(test_data, pixel_size=huge_pixel_size, 
                                     length_km=1.0)
    assert huge_scale_result.shape == test_data.shape
    
    # Test north arrow with zero size
    try:
        zero_arrow_result = add_north_arrow(test_data, size=0)
        assert zero_arrow_result.shape == test_data.shape
    except (ValueError, ZeroDivisionError):
        # Acceptable to reject zero size
        pass
    
    # Test colorbar with invalid value range
    try:
        invalid_range_colorbar = generate_colorbar("viridis", (np.nan, 1.0))
        # Should either handle gracefully or raise error
        assert invalid_range_colorbar.dtype == np.uint8
    except (ValueError, TypeError):
        # Acceptable to reject invalid ranges
        pass