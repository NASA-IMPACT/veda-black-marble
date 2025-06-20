"""Behavioral tests for enhancement components.

These tests use predictable artificial data to verify exact behavior
of enhancement components without external dependencies.
"""

import numpy as np
import pytest

from blackmarble.components.enhancement import (
    RoadEnhancer,
    ContrastEnhancer,
    ColorMapper,
    CompositeEnhancer,
    EnhancementResult,
)


class TestRoadEnhancer:
    """Test RoadEnhancer multiplies values correctly."""
    
    def test_basic_road_enhancement(self):
        """Test basic road enhancement with default factors."""
        enhancer = RoadEnhancer(major_factor=1.3, minor_factor=1.1)
        
        # Create test data
        data = np.array([
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90]
        ], dtype=np.float32)
        
        # Road classification: 0=no road, 1=motorway, 3=secondary
        roads = np.array([
            [0, 1, 0],
            [3, 0, 1],
            [0, 3, 0]
        ], dtype=np.int32)
        
        result = enhancer.enhance(data, roads)
        
        # Check enhanced values
        assert pytest.approx(result.data[0, 0]) == 10  # No road, unchanged
        assert result.data[0, 1] == pytest.approx(26.0)  # 20 * 1.3 (major)
        assert result.data[1, 0] == pytest.approx(44.0)  # 40 * 1.1 (minor)
        assert result.data[1, 2] == pytest.approx(78.0)  # 60 * 1.3 (major)
        
    def test_custom_enhancement_factors(self):
        """Test road enhancement with custom factors."""
        enhancer = RoadEnhancer()
        
        data = np.array([
            [100, 100, 100],
            [100, 100, 100]
        ], dtype=np.float32)
        
        roads = np.array([
            [1, 2, 3],
            [4, 5, 0]
        ], dtype=np.int32)
        
        custom_factors = {
            1: 1.5,
            2: 1.4,
            3: 1.3,
            4: 1.2,
            5: 1.1
        }
        
        result = enhancer.enhance(data, roads, custom_factors=custom_factors)
        
        assert pytest.approx(result.data[0, 0]) == 150  # 100 * 1.5
        assert pytest.approx(result.data[0, 1]) == 140  # 100 * 1.4
        assert pytest.approx(result.data[0, 2]) == 130  # 100 * 1.3
        assert pytest.approx(result.data[1, 0]) == 120  # 100 * 1.2
        assert pytest.approx(result.data[1, 1]) == 110  # 100 * 1.1
        assert pytest.approx(result.data[1, 2]) == 100  # No road
        
    def test_enhancement_statistics(self):
        """Test enhancement statistics calculation."""
        enhancer = RoadEnhancer(major_factor=2.0, minor_factor=1.5)
        
        data = np.ones((10, 10), dtype=np.float32) * 50
        
        # Create pattern with known road counts
        roads = np.zeros((10, 10), dtype=np.int32)
        roads[0:2, :] = 1  # 20 motorway pixels
        roads[2:5, :] = 3  # 30 secondary pixels
        # 50 no-road pixels
        
        result = enhancer.enhance(data, roads)
        
        stats = result.statistics
        assert stats["road_pixels"] == 50
        assert stats["total_pixels"] == 100
        assert stats["road_percentage"] == 50.0
        assert stats["road_counts"]["motorway"] == 20
        assert stats["road_counts"]["secondary"] == 30
        assert stats["road_counts"]["no_road"] == 50
        
    def test_zero_data_enhancement(self):
        """Test enhancement of zero values."""
        enhancer = RoadEnhancer(major_factor=1.5)
        
        data = np.zeros((3, 3), dtype=np.float32)
        roads = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ], dtype=np.int32)
        
        result = enhancer.enhance(data, roads)
        
        # Zero times any factor is still zero
        assert np.all(result.data == 0)
        
    def test_negative_value_enhancement(self):
        """Test enhancement preserves sign of negative values."""
        enhancer = RoadEnhancer(major_factor=1.5, minor_factor=1.2)
        
        data = np.array([
            [-10, -20],
            [-30, -40]
        ], dtype=np.float32)
        
        roads = np.array([
            [1, 0],
            [3, 0]
        ], dtype=np.int32)
        
        result = enhancer.enhance(data, roads)
        
        assert result.data[0, 0] == pytest.approx(-15)  # -10 * 1.5
        assert result.data[1, 0] == pytest.approx(-36)  # -30 * 1.2
        
    def test_process_method_mask_generation(self):
        """Test process method generates correct mask."""
        enhancer = RoadEnhancer()
        
        data = np.ones((5, 5), dtype=np.float32)
        roads = np.array([
            [0, 1, 0, 0, 0],
            [0, 0, 2, 0, 0],
            [0, 0, 0, 3, 0],
            [0, 0, 0, 0, 4],
            [5, 0, 0, 0, 0]
        ], dtype=np.int32)
        
        result = enhancer.process(data, roads=roads)
        
        # Mask should be 1 where roads > 0
        expected_mask = (roads > 0).astype(float)
        assert np.array_equal(result.mask, expected_mask)
        assert result.statistics.valid_pixels == 5  # 5 road pixels


class TestContrastEnhancer:
    """Test ContrastEnhancer scaling methods."""
    
    def test_percentile_stretch(self):
        """Test percentile-based contrast stretch."""
        enhancer = ContrastEnhancer(
            method="percentile",
            clip_percentiles=(10.0, 90.0)
        )
        
        # Create data with outliers
        data = np.array([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,  # Normal range
            100, 200  # Outliers
        ], dtype=np.float32).reshape(3, 4)
        
        result = enhancer.enhance(data, output_range=(0.0, 1.0))
        
        # 10th percentile ≈ 1, 90th percentile ≈ 8
        # Values should be stretched between these
        enhanced = result.data
        
        # Check clipping of outliers
        assert enhanced[2, 2] >= 1.0  # 100 should be at or above max
        assert enhanced[2, 3] >= 1.0  # 200 should be at or above max
        
        # Check stretched values
        assert enhanced[0, 1] == pytest.approx(0.0, abs=0.1)  # 1 -> ~0
        assert enhanced[1, 2] == pytest.approx(0.71, abs=0.1)  # 6 -> ~0.71
        
    def test_minmax_stretch(self):
        """Test min-max contrast stretch."""
        enhancer = ContrastEnhancer(method="minmax")
        
        data = np.array([
            [5, 10, 15],
            [20, 25, 30]
        ], dtype=np.float32)
        
        result = enhancer.enhance(data, output_range=(0.0, 1.0))
        
        # Min=5, Max=30, so range is 25
        assert result.data[0, 0] == 0.0  # (5-5)/25 = 0
        assert result.data[0, 1] == 0.2  # (10-5)/25 = 0.2
        assert result.data[1, 2] == 1.0  # (30-5)/25 = 1
        
    def test_adaptive_stretch(self):
        """Test adaptive contrast stretch."""
        enhancer = ContrastEnhancer(method="adaptive")
        
        # Create data with specific mean and std
        data = np.array([
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90]
        ], dtype=np.float32)
        
        result = enhancer.enhance(
            data,
            output_range=(0.0, 1.0),
            target_mean=0.5,
            target_std=0.2
        )
        
        enhanced_flat = result.data.flatten()
        valid_data = enhanced_flat[~np.isnan(enhanced_flat)]
        
        # Check that we achieved target statistics (approximately)
        assert np.mean(valid_data) == pytest.approx(0.5, abs=0.1)
        assert np.std(valid_data) == pytest.approx(0.2, abs=0.05)
        
    def test_masked_enhancement(self):
        """Test enhancement with mask."""
        enhancer = ContrastEnhancer(method="percentile")
        
        data = np.array([
            [1, 2, 100],  # 100 is outlier
            [3, 4, 200],  # 200 is outlier
            [5, 6, 300]   # 300 is outlier
        ], dtype=np.float32)
        
        # Mask excludes outliers
        mask = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0]
        ], dtype=float)
        
        result = enhancer.enhance(data, mask=mask)
        
        # Should use only masked values for statistics
        # Valid data: 1-6, so stretch should map these to 0-1
        assert result.data[0, 0] == pytest.approx(0.0, abs=0.1)  # 1 -> 0
        assert result.data[2, 1] == pytest.approx(1.0, abs=0.1)  # 6 -> 1
        
    def test_no_valid_data(self):
        """Test enhancement with no valid data."""
        enhancer = ContrastEnhancer()
        
        data = np.full((3, 3), np.nan, dtype=np.float32)
        
        result = enhancer.enhance(data)
        
        assert np.all(result.data == 0)
        assert result.statistics["valid_pixels"] == 0
        
    def test_constant_data(self):
        """Test enhancement of constant values."""
        enhancer = ContrastEnhancer(method="adaptive")
        
        data = np.ones((5, 5), dtype=np.float32) * 42
        
        result = enhancer.enhance(data, target_mean=0.7)
        
        # With no variation, should return target mean
        assert np.all(result.data == 0.7)
        
    def test_output_range_scaling(self):
        """Test different output ranges."""
        enhancer = ContrastEnhancer(method="minmax")
        
        data = np.array([0, 50, 100], dtype=np.float32)
        
        # Scale to 0-255
        result1 = enhancer.enhance(data, output_range=(0, 255))
        assert result1.data[0] == 0
        assert result1.data[1] == 127.5
        assert result1.data[2] == 255
        
        # Scale to -1 to 1
        result2 = enhancer.enhance(data, output_range=(-1, 1))
        assert result2.data[0] == -1
        assert result2.data[1] == 0
        assert result2.data[2] == 1
        
    def test_statistics_reporting(self):
        """Test enhancement statistics."""
        enhancer = ContrastEnhancer(method="percentile")
        
        data = np.linspace(0, 100, 100).reshape(10, 10).astype(np.float32)
        
        result = enhancer.enhance(data)
        stats = result.statistics
        
        assert stats["valid_pixels"] == 100
        assert stats["total_pixels"] == 100
        assert "original_range" in stats
        assert "enhanced_range" in stats
        assert "original_mean" in stats
        assert "enhanced_mean" in stats


class TestColorMapper:
    """Test ColorMapper produces expected colors."""
    
    def test_grayscale_colormap(self):
        """Test default grayscale colormap."""
        mapper = ColorMapper()
        
        data = np.array([
            [0.0, 0.5, 1.0],
            [0.25, 0.75, 0.5]
        ], dtype=np.float32)
        
        rgb = mapper.apply_colormap(data, scale_range=(0.0, 1.0))
        
        assert rgb.shape == (3, 2, 3)  # RGB channels
        assert rgb.dtype == np.uint8
        
        # Check grayscale values
        assert rgb[0, 0, 0] == 0    # Black
        assert rgb[0, 0, 2] == 255  # White
        assert rgb[0, 1, 0] == 64   # 25% gray
        assert rgb[0, 1, 1] == 191  # 75% gray
        
        # All channels should be equal (grayscale)
        assert np.array_equal(rgb[0], rgb[1])
        assert np.array_equal(rgb[1], rgb[2])
        
    def test_custom_colormap(self):
        """Test custom colormap application."""
        # Create blue-white-red colormap
        colormap_data = {
            "r": np.concatenate([np.zeros(128), np.linspace(0, 255, 128)]).astype(np.uint8),
            "g": np.concatenate([np.zeros(128), np.linspace(0, 255, 128)]).astype(np.uint8),
            "b": np.concatenate([np.linspace(255, 0, 128), np.zeros(128)]).astype(np.uint8)
        }
        
        mapper = ColorMapper(colormap_data=colormap_data)
        
        data = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        
        rgb = mapper.apply_colormap(data, scale_range=(0.0, 1.0))
        
        # Check colors
        # 0.0 -> Blue (0, 0, 255)
        assert rgb[0, 0, 0] == 0
        assert rgb[1, 0, 0] == 0
        assert rgb[2, 0, 0] == 255
        
        # 0.5 -> ~White (127, 127, 127)
        assert 120 < rgb[0, 0, 1] < 135
        assert 120 < rgb[1, 0, 1] < 135
        assert 120 < rgb[2, 0, 1] < 135
        
        # 1.0 -> Red (255, 255, 0)
        assert rgb[0, 0, 2] == 255
        assert rgb[1, 0, 2] == 255
        assert rgb[2, 0, 2] == 0
        
    def test_alpha_channel(self):
        """Test alpha channel generation."""
        mapper = ColorMapper(add_alpha=True)
        
        data = np.array([
            [1.0, np.nan],
            [0.5, 0.0]
        ], dtype=np.float32)
        
        rgba = mapper.apply_colormap(data, scale_range=(0.0, 1.0))
        
        assert rgba.shape == (4, 2, 2)  # RGBA channels
        
        # Check alpha channel
        assert rgba[3, 0, 0] == 255  # Valid pixel
        assert rgba[3, 0, 1] == 0    # NaN pixel
        assert rgba[3, 1, 0] == 255  # Valid pixel
        assert rgba[3, 1, 1] == 255  # Valid pixel (0 is valid)
        
    def test_custom_alpha_mask(self):
        """Test custom alpha mask."""
        mapper = ColorMapper(add_alpha=True)
        
        data = np.ones((3, 3), dtype=np.float32)
        alpha_mask = np.array([
            [1.0, 0.5, 0.0],
            [0.75, 0.25, 0.5],
            [0.0, 1.0, 0.5]
        ], dtype=np.float32)
        
        rgba = mapper.apply_colormap(data, alpha_mask=alpha_mask)
        
        # Check alpha values
        assert rgba[3, 0, 0] == 255  # 1.0 -> 255
        assert rgba[3, 0, 1] == 127  # 0.5 -> 127
        assert rgba[3, 0, 2] == 0    # 0.0 -> 0
        assert rgba[3, 1, 0] == 191  # 0.75 -> 191
        assert rgba[3, 1, 1] == 64   # 0.25 -> 64
        
    def test_auto_scaling(self):
        """Test automatic range scaling."""
        mapper = ColorMapper()
        
        # Data not in 0-1 range
        data = np.array([
            [10, 20, 30],
            [15, 25, 35]
        ], dtype=np.float32)
        
        rgb = mapper.apply_colormap(data)  # No scale_range provided
        
        # Should auto-scale from 10-35
        assert rgb[0, 0, 0] == 0    # 10 -> 0
        assert rgb[0, 0, 2] == 255  # 35 -> 255
        # Middle values
        assert 90 < rgb[0, 0, 1] < 110   # ~102 for value 20
        assert 140 < rgb[0, 1, 1] < 160  # ~153 for value 25
        
    def test_nan_handling(self):
        """Test NaN value handling in colormap."""
        mapper = ColorMapper()
        
        data = np.array([
            [0.0, np.nan, 1.0],
            [np.nan, 0.5, np.nan]
        ], dtype=np.float32)
        
        rgb = mapper.apply_colormap(data, scale_range=(0.0, 1.0))
        
        # NaN should map to 0 (black)
        assert rgb[0, 0, 1] == 0
        assert rgb[1, 0, 1] == 0
        assert rgb[2, 0, 1] == 0
        
        assert rgb[0, 1, 0] == 0
        assert rgb[1, 1, 0] == 0
        assert rgb[2, 1, 0] == 0
        
    def test_colorbar_generation(self):
        """Test colorbar image generation."""
        # Custom colormap for testing
        colormap_data = {
            "r": np.linspace(0, 255, 256).astype(np.uint8),
            "g": np.linspace(255, 0, 256).astype(np.uint8),
            "b": np.zeros(256, dtype=np.uint8)
        }
        
        mapper = ColorMapper(colormap_data=colormap_data)
        
        # Vertical colorbar
        colorbar_v = mapper.create_colorbar(height=256, width=32, orientation="vertical")
        assert colorbar_v.shape == (3, 256, 32)
        # Top should be red (255, 0, 0)
        assert colorbar_v[0, 0, 0] == 255
        assert colorbar_v[1, 0, 0] == 0
        # Bottom should be green (0, 255, 0)
        assert colorbar_v[0, 255, 0] == 0
        assert colorbar_v[1, 255, 0] == 255
        
        # Horizontal colorbar
        colorbar_h = mapper.create_colorbar(height=32, width=256, orientation="horizontal")
        assert colorbar_h.shape == (3, 32, 256)
        # Left should be green
        assert colorbar_h[0, 0, 0] == 0
        assert colorbar_h[1, 0, 0] == 255
        # Right should be red
        assert colorbar_h[0, 0, 255] == 255
        assert colorbar_h[1, 0, 255] == 0


class TestCompositeEnhancer:
    """Test CompositeEnhancer chaining operations."""
    
    def test_enhancement_chain(self):
        """Test chaining multiple enhancements."""
        # Create enhancement pipeline
        contrast = ContrastEnhancer(method="minmax")
        road_enhancer = RoadEnhancer(major_factor=1.5)
        color_mapper = ColorMapper()
        
        composite = CompositeEnhancer(enhancers=[contrast, road_enhancer])
        
        # Test data
        data = np.array([
            [10, 20, 30],
            [40, 50, 60]
        ], dtype=np.float32)
        
        roads = np.array([
            [0, 1, 0],
            [1, 0, 1]
        ], dtype=np.int32)
        
        result = composite.enhance(data, roads=roads)
        
        # First contrast stretch (10-60 -> 0-1)
        # Then road enhancement (multiply by 1.5 where roads=1)
        expected = np.array([
            [0.0, 0.3, 0.4],     # 20 stretched to 0.2, then * 1.5 = 0.3
            [0.9, 0.8, 1.5]      # 40->0.6*1.5=0.9, 50->0.8, 60->1.0*1.5=1.5
        ], dtype=np.float32)
        
        assert np.allclose(result.data, expected, atol=0.01)
        
    def test_add_enhancer_dynamically(self):
        """Test adding enhancers dynamically."""
        composite = CompositeEnhancer()
        
        assert len(composite.enhancers) == 0
        
        # Add enhancers
        composite.add_enhancer(ContrastEnhancer())
        assert len(composite.enhancers) == 1
        
        composite.add_enhancer(RoadEnhancer())
        assert len(composite.enhancers) == 2
        
    def test_statistics_aggregation(self):
        """Test that statistics from all steps are collected."""
        enhancer1 = ContrastEnhancer(method="percentile")
        enhancer2 = ContrastEnhancer(method="adaptive")
        
        composite = CompositeEnhancer(enhancers=[enhancer1, enhancer2])
        
        data = np.random.rand(10, 10).astype(np.float32)
        
        result = composite.enhance(data)
        
        # Should have statistics from both steps
        assert "step_0_contrast_enhancer" in result.statistics
        assert "step_1_contrast_enhancer" in result.statistics
        
    def test_colormap_in_chain(self):
        """Test including ColorMapper in enhancement chain."""
        contrast = ContrastEnhancer(method="minmax")
        mapper = ColorMapper()
        
        composite = CompositeEnhancer(enhancers=[contrast, mapper])
        
        data = np.array([[0, 50], [75, 100]], dtype=np.float32)
        
        result = composite.enhance(data)
        
        # Result should be RGB
        assert result.data.shape == (3, 2, 2)
        assert result.data.dtype == np.uint8
        
    def test_empty_enhancer_chain(self):
        """Test composite enhancer with no enhancers."""
        composite = CompositeEnhancer()
        
        data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        
        result = composite.enhance(data)
        
        # Should return data unchanged
        assert np.array_equal(result.data, data)
        assert len(result.statistics) == 0