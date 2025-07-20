"""Test enhancement operations for visualization.

The pipeline applies contrast enhancement and color mapping
to create visually appealing outputs.
"""

import numpy as np
import pytest

from blackmarble.enhance import (
    enhance_contrast,
    apply_inferno_colormap,
    enhance_roads,
    enhance_urban_features
)


class TestContrastEnhancement:
    """Test contrast enhancement functionality."""
    
    def test_percentile_contrast_enhancement(self):
        """Test percentile-based contrast enhancement."""
        # Create data with outliers
        data = np.random.normal(50, 10, (100, 100)).astype(np.float32)
        data[0, 0] = 200  # Outlier
        data[99, 99] = -50  # Outlier
        
        # Enhance using percentiles
        enhanced = enhance_contrast(
            data,
            method="percentile",
            percentiles=(2.0, 98.0),
            output_range=(0.0, 1.0)
        )
        
        # Check output range
        assert enhanced.min() >= 0.0
        assert enhanced.max() <= 1.0
        
        # Most values should span the range (outliers clipped)
        assert enhanced[50, 50] > 0.3  # Near middle value
        assert enhanced[50, 50] < 0.7
        
        # Outliers should be clipped
        assert enhanced[0, 0] == 1.0  # Max outlier
        assert enhanced[99, 99] == 0.0  # Min outlier
    
    def test_minmax_contrast_enhancement(self):
        """Test min-max contrast enhancement."""
        # Create data with known range
        data = np.linspace(10, 90, 100).reshape(10, 10).astype(np.float32)
        
        enhanced = enhance_contrast(
            data,
            method="minmax",
            output_range=(0.0, 1.0)
        )
        
        # Should use full range
        assert enhanced.min() == 0.0
        assert enhanced.max() == 1.0
        
        # Linear scaling
        assert np.allclose(enhanced[5, 0], 0.5, rtol=0.01)  # Middle value
    
    def test_contrast_with_nan_values(self):
        """Test contrast enhancement handles NaN values."""
        data = np.ones((50, 50), dtype=np.float32) * 50
        data[10:20, 10:20] = np.nan
        
        enhanced = enhance_contrast(
            data,
            method="percentile",
            percentiles=(5.0, 95.0)
        )
        
        # NaN should be preserved
        assert np.all(np.isnan(enhanced[10:20, 10:20]))
        
        # Non-NaN values should be enhanced
        assert not np.any(np.isnan(enhanced[0:10, 0:10]))


class TestColormapApplication:
    """Test colormap application for visualization."""
    
    def test_inferno_colormap_basic(self):
        """Test applying inferno colormap to data."""
        # Create gradient data
        data = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)
        
        # Apply colormap
        colored = apply_inferno_colormap(
            data,
            scale_range=(0.0, 1.0)
        )
        
        # Check output shape (should be 3-channel RGB)
        assert colored.shape == (3, 10, 10)
        assert colored.dtype == np.uint8
        
        # Check value range
        assert colored.min() >= 0
        assert colored.max() <= 255
        
        # Inferno colormap characteristics:
        # Low values -> dark (purple/black)
        # High values -> bright (yellow/white)
        assert np.mean(colored[:, 0, 0]) < np.mean(colored[:, 9, 9])
    
    def test_colormap_with_scale_range(self):
        """Test colormap with custom scale range."""
        # Data mostly in middle range
        data = np.ones((20, 20), dtype=np.float32) * 0.5
        data[5:10, 5:10] = 0.7  # Bright spot
        data[10:15, 10:15] = 0.3  # Dark spot
        
        # Apply with focused range
        colored = apply_inferno_colormap(
            data,
            scale_range=(0.3, 0.8)  # Focus on urban range
        )
        
        # Areas outside range should be clipped
        # 0.3 maps to 0.0 (darkest)
        # 0.8 maps to 1.0 (brightest)
        # 0.5 maps to 0.4 (mid-range)
        
        bright_mean = np.mean(colored[:, 7, 7])
        mid_mean = np.mean(colored[:, 0, 0])
        dark_mean = np.mean(colored[:, 12, 12])
        
        assert bright_mean > mid_mean > dark_mean
    
    def test_colormap_preserves_nan(self):
        """Test colormap handling of NaN values."""
        data = np.ones((10, 10), dtype=np.float32) * 0.5
        data[3:6, 3:6] = np.nan
        
        colored = apply_inferno_colormap(data)
        
        # NaN should become black (0, 0, 0)
        nan_region = colored[:, 3:6, 3:6]
        assert np.all(nan_region == 0)


class TestRoadEnhancement:
    """Test road network enhancement in NTL data."""
    
    def test_basic_road_enhancement(self):
        """Test enhancing roads in NTL data."""
        # Create NTL data
        ntl = np.ones((50, 50), dtype=np.float32) * 0.5
        
        # Create road mask (1=major, 2=minor)
        roads = np.zeros((50, 50), dtype=np.uint8)
        roads[25, :] = 1  # Major road
        roads[:, 25] = 2  # Minor road
        
        # Enhance
        enhanced = enhance_roads(
            data=ntl,
            road_mask=roads,
            major_factor=1.5,
            minor_factor=1.2
        )
        
        # Check enhancement
        assert enhanced[25, 10] == 0.5 * 1.5  # Major road
        assert enhanced[10, 25] == 0.5 * 1.2  # Minor road
        assert enhanced[10, 10] == 0.5  # No road
        
        # Intersection gets major factor
        assert enhanced[25, 25] == 0.5 * 1.5
    
    def test_custom_road_factors(self):
        """Test custom enhancement factors by road type."""
        ntl = np.ones((30, 30), dtype=np.float32)
        
        # Different road types
        roads = np.zeros((30, 30), dtype=np.uint8)
        roads[10, :] = 1  # Motorway
        roads[15, :] = 3  # Primary
        roads[20, :] = 5  # Residential
        
        # Custom factors
        custom_factors = {
            1: 2.0,   # Motorway
            3: 1.5,   # Primary
            5: 1.1    # Residential
        }
        
        enhanced = enhance_roads(
            data=ntl,
            road_mask=roads,
            custom_factors=custom_factors
        )
        
        assert enhanced[10, 10] == 2.0  # Motorway
        assert enhanced[15, 10] == 1.5  # Primary
        assert enhanced[20, 10] == 1.1  # Residential


class TestUrbanFeatureEnhancement:
    """Test urban feature enhancement."""
    
    def test_urban_enhancement(self):
        """Test enhancing urban features."""
        # Create test data
        shape = (40, 40)
        ntl = np.ones(shape, dtype=np.float32) * 0.5
        
        # Urban mask (binary)
        urban_mask = np.zeros(shape, dtype=bool)
        urban_mask[15:25, 15:25] = True
        
        # Road mask
        road_mask = np.zeros(shape, dtype=np.uint8)
        road_mask[20, :] = 1  # Road through urban area
        
        # Enhance
        enhanced = enhance_urban_features(
            ntl_data=ntl,
            urban_mask=urban_mask,
            road_mask=road_mask,
            enhancement_factor=1.2,
            road_factor=1.1
        )
        
        # Urban areas enhanced
        assert enhanced[20, 20] > ntl[20, 20]
        
        # Roads in urban areas get additional enhancement
        urban_road_value = enhanced[20, 20]
        urban_nonroad_value = enhanced[18, 20]
        assert urban_road_value > urban_nonroad_value
        
        # Non-urban unchanged
        assert enhanced[5, 5] == ntl[5, 5]


class TestEnhancementIntegration:
    """Test enhancement pipeline integration."""
    
    def test_full_enhancement_pipeline(self):
        """Test complete enhancement workflow as in pipeline."""
        # Create synthetic data
        shape = (100, 100)
        
        # NTL data with urban bright spots
        ntl = np.random.rand(*shape).astype(np.float32) * 10
        y, x = np.ogrid[:shape[0], :shape[1]]
        mask = (x - 50)**2 + (y - 50)**2 < 400
        ntl[mask] = 20  # Bright center
        
        # Roads
        roads = np.zeros(shape, dtype=np.uint8)
        roads[50, :] = 1  # Major road
        roads[:, 50] = 2  # Minor road
        
        # Urban mask from NDUI
        urban_mask = ntl > 15
        
        # Step 1: Contrast enhancement
        ntl_enhanced = enhance_contrast(
            ntl,
            method="percentile",
            percentiles=(2.0, 98.0),
            output_range=(0.0, 1.0)
        )
        
        # Step 2: Road enhancement
        ntl_roads = enhance_roads(
            ntl_enhanced,
            road_mask=roads,
            major_factor=1.3,
            minor_factor=1.1
        )
        
        # Step 3: Urban enhancement
        ntl_urban = enhance_urban_features(
            ntl_roads,
            urban_mask=urban_mask,
            road_mask=roads,
            enhancement_factor=1.2,
            road_factor=1.1
        )
        
        # Step 4: Colormap
        final = apply_inferno_colormap(
            ntl_urban,
            scale_range=(0.0, 1.0)
        )
        
        # Verify enhancements compound
        # Roads in urban areas should be brightest
        urban_road_brightness = np.mean(final[:, 50, 50])
        urban_nonroad_brightness = np.mean(final[:, 55, 55])
        rural_brightness = np.mean(final[:, 10, 10])
        
        assert urban_road_brightness > urban_nonroad_brightness > rural_brightness