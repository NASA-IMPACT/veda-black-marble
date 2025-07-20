"""Test urban field enhancement for NTL data.

The pipeline uses multi-scale Gaussian smoothing of road networks
to create urban density fields that enhance nighttime lights data.
"""

import numpy as np
import pytest

from blackmarble.analyze.urban_fields import (
    compute_urban_fields,
    enhance_ntl_with_urban_field
)


class TestUrbanFieldComputation:
    """Test urban field computation from road networks."""
    
    def test_basic_urban_field_computation(self):
        """Test basic urban field computation with single scale."""
        shape = (50, 50)
        
        # Create simple road pattern
        roads = np.zeros(shape, dtype=np.float32)
        roads[25, :] = 1.0  # Horizontal road
        roads[:, 25] = 1.0  # Vertical road (crossroads)
        
        # Compute urban field with single sigma
        urban_field = compute_urban_fields(
            road_fraction=roads,
            sigmas=[2.0],
            weights=[1.0]
        )
        
        # Check output
        assert urban_field.shape == shape
        assert urban_field.dtype == np.float32
        
        # Center (crossroads) should have highest value
        assert urban_field[25, 25] == np.max(urban_field)
        
        # Should decay away from roads
        assert urban_field[25, 30] < urban_field[25, 25]
        assert urban_field[30, 30] < urban_field[25, 30]
        
        # Far corners should have lowest values
        assert urban_field[0, 0] < 0.1
        assert urban_field[49, 49] < 0.1
    
    def test_multi_scale_urban_fields(self):
        """Test multi-scale urban field computation."""
        shape = (100, 100)
        
        # Create urban area with roads
        roads = np.zeros(shape, dtype=np.float32)
        # Dense grid in center
        roads[40:60, 40:60] = 0.8  # Urban core
        roads[30:70, 30:70] = np.maximum(roads[30:70, 30:70], 0.3)  # Suburban
        
        # Add some highways
        roads[50, :] = np.maximum(roads[50, :], 0.5)
        roads[:, 50] = np.maximum(roads[:, 50], 0.5)
        
        # Compute with multiple scales
        sigmas = [2.0, 5.0, 10.0]
        weights = [0.5, 0.3, 0.2]
        
        urban_field = compute_urban_fields(
            road_fraction=roads,
            sigmas=sigmas,
            weights=weights
        )
        
        # Check weighted combination
        assert np.sum(weights) == 1.0, "Weights should sum to 1"
        
        # Urban core should have high values
        core_mean = urban_field[45:55, 45:55].mean()
        assert core_mean > 0.6, f"Urban core too low: {core_mean}"
        
        # Should have smooth gradient
        # Sample along a line from center outward
        center_line = urban_field[50, 50:80]
        diffs = np.diff(center_line)
        assert np.all(diffs <= 0), "Urban field should decrease from center"
        
        # Different scales should create different decay rates
        # Large sigma should extend influence further
        assert urban_field[50, 70] > urban_field[50, 90], "Should decay with distance"
    
    def test_urban_field_with_fractional_roads(self):
        """Test urban field with fractional road coverage values."""
        shape = (50, 50)
        
        # Create fractional road coverage (0-1 values)
        roads = np.zeros(shape, dtype=np.float32)
        
        # Main road with full coverage
        roads[25, 20:30] = 1.0
        
        # Secondary road with partial coverage
        roads[20:30, 25] = 0.5
        
        # Residential streets with low coverage
        roads[15:20, 15:20] = 0.2
        
        urban_field = compute_urban_fields(
            road_fraction=roads,
            sigmas=[3.0],
            weights=[1.0]
        )
        
        # Higher road coverage should produce higher urban field
        main_road_field = urban_field[25, 25]
        secondary_road_field = urban_field[22, 25]  # On 0.5 coverage road
        residential_field = urban_field[17, 17]  # On 0.2 coverage area
        
        # Approximate expected relationships
        assert main_road_field > secondary_road_field * 1.5
        assert secondary_road_field > residential_field * 2
    
    def test_urban_field_normalization(self):
        """Test that urban fields are properly normalized."""
        shape = (30, 30)
        
        # Create roads with known total coverage
        roads = np.zeros(shape, dtype=np.float32)
        roads[10:20, 10:20] = 1.0  # 100 pixels at full coverage
        
        urban_field = compute_urban_fields(
            road_fraction=roads,
            sigmas=[1.0, 3.0],
            weights=[0.7, 0.3]
        )
        
        # Should be in reasonable range [0, 1]
        assert np.min(urban_field) >= 0.0
        assert np.max(urban_field) <= 1.0
        
        # With normalization, max should be close to 1
        assert np.max(urban_field) > 0.8, f"Max urban field too low: {np.max(urban_field)}"


class TestNTLEnhancement:
    """Test enhancement of NTL data using urban fields."""
    
    def test_multiplicative_enhancement(self):
        """Test multiplicative enhancement mode."""
        shape = (50, 50)
        
        # Create NTL data with some bright spots
        ntl = np.ones(shape, dtype=np.float32) * 2.0
        ntl[20:30, 20:30] = 10.0  # Bright area
        
        # Create urban field
        urban_field = np.zeros(shape, dtype=np.float32)
        urban_field[20:30, 20:30] = 0.8  # High urban density
        urban_field[15:35, 15:35] = np.maximum(urban_field[15:35, 15:35], 0.3)
        
        # Apply multiplicative enhancement
        enhanced = enhance_ntl_with_urban_field(
            ntl=ntl,
            urban_field=urban_field,
            enhancement_mode='multiplicative',
            multiplicative_factor=0.5
        )
        
        # Check enhancement
        assert enhanced.shape == shape
        
        # High urban areas should be enhanced more
        urban_enhancement = enhanced[25, 25] / ntl[25, 25]
        rural_enhancement = enhanced[5, 5] / ntl[5, 5]
        
        assert urban_enhancement > rural_enhancement
        assert urban_enhancement == pytest.approx(1 + 0.5 * 0.8, rel=0.01)
        assert rural_enhancement == pytest.approx(1.0, rel=0.01)
    
    def test_additive_enhancement(self):
        """Test additive enhancement mode."""
        shape = (40, 40)
        
        # Create NTL data
        ntl = np.ones(shape, dtype=np.float32) * 1.0
        
        # Create urban field
        urban_field = np.zeros(shape, dtype=np.float32)
        urban_field[15:25, 15:25] = 0.5
        
        # Apply additive enhancement
        enhanced = enhance_ntl_with_urban_field(
            ntl=ntl,
            urban_field=urban_field,
            enhancement_mode='additive',
            additive_factor=10.0
        )
        
        # Check additive enhancement
        assert enhanced[20, 20] == pytest.approx(1.0 + 10.0 * 0.5, rel=0.01)
        assert enhanced[5, 5] == pytest.approx(1.0, rel=0.01)
    
    def test_hybrid_enhancement(self):
        """Test hybrid enhancement mode (default in pipeline)."""
        shape = (60, 60)
        
        # Create varied NTL data
        ntl = np.zeros(shape, dtype=np.float32)
        # Dim areas
        ntl[10:20, 10:20] = 0.5
        # Moderate areas
        ntl[25:35, 25:35] = 5.0
        # Bright areas
        ntl[40:50, 40:50] = 20.0
        
        # Create urban field covering all areas
        urban_field = np.zeros(shape, dtype=np.float32)
        urban_field[5:55, 5:55] = 0.6  # Uniform urban density
        
        # Apply hybrid enhancement
        enhanced = enhance_ntl_with_urban_field(
            ntl=ntl,
            urban_field=urban_field,
            enhancement_mode='hybrid',
            multiplicative_factor=0.3,
            additive_factor=0.02,
            hybrid_expected_scale=0.2,
            hybrid_expected_sqrt_scale=0.1,
            hybrid_application_factor=1.0
        )
        
        # Check that all areas are enhanced
        assert np.all(enhanced >= ntl)
        
        # Dim areas should get more relative enhancement
        dim_ratio = enhanced[15, 15] / ntl[15, 15]
        bright_ratio = enhanced[45, 45] / ntl[45, 45]
        
        assert dim_ratio > bright_ratio, "Dim areas should get more relative enhancement"
    
    def test_enhancement_with_zero_ntl(self):
        """Test enhancement behavior with zero NTL values."""
        shape = (30, 30)
        
        # NTL with zeros
        ntl = np.zeros(shape, dtype=np.float32)
        ntl[15:20, 15:20] = 5.0  # Some lit areas
        
        # Urban field
        urban_field = np.ones(shape, dtype=np.float32) * 0.5
        
        # Test different modes with zero NTL
        for mode in ['multiplicative', 'additive', 'hybrid']:
            enhanced = enhance_ntl_with_urban_field(
                ntl=ntl,
                urban_field=urban_field,
                enhancement_mode=mode,
                multiplicative_factor=0.5,
                additive_factor=1.0
            )
            
            # Should handle zeros gracefully
            assert np.all(np.isfinite(enhanced))
            assert np.all(enhanced >= 0)
            
            if mode == 'additive':
                # Additive should add even to zero areas
                assert enhanced[0, 0] > 0
            elif mode == 'multiplicative':
                # Multiplicative should keep zeros as zeros
                assert enhanced[0, 0] == 0
    
    def test_enhancement_preserves_spatial_patterns(self):
        """Test that enhancement preserves spatial patterns in NTL."""
        shape = (80, 80)
        
        # Create NTL with specific pattern
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        # Circular bright spot
        center = shape[0] // 2
        radius = 20
        distance = np.sqrt((x - center)**2 + (y - center)**2)
        ntl = np.maximum(0, 10 * (1 - distance / radius))
        
        # Urban field that's stronger in center
        urban_field = np.maximum(0, 0.8 * (1 - distance / (radius * 2)))
        
        # Enhance
        enhanced = enhance_ntl_with_urban_field(
            ntl=ntl,
            urban_field=urban_field,
            enhancement_mode='hybrid'
        )
        
        # Check that circular pattern is preserved
        # Find brightest point
        max_pos = np.unravel_index(np.argmax(enhanced), enhanced.shape)
        assert max_pos == (center, center), "Brightest point should remain at center"
        
        # Check radial decay is preserved
        radial_line_orig = ntl[center, center:]
        radial_line_enh = enhanced[center, center:]
        
        # Both should decay
        orig_diffs = np.diff(radial_line_orig[:radius])
        enh_diffs = np.diff(radial_line_enh[:radius])
        
        assert np.all(orig_diffs <= 0), "Original should decay"
        assert np.all(enh_diffs <= 0), "Enhanced should also decay"


class TestUrbanFieldEdgeCases:
    """Test edge cases in urban field enhancement."""
    
    def test_empty_roads(self):
        """Test urban field with no roads."""
        shape = (20, 20)
        roads = np.zeros(shape, dtype=np.float32)
        
        urban_field = compute_urban_fields(
            road_fraction=roads,
            sigmas=[5.0],
            weights=[1.0]
        )
        
        # Should be all zeros
        assert np.all(urban_field == 0)
    
    def test_full_road_coverage(self):
        """Test urban field with complete road coverage."""
        shape = (20, 20)
        roads = np.ones(shape, dtype=np.float32)
        
        urban_field = compute_urban_fields(
            road_fraction=roads,
            sigmas=[2.0],
            weights=[1.0]
        )
        
        # Should be high everywhere, relatively uniform
        assert np.min(urban_field) > 0.8
        assert np.std(urban_field) < 0.1
    
    def test_invalid_weights(self):
        """Test handling of invalid weight configurations."""
        shape = (10, 10)
        roads = np.ones(shape, dtype=np.float32) * 0.5
        
        # Weights don't sum to 1 - should be normalized
        urban_field = compute_urban_fields(
            road_fraction=roads,
            sigmas=[1.0, 2.0],
            weights=[1.0, 1.0]  # Sum = 2, will be normalized
        )
        
        # Should still produce valid output
        assert np.all(urban_field >= 0)
        assert np.all(urban_field <= 1)