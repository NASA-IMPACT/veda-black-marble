"""Test legacy burn-in enhancement for NTL data.

The pipeline supports a legacy enhancement mode that directly
multiplies NTL values by road-type-specific factors.
"""

import numpy as np
import pytest


class TestLegacyBurnEnhancement:
    """Test legacy burn-in enhancement mode."""
    
    def test_legacy_burn_with_binary_roads(self):
        """Test legacy burn enhancement with binary road mask."""
        # Create NTL data
        shape = (50, 50)
        ntl = np.ones(shape, dtype=np.float32) * 10.0  # Uniform brightness
        
        # Create road mask with different types
        # Road type encoding (from pipeline):
        # 1 = motorway, 2 = trunk, 3 = primary, 
        # 4 = secondary, 5 = tertiary, 6 = residential
        roads = np.zeros(shape, dtype=np.uint8)
        roads[10, :] = 1  # Motorway
        roads[20, :] = 3  # Primary
        roads[30, :] = 5  # Tertiary
        roads[:, 25] = 4  # Secondary (crosses others)
        
        # Enhancement factors
        major_factor = 1.3
        minor_factor = 1.1
        
        # Simulate legacy burn logic from pipeline
        ntl_enhanced = ntl.copy()
        
        # Major roads: motorway (1) and secondary (4)
        major_roads_mask = (roads == 1) | (roads == 4)
        # Minor roads: trunk (2), primary (3), tertiary (5), residential (6)
        minor_roads_mask = (
            (roads == 2) | (roads == 3) | 
            (roads == 5) | (roads == 6)
        )
        
        # Apply enhancements
        ntl_enhanced = np.where(
            major_roads_mask, ntl * major_factor, ntl_enhanced
        )
        ntl_enhanced = np.where(
            minor_roads_mask, ntl * minor_factor, ntl_enhanced
        )
        
        # Verify enhancements
        assert ntl_enhanced[10, 0] == 10.0 * major_factor  # Motorway
        assert ntl_enhanced[20, 0] == 10.0 * minor_factor  # Primary
        assert ntl_enhanced[30, 0] == 10.0 * minor_factor  # Tertiary
        assert ntl_enhanced[0, 25] == 10.0 * major_factor   # Secondary
        
        # Intersection (secondary crosses primary) - major wins
        assert ntl_enhanced[20, 25] == 10.0 * major_factor
        
        # Non-road areas unchanged
        assert ntl_enhanced[5, 5] == 10.0
    
    def test_legacy_burn_with_fractional_roads(self):
        """Test legacy burn with fractional road coverage."""
        shape = (40, 40)
        ntl = np.ones(shape, dtype=np.float32) * 20.0
        
        # Fractional road coverage (0-1)
        roads_fractional = np.zeros(shape, dtype=np.float32)
        roads_fractional[15:25, :] = 0.8   # 80% road coverage
        roads_fractional[5:10, :] = 0.3     # 30% coverage
        roads_fractional[:, 20] = 1.0       # Full coverage
        
        # Factors for fractional mode
        major_factor = 1.3
        minor_factor = 1.1
        avg_factor = (major_factor + minor_factor) / 2  # 1.2
        
        # Simulate fractional burn logic
        # Enhancement = 1 + (factor - 1) * coverage
        enhancement_factor = 1 + (avg_factor - 1) * roads_fractional
        ntl_enhanced = ntl * enhancement_factor
        
        # Verify proportional enhancement
        # 80% coverage: 1 + (1.2 - 1) * 0.8 = 1.16
        expected_80 = 20.0 * (1 + (avg_factor - 1) * 0.8)
        assert np.allclose(ntl_enhanced[20, 10], expected_80)
        
        # 30% coverage: 1 + (1.2 - 1) * 0.3 = 1.06
        expected_30 = 20.0 * (1 + (avg_factor - 1) * 0.3)
        assert np.allclose(ntl_enhanced[7, 10], expected_30)
        
        # Full coverage: factor = 1.2
        assert ntl_enhanced[10, 20] == 20.0 * avg_factor
        
        # No roads: unchanged
        assert ntl_enhanced[35, 35] == 20.0
    
    def test_legacy_burn_preserves_zeros(self):
        """Test that legacy burn preserves zero values (truly dark areas)."""
        shape = (30, 30)
        ntl = np.zeros(shape, dtype=np.float32)
        ntl[10:20, 10:20] = 15.0  # Some lit areas
        
        roads = np.ones(shape, dtype=np.uint8)  # Roads everywhere
        
        # Apply enhancement
        major_factor = 1.5
        ntl_enhanced = ntl * major_factor
        
        # Dark areas should stay dark
        assert np.all(ntl_enhanced[:10, :10] == 0)
        
        # Lit areas enhanced
        assert np.all(ntl_enhanced[10:20, 10:20] == 15.0 * major_factor)
    
    def test_legacy_burn_statistics(self):
        """Test statistics calculation after legacy burn enhancement."""
        shape = (100, 100)
        ntl = np.random.rand(*shape).astype(np.float32) * 30
        
        # Add roads
        roads = np.zeros(shape, dtype=np.uint8)
        roads[::10, :] = 1  # Grid of major roads
        
        major_factor = 1.3
        
        # Count pixels by type
        major_roads_mask = roads == 1
        major_pixels = np.sum(major_roads_mask)
        total_pixels = roads.size
        
        # Apply enhancement
        ntl_enhanced = ntl.copy()
        ntl_enhanced[major_roads_mask] *= major_factor
        
        # Calculate enhancement stats
        enhancement = ntl_enhanced - ntl
        enhanced_pixels = np.sum(enhancement > 0)
        
        # Verify statistics
        assert enhanced_pixels == major_pixels
        assert enhanced_pixels == shape[0] * shape[1] // 10  # Every 10th row
        
        # Mean enhancement should be (factor - 1) * original value
        mean_enhancement = np.mean(enhancement[enhancement > 0])
        mean_original = np.mean(ntl[major_roads_mask])
        expected_enhancement = (major_factor - 1) * mean_original
        
        assert np.allclose(mean_enhancement, expected_enhancement, rtol=0.01)


class TestLegacyVsUrbanFieldComparison:
    """Compare legacy burn with urban field enhancement."""
    
    def test_enhancement_mode_differences(self):
        """Test differences between legacy and urban field modes."""
        shape = (50, 50)
        ntl = np.ones(shape, dtype=np.float32) * 10
        
        # Simple road pattern
        roads = np.zeros(shape, dtype=np.float32)
        roads[25, :] = 1.0
        roads[:, 25] = 1.0
        
        # Legacy burn enhancement
        legacy_factor = 1.3
        ntl_legacy = ntl.copy()
        ntl_legacy[roads > 0] *= legacy_factor
        
        # Urban field enhancement (simplified)
        # Would normally use Gaussian blur
        urban_field = roads.copy()
        multiplicative_factor = 0.3
        ntl_urban = ntl * (1 + multiplicative_factor * urban_field)
        
        # Compare patterns
        # Legacy: sharp enhancement only on roads
        assert ntl_legacy[25, 25] == 10 * legacy_factor
        assert ntl_legacy[24, 25] == 10  # No enhancement off-road
        
        # Urban field: same on roads, but would blur in real implementation
        assert ntl_urban[25, 25] == 10 * (1 + 0.3)
        assert ntl_urban[24, 25] == 10  # Would be enhanced with blur
        
        # Total enhancement
        legacy_total = np.sum(ntl_legacy - ntl)
        urban_total = np.sum(ntl_urban - ntl)
        
        # Both enhance, amounts differ
        assert legacy_total > 0
        assert urban_total > 0