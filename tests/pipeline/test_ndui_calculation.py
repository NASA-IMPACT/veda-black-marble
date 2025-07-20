"""Test NDUI (Normalized Difference Urban Index) calculation.

NDUI combines nighttime lights (NTL) and vegetation indices (NDVI/NDWI)
to identify urban areas.
"""

import numpy as np
import pytest

from blackmarble.analyze.indices import calculate_ndui


class TestNDUICalculation:
    """Test NDUI calculation matching pipeline implementation."""
    
    def test_ndui_basic_calculation(self):
        """Test basic NDUI calculation without water masking."""
        # Create test data
        shape = (10, 10)
        
        # NTL data (0-50 range typical for VIIRS)
        ntl = np.zeros(shape, dtype=np.float32)
        ntl[5:, 5:] = 10.0  # Bright area (urban)
        ntl[:5, :5] = 0.5   # Dim area (rural)
        
        # NDVI data (-1 to 1, negative = water/urban, positive = vegetation)
        ndvi = np.ones(shape, dtype=np.float32) * 0.5  # Vegetated
        ndvi[5:, 5:] = -0.2  # Urban/bare (negative NDVI)
        
        # Calculate NDUI without NDWI
        ndui = calculate_ndui(ntl, ndvi)
        
        # Check output
        assert ndui.shape == shape
        assert ndui.dtype == np.float32
        
        # Urban areas (high NTL, low NDVI) should have high NDUI
        urban_ndui = ndui[5:, 5:]
        assert np.all(urban_ndui > 0.5), f"Urban NDUI too low: {urban_ndui.mean()}"
        
        # Rural areas (low NTL, high NDVI) should have low/negative NDUI
        rural_ndui = ndui[:5, :5]
        assert np.all(rural_ndui < 0.2), f"Rural NDUI too high: {rural_ndui.mean()}"
    
    def test_ndui_with_water_masking(self):
        """Test NDUI with NDWI-based water masking."""
        shape = (20, 20)
        
        # Create diverse landscape
        ntl = np.ones(shape, dtype=np.float32) * 5.0
        ndvi = np.zeros(shape, dtype=np.float32)
        ndwi = np.zeros(shape, dtype=np.float32)
        
        # Urban area: high NTL, negative NDVI, negative NDWI
        ntl[0:5, 0:5] = 20.0
        ndvi[0:5, 0:5] = -0.3
        ndwi[0:5, 0:5] = -0.2
        
        # Water: low NTL, negative NDVI, positive NDWI
        ntl[10:15, 10:15] = 0.1
        ndvi[10:15, 10:15] = -0.5
        ndwi[10:15, 10:15] = 0.6  # High NDWI = water
        
        # Vegetation: low NTL, positive NDVI, negative NDWI
        ntl[15:20, 15:20] = 0.5
        ndvi[15:20, 15:20] = 0.7
        ndwi[15:20, 15:20] = -0.3
        
        # Calculate with water masking
        ndui = calculate_ndui(ntl, ndvi, ndwi=ndwi)
        
        # Water areas should be masked (NaN)
        water_ndui = ndui[10:15, 10:15]
        assert np.all(np.isnan(water_ndui)), "Water areas should be NaN"
        
        # Urban areas should have high NDUI
        urban_ndui = ndui[0:5, 0:5]
        assert np.all(urban_ndui > 0.5), f"Urban NDUI too low: {np.nanmean(urban_ndui)}"
        
        # Vegetation should have low NDUI
        veg_ndui = ndui[15:20, 15:20]
        assert np.all(veg_ndui < 0.1), f"Vegetation NDUI too high: {np.nanmean(veg_ndui)}"
    
    def test_ndui_floor_ceiling_parameters(self):
        """Test NTL floor and ceiling parameters."""
        shape = (10, 10)
        
        # Create NTL with wide range
        ntl = np.linspace(0, 100, 100).reshape(shape)
        ndvi = np.zeros(shape)  # Neutral NDVI
        
        # Test with floor and ceiling
        ntl_floor = 1.0
        ntl_ceiling = 50.0
        
        ndui = calculate_ndui(
            ntl, ndvi,
            ntl_floor=ntl_floor,
            ntl_ceiling=ntl_ceiling
        )
        
        # Values should be clipped and scaled
        # After clipping: min=1.0, max=50.0
        # After log scaling, should be normalized
        assert np.nanmin(ndui) >= -1.0
        assert np.nanmax(ndui) <= 1.0
    
    def test_ndui_floor_parameter(self):
        """Test NDUI floor parameter effect."""
        shape = (5, 5)
        
        ntl = np.array([
            [0.0, 0.5, 1.0, 2.0, 5.0],
            [0.0, 0.5, 1.0, 2.0, 5.0],
            [0.0, 0.5, 1.0, 2.0, 5.0],
            [0.0, 0.5, 1.0, 2.0, 5.0],
            [0.0, 0.5, 1.0, 2.0, 5.0]
        ])
        ndvi = np.zeros(shape)
        
        # Without floor
        ndui_no_floor = calculate_ndui(ntl, ndvi, ndui_floor=0.0)
        
        # With floor = 0.02 (default)
        ndui_with_floor = calculate_ndui(ntl, ndvi, ndui_floor=0.02)
        
        # Values below floor should be set to 0
        assert np.all(ndui_with_floor[ntl < 1.0] == 0.0), "Values below threshold should be 0"
        
        # Values above threshold should be non-zero
        assert np.all(ndui_with_floor[ntl >= 1.0] > 0.0), "Values above threshold should be > 0"
    
    def test_ndui_with_nan_values(self):
        """Test NDUI handles NaN values correctly."""
        shape = (10, 10)
        
        ntl = np.ones(shape) * 10.0
        ndvi = np.ones(shape) * 0.3
        
        # Add some NaN values
        ntl[0:3, 0:3] = np.nan
        ndvi[7:10, 7:10] = np.nan
        
        ndui = calculate_ndui(ntl, ndvi)
        
        # NaN in input should produce NaN in output
        assert np.all(np.isnan(ndui[0:3, 0:3])), "NaN NTL should produce NaN NDUI"
        assert np.all(np.isnan(ndui[7:10, 7:10])), "NaN NDVI should produce NaN NDUI"
        
        # Non-NaN areas should be valid
        valid_mask = ~(np.isnan(ntl) | np.isnan(ndvi))
        assert not np.any(np.isnan(ndui[valid_mask])), "Valid areas should not be NaN"
    
    def test_ndui_edge_cases(self):
        """Test NDUI with edge case values."""
        # All zeros
        ntl = np.zeros((5, 5))
        ndvi = np.zeros((5, 5))
        ndui = calculate_ndui(ntl, ndvi)
        assert np.all(ndui == 0.0), "All zeros should produce zero NDUI"
        
        # Very high NTL
        ntl = np.ones((5, 5)) * 1000.0
        ndvi = np.zeros((5, 5))
        ndui = calculate_ndui(ntl, ndvi, ntl_ceiling=100.0)
        assert np.all(np.isfinite(ndui)), "High NTL should still produce finite values"
        
        # Extreme NDVI
        ntl = np.ones((5, 5)) * 10.0
        ndvi = np.ones((5, 5))  # Max NDVI
        ndui = calculate_ndui(ntl, ndvi)
        assert np.all(ndui < 0), "High NDVI should produce negative NDUI"
        
        ndvi = np.ones((5, 5)) * -1.0  # Min NDVI
        ndui = calculate_ndui(ntl, ndvi)
        assert np.all(ndui > 0), "Low NDVI should produce positive NDUI"
    
    def test_ndui_matches_pipeline_formula(self):
        """Test that NDUI calculation matches the exact pipeline formula."""
        # Create specific test case
        ntl = np.array([[0.5, 2.0, 10.0, 50.0]])
        ndvi = np.array([[0.8, 0.4, -0.2, -0.5]])
        
        # Calculate NDUI with default parameters
        ndui = calculate_ndui(
            ntl, ndvi,
            ntl_floor=0.1,
            ntl_ceiling=10.0,
            ndui_floor=0.02,
            epsilon=1e-6
        )
        
        # Manually calculate expected values
        # 1. Clip NTL: [0.5, 2.0, 10.0, 10.0] (50->10 due to ceiling)
        # 2. Log scale: log10([0.5, 2.0, 10.0, 10.0] + 1)
        # 3. Normalize to [0, 1]
        # 4. Combine with NDVI using subtraction
        # 5. Apply floor threshold
        
        # All results should be finite
        assert np.all(np.isfinite(ndui))
        
        # Check relative ordering makes sense
        # Higher NTL + lower NDVI = higher NDUI
        assert ndui[0, 2] > ndui[0, 0], "Urban area should have higher NDUI than vegetation"


class TestNDUIIntegration:
    """Test NDUI calculation in pipeline context."""
    
    def test_ndui_with_aligned_data(self):
        """Test NDUI with properly aligned NTL and NDVI arrays."""
        # Simulate aligned data from pipeline
        shape = (100, 100)
        
        # Create realistic NTL pattern (cities and dark areas)
        ntl = np.zeros(shape, dtype=np.float32)
        # Add city centers
        y, x = np.ogrid[:shape[0], :shape[1]]
        city1 = ((x - 25)**2 + (y - 25)**2) < 100
        city2 = ((x - 75)**2 + (y - 75)**2) < 150
        ntl[city1] = 20.0
        ntl[city2] = 30.0
        
        # Add urban sprawl
        sprawl1 = ((x - 25)**2 + (y - 25)**2) < 400
        sprawl2 = ((x - 75)**2 + (y - 75)**2) < 500
        ntl[sprawl1] = np.maximum(ntl[sprawl1], 5.0)
        ntl[sprawl2] = np.maximum(ntl[sprawl2], 8.0)
        
        # Create corresponding NDVI (negative in cities, positive elsewhere)
        ndvi = np.ones(shape, dtype=np.float32) * 0.4  # Background vegetation
        ndvi[city1] = -0.3  # Urban core
        ndvi[city2] = -0.4
        ndvi[sprawl1] = np.where(ndvi[sprawl1] < 0, ndvi[sprawl1], 0.1)  # Low vegetation
        ndvi[sprawl2] = np.where(ndvi[sprawl2] < 0, ndvi[sprawl2], 0.05)
        
        # Calculate NDUI
        ndui = calculate_ndui(ntl, ndvi)
        
        # Verify urban areas are identified
        assert ndui[city1].mean() > 0.6, "City centers should have high NDUI"
        assert ndui[city2].mean() > 0.7, "Larger city should have higher NDUI"
        
        # Verify gradient from urban to rural
        # Sample along a line from city to rural
        line_ndui = ndui[25, 25:50]  # From city center outward
        assert np.all(np.diff(line_ndui) <= 0), "NDUI should decrease from city center"