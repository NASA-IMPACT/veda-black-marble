"""Test statistics calculation at the end of the pipeline.

The pipeline calculates urban area, mean radiance, and other
statistics for the final output.
"""

import numpy as np
import pytest
from rasterio.transform import from_origin


class TestStatisticsCalculation:
    """Test final statistics calculation."""
    
    def test_urban_area_calculation(self):
        """Test urban area calculation in km²."""
        # Create urban mask
        shape = (100, 100)
        urban_mask = np.zeros(shape, dtype=bool)
        # 30x30 urban area in center
        urban_mask[35:65, 35:65] = True
        
        # Create transform with known pixel size
        pixel_size_m = 30  # 30m pixels (Landsat resolution)
        transform = from_origin(0, 3000, pixel_size_m, pixel_size_m)
        
        # Calculate urban area (from pipeline logic)
        pixel_area_m2 = abs(transform.a * transform.e)  # 30 * 30 = 900 m²
        urban_pixels = np.sum(urban_mask)  # 30 * 30 = 900 pixels
        urban_area_km2 = float((urban_pixels * pixel_area_m2) / 1e6)
        
        # Verify calculation
        assert pixel_area_m2 == 900
        assert urban_pixels == 900
        expected_area_km2 = (900 * 900) / 1e6  # 0.81 km²
        assert urban_area_km2 == expected_area_km2
        assert abs(urban_area_km2 - 0.81) < 0.001
    
    def test_mean_radiance_calculation(self):
        """Test mean radiance calculation for NTL."""
        # Create NTL data with known distribution
        shape = (50, 50)
        ntl = np.zeros(shape, dtype=np.float32)
        
        # Dark rural areas
        ntl[:25, :] = 0.5
        
        # Bright urban areas
        ntl[25:, :] = 20.0
        
        # Add some very bright spots
        ntl[30:35, 30:35] = 50.0
        
        # Calculate mean (from pipeline)
        mean_radiance = float(np.nanmean(ntl))
        
        # Verify
        # Total: 25*50*0.5 + 25*50*20 - 5*5*20 + 5*5*50
        # = 625 + 25000 - 500 + 1250 = 26375
        # Mean: 26375 / 2500 = 10.55
        assert abs(mean_radiance - 10.55) < 0.01
    
    def test_statistics_with_nan_values(self):
        """Test statistics handle NaN values correctly."""
        shape = (40, 40)
        
        # Urban mask with some invalid areas
        urban_mask = np.ones(shape, dtype=bool)
        urban_mask[10:20, 10:20] = False  # Non-urban
        
        # NTL with NaN values
        ntl = np.ones(shape, dtype=np.float32) * 15.0
        ntl[5:15, 5:15] = np.nan  # Invalid data
        
        # Calculate statistics
        transform = from_origin(0, 0, 30, 30)
        pixel_area_m2 = abs(transform.a * transform.e)
        
        # Urban area (NaN doesn't affect boolean mask)
        urban_pixels = np.sum(urban_mask)
        urban_area_km2 = float((urban_pixels * pixel_area_m2) / 1e6)
        
        # Mean radiance (should ignore NaN)
        mean_radiance = float(np.nanmean(ntl))
        
        # Verify
        assert urban_pixels == shape[0] * shape[1] - 100  # Minus non-urban square
        assert mean_radiance == 15.0  # All non-NaN values are 15
    
    def test_statistics_edge_cases(self):
        """Test statistics with edge case inputs."""
        # All zero urban mask
        urban_mask = np.zeros((10, 10), dtype=bool)
        transform = from_origin(0, 0, 30, 30)
        pixel_area_m2 = abs(transform.a * transform.e)
        urban_pixels = np.sum(urban_mask)
        urban_area_km2 = float((urban_pixels * pixel_area_m2) / 1e6)
        
        assert urban_area_km2 == 0.0
        
        # All NaN NTL
        ntl_all_nan = np.full((10, 10), np.nan, dtype=np.float32)
        mean_radiance = float(np.nanmean(ntl_all_nan))
        
        assert np.isnan(mean_radiance)
        
        # Single pixel urban
        urban_single = np.zeros((100, 100), dtype=bool)
        urban_single[50, 50] = True
        urban_pixels = np.sum(urban_single)
        urban_area_km2 = float((urban_pixels * pixel_area_m2) / 1e6)
        
        assert urban_area_km2 == 900 / 1e6  # 0.0009 km²
    
    def test_statistics_dict_structure(self):
        """Test the structure of returned statistics dictionary."""
        # Simulate pipeline statistics calculation
        urban_mask = np.ones((50, 50), dtype=bool)
        ntl = np.ones((50, 50), dtype=np.float32) * 10
        transform = from_origin(0, 0, 30, 30)
        
        # Calculate statistics (matching pipeline structure)
        statistics = {
            "urban_area_km2": None,
            "mean_radiance": None,
            "road_length_km": None,
        }
        
        # Urban area
        pixel_area_m2 = abs(transform.a * transform.e)
        urban_pixels = np.sum(urban_mask)
        statistics["urban_area_km2"] = float((urban_pixels * pixel_area_m2) / 1e6)
        
        # Mean radiance
        statistics["mean_radiance"] = float(np.nanmean(ntl))
        
        # Verify structure
        assert isinstance(statistics, dict)
        assert "urban_area_km2" in statistics
        assert "mean_radiance" in statistics
        assert "road_length_km" in statistics
        
        # Verify types
        assert isinstance(statistics["urban_area_km2"], float)
        assert isinstance(statistics["mean_radiance"], float)
        assert statistics["road_length_km"] is None  # Not calculated in basic test


class TestStatisticsAccuracy:
    """Test accuracy of statistics calculations."""
    
    def test_precise_area_calculation(self):
        """Test area calculation with precise known values."""
        # Create checkerboard urban pattern
        shape = (10, 10)
        urban_mask = np.zeros(shape, dtype=bool)
        urban_mask[::2, ::2] = True  # Every other pixel
        
        # Exactly 100m pixels
        transform = from_origin(0, 0, 100, 100)
        
        # Calculate
        pixel_area_m2 = 100 * 100  # 10,000 m²
        urban_pixels = np.sum(urban_mask)  # 5*5 = 25 pixels
        urban_area_km2 = (urban_pixels * pixel_area_m2) / 1e6
        
        # Verify
        assert urban_pixels == 25
        assert pixel_area_m2 == 10000
        assert urban_area_km2 == 0.25  # Exactly 0.25 km²
    
    def test_radiance_with_outliers(self):
        """Test mean radiance robust to outliers."""
        shape = (100, 100)
        ntl = np.random.normal(10, 2, shape).astype(np.float32)
        
        # Add outliers
        ntl[0, 0] = 1000  # Extremely bright
        ntl[1, 1] = -50   # Invalid negative
        
        # In practice, should clip or mask outliers
        ntl_clipped = np.clip(ntl, 0, 100)
        
        mean_radiance = float(np.nanmean(ntl_clipped))
        
        # Should be close to 10 (the normal distribution mean)
        assert 9 < mean_radiance < 11