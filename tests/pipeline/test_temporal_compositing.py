"""Test temporal compositing of Landsat data.

The pipeline creates median composites from multiple Landsat scenes
to reduce cloud cover and improve data quality.
"""

import numpy as np
import pytest

from blackmarble.analyze.temporal import create_landsat_median_composite


class TestTemporalCompositing:
    """Test Landsat temporal compositing matching pipeline implementation."""
    
    def test_basic_median_composite(self):
        """Test basic median compositing with multiple dates."""
        # Create test data for 3 dates
        shape = (50, 50)
        
        # Create data stacks as list of arrays (already masked with NaN)
        data_list = []
        
        # Date 1: Some cloud-masked areas (NaN)
        data1 = np.full(shape, 10000, dtype=np.float32)
        data1[10:20, 10:20] = np.nan  # Cloud area
        data_list.append(data1)
        
        # Date 2: Different cloud areas
        data2 = np.full(shape, 11000, dtype=np.float32)
        data2[30:40, 30:40] = np.nan  # Different cloud area
        data_list.append(data2)
        
        # Date 3: Clear everywhere
        data3 = np.full(shape, 12000, dtype=np.float32)
        data_list.append(data3)
        
        # Create composite with Landsat scaling
        scale_factor = 0.0000275
        add_offset = -0.2
        composite = create_landsat_median_composite(
            data_list,
            scale_factor=scale_factor,
            add_offset=add_offset
        )
        
        # Check output
        assert composite.shape == shape
        assert composite.dtype == np.float32
        
        # Check scaling was applied
        # Raw values [10000, 11000, 12000] -> scaled values
        expected_clear = np.median([
            10000 * scale_factor + add_offset,
            11000 * scale_factor + add_offset,
            12000 * scale_factor + add_offset
        ])
        
        # Clear areas should have median of all 3 dates
        assert np.abs(composite[5, 5] - expected_clear) < 0.001
        
        # Areas cloudy in date 1 should have median of dates 2 & 3
        expected_cloud1 = np.median([
            11000 * scale_factor + add_offset,
            12000 * scale_factor + add_offset
        ])
        assert np.abs(composite[15, 15] - expected_cloud1) < 0.001
        
        # Areas cloudy in date 2 should have median of dates 1 & 3
        expected_cloud2 = np.median([
            10000 * scale_factor + add_offset,
            12000 * scale_factor + add_offset
        ])
        assert np.abs(composite[35, 35] - expected_cloud2) < 0.001
    
    def test_composite_handles_all_nan(self):
        """Test compositing when all dates are NaN for some pixels."""
        shape = (20, 20)
        
        data_list = []
        
        # All dates have same cloud area
        for i in range(3):
            data = np.full(shape, 10000 + i * 1000, dtype=np.float32)
            data[5:10, 5:10] = np.nan  # Persistent clouds
            data_list.append(data)
        
        composite = create_landsat_median_composite(data_list)
        
        # Persistently cloudy areas should remain NaN
        assert np.all(np.isnan(composite[5:10, 5:10]))
        
        # Clear areas should have valid data
        assert not np.any(np.isnan(composite[15:20, 15:20]))
    
    def test_composite_with_single_date(self):
        """Test composite with only one date."""
        shape = (10, 10)
        
        data = np.full(shape, 10000, dtype=np.float32)
        data[0:3, 0:3] = np.nan
        
        composite = create_landsat_median_composite([data])
        
        # Should work with single date (no median needed)
        assert composite.shape == shape
        
        # Check scaling
        expected = 10000 * 0.0000275 + (-0.2)
        assert np.abs(composite[5, 5] - expected) < 0.001
        
        # NaN areas preserved
        assert np.all(np.isnan(composite[0:3, 0:3]))
    
    def test_composite_with_different_shapes(self):
        """Test compositing handles slightly different array shapes."""
        # This can happen with different scene boundaries
        
        data_list = [
            np.full((100, 100), 10000, dtype=np.float32),
            np.full((100, 95), 11000, dtype=np.float32),  # Narrower
            np.full((95, 100), 12000, dtype=np.float32),  # Shorter
        ]
        
        composite = create_landsat_median_composite(data_list)
        
        # Should crop to common size (95x95)
        assert composite.shape == (95, 95)
        
        # Values should be median of available data
        scale_factor = 0.0000275
        add_offset = -0.2
        expected = np.median([
            10000 * scale_factor + add_offset,
            11000 * scale_factor + add_offset,
            12000 * scale_factor + add_offset
        ])
        assert np.abs(composite[50, 50] - expected) < 0.001
    
    def test_composite_preserves_landsat_offset_values(self):
        """Test that compositing preserves values near Landsat offset."""
        shape = (10, 10)
        
        # Create data with DN values that produce reflectance near -0.2
        # DN = 0 -> reflectance = -0.2 (at offset)
        # DN = 36 -> reflectance ≈ -0.199
        
        data_list = [
            np.full(shape, 0, dtype=np.float32),     # All at offset
            np.full(shape, 36, dtype=np.float32),    # Just above offset
            np.full(shape, 100, dtype=np.float32),   # Further from offset
        ]
        
        composite = create_landsat_median_composite(data_list)
        
        # Check that median preserves these edge values
        # Median of [0, 36, 100] = 36
        expected_dn = 36
        expected_reflectance = expected_dn * 0.0000275 + (-0.2)
        
        assert np.allclose(composite, expected_reflectance, rtol=0.001)
    
    def test_empty_data_list(self):
        """Test handling of empty data list."""
        with pytest.raises(ValueError, match="No data provided"):
            create_landsat_median_composite([])
    
    def test_composite_with_extreme_values(self):
        """Test compositing with extreme DN values."""
        shape = (5, 5)
        
        data_list = [
            np.full(shape, 0, dtype=np.float32),      # Minimum DN
            np.full(shape, 30000, dtype=np.float32),  # Medium DN
            np.full(shape, 65535, dtype=np.float32),  # Maximum uint16 DN
        ]
        
        composite = create_landsat_median_composite(data_list)
        
        # Should handle full DN range
        assert np.all(np.isfinite(composite))
        
        # Median DN = 30000
        expected = 30000 * 0.0000275 + (-0.2)
        assert np.allclose(composite, expected, rtol=0.001)


class TestTemporalCompositingIntegration:
    """Test temporal compositing in pipeline context."""
    
    def test_composite_for_index_calculation(self):
        """Test that composite output is suitable for NDVI calculation."""
        shape = (30, 30)
        
        # Create NIR (B5) data stack
        nir_list = []
        for i in range(3):
            data = np.full(shape, 12000 + i * 500, dtype=np.float32)
            # Add some clouds
            if i == 0:
                data[5:10, 5:10] = np.nan
            nir_list.append(data)
        
        # Create Red (B4) data stack
        red_list = []
        for i in range(3):
            data = np.full(shape, 8000 + i * 300, dtype=np.float32)
            # Same cloud pattern
            if i == 0:
                data[5:10, 5:10] = np.nan
            red_list.append(data)
        
        # Create composites
        nir_composite = create_landsat_median_composite(nir_list)
        red_composite = create_landsat_median_composite(red_list)
        
        # Check that we can calculate NDVI
        # This simulates what happens in the pipeline
        epsilon = 0.001
        denominator = nir_composite + red_composite
        safe_denominator = np.where(
            np.abs(denominator) < epsilon,
            np.copysign(epsilon, denominator),
            denominator
        )
        ndvi = (nir_composite - red_composite) / safe_denominator
        
        # NDVI should be valid and bounded
        assert np.all(np.abs(ndvi) <= 1.0)
        
        # Check specific values
        # Clear area NDVI (using median scaled values)
        nir_median = np.median([12000, 12500, 13000]) * 0.0000275 + (-0.2)
        red_median = np.median([8000, 8300, 8600]) * 0.0000275 + (-0.2)
        expected_ndvi = (nir_median - red_median) / (nir_median + red_median)
        
        assert np.abs(ndvi[20, 20] - expected_ndvi) < 0.01
    
    def test_composite_maintains_band_relationships(self):
        """Test that compositing maintains relationships between bands."""
        shape = (20, 20)
        
        # Create correlated band data (as would occur in real Landsat)
        dates = 3
        bands = {}
        
        for band_name, base_dn in [('B3', 9000), ('B4', 10000), ('B5', 15000)]:
            band_list = []
            for i in range(dates):
                # Add temporal variation
                data = np.full(shape, base_dn + i * 500, dtype=np.float32)
                # Add spatial gradient
                data += np.arange(shape[0])[:, np.newaxis] * 10
                band_list.append(data)
            
            bands[band_name] = create_landsat_median_composite(band_list)
        
        # Check that band relationships are preserved
        # B5 (NIR) should be brighter than B4 (Red) for vegetation
        assert np.all(bands['B5'] > bands['B4'])
        
        # Spatial gradients should be preserved
        assert np.all(np.diff(bands['B4'][:, 0]) > 0)  # Increasing down column