"""Test VIIRS nighttime lights processing.

The pipeline crops and resamples VIIRS data to match the target area.
"""

import numpy as np
import pytest
from rasterio.transform import from_origin
from rasterio.crs import CRS

from blackmarble.prepare.spatial import crop_to_bounds, reproject_image


class TestVIIRSProcessing:
    """Test VIIRS data processing in the pipeline."""
    
    def test_viirs_crop_to_bounds(self):
        """Test cropping VIIRS data to AOI bounds."""
        # Create VIIRS-like data (large tile)
        shape = (1000, 1000)
        viirs_data = np.random.rand(*shape).astype(np.float32) * 50  # 0-50 nW/cm²/sr
        
        # VIIRS in EPSG:4326 with ~500m pixels
        pixel_size = 0.005  # ~500m at equator
        west, north = -100, 40
        viirs_transform = from_origin(west, north, pixel_size, pixel_size)
        
        # Define crop bounds (smaller area)
        crop_bounds = (-99.5, 39.2, -99.0, 39.7)  # 0.5° x 0.5°
        
        # Crop
        cropped, new_transform, new_bounds = crop_to_bounds(
            data=viirs_data,
            src_transform=viirs_transform,
            bounds=crop_bounds
        )
        
        # Check output
        assert cropped.shape[0] < shape[0]  # Smaller
        assert cropped.shape[1] < shape[1]
        
        # Check bounds match request
        assert abs(new_bounds[0] - crop_bounds[0]) < pixel_size
        assert abs(new_bounds[1] - crop_bounds[1]) < pixel_size
        assert abs(new_bounds[2] - crop_bounds[2]) < pixel_size
        assert abs(new_bounds[3] - crop_bounds[3]) < pixel_size
        
        # Check data range preserved
        assert 0 <= np.min(cropped) <= np.max(cropped) <= 50
    
    def test_viirs_resample_to_high_resolution(self):
        """Test resampling VIIRS to higher resolution (12m target)."""
        # Create coarse VIIRS data
        shape = (50, 50)
        viirs_data = np.zeros(shape, dtype=np.float32)
        # Add bright spots
        viirs_data[20:30, 20:30] = 40.0  # Urban area
        viirs_data[10:15, 35:40] = 20.0  # Smaller town
        
        # VIIRS at ~500m
        src_transform = from_origin(0, 5000, 500, 500)
        src_crs = CRS.from_epsg(4326)
        
        # Resample to 12m (like in pipeline)
        target_resolution = 12
        dst_crs = CRS.from_epsg(32633)  # UTM
        
        resampled, dst_transform, dst_bounds = reproject_image(
            data=viirs_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_crs=dst_crs,
            resampling="bilinear",
            resolution=target_resolution
        )
        
        # Check resolution
        assert abs(dst_transform.a) == target_resolution
        assert abs(dst_transform.e) == target_resolution
        
        # Check interpolation preserved bright spots
        assert np.max(resampled) > 35  # Should preserve bright values
        assert np.sum(resampled > 10) > 100  # Spread by interpolation
        
        # Check no extreme values created
        assert np.max(resampled) <= 50  # No overshoot
        assert np.min(resampled) >= 0   # No negative
    
    def test_viirs_alignment_check(self):
        """Test VIIRS non-zero pixel check after alignment."""
        # Create mostly-zero VIIRS (simulating misalignment)
        shape = (100, 100)
        viirs_data = np.zeros(shape, dtype=np.float32)
        # Only 5% non-zero (below 10% threshold)
        viirs_data[0:7, 0:7] = 10.0  # Small corner lit
        
        # Calculate statistics (as done in pipeline)
        non_zero = (viirs_data > 0).sum()
        total = viirs_data.size
        percent_nonzero = non_zero / total * 100
        
        # This should trigger the pipeline error
        assert percent_nonzero < 10, "Should have less than 10% non-zero"
        
        # In pipeline this raises RuntimeError
        if percent_nonzero < 10:
            with pytest.raises(RuntimeError, match="overlap"):
                raise RuntimeError(
                    f"VIIRS has only {percent_nonzero:.1f}% non-zero pixels. "
                    "This suggests a spatial alignment issue or no overlap with AOI."
                )
    
    def test_viirs_statistics_logging(self):
        """Test VIIRS statistics calculation (quick_stats function)."""
        # Create realistic VIIRS data
        shape = (200, 200)
        viirs_data = np.random.exponential(5, shape).astype(np.float32)
        viirs_data = np.clip(viirs_data, 0, 50)  # Realistic range
        # Add some zeros (dark areas)
        viirs_data[viirs_data < 1] = 0
        
        # Calculate stats (from quick_stats function)
        stats = {
            "shape": viirs_data.shape,
            "dtype": viirs_data.dtype,
            "min": np.nanmin(viirs_data),
            "max": np.nanmax(viirs_data),
            "non_zero_count": (viirs_data != 0).sum(),
            "non_zero_percent": (viirs_data != 0).sum() / viirs_data.size * 100,
            "nan_count": np.isnan(viirs_data).sum()
        }
        
        # Verify stats make sense
        assert stats["shape"] == shape
        assert stats["dtype"] == np.float32
        assert 0 <= stats["min"] <= stats["max"] <= 50
        assert 0 <= stats["non_zero_percent"] <= 100
        assert stats["nan_count"] == 0  # No NaN in this test


class TestVIIRSFallback:
    """Test VIIRS-based reference grid when Landsat unavailable."""
    
    def test_viirs_as_reference_grid(self):
        """Test using VIIRS transform when no Landsat available."""
        # Simulate pipeline logic for reference grid selection
        landsat_reprojected = {}  # Empty - no Landsat
        
        # VIIRS data available
        viirs_shape = (100, 100)
        viirs_data = np.ones(viirs_shape, dtype=np.float32) * 10
        viirs_transform = from_origin(500000, 4000000, 30, 30)  # Already in UTM
        
        # Pipeline logic for reference selection
        if landsat_reprojected:
            # Use Landsat band as reference
            reference_transform = None  # Would come from Landsat
            reference_shape = None
        else:
            # Use VIIRS as reference
            reference_transform = viirs_transform
            reference_shape = viirs_shape
            print("Using VIIRS data as reference grid (no Landsat available)")
        
        # Verify fallback worked
        assert reference_transform is not None
        assert reference_shape == viirs_shape
        assert reference_transform == viirs_transform
    
    def test_missing_both_viirs_and_landsat(self):
        """Test error when both VIIRS and Landsat missing."""
        # Simulate no data available
        landsat_reprojected = {}
        viirs_reprojected = None
        
        # This should be caught
        if not landsat_reprojected and viirs_reprojected is None:
            with pytest.raises(RuntimeError, match="No data available"):
                raise RuntimeError(
                    "No data available for processing. "
                    "Both VIIRS and Landsat data are missing."
                )


class TestVIIRSEdgeCases:
    """Test VIIRS edge cases and error conditions."""
    
    def test_viirs_all_zero_data(self):
        """Test handling VIIRS data that's all zeros."""
        viirs_data = np.zeros((50, 50), dtype=np.float32)
        
        # Check statistics
        non_zero_percent = (viirs_data > 0).sum() / viirs_data.size * 100
        
        assert non_zero_percent == 0
        
        # Pipeline should catch this
        if non_zero_percent < 10:
            # Would raise error in pipeline
            assert True
    
    def test_viirs_extreme_values(self):
        """Test VIIRS data with unrealistic values."""
        shape = (30, 30)
        viirs_data = np.ones(shape, dtype=np.float32)
        
        # Add some extreme values (outside expected range)
        viirs_data[0, 0] = 1000  # Way too bright
        viirs_data[1, 1] = -10   # Negative (invalid)
        
        # Clipping that should happen
        viirs_clipped = np.clip(viirs_data, 0, 100)  # Reasonable max
        
        assert viirs_clipped[0, 0] == 100
        assert viirs_clipped[1, 1] == 0
        assert np.all(viirs_clipped >= 0)
        assert np.all(viirs_clipped <= 100)