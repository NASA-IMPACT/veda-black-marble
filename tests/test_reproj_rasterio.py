"""Tests for rasterio-based reprojection implementations."""

import os
import tempfile
from typing import Tuple

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds, Affine
from rasterio.warp import calculate_default_transform, reproject, Resampling

from blackmarble.reproj_rasterio import (
    subset_raster,
    subset_raster_normal,
    convert_to_wgs84,
    crop_to_bb_utm,
)


class TestRasterioReprojection:
    """Test suite for rasterio reprojection functions."""

    @pytest.fixture
    def utm_test_data(self) -> Tuple[np.ndarray, str, Tuple[float, ...]]:
        """Create UTM test data."""
        array = np.random.rand(500, 500).astype(np.float32) * 1000
        # Add some features
        array[100:200, 100:200] = 5000  # High value region
        array[300:350, 300:350] = 0  # Zero region
        array[50:100, 50:100] = np.nan  # NaN region

        # UTM Zone 30N (covers UK/Western Europe)
        projection = "EPSG:32630"
        # Centered around 500000, 4000000 (roughly central UK)
        geotransform = (450000.0, 100.0, 0.0, 4050000.0, 0.0, -100.0)

        return array, projection, geotransform

    @pytest.fixture
    def wgs84_test_data(self) -> Tuple[np.ndarray, str, Tuple[float, ...]]:
        """Create WGS84 test data."""
        array = np.random.rand(360, 360).astype(np.float32) * 100
        projection = "EPSG:4326"
        # 1 degree resolution
        geotransform = (-180.0, 1.0, 0.0, 90.0, 0.0, -1.0)

        return array, projection, geotransform

    def create_test_file(self, array, projection, geotransform, filepath):
        """Helper to create test raster files."""
        transform = Affine.from_gdal(*geotransform)

        with rasterio.open(
            filepath,
            "w",
            driver="GTiff",
            height=array.shape[0],
            width=array.shape[1],
            count=1,
            dtype=array.dtype,
            crs=projection,
            transform=transform,
            nodata=-9999.0 if array.dtype in [np.float32, np.float64] else None,
        ) as dst:
            # Convert NaN to nodata
            if array.dtype in [np.float32, np.float64]:
                write_array = np.where(np.isnan(array), -9999.0, array)
            else:
                write_array = array
            dst.write(write_array, 1)

    def test_subset_raster_basic(self, utm_test_data):
        """Test basic subset_raster functionality."""
        array, projection, geotransform = utm_test_data

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "input.tif")
            output_file = os.path.join(tmpdir, "output.tif")

            # Create input file
            self.create_test_file(array, projection, geotransform, input_file)

            # Define subset bounds (smaller area in center)
            minX = 470000
            maxX = 480000
            minY = 4020000
            maxY = 4030000
            scale = 200  # 200m resolution

            # Run subset
            subset_raster(
                input_file, output_file, minX, maxY, maxX, minY, scale, tmpdir
            )

            # Verify output
            with rasterio.open(output_file) as src:
                # Should be reprojected to WGS84
                assert src.crs.to_epsg() == 4326

                # Check bounds are reasonable
                bounds = src.bounds
                assert bounds.left < bounds.right
                assert bounds.bottom < bounds.top

                # Check resolution
                pixel_width = (bounds.right - bounds.left) / src.width
                pixel_height = (bounds.top - bounds.bottom) / src.height

                # Resolution should be approximately the requested scale
                # (converted from meters to degrees)
                expected_deg_scale = scale / 111000  # Rough conversion
                assert abs(pixel_width - expected_deg_scale) / expected_deg_scale < 0.5

                # Check data integrity
                data = src.read(1)
                assert not np.all(np.isnan(data))
                assert data.shape[0] > 0 and data.shape[1] > 0

    def test_subset_raster_normal_multistep(self, utm_test_data):
        """Test subset_raster_normal with multiple processing steps."""
        array, projection, geotransform = utm_test_data

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "input.tif")
            output_file = os.path.join(tmpdir, "output.tif")

            # Create input file
            self.create_test_file(array, projection, geotransform, input_file)

            # Define subset bounds
            minX = 460000
            maxX = 490000
            minY = 4010000
            maxY = 4040000
            scale = 150

            # Run subset_normal (multi-step process)
            subset_raster_normal(
                input_file, output_file, minX, maxY, maxX, minY, scale, tmpdir
            )

            # Verify output
            assert os.path.exists(output_file)

            with rasterio.open(output_file) as src:
                # Check bounds match request
                transform = src.transform

                # Verify resolution
                assert abs(transform.a - scale) < 1e-6
                assert abs(transform.e + scale) < 1e-6

                # Check data
                data = src.read(1)
                assert data.shape[0] > 0 and data.shape[1] > 0

            # Verify temp files are cleaned up
            temp_files = [f for f in os.listdir(tmpdir) if f.startswith("new")]
            assert len(temp_files) == 0

    def test_convert_to_wgs84_from_utm(self, utm_test_data):
        """Test UTM to WGS84 conversion."""
        array, projection, geotransform = utm_test_data

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "utm_input.tif")
            output_file = os.path.join(tmpdir, "wgs84_output.tif")

            # Create UTM input file
            self.create_test_file(array, projection, geotransform, input_file)

            # Define output bounds in WGS84
            # These roughly correspond to the UTM area
            lon1, lat1 = -3.5, 36.0
            lon2, lat2 = -2.5, 36.5

            # Convert
            convert_to_wgs84(input_file, output_file, lon1, lat1, lon2, lat2, tmpdir)

            # Verify output
            with rasterio.open(output_file) as src:
                # Should be WGS84
                assert src.crs.to_epsg() == 4326

                # Check bounds
                bounds = src.bounds
                assert bounds.left >= min(lon1, lon2)
                assert bounds.right <= max(lon1, lon2)
                assert bounds.bottom >= min(lat1, lat2)
                assert bounds.top <= max(lat1, lat2)

                # Data should be valid
                data = src.read(1)
                valid_pixels = (
                    ~np.isnan(data) if src.nodata is None else data != src.nodata
                )
                assert np.any(valid_pixels)

    def test_convert_to_wgs84_missing_projection(self):
        """Test handling of files with missing projection info."""
        array = np.random.rand(100, 100).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "no_proj.tif")
            output_file = os.path.join(tmpdir, "output.tif")

            # Create file without projection
            with rasterio.open(
                input_file,
                "w",
                driver="GTiff",
                height=100,
                width=100,
                count=1,
                dtype=array.dtype,
            ) as dst:
                dst.write(array, 1)

            # This should raise an error
            with pytest.raises(Exception):
                convert_to_wgs84(input_file, output_file, -1, -1, 1, 1, tmpdir)

    def test_crop_to_bb_utm_with_margin(self, utm_test_data):
        """Test UTM cropping with margin."""
        array, projection, geotransform = utm_test_data

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "utm_input.tif")
            output_file = os.path.join(tmpdir, "utm_cropped.tif")

            # Create input file
            self.create_test_file(array, projection, geotransform, input_file)

            # Define crop bounds (smaller than full extent)
            x1, y1 = 470000, 4020000
            x2, y2 = 480000, 4030000

            # Crop with margin
            crop_to_bb_utm(input_file, output_file, x1, y1, x2, y2)

            # Verify output
            with rasterio.open(output_file) as src:
                # Should maintain same CRS
                assert src.crs.to_string() == projection

                # Check bounds include margin (1000m)
                bounds = src.bounds
                assert bounds.left <= x1 - 1000
                assert bounds.right >= x2 + 1000
                assert bounds.bottom <= y1 - 1000
                assert bounds.top >= y2 + 1000

                # Data should be preserved
                data = src.read(1)
                assert data.shape[0] > 0 and data.shape[1] > 0

    def test_edge_case_dateline_crossing(self):
        """Test handling of dateline crossing (180/-180 longitude)."""
        # Create data that spans the dateline
        array = np.random.rand(100, 200).astype(np.float32)
        projection = "EPSG:4326"
        # Spans from 170 to -170 (crosses dateline)
        geotransform = (170.0, 0.1, 0.0, 10.0, 0.0, -0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "dateline.tif")
            output_file = os.path.join(tmpdir, "dateline_subset.tif")

            self.create_test_file(array, projection, geotransform, input_file)

            # Try to subset across dateline
            subset_raster(
                input_file,
                output_file,
                175.0,
                5.0,
                -175.0,
                -5.0,  # Crosses dateline
                0.1,
                tmpdir,
            )

            # Should handle gracefully
            assert os.path.exists(output_file)
            with rasterio.open(output_file) as src:
                data = src.read(1)
                assert data.shape[0] > 0 and data.shape[1] > 0

    def test_edge_case_polar_regions(self):
        """Test handling of polar regions."""
        # Create data near north pole
        array = np.random.rand(100, 100).astype(np.float32)
        projection = "EPSG:3413"  # NSIDC Polar Stereographic North
        geotransform = (-1000000.0, 1000.0, 0.0, 1000000.0, 0.0, -1000.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "polar.tif")
            output_file = os.path.join(tmpdir, "polar_wgs84.tif")

            self.create_test_file(array, projection, geotransform, input_file)

            # Convert to WGS84 near pole
            convert_to_wgs84(
                input_file,
                output_file,
                -180.0,
                85.0,
                180.0,
                89.0,  # Near north pole
                tmpdir,
            )

            # Should handle high latitude projections
            with rasterio.open(output_file) as src:
                assert src.crs.to_epsg() == 4326
                bounds = src.bounds
                assert bounds.top <= 90.0  # Can't exceed pole

    def test_performance_large_raster(self):
        """Test performance with larger rasters."""
        # Create a larger raster
        array = np.random.rand(5000, 5000).astype(np.float32)
        projection = "EPSG:32630"
        geotransform = (400000.0, 10.0, 0.0, 4500000.0, 0.0, -10.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "large.tif")
            output_file = os.path.join(tmpdir, "large_subset.tif")

            # Create with compression for space efficiency
            transform = Affine.from_gdal(*geotransform)
            with rasterio.open(
                input_file,
                "w",
                driver="GTiff",
                height=5000,
                width=5000,
                count=1,
                dtype=array.dtype,
                crs=projection,
                transform=transform,
                compress="deflate",
            ) as dst:
                dst.write(array, 1)

            # Subset a portion
            import time

            start_time = time.time()

            subset_raster(
                input_file, output_file, 420000, 4480000, 480000, 4420000, 50, tmpdir
            )

            elapsed = time.time() - start_time

            # Should complete in reasonable time
            assert elapsed < 30  # 30 seconds max

            # Verify output
            with rasterio.open(output_file) as src:
                assert src.crs.to_epsg() == 4326
                data = src.read(1)
                assert data.shape[0] > 0 and data.shape[1] > 0

    def test_error_handling_invalid_bounds(self, utm_test_data):
        """Test error handling for invalid bounds."""
        array, projection, geotransform = utm_test_data

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "input.tif")
            output_file = os.path.join(tmpdir, "output.tif")

            self.create_test_file(array, projection, geotransform, input_file)

            # Test completely outside bounds
            with pytest.raises(Exception):
                subset_raster(
                    input_file,
                    output_file,
                    0,
                    0,
                    10,
                    10,  # Way outside UTM zone 30N
                    100,
                    tmpdir,
                )

    def test_temporary_file_cleanup(self, utm_test_data):
        """Ensure temporary files are properly cleaned up."""
        array, projection, geotransform = utm_test_data

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "input.tif")
            output_file = os.path.join(tmpdir, "output.tif")

            self.create_test_file(array, projection, geotransform, input_file)

            # Get initial file count
            initial_files = set(os.listdir(tmpdir))

            # Run operations
            subset_raster(
                input_file, output_file, 470000, 4030000, 480000, 4020000, 100, tmpdir
            )

            # Check no temp files remain
            final_files = set(os.listdir(tmpdir))
            new_files = final_files - initial_files

            # Only output file should be new
            assert new_files == {os.path.basename(output_file)}

            # No temp files should remain
            temp_patterns = ["new1", "new2", "temp", "conv", "tmp"]
            remaining_temps = [
                f for f in final_files if any(pattern in f for pattern in temp_patterns)
            ]
            assert len(remaining_temps) == 0
