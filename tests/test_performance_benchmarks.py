"""Performance benchmarks for GDAL vs rasterio implementations."""

import os
import tempfile
import time
from contextlib import contextmanager
from typing import Dict, List, Tuple

import numpy as np
import psutil
import pytest
import rasterio
from osgeo import gdal
from rasterio.transform import Affine

# Enable GDAL exceptions
gdal.UseExceptions()


@contextmanager
def measure_performance():
    """Context manager to measure execution time and memory usage."""
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    start_time = time.time()

    yield

    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB

    return {
        "time": end_time - start_time,
        "memory_peak": end_memory,
        "memory_delta": end_memory - start_memory,
    }


class TestPerformanceBenchmarks:
    """Performance comparison between GDAL and rasterio."""

    @pytest.fixture(
        params=[
            (1000, 1000),  # Small: 1 megapixel
            (5000, 5000),  # Medium: 25 megapixels
            (10000, 10000),  # Large: 100 megapixels
        ]
    )
    def test_sizes(self, request):
        """Different raster sizes for benchmarking."""
        return request.param

    @pytest.fixture
    def test_data(self, test_sizes):
        """Generate test data for given size."""
        height, width = test_sizes
        array = np.random.rand(height, width).astype(np.float32) * 1000

        # Add some features
        # NaN region (10% of image)
        nan_size = int(height * 0.1)
        array[:nan_size, :nan_size] = np.nan

        # Zero region
        array[height // 2 : height // 2 + 100, width // 2 : width // 2 + 100] = 0

        projection = "EPSG:32630"
        geotransform = (500000.0, 30.0, 0.0, 4000000.0, 0.0, -30.0)

        return array, projection, geotransform

    def benchmark_gdal_write(self, array, projection, geotransform, output_path):
        """Benchmark GDAL write operation."""
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(
            output_path, array.shape[1], array.shape[0], 1, gdal.GDT_Float32
        )
        ds.SetProjection(projection)
        ds.SetGeoTransform(geotransform)
        band = ds.GetRasterBand(1)
        band.SetNoDataValue(-9999.0)
        band.WriteArray(array)
        band.FlushCache()
        ds.FlushCache()
        ds = None

    def benchmark_rasterio_write(self, array, projection, geotransform, output_path):
        """Benchmark rasterio write operation."""
        from rasterio.transform import Affine

        transform = Affine.from_gdal(*geotransform)

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=array.shape[0],
            width=array.shape[1],
            count=1,
            dtype=array.dtype,
            crs=projection,
            transform=transform,
            nodata=-9999.0,
        ) as dst:
            dst.write(array, 1)

    def benchmark_gdal_read(self, input_path):
        """Benchmark GDAL read operation."""
        ds = gdal.Open(input_path, gdal.GA_ReadOnly)
        array = ds.ReadAsArray()
        geotransform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        ds = None
        return array, projection, geotransform

    def benchmark_rasterio_read(self, input_path):
        """Benchmark rasterio read operation."""
        with rasterio.open(input_path) as src:
            array = src.read(1)
            projection = src.crs.to_string()
            geotransform = src.transform.to_gdal()
        return array, projection, geotransform

    def test_write_performance(self, test_data):
        """Compare write performance between GDAL and rasterio."""
        array, projection, geotransform = test_data
        size_mb = array.nbytes / 1024 / 1024

        results = {
            "size": f"{array.shape[0]}x{array.shape[1]}",
            "size_mb": size_mb,
            "gdal": {},
            "rasterio": {},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Benchmark GDAL
            gdal_path = os.path.join(tmpdir, "gdal_write.tif")

            process = psutil.Process()
            start_mem = process.memory_info().rss / 1024 / 1024
            start_time = time.time()

            self.benchmark_gdal_write(array, projection, geotransform, gdal_path)

            gdal_time = time.time() - start_time
            gdal_mem = process.memory_info().rss / 1024 / 1024 - start_mem

            results["gdal"] = {
                "time": gdal_time,
                "memory_delta": gdal_mem,
                "mb_per_sec": size_mb / gdal_time,
            }

            # Benchmark rasterio
            rio_path = os.path.join(tmpdir, "rio_write.tif")

            start_mem = process.memory_info().rss / 1024 / 1024
            start_time = time.time()

            self.benchmark_rasterio_write(array, projection, geotransform, rio_path)

            rio_time = time.time() - start_time
            rio_mem = process.memory_info().rss / 1024 / 1024 - start_mem

            results["rasterio"] = {
                "time": rio_time,
                "memory_delta": rio_mem,
                "mb_per_sec": size_mb / rio_time,
            }

            # Calculate performance ratios
            # Handle measurement noise for very small values
            effective_gdal_time = max(
                gdal_time, 0.001
            )  # Minimum 1ms for ratio calculation
            effective_gdal_mem = max(
                abs(gdal_mem), 1.0
            )  # Minimum 1MB, handle negative values

            results["ratios"] = {
                "time_ratio": rio_time / effective_gdal_time,
                "memory_ratio": rio_mem / effective_gdal_mem,
            }

            # Print results
            print(f"\nWrite Performance - {results['size']} ({size_mb:.1f} MB):")
            print(
                f"  GDAL:     {gdal_time:.3f}s, {gdal_mem:.1f} MB, {results['gdal']['mb_per_sec']:.1f} MB/s"
            )
            print(
                f"  Rasterio: {rio_time:.3f}s, {rio_mem:.1f} MB, {results['rasterio']['mb_per_sec']:.1f} MB/s"
            )
            print(
                f"  Ratio:    {results['ratios']['time_ratio']:.2f}x time, {results['ratios']['memory_ratio']:.2f}x memory"
            )

            # Assert performance is within acceptable bounds
            # Skip performance assertions for operations too fast to measure accurately
            if gdal_time < 0.001:  # Less than 1ms - measurement noise territory
                print(
                    f"  WARNING: GDAL time {gdal_time:.4f}s is below measurement precision"
                )
                # Just ensure rasterio completes in reasonable time
                assert rio_time < 0.050, (
                    f"Rasterio write took {rio_time:.3f}s (should complete within 50ms for small files)"
                )
            elif gdal_time < 0.010:  # 1-10ms range
                # Fixed overhead dominates
                assert rio_time < 0.030, (
                    f"Rasterio write took {rio_time:.3f}s for a {gdal_time:.3f}s operation (should complete within 30ms)"
                )
            elif size_mb > 100:
                # Large files - proportional overhead matters
                assert results["ratios"]["time_ratio"] <= 2.0, (
                    f"Rasterio write is {results['ratios']['time_ratio']:.2f}x slower than GDAL (limit: 2.0x for large files)"
                )
            else:
                # Medium files (10-100 MB)
                # Allow slightly more overhead as file size approaches 100MB
                limit = 1.80 if size_mb > 50 else 1.50
                assert results["ratios"]["time_ratio"] <= limit, (
                    f"Rasterio write is {results['ratios']['time_ratio']:.2f}x slower than GDAL (limit: {limit}x)"
                )

            # Memory usage: Skip unreliable measurements
            if abs(gdal_mem) < 1.0:  # Less than 1MB change - measurement noise
                print(
                    f"  WARNING: GDAL memory delta {gdal_mem:.1f}MB is below measurement precision"
                )
            elif gdal_mem < 0:  # Negative memory delta - measurement error
                print(
                    f"  WARNING: Negative GDAL memory delta {gdal_mem:.1f}MB indicates measurement error"
                )
            else:
                # Only check memory ratio for reliable measurements
                if size_mb < 10:  # Small files
                    assert results["ratios"]["memory_ratio"] <= 4.0, (
                        f"Rasterio uses {results['ratios']['memory_ratio']:.2f}x more memory than GDAL (limit: 4.0x for small files)"
                    )
                else:
                    assert results["ratios"]["memory_ratio"] <= 2.5, (
                        f"Rasterio uses {results['ratios']['memory_ratio']:.2f}x more memory than GDAL (limit: 2.5x)"
                    )

    def test_read_performance(self, test_data):
        """Compare read performance between GDAL and rasterio."""
        array, projection, geotransform = test_data
        size_mb = array.nbytes / 1024 / 1024

        results = {
            "size": f"{array.shape[0]}x{array.shape[1]}",
            "size_mb": size_mb,
            "gdal": {},
            "rasterio": {},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # First create a test file
            test_file = os.path.join(tmpdir, "test_read.tif")
            self.benchmark_gdal_write(array, projection, geotransform, test_file)

            # Benchmark GDAL read
            process = psutil.Process()
            start_mem = process.memory_info().rss / 1024 / 1024
            start_time = time.time()

            gdal_array, _, _ = self.benchmark_gdal_read(test_file)

            gdal_time = time.time() - start_time
            gdal_mem = process.memory_info().rss / 1024 / 1024 - start_mem

            results["gdal"] = {
                "time": gdal_time,
                "memory_delta": gdal_mem,
                "mb_per_sec": size_mb / gdal_time,
            }

            # Clear memory
            del gdal_array

            # Benchmark rasterio read
            start_mem = process.memory_info().rss / 1024 / 1024
            start_time = time.time()

            rio_array, _, _ = self.benchmark_rasterio_read(test_file)

            rio_time = time.time() - start_time
            rio_mem = process.memory_info().rss / 1024 / 1024 - start_mem

            results["rasterio"] = {
                "time": rio_time,
                "memory_delta": rio_mem,
                "mb_per_sec": size_mb / rio_time,
            }

            # Calculate performance ratios
            # Handle measurement noise
            effective_gdal_time = max(gdal_time, 0.001)  # Minimum 1ms
            effective_gdal_mem = max(
                abs(gdal_mem), 1.0
            )  # Minimum 1MB, handle negatives

            results["ratios"] = {
                "time_ratio": rio_time / effective_gdal_time,
                "memory_ratio": rio_mem / effective_gdal_mem,
            }

            # Print results
            print(f"\nRead Performance - {results['size']} ({size_mb:.1f} MB):")
            print(
                f"  GDAL:     {gdal_time:.3f}s, {gdal_mem:.1f} MB, {results['gdal']['mb_per_sec']:.1f} MB/s"
            )
            print(
                f"  Rasterio: {rio_time:.3f}s, {rio_mem:.1f} MB, {results['rasterio']['mb_per_sec']:.1f} MB/s"
            )
            print(
                f"  Ratio:    {results['ratios']['time_ratio']:.2f}x time, {results['ratios']['memory_ratio']:.2f}x memory"
            )

            # Assert performance is within acceptable bounds
            if gdal_time < 0.001:  # Below measurement precision
                print(
                    f"  WARNING: GDAL time {gdal_time:.4f}s is below measurement precision"
                )
                assert rio_time < 0.050, (
                    f"Rasterio read took {rio_time:.3f}s (should complete within 50ms)"
                )
            elif gdal_time < 0.020:  # Fast operations
                assert rio_time < 0.040, (
                    f"Rasterio read took {rio_time:.3f}s for a {gdal_time:.3f}s operation (should complete within 40ms)"
                )
            else:
                # Larger operations - check ratio
                assert results["ratios"]["time_ratio"] <= 1.50, (
                    f"Rasterio read is {results['ratios']['time_ratio']:.2f}x slower than GDAL (limit: 1.50x)"
                )

            # Memory ratio check - skip unreliable measurements
            if abs(gdal_mem) < 1.0:
                print(
                    f"  WARNING: GDAL memory delta {gdal_mem:.1f}MB is below measurement precision"
                )
            elif gdal_mem < 0:
                print(
                    f"  WARNING: Negative GDAL memory delta {gdal_mem:.1f}MB indicates measurement error"
                )
            else:
                # Only check for reliable measurements
                if size_mb < 10:
                    assert results["ratios"]["memory_ratio"] <= 3.0, (
                        f"Rasterio uses {results['ratios']['memory_ratio']:.2f}x more memory than GDAL (limit: 3.0x for small files)"
                    )
                else:
                    assert results["ratios"]["memory_ratio"] <= 2.0, (
                        f"Rasterio uses {results['ratios']['memory_ratio']:.2f}x more memory than GDAL (limit: 2.0x)"
                    )

    def test_reprojection_performance(self):
        """Compare reprojection performance."""
        # Create test data in UTM
        array = np.random.rand(2000, 2000).astype(np.float32) * 1000
        utm_projection = "EPSG:32630"
        utm_geotransform = (500000.0, 30.0, 0.0, 4000000.0, 0.0, -30.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create source file
            source_file = os.path.join(tmpdir, "source_utm.tif")
            self.benchmark_gdal_write(
                array, utm_projection, utm_geotransform, source_file
            )

            # Benchmark GDAL reprojection
            gdal_output = os.path.join(tmpdir, "gdal_wgs84.tif")

            start_time = time.time()

            warp_opts = gdal.WarpOptions(
                dstSRS="EPSG:4326",
                resampleAlg="bilinear",
                warpMemoryLimit=4000,
                multithread=True,
            )
            ds = gdal.Warp(gdal_output, source_file, options=warp_opts)
            ds = None

            gdal_time = time.time() - start_time

            # Benchmark rasterio reprojection
            rio_output = os.path.join(tmpdir, "rio_wgs84.tif")

            start_time = time.time()

            with rasterio.open(source_file) as src:
                # Calculate transform for WGS84
                from rasterio.warp import calculate_default_transform, reproject, Resampling

                dst_crs = "EPSG:4326"
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds
                )

                # Create output
                profile = src.profile.copy()
                profile.update(
                    {
                        "crs": dst_crs,
                        "transform": transform,
                        "width": width,
                        "height": height,
                    }
                )

                with rasterio.open(rio_output, "w", **profile) as dst:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear,
                    )

            rio_time = time.time() - start_time

            # Results
            ratio = rio_time / gdal_time
            print(f"\nReprojection Performance (UTM → WGS84):")
            print(f"  GDAL:     {gdal_time:.3f}s")
            print(f"  Rasterio: {rio_time:.3f}s")
            print(f"  Ratio:    {ratio:.2f}x")

            # Assert performance is comparable
            # Reprojection is more complex, so we check absolute time first
            if gdal_time < 0.050:  # Less than 50ms
                assert rio_time < 0.100, (
                    f"Rasterio reprojection took {rio_time:.3f}s for a {gdal_time:.3f}s operation (should complete within 100ms)"
                )
            else:
                # For longer operations, use ratio
                assert ratio <= 1.50, (
                    f"Rasterio reprojection is {ratio:.2f}x slower than GDAL (limit: 1.50x)"
                )

    def test_compression_performance(self):
        """Compare performance with different compression options."""
        array = np.random.rand(3000, 3000).astype(np.float32) * 1000
        projection = "EPSG:4326"
        geotransform = (0.0, 0.01, 0.0, 40.0, 0.0, -0.01)

        compression_options = [
            ("NONE", {}),
            ("LZW", {"compress": "lzw"}),
            ("DEFLATE", {"compress": "deflate", "zlevel": 6}),
        ]

        results = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for comp_name, rio_opts in compression_options:
                gdal_file = os.path.join(tmpdir, f"gdal_{comp_name}.tif")
                rio_file = os.path.join(tmpdir, f"rio_{comp_name}.tif")

                # GDAL write
                start_time = time.time()

                driver = gdal.GetDriverByName("GTiff")
                options = [] if comp_name == "NONE" else [f"COMPRESS={comp_name}"]
                if comp_name == "DEFLATE":
                    options.append("ZLEVEL=6")

                ds = driver.Create(
                    gdal_file,
                    array.shape[1],
                    array.shape[0],
                    1,
                    gdal.GDT_Float32,
                    options=options,
                )
                ds.SetProjection(projection)
                ds.SetGeoTransform(geotransform)
                band = ds.GetRasterBand(1)
                band.WriteArray(array)
                ds = None

                gdal_time = time.time() - start_time
                gdal_size = os.path.getsize(gdal_file) / 1024 / 1024  # MB

                # Rasterio write
                start_time = time.time()

                transform = Affine.from_gdal(*geotransform)
                with rasterio.open(
                    rio_file,
                    "w",
                    driver="GTiff",
                    height=array.shape[0],
                    width=array.shape[1],
                    count=1,
                    dtype=array.dtype,
                    crs=projection,
                    transform=transform,
                    **rio_opts,
                ) as dst:
                    dst.write(array, 1)

                rio_time = time.time() - start_time
                rio_size = os.path.getsize(rio_file) / 1024 / 1024  # MB

                result = {
                    "compression": comp_name,
                    "gdal_time": gdal_time,
                    "gdal_size": gdal_size,
                    "rio_time": rio_time,
                    "rio_size": rio_size,
                    "time_ratio": rio_time / gdal_time,
                    "size_ratio": rio_size / gdal_size,
                }
                results.append(result)

                print(f"\nCompression: {comp_name}")
                print(f"  GDAL:     {gdal_time:.3f}s, {gdal_size:.1f} MB")
                print(f"  Rasterio: {rio_time:.3f}s, {rio_size:.1f} MB")
                print(
                    f"  Ratios:   {result['time_ratio']:.2f}x time, {result['size_ratio']:.2f}x size"
                )

        # Verify compression efficiency is similar
        for result in results:
            assert abs(result["size_ratio"] - 1.0) < 0.1, (
                f"Size difference too large for {result['compression']}: {result['size_ratio']:.2f}x"
            )


class TestMemoryLeaks:
    """Test for memory leaks in repeated operations."""

    def test_repeated_writes_memory(self):
        """Test memory usage doesn't grow with repeated writes."""
        array = np.random.rand(1000, 1000).astype(np.float32)
        projection = "EPSG:4326"
        geotransform = (0, 1, 0, 0, 0, -1)

        process = psutil.Process()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Measure initial memory
            initial_memory = process.memory_info().rss / 1024 / 1024

            # Perform many write operations
            for i in range(50):
                # GDAL write
                gdal_file = os.path.join(tmpdir, f"gdal_{i}.tif")
                driver = gdal.GetDriverByName("GTiff")
                ds = driver.Create(gdal_file, 1000, 1000, 1, gdal.GDT_Float32)
                ds.SetProjection(projection)
                ds.SetGeoTransform(geotransform)
                band = ds.GetRasterBand(1)
                band.WriteArray(array)
                ds = None

                # Rasterio write
                rio_file = os.path.join(tmpdir, f"rio_{i}.tif")
                transform = Affine.from_gdal(*geotransform)
                with rasterio.open(
                    rio_file,
                    "w",
                    driver="GTiff",
                    height=1000,
                    width=1000,
                    count=1,
                    dtype=array.dtype,
                    crs=projection,
                    transform=transform,
                ) as dst:
                    dst.write(array, 1)

                # Clean up files to avoid disk space issues
                os.unlink(gdal_file)
                os.unlink(rio_file)

            # Check final memory
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = final_memory - initial_memory

            print(f"\nMemory leak test (50 iterations):")
            print(f"  Initial: {initial_memory:.1f} MB")
            print(f"  Final:   {final_memory:.1f} MB")
            print(f"  Growth:  {memory_growth:.1f} MB")

            # Memory growth should be minimal
            assert memory_growth < 50, (
                f"Memory grew by {memory_growth:.1f} MB over 50 operations"
            )
