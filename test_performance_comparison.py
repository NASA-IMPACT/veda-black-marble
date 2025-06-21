#!/usr/bin/env python3
"""
Direct performance comparison between GDAL and rasterio.
This test isolates the write operations to understand why we're seeing
performance differences when rasterio should be nearly identical to GDAL.
"""

import time
import tempfile
import os
import numpy as np
from osgeo import gdal
import rasterio
from rasterio.transform import Affine


def benchmark_gdal_write(array, projection, geotransform, output_path):
    """Direct GDAL write - minimal operations."""
    start = time.time()
    
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        output_path, 
        array.shape[1], 
        array.shape[0], 
        1, 
        gdal.GDT_Float32
    )
    ds.SetProjection(projection)
    ds.SetGeoTransform(geotransform)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(-9999.0)
    band.WriteArray(array)
    band.FlushCache()
    ds.FlushCache()
    ds = None
    
    return time.time() - start


def benchmark_rasterio_write(array, projection, geotransform, output_path):
    """Direct rasterio write - minimal operations."""
    start = time.time()
    
    transform = Affine.from_gdal(*geotransform)
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=projection,
        transform=transform,
        nodata=-9999.0
    ) as dst:
        dst.write(array, 1)
    
    return time.time() - start


def benchmark_rasterio_write_optimized(array, projection, geotransform, output_path):
    """Rasterio write with explicit settings to match GDAL behavior."""
    start = time.time()
    
    transform = Affine.from_gdal(*geotransform)
    
    # Explicitly set options that might affect performance
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=projection,
        transform=transform,
        nodata=-9999.0,
        # Disable features that might add overhead
        tiled=False,  # GDAL doesn't tile by default
        compress=None,  # No compression
        interleave='band',
        # GDAL-like creation options
        BIGTIFF='NO',
        NUM_THREADS=1
    ) as dst:
        dst.write(array, 1)
    
    return time.time() - start


def main():
    # Create test data
    sizes = [(100, 100), (1000, 1000), (2000, 2000)]
    projection = 'EPSG:32630'
    geotransform = (500000.0, 30.0, 0.0, 4000000.0, 0.0, -30.0)
    
    print("Performance comparison: GDAL vs Rasterio")
    print("=" * 60)
    
    for size in sizes:
        array = np.random.rand(*size).astype(np.float32) * 1000
        array[array < 100] = np.nan  # Add some NaN values
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run each benchmark multiple times
            runs = 5
            gdal_times = []
            rio_times = []
            rio_opt_times = []
            
            for i in range(runs):
                # GDAL
                gdal_path = os.path.join(tmpdir, f"gdal_{i}.tif")
                gdal_time = benchmark_gdal_write(array, projection, geotransform, gdal_path)
                gdal_times.append(gdal_time)
                
                # Rasterio (basic)
                rio_path = os.path.join(tmpdir, f"rio_{i}.tif")
                rio_time = benchmark_rasterio_write(array, projection, geotransform, rio_path)
                rio_times.append(rio_time)
                
                # Rasterio (optimized)
                rio_opt_path = os.path.join(tmpdir, f"rio_opt_{i}.tif")
                rio_opt_time = benchmark_rasterio_write_optimized(array, projection, geotransform, rio_opt_path)
                rio_opt_times.append(rio_opt_time)
                
                # Verify files are identical in size
                gdal_size = os.path.getsize(gdal_path)
                rio_size = os.path.getsize(rio_path)
                rio_opt_size = os.path.getsize(rio_opt_path)
                
                if i == 0:  # Only print once
                    print(f"\nFile sizes for {size[0]}x{size[1]}:")
                    print(f"  GDAL:          {gdal_size:,} bytes")
                    print(f"  Rasterio:      {rio_size:,} bytes")
                    print(f"  Rasterio opt:  {rio_opt_size:,} bytes")
            
            # Calculate averages (excluding first run for warmup)
            avg_gdal = np.mean(gdal_times[1:])
            avg_rio = np.mean(rio_times[1:])
            avg_rio_opt = np.mean(rio_opt_times[1:])
            
            print(f"\nArray size: {size[0]}x{size[1]} ({array.nbytes / 1024 / 1024:.1f} MB)")
            print(f"  GDAL:           {avg_gdal:.4f}s")
            print(f"  Rasterio:       {avg_rio:.4f}s ({avg_rio/avg_gdal:.2f}x)")
            print(f"  Rasterio opt:   {avg_rio_opt:.4f}s ({avg_rio_opt/avg_gdal:.2f}x)")
            
            # If rasterio is significantly slower, investigate why
            if avg_rio / avg_gdal > 1.5:
                print(f"  ⚠️  Rasterio is {avg_rio/avg_gdal:.1f}x slower - investigating...")
                
                # Check what might be different
                with rasterio.open(rio_path) as src:
                    print(f"     Tiled: {src.profile.get('tiled', False)}")
                    print(f"     Blocksize: {src.profile.get('blockxsize', 'N/A')} x {src.profile.get('blockysize', 'N/A')}")
                    print(f"     Compress: {src.profile.get('compress', 'None')}")


if __name__ == "__main__":
    main()