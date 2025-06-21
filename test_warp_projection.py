#!/usr/bin/env python3
"""Test GDAL Warp projection handling."""

import os
import numpy as np
from osgeo import gdal, osr
import rasterio
import tempfile

# Enable GDAL exceptions
gdal.UseExceptions()

def test_warp_projection_handling():
    """Test how gdal.Warp handles projection when not explicitly specified."""
    
    # Create test data with projection
    test_array = np.random.rand(100, 100).astype(np.float32)
    geotransform = (431500.5269564087, 29.989469284937346, 0.0, 
                     4480477.353728986, 0.0, -29.99918630562978)
    
    # Get proper WKT for EPSG:32630
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32630)
    projection = srs.ExportToWkt()
    
    with tempfile.NamedTemporaryFile(suffix='_source.tif', delete=False) as tmp:
        source_file = tmp.name
    
    with tempfile.NamedTemporaryFile(suffix='_warped.tif', delete=False) as tmp:
        warped_file = tmp.name
    
    try:
        # Write source file with proper projection
        print("Creating source file with projection...")
        driver = gdal.GetDriverByName("GTiff")
        dst = driver.Create(source_file, 100, 100, 1, gdal.GDT_Float32)
        dst.SetProjection(projection)
        dst.SetGeoTransform(geotransform)
        band = dst.GetRasterBand(1)
        band.WriteArray(test_array)
        band.FlushCache()
        dst.FlushCache()
        dst = None
        
        # Verify source file
        print("\nVerifying source file:")
        ds = gdal.Open(source_file)
        print(f"  Has projection: {'Yes' if ds.GetProjection() else 'No'}")
        print(f"  Has geotransform: {ds.GetGeoTransform()}")
        ds = None
        
        # Test 1: Warp without specifying target SRS
        print("\nTest 1: Warp without specifying target SRS")
        warp_opts = gdal.WarpOptions(
            format="GTiff",
            creationOptions=["COMPRESS=LZW", "TILED=YES"],
            resampleAlg="bilinear",
            multithread=True,
        )
        gdal.Warp(warped_file, source_file, options=warp_opts)
        
        # Check result with GDAL
        print("  GDAL verification:")
        ds = gdal.Open(warped_file)
        proj = ds.GetProjection()
        print(f"    Has projection: {'Yes' if proj else 'No'}")
        print(f"    Projection matches source: {proj == projection}")
        ds = None
        
        # Check with rasterio
        print("  Rasterio verification:")
        with rasterio.open(warped_file) as src:
            print(f"    CRS: {src.crs}")
            print(f"    Has valid geotransform: {src.transform != rasterio.Affine.identity()}")
            if src.crs is None and src.transform != rasterio.Affine.identity():
                print("    ⚠️  WARNING: This will trigger rasterio warning!")
        
        # Clean up for next test
        os.remove(warped_file)
        
        # Test 2: Warp from multiple sources (like in process_band_data)
        print("\nTest 2: Warp from multiple sources")
        source_files = [source_file]  # Simulating multiple sources
        
        warp_opts = gdal.WarpOptions(
            format="GTiff",
            creationOptions=["COMPRESS=LZW", "TILED=YES"],
            resampleAlg="bilinear",
            multithread=True,
        )
        gdal.Warp(warped_file, source_files, options=warp_opts)
        
        # Check result
        print("  GDAL verification:")
        ds = gdal.Open(warped_file)
        proj = ds.GetProjection()
        print(f"    Has projection: {'Yes' if proj else 'No'}")
        ds = None
        
        print("  Rasterio verification:")
        with rasterio.open(warped_file) as src:
            print(f"    CRS: {src.crs}")
            if src.crs is None:
                print("    ⚠️  WARNING: No CRS in warped file!")
                
    finally:
        # Clean up
        for f in [source_file, warped_file]:
            if os.path.exists(f):
                os.remove(f)

if __name__ == "__main__":
    test_warp_projection_handling()