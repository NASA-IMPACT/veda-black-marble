#!/usr/bin/env python3
"""Test different projection format handling between GDAL and rasterio."""

import os
import numpy as np
from osgeo import gdal, osr
import rasterio
import tempfile

# Enable GDAL exceptions
gdal.UseExceptions()

def test_projection_formats():
    """Test different ways of setting projection in GDAL."""
    
    # Create test data
    test_array = np.random.rand(100, 100).astype(np.float32)
    geotransform = (431500.5269564087, 29.989469284937346, 0.0, 
                     4480477.353728986, 0.0, -29.99918630562978)
    
    # Test different projection formats
    projections = {
        "EPSG string": "EPSG:32630",
        "Empty string": "",
        "None": None,
        "WKT from EPSG": None,  # Will be set below
    }
    
    # Get proper WKT for EPSG:32630
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32630)
    projections["WKT from EPSG"] = srs.ExportToWkt()
    
    for proj_name, projection in projections.items():
        print(f"\n{'='*60}")
        print(f"Testing projection format: {proj_name}")
        print(f"Projection value: {projection if projection else 'None/Empty'}")
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            temp_file = tmp.name
        
        try:
            # Write with GDAL
            driver = gdal.GetDriverByName("GTiff")
            dst = driver.Create(temp_file, 100, 100, 1, gdal.GDT_Float32)
            
            # Set geotransform
            dst.SetGeoTransform(geotransform)
            
            # Set projection (handle None case)
            if projection is not None:
                dst.SetProjection(projection)
            
            # Write data
            band = dst.GetRasterBand(1)
            band.SetNoDataValue(-9999.0)
            band.WriteArray(test_array)
            
            # Flush and close
            band.FlushCache()
            dst.FlushCache()
            dst = None
            
            # Verify with GDAL
            print("\nGDAL verification:")
            verify_ds = gdal.Open(temp_file, gdal.GA_ReadOnly)
            if verify_ds:
                proj = verify_ds.GetProjection()
                gt = verify_ds.GetGeoTransform()
                print(f"  Has projection: {'Yes' if proj else 'No'}")
                print(f"  Projection length: {len(proj) if proj else 0}")
                print(f"  GeoTransform: {gt}")
                verify_ds = None
            
            # Read with rasterio
            print("\nRasterio verification:")
            try:
                with rasterio.open(temp_file) as src:
                    print(f"  CRS: {src.crs}")
                    print(f"  Has transform: {src.transform != rasterio.Affine.identity()}")
                    print(f"  Transform: {src.transform}")
                    
                    # Check for warning conditions
                    if src.crs is None and src.transform != rasterio.Affine.identity():
                        print("  ⚠️  WARNING: Has transform but no CRS - this may trigger rasterio warning!")
            except Exception as e:
                print(f"  ERROR reading with rasterio: {e}")
                
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

if __name__ == "__main__":
    test_projection_formats()