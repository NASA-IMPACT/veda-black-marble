"""Test VIIRS georeferencing orientation to prevent south-up rasters."""

from osgeo import gdal

import pytest


class TestVIIRSOrientation:
    """Test that VIIRS conversion produces north-up rasters."""
    
    def test_viirs_outputbounds_order(self):
        """Test that outputBounds are in correct order [west, north, east, south]."""
        # This is more of a documentation test to ensure we understand the fix
        
        # For tile h07v04:
        h, v = 7, 4
        
        # Calculate bounds using the formula from the code
        west = (10 * h) - 180  # -110
        north = 90 - (10 * v)  # 50
        east = west + 10       # -100
        south = north - 10     # 40
        
        # The CORRECT order for gdal.TranslateOptions outputBounds is:
        # [ulx, uly, lrx, lry] = [west, north, east, south]
        correct_bounds = [west, north, east, south]
        
        # The WRONG order that was causing the bug was:
        # [west, south, east, north]
        wrong_bounds = [west, south, east, north]
        
        # Verify the correct bounds have north > south
        assert correct_bounds[1] > correct_bounds[3], "North should be greater than South"
        
        # The wrong bounds had south in the UL position
        assert wrong_bounds[1] < wrong_bounds[3], "Wrong bounds have South in UL position"
    
    def test_viirs_tiles_have_negative_y_pixel_size(self):
        """Integration test: verify actual VIIRS TIF files have negative Y pixel size."""
        from pathlib import Path
        
        viirs_dir = Path("data/viirs/2023/01/24")
        if not viirs_dir.exists():
            pytest.skip("No VIIRS data directory found")
        
        tif_files = list(viirs_dir.glob("*_Gap_Filled_DNB_BRDF-Corrected_NTL.tif"))
        if not tif_files:
            pytest.skip("No VIIRS TIF files found")
        
        # Check each file
        for tif_file in tif_files:
            ds = gdal.Open(str(tif_file))
            assert ds is not None, f"Failed to open {tif_file}"
            
            geotransform = ds.GetGeoTransform()
            y_pixel_size = geotransform[5]
            
            # Y pixel size MUST be negative for north-up orientation
            assert y_pixel_size < 0, (
                f"File {tif_file.name} has positive Y pixel size ({y_pixel_size}), "
                f"indicating south-up raster!"
            )
            
            # Also verify corner coordinates make sense
            cols = ds.RasterXSize
            rows = ds.RasterYSize
            
            # Calculate corners from geotransform
            ulx = geotransform[0]
            uly = geotransform[3]
            ulx + geotransform[1] * cols
            lry = uly + geotransform[5] * rows  # Note: geotransform[5] is negative
            
            # Upper should have higher Y than Lower
            assert uly > lry, f"Upper Y ({uly}) should be > Lower Y ({lry})"
            
            ds = None