"""Compatibility tests between GDAL and rasterio implementations."""

import os
import tempfile
from typing import Tuple

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine
from osgeo import gdal

# Enable GDAL exceptions
gdal.UseExceptions()


class TestGDALRasterioCompatibility:
    """Test suite for GDAL/rasterio compatibility."""

    @pytest.fixture
    def sample_data(self) -> Tuple[np.ndarray, str, Tuple[float, ...]]:
        """Create sample raster data for testing."""
        # Create test array with various features
        array = np.random.rand(100, 100).astype(np.float32) * 1000
        array[10:20, 10:20] = np.nan  # NaN region
        array[30:40, 30:40] = 0  # Zero region
        array[50:60, 50:60] = -9999  # NoData-like values
        
        # UTM Zone 30N projection
        projection = "EPSG:32630"
        geotransform = (500000.0, 30.0, 0.0, 4000000.0, 0.0, -30.0)
        
        return array, projection, geotransform

    @pytest.fixture
    def wkt_projection(self) -> str:
        """Full WKT projection string for testing."""
        return """PROJCS["WGS 84 / UTM zone 30N",
            GEOGCS["WGS 84",
                DATUM["WGS_1984",
                    SPHEROID["WGS 84",6378137,298.257223563,
                        AUTHORITY["EPSG","7030"]],
                    AUTHORITY["EPSG","6326"]],
                PRIMEM["Greenwich",0,
                    AUTHORITY["EPSG","8901"]],
                UNIT["degree",0.0174532925199433,
                    AUTHORITY["EPSG","9122"]],
                AUTHORITY["EPSG","4326"]],
            PROJECTION["Transverse_Mercator"],
            PARAMETER["latitude_of_origin",0],
            PARAMETER["central_meridian",-3],
            PARAMETER["scale_factor",0.9996],
            PARAMETER["false_easting",500000],
            PARAMETER["false_northing",0],
            UNIT["metre",1,
                AUTHORITY["EPSG","9001"]],
            AXIS["Easting",EAST],
            AXIS["Northing",NORTH],
            AUTHORITY["EPSG","32630"]]"""

    def test_gdal_write_rasterio_read(self, sample_data):
        """Test GDAL write -> rasterio read compatibility."""
        array, projection, geotransform = sample_data
        
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            # Write with GDAL
            driver = gdal.GetDriverByName("GTiff")
            ds = driver.Create(tmp.name, array.shape[1], array.shape[0], 1, gdal.GDT_Float32)
            ds.SetProjection(projection)
            ds.SetGeoTransform(geotransform)
            band = ds.GetRasterBand(1)
            band.SetNoDataValue(-9999.0)
            band.WriteArray(array)
            ds = None
            
            # Read with rasterio
            with rasterio.open(tmp.name) as src:
                # Check metadata
                assert src.crs is not None
                assert src.crs.to_epsg() == 32630
                assert src.transform.almost_equals(Affine.from_gdal(*geotransform))
                assert src.nodata == -9999.0
                
                # Check data
                data = src.read(1)
                # Handle NaN comparison
                mask = ~np.isnan(array)
                np.testing.assert_array_almost_equal(data[mask], array[mask], decimal=5)
            
            os.unlink(tmp.name)

    def test_rasterio_write_gdal_read(self, sample_data):
        """Test rasterio write -> GDAL read compatibility."""
        array, projection, geotransform = sample_data
        
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            # Write with rasterio
            transform = Affine.from_gdal(*geotransform)
            
            with rasterio.open(
                tmp.name,
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
                # Convert NaN to nodata before writing
                write_array = np.where(np.isnan(array), -9999.0, array)
                dst.write(write_array, 1)
            
            # Read with GDAL
            ds = gdal.Open(tmp.name)
            assert ds is not None
            
            # Check metadata
            read_projection = ds.GetProjection()
            assert read_projection != ""
            
            read_geotransform = ds.GetGeoTransform()
            np.testing.assert_array_almost_equal(read_geotransform, geotransform, decimal=10)
            
            band = ds.GetRasterBand(1)
            assert band.GetNoDataValue() == -9999.0
            
            # Check data
            data = band.ReadAsArray()
            write_array = np.where(np.isnan(array), -9999.0, array)
            np.testing.assert_array_almost_equal(data, write_array, decimal=5)
            
            ds = None
            os.unlink(tmp.name)

    def test_round_trip_float32(self, sample_data):
        """Test GDAL -> rasterio -> GDAL round trip for float32 data."""
        array, projection, geotransform = sample_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "gdal_original.tif")
            file2 = os.path.join(tmpdir, "rasterio_copy.tif")
            file3 = os.path.join(tmpdir, "gdal_final.tif")
            
            # Step 1: Write with GDAL
            driver = gdal.GetDriverByName("GTiff")
            ds = driver.Create(file1, array.shape[1], array.shape[0], 1, gdal.GDT_Float32)
            ds.SetProjection(projection)
            ds.SetGeoTransform(geotransform)
            band = ds.GetRasterBand(1)
            band.SetNoDataValue(-9999.0)
            # Convert NaN to nodata
            write_array = np.where(np.isnan(array), -9999.0, array)
            band.WriteArray(write_array)
            ds = None
            
            # Step 2: Read with rasterio and write back
            with rasterio.open(file1) as src:
                profile = src.profile
                data = src.read(1)
                
                with rasterio.open(file2, 'w', **profile) as dst:
                    dst.write(data, 1)
            
            # Step 3: Read with GDAL and write back
            ds1 = gdal.Open(file2)
            driver = gdal.GetDriverByName("GTiff")
            ds2 = driver.CreateCopy(file3, ds1)
            ds1 = None
            ds2 = None
            
            # Compare original and final
            ds_orig = gdal.Open(file1)
            ds_final = gdal.Open(file3)
            
            orig_data = ds_orig.ReadAsArray()
            final_data = ds_final.ReadAsArray()
            
            np.testing.assert_array_almost_equal(orig_data, final_data, decimal=5)
            
            ds_orig = None
            ds_final = None

    def test_wkt_projection_compatibility(self, sample_data, wkt_projection):
        """Test WKT projection string compatibility."""
        array, _, geotransform = sample_data
        
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            # Write with GDAL using WKT
            driver = gdal.GetDriverByName("GTiff")
            ds = driver.Create(tmp.name, array.shape[1], array.shape[0], 1, gdal.GDT_Float32)
            ds.SetProjection(wkt_projection)
            ds.SetGeoTransform(geotransform)
            band = ds.GetRasterBand(1)
            band.WriteArray(array)
            ds = None
            
            # Read with rasterio
            with rasterio.open(tmp.name) as src:
                # Should correctly parse the WKT
                assert src.crs is not None
                epsg = src.crs.to_epsg()
                assert epsg == 32630 or src.crs.to_string() != ""
            
            os.unlink(tmp.name)

    def test_nodata_handling_compatibility(self):
        """Test NoData value handling between libraries."""
        test_cases = [
            (np.float32, -9999.0),
            (np.float32, np.nan),
            (np.float64, -9999.0),
            (np.uint16, 0),
            (np.uint16, 65535),
            (np.int16, -32768),
        ]
        
        for dtype, nodata_value in test_cases:
            array = np.ones((50, 50), dtype=dtype) * 100
            
            # Set some nodata
            if dtype in [np.float32, np.float64] and np.isnan(nodata_value):
                array[10:20, 10:20] = np.nan
            else:
                array[10:20, 10:20] = nodata_value
            
            geotransform = (0, 1, 0, 0, 0, -1)
            projection = "EPSG:4326"
            
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                # Write with GDAL
                driver = gdal.GetDriverByName("GTiff")
                gdal_dtype = gdal.GDT_Float32 if dtype == np.float32 else \
                            gdal.GDT_Float64 if dtype == np.float64 else \
                            gdal.GDT_UInt16 if dtype == np.uint16 else \
                            gdal.GDT_Int16
                
                ds = driver.Create(tmp.name, 50, 50, 1, gdal_dtype)
                ds.SetProjection(projection)
                ds.SetGeoTransform(geotransform)
                band = ds.GetRasterBand(1)
                
                if not (dtype in [np.float32, np.float64] and np.isnan(nodata_value)):
                    band.SetNoDataValue(float(nodata_value))
                
                band.WriteArray(array)
                ds = None
                
                # Read with rasterio
                with rasterio.open(tmp.name) as src:
                    data = src.read(1)
                    
                    if not (dtype in [np.float32, np.float64] and np.isnan(nodata_value)):
                        assert src.nodata == nodata_value
                        # Check that nodata regions are preserved
                        assert np.sum(data == nodata_value) == 100  # 10x10 region
                
                os.unlink(tmp.name)

    def test_compression_compatibility(self, sample_data):
        """Test different compression options compatibility."""
        array, projection, geotransform = sample_data
        
        compression_options = [
            ("COMPRESS=NONE", {}),
            ("COMPRESS=LZW", {"compress": "lzw"}),
            ("COMPRESS=DEFLATE", {"compress": "deflate"}),
            ("COMPRESS=PACKBITS", {"compress": "packbits"}),
        ]
        
        for gdal_opt, rio_opt in compression_options:
            with tempfile.TemporaryDirectory() as tmpdir:
                gdal_file = os.path.join(tmpdir, f"gdal_{gdal_opt.split('=')[1]}.tif")
                rio_file = os.path.join(tmpdir, f"rio_{gdal_opt.split('=')[1]}.tif")
                
                # Write with GDAL
                driver = gdal.GetDriverByName("GTiff")
                ds = driver.Create(
                    gdal_file, 
                    array.shape[1], 
                    array.shape[0], 
                    1, 
                    gdal.GDT_Float32,
                    options=[gdal_opt]
                )
                ds.SetProjection(projection)
                ds.SetGeoTransform(geotransform)
                band = ds.GetRasterBand(1)
                band.WriteArray(array)
                ds = None
                
                # Write with rasterio
                transform = Affine.from_gdal(*geotransform)
                with rasterio.open(
                    rio_file,
                    'w',
                    driver='GTiff',
                    height=array.shape[0],
                    width=array.shape[1],
                    count=1,
                    dtype=array.dtype,
                    crs=projection,
                    transform=transform,
                    **rio_opt
                ) as dst:
                    dst.write(array, 1)
                
                # Cross-read and compare
                # GDAL reads rasterio file
                ds = gdal.Open(rio_file)
                gdal_reads_rio = ds.ReadAsArray()
                ds = None
                
                # Rasterio reads GDAL file
                with rasterio.open(gdal_file) as src:
                    rio_reads_gdal = src.read(1)
                
                # Data should be identical (within floating point tolerance)
                mask = ~np.isnan(array)
                np.testing.assert_array_almost_equal(
                    gdal_reads_rio[mask], 
                    array[mask], 
                    decimal=5
                )
                np.testing.assert_array_almost_equal(
                    rio_reads_gdal[mask], 
                    array[mask], 
                    decimal=5
                )

    def test_multiband_compatibility(self):
        """Test multi-band raster compatibility."""
        # Create 3-band test data (RGB)
        height, width = 100, 100
        r_band = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        g_band = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        b_band = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        
        geotransform = (0, 1, 0, 0, 0, -1)
        projection = "EPSG:4326"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            gdal_file = os.path.join(tmpdir, "gdal_rgb.tif")
            rio_file = os.path.join(tmpdir, "rio_rgb.tif")
            
            # Write with GDAL
            driver = gdal.GetDriverByName("GTiff")
            ds = driver.Create(gdal_file, width, height, 3, gdal.GDT_Byte)
            ds.SetProjection(projection)
            ds.SetGeoTransform(geotransform)
            ds.GetRasterBand(1).WriteArray(r_band)
            ds.GetRasterBand(2).WriteArray(g_band)
            ds.GetRasterBand(3).WriteArray(b_band)
            ds = None
            
            # Write with rasterio
            transform = Affine.from_gdal(*geotransform)
            with rasterio.open(
                rio_file,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=3,
                dtype=np.uint8,
                crs=projection,
                transform=transform
            ) as dst:
                dst.write(r_band, 1)
                dst.write(g_band, 2)
                dst.write(b_band, 3)
            
            # Cross-read
            # GDAL reads rasterio
            ds = gdal.Open(rio_file)
            gdal_r = ds.GetRasterBand(1).ReadAsArray()
            gdal_g = ds.GetRasterBand(2).ReadAsArray()
            gdal_b = ds.GetRasterBand(3).ReadAsArray()
            ds = None
            
            # Rasterio reads GDAL
            with rasterio.open(gdal_file) as src:
                rio_rgb = src.read()
            
            # Compare
            np.testing.assert_array_equal(gdal_r, r_band)
            np.testing.assert_array_equal(gdal_g, g_band)
            np.testing.assert_array_equal(gdal_b, b_band)
            np.testing.assert_array_equal(rio_rgb[0], r_band)
            np.testing.assert_array_equal(rio_rgb[1], g_band)
            np.testing.assert_array_equal(rio_rgb[2], b_band)


class TestQABandCompatibility:
    """Test QA band handling compatibility."""
    
    def test_qa_band_zero_preservation(self):
        """Test that QA bands with 0 values are preserved correctly."""
        # Create QA band data where 0 is a valid value
        qa_data = np.array([
            [0, 1, 0, 1],    # 0 = clear, 1 = cloud
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ], dtype=np.uint16)
        
        geotransform = (0, 1, 0, 4, 0, -1)
        projection = "EPSG:4326"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with a QA band filename
            qa_file = os.path.join(tmpdir, "LC08_QA_PIXEL.tif")
            
            # Write with GDAL (0 should NOT be NoData for QA bands)
            driver = gdal.GetDriverByName("GTiff")
            ds = driver.Create(qa_file, 4, 4, 1, gdal.GDT_UInt16)
            ds.SetProjection(projection)
            ds.SetGeoTransform(geotransform)
            band = ds.GetRasterBand(1)
            # Don't set NoData for QA bands
            band.WriteArray(qa_data)
            ds = None
            
            # Read with rasterio
            with rasterio.open(qa_file) as src:
                data = src.read(1)
                # All values including 0 should be preserved
                np.testing.assert_array_equal(data, qa_data)
                # Should have no NoData value set
                assert src.nodata is None
            
            # Now test with blackmarble.io.get_geo
            from blackmarble.io import get_geo
            
            # This should recognize it's a QA band and not convert 0 to NaN
            array, _, _ = get_geo(qa_file, 1)
            
            # Check that zeros are preserved
            assert np.sum(array == 0) == np.sum(qa_data == 0)
            assert not np.any(np.isnan(array))