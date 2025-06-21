"""Tests for rasterio-based save_geotiff_single implementation."""

import os
import tempfile

import numpy as np
import pytest
import rasterio
from osgeo import gdal

# Enable GDAL exceptions for tests
gdal.UseExceptions()


class TestRasterioImplementation:
    """Test suite for rasterio-based save_geotiff_single."""

    @pytest.fixture
    def test_data(self):
        """Create test data for all tests."""
        array = np.random.rand(100, 100).astype(np.float32) * 100
        array[10:20, 10:20] = np.nan  # Add NaN values

        geotransform = (431500.0, 30.0, 0.0, 4480000.0, 0.0, -30.0)
        projection = "EPSG:32630"

        return array, projection, geotransform

    @pytest.fixture
    def save_function(self):
        """Get the rasterio save function."""
        # Force rasterio implementation
        os.environ["BM_USE_RASTERIO_WRITER"] = "true"

        # Import and reload to pick up env var
        import importlib

        import blackmarble.io

        importlib.reload(blackmarble.io)

        from blackmarble.io import save_geotiff_single

        return save_geotiff_single

    def test_float32_with_nan(self, save_function, test_data):
        """Test saving float32 data with NaN values."""
        array, projection, geotransform = test_data

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            save_function(tmp.name, array, projection, geotransform)

            # Verify with GDAL
            ds = gdal.Open(tmp.name)
            assert ds is not None
            assert ds.GetProjection() != ""
            assert ds.GetGeoTransform() == geotransform

            band = ds.GetRasterBand(1)
            assert band.GetNoDataValue() == -9999.0

            data = band.ReadAsArray()
            # Check that NaN values were converted to NoData
            assert np.sum(data == -9999.0) == 100  # 10x10 NaN region

            ds = None
            os.unlink(tmp.name)

    def test_uint16_data(self, save_function):
        """Test saving uint16 data."""
        array = (np.random.rand(50, 50) * 10000).astype(np.uint16)
        geotransform = (500000.0, 30.0, 0.0, 4000000.0, 0.0, -30.0)
        projection = "EPSG:32630"

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            save_function(tmp.name, array, projection, geotransform)

            # Verify data integrity
            with rasterio.open(tmp.name) as src:
                assert src.crs is not None
                assert src.transform != rasterio.Affine.identity()
                assert src.nodata == 0
                assert src.dtypes[0] == "uint16"

                data = src.read(1)
                np.testing.assert_array_equal(data, array)

            os.unlink(tmp.name)

    def test_wkt_projection(self, save_function):
        """Test with WKT projection string."""
        array = np.random.rand(50, 50).astype(np.float32)
        geotransform = (500000.0, 30.0, 0.0, 4000000.0, 0.0, -30.0)

        # Full WKT string
        projection = """PROJCS["WGS 84 / UTM zone 30N",
            GEOGCS["WGS 84",
                DATUM["WGS_1984",
                    SPHEROID["WGS 84",6378137,298.257223563]],
                PRIMEM["Greenwich",0],
                UNIT["degree",0.0174532925199433]],
            PROJECTION["Transverse_Mercator"],
            PARAMETER["latitude_of_origin",0],
            PARAMETER["central_meridian",-3],
            PARAMETER["scale_factor",0.9996],
            PARAMETER["false_easting",500000],
            PARAMETER["false_northing",0],
            UNIT["metre",1]]"""

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            save_function(tmp.name, array, projection, geotransform)

            # Should be readable by both GDAL and rasterio
            ds = gdal.Open(tmp.name)
            assert ds is not None
            assert ds.GetProjection() != ""
            ds = None

            with rasterio.open(tmp.name) as src:
                assert src.crs is not None
                assert "utm" in str(src.crs).lower() or "32630" in str(src.crs)

            os.unlink(tmp.name)

    def test_reprojection_compatibility(self, save_function, test_data):
        """Test that saved files work with convert_to_wgs84."""
        from blackmarble.reproj import convert_to_wgs84

        array, projection, geotransform = test_data

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "input.tif")
            output_file = os.path.join(tmpdir, "output_wgs84.tif")

            # Save with rasterio
            save_function(input_file, array, projection, geotransform)

            # Try reprojection
            convert_to_wgs84(input_file, output_file, -3.5, 40.0, -2.5, 40.5, tmpdir)

            # Verify output
            assert os.path.exists(output_file)

            with rasterio.open(output_file) as src:
                assert src.crs.to_epsg() == 4326
                assert src.shape[0] > 0 and src.shape[1] > 0

    def test_edge_case_all_nan(self, save_function):
        """Test array with all NaN values."""
        array = np.full((50, 50), np.nan, dtype=np.float32)
        geotransform = (500000.0, 30.0, 0.0, 4000000.0, 0.0, -30.0)
        projection = "EPSG:32630"

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            save_function(tmp.name, array, projection, geotransform)

            with rasterio.open(tmp.name) as src:
                data = src.read(1)
                assert np.all(data == src.nodata)

            os.unlink(tmp.name)

    def test_minimal_interface_compatibility(self, save_function, test_data):
        """Test that the interface matches GDAL version exactly."""
        array, projection, geotransform = test_data

        # Should accept exactly 4 arguments
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            # This should work
            save_function(tmp.name, array, projection, geotransform)

            # Verify file was created
            assert os.path.exists(tmp.name)
            assert os.path.getsize(tmp.name) > 0

            os.unlink(tmp.name)
