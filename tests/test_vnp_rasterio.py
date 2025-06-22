"""Tests for rasterio-based VIIRS (VNP46A2) HDF5 handling."""

import os
import tempfile
from unittest.mock import Mock, patch, MagicMock

import h5py
import numpy as np
import pytest
import rasterio
from osgeo import gdal

from blackmarble.vnp_rasterio import convert_VNP46A2_HDF2TIFF, VNP46A2Util


class TestVNP46A2RasterioImplementation:
    """Test suite for rasterio-based VIIRS HDF5 processing."""

    @pytest.fixture
    def mock_hdf5_file(self):
        """Create a mock HDF5 file structure similar to VNP46A2."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            with h5py.File(tmp.name, "w") as f:
                # Create groups mimicking VNP46A2 structure
                grids = f.create_group("HDFEOS/GRIDS/VNP_Grid_DNB")

                # Add metadata attributes
                grids.attrs["HorizontalTileNumber"] = 17
                grids.attrs["VerticalTileNumber"] = 7

                # Create data fields
                data_fields = grids.create_group("Data Fields")

                # DNB_At_Sensor_Radiance_500m (main data)
                dnb_data = np.random.rand(2400, 2400).astype(np.float32) * 100
                dnb_data[100:200, 100:200] = 65535  # Add some extreme values
                data_fields.create_dataset(
                    "DNB_At_Sensor_Radiance_500m", data=dnb_data, compression="gzip"
                )

                # QF_Cloud_Mask (quality flag)
                qf_data = np.random.randint(0, 4, (2400, 2400), dtype=np.uint8)
                data_fields.create_dataset(
                    "QF_Cloud_Mask", data=qf_data, compression="gzip"
                )

                # Gap_Filled_DNB_BRDF_Corrected_NTL (processed data)
                gap_filled = np.random.rand(2400, 2400).astype(np.float32) * 50
                data_fields.create_dataset(
                    "Gap_Filled_DNB_BRDF_Corrected_NTL",
                    data=gap_filled,
                    compression="gzip",
                )

            yield tmp.name

        # Cleanup
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)

    @pytest.fixture
    def vnp_util(self, tmp_path):
        """Create VNP46A2Util instance."""
        return VNP46A2Util(output_dir=str(tmp_path))

    def test_coordinates_to_tile_id(self, vnp_util):
        """Test coordinate to tile ID conversion.

        VIIRS uses a 10-degree geographic grid (36x18 tiles).
        v = int((90 - lat) / 10)  # simplified: int(((-lat + 90) / 180) * 18)
        h = int((lon + 180) / 10) # simplified: int(((lon + 180) / 360) * 36)
        """
        # Test known coordinates with correct VIIRS tile positions
        test_cases = [
            # (lat, lon, expected_v, expected_h)
            (
                40.7128,
                -74.0060,
                4,
                10,
            ),  # New York: v=int((90-40.7)/10)=4, h=int((180-74)/10)=10
            (
                51.5074,
                -0.1278,
                3,
                17,
            ),  # London: v=int((90-51.5)/10)=3, h=int((180-0.1)/10)=17
            (
                -33.8688,
                151.2093,
                12,
                33,
            ),  # Sydney: v=int((90-(-33.9))/10)=12, h=int((180+151.2)/10)=33
            (
                35.6762,
                139.6503,
                5,
                31,
            ),  # Tokyo: v=int((90-35.7)/10)=5, h=int((180+139.7)/10)=31
            (0.0, 0.0, 9, 18),  # Equator/Prime Meridian: v=9, h=18
            (-90.0, 0.0, 18, 18),  # South Pole: v=18, h=18
            (90.0, 0.0, 0, 18),  # North Pole: v=0, h=18
        ]

        for lat, lon, expected_v, expected_h in test_cases:
            v, h = vnp_util.coordinates_to_tile_id(lat, lon)
            assert v == expected_v, (
                f"Failed for lat={lat}: expected v={expected_v}, got v={v}"
            )
            assert h == expected_h, (
                f"Failed for lon={lon}: expected h={expected_h}, got h={h}"
            )

    def test_get_tiles_for_bbox(self, vnp_util):
        """Test bounding box to tile list conversion.

        VIIRS 10-degree tiles: each tile covers 10x10 degrees.
        """
        # Test small area near New York (may span tile boundary)
        tiles = vnp_util.get_tiles_for_bbox(40.0, -75.0, 41.0, -74.0)
        # This bbox spans the boundary between v4 and v5 at lat 40
        assert len(tiles) == 2  # Tiles (4,10) and (5,10)
        assert (4, 10) in tiles
        assert (5, 10) in tiles

        # Test area spanning multiple tiles (10 degree span)
        tiles = vnp_util.get_tiles_for_bbox(35.0, -80.0, 45.0, -70.0)
        # v4: lats 40-50, v5: lats 30-40
        # h10: lons -80 to -70, h11: lons -70 to -60
        assert (
            len(tiles) == 4
        )  # 2 vertical x 2 horizontal = (4,10),(4,11),(5,10),(5,11)

        # Test area crossing equator
        tiles = vnp_util.get_tiles_for_bbox(-5.0, 10.0, 5.0, 20.0)
        # Spans v8-9 (equator at v9) and h19-20
        assert len(tiles) == 4  # 2x2 tiles

        # Test dateline crossing
        tiles = vnp_util.get_tiles_for_bbox(-10.0, 170.0, 10.0, -170.0)
        # Should include tiles on both sides of dateline
        h_values = [h for _, h in tiles]
        assert any(h >= 35 for h in h_values)  # Eastern hemisphere (170-180)
        assert any(h <= 1 for h in h_values)  # Western hemisphere (-180 to -170)

    def test_convert_hdf_to_tiff_basic(self, vnp_util, mock_hdf5_file):
        """Test basic HDF5 to GeoTIFF conversion."""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            output_file = tmp.name

            # Convert layer 2 (Gap_Filled_DNB_BRDF_Corrected_NTL)
            vnp_util.convert_hdf_to_tiff(mock_hdf5_file, output_file, layer=2)

            # Verify output with rasterio
            with rasterio.open(output_file) as src:
                # Check CRS (should be WGS84)
                assert src.crs is not None
                assert src.crs.to_epsg() == 4326

                # Check dimensions
                assert src.width == 2400
                assert src.height == 2400

                # Check georeferencing
                bounds = src.bounds
                # Tile h17v7 in VIIRS 10-degree grid:
                # h17: west = (10 * 17) - 180 = -10, east = 0
                # v7: north = 90 - (10 * 7) = 20, south = 10
                # This is in the Atlantic Ocean near Africa, NOT Eastern US
                assert bounds.left == -10
                assert bounds.right == 0
                assert bounds.bottom == 10
                assert bounds.top == 20

                # Check data
                data = src.read(1)
                assert data.shape == (2400, 2400)
                assert not np.all(data == 0)

            os.unlink(output_file)

    def test_convert_hdf_to_tiff_different_layers(self, vnp_util, mock_hdf5_file):
        """Test conversion of different HDF5 layers."""
        layers_to_test = [0, 1, 2]  # Different subdatasets

        for layer in layers_to_test:
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                output_file = tmp.name

                vnp_util.convert_hdf_to_tiff(mock_hdf5_file, output_file, layer=layer)

                # Each layer should produce valid output
                assert os.path.exists(output_file)

                with rasterio.open(output_file) as src:
                    assert src.count == 1
                    assert src.crs.to_epsg() == 4326
                    data = src.read(1)
                    assert data.shape == (2400, 2400)

                os.unlink(output_file)

    def test_convert_VNP46A2_HDF2TIFF_function(self, mock_hdf5_file):
        """Test the standalone conversion function."""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            output_file = tmp.name

            # Convert using the function
            convert_VNP46A2_HDF2TIFF(mock_hdf5_file, output_file)

            # Verify output
            with rasterio.open(output_file) as src:
                assert src.crs.to_epsg() == 4326
                assert src.width == 2400
                assert src.height == 2400

                # Check that extreme values were handled
                data = src.read(1)
                # Original had values up to 65535, these should be clipped
                assert np.max(data) < 65535

            os.unlink(output_file)

    @patch("requests.get")
    @patch("requests.Session")
    def test_download_h5_success(self, mock_session_class, mock_get, vnp_util):
        """Test successful HDF5 download."""
        # Mock the API response
        mock_api_response = Mock()
        mock_api_response.json.return_value = {
            "content": [
                {
                    "name": "VNP46A2.A2023001.h17v07.001.2023002123456.h5",
                    "downloadsLink": "https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/archives/...",
                }
            ]
        }
        mock_api_response.raise_for_status = Mock()
        mock_get.return_value = mock_api_response

        # Mock the download session
        mock_session = Mock()
        mock_download_response = Mock()
        mock_download_response.iter_content = Mock(return_value=[b"test data chunk"])
        mock_download_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_download_response
        mock_session_class.return_value = mock_session

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "test.h5")

            # Download
            vnp_util.download_h5(
                earthdata_token="test_token",
                year=2023,
                month=1,
                day=1,
                vertical=7,
                horizontal=17,
                target_filepath=output_file,
            )

            # Verify file was created
            assert os.path.exists(output_file)

            # Verify API was called correctly
            assert mock_get.called
            api_url = mock_get.call_args[0][0]
            assert "VNP46A2/2023/1" in api_url

            # Verify download session was configured
            mock_session.headers.update.assert_called_with(
                {"Authorization": "Bearer test_token"}
            )

    @patch("requests.get")
    def test_download_h5_no_data_found(self, mock_get, vnp_util):
        """Test handling when no data is found for the requested tile/date."""
        # Mock empty API response
        mock_response = Mock()
        mock_response.json.return_value = {"content": []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "test.h5")

            # Should raise RuntimeError
            with pytest.raises(RuntimeError, match="No VNP46A2 product found"):
                vnp_util.download_h5(
                    earthdata_token="test_token",
                    year=2023,
                    month=1,
                    day=1,
                    vertical=7,
                    horizontal=17,
                    target_filepath=output_file,
                )

    def test_edge_case_missing_metadata(self):
        """Test handling of HDF5 files with missing metadata."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            # Create HDF5 without proper metadata
            with h5py.File(tmp.name, "w") as f:
                grids = f.create_group("HDFEOS/GRIDS/VNP_Grid_DNB")
                # Missing tile number attributes
                data_fields = grids.create_group("Data Fields")
                data = np.random.rand(100, 100).astype(np.float32)
                data_fields.create_dataset("test_data", data=data)

            vnp_util = VNP46A2Util(output_dir=tmp.name + "_output")

            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as out_tmp:
                # Should handle missing metadata gracefully
                vnp_util.convert_hdf_to_tiff(tmp.name, out_tmp.name, layer=0)

                # Output should still be created (with default georeferencing)
                assert os.path.exists(out_tmp.name)

                with rasterio.open(out_tmp.name) as src:
                    # Should have some CRS even if metadata was missing
                    assert src.crs is not None

                os.unlink(out_tmp.name)

            os.unlink(tmp.name)

    def test_compatibility_with_gdal_output(self, vnp_util, mock_hdf5_file):
        """Test that rasterio output can be read by GDAL."""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            output_file = tmp.name

            # Convert with rasterio implementation
            vnp_util.convert_hdf_to_tiff(mock_hdf5_file, output_file, layer=2)

            # Read with GDAL
            ds = gdal.Open(output_file)
            assert ds is not None

            # Check georeferencing
            projection = ds.GetProjection()
            assert projection != ""
            assert "4326" in projection or "WGS" in projection

            geotransform = ds.GetGeoTransform()
            assert geotransform != (0, 1, 0, 0, 0, 1)  # Not default

            # Check data
            band = ds.GetRasterBand(1)
            data = band.ReadAsArray()
            assert data.shape == (2400, 2400)
            assert not np.all(data == 0)

            ds = None
            os.unlink(output_file)

    def test_memory_efficiency_large_hdf5(self):
        """Test memory efficiency with larger HDF5 files."""
        import psutil

        process = psutil.Process()

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            # Create larger HDF5 file
            with h5py.File(tmp.name, "w") as f:
                grids = f.create_group("HDFEOS/GRIDS/VNP_Grid_DNB")
                grids.attrs["HorizontalTileNumber"] = 17
                grids.attrs["VerticalTileNumber"] = 7

                data_fields = grids.create_group("Data Fields")
                # Create 5000x5000 dataset
                large_data = np.random.rand(5000, 5000).astype(np.float32)
                data_fields.create_dataset(
                    "Large_Dataset",
                    data=large_data,
                    compression="gzip",
                    chunks=(500, 500),  # Chunked for efficiency
                )

            vnp_util = VNP46A2Util(output_dir=tmp.name + "_output")

            # Measure memory before
            mem_before = process.memory_info().rss / 1024 / 1024  # MB

            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as out_tmp:
                vnp_util.convert_hdf_to_tiff(tmp.name, out_tmp.name, layer=0)

                # Measure memory after
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                mem_used = mem_after - mem_before

                # Memory usage should be reasonable (not loading entire dataset)
                # 5000x5000 float32 = 100MB, so memory use should be < 200MB
                assert mem_used < 200, f"Used too much memory: {mem_used:.1f} MB"

                os.unlink(out_tmp.name)

            os.unlink(tmp.name)

    def test_handle_corrupted_hdf5(self, vnp_util):
        """Test handling of corrupted HDF5 files."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            # Write garbage data
            tmp.write(b"This is not a valid HDF5 file!")
            tmp.flush()

            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as out_tmp:
                # Should raise an appropriate error
                with pytest.raises(Exception):
                    vnp_util.convert_hdf_to_tiff(tmp.name, out_tmp.name)

                # Cleanup
                if os.path.exists(out_tmp.name):
                    os.unlink(out_tmp.name)

            os.unlink(tmp.name)
