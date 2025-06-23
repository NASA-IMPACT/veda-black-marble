"""Test functions in blackmarble.acquire.landsat module.

This module tests Landsat data acquisition and tile selection functionality.
"""

import pytest
from unittest.mock import Mock, patch
import tempfile
import os
from datetime import datetime

# Import the module under test
from blackmarble.acquire.landsat import (
    get_wrs_path_row,
    filter_landsat_scenes,
    download_landsat_band,
    download_tiles_for_band,
    download_tiles_with_composites,
    get_landsat_metadata,
)


def test_get_wrs_path_row():
    """Test WRS-2 path and row calculation for coordinates.
    
    This test should verify that:
    - Valid coordinates return expected path/row combinations
    - Edge cases (poles, date line) are handled correctly
    - Multiple overlapping tiles are returned when appropriate
    - Invalid coordinates raise appropriate errors
    - Output format matches expected schema (list of dicts with 'path', 'row' keys)
    """
    with patch('blackmarble.acquire.landsat.landsat_module') as mock_landsat_module:
        # Mock the LandsatUtil class and its methods
        mock_util = Mock()
        mock_landsat_module.LandsatUtil.return_value = mock_util
        
        # Test Manhattan coordinates - should return path 014, row 032
        manhattan_lat, manhattan_lon = 40.7589, -73.9851
        expected_manhattan = [{"path": 14, "row": 32}]
        mock_util.get_wrs.return_value = expected_manhattan
        
        result = get_wrs_path_row(manhattan_lat, manhattan_lon)
        
        # Verify output format matches expected schema
        assert isinstance(result, list)
        assert len(result) == 1
        assert "path" in result[0]
        assert "row" in result[0]
        assert result[0]["path"] == 14
        assert result[0]["row"] == 32
        
        # Verify the function was called with correct coordinates
        mock_util.get_wrs.assert_called_once_with(manhattan_lat, manhattan_lon)
        
        # Test London coordinates - should return path 201, row 024
        london_lat, london_lon = 51.5074, -0.1278
        expected_london = [{"path": 201, "row": 24}]
        mock_util.get_wrs.return_value = expected_london
        
        result = get_wrs_path_row(london_lat, london_lon)
        
        assert result == expected_london
        
        # Test coordinates that might return no tiles (water/polar regions)
        arctic_lat, arctic_lon = 85.0, 0.0
        mock_util.get_wrs.return_value = []
        
        result = get_wrs_path_row(arctic_lat, arctic_lon)
        
        assert result == []
        assert isinstance(result, list)


def test_get_wrs_path_row_multiple_tiles():
    """Test coordinates that fall in multiple WRS tiles.
    
    This test should verify that:
    - Coordinates on tile boundaries return multiple path/row combinations
    - All overlapping tiles are included in results
    - Results are properly sorted/ordered
    """
    with patch('blackmarble.acquire.landsat.landsat_module') as mock_landsat_module:
        mock_util = Mock()
        mock_landsat_module.LandsatUtil.return_value = mock_util
        
        # Test coordinates on tile boundary that overlap multiple tiles
        boundary_lat, boundary_lon = 40.0, -100.0
        expected_multiple = [
            {"path": 28, "row": 33},
            {"path": 29, "row": 33},
            {"path": 28, "row": 34}
        ]
        mock_util.get_wrs.return_value = expected_multiple
        
        result = get_wrs_path_row(boundary_lat, boundary_lon)
        
        # Verify multiple tiles are returned
        assert isinstance(result, list)
        assert len(result) == 3
        
        # Verify all expected tiles are included
        for expected_tile in expected_multiple:
            assert expected_tile in result
        
        # Verify each result has the correct format
        for tile in result:
            assert "path" in tile
            assert "row" in tile
            assert isinstance(tile["path"], int)
            assert isinstance(tile["row"], int)
        
        # Test that function was called correctly
        mock_util.get_wrs.assert_called_once_with(boundary_lat, boundary_lon)


def test_filter_landsat_scenes():
    """Test Landsat scene filtering by date and location.
    
    This test should verify that:
    - Valid date/path/row combinations return expected scene directories
    - Different Landsat collections are handled correctly
    - Non-existent scenes return None
    - Date edge cases (leap years, invalid dates) are handled
    - AWS S3 bucket structure assumptions are validated
    """
    with patch('boto3.Session') as mock_session:
        # Mock S3 client and responses
        mock_s3 = Mock()
        mock_session.return_value.client.return_value = mock_s3
        
        # Test valid scene exists
        expected_scene_dir = "collection02/level-2/standard/oli-tirs/2023/014/032/LC08_L2SP_014032_20230915_20231017_02_T1/"
        mock_s3.list_objects_v2.return_value = {
            "CommonPrefixes": [
                {"Prefix": expected_scene_dir}
            ]
        }
        
        result = filter_landsat_scenes(
            year=2023, month=9, day=15,
            path=14, row=32
        )
        
        # Verify AWS S3 call was made correctly
        mock_s3.list_objects_v2.assert_called_once_with(
            Bucket="usgs-landsat",
            Prefix="collection02/level-2/standard/oli-tirs/2023/014/032/",
            RequestPayer="requester",
            Delimiter="/"
        )
        
        # Verify scene directory is returned
        assert result == expected_scene_dir
        
        # Test non-existent scene returns None
        mock_s3.list_objects_v2.return_value = {"CommonPrefixes": []}
        
        result = filter_landsat_scenes(
            year=2023, month=1, day=1,
            path=999, row=999  # Invalid path/row
        )
        
        assert result is None
        
        # Test date edge case - leap year
        mock_s3.list_objects_v2.return_value = {
            "CommonPrefixes": [
                {"Prefix": "collection02/level-2/standard/oli-tirs/2020/014/032/LC08_L2SP_014032_20200229_20200305_02_T1/"}
            ]
        }
        
        result = filter_landsat_scenes(
            year=2020, month=2, day=29,  # Leap year date
            path=14, row=32
        )
        
        assert result is not None
        assert "20200229" in result


def test_filter_landsat_scenes_collection_types():
    """Test filtering across different Landsat collections.
    
    This test should verify that:
    - Collection01 and Collection02 path structures are handled
    - Invalid collection names raise appropriate errors
    - Collection-specific naming conventions are respected
    """
    with patch('boto3.Session') as mock_session:
        mock_s3 = Mock()
        mock_session.return_value.client.return_value = mock_s3
        
        # Test Collection02 (default)
        collection02_scene = "collection02/level-2/standard/oli-tirs/2023/014/032/LC08_L2SP_014032_20230915_20231017_02_T1/"
        mock_s3.list_objects_v2.return_value = {
            "CommonPrefixes": [{"Prefix": collection02_scene}]
        }
        
        result = filter_landsat_scenes(
            year=2023, month=9, day=15,
            path=14, row=32,
            collection="collection02"
        )
        
        # Verify Collection02 path structure is used
        expected_prefix = "collection02/level-2/standard/oli-tirs/2023/014/032/"
        mock_s3.list_objects_v2.assert_called_with(
            Bucket="usgs-landsat",
            Prefix=expected_prefix,
            RequestPayer="requester",
            Delimiter="/"
        )
        
        assert result == collection02_scene
        
        # Note: The current implementation only supports Collection02
        # Testing Collection01 would require modifying the function
        # For now, we verify that the function uses the expected structure
        
        # Test that path and row are properly zero-padded
        result = filter_landsat_scenes(
            year=2023, month=9, day=15,
            path=1, row=1  # Single digits should be zero-padded
        )
        
        # Verify that single-digit path/row get zero-padded to 3 digits
        expected_prefix_padded = "collection02/level-2/standard/oli-tirs/2023/001/001/"
        calls = mock_s3.list_objects_v2.call_args_list
        found_padded_call = any(
            call[1]["Prefix"] == expected_prefix_padded 
            for call in calls
        )
        assert found_padded_call, f"Expected padded prefix {expected_prefix_padded} not found in calls"


def test_download_landsat_band():
    """Test downloading individual Landsat bands from AWS.
    
    This test should verify that:
    - Valid scene/band combinations download successfully
    - Different band types (spectral, QA) are handled correctly
    - AWS credentials are properly utilized when provided
    - Output files are created with correct format
    - Network errors are handled gracefully
    - File permissions and directory creation work correctly
    """
    with patch('boto3.Session') as mock_session, \
         patch('os.makedirs') as mock_makedirs:
        
        mock_s3 = Mock()
        mock_session.return_value.client.return_value = mock_s3
        
        # Test downloading spectral band
        scene_dir = "collection02/level-2/standard/oli-tirs/2023/014/032/LC08_L2SP_014032_20230915_20231017_02_T1/"
        band = 4  # Red band
        output_path = "/tmp/test_band4.tif"
        
        result = download_landsat_band(scene_dir, band, output_path)
        
        # Verify directory creation
        mock_makedirs.assert_called_once_with("/tmp", exist_ok=True)
        
        # Verify S3 download call
        expected_key = scene_dir + "LC08_L2SP_014032_20230915_20231017_02_T1_SR_B4.TIF"
        mock_s3.download_file.assert_called_once_with(
            "usgs-landsat",
            expected_key,
            output_path,
            ExtraArgs={"RequestPayer": "requester"}
        )
        
        # Verify return value
        assert result == output_path
        
        # Test downloading QA_PIXEL band
        mock_s3.reset_mock()
        mock_makedirs.reset_mock()
        
        qa_output_path = "/tmp/test_qa_pixel.tif"
        result = download_landsat_band(scene_dir, "QA_PIXEL", qa_output_path)
        
        expected_qa_key = scene_dir + "LC08_L2SP_014032_20230915_20231017_02_T1_QA_PIXEL.TIF"
        mock_s3.download_file.assert_called_once_with(
            "usgs-landsat",
            expected_qa_key,
            qa_output_path,
            ExtraArgs={"RequestPayer": "requester"}
        )
        
        # Test downloading QA_RADSAT band
        mock_s3.reset_mock()
        
        radsat_output_path = "/tmp/test_qa_radsat.tif"
        result = download_landsat_band(scene_dir, "QA_RADSAT", radsat_output_path)
        
        expected_radsat_key = scene_dir + "LC08_L2SP_014032_20230915_20231017_02_T1_QA_RADSAT.TIF"
        mock_s3.download_file.assert_called_once_with(
            "usgs-landsat",
            expected_radsat_key,
            radsat_output_path,
            ExtraArgs={"RequestPayer": "requester"}
        )
        
        # Test downloading QA_AEROSOL band
        mock_s3.reset_mock()
        
        aerosol_output_path = "/tmp/test_qa_aerosol.tif"
        result = download_landsat_band(scene_dir, "QA_AEROSOL", aerosol_output_path)
        
        expected_aerosol_key = scene_dir + "LC08_L2SP_014032_20230915_20231017_02_T1_SR_QA_AEROSOL.TIF"
        mock_s3.download_file.assert_called_once_with(
            "usgs-landsat",
            expected_aerosol_key,
            aerosol_output_path,
            ExtraArgs={"RequestPayer": "requester"}
        )


def test_download_landsat_band_with_credentials():
    """Test band download with custom AWS credentials.
    
    This test should verify that:
    - Custom AWS credentials override default credentials
    - Credential validation occurs before download attempts
    - Invalid credentials result in appropriate error messages
    """
    with patch('boto3.Session') as mock_session, \
         patch('os.makedirs'):
        
        mock_s3 = Mock()
        mock_session.return_value.client.return_value = mock_s3
        
        # Test with custom credentials (note: current implementation doesn't use them)
        scene_dir = "collection02/level-2/standard/oli-tirs/2023/014/032/LC08_L2SP_014032_20230915_20231017_02_T1/"
        band = 4
        output_path = "/tmp/test_with_creds.tif"
        
        # Custom AWS credentials
        aws_credentials = {
            "aws_access_key_id": "test_key",
            "aws_secret_access_key": "test_secret",
            "aws_session_token": "test_token"
        }
        
        result = download_landsat_band(
            scene_dir, band, output_path, aws_credentials
        )
        
        # Verify boto3 Session was created (current implementation uses default)
        mock_session.assert_called_once()
        
        # Verify S3 client was created
        mock_session.return_value.client.assert_called_once_with("s3")
        
        # Verify download was attempted
        assert mock_s3.download_file.called
        
        # Verify return value
        assert result == output_path
        
        # Note: The current implementation doesn't actually use the provided credentials
        # This is a limitation that could be addressed in future improvements
        # For now, we verify that the function accepts the parameter without error


def test_download_tiles_for_band():
    """Test downloading all tiles covering a bounding box.
    
    This test should verify that:
    - Bounding box to WRS tile conversion works correctly
    - All necessary tiles are identified and downloaded
    - Missing tiles are handled according to ignore_missing parameter
    - Output file organization follows expected structure
    - Partial downloads can be resumed
    - Return values include all expected file paths
    """
    with patch('blackmarble.acquire.landsat._download_tiles_for_band') as mock_download, \
         patch('tempfile.mkdtemp') as mock_mkdtemp:
        
        # Mock temporary directory creation
        mock_temp_dir = "/tmp/landsat_test123"
        mock_mkdtemp.return_value = mock_temp_dir
        
        # Mock the underlying download function return value
        expected_band_files = ["/tmp/landsat_test123/LC08_L2SP_014032_20230915_20231017_02_T1_SR_B4.TIF"]
        expected_qa_pixel_files = ["/tmp/landsat_test123/LC08_L2SP_014032_20230915_20231017_02_T1_QA_PIXEL.TIF"]
        expected_qa_radsat_files = ["/tmp/landsat_test123/LC08_L2SP_014032_20230915_20231017_02_T1_QA_RADSAT.TIF"]
        expected_qa_aerosol_files = ["/tmp/landsat_test123/LC08_L2SP_014032_20230915_20231017_02_T1_SR_QA_AEROSOL.TIF"]
        
        mock_download.return_value = (
            expected_band_files,
            expected_qa_pixel_files, 
            expected_qa_radsat_files,
            expected_qa_aerosol_files
        )
        
        # Test bounding box covering Manhattan area
        bbox = (-74.0, 40.7, -73.9, 40.8)  # (min_lon, min_lat, max_lon, max_lat)
        band = 4
        year, month, day = 2023, 9, 15
        
        result = download_tiles_for_band(
            band, year, month, day, bbox, ignore_missing=False
        )
        
        # Verify temporary directory was created
        mock_mkdtemp.assert_called_once_with(prefix="landsat_")
        
        # Verify the underlying function was called with correct parameters
        # Note: bbox (min_lon, min_lat, max_lon, max_lat) -> (lat1, lon1, lat2, lon2)
        mock_download.assert_called_once_with(
            band, year, month, day,
            40.7, -74.0,  # min_lat, min_lon (lat1, lon1)
            40.8, -73.9,  # max_lat, max_lon (lat2, lon2)
            mock_temp_dir,
            False  # ignore_missing
        )
        
        # Verify return value format
        assert isinstance(result, tuple)
        assert len(result) == 4
        
        band_files, qa_pixel_files, qa_radsat_files, qa_aerosol_files = result
        assert band_files == expected_band_files
        assert qa_pixel_files == expected_qa_pixel_files
        assert qa_radsat_files == expected_qa_radsat_files
        assert qa_aerosol_files == expected_qa_aerosol_files
        
        # Test with ignore_missing=True
        mock_download.reset_mock()
        
        result = download_tiles_for_band(
            band, year, month, day, bbox, ignore_missing=True
        )
        
        # Verify ignore_missing parameter is passed correctly
        mock_download.assert_called_once_with(
            band, year, month, day,
            40.7, -74.0,  # min_lat, min_lon
            40.8, -73.9,  # max_lat, max_lon
            mock_temp_dir,
            True  # ignore_missing
        )


def test_download_tiles_for_band_missing_tiles():
    """Test behavior when some tiles are missing from AWS.
    
    This test should verify that:
    - ignore_missing=True allows processing to continue
    - Partial results are properly reported
    - Download status is accurately tracked
    """
    with patch('blackmarble.acquire.landsat._download_tiles_for_band') as mock_download, \
         patch('tempfile.mkdtemp') as mock_mkdtemp:
        
        mock_temp_dir = "/tmp/landsat_missing_test"
        mock_mkdtemp.return_value = mock_temp_dir
        
        bbox = (-74.0, 40.7, -73.9, 40.8)
        band = 4
        year, month, day = 2023, 9, 15
        
        # Test ignore_missing=True - should continue with partial results
        partial_band_files = ["/tmp/landsat_missing_test/LC08_L2SP_014032_20230915_20231017_02_T1_SR_B4.TIF"]
        partial_qa_pixel_files = []  # Some QA files missing
        partial_qa_radsat_files = ["/tmp/landsat_missing_test/LC08_L2SP_014032_20230915_20231017_02_T1_QA_RADSAT.TIF"]
        partial_qa_aerosol_files = []
        
        mock_download.return_value = (
            partial_band_files,
            partial_qa_pixel_files,
            partial_qa_radsat_files, 
            partial_qa_aerosol_files
        )
        
        result = download_tiles_for_band(
            band, year, month, day, bbox, ignore_missing=True
        )
        
        # Verify partial results are returned
        band_files, qa_pixel_files, qa_radsat_files, qa_aerosol_files = result
        
        assert len(band_files) == 1
        assert len(qa_pixel_files) == 0  # Missing files
        assert len(qa_radsat_files) == 1
        assert len(qa_aerosol_files) == 0  # Missing files
        
        # Verify ignore_missing was passed correctly
        mock_download.assert_called_once_with(
            band, year, month, day,
            40.7, -74.0,  # lat1, lon1
            40.8, -73.9,  # lat2, lon2
            mock_temp_dir,
            True  # ignore_missing=True
        )
        
        # Test ignore_missing=False - should raise error when tiles missing
        mock_download.reset_mock()
        mock_download.side_effect = RuntimeError("No data for tile P014R032 on 2023-9-15")
        
        with pytest.raises(RuntimeError, match="No data for tile P014R032 on 2023-9-15"):
            download_tiles_for_band(
                band, year, month, day, bbox, ignore_missing=False
            )
        
        # Verify the underlying function was called with ignore_missing=False
        mock_download.assert_called_once_with(
            band, year, month, day,
            40.7, -74.0,
            40.8, -73.9,
            mock_temp_dir,
            False  # ignore_missing=False
        )


def test_download_tiles_with_composites():
    """Test downloading tiles with historical composite support.
    
    This test should verify that:
    - Historical month parameter affects tile selection
    - Composite fallback works when primary date unavailable
    - Date range calculations respect month boundaries
    - Quality ranking prioritizes recent over historical data
    - Composite metadata is properly tracked
    """
    with patch('blackmarble.acquire.landsat._download_tiles_for_band_with_composites') as mock_download, \
         patch('tempfile.mkdtemp') as mock_mkdtemp:
        
        mock_temp_dir = "/tmp/landsat_composite_test"
        mock_mkdtemp.return_value = mock_temp_dir
        
        # Mock return value - list of composite results
        primary_files = (
            ["/tmp/primary_B4.TIF"],
            ["/tmp/primary_QA_PIXEL.TIF"],
            ["/tmp/primary_QA_RADSAT.TIF"],
            ["/tmp/primary_QA_AEROSOL.TIF"]
        )
        composite1_files = (
            ["/tmp/composite1_B4.TIF"],
            ["/tmp/composite1_QA_PIXEL.TIF"],
            ["/tmp/composite1_QA_RADSAT.TIF"],
            ["/tmp/composite1_QA_AEROSOL.TIF"]
        )
        composite2_files = (
            ["/tmp/composite2_B4.TIF"],
            ["/tmp/composite2_QA_PIXEL.TIF"],
            ["/tmp/composite2_QA_RADSAT.TIF"],
            ["/tmp/composite2_QA_AEROSOL.TIF"]
        )
        
        mock_download.return_value = [primary_files, composite1_files, composite2_files]
        
        bbox = (-74.0, 40.7, -73.9, 40.8)
        band = 4
        year, month, day = 2023, 9, 15
        composite_history_months = 2
        
        result = download_tiles_with_composites(
            band, year, month, day, bbox, composite_history_months
        )
        
        # Verify temporary directory creation
        mock_mkdtemp.assert_called_once_with(prefix="landsat_composite_")
        
        # Verify underlying function was called with correct parameters
        mock_download.assert_called_once_with(
            band, year, month, day,
            40.7, -74.0,  # min_lat, min_lon (lat1, lon1)
            40.8, -73.9,  # max_lat, max_lon (lat2, lon2)
            mock_temp_dir,
            composite_history_months
        )
        
        # Verify return value format - list of tuples
        assert isinstance(result, list)
        assert len(result) == 3  # Primary + 2 composites
        
        # Verify each result is a tuple with 4 elements
        for composite_result in result:
            assert isinstance(composite_result, tuple)
            assert len(composite_result) == 4
            band_files, qa_pixel, qa_radsat, qa_aerosol = composite_result
            assert isinstance(band_files, list)
            assert isinstance(qa_pixel, list)
            assert isinstance(qa_radsat, list)
            assert isinstance(qa_aerosol, list)
        
        # Verify first result is primary (most recent)
        assert result[0] == primary_files
        assert result[1] == composite1_files
        assert result[2] == composite2_files


def test_download_tiles_with_composites_date_ranges():
    """Test composite date range calculations.
    
    This test should verify that:
    - Different composite_history_months values work correctly
    - Year boundaries are handled properly
    - Seasonal patterns can be captured
    - Date arithmetic edge cases are handled
    """
    with patch('blackmarble.acquire.landsat._download_tiles_for_band_with_composites') as mock_download, \
         patch('tempfile.mkdtemp') as mock_mkdtemp:
        
        mock_temp_dir = "/tmp/landsat_date_test"
        mock_mkdtemp.return_value = mock_temp_dir
        
        # Mock return value for different date ranges
        mock_download.return_value = [(
            ["/tmp/test_B4.TIF"],
            ["/tmp/test_QA_PIXEL.TIF"],
            ["/tmp/test_QA_RADSAT.TIF"],
            ["/tmp/test_QA_AEROSOL.TIF"]
        )]
        
        bbox = (-74.0, 40.7, -73.9, 40.8)
        band = 4
        
        # Test different composite_history_months values
        for history_months in [1, 6, 12, 24]:
            mock_download.reset_mock()
            
            result = download_tiles_with_composites(
                band, 2023, 9, 15, bbox, history_months
            )
            
            # Verify the parameter was passed correctly
            mock_download.assert_called_once_with(
                band, 2023, 9, 15,
                40.7, -74.0,
                40.8, -73.9,
                mock_temp_dir,
                history_months
            )
            
            assert isinstance(result, list)
        
        # Test year boundary cases - January date
        mock_download.reset_mock()
        
        result = download_tiles_with_composites(
            band, 2023, 1, 15, bbox, 6  # 6 months back from January
        )
        
        # Should handle crossing year boundary (back to 2022)
        mock_download.assert_called_once_with(
            band, 2023, 1, 15,
            40.7, -74.0,
            40.8, -73.9,
            mock_temp_dir,
            6
        )
        
        # Test seasonal pattern capture - different seasons
        mock_download.reset_mock()
        
        # Summer date with 12 months history (captures full year)
        result = download_tiles_with_composites(
            band, 2023, 7, 15, bbox, 12
        )
        
        mock_download.assert_called_once_with(
            band, 2023, 7, 15,
            40.7, -74.0,
            40.8, -73.9,
            mock_temp_dir,
            12
        )
        
        # Test edge case - zero history months
        mock_download.reset_mock()
        
        result = download_tiles_with_composites(
            band, 2023, 9, 15, bbox, 0
        )
        
        mock_download.assert_called_once_with(
            band, 2023, 9, 15,
            40.7, -74.0,
            40.8, -73.9,
            mock_temp_dir,
            0
        )


def test_get_landsat_metadata():
    """Test extraction of metadata from Landsat scenes.
    
    This test should verify that:
    - Standard metadata fields are extracted correctly
    - Different Landsat sensor types are supported
    - Coordinate reference system information is preserved
    - Acquisition parameters (sun angles, etc.) are included
    - Missing metadata fields are handled gracefully
    - Output format is consistent and standardized
    """
    with patch('osgeo.gdal.Open') as mock_gdal_open:
        # Mock GDAL dataset
        mock_dataset = Mock()
        mock_gdal_open.return_value = mock_dataset
        
        # Configure mock dataset properties
        mock_dataset.RasterXSize = 7761
        mock_dataset.RasterYSize = 7621
        mock_dataset.RasterCount = 7
        mock_dataset.GetProjection.return_value = 'PROJCS["WGS 84 / UTM zone 18N",GEOGCS["WGS 84",DATUM["WGS_1984"]]'
        mock_dataset.GetGeoTransform.return_value = (419385.0, 30.0, 0.0, 4582815.0, 0.0, -30.0)
        
        # Mock Landsat-specific metadata
        landsat_metadata = {
            'LANDSAT_SCENE_ID': 'LC08_L2SP_014032_20230915_20231017_02_T1',
            'SPACECRAFT_ID': 'LANDSAT_8',
            'SENSOR_ID': 'OLI_TIRS',
            'SUN_ELEVATION': '45.678',
            'SUN_AZIMUTH': '156.789'
        }
        mock_dataset.GetMetadata.return_value = landsat_metadata
        
        # Test with a realistic Landsat scene path
        scene_path = "/tmp/LC08_L2SP_014032_20230915_20231017_02_T1_SR_B4.TIF"
        
        result = get_landsat_metadata(scene_path)
        
        # Verify GDAL open was called
        mock_gdal_open.assert_called_once_with(scene_path)
        
        # Verify standard metadata fields are extracted
        assert result["width"] == 7761
        assert result["height"] == 7621
        assert result["bands"] == 7
        assert "WGS 84 / UTM zone 18N" in result["projection"]
        assert result["geotransform"] == (419385.0, 30.0, 0.0, 4582815.0, 0.0, -30.0)
        
        # Verify Landsat Collection 2 standard scaling factors
        assert result["scale_factor"] == 0.0000275
        assert result["add_offset"] == -0.2
        assert result["valid_range"] == [7273, 43636]
        assert result["fill_value"] == 0
        
        # Verify acquisition date extraction from filename
        assert result["acquisition_date"] == "20230915"
        
        # Verify GDAL metadata is included
        assert result["gdal_metadata"] == landsat_metadata
        
        # Verify output format is consistent
        required_fields = [
            "width", "height", "bands", "projection", "geotransform",
            "scale_factor", "add_offset", "valid_range", "fill_value"
        ]
        for field in required_fields:
            assert field in result
        
        # Test different sensor types - Landsat 9
        mock_dataset.reset_mock()
        landsat9_metadata = {
            'LANDSAT_SCENE_ID': 'LC09_L2SP_014032_20230915_20231017_02_T1',
            'SPACECRAFT_ID': 'LANDSAT_9',
            'SENSOR_ID': 'OLI_TIRS'
        }
        mock_dataset.GetMetadata.return_value = landsat9_metadata
        
        l9_scene_path = "/tmp/LC09_L2SP_014032_20230915_20231017_02_T1_SR_B4.TIF"
        result = get_landsat_metadata(l9_scene_path)
        
        # Verify same standard fields are present
        assert result["scale_factor"] == 0.0000275
        assert result["gdal_metadata"] == landsat9_metadata


def test_get_landsat_metadata_error_handling():
    """Test metadata extraction error handling.
    
    This test should verify that:
    - Corrupted or incomplete metadata files are handled
    - Network timeouts during metadata access are managed
    - Invalid file paths result in appropriate errors
    - Missing required metadata fields are detected
    """
    with patch('osgeo.gdal.Open') as mock_gdal_open:
        
        # Test invalid file path - GDAL returns None
        mock_gdal_open.return_value = None
        invalid_path = "/nonexistent/path/file.tif"
        
        with pytest.raises(ValueError, match="Could not open scene file"):
            get_landsat_metadata(invalid_path)
        
        mock_gdal_open.assert_called_once_with(invalid_path)
        
        # Test corrupted/incomplete metadata - dataset opens but has minimal data
        mock_gdal_open.reset_mock()
        mock_dataset = Mock()
        mock_gdal_open.return_value = mock_dataset
        
        # Configure dataset with minimal valid data
        mock_dataset.RasterXSize = 1000
        mock_dataset.RasterYSize = 1000
        mock_dataset.RasterCount = 1
        mock_dataset.GetProjection.return_value = ""  # Empty projection
        mock_dataset.GetGeoTransform.return_value = (0, 1, 0, 0, 0, -1)  # Default geotransform
        mock_dataset.GetMetadata.return_value = {}  # No GDAL metadata
        
        corrupt_path = "/tmp/corrupt_landsat.tif"
        result = get_landsat_metadata(corrupt_path)
        
        # Verify function handles missing metadata gracefully
        assert result["width"] == 1000
        assert result["height"] == 1000
        assert result["bands"] == 1
        assert result["projection"] == ""
        
        # Standard Landsat scaling factors should still be present
        assert result["scale_factor"] == 0.0000275
        assert result["add_offset"] == -0.2
        
        # GDAL metadata should not be included when empty (falsy check in implementation)
        assert "gdal_metadata" not in result
        
        # Test file with invalid filename format
        mock_gdal_open.reset_mock()
        mock_dataset.reset_mock()
        
        invalid_filename_path = "/tmp/invalid_name_format.tif"
        result = get_landsat_metadata(invalid_filename_path)
        
        # Should not have acquisition_date since filename doesn't match pattern
        assert "acquisition_date" not in result
        
        # But other standard fields should be present
        assert "width" in result
        assert "height" in result
        assert "scale_factor" in result
        
        # Test file with short filename (edge case)
        mock_gdal_open.reset_mock()
        mock_dataset.reset_mock()
        
        short_filename_path = "/tmp/short.tif"
        result = get_landsat_metadata(short_filename_path)
        
        # Should handle gracefully without acquisition_date
        assert "acquisition_date" not in result
        assert "width" in result
        
        # Test dataset that raises exception on metadata access
        mock_gdal_open.reset_mock()
        mock_dataset.reset_mock()
        mock_dataset.GetMetadata.side_effect = Exception("Metadata access failed")
        
        exception_path = "/tmp/metadata_exception.tif"
        
        # The current implementation doesn't handle GetMetadata exceptions
        # So we expect the exception to propagate
        with pytest.raises(Exception, match="Metadata access failed"):
            get_landsat_metadata(exception_path)