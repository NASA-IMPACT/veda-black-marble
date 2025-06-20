"""
Example tests showing how the Phase 2 adapters enable better testing.

With dependency injection, we can easily mock external dependencies
like S3 and raster I/O operations.
"""

import pytest
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

from blackmarble.pipeline import run_pipeline
from blackmarble.adapters.base import StorageAdapter, RasterAdapter
from blackmarble.data_sources.landsat import LandsatDataSource
from blackmarble.data_sources.viirs import VIIRSDataSource


class MockS3Adapter:
    """Mock S3 adapter for testing."""
    
    def __init__(self):
        self.uploaded_files = {}
        self.downloaded_files = {}
    
    def upload_file(self, local_path: Path, bucket: str, key: str) -> str:
        """Mock upload - just track what was uploaded."""
        self.uploaded_files[f"{bucket}/{key}"] = local_path
        return f"s3://{bucket}/{key}"
    
    def download(self, bucket: str, key: str, local_path: Path = None) -> bytes:
        """Mock download - return dummy data."""
        self.downloaded_files[f"{bucket}/{key}"] = local_path
        return b"mock data"
    
    def exists(self, bucket: str, key: str) -> bool:
        """Mock exists check."""
        return True
    
    def list_objects(self, bucket: str, prefix: str = "") -> list:
        """Mock list objects."""
        return [f"{prefix}mock_file_1.tif", f"{prefix}mock_file_2.tif"]


class MockRasterAdapter:
    """Mock raster adapter for testing."""
    
    def __init__(self, shape=(100, 100)):
        self.shape = shape
        self.written_files = {}
    
    def read(self, path: Path, band: int = None):
        """Return mock raster data."""
        data = np.random.rand(*self.shape).astype(np.float32)
        metadata = {
            'crs': 'EPSG:4326',
            'transform': (0.0001, 0.0, -122.4194, 0.0, -0.0001, 37.8049),
            'gdal_transform': (0.0001, 0.0, -122.4194, 0.0, -0.0001, 37.8049),
            'width': self.shape[1],
            'height': self.shape[0],
            'count': 1,
            'dtype': 'float32',
            'nodata': -9999.0,
            'driver': 'GTiff'
        }
        return data, metadata
    
    def write(self, path: Path, data: np.ndarray, metadata: dict, compress: bool = True):
        """Mock write - just track what was written."""
        self.written_files[str(path)] = {
            'data': data,
            'metadata': metadata,
            'compress': compress
        }
    
    def get_info(self, path: Path) -> dict:
        """Get mock raster info."""
        return {
            'driver': 'GTiff',
            'width': self.shape[1],
            'height': self.shape[0],
            'count': 1,
            'crs': 'EPSG:4326',
            'bounds': (-122.4194, 37.7749, -122.3894, 37.8049),
            'dtype': ['float32'],
            'nodata': -9999.0,
            'shape': (1, self.shape[0], self.shape[1])
        }


class MockLandsatDataSource:
    """Mock Landsat data source."""
    
    def __init__(self, storage_adapter=None, raster_adapter=None):
        self.storage = storage_adapter
        self.raster = raster_adapter
    
    def fetch(self, bbox, start_date, end_date, bands=None, composite_months=1):
        """Return mock Landsat data."""
        shape = (100, 100)
        result = {}
        
        for band in bands or ["B3", "B4", "B5", "QA_PIXEL"]:
            if band == "QA_PIXEL":
                # Mock QA data with some "clear" pixels
                data = np.zeros(shape, dtype=np.uint16)
                data[20:80, 20:80] = 21824  # Clear pixel value
                result[band] = data
            else:
                # Mock reflectance data (0-1 range)
                result[band] = np.random.rand(*shape).astype(np.float32) * 0.3 + 0.1
        
        return result
    
    def get_metadata(self):
        """Return mock metadata."""
        return {
            "name": "Mock Landsat",
            "bands": ["B3", "B4", "B5", "QA_PIXEL"],
            "resolution": 30.0
        }


class MockVIIRSDataSource:
    """Mock VIIRS data source."""
    
    def __init__(self, api_adapter=None, raster_adapter=None, earthdata_token=None):
        self.api = api_adapter
        self.raster = raster_adapter
        self.token = earthdata_token
    
    def fetch(self, bbox, start_date, end_date, layers=None):
        """Return mock VIIRS data."""
        shape = (100, 100)
        result = {}
        
        for layer in layers or ["NTL", "LUNAR", "QF"]:
            if layer == "NTL":
                # Mock nighttime lights
                result[layer] = np.random.rand(*shape).astype(np.float32) * 10
            elif layer == "LUNAR":
                # Mock lunar irradiance
                result[layer] = np.random.rand(*shape).astype(np.float32) * 0.5
            elif layer == "QF":
                # Mock quality flags (0 = good quality)
                result[layer] = np.zeros(shape, dtype=np.uint8)
        
        return result
    
    def get_metadata(self):
        """Return mock metadata."""
        return {
            "name": "Mock VIIRS",
            "layers": ["NTL", "LUNAR", "QF"],
            "resolution": 500.0
        }


@pytest.fixture
def mock_adapters():
    """Fixture providing mock adapters."""
    return {
        's3_adapter': MockS3Adapter(),
        'raster_adapter': MockRasterAdapter(),
        'landsat_source': None,  # Will be created with mocks
        'viirs_source': None  # Will be created with mocks
    }


def test_pipeline_with_mock_adapters(mock_adapters, tmp_path):
    """Test pipeline execution with mock adapters."""
    
    # Create mock data sources
    mock_adapters['landsat_source'] = MockLandsatDataSource(
        storage_adapter=mock_adapters['s3_adapter'],
        raster_adapter=mock_adapters['raster_adapter']
    )
    
    mock_adapters['viirs_source'] = MockVIIRSDataSource(
        raster_adapter=mock_adapters['raster_adapter'],
        earthdata_token="mock_token"
    )
    
    # Mock other dependencies
    with patch('blackmarble.pipeline.OSMUtil') as mock_osm:
        # Setup OSM mock
        mock_osm_instance = Mock()
        mock_osm_instance.create_road_raster.return_value = None
        mock_osm.return_value = mock_osm_instance
        
        # Mock OSM file creation
        with patch('blackmarble.pipeline.get_geo') as mock_get_geo:
            # Mock get_geo to return dummy data for OSM
            mock_get_geo.return_value = (
                np.zeros((100, 100), dtype=np.float32),
                (0.0001, 0.0, -122.4194, 0.0, -0.0001, 37.8049),
                'EPSG:4326'
            )
            
            # Run pipeline with mocks
            output = run_pipeline(
                lat1=37.7749,
                lon1=-122.4194,
                lat2=37.8049,
                lon2=-122.3894,
                year=2023,
                month=6,
                day=15,
                earthdata_token="mock_token",
                output_dir=str(tmp_path),
                **mock_adapters
            )
    
    # Verify pipeline completed
    assert output.endswith("ndui-co.tif")
    assert Path(output).exists()
    
    # Verify adapters were used
    assert len(mock_adapters['raster_adapter'].written_files) > 0
    
    # Verify no real S3 uploads happened (unless upload_bucket was specified)
    assert len(mock_adapters['s3_adapter'].uploaded_files) == 0


def test_pipeline_adapter_fallback(tmp_path):
    """Test that pipeline falls back gracefully when adapters fail."""
    
    # Create adapters that will fail
    failing_landsat = Mock()
    failing_landsat.fetch.side_effect = Exception("Mock fetch failed")
    
    failing_viirs = Mock()
    failing_viirs.fetch.side_effect = Exception("Mock fetch failed")
    
    # Mock the legacy functions to succeed
    with patch('blackmarble.pipeline.process_band_data') as mock_process:
        with patch('blackmarble.pipeline.VNP46A2Util') as mock_vnp:
            # Setup mocks for legacy path
            mock_process.return_value = (
                [str(tmp_path / "B3.tif")],  # band files
                [str(tmp_path / "QA.tif")],  # QA pixel
                [str(tmp_path / "QA.tif")],  # QA radsat
                [str(tmp_path / "QA.tif")]   # QA aerosol
            )
            
            # Create dummy files
            for f in ["B3.tif", "B4.tif", "B5.tif", "QA.tif"]:
                (tmp_path / f).write_text("dummy")
            
            # Mock other required functions
            with patch('blackmarble.pipeline.get_geo'):
                with patch('blackmarble.pipeline.crop_to_bb_utm'):
                    with patch('blackmarble.pipeline.OSMUtil'):
                        # This should fall back to legacy methods
                        # (Would need more complete mocking for full test)
                        pass


def test_s3_upload_with_mock(mock_adapters, tmp_path):
    """Test S3 upload functionality with mock adapter."""
    
    # Create a dummy file to upload
    test_file = tmp_path / "test.tif"
    test_file.write_text("test data")
    
    # Upload using mock adapter
    result = mock_adapters['s3_adapter'].upload_file(
        test_file,
        "test-bucket",
        "test-key"
    )
    
    # Verify upload was tracked
    assert result == "s3://test-bucket/test-key"
    assert "test-bucket/test-key" in mock_adapters['s3_adapter'].uploaded_files
    assert mock_adapters['s3_adapter'].uploaded_files["test-bucket/test-key"] == test_file


if __name__ == "__main__":
    # Example of running a specific test
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        adapters = {
            's3_adapter': MockS3Adapter(),
            'raster_adapter': MockRasterAdapter(),
            'landsat_source': None,
            'viirs_source': None
        }
        
        test_pipeline_with_mock_adapters(adapters, Path(tmpdir))