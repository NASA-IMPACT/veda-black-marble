"""Pytest configuration and shared fixtures for Black Marble tests."""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def test_shape():
    """Default shape for test arrays."""
    return (100, 100)


@pytest.fixture
def manhattan_bbox():
    """Small Manhattan area bounding box for testing."""
    return (-74.05, 40.69, -73.95, 40.79)


@pytest.fixture
def sf_bbox():
    """Small San Francisco area bounding box for testing."""
    return (-122.42, 37.77, -122.40, 37.79)


@pytest.fixture
def test_date():
    """Default test date."""
    return datetime(2023, 9, 15)


@pytest.fixture
def temp_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_s3_adapter():
    """Mock S3 adapter for testing."""
    from unittest.mock import Mock

    adapter = Mock()
    adapter.anonymous = False
    adapter.upload_file.return_value = True
    adapter.download_file.return_value = True

    return adapter


@pytest.fixture
def mock_raster_adapter():
    """Mock raster adapter for testing."""
    from unittest.mock import Mock

    adapter = Mock()
    adapter.use_rasterio = True

    # Mock read operation
    def mock_read(filepath, band=None):
        # Return random data of appropriate shape
        return np.random.rand(100, 100).astype(np.float32)

    adapter.read = mock_read

    # Mock write operation
    adapter.write.return_value = True

    return adapter


@pytest.fixture
def pipeline_config(manhattan_bbox, test_date, temp_dir):
    """Default pipeline configuration for testing."""
    from blackmarble.config import PipelineConfig

    output_path = str(temp_dir / "test_output.tif")
    return PipelineConfig.for_bm_parity(manhattan_bbox, test_date, output_path)


@pytest.fixture
def enhanced_pipeline_config(manhattan_bbox, test_date, temp_dir):
    """Enhanced pipeline configuration for testing."""
    from blackmarble.config import PipelineConfig

    output_path = str(temp_dir / "test_enhanced.tif")
    return PipelineConfig.for_enhanced_quality(manhattan_bbox, test_date, output_path)


@pytest.fixture
def mock_cog_generator():
    """Mock COG generator for testing."""
    from unittest.mock import MagicMock, Mock

    # Create a mock COGProfile
    COGProfile = Mock()
    COGProfile.profile_name = "deflate"

    generator = Mock()
    generator.profile = COGProfile

    # Mock process method
    def mock_process(data, output_path, **kwargs):
        result = MagicMock()
        result.output_path = output_path
        result.success = True
        result.metadata = {
            "bands": data.shape[0] if data.ndim == 3 else 1,
            "dtype": str(data.dtype),
            "shape": data.shape,
        }
        return result

    generator.process = mock_process

    return generator


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "requires_gdal: mark test as requiring GDAL")
