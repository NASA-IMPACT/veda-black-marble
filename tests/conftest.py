"""Pytest configuration and shared fixtures for Black Marble tests."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock

import numpy as np
import numpy.typing as npt
import pytest

from blackmarble.config import PipelineConfig


@pytest.fixture
def test_shape() -> tuple[int, int]:
    """Default shape for test arrays."""
    return (100, 100)


@pytest.fixture
def manhattan_bbox() -> tuple[float, float, float, float]:
    """Small Manhattan area bounding box for testing."""
    return (-74.05, 40.69, -73.95, 40.79)


@pytest.fixture
def sf_bbox() -> tuple[float, float, float, float]:
    """Small San Francisco area bounding box for testing."""
    return (-122.42, 37.77, -122.40, 37.79)


@pytest.fixture
def test_date() -> datetime:
    """Default test date."""
    return datetime(2023, 9, 15)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_s3_adapter() -> Mock:
    """Mock S3 adapter for testing."""

    adapter = Mock()
    adapter.anonymous = False
    adapter.upload_file.return_value = True
    adapter.download_file.return_value = True

    return adapter


@pytest.fixture
def mock_raster_adapter() -> Mock:
    """Mock raster adapter for testing."""

    adapter = Mock()
    adapter.use_rasterio = True

    # Mock read operation
    def mock_read(filepath: str, band: int | None = None) -> npt.NDArray[np.float32]:
        # Return random data of appropriate shape
        return np.random.rand(100, 100).astype(np.float32)

    adapter.read = mock_read

    # Mock write operation
    adapter.write.return_value = True

    return adapter


@pytest.fixture
def pipeline_config(
    manhattan_bbox: tuple[float, float, float, float],
    test_date: datetime,
    temp_dir: Path,
):
    """Default pipeline configuration for testing."""

    output_path = str(temp_dir / "test_output.tif")
    return PipelineConfig.for_bm_parity(manhattan_bbox, test_date, output_path)


@pytest.fixture
def enhanced_pipeline_config(
    manhattan_bbox: tuple[float, float, float, float],
    test_date: datetime,
    temp_dir: Path,
):
    """Enhanced pipeline configuration for testing."""
    output_path = str(temp_dir / "test_enhanced.tif")
    return PipelineConfig.for_enhanced_quality(manhattan_bbox, test_date, output_path)


@pytest.fixture
def mock_cog_generator() -> Mock:
    """Mock COG generator for testing."""
    # Create a mock COGProfile
    COGProfile = Mock()
    COGProfile.profile_name = "deflate"

    generator = Mock()
    generator.profile = COGProfile

    # Mock process method
    def mock_process(data: npt.NDArray[Any], output_path: str, **kwargs: Any) -> MagicMock:
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


@pytest.fixture(scope="session")
def reference_data_paths() -> dict[str, Path]:
    """Verify and return paths to reference data files.

    This fixture runs once per test session and ensures all reference
    data is present and valid before tests run.
    """

    # Get reference data directory
    test_dir = Path(__file__).parent
    reference_dir = test_dir / "data" / "reference"

    # Run verification
    result = subprocess.run(
        [sys.executable, str(test_dir.parent / "tools" / "fetch_reference_data.py")],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.fail(
            "Reference data verification failed!\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}\n"
            "Run 'python tools/fetch_reference_data.py --force' to regenerate"
        )

    # Return paths
    return {
        "viirs": reference_dir / "viirs" / "VNP46A2_h08v05_20231015.tif",
        "osm": reference_dir / "osm" / "test_roads.gpkg",
    }


# Pytest configuration
def pytest_configure(config: Any) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
