"""Tests for I/O adapters."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from blackmarble.adapters.mock import (
    MockS3Adapter,
    MockNASAAPIAdapter,
    MockRasterAdapter,
)
from blackmarble.adapters.base import StorageError, APIError, RasterIOError


class TestMockS3Adapter:
    """Test mock S3 adapter."""

    def test_upload_download(self):
        """Test basic upload and download."""
        adapter = MockS3Adapter()

        # Upload data
        bucket = "test-bucket"
        key = "test-file.txt"
        data = b"Hello, World!"

        uri = adapter.upload(bucket, key, data)
        assert uri == f"s3://{bucket}/{key}"

        # Download data
        downloaded = adapter.download(bucket, key)
        assert downloaded == data

        # Check call history
        assert len(adapter.calls) == 2
        assert adapter.calls[0][0] == "upload"
        assert adapter.calls[1][0] == "download"

    def test_download_to_file(self):
        """Test download to local file."""
        adapter = MockS3Adapter()
        adapter.add_mock_object("bucket", "key", b"test data")

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "downloaded.txt"
            data = adapter.download("bucket", "key", local_path)

            assert local_path.exists()
            assert local_path.read_bytes() == b"test data"
            assert data == b"test data"

    def test_exists(self):
        """Test existence check."""
        adapter = MockS3Adapter()

        assert not adapter.exists("bucket", "key")

        adapter.add_mock_object("bucket", "key", b"data")
        assert adapter.exists("bucket", "key")
        assert not adapter.exists("bucket", "other-key")
        assert not adapter.exists("other-bucket", "key")

    def test_list_objects(self):
        """Test object listing."""
        adapter = MockS3Adapter()

        # Add some objects
        adapter.add_mock_object("bucket", "dir/file1.txt", b"data1")
        adapter.add_mock_object("bucket", "dir/file2.txt", b"data2")
        adapter.add_mock_object("bucket", "other/file3.txt", b"data3")

        # List with prefix
        objects = adapter.list_objects("bucket", "dir/")
        assert len(objects) == 2
        assert "dir/file1.txt" in objects
        assert "dir/file2.txt" in objects

        # List all
        objects = adapter.list_objects("bucket", "")
        assert len(objects) == 3

    def test_error_handling(self):
        """Test error conditions."""
        adapter = MockS3Adapter()

        # Download non-existent object
        with pytest.raises(StorageError, match="Object not found"):
            adapter.download("bucket", "missing")

        # Test failure mode
        adapter.fail_on_next_call = True
        with pytest.raises(StorageError, match="Mock download failure"):
            adapter.download("bucket", "key")


class TestMockNASAAPIAdapter:
    """Test mock NASA API adapter."""

    def test_authentication(self):
        """Test authentication."""
        adapter = MockNASAAPIAdapter()

        # Invalid credentials
        with pytest.raises(APIError):
            adapter.authenticate({"token": "invalid"})

        # Valid credentials
        adapter.authenticate({"token": "valid_token"})
        assert adapter.authenticated

    def test_search(self):
        """Test granule search."""
        adapter = MockNASAAPIAdapter()

        # Add mock granules
        adapter.add_mock_granule({"collection_short_name": "VNP46A2", "id": "granule1"})
        adapter.add_mock_granule({"collection_short_name": "VNP46A2", "id": "granule2"})
        adapter.add_mock_granule({"collection_short_name": "OTHER", "id": "granule3"})

        # Search by collection
        results = adapter.search(short_name="VNP46A2")
        assert len(results) == 2
        assert results[0]["id"] == "granule1"
        assert results[1]["id"] == "granule2"

        # Search all
        results = adapter.search()
        assert len(results) == 3

    def test_download(self):
        """Test resource download."""
        adapter = MockNASAAPIAdapter()

        # Should fail without authentication
        with pytest.raises(APIError, match="Not authenticated"):
            adapter.download("resource1")

        # Authenticate and download
        adapter.authenticate({"token": "valid_token"})
        data = adapter.download("resource1")
        assert b"Mock data for resource1" in data

        # Download to file
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "download.dat"
            data = adapter.download("resource2", output_path)
            assert output_path.exists()
            assert b"Mock data for resource2" in output_path.read_bytes()

    def test_error_handling(self):
        """Test error conditions."""
        adapter = MockNASAAPIAdapter()

        # Test failure mode
        adapter.fail_on_next_call = True
        with pytest.raises(APIError, match="Mock search failure"):
            adapter.search()


class TestMockRasterAdapter:
    """Test mock raster adapter."""

    def setup_method(self):
        """Set up test data."""
        self.adapter = MockRasterAdapter()

        # Create test raster
        self.test_data = np.random.rand(100, 100).astype(np.float32)
        self.test_metadata = {
            "crs": "EPSG:4326",
            "transform": [0.1, 0, -180, 0, -0.1, 90],
            "width": 100,
            "height": 100,
            "count": 1,
            "dtype": "float32",
            "nodata": -9999,
        }

        self.adapter.add_mock_file(
            "/test/raster.tif", self.test_data, self.test_metadata
        )

    def test_read(self):
        """Test raster read."""
        # Read full raster
        data, metadata = self.adapter.read(Path("/test/raster.tif"))
        assert data.shape == (100, 100)
        assert metadata["crs"] == "EPSG:4326"

        # Read specific band
        data, metadata = self.adapter.read(Path("/test/raster.tif"), band=1)
        assert data.shape == (100, 100)

        # Check call history
        assert len(self.adapter.calls) == 2
        assert self.adapter.calls[0][0] == "read"

    def test_write(self):
        """Test raster write."""
        new_data = np.ones((50, 50), dtype=np.float32)
        new_metadata = {"crs": "EPSG:3857", "width": 50, "height": 50}

        self.adapter.write(
            Path("/test/new_raster.tif"), new_data, new_metadata, compress=True
        )

        # Verify it was written
        read_data, read_metadata = self.adapter.read(Path("/test/new_raster.tif"))
        assert read_data.shape == (50, 50)
        assert read_metadata["crs"] == "EPSG:3857"
        assert np.all(read_data == 1.0)

    def test_get_info(self):
        """Test getting raster info."""
        info = self.adapter.get_info(Path("/test/raster.tif"))

        assert info["shape"] == (100, 100)
        assert info["dtype"] == "float32"
        assert info["crs"] == "EPSG:4326"
        assert "size_bytes" in info

    def test_reproject(self):
        """Test reprojection."""
        # Reproject to memory
        data, metadata = self.adapter.reproject(Path("/test/raster.tif"), "EPSG:3857")

        assert data.shape == self.test_data.shape
        assert metadata["crs"] == "EPSG:3857"

        # Reproject to file
        output_path = Path("/test/reprojected.tif")
        data, metadata = self.adapter.reproject(
            Path("/test/raster.tif"), "EPSG:3857", output_path=output_path
        )

        # Verify output file exists in mock
        assert self.adapter.get_info(output_path)["crs"] == "EPSG:3857"

    def test_multiband(self):
        """Test multiband raster handling."""
        # Add multiband raster
        multiband_data = np.random.rand(3, 50, 50).astype(np.float32)
        multiband_metadata = self.test_metadata.copy()
        multiband_metadata["count"] = 3

        self.adapter.add_mock_file(
            "/test/multiband.tif", multiband_data, multiband_metadata
        )

        # Read all bands
        data, metadata = self.adapter.read(Path("/test/multiband.tif"))
        assert data.shape == (3, 50, 50)

        # Read specific band
        data, metadata = self.adapter.read(Path("/test/multiband.tif"), band=2)
        assert data.shape == (50, 50)
        assert np.array_equal(data, multiband_data[1])

    def test_error_handling(self):
        """Test error conditions."""
        # Read non-existent file
        with pytest.raises(RasterIOError, match="File not found"):
            self.adapter.read(Path("/missing/file.tif"))

        # Invalid band
        with pytest.raises(RasterIOError, match="Invalid band"):
            self.adapter.read(Path("/test/raster.tif"), band=2)

        # Test failure mode
        self.adapter.fail_on_next_call = True
        with pytest.raises(RasterIOError, match="Mock read failure"):
            self.adapter.read(Path("/test/raster.tif"))
