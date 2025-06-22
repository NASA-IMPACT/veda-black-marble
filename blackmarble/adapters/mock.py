"""Mock adapters for testing."""

import logging
from typing import Any
from pathlib import Path
import numpy as np

from .base import (
    StorageAdapter,
    APIAdapter,
    RasterAdapter,
    StorageError,
    APIError,
    RasterIOError,
)

logger = logging.getLogger(__name__)


class MockS3Adapter(StorageAdapter):
    """Mock S3 adapter for testing.

    Stores objects in memory and tracks all operations.
    """

    def __init__(self):
        """Initialize mock S3 adapter."""
        self.storage = {}  # {bucket: {key: data}}
        self.calls = []  # Track all method calls
        self.fail_on_next_call = False  # For testing error handling

    def download(self, bucket: str, key: str, local_path: Path | None = None) -> bytes:
        """Mock download from S3."""
        self.calls.append(("download", bucket, key, local_path))

        if self.fail_on_next_call:
            self.fail_on_next_call = False
            raise StorageError("Mock download failure")

        if bucket not in self.storage or key not in self.storage[bucket]:
            raise StorageError(f"Object not found: s3://{bucket}/{key}")

        data = self.storage[bucket][key]

        if local_path:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(data)

        return data

    def upload(self, bucket: str, key: str, data: bytes) -> str:
        """Mock upload to S3."""
        self.calls.append(("upload", bucket, key, len(data)))

        if self.fail_on_next_call:
            self.fail_on_next_call = False
            raise StorageError("Mock upload failure")

        if bucket not in self.storage:
            self.storage[bucket] = {}

        self.storage[bucket][key] = data
        return f"s3://{bucket}/{key}"

    def exists(self, bucket: str, key: str) -> bool:
        """Check if mock object exists."""
        self.calls.append(("exists", bucket, key))
        return bucket in self.storage and key in self.storage.get(bucket, {})

    def list_objects(self, bucket: str, prefix: str = "") -> list[str]:
        """List mock objects with prefix."""
        self.calls.append(("list_objects", bucket, prefix))

        if bucket not in self.storage:
            return []

        keys = []
        for key in self.storage[bucket]:
            if key.startswith(prefix):
                keys.append(key)

        return sorted(keys)

    def add_mock_object(self, bucket: str, key: str, data: bytes) -> None:
        """Add object to mock storage."""
        if bucket not in self.storage:
            self.storage[bucket] = {}
        self.storage[bucket][key] = data

    def clear(self) -> None:
        """Clear all mock data and call history."""
        self.storage.clear()
        self.calls.clear()
        self.fail_on_next_call = False


class MockNASAAPIAdapter(APIAdapter):
    """Mock NASA Earthdata API adapter for testing."""

    def __init__(self):
        """Initialize mock API adapter."""
        self.mock_granules = []
        self.calls = []
        self.authenticated = False
        self.fail_on_next_call = False

    def search(self, **params) -> list[dict[str, Any]]:
        """Mock granule search."""
        self.calls.append(("search", params))

        if self.fail_on_next_call:
            self.fail_on_next_call = False
            raise APIError("Mock search failure")

        # Filter mock granules based on parameters
        results = []
        for granule in self.mock_granules:
            # Simple filtering by collection name
            if "short_name" in params:
                if granule.get("collection_short_name") != params["short_name"]:
                    continue

            results.append(granule)

        return results

    def download(self, resource_id: str, output_path: Path | None = None) -> bytes:
        """Mock resource download."""
        self.calls.append(("download", resource_id, output_path))

        if self.fail_on_next_call:
            self.fail_on_next_call = False
            raise APIError("Mock download failure")

        if not self.authenticated:
            raise APIError("Not authenticated")

        # Return mock data based on resource_id
        data = f"Mock data for {resource_id}".encode()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(data)

        return data

    def authenticate(self, credentials: dict[str, str]) -> None:
        """Mock authentication."""
        self.calls.append(("authenticate", credentials))

        if "token" in credentials and credentials["token"] == "valid_token":
            self.authenticated = True
        else:
            raise APIError("Invalid credentials")

    def add_mock_granule(self, granule: dict[str, Any]) -> None:
        """Add granule to mock results."""
        self.mock_granules.append(granule)

    def clear(self) -> None:
        """Clear mock data and call history."""
        self.mock_granules.clear()
        self.calls.clear()
        self.authenticated = False
        self.fail_on_next_call = False


class MockRasterAdapter(RasterAdapter):
    """Mock raster adapter for testing."""

    def __init__(self):
        """Initialize mock raster adapter."""
        self.mock_files = {}  # {path: (data, metadata)}
        self.calls = []
        self.fail_on_next_call = False

    def read(
        self, path: Path, band: int | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Mock raster read."""
        self.calls.append(("read", str(path), band))

        if self.fail_on_next_call:
            self.fail_on_next_call = False
            raise RasterIOError("Mock read failure")

        if str(path) not in self.mock_files:
            raise RasterIOError(f"File not found: {path}")

        data, metadata = self.mock_files[str(path)]

        if band is not None:
            if data.ndim == 3 and 1 <= band <= data.shape[0]:
                return data[band - 1], metadata
            elif data.ndim == 2 and band == 1:
                return data, metadata
            else:
                raise RasterIOError(f"Invalid band {band}")

        return data, metadata

    def write(
        self,
        path: Path,
        data: np.ndarray,
        metadata: dict[str, Any],
        compress: bool = True,
    ) -> None:
        """Mock raster write."""
        self.calls.append(("write", str(path), data.shape, compress))

        if self.fail_on_next_call:
            self.fail_on_next_call = False
            raise RasterIOError("Mock write failure")

        self.mock_files[str(path)] = (data.copy(), metadata.copy())

    def get_info(self, path: Path) -> dict[str, Any]:
        """Get mock raster info."""
        self.calls.append(("get_info", str(path)))

        if str(path) not in self.mock_files:
            raise RasterIOError(f"File not found: {path}")

        data, metadata = self.mock_files[str(path)]

        info = metadata.copy()
        info.update(
            {"shape": data.shape, "dtype": str(data.dtype), "size_bytes": data.nbytes}
        )

        return info

    def reproject(
        self,
        source_path: Path,
        target_crs: str,
        output_path: Path | None = None,
        resolution: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Mock reprojection."""
        self.calls.append(
            ("reproject", str(source_path), target_crs, output_path, resolution)
        )

        if self.fail_on_next_call:
            self.fail_on_next_call = False
            raise RasterIOError("Mock reproject failure")

        if str(source_path) not in self.mock_files:
            raise RasterIOError(f"File not found: {source_path}")

        # Return same data with updated CRS (mock behavior)
        data, metadata = self.mock_files[str(source_path)]
        new_metadata = metadata.copy()
        new_metadata["crs"] = target_crs

        if output_path:
            self.mock_files[str(output_path)] = (data.copy(), new_metadata)

        return data.copy(), new_metadata

    def add_mock_file(
        self, path: str, data: np.ndarray, metadata: dict[str, Any]
    ) -> None:
        """Add file to mock storage."""
        self.mock_files[path] = (data.copy(), metadata.copy())

    def clear(self) -> None:
        """Clear mock data and call history."""
        self.mock_files.clear()
        self.calls.clear()
        self.fail_on_next_call = False
