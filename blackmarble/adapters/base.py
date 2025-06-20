"""Base adapter interfaces for I/O operations.

These Protocol classes define the interfaces that all adapters must implement,
enabling dependency injection and easy mocking for tests.
"""

from typing import Protocol, Dict, List, Tuple, Optional, Any, BinaryIO
from pathlib import Path
import numpy as np
from datetime import datetime


class StorageAdapter(Protocol):
    """Interface for cloud storage operations (S3, GCS, etc.)."""

    def download(
        self, bucket: str, key: str, local_path: Optional[Path] = None
    ) -> bytes:
        """Download object from storage.

        Args:
            bucket: Storage bucket name
            key: Object key/path
            local_path: Optional local path to save file

        Returns:
            Object contents as bytes

        Raises:
            StorageError: If download fails
        """
        ...

    def upload(self, bucket: str, key: str, data: bytes) -> str:
        """Upload data to storage.

        Args:
            bucket: Storage bucket name
            key: Object key/path
            data: Data to upload

        Returns:
            Object URL or identifier

        Raises:
            StorageError: If upload fails
        """
        ...

    def exists(self, bucket: str, key: str) -> bool:
        """Check if object exists in storage.

        Args:
            bucket: Storage bucket name
            key: Object key/path

        Returns:
            True if object exists
        """
        ...

    def list_objects(self, bucket: str, prefix: str) -> List[str]:
        """List objects with given prefix.

        Args:
            bucket: Storage bucket name
            prefix: Key prefix to filter objects

        Returns:
            List of object keys
        """
        ...


class APIAdapter(Protocol):
    """Interface for external API operations."""

    def search(self, **params) -> List[Dict[str, Any]]:
        """Search for resources matching parameters.

        Args:
            **params: Search parameters (API-specific)

        Returns:
            List of matching resources

        Raises:
            APIError: If search fails
        """
        ...

    def download(self, resource_id: str, output_path: Optional[Path] = None) -> bytes:
        """Download resource by ID.

        Args:
            resource_id: Resource identifier
            output_path: Optional local path to save

        Returns:
            Resource data as bytes

        Raises:
            APIError: If download fails
        """
        ...

    def authenticate(self, credentials: Dict[str, str]) -> None:
        """Authenticate with the API.

        Args:
            credentials: Authentication credentials

        Raises:
            AuthenticationError: If authentication fails
        """
        ...


class RasterAdapter(Protocol):
    """Interface for raster I/O operations."""

    def read(
        self, path: Path, band: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Read raster data and metadata.

        Args:
            path: Path to raster file
            band: Optional band number (1-indexed)

        Returns:
            Tuple of (data array, metadata dict)

        Raises:
            RasterIOError: If read fails
        """
        ...

    def write(
        self,
        path: Path,
        data: np.ndarray,
        metadata: Dict[str, Any],
        compress: bool = True,
    ) -> None:
        """Write raster data to file.

        Args:
            path: Output file path
            data: Array data to write
            metadata: Raster metadata (CRS, transform, etc.)
            compress: Whether to compress output

        Raises:
            RasterIOError: If write fails
        """
        ...

    def get_info(self, path: Path) -> Dict[str, Any]:
        """Get raster information without reading data.

        Args:
            path: Path to raster file

        Returns:
            Dictionary with raster info (shape, CRS, bands, etc.)

        Raises:
            RasterIOError: If file cannot be accessed
        """
        ...

    def reproject(
        self,
        source_path: Path,
        target_crs: str,
        output_path: Optional[Path] = None,
        resolution: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reproject raster to different CRS.

        Args:
            source_path: Input raster path
            target_crs: Target coordinate reference system
            output_path: Optional output path
            resolution: Optional output resolution

        Returns:
            Tuple of (reprojected data, updated metadata)

        Raises:
            RasterIOError: If reprojection fails
        """
        ...


class TileSystemAdapter(Protocol):
    """Interface for tile system operations (WRS-2, MODIS, etc.)."""

    def get_tiles_for_bbox(
        self, west: float, south: float, east: float, north: float
    ) -> List[str]:
        """Get tile identifiers covering bounding box.

        Args:
            west: Western longitude
            south: Southern latitude
            east: Eastern longitude
            north: Northern latitude

        Returns:
            List of tile identifiers
        """
        ...

    def get_tile_bounds(self, tile_id: str) -> Tuple[float, float, float, float]:
        """Get geographic bounds of a tile.

        Args:
            tile_id: Tile identifier

        Returns:
            Tuple of (west, south, east, north)
        """
        ...


# Custom exceptions for adapters
class StorageError(Exception):
    """Raised when storage operations fail."""

    pass


class APIError(Exception):
    """Raised when API operations fail."""

    pass


class AuthenticationError(APIError):
    """Raised when API authentication fails."""

    pass


class RasterIOError(Exception):
    """Raised when raster I/O operations fail."""

    pass
