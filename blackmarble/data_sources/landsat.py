"""Landsat data source implementation."""

import logging
from typing import Any
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import tempfile
import shutil

from ..adapters.base import StorageAdapter, RasterAdapter
from ..adapters.s3 import S3Adapter
from ..adapters.raster_io import GDALRasterAdapter
from .base import DataSource, DataSourceError, DataNotAvailableError

logger = logging.getLogger(__name__)


class LandsatDataSource:
    """Landsat data source using S3 and raster adapters.

    Fetches Landsat Collection 2 Level-2 data from AWS S3.
    """

    # USGS Landsat S3 bucket
    LANDSAT_BUCKET = "usgs-landsat"

    # Band mapping
    BAND_MAPPING = {
        "B3": "SR_B3",  # Green
        "B4": "SR_B4",  # Red
        "B5": "SR_B5",  # NIR
        "QA_PIXEL": "QA_PIXEL",  # Quality assessment
    }

    def __init__(
        self,
        storage_adapter: StorageAdapter | None = None,
        raster_adapter: RasterAdapter | None = None,
        collection: str = "02",
        level: str = "L2",
        sensor: str = "LC09",
    ):
        """Initialize Landsat data source.

        Args:
            storage_adapter: Storage adapter for S3 access
            raster_adapter: Raster adapter for reading files
            collection: Landsat collection (01 or 02)
            level: Processing level (L1 or L2)
            sensor: Sensor ID (LC08, LC09, etc.)
        """
        self.storage = storage_adapter or S3Adapter(anonymous=True)
        self.raster = raster_adapter or GDALRasterAdapter()
        self.collection = collection
        self.level = level
        self.sensor = sensor

        self._wrs2_tiles = None  # Cache for WRS-2 tile lookup

    def fetch(
        self,
        bbox: tuple[float, float, float, float],
        start_date: datetime,
        end_date: datetime,
        bands: list[str] | None = None,
        composite_months: int = 1,
    ) -> dict[str, np.ndarray]:
        """Fetch Landsat data for given bounds.

        Args:
            bbox: Bounding box (west, south, east, north)
            start_date: Start date
            end_date: End date
            bands: List of bands to fetch (default: all)
            composite_months: Number of months to search for compositing

        Returns:
            Dictionary mapping band names to arrays
        """
        self.validate_request(bbox, start_date, end_date)

        if bands is None:
            bands = list(self.BAND_MAPPING.keys())

        # Get WRS-2 tiles covering bbox
        tiles = self._get_wrs2_tiles(bbox)
        logger.info(f"Found {len(tiles)} WRS-2 tiles for bbox")

        # Expand date range for compositing
        search_start = start_date - timedelta(days=composite_months * 30)
        search_end = end_date

        # Collect scene URLs for each tile
        all_scenes = []
        for path, row in tiles:
            scenes = self._search_scenes(path, row, search_start, search_end)
            all_scenes.extend(scenes)

        logger.info(f"Found {len(all_scenes)} Landsat scenes")

        if not all_scenes:
            raise DataNotAvailableError(
                f"No Landsat scenes found for {bbox} between "
                f"{search_start} and {search_end}"
            )

        # Download and merge scenes
        result = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            for band in bands:
                logger.info(f"Processing band {band}")
                band_arrays = []

                for scene in all_scenes:
                    try:
                        # Download band
                        band_file = self._download_band(scene, band, tmppath)

                        # Read raster
                        data, metadata = self.raster.read(band_file)
                        band_arrays.append((data, metadata))

                    except Exception as e:
                        logger.warning(
                            f"Failed to process {scene['id']} band {band}: {e}"
                        )
                        continue

                if not band_arrays:
                    raise DataSourceError(
                        f"Failed to download any valid data for band {band}"
                    )

                # Merge tiles (simplified - just take first for now)
                # TODO: Implement proper mosaicking
                result[band] = band_arrays[0][0]

        return result

    def get_metadata(self) -> dict[str, Any]:
        """Get Landsat data source metadata."""
        return {
            "name": "Landsat Collection 2 Level-2",
            "description": "Surface reflectance and quality bands from Landsat 8/9",
            "bands": list(self.BAND_MAPPING.keys()),
            "resolution": 30.0,  # meters
            "crs": "EPSG:32600",  # UTM (zone varies)
            "collection": self.collection,
            "level": self.level,
            "sensor": self.sensor,
        }

    def validate_request(
        self,
        bbox: tuple[float, float, float, float],
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        """Validate data request parameters."""
        west, south, east, north = bbox

        # Check bbox validity
        if west >= east or south >= north:
            raise DataSourceError(f"Invalid bounding box: {bbox}")

        if west < -180 or east > 180 or south < -90 or north > 90:
            raise DataSourceError(f"Bounding box out of range: {bbox}")

        # Check date validity
        if start_date >= end_date:
            raise DataSourceError(
                f"Start date {start_date} must be before end date {end_date}"
            )

        # Check if dates are reasonable (Landsat 8 launch: 2013)
        if start_date.year < 2013:
            raise DataSourceError(
                f"Landsat {self.sensor} data not available before 2013"
            )

    def _get_wrs2_tiles(
        self, bbox: tuple[float, float, float, float]
    ) -> list[tuple[int, int]]:
        """Get WRS-2 path/row tiles covering bbox.

        This is a simplified implementation. In production, you would
        use a proper WRS-2 shapefile or API.
        """
        west, south, east, north = bbox

        # Simplified calculation - this is not accurate!
        # Real implementation would use WRS-2 shapefile
        path_min = int((west + 180) * 233 / 360)
        path_max = int((east + 180) * 233 / 360) + 1
        row_min = int((90 - north) * 248 / 180)
        row_max = int((90 - south) * 248 / 180) + 1

        tiles = []
        for path in range(max(1, path_min), min(234, path_max)):
            for row in range(max(1, row_min), min(249, row_max)):
                tiles.append((path, row))

        return tiles

    def _search_scenes(
        self, path: int, row: int, start_date: datetime, end_date: datetime
    ) -> list[dict[str, Any]]:
        """Search for Landsat scenes in S3.

        This is a simplified implementation that constructs S3 paths.
        A real implementation would use STAC or other catalog API.
        """
        scenes = []

        # Check each date in range (simplified - check weekly)
        current_date = start_date
        while current_date <= end_date:
            year = current_date.year

            # Construct S3 prefix
            prefix = (
                f"collection{self.collection}/{self.level}/"
                f"{self.sensor}/{year}/{path:03d}/{row:03d}/"
            )

            try:
                # List objects in S3
                keys = self.storage.list_objects(self.LANDSAT_BUCKET, prefix)

                # Filter for scene directories
                scene_dirs = set()
                for key in keys:
                    parts = key.split("/")
                    if len(parts) > 7:
                        scene_id = parts[7]
                        if scene_id.startswith(self.sensor):
                            scene_dirs.add(scene_id)

                # Add scene info
                for scene_id in scene_dirs:
                    # Extract date from scene ID
                    # Format: LC09_L2SP_001002_20230615_20230617_02_T1
                    try:
                        date_str = scene_id.split("_")[3]
                        scene_date = datetime.strptime(date_str, "%Y%m%d")

                        if start_date <= scene_date <= end_date:
                            scenes.append(
                                {
                                    "id": scene_id,
                                    "path": path,
                                    "row": row,
                                    "date": scene_date,
                                    "prefix": f"{prefix}{scene_id}/",
                                }
                            )
                    except:
                        continue

            except Exception as e:
                logger.debug(f"Error searching {prefix}: {e}")

            current_date += timedelta(days=7)

        return scenes

    def _download_band(self, scene: dict[str, Any], band: str, tmpdir: Path) -> Path:
        """Download a band file from S3."""
        # Get internal band name
        internal_band = self.BAND_MAPPING.get(band, band)

        # Construct S3 key
        s3_key = f"{scene['prefix']}{scene['id']}_{internal_band}.TIF"

        # Local path
        local_path = tmpdir / f"{scene['id']}_{band}.TIF"

        # Download
        logger.debug(f"Downloading s3://{self.LANDSAT_BUCKET}/{s3_key}")
        self.storage.download(self.LANDSAT_BUCKET, s3_key, local_path)

        return local_path

    def get_scene_metadata(self, scene_id: str) -> dict[str, Any]:
        """Get metadata for a specific scene."""
        # Parse scene ID for metadata
        # Format: LC09_L2SP_001002_20230615_20230617_02_T1
        parts = scene_id.split("_")

        return {
            "sensor": parts[0],
            "level": parts[1],
            "path": int(parts[2][:3]),
            "row": int(parts[2][3:]),
            "acquisition_date": datetime.strptime(parts[3], "%Y%m%d"),
            "processing_date": datetime.strptime(parts[4], "%Y%m%d"),
            "collection": parts[5],
            "tier": parts[6],
        }
