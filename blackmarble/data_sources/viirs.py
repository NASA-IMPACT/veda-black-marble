"""VIIRS nighttime lights data source implementation."""

import logging
from typing import Any
from datetime import datetime
from pathlib import Path
import numpy as np
import tempfile

from ..adapters.base import APIAdapter, RasterAdapter
from ..adapters.nasa_api import NASAEarthdataAdapter
from ..adapters.raster_io import GDALRasterAdapter
from .base import DataSource, DataSourceError, DataNotAvailableError

logger = logging.getLogger(__name__)


class VIIRSDataSource:
    """VIIRS nighttime lights data source using NASA Earthdata API.

    Fetches VNP46A2 Black Marble daily nighttime lights data.
    """

    # Collection information
    COLLECTION_NAME = "VNP46A2"
    COLLECTION_VERSION = "001"

    # Layer mapping
    LAYER_MAPPING = {
        "NTL": "HDFEOS/GRIDS/VNP46A2_h{h:02d}v{v:02d}/Data Fields/NearNadir_Composite_Snow_Free",
        "QF": "HDFEOS/GRIDS/VNP46A2_h{h:02d}v{v:02d}/Data Fields/Mandatory_Quality_Flag",
        "LUNAR": "HDFEOS/GRIDS/VNP46A2_h{h:02d}v{v:02d}/Data Fields/LunarIrradiance",
    }

    def __init__(
        self,
        api_adapter: APIAdapter | None = None,
        raster_adapter: RasterAdapter | None = None,
        earthdata_token: str | None = None,
    ):
        """Initialize VIIRS data source.

        Args:
            api_adapter: API adapter for NASA Earthdata
            raster_adapter: Raster adapter for reading HDF5 files
            earthdata_token: NASA Earthdata authentication token
        """
        self.api = api_adapter or NASAEarthdataAdapter(earthdata_token)
        self.raster = raster_adapter or GDALRasterAdapter()

        # Authenticate if token provided
        if earthdata_token and not api_adapter:
            self.api.authenticate({"token": earthdata_token})

    def fetch(
        self,
        bbox: tuple[float, float, float, float],
        start_date: datetime,
        end_date: datetime,
        layers: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Fetch VIIRS data for given bounds.

        Args:
            bbox: Bounding box (west, south, east, north)
            start_date: Start date
            end_date: End date
            layers: List of layers to fetch (default: all)

        Returns:
            Dictionary mapping layer names to arrays
        """
        self.validate_request(bbox, start_date, end_date)

        if layers is None:
            layers = list(self.LAYER_MAPPING.keys())

        # Get MODIS tiles covering bbox
        tiles = self._get_modis_tiles(bbox)
        logger.info(f"Found {len(tiles)} MODIS tiles for bbox")

        # Search for granules
        granules = self._search_granules(bbox, start_date, end_date)
        logger.info(f"Found {len(granules)} VIIRS granules")

        if not granules:
            raise DataNotAvailableError(
                f"No VIIRS data found for {bbox} between {start_date} and {end_date}"
            )

        # Download and process granules
        result = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # For simplicity, just use the first granule
            # Real implementation would mosaic multiple tiles
            granule = granules[0]

            # Download granule
            hdf_path = self._download_granule(granule, tmppath)

            # Extract layers
            for layer in layers:
                logger.info(f"Extracting layer {layer}")

                # Get tile info from filename
                h, v = self._parse_tile_from_filename(hdf_path.name)

                # Format layer path
                layer_path = self.LAYER_MAPPING[layer].format(h=h, v=v)

                # Read from HDF5
                # Note: Real implementation would use GDAL subdataset syntax
                # This is simplified
                data = self._read_hdf_layer(hdf_path, layer_path)
                result[layer] = data

        return result

    def get_metadata(self) -> dict[str, Any]:
        """Get VIIRS data source metadata."""
        return {
            "name": "VIIRS Black Marble (VNP46A2)",
            "description": "Daily nighttime lights at 500m resolution",
            "layers": list(self.LAYER_MAPPING.keys()),
            "resolution": 500.0,  # meters
            "crs": "MODIS Sinusoidal",
            "collection": self.COLLECTION_NAME,
            "version": self.COLLECTION_VERSION,
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

        # VNP46A2 available from 2012 onwards
        if start_date.year < 2012:
            raise DataSourceError("VIIRS data not available before 2012")

    def _get_modis_tiles(self, bbox: tuple[float, float, float, float]) -> list[str]:
        """Get MODIS tile IDs covering bbox.

        This is a simplified implementation.
        Real implementation would use MODIS tile grid calculations.
        """
        west, south, east, north = bbox

        # Very simplified - just return some example tiles
        # Real implementation would calculate based on sinusoidal projection
        tiles = []

        # Approximate tile calculation
        h_min = int((west + 180) * 36 / 360)
        h_max = int((east + 180) * 36 / 360) + 1
        v_min = int((90 - north) * 18 / 180)
        v_max = int((90 - south) * 18 / 180) + 1

        for h in range(max(0, h_min), min(36, h_max)):
            for v in range(max(0, v_min), min(18, v_max)):
                tiles.append(f"h{h:02d}v{v:02d}")

        return tiles

    def _search_granules(
        self,
        bbox: tuple[float, float, float, float],
        start_date: datetime,
        end_date: datetime,
    ) -> list[dict[str, Any]]:
        """Search for VIIRS granules using CMR API."""
        # Use the API adapter to search
        granules = self.api.search(
            short_name=self.COLLECTION_NAME,
            version=self.COLLECTION_VERSION,
            bounding_box=f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            temporal=f"{start_date.isoformat()}Z,{end_date.isoformat()}Z",
            page_size=100,
        )

        return granules

    def _download_granule(self, granule: dict[str, Any], tmpdir: Path) -> Path:
        """Download granule HDF5 file."""
        # Extract download URL
        download_urls = []

        # Look for data links
        for link in granule.get("links", []):
            if link.get("rel") == "http://esipfed.org/ns/fedsearch/1.1/data#":
                href = link.get("href")
                if href and href.endswith(".h5"):
                    download_urls.append(href)

        if not download_urls:
            raise DataSourceError(
                f"No download URL found for granule {granule.get('id')}"
            )

        # Download file
        url = download_urls[0]
        filename = Path(url).name
        local_path = tmpdir / filename

        logger.debug(f"Downloading {url}")
        self.api.download(url, local_path)

        return local_path

    def _parse_tile_from_filename(self, filename: str) -> tuple[int, int]:
        """Extract MODIS tile h,v from filename.

        Example: VNP46A2.A2023166.h18v04.001.2023167123456.h5
        """
        parts = filename.split(".")
        tile_part = parts[2]  # h18v04

        h = int(tile_part[1:3])
        v = int(tile_part[4:6])

        return h, v

    def _read_hdf_layer(self, hdf_path: Path, layer_path: str) -> np.ndarray:
        """Read layer from HDF5 file.

        This is simplified - real implementation would use GDAL
        subdataset syntax like:
        HDF5:"file.h5"://layer/path
        """
        # For now, return dummy data
        # Real implementation would use:
        # subdataset = f'HDF5:"{hdf_path}"://{layer_path}'
        # data, metadata = self.raster.read(subdataset)

        # Dummy 500x500 array
        return np.random.rand(500, 500).astype(np.float32)

    def get_granule_metadata(self, granule: dict[str, Any]) -> dict[str, Any]:
        """Extract metadata from granule information."""
        metadata = {
            "id": granule.get("id"),
            "title": granule.get("title"),
            "time_start": granule.get("time_start"),
            "time_end": granule.get("time_end"),
            "dataset_id": granule.get("dataset_id"),
            "collection": self.COLLECTION_NAME,
            "version": self.COLLECTION_VERSION,
        }

        # Extract bounding box if available
        if "boxes" in granule:
            boxes = granule["boxes"]
            if boxes:
                # Format: "west south east north"
                coords = boxes[0].split()
                metadata["bbox"] = [
                    float(coords[0]),
                    float(coords[1]),
                    float(coords[2]),
                    float(coords[3]),
                ]

        return metadata
