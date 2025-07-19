"""Data acquisition module for Black Marble pipeline.

This module handles fetching data from various sources:
- Landsat imagery from AWS
- VIIRS nighttime lights from NASA LAADS
- OpenStreetMap road networks
"""

from .async_orchestrator import DownloadPlan, DownloadResults, download_all_data_sync
from .landsat import download_landsat
from .osm import buffer_and_rasterize_roads, clear_osm_cache, fetch_road_network
from .osm_fractional import buffer_and_rasterize_roads_fractional
from .viirs import download_viirs, download_viirs_h5

__all__ = [
    # Landsat functions (used by pipeline)
    "download_landsat",
    # OSM functions (used by pipeline)
    "fetch_road_network",
    "buffer_and_rasterize_roads",
    "buffer_and_rasterize_roads_fractional",
    "clear_osm_cache",
    # VIIRS functions (used by pipeline)
    "download_viirs_h5",
    "download_viirs",
    # Async download orchestration
    "download_all_data_sync",
    "DownloadPlan",
    "DownloadResults",
]
