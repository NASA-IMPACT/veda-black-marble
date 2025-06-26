"""Data acquisition module for Black Marble pipeline.

This module handles fetching data from various sources:
- Landsat imagery from AWS
- VIIRS nighttime lights from NASA LAADS
- OpenStreetMap road networks
"""

from .landsat import (
    filter_landsat_scenes,
    download_all_bands_for_date,
)

from .osm import (
    fetch_road_network,
    buffer_and_rasterize_roads,
)

from .viirs import (
    download_viirs_h5,
    download_vnp46a2_tiles,
)

from .async_orchestrator import (
    download_all_data_sync,
    DownloadPlan,
    DownloadResults,
)

__all__ = [
    # Landsat functions (used by pipeline)
    "filter_landsat_scenes",
    "download_all_bands_for_date",
    # OSM functions (used by pipeline)
    "fetch_road_network",
    "buffer_and_rasterize_roads",
    # VIIRS functions (used by pipeline)
    "download_viirs_h5",
    "download_vnp46a2_tiles",
    # Async download orchestration
    "download_all_data_sync",
    "DownloadPlan",
    "DownloadResults",
]