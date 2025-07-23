"""Data acquisition module for Black Marble pipeline.

This module handles fetching data from various sources:
- Landsat imagery from AWS
- VIIRS nighttime lights from NASA LAADS
- OpenStreetMap road networks
"""

from .async_orchestrator import AsyncDownloadOrchestrator, DownloadPlan
from .osm import buffer_and_rasterize_roads, fetch_road_network
from .osm_fractional import buffer_and_rasterize_roads_fractional

__all__ = [
    # Async download orchestration (used by pipeline)
    "AsyncDownloadOrchestrator",
    "DownloadPlan",
    # OSM functions (used by pipeline and tests)
    "fetch_road_network",
    "buffer_and_rasterize_roads",  # Used in tests
    "buffer_and_rasterize_roads_fractional",
]
