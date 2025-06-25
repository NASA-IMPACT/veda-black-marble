"""Data acquisition module for Black Marble pipeline.

This module handles downloading and fetching data from various sources:
- Landsat imagery from AWS
- VIIRS nighttime lights from NASA LAADS
- OpenStreetMap road networks
"""

from .landsat import (
    filter_landsat_scenes,
)

from .osm import (
    fetch_road_network,
    roads_graph_to_shapes,
)

from .viirs import (
    download_viirs_h5,
)

__all__ = [
    # Landsat functions (used by pipeline)
    "filter_landsat_scenes",
    # OSM functions (used by pipeline)
    "fetch_road_network",
    "roads_graph_to_shapes",
    # VIIRS functions (used by pipeline)
    "download_viirs_h5",
]