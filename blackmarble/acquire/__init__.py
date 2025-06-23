"""Data acquisition module for Black Marble pipeline.

This module handles downloading and fetching data from various sources:
- Landsat imagery from AWS
- VIIRS nighttime lights from NASA LAADS
- OpenStreetMap road networks
"""

from .landsat import (
    download_landsat_band,
    download_tiles_for_band,
    download_tiles_with_composites,
    filter_landsat_scenes,
    get_landsat_metadata,
    get_wrs_path_row,
)

from .osm import (
    create_road_raster,
    create_simple_road_mask,
    fetch_road_network,
    get_hierarchical_road_filters,
    rasterize_road_shapes,
    roads_graph_to_shapes,
)

from .viirs import (
    convert_viirs_hdf_to_tiff,
    coordinates_to_viirs_tile,
    download_viirs_h5,
    download_vnp46a2_tiles,
    extract_viirs_layers,
    get_tiles_for_bbox,
    get_viirs_metadata,
    read_viirs_with_rasterio,
)

__all__ = [
    # Landsat functions
    "download_landsat_band",
    "download_tiles_for_band",
    "download_tiles_with_composites",
    "filter_landsat_scenes",
    "get_landsat_metadata",
    "get_wrs_path_row",
    # OSM functions
    "create_road_raster",
    "create_simple_road_mask",
    "fetch_road_network",
    "get_hierarchical_road_filters",
    "rasterize_road_shapes",
    "roads_graph_to_shapes",
    # VIIRS functions
    "convert_viirs_hdf_to_tiff",
    "coordinates_to_viirs_tile",
    "download_viirs_h5",
    "download_vnp46a2_tiles",
    "extract_viirs_layers",
    "get_tiles_for_bbox",
    "get_viirs_metadata",
    "read_viirs_with_rasterio",
]