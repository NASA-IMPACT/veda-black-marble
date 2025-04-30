"""
OpenStreetMap utilities:
  - OSMUtil: rasterize roads (flat and layered) into a uint8 mask
"""
# OpenStreetMap utilities for road rasterization
import os
import numpy as np
import rasterio
import rasterio.features
import fiona
import osmnx as ox
import shapely.geometry
import shutil # Import shutil for directory removal
from typing import List, Dict, Any, Union  # Add typing imports

class OSMUtil:
    """Rasterize OpenStreetMap roads into a uint8 mask."""

    def create_road_raster_flattened(self, lat1: float, lon1: float, lat2: float, lon2: float,
                                     raster_value: int = 5,
                                     reference_raster: str = 'outputs/ntl.tif', # Keep default for reference if needed outside pipeline context
                                     output_path: str = 'outputs/osm.tif', # Keep default for output if needed outside pipeline context
                                     output_dir: str = 'outputs', # Add output_dir
                                     path_thickness: float = 0.00005) -> None:  # Add type hints
        """Rasterize all roads to a single mask value."""
        with rasterio.open(reference_raster) as src:
            height: int = src.height  # Add type hint
            width: int = src.width  # Add type hint
            transform: rasterio.Affine = src.transform  # Add type hint
            profile: Dict[str, Any] = src.profile.copy()  # Add type hint

        # Define shapefile output path within output_dir/temp
        temp_dir = os.path.join(output_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        shapefile_dir = os.path.join(temp_dir, 'roads_osm_flat')

        # Download/Ox graph for full bbox
        try:
            G = ox.graph_from_bbox(max(lat1, lat2), min(lat1, lat2),
                                   max(lon1, lon2), min(lon1, lon2),
                                   retain_all=True, network_type='all')
            if G.number_of_edges() == 0:
                 print("Warning: No road edges found in the specified bounding box.")
                 # Create an empty mask and return
                 mask = np.zeros((height, width), dtype=np.uint8)
            else:
                ox.save_graph_shapefile(G, filepath=shapefile_dir) # Use filepath argument

                shp: str = os.path.join(shapefile_dir, 'edges.shp')  # Add type hint
                road_shapes: List[Dict[str, Any]] = []  # Add type hint
                with fiona.open(shp) as feats:
                    for feat in feats:
                        geom: Dict[str, Any] = feat['geometry']  # Add type hint
                        poly: shapely.geometry.Polygon = shapely.geometry.shape(geom).buffer(path_thickness)  # Add type hint
                        road_shapes.append(shapely.geometry.mapping(poly))

                mask: np.ndarray = rasterio.features.rasterize(  # Add type hint
                    road_shapes,
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    all_touched=False,
                    default_value=raster_value,
                    dtype=np.uint8)
        except Exception as e:
             print(f"Error during OSM download/processing (flattened): {e}. Creating empty mask.")
             mask = np.zeros((height, width), dtype=np.uint8)


        # Prepare output profile: single-band uint8 with no nodata value
        profile.pop('nodata', None)
        profile.update(count=1, dtype='uint8')
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(mask, 1)

        # Clean up temporary shapefile directory
        if os.path.exists(shapefile_dir):
            try:
                shutil.rmtree(shapefile_dir)
            except OSError as e:
                print(f"Warning: Could not remove temporary OSM shapefile directory {shapefile_dir}: {e}")


    def create_road_raster(self, lat1: float, lon1: float, lat2: float, lon2: float,
                           raster_value: int = 5,  # Note: raster_value seems unused here?
                           reference_raster: str = 'outputs/ntl.tif', # Keep default
                           output_path: str = 'outputs/osm.tif', # Keep default
                           output_dir: str = 'outputs', # Add output_dir
                           path_thickness: float = 0.00005) -> None:  # Add type hints
        """Rasterize roads by type, layering higher-index roads over lower."""
        with rasterio.open(reference_raster) as src:
            height: int = src.height  # Add type hint
            width: int = src.width  # Add type hint
            transform: rasterio.Affine = src.transform  # Add type hint
            profile: Dict[str, Any] = src.profile.copy()  # Add type hint

        # Define shapefile output path within output_dir/temp
        temp_dir = os.path.join(output_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        shapefile_dir = os.path.join(temp_dir, 'roads_osm_typed')


        base: np.ndarray = np.zeros((height, width), dtype=np.uint8)  # Add type hint
        filters: List[str] = [  # Add type hint
            '["highway"~"primary_link|primary|secondary|secondary_link|tertiary|tertiary_link"]',
            '["highway"~"motorway|motorway_link"]',
            '["highway"~"residential"]',
            '["highway"~"trunk|trunk_link"]',
            '["highway"~"service|unclassified|road|busway"]'
        ]
        for idx, custom_filter in enumerate(filters, start=1):
            try:
                print(f"Downloading OSM data for filter: {custom_filter}")
                G = ox.graph_from_bbox(max(lat1, lat2), min(lat1, lat2),
                                       max(lon1, lon2), min(lon1, lon2),
                                       retain_all=True,
                                       network_type='all',
                                       custom_filter=custom_filter)

                if G.number_of_edges() == 0:
                    print(f'No data for filter {custom_filter}')
                    continue # Skip if no edges found for this filter

                # Clean previous shapefile dir for this type if it exists
                if os.path.exists(shapefile_dir):
                     shutil.rmtree(shapefile_dir)

                ox.save_graph_shapefile(G, filepath=shapefile_dir) # Use filepath argument

                shp: str = os.path.join(shapefile_dir, 'edges.shp')  # Add type hint
                shapes: List[Dict[str, Any]] = []  # Add type hint
                with fiona.open(shp) as feats:
                    for feat in feats:
                        geom: Dict[str, Any] = feat['geometry']  # Add type hint
                        poly: shapely.geometry.Polygon = shapely.geometry.shape(geom).buffer(path_thickness)  # Add type hint
                        shapes.append(shapely.geometry.mapping(poly))

                print(f"Rasterizing {len(shapes)} shapes for filter index {idx}...")
                mask: np.ndarray = rasterio.features.rasterize(  # Add type hint
                    shapes,
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    all_touched=False,
                    default_value=idx,
                    dtype=np.uint8)
                # overlay
                base = np.maximum(base, mask)
                print(f"Finished rasterizing for filter index {idx}.")

            except Exception as e:
                print(f'Error processing filter {custom_filter}: {e}')

        # Prepare output profile: single-band uint8 with no nodata value
        profile.pop('nodata', None)
        profile.update(count=1, dtype='uint8')
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(base, 1)

        # Clean up temporary shapefile directory
        if os.path.exists(shapefile_dir):
            try:
                shutil.rmtree(shapefile_dir)
            except OSError as e:
                print(f"Warning: Could not remove temporary OSM shapefile directory {shapefile_dir}: {e}")