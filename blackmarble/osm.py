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

class OSMUtil:
    """Rasterize OpenStreetMap roads into a uint8 mask."""

    def create_road_raster_flattened(self, lat1, lon1, lat2, lon2,
                                     raster_value=5,
                                     reference_raster='outputs/ntl.tif',
                                     output_path='outputs/osm.tif',
                                     path_thickness=0.00005):
        """Rasterize all roads to a single mask value."""
        with rasterio.open(reference_raster) as src:
            height = src.height
            width = src.width
            transform = src.transform
            profile = src.profile.copy()

        # Download/Ox graph for full bbox
        G = ox.graph_from_bbox(max(lat1, lat2), min(lat1, lat2),
                               max(lon1, lon2), min(lon1, lon2),
                               retain_all=True, network_type='all')
        ox.save_graph_shapefile(G, 'roads')

        shp = 'roads/edges.shp'
        road_shapes = []
        with fiona.open(shp) as feats:
            for feat in feats:
                geom = feat['geometry']
                poly = shapely.geometry.shape(geom).buffer(path_thickness)
                road_shapes.append(shapely.geometry.mapping(poly))

        mask = rasterio.features.rasterize(
            road_shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=False,
            default_value=raster_value,
            dtype=np.uint8)

        # Prepare output profile: single-band uint8 with no nodata value
        profile.pop('nodata', None)
        profile.update(count=1, dtype='uint8')
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(mask, 1)

    def create_road_raster(self, lat1, lon1, lat2, lon2,
                           raster_value=5,
                           reference_raster='outputs/ntl.tif',
                           output_path='outputs/osm.tif',
                           path_thickness=0.00005):
        """Rasterize roads by type, layering higher-index roads over lower."""
        with rasterio.open(reference_raster) as src:
            height = src.height
            width = src.width
            transform = src.transform
            profile = src.profile.copy()

        base = np.zeros((height, width), dtype=np.uint8)
        filters = [
            '["highway"~"primary_link|primary|secondary|secondary_link|tertiary|tertiary_link"]',
            '["highway"~"motorway|motorway_link"]',
            '["highway"~"residential"]',
            '["highway"~"trunk|trunk_link"]',
            '["highway"~"service|unclassified|road|busway"]'
        ]
        for idx, custom_filter in enumerate(filters, start=1):
            try:
                G = ox.graph_from_bbox(max(lat1, lat2), min(lat1, lat2),
                                       max(lon1, lon2), min(lon1, lon2),
                                       retain_all=True,
                                       network_type='all',
                                       custom_filter=custom_filter)
                ox.save_graph_shapefile(G, 'roads')

                shp = 'roads/edges.shp'
                shapes = []
                with fiona.open(shp) as feats:
                    for feat in feats:
                        geom = feat['geometry']
                        poly = shapely.geometry.shape(geom).buffer(path_thickness)
                        shapes.append(shapely.geometry.mapping(poly))

                mask = rasterio.features.rasterize(
                    shapes,
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    all_touched=False,
                    default_value=idx,
                    dtype=np.uint8)
                # overlay
                base = np.maximum(base, mask)
            except Exception:
                print(f'No data for filter {custom_filter}')

        # Prepare output profile: single-band uint8 with no nodata value
        profile.pop('nodata', None)
        profile.update(count=1, dtype='uint8')
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(base, 1)