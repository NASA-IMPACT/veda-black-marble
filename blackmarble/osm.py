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
import shutil  # Import shutil for directory removal
from typing import Any


class OSMUtil:
    """Rasterize OpenStreetMap roads into a uint8 mask."""

    def create_road_raster_flattened(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        reference_raster: str = "outputs/ntl.tif",
        output_path: str = "outputs/osm.tif",
        output_dir: str = "outputs",
        path_thickness: float = 0.00005,
    ) -> None:
        """Rasterize all roads to a single mask value."""
        with rasterio.open(reference_raster) as src:
            height: int = src.height
            width: int = src.width
            transform: rasterio.Affine = src.transform
            profile: dict[str, Any] = src.profile.copy()
        temp_dir = os.path.join(output_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        shapefile_dir = os.path.join(temp_dir, "roads_osm_flat")
        try:
            print(
                f"\nDEBUG OSM: Downloading roads for bbox: ({lat1}, {lon1}) to ({lat2}, {lon2})"
            )
            # osmnx expects bbox as (west, south, east, north)
            G = ox.graph_from_bbox(
                bbox=(
                    min(lon1, lon2),  # west
                    min(lat1, lat2),  # south
                    max(lon1, lon2),  # east
                    max(lat1, lat2),  # north
                ),
                retain_all=True,
                network_type="all",
            )
            print(
                f"DEBUG OSM: Downloaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
            )
            if G.number_of_edges() == 0:
                print("WARNING OSM: No roads found in the area!")
                mask: np.ndarray = np.zeros((height, width), dtype=np.uint8)
            else:
                print(f"DEBUG OSM: Processing {G.number_of_edges()} road edges")
                # Convert graph to GeoDataFrames and save as shapefiles
                nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
                os.makedirs(shapefile_dir, exist_ok=True)
                edges_gdf.to_file(os.path.join(shapefile_dir, "edges.shp"))
                shp: str = os.path.join(shapefile_dir, "edges.shp")
                road_shapes: list[dict[str, Any]] = []
                with fiona.open(shp) as feats:
                    for feat in feats:
                        geom: dict[str, Any] = feat["geometry"]
                        poly: shapely.geometry.Polygon = shapely.geometry.shape(
                            geom
                        ).buffer(path_thickness)
                        road_shapes.append(shapely.geometry.mapping(poly))
                print(f"DEBUG OSM: Rasterizing {len(road_shapes)} road shapes")
                print(f"DEBUG OSM: Path thickness: {path_thickness} degrees")

                mask = rasterio.features.rasterize(
                    road_shapes,
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    all_touched=True,  # Changed to True to capture more pixels
                    default_value=5,
                    dtype=np.uint8,
                )

                road_pixel_count = np.sum(mask > 0)
                print(
                    f"DEBUG OSM: Rasterized {road_pixel_count} road pixels ({100.0 * road_pixel_count / mask.size:.2f}%)"
                )
        except Exception as e:
            print(
                f"ERROR OSM: Failed to download/rasterize roads: {type(e).__name__}: {e}"
            )
            import traceback

            traceback.print_exc()
            mask = np.zeros((height, width), dtype=np.uint8)
        profile.pop("nodata", None)
        profile.update(count=1, dtype="uint8")

        print(f"DEBUG OSM: Saving road raster to {output_path}")
        print(
            f"DEBUG OSM: Final mask - min: {mask.min()}, max: {mask.max()}, non-zero: {np.sum(mask > 0)}"
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mask, 1)
        if os.path.exists(shapefile_dir):
            shutil.rmtree(shapefile_dir)

    def create_road_raster(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        reference_raster: str = "outputs/ntl.tif",
        output_path: str = "outputs/osm.tif",
        output_dir: str = "outputs",
        path_thickness: float = 0.00005,
    ) -> None:
        """Rasterize roads by type, layering higher-index roads over lower."""
        with rasterio.open(reference_raster) as src:
            height: int = src.height
            width: int = src.width
            transform: rasterio.Affine = src.transform
            profile: dict[str, Any] = src.profile.copy()
        temp_dir = os.path.join(output_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        shapefile_dir = os.path.join(temp_dir, "roads_osm_typed")
        base: np.ndarray = np.zeros((height, width), dtype=np.uint8)
        filters: list[str] = [
            '["highway"~"primary_link|primary|secondary|secondary_link|tertiary|tertiary_link"]',
            '["highway"~"motorway|motorway_link"]',
            '["highway"~"residential"]',
            '["highway"~"trunk|trunk_link"]',
            '["highway"~"service|unclassified|road|busway"]',
        ]
        for idx, custom_filter in enumerate(filters, start=1):
            try:
                G = ox.graph_from_bbox(
                    bbox=(
                        min(lon1, lon2),
                        min(lat1, lat2),
                        max(lon1, lon2),
                        max(lat1, lat2),
                    ),
                    retain_all=True,
                    network_type="all",
                    custom_filter=custom_filter,
                )
                if G.number_of_edges() == 0:
                    continue
                if os.path.exists(shapefile_dir):
                    shutil.rmtree(shapefile_dir)
                # Convert graph to GeoDataFrames and save as shapefiles
                nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
                os.makedirs(shapefile_dir, exist_ok=True)
                edges_gdf.to_file(os.path.join(shapefile_dir, "edges.shp"))
                shp: str = os.path.join(shapefile_dir, "edges.shp")
                shapes: list[dict[str, Any]] = []
                with fiona.open(shp) as feats:
                    for feat in feats:
                        geom: dict[str, Any] = feat["geometry"]
                        poly: shapely.geometry.Polygon = shapely.geometry.shape(
                            geom
                        ).buffer(path_thickness)
                        shapes.append(shapely.geometry.mapping(poly))
                print(
                    f"DEBUG OSM: Rasterizing {len(shapes)} shapes for road type {idx}"
                )

                mask: np.ndarray = rasterio.features.rasterize(
                    shapes,
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    all_touched=True,  # Changed to True to capture more pixels
                    default_value=idx,
                    dtype=np.uint8,
                )

                type_pixels = np.sum(mask > 0)
                print(f"DEBUG OSM: Road type {idx} has {type_pixels} pixels")
                base = np.maximum(base, mask)
            except Exception as e:
                print(
                    f"DEBUG OSM: Error with filter {custom_filter}: {type(e).__name__}: {e}"
                )
                continue
        profile.pop("nodata", None)
        profile.update(count=1, dtype="uint8")

        total_road_pixels = np.sum(base > 0)
        print(f"\nDEBUG OSM: Final road raster summary:")
        print(
            f"  Total road pixels: {total_road_pixels} ({100.0 * total_road_pixels / base.size:.2f}%)"
        )
        print(f"  Output path: {output_path}")

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(base, 1)
        if os.path.exists(shapefile_dir):
            shutil.rmtree(shapefile_dir)
