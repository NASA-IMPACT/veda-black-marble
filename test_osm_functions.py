#!/usr/bin/env python3
"""Test the OSM pure functions to debug road data issues."""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from blackmarble.osm_functions import (
    download_osm_roads,
    graph_to_shapes,
    rasterize_shapes,
    create_multi_type_road_raster,
    bbox_area_km2,
)


def test_osm_download():
    """Test downloading OSM data for a known area with roads."""

    # Test 1: Small area in Manhattan, NYC (should have lots of roads)
    print("Test 1: Manhattan, NYC")
    north, south = 40.7589, 40.7489  # ~1km N-S
    east, west = -73.9842, -73.9942  # ~1km E-W

    try:
        G = download_osm_roads(north, south, east, west)
        print(f"✓ Downloaded {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Test with filter
        G_primary = download_osm_roads(
            north, south, east, west, '["highway"~"primary|secondary|tertiary"]'
        )
        print(f"✓ With filter: {G_primary.number_of_edges()} edges")

    except Exception as e:
        print(f"✗ Failed: {e}")

    # Test 2: Rural area (fewer roads expected)
    print("\nTest 2: Rural area in Wyoming")
    north, south = 41.0, 40.9
    east, west = -104.0, -104.1

    try:
        G = download_osm_roads(north, south, east, west)
        print(f"✓ Downloaded {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except Exception as e:
        print(f"✗ Failed: {e}")


def test_shape_conversion():
    """Test converting graph to shapes."""

    print("\nTesting shape conversion...")

    # Download a small test area
    north, south = 40.7589, 40.7489
    east, west = -73.9842, -73.9942

    G = download_osm_roads(north, south, east, west)

    # Test different buffer distances
    for buffer in [0.00005, 0.0001, 0.0002]:
        shapes = graph_to_shapes(G, buffer)
        print(f"✓ Buffer {buffer}: Created {len(shapes)} shapes")


def test_rasterization():
    """Test rasterizing shapes to array."""

    print("\nTesting rasterization...")

    # Define test area
    north, south = 40.7589, 40.7489
    east, west = -73.9842, -73.9942

    # Create transform for 100x100 pixel raster
    height, width = 100, 100
    transform = from_bounds(west, south, east, north, width, height)

    # Download and convert
    G = download_osm_roads(north, south, east, west)
    shapes = graph_to_shapes(G, 0.00005)

    # Test rasterization
    mask = rasterize_shapes(shapes, height, width, transform)

    print(f"✓ Raster shape: {mask.shape}")
    print(
        f"✓ Non-zero pixels: {np.sum(mask > 0)} ({100.0 * np.sum(mask > 0) / mask.size:.1f}%)"
    )
    print(f"✓ Unique values: {np.unique(mask)}")


def test_multi_type_roads():
    """Test creating multi-type road raster."""

    print("\nTesting multi-type road raster...")

    # Manhattan test area
    north, south = 40.7589, 40.7489
    east, west = -73.9842, -73.9942

    height, width = 200, 200
    transform = from_bounds(west, south, east, north, width, height)

    # Create raster
    road_raster = create_multi_type_road_raster(
        north, south, east, west, height, width, transform, buffer_distance=0.00005
    )

    print(f"\n✓ Final shape: {road_raster.shape}")
    print(f"✓ Unique values: {np.unique(road_raster)}")

    # Save for visual inspection
    import os

    os.makedirs("test_outputs", exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": transform,
    }

    with rasterio.open("test_outputs/test_roads.tif", "w", **profile) as dst:
        dst.write(road_raster, 1)
    print("✓ Saved test raster to test_outputs/test_roads.tif")


def test_actual_pipeline_area():
    """Test with the actual coordinates from the pipeline run."""

    print("\nTesting with actual pipeline coordinates...")

    # These would be the actual coordinates from your pipeline
    # You'll need to fill these in based on your LAT1, LON1, LAT2, LON2
    lat1, lon1 = 40.0, -3.8  # Example - replace with actual
    lat2, lon2 = 40.5, -3.7  # Example - replace with actual

    north = max(lat1, lat2)
    south = min(lat1, lat2)
    east = max(lon1, lon2)
    west = min(lon1, lon2)

    area = bbox_area_km2(north, south, east, west)
    print(f"Bounding box area: {area:.1f} km²")

    # Test at different resolutions
    for resolution in [100, 500, 1000]:
        print(f"\nTesting at {resolution}x{resolution} resolution:")

        height = width = resolution
        transform = from_bounds(west, south, east, north, width, height)
        pixel_size_m = area * 1e6 / (height * width)
        print(f"Pixel size: ~{np.sqrt(pixel_size_m):.1f} meters")

        try:
            road_raster = create_multi_type_road_raster(
                north,
                south,
                east,
                west,
                height,
                width,
                transform,
                buffer_distance=0.00005,
            )

            if np.sum(road_raster > 0) == 0:
                print("⚠️  WARNING: No roads found at this resolution!")

        except Exception as e:
            print(f"✗ Error: {e}")


if __name__ == "__main__":
    print("Testing OSM functions...\n")

    test_osm_download()
    test_shape_conversion()
    test_rasterization()
    test_multi_type_roads()

    # Uncomment and fill in actual coordinates to test your specific area
    # test_actual_pipeline_area()

    print("\nDone!")
