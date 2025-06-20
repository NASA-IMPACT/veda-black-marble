#!/usr/bin/env python3
"""Test OSM functions with small, manageable areas."""

import numpy as np
from blackmarble.osm_functions import (
    download_osm_roads,
    graph_to_shapes,
    rasterize_shapes,
)
from rasterio.transform import from_bounds


def test_tiny_area():
    """Test with a very small area (few hundred meters)."""

    # Times Square, NYC - very small area
    center_lat, center_lon = 40.7580, -73.9855

    # Create a ~200m x 200m box
    offset = 0.001  # roughly 100m at this latitude

    north = center_lat + offset
    south = center_lat - offset
    east = center_lon + offset
    west = center_lon - offset

    print(f"Testing area: ({south:.4f}, {west:.4f}) to ({north:.4f}, {east:.4f})")
    print(
        f"Approximate size: {2 * offset * 111}km x {2 * offset * 111 * np.cos(np.radians(center_lat))}km"
    )

    try:
        # Download all roads
        print("\n1. Downloading all roads...")
        G = download_osm_roads(north, south, east, west)

        if G.number_of_edges() == 0:
            print("ERROR: No roads found even in Times Square!")
            return False

        # Convert to shapes
        print("\n2. Converting to shapes...")
        shapes = graph_to_shapes(G, buffer_distance=0.00001)  # ~1m buffer

        # Rasterize at high resolution
        print("\n3. Rasterizing...")
        height = width = 100
        transform = from_bounds(west, south, east, north, width, height)

        mask = rasterize_shapes(shapes, height, width, transform)

        print(f"\nResults:")
        print(f"  Roads found: {G.number_of_edges()}")
        print(f"  Shapes created: {len(shapes)}")
        print(
            f"  Pixels rasterized: {np.sum(mask > 0)}/{mask.size} ({100.0 * np.sum(mask > 0) / mask.size:.1f}%)"
        )

        return True

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        return False


def test_buffer_effect():
    """Test how buffer size affects rasterization."""

    # Small area in Manhattan
    center_lat, center_lon = 40.7580, -73.9855
    offset = 0.002  # ~200m

    north = center_lat + offset
    south = center_lat - offset
    east = center_lon + offset
    west = center_lon - offset

    print(
        f"\nTesting buffer effects on area: ({south:.4f}, {west:.4f}) to ({north:.4f}, {east:.4f})"
    )

    try:
        G = download_osm_roads(north, south, east, west)

        if G.number_of_edges() == 0:
            print("No roads found!")
            return

        height = width = 200
        transform = from_bounds(west, south, east, north, width, height)

        # Test different buffer sizes
        for buffer_m in [1, 5, 10, 20]:
            # Convert meters to degrees (approximate)
            buffer_deg = buffer_m / 111000.0

            shapes = graph_to_shapes(G, buffer_deg)
            mask = rasterize_shapes(shapes, height, width, transform)

            pixels = np.sum(mask > 0)
            print(
                f"  Buffer {buffer_m}m (~{buffer_deg:.6f}°): {pixels} pixels ({100.0 * pixels / mask.size:.1f}%)"
            )

    except Exception as e:
        print(f"ERROR: {e}")


def diagnose_actual_area(lat1, lon1, lat2, lon2):
    """Diagnose what's happening with the actual pipeline coordinates."""

    north = max(lat1, lat2)
    south = min(lat1, lat2)
    east = max(lon1, lon2)
    west = min(lon1, lon2)

    print(f"\nDiagnosing area: ({south:.4f}, {west:.4f}) to ({north:.4f}, {east:.4f})")

    # Calculate area
    height_km = (north - south) * 111
    width_km = (east - west) * 111 * np.cos(np.radians((north + south) / 2))
    print(
        f"Area size: {height_km:.1f}km x {width_km:.1f}km = {height_km * width_km:.1f} km²"
    )

    # Try downloading a small subset
    subset_size = 0.01  # ~1km
    center_lat = (north + south) / 2
    center_lon = (east + west) / 2

    mini_north = center_lat + subset_size
    mini_south = center_lat - subset_size
    mini_east = center_lon + subset_size
    mini_west = center_lon - subset_size

    print(f"\nTesting 1km² subset at center...")

    try:
        G = download_osm_roads(mini_north, mini_south, mini_east, mini_west)

        if G.number_of_edges() == 0:
            print("WARNING: No roads found even in subset!")
            print("Possible issues:")
            print("  - Area might be ocean/desert/wilderness")
            print("  - OSM coverage might be poor")
            print("  - Network connectivity issues")
        else:
            print(f"✓ Found {G.number_of_edges()} road segments in subset")

    except Exception as e:
        print(f"ERROR downloading subset: {e}")


if __name__ == "__main__":
    print("Testing OSM with small areas...\n")

    # Test 1: Tiny area
    success = test_tiny_area()

    if success:
        # Test 2: Buffer effects
        test_buffer_effect()

    # Test 3: Diagnose your actual area
    # Replace with your actual coordinates
    print("\n" + "=" * 50)
    print("To diagnose your specific area, update these coordinates:")
    print("lat1, lon1 = <your_lat1>, <your_lon1>")
    print("lat2, lon2 = <your_lat2>, <your_lon2>")
    print("diagnose_actual_area(lat1, lon1, lat2, lon2)")

    # Example - uncomment and update with actual values:
    # diagnose_actual_area(40.0, -3.8, 40.5, -3.7)
