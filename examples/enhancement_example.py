#!/usr/bin/env python3
"""Example demonstrating the use of enhancement functions in Black Marble processing.

This shows how to use the new simple enhancement functions that replace
the overly complex component system.
"""

import numpy as np
from pathlib import Path

# Import the enhancement functions
from blackmarble.processing import (
    enhance_contrast,
    normalize_data,
    scale_to_byte,
    apply_colormap,
    enhance_roads,
    enhance_urban_features,
)


def demo_contrast_enhancement():
    """Demonstrate contrast enhancement on sample data."""
    print("=== Contrast Enhancement Demo ===")

    # Create sample data with poor contrast
    data = np.random.normal(0.5, 0.1, (100, 100))
    data = np.clip(data, 0.2, 0.8)  # Limit to narrow range

    # Apply different enhancement methods
    enhanced_percentile = enhance_contrast(
        data, method="percentile", clip_percentiles=(2, 98)
    )
    enhanced_minmax = enhance_contrast(data, method="minmax")
    enhanced_manual = enhance_contrast(data, min_value=0.3, max_value=0.7)

    print(f"Original data range: [{data.min():.3f}, {data.max():.3f}]")
    print(
        f"Percentile enhanced range: [{enhanced_percentile.min():.3f}, {enhanced_percentile.max():.3f}]"
    )
    print(
        f"Min-max enhanced range: [{enhanced_minmax.min():.3f}, {enhanced_minmax.max():.3f}]"
    )
    print(
        f"Manual enhanced range: [{enhanced_manual.min():.3f}, {enhanced_manual.max():.3f}]"
    )
    print()


def demo_colormap_application():
    """Demonstrate colormap application for visualization."""
    print("=== Colormap Application Demo ===")

    # Create gradient data
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    data = (X + Y) / 2  # Diagonal gradient

    # Apply different colormaps
    rgb_gray = apply_colormap(data, colormap="grayscale")
    rgb_viridis = apply_colormap(data, colormap="viridis")
    rgb_inferno = apply_colormap(data, colormap="inferno", vmin=0.2, vmax=0.8)

    print(f"Input data shape: {data.shape}")
    print(f"Output RGB shape: {rgb_gray.shape}")
    print(f"Output dtype: {rgb_gray.dtype}")
    print(
        f"Channels: RGB" + (" (3 bands)" if rgb_gray.shape[0] == 3 else "A (4 bands)")
    )
    print()


def demo_road_enhancement():
    """Demonstrate road enhancement for urban areas."""
    print("=== Road Enhancement Demo ===")

    # Create sample NDUI data
    ndui = np.random.uniform(0.3, 0.7, (50, 50))

    # Create sample road network
    roads = np.zeros((50, 50), dtype=int)
    # Add a motorway (type 1) horizontally
    roads[25, :] = 1
    # Add a primary road (type 2) vertically
    roads[:, 25] = 2
    # Add some secondary roads (type 3)
    roads[10:15, 10:40] = 3
    roads[35:40, 10:40] = 3

    # Apply road enhancement
    enhanced = enhance_roads(ndui, roads)

    # Calculate statistics
    motorway_pixels = np.sum(roads == 1)
    primary_pixels = np.sum(roads == 2)
    secondary_pixels = np.sum(roads == 3)

    print(f"Road network statistics:")
    print(f"  Motorway pixels: {motorway_pixels}")
    print(f"  Primary road pixels: {primary_pixels}")
    print(f"  Secondary road pixels: {secondary_pixels}")
    print(f"  Total road pixels: {np.sum(roads > 0)}")
    print()

    print(f"Enhancement results:")
    print(f"  Mean NDUI (original): {ndui.mean():.3f}")
    print(f"  Mean NDUI (enhanced): {enhanced.mean():.3f}")
    print(f"  Mean along motorway: {enhanced[roads == 1].mean():.3f}")
    print(f"  Mean along primary roads: {enhanced[roads == 2].mean():.3f}")
    print()


def demo_urban_feature_enhancement():
    """Demonstrate comprehensive urban feature enhancement."""
    print("=== Urban Feature Enhancement Demo ===")

    # Create sample data
    size = (100, 100)
    ndui = np.random.uniform(0.2, 0.8, size)

    # Create urban core with higher NDUI
    center_y, center_x = 50, 50
    Y, X = np.ogrid[: size[0], : size[1]]
    dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    ndui = ndui + 0.3 * np.exp(-dist_from_center / 20)
    ndui = np.clip(ndui, 0, 1)

    # Create road network radiating from center
    roads = np.zeros(size, dtype=int)
    # Main highways
    roads[48:52, :] = 1  # East-west motorway
    roads[:, 48:52] = 1  # North-south motorway
    # Secondary roads
    roads[30:32, 20:80] = 2
    roads[68:70, 20:80] = 2
    roads[20:80, 30:32] = 2
    roads[20:80, 68:70] = 2

    # Create nighttime lights (correlated with urban core)
    ntl = 0.1 + 0.8 * np.exp(-dist_from_center / 15)
    ntl = np.clip(ntl + np.random.normal(0, 0.1, size), 0, 1)

    # Apply urban feature enhancement
    enhanced = enhance_urban_features(ndui, roads, ntl=ntl, enhancement_strength=0.5)

    print(f"Urban enhancement statistics:")
    print(f"  Original NDUI range: [{ndui.min():.3f}, {ndui.max():.3f}]")
    print(f"  Enhanced NDUI range: [{enhanced.min():.3f}, {enhanced.max():.3f}]")
    print(f"  Mean enhancement factor: {(enhanced.mean() / ndui.mean()):.3f}")
    print()

    # Analyze enhancement by region
    urban_core_mask = dist_from_center < 20
    suburban_mask = (dist_from_center >= 20) & (dist_from_center < 40)
    rural_mask = dist_from_center >= 40

    print(f"Enhancement by region:")
    print(
        f"  Urban core: {enhanced[urban_core_mask].mean():.3f} (was {ndui[urban_core_mask].mean():.3f})"
    )
    print(
        f"  Suburban: {enhanced[suburban_mask].mean():.3f} (was {ndui[suburban_mask].mean():.3f})"
    )
    print(
        f"  Rural: {enhanced[rural_mask].mean():.3f} (was {ndui[rural_mask].mean():.3f})"
    )
    print()


def demo_visualization_pipeline():
    """Demonstrate complete visualization pipeline."""
    print("=== Complete Visualization Pipeline Demo ===")

    # Create sample NDUI data
    size = (200, 200)
    ndui = np.random.uniform(0.1, 0.9, size)

    # Add some structure
    ndui[50:150, 50:150] += 0.2  # Urban area
    ndui[80:120, 80:120] += 0.3  # City center
    ndui = np.clip(ndui, 0, 1)

    # Step 1: Enhance contrast
    enhanced = enhance_contrast(ndui, method="percentile", clip_percentiles=(5, 95))

    # Step 2: Normalize to display range
    normalized = normalize_data(enhanced, output_min=0.1, output_max=0.9)

    # Step 3: Convert to byte for visualization
    byte_data = scale_to_byte(normalized)

    # Step 4: Apply colormap
    rgb = apply_colormap(normalized, colormap="plasma")

    print(f"Visualization pipeline results:")
    print(f"  Input shape: {ndui.shape}, dtype: {ndui.dtype}")
    print(f"  Enhanced range: [{enhanced.min():.3f}, {enhanced.max():.3f}]")
    print(f"  Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    print(
        f"  Byte data range: [{byte_data.min()}, {byte_data.max()}], dtype: {byte_data.dtype}"
    )
    print(f"  RGB output shape: {rgb.shape}, dtype: {rgb.dtype}")
    print()


def main():
    """Run all enhancement demonstrations."""
    print("Black Marble Enhancement Functions Demo")
    print("=" * 50)
    print()

    # Run each demo
    demo_contrast_enhancement()
    demo_colormap_application()
    demo_road_enhancement()
    demo_urban_feature_enhancement()
    demo_visualization_pipeline()

    print("Demo complete!")


if __name__ == "__main__":
    main()
