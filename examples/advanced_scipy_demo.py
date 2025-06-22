#!/usr/bin/env python
"""Demonstration of advanced scipy features in Black Marble temporal processing.

This script shows how to use the scipy-based features for:
1. Advanced gap filling with nearest neighbor interpolation
2. Feather blending for seamless scene merging
"""

import numpy as np
from blackmarble.components.temporal import TemporalCompositor, SceneMerger


def demo_gap_filling():
    """Demonstrate scipy-based gap filling."""
    print("\n=== Gap Filling Demo ===")

    # Create synthetic time series data with gaps
    # Simulating 5 time steps with some missing data
    data = np.array(
        [
            [[10, 20, np.nan, 40], [50, np.nan, np.nan, 80], [90, 100, 110, np.nan]],
            [[np.nan, 22, 33, np.nan], [55, 66, 77, 88], [np.nan, 102, np.nan, 122]],
            [[11, np.nan, 33, 44], [np.nan, 66, np.nan, 88], [99, np.nan, 111, 122]],
            [[12, 23, 34, 45], [56, 67, 78, 89], [100, 101, 112, 123]],
            [[13, 24, np.nan, 46], [57, np.nan, 79, 90], [101, 102, 113, np.nan]],
        ]
    )

    print(f"Input shape: {data.shape} (5 time steps, 3x4 pixels)")
    print(f"Total gaps: {np.sum(np.isnan(data))}")

    # Create compositor without scipy gap filling
    compositor_basic = TemporalCompositor(
        method="median", use_scipy_fill_gaps=False, min_valid_observations=1
    )

    # Create compositor with scipy gap filling
    compositor_scipy = TemporalCompositor(
        method="median", use_scipy_fill_gaps=True, min_valid_observations=1
    )

    # Process with both methods
    result_basic = compositor_basic.process(data)
    result_scipy = compositor_scipy.process(data)

    if result_basic.metadata is None or "composite" not in result_basic.metadata:
        raise ValueError("Basic compositor did not return composite in metadata")
    if result_scipy.metadata is None or "composite" not in result_scipy.metadata:
        raise ValueError("Scipy compositor did not return composite in metadata")

    composite_basic = result_basic.metadata["composite"]
    composite_scipy = result_scipy.metadata["composite"]

    print("\nBasic method - remaining gaps:", np.sum(np.isnan(composite_basic)))
    print("Scipy method - remaining gaps:", np.sum(np.isnan(composite_scipy)))

    # Show a specific example
    print("\nExample pixel [0, 2] across time:")
    print(f"  Time series: {data[:, 0, 2]}")
    print(f"  Basic composite: {composite_basic[0, 2]}")
    print(f"  Scipy composite: {composite_scipy[0, 2]} (filled with nearest neighbor)")


def demo_feather_blending():
    """Demonstrate feather blending for scene merging."""
    print("\n\n=== Feather Blending Demo ===")

    # Create two overlapping scenes with different values
    scene1 = np.ones((10, 10)) * 100  # Western scene
    scene2 = np.ones((10, 10)) * 200  # Eastern scene

    # Add some texture to make blending more visible
    scene1[:, -3:] = 150  # Gradient at edge
    scene2[:, :3] = 150  # Gradient at edge

    # Create transforms (scene2 is offset by 7 pixels east)
    from rasterio.transform import Affine

    transform1 = Affine(1, 0, 0, 0, -1, 10)
    transform2 = Affine(1, 0, 7, 0, -1, 10)

    bounds1 = (0, 0, 10, 10)
    bounds2 = (7, 0, 17, 10)

    print(f"Scene 1: {scene1.shape}, bounds {bounds1}")
    print(f"Scene 2: {scene2.shape}, bounds {bounds2}")
    print(f"Overlap region: x=7 to x=10")

    # Test different blending methods
    for method, distance in [("max", None), ("mean", None), ("feather", 5)]:
        if method == "feather":
            print(f"\nMethod: {method} (distance={distance} pixels)")
            merger = SceneMerger(blend_method=method, feather_distance=distance)
        else:
            print(f"\nMethod: {method}")
            merger = SceneMerger(blend_method=method)

        merged, _, _ = merger.merge_scenes(
            [scene1, scene2], [transform1, transform2], [bounds1, bounds2]
        )

        # Show values in overlap region
        overlap_values = merged[5, 7:10]
        print(f"  Overlap values at row 5: {overlap_values}")
        print(f"  Unique values in overlap: {np.unique(merged[:, 7:10])[:5]}...")


def main():
    """Run all demonstrations."""
    print("Black Marble Advanced Features Demo")
    print("====================================")

    import scipy

    print(f"scipy version: {scipy.__version__}")

    demo_gap_filling()
    demo_feather_blending()

    print("\n\nFor more information, see the module documentation:")
    print("  from blackmarble.components import temporal")
    print("  help(temporal)")


if __name__ == "__main__":
    main()
