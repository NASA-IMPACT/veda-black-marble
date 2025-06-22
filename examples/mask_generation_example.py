"""Example usage of the new mask generation functions in processing.py

This demonstrates how the simplified mask functions replace the complex
component system that always returns MaskResult objects.
"""

import numpy as np
from blackmarble.processing import (
    create_cloud_mask,
    create_quality_mask,
    create_aerosol_mask,
    create_saturation_mask,
    create_water_mask,
    create_viirs_quality_mask,
    combine_masks,
    dilate_mask,
)


def example_cloud_masking():
    """Example: Create cloud mask from Landsat QA_PIXEL band."""
    print("=== Cloud Masking Example ===")

    # Simulate QA_PIXEL data with various cloud conditions
    qa_pixel = np.array(
        [
            [21824, 21824, 21952, 55052],  # Clear land, clear land, clear water, cloud
            [22280, 54596, 21824, 21824],  # Low conf cloud, medium conf cloud, clear
            [54852, 55052, 21952, 21824],  # High conf clouds, clear water, clear land
        ]
    )

    # Conservative strategy (only clear pixels)
    conservative_mask = create_cloud_mask(qa_pixel, conservative=True)
    print(f"Conservative mask:\n{conservative_mask}")
    print(f"Valid pixels: {np.sum(conservative_mask)} / {conservative_mask.size}")

    # Permissive strategy with dilation
    permissive_mask = create_cloud_mask(qa_pixel, conservative=False, dilate_pixels=1)
    print(f"\nPermissive mask with 1-pixel dilation:\n{permissive_mask}")
    print(f"Valid pixels: {np.sum(permissive_mask)} / {permissive_mask.size}")


def example_quality_composite():
    """Example: Create composite quality mask from multiple QA bands."""
    print("\n=== Composite Quality Mask Example ===")

    # Simulate multiple QA bands
    qa_bands = {
        "qa_pixel": np.array(
            [
                [21824, 55052, 21824],
                [21824, 21824, 54596],
            ]
        ),
        "qa_aerosol": np.array(
            [
                [0, 192, 0],  # Low, medium, low aerosol
                [0, 0, 194],  # Low, low, medium aerosol
            ]
        ),
        "qa_radsat": np.array(
            [
                [0, 4, 0],  # No sat, band 3 sat, no sat
                [0, 0, 8],  # No sat, no sat, band 4 sat
            ]
        ),
    }

    # Create composite mask with all quality checks
    composite_mask = create_quality_mask(
        qa_bands,
        combine_method="and",
        cloud_strategy="conservative",
        aerosol_threshold="standard",
        include_saturation=True,
        saturation_bands=[3, 4],  # Check bands 3 and 4 for saturation
    )

    print(f"Composite quality mask:\n{composite_mask}")
    print(f"Valid pixels: {np.sum(composite_mask)} / {composite_mask.size}")

    # Show which pixels failed which test
    cloud_mask = create_cloud_mask(qa_bands["qa_pixel"])
    aerosol_mask = create_aerosol_mask(qa_bands["qa_aerosol"])
    sat3_mask = create_saturation_mask(qa_bands["qa_radsat"], band=3)
    sat4_mask = create_saturation_mask(qa_bands["qa_radsat"], band=4)

    print("\nIndividual mask results:")
    print(f"Cloud mask:\n{cloud_mask}")
    print(f"Aerosol mask:\n{aerosol_mask}")
    print(f"Band 3 saturation mask:\n{sat3_mask}")
    print(f"Band 4 saturation mask:\n{sat4_mask}")


def example_water_masking():
    """Example: Create water mask from NDWI."""
    print("\n=== Water Masking Example ===")

    # Simulate NDWI values (positive = water, negative = land)
    ndwi = np.array(
        [
            [-0.5, -0.2, 0.0, 0.1],
            [0.3, -0.1, 0.5, -0.3],
        ]
    )

    # Create water mask (True = land, False = water)
    water_mask = create_water_mask(ndwi, threshold=0.0)
    print(f"NDWI values:\n{ndwi}")
    print(f"Water mask (True=land):\n{water_mask}")
    print(f"Land pixels: {np.sum(water_mask)} / {water_mask.size}")


def example_viirs_quality():
    """Example: Create quality mask for VIIRS data."""
    print("\n=== VIIRS Quality Masking Example ===")

    # Simulate VIIRS quality flags
    quality_flag = np.array(
        [
            [0, 1, 2, 3],  # High quality, unknown, moderate, poor
            [2, 4, 0, 5],  # Moderate, poor, high quality, poor
        ]
    )

    # Use default valid flags (0 and 2)
    viirs_mask = create_viirs_quality_mask(quality_flag)
    print(f"Quality flags:\n{quality_flag}")
    print(f"Quality mask (default flags [0,2]):\n{viirs_mask}")

    # Use custom valid flags (only highest quality)
    strict_mask = create_viirs_quality_mask(quality_flag, valid_flags=[0])
    print(f"Strict quality mask (only flag 0):\n{strict_mask}")


def example_mask_combination():
    """Example: Combine multiple masks with different methods."""
    print("\n=== Mask Combination Example ===")

    # Create some sample masks
    mask1 = np.array([[True, True, False], [False, True, True]])
    mask2 = np.array([[True, False, True], [True, True, False]])
    mask3 = np.array([[True, True, True], [False, False, True]])

    print(f"Mask 1:\n{mask1}")
    print(f"Mask 2:\n{mask2}")
    print(f"Mask 3:\n{mask3}")

    # Combine with AND (all must be valid)
    and_result = combine_masks([mask1, mask2, mask3], method="and")
    print(f"\nAND combination (all must be True):\n{and_result}")

    # Combine with OR (any can be valid)
    or_result = combine_masks([mask1, mask2, mask3], method="or")
    print(f"\nOR combination (any can be True):\n{or_result}")


def compare_with_old_components():
    """Show how the new functions compare to the old component system."""
    print("\n=== Comparison with Old Component System ===")

    print("OLD WAY (using components):")
    print("```python")
    print("# Everything returns MaskResult objects")
    print("cloud_masker = CloudMasker(strategy='conservative')")
    print("result = cloud_masker.process(qa_pixel)")
    print("mask = result.mask  # Extract mask from MaskResult")
    print("stats = result.statistics  # Complex statistics object")
    print("```")

    print("\nNEW WAY (using functions):")
    print("```python")
    print("# Functions return exactly what you need")
    print("mask = create_cloud_mask(qa_pixel, conservative=True)")
    print("# That's it! Just a boolean numpy array")
    print("```")

    print("\nBenefits of the new approach:")
    print("- Simpler: Functions return masks directly, not wrapper objects")
    print("- Clearer: Function names say what they return")
    print("- Testable: Easy to test individual functions")
    print("- Composable: Easy to combine masks with standard numpy operations")
    print("- No hidden state: Pure functions with no side effects")


if __name__ == "__main__":
    example_cloud_masking()
    example_quality_composite()
    example_water_masking()
    example_viirs_quality()
    example_mask_combination()
    compare_with_old_components()
