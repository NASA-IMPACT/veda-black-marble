#!/usr/bin/env python3

import rasterio
import numpy as np
import argparse


def compare_raster_metadata(src1, src2, file1_path, file2_path):
    """Prints a comparison of metadata for two rasterio dataset readers."""
    print(f"--- Metadata Comparison ---")
    print(f"\nFile 1: {file1_path}")
    print(f"  Shape: {src1.shape}")
    print(f"  Band count: {src1.count}")
    print(f"  CRS: {src1.crs}")
    print(f"  Transform: \n{src1.transform}")
    print(f"  Dtypes: {src1.dtypes}")
    print(f"  Nodata values: {src1.nodatavals}")

    print(f"\nFile 2: {file2_path}")
    print(f"  Shape: {src2.shape}")
    print(f"  Band count: {src2.count}")
    print(f"  CRS: {src2.crs}")
    print(f"  Transform: \n{src2.transform}")
    print(f"  Dtypes: {src2.dtypes}")
    print(f"  Nodata values: {src2.nodatavals}")

    print("\nSummary of Metadata Differences:")
    shape_match = src1.shape == src2.shape
    count_match = src1.count == src2.count
    crs_match = src1.crs == src2.crs
    transform_match = src1.transform == src2.transform
    dtypes_match = src1.dtypes == src2.dtypes

    print(f"  Shapes match: {'Yes' if shape_match else 'No'}")
    print(f"  Band counts match: {'Yes' if count_match else 'No'}")
    print(f"  CRSs match: {'Yes' if crs_match else 'No'}")
    print(f"  Transforms match: {'Yes' if transform_match else 'No'}")
    if not dtypes_match:
        print(f"  Dtypes match: No")
        for i in range(min(src1.count, src2.count)):
            if src1.dtypes[i] != src2.dtypes[i]:
                print(
                    f"    Band {i + 1}: File1 dtype '{src1.dtypes[i]}', File2 dtype '{src2.dtypes[i]}'"
                )
    else:
        print(f"  Dtypes match: Yes")

    return shape_match and count_match


def compare_rasters(file1_path, file2_path, tolerance=1e-6, diff_output_path=None):
    """
    Compares two raster files, prints metadata and data differences,
    and optionally saves a difference raster.
    """
    try:
        with rasterio.open(file1_path) as src1, rasterio.open(file2_path) as src2:
            metadata_compatible = compare_raster_metadata(
                src1, src2, file1_path, file2_path
            )

            if not metadata_compatible:
                print(
                    "\nMetadata (shape or band count) differs significantly. Skipping pixel data comparison."
                )
                return

            if src1.crs != src2.crs or src1.transform != src2.transform:
                print(
                    "\nWarning: CRS or Transforms differ. Pixel comparison might be misleading."
                )

            print(f"\n--- Pixel Data Comparison (Tolerance: {tolerance}) ---")
            for band_idx in range(1, src1.count + 1):
                print(f"\nComparing Band {band_idx}:")

                arr1 = src1.read(band_idx, masked=True)
                arr2 = src2.read(band_idx, masked=True)

                # Convert to float64 for consistent subtraction
                arr1_float = arr1.astype(np.float64)
                arr2_float = arr2.astype(np.float64)

                # Difference array (MaskedArray)
                # The mask of diff_arr will be the union of arr1_float.mask and arr2_float.mask
                diff_arr = arr1_float - arr2_float

                if diff_arr.mask is np.ma.nomask:  # All pixels are valid (not masked)
                    num_valid_pixels = diff_arr.size
                else:
                    # Count only unmasked pixels
                    num_valid_pixels = np.sum(~diff_arr.mask)

                if num_valid_pixels == 0:
                    print("  No valid (unmasked) pixels to compare in this band.")
                    continue

                print(
                    f"  Number of valid (unmasked) pixels for comparison: {num_valid_pixels}"
                )

                absolute_diff = np.abs(diff_arr)
                # Pixels where the absolute difference is greater than the tolerance
                # .filled(False) ensures masked values in `absolute_diff` are not counted as differing
                differing_pixels_map = absolute_diff.filled(np.nan) > tolerance
                num_diff_pixels = np.sum(differing_pixels_map)

                percent_diff_pixels = (
                    (num_diff_pixels / num_valid_pixels) * 100
                    if num_valid_pixels > 0
                    else 0
                )

                print(
                    f"  Number of differing pixels: {num_diff_pixels} ({percent_diff_pixels:.2f}%)"
                )

                if (
                    num_diff_pixels > 0 or True
                ):  # Always print stats if valid pixels exist
                    # Stats for all valid (unmasked) differences
                    min_val_diff = diff_arr.min()
                    max_val_diff = diff_arr.max()
                    mean_val_diff = diff_arr.mean()
                    mean_abs_val_diff = (
                        absolute_diff.mean()
                    )  # Mean of absolute differences
                    std_val_diff = diff_arr.std()

                    print(f"  Min difference: {min_val_diff}")
                    print(f"  Max difference: {max_val_diff}")
                    print(f"  Mean difference: {mean_val_diff}")
                    print(f"  Mean absolute difference: {mean_abs_val_diff}")
                    print(f"  Std dev of difference: {std_val_diff}")

                if num_diff_pixels == 0 and num_valid_pixels > 0:
                    print("  All valid pixels are within tolerance.")

                # Save difference raster for the first band if path provided
                if diff_output_path and band_idx == 1:
                    print(
                        f"  Saving difference for band {band_idx} to {diff_output_path}"
                    )
                    profile = src1.profile.copy()
                    profile.update(dtype=rasterio.float64, count=1)

                    # Choose a nodata value for the output difference map
                    diff_nodata = -9999.0
                    profile["nodata"] = diff_nodata

                    with rasterio.open(diff_output_path, "w", **profile) as dst:
                        # .filled() replaces masked values with fill_value
                        dst.write(diff_arr.filled(diff_nodata).astype(np.float64), 1)

    except FileNotFoundError as e:
        print(f"Error: {e}. Please check file paths.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare two raster files and report differences.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("file1", help="Path to the first raster file.")
    parser.add_argument("file2", help="Path to the second raster file.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Tolerance for comparing float values. For integer comparison, use 0. Default: 1e-6.",
    )
    parser.add_argument(
        "--diff_output",
        help="Optional. Path to save the difference raster (first band only if multi-band).",
    )

    args = parser.parse_args()

    print(f"Comparing {args.file1} and {args.file2}\n")
    compare_rasters(args.file1, args.file2, args.tolerance, args.diff_output)
