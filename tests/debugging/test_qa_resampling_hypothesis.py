#!/usr/bin/env python3

import os
import tempfile
import shutil
import numpy as np
import rasterio
from osgeo import gdal
from typing import Optional # Required for Optional type hint

def create_sample_qa_tiff(filepath: str, width: int, height: int, values: np.ndarray, nodata_val: Optional[int] = None) -> None:
    """Creates a simple single-band integer GeoTIFF."""
    profile = {
        'driver': 'GTiff',
        'dtype': values.dtype.name,
        'nodata': nodata_val,
        'width': width,
        'height': height,
        'count': 1,
        'crs': rasterio.crs.CRS.from_epsg(4326), # Dummy CRS
        'transform': rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0) # Dummy transform
    }
    with rasterio.open(filepath, 'w', **profile) as dst:
        dst.write(values.astype(profile['dtype']), 1)
    print(f"Created sample QA TIFF: {filepath} with values:\n{values}")

def main():
    """
    Tests the effect of 'bilinear' vs 'near' resampling on a sample QA TIFF,
    focusing on metadata (dtype) and preservation of integer values.
    """
    temp_dir = tempfile.mkdtemp(prefix="qa_resample_test_")
    print(f"Using temporary directory: {temp_dir}")

    try:
        # 1. Create a sample QA TIFF
        sample_qa_data = np.array([
            [1, 1, 2, 2],
            [1, 10, 20, 2],
            [3, 30, 40, 4],
            [3, 3, 4, 4]
        ], dtype=np.uint8)
        nodata_value = 0 # Define a NoData value for the sample
        input_qa_path = os.path.join(temp_dir, "sample_qa_input.tif")
        create_sample_qa_tiff(input_qa_path, sample_qa_data.shape[1], sample_qa_data.shape[0], sample_qa_data, nodata_val=nodata_value)

        # Define output paths
        output_bilinear_path = os.path.join(temp_dir, "qa_warped_bilinear.tif")
        output_near_path = os.path.join(temp_dir, "qa_warped_near.tif")

        # Common GDAL Warp options: upscale slightly to ensure interpolation effects for bilinear
        target_width = int(sample_qa_data.shape[1] * 1.5)
        target_height = int(sample_qa_data.shape[0] * 1.5)

        common_warp_options = {
            "format": "GTiff",
            "width": target_width,
            "height": target_height,
            "srcNodata": nodata_value,
            "dstNodata": nodata_value
        }

        # 2. Warp with resampleAlg='bilinear'
        print(f"\nWarping {input_qa_path} to {output_bilinear_path} using 'bilinear'...")
        warp_opts_bilinear = gdal.WarpOptions(resampleAlg='bilinear', **common_warp_options)
        gdal.Warp(output_bilinear_path, input_qa_path, options=warp_opts_bilinear)
        print("Bilinear warp complete.")

        # 3. Warp with resampleAlg='near'
        print(f"\nWarping {input_qa_path} to {output_near_path} using 'near'...")
        warp_opts_near = gdal.WarpOptions(resampleAlg='near', **common_warp_options)
        gdal.Warp(output_near_path, input_qa_path, options=warp_opts_near)
        print("Near warp complete.")

        # 4. Analyze the outputs
        print("\n--- Analysis of Warped QA Outputs ---")

        print(f"\nAnalysis of 'bilinear' resampled output ({output_bilinear_path}):")
        with rasterio.open(output_bilinear_path) as src_bilinear:
            data_bilinear = src_bilinear.read(1, masked=True)
            print(f"  Output Dtype: {src_bilinear.dtypes[0]}")
            
            # Check if values are still integers (ignoring masked NoData values)
            unmasked_bilinear_values = data_bilinear[~data_bilinear.mask] if data_bilinear.mask is not np.ma.nomask else data_bilinear.flatten()
            if unmasked_bilinear_values.size > 0:
                are_bilinear_values_integer = np.array_equal(unmasked_bilinear_values, unmasked_bilinear_values.astype(int))
                if not are_bilinear_values_integer:
                    print("  WARNING: Bilinear output contains non-integer (interpolated) values where data is present.")
                    non_integer_examples = unmasked_bilinear_values[unmasked_bilinear_values != unmasked_bilinear_values.astype(int)][:5]
                    print(f"    Examples of non-integer values: {non_integer_examples}")
                else:
                    print("  INFO: Bilinear output values appear to be integers (after potential rounding/truncation by GDAL/rasterio).")
            else:
                print("  INFO: No unmasked values to check for integer property in bilinear output.")


        print(f"\nAnalysis of 'near' resampled output ({output_near_path}):")
        with rasterio.open(output_near_path) as src_near:
            data_near = src_near.read(1, masked=True)
            print(f"  Output Dtype: {src_near.dtypes[0]}")

            unmasked_near_values = data_near[~data_near.mask] if data_near.mask is not np.ma.nomask else data_near.flatten()
            if unmasked_near_values.size > 0:
                are_near_values_integer = np.array_equal(unmasked_near_values, unmasked_near_values.astype(int))
                if are_near_values_integer:
                    print("  INFO: Near output values are integers, as expected.")
                else:
                    print("  WARNING: Near output (surprisingly) contains non-integer values.")

                # Check if unique values in 'near' output are a subset of original (excluding nodata)
                original_unique_values = np.unique(sample_qa_data[sample_qa_data != nodata_value])
                near_output_unique_values = np.unique(unmasked_near_values)
                
                print(f"  Original unique QA values (excluding nodata): {original_unique_values}")
                print(f"  'Near' resampled unique QA values (excluding nodata): {near_output_unique_values}")
                
                if set(near_output_unique_values).issubset(set(original_unique_values)):
                    print("  INFO: Unique values in 'near' output are a subset of original values (good).")
                else:
                    newly_introduced = set(near_output_unique_values) - set(original_unique_values)
                    print(f"  WARNING: 'Near' output introduced new values not in the original QA flags: {newly_introduced} (bad).")
            else:
                print("  INFO: No unmasked values to check for integer property or unique values in near output.")

        print("\n--- Conclusion ---")
        print("If 'bilinear' output shows a float dtype or non-integer values, it confirms that")
        print("bilinear resampling is unsuitable for categorical QA data as it interpolates")
        print("and alters the discrete flag values.")
        print("'Near' resampling should preserve the original integer values and their categories.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    gdal.UseExceptions()
    main()