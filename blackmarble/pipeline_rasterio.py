"""
Example of pipeline.py modifications to use rasterio for mosaicing.
This file demonstrates the changes needed - not a complete replacement.
"""

import os


def mosaic_viirs_with_rasterio(
    ntl_tiles: list[str],
    lunar_tiles: list[str],
    quality_tiles: list[str],
    temp_dir: str,
    use_rasterio: bool = True,
) -> tuple[str, str, str]:
    """
    Demonstrate how to replace GDAL mosaic operations with rasterio.

    This function shows the migration path for lines 806-827 in pipeline.py.
    """

    if use_rasterio and len(ntl_tiles) > 1:
        print("Mosaicing VIIRS tiles using rasterio...")
        try:
            from blackmarble.mosaic_rasterio import mosaic_viirs_tiles

            # Use the specialized VIIRS mosaic function
            from typing import cast
            from pathlib import Path
            ntl_temp_tif, lunar_temp_tif, quality_temp_tif = mosaic_viirs_tiles(
                cast(list[str | Path], ntl_tiles),
                cast(list[str | Path], lunar_tiles),
                cast(list[str | Path], quality_tiles),
                temp_dir
            )

            print("Successfully mosaiced using rasterio")
            return ntl_temp_tif, lunar_temp_tif, quality_temp_tif

        except Exception as e:
            print(f"Rasterio mosaic failed: {e}. Falling back to GDAL...")
            use_rasterio = False

    if not use_rasterio and len(ntl_tiles) > 1:
        # Original GDAL code
        from osgeo import gdal

        print("Mosaicing VIIRS tiles using GDAL...")
        ntl_temp_tif = os.path.join(temp_dir, "ntl_raw_mosaic.tif")
        gdal.Warp(ntl_temp_tif, ntl_tiles, format="GTiff", resampleAlg="bilinear")

        lunar_temp_tif = os.path.join(temp_dir, "lunar_irradiance_mosaic.tif")
        gdal.Warp(lunar_temp_tif, lunar_tiles, format="GTiff", resampleAlg="bilinear")

        quality_temp_tif = os.path.join(temp_dir, "mandatory_quality_flag_mosaic.tif")
        gdal.Warp(
            quality_temp_tif, quality_tiles, format="GTiff", resampleAlg="nearest"
        )

        return ntl_temp_tif, lunar_temp_tif, quality_temp_tif

    else:
        # Single tile, use directly
        return ntl_tiles[0], lunar_tiles[0], quality_tiles[0]


def resample_final_with_rasterio(
    inputs: list[str],
    outputs: list[str],
    desired_res: tuple[float, float],
    use_rasterio: bool = True,
) -> None:
    """
    Demonstrate how to replace GDAL resample operations with rasterio.

    This function shows the migration path for lines 913-932 in pipeline.py.
    """

    print(f"Resampling {len(inputs)} files to resolution {desired_res}")

    for i, (inp, outp) in enumerate(zip(inputs, outputs)):
        print(f"\nResampling file {i + 1}/{len(inputs)}: {os.path.basename(inp)}")

        if use_rasterio:
            try:
                from blackmarble.mosaic_rasterio import resample_to_resolution

                # Use rasterio-based resampling
                resample_to_resolution(inp, outp, resolution=desired_res)
                print(f"  Successfully resampled using rasterio to {outp}")

            except Exception as e:
                print(f"  Rasterio resample failed: {e}. Falling back to GDAL...")

                # Fall back to GDAL
                from osgeo import gdal

                warp_opts = gdal.WarpOptions(
                    format="GTiff", xRes=desired_res[0], yRes=desired_res[1]
                )
                result = gdal.Warp(outp, inp, options=warp_opts)

                if result is None:
                    print(f"  ERROR: Failed to resample {inp}!")
                else:
                    print(f"  Successfully resampled using GDAL")
        else:
            # Use GDAL directly
            from osgeo import gdal

            warp_opts = gdal.WarpOptions(
                format="GTiff", xRes=desired_res[0], yRes=desired_res[1]
            )
            result = gdal.Warp(outp, inp, options=warp_opts)

            if result is None:
                print(f"  ERROR: Failed to resample {inp}!")
            else:
                print(f"  Successfully resampled to {outp}")


# Example of how to integrate into run_pipeline():
"""
# In run_pipeline(), around line 806:

# Check for feature flag
use_rasterio_mosaic = os.environ.get("USE_RASTERIO_MOSAIC", "false").lower() == "true"

if use_rasterio_mosaic:
    print("INFO: Using rasterio for mosaic operations")

# Replace the mosaic section with:
ntl_temp_tif, lunar_temp_tif, quality_temp_tif = mosaic_viirs_with_rasterio(
    ntl_tiles, lunar_tiles, quality_tiles, temp_dir, use_rasterio=use_rasterio_mosaic
)

# Similarly for final resampling around line 895:
resample_final_with_rasterio(
    inputs, outputs_final, desired_res, use_rasterio=use_rasterio_mosaic
)
"""
