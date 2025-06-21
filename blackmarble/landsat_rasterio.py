"""
Rasterio-based Landsat processing functions.

Drop-in replacements for GDAL-based operations in landsat.py.
"""

import os
from typing import List, Tuple, Union
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.crs import CRS


def process_band_data(
    band: Union[int, str],
    year: int,
    month: int,
    day: int,
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    output_prefix: str,
    temp_dir: str,
    composite_months: int = 12,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Download, warp, and merge source + QA into multi-scene TIFFs using rasterio.
    
    This is a drop-in replacement for the GDAL-based version.
    """
    # Import here to avoid circular imports
    from .landsat import download_tiles_for_band_with_composites
    
    composites: List[Tuple[List[str], List[str], List[str], List[str]]] = (
        download_tiles_for_band_with_composites(
            band, year, month, day, lat1, lon1, lat2, lon2, temp_dir, composite_months
        )
    )
    
    outputs: List[str] = []
    pixel_tifs: List[str] = []
    radsat_tifs: List[str] = []
    aerosol_tifs: List[str] = []
    
    for idx, (srcs, pxs, rxs, axs) in enumerate(composites):
        # Skip processing if no source files were found for this composite period
        if not srcs:
            print(
                f"Skipping processing for composite index {idx} due to missing source files."
            )
            continue

        print(f"Merging base {idx}")
        outb: str = f"{output_prefix}{idx}.TIFF"
        
        # Merge source band files using rasterio
        _merge_rasters(
            srcs, 
            outb,
            resampling=Resampling.bilinear,
            compress='lzw',
            tiled=True
        )
        outputs.append(outb)

        # Process QA bands
        for name, tifs, output_list in (
            ("P", pxs, pixel_tifs),
            ("R", rxs, radsat_tifs),
            ("A", axs, aerosol_tifs),
        ):
            # Skip QA processing if no corresponding QA files were found
            if not tifs:
                print(
                    f"Skipping processing for QA type {name}, index {idx} due to missing files."
                )
                # Append a placeholder or handle downstream logic if needed
                output_list.append("")  # Or None, depending on how it's handled
                continue

            print(f"Merging {name} {idx}")
            outf: str = f"{output_prefix}{idx}{name}.TIFF"
            
            # Merge QA bands with nearest neighbor resampling
            _merge_rasters(
                tifs,
                outf,
                resampling=Resampling.nearest,  # Use nearest for QA flags
                compress='lzw',
                tiled=True,
                nodata=0  # Set nodata to 0 for QA bands
            )
            output_list.append(outf)

    # Filter out empty placeholders if added
    pixel_tifs = [p for p in pixel_tifs if p]
    radsat_tifs = [r for r in radsat_tifs if r]
    aerosol_tifs = [a for a in aerosol_tifs if a]

    return outputs, pixel_tifs, radsat_tifs, aerosol_tifs


def _merge_rasters(
    input_files: List[str],
    output_file: str,
    resampling: Resampling = Resampling.bilinear,
    compress: str = 'lzw',
    tiled: bool = True,
    nodata: Union[int, float, None] = None
) -> None:
    """
    Merge multiple raster files into a single output file using rasterio.
    
    Args:
        input_files: List of input file paths
        output_file: Output file path
        resampling: Resampling method
        compress: Compression type
        tiled: Whether to tile the output
        nodata: NoData value for the output
    """
    # Open all source files
    src_files_to_mosaic = []
    for fp in input_files:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)
    
    try:
        # Merge the rasters
        mosaic, out_trans = merge(
            src_files_to_mosaic,
            resampling=resampling,
            nodata=nodata
        )
        
        # Get metadata from first file and update
        out_meta = src_files_to_mosaic[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "compress": compress,
            "tiled": tiled,
            "blockxsize": 256 if tiled else None,
            "blockysize": 256 if tiled else None,
        })
        
        if nodata is not None:
            out_meta["nodata"] = nodata
            
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Write the mosaic
        with rasterio.open(output_file, "w", **out_meta) as dest:
            dest.write(mosaic)
            
    finally:
        # Close all source files
        for src in src_files_to_mosaic:
            src.close()