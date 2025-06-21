"""
Rasterio-based mosaic utilities for replacing GDAL mosaic operations.
"""

import os
from typing import List, Union, Optional
from pathlib import Path
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio import warp
from rasterio.crs import CRS


def mosaic_tiles(
    input_tiles: List[Union[str, Path]],
    output_path: Union[str, Path],
    resampling: str = "bilinear",
    nodata: Optional[float] = None,
    dtype: Optional[str] = None,
    bounds: Optional[tuple] = None,
    res: Optional[float] = None,
) -> None:
    """
    Mosaic multiple raster tiles into a single output file using rasterio.
    
    This function replaces gdal.Warp() for mosaicing operations.
    
    Args:
        input_tiles: List of input raster file paths
        output_path: Output raster file path
        resampling: Resampling method ('bilinear', 'nearest', 'cubic', etc.)
        nodata: NoData value for output (if None, uses first input's nodata)
        dtype: Output data type (if None, uses first input's dtype)
        bounds: Optional output bounds (minx, miny, maxx, maxy)
        res: Optional output resolution
    
    Raises:
        ValueError: If no input tiles provided or resampling method invalid
        RuntimeError: If mosaic operation fails
    """
    if not input_tiles:
        raise ValueError("No input tiles provided for mosaicing")
    
    # Convert resampling string to Resampling enum
    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "cubic_spline": Resampling.cubic_spline,
        "lanczos": Resampling.lanczos,
        "average": Resampling.average,
        "mode": Resampling.mode,
        "gauss": Resampling.gauss,
    }
    
    if resampling not in resampling_map:
        raise ValueError(f"Invalid resampling method: {resampling}")
    
    resampling_enum = resampling_map[resampling]
    
    # Open all input datasets
    datasets = []
    for tile_path in input_tiles:
        if not os.path.exists(tile_path):
            print(f"Warning: Tile {tile_path} not found, skipping")
            continue
        datasets.append(rasterio.open(tile_path))
    
    if not datasets:
        raise RuntimeError("No valid input tiles found")
    
    try:
        # Perform the merge
        mosaic, out_transform = merge(
            datasets,
            bounds=bounds,
            res=res,
            nodata=nodata,
            dtype=dtype,
            method='first',  # Use first valid pixel
            resampling=resampling_enum
        )
        
        # Get output profile from first dataset
        out_profile = datasets[0].profile.copy()
        
        # Update profile with merge results
        out_profile.update({
            'driver': 'GTiff',
            'height': mosaic.shape[1],
            'width': mosaic.shape[2],
            'transform': out_transform,
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512,
        })
        
        if dtype:
            out_profile['dtype'] = dtype
        
        if nodata is not None:
            out_profile['nodata'] = nodata
        
        # Write the mosaic
        with rasterio.open(output_path, 'w', **out_profile) as dst:
            dst.write(mosaic)
            
    finally:
        # Close all datasets
        for ds in datasets:
            ds.close()


def mosaic_viirs_tiles(
    ntl_tiles: List[Union[str, Path]],
    lunar_tiles: List[Union[str, Path]], 
    quality_tiles: List[Union[str, Path]],
    output_dir: Union[str, Path],
) -> tuple[str, str, str]:
    """
    Mosaic VIIRS NTL, lunar irradiance, and quality flag tiles.
    
    This is a specialized function for VIIRS data that handles the different
    resampling requirements for each layer type.
    
    Args:
        ntl_tiles: List of NTL (nighttime lights) tile paths
        lunar_tiles: List of lunar irradiance tile paths
        quality_tiles: List of quality flag tile paths
        output_dir: Directory for output files
        
    Returns:
        Tuple of (ntl_mosaic_path, lunar_mosaic_path, quality_mosaic_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mosaic NTL tiles with bilinear resampling
    ntl_output = output_dir / "ntl_raw_mosaic.tif"
    mosaic_tiles(
        ntl_tiles,
        ntl_output,
        resampling="bilinear",
        nodata=65535  # VIIRS fill value
    )
    
    # Mosaic lunar irradiance with bilinear resampling
    lunar_output = output_dir / "lunar_irradiance_mosaic.tif"
    mosaic_tiles(
        lunar_tiles,
        lunar_output,
        resampling="bilinear",
        nodata=65535  # VIIRS fill value
    )
    
    # Mosaic quality flags with nearest neighbor (preserve exact values)
    quality_output = output_dir / "mandatory_quality_flag_mosaic.tif"
    mosaic_tiles(
        quality_tiles,
        quality_output,
        resampling="nearest",  # Critical: preserve flag values
        dtype="uint16"  # Ensure integer type for flags
    )
    
    return str(ntl_output), str(lunar_output), str(quality_output)


def resample_to_resolution(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    resolution: Union[float, tuple[float, float]],
    resampling: str = "bilinear",
) -> None:
    """
    Resample a raster to a specific resolution.
    
    This function replaces gdal.Warp() for resolution changes.
    
    Args:
        input_path: Input raster file path
        output_path: Output raster file path
        resolution: Target resolution (single value or (xres, yres) tuple)
        resampling: Resampling method (default: 'bilinear')
    """
    # Handle resolution parameter
    if isinstance(resolution, (int, float)):
        xres, yres = resolution, -resolution
    else:
        xres, yres = resolution[0], -abs(resolution[1])
    
    # Get resampling enum
    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "cubic_spline": Resampling.cubic_spline,
        "lanczos": Resampling.lanczos,
        "average": Resampling.average,
    }
    
    resampling_enum = resampling_map.get(resampling, Resampling.bilinear)
    
    with rasterio.open(input_path) as src:
        # Calculate new dimensions
        bounds = src.bounds
        new_width = int((bounds.right - bounds.left) / xres)
        new_height = int((bounds.top - bounds.bottom) / abs(yres))
        
        # Create new transform
        new_transform = Affine(xres, 0, bounds.left, 0, yres, bounds.top)
        
        # Update profile
        profile = src.profile.copy()
        profile.update({
            'width': new_width,
            'height': new_height,
            'transform': new_transform,
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
        })
        
        # Perform resampling
        with rasterio.open(output_path, 'w', **profile) as dst:
            for band_idx in range(1, src.count + 1):
                # Read and resample each band
                warp.reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=new_transform,
                    dst_crs=src.crs,
                    resampling=resampling_enum,
                    num_threads=4,
                )