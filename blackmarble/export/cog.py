"""Cloud-Optimized GeoTIFF (COG) generation and export utilities.

This module provides functions for creating COGs with proper tiling,
compression, and overview levels for efficient cloud-based access.
"""

from typing import Any, Literal
import numpy.typing as npt
import numpy as np
import os
import rasterio
from rasterio.crs import CRS as RioCRS
from rasterio.io import MemoryFile
from rasterio.shutil import copy
from rasterio.enums import Resampling
from concurrent.futures import ProcessPoolExecutor, as_completed

# Type aliases
ArrayLike = npt.NDArray[Any]
Transform = Any  # rasterio.transform.Affine
CRS = Any  # rasterio.crs.CRS
BBox = tuple[float, float, float, float]


def create_cog(
    data: ArrayLike | str,
    output_path: str,
    transform: Transform,
    crs: CRS,
    metadata: dict[str, Any] | None = None,
    compress: Literal["deflate", "lzw", "jpeg", "webp", "zstd"] = "deflate",
    tiled: bool = True,
    blocksize: int = 512,
    overviews: list[int] | None = None,
    nodata: float | None = None
) -> str:
    """Create a Cloud-Optimized GeoTIFF from data.
    
    Args:
        data: Input data array or path to existing raster
        output_path: Path for output COG file
        transform: Affine transform for georeferencing
        crs: Coordinate reference system
        metadata: Optional metadata to embed in file
        compress: Compression method. DEFLATE is default because:
            - Read performance: Faster decompression for cloud workflows
            - File size: 10-20% better compression than LZW
            - COG optimization: Better for "write once, read many" pattern
            LZW compresses faster but decompresses slower. Consider LZW
            only if compression time is critical (e.g., real-time processing).
        tiled: Whether to create a tiled TIFF
        blocksize: Tile size for internal blocks
        overviews: Overview levels (default: [2, 4, 8, 16])
        nodata: NoData value
        
    Returns:
        Path to created COG file
    """
    
    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Validate compression choice for scientific data
    # CRITICAL: For Black Marble nighttime lights, we need lossless compression
    # to preserve precise radiance values for scientific analysis
    if compress in ["jpeg", "webp"]:
        raise ValueError(
            f"Compression '{compress}' is lossy and not suitable for scientific data. "
            f"Use 'deflate' or 'lzw' for lossless compression."
        )
    
    # Default overview levels following COG best practices
    # Powers of 2 enable efficient multi-resolution access
    if overviews is None:
        overviews = [2, 4, 8, 16, 32]
    
    # Process input data
    if isinstance(data, str):
        # Input is a file path - convert existing raster to COG
        with rasterio.open(data) as src:
            # Read metadata from source
            profile = src.profile.copy()
            profile.update({
                "driver": "GTiff",
                "tiled": tiled,
                "blockxsize": blocksize,
                "blockysize": blocksize,
                "compress": compress,
                "BIGTIFF": "IF_SAFER",  # Use BigTIFF for large files
                "interleave": "pixel" if src.count > 1 else "band"
            })
            
            # Copy to temporary file with COG structure
            with MemoryFile() as memfile:
                with memfile.open(**profile) as dst:
                    # Copy data band by band
                    for i in range(1, src.count + 1):
                        dst.write(src.read(i), i)
                    
                    # Add metadata tags
                    if metadata:
                        dst.update_tags(**metadata)
                    
                    # Build overviews with appropriate resampling
                    # Use NEAREST for categorical data, BILINEAR for continuous
                    resampling = Resampling.bilinear
                    if np.issubdtype(src.dtypes[0], np.integer):
                        resampling = Resampling.nearest
                    
                    dst.build_overviews(overviews, resampling)
                    
                    # Configure COG-specific options
                    dst.update_tags(ns='rio_overview', resampling=resampling.name)
                
                # Copy to final destination with COG layout
                copy(memfile, output_path, copy_src_overviews=True, **profile)
    
    else:
        # Input is array data
        data = np.asarray(data)
        
        # Determine data type
        if np.issubdtype(data.dtype, np.floating):
            dtype = rasterio.float32 if data.dtype == np.float32 else rasterio.float64
        elif np.issubdtype(data.dtype, np.integer):
            if data.dtype == np.uint8:
                dtype = rasterio.uint8
            elif data.dtype == np.uint16:
                dtype = rasterio.uint16
            elif data.dtype == np.int16:
                dtype = rasterio.int16
            else:
                dtype = rasterio.int32
        else:
            dtype = rasterio.float32
        
        # Handle 2D vs 3D arrays
        if data.ndim == 2:
            count = 1
            height, width = data.shape
        elif data.ndim == 3:
            count, height, width = data.shape
        else:
            raise ValueError(f"Data must be 2D or 3D, got shape {data.shape}")
        
        # Create output profile
        profile = {
            "driver": "GTiff",
            "dtype": dtype,
            "count": count,
            "height": height,
            "width": width,
            "crs": crs,
            "transform": transform,
            "tiled": tiled,
            "blockxsize": blocksize,
            "blockysize": blocksize,
            "compress": compress,
            "nodata": nodata,
            "BIGTIFF": "IF_SAFER",
            "interleave": "pixel" if count > 1 else "band"
        }
        
        # Set compression-specific options for optimal performance
        if compress == "deflate":
            # Predictor 3 for floating point gives ~50% better compression
            # with minimal computational overhead
            if np.issubdtype(data.dtype, np.floating):
                profile["predictor"] = 3
            else:
                profile["predictor"] = 2
            profile["zlevel"] = 6  # Good balance of speed vs compression
        elif compress == "lzw":
            if np.issubdtype(data.dtype, np.floating):
                profile["predictor"] = 3
            else:
                profile["predictor"] = 2
        elif compress == "zstd":
            profile["zstd_level"] = 9  # Higher compression for archival
        
        # Create directory if needed
        # Use abspath to ensure we have a proper directory path
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        
        # Write COG with overviews
        with rasterio.open(output_path, "w", **profile) as dst:
            # Write data
            if data.ndim == 2:
                dst.write(data, 1)
            else:
                for i in range(count):
                    dst.write(data[i], i + 1)
            
            # Add metadata
            if metadata:
                dst.update_tags(**metadata)
            
            # Build overviews
            resampling = Resampling.bilinear
            if np.issubdtype(data.dtype, np.integer):
                resampling = Resampling.nearest
            
            dst.build_overviews(overviews, resampling)
            dst.update_tags(ns='rio_overview', resampling=resampling.name)
    
    return output_path


def export_single_band(
    data: ArrayLike,
    output_path: str,
    transform: Transform,
    crs: CRS,
    band_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    compress: Literal["deflate", "lzw"] = "deflate"
) -> str:
    """Export single-band data as Cloud-Optimized GeoTIFF.
    
    Args:
        data: 2D array of data values
        output_path: Output file path
        transform: Affine transform
        crs: Coordinate reference system
        band_name: Optional name for the band
        metadata: Additional metadata
        compress: Compression method (deflate or lzw only)
        
    Returns:
        Path to created file
    """
    
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")
    
    # Add band name to metadata
    if metadata is None:
        metadata = {}
    
    if band_name:
        metadata["band_1"] = band_name
    
    # Ensure lossless compression
    if compress not in ["deflate", "lzw"]:
        compress = "deflate"
    
    return create_cog(
        data=data,
        output_path=output_path,
        transform=transform,
        crs=crs,
        metadata=metadata,
        compress=compress
    )


def export_multi_band(
    bands: dict[str, ArrayLike],
    output_path: str,
    transform: Transform,
    crs: CRS,
    metadata: dict[str, Any] | None = None,
    compress: Literal["deflate", "lzw"] = "deflate"
) -> str:
    """Export multiple bands as a single Cloud-Optimized GeoTIFF.
    
    Args:
        bands: Dictionary mapping band names to 2D arrays
        output_path: Output file path
        transform: Affine transform
        crs: Coordinate reference system
        metadata: Additional metadata
        compress: Compression method (deflate or lzw only)
        
    Returns:
        Path to created file
    """
    
    if not bands:
        raise ValueError("No bands provided")
    
    # Validate all bands have same shape
    shapes = [b.shape for b in bands.values()]
    if not all(s == shapes[0] for s in shapes):
        raise ValueError(f"All bands must have same shape, got {shapes}")
    
    # Stack bands into 3D array
    band_names = list(bands.keys())
    band_arrays = [np.asarray(bands[name]) for name in band_names]
    stacked = np.stack(band_arrays, axis=0)
    
    # Add band names to metadata
    if metadata is None:
        metadata = {}
    
    for i, name in enumerate(band_names, 1):
        metadata[f"band_{i}"] = name
    
    # Ensure lossless compression
    if compress not in ["deflate", "lzw"]:
        compress = "deflate"
    
    return create_cog(
        data=stacked,
        output_path=output_path,
        transform=transform,
        crs=crs,
        metadata=metadata,
        compress=compress
    )


def create_overviews(
    raster_path: str,
    levels: list[int] | None = None,
    resampling: str = "bilinear"
) -> None:
    """Add overview levels to an existing raster.
    
    Args:
        raster_path: Path to raster file
        levels: Overview levels (default: [2, 4, 8, 16])
        resampling: Resampling method
    """
    
    if levels is None:
        levels = [2, 4, 8, 16, 32]
    
    # Map resampling string to enum
    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
        "mode": Resampling.mode
    }
    
    resampling_enum = resampling_map.get(resampling.lower(), Resampling.bilinear)
    
    # Add overviews
    with rasterio.open(raster_path, "r+") as dst:
        dst.build_overviews(levels, resampling_enum)
        dst.update_tags(ns='rio_overview', resampling=resampling)


def optimize_for_web(
    input_path: str,
    output_path: str,
    tile_size: int = 256,
    compress: Literal["deflate", "lzw"] = "deflate"
) -> str:
    """Optimize a GeoTIFF for web serving as COG.
    
    This function ensures the output is a proper Cloud-Optimized GeoTIFF
    with appropriate tiling, compression, and overview levels for efficient
    web-based access and partial reading.
    
    Args:
        input_path: Path to input raster
        output_path: Path for optimized output
        tile_size: Internal tile size (256 or 512 recommended)
        compress: Compression method
        
    Returns:
        Path to optimized file
    """
    
    with rasterio.open(input_path) as src:
        # Read profile and update for COG
        profile = src.profile.copy()
        
        # Ensure COG-compatible settings
        profile.update({
            "driver": "GTiff",
            "tiled": True,
            "blockxsize": tile_size,
            "blockysize": tile_size,
            "compress": compress,
            "BIGTIFF": "IF_SAFER"
        })
        
        # Set predictor for better compression
        if compress in ["deflate", "lzw"]:
            if np.issubdtype(src.dtypes[0], np.floating):
                profile["predictor"] = 3
            else:
                profile["predictor"] = 2
        
        # Copy with COG layout
        copy(
            src,
            output_path,
            copy_src_overviews=True,
            **profile
        )
        
        # Ensure overviews exist
        with rasterio.open(output_path, "r+") as dst:
            if not dst.overviews(1):
                # Build overviews if missing
                resampling = Resampling.bilinear
                if np.issubdtype(dst.dtypes[0], np.integer):
                    resampling = Resampling.nearest
                
                dst.build_overviews([2, 4, 8, 16, 32], resampling)
    
    return output_path


def batch_export_cogs(
    input_files: dict[str, str],
    output_dir: str,
    transform: Transform,
    crs: CRS,
    compress: Literal["deflate", "lzw"] = "deflate",
    max_workers: int = 4
) -> dict[str, str]:
    """Batch export multiple files as COGs.
    
    Args:
        input_files: Dict mapping output names to input file paths
        output_dir: Directory for output files
        transform: Common transform for all files
        crs: Common CRS for all files
        compress: Compression method
        max_workers: Number of parallel workers
        
    Returns:
        Dict mapping names to output file paths
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare tasks
    tasks = []
    for name, input_path in input_files.items():
        output_path = os.path.join(output_dir, f"{name}.tif")
        tasks.append((name, input_path, output_path))
    
    # Process function for parallel execution
    def process_file(name: str, input_path: str, output_path: str) -> tuple[str, str, str | None]:
        try:
            create_cog(
                data=input_path,
                output_path=output_path,
                transform=transform,
                crs=crs,
                compress=compress
            )
            return name, output_path, None
        except Exception as e:
            return name, output_path, str(e)
    
    # Execute in parallel
    output_files = {}
    
    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_file, name, inp, outp): (name, outp)
                for name, inp, outp in tasks
            }
            
            # Collect results
            errors = []
            for future in as_completed(futures):
                name, output_path = futures[future]
                try:
                    _, _, error = future.result()
                    if error:
                        errors.append(f"{name}: {error}")
                    else:
                        output_files[name] = output_path
                        print(f"Completed: {name}")
                except Exception as e:
                    errors.append(f"{name}: {str(e)}")
        
        if errors:
            print(f"\nErrors encountered:")
            for err in errors:
                print(f"  {err}")
    
    else:
        # Sequential processing
        for name, input_path, output_path in tasks:
            name, output_path, error = process_file(name, input_path, output_path)
            
            if error:
                print(f"Error processing {name}: {error}")
                output_files.pop(name, None)
            else:
                print(f"Completed: {name} -> {output_path}")
    
    return output_files