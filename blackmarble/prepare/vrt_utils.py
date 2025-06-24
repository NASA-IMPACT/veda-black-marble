"""VRT utilities for efficient large-scale data processing.

This module provides utilities for creating and working with GDAL Virtual Rasters (VRT),
enabling efficient processing of large datasets without loading everything into memory.
"""

from typing import Any, Callable
import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.windows import Window
from rasterio import Affine
import tempfile
import os

# Type aliases
ArrayLike = npt.NDArray[Any]
Transform = Any  # rasterio.transform.Affine
CRS = Any  # rasterio.crs.CRS
BBox = tuple[float, float, float, float]


def create_landsat_vrt_mosaic(
    band_files: dict[str, list[str]],
    qa_files: list[str] | None = None,
    output_dir: str = "data/temp",
    target_bbox: BBox | None = None,
    target_crs: CRS | None = None
) -> dict[str, str]:
    """Create VRT mosaics for Landsat bands with optional clipping.
    
    Args:
        band_files: Dict mapping band names to file lists
        qa_files: List of QA files
        output_dir: Directory for VRT files
        target_bbox: Optional bbox to clip to (in target_crs)
        target_crs: Optional target CRS for reprojection
        
    Returns:
        Dict mapping band names to VRT file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    vrt_paths = {}
    
    # Create VRT for each band
    for band_name, files in band_files.items():
        if not files:
            continue
            
        vrt_path = os.path.join(output_dir, f"{band_name}_mosaic.vrt")
        
        # Import here to avoid circular dependency
        from . import create_virtual_raster
        
        # Create VRT with optional extent clipping
        vrt_paths[band_name] = create_virtual_raster(
            file_paths=files,
            output_vrt=vrt_path,
            extent=target_bbox,
            crs=target_crs
        )
    
    # Create VRT for QA if provided
    if qa_files:
        vrt_paths["QA"] = create_virtual_raster(
            file_paths=qa_files,
            output_vrt=os.path.join(output_dir, "QA_mosaic.vrt"),
            extent=target_bbox,
            crs=target_crs
        )
    
    return vrt_paths


def process_vrt_by_chunks(
    vrt_path: str,
    process_func: Callable[[ArrayLike], ArrayLike],
    output_path: str,
    chunk_size: int = 2048,
    overlap: int = 0,
    dtype: str | None = None
) -> None:
    """Process a VRT in chunks to handle very large datasets.
    
    Args:
        vrt_path: Path to VRT file
        process_func: Function to apply to each chunk
        output_path: Output file path
        chunk_size: Size of chunks to process
        overlap: Overlap between chunks (for filters that need context)
        dtype: Output data type
    """
    with rasterio.open(vrt_path) as src:
        # Prepare output
        profile = src.profile.copy()
        if dtype:
            profile['dtype'] = dtype
            
        # Calculate chunks
        height, width = src.shape
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            # Process in chunks
            for row_off in range(0, height, chunk_size - overlap):
                for col_off in range(0, width, chunk_size - overlap):
                    # Calculate window
                    row_end = min(row_off + chunk_size, height)
                    col_end = min(col_off + chunk_size, width)
                    
                    window = Window(col_off, row_off, 
                                  col_end - col_off, 
                                  row_end - row_off)
                    
                    # Read chunk
                    chunk = src.read(1, window=window)
                    
                    # Process chunk
                    processed = process_func(chunk)
                    
                    # Handle overlap by trimming edges
                    if overlap > 0 and (row_off > 0 or col_off > 0):
                        trim_top = overlap if row_off > 0 else 0
                        trim_left = overlap if col_off > 0 else 0
                        processed = processed[trim_top:, trim_left:]
                        
                        # Adjust window
                        window = Window(
                            col_off + trim_left,
                            row_off + trim_top,
                            processed.shape[1],
                            processed.shape[0]
                        )
                    
                    # Write processed chunk
                    dst.write(processed, 1, window=window)


def merge_vrts_with_function(
    vrt_paths: list[str],
    merge_func: Callable[[list[ArrayLike]], ArrayLike],
    output_path: str,
    window_size: int = 2048
) -> str:
    """Merge multiple VRTs using a custom function (e.g., median).
    
    Args:
        vrt_paths: List of VRT file paths
        merge_func: Function to merge arrays (e.g., np.nanmedian)
        output_path: Output file path
        window_size: Size of windows to process
        
    Returns:
        Path to merged output
    """
    if not vrt_paths:
        raise ValueError("No VRT paths provided")
    
    # Open all VRTs
    datasets = [rasterio.open(vrt) for vrt in vrt_paths]
    
    try:
        # Use first dataset as reference
        ref_ds = datasets[0]
        profile = ref_ds.profile.copy()
        
        # Ensure all datasets have same shape
        for ds in datasets[1:]:
            if ds.shape != ref_ds.shape:
                raise ValueError(f"Shape mismatch: {ds.shape} vs {ref_ds.shape}")
        
        # Process by windows
        with rasterio.open(output_path, 'w', **profile) as dst:
            for row_off in range(0, ref_ds.height, window_size):
                for col_off in range(0, ref_ds.width, window_size):
                    # Calculate window
                    window = Window(
                        col_off, row_off,
                        min(window_size, ref_ds.width - col_off),
                        min(window_size, ref_ds.height - row_off)
                    )
                    
                    # Read from all datasets
                    chunks = []
                    for ds in datasets:
                        chunk = ds.read(1, window=window)
                        chunks.append(chunk)
                    
                    # Apply merge function
                    merged = merge_func(chunks)
                    
                    # Write result
                    dst.write(merged, 1, window=window)
                    
    finally:
        # Close all datasets
        for ds in datasets:
            ds.close()
    
    return output_path


def create_temporal_vrt_stack(
    file_lists: list[list[str]],
    band_name: str,
    output_dir: str = "data/temp"
) -> str:
    """Create a VRT stack for temporal compositing.
    
    Args:
        file_lists: List of file lists (one per time step)
        band_name: Band identifier
        output_dir: Output directory
        
    Returns:
        Path to temporal stack VRT
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # First create VRT for each time step
    time_vrts = []
    for i, files in enumerate(file_lists):
        if files:
            vrt_path = os.path.join(output_dir, f"{band_name}_t{i:03d}.vrt")
            from . import create_virtual_raster
            time_vrts.append(
                create_virtual_raster(files, vrt_path)
            )
    
    # Then stack them as a multi-band VRT
    stack_vrt = os.path.join(output_dir, f"{band_name}_temporal_stack.vrt")
    
    # This would require GDAL Python bindings to create a proper stack
    # For now, return list of VRTs
    return time_vrts