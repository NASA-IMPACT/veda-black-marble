"""
Rasterio-based preprocessing components.
"""

import numpy as np
from typing import Optional
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling as ResamplingEnum
from rasterio import MemoryFile
import rasterio


def resample_array(
    data: np.ndarray,
    new_width: int,
    new_height: int,
    transform: Affine = None,
    nodata: Optional[float] = None,
    method: str = 'bilinear',
) -> np.ndarray:
    """
    Resample a 2D array using rasterio.
    
    This provides consistent behavior with all other rasterio operations
    in the codebase.
    """
    # Map method names to rasterio resampling methods
    resample_map = {
        'nearest': Resampling.nearest,
        'bilinear': Resampling.bilinear,
        'cubic': Resampling.cubic,
        'average': Resampling.average,
    }
    resample_alg = resample_map.get(method, Resampling.bilinear)
    
    # Use MemoryFile for in-memory operations
    height, width = data.shape
    
    # Create source profile
    if transform is None:
        transform = Affine.identity()
    
    profile = {
        'driver': 'GTiff',
        'width': width,
        'height': height,
        'count': 1,
        'dtype': data.dtype,
        'transform': transform,
        'nodata': nodata,
    }
    
    # Write source to memory and read back resampled
    with MemoryFile() as memfile:
        with memfile.open(**profile) as src:
            src.write(data, 1)
            
            # Read with resampling
            out_shape = (1, new_height, new_width)
            
            # Calculate scale factors for window reading
            scale_x = width / new_width
            scale_y = height / new_height
            
            # Read the whole dataset resampled to new dimensions
            resampled = src.read(
                out_shape=out_shape,
                resampling=resample_alg,
            )[0]  # Get the first (and only) band
            
    return resampled