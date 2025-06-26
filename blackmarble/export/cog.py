"""Cloud-Optimized GeoTIFF (COG) generation - SIMPLIFIED."""

from typing import Any, Literal
import numpy.typing as npt
import numpy as np
import os
import rasterio
from rasterio.io import MemoryFile
from rasterio.shutil import copy
from rasterio.enums import Resampling

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
    """Create a Cloud-Optimized GeoTIFF from data."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    if compress in ["jpeg", "webp"]:
        raise ValueError(f"Lossy compression '{compress}' not suitable for scientific data")
    
    if overviews is None:
        overviews = [2, 4, 8, 16, 32]
    
    if isinstance(data, str):
        # Convert existing file to COG
        with rasterio.open(data) as src:
            profile = src.profile.copy()
            profile.update({
                "driver": "GTiff",
                "tiled": tiled,
                "blockxsize": blocksize,
                "blockysize": blocksize,
                "compress": compress,
                "BIGTIFF": "IF_SAFER",
            })
            
            with MemoryFile() as memfile:
                with memfile.open(**profile) as dst:
                    for i in range(1, src.count + 1):
                        dst.write(src.read(i), i)
                    
                    if metadata:
                        dst.update_tags(**metadata)
                    
                    resampling = Resampling.nearest if np.issubdtype(src.dtypes[0], np.integer) else Resampling.bilinear
                    dst.build_overviews(overviews, resampling)
                
                copy(memfile, output_path, copy_src_overviews=True, **profile)
    else:
        # Create COG from array
        data = np.asarray(data)
        
        if data.ndim == 2:
            count, height, width = 1, data.shape[0], data.shape[1]
        elif data.ndim == 3:
            count, height, width = data.shape
        else:
            raise ValueError(f"Data must be 2D or 3D, got shape {data.shape}")
        
        profile = {
            "driver": "GTiff",
            "dtype": data.dtype,
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
        }
        
        if compress in ["deflate", "lzw"] and np.issubdtype(data.dtype, np.floating):
            profile["predictor"] = 3
        
        with rasterio.open(output_path, "w", **profile) as dst:
            if data.ndim == 2:
                dst.write(data, 1)
            else:
                for i in range(count):
                    dst.write(data[i], i + 1)
            
            if metadata:
                dst.update_tags(**metadata)
            
            resampling = Resampling.nearest if np.issubdtype(data.dtype, np.integer) else Resampling.bilinear
            dst.build_overviews(overviews, resampling)
    
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
    """Export single-band data as COG."""
    if metadata is None:
        metadata = {}
    if band_name:
        metadata["band_name"] = band_name
    
    return create_cog(data, output_path, transform, crs, metadata, compress)


def export_multi_band(
    bands: dict[str, ArrayLike] | list[ArrayLike],
    output_path: str,
    transform: Transform,
    crs: CRS,
    band_names: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    compress: Literal["deflate", "lzw"] = "deflate",
    nodata: float | None = None
) -> str:
    """Export multi-band data as COG.
    
    Args:
        bands: Either a dict mapping band names to arrays or a list of arrays
        output_path: Output file path
        transform: Affine transform
        crs: Coordinate reference system
        band_names: Band names (used if bands is a list)
        metadata: Additional metadata
        compress: Compression method
        nodata: NoData value
        
    Returns:
        Output file path
    """
    if isinstance(bands, dict):
        band_names = list(bands.keys())
        bands_list = list(bands.values())
    else:
        bands_list = bands
    
    data = np.stack(bands_list, axis=0)
    
    if metadata is None:
        metadata = {}
    if band_names:
        for i, name in enumerate(band_names):
            metadata[f"band_{i+1}_name"] = name
    
    return create_cog(data, output_path, transform, crs, metadata, compress, nodata=nodata)


def create_overviews(
    input_path: str,
    overviews: list[int] | None = None,
    resampling: Literal["nearest", "bilinear", "cubic", "average"] = "bilinear"
) -> None:
    """Add overviews to existing GeoTIFF."""
    if overviews is None:
        overviews = [2, 4, 8, 16, 32]
    
    resample_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average
    }
    
    with rasterio.open(input_path, "r+") as dst:
        dst.build_overviews(overviews, resample_map[resampling])


def optimize_for_web(
    input_path: str,
    output_path: str,
    tile_size: int = 256,
    jpeg_quality: int = 85
) -> str:
    """Optimize GeoTIFF for web serving."""
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        
        if src.count == 3 and src.dtypes[0] == rasterio.uint8:
            # RGB imagery - use JPEG compression
            profile.update({
                "driver": "GTiff",
                "tiled": True,
                "blockxsize": tile_size,
                "blockysize": tile_size,
                "compress": "jpeg",
                "jpeg_quality": jpeg_quality,
                "photometric": "rgb"
            })
        else:
            # Scientific data - use DEFLATE
            profile.update({
                "driver": "GTiff", 
                "tiled": True,
                "blockxsize": tile_size,
                "blockysize": tile_size,
                "compress": "deflate",
                "predictor": 3 if np.issubdtype(src.dtypes[0], np.floating) else 2
            })
        
        with rasterio.open(output_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                dst.write(src.read(i), i)
            
            dst.build_overviews([2, 4, 8, 16], Resampling.bilinear)
    
    return output_path


def batch_export_cogs(
    data_paths: list[tuple[str, Transform, CRS]],
    output_dir: str,
    compress: Literal["deflate", "lzw"] = "deflate",
    max_workers: int = 4
) -> list[str]:
    """Export multiple files as COGs in parallel."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    os.makedirs(output_dir, exist_ok=True)
    output_paths = []
    
    def process_file(item):
        path, transform, crs = item
        base_name = os.path.splitext(os.path.basename(path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_cog.tif")
        return create_cog(path, output_path, transform, crs, compress=compress)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, item): item for item in data_paths}
        
        for future in as_completed(futures):
            try:
                result = future.result()
                output_paths.append(result)
                print(f"Created COG: {result}")
            except Exception as e:
                item = futures[future]
                print(f"Failed to process {item[0]}: {e}")
    
    return output_paths