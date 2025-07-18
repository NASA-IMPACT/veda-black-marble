"""Cloud-Optimized GeoTIFF (COG) generation - SIMPLIFIED."""

import os
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.shutil import copy

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
    nodata: float | None = None,
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
            profile.update(
                {
                    "driver": "GTiff",
                    "tiled": tiled,
                    "blockxsize": blocksize,
                    "blockysize": blocksize,
                    "compress": compress,
                    "BIGTIFF": "IF_SAFER",
                }
            )

            with MemoryFile() as memfile:
                with memfile.open(**profile) as dst:
                    for i in range(1, src.count + 1):
                        dst.write(src.read(i), i)

                    if metadata:
                        dst.update_tags(**metadata)

                    resampling = (
                        Resampling.nearest
                        if np.issubdtype(src.dtypes[0], np.integer)
                        else Resampling.bilinear
                    )
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

            resampling = (
                Resampling.nearest if np.issubdtype(data.dtype, np.integer) else Resampling.bilinear
            )
            dst.build_overviews(overviews, resampling)

    return output_path





