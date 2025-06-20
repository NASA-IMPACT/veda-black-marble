"""GDAL raster I/O adapter implementation."""

import logging
from typing import Tuple, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np
from osgeo import gdal, osr, gdal_array
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import Affine

from .base import RasterAdapter, RasterIOError

logger = logging.getLogger(__name__)

# Enable GDAL exceptions
gdal.UseExceptions()


class GDALRasterAdapter:
    """GDAL/Rasterio adapter for raster I/O operations.

    Implements the RasterAdapter interface using GDAL and rasterio
    for comprehensive raster data handling.
    """

    def __init__(self, use_rasterio: bool = True):
        """Initialize raster adapter.

        Args:
            use_rasterio: Whether to prefer rasterio over pure GDAL
        """
        self.use_rasterio = use_rasterio

    def read(
        self, path: Path, band: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Read raster data and metadata.

        Args:
            path: Path to raster file
            band: Optional band number (1-indexed)

        Returns:
            Tuple of (data array, metadata dict)

        Raises:
            RasterIOError: If read fails
        """
        if not path.exists():
            raise RasterIOError(f"File not found: {path}")

        try:
            if self.use_rasterio:
                return self._read_with_rasterio(path, band)
            else:
                return self._read_with_gdal(path, band)
        except Exception as e:
            raise RasterIOError(f"Failed to read {path}: {e}")

    def _read_with_rasterio(
        self, path: Path, band: Optional[int]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Read using rasterio."""
        with rasterio.open(path) as src:
            # Read data
            if band is not None:
                data = src.read(band)
            else:
                data = src.read()
                if data.shape[0] == 1:
                    data = data[0]  # Remove single band dimension

            # Extract metadata
            metadata = {
                "crs": str(src.crs) if src.crs else None,
                "transform": src.transform,
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": str(src.dtypes[0]),
                "nodata": src.nodata,
                "bounds": src.bounds,
                "driver": src.driver,
                "profile": src.profile.copy(),
            }

            logger.debug(
                f"Read {path} with rasterio: shape={data.shape}, dtype={data.dtype}"
            )
            return data, metadata

    def _read_with_gdal(
        self, path: Path, band: Optional[int]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Read using GDAL."""
        dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
        if dataset is None:
            raise RasterIOError(f"GDAL failed to open {path}")

        try:
            # Get georeferencing info
            transform = dataset.GetGeoTransform()
            projection = dataset.GetProjection()

            # Read data
            if band is not None:
                band_obj = dataset.GetRasterBand(band)
                data = band_obj.ReadAsArray()
                nodata = band_obj.GetNoDataValue()
            else:
                # Read all bands
                data = dataset.ReadAsArray()
                if dataset.RasterCount == 1:
                    nodata = dataset.GetRasterBand(1).GetNoDataValue()
                else:
                    nodata = None

            # Build metadata
            metadata = {
                "crs": projection,
                "transform": Affine.from_gdal(*transform),
                "width": dataset.RasterXSize,
                "height": dataset.RasterYSize,
                "count": dataset.RasterCount,
                "dtype": str(data.dtype),
                "nodata": nodata,
                "driver": dataset.GetDriver().ShortName,
                "gdal_transform": transform,
            }

            # Calculate bounds
            west = transform[0]
            north = transform[3]
            east = west + transform[1] * dataset.RasterXSize
            south = north + transform[5] * dataset.RasterYSize
            metadata["bounds"] = (west, south, east, north)

            logger.debug(
                f"Read {path} with GDAL: shape={data.shape}, dtype={data.dtype}"
            )
            return data, metadata

        finally:
            dataset = None  # Close dataset

    def write(
        self,
        path: Path,
        data: np.ndarray,
        metadata: Dict[str, Any],
        compress: bool = True,
    ) -> None:
        """Write raster data to file.

        Args:
            path: Output file path
            data: Array data to write
            metadata: Raster metadata (CRS, transform, etc.)
            compress: Whether to compress output

        Raises:
            RasterIOError: If write fails
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            if self.use_rasterio:
                self._write_with_rasterio(path, data, metadata, compress)
            else:
                self._write_with_gdal(path, data, metadata, compress)

            logger.debug(f"Wrote raster to {path}: shape={data.shape}")

        except Exception as e:
            raise RasterIOError(f"Failed to write {path}: {e}")

    def _write_with_rasterio(
        self, path: Path, data: np.ndarray, metadata: Dict[str, Any], compress: bool
    ) -> None:
        """Write using rasterio."""
        # Build profile from metadata
        if "profile" in metadata:
            profile = metadata["profile"].copy()
        else:
            profile = {
                "driver": "GTiff",
                "width": metadata.get("width", data.shape[-1]),
                "height": metadata.get("height", data.shape[-2]),
                "count": 1 if data.ndim == 2 else data.shape[0],
                "dtype": data.dtype,
                "crs": metadata.get("crs"),
                "transform": metadata.get("transform"),
                "nodata": metadata.get("nodata"),
            }

        # Update compression
        if compress:
            profile.update(
                {
                    "compress": "deflate",
                    "predictor": 2 if "int" in str(data.dtype) else 3,
                }
            )

        # Ensure data has correct shape
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data)

    def _write_with_gdal(
        self, path: Path, data: np.ndarray, metadata: Dict[str, Any], compress: bool
    ) -> None:
        """Write using GDAL."""
        # Determine driver and options
        driver = gdal.GetDriverByName("GTiff")
        options = []
        if compress:
            options.extend(["COMPRESS=DEFLATE", "PREDICTOR=2"])

        # Get dimensions
        if data.ndim == 2:
            bands, height, width = 1, data.shape[0], data.shape[1]
            data = data[np.newaxis, ...]
        else:
            bands, height, width = data.shape

        # Create dataset
        dataset = driver.Create(
            str(path),
            width,
            height,
            bands,
            gdal_array.NumericTypeCodeToGDALTypeCode(data.dtype),
            options,
        )

        if dataset is None:
            raise RasterIOError(f"Failed to create dataset at {path}")

        try:
            # Set projection and transform
            if "crs" in metadata and metadata["crs"]:
                dataset.SetProjection(str(metadata["crs"]))

            if "gdal_transform" in metadata:
                dataset.SetGeoTransform(metadata["gdal_transform"])
            elif "transform" in metadata:
                # Convert from Affine to GDAL transform
                t = metadata["transform"]
                gdal_transform = (t.c, t.a, t.b, t.f, t.d, t.e)
                dataset.SetGeoTransform(gdal_transform)

            # Write bands
            for i in range(bands):
                band = dataset.GetRasterBand(i + 1)
                band.WriteArray(data[i])

                if "nodata" in metadata and metadata["nodata"] is not None:
                    band.SetNoDataValue(float(metadata["nodata"]))

                band.FlushCache()

        finally:
            dataset = None  # Close dataset

    def get_info(self, path: Path) -> Dict[str, Any]:
        """Get raster information without reading data.

        Args:
            path: Path to raster file

        Returns:
            Dictionary with raster info

        Raises:
            RasterIOError: If file cannot be accessed
        """
        if not path.exists():
            raise RasterIOError(f"File not found: {path}")

        try:
            with rasterio.open(path) as src:
                info = {
                    "driver": src.driver,
                    "width": src.width,
                    "height": src.height,
                    "count": src.count,
                    "crs": str(src.crs) if src.crs else None,
                    "bounds": src.bounds,
                    "transform": src.transform,
                    "dtype": [str(dt) for dt in src.dtypes],
                    "nodata": src.nodata,
                    "shape": (src.count, src.height, src.width),
                    "size_bytes": path.stat().st_size,
                }

                # Add band descriptions if available
                if src.descriptions:
                    info["band_descriptions"] = list(src.descriptions)

                return info

        except Exception as e:
            raise RasterIOError(f"Failed to get info for {path}: {e}")

    def reproject(
        self,
        source_path: Path,
        target_crs: str,
        output_path: Optional[Path] = None,
        resolution: Optional[Tuple[float, float]] = None,
        resampling: str = "bilinear",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reproject raster to different CRS.

        Args:
            source_path: Input raster path
            target_crs: Target coordinate reference system
            output_path: Optional output path
            resolution: Optional output resolution (x, y)
            resampling: Resampling method

        Returns:
            Tuple of (reprojected data, updated metadata)

        Raises:
            RasterIOError: If reprojection fails
        """
        try:
            # Map resampling methods
            resampling_methods = {
                "nearest": Resampling.nearest,
                "bilinear": Resampling.bilinear,
                "cubic": Resampling.cubic,
                "average": Resampling.average,
                "mode": Resampling.mode,
            }
            resample_method = resampling_methods.get(resampling, Resampling.bilinear)

            with rasterio.open(source_path) as src:
                # Calculate transform for target CRS
                transform, width, height = calculate_default_transform(
                    src.crs,
                    target_crs,
                    src.width,
                    src.height,
                    *src.bounds,
                    resolution=resolution,
                )

                # Update metadata
                kwargs = src.meta.copy()
                kwargs.update(
                    {
                        "crs": target_crs,
                        "transform": transform,
                        "width": width,
                        "height": height,
                    }
                )

                # Reproject
                if output_path:
                    # Reproject to file
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with rasterio.open(output_path, "w", **kwargs) as dst:
                        for i in range(1, src.count + 1):
                            reproject(
                                source=rasterio.band(src, i),
                                destination=rasterio.band(dst, i),
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=transform,
                                dst_crs=target_crs,
                                resampling=resample_method,
                            )

                    # Read back the reprojected data
                    return self.read(output_path)
                else:
                    # Reproject to memory
                    data = np.empty((src.count, height, width), dtype=src.dtypes[0])

                    for i in range(src.count):
                        reproject(
                            source=src.read(i + 1),
                            destination=data[i],
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_crs,
                            resampling=resample_method,
                        )

                    if src.count == 1:
                        data = data[0]

                    return data, kwargs

        except Exception as e:
            raise RasterIOError(f"Failed to reproject {source_path}: {e}")
