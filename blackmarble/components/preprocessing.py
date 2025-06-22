"""Preprocessing components for spatial operations.

This module provides components for coordinate system transformations,
resampling, cropping, and spatial alignment of raster data.
"""

from typing import Any
from dataclasses import dataclass
import numpy as np
from osgeo import gdal, osr
from pyproj import Proj, Transformer
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import Affine

from .base import ComposableComponent, MaskResult, QualityStatistics


@dataclass
class SpatialResult:
    """Result of a spatial preprocessing operation.

    Attributes:
        data: Processed array data
        transform: Affine transformation matrix
        crs: Coordinate reference system
        bounds: (min_x, min_y, max_x, max_y) in CRS units
        metadata: Additional metadata about the operation
    """

    data: np.ndarray
    transform: Affine
    crs: str
    bounds: tuple[float, float, float, float]
    metadata: dict[str, Any] | None = None


class ProjectionHandler(ComposableComponent):
    """Handle coordinate reference system transformations.

    Supports transformations between different coordinate systems including
    geographic (lat/lon) and projected (UTM, etc.) systems.
    """

    def __init__(
        self,
        name: str = "projection_handler",
        target_crs: str | None = None,
        resampling: str = "bilinear",
    ):
        """Initialize projection handler.

        Args:
            name: Component identifier
            target_crs: Target CRS (e.g., "EPSG:4326" for WGS84)
            resampling: Resampling method (nearest, bilinear, cubic)
        """
        super().__init__(name)
        self.target_crs = target_crs
        self.resampling = resampling
        self._config.update({"target_crs": target_crs, "resampling": resampling})

    def process(self, data: np.ndarray, **kwargs) -> MaskResult:
        """Transform data to target CRS.

        Args:
            data: Input raster data
            source_crs: Source coordinate reference system
            source_transform: Source affine transform
            source_bounds: Optional source bounds

        Returns:
            MaskResult with success mask (1.0 for all valid pixels)
        """
        # Extract spatial metadata
        source_crs = kwargs.get("source_crs")
        source_transform = kwargs.get("source_transform")

        if not source_crs or not source_transform:
            raise ValueError("source_crs and source_transform required")

        if not self.target_crs:
            raise ValueError("target_crs not specified")

        # Get resampling method
        resampling_map = {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
            "average": Resampling.average,
        }
        resample_method = resampling_map.get(self.resampling, Resampling.bilinear)

        # Calculate target transform
        height, width = data.shape[-2:]
        bounds = kwargs.get("source_bounds")
        if not bounds:
            # Calculate bounds from transform
            west = source_transform.c
            north = source_transform.f
            east = west + source_transform.a * width
            south = north + source_transform.e * height
            bounds = (west, south, east, north)

        # Calculate transform for target CRS
        result = calculate_default_transform(
            source_crs, self.target_crs, width, height, *bounds
        )
        dst_transform = result[0]
        assert result[1] is not None and result[2] is not None
        dst_width = int(result[1])
        dst_height = int(result[2])

        # Create output array
        if data.ndim == 2:
            dst_data = np.empty((dst_height, dst_width), dtype=data.dtype)
            reproject(
                source=data,
                destination=dst_data,
                src_transform=source_transform,
                src_crs=source_crs,
                dst_transform=dst_transform,
                dst_crs=self.target_crs,
                resampling=resample_method,
            )
        else:
            # Handle multi-band data
            dst_data = np.empty(
                (data.shape[0], dst_height, dst_width), dtype=data.dtype
            )
            for i in range(data.shape[0]):
                reproject(
                    source=data[i],
                    destination=dst_data[i],
                    src_transform=source_transform,
                    src_crs=source_crs,
                    dst_transform=dst_transform,
                    dst_crs=self.target_crs,
                    resampling=resample_method,
                )

        # Create mask for valid pixels (non-nodata)
        nodata = kwargs.get("nodata", np.nan)
        if np.isnan(nodata):
            mask = (~np.isnan(dst_data)).astype(float)
        else:
            mask = (dst_data != nodata).astype(float)

        if data.ndim > 2:
            mask = np.all(mask, axis=0).astype(float)

        # Calculate new bounds
        west = dst_transform.c
        north = dst_transform.f
        east = west + dst_transform.a * dst_width
        south = north + dst_transform.e * dst_height
        new_bounds = (west, south, east, north)

        # Store transformed data in kwargs for downstream use
        kwargs["transformed_data"] = dst_data
        kwargs["transform"] = dst_transform
        kwargs["crs"] = self.target_crs
        kwargs["bounds"] = new_bounds

        # Generate statistics
        stats = QualityStatistics.from_mask(mask)

        metadata = {
            "source_crs": str(source_crs),
            "target_crs": self.target_crs,
            "original_shape": data.shape,
            "transformed_shape": dst_data.shape,
            "resampling": self.resampling,
            "bounds": new_bounds,
        }

        return MaskResult(mask=mask, statistics=stats.__dict__, metadata=metadata)


class ResolutionResampler(ComposableComponent):
    """Handle multi-resolution resampling operations.

    Resamples raster data to different spatial resolutions while
    preserving data quality and spatial alignment.
    """

    def __init__(
        self,
        name: str = "resolution_resampler",
        target_resolution: tuple[float, float] | None = None,
        method: str = "bilinear",
    ):
        """Initialize resolution resampler.

        Args:
            name: Component identifier
            target_resolution: Target resolution (x_res, y_res) in CRS units
            method: Resampling method
        """
        super().__init__(name)
        self.target_resolution = target_resolution
        self.method = method
        self._config.update({"target_resolution": target_resolution, "method": method})

    def process(self, data: np.ndarray, **kwargs) -> MaskResult:
        """Resample data to target resolution.

        Args:
            data: Input raster data
            transform: Current affine transform
            crs: Coordinate reference system

        Returns:
            MaskResult with resampling success mask
        """
        if not self.target_resolution:
            raise ValueError("target_resolution not specified")

        transform = kwargs.get("transform")
        crs = kwargs.get("crs")

        if not transform:
            raise ValueError("transform required for resampling")

        # Current resolution
        current_res = (abs(transform.a), abs(transform.e))

        # Calculate scaling factors
        scale_x = current_res[0] / self.target_resolution[0]
        scale_y = current_res[1] / self.target_resolution[1]

        # New dimensions
        height, width = data.shape[-2:]
        new_width = int(width * scale_x)
        new_height = int(height * scale_y)

        # Use GDAL for resampling
        nodata = kwargs.get("nodata", np.nan)

        # First create a mask of valid pixels in the original data
        if np.isnan(nodata):
            original_valid_mask = (~np.isnan(data)).astype(float)
        else:
            original_valid_mask = (data != nodata).astype(float)

        # For multi-band data, combine masks
        if data.ndim > 2:
            original_valid_mask = np.all(original_valid_mask, axis=0).astype(float)

        # Resample the data
        if data.ndim == 2:
            resampled = self._resample_array(
                data, new_width, new_height, transform, nodata
            )
        else:
            # Handle multi-band
            resampled = np.empty(
                (data.shape[0], new_height, new_width), dtype=data.dtype
            )
            for i in range(data.shape[0]):
                resampled[i] = self._resample_array(
                    data[i], new_width, new_height, transform, nodata
                )

        # Resample the mask using average method to get proportion of valid pixels
        resampled_mask = self._resample_array(
            original_valid_mask,
            new_width,
            new_height,
            transform,
            None,
            method="average",
        )
        # Conservative approach: require ALL source pixels to be valid
        # Average resampling gives exact proportion of valid pixels
        # A value of 1.0 means all contributing pixels were valid
        mask = (resampled_mask >= 0.999).astype(float)

        # Update transform
        new_transform = Affine(
            self.target_resolution[0] * (1 if transform.a > 0 else -1),
            transform.b,
            transform.c,
            transform.d,
            self.target_resolution[1] * (1 if transform.e > 0 else -1),
            transform.f,
        )

        # Store resampled data
        kwargs["resampled_data"] = resampled
        kwargs["transform"] = new_transform
        kwargs["resolution"] = self.target_resolution

        # Generate statistics
        stats = QualityStatistics.from_mask(mask)

        metadata = {
            "original_resolution": current_res,
            "target_resolution": self.target_resolution,
            "scale_factors": (scale_x, scale_y),
            "original_shape": data.shape,
            "resampled_shape": resampled.shape,
            "method": self.method,
        }

        return MaskResult(mask=mask, statistics=stats.__dict__, metadata=metadata)

    def _resample_array(
        self,
        data: np.ndarray,
        new_width: int,
        new_height: int,
        transform: Affine | None = None,
        nodata: float | None = None,
        method: str | None = None,
    ) -> np.ndarray:
        """Resample a 2D array using scipy/rasterio."""
        # Import here to allow gradual migration
        try:
            from .preprocessing_rasterio import resample_array

            use_method = method or self.method
            if transform is not None:
                return resample_array(
                    data, new_width, new_height, transform, nodata, use_method
                )
            else:
                # Use identity transform if none provided
                from rasterio.transform import Affine as RioAffine
                identity = RioAffine.identity()
                return resample_array(
                    data, new_width, new_height, identity, nodata, use_method
                )
        except ImportError:
            # Fall back to GDAL implementation
            return self._resample_array_gdal(
                data, new_width, new_height, transform, nodata, method
            )

    def _resample_array_gdal(
        self,
        data: np.ndarray,
        new_width: int,
        new_height: int,
        transform: Affine | None = None,
        nodata: float | None = None,
        method: str | None = None,
    ) -> np.ndarray:
        """Original GDAL implementation kept as fallback."""
        # Create in-memory dataset
        driver = gdal.GetDriverByName("MEM")
        src_ds = driver.Create("", data.shape[1], data.shape[0], 1, gdal.GDT_Float32)

        # Set geotransform if provided
        if transform is not None:
            gt = (
                transform.c,
                transform.a,
                transform.b,
                transform.f,
                transform.d,
                transform.e,
            )
            src_ds.SetGeoTransform(gt)

        src_band = src_ds.GetRasterBand(1)
        src_band.WriteArray(data)

        # Set nodata value if provided
        if nodata is not None and not np.isnan(nodata):
            src_band.SetNoDataValue(float(nodata))

        # Create output dataset
        dst_ds = driver.Create("", new_width, new_height, 1, gdal.GDT_Float32)

        # Calculate output geotransform
        if transform is not None:
            scale_x = data.shape[1] / new_width
            scale_y = data.shape[0] / new_height
            dst_gt = (
                transform.c,
                transform.a * scale_x,
                transform.b,
                transform.f,
                transform.d,
                transform.e * scale_y,
            )
            dst_ds.SetGeoTransform(dst_gt)

        dst_band = dst_ds.GetRasterBand(1)
        if nodata is not None and not np.isnan(nodata):
            dst_band.SetNoDataValue(float(nodata))

        # Map resampling methods
        resample_map = {
            "nearest": gdal.GRA_NearestNeighbour,
            "bilinear": gdal.GRA_Bilinear,
            "cubic": gdal.GRA_Cubic,
            "average": gdal.GRA_Average,
        }
        # Use provided method or fall back to instance method
        use_method = method or self.method
        resample_alg = resample_map.get(use_method, gdal.GRA_Bilinear)

        # Perform resampling
        gdal.ReprojectImage(src_ds, dst_ds, None, None, resample_alg)

        # Read result
        result = dst_band.ReadAsArray()

        # Cleanup
        src_ds = None
        dst_ds = None

        return result


class BoundsCropper(ComposableComponent):
    """Crop rasters to specified spatial bounds.

    Handles spatial subsetting with proper alignment and
    optional buffer zones.
    """

    def __init__(
        self,
        name: str = "bounds_cropper",
        bounds: tuple[float, float, float, float] | None = None,
        buffer: float = 0.0,
        buffer_units: str = "crs",
    ):
        """Initialize bounds cropper.

        Args:
            name: Component identifier
            bounds: Target bounds (min_x, min_y, max_x, max_y)
            buffer: Buffer distance to add around bounds
            buffer_units: Units for buffer ("crs" or "pixels")
        """
        super().__init__(name)
        self.bounds = bounds
        self.buffer = buffer
        self.buffer_units = buffer_units
        self._config.update(
            {"bounds": bounds, "buffer": buffer, "buffer_units": buffer_units}
        )

    def process(self, data: np.ndarray, **kwargs) -> MaskResult:
        """Crop data to specified bounds.

        Args:
            data: Input raster data
            transform: Affine transform

        Returns:
            MaskResult indicating cropped area
        """
        if not self.bounds:
            raise ValueError("bounds not specified")

        transform = kwargs.get("transform")
        if not transform:
            raise ValueError("transform required for cropping")

        # Calculate original bounds from transform and data shape
        height, width = data.shape[-2:]
        west = transform.c
        north = transform.f
        east = west + transform.a * width
        south = north + transform.e * height
        original_bounds = (west, south, east, north)

        # Apply buffer to bounds
        min_x, min_y, max_x, max_y = self.bounds
        if self.buffer > 0:
            if self.buffer_units == "crs":
                min_x -= self.buffer
                min_y -= self.buffer
                max_x += self.buffer
                max_y += self.buffer
            else:  # pixels
                pixel_buffer_x = self.buffer * abs(transform.a)
                pixel_buffer_y = self.buffer * abs(transform.e)
                min_x -= pixel_buffer_x
                min_y -= pixel_buffer_y
                max_x += pixel_buffer_x
                max_y += pixel_buffer_y

        # Convert bounds to pixel coordinates
        inv_transform = ~transform
        col_min, row_max = inv_transform * (min_x, min_y)
        col_max, row_min = inv_transform * (max_x, max_y)

        # Round to integer pixels
        col_min = max(0, int(np.floor(col_min)))
        col_max = min(data.shape[-1], int(np.ceil(col_max)))
        row_min = max(0, int(np.floor(row_min)))
        row_max = min(data.shape[-2], int(np.ceil(row_max)))

        # Crop data
        if data.ndim == 2:
            cropped = data[row_min:row_max, col_min:col_max]
        else:
            cropped = data[:, row_min:row_max, col_min:col_max]

        # Update transform for cropped data
        new_transform = transform * Affine.translation(col_min, row_min)

        # Create mask (all valid pixels in cropped area)
        if data.ndim == 2:
            mask = np.ones(cropped.shape, dtype=float)
        else:
            mask = np.ones(cropped.shape[-2:], dtype=float)

        # Store cropped data
        kwargs["cropped_data"] = cropped
        kwargs["transform"] = new_transform
        kwargs["bounds"] = (
            new_transform.c,
            new_transform.f + new_transform.e * cropped.shape[-2],
            new_transform.c + new_transform.a * cropped.shape[-1],
            new_transform.f,
        )

        # Generate statistics
        original_pixels = data.shape[-2] * data.shape[-1]
        cropped_pixels = cropped.shape[-2] * cropped.shape[-1]

        stats = {
            "total_pixels": cropped_pixels,
            "valid_pixels": cropped_pixels,
            "invalid_pixels": 0,
            "valid_percentage": 100.0,
            "pixels_removed": original_pixels - cropped_pixels,
        }

        metadata = {
            "original_bounds": original_bounds,
            "target_bounds": self.bounds,
            "applied_bounds": (min_x, min_y, max_x, max_y),
            "buffer": self.buffer,
            "buffer_units": self.buffer_units,
            "pixel_window": {
                "col_min": col_min,
                "col_max": col_max,
                "row_min": row_min,
                "row_max": row_max,
            },
            "original_shape": data.shape,
            "cropped_shape": cropped.shape,
        }

        return MaskResult(mask=mask, statistics=stats, metadata=metadata)


class SpatialAligner(ComposableComponent):
    """Ensure multiple rasters are spatially aligned.

    Handles alignment of multiple rasters to a common grid,
    resolution, and extent for consistent processing.
    """

    def __init__(
        self,
        name: str = "spatial_aligner",
        reference_idx: int = 0,
        method: str = "nearest",
    ):
        """Initialize spatial aligner.

        Args:
            name: Component identifier
            reference_idx: Index of reference raster for alignment
            method: Resampling method for alignment
        """
        super().__init__(name)
        self.reference_idx = reference_idx
        self.method = method
        self._config.update({"reference_idx": reference_idx, "method": method})

    def process(self, data: np.ndarray, **kwargs) -> MaskResult:
        """Process is not used directly for alignment."""
        raise NotImplementedError(
            "SpatialAligner requires align_rasters() method instead of process()"
        )

    def align_rasters(
        self,
        rasters: list[np.ndarray],
        transforms: list[Affine],
        crs_list: list[str] | None = None,
    ) -> list[SpatialResult]:
        """Align multiple rasters to common grid.

        Args:
            rasters: List of raster arrays
            transforms: List of affine transforms
            crs_list: Optional list of CRS strings

        Returns:
            List of SpatialResult objects with aligned data
        """
        if not rasters:
            raise ValueError("No rasters provided for alignment")

        if len(rasters) != len(transforms):
            raise ValueError("Number of rasters must match number of transforms")

        # Use reference raster for alignment target
        ref_raster = rasters[self.reference_idx]
        ref_transform = transforms[self.reference_idx]
        ref_crs = crs_list[self.reference_idx] if crs_list else None

        # Calculate reference bounds
        height, width = ref_raster.shape[-2:]
        west = ref_transform.c
        north = ref_transform.f
        east = west + ref_transform.a * width
        south = north + ref_transform.e * height
        ref_bounds = (west, south, east, north)

        # Align each raster
        results = []
        for i, (raster, transform) in enumerate(zip(rasters, transforms)):
            if i == self.reference_idx:
                # Reference raster stays unchanged
                result = SpatialResult(
                    data=raster,
                    transform=transform,
                    crs=ref_crs or "unknown",
                    bounds=ref_bounds,
                    metadata={"is_reference": True},
                )
            else:
                # Align to reference
                aligned = self._align_to_reference(
                    raster,
                    transform,
                    ref_raster.shape[-2:],
                    ref_transform,
                    crs_list[i] if crs_list else None,
                    ref_crs,
                )

                result = SpatialResult(
                    data=aligned,
                    transform=ref_transform,
                    crs=ref_crs or "unknown",
                    bounds=ref_bounds,
                    metadata={
                        "is_reference": False,
                        "aligned_to": self.reference_idx,
                        "original_shape": raster.shape,
                        "method": self.method,
                    },
                )

            results.append(result)

        return results

    def _align_to_reference(
        self,
        data: np.ndarray,
        src_transform: Affine,
        ref_shape: tuple[int, int],
        ref_transform: Affine,
        src_crs: str | None,
        ref_crs: str | None,
    ) -> np.ndarray:
        """Align a single raster to reference grid."""
        # Map resampling methods
        resample_map = {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
            "average": Resampling.average,
        }
        resample_method = resample_map.get(self.method, Resampling.nearest)

        # Check if we have CRS information
        if not src_crs and not ref_crs:
            # No CRS provided - use a dummy CRS for alignment
            dummy_crs = "EPSG:4326"
            src_crs = dummy_crs
            ref_crs = dummy_crs

        # Create output array
        if data.ndim == 2:
            aligned = np.empty(ref_shape, dtype=data.dtype)
            reproject(
                source=data,
                destination=aligned,
                src_transform=src_transform,
                src_crs=src_crs or ref_crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs or src_crs,
                resampling=resample_method,
            )
        else:
            # Handle multi-band
            aligned = np.empty((data.shape[0],) + ref_shape, dtype=data.dtype)
            for i in range(data.shape[0]):
                reproject(
                    source=data[i],
                    destination=aligned[i],
                    src_transform=src_transform,
                    src_crs=src_crs or ref_crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs or src_crs,
                    resampling=resample_method,
                )

        return aligned
