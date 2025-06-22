"""Output generation components for Black Marble processing.

This module provides components for generating final output products including
Cloud-Optimized GeoTIFFs, metadata, and visualizations.
"""

from typing import Protocol, Any
from dataclasses import dataclass
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.crs import CRS
from rasterio.transform import Affine
from rio_cogeo.profiles import cog_profiles
from rio_cogeo.cogeo import cog_translate
from rasterio.enums import Resampling as RIOResampling
import os
from pathlib import Path

from .base import ComposableComponent, MaskResult, QualityStatistics


@dataclass
class OutputMetadata:
    """Metadata for output products.

    Attributes:
        crs: Coordinate reference system
        transform: Affine transform for georeferencing
        bounds: Bounding box (minx, miny, maxx, maxy)
        shape: Array shape (height, width)
        dtype: Data type
        nodata: NoData value
        tags: Additional metadata tags
    """

    crs: CRS | str
    transform: Affine | tuple[float, ...]
    bounds: tuple[float, float, float, float]
    shape: tuple[int, int]
    dtype: np.dtype
    nodata: float | None = None
    tags: dict[str, Any] | None = None

    def to_rasterio_profile(self) -> dict[str, Any]:
        """Convert to rasterio profile dictionary."""
        from rasterio.transform import Affine

        # Convert transform to Affine if it's a tuple
        if isinstance(self.transform, Affine):
            # Already an Affine object, use directly
            transform = self.transform
        elif isinstance(self.transform, tuple):
            # Convert from GDAL tuple format
            transform = Affine.from_gdal(*self.transform)
        else:
            transform = self.transform

        # Ensure dtype is properly formatted for rasterio
        if isinstance(self.dtype, type) and issubclass(self.dtype, np.generic):
            # It's a numpy type like np.uint8
            dtype_str = np.dtype(self.dtype).name
        elif hasattr(self.dtype, "name"):
            # It's already a numpy dtype object
            dtype_str = self.dtype.name
        else:
            # Try to convert to numpy dtype first
            dtype_str = np.dtype(self.dtype).name

        profile = {
            "driver": "GTiff",
            "dtype": dtype_str,
            "width": self.shape[1],
            "height": self.shape[0],
            "count": 1,
            "crs": self.crs,
            "transform": transform,
            "nodata": self.nodata,
        }
        return profile


@dataclass
class COGProfile:
    """Cloud-Optimized GeoTIFF profile configuration.

    Attributes:
        profile_name: Name of COG profile (e.g., 'deflate', 'lzw', 'webp')
        blocksize: Tile size for internal tiling
        overview_level: Number of overview levels
        overview_resampling: Resampling method for overviews
        web_optimized: Enable web optimization
        in_memory: Process in memory (faster but uses more RAM)
    """

    profile_name: str = "deflate"
    blocksize: int = 512
    overview_level: int | None = None
    overview_resampling: str = "bilinear"
    web_optimized: bool = True  # Always true for COGs
    in_memory: bool = True


class OutputComponent(Protocol):
    """Protocol for output generation components."""

    def generate(self, data: np.ndarray, metadata: OutputMetadata, **kwargs) -> Any:
        """Generate output from data and metadata.

        Args:
            data: Input data array
            metadata: Output metadata
            **kwargs: Additional parameters

        Returns:
            Generated output (varies by component)
        """
        ...


class COGGenerator(ComposableComponent):
    """Generate Cloud-Optimized GeoTIFFs from arrays.

    This component creates COG files with proper tiling, compression,
    and overviews for efficient cloud access.
    """

    def __init__(
        self,
        name: str = "cog_generator",
        profile: COGProfile | None = None,
    ):
        """Initialize COG generator.

        Args:
            name: Component name
            profile: COG profile configuration
        """
        super().__init__(name)
        self.profile = profile or COGProfile()
        self._config = {"profile": self.profile.__dict__}

    def generate(
        self,
        data: np.ndarray,
        metadata: OutputMetadata,
        output_path: str | Path,
        **kwargs,
    ) -> Path:
        """Generate a Cloud-Optimized GeoTIFF.

        Args:
            data: Input data array (2D for single band, 3D for multi-band)
            metadata: Output metadata
            output_path: Path for output COG file
            **kwargs: Additional COG options

        Returns:
            Path to generated COG file
        """
        output_path = Path(output_path)

        # Determine number of bands
        if data.ndim == 2:
            count = 1
            data_to_write = data
        elif data.ndim == 3:
            count = data.shape[0]
            data_to_write = data
        else:
            raise ValueError(f"Data must be 2D or 3D, got shape {data.shape}")

        # Create rasterio profile
        profile = metadata.to_rasterio_profile()
        profile["count"] = count

        # Get COG profile options
        cog_profile = cog_profiles.get(self.profile.profile_name)
        if cog_profile is None:
            cog_profile = cog_profiles["deflate"]

        # COG profiles don't include dtype, so we need to ensure it's preserved
        cog_profile = cog_profile.copy()  # Don't modify the original

        # Create config dict for cog_translate
        config = {
            "GDAL_NUM_THREADS": "ALL_CPUS",
            "GDAL_TIFF_INTERNAL_MASK": os.environ.get("GDAL_TIFF_INTERNAL_MASK", "YES"),
            "GDAL_TIFF_OVR_BLOCKSIZE": str(self.profile.blocksize),
        }

        # Create COG using rio-cogeo
        with MemoryFile() as memfile:
            with memfile.open(**profile) as mem:
                if count == 1:
                    mem.write(data_to_write, 1)
                else:
                    for i in range(count):
                        mem.write(data_to_write[i], i + 1)

                # Add tags if provided
                if metadata.tags:
                    mem.update_tags(**metadata.tags)

            # Calculate overview levels if not specified
            overview_level = self.profile.overview_level
            if overview_level is None:
                # Calculate appropriate overview levels based on image size
                max_dim = max(metadata.shape)
                overview_level = 0
                dim = max_dim
                while dim > self.profile.blocksize:
                    overview_level += 1
                    dim //= 2

            # Re-open the memory file in read mode for cog_translate
            with memfile.open() as src:
                # Translate to COG
                cog_translate(
                    src,
                    str(output_path),
                    cog_profile,  # This is dst_kwargs
                    config=config,
                    overview_level=overview_level,
                    overview_resampling=self.profile.overview_resampling,  # type: ignore
                    web_optimized=self.profile.web_optimized,
                    in_memory=self.profile.in_memory,
                )

        return output_path

    def process(self, data: np.ndarray, **kwargs) -> MaskResult:
        """Process method for protocol compliance.

        This method is not used for COG generation but is required
        by the QualityComponent protocol.
        """
        # Return a dummy result since this component doesn't create masks
        return MaskResult(
            mask=np.ones(data.shape[:2], dtype=bool),
            statistics=QualityStatistics.from_mask(np.ones(data.shape[:2])).to_dict(),
            metadata={"component": "cog_generator", "action": "none"},
        )


class MetadataGenerator(ComposableComponent):
    """Generate metadata for output products.

    This component creates STAC items, processing metadata,
    and other metadata formats.
    """

    def __init__(
        self,
        name: str = "metadata_generator",
        metadata_format: str = "stac",
    ):
        """Initialize metadata generator.

        Args:
            name: Component name
            metadata_format: Format for metadata ('stac', 'json', 'xml')
        """
        super().__init__(name)
        self.metadata_format = metadata_format
        self._config = {"format": metadata_format}

    def generate(
        self,
        data: np.ndarray,
        metadata: OutputMetadata,
        processing_info: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Generate metadata for the output.

        Args:
            data: Input data array
            metadata: Output metadata
            processing_info: Information about processing steps
            **kwargs: Additional metadata fields

        Returns:
            Metadata dictionary
        """
        # Calculate statistics
        stats = self._calculate_statistics(data)

        # Base metadata
        base_metadata = {
            "type": "Feature",
            "properties": {
                "datetime": kwargs.get("datetime", None),
                "created": kwargs.get("created", None),
                "updated": kwargs.get("updated", None),
                "title": kwargs.get("title", "Black Marble Output"),
                "description": kwargs.get("description", ""),
                "providers": kwargs.get("providers", []),
                "statistics": stats,
                "shape": metadata.shape,
                "dtype": np.dtype(metadata.dtype).name,
                "nodata": metadata.nodata,
            },
            "geometry": self._bounds_to_geojson(metadata.bounds),
            "bbox": list(metadata.bounds),
        }

        # Add processing information
        if processing_info:
            base_metadata["properties"]["processing"] = processing_info

        # Add custom tags
        if metadata.tags:
            base_metadata["properties"]["tags"] = metadata.tags

        # Format-specific processing
        if self.metadata_format == "stac":
            return self._format_as_stac(base_metadata, **kwargs)
        elif self.metadata_format == "json":
            return base_metadata
        else:
            raise ValueError(f"Unsupported metadata format: {self.metadata_format}")

    def _calculate_statistics(self, data: np.ndarray) -> dict[str, Any]:
        """Calculate statistics for the data."""
        # Handle multi-band data
        if data.ndim == 3:
            stats = {}
            for i in range(data.shape[0]):
                band_data = data[i]
                stats[f"band_{i + 1}"] = self._band_statistics(band_data)
            return stats
        else:
            return self._band_statistics(data)

    def _band_statistics(self, data: np.ndarray) -> dict[str, float | None]:
        """Calculate statistics for a single band."""
        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            return {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "count": 0,
                "valid_percentage": 0.0,
            }

        return {
            "min": float(np.min(valid_data)),
            "max": float(np.max(valid_data)),
            "mean": float(np.mean(valid_data)),
            "std": float(np.std(valid_data)),
            "count": int(len(valid_data)),
            "valid_percentage": float(100.0 * len(valid_data) / data.size),
        }

    def _bounds_to_geojson(
        self, bounds: tuple[float, float, float, float]
    ) -> dict[str, Any]:
        """Convert bounds to GeoJSON polygon."""
        minx, miny, maxx, maxy = bounds
        return {
            "type": "Polygon",
            "coordinates": [
                [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]
            ],
        }

    def _format_as_stac(self, metadata: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Format metadata as STAC item."""
        # Add STAC-specific fields
        metadata["stac_version"] = "1.0.0"
        metadata["id"] = kwargs.get("id", "black-marble-output")
        metadata["collection"] = kwargs.get("collection", "black-marble")

        # Add assets
        metadata["assets"] = kwargs.get("assets", {})

        # Add links
        metadata["links"] = kwargs.get("links", [])

        return metadata

    def process(self, data: np.ndarray, **kwargs) -> MaskResult:
        """Process method for protocol compliance."""
        # Return a dummy result
        return MaskResult(
            mask=np.ones(data.shape[:2], dtype=bool),
            statistics=QualityStatistics.from_mask(np.ones(data.shape[:2])).to_dict(),
            metadata={"component": "metadata_generator", "action": "none"},
        )


class VisualizationEnhancer(ComposableComponent):
    """Apply colormaps and display enhancements to data.

    This component handles visualization preparation including
    colormap application, contrast enhancement, and scaling.
    """

    def __init__(
        self,
        name: str = "visualization_enhancer",
        colormap: str | None = None,
        colormap_arrays: dict[str, np.ndarray] | None = None,
    ):
        """Initialize visualization enhancer.

        Args:
            name: Component name
            colormap: Name of colormap to use
            colormap_arrays: Custom colormap arrays {'r': array, 'g': array, 'b': array}
        """
        super().__init__(name)
        self.colormap = colormap
        self.colormap_arrays = colormap_arrays
        self._config = {
            "colormap": colormap,
            "has_custom_arrays": colormap_arrays is not None,
        }

    def enhance(
        self,
        data: np.ndarray,
        display_min: float = 0.0,
        display_max: float = 1.0,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Apply display enhancements to data.

        Args:
            data: Input data array (2D)
            display_min: Minimum display value
            display_max: Maximum display value
            gamma: Gamma correction factor
            **kwargs: Additional enhancement parameters

        Returns:
            Enhanced data array (uint8)
        """
        # Apply display range scaling
        scaled = self._scale_to_display_range(data, display_min, display_max)

        # Apply gamma correction if needed
        if gamma != 1.0:
            scaled = np.power(scaled, gamma)

        # Convert to uint8
        enhanced = (scaled * 255).astype(np.uint8)

        return enhanced

    def apply_colormap(
        self,
        data: np.ndarray,
        display_min: float = 0.0,
        display_max: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Apply colormap to create RGB visualization.

        Args:
            data: Input data array (2D)
            display_min: Minimum display value
            display_max: Maximum display value
            **kwargs: Additional parameters

        Returns:
            RGB array (3, height, width) as uint8
        """
        # First enhance the data
        enhanced = self.enhance(data, display_min, display_max, **kwargs)

        # Apply colormap
        if self.colormap_arrays:
            # Use custom colormap arrays
            r = self.colormap_arrays["r"][enhanced]
            g = self.colormap_arrays["g"][enhanced]
            b = self.colormap_arrays["b"][enhanced]
        else:
            # For now, return grayscale if no colormap specified
            r = g = b = enhanced

        # Stack into RGB array
        rgb = np.stack([r, g, b], axis=0)

        return rgb

    def _scale_to_display_range(
        self, data: np.ndarray, display_min: float, display_max: float
    ) -> np.ndarray:
        """Scale data to display range [0, 1]."""
        # Clip to display range
        clipped = np.clip(data, display_min, display_max)

        # Handle case where min equals max
        if display_max == display_min:
            return np.zeros_like(clipped)

        # Scale to [0, 1]
        scaled = (clipped - display_min) / (display_max - display_min)

        # Handle NaN values
        scaled = np.where(np.isnan(data), 0, scaled)

        return scaled

    def process(self, data: np.ndarray, **kwargs) -> MaskResult:
        """Process method for protocol compliance."""
        # Return a dummy result
        return MaskResult(
            mask=np.ones(data.shape[:2], dtype=bool),
            statistics=QualityStatistics.from_mask(np.ones(data.shape[:2])).to_dict(),
            metadata={"component": "visualization_enhancer", "action": "none"},
        )


class OutputValidator(ComposableComponent):
    """Validate outputs before saving.

    This component performs validation checks on output data
    to ensure quality and correctness.
    """

    def __init__(
        self,
        name: str = "output_validator",
        min_valid_percentage: float = 10.0,
        check_range: bool = True,
        check_nodata: bool = True,
    ):
        """Initialize output validator.

        Args:
            name: Component name
            min_valid_percentage: Minimum percentage of valid pixels required
            check_range: Check if values are in expected range
            check_nodata: Check for proper NoData handling
        """
        super().__init__(name)
        self.min_valid_percentage = min_valid_percentage
        self.check_range = check_range
        self.check_nodata = check_nodata
        self._config = {
            "min_valid_percentage": min_valid_percentage,
            "check_range": check_range,
            "check_nodata": check_nodata,
        }

    def validate(
        self,
        data: np.ndarray,
        metadata: OutputMetadata,
        expected_range: tuple[float, float] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Validate output data.

        Args:
            data: Data array to validate
            metadata: Output metadata
            expected_range: Expected value range (min, max)
            **kwargs: Additional validation parameters

        Returns:
            Validation results dictionary

        Raises:
            ValueError: If validation fails
        """
        results = {"valid": True, "errors": [], "warnings": [], "statistics": {}}

        # Check data shape matches metadata
        if data.ndim == 2:
            data_shape = data.shape
        else:
            data_shape = data.shape[1:]  # For multi-band data

        if data_shape != metadata.shape:
            results["valid"] = False
            results["errors"].append(
                f"Data shape {data_shape} doesn't match metadata shape {metadata.shape}"
            )

        # Calculate valid pixel percentage
        if data.ndim == 2:
            valid_mask = ~np.isnan(data)
            if metadata.nodata is not None:
                valid_mask &= data != metadata.nodata

            valid_percentage = 100.0 * np.sum(valid_mask) / valid_mask.size
            results["statistics"]["valid_percentage"] = valid_percentage

            if valid_percentage < self.min_valid_percentage:
                results["valid"] = False
                results["errors"].append(
                    f"Valid pixel percentage ({valid_percentage:.1f}%) is below "
                    f"minimum threshold ({self.min_valid_percentage}%)"
                )
        else:
            # For multi-band data, check each band individually
            band_percentages = []
            for band_idx in range(data.shape[0]):
                band_data = data[band_idx]
                valid_mask = ~np.isnan(band_data)
                if metadata.nodata is not None:
                    valid_mask &= band_data != metadata.nodata

                band_valid_percentage = 100.0 * np.sum(valid_mask) / valid_mask.size
                band_percentages.append(band_valid_percentage)

                if band_valid_percentage < self.min_valid_percentage:
                    results["valid"] = False
                    results["errors"].append(
                        f"Band {band_idx + 1} valid pixel percentage ({band_valid_percentage:.1f}%) is below "
                        f"minimum threshold ({self.min_valid_percentage}%)"
                    )

            results["statistics"]["band_valid_percentages"] = band_percentages
            # Overall valid percentage is the average across bands
            results["statistics"]["valid_percentage"] = np.mean(band_percentages)

        # Check value range if specified
        if self.check_range and expected_range is not None:
            min_val, max_val = expected_range

            if data.ndim == 2:
                actual_min = np.nanmin(data)
                actual_max = np.nanmax(data)
            else:
                actual_min = np.nanmin(data)
                actual_max = np.nanmax(data)

            results["statistics"]["actual_range"] = (actual_min, actual_max)

            if actual_min < min_val or actual_max > max_val:
                results["warnings"].append(
                    f"Data range [{actual_min:.3f}, {actual_max:.3f}] is outside "
                    f"expected range [{min_val:.3f}, {max_val:.3f}]"
                )

        # Check NoData handling
        if self.check_nodata and metadata.nodata is not None:
            if data.dtype in [np.float32, np.float64]:
                # For float types, check if NoData is being used instead of NaN
                nodata_count = np.sum(data == metadata.nodata)
                if nodata_count > 0:
                    results["warnings"].append(
                        f"Found {nodata_count} pixels with NoData value {metadata.nodata}. "
                        "Consider using NaN for float data."
                    )

        # Raise exception if validation failed
        if not results["valid"]:
            error_msg = "Output validation failed:\n" + "\n".join(results["errors"])
            raise ValueError(error_msg)

        return results

    def process(self, data: np.ndarray, **kwargs) -> MaskResult:
        """Process method for protocol compliance."""
        # Perform validation
        metadata = kwargs.get(
            "metadata",
            OutputMetadata(
                crs="EPSG:4326",
                transform=(0, 1, 0, 0, 0, -1),
                bounds=(0, 0, data.shape[1], data.shape[0]),
                shape=data.shape[:2] if data.ndim == 2 else data.shape[1:],
                dtype=data.dtype,
            ),
        )

        validation_results = None
        try:
            validation_results = self.validate(data, metadata, **kwargs)
            valid = validation_results["valid"]
        except ValueError:
            valid = False

        # Create mask based on validation
        mask = (
            np.ones(data.shape[:2], dtype=bool)
            if valid
            else np.zeros(data.shape[:2], dtype=bool)
        )

        return MaskResult(
            mask=mask,
            statistics=QualityStatistics.from_mask(mask).to_dict(),
            metadata={
                "component": "output_validator",
                "validation_results": validation_results if valid else None,
            },
        )
