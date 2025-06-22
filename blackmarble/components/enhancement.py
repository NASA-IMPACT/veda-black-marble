"""Enhancement components for Black Marble processing.

This module provides components for enhancing data through various
techniques including road enhancement, contrast adjustment, and colormap application.
"""

from typing import Any
from dataclasses import dataclass
import numpy as np

from .base import ComposableComponent, MaskResult, QualityStatistics
from ..calculations import enhance_with_roads, normalize_to_range


@dataclass
class EnhancementResult:
    """Result of an enhancement operation.

    Attributes:
        data: Enhanced data array
        statistics: Statistics about the enhancement
        metadata: Additional metadata about the operation
    """

    data: np.ndarray
    statistics: dict[str, Any]
    metadata: dict[str, Any] | None = None


class RoadEnhancer(ComposableComponent):
    """Apply OSM road enhancement to data.

    This component enhances values along road networks to better
    highlight urban corridors and transportation infrastructure.
    """

    def __init__(
        self,
        name: str = "road_enhancer",
        major_factor: float = 1.3,
        minor_factor: float = 1.1,
        road_classes: dict[int, str] | None = None,
    ):
        """Initialize road enhancer.

        Args:
            name: Component name
            major_factor: Enhancement factor for major roads
            minor_factor: Enhancement factor for minor roads
            road_classes: Mapping of road class numbers to types
        """
        super().__init__(name)
        self.major_factor = major_factor
        self.minor_factor = minor_factor
        self.road_classes = road_classes or {
            0: "no_road",
            1: "motorway",
            2: "primary",
            3: "secondary",
            4: "trunk",
            5: "tertiary",
        }
        self._config = {
            "major_factor": major_factor,
            "minor_factor": minor_factor,
            "road_classes": self.road_classes,
        }

    def enhance(
        self,
        data: np.ndarray,
        roads: np.ndarray,
        custom_factors: dict[int, float] | None = None,
        **kwargs,
    ) -> EnhancementResult:
        """Apply road enhancement to data.

        Args:
            data: Base data to enhance
            roads: Road classification array (0=no road, 1-5=road types)
            custom_factors: Custom enhancement factors by road class
            **kwargs: Additional parameters

        Returns:
            EnhancementResult with enhanced data and statistics
        """
        # Use custom factors if provided, otherwise use defaults
        if custom_factors:
            enhanced = data.copy()
            for road_class, factor in custom_factors.items():
                enhanced = np.where(roads == road_class, enhanced * factor, enhanced)
        else:
            # Use the standard enhancement function
            enhanced = enhance_with_roads(
                data, roads, self.major_factor, self.minor_factor
            )

        # Calculate statistics
        stats = self._calculate_statistics(data, enhanced, roads)

        # Create metadata
        metadata = {
            "major_factor": self.major_factor,
            "minor_factor": self.minor_factor,
            "custom_factors": custom_factors,
            "road_pixels_enhanced": stats["road_pixels"],
        }

        return EnhancementResult(data=enhanced, statistics=stats, metadata=metadata)

    def _calculate_statistics(
        self, original: np.ndarray, enhanced: np.ndarray, roads: np.ndarray
    ) -> dict[str, Any]:
        """Calculate enhancement statistics."""
        # Count pixels by road type
        road_counts = {}
        for road_class, name in self.road_classes.items():
            count = np.sum(roads == road_class)
            road_counts[name] = int(count)

        # Calculate enhancement statistics
        difference = enhanced - original

        stats = {
            "road_pixels": int(np.sum(roads > 0)),
            "total_pixels": int(roads.size),
            "road_percentage": float(100.0 * np.sum(roads > 0) / roads.size),
            "road_counts": road_counts,
            "mean_enhancement": float(np.mean(difference[roads > 0]))
            if np.any(roads > 0)
            else 0.0,
            "max_enhancement": float(np.max(difference))
            if difference.size > 0
            else 0.0,
        }

        return stats

    def process(self, data: np.ndarray, **kwargs) -> MaskResult:
        """Process method for protocol compliance.

        For enhancement, we don't create masks but need to comply with protocol.
        """
        # Roads data must be provided
        roads = kwargs.get("roads")
        if roads is None:
            raise ValueError("Roads data must be provided in kwargs")

        # Perform enhancement (remove roads from kwargs to avoid duplicate)
        enhancement_kwargs = {k: v for k, v in kwargs.items() if k != "roads"}
        result = self.enhance(data, roads, **enhancement_kwargs)

        # Create a mask indicating enhanced pixels
        mask = (roads > 0).astype(float)

        return MaskResult(
            mask=mask,
            statistics=QualityStatistics.from_mask(mask).to_dict(),
            metadata=result.metadata,
        )


class ContrastEnhancer(ComposableComponent):
    """Dynamic range adjustment for improved visualization.

    This component provides various contrast enhancement techniques
    including histogram stretching, percentile clipping, and adaptive enhancement.
    """

    def __init__(
        self,
        name: str = "contrast_enhancer",
        method: str = "percentile",
        clip_percentiles: tuple[float, float] = (2.0, 98.0),
    ):
        """Initialize contrast enhancer.

        Args:
            name: Component name
            method: Enhancement method ('percentile', 'minmax', 'adaptive')
            clip_percentiles: Percentiles for clipping (low, high)
        """
        super().__init__(name)
        self.method = method
        self.clip_percentiles = clip_percentiles
        self._config = {"method": method, "clip_percentiles": clip_percentiles}

    def enhance(
        self,
        data: np.ndarray,
        output_range: tuple[float, float] = (0.0, 1.0),
        mask: np.ndarray | None = None,
        **kwargs,
    ) -> EnhancementResult:
        """Apply contrast enhancement.

        Args:
            data: Input data array
            output_range: Desired output range (min, max)
            mask: Optional mask for valid pixels
            **kwargs: Method-specific parameters

        Returns:
            EnhancementResult with enhanced data
        """
        # Apply mask if provided
        if mask is not None:
            valid_data = data[mask > 0]
        else:
            valid_data = data[~np.isnan(data)]

        if len(valid_data) == 0:
            # No valid data, return zeros
            enhanced = np.zeros_like(data)
            stats = {"valid_pixels": 0, "method": self.method}
            return EnhancementResult(enhanced, stats)

        # Apply enhancement based on method
        if self.method == "percentile":
            enhanced = self._percentile_stretch(data, valid_data, output_range)
        elif self.method == "minmax":
            enhanced = self._minmax_stretch(data, valid_data, output_range)
        elif self.method == "adaptive":
            enhanced = self._adaptive_stretch(data, valid_data, output_range, **kwargs)
        else:
            raise ValueError(f"Unknown enhancement method: {self.method}")

        # Calculate statistics
        stats = self._calculate_statistics(data, enhanced, valid_data)

        return EnhancementResult(
            data=enhanced,
            statistics=stats,
            metadata={"method": self.method, "output_range": output_range},
        )

    def _percentile_stretch(
        self,
        data: np.ndarray,
        valid_data: np.ndarray,
        output_range: tuple[float, float],
    ) -> np.ndarray:
        """Apply percentile-based contrast stretch."""
        # Calculate percentiles from valid data
        low_percentile, high_percentile = self.clip_percentiles
        vmin = np.percentile(valid_data, low_percentile)
        vmax = np.percentile(valid_data, high_percentile)

        # Apply normalization
        return normalize_to_range(
            data, output_range[0], output_range[1], float(vmin), float(vmax)
        )

    def _minmax_stretch(
        self,
        data: np.ndarray,
        valid_data: np.ndarray,
        output_range: tuple[float, float],
    ) -> np.ndarray:
        """Apply min-max contrast stretch."""
        vmin = np.min(valid_data)
        vmax = np.max(valid_data)

        return normalize_to_range(data, output_range[0], output_range[1], vmin, vmax)

    def _adaptive_stretch(
        self,
        data: np.ndarray,
        valid_data: np.ndarray,
        output_range: tuple[float, float],
        target_mean: float = 0.5,
        target_std: float = 0.15,
        **kwargs,
    ) -> np.ndarray:
        """Apply adaptive contrast stretch to achieve target statistics."""
        # Calculate current statistics
        current_mean = np.mean(valid_data)
        current_std = np.std(valid_data)

        if current_std == 0:
            # No variation, return constant value
            return np.full_like(data, target_mean)

        # Calculate scaling to achieve target statistics
        scale = target_std / current_std
        shift = target_mean - (current_mean * scale)

        # Apply transformation
        enhanced = data * scale + shift

        # Clip to output range
        enhanced = np.clip(enhanced, output_range[0], output_range[1])

        return enhanced

    def _calculate_statistics(
        self, original: np.ndarray, enhanced: np.ndarray, valid_data: np.ndarray
    ) -> dict[str, Any]:
        """Calculate enhancement statistics."""
        valid_enhanced = enhanced[~np.isnan(enhanced)]

        return {
            "valid_pixels": len(valid_data),
            "total_pixels": original.size,
            "original_range": (float(np.min(valid_data)), float(np.max(valid_data))),
            "enhanced_range": (
                float(np.min(valid_enhanced)),
                float(np.max(valid_enhanced)),
            ),
            "original_mean": float(np.mean(valid_data)),
            "enhanced_mean": float(np.mean(valid_enhanced)),
            "original_std": float(np.std(valid_data)),
            "enhanced_std": float(np.std(valid_enhanced)),
        }

    def process(self, data: np.ndarray, **kwargs) -> MaskResult:
        """Process method for protocol compliance."""
        # Perform enhancement
        result = self.enhance(data, **kwargs)

        # Create a mask of valid pixels
        mask = (~np.isnan(result.data)).astype(float)

        return MaskResult(
            mask=mask,
            statistics=QualityStatistics.from_mask(mask).to_dict(),
            metadata=result.metadata,
        )


class ColorMapper(ComposableComponent):
    """Apply colormaps for visualization.

    This component handles colormap application with support for
    custom colormaps, transparency, and multi-band output.
    """

    def __init__(
        self,
        name: str = "color_mapper",
        colormap_name: str | None = None,
        colormap_data: dict[str, np.ndarray] | None = None,
        add_alpha: bool = False,
    ):
        """Initialize color mapper.

        Args:
            name: Component name
            colormap_name: Name of standard colormap
            colormap_data: Custom colormap data {'r': array, 'g': array, 'b': array}
            add_alpha: Add alpha channel for transparency
        """
        super().__init__(name)
        self.colormap_name = colormap_name
        self.colormap_data = colormap_data
        self.add_alpha = add_alpha
        self._config = {
            "colormap_name": colormap_name,
            "has_custom_colormap": colormap_data is not None,
            "add_alpha": add_alpha,
        }

        # Initialize colormap
        self._init_colormap()

    def _init_colormap(self):
        """Initialize the colormap arrays."""
        if self.colormap_data:
            # Use provided custom colormap
            self.cmap_r = self.colormap_data["r"]
            self.cmap_g = self.colormap_data["g"]
            self.cmap_b = self.colormap_data["b"]
        else:
            # Create a default grayscale colormap
            self.cmap_r = np.arange(256, dtype=np.uint8)
            self.cmap_g = np.arange(256, dtype=np.uint8)
            self.cmap_b = np.arange(256, dtype=np.uint8)

    def apply_colormap(
        self,
        data: np.ndarray,
        scale_range: tuple[float, float] | None = None,
        alpha_mask: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Apply colormap to data.

        Args:
            data: Input data array (2D)
            scale_range: Range to scale data before colormap (min, max)
            alpha_mask: Optional alpha channel mask
            **kwargs: Additional parameters

        Returns:
            RGB(A) array with shape (3 or 4, height, width) as uint8
        """
        # Scale data to 0-255
        if scale_range:
            scaled = normalize_to_range(data, 0, 255, scale_range[0], scale_range[1])
        else:
            # Auto-scale based on data range
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                vmin, vmax = np.min(valid_data), np.max(valid_data)
                scaled = normalize_to_range(data, 0, 255, vmin, vmax)
            else:
                scaled = np.zeros_like(data)

        # Handle NaN values before rounding to avoid warnings
        valid_mask = ~np.isnan(scaled)

        # Convert to uint8 indices with proper rounding
        indices = np.zeros_like(scaled, dtype=np.uint8)
        if np.any(valid_mask):
            indices[valid_mask] = np.clip(np.round(scaled[valid_mask]), 0, 255).astype(
                np.uint8
            )

        # Apply colormap
        r = self.cmap_r[indices]
        g = self.cmap_g[indices]
        b = self.cmap_b[indices]

        # Create output array
        if self.add_alpha:
            # Create alpha channel
            if alpha_mask is not None:
                alpha = (alpha_mask * 255).astype(np.uint8)
            else:
                # Use data validity as alpha
                alpha = np.where(np.isnan(data), 0, 255).astype(np.uint8)

            # Stack RGBA
            output = np.stack([r, g, b, alpha], axis=0)
        else:
            # Stack RGB
            output = np.stack([r, g, b], axis=0)

        return output

    def create_colorbar(
        self,
        height: int = 256,
        width: int = 32,
        orientation: str = "vertical",
        labels: dict[float, str] | None = None,
    ) -> np.ndarray:
        """Create a colorbar image.

        Args:
            height: Height of colorbar
            width: Width of colorbar
            orientation: 'vertical' or 'horizontal'
            labels: Optional labels for specific values

        Returns:
            RGB colorbar array
        """
        if orientation == "vertical":
            # Create gradient from top to bottom
            indices = np.linspace(255, 0, height).astype(np.uint8)
            indices = np.tile(indices[:, np.newaxis], (1, width))
        else:
            # Create gradient from left to right
            indices = np.linspace(0, 255, width).astype(np.uint8)
            indices = np.tile(indices[np.newaxis, :], (height, 1))

        # Apply colormap
        r = self.cmap_r[indices]
        g = self.cmap_g[indices]
        b = self.cmap_b[indices]

        # Stack RGB
        colorbar = np.stack([r, g, b], axis=0)

        return colorbar

    def process(self, data: np.ndarray, **kwargs) -> MaskResult:
        """Process method for protocol compliance."""
        # Apply colormap
        result = self.apply_colormap(data, **kwargs)

        # Create a mask of valid pixels from original data
        mask = (~np.isnan(data)).astype(float)

        return MaskResult(
            mask=mask,
            statistics=QualityStatistics.from_mask(mask).to_dict(),
            metadata={
                "component": "color_mapper",
                "output_shape": result.shape,
                "channels": result.shape[0],
            },
        )


class CompositeEnhancer(ComposableComponent):
    """Composite enhancer that chains multiple enhancement operations.

    This component allows combining multiple enhancement steps
    in a configurable pipeline.
    """

    def __init__(
        self,
        name: str = "composite_enhancer",
        enhancers: list[ComposableComponent] | None = None,
    ):
        """Initialize composite enhancer.

        Args:
            name: Component name
            enhancers: List of enhancement components to chain
        """
        super().__init__(name)
        self.enhancers = enhancers or []
        self._config = {
            "num_enhancers": len(self.enhancers),
            "enhancer_names": [e.name for e in self.enhancers],
        }

    def add_enhancer(self, enhancer: ComposableComponent):
        """Add an enhancer to the pipeline."""
        self.enhancers.append(enhancer)
        self._config["num_enhancers"] = len(self.enhancers)
        self._config["enhancer_names"] = [e.name for e in self.enhancers]

    def enhance(self, data: np.ndarray, **kwargs) -> EnhancementResult:
        """Apply all enhancers in sequence.

        Args:
            data: Input data array
            **kwargs: Parameters passed to all enhancers

        Returns:
            EnhancementResult with final enhanced data
        """
        current_data = data.copy()
        all_stats = {}
        all_metadata = {}

        # Apply each enhancer in sequence
        for i, enhancer in enumerate(self.enhancers):
            # Check if enhancer has appropriate method
            if hasattr(enhancer, "enhance") and callable(
                getattr(enhancer, "enhance", None)
            ):
                result = getattr(enhancer, "enhance")(current_data, **kwargs)
                current_data = result.data
                all_stats[f"step_{i}_{enhancer.name}"] = result.statistics
                if result.metadata:
                    all_metadata[f"step_{i}_{enhancer.name}"] = result.metadata
            elif hasattr(enhancer, "apply_colormap") and isinstance(
                enhancer, ColorMapper
            ):
                # Special handling for ColorMapper
                current_data = enhancer.apply_colormap(current_data, **kwargs)
                all_stats[f"step_{i}_{enhancer.name}"] = {
                    "output_shape": current_data.shape,
                    "channels": current_data.shape[0] if current_data.ndim > 2 else 1,
                }
            else:
                # Try process method as fallback
                mask_result = enhancer.process(current_data, **kwargs)
                all_stats[f"step_{i}_{enhancer.name}"] = mask_result.statistics
                if mask_result.metadata:
                    all_metadata[f"step_{i}_{enhancer.name}"] = mask_result.metadata

        return EnhancementResult(
            data=current_data, statistics=all_stats, metadata=all_metadata
        )

    def process(self, data: np.ndarray, **kwargs) -> MaskResult:
        """Process method for protocol compliance."""
        # Apply enhancement pipeline
        result = self.enhance(data, **kwargs)

        # Create a mask indicating valid output pixels
        if result.data.ndim == 2:
            mask = (~np.isnan(result.data)).astype(float)
        else:
            # For multi-channel output, use first channel
            mask = np.ones(result.data.shape[1:], dtype=float)

        return MaskResult(
            mask=mask,
            statistics=QualityStatistics.from_mask(mask).to_dict(),
            metadata=result.metadata,
        )
