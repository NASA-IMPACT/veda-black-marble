"""Quality control components for satellite imagery processing.

This module provides modular components for quality assessment and masking,
extracted from the main pipeline for better testability and reusability.
"""

from typing import Any
from dataclasses import dataclass
import numpy as np

from .base import (
    ComposableComponent,
    MaskResult,
    QualityStatistics,
    combine_masks,
    dilate_mask,
    apply_mask,
)
from ..config import LandsatConfig, VIIRSConfig


class CloudMasker(ComposableComponent):
    """Component for masking clouds in Landsat imagery using QA bands.

    This component processes Landsat QA_PIXEL bands to identify and mask
    cloud-contaminated pixels. It supports multiple masking strategies.
    """

    def __init__(
        self,
        name: str = "cloud_masker",
        strategy: str = "conservative",
        dilate_radius: int = 0,
        config: LandsatConfig | None = None,
    ):
        """Initialize CloudMasker.

        Args:
            name: Component name for identification
            strategy: Masking strategy - "conservative" (default) or "permissive"
            dilate_radius: Radius for dilating cloud masks (0 = no dilation)
            config: Landsat configuration (uses defaults if None)
        """
        super().__init__(name)
        self.strategy = strategy
        self.dilate_radius = dilate_radius
        self.config = config or LandsatConfig()

        self._config = {
            "strategy": strategy,
            "dilate_radius": dilate_radius,
        }

        # Define QA values for different strategies
        # Based on Landsat Collection 2 QA_PIXEL bit definitions
        if strategy == "conservative":
            # Only keep clear land (21824) and clear water (21952)
            self.valid_values = {21824, 21952}
        elif strategy == "permissive":
            # Keep clear pixels and low confidence clouds
            # This is closer to the mark_cloud2 function
            self.invalid_values = {55052, 54852, 54596, 22280}
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def process(self, data: np.ndarray, **kwargs) -> MaskResult:
        """Process QA_PIXEL band to create cloud mask.

        Args:
            data: QA_PIXEL band array
            **kwargs: Additional parameters (unused)

        Returns:
            MaskResult with cloud mask and statistics
        """
        # Create initial mask based on strategy
        if self.strategy == "conservative":
            # Only pixels with exact clear values are kept
            mask = np.isin(data, list(self.valid_values)).astype(float)
        else:  # permissive
            # All pixels except known cloud values are kept
            mask = (~np.isin(data, list(self.invalid_values))).astype(float)

        # Apply dilation if requested
        if self.dilate_radius > 0:
            mask = dilate_mask(mask, self.dilate_radius)

        # Calculate statistics
        unique_values, counts = np.unique(data, return_counts=True)
        value_counts = dict(zip(unique_values.tolist(), counts.tolist()))

        # Categorize pixels
        categories = {
            "clear_land": value_counts.get(21824, 0),
            "clear_water": value_counts.get(21952, 0),
            "cloud_low_confidence": value_counts.get(22280, 0),
            "cloud_medium_confidence": value_counts.get(54596, 0),
            "cloud_high_confidence": value_counts.get(54852, 0)
            + value_counts.get(55052, 0),
            "other": sum(
                c
                for v, c in value_counts.items()
                if v not in {21824, 21952, 22280, 54596, 54852, 55052}
            ),
        }

        stats = QualityStatistics.from_mask(mask, categories)

        metadata = {
            "qa_value_distribution": value_counts,
            "strategy": self.strategy,
            "dilate_radius": self.dilate_radius,
        }

        return MaskResult(mask=mask, statistics=stats.__dict__, metadata=metadata)


class AerosolMasker(ComposableComponent):
    """Component for masking aerosol-contaminated pixels."""

    def __init__(
        self,
        name: str = "aerosol_masker",
        threshold: str = "standard",
        config: LandsatConfig | None = None,
    ):
        """Initialize AerosolMasker.

        Args:
            name: Component name
            threshold: Threshold level - "standard" or "strict"
            config: Landsat configuration
        """
        super().__init__(name)
        self.threshold = threshold
        self.config = config or LandsatConfig()

        self._config = {"threshold": threshold}

        # Define aerosol QA values to mask
        if threshold == "standard":
            self.invalid_values = {192, 194, 196, 228}
        else:  # strict
            self.invalid_values = {192, 194, 196, 224, 228}

    def process(self, data: np.ndarray, **kwargs) -> MaskResult:
        """Process QA_AEROSOL band to create aerosol mask.

        Args:
            data: QA_AEROSOL band array
            **kwargs: Additional parameters (unused)

        Returns:
            MaskResult with aerosol mask and statistics
        """
        # Create mask - pixels NOT in invalid set are kept
        mask = (~np.isin(data, list(self.invalid_values))).astype(float)

        # Calculate statistics
        unique_values, counts = np.unique(data, return_counts=True)
        value_counts = dict(zip(unique_values.tolist(), counts.tolist()))

        categories = {
            "low_aerosol": sum(
                c
                for v, c in value_counts.items()
                if v not in self.invalid_values and v < 150
            ),
            "medium_aerosol": sum(
                c for v, c in value_counts.items() if v in {192, 194}
            ),
            "high_aerosol": sum(
                c for v, c in value_counts.items() if v in {196, 224, 228}
            ),
        }

        stats = QualityStatistics.from_mask(mask, categories)

        metadata = {
            "qa_value_distribution": value_counts,
            "threshold": self.threshold,
        }

        return MaskResult(mask=mask, statistics=stats.__dict__, metadata=metadata)


class RadianceSaturationMasker(ComposableComponent):
    """Component for masking radiometrically saturated pixels."""

    def __init__(self, name: str = "radsat_masker"):
        """Initialize RadianceSaturationMasker.

        Args:
            name: Component name
        """
        super().__init__(name)
        self._config = {}

    def process(
        self, data: np.ndarray, band: int | None = None, **kwargs
    ) -> MaskResult:
        """Process QA_RADSAT band to create saturation masks.

        Args:
            data: QA_RADSAT band array
            band: Specific band number (3, 4, or 5) or None for all
            **kwargs: Additional parameters (unused)

        Returns:
            MaskResult with saturation mask(s) and statistics
        """
        # Handle NaN values
        data_safe = np.where(np.isnan(data), -1, data).astype(np.int32)

        if band is not None:
            # Single band mask
            if band not in {3, 4, 5}:
                raise ValueError(f"Band must be 3, 4, or 5, got {band}")

            bit_position = band - 1  # Band 3 = bit 2, Band 4 = bit 3, Band 5 = bit 4
            bit_value = 1 << bit_position

            # Pixel is good if bit is NOT set
            mask = ((data_safe & bit_value) == 0).astype(float)

            # Also mask out original NaN pixels
            mask = np.where(np.isnan(data), 0.0, mask)

            stats = QualityStatistics.from_mask(mask)
            metadata = {
                "band": band,
                "saturated_pixels": np.sum((data_safe & bit_value) > 0).item(),
            }

        else:
            # Return masks for all bands
            masks = []
            all_stats = {}

            for b in [3, 4, 5]:
                bit_position = b - 1
                bit_value = 1 << bit_position

                band_mask = ((data_safe & bit_value) == 0).astype(float)
                band_mask = np.where(np.isnan(data), 0.0, band_mask)

                masks.append(band_mask)
                all_stats[f"band_{b}"] = {
                    "saturated_pixels": np.sum((data_safe & bit_value) > 0).item(),
                    "valid_pixels": np.sum(band_mask > 0).item(),
                    "valid_percentage": 100.0 * np.sum(band_mask > 0) / band_mask.size,
                }

            # Stack masks as a 3D array
            mask = np.stack(masks, axis=0)

            # Overall statistics
            total_pixels = data.size
            any_saturated = np.sum(
                (data_safe & 0b11100) > 0
            ).item()  # Any of bits 2-4 set

            stats = {
                "total_pixels": total_pixels,
                "any_band_saturated": any_saturated,
                "all_bands_valid": np.sum((data_safe & 0b11100) == 0).item(),
                "band_statistics": all_stats,
            }

            metadata = {"multi_band": True}

        return MaskResult(
            mask=mask,
            statistics=stats if isinstance(stats, dict) else stats.__dict__,
            metadata=metadata,
        )


class VIIRSQualityMasker(ComposableComponent):
    """Component for masking VIIRS nighttime lights based on quality flags."""

    def __init__(
        self, name: str = "viirs_quality_masker", config: VIIRSConfig | None = None
    ):
        """Initialize VIIRSQualityMasker.

        Args:
            name: Component name
            config: VIIRS configuration
        """
        super().__init__(name)
        self.config = config or VIIRSConfig()
        self._config = {"valid_flags": self.config.quality_valid_flags}

    def process(self, data: np.ndarray, **kwargs) -> MaskResult:
        """Process VIIRS quality flag band.

        Args:
            data: Mandatory_Quality_Flag array from VIIRS
            **kwargs: Additional parameters (unused)

        Returns:
            MaskResult with quality mask and statistics
        """
        # Create mask for valid quality flags
        mask = np.isin(data, self.config.quality_valid_flags).astype(float)

        # Calculate statistics
        unique_values, counts = np.unique(data, return_counts=True)
        value_counts = dict(zip(unique_values.tolist(), counts.tolist()))

        categories = {
            "high_quality": value_counts.get(0, 0),  # Best quality
            "moderate_quality": value_counts.get(2, 0),  # Moderate quality
            "poor_quality": sum(
                c
                for v, c in value_counts.items()
                if v not in self.config.quality_valid_flags
            ),
        }

        stats = QualityStatistics.from_mask(mask, categories)

        metadata = {
            "qa_value_distribution": value_counts,
            "valid_flags": self.config.quality_valid_flags,
        }

        return MaskResult(mask=mask, statistics=stats.__dict__, metadata=metadata)


class QualityCompositor(ComposableComponent):
    """Component for combining multiple quality masks."""

    def __init__(
        self,
        name: str = "quality_compositor",
        components: list[ComposableComponent] | None = None,
        combination_method: str = "and",
    ):
        """Initialize QualityCompositor.

        Args:
            name: Component name
            components: List of quality components to combine
            combination_method: How to combine masks - "and" or "or"
        """
        super().__init__(name)
        self.components = components or []
        self.combination_method = combination_method

        self._config = {
            "combination_method": combination_method,
            "components": [c.name for c in self.components],
        }

    def add_component(self, component: ComposableComponent) -> None:
        """Add a component to the compositor.

        Args:
            component: Quality component to add
        """
        self.components.append(component)
        self._config["components"].append(component.name)

    def process(self, data: dict[str, np.ndarray], **kwargs) -> MaskResult:  # type: ignore[override]
        """Process multiple QA bands through their respective components.

        Args:
            data: Dictionary mapping component names to their input arrays
            **kwargs: Additional parameters passed to components

        Returns:
            MaskResult with combined mask and statistics
        """
        if not self.components:
            raise ValueError("No components configured in compositor")

        results = []
        component_stats = {}

        # Process each component
        for component in self.components:
            if component.name not in data:
                raise ValueError(f"No data provided for component '{component.name}'")

            result = component.process(data[component.name], **kwargs)
            results.append(result)
            component_stats[component.name] = result.statistics

        # Combine masks
        masks = [r.mask for r in results]
        combined_mask = combine_masks(masks, method=self.combination_method)

        # Calculate combined statistics
        stats = QualityStatistics.from_mask(combined_mask)

        metadata = {
            "combination_method": self.combination_method,
            "component_statistics": component_stats,
            "component_metadata": {
                c.name: r.metadata for c, r in zip(self.components, results)
            },
        }

        return MaskResult(
            mask=combined_mask, statistics=stats.__dict__, metadata=metadata
        )


class MaskStatistics(ComposableComponent):
    """Component for tracking and reporting masking statistics."""

    def __init__(self, name: str = "mask_statistics"):
        """Initialize MaskStatistics.

        Args:
            name: Component name
        """
        super().__init__(name)
        self.history: list[dict[str, Any]] = []
        self._config = {}

    def process(
        self, data: np.ndarray, mask_name: str = "mask", **kwargs
    ) -> MaskResult:
        """Calculate and track statistics for a mask.

        Args:
            data: Mask array
            mask_name: Name for this mask in tracking
            **kwargs: Additional parameters (unused)

        Returns:
            MaskResult with statistics
        """
        stats = QualityStatistics.from_mask(data)

        # Track in history
        record = {
            "name": mask_name,
            "statistics": stats.__dict__,
            "timestamp": np.datetime64("now"),
        }
        self.history.append(record)

        metadata = {"mask_name": mask_name, "history_length": len(self.history)}

        return MaskResult(mask=data, statistics=stats.__dict__, metadata=metadata)

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all tracked masks.

        Returns:
            Summary dictionary with overall statistics
        """
        if not self.history:
            return {"total_masks": 0, "message": "No masks tracked"}

        total_valid = sum(h["statistics"]["valid_pixels"] for h in self.history)
        total_pixels = sum(h["statistics"]["total_pixels"] for h in self.history)

        return {
            "total_masks": len(self.history),
            "total_pixels_processed": total_pixels,
            "total_valid_pixels": total_valid,
            "overall_valid_percentage": 100.0 * total_valid / total_pixels
            if total_pixels > 0
            else 0.0,
            "mask_names": [h["name"] for h in self.history],
            "individual_stats": {h["name"]: h["statistics"] for h in self.history},
        }

    def clear_history(self) -> None:
        """Clear the tracking history."""
        self.history.clear()


# Convenience function to create a standard Landsat QA pipeline
def create_landsat_qa_pipeline(
    strategy: str = "conservative",
    dilate_clouds: int = 0,
    config: LandsatConfig | None = None,
) -> QualityCompositor:
    """Create a standard Landsat quality assessment pipeline.

    Args:
        strategy: Cloud masking strategy
        dilate_clouds: Dilation radius for cloud masks
        config: Landsat configuration

    Returns:
        Configured QualityCompositor
    """
    cloud_masker = CloudMasker(
        name="qa_pixel", strategy=strategy, dilate_radius=dilate_clouds, config=config
    )

    aerosol_masker = AerosolMasker(
        name="qa_aerosol", threshold="standard", config=config
    )

    radsat_masker = RadianceSaturationMasker(name="qa_radsat")

    compositor = QualityCompositor(
        name="landsat_qa",
        components=[cloud_masker, aerosol_masker],
        combination_method="and",
    )

    # Note: radsat is handled separately per band
    return compositor
