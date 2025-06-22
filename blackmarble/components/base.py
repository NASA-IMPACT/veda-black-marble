"""Base protocol definitions for quality control components.

This module defines the interfaces and data structures used by all
quality control components in the Black Marble pipeline.
"""

from typing import Protocol, Any, cast
from dataclasses import dataclass
import numpy as np


@dataclass
class MaskResult:
    """Result of a masking operation.

    Attributes:
        mask: Boolean or float array where True/1.0 indicates pixels to keep
        statistics: Statistics about the masking operation
        metadata: Additional metadata about the operation
    """

    mask: np.ndarray
    statistics: dict[str, Any]
    metadata: dict[str, Any] | None = None


@dataclass
class QualityStatistics:
    """Statistics from quality control operations.

    Attributes:
        total_pixels: Total number of pixels processed
        valid_pixels: Number of pixels marked as valid/kept
        invalid_pixels: Number of pixels marked as invalid/masked
        valid_percentage: Percentage of valid pixels
        categories: Optional breakdown by category (e.g., cloud types)
    """

    total_pixels: int
    valid_pixels: int
    invalid_pixels: int
    valid_percentage: float
    categories: dict[str, int] | None = None

    @classmethod
    def from_mask(
        cls, mask: np.ndarray, categories: dict[str, int] | None = None
    ) -> "QualityStatistics":
        """Create statistics from a mask array.

        Args:
            mask: Boolean or float array where True/1.0 indicates valid pixels
            categories: Optional category breakdown

        Returns:
            QualityStatistics instance
        """
        total = mask.size
        valid = np.sum(mask > 0).item()
        invalid = total - valid
        percentage = 100.0 * valid / total if total > 0 else 0.0

        return cls(
            total_pixels=total,
            valid_pixels=valid,
            invalid_pixels=invalid,
            valid_percentage=percentage,
            categories=categories,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            "total_pixels": self.total_pixels,
            "valid_pixels": self.valid_pixels,
            "invalid_pixels": self.invalid_pixels,
            "valid_percentage": self.valid_percentage,
            "categories": self.categories,
        }


class QualityComponent(Protocol):
    """Protocol for quality control components.

    All quality control components should implement this interface
    to ensure consistency and composability.
    """

    def process(self, data: np.ndarray, **kwargs) -> MaskResult:
        """Process data and return quality mask.

        Args:
            data: Input data array to process
            **kwargs: Additional parameters specific to the component

        Returns:
            MaskResult containing the mask and statistics
        """
        ...

    def get_config(self) -> dict[str, Any]:
        """Get the current configuration of the component.

        Returns:
            Dictionary of configuration parameters
        """
        ...


class ComposableComponent(QualityComponent):
    """Base class for components that can be composed together."""

    def __init__(self, name: str):
        """Initialize component with a name.

        Args:
            name: Name of the component for identification
        """
        self.name = name
        self._config: dict[str, Any] = {}

    def get_config(self) -> dict[str, Any]:
        """Get the current configuration."""
        return self._config.copy()

    def __repr__(self) -> str:
        """String representation of the component."""
        return f"{self.__class__.__name__}(name='{self.name}')"


def combine_masks(masks: list[np.ndarray], method: str = "and") -> np.ndarray:
    """Combine multiple masks using the specified method.

    Args:
        masks: List of mask arrays to combine
        method: Combination method - "and" (all must be valid) or "or" (any can be valid)

    Returns:
        Combined mask array

    Raises:
        ValueError: If method is not recognized or masks have different shapes
    """
    if not masks:
        raise ValueError("No masks provided to combine")

    # Ensure all masks have the same shape
    shape = masks[0].shape
    for i, mask in enumerate(masks[1:], 1):
        if mask.shape != shape:
            raise ValueError(f"Mask {i} has shape {mask.shape}, expected {shape}")

    if method == "and":
        # All masks must be valid (1.0) for pixel to be kept
        result = np.ones_like(masks[0], dtype=float)
        for mask in masks:
            result *= mask
    elif method == "or":
        # Any mask being valid (1.0) means pixel is kept
        result = np.zeros_like(masks[0], dtype=float)
        for mask in masks:
            result = np.maximum(result, mask)
    else:
        raise ValueError(f"Unknown combination method: {method}")

    return result


def dilate_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Dilate a mask to expand masked regions.

    This is useful for conservative masking around clouds and other
    artifacts that may have edge effects.

    Args:
        mask: Input mask where 0 indicates pixels to mask
        radius: Dilation radius in pixels

    Returns:
        Dilated mask
    """
    if radius == 0:
        return mask.copy()

    # Simple dilation implementation without scipy
    # This is a basic box filter approach
    result = mask.copy()

    for _ in range(radius):
        # Create a temporary mask for this iteration
        temp_mask = result.copy()

        # For each pixel, if any neighbor is masked (0), mask this pixel too
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                # Check if any neighbor is masked
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ni, nj = i + di, j + dj
                        # Check bounds
                        if 0 <= ni < mask.shape[0] and 0 <= nj < mask.shape[1]:
                            if result[ni, nj] == 0:
                                temp_mask[i, j] = 0
                                break
                    if temp_mask[i, j] == 0:
                        break

        result = temp_mask

    return result


def mask_to_indices(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a boolean mask to array indices.

    Args:
        mask: Boolean mask array

    Returns:
        Tuple of (row_indices, col_indices) for True values
    """
    indices = np.where(mask > 0)
    return cast(tuple[np.ndarray, np.ndarray], indices)


def apply_mask(
    data: np.ndarray, mask: np.ndarray, fill_value: float = np.nan
) -> np.ndarray:
    """Apply a mask to data, filling masked pixels with fill_value.

    Args:
        data: Data array to mask
        mask: Mask array where 0 indicates pixels to mask
        fill_value: Value to use for masked pixels

    Returns:
        Masked data array
    """
    result = data.copy()
    result[mask == 0] = fill_value
    return result
