"""Temporal processing components for time series operations.

This module provides components for temporal compositing, scene merging,
and date range selection for satellite imagery time series.

Advanced Features:
------------------
This module includes advanced features using scipy:

1. Gap Filling with Nearest Neighbor Interpolation:
   - Enabled via `use_scipy_fill_gaps=True` in TemporalCompositor
   - Uses scipy's Euclidean distance transform for sophisticated gap filling
   - Finds the nearest valid pixel for each invalid pixel
   - Produces smoother, more accurate results than simple methods

2. Feather Blending for Scene Merging:
   - Enabled by setting `blend_method="feather"` in SceneMerger
   - Uses scipy's distance transform to create smooth transitions
   - Prevents visible seams when merging overlapping satellite scenes
   - Provides professional-quality mosaicking

Example:
    # With scipy gap filling
    compositor = TemporalCompositor(
        method="median",
        use_scipy_fill_gaps=True
    )
    
    # With feather blending
    merger = SceneMerger(
        blend_method="feather",
        feather_distance=10
    )
"""

from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from enum import Enum

from .base import ComposableComponent, MaskResult, QualityStatistics


class CompositeMethod(Enum):
    """Available temporal compositing methods."""

    MEDIAN = "median"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    PERCENTILE = "percentile"
    QUALITY_WEIGHTED = "quality_weighted"


@dataclass
class TemporalResult:
    """Result of a temporal processing operation.

    Attributes:
        data: Processed array data
        dates: List of dates included in processing
        coverage: Temporal coverage statistics
        method: Compositing method used
        metadata: Additional metadata
    """

    data: np.ndarray
    dates: List[datetime]
    coverage: Dict[str, Any]
    method: str
    metadata: Optional[Dict[str, Any]] = None


class TemporalCompositor(ComposableComponent):
    """Create temporal composites from time series data.

    Supports various compositing methods including median, mean,
    and quality-weighted composites for reducing cloud cover and
    improving data quality.
    """

    def __init__(
        self,
        name: str = "temporal_compositor",
        method: Union[str, CompositeMethod] = CompositeMethod.MEDIAN,
        percentile: Optional[int] = None,
        min_valid_observations: int = 1,
        fill_nodata: bool = True,
        use_scipy_fill_gaps: bool = False,
    ):
        """Initialize temporal compositor.

        Args:
            name: Component identifier
            method: Compositing method to use for combining temporal observations.
                Options: median, mean, max, min, percentile, quality_weighted
            percentile: Percentile value if method is PERCENTILE (0-100)
            min_valid_observations: Minimum observations for valid composite pixel.
                Pixels with fewer observations will be marked as invalid.
            fill_nodata: Whether to fill nodata values across time using valid
                observations from other dates
            use_scipy_fill_gaps: Enable advanced gap filling using scipy's nearest
                neighbor interpolation. This produces higher quality results by
                finding the nearest valid pixel for each gap. Default: False
        """
        super().__init__(name)
        if isinstance(method, str):
            method = CompositeMethod(method)
        self.method = method
        self.percentile = percentile
        self.min_valid_observations = min_valid_observations
        self.fill_nodata = fill_nodata
        self.use_scipy_fill_gaps = use_scipy_fill_gaps
        
        # scipy is now a required dependency, so no need to check

        self._config.update(
            {
                "method": method.value,
                "percentile": percentile,
                "min_valid_observations": min_valid_observations,
                "fill_nodata": fill_nodata,
                "use_scipy_fill_gaps": use_scipy_fill_gaps,
            }
        )

    def process(self, data: np.ndarray, **kwargs) -> MaskResult:
        """Create temporal composite from stacked time series.

        Args:
            data: 3D array (time, height, width) or 4D (time, bands, height, width)
            dates: List of acquisition dates
            quality_masks: Optional quality masks for each time step
            weights: Optional weights for quality-weighted composite
            nodata: Nodata value (default: np.nan)

        Returns:
            MaskResult with composite validity mask
        """
        if data.ndim < 3:
            raise ValueError(f"Expected 3D or 4D array, got shape {data.shape}")

        dates = kwargs.get("dates", [])
        quality_masks = kwargs.get("quality_masks")
        weights = kwargs.get("weights")
        nodata = kwargs.get("nodata", np.nan)

        # Handle nodata values
        if np.isnan(nodata):
            valid_mask = ~np.isnan(data)
        else:
            valid_mask = data != nodata

        # Apply quality masks if provided
        if quality_masks is not None:
            if quality_masks.shape[0] != data.shape[0]:
                raise ValueError("Number of quality masks must match time dimension")
            # Expand quality masks to match data dimensions
            if data.ndim == 4 and quality_masks.ndim == 3:
                quality_masks = quality_masks[:, np.newaxis, :, :]
            valid_mask = valid_mask & (quality_masks > 0)

        # Create composite based on method
        if self.method == CompositeMethod.MEDIAN:
            composite = self._median_composite(data, valid_mask)
        elif self.method == CompositeMethod.MEAN:
            composite = self._mean_composite(data, valid_mask)
        elif self.method == CompositeMethod.MAX:
            composite = self._max_composite(data, valid_mask)
        elif self.method == CompositeMethod.MIN:
            composite = self._min_composite(data, valid_mask)
        elif self.method == CompositeMethod.PERCENTILE:
            if self.percentile is None:
                raise ValueError("Percentile value required for PERCENTILE method")
            composite = self._percentile_composite(data, valid_mask, self.percentile)
        elif self.method == CompositeMethod.QUALITY_WEIGHTED:
            if weights is None:
                raise ValueError("Weights required for quality-weighted composite")
            composite = self._quality_weighted_composite(data, valid_mask, weights)
        else:
            raise ValueError(f"Unknown composite method: {self.method}")

        # Count valid observations per pixel
        if data.ndim == 3:
            valid_count = np.sum(valid_mask, axis=0)
        else:
            # For multi-band, use minimum valid count across bands
            valid_count = np.min(np.sum(valid_mask, axis=(0, 1)), axis=0)

        # Create output mask based on minimum observations
        output_mask = (valid_count >= self.min_valid_observations).astype(float)

        # Fill nodata if requested
        if self.fill_nodata and np.any(output_mask == 0):
            if self.use_scipy_fill_gaps:
                composite = self._fill_gaps_scipy(composite, output_mask)
            else:
                # Simple filling - just leave gaps as-is
                # The basic temporal compositing already fills many gaps
                # by using valid observations from other dates
                pass

        # Store composite in kwargs
        kwargs["composite"] = composite
        kwargs["valid_count"] = valid_count

        # Calculate temporal coverage statistics
        coverage_stats = self._calculate_coverage(valid_mask, dates)

        # Generate statistics
        stats = QualityStatistics.from_mask(output_mask)

        metadata = {
            "method": self.method.value,
            "time_steps": data.shape[0],
            "dates_provided": len(dates),
            "min_observations": self.min_valid_observations,
            "mean_observations": float(np.mean(valid_count)),
            "max_observations": int(np.max(valid_count)),
            "coverage": coverage_stats,
            "composite": composite,
            "valid_count": valid_count,
        }

        return MaskResult(
            mask=output_mask, statistics=stats.__dict__, metadata=metadata
        )

    def _median_composite(self, data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Create median composite."""
        # Use masked array for efficient computation
        masked_data = np.ma.masked_where(~valid_mask, data)
        return np.ma.median(masked_data, axis=0).filled(np.nan)

    def _mean_composite(self, data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Create mean composite."""
        masked_data = np.ma.masked_where(~valid_mask, data)
        return np.ma.mean(masked_data, axis=0).filled(np.nan)

    def _max_composite(self, data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Create maximum value composite."""
        masked_data = np.ma.masked_where(~valid_mask, data)
        return np.ma.max(masked_data, axis=0).filled(np.nan)

    def _min_composite(self, data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Create minimum value composite."""
        masked_data = np.ma.masked_where(~valid_mask, data)
        return np.ma.min(masked_data, axis=0).filled(np.nan)

    def _percentile_composite(
        self, data: np.ndarray, valid_mask: np.ndarray, percentile: int
    ) -> np.ndarray:
        """Create percentile composite."""
        # For percentile, we need to handle each pixel separately
        # due to varying number of valid observations
        output_shape = data.shape[1:]
        composite = np.full(output_shape, np.nan)

        # Process each pixel
        if data.ndim == 3:
            for i in range(output_shape[0]):
                for j in range(output_shape[1]):
                    valid_data = data[valid_mask[:, i, j], i, j]
                    if len(valid_data) > 0:
                        composite[i, j] = np.percentile(valid_data, percentile)
        else:
            # Multi-band case
            for b in range(output_shape[0]):
                for i in range(output_shape[1]):
                    for j in range(output_shape[2]):
                        valid_data = data[valid_mask[:, b, i, j], b, i, j]
                        if len(valid_data) > 0:
                            composite[b, i, j] = np.percentile(valid_data, percentile)

        return composite

    def _quality_weighted_composite(
        self, data: np.ndarray, valid_mask: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Create quality-weighted composite."""
        # Ensure weights match data shape
        if weights.shape[0] != data.shape[0]:
            raise ValueError("Weights must match time dimension")

        # Expand weights to match data dimensions if needed
        if data.ndim == 4 and weights.ndim == 3:
            weights = weights[:, np.newaxis, :, :]
        elif data.ndim == 3 and weights.ndim == 1:
            weights = weights[:, np.newaxis, np.newaxis]

        # Apply valid mask to weights
        masked_weights = weights * valid_mask

        # Normalize weights
        weight_sum = np.sum(masked_weights, axis=0)
        weight_sum[weight_sum == 0] = 1  # Avoid division by zero

        # Compute weighted average
        weighted_data = data * masked_weights
        composite = np.sum(weighted_data, axis=0) / weight_sum

        return composite

    def _fill_gaps_scipy(self, composite: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Fill gaps in composite using scipy's nearest neighbor interpolation.
        
        Uses scipy's Euclidean distance transform to find the nearest valid pixel
        for each gap. This produces high-quality results that preserve spatial
        patterns and avoid artifacts.
        
        Args:
            composite: The composite array with gaps (2D or multi-band)
            valid_mask: Boolean mask indicating valid pixels
            
        Returns:
            Composite array with gaps filled using nearest neighbor interpolation
            
        Note:
            This method requires scipy to be installed and use_scipy_fill_gaps=True
        """
        from scipy.ndimage import distance_transform_edt

        # Find indices of valid pixels
        if composite.ndim == 2:
            valid_indices = np.where(valid_mask > 0)
            invalid_indices = np.where(valid_mask == 0)

            if len(valid_indices[0]) > 0 and len(invalid_indices[0]) > 0:
                # Use distance transform to find nearest valid pixel
                indices = distance_transform_edt(
                    valid_mask == 0, return_distances=False, return_indices=True
                )
                # indices is shape (2, height, width) where indices[d, i, j]
                # gives the d-th coordinate of nearest valid pixel to (i, j)
                # Extract the nearest neighbor coordinates for invalid pixels
                row_coords = invalid_indices[0]
                col_coords = invalid_indices[1]
                nearest_rows = indices[0, row_coords, col_coords]
                nearest_cols = indices[1, row_coords, col_coords]
                composite[row_coords, col_coords] = composite[
                    nearest_rows, nearest_cols
                ]
        else:
            # Multi-band case - fill each band separately
            for b in range(composite.shape[0]):
                band_mask = valid_mask if valid_mask.ndim == 2 else valid_mask[b]
                valid_indices = np.where(band_mask > 0)
                invalid_indices = np.where(band_mask == 0)

                if len(valid_indices[0]) > 0 and len(invalid_indices[0]) > 0:
                    indices = distance_transform_edt(
                        band_mask == 0, return_distances=False, return_indices=True
                    )
                    row_coords = invalid_indices[0]
                    col_coords = invalid_indices[1]
                    nearest_rows = indices[0, row_coords, col_coords]
                    nearest_cols = indices[1, row_coords, col_coords]
                    composite[b, row_coords, col_coords] = composite[
                        b, nearest_rows, nearest_cols
                    ]

        return composite

    def _calculate_coverage(
        self, valid_mask: np.ndarray, dates: List[datetime]
    ) -> Dict[str, Any]:
        """Calculate temporal coverage statistics."""
        coverage = {
            "temporal_range_days": 0,
            "observation_frequency": 0,
            "coverage_percentage": 0,
        }

        if dates and len(dates) == valid_mask.shape[0]:
            # Calculate temporal range
            date_range = max(dates) - min(dates)
            coverage["temporal_range_days"] = date_range.days

            # Calculate observation frequency
            if coverage["temporal_range_days"] > 0:
                coverage["observation_frequency"] = (
                    len(dates) / coverage["temporal_range_days"]
                )

            # Calculate overall coverage percentage
            total_possible = valid_mask.size
            total_valid = np.sum(valid_mask)
            coverage["coverage_percentage"] = 100.0 * total_valid / total_possible

        return coverage


class SceneMerger(ComposableComponent):
    """Merge multiple satellite scenes into a single mosaic.

    Handles overlapping scenes with various blending methods
    and seamline optimization.
    """

    def __init__(
        self,
        name: str = "scene_merger",
        blend_method: str = "feather",
        feather_distance: int = 10,
        priority_order: Optional[List[int]] = None,
    ):
        """Initialize scene merger.

        Args:
            name: Component identifier
            blend_method: Method for blending overlapping scenes:
                - "feather": Smooth transitions using distance-based weighting.
                  Produces professional seamless mosaics.
                - "max": Take maximum value at each pixel
                - "min": Take minimum value at each pixel  
                - "mean": Average overlapping pixels
            feather_distance: Distance for feathering in pixels. Only used when
                blend_method="feather". Larger values create smoother but slower
                transitions. Default: 10 pixels
            priority_order: Optional priority order for scenes. Lower indices have
                higher priority when blend_method is not "mean".
        """
        super().__init__(name)
        self.blend_method = blend_method
        self.feather_distance = feather_distance
        self.priority_order = priority_order
        
        # scipy is now a required dependency, so no need to check

        self._config.update(
            {
                "blend_method": blend_method,
                "feather_distance": feather_distance,
                "priority_order": priority_order,
            }
        )

    def process(self, data: np.ndarray, **kwargs) -> MaskResult:
        """Process is not used directly for merging."""
        raise NotImplementedError(
            "SceneMerger requires merge_scenes() method instead of process()"
        )

    def merge_scenes(
        self,
        scenes: List[np.ndarray],
        transforms: List[Any],
        bounds: List[Tuple[float, float, float, float]],
        nodata: Optional[float] = None,
    ) -> Tuple[np.ndarray, Any, Tuple[float, float, float, float]]:
        """Merge multiple scenes into a mosaic.

        Args:
            scenes: List of scene arrays
            transforms: List of affine transforms
            bounds: List of (min_x, min_y, max_x, max_y) tuples
            nodata: Nodata value

        Returns:
            Tuple of (merged_data, merged_transform, merged_bounds)
        """
        if not scenes:
            raise ValueError("No scenes provided for merging")

        if len(scenes) != len(transforms) or len(scenes) != len(bounds):
            raise ValueError("Number of scenes, transforms, and bounds must match")

        # Calculate output bounds
        min_x = min(b[0] for b in bounds)
        min_y = min(b[1] for b in bounds)
        max_x = max(b[2] for b in bounds)
        max_y = max(b[3] for b in bounds)
        merged_bounds = (min_x, min_y, max_x, max_y)

        # Use first transform's resolution
        res_x = abs(transforms[0].a)
        res_y = abs(transforms[0].e)

        # Calculate output dimensions
        width = int(np.ceil((max_x - min_x) / res_x))
        height = int(np.ceil((max_y - min_y) / res_y))

        # Create output transform
        from rasterio.transform import from_bounds

        merged_transform = from_bounds(min_x, min_y, max_x, max_y, width, height)

        # Initialize output array with appropriate fill values based on blend method
        shape = scenes[0].shape
        if self.blend_method == "max":
            fill_value = -np.inf
        elif self.blend_method == "min":
            fill_value = np.inf
        else:
            fill_value = nodata or np.nan

        if len(shape) == 2:
            merged = np.full((height, width), fill_value, dtype=scenes[0].dtype)
        else:
            merged = np.full(
                (shape[0], height, width), fill_value, dtype=scenes[0].dtype
            )

        # Apply priority order if specified
        scene_order = self.priority_order or list(range(len(scenes)))

        # Merge scenes based on method
        if self.blend_method == "feather":
            merged = self._feather_blend(
                merged, scenes, transforms, merged_transform, scene_order, nodata
            )
        elif self.blend_method == "max":
            merged = self._max_merge(
                merged, scenes, transforms, merged_transform, scene_order, nodata
            )
        elif self.blend_method == "min":
            merged = self._min_merge(
                merged, scenes, transforms, merged_transform, scene_order, nodata
            )
        elif self.blend_method == "mean":
            merged = self._mean_merge(
                merged, scenes, transforms, merged_transform, scene_order, nodata
            )
        else:
            raise ValueError(f"Unknown blend method: {self.blend_method}")

        return merged, merged_transform, merged_bounds

    def _feather_blend(
        self,
        output: np.ndarray,
        scenes: List[np.ndarray],
        transforms: List[Any],
        out_transform: Any,
        order: List[int],
        nodata: Optional[float],
    ) -> np.ndarray:
        """Blend scenes using feathering for smooth seamless transitions.
        
        Uses scipy's Euclidean distance transform to calculate weights based on
        distance from edges. This creates smooth transitions between overlapping
        scenes, eliminating visible seams.
        
        The algorithm:
        1. For each scene, calculate distance from valid pixel edges
        2. Normalize distances to create blend weights (0-1)
        3. In overlap regions, blend pixels based on distance weights
        4. Scenes are processed in priority order
        
        Args:
            output: Output array to fill
            scenes: List of scene arrays to merge
            transforms: Affine transforms for each scene
            out_transform: Output affine transform
            order: Processing order for scenes (priority)
            nodata: Value representing no data
            
        Returns:
            Blended output array with smooth transitions
            
        Note:
            Requires scipy to be installed and blend_method="feather"
        """
        from scipy.ndimage import distance_transform_edt

        # Process scenes in priority order
        for idx in order:
            scene = scenes[idx]
            transform = transforms[idx]

            # Calculate offset in output array
            col_off = int((transform.c - out_transform.c) / out_transform.a)
            row_off = int((transform.f - out_transform.f) / out_transform.e)

            # Get valid data mask
            if nodata is None or np.isnan(nodata):
                valid_mask = ~np.isnan(scene)
            else:
                valid_mask = scene != nodata

            if scene.ndim > 2:
                valid_mask = np.all(valid_mask, axis=0)

            # Calculate distance from edges for feathering
            distance = distance_transform_edt(valid_mask)
            distance = (
                np.minimum(distance, self.feather_distance) / self.feather_distance
            )

            # Apply feathered blend
            if scene.ndim == 2:
                # Get slice of output
                out_slice = output[
                    row_off : row_off + scene.shape[0],
                    col_off : col_off + scene.shape[1],
                ]

                # Blend with distance weights
                # Check where we have overlapping data
                existing_valid = ~np.isnan(out_slice)
                blend_mask = valid_mask & existing_valid

                # For overlapping regions, blend based on distance
                if np.any(blend_mask):
                    out_slice[blend_mask] = (
                        distance[blend_mask] * scene[blend_mask]
                        + (1 - distance[blend_mask]) * out_slice[blend_mask]
                    )

                # For non-overlapping regions, just copy the data
                new_data_mask = valid_mask & ~existing_valid
                out_slice[new_data_mask] = scene[new_data_mask]
            else:
                # Multi-band case
                for b in range(scene.shape[0]):
                    out_slice = output[
                        b,
                        row_off : row_off + scene.shape[1],
                        col_off : col_off + scene.shape[2],
                    ]

                    band_valid = valid_mask if valid_mask.ndim == 2 else valid_mask[b]

                    # Check where we have overlapping data
                    existing_valid = ~np.isnan(out_slice)
                    blend_mask = band_valid & existing_valid

                    # For overlapping regions, blend based on distance
                    if np.any(blend_mask):
                        out_slice[blend_mask] = (
                            distance[blend_mask] * scene[b][blend_mask]
                            + (1 - distance[blend_mask]) * out_slice[blend_mask]
                        )

                    # For non-overlapping regions, just copy the data
                    new_data_mask = band_valid & ~existing_valid
                    out_slice[new_data_mask] = scene[b][new_data_mask]

        return output

    def _max_merge(
        self,
        output: np.ndarray,
        scenes: List[np.ndarray],
        transforms: List[Any],
        out_transform: Any,
        order: List[int],
        nodata: Optional[float],
    ) -> np.ndarray:
        """Merge using maximum values."""
        for idx in order:
            scene = scenes[idx]
            transform = transforms[idx]

            # Calculate offset
            col_off = int((transform.c - out_transform.c) / out_transform.a)
            row_off = int((transform.f - out_transform.f) / out_transform.e)

            # Get valid data mask
            if nodata is None or np.isnan(nodata):
                valid_mask = ~np.isnan(scene)
            else:
                valid_mask = scene != nodata

            # Apply maximum
            if scene.ndim == 2:
                out_slice = output[
                    row_off : row_off + scene.shape[0],
                    col_off : col_off + scene.shape[1],
                ]
                # Use where to only update valid pixels
                np.maximum(out_slice, scene, out=out_slice, where=valid_mask)
            else:
                for b in range(scene.shape[0]):
                    out_slice = output[
                        b,
                        row_off : row_off + scene.shape[1],
                        col_off : col_off + scene.shape[2],
                    ]
                    band_valid = valid_mask if valid_mask.ndim == 2 else valid_mask[b]
                    np.maximum(out_slice, scene[b], out=out_slice, where=band_valid)

        # Replace -inf with nodata
        output[output == -np.inf] = nodata or np.nan
        return output

    def _min_merge(
        self,
        output: np.ndarray,
        scenes: List[np.ndarray],
        transforms: List[Any],
        out_transform: Any,
        order: List[int],
        nodata: Optional[float],
    ) -> np.ndarray:
        """Merge using minimum values."""
        # Similar to max but using minimum
        for idx in order:
            scene = scenes[idx]
            transform = transforms[idx]

            col_off = int((transform.c - out_transform.c) / out_transform.a)
            row_off = int((transform.f - out_transform.f) / out_transform.e)

            if nodata is None or np.isnan(nodata):
                valid_mask = ~np.isnan(scene)
            else:
                valid_mask = scene != nodata

            if scene.ndim == 2:
                out_slice = output[
                    row_off : row_off + scene.shape[0],
                    col_off : col_off + scene.shape[1],
                ]
                # Use where to only update valid pixels
                np.minimum(out_slice, scene, out=out_slice, where=valid_mask)
            else:
                for b in range(scene.shape[0]):
                    out_slice = output[
                        b,
                        row_off : row_off + scene.shape[1],
                        col_off : col_off + scene.shape[2],
                    ]
                    band_valid = valid_mask if valid_mask.ndim == 2 else valid_mask[b]
                    np.minimum(out_slice, scene[b], out=out_slice, where=band_valid)

        output[output == np.inf] = nodata or np.nan
        return output

    def _mean_merge(
        self,
        output: np.ndarray,
        scenes: List[np.ndarray],
        transforms: List[Any],
        out_transform: Any,
        order: List[int],
        nodata: Optional[float],
    ) -> np.ndarray:
        """Merge using mean values."""
        # Initialize sum and count arrays
        if output.ndim == 2:
            sum_array = np.zeros_like(output)
            count_array = np.zeros_like(output, dtype=int)
        else:
            sum_array = np.zeros_like(output)
            count_array = np.zeros(output.shape[1:], dtype=int)

        # Accumulate values
        for idx in order:
            scene = scenes[idx]
            transform = transforms[idx]

            col_off = int((transform.c - out_transform.c) / out_transform.a)
            row_off = int((transform.f - out_transform.f) / out_transform.e)

            if nodata is None or np.isnan(nodata):
                valid_mask = ~np.isnan(scene)
            else:
                valid_mask = scene != nodata

            if scene.ndim == 2:
                sum_slice = sum_array[
                    row_off : row_off + scene.shape[0],
                    col_off : col_off + scene.shape[1],
                ]
                count_slice = count_array[
                    row_off : row_off + scene.shape[0],
                    col_off : col_off + scene.shape[1],
                ]

                sum_slice[valid_mask] += scene[valid_mask]
                count_slice[valid_mask] += 1
            else:
                for b in range(scene.shape[0]):
                    sum_slice = sum_array[
                        b,
                        row_off : row_off + scene.shape[1],
                        col_off : col_off + scene.shape[2],
                    ]
                    band_valid = valid_mask if valid_mask.ndim == 2 else valid_mask[b]
                    sum_slice[band_valid] += scene[b][band_valid]

                count_slice = count_array[
                    row_off : row_off + scene.shape[1],
                    col_off : col_off + scene.shape[2],
                ]
                if valid_mask.ndim == 2:
                    count_slice[valid_mask] += 1
                else:
                    count_slice[np.any(valid_mask, axis=0)] += 1

        # Calculate mean and return the result
        if output.ndim == 2:
            result = np.where(
                count_array > 0, sum_array / count_array, nodata or np.nan
            )
        else:
            result = np.zeros_like(output)
            for b in range(output.shape[0]):
                result[b] = np.where(
                    count_array > 0, sum_array[b] / count_array, nodata or np.nan
                )

        return result


class DateRangeSelector(ComposableComponent):
    """Select appropriate scenes based on date criteria.

    Filters and selects satellite scenes based on temporal
    criteria such as date ranges, seasonal constraints, and
    cloud cover thresholds.
    """

    def __init__(
        self,
        name: str = "date_range_selector",
        target_date: Optional[datetime] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        max_days_from_target: Optional[int] = None,
        season_filter: Optional[List[str]] = None,
        cloud_threshold: Optional[float] = None,
        min_scenes: int = 1,
        max_scenes: Optional[int] = None,
    ):
        """Initialize date range selector.

        Args:
            name: Component identifier
            target_date: Target acquisition date
            date_range: Explicit date range (start, end)
            max_days_from_target: Maximum days from target date
            season_filter: List of seasons to include (spring, summer, fall, winter)
            cloud_threshold: Maximum cloud cover percentage
            min_scenes: Minimum number of scenes required
            max_scenes: Maximum number of scenes to select
        """
        super().__init__(name)
        self.target_date = target_date
        self.date_range = date_range
        self.max_days_from_target = max_days_from_target
        self.season_filter = season_filter
        self.cloud_threshold = cloud_threshold
        self.min_scenes = min_scenes
        self.max_scenes = max_scenes

        self._config.update(
            {
                "target_date": target_date.isoformat() if target_date else None,
                "date_range": (
                    (date_range[0].isoformat(), date_range[1].isoformat())
                    if date_range
                    else None
                ),
                "max_days_from_target": max_days_from_target,
                "season_filter": season_filter,
                "cloud_threshold": cloud_threshold,
                "min_scenes": min_scenes,
                "max_scenes": max_scenes,
            }
        )

    def process(self, data: np.ndarray, **kwargs) -> MaskResult:
        """Process is not used directly for selection."""
        raise NotImplementedError(
            "DateRangeSelector requires select_scenes() method instead of process()"
        )

    def select_scenes(
        self,
        scene_dates: List[datetime],
        scene_metadata: Optional[List[Dict[str, Any]]] = None,
        quality_scores: Optional[List[float]] = None,
    ) -> List[int]:
        """Select scene indices based on date criteria.

        Args:
            scene_dates: List of scene acquisition dates
            scene_metadata: Optional metadata for each scene
            quality_scores: Optional quality scores for ranking

        Returns:
            List of selected scene indices
        """
        if not scene_dates:
            return []

        # Start with all scenes
        candidates = list(range(len(scene_dates)))

        # Apply date range filter
        if self.date_range:
            candidates = [
                i
                for i in candidates
                if self.date_range[0] <= scene_dates[i] <= self.date_range[1]
            ]

        # Apply target date filter
        if self.target_date and self.max_days_from_target:
            candidates = [
                i
                for i in candidates
                if abs((scene_dates[i] - self.target_date).days)
                <= self.max_days_from_target
            ]

        # Apply season filter
        if self.season_filter:
            candidates = [
                i
                for i in candidates
                if self._get_season(scene_dates[i]) in self.season_filter
            ]

        # Apply cloud cover filter
        if self.cloud_threshold is not None and scene_metadata:
            candidates = [
                i
                for i in candidates
                if i < len(scene_metadata)
                and scene_metadata[i].get("cloud_cover", 100) <= self.cloud_threshold
            ]

        # Check minimum scenes requirement
        if len(candidates) < self.min_scenes:
            # Relax constraints and try again
            # Start with original list and only apply essential filters
            candidates = list(range(len(scene_dates)))
            if self.date_range:
                candidates = [
                    i
                    for i in candidates
                    if self.date_range[0] <= scene_dates[i] <= self.date_range[1]
                ]

        # Sort by quality if scores provided
        if quality_scores:
            candidates.sort(key=lambda i: quality_scores[i], reverse=True)
        elif self.target_date:
            # Sort by distance from target date
            candidates.sort(key=lambda i: abs((scene_dates[i] - self.target_date).days))

        # Apply maximum scenes limit
        if self.max_scenes and len(candidates) > self.max_scenes:
            candidates = candidates[: self.max_scenes]

        return candidates

    def _get_season(self, date: datetime) -> str:
        """Get season for a given date (Northern Hemisphere)."""
        month = date.month
        if month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        elif month in [9, 10, 11]:
            return "fall"
        else:
            return "winter"
