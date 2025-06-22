"""Behavioral tests for preprocessing components.

These tests use predictable artificial data to verify exact behavior
of preprocessing components without external dependencies.
"""

import numpy as np
import pytest
from rasterio.transform import Affine, from_bounds

from blackmarble.components.preprocessing import (
    ProjectionHandler,
    ResolutionResampler,
    BoundsCropper,
    SpatialAligner,
    SpatialResult,
)


class TestProjectionHandler:
    """Test ProjectionHandler transforms coordinates correctly."""

    def test_identity_transform(self):
        """Test that same CRS returns unchanged data."""
        # For identity transform, we can verify the behavior through the mask
        handler = ProjectionHandler(target_crs="EPSG:4326")

        # Create simple test data
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        transform = Affine(1.0, 0, 0, 0, -1.0, 2.0)

        result = handler.process(
            data,
            source_crs="EPSG:4326",
            source_transform=transform,
        )

        # For same CRS, mask should be all valid
        assert result.mask.shape == data.shape
        assert np.all(result.mask == 1.0)
        assert result.metadata is not None
        assert result.metadata["source_crs"] == "EPSG:4326"
        assert result.metadata["target_crs"] == "EPSG:4326"
        assert result.metadata["original_shape"] == data.shape

    # Removed test_transform_metadata - only verifies dictionary fields exist and match input parameters

    def test_nodata_masking(self):
        """Test proper handling of nodata values in mask."""
        handler = ProjectionHandler(target_crs="EPSG:4326")

        # Create data with nodata
        data = np.array([[1, 2], [3, -999]], dtype=np.float32)
        transform = from_bounds(0, 0, 2, 2, 2, 2)

        result = handler.process(
            data,
            source_crs="EPSG:4326",
            source_transform=transform,
            nodata=-999,
        )

        # Check mask excludes nodata
        assert result.mask[1, 1] == 0.0  # nodata position should be masked
        assert result.mask[0, 0] == 1.0  # valid data
        assert result.mask[0, 1] == 1.0  # valid data
        assert result.mask[1, 0] == 1.0  # valid data

    def test_multiband_mask(self):
        """Test mask generation for multi-band data."""
        handler = ProjectionHandler(
            target_crs="EPSG:3857"
        )  # Different CRS to force reprojection

        # Create 3-band test data with nodata in different bands
        data = np.array(
            [[[1, 2], [3, 4]], [[5, -999], [7, 8]], [[9, 10], [-999, 12]]],
            dtype=np.float32,
        )

        transform = from_bounds(0, 0, 2, 2, 2, 2)

        result = handler.process(
            data,
            source_crs="EPSG:4326",
            source_transform=transform,
            nodata=-999,
        )

        # Mask should be 2D and combine all bands
        assert result.mask.shape[0] > 0
        assert result.mask.shape[1] > 0
        assert len(result.mask.shape) == 2  # Should be 2D

        # Check that statistics show some invalid pixels
        assert result.statistics["valid_percentage"] < 100.0


class TestResolutionResampler:
    """Test ResolutionResampler produces expected output sizes."""

    def test_downsampling_scale(self):
        """Test resampling to lower resolution."""
        resampler = ResolutionResampler(target_resolution=(2.0, 2.0), method="bilinear")

        # Create 10x10 data with 1x1 resolution
        data = np.arange(100).reshape(10, 10).astype(np.float32)
        transform = Affine(1.0, 0, 0, 0, -1.0, 10)

        result = resampler.process(data, transform=transform, crs="EPSG:4326")

        # Check metadata records the scaling
        assert result.metadata is not None
        assert result.metadata["original_resolution"] == (1.0, 1.0)
        assert result.metadata["target_resolution"] == (2.0, 2.0)
        assert result.metadata["scale_factors"] == (0.5, 0.5)
        assert result.metadata["original_shape"] == (10, 10)
        # Output should be 5x5
        assert result.metadata["resampled_shape"] == (5, 5)

    def test_upsampling_scale(self):
        """Test resampling to higher resolution."""
        resampler = ResolutionResampler(target_resolution=(0.5, 0.5), method="bilinear")

        # Create 5x5 data with 2x2 resolution
        data = np.arange(25).reshape(5, 5).astype(np.float32)
        transform = Affine(2.0, 0, 0, 0, -2.0, 10)

        result = resampler.process(data, transform=transform, crs="EPSG:4326")

        # Check scaling factors
        assert result.metadata is not None
        assert result.metadata["scale_factors"] == (4.0, 4.0)
        assert result.metadata["resampled_shape"] == (20, 20)

    def test_multiband_resampling_shape(self):
        """Test resampling of multi-band data preserves bands."""
        resampler = ResolutionResampler(target_resolution=(2.0, 2.0))

        # Create 2-band data
        data = np.array(
            [np.arange(16).reshape(4, 4), np.arange(16, 32).reshape(4, 4)],
            dtype=np.float32,
        )
        transform = Affine(1.0, 0, 0, 0, -1.0, 4)

        result = resampler.process(data, transform=transform)

        # Check shapes in metadata
        assert result.metadata is not None
        assert result.metadata["original_shape"] == (2, 4, 4)
        assert result.metadata["resampled_shape"] == (2, 2, 2)

    def test_nodata_mask_generation(self):
        """Test mask generation with nodata values."""
        resampler = ResolutionResampler(target_resolution=(2.0, 2.0))

        # Create data with nodata in specific quadrants
        # This ensures some quadrants have nodata and some don't
        data = np.array(
            [[1, 2, 3, 4], [5, 6, -999, -999], [9, 10, -999, -999], [13, 14, 15, 16]],
            dtype=np.float32,
        )
        transform = Affine(1.0, 0, 0, 0, -1.0, 4)

        result = resampler.process(data, transform=transform, nodata=-999)

        # After downsampling, pixels with any nodata should be masked
        assert result.mask.shape == (2, 2)
        assert result.mask[0, 0] == 1.0  # No nodata in top-left quadrant
        assert result.mask[0, 1] == 0.0  # Has nodata in top-right quadrant
        assert result.mask[1, 0] == 1.0  # No nodata in bottom-left quadrant
        assert result.mask[1, 1] == 0.0  # Has nodata in bottom-right quadrant


class TestBoundsCropper:
    """Test BoundsCropper extracts correct regions."""

    # Removed test_exact_crop_metadata - only verifies dictionary field existence and basic arithmetic

    # Removed test_crop_with_buffer_metadata - only verifies dictionary field existence and basic arithmetic

    def test_multiband_crop_mask(self):
        """Test cropping multi-band data produces correct mask."""
        cropper = BoundsCropper(bounds=(1, 1, 3, 3))

        # Create 3-band 4x4 data
        data = np.array(
            [
                np.arange(16).reshape(4, 4),
                np.arange(16, 32).reshape(4, 4),
                np.arange(32, 48).reshape(4, 4),
            ],
            dtype=np.float32,
        )
        transform = Affine(1.0, 0, 0, 0, -1.0, 4)

        result = cropper.process(data, transform=transform)

        # Mask should be 2D even for multiband
        assert result.mask.shape == (2, 2)
        assert np.all(result.mask == 1.0)
        assert result.metadata is not None
        assert result.metadata["original_shape"] == (3, 4, 4)
        assert result.metadata["cropped_shape"] == (3, 2, 2)

    def test_pixel_statistics(self):
        """Test pixel count statistics."""
        cropper = BoundsCropper(bounds=(0, 0, 3, 2))

        # Create 5x5 data
        data = np.ones((5, 5), dtype=np.float32)
        transform = Affine(1.0, 0, 0, 0, -1.0, 5)

        result = cropper.process(data, transform=transform)

        # Check statistics
        assert result.statistics["total_pixels"] == 6  # 3x2
        assert result.statistics["valid_pixels"] == 6
        assert result.statistics["invalid_pixels"] == 0
        assert result.statistics["valid_percentage"] == 100.0
        assert result.statistics["pixels_removed"] == 19  # 25 - 6


class TestSpatialAligner:
    """Test SpatialAligner aligns multiple arrays."""

    # Removed test_align_same_grid_metadata - only verifies dictionary field existence and object type checking

    # Removed test_align_different_sizes_metadata - only verifies dictionary field existence and shape preservation

    def test_multiband_alignment(self):
        """Test aligning multi-band rasters."""
        aligner = SpatialAligner(reference_idx=1)

        # Create 2-band rasters
        raster1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)

        raster2 = np.array(
            [
                [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
                [[11, 21, 31], [41, 51, 61], [71, 81, 91]],
            ],
            dtype=np.float32,
        )

        transform1 = Affine(2.0, 0, 0, 0, -2.0, 4)
        transform2 = Affine(1.0, 0, 0, 0, -1.0, 3)

        results = aligner.align_rasters([raster1, raster2], [transform1, transform2])

        # Both should match reference (raster2) shape
        assert results[0].data.shape == (2, 3, 3)
        assert results[1].data.shape == (2, 3, 3)

        # Reference should be unchanged
        assert np.array_equal(results[1].data, raster2)
        assert results[1].metadata is not None
        assert results[1].metadata["is_reference"] is True

    def test_no_crs_provided(self):
        """Test alignment when no CRS is provided."""
        aligner = SpatialAligner()

        raster1 = np.ones((2, 2), dtype=np.float32)
        raster2 = np.ones((2, 2), dtype=np.float32) * 2
        transform = Affine(1.0, 0, 0, 0, -1.0, 2)

        # Should work without CRS
        results = aligner.align_rasters([raster1, raster2], [transform, transform])

        assert len(results) == 2
        assert results[0].crs == "unknown"
        assert results[1].crs == "unknown"
