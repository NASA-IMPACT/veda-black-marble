"""Behavioral tests for preprocessing components.

These tests use predictable artificial data to verify exact behavior
of preprocessing components without external dependencies.
"""

import numpy as np
import pytest
from rasterio.transform import Affine, from_bounds

from blackmarble.processing import (
    reproject_image,
    resample_image,
    crop_to_bounds,
    align_rasters,
)


class TestProjectionHandler:
    """Test projection/reprojection functions."""

    def test_identity_transform(self):
        """Test that same CRS returns unchanged data."""
        # Create simple test data
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        transform = Affine(1.0, 0, 0, 0, -1.0, 2.0)

        reprojected, new_transform, new_bounds = reproject_image(
            data,
            source_crs="EPSG:4326",
            target_crs="EPSG:4326",
            source_transform=transform,
        )

        # For same CRS, data should be unchanged
        assert reprojected.shape == data.shape
        np.testing.assert_array_almost_equal(reprojected, data)

    # Removed test_transform_metadata - only verifies dictionary fields exist and match input parameters

    def test_nodata_handling(self):
        """Test proper handling of nodata values."""
        # Create data with nodata
        data = np.array([[1, 2], [3, -999]], dtype=np.float32)
        transform = from_bounds(0, 0, 2, 2, 2, 2)

        reprojected, new_transform, new_bounds = reproject_image(
            data,
            source_crs="EPSG:4326",
            target_crs="EPSG:4326",
            source_transform=transform,
            nodata=-999,
        )

        # Check that nodata is preserved
        assert reprojected[1, 1] == -999  # nodata position should be preserved

    def test_multiband_reprojection(self):
        """Test reprojection of multi-band data."""
        # Create 3-band test data with nodata in different bands
        data = np.array(
            [[[1, 2], [3, 4]], [[5, -999], [7, 8]], [[9, 10], [-999, 12]]],
            dtype=np.float32,
        )

        transform = from_bounds(0, 0, 2, 2, 2, 2)

        reprojected, new_transform, new_bounds = reproject_image(
            data,
            source_crs="EPSG:4326",
            target_crs="EPSG:3857",  # Different CRS to force reprojection
            source_transform=transform,
            nodata=-999,
        )

        # Should preserve band dimension
        assert len(reprojected.shape) == 3  # Should be 3D
        assert reprojected.shape[0] == 3  # 3 bands


class TestResolutionResampler:
    """Test resolution resampling functions."""

    def test_downsampling_scale(self):
        """Test resampling to lower resolution."""
        # Create 10x10 data with 1x1 resolution
        data = np.arange(100).reshape(10, 10).astype(np.float32)
        transform = Affine(1.0, 0, 0, 0, -1.0, 10)

        resampled, new_transform = resample_image(
            data,
            source_resolution=1.0,
            target_resolution=2.0,
            method="bilinear",
            transform=transform,
        )

        # Output should be 5x5 (downsampled by factor of 2)
        assert resampled.shape == (5, 5)

    def test_upsampling_scale(self):
        """Test resampling to higher resolution."""
        # Create 5x5 data with 2x2 resolution
        data = np.arange(25).reshape(5, 5).astype(np.float32)
        transform = Affine(2.0, 0, 0, 0, -2.0, 10)

        resampled, new_transform = resample_image(
            data,
            source_resolution=2.0,
            target_resolution=0.5,
            method="bilinear",
            transform=transform,
        )

        # Output should be 20x20 (upsampled by factor of 4)
        assert resampled.shape == (20, 20)

    def test_multiband_resampling_shape(self):
        """Test resampling of multi-band data preserves bands."""
        # Create 2-band data
        data = np.array(
            [np.arange(16).reshape(4, 4), np.arange(16, 32).reshape(4, 4)],
            dtype=np.float32,
        )
        transform = Affine(1.0, 0, 0, 0, -1.0, 4)

        resampled, new_transform = resample_image(
            data,
            source_resolution=1.0,
            target_resolution=2.0,
            transform=transform,
        )

        # Check shapes - should preserve bands
        assert resampled.shape == (2, 2, 2)  # 2 bands, 2x2 pixels

    def test_nodata_handling_in_resampling(self):
        """Test handling of nodata values during resampling."""
        # Create data with nodata in specific quadrants
        # This ensures some quadrants have nodata and some don't
        data = np.array(
            [[1, 2, 3, 4], [5, 6, -999, -999], [9, 10, -999, -999], [13, 14, 15, 16]],
            dtype=np.float32,
        )
        transform = Affine(1.0, 0, 0, 0, -1.0, 4)

        resampled, new_transform = resample_image(
            data,
            source_resolution=1.0,
            target_resolution=2.0,
            transform=transform,
            nodata=-999,
        )

        # After downsampling, check that nodata regions are handled
        assert resampled.shape == (2, 2)
        # Top-right and bottom-right quadrants had nodata
        assert not np.isnan(resampled[0, 0])  # No nodata in top-left
        assert not np.isnan(resampled[1, 0])  # No nodata in bottom-left


class TestBoundsCropper:
    """Test bounds cropping functions."""

    # Removed test_exact_crop_metadata - only verifies dictionary field existence and basic arithmetic

    # Removed test_crop_with_buffer_metadata - only verifies dictionary field existence and basic arithmetic

    def test_multiband_crop(self):
        """Test cropping multi-band data."""
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

        cropped, new_transform, actual_bounds = crop_to_bounds(
            data,
            transform=transform,
            bounds=(1, 1, 3, 3),
        )

        # Should preserve bands and crop spatial dimensions
        assert cropped.shape == (3, 2, 2)
        assert actual_bounds == (1, 1, 3, 3)

    def test_crop_size(self):
        """Test cropped output size."""
        # Create 5x5 data
        data = np.ones((5, 5), dtype=np.float32)
        transform = Affine(1.0, 0, 0, 0, -1.0, 5)

        cropped, new_transform, actual_bounds = crop_to_bounds(
            data,
            transform=transform,
            bounds=(0, 0, 3, 2),
        )

        # Check output size
        assert cropped.shape == (2, 3)  # 2 rows, 3 columns
        assert cropped.size == 6  # 3x2 = 6 pixels


class TestSpatialAligner:
    """Test raster alignment functions."""

    # Removed test_align_same_grid_metadata - only verifies dictionary field existence and object type checking

    # Removed test_align_different_sizes_metadata - only verifies dictionary field existence and shape preservation

    def test_multiband_alignment(self):
        """Test aligning multi-band rasters."""
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

        aligned_results = align_rasters(
            [raster1, raster2], [transform1, transform2], reference_idx=1
        )

        # Both should match reference (raster2) shape
        assert aligned_results[0][0].shape == (2, 3, 3)
        assert aligned_results[1][0].shape == (2, 3, 3)

        # Reference should be unchanged
        assert np.array_equal(aligned_results[1][0], raster2)

    def test_no_crs_provided(self):
        """Test alignment when no CRS is provided."""
        raster1 = np.ones((2, 2), dtype=np.float32)
        raster2 = np.ones((2, 2), dtype=np.float32) * 2
        transform = Affine(1.0, 0, 0, 0, -1.0, 2)

        # Should work without CRS
        aligned_results = align_rasters([raster1, raster2], [transform, transform])

        assert len(aligned_results) == 2
        # Check that data is returned
        assert aligned_results[0][0].shape == (2, 2)
        assert aligned_results[1][0].shape == (2, 2)
