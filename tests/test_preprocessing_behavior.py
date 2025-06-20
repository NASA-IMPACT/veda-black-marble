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
        assert result.metadata["source_crs"] == "EPSG:4326"
        assert result.metadata["target_crs"] == "EPSG:4326"
        assert result.metadata["original_shape"] == data.shape
        
    def test_transform_metadata(self):
        """Test that projection metadata is correct."""
        handler = ProjectionHandler(target_crs="EPSG:3857", resampling="bilinear")
        
        # Create test data in lat/lon
        data = np.ones((10, 10), dtype=np.float32)
        transform = from_bounds(-1, -1, 1, 1, 10, 10)
        
        result = handler.process(
            data,
            source_crs="EPSG:4326",
            source_transform=transform,
        )
        
        # Check metadata
        assert result.metadata["source_crs"] == "EPSG:4326"
        assert result.metadata["target_crs"] == "EPSG:3857"
        assert result.metadata["resampling"] == "bilinear"
        assert "transformed_shape" in result.metadata
        assert "bounds" in result.metadata
        
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
        handler = ProjectionHandler(target_crs="EPSG:3857")  # Different CRS to force reprojection
        
        # Create 3-band test data with nodata in different bands
        data = np.array([
            [[1, 2], [3, 4]],
            [[5, -999], [7, 8]],
            [[9, 10], [-999, 12]]
        ], dtype=np.float32)
        
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
        resampler = ResolutionResampler(
            target_resolution=(2.0, 2.0),
            method="bilinear"
        )
        
        # Create 10x10 data with 1x1 resolution
        data = np.arange(100).reshape(10, 10).astype(np.float32)
        transform = Affine(1.0, 0, 0, 0, -1.0, 10)
        
        result = resampler.process(
            data,
            transform=transform,
            crs="EPSG:4326"
        )
        
        # Check metadata records the scaling
        assert result.metadata["original_resolution"] == (1.0, 1.0)
        assert result.metadata["target_resolution"] == (2.0, 2.0)
        assert result.metadata["scale_factors"] == (0.5, 0.5)
        assert result.metadata["original_shape"] == (10, 10)
        # Output should be 5x5
        assert result.metadata["resampled_shape"] == (5, 5)
        
    def test_upsampling_scale(self):
        """Test resampling to higher resolution."""
        resampler = ResolutionResampler(
            target_resolution=(0.5, 0.5),
            method="bilinear"
        )
        
        # Create 5x5 data with 2x2 resolution
        data = np.arange(25).reshape(5, 5).astype(np.float32)
        transform = Affine(2.0, 0, 0, 0, -2.0, 10)
        
        result = resampler.process(
            data,
            transform=transform,
            crs="EPSG:4326"
        )
        
        # Check scaling factors
        assert result.metadata["scale_factors"] == (4.0, 4.0)
        assert result.metadata["resampled_shape"] == (20, 20)
        
    def test_multiband_resampling_shape(self):
        """Test resampling of multi-band data preserves bands."""
        resampler = ResolutionResampler(target_resolution=(2.0, 2.0))
        
        # Create 2-band data
        data = np.array([
            np.arange(16).reshape(4, 4),
            np.arange(16, 32).reshape(4, 4)
        ], dtype=np.float32)
        transform = Affine(1.0, 0, 0, 0, -1.0, 4)
        
        result = resampler.process(data, transform=transform)
        
        # Check shapes in metadata
        assert result.metadata["original_shape"] == (2, 4, 4)
        assert result.metadata["resampled_shape"] == (2, 2, 2)
        
    def test_nodata_mask_generation(self):
        """Test mask generation with nodata values."""
        resampler = ResolutionResampler(target_resolution=(2.0, 2.0))
        
        # Create data with some nodata
        data = np.array([
            [1, 2, 3, 4],
            [5, -999, -999, 8],
            [9, -999, -999, 12],
            [13, 14, 15, 16]
        ], dtype=np.float32)
        transform = Affine(1.0, 0, 0, 0, -1.0, 4)
        
        result = resampler.process(data, transform=transform, nodata=-999)
        
        # After downsampling, pixels with any nodata should be masked
        assert result.mask.shape == (2, 2)
        assert result.mask[0, 0] == 1.0  # No nodata in this quadrant
        assert result.mask[0, 1] == 0.0  # Has nodata
        assert result.mask[1, 0] == 0.0  # Has nodata
        assert result.mask[1, 1] == 1.0  # No nodata


class TestBoundsCropper:
    """Test BoundsCropper extracts correct regions."""
    
    def test_exact_crop_metadata(self):
        """Test cropping metadata is correct."""
        cropper = BoundsCropper(bounds=(2, 2, 4, 4))
        
        # Create 6x6 data
        data = np.arange(36).reshape(6, 6).astype(np.float32)
        transform = Affine(1.0, 0, 0, 0, -1.0, 6)
        
        result = cropper.process(data, transform=transform)
        
        # Check metadata
        assert result.metadata["original_bounds"] == (0, 0, 6, 6)
        assert result.metadata["target_bounds"] == (2, 2, 4, 4)
        assert result.metadata["applied_bounds"] == (2, 2, 4, 4)  # No buffer
        assert result.metadata["buffer"] == 0.0
        assert result.metadata["original_shape"] == (6, 6)
        assert result.metadata["cropped_shape"] == (2, 2)
        
        # Check pixel window
        window = result.metadata["pixel_window"]
        assert window["col_min"] == 2
        assert window["col_max"] == 4
        assert window["row_min"] == 2
        assert window["row_max"] == 4
        
        # All pixels in cropped area should be valid
        assert np.all(result.mask == 1.0)
        
    def test_crop_with_buffer_metadata(self):
        """Test cropping with buffer zone."""
        cropper = BoundsCropper(
            bounds=(2, 2, 4, 4),
            buffer=1.0,
            buffer_units="crs"
        )
        
        # Create 8x8 data
        data = np.arange(64).reshape(8, 8).astype(np.float32)
        transform = Affine(1.0, 0, 0, 0, -1.0, 8)
        
        result = cropper.process(data, transform=transform)
        
        # With buffer of 1, applied bounds should be expanded
        assert result.metadata["target_bounds"] == (2, 2, 4, 4)
        assert result.metadata["applied_bounds"] == (1, 1, 5, 5)
        assert result.metadata["buffer"] == 1.0
        assert result.metadata["buffer_units"] == "crs"
        assert result.metadata["cropped_shape"] == (4, 4)
        
    def test_multiband_crop_mask(self):
        """Test cropping multi-band data produces correct mask."""
        cropper = BoundsCropper(bounds=(1, 1, 3, 3))
        
        # Create 3-band 4x4 data
        data = np.array([
            np.arange(16).reshape(4, 4),
            np.arange(16, 32).reshape(4, 4),
            np.arange(32, 48).reshape(4, 4)
        ], dtype=np.float32)
        transform = Affine(1.0, 0, 0, 0, -1.0, 4)
        
        result = cropper.process(data, transform=transform)
        
        # Mask should be 2D even for multiband
        assert result.mask.shape == (2, 2)
        assert np.all(result.mask == 1.0)
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
    
    def test_align_same_grid_metadata(self):
        """Test aligning arrays already on same grid."""
        aligner = SpatialAligner(reference_idx=0)
        
        # Create two identical grids
        raster1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
        raster2 = np.array([[5, 6], [7, 8]], dtype=np.float32)
        transform = Affine(1.0, 0, 0, 0, -1.0, 2)
        
        results = aligner.align_rasters(
            [raster1, raster2],
            [transform, transform],
            ["EPSG:4326", "EPSG:4326"]
        )
        
        assert len(results) == 2
        assert isinstance(results[0], SpatialResult)
        assert isinstance(results[1], SpatialResult)
        
        # Check reference raster metadata
        assert results[0].metadata["is_reference"] is True
        assert results[0].transform == transform
        assert results[0].crs == "EPSG:4326"
        assert results[0].bounds == (0, 0, 2, 2)
        
        # Check aligned raster metadata
        assert results[1].metadata["is_reference"] is False
        assert results[1].metadata["aligned_to"] == 0
        assert results[1].metadata["method"] == "nearest"
        
    def test_align_different_sizes_metadata(self):
        """Test aligning arrays of different sizes."""
        aligner = SpatialAligner(reference_idx=0, method="bilinear")
        
        # Reference: 2x2, Other: 4x4 covering same area
        ref_raster = np.array([[1, 2], [3, 4]], dtype=np.float32)
        ref_transform = Affine(2.0, 0, 0, 0, -2.0, 4)
        
        other_raster = np.arange(16).reshape(4, 4).astype(np.float32)
        other_transform = Affine(1.0, 0, 0, 0, -1.0, 4)
        
        results = aligner.align_rasters(
            [ref_raster, other_raster],
            [ref_transform, other_transform]
        )
        
        # Both should have reference shape
        assert results[0].data.shape == (2, 2)
        assert results[1].data.shape == (2, 2)
        
        # Check alignment metadata
        assert results[1].metadata["original_shape"] == (4, 4)
        assert results[1].metadata["method"] == "bilinear"
        assert results[1].transform == ref_transform
        
    def test_multiband_alignment(self):
        """Test aligning multi-band rasters."""
        aligner = SpatialAligner(reference_idx=1)
        
        # Create 2-band rasters
        raster1 = np.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ], dtype=np.float32)
        
        raster2 = np.array([
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            [[11, 21, 31], [41, 51, 61], [71, 81, 91]]
        ], dtype=np.float32)
        
        transform1 = Affine(2.0, 0, 0, 0, -2.0, 4)
        transform2 = Affine(1.0, 0, 0, 0, -1.0, 3)
        
        results = aligner.align_rasters(
            [raster1, raster2],
            [transform1, transform2]
        )
        
        # Both should match reference (raster2) shape
        assert results[0].data.shape == (2, 3, 3)
        assert results[1].data.shape == (2, 3, 3)
        
        # Reference should be unchanged
        assert np.array_equal(results[1].data, raster2)
        assert results[1].metadata["is_reference"] is True
        
    def test_no_crs_provided(self):
        """Test alignment when no CRS is provided."""
        aligner = SpatialAligner()
        
        raster1 = np.ones((2, 2), dtype=np.float32)
        raster2 = np.ones((2, 2), dtype=np.float32) * 2
        transform = Affine(1.0, 0, 0, 0, -1.0, 2)
        
        # Should work without CRS
        results = aligner.align_rasters(
            [raster1, raster2],
            [transform, transform]
        )
        
        assert len(results) == 2
        assert results[0].crs == "unknown"
        assert results[1].crs == "unknown"