"""Test critical spatial processing functions for bm.py parity.

This module tests only the essential spatial functionality needed for bm.py parity.
"""

import pytest
import numpy as np
from rasterio.transform import from_bounds
from rasterio.crs import CRS

from blackmarble.prepare.spatial import (
    reproject_image,
    crop_to_bounds,
    align_rasters,
)


def test_reproject_image():
    """Test image reprojection - CRITICAL for coordinate system alignment."""
    # Create test raster data
    data = np.ones((10, 10), dtype=np.float32)
    transform = from_bounds(0, 0, 10, 10, 10, 10)
    src_crs = CRS.from_epsg(4326)
    dst_crs = CRS.from_epsg(3857)  # Web Mercator

    # Reproject
    reprojected_data, new_transform, new_bounds = reproject_image(
        data, transform, src_crs, dst_crs, resampling="nearest"
    )

    # Basic validation
    assert reprojected_data.shape[0] > 0
    assert reprojected_data.shape[1] > 0
    assert not np.all(np.isnan(reprojected_data))
    assert new_transform is not None
    assert new_bounds is not None


def test_crop_to_bounds():
    """Test cropping to bounds - CRITICAL for area of interest extraction."""
    # Create test raster
    data = np.arange(100).reshape(10, 10).astype(np.float32)
    transform = from_bounds(0, 0, 10, 10, 10, 10)

    # Define crop bounds (center region)
    bounds = (3, 3, 7, 7)

    # Crop
    cropped_data, new_transform, actual_bounds = crop_to_bounds(data, transform, bounds)

    # Verify shape is smaller
    assert cropped_data.shape[0] < data.shape[0]
    assert cropped_data.shape[1] < data.shape[1]

    # Verify transform is updated
    assert new_transform != transform
    assert actual_bounds is not None


def test_align_rasters():
    """Test raster alignment - needed for multi-source data integration."""
    # Create two misaligned rasters
    data1 = np.ones((10, 10))
    transform1 = from_bounds(0, 0, 10, 10, 10, 10)

    data2 = np.ones((8, 8)) * 2
    transform2 = from_bounds(1, 1, 9, 9, 8, 8)

    # Align rasters
    aligned = align_rasters(
        rasters=[data1, data2],
        transforms=[transform1, transform2],
        crs_list=["EPSG:4326", "EPSG:4326"],
        reference_idx=0,
    )

    # Verify alignment
    assert len(aligned) == 2
    aligned_data1, aligned_transform1 = aligned[0]
    aligned_data2, aligned_transform2 = aligned[1]

    assert aligned_data1.shape == aligned_data2.shape
    assert np.array_equal(aligned_transform1, aligned_transform2)
