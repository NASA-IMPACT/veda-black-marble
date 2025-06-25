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
    raster = {
        'data': np.ones((10, 10), dtype=np.float32),
        'transform': from_bounds(0, 0, 10, 10, 10, 10),
        'crs': CRS.from_epsg(4326)
    }
    
    dst_crs = CRS.from_epsg(3857)  # Web Mercator
    
    # Reproject
    reprojected = reproject_image(
        raster,
        dst_crs,
        resampling='nearest'
    )
    
    # Basic validation
    assert reprojected['data'].shape[0] > 0
    assert reprojected['data'].shape[1] > 0
    assert not np.all(np.isnan(reprojected['data']))
    assert reprojected['transform'] is not None
    assert reprojected['crs'] == dst_crs


def test_crop_to_bounds():
    """Test cropping to bounds - CRITICAL for area of interest extraction."""
    # Create test raster
    raster = {
        'data': np.arange(100).reshape(10, 10).astype(np.float32),
        'transform': from_bounds(0, 0, 10, 10, 10, 10),
        'crs': CRS.from_epsg(4326)
    }
    
    # Define crop bounds (center region)
    bounds = (3, 3, 7, 7)
    
    # Crop
    cropped = crop_to_bounds(raster, bounds)
    
    # Verify shape is smaller
    assert cropped['data'].shape[0] < raster['data'].shape[0]
    assert cropped['data'].shape[1] < raster['data'].shape[1]
    
    # Verify transform is updated
    assert cropped['transform'] != raster['transform']


def test_align_rasters():
    """Test raster alignment - needed for multi-source data integration."""
    # Create two misaligned rasters
    raster1 = {
        'data': np.ones((10, 10)),
        'transform': from_bounds(0, 0, 10, 10, 10, 10),
        'crs': CRS.from_epsg(4326)
    }
    
    raster2 = {
        'data': np.ones((8, 8)) * 2,
        'transform': from_bounds(1, 1, 9, 9, 8, 8),
        'crs': CRS.from_epsg(4326)
    }
    
    # Align rasters
    aligned = align_rasters(
        [raster1, raster2],
        target_crs=CRS.from_epsg(4326)
    )
    
    # Verify alignment
    assert len(aligned) == 2
    assert aligned[0]['data'].shape == aligned[1]['data'].shape
    assert aligned[0]['transform'] == aligned[1]['transform']
    assert aligned[0]['crs'] == aligned[1]['crs']