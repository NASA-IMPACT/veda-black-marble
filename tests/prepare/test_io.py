"""Test critical I/O functions for bm.py parity.

This module tests only the essential I/O functionality needed for bm.py parity.
"""

import pytest
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import tempfile
import os

from blackmarble.prepare.io import (
    load_with_gdal,
    load_with_rasterio,
    create_virtual_raster,
    read_metadata,
    setup_output_profile,
)


def test_load_with_rasterio():
    """Test loading raster data with rasterio - CRITICAL for pipeline."""
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        # Create test data
        data = np.random.rand(10, 10).astype(np.float32)
        transform = from_bounds(0, 0, 10, 10, 10, 10)
        
        # Write test file
        with rasterio.open(
            tmp_path,
            'w',
            driver='GTiff',
            height=10,
            width=10,
            count=1,
            dtype='float32',
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            dst.write(data, 1)
        
        # Test loading
        data_array, metadata = load_with_rasterio(tmp_path)
        
        # Verify result
        assert data_array is not None
        assert metadata is not None
        assert 'crs' in metadata
        assert 'transform' in metadata
        
        # Data shape depends on how rasterio reads it
        # For single band it might be 2D
        if data_array.ndim == 2:
            assert data_array.shape == (10, 10)
            assert np.array_equal(data_array, data)
        else:
            assert data_array.shape == (1, 10, 10)
            assert np.array_equal(data_array[0], data)
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_read_metadata():
    """Test reading raster metadata - needed for pipeline processing."""
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        # Create test data
        transform = from_bounds(0, 0, 10, 10, 10, 10)
        
        # Write test file
        with rasterio.open(
            tmp_path,
            'w',
            driver='GTiff',
            height=10,
            width=10,
            count=2,
            dtype='float32',
            crs='EPSG:4326',
            transform=transform,
            nodata=-9999
        ) as dst:
            dst.write(np.ones((10, 10), dtype=np.float32), 1)
            dst.write(np.ones((10, 10), dtype=np.float32) * 2, 2)
        
        # Test metadata reading
        metadata = read_metadata(tmp_path)
        
        # Verify essential metadata
        # Check basic structure
        assert 'spatial' in metadata
        assert 'bands' in metadata
        assert 'format' in metadata
        
        # Check spatial info
        assert metadata['spatial']['shape'] == (10, 10)
        assert metadata['spatial']['crs'] == 'EPSG:4326'
        assert 'bounds' in metadata['spatial']
        assert 'transform' in metadata['spatial']
        
        # Check band info
        assert metadata['bands']['count'] == 2
        assert len(metadata['bands']['info']) == 2
        assert metadata['bands']['info'][0]['nodata'] == -9999
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_setup_output_profile():
    """Test output profile setup - CRITICAL for correct output format."""
    # Base profile
    profile = {
        'driver': 'GTiff',
        'width': 100,
        'height': 100,
        'count': 1,
        'dtype': 'float32',
        'crs': 'EPSG:4326',
        'transform': from_bounds(0, 0, 10, 10, 100, 100)
    }
    
    # Test COG profile
    cog_profile = setup_output_profile(
        template_meta=profile,
        modifications={
            'compress': 'DEFLATE',
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512
        }
    )
    
    assert cog_profile['driver'] == 'GTiff'
    assert cog_profile['compress'] == 'DEFLATE'
    assert cog_profile['tiled'] is True
    assert cog_profile['blockxsize'] == 512
    assert cog_profile['blockysize'] == 512
    
    # Test compression options
    profile_lzw = setup_output_profile(
        template_meta=profile,
        modifications={'compress': 'LZW'}
    )
    
    assert profile_lzw['compress'] == 'LZW'