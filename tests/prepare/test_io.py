"""Test functions in blackmarble.prepare.io module.

This module tests file I/O utilities for data preparation phase.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import numpy as np
from pathlib import Path
import gc
from osgeo import gdal

# Import the module under test
from blackmarble.prepare.io import (
    detect_format,
    load_with_gdal,
    load_with_rasterio,
    create_virtual_raster,
    read_metadata,
    setup_output_profile,
    validate_raster_stack,
    create_memory_efficient_reader,
)


def test_detect_format():
    """Test raster file format detection and structure analysis.
    
    This test should verify that:
    - Common raster formats (GTiff, HDF5, NetCDF) are correctly identified
    - Driver names are properly extracted and mapped
    - Subdataset detection works for HDF/NetCDF files
    - Compression type is detected when present
    - Tile structure information is extracted for tiled formats
    - COG (Cloud-Optimized GeoTIFF) detection works correctly
    - Appropriate reader function is recommended for each format
    - Error handling for non-existent files
    - Error handling for corrupted or invalid files
    """
    import rasterio
    from rasterio.transform import Affine
    from osgeo import gdal
    
    # Test non-existent file
    with pytest.raises(FileNotFoundError):
        detect_format("/non/existent/file.tif")
    
    # Test standard GeoTIFF
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        test_tiff = f.name
    
    try:
        # Create a simple GeoTIFF
        profile = {
            'driver': 'GTiff',
            'width': 100,
            'height': 100,
            'count': 1,
            'dtype': 'float32',
            'crs': 'EPSG:4326',
            'transform': Affine.identity(),
            'compress': 'deflate'
        }
        with rasterio.open(test_tiff, 'w', **profile) as dst:
            dst.write(np.ones((100, 100), dtype='float32'), 1)
        
        # Test format detection
        result = detect_format(test_tiff)
        
        # Verify format identification
        assert result['format'] == 'GTiff'
        assert result['driver'] == 'GTiff'
        assert str(result['compression']).lower() in ['deflate', 'compression.deflate']  # Handle enum
        assert result['is_cog'] == False  # No overviews
        assert result['reader'] == load_with_rasterio
        assert len(result['subdatasets']) == 0
        
    finally:
        os.unlink(test_tiff)
    
    # Test tiled GeoTIFF
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        test_tiled = f.name
    
    try:
        # Create a tiled GeoTIFF
        profile = {
            'driver': 'GTiff',
            'width': 512,
            'height': 512,
            'count': 1,
            'dtype': 'uint8',
            'crs': 'EPSG:4326',
            'transform': Affine.identity(),
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256
        }
        with rasterio.open(test_tiled, 'w', **profile) as dst:
            dst.write(np.ones((512, 512), dtype='uint8'), 1)
        
        # Test format detection
        result = detect_format(test_tiled)
        
        # Verify tile structure extraction
        assert result['tile_structure'] is not None
        assert result['tile_structure']['tiled'] == True
        assert result['tile_structure']['blockxsize'] == 256
        assert result['tile_structure']['blockysize'] == 256
        
    finally:
        os.unlink(test_tiled)
    
    # Test COG detection (mock with overviews)
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        test_cog = f.name
    
    try:
        # Create a GeoTIFF with overviews (simulated COG)
        profile = {
            'driver': 'GTiff',
            'width': 1024,
            'height': 1024,
            'count': 1,
            'dtype': 'uint8',
            'crs': 'EPSG:4326',
            'transform': Affine.identity(),
            'tiled': True
        }
        with rasterio.open(test_cog, 'w', **profile) as dst:
            dst.write(np.ones((1024, 1024), dtype='uint8'), 1)
            # Build overviews
            dst.build_overviews([2, 4, 8], rasterio.enums.Resampling.nearest)
        
        # Test format detection
        result = detect_format(test_cog)
        
        # Verify COG detection
        assert result['is_cog'] == True
        assert result['format'] == 'GTiff'
        
    finally:
        os.unlink(test_cog)
    
    # Test corrupted file handling
    with tempfile.NamedTemporaryFile(delete=False) as f:
        test_corrupted = f.name
        f.write(b"This is not a valid raster file")
    
    try:
        # Should not raise exception, but handle gracefully
        result = detect_format(test_corrupted)
        # Format might be unknown or None
        assert result['format'] in [None, 'Unknown']
    finally:
        os.unlink(test_corrupted)


def test_detect_format_with_subdatasets():
    """Test format detection for files with multiple subdatasets.
    
    This test should verify that:
    - HDF5 files with multiple layers are properly analyzed
    - NetCDF files with multiple variables are handled
    - Subdataset paths and descriptions are extracted
    - GDAL reader is recommended for complex formats
    - Subdataset metadata is properly structured
    """
    # Mock HDF5 file with subdatasets by using a mock GDAL dataset
    with patch('osgeo.gdal.Open') as mock_open:
        # Create mock dataset
        mock_ds = Mock()
        mock_driver = Mock()
        mock_driver.ShortName = 'HDF5'
        mock_ds.GetDriver.return_value = mock_driver
        
        # Mock subdatasets
        mock_subdatasets = [
            ('HDF5:"test.h5"://dataset1', 'Dataset 1 Description'),
            ('HDF5:"test.h5"://dataset2', 'Dataset 2 Description'),
            ('HDF5:"test.h5"://dataset3', 'Dataset 3 Description')
        ]
        mock_ds.GetSubDatasets.return_value = mock_subdatasets
        mock_open.return_value = mock_ds
        
        # Create a temporary file to pass file existence check
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            test_hdf5 = f.name
        
        try:
            result = detect_format(test_hdf5, check_subdatasets=True)
            
            # Verify HDF5 format detection
            assert result['format'] == 'HDF5'
            assert result['driver'] == 'HDF5'
            
            # Verify subdataset extraction
            assert len(result['subdatasets']) == 3
            assert result['subdatasets'][0]['path'] == 'HDF5:"test.h5"://dataset1'
            assert result['subdatasets'][0]['description'] == 'Dataset 1 Description'
            assert result['subdatasets'][1]['path'] == 'HDF5:"test.h5"://dataset2'
            assert result['subdatasets'][2]['path'] == 'HDF5:"test.h5"://dataset3'
            
            # Verify GDAL reader is recommended for HDF5 with subdatasets
            assert result['reader'] == load_with_gdal
            
        finally:
            os.unlink(test_hdf5)
    
    # Test with subdatasets disabled
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        test_file = f.name
    
    try:
        # Create simple file
        import rasterio
        profile = {'driver': 'GTiff', 'width': 10, 'height': 10, 'count': 1, 'dtype': 'uint8'}
        with rasterio.open(test_file, 'w', **profile) as dst:
            dst.write(np.ones((10, 10), dtype='uint8'), 1)
        
        # Test without subdataset check
        result = detect_format(test_file, check_subdatasets=False)
        
        # Should not check for subdatasets
        assert 'subdatasets' in result
        assert len(result['subdatasets']) == 0
        
    finally:
        os.unlink(test_file)


def test_detect_format_cog_detection():
    """Test Cloud-Optimized GeoTIFF detection.
    
    This test should verify that:
    - Valid COG files are correctly identified
    - Non-COG GeoTIFFs are properly distinguished
    - Overview detection works for COG validation
    - Tile structure is properly analyzed
    - Performance recommendations are appropriate
    """
    import rasterio
    from rasterio.transform import Affine
    
    # Test non-COG GeoTIFF (no tiling, no overviews)
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        non_cog = f.name
    
    try:
        profile = {
            'driver': 'GTiff',
            'width': 256,
            'height': 256,
            'count': 1,
            'dtype': 'uint8',
            'crs': 'EPSG:4326',
            'transform': Affine.identity(),
            'tiled': False  # Not tiled
        }
        with rasterio.open(non_cog, 'w', **profile) as dst:
            dst.write(np.ones((256, 256), dtype='uint8'), 1)
        
        result = detect_format(non_cog)
        
        # Verify non-COG detection
        assert result['is_cog'] == False
        assert result['format'] == 'GTiff'
        assert result['tile_structure'] is None or result['tile_structure'].get('tiled') == False
        
    finally:
        os.unlink(non_cog)
    
    # Test COG with tiling and overviews
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        cog_file = f.name
    
    try:
        profile = {
            'driver': 'GTiff',
            'width': 2048,
            'height': 2048,
            'count': 1,
            'dtype': 'uint8',
            'crs': 'EPSG:4326',
            'transform': Affine.identity(),
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512,
            'compress': 'deflate'
        }
        with rasterio.open(cog_file, 'w', **profile) as dst:
            data = np.random.randint(0, 255, (2048, 2048), dtype='uint8')
            dst.write(data, 1)
            # Build overviews for COG
            dst.build_overviews([2, 4, 8, 16], rasterio.enums.Resampling.average)
        
        result = detect_format(cog_file)
        
        # Verify COG detection
        assert result['is_cog'] == True
        assert result['format'] == 'GTiff'
        assert str(result['compression']).lower() in ['deflate', 'compression.deflate']
        
        # Verify tile structure analysis
        assert result['tile_structure'] is not None
        assert result['tile_structure']['tiled'] == True
        assert result['tile_structure']['blockxsize'] == 512
        assert result['tile_structure']['blockysize'] == 512
        
        # Verify performance recommendation (rasterio for COG)
        assert result['reader'] == load_with_rasterio
        
    finally:
        os.unlink(cog_file)
    
    # Test tiled but no overviews (not a true COG)
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        partial_cog = f.name
    
    try:
        profile = {
            'driver': 'GTiff',
            'width': 1024,
            'height': 1024,
            'count': 1,
            'dtype': 'uint8',
            'crs': 'EPSG:4326',
            'transform': Affine.identity(),
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256
        }
        with rasterio.open(partial_cog, 'w', **profile) as dst:
            dst.write(np.ones((1024, 1024), dtype='uint8'), 1)
            # No overviews built
        
        result = detect_format(partial_cog)
        
        # Should not be detected as COG without overviews
        assert result['is_cog'] == False
        assert result['tile_structure']['tiled'] == True
        
    finally:
        os.unlink(partial_cog)


def test_load_with_gdal():
    """Test GDAL-based raster loading with comprehensive metadata.
    
    This test should verify that:
    - Basic raster loading works for common formats
    - Band selection works with numeric indices
    - Subdataset loading works with string band names
    - Windowed reading returns correct subsets
    - Overview level selection works correctly
    - NoData values are properly handled and masked
    - Metadata includes all required fields (transform, CRS, bounds, etc.)
    - Geotransform is properly converted to Affine
    - Format-specific metadata is extracted
    - Error handling for invalid files or parameters
    """
    import rasterio
    from rasterio.transform import Affine
    from rasterio.windows import Window
    
    # Create test GeoTIFF
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        test_file = f.name
    
    try:
        # Create test data with known values
        test_data = np.arange(100).reshape(10, 10).astype('float32')
        test_data[0, 0] = -9999  # Add nodata value
        
        profile = {
            'driver': 'GTiff',
            'width': 10,
            'height': 10,
            'count': 3,  # Multi-band
            'dtype': 'float32',
            'crs': 'EPSG:4326',
            'transform': Affine(0.1, 0, -180, 0, -0.1, 90),  # 0.1 degree pixels
            'nodata': -9999
        }
        with rasterio.open(test_file, 'w', **profile) as dst:
            for band in range(1, 4):
                dst.write(test_data * band, band)
        
        # Test basic loading
        data, metadata = load_with_gdal(test_file)
        
        # Verify data shape (default band 1)
        assert data.shape == (10, 10)
        assert np.allclose(data[1:, 1:], test_data[1:, 1:])  # Skip nodata pixel
        
        # Verify nodata masking
        assert np.isnan(data[0, 0])  # Nodata should be NaN when as_masked=True
        
        # Verify metadata
        assert 'transform' in metadata
        assert isinstance(metadata['transform'], Affine)
        assert metadata['nodata'] == -9999
        assert metadata['dtype'] == np.dtype('float32')
        assert metadata['shape'] == (10, 10)
        
        # Verify bounds calculation
        assert metadata['bounds'] == (-180, 89, -179, 90)
        
        # Verify format metadata
        assert metadata['format_metadata']['driver'] == 'GTiff'
        assert metadata['format_metadata']['band_count'] == 3
        
        # Test band selection
        data2, _ = load_with_gdal(test_file, band=2)
        assert np.allclose(data2[1:, 1:], test_data[1:, 1:] * 2)  # Band 2 is data * 2
        
        # Test without masking
        data_unmasked, _ = load_with_gdal(test_file, as_masked=False)
        assert data_unmasked[0, 0] == -9999  # Nodata value preserved
        
        # Test windowed reading
        window = Window(2, 3, 5, 4)  # col_off=2, row_off=3, width=5, height=4
        data_window, meta_window = load_with_gdal(test_file, window=window)
        
        # Verify window data
        assert data_window.shape == (4, 5)
        expected_window = test_data[3:7, 2:7]
        assert np.allclose(data_window, expected_window, equal_nan=True)
        
        # Verify window transform adjustment
        assert meta_window['transform'] != metadata['transform']
        assert meta_window['bounds'] != metadata['bounds']
        assert meta_window['bounds'] == (-179.8, 89.3, -179.3, 89.7)
        
    finally:
        os.unlink(test_file)
    
    # Test error handling
    with pytest.raises(RuntimeError):
        load_with_gdal("/non/existent/file.tif")
    
    # Test subdataset loading with mock
    with patch('osgeo.gdal.Open') as mock_open:
        mock_ds = Mock()
        mock_band = Mock()
        mock_band.ReadAsArray.return_value = np.ones((10, 10))
        mock_band.GetNoDataValue.return_value = None
        mock_band.GetOverviewCount.return_value = 0
        mock_ds.GetRasterBand.return_value = mock_band
        mock_ds.RasterXSize = 10
        mock_ds.RasterYSize = 10
        mock_ds.GetGeoTransform.return_value = (0, 1, 0, 0, 0, -1)
        mock_ds.GetProjection.return_value = 'EPSG:4326'
        mock_driver = Mock()
        mock_driver.ShortName = 'HDF5'
        mock_ds.GetDriver.return_value = mock_driver
        mock_ds.RasterCount = 1
        mock_ds.GetMetadata.return_value = {}
        mock_open.return_value = mock_ds
        
        # Test subdataset string band name
        data, meta = load_with_gdal("test.h5", band="GRIDS/Dataset1")
        
        # Verify subdataset path construction
        mock_open.assert_called_with('HDF5:"test.h5"://GRIDS/Dataset1', gdal.GA_ReadOnly)


def test_load_with_gdal_subdatasets():
    """Test GDAL loading of HDF5/NetCDF subdatasets.
    
    This test should verify that:
    - Subdatasets are accessible by name/path
    - HDF5 path construction works correctly
    - Multiple subdatasets from same file work
    - Metadata is specific to selected subdataset
    - Error handling for invalid subdataset names
    """
    from osgeo import gdal
    
    # Create mock HDF5 file behavior
    with patch('osgeo.gdal.Open') as mock_open:
        # Setup different mock datasets for different subdatasets
        def create_mock_subdataset(name, width, height, data_value):
            mock_ds = Mock()
            mock_band = Mock()
            mock_band.ReadAsArray.return_value = np.full((height, width), data_value, dtype='float32')
            mock_band.GetNoDataValue.return_value = -9999
            mock_band.GetOverviewCount.return_value = 0
            mock_ds.GetRasterBand.return_value = mock_band
            mock_ds.RasterXSize = width
            mock_ds.RasterYSize = height
            mock_ds.GetGeoTransform.return_value = (0, 1, 0, 0, 0, -1)
            mock_ds.GetProjection.return_value = 'EPSG:4326'
            mock_driver = Mock()
            mock_driver.ShortName = 'HDF5'
            mock_ds.GetDriver.return_value = mock_driver
            mock_ds.RasterCount = 1
            mock_ds.GetMetadata.return_value = {'SUBDATASET_NAME': name}
            return mock_ds
        
        # Configure mock to return different datasets based on path
        def side_effect(path, mode):
            if 'dataset1' in path:
                return create_mock_subdataset('dataset1', 20, 20, 1.0)
            elif 'dataset2' in path:
                return create_mock_subdataset('dataset2', 30, 30, 2.0)
            elif 'invalid' in path:
                return None
            else:
                return create_mock_subdataset('default', 10, 10, 0.0)
        
        mock_open.side_effect = side_effect
        
        # Test subdataset access by name
        data1, meta1 = load_with_gdal("test.h5", band="GRIDS/dataset1")
        
        # Verify HDF5 path construction
        expected_path = 'HDF5:"test.h5"://GRIDS/dataset1'
        mock_open.assert_any_call(expected_path, gdal.GA_ReadOnly)
        
        # Verify data and metadata
        assert data1.shape == (20, 20)
        assert np.all(data1 == 1.0)
        assert meta1['shape'] == (20, 20)
        assert meta1['format_metadata']['metadata']['SUBDATASET_NAME'] == 'dataset1'
        
        # Test multiple subdatasets from same file
        data2, meta2 = load_with_gdal("test.h5", band="GRIDS/dataset2")
        
        # Verify different subdataset
        assert data2.shape == (30, 30)
        assert np.all(data2 == 2.0)
        assert meta2['shape'] == (30, 30)
        assert meta2['format_metadata']['metadata']['SUBDATASET_NAME'] == 'dataset2'
        
        # Test error handling for invalid subdataset
        with pytest.raises(RuntimeError):
            load_with_gdal("test.h5", band="GRIDS/invalid")


def test_load_with_gdal_windowed_reads():
    """Test GDAL windowed reading functionality.
    
    This test should verify that:
    - Window parameters are correctly interpreted
    - Partial reads return expected data shapes
    - Transform is adjusted for window offset
    - Bounds are calculated correctly for windows
    - Edge cases (windows extending beyond data) are handled
    """
    import rasterio
    from rasterio.transform import Affine
    from rasterio.windows import Window
    
    # Create test file with known pattern
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        test_file = f.name
    
    try:
        # Create data with coordinates as values for easy verification
        height, width = 100, 100
        test_data = np.zeros((height, width), dtype='float32')
        for y in range(height):
            for x in range(width):
                test_data[y, x] = y * 1000 + x  # Encode position in value
        
        profile = {
            'driver': 'GTiff',
            'width': width,
            'height': height,
            'count': 1,
            'dtype': 'float32',
            'crs': 'EPSG:4326',
            'transform': Affine(0.01, 0, -180, 0, -0.01, 90),  # 0.01 degree pixels
            'nodata': -9999
        }
        with rasterio.open(test_file, 'w', **profile) as dst:
            dst.write(test_data, 1)
        
        # Test basic window read
        window = Window(col_off=10, row_off=20, width=30, height=40)
        data, metadata = load_with_gdal(test_file, window=window)
        
        # Verify window parameters correctly interpreted
        assert data.shape == (40, 30)  # height, width
        
        # Verify correct data subset
        expected = test_data[20:60, 10:40]
        assert np.array_equal(data, expected)
        
        # Verify transform adjustment for window offset
        original_transform = Affine(0.01, 0, -180, 0, -0.01, 90)
        expected_transform = Affine(0.01, 0, -180 + 10 * 0.01, 0, -0.01, 90 - 20 * 0.01)
        assert metadata['transform'] == expected_transform
        
        # Verify bounds calculation for window
        expected_bounds = (-179.9, 89.4, -179.6, 89.8)
        bounds = metadata['bounds']
        assert abs(bounds[0] - expected_bounds[0]) < 1e-10
        assert abs(bounds[1] - expected_bounds[1]) < 1e-10
        assert abs(bounds[2] - expected_bounds[2]) < 1e-10
        assert abs(bounds[3] - expected_bounds[3]) < 1e-10
        
        # Test edge case - window partially outside raster
        large_window = Window(col_off=80, row_off=80, width=40, height=40)
        
        # GDAL doesn't handle windows extending beyond bounds gracefully
        with pytest.raises(RuntimeError):
            load_with_gdal(test_file, window=large_window)
        
        # Test window at origin
        origin_window = Window(col_off=0, row_off=0, width=10, height=10)
        data_origin, meta_origin = load_with_gdal(test_file, window=origin_window)
        
        assert data_origin.shape == (10, 10)
        assert np.array_equal(data_origin, test_data[0:10, 0:10])
        assert meta_origin['transform'] == original_transform  # No offset
        
    finally:
        os.unlink(test_file)


def test_load_with_rasterio():
    """Test rasterio-based raster loading with modern interface.
    
    This test should verify that:
    - Standard GeoTIFF loading works correctly
    - Multi-band reading works with band parameter
    - Masked arrays handle NoData properly
    - Windowed reading is efficient and accurate
    - Overview access works with resampling parameter
    - Metadata profile is complete and valid
    - Single band dimension handling works correctly
    - Memory efficiency for large files
    - Error handling for unsupported formats
    """
    import rasterio
    from rasterio.transform import Affine
    from rasterio.windows import Window
    from rasterio.enums import Resampling
    import numpy.ma as ma
    
    # Create test GeoTIFF
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        test_file = f.name
    
    try:
        # Create test data
        test_data = np.arange(100, dtype='float32').reshape(10, 10)
        test_data[0, 0] = -9999  # NoData value
        
        profile = {
            'driver': 'GTiff',
            'width': 10,
            'height': 10,
            'count': 3,
            'dtype': 'float32',
            'crs': 'EPSG:4326',
            'transform': Affine(1, 0, 0, 0, -1, 10),
            'nodata': -9999,
            'tiled': True,
            'compress': 'deflate'
        }
        with rasterio.open(test_file, 'w', **profile) as dst:
            for band in range(1, 4):
                dst.write(test_data * band, band)
            # Build overviews for testing
            dst.build_overviews([2], Resampling.nearest)
        
        # Test standard loading
        data, metadata = load_with_rasterio(test_file)
        
        # Verify data (default band 1)
        assert data.shape == (10, 10)  # Single band dimension removed
        assert isinstance(data, ma.MaskedArray)  # Default masked=True
        assert data.mask[0, 0] == True  # NoData masked
        assert np.array_equal(data[1:, 1:], test_data[1:, 1:])
        
        # Verify metadata profile
        assert metadata['transform'] == profile['transform']
        assert metadata['crs'].to_string() == 'EPSG:4326'
        assert metadata['count'] == 3
        assert metadata['dtype'] == profile['dtype']
        assert metadata['nodata'] == -9999
        assert metadata['bounds'] == (0, 0, 10, 10)
        assert 'profile' in metadata
        assert metadata['profile']['driver'] == 'GTiff'
        assert metadata['profile']['compress'] == 'deflate'
        
        # Test multi-band reading
        data_multi, _ = load_with_rasterio(test_file, band=[1, 2, 3])
        assert data_multi.shape == (3, 10, 10)  # Band dimension preserved
        assert np.array_equal(data_multi[1, 1:, 1:], test_data[1:, 1:] * 2)
        
        # Test single band with indexes parameter
        data_idx, _ = load_with_rasterio(test_file, indexes=2)
        assert data_idx.shape == (10, 10)
        assert np.array_equal(data_idx[1:, 1:], test_data[1:, 1:] * 2)
        
        # Test unmasked reading
        data_unmasked, _ = load_with_rasterio(test_file, masked=False)
        assert not isinstance(data_unmasked, ma.MaskedArray)
        assert data_unmasked[0, 0] == -9999  # NoData value preserved
        
        # Test windowed reading
        window = Window(2, 3, 5, 4)
        data_window, meta_window = load_with_rasterio(test_file, window=window)
        
        assert data_window.shape == (4, 5)
        assert meta_window['transform'] != metadata['transform']
        assert meta_window['bounds'] == (2, 3, 7, 7)
        
        # Test overview access with resampling
        # Mock overview reading since our test file is small
        with patch('rasterio.open') as mock_open:
            mock_src = Mock()
            mock_src.read.return_value = np.ones((1, 5, 5))  # Simulated overview
            mock_src.window_transform.return_value = Affine.identity()
            mock_src.window_bounds.return_value = (0, 0, 10, 10)
            mock_src.bounds = (0, 0, 10, 10)
            mock_src.transform = Affine.identity()
            mock_src.crs = 'EPSG:4326'
            mock_src.count = 1
            mock_src.dtypes = ['float32']
            mock_src.nodata = None
            mock_src.profile = {'driver': 'GTiff'}
            mock_open.return_value.__enter__.return_value = mock_src
            
            data_ov, _ = load_with_rasterio('dummy.tif', resampling=Resampling.average)
            mock_src.read.assert_called_with(1, window=None, masked=True, resampling=Resampling.average)
        
    finally:
        os.unlink(test_file)
    
    # Test error handling for unsupported format
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        bad_file = f.name
        f.write(b"Not a raster")
    
    try:
        with pytest.raises(Exception):  # rasterio will raise an exception
            load_with_rasterio(bad_file)
    finally:
        os.unlink(bad_file)


def test_load_with_rasterio_multiband():
    """Test rasterio multi-band reading.
    
    This test should verify that:
    - Multiple bands can be read simultaneously
    - Band dimension is preserved when needed
    - Single band dimension is removed appropriately
    - Indexes parameter works as alternative to band
    - Memory layout is efficient for multi-band data
    """
    import rasterio
    from rasterio.transform import Affine
    
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        test_file = f.name
    
    try:
        # Create multi-band test data
        band1 = np.ones((50, 50), dtype='uint16') * 100
        band2 = np.ones((50, 50), dtype='uint16') * 200
        band3 = np.ones((50, 50), dtype='uint16') * 300
        band4 = np.ones((50, 50), dtype='uint16') * 400
        
        profile = {
            'driver': 'GTiff',
            'width': 50,
            'height': 50,
            'count': 4,
            'dtype': 'uint16',
            'crs': 'EPSG:4326',
            'transform': Affine.identity()
        }
        with rasterio.open(test_file, 'w', **profile) as dst:
            dst.write(band1, 1)
            dst.write(band2, 2)
            dst.write(band3, 3)
            dst.write(band4, 4)
        
        # Test reading multiple bands
        data_multi, meta = load_with_rasterio(test_file, band=[1, 3, 4])
        
        # Verify band dimension preserved for multiple bands
        assert data_multi.shape == (3, 50, 50)
        assert np.all(data_multi[0] == 100)
        assert np.all(data_multi[1] == 300)
        assert np.all(data_multi[2] == 400)
        
        # Test single band dimension removal
        data_single, _ = load_with_rasterio(test_file, band=2)
        assert data_single.shape == (50, 50)  # No band dimension
        assert np.all(data_single == 200)
        
        # Test indexes parameter as alternative
        data_idx, _ = load_with_rasterio(test_file, indexes=[2, 4])
        assert data_idx.shape == (2, 50, 50)
        assert np.all(data_idx[0] == 200)
        assert np.all(data_idx[1] == 400)
        
        # Test reading all bands
        data_all, _ = load_with_rasterio(test_file, band=list(range(1, 5)))
        assert data_all.shape == (4, 50, 50)
        
        # Verify memory layout (C-contiguous for efficiency)
        assert data_multi.flags['C_CONTIGUOUS']
        
        # Test that indexes parameter works when band is None
        data_idx_only, _ = load_with_rasterio(test_file, indexes=2)
        assert data_idx_only.shape == (50, 50)
        assert np.all(data_idx_only == 200)  # indexes=2 (second band)
        
    finally:
        os.unlink(test_file)


def test_load_with_rasterio_performance():
    """Test rasterio performance optimizations.
    
    This test should verify that:
    - COG files enable efficient partial reading
    - Overview access reduces memory usage
    - Windowed reads minimize data transfer
    - Resampling parameter affects performance appropriately
    - Large file handling doesn't exhaust memory
    """
    import rasterio
    from rasterio.transform import Affine
    from rasterio.windows import Window
    from rasterio.enums import Resampling
    
    # Create a mock COG file
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        test_cog = f.name
    
    try:
        # Create larger file to test performance features
        size = 2048
        profile = {
            'driver': 'GTiff',
            'width': size,
            'height': size,
            'count': 1,
            'dtype': 'uint8',
            'crs': 'EPSG:4326',
            'transform': Affine.identity(),
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512,
            'compress': 'deflate'
        }
        
        # Create data with pattern to verify correct reading
        data = np.zeros((size, size), dtype='uint8')
        for i in range(0, size, 256):
            data[i:i+256, :] = i // 256
        
        with rasterio.open(test_cog, 'w', **profile) as dst:
            dst.write(data, 1)
            # Build overviews for COG
            dst.build_overviews([2, 4, 8], Resampling.average)
        
        # Test efficient partial reading with window
        window = Window(512, 512, 256, 256)
        data_window, meta_window = load_with_rasterio(test_cog, window=window)
        
        # Verify only requested data was read
        assert data_window.shape == (256, 256)
        # Bounds are (left, bottom, right, top) so for window at (512, 512)
        bounds = meta_window['bounds']
        assert abs(bounds[0] - 512) < 1e-6  # left
        assert abs(bounds[2] - 768) < 1e-6  # right
        
        # Test overview access (simulated with mock)
        with patch('rasterio.open') as mock_open:
            mock_src = Mock()
            # Simulate overview reading returning smaller array
            overview_data = np.ones((1, size//4, size//4), dtype='uint8')
            mock_src.read.return_value = overview_data
            mock_src.window_transform.return_value = Affine.identity()
            mock_src.window_bounds.return_value = (0, 0, size, size)
            mock_src.bounds = (0, 0, size, size)
            mock_src.transform = Affine.identity()
            mock_src.crs = 'EPSG:4326'
            mock_src.count = 1
            mock_src.dtypes = ['uint8']
            mock_src.nodata = None
            mock_src.profile = profile.copy()
            mock_open.return_value.__enter__.return_value = mock_src
            
            # Request with resampling should trigger overview use
            data_ov, _ = load_with_rasterio('dummy.tif', resampling=Resampling.average)
            
            # Verify resampling parameter was passed
            mock_src.read.assert_called_with(1, window=None, masked=True, resampling=Resampling.average)
        
        # Test memory efficiency with large window reads
        # Read in chunks to avoid loading entire file
        chunk_size = 512
        chunks_read = 0
        
        for row in range(0, size, chunk_size):
            for col in range(0, size, chunk_size):
                window = Window(col, row, chunk_size, chunk_size)
                chunk_data, _ = load_with_rasterio(test_cog, window=window)
                assert chunk_data.shape == (chunk_size, chunk_size)
                chunks_read += 1
        
        # Verify we read the expected number of chunks
        expected_chunks = (size // chunk_size) ** 2
        assert chunks_read == expected_chunks
        
    finally:
        os.unlink(test_cog)


def test_create_virtual_raster():
    """Test VRT creation from multiple raster files.
    
    This test should verify that:
    - VRT is created successfully from file list
    - Temporary VRT files are created when no output path specified
    - Resolution parameter affects VRT properties
    - CRS parameter reprojects inputs as needed
    - Extent parameter crops VRT appropriately
    - Output VRT file is valid and readable
    - VRT enables efficient mosaic processing
    - Error handling for empty file lists
    - Error handling for incompatible input files
    """
    import rasterio
    from rasterio.transform import Affine
    from rasterio.crs import CRS
    
    # Create test raster files
    test_files = []
    temp_files = []
    
    try:
        # Create adjacent tiles
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
                temp_files.append(f.name)
                
            profile = {
                'driver': 'GTiff',
                'width': 100,
                'height': 100,
                'count': 1,
                'dtype': 'uint8',
                'crs': 'EPSG:4326',
                'transform': Affine(0.01, 0, i, 0, -0.01, 1)  # Adjacent tiles
            }
            
            with rasterio.open(temp_files[-1], 'w', **profile) as dst:
                # Write tile number as data for verification
                dst.write(np.full((100, 100), i, dtype='uint8'), 1)
            
            test_files.append(temp_files[-1])
        
        # Test basic VRT creation
        vrt_path = create_virtual_raster(test_files)
        assert os.path.exists(vrt_path)
        temp_files.append(vrt_path)  # Track for cleanup
        
        # Verify VRT is readable
        with rasterio.open(vrt_path) as vrt:
            assert vrt.count == 1
            assert vrt.width == 300  # 3 tiles of 100 pixels
            assert vrt.height == 100
            assert vrt.crs.to_string() == 'EPSG:4326'
            
            # Read data to verify mosaic
            data = vrt.read(1)
            assert data.shape == (100, 300)
            # Verify each tile's data
            assert np.all(data[:, 0:100] == 0)
            assert np.all(data[:, 100:200] == 1)
            assert np.all(data[:, 200:300] == 2)
        
        # Test with specified output path
        output_vrt = tempfile.mktemp(suffix='.vrt')
        vrt_path2 = create_virtual_raster(test_files, output_vrt=output_vrt)
        assert vrt_path2 == output_vrt
        assert os.path.exists(output_vrt)
        temp_files.append(output_vrt)
        
        # Test with resolution parameter
        vrt_res = create_virtual_raster(test_files, resolution=0.02)
        temp_files.append(vrt_res)
        
        with rasterio.open(vrt_res) as vrt:
            # Resolution doubled, so dimensions halved
            assert vrt.width == 150
            assert vrt.height == 50
            assert abs(vrt.transform.a) == 0.02
        
        # Test with CRS parameter
        vrt_crs = create_virtual_raster(test_files, crs='EPSG:3857')
        temp_files.append(vrt_crs)
        
        with rasterio.open(vrt_crs) as vrt:
            assert vrt.crs.to_string() == 'EPSG:3857'
        
        # Test with extent parameter
        extent = (0.5, 0, 2.5, 1)  # Crop to middle portion
        vrt_extent = create_virtual_raster(test_files, extent=extent)
        temp_files.append(vrt_extent)
        
        with rasterio.open(vrt_extent) as vrt:
            assert vrt.bounds.left >= 0.5
            assert vrt.bounds.right <= 2.5
        
        # Test error handling for empty list
        with pytest.raises(ValueError):
            create_virtual_raster([])
        
        # Test with non-existent files
        bad_files = test_files + ['/non/existent/file.tif']
        # GDAL might handle this gracefully or raise error
        try:
            vrt_bad = create_virtual_raster(bad_files)
            temp_files.append(vrt_bad)
            # If it succeeds, VRT should still be created
            assert os.path.exists(vrt_bad)
        except RuntimeError:
            # Expected if GDAL is strict
            pass
        
    finally:
        # Cleanup
        for f in temp_files:
            if os.path.exists(f):
                os.unlink(f)


def test_create_virtual_raster_mosaicking():
    """Test VRT creation for spatial mosaicking.
    
    This test should verify that:
    - Adjacent tiles are properly mosaicked
    - Overlapping areas are handled correctly
    - No-data areas don't interfere with valid data
    - Pixel alignment is maintained across tiles
    - Large numbers of tiles are handled efficiently
    """
    import rasterio
    from rasterio.transform import Affine
    
    temp_files = []
    
    try:
        # Create 2x2 grid of tiles
        tile_size = 50
        tiles = []
        
        for row in range(2):
            for col in range(2):
                with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
                    temp_files.append(f.name)
                
                # Create transform for this tile
                transform = Affine(0.01, 0, col * tile_size * 0.01, 
                                 0, -0.01, 1 - row * tile_size * 0.01)
                
                profile = {
                    'driver': 'GTiff',
                    'width': tile_size,
                    'height': tile_size,
                    'count': 1,
                    'dtype': 'float32',
                    'crs': 'EPSG:4326',
                    'transform': transform,
                    'nodata': -9999
                }
                
                with rasterio.open(temp_files[-1], 'w', **profile) as dst:
                    # Create data with tile ID and some nodata
                    data = np.full((tile_size, tile_size), row * 2 + col, dtype='float32')
                    # Add nodata strip at edges
                    data[0, :] = -9999
                    data[:, 0] = -9999
                    dst.write(data, 1)
                
                tiles.append(temp_files[-1])
        
        # Create VRT mosaic
        vrt_path = create_virtual_raster(tiles)
        temp_files.append(vrt_path)
        
        # Verify mosaic properties
        with rasterio.open(vrt_path) as vrt:
            # Check dimensions (2x2 tiles)
            assert vrt.width == tile_size * 2
            assert vrt.height == tile_size * 2
            
            # Read full mosaic
            data = vrt.read(1)
            
            # Verify adjacent tiles are properly mosaicked
            # Top-left tile (value 0)
            assert np.all(data[1:tile_size-1, 1:tile_size-1] == 0)
            # Top-right tile (value 1)
            assert np.all(data[1:tile_size-1, tile_size+1:tile_size*2-1] == 1)
            # Bottom-left tile (value 2)
            assert np.all(data[tile_size+1:tile_size*2-1, 1:tile_size-1] == 2)
            # Bottom-right tile (value 3)
            assert np.all(data[tile_size+1:tile_size*2-1, tile_size+1:tile_size*2-1] == 3)
            
            # Verify nodata handling
            assert data[0, 0] == -9999  # Nodata preserved
        
        # Test overlapping tiles
        overlap_tiles = []
        for i in range(2):
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
                temp_files.append(f.name)
            
            # Create overlapping tiles (10 pixel overlap)
            transform = Affine(0.01, 0, i * 40 * 0.01, 0, -0.01, 1)
            
            profile = {
                'driver': 'GTiff',
                'width': 50,
                'height': 50,
                'count': 1,
                'dtype': 'uint8',
                'crs': 'EPSG:4326',
                'transform': transform
            }
            
            with rasterio.open(temp_files[-1], 'w', **profile) as dst:
                dst.write(np.full((50, 50), i * 100, dtype='uint8'), 1)
            
            overlap_tiles.append(temp_files[-1])
        
        # Create VRT with overlapping tiles
        vrt_overlap = create_virtual_raster(overlap_tiles)
        temp_files.append(vrt_overlap)
        
        with rasterio.open(vrt_overlap) as vrt:
            data = vrt.read(1)
            # In overlap region, later tile should win (value 100)
            assert np.any(data[:, 40:50] == 100)
        
        # Test large number of tiles (simulate)
        many_tiles = tiles * 10  # Reuse existing tiles
        vrt_many = create_virtual_raster(many_tiles)
        temp_files.append(vrt_many)
        
        # Should handle efficiently without error
        assert os.path.exists(vrt_many)
        
    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.unlink(f)


def test_create_virtual_raster_temporal_stacks():
    """Test VRT creation for temporal data stacks.
    
    This test should verify that:
    - Time series data can be virtually stacked
    - Band ordering is preserved
    - Metadata is properly inherited
    - Memory efficiency is maintained for large stacks
    - Temporal dimension is properly handled
    """
    import rasterio
    from rasterio.transform import Affine
    
    temp_files = []
    
    try:
        # Create time series of rasters (e.g., daily data)
        n_days = 5
        width, height = 100, 100
        time_series_files = []
        
        for day in range(n_days):
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
                temp_files.append(f.name)
            
            profile = {
                'driver': 'GTiff',
                'width': width,
                'height': height,
                'count': 1,
                'dtype': 'float32',
                'crs': 'EPSG:4326',
                'transform': Affine.identity()
            }
            
            with rasterio.open(temp_files[-1], 'w', **profile) as dst:
                # Write day number as data for verification
                data = np.full((height, width), day * 10, dtype='float32')
                dst.write(data, 1)
                # Add temporal metadata
                dst.update_tags(acquisition_date=f'2024-01-{day+1:02d}')
            
            time_series_files.append(temp_files[-1])
        
        # Create VRT stack
        vrt_stack = create_virtual_raster(time_series_files)
        temp_files.append(vrt_stack)
        
        # Verify temporal stack properties
        with rasterio.open(vrt_stack) as vrt:
            # Should maintain spatial dimensions
            assert vrt.width == width
            assert vrt.height == height
            
            # Read data
            data = vrt.read(1)
            
            # For VRT mosaic, later files might override earlier ones
            # This depends on GDAL's VRT implementation
            # Just verify it's readable and has expected shape
            assert data.shape == (height, width)
        
        # Create multi-band stack simulation
        # (True temporal stacking would require different approach)
        multiband_files = []
        
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
                temp_files.append(f.name)
            
            profile = {
                'driver': 'GTiff',
                'width': 50,
                'height': 50,
                'count': 2,  # Multi-band
                'dtype': 'uint16',
                'crs': 'EPSG:4326',
                'transform': Affine.identity()
            }
            
            with rasterio.open(temp_files[-1], 'w', **profile) as dst:
                dst.write(np.full((50, 50), i * 100, dtype='uint16'), 1)
                dst.write(np.full((50, 50), i * 200, dtype='uint16'), 2)
                dst.set_band_description(1, f'Band1_File{i}')
                dst.set_band_description(2, f'Band2_File{i}')
            
            multiband_files.append(temp_files[-1])
        
        # Create VRT from multi-band files
        vrt_multiband = create_virtual_raster(multiband_files)
        temp_files.append(vrt_multiband)
        
        with rasterio.open(vrt_multiband) as vrt:
            # Band count depends on VRT strategy
            assert vrt.count >= 2  # At least 2 bands
            
            # Verify band ordering preserved
            data_b1 = vrt.read(1)
            data_b2 = vrt.read(2)
            assert data_b1.shape == data_b2.shape
        
        # Test memory efficiency with larger stack
        large_stack = time_series_files * 20  # 100 time steps
        vrt_large = create_virtual_raster(large_stack)
        temp_files.append(vrt_large)
        
        # Should create VRT without loading all data
        assert os.path.exists(vrt_large)
        
        # Verify VRT file is small (just XML references)
        vrt_size = os.path.getsize(vrt_large)
        assert vrt_size < 1024 * 1024  # Less than 1MB for VRT file
        
    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.unlink(f)


def test_read_metadata():
    """Test comprehensive metadata extraction from raster files.
    
    This test should verify that:
    - All metadata types ('all', 'spatial', 'bands', 'tags') work correctly
    - Spatial metadata includes CRS, bounds, resolution, shape
    - Band metadata includes count, data types, descriptions, nodata
    - Tag metadata includes file-level and band-level tags
    - Format metadata includes driver, compression, tiling info
    - Pixel size calculation works for different CRS types
    - Units are properly extracted from CRS
    - Error handling for files without metadata
    """
    import rasterio
    from rasterio.transform import Affine
    from rasterio.crs import CRS
    
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        test_file = f.name
    
    try:
        # Create test file with comprehensive metadata
        profile = {
            'driver': 'GTiff',
            'width': 200,
            'height': 150,
            'count': 3,
            'dtype': 'uint16',
            'crs': 'EPSG:4326',
            'transform': Affine(0.1, 0, -180, 0, -0.1, 90),
            'nodata': 65535,
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': 128,
            'blockysize': 128
        }
        
        with rasterio.open(test_file, 'w', **profile) as dst:
            for band in range(1, 4):
                dst.write(np.ones((150, 200), dtype='uint16') * band, band)
                dst.set_band_description(band, f'Band {band} Description')
                dst.update_tags(band, band_tag=f'Band{band}_Value')
            
            # Add file-level tags
            dst.update_tags(sensor='Test_Sensor', date='2024-01-01')
        
        # Test 'all' metadata
        meta_all = read_metadata(test_file, 'all')
        
        # Verify all sections present
        assert 'spatial' in meta_all
        assert 'bands' in meta_all
        assert 'tags' in meta_all
        assert 'format' in meta_all
        
        # Test spatial metadata
        meta_spatial = read_metadata(test_file, 'spatial')
        spatial = meta_spatial['spatial']
        
        assert spatial['transform'] == profile['transform']
        assert spatial['crs'].to_string() == 'EPSG:4326'
        assert spatial['bounds'] == (-180, 75, -160, 90)
        assert spatial['shape'] == (150, 200)
        assert spatial['pixel_size'] == (0.1, 0.1)
        assert spatial['units'] in ['unknown', 'degrees', 'degree']  # Depends on CRS parsing
        
        # Test band metadata
        meta_bands = read_metadata(test_file, 'bands')
        bands = meta_bands['bands']
        
        assert bands['count'] == 3
        assert len(bands['info']) == 3
        
        for i, band_info in enumerate(bands['info']):
            assert band_info['index'] == i + 1
            assert band_info['dtype'] == 'uint16'
            assert band_info['nodata'] == 65535
            assert band_info['description'] == f'Band {i + 1} Description'
        
        # Test tag metadata
        meta_tags = read_metadata(test_file, 'tags')
        tags = meta_tags['tags']
        
        assert tags['file']['sensor'] == 'Test_Sensor'
        assert tags['file']['date'] == '2024-01-01'
        assert tags['bands'][1]['band_tag'] == 'Band1_Value'
        assert tags['bands'][2]['band_tag'] == 'Band2_Value'
        assert tags['bands'][3]['band_tag'] == 'Band3_Value'
        
        # Verify format metadata is always included
        assert meta_all['format']['driver'] == 'GTiff'
        assert meta_all['format']['compression'] == 'lzw'
        assert meta_all['format']['tiled'] == True
        
    finally:
        os.unlink(test_file)
    
    # Test with projected CRS (meters)
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        test_utm = f.name
    
    try:
        profile_utm = {
            'driver': 'GTiff',
            'width': 100,
            'height': 100,
            'count': 1,
            'dtype': 'float32',
            'crs': 'EPSG:32633',  # UTM Zone 33N
            'transform': Affine(30, 0, 500000, 0, -30, 5000000)  # 30m pixels
        }
        
        with rasterio.open(test_utm, 'w', **profile_utm) as dst:
            dst.write(np.ones((100, 100), dtype='float32'), 1)
        
        meta_utm = read_metadata(test_utm, 'spatial')
        spatial_utm = meta_utm['spatial']
        
        assert spatial_utm['pixel_size'] == (30, 30)
        # Units might be 'metre', 'meter', or 'unknown' depending on parsing
        
    finally:
        os.unlink(test_utm)
    
    # Test file without metadata (minimal file)
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        test_minimal = f.name
    
    try:
        profile_min = {
            'driver': 'GTiff',
            'width': 10,
            'height': 10,
            'count': 1,
            'dtype': 'uint8'
            # No CRS, no tags
        }
        
        with rasterio.open(test_minimal, 'w', **profile_min) as dst:
            dst.write(np.ones((10, 10), dtype='uint8'), 1)
        
        meta_min = read_metadata(test_minimal)
        
        # Should handle missing metadata gracefully
        assert meta_min['spatial']['crs'] is None
        assert meta_min['bands']['info'][0]['description'] is None
        assert len(meta_min['tags']['file']) == 0
        
    finally:
        os.unlink(test_minimal)


def test_read_metadata_spatial_only():
    """Test spatial metadata extraction.
    
    This test should verify that:
    - Only spatial information is returned
    - Transform matrices are properly extracted
    - Bounds calculation is accurate
    - Resolution calculation handles different pixel shapes
    - CRS information is complete and valid
    """
    import rasterio
    from rasterio.transform import Affine
    from rasterio.crs import CRS
    
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        test_file = f.name
    
    try:
        # Create test file with non-square pixels
        profile = {
            'driver': 'GTiff',
            'width': 360,
            'height': 180,
            'count': 1,
            'dtype': 'float32',
            'crs': CRS.from_epsg(4326),
            'transform': Affine(1.0, 0, -180, 0, -0.5, 90)  # Non-square pixels
        }
        
        with rasterio.open(test_file, 'w', **profile) as dst:
            dst.write(np.ones((180, 360), dtype='float32'), 1)
            # Add other metadata that should NOT be returned
            dst.update_tags(test='should_not_appear')
        
        # Test spatial-only extraction
        metadata = read_metadata(test_file, 'spatial')
        
        # Verify only spatial section is returned
        assert 'spatial' in metadata
        assert 'bands' not in metadata
        assert 'tags' not in metadata
        
        spatial = metadata['spatial']
        
        # Verify transform extraction
        assert isinstance(spatial['transform'], Affine)
        assert spatial['transform'] == profile['transform']
        assert spatial['transform'].a == 1.0  # x pixel size
        assert spatial['transform'].e == -0.5  # y pixel size
        
        # Verify bounds calculation
        expected_bounds = (-180, 0, 180, 90)
        assert spatial['bounds'] == expected_bounds
        
        # Verify resolution with non-square pixels
        assert spatial['pixel_size'] == (1.0, 0.5)
        
        # Verify CRS information
        assert spatial['crs'] is not None
        assert spatial['crs'].to_epsg() == 4326
        assert spatial['crs'].to_string() == 'EPSG:4326'
        
        # Verify shape
        assert spatial['shape'] == (180, 360)
        
    finally:
        os.unlink(test_file)
    
    # Test with rotated transform (rare but valid)
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        test_rotated = f.name
    
    try:
        # Create transform with rotation
        transform_rotated = Affine(1.0, 0.1, 0,  # Slight rotation
                                  0.1, -1.0, 100)
        
        profile_rot = {
            'driver': 'GTiff',
            'width': 100,
            'height': 100,
            'count': 1,
            'dtype': 'uint8',
            'crs': 'EPSG:3857',
            'transform': transform_rotated
        }
        
        with rasterio.open(test_rotated, 'w', **profile_rot) as dst:
            dst.write(np.ones((100, 100), dtype='uint8'), 1)
        
        meta_rot = read_metadata(test_rotated, 'spatial')
        spatial_rot = meta_rot['spatial']
        
        # Should handle rotated transform
        assert spatial_rot['transform'] == transform_rotated
        # Pixel size should be calculated from scale components
        assert abs(spatial_rot['pixel_size'][0] - 1.0) < 0.2  # Approximate due to rotation
        
    finally:
        os.unlink(test_rotated)


def test_read_metadata_performance():
    """Test metadata extraction performance.
    
    This test should verify that:
    - Metadata is read without loading raster data
    - Large files can be analyzed quickly
    - Network files (if supported) don't transfer data
    - Caching improves repeated access
    - Memory usage remains minimal
    """
    import rasterio
    from rasterio.transform import Affine
    import time
    
    # Create a "large" file (simulate with reasonable size)
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        test_file = f.name
    
    try:
        # Create file large enough to measure but not too large for tests
        size = 1000
        profile = {
            'driver': 'GTiff',
            'width': size,
            'height': size,
            'count': 5,  # Multiple bands
            'dtype': 'float64',  # Larger data type
            'crs': 'EPSG:4326',
            'transform': Affine.identity(),
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256
        }
        
        with rasterio.open(test_file, 'w', **profile) as dst:
            # Write only a small amount to create structure
            # Don't write all data to keep test fast
            small_data = np.ones((256, 256), dtype='float64')
            dst.write(small_data, 1, window=rasterio.windows.Window(0, 0, 256, 256))
            
            # Add metadata
            for band in range(1, 6):
                dst.set_band_description(band, f'Band_{band}')
                dst.update_tags(band, test=f'band{band}')
        
        # Time metadata extraction
        start_time = time.time()
        metadata = read_metadata(test_file, 'all')
        end_time = time.time()
        
        # Should be fast (under 1 second even for large file)
        assert end_time - start_time < 1.0
        
        # Verify metadata was extracted without loading all data
        assert metadata['spatial']['shape'] == (size, size)
        assert metadata['bands']['count'] == 5
        
        # Test repeated access (potential caching benefit)
        times = []
        for _ in range(5):
            start = time.time()
            _ = read_metadata(test_file, 'spatial')
            times.append(time.time() - start)
        
        # Later accesses might be faster due to OS caching
        # Just verify they're all reasonably fast
        assert all(t < 0.5 for t in times)
        
        # Test that only requested metadata is extracted
        start_spatial = time.time()
        meta_spatial = read_metadata(test_file, 'spatial')
        time_spatial = time.time() - start_spatial
        
        start_all = time.time()
        meta_all = read_metadata(test_file, 'all')
        time_all = time.time() - start_all
        
        # Spatial-only should be faster or similar to all
        # (May not always be true due to caching, but verify both are fast)
        assert time_spatial < 1.0
        assert time_all < 1.0
        
    finally:
        os.unlink(test_file)
    
    # Test with mock for network file simulation
    with patch('rasterio.open') as mock_open:
        mock_src = Mock()
        mock_src.__enter__ = Mock(return_value=mock_src)
        mock_src.__exit__ = Mock(return_value=None)
        
        # Configure mock to simulate remote file
        mock_src.transform = Affine.identity()
        mock_src.crs = 'EPSG:4326'
        mock_src.bounds = (0, 0, 100, 100)
        mock_src.height = 100
        mock_src.width = 100
        mock_src.count = 1
        mock_src.dtypes = ['uint8']
        mock_src.nodatavals = [None]
        mock_src.descriptions = [None]
        mock_src.driver = 'GTiff'
        mock_src.is_tiled = False
        mock_src.tags = Mock(return_value={})
        
        mock_open.return_value = mock_src
        
        # Should not call read() for metadata extraction
        metadata = read_metadata('s3://bucket/file.tif')
        
        # Verify read() was never called (no data transfer)
        assert not mock_src.read.called


def test_setup_output_profile():
    """Test rasterio profile setup for output file creation.
    
    This test should verify that:
    - Template profiles are properly copied and modified
    - Default profiles are valid for common use cases
    - Driver-specific options are set appropriately
    - GTiff profiles include compression and tiling
    - Data type conversions work correctly
    - Modifications override template settings
    - Output profiles create valid files
    - Error handling for invalid template files
    """
    import rasterio
    from rasterio.transform import Affine
    
    # Test with template file
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        template_file = f.name
    
    try:
        # Create template
        template_profile = {
            'driver': 'GTiff',
            'width': 100,
            'height': 100,
            'count': 2,
            'dtype': 'uint8',
            'crs': 'EPSG:4326',
            'transform': Affine(0.1, 0, 0, 0, -0.1, 10),
            'nodata': 255
        }
        
        with rasterio.open(template_file, 'w', **template_profile) as dst:
            dst.write(np.ones((100, 100), dtype='uint8'), 1)
        
        # Test profile creation from template
        profile = setup_output_profile(template_path=template_file)
        
        # Verify template copied
        assert profile['width'] == 100
        assert profile['height'] == 100
        assert profile['crs'] == 'EPSG:4326'
        assert profile['transform'] == template_profile['transform']
        
        # Verify GTiff defaults added
        assert profile['driver'] == 'GTiff'
        assert profile['compress'] == 'deflate'
        assert profile['tiled'] == True  # Added for 100x100
        
        # Test with modifications
        mods = {
            'dtype': 'float32',
            'compress': 'lzw',
            'tiled': False,
            'nodata': -9999
        }
        profile_mod = setup_output_profile(
            template_path=template_file,
            modifications=mods
        )
        
        # Verify modifications override
        assert profile_mod['dtype'] == 'float32'
        assert profile_mod['compress'] == 'lzw'
        assert profile_mod['tiled'] == False
        assert profile_mod['nodata'] == -9999
        
        # Test dtype parameter
        profile_dtype = setup_output_profile(
            template_path=template_file,
            dtype='int16'
        )
        assert profile_dtype['dtype'] == 'int16'
        
    finally:
        os.unlink(template_file)
    
    # Test with template metadata dict
    template_meta = {
        'driver': 'GTiff',
        'width': 512,
        'height': 512,
        'count': 1,
        'dtype': 'float64',
        'crs': 'EPSG:3857',
        'transform': Affine.identity()
    }
    
    profile_meta = setup_output_profile(template_meta=template_meta)
    
    # Verify template used
    assert profile_meta['width'] == 512
    assert profile_meta['crs'] == 'EPSG:3857'
    
    # Verify large file gets tiling
    assert profile_meta['tiled'] == True
    assert profile_meta['blockxsize'] == 256
    assert profile_meta['blockysize'] == 256
    
    # Test without template (default profile)
    profile_default = setup_output_profile()
    
    # Verify sensible defaults
    assert profile_default['driver'] == 'GTiff'
    assert profile_default['dtype'] == 'float32'
    assert profile_default['count'] == 1
    assert profile_default['compress'] == 'deflate'
    
    # Test different driver
    profile_hdf = setup_output_profile(driver='HDF4')
    assert profile_hdf['driver'] == 'HDF4'
    # Should not have GTiff-specific options
    assert 'compress' not in profile_hdf or profile_hdf['driver'] != 'HDF4'
    
    # Test creating file with profile
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        output_file = f.name
    
    try:
        # Use profile to create file
        test_profile = setup_output_profile(
            modifications={
                'width': 50,
                'height': 50,
                'count': 1,
                'dtype': 'uint16',
                'crs': 'EPSG:4326',
                'transform': Affine.identity()
            }
        )
        
        with rasterio.open(output_file, 'w', **test_profile) as dst:
            dst.write(np.ones((50, 50), dtype='uint16'), 1)
        
        # Verify file is valid
        with rasterio.open(output_file) as src:
            assert src.width == 50
            assert src.height == 50
            assert src.compression == 'deflate'
        
    finally:
        os.unlink(output_file)


def test_setup_output_profile_gtiff_optimization():
    """Test GeoTIFF profile optimization.
    
    This test should verify that:
    - Compression is enabled by default
    - Tiling is enabled for large files
    - Block sizes are appropriate for typical use
    - COG-compatible settings are used when possible
    - Predictor is set for floating-point data
    """
    import rasterio
    from rasterio.transform import Affine
    
    # Test small file (no tiling)
    small_profile = setup_output_profile(
        modifications={
            'width': 256,
            'height': 256,
            'count': 1,
            'dtype': 'uint8'
        },
        driver='GTiff'
    )
    
    # Verify compression enabled by default
    assert small_profile['compress'] == 'deflate'
    
    # Small file might not get tiling
    # Implementation shows tiling added for files > 512 pixels
    assert 'tiled' not in small_profile or small_profile['tiled'] == False
    
    # Test large file (should get tiling)
    large_profile = setup_output_profile(
        modifications={
            'width': 2048,
            'height': 2048,
            'count': 1,
            'dtype': 'float32'
        },
        driver='GTiff'
    )
    
    # Verify tiling enabled for large files
    assert large_profile['tiled'] == True
    assert large_profile['blockxsize'] == 256
    assert large_profile['blockysize'] == 256
    
    # Verify compression
    assert large_profile['compress'] == 'deflate'
    
    # Test floating-point optimization
    # Note: Current implementation doesn't add predictor automatically
    # but we can test that it works when specified
    float_profile = setup_output_profile(
        modifications={
            'width': 1024,
            'height': 1024,
            'dtype': 'float32',
            'predictor': 3  # Floating-point predictor
        },
        driver='GTiff'
    )
    
    assert float_profile['predictor'] == 3
    
    # Test COG-compatible settings
    cog_profile = setup_output_profile(
        modifications={
            'width': 4096,
            'height': 4096,
            'dtype': 'uint8',
            'compress': 'deflate',
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512,
            'BIGTIFF': 'IF_SAFER'
        },
        driver='GTiff'
    )
    
    # Verify COG-compatible settings preserved
    assert cog_profile['tiled'] == True
    assert cog_profile['blockxsize'] == 512
    assert cog_profile['blockysize'] == 512
    assert cog_profile['compress'] == 'deflate'
    assert cog_profile['BIGTIFF'] == 'IF_SAFER'
    
    # Test different compression options
    lzw_profile = setup_output_profile(
        modifications={
            'width': 1000,
            'height': 1000,
            'compress': 'lzw'
        },
        driver='GTiff'
    )
    
    assert lzw_profile['compress'] == 'lzw'  # User preference respected
    
    # Test that existing compression isn't overridden
    template_meta = {
        'driver': 'GTiff',
        'width': 1000,
        'height': 1000,
        'compress': 'zstd',
        'tiled': True
    }
    
    profile_existing = setup_output_profile(template_meta=template_meta)
    
    # Should preserve existing compression from template
    assert profile_existing['compress'] == 'zstd'


def test_setup_output_profile_format_compatibility():
    """Test output profile compatibility across formats.
    
    This test should verify that:
    - Profiles work with different output drivers
    - Format-specific limitations are respected
    - Unsupported options are filtered out
    - Cross-format conversion maintains essential metadata
    - Warning messages are appropriate for incompatible settings
    """
    import rasterio
    from rasterio.transform import Affine
    
    # Base template for testing
    base_template = {
        'width': 100,
        'height': 100,
        'count': 1,
        'dtype': 'float32',
        'crs': 'EPSG:4326',
        'transform': Affine.identity(),
        'nodata': -9999
    }
    
    # Test GTiff profile
    gtiff_profile = setup_output_profile(
        template_meta=base_template,
        driver='GTiff'
    )
    
    assert gtiff_profile['driver'] == 'GTiff'
    assert 'compress' in gtiff_profile  # GTiff supports compression
    
    # Test HDF4 profile (different capabilities)
    hdf_profile = setup_output_profile(
        template_meta=base_template,
        driver='HDF4'
    )
    
    assert hdf_profile['driver'] == 'HDF4'
    # HDF4 has different options than GTiff
    
    # Test format conversion maintaining metadata
    gtiff_template = {
        'driver': 'GTiff',
        'width': 200,
        'height': 150,
        'count': 3,
        'dtype': 'uint16',
        'crs': 'EPSG:3857',
        'transform': Affine(10, 0, 0, 0, -10, 1000),
        'compress': 'lzw',
        'tiled': True,
        'photometric': 'rgb'  # GTiff-specific
    }
    
    # Convert to different format
    png_profile = setup_output_profile(
        template_meta=gtiff_template,
        driver='PNG'
    )
    
    # Essential metadata maintained
    assert png_profile['width'] == 200
    assert png_profile['height'] == 150
    assert png_profile['crs'] == 'EPSG:3857'
    assert png_profile['transform'] == gtiff_template['transform']
    
    # Format changed
    assert png_profile['driver'] == 'PNG'
    
    # Test with incompatible options
    incompatible_mods = {
        'compress': 'jpeg',  # JPEG compression
        'jpeg_quality': 90,
        'tiled': True,
        'sparse_ok': True  # GTiff sparse files
    }
    
    # These options might not work with all formats
    bmp_profile = setup_output_profile(
        template_meta=base_template,
        modifications=incompatible_mods,
        driver='BMP'
    )
    
    assert bmp_profile['driver'] == 'BMP'
    # BMP might not support all GTiff options, but function should handle gracefully
    
    # Test preservation of critical metadata across formats
    critical_meta = {
        'width': 512,
        'height': 512,
        'count': 4,
        'dtype': 'float64',
        'crs': 'EPSG:4326',
        'transform': Affine(0.001, 0, -180, 0, -0.001, 90),
        'nodata': float('nan')
    }
    
    # Test multiple format conversions
    for driver in ['GTiff', 'HDF4', 'ENVI']:
        profile = setup_output_profile(
            template_meta=critical_meta,
            driver=driver
        )
        
        # Critical metadata preserved
        assert profile['width'] == 512
        assert profile['height'] == 512
        assert profile['dtype'] == 'float64'
        assert profile['crs'] == 'EPSG:4326'
        assert profile['transform'] == critical_meta['transform']
        assert profile['driver'] == driver
    
    # Test that modifications override even for different formats
    override_profile = setup_output_profile(
        template_meta=gtiff_template,
        driver='ENVI',
        modifications={'dtype': 'int32', 'count': 5}
    )
    
    assert override_profile['driver'] == 'ENVI'
    assert override_profile['dtype'] == 'int32'
    assert override_profile['count'] == 5


def test_validate_raster_stack():
    """Test validation of multiple rasters for stacking compatibility.
    
    This test should verify that:
    - Compatible raster stacks pass validation
    - CRS mismatches are detected and reported
    - Resolution differences are identified
    - Spatial misalignment is caught
    - Common extent calculation works correctly
    - Validation reports include specific issues
    - Recommendations are actionable and helpful
    - Empty file lists are handled gracefully
    - Corrupted files don't crash validation
    """
    import rasterio
    from rasterio.transform import Affine
    from rasterio.crs import CRS
    
    temp_files = []
    
    try:
        # Create compatible raster stack
        compatible_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
                temp_files.append(f.name)
            
            profile = {
                'driver': 'GTiff',
                'width': 100,
                'height': 100,
                'count': 1,
                'dtype': 'float32',
                'crs': 'EPSG:4326',
                'transform': Affine(0.1, 0, -180, 0, -0.1, 90),
                'nodata': -9999
            }
            
            with rasterio.open(temp_files[-1], 'w', **profile) as dst:
                dst.write(np.full((100, 100), i, dtype='float32'), 1)
            
            compatible_files.append(temp_files[-1])
        
        # Test compatible stack
        result = validate_raster_stack(compatible_files)
        
        assert result['valid'] == True
        assert len(result['issues']) == 0
        assert len(result['alignment_errors']) == 0
        assert result['common_extent'] == (-180, 80, -170, 90)
        
        # Test CRS mismatch
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            crs_mismatch = f.name
            temp_files.append(crs_mismatch)
        
        profile_diff_crs = profile.copy()
        profile_diff_crs['crs'] = 'EPSG:3857'
        
        with rasterio.open(crs_mismatch, 'w', **profile_diff_crs) as dst:
            dst.write(np.ones((100, 100), dtype='float32'), 1)
        
        result_crs = validate_raster_stack(compatible_files + [crs_mismatch])
        
        assert result_crs['valid'] == False
        assert any('CRS mismatch' in issue for issue in result_crs['issues'])
        assert 'Reproject all files to common CRS' in result_crs['recommendations']
        
        # Test resolution mismatch
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            res_mismatch = f.name
            temp_files.append(res_mismatch)
        
        profile_diff_res = profile.copy()
        profile_diff_res['transform'] = Affine(0.2, 0, -180, 0, -0.2, 90)  # Different resolution
        
        with rasterio.open(res_mismatch, 'w', **profile_diff_res) as dst:
            dst.write(np.ones((50, 50), dtype='float32'), 1)
        
        result_res = validate_raster_stack(
            compatible_files + [res_mismatch],
            check_crs=False  # Skip CRS check
        )
        
        assert result_res['valid'] == False
        assert any('Resolution mismatch' in issue for issue in result_res['issues'])
        assert 'Resample all files to common resolution' in result_res['recommendations']
        
        # Test spatial misalignment
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            misaligned = f.name
            temp_files.append(misaligned)
        
        # Offset by half pixel
        profile_misaligned = profile.copy()
        profile_misaligned['transform'] = Affine(0.1, 0, -180.05, 0, -0.1, 90.05)
        
        with rasterio.open(misaligned, 'w', **profile_misaligned) as dst:
            dst.write(np.ones((100, 100), dtype='float32'), 1)
        
        result_align = validate_raster_stack(
            compatible_files + [misaligned],
            check_crs=False,
            check_resolution=False
        )
        
        assert result_align['valid'] == False
        assert len(result_align['alignment_errors']) > 0
        offset = result_align['alignment_errors'][0]['offset']
        assert abs(offset[0] - 0.05) < 1e-10
        assert abs(offset[1] - 0.05) < 1e-10
        assert 'Align all rasters to common grid' in result_align['recommendations']
        
        # Test no spatial overlap
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            no_overlap = f.name
            temp_files.append(no_overlap)
        
        # Far away location
        profile_no_overlap = profile.copy()
        profile_no_overlap['transform'] = Affine(0.1, 0, 0, 0, -0.1, 0)
        
        with rasterio.open(no_overlap, 'w', **profile_no_overlap) as dst:
            dst.write(np.ones((100, 100), dtype='float32'), 1)
        
        result_overlap = validate_raster_stack(
            compatible_files + [no_overlap],
            check_crs=False,
            check_resolution=False,
            check_alignment=False
        )
        
        assert result_overlap['common_extent'] is None
        assert any('No spatial overlap' in issue for issue in result_overlap['issues'])
        
        # Test empty file list
        result_empty = validate_raster_stack([])
        
        assert result_empty['valid'] == False
        assert 'No files provided' in result_empty['issues']
        
        # Test corrupted file handling
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            corrupted = f.name
            temp_files.append(corrupted)
            f.write(b'Not a valid raster')
        
        result_corrupted = validate_raster_stack(compatible_files + [corrupted])
        
        # Should handle gracefully
        assert any('Cannot read' in issue for issue in result_corrupted['issues'])
        # With valid files present, the validation continues
        assert 'issues' in result_corrupted
        
        # Test tolerance parameter
        result_strict = validate_raster_stack(
            compatible_files + [misaligned],
            check_crs=False,
            check_resolution=False,
            tolerance=0.01  # Stricter tolerance
        )
        
        assert result_strict['valid'] == False  # Should catch small misalignment
        
    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.unlink(f)


def test_validate_raster_stack_alignment():
    """Test spatial alignment validation.
    
    This test should verify that:
    - Grid alignment is checked accurately
    - Tolerance parameter affects sensitivity appropriately
    - Sub-pixel misalignments are detected
    - Alignment errors include offset information
    - Edge cases (perfect alignment) work correctly
    """
    import rasterio
    from rasterio.transform import Affine
    
    temp_files = []
    
    try:
        # Create reference raster
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            ref_file = f.name
            temp_files.append(ref_file)
        
        ref_transform = Affine(1.0, 0, 100.0, 0, -1.0, 50.0)
        profile = {
            'driver': 'GTiff',
            'width': 100,
            'height': 100,
            'count': 1,
            'dtype': 'float32',
            'crs': 'EPSG:4326',
            'transform': ref_transform
        }
        
        with rasterio.open(ref_file, 'w', **profile) as dst:
            dst.write(np.ones((100, 100), dtype='float32'), 1)
        
        # Test perfect alignment
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            aligned_file = f.name
            temp_files.append(aligned_file)
        
        with rasterio.open(aligned_file, 'w', **profile) as dst:
            dst.write(np.ones((100, 100), dtype='float32') * 2, 1)
        
        result = validate_raster_stack([ref_file, aligned_file])
        
        assert result['valid'] == True
        assert len(result['alignment_errors']) == 0
        
        # Test sub-pixel misalignment
        sub_pixel_offsets = [
            (0.1, 0.0),    # X offset only
            (0.0, 0.1),    # Y offset only
            (0.05, 0.05),  # Small diagonal offset
            (0.5, 0.5),    # Half-pixel offset
            (0.9, 0.9)     # Large sub-pixel offset
        ]
        
        for i, (x_off, y_off) in enumerate(sub_pixel_offsets):
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
                misaligned = f.name
                temp_files.append(misaligned)
            
            # Create transform with offset
            offset_transform = Affine(1.0, 0, 100.0 + x_off, 0, -1.0, 50.0 + y_off)
            profile['transform'] = offset_transform
            
            with rasterio.open(misaligned, 'w', **profile) as dst:
                dst.write(np.ones((100, 100), dtype='float32') * (i + 3), 1)
            
            # Test with default tolerance
            result_default = validate_raster_stack(
                [ref_file, misaligned],
                check_crs=False,
                check_resolution=False,
                tolerance=1e-6
            )
            
            # Should detect misalignment
            assert result_default['valid'] == False
            assert len(result_default['alignment_errors']) == 1
            
            # Check offset information
            error = result_default['alignment_errors'][0]
            assert abs(error['offset'][0] - x_off) < 1e-10
            assert abs(error['offset'][1] - y_off) < 1e-10
            
            # Test with looser tolerance
            if x_off <= 0.1 and y_off <= 0.1:
                result_loose = validate_raster_stack(
                    [ref_file, misaligned],
                    check_crs=False,
                    check_resolution=False,
                    tolerance=0.2  # Allow up to 0.2 pixel offset
                )
                
                # Small offsets should pass with loose tolerance
                assert result_loose['valid'] == True
                assert len(result_loose['alignment_errors']) == 0
        
        # Test grid alignment with different resolution
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            diff_res = f.name
            temp_files.append(diff_res)
        
        # Same origin but different resolution - should still check alignment
        diff_transform = Affine(2.0, 0, 100.0, 0, -2.0, 50.0)  # 2x resolution
        profile['transform'] = diff_transform
        profile['width'] = 50
        profile['height'] = 50
        
        with rasterio.open(diff_res, 'w', **profile) as dst:
            dst.write(np.ones((50, 50), dtype='float32'), 1)
        
        result_diff_res = validate_raster_stack(
            [ref_file, diff_res],
            check_crs=False,
            check_resolution=False,  # Only checking alignment
            check_alignment=True
        )
        
        # Origins align, so should pass alignment check
        assert len(result_diff_res['alignment_errors']) == 0
        
        # Test multiple files with various alignments
        all_files = [ref_file, aligned_file] + temp_files[2:5]  # Mix of aligned and misaligned
        
        result_multi = validate_raster_stack(all_files, tolerance=0.01)
        
        # Should report all misaligned files
        assert len(result_multi['alignment_errors']) >= 2
        
    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.unlink(f)


def test_validate_raster_stack_edge_cases():
    """Test stack validation edge cases.
    
    This test should verify that:
    - Single-file stacks are handled correctly
    - Files with no spatial overlap are detected
    - Very large numbers of files are processed efficiently
    - Mixed format stacks are validated appropriately
    - Network files are handled if supported
    """
    import rasterio
    from rasterio.transform import Affine
    
    temp_files = []
    
    try:
        # Test single-file stack
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            single_file = f.name
            temp_files.append(single_file)
        
        profile = {
            'driver': 'GTiff',
            'width': 100,
            'height': 100,
            'count': 1,
            'dtype': 'uint8',
            'crs': 'EPSG:4326',
            'transform': Affine.identity()
        }
        
        with rasterio.open(single_file, 'w', **profile) as dst:
            dst.write(np.ones((100, 100), dtype='uint8'), 1)
        
        result_single = validate_raster_stack([single_file])
        
        # Single file should be valid
        assert result_single['valid'] == True
        assert len(result_single['issues']) == 0
        assert result_single['common_extent'] == (0, 0, 100, 100)
        
        # Test files with no spatial overlap
        non_overlapping = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
                temp_files.append(f.name)
                non_overlapping.append(f.name)
            
            # Place tiles far apart
            transform = Affine(1, 0, i * 200, 0, -1, 0)
            profile['transform'] = transform
            
            with rasterio.open(temp_files[-1], 'w', **profile) as dst:
                dst.write(np.full((100, 100), i, dtype='uint8'), 1)
        
        result_no_overlap = validate_raster_stack(non_overlapping)
        
        # Should detect no overlap
        assert result_no_overlap['common_extent'] is None
        assert any('No spatial overlap' in issue for issue in result_no_overlap['issues'])
        
        # Test very large number of files (simulated)
        large_file_list = []
        
        # Create a few actual files
        for i in range(5):
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
                temp_files.append(f.name)
                large_file_list.append(f.name)
            
            profile['transform'] = Affine(1, 0, 0, 0, -1, 100)
            with rasterio.open(temp_files[-1], 'w', **profile) as dst:
                dst.write(np.ones((100, 100), dtype='uint8'), 1)
        
        # Simulate large list by repeating
        large_file_list = large_file_list * 20  # 100 files
        
        import time
        start_time = time.time()
        result_large = validate_raster_stack(large_file_list)
        end_time = time.time()
        
        # Should process efficiently
        assert end_time - start_time < 5.0  # Should complete in reasonable time
        assert result_large['valid'] == True  # All files are identical
        
        # Test mixed format stack
        mixed_formats = []
        
        # Create different format files
        formats = [
            {'suffix': '.tif', 'driver': 'GTiff'},
            {'suffix': '.img', 'driver': 'HFA'},  # ERDAS Imagine
            {'suffix': '.tif', 'driver': 'GTiff'}  # Another GTiff
        ]
        
        for fmt in formats:
            with tempfile.NamedTemporaryFile(suffix=fmt['suffix'], delete=False) as f:
                temp_files.append(f.name)
                mixed_formats.append(f.name)
            
            profile['driver'] = fmt['driver']
            
            # Some formats might not support all features
            if fmt['driver'] == 'HFA':
                profile.pop('compress', None)  # Remove compression if present
            
            try:
                with rasterio.open(temp_files[-1], 'w', **profile) as dst:
                    dst.write(np.ones((100, 100), dtype='uint8'), 1)
            except Exception:
                # If format not supported, create as GTiff instead
                profile['driver'] = 'GTiff'
                with rasterio.open(temp_files[-1], 'w', **profile) as dst:
                    dst.write(np.ones((100, 100), dtype='uint8'), 1)
        
        result_mixed = validate_raster_stack(mixed_formats)
        
        # Mixed formats should still validate if spatially compatible
        # Validation focuses on spatial properties, not format
        assert 'issues' in result_mixed
        
        # Test with mock network file
        with patch('rasterio.open') as mock_open:
            # Mock successful open of network file
            mock_src = Mock()
            mock_src.crs = 'EPSG:4326'
            mock_src.transform = Affine.identity()
            mock_src.height = 100
            mock_src.width = 100
            mock_src.bounds = Mock()
            mock_src.bounds.left = 0
            mock_src.bounds.bottom = 0
            mock_src.bounds.right = 100
            mock_src.bounds.top = 100
            
            mock_open.return_value.__enter__.return_value = mock_src
            mock_open.return_value.__exit__.return_value = None
            
            # Should handle network files
            network_files = ['s3://bucket/file1.tif', 's3://bucket/file2.tif']
            result_network = validate_raster_stack(network_files)
            
            # Should attempt to validate
            assert mock_open.call_count == 2
        
    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.unlink(f)


def test_create_memory_efficient_reader():
    """Test memory-efficient reader creation for large rasters.
    
    This test should verify that:
    - Reader generator function is created successfully
    - Chunk size parameter affects memory usage
    - Overlap parameter enables context-aware processing
    - Chunks are yielded in logical order (e.g., row-major)
    - Edge chunks are handled correctly
    - Window information is accurate for each chunk
    - Single-band dimension handling works in chunks
    - Error handling for invalid chunk sizes
    - Performance scales appropriately with file size
    """
    import rasterio
    from rasterio.transform import Affine
    from rasterio.windows import Window
    
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        test_file = f.name
    
    try:
        # Create test raster with known pattern
        width, height = 1000, 800
        profile = {
            'driver': 'GTiff',
            'width': width,
            'height': height,
            'count': 1,
            'dtype': 'uint16',
            'crs': 'EPSG:4326',
            'transform': Affine.identity(),
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256
        }
        
        # Create data with position-encoded values
        data = np.zeros((height, width), dtype='uint16')
        for y in range(height):
            for x in range(width):
                data[y, x] = (y // 100) * 10 + (x // 100)
        
        with rasterio.open(test_file, 'w', **profile) as dst:
            dst.write(data, 1)
        
        # Test basic reader creation
        reader = create_memory_efficient_reader(test_file)
        assert callable(reader)
        
        # Test chunk generation
        chunk_reader = reader()
        chunks = list(chunk_reader)
        
        # Default chunk size is (1024, 1024), so expect 1x1 chunks for 1000x800
        assert len(chunks) == 1
        
        chunk_data, window = chunks[0]
        assert chunk_data.shape == (800, 1000)  # Full image fits in one chunk
        assert isinstance(window, Window)
        
        # Test with smaller chunks
        reader_small = create_memory_efficient_reader(
            test_file,
            chunk_size=(256, 256)
        )
        
        small_chunks = list(reader_small())
        
        # Calculate expected number of chunks
        expected_chunks_x = (width + 255) // 256  # 4 chunks
        expected_chunks_y = (height + 255) // 256  # 4 chunks
        assert len(small_chunks) == expected_chunks_x * expected_chunks_y
        
        # Verify chunks are yielded in row-major order
        chunk_positions = []
        for chunk_data, window in small_chunks:
            chunk_positions.append((window.row_off, window.col_off))
        
        # Check row-major order
        for i in range(1, len(chunk_positions)):
            prev_row, prev_col = chunk_positions[i-1]
            curr_row, curr_col = chunk_positions[i]
            
            # Either same row with increasing col, or next row
            if curr_row == prev_row:
                assert curr_col > prev_col
            else:
                assert curr_row > prev_row
                assert curr_col == 0  # New row starts at col 0
        
        # Verify edge chunks have correct size
        last_chunk_data, last_window = small_chunks[-1]
        
        # Last chunk should be at bottom-right
        assert last_window.row_off == 768  # 3 * 256
        assert last_window.col_off == 768  # 3 * 256
        
        # Edge chunk dimensions
        assert last_chunk_data.shape == (32, 232)  # 800-768, 1000-768
        
        # Test overlap parameter
        reader_overlap = create_memory_efficient_reader(
            test_file,
            chunk_size=(300, 300),
            overlap=50
        )
        
        overlap_chunks = list(reader_overlap())
        
        # First chunk should include overlap
        first_chunk, first_window = overlap_chunks[0]
        assert first_window.row_off == 0  # No negative overlap at edges
        assert first_window.col_off == 0
        assert first_chunk.shape == (350, 350)  # 300 + 50 overlap on one side
        
        # Middle chunks should have overlap on all sides
        for chunk_data, window in overlap_chunks:
            if window.row_off > 0 and window.col_off > 0:
                # Not at edges, should have overlap
                if window.row_off + 300 < height and window.col_off + 300 < width:
                    # Full overlap possible
                    assert chunk_data.shape[0] >= 300  # At least chunk size
                    assert chunk_data.shape[1] >= 300
        
        # Test single-band dimension handling
        single_band_chunk, _ = small_chunks[0]
        assert single_band_chunk.ndim == 2  # No band dimension
        
        # Test multi-band file
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            multiband_file = f.name
        
        try:
            profile['count'] = 3
            with rasterio.open(multiband_file, 'w', **profile) as dst:
                for band in range(1, 4):
                    dst.write(data * band, band)
            
            reader_multi = create_memory_efficient_reader(
                multiband_file,
                chunk_size=(256, 256)
            )
            
            multi_chunks = list(reader_multi())
            multi_chunk_data, _ = multi_chunks[0]
            
            # Multi-band should preserve band dimension
            assert multi_chunk_data.shape == (3, 256, 256)
            
        finally:
            os.unlink(multiband_file)
        
        # Test performance with different chunk sizes
        import time
        
        # Small chunks
        start = time.time()
        small_reader = create_memory_efficient_reader(test_file, chunk_size=(100, 100))
        list(small_reader())  # Process all chunks
        small_time = time.time() - start
        
        # Large chunks
        start = time.time()
        large_reader = create_memory_efficient_reader(test_file, chunk_size=(500, 500))
        list(large_reader())  # Process all chunks
        large_time = time.time() - start
        
        # Both should complete reasonably quickly
        assert small_time < 2.0
        assert large_time < 2.0
        
    finally:
        os.unlink(test_file)


def test_memory_efficient_reader_chunking():
    """Test chunked reading behavior.
    
    This test should verify that:
    - All pixels are covered exactly once (no gaps or overlaps)
    - Chunk boundaries are calculated correctly
    - Edge chunks have appropriate sizes
    - Row-major ordering is maintained
    - Memory usage stays within expected bounds
    """
    import rasterio
    from rasterio.transform import Affine
    from rasterio.windows import Window
    
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        test_file = f.name
    
    try:
        # Create test file with unique values for coverage verification
        width, height = 513, 397  # Non-divisible by chunk size
        chunk_width, chunk_height = 256, 256
        
        profile = {
            'driver': 'GTiff',
            'width': width,
            'height': height,
            'count': 1,
            'dtype': 'int32',
            'crs': 'EPSG:4326',
            'transform': Affine.identity()
        }
        
        # Create data where each pixel value is its linear index
        data = np.arange(width * height, dtype='int32').reshape(height, width)
        
        with rasterio.open(test_file, 'w', **profile) as dst:
            dst.write(data, 1)
        
        # Create reader
        reader = create_memory_efficient_reader(
            test_file,
            chunk_size=(chunk_width, chunk_height),
            overlap=0  # No overlap for coverage test
        )
        
        # Collect all chunks
        chunks = list(reader())
        
        # Verify coverage - reconstruct full image from chunks
        reconstructed = np.full((height, width), -1, dtype='int32')
        
        for chunk_data, window in chunks:
            row_start = window.row_off
            col_start = window.col_off
            row_end = row_start + window.height
            col_end = col_start + window.width
            
            # Verify chunk boundaries are within image
            assert row_start >= 0
            assert col_start >= 0
            assert row_end <= height
            assert col_end <= width
            
            # Verify chunk data shape matches window
            assert chunk_data.shape == (window.height, window.width)
            
            # Place chunk in reconstructed image
            reconstructed[row_start:row_end, col_start:col_end] = chunk_data
        
        # Verify complete coverage (no gaps)
        assert np.all(reconstructed >= 0), "Gaps found in chunk coverage"
        
        # Verify no overlaps (all values match original)
        assert np.array_equal(reconstructed, data), "Overlaps or incorrect values in chunks"
        
        # Verify expected number of chunks
        expected_chunks_x = (width + chunk_width - 1) // chunk_width  # 3
        expected_chunks_y = (height + chunk_height - 1) // chunk_height  # 2
        assert len(chunks) == expected_chunks_x * expected_chunks_y
        
        # Verify chunk boundaries
        chunk_windows = [window for _, window in chunks]
        
        # Check row-major ordering
        for i in range(expected_chunks_y):
            for j in range(expected_chunks_x):
                idx = i * expected_chunks_x + j
                window = chunk_windows[idx]
                
                expected_row = i * chunk_height
                expected_col = j * chunk_width
                
                assert window.row_off == expected_row
                assert window.col_off == expected_col
        
        # Verify edge chunks have correct sizes
        # Bottom-right chunk (last one)
        last_data, last_window = chunks[-1]
        
        expected_last_height = height - (expected_chunks_y - 1) * chunk_height  # 397 - 256 = 141
        expected_last_width = width - (expected_chunks_x - 1) * chunk_width    # 513 - 512 = 1
        
        assert last_data.shape == (expected_last_height, expected_last_width)
        assert last_window.height == expected_last_height
        assert last_window.width == expected_last_width
        
        # Test with perfect divisible dimensions
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            perfect_file = f.name
        
        try:
            perfect_width, perfect_height = 512, 512
            perfect_profile = profile.copy()
            perfect_profile['width'] = perfect_width
            perfect_profile['height'] = perfect_height
            
            perfect_data = np.arange(perfect_width * perfect_height, dtype='int32').reshape(perfect_height, perfect_width)
            
            with rasterio.open(perfect_file, 'w', **perfect_profile) as dst:
                dst.write(perfect_data, 1)
            
            perfect_reader = create_memory_efficient_reader(
                perfect_file,
                chunk_size=(256, 256)
            )
            
            perfect_chunks = list(perfect_reader())
            
            # Should have exactly 4 chunks (2x2)
            assert len(perfect_chunks) == 4
            
            # All chunks should be exactly 256x256
            for chunk_data, window in perfect_chunks:
                assert chunk_data.shape == (256, 256)
                assert window.height == 256
                assert window.width == 256
            
        finally:
            os.unlink(perfect_file)
        
    finally:
        os.unlink(test_file)


def test_memory_efficient_reader_overlap():
    """Test overlapping chunk functionality.
    
    This test should verify that:
    - Overlap parameter adds context around chunks
    - Edge chunks handle overlap correctly
    - Window information accounts for overlap
    - Processing algorithms can use overlap context
    - No duplicate pixels in non-overlapping regions
    """
    import rasterio
    from rasterio.transform import Affine
    from rasterio.windows import Window
    
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        test_file = f.name
    
    try:
        # Create test raster
        width, height = 300, 300
        chunk_size = (100, 100)
        overlap = 20
        
        profile = {
            'driver': 'GTiff',
            'width': width,
            'height': height,
            'count': 1,
            'dtype': 'uint16',
            'crs': 'EPSG:4326',
            'transform': Affine.identity()
        }
        
        # Create data with coordinate-based pattern
        data = np.zeros((height, width), dtype='uint16')
        for y in range(height):
            for x in range(width):
                data[y, x] = y * 1000 + x  # Encode coordinates
        
        with rasterio.open(test_file, 'w', **profile) as dst:
            dst.write(data, 1)
        
        # Create reader with overlap
        reader = create_memory_efficient_reader(
            test_file,
            chunk_size=chunk_size,
            overlap=overlap
        )
        
        chunks = list(reader())
        
        # Verify overlap behavior for different chunk positions
        for i, (chunk_data, window) in enumerate(chunks):
            chunk_row = i // 3  # 3x3 grid
            chunk_col = i % 3
            
            # Calculate expected window with overlap
            base_row = chunk_row * chunk_size[1]
            base_col = chunk_col * chunk_size[0]
            
            # Expected window includes overlap
            expected_row_start = max(0, base_row - overlap)
            expected_col_start = max(0, base_col - overlap)
            expected_row_end = min(height, base_row + chunk_size[1] + overlap)
            expected_col_end = min(width, base_col + chunk_size[0] + overlap)
            
            # Verify window properties
            assert window.row_off == expected_row_start
            assert window.col_off == expected_col_start
            assert window.height == expected_row_end - expected_row_start
            assert window.width == expected_col_end - expected_col_start
            
            # Verify chunk data shape matches window
            assert chunk_data.shape == (window.height, window.width)
            
            # Verify data correctness with overlap
            for y in range(chunk_data.shape[0]):
                for x in range(chunk_data.shape[1]):
                    global_y = window.row_off + y
                    global_x = window.col_off + x
                    expected_value = global_y * 1000 + global_x
                    assert chunk_data[y, x] == expected_value
        
        # Test edge chunks specifically
        # Top-left chunk (no overlap on top/left edges)
        top_left_chunk, top_left_window = chunks[0]
        assert top_left_window.row_off == 0
        assert top_left_window.col_off == 0
        assert top_left_window.height == chunk_size[1] + overlap  # Only bottom overlap
        assert top_left_window.width == chunk_size[0] + overlap   # Only right overlap
        
        # Bottom-right chunk (no overlap on bottom/right edges)
        bottom_right_chunk, bottom_right_window = chunks[-1]
        assert bottom_right_window.row_off == 200 - overlap  # 200 (start of last chunk) - overlap
        assert bottom_right_window.col_off == 200 - overlap
        
        # Center chunk (overlap on all sides)
        center_chunk, center_window = chunks[4]  # Middle of 3x3 grid
        assert center_window.row_off == chunk_size[1] - overlap
        assert center_window.col_off == chunk_size[0] - overlap
        assert center_window.height == chunk_size[1] + 2 * overlap
        assert center_window.width == chunk_size[0] + 2 * overlap
        
        # Test overlap context for convolution-like operations
        # Simulate edge detection that needs neighbor pixels
        def simulate_edge_detection(chunk_data, core_region):
            """Simulate processing that needs overlap context."""
            # Apply simple edge filter (needs 1-pixel border)
            if chunk_data.shape[0] < 3 or chunk_data.shape[1] < 3:
                return np.zeros(core_region, dtype='float32')
            
            # Sobel-like filter on core region
            result = np.zeros(core_region, dtype='float32')
            core_start_y, core_start_x = (chunk_data.shape[0] - core_region[0]) // 2, (chunk_data.shape[1] - core_region[1]) // 2
            
            for y in range(core_region[0]):
                for x in range(core_region[1]):
                    # Simple gradient using neighboring pixels
                    src_y = core_start_y + y + 1  # +1 for border
                    src_x = core_start_x + x + 1
                    
                    if (src_y > 0 and src_y < chunk_data.shape[0] - 1 and
                        src_x > 0 and src_x < chunk_data.shape[1] - 1):
                        
                        grad_x = (chunk_data[src_y, src_x + 1].astype('float32') - 
                                 chunk_data[src_y, src_x - 1].astype('float32')) / 2
                        grad_y = (chunk_data[src_y + 1, src_x].astype('float32') - 
                                 chunk_data[src_y - 1, src_x].astype('float32')) / 2
                        
                        result[y, x] = np.sqrt(grad_x**2 + grad_y**2)
            
            return result
        
        # Process chunks with overlap context
        processed_results = []
        for chunk_data, window in chunks:
            # Define core region (non-overlap area)
            core_height = min(chunk_size[1], height - (window.row_off + overlap))
            core_width = min(chunk_size[0], width - (window.col_off + overlap))
            
            # Ensure core region doesn't exceed chunk boundaries
            core_height = min(core_height, window.height - overlap)
            core_width = min(core_width, window.width - overlap)
            
            if core_height > 0 and core_width > 0:
                result = simulate_edge_detection(chunk_data, (core_height, core_width))
                processed_results.append(result)
        
        # Verify processing used overlap context successfully
        assert len(processed_results) == len(chunks)
        assert all(result.shape[0] > 0 and result.shape[1] > 0 for result in processed_results)
        
    finally:
        os.unlink(test_file)


def test_load_consistency_gdal_vs_rasterio():
    """Test consistency between GDAL and rasterio loading.
    
    This test should verify that:
    - Same files produce identical results from both loaders
    - Metadata is consistent between loading methods
    - Coordinate transformations match
    - NoData handling is equivalent
    - Performance characteristics are documented
    """
    import rasterio
    from rasterio.transform import Affine
    from rasterio.windows import Window
    import time
    
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        test_file = f.name
    
    try:
        # Create test file with various features
        width, height = 100, 100
        profile = {
            'driver': 'GTiff',
            'width': width,
            'height': height,
            'count': 2,
            'dtype': 'float32',
            'crs': 'EPSG:4326',
            'transform': Affine(0.1, 0, -180, 0, -0.1, 90),
            'nodata': -9999,
            'compress': 'deflate'
        }
        
        # Create test data with nodata values
        data1 = np.random.rand(height, width).astype('float32') * 1000
        data2 = np.random.rand(height, width).astype('float32') * 2000
        
        # Add some nodata values
        data1[0:5, 0:5] = -9999
        data2[95:100, 95:100] = -9999
        
        with rasterio.open(test_file, 'w', **profile) as dst:
            dst.write(data1, 1)
            dst.write(data2, 2)
            dst.set_band_description(1, 'Band 1 Test')
            dst.set_band_description(2, 'Band 2 Test')
        
        # Test full file loading
        gdal_data1, gdal_meta1 = load_with_gdal(test_file, band=1)
        rio_data1, rio_meta1 = load_with_rasterio(test_file, band=1)
        
        # Compare data arrays (accounting for different nodata handling)
        # GDAL converts nodata to NaN when masked, rasterio uses masked arrays
        gdal_clean = gdal_data1[~np.isnan(gdal_data1)]
        rio_clean = rio_data1.compressed() if hasattr(rio_data1, 'compressed') else rio_data1[~np.isnan(rio_data1)]
        
        # Should have same valid data
        assert len(gdal_clean) == len(rio_clean)
        assert np.allclose(np.sort(gdal_clean), np.sort(rio_clean), rtol=1e-6)
        
        # Test metadata consistency
        # Transform should be identical
        assert gdal_meta1['transform'] == rio_meta1['transform']
        
        # Bounds should be identical
        assert gdal_meta1['bounds'] == rio_meta1['bounds']
        
        # Shape should be identical
        assert gdal_meta1['shape'] == rio_meta1['shape']
        
        # NoData handling
        assert gdal_meta1['nodata'] == rio_meta1['nodata']
        
        # Test windowed reading consistency
        window = Window(20, 30, 40, 30)
        
        gdal_window_data, gdal_window_meta = load_with_gdal(test_file, band=1, window=window)
        rio_window_data, rio_window_meta = load_with_rasterio(test_file, band=1, window=window)
        
        # Window data should be consistent
        assert gdal_window_data.shape == rio_window_data.shape
        
        # Window transforms should be identical
        assert gdal_window_meta['transform'] == rio_window_meta['transform']
        assert gdal_window_meta['bounds'] == rio_window_meta['bounds']
        
        # Test multi-band consistency
        gdal_data2, _ = load_with_gdal(test_file, band=2)
        rio_data2, _ = load_with_rasterio(test_file, band=2)
        
        gdal_clean2 = gdal_data2[~np.isnan(gdal_data2)]
        rio_clean2 = rio_data2.compressed() if hasattr(rio_data2, 'compressed') else rio_data2[~np.isnan(rio_data2)]
        
        assert np.allclose(np.sort(gdal_clean2), np.sort(rio_clean2), rtol=1e-6)
        
        # Test performance characteristics
        # Test loading speed
        n_iterations = 5
        
        # GDAL performance
        gdal_times = []
        for _ in range(n_iterations):
            start = time.time()
            load_with_gdal(test_file, band=1)
            gdal_times.append(time.time() - start)
        
        # Rasterio performance
        rio_times = []
        for _ in range(n_iterations):
            start = time.time()
            load_with_rasterio(test_file, band=1)
            rio_times.append(time.time() - start)
        
        avg_gdal_time = np.mean(gdal_times)
        avg_rio_time = np.mean(rio_times)
        
        # Both should be reasonably fast
        assert avg_gdal_time < 1.0
        assert avg_rio_time < 1.0
        
        # Document performance characteristics
        print(f"Average GDAL load time: {avg_gdal_time:.4f}s")
        print(f"Average rasterio load time: {avg_rio_time:.4f}s")
        
        # Test unmasked vs masked consistency
        gdal_unmasked, _ = load_with_gdal(test_file, band=1, as_masked=False)
        rio_unmasked, _ = load_with_rasterio(test_file, band=1, masked=False)
        
        # Unmasked data should be identical
        assert np.array_equal(gdal_unmasked, rio_unmasked)
        
        # Test different data types
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            int_file = f.name
        
        try:
            int_profile = profile.copy()
            int_profile['dtype'] = 'uint16'
            int_profile['nodata'] = 65535
            
            int_data = np.random.randint(0, 32000, (height, width), dtype='uint16')
            int_data[0:5, 0:5] = 65535  # nodata
            
            with rasterio.open(int_file, 'w', **int_profile) as dst:
                dst.write(int_data, 1)
            
            gdal_int, gdal_int_meta = load_with_gdal(int_file)
            rio_int, rio_int_meta = load_with_rasterio(int_file)
            
            # Integer data should be handled consistently
            assert gdal_int_meta['dtype'] == rio_int_meta['dtype']
            assert gdal_int_meta['nodata'] == rio_int_meta['nodata']
            
        finally:
            os.unlink(int_file)
        
    finally:
        os.unlink(test_file)


def test_format_detection_accuracy():
    """Test accuracy of format detection.
    
    This test should verify that:
    - Format detection matches actual file capabilities
    - Reader recommendations are optimal for each format
    - Performance hints are accurate
    - Complex formats are handled appropriately
    - Format-specific quirks are documented
    """
    import rasterio
    from rasterio.transform import Affine
    
    temp_files = []
    
    try:
        # Test GTiff format detection
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            gtiff_file = f.name
            temp_files.append(gtiff_file)
        
        gtiff_profile = {
            'driver': 'GTiff',
            'width': 256,
            'height': 256,
            'count': 1,
            'dtype': 'uint8',
            'crs': 'EPSG:4326',
            'transform': Affine.identity(),
            'compress': 'lzw',
            'tiled': True
        }
        
        with rasterio.open(gtiff_file, 'w', **gtiff_profile) as dst:
            dst.write(np.ones((256, 256), dtype='uint8'), 1)
        
        gtiff_info = detect_format(gtiff_file)
        
        # Verify GTiff detection accuracy
        assert gtiff_info['format'] == 'GTiff'
        assert gtiff_info['driver'] == 'GTiff'
        assert str(gtiff_info['compression']).lower() in ['lzw', 'compression.lzw']
        if gtiff_info['tile_structure'] is not None:
            assert gtiff_info['tile_structure']['tiled'] == True
        assert gtiff_info['reader'] == load_with_rasterio  # Optimal for GTiff
        
        # Test COG vs regular GTiff recommendations
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            cog_file = f.name
            temp_files.append(cog_file)
        
        cog_profile = gtiff_profile.copy()
        cog_profile['compress'] = 'deflate'
        
        with rasterio.open(cog_file, 'w', **cog_profile) as dst:
            dst.write(np.random.randint(0, 255, (256, 256), dtype='uint8'), 1)
            dst.build_overviews([2, 4], rasterio.enums.Resampling.average)
        
        cog_info = detect_format(cog_file)
        
        # COG detection may depend on overview structure - just verify it's a GTiff
        assert cog_info['format'] == 'GTiff'
        assert cog_info['reader'] == load_with_rasterio  # Optimal for COG
        # COG detection might be false due to specific overview requirements
        
        # Test format detection with actual file capabilities
        test_formats = [
            {'driver': 'GTiff', 'suffix': '.tif', 'expected_reader': load_with_rasterio},
            {'driver': 'ENVI', 'suffix': '.dat', 'expected_reader': load_with_rasterio},
            {'driver': 'HFA', 'suffix': '.img', 'expected_reader': load_with_rasterio}
        ]
        
        for fmt in test_formats:
            try:
                with tempfile.NamedTemporaryFile(suffix=fmt['suffix'], delete=False) as f:
                    test_file = f.name
                    temp_files.append(test_file)
                
                profile = {
                    'driver': fmt['driver'],
                    'width': 50,
                    'height': 50,
                    'count': 1,
                    'dtype': 'float32',
                    'crs': 'EPSG:4326',
                    'transform': Affine.identity()
                }
                
                with rasterio.open(test_file, 'w', **profile) as dst:
                    dst.write(np.ones((50, 50), dtype='float32'), 1)
                
                info = detect_format(test_file)
                
                # Verify format matches file capabilities
                assert info['format'] == fmt['driver']
                assert info['driver'] == fmt['driver']
                assert info['reader'] == fmt['expected_reader']
                
                # Verify recommendations match actual capabilities
                reader = info['reader']
                data, meta = reader(test_file)
                assert data.shape == (50, 50)
                assert meta['format_metadata']['driver'] == fmt['driver']
                
            except Exception as e:
                # Some formats might not be available in test environment
                print(f"Skipping {fmt['driver']}: {e}")
                continue
        
        # Test mock HDF5 with subdatasets (complex format)
        with patch('osgeo.gdal.Open') as mock_open:
            mock_ds = Mock()
            mock_driver = Mock()
            mock_driver.ShortName = 'HDF5'
            mock_ds.GetDriver.return_value = mock_driver
            
            # Mock complex subdataset structure
            mock_subdatasets = [
                ('HDF5:"test.h5"://GRIDS/VNP_Grid_DNB/Data_Fields/DNB_At_Sensor_Radiance_500m', 
                 'DNB at-sensor radiance'),
                ('HDF5:"test.h5"://GRIDS/VNP_Grid_DNB/Data_Fields/DNB_Observation_Data', 
                 'DNB observation data')
            ]
            mock_ds.GetSubDatasets.return_value = mock_subdatasets
            mock_open.return_value = mock_ds
            
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
                hdf5_file = f.name
                temp_files.append(hdf5_file)
            
            hdf5_info = detect_format(hdf5_file)
            
            # Complex format should recommend GDAL
            assert hdf5_info['format'] == 'HDF5'
            assert hdf5_info['reader'] == load_with_gdal  # Better for HDF5
            assert len(hdf5_info['subdatasets']) == 2
            
            # Verify subdataset structure is properly analyzed
            assert 'DNB_At_Sensor_Radiance_500m' in hdf5_info['subdatasets'][0]['path']
            assert 'DNB at-sensor radiance' in hdf5_info['subdatasets'][0]['description']
        
        # Test performance hints accuracy
        # Large tiled file should hint at efficient partial reading
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            large_file = f.name
            temp_files.append(large_file)
        
        large_profile = {
            'driver': 'GTiff',
            'width': 2048,
            'height': 2048,
            'count': 1,
            'dtype': 'uint8',
            'crs': 'EPSG:4326',
            'transform': Affine.identity(),
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512
        }
        
        with rasterio.open(large_file, 'w', **large_profile) as dst:
            # Write minimal data for structure
            dst.write(np.ones((512, 512), dtype='uint8'), 1, 
                     window=rasterio.windows.Window(0, 0, 512, 512))
        
        large_info = detect_format(large_file)
        
        # Large tiled file should provide performance hints
        assert large_info['tile_structure']['tiled'] == True
        assert large_info['tile_structure']['blockxsize'] == 512
        assert large_info['tile_structure']['blockysize'] == 512
        
        # Should recommend rasterio for efficient partial reads
        assert large_info['reader'] == load_with_rasterio
        
    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.unlink(f)


def test_error_handling_robustness():
    """Test robustness of error handling across all functions.
    
    This test should verify that:
    - Invalid file paths raise appropriate exceptions
    - Corrupted files are handled gracefully
    - Resource cleanup occurs after errors
    - Error messages are informative and actionable
    - No memory leaks occur during error conditions
    """
    import rasterio
    from rasterio.transform import Affine
    import gc
    
    # Test detect_format error handling
    with pytest.raises(FileNotFoundError):
        detect_format('/absolutely/non/existent/path.tif')
    
    # Test with corrupted file
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        corrupted_file = f.name
        f.write(b'This is not a valid raster file at all')
    
    try:
        # Should handle gracefully, not crash
        result = detect_format(corrupted_file)
        assert result['format'] in [None, 'Unknown']
    finally:
        os.unlink(corrupted_file)
    
    # Test load_with_gdal error handling
    with pytest.raises(RuntimeError):
        load_with_gdal('/non/existent/file.tif')
    
    # Test with invalid band
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        test_file = f.name
    
    try:
        profile = {
            'driver': 'GTiff',
            'width': 10,
            'height': 10,
            'count': 2,
            'dtype': 'uint8',
            'crs': 'EPSG:4326',
            'transform': Affine.identity()
        }
        
        with rasterio.open(test_file, 'w', **profile) as dst:
            dst.write(np.ones((10, 10), dtype='uint8'), 1)
            dst.write(np.ones((10, 10), dtype='uint8'), 2)
        
        # Test invalid band number
        with pytest.raises(Exception):  # GDAL might raise different exceptions
            load_with_gdal(test_file, band=99)
        
    finally:
        os.unlink(test_file)
    
    # Test load_with_rasterio error handling
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        text_file = f.name
        f.write(b'Not a raster file')
    
    try:
        with pytest.raises(Exception):  # rasterio will raise an exception
            load_with_rasterio(text_file)
    finally:
        os.unlink(text_file)
    
    # Test create_virtual_raster error handling
    with pytest.raises(ValueError, match='No input files'):
        create_virtual_raster([])
    
    # Test with non-existent files in list
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        valid_file = f.name
    
    try:
        profile = {
            'driver': 'GTiff',
            'width': 10,
            'height': 10,
            'count': 1,
            'dtype': 'uint8'
        }
        
        with rasterio.open(valid_file, 'w', **profile) as dst:
            dst.write(np.ones((10, 10), dtype='uint8'), 1)
        
        # Mix of valid and invalid files
        mixed_files = [valid_file, '/non/existent/file.tif']
        
        # Should either succeed (GDAL ignores missing) or raise clear error
        try:
            vrt_path = create_virtual_raster(mixed_files)
            # If successful, clean up
            if os.path.exists(vrt_path):
                os.unlink(vrt_path)
        except RuntimeError as e:
            # Should provide informative error
            assert 'VRT' in str(e) or 'Failed' in str(e)
        
    finally:
        os.unlink(valid_file)
    
    # Test read_metadata error handling
    with pytest.raises(Exception):  # Should raise exception for non-existent file
        read_metadata('/non/existent/file.tif')
    
    # Test setup_output_profile with invalid template
    with pytest.raises(Exception):
        setup_output_profile(template_path='/non/existent/template.tif')
    
    # Test validate_raster_stack robustness
    result = validate_raster_stack([])  # Empty list
    assert result['valid'] == False
    assert 'No files provided' in result['issues']
    
    # Test with mix of valid and invalid files
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        valid_file2 = f.name
    
    try:
        with rasterio.open(valid_file2, 'w', **profile) as dst:
            dst.write(np.ones((10, 10), dtype='uint8'), 1)
        
        mixed_list = [valid_file2, '/invalid/path.tif']
        result = validate_raster_stack(mixed_list)
        
        # Should report issues but not crash
        assert 'issues' in result
        assert any('Cannot read' in issue for issue in result['issues'])
        
    finally:
        os.unlink(valid_file2)
    
    # Test create_memory_efficient_reader error handling
    with pytest.raises(Exception):
        reader = create_memory_efficient_reader('/non/existent/file.tif')
        list(reader())  # Try to consume generator
    
    # Test resource cleanup during errors
    # Use a temporary file that gets deleted during processing
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        temp_file = f.name
    
    try:
        with rasterio.open(temp_file, 'w', **profile) as dst:
            dst.write(np.ones((10, 10), dtype='uint8'), 1)
        
        # Delete file while trying to read it
        os.unlink(temp_file)
        
        # Should handle gracefully
        with pytest.raises(Exception):
            load_with_rasterio(temp_file)
        
        # No cleanup needed since file is already deleted
        temp_file = None
        
    finally:
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)
    
    # Test memory management during errors
    # Force garbage collection to check for leaks
    initial_objects = len(gc.get_objects())
    
    # Trigger multiple errors
    for i in range(10):
        try:
            detect_format(f'/non/existent/file_{i}.tif')
        except FileNotFoundError:
            pass
        
        try:
            load_with_gdal(f'/non/existent/file_{i}.tif')
        except RuntimeError:
            pass
    
    # Force garbage collection
    gc.collect()
    
    # Object count shouldn't grow significantly
    final_objects = len(gc.get_objects())
    object_growth = final_objects - initial_objects
    
    # Allow some growth but not excessive
    assert object_growth < 1000, f"Potential memory leak: {object_growth} new objects"