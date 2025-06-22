# GDAL to Rasterio Migration Guide

## Overview

This guide documents the migration from GDAL to rasterio in the Black Marble pipeline. The migration addresses several issues:
- Georeferencing metadata not being written correctly
- Need for more Pythonic API
- Better error handling and memory management
- Consistent behavior across different operations

## Migration Status

### ✅ Completed Migrations

1. **I/O Operations** (`save_geotiff_single`)
   - File: `blackmarble/io_rasterio.py`
   - Status: Complete, tested, backward compatible
   - Toggle: `export BM_USE_RASTERIO_WRITER=true`

2. **Reprojection Operations** 
   - File: `blackmarble/reproj_rasterio.py`
   - Functions: `crop_to_bb_utm`, `convert_to_wgs84`, `subset_raster_normal`
   - Status: Complete with full test coverage

3. **Mosaic Operations**
   - File: `blackmarble/mosaic_rasterio.py`
   - Functions: `mosaic_tiles`, `mosaic_viirs_tiles`, `resample_to_resolution`
   - Status: Complete with VIIRS-specific handling

4. **Landsat Processing**
   - File: `blackmarble/landsat_rasterio.py`
   - Status: Complete, replaces gdal.Warp with rasterio.merge

5. **VIIRS Processing**
   - File: `blackmarble/vnp_rasterio.py`
   - Status: Complete with HDF5 subdataset handling

## Implementation Details

### Architecture

```
blackmarble/
├── io.py                   # Original GDAL implementation
├── io_rasterio.py          # Rasterio implementation
├── io_compat.py            # Compatibility layer
├── reproj.py               # Original GDAL reprojection
├── reproj_rasterio.py      # Rasterio reprojection
├── mosaic_rasterio.py      # Rasterio mosaic operations
├── pipeline_ops.py         # Dynamic implementation switching
└── pipeline_rasterio.py    # Integration examples
```

### Feature Flags

Different components can be toggled independently:

```bash
# Enable rasterio for I/O operations
export BM_USE_RASTERIO_WRITER=true

# Enable rasterio for mosaic operations (in pipeline)
export USE_RASTERIO_MOSAIC=true

# Enable all rasterio features
export BM_USE_RASTERIO=true
```

### Key Improvements

1. **Georeferencing Reliability**
   - Rasterio always writes CRS and transform metadata correctly
   - Fixes issues with downstream reprojection operations

2. **Better Compression**
   - Default deflate compression reduces file sizes
   - Tiled GeoTIFFs for improved performance

3. **Cleaner API**
   ```python
   # GDAL approach
   ds = gdal.GetDriverByName("GTiff").Create(...)
   ds.SetProjection(projection)
   ds.SetGeoTransform(geotransform)
   band = ds.GetRasterBand(1)
   band.WriteArray(data)
   ds = None  # Manual cleanup
   
   # Rasterio approach
   with rasterio.open(path, 'w', **profile) as dst:
       dst.write(data, 1)  # Automatic cleanup
   ```

4. **Type Safety**
   - Better integration with type hints
   - More predictable error handling

## Migration Path

### Phase 1: I/O Operations (Complete)
Enable rasterio for writing GeoTIFFs:
```bash
export BM_USE_RASTERIO_WRITER=true
```

### Phase 2: Mosaic Operations (Complete)
Enable rasterio for mosaicing:
```python
# In pipeline.py
from blackmarble.pipeline_ops import mosaic_viirs_tiles
```

### Phase 3: Full Migration (Ready)
Once validated, remove GDAL dependencies:
```bash
# Future: Remove gdal from pyproject.toml
uv remove gdal
```

## Testing

### Unit Tests
```bash
# Test rasterio implementations
uv run pytest tests/test_io_rasterio.py
uv run pytest tests/test_mosaic_rasterio.py
uv run pytest tests/test_reproj_rasterio.py

# Test compatibility
uv run pytest tests/test_gdal_rasterio_compat.py
```

### Integration Tests
```bash
# Run with rasterio
export BM_USE_RASTERIO_WRITER=true
uv run pytest tests/test_pipeline_integration.py

# Run with GDAL (default)
unset BM_USE_RASTERIO_WRITER
uv run pytest tests/test_pipeline_integration.py
```

### Performance Benchmarks
```bash
uv run pytest tests/test_performance_benchmarks.py -v
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure rasterio is installed: `uv add rasterio`
   - Check GDAL system dependencies

2. **Different Output Sizes**
   - Rasterio and GDAL may differ by a few bytes due to TIFF metadata
   - Pixel data should be identical

3. **Performance Differences**
   - Small files: Rasterio has ~3-5ms Python overhead
   - Large files: Performance is comparable

### Debugging

Enable verbose output:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check which implementation is active:
```python
from blackmarble.io import save_geotiff_single
print(save_geotiff_single.__module__)
# 'blackmarble.io_rasterio' if using rasterio
# 'blackmarble.io' if using GDAL
```

## Future Work

1. **Remove GDAL Dependency**
   - Once rasterio is fully validated
   - Will reduce package size significantly

2. **Cloud Optimized GeoTIFF (COG)**
   - Native COG support in rasterio
   - Better for cloud deployment

3. **Memory Optimization**
   - Windowed reading/writing for large files
   - Reduced memory footprint

## References

- [Rasterio Documentation](https://rasterio.readthedocs.io/)
- [Migration Examples](../blackmarble/pipeline_rasterio.py)
- [Performance Benchmarks](../tests/test_performance_benchmarks.py)