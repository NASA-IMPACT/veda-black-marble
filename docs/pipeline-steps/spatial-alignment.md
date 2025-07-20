# Spatial Alignment

This step ensures all raster datasets share a common grid for pixel-perfect calculations.

## Overview

Spatial alignment is critical for combining datasets from different sources (VIIRS, Landsat, OSM) that have different native resolutions and grids. This step uses the `blackmarble.prepare.spatial.align_rasters()` function to reproject and resample all inputs to a common grid.

## Processing Steps

### TODO: Document the following sections
- Reference grid selection logic
- Resampling method choices (bilinear vs nearest)
- Memory optimization strategies
- Handling of nodata values

## Key Functions

**Primary Function**: `blackmarble.prepare.spatial.align_rasters()`

**Supporting Functions**:
- `get_common_bounds()` - Calculate intersection of all raster bounds
- `create_reference_grid()` - Define target grid parameters

## Test Coverage

**Primary Test**: `tests/pipeline/test_alignment.py`
- Tests multi-raster alignment scenarios
- Verifies grid consistency
- Checks nodata preservation

## Configuration

```python
# In PipelineConfig
config.spatial.reference_resolution = 0.000111111  # 12m in degrees
config.spatial.alignment_method = "bilinear"
config.spatial.force_common_extent = True
```

## Related Steps

- Previous: [VIIRS Preprocessing](./viirs-preprocessing.md)
- Next: [Spectral Indices](./spectral-indices.md)
- Uses: All preprocessed raster data
- Outputs to: All subsequent analysis steps