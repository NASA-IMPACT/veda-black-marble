# Spatial Alignment

This step ensures all raster datasets share a common grid for pixel-perfect calculations.

## Overview

Spatial alignment is critical for combining datasets from different sources (VIIRS, Landsat, OSM) that have different native resolutions and grids. The pipeline ensures all data is aligned to the same 30m grid in a locally-optimized Albers Equal Area projection.

## Processing Flow

1. **Reference Grid Creation**: Processing grid is established early in pipeline
2. **Per-Dataset Reprojection**: Each dataset is reprojected to the common CRS
3. **Final Alignment**: VIIRS is aligned to Landsat's grid as the final step

### Why Align VIIRS to Landsat?

- Landsat provides the spatial framework (multiple tiles, 30m native resolution)
- VIIRS is a single, coarser dataset that needs to match Landsat's grid
- Using Landsat as reference ensures consistent spatial extent

## Algorithm

**Function**: `blackmarble.prepare.spatial.align_rasters()`

### Key Features

1. **Reference-Based Alignment**:
   - One raster serves as the reference (typically Landsat)
   - All others are reprojected to match its grid exactly
   - Ensures pixel-perfect alignment

2. **Resampling Methods**:
   - **Bilinear**: Used for continuous data (VIIRS NTL, indices)
   - **Nearest**: Used for categorical data (masks, classifications)
   - Preserves data characteristics during resampling

3. **Memory Efficiency**:
   - Processes bands individually to reduce memory usage
   - Uses rasterio's windowed operations internally

## Pipeline Implementation

```python
# From blackmarble/pipeline.py
if viirs_reprojected is not None and reference_band is not None:
    # Align VIIRS to Landsat reference grid
    aligned_rasters = prepare.align_rasters(
        rasters=[reference_band, viirs_reprojected],
        transforms=[reference_transform, viirs_transform],
        reference_idx=0,  # Use Landsat as reference
        resampling="bilinear",  # Smooth resampling for VIIRS
    )
    
    # Extract aligned VIIRS
    viirs_reprojected = aligned_rasters[1][0]
    viirs_transform = aligned_rasters[1][1]
    
    # Verify alignment success
    nonzero_pct = (viirs_reprojected > 0).sum() / viirs_reprojected.size * 100
    if nonzero_pct < 10.0:
        raise RuntimeError(f"VIIRS alignment failed - only {nonzero_pct:.1f}% non-zero pixels")
```

## Test Coverage

**Unit Tests**: `tests/prepare/test_spatial.py`
- Multi-raster alignment scenarios
- Different resampling methods
- CRS transformation handling
- Edge cases: single raster, mismatched sizes

**Integration**: Pipeline tests verify end-to-end alignment

## Configuration

```python
# No direct configuration - alignment parameters are hardcoded:
# - 30m resolution (set during grid creation)
# - Bilinear resampling for continuous data
# - Landsat as reference grid
```

## Common Issues

1. **VIIRS Coverage Warnings**
   - Cause: VIIRS tile doesn't fully cover the bounding box
   - Impact: Missing data at edges
   - Solution: Pipeline continues with available data

2. **Low Non-Zero Pixel Count**
   - Cause: Misalignment or incorrect tile selection
   - Error: "VIIRS alignment failed - only X% non-zero pixels"
   - Solution: Check VIIRS tile selection and bounds

3. **Shape Mismatches**
   - Cause: Arrays not properly aligned to reference
   - Solution: Ensure all data goes through alignment step
   - Note: Pipeline validates shapes before NDUI calculation

4. **Memory Usage**
   - Large areas may require significant memory
   - Solution: Process smaller regions or use windowed operations

## Performance Notes

- Alignment is relatively fast (< 1s for typical areas)
- Bilinear resampling adds minimal overhead
- Most time spent in reprojection calculations
- Band-by-band processing reduces peak memory usage

## Diagnostic Outputs

When alignment issues occur, the pipeline logs:
- Input and output shapes
- Non-zero pixel percentages
- Coverage warnings
- Transform parameters

## Related Steps

- Previous: [VIIRS Preprocessing](./viirs-preprocessing.md) - Reprojected VIIRS
- Previous: [Temporal Compositing](./temporal-compositing.md) - Composite indices
- Next: [NDUI Calculation](./ndui-calculation.md) - Uses aligned data
- Uses: Reprojected VIIRS and Landsat composite data
- Outputs: Pixel-aligned arrays for analysis