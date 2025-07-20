# VIIRS Preprocessing

This step prepares VIIRS nighttime lights data for integration with Landsat imagery.

## Overview

VIIRS (Visible Infrared Imaging Radiometer Suite) provides nighttime lights data at ~500m resolution. We preprocess it to a 12m resolution grid (≈ 0.000111111°) for enhanced spatial detail in urban corridors.

## Processing Steps

### 1. Crop to Bounds

**Function**: `blackmarble.prepare.spatial.crop_to_bounds()`

**Purpose**: Reduce data volume by cropping to area of interest

**Parameters**:
- `bounds`: Target bounding box in WGS84
- `buffer_distance`: Optional buffer around bounds

**Implementation**:
```python
viirs_cropped, viirs_transform, viirs_bounds = crop_to_bounds(
    data=viirs_data,
    src_transform=viirs_profile["transform"],
    bounds=bbox
)
```

### 2. Reprojection {#reprojection}

**Function**: `blackmarble.prepare.spatial.reproject_image()`

**Purpose**: Transform from geographic (WGS84) to projected coordinate system

**Key Decisions**:
- **Target CRS**: Automatically selected based on:
  - Landsat UTM zone if available
  - Local UTM zone based on bbox centroid
  - Rationale: Minimize distortion, enable accurate distance calculations
- **Resampling During Reprojection**: Bilinear
  - Maintains smooth radiance gradients

**Implementation**:
```python
viirs_reprojected, viirs_transform, viirs_bounds = reproject_image(
    data=viirs_resampled,
    src_transform=viirs_transform,
    src_crs="EPSG:4326",
    dst_crs=target_crs,  # e.g., "EPSG:32610" for UTM Zone 10N
    resampling="bilinear"
)
```

### 3. Spatial Alignment to Landsat Grid

**Function**: `blackmarble.prepare.spatial.align_rasters()`

**Purpose**: Align VIIRS to a common 12m grid for consistent analysis

**Key Decisions**:
- **Target Resolution**: 12m (≈ 0.000111111°)
  - Rationale: Provides better spatial detail for urban areas while maintaining computational efficiency
  - Process: VIIRS is resampled from ~500m to 12m during alignment
- **Resampling Method**: Bilinear
  - Rationale: Smooth interpolation for continuous radiance values
  - Preserves nighttime light gradients

**Implementation**:
```python
# After reprojecting both to UTM
aligned_rasters = align_rasters(
    rasters=[landsat_band, viirs_reprojected],
    transforms=[landsat_transform, viirs_transform],
    reference_idx=0,  # Use Landsat as reference
    resampling="bilinear"
)
viirs_aligned = aligned_rasters[1][0]
```

**Note**: The resolution change happens during alignment, not as a separate resampling step.

## Test Coverage

**Primary Test**: `tests/pipeline/test_pipeline_flow.py::test_03_viirs_processing`
- Verifies cropping reduces data size correctly
- Checks reprojection to UTM
- Tests alignment with common grid (12m resolution)

**Additional Tests**: 
- `tests/pipeline/test_viirs_processing.py`
  - Edge cases: Empty bounds, polar regions
  - Performance: Large area processing
- `tests/pipeline/test_alignment.py`
  - Multi-raster alignment scenarios

## Common Issues

1. **Memory Usage**: Alignment to 12m can increase memory ~1700x (500m→12m)
   - Solution: Process in tiles for large areas

2. **Nodata Handling**: VIIRS uses 0 for no data
   - Solution: Preserve zeros during alignment

3. **CRS Selection**: Automatic UTM selection may cross zone boundaries
   - Solution: Allow manual CRS override in config

4. **Alignment Failure**: VIIRS and Landsat don't overlap spatially
   - Solution: Check spatial footprints before processing

## Configuration

```python
# In PipelineConfig
config.preparation.viirs_resampling = "bilinear"  # or "cubic"
config.preparation.auto_select_utm = True  # Auto-select UTM zone
config.spatial.alignment_nodata = 0.0  # Preserve VIIRS zeros
```

## Performance Notes

- Cropping first reduces processing time by 10-100x
- Bilinear resampling is 3x faster than cubic with minimal quality loss
- Alignment is the slowest step (~70% of preprocessing time)
- Reprojection before alignment reduces memory usage

## Related Steps

- Next: [Spatial Alignment](./spatial-alignment.md) - Align with Landsat
- Uses: VIIRS data from acquisition phase
- Outputs to: NDUI calculation, NTL enhancement