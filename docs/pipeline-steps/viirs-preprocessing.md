# VIIRS Preprocessing

This step prepares VIIRS nighttime lights data for integration with Landsat imagery.

## Overview

VIIRS (Visible Infrared Imaging Radiometer Suite) provides nighttime lights data at ~500m resolution. We preprocess it to match a 30m resolution grid in a locally-optimized Albers Equal Area projection for accurate area calculations and consistent spatial analysis.

## Processing Steps

### 1. Data Acquisition and Mosaicking

**Function**: `blackmarble.acquire.viirs` + `gdal.BuildVRT()`

**Purpose**: Download VIIRS gap-filled NTL data and create mosaic if multiple tiles

**Key Points**:
- Uses NASA's gap-filled product (VNP46A2) that already incorporates quality filtering
- Lunar irradiance is already removed in VNP46A2 (pipeline does NOT subtract it)
- Data is in DN units with 0.1 scale factor (converted to radiance later in NDUI calculation)
- Creates VRT mosaic for multiple tiles
- Fill value: -999.9 (handled during mosaicking)

**Implementation**:
```python
if len(viirs_ntl_files) > 1:
    vrt_options = gdal.BuildVRTOptions(
        separate=False,  # Mosaic tiles together
        resolution='highest'  # Use highest resolution from input files
    )
    gdal.BuildVRT(viirs_ntl_vrt, [str(f) for f in viirs_ntl_files], options=vrt_options)
```

### 2. Crop to Bounds

**Function**: `blackmarble.prepare.spatial.crop_to_bounds()`

**Purpose**: Reduce data volume by cropping to area of interest

**Parameters**:
- `bounds`: Target bounding box in WGS84
- `buffer_distance`: Optional buffer around bounds

**Implementation**:
```python
viirs_ntl, viirs_transform, viirs_bounds = prepare.crop_to_bounds(
    data=viirs_ntl,
    src_transform=viirs_profile["transform"],
    bounds=bbox
)
```

### 3. Reprojection to Processing CRS {#reprojection}

**Function**: `blackmarble.prepare.spatial.reproject_image()`

**Purpose**: Transform from geographic (WGS84) to locally-optimized Albers Equal Area projection

**Key Decisions**:
- **Target CRS**: Locally-optimized Albers Equal Area
  - Created by `get_processing_crs()` based on bbox centroid
  - Rationale: Minimize distortion, enable accurate area calculations
  - Resolution: 30m default (can override via config.preparation.target_resolution)
- **Resampling During Reprojection**: Bilinear
  - Maintains smooth radiance gradients

**Implementation**:
```python
# Get processing CRS and grid
processing_crs = get_processing_crs(bbox)
processing_crs, reference_transform, reference_shape = create_processing_grid(bbox, 30.0)

# Reproject VIIRS to processing CRS
viirs_reprojected, viirs_transform, viirs_bounds = prepare.reproject_image(
    data=viirs_corrected,
    src_transform=viirs_profile["transform"],
    src_crs=viirs_profile["crs"],
    dst_crs=processing_crs,
    dst_transform=reference_transform,
    dst_shape=reference_shape,
    resampling="bilinear"
)
```

### 4. Grid Alignment

**Purpose**: Ensure VIIRS aligns perfectly with the processing grid

**Key Points**:
- Uses pre-computed reference grid from `create_processing_grid()`
- All rasters are aligned to this common 30m grid
- No additional resampling needed - alignment happens during reprojection

**Note**: The pipeline no longer resamples to 12m. Instead, it uses 30m resolution throughout for consistency with Landsat's native resolution.

## Test Coverage

**Primary Test**: `tests/acquire/test_viirs_orientation.py`
- Verifies VIIRS data loading and orientation
- Checks coordinate transformations

**Pipeline Integration**: 
- Full pipeline test in `tests/pipeline/` verifies end-to-end processing
- Includes cropping, reprojection, and alignment

## Common Issues

1. **Memory Usage**: Processing large areas at 30m resolution
   - Solution: Use VRT mosaics to handle tiles efficiently
   - Pipeline automatically creates VRTs for multiple tiles

2. **Nodata Handling**: VIIRS uses fill value of -999.9
   - Solution: Gap-filled product handles this automatically
   - Zeros are preserved as valid data (no lights)

3. **CRS Selection**: Albers Equal Area parameters optimized per region
   - Solution: Automatic optimization based on bbox centroid
   - Ensures minimal distortion for the area of interest

4. **Multiple Tiles**: VIIRS data may span multiple tiles
   - Solution: Automatic VRT mosaicking handles this seamlessly

## Configuration

```python
# No configuration needed - pipeline handles automatically
# Processing CRS is determined by get_processing_crs(bbox)
# Resolution is fixed at 30m to match Landsat
```

## Performance Notes

- Cropping first reduces processing time significantly
- VRT mosaics avoid loading all tiles into memory
- Bilinear resampling preserves smooth radiance gradients
- Single reprojection step (no separate resampling)

## Related Steps

- Next: [Landsat Processing](./landsat-qa-masking.md) - Process Landsat imagery
- Uses: VIIRS gap-filled NTL data from NASA Earthdata
- Outputs to: NDUI calculation, NTL enhancement