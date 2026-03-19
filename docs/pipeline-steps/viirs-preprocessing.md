# VIIRS Preprocessing

This step prepares VIIRS nighttime lights data for integration with Landsat imagery.

## Overview

VIIRS (Visible Infrared Imaging Radiometer Suite) provides nighttime lights data at ~500m resolution. We preprocess it to match a 30m resolution grid in a locally-optimized Albers Equal Area projection for accurate area calculations and consistent spatial analysis.

## Processing Steps

### 1. Data Acquisition and Mosaicking

**Function**: `blackmarble.acquire.viirs` + `blackmarble.pipeline.mosaic_viirs`

**Purpose**: Download VIIRS gap-filled NTL data and mosaic tiles when multiple files are returned

**Key Points**:
- Uses NASA's gap-filled product (VNP46A2) that already incorporates quality filtering
- Lunar irradiance is already removed in VNP46A2 (pipeline does NOT subtract it)
- Data is in DN units with 0.1 scale factor (converted to radiance later in NDUI calculation)
- Mosaicking is handled in the pipeline when multiple tiles are present
- Fill value: -999.9 is preserved and handled as nodata in downstream processing

### 2. Crop to Bounds

**Function**: `blackmarble.prepare.spatial.crop_to_bounds()`

**Purpose**: Reduce data volume by cropping to area of interest

**Parameters**:
- `bounds`: Target bounding box in WGS84
- `buffer_distance`: Optional buffer around bounds

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
  - Solution: Mosaic multi-tile inputs before downstream processing
  - Pipeline handles tile combination automatically

2. **Nodata Handling**: VIIRS uses fill value of -999.9
   - Solution: Gap-filled product handles this automatically
   - Zeros are preserved as valid data (no lights)

3. **CRS Selection**: Albers Equal Area parameters optimized per region
   - Solution: Automatic optimization based on bbox centroid
   - Ensures minimal distortion for the area of interest

4. **Multiple Tiles**: VIIRS data may span multiple tiles
  - Solution: Automatic mosaicking combines tiles before downstream processing

## Configuration

No manual configuration is required. The pipeline determines processing CRS from the area of interest and uses 30m resolution to match Landsat.

## Performance Notes

- Cropping first reduces processing time significantly
- Mosaicking handles multi-tile scenes for downstream processing
- Bilinear resampling preserves smooth radiance gradients
- Single reprojection step (no separate resampling)

## Related Steps

- Next: [Landsat Processing](./landsat-qa-masking.md) - Process Landsat imagery
- Uses: VIIRS gap-filled NTL data from NASA Earthdata
- Outputs to: NDUI calculation, NTL enhancement