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

## Pipeline Behavior

- Landsat-aligned data provides the reference grid for final VIIRS alignment.
- Continuous inputs use bilinear resampling during alignment.
- The pipeline performs sanity checks on aligned outputs to detect sparse or failed alignment.

## Practical Notes

- Resolution and grid are fixed by the pipeline (30m processing grid).
- Continuous rasters use bilinear resampling.
- Alignment warnings usually indicate sparse/partial VIIRS coverage near edges.

## Related Steps

- [VIIRS Preprocessing](./viirs-preprocessing.md)
- [Temporal Compositing](./temporal-compositing.md)
- [NDUI Calculation](./ndui-calculation.md)