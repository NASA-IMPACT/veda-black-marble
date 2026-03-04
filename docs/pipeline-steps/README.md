# Black Marble Pipeline Steps

This directory documents every step in the Black Marble processing pipeline, from data acquisition through final export. Each processing step that involves decisions or complex logic has its own detailed documentation.

## Pipeline Overview

The Black Marble pipeline processes nighttime lights (VIIRS), daytime imagery (Landsat), and road networks (OSM) to create enhanced urban nighttime visualizations.

## Pipeline Steps

### 1. Data Acquisition (I/O)
- Download VIIRS nighttime lights from NASA Earthdata
- Download Landsat Collection 2 imagery from USGS
- Download OpenStreetMap road networks
- *No processing decisions - pure data fetching*

### 2. VIIRS Processing
- Load VIIRS HDF5 and convert to GeoTIFF
- **[Crop and Resample VIIRS](./viirs-preprocessing.md)** ⚙️
  - Decisions: Target resolution (30m), resampling method
- **[Reproject VIIRS](./viirs-preprocessing.md#reprojection)** ⚙️
  - Decisions: Target CRS selection

### 3. Landsat Processing  
- **[Scene Search and Selection](./landsat-search.md)** ⚙️
  - Decisions: Time window (±15 days), quality weights, tile coverage
- Load Landsat bands (B3, B4, B5, QA_PIXEL)
- **[Cloud Masking](./landsat-qa-masking.md)** ⚙️
  - Decisions: QA strategy (conservative/moderate/permissive), dilation
- **[Temporal Compositing](./temporal-compositing.md)** ⚙️
  - Decisions: Composite methods (NDVI 85th percentile, NDWI median)
- Apply scaling factors (0.0000275, -0.2)
- Reproject to target CRS

### 4. Spatial Alignment
- **[Align All Rasters](./spatial-alignment.md)** ⚙️
  - Decisions: Reference raster selection, resampling method

### 5. Road Processing
- Load OSM road network
- **Fractional Road Rasterization** ⚙️
  - Decisions: Buffer size, sub-pixel grid resolution

### 6. Index Calculations
- **[Calculate NDVI/NDWI](./spectral-indices.md)** ⚙️
  - Decisions: Epsilon handling for near-zero denominators
- **[Calculate NDUI](./ndui-calculation.md)** ⚙️
  - Decisions: Water masking, parameter tuning

### 7. Urban Enhancement
- **[Compute Urban Fields](./urban-field-enhancement.md)** ⚙️
  - Decisions: Multi-scale parameters, weighting
- **[Enhance NTL](./urban-field-enhancement.md#ntl-enhancement)** ⚙️
  - Decisions: Enhancement mode (current default: multiplicative)

### 8. Visualization
- **Contrast Enhancement** ⚙️
  - Decisions: Method (percentile/minmax), clipping range
- **Colormap Application** ⚙️
  - Decisions: Colormap choice, scaling range

### 9. Export
- **Create COG** ⚙️
  - Decisions: Compression, tiling, overviews
- Generate metadata
- Optional: Export auxiliary products (urban mask, indices)

## Legend

- ⚙️ **Processing Step** - Click for detailed documentation including:
  - Algorithm description
  - Decision points and parameters
  - Test coverage references
  - Code examples

## Quick Navigation

| Phase | Key Processing Steps | Primary Decisions |
|-------|---------------------|-------------------|
| Preprocessing | [VIIRS Preprocessing](./viirs-preprocessing.md), [QA Masking](./landsat-qa-masking.md) | Resolution, QA strategy |
| Analysis | [Spectral Indices](./spectral-indices.md), [NDUI](./ndui-calculation.md) | Epsilon handling, water masking |
| Enhancement | [Urban Fields](./urban-field-enhancement.md) | Scale parameters, enhancement mode |
| Visualization | Contrast Enhancement, Colormap Application | Percentiles, color range |

## Running Tests

Run tests from the repository root:

```bash
# Run all tests
uv run pytest

# Run pipeline statistics tests
uv run pytest tests/pipeline/test_statistics.py -v
```

## Reference Data

Test data is committed to the repository in `tests/data/reference/` including:
- VIIRS nighttime lights (150×150 px, ~7.4m resolution)
- Landsat 8 bands B3/B4/B5/QA_PIXEL (150×150 px, 30m resolution) 
- OSM road network (12 segments)

**Data Characteristics:**
- Synthetic but realistic patterns (deterministic generation)
- Small size suitable for version control (~500 KB total)
- Includes test scenarios: clouds, water, vegetation, urban areas
- SHA-256 checksums in `manifest.json` for integrity verification

**Verification:**
```bash
# Verify data integrity (default mode)
uv run python tools/fetch_reference_data.py

# Force regeneration (uses fixed seeds)
uv run python tools/fetch_reference_data.py --force
```

See [Reference Data README](../../tests/data/reference/README.md) for details.