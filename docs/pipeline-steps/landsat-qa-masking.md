# Landsat QA Masking

This step identifies and masks clouds, shadows, and other artifacts in Landsat imagery using the QA_PIXEL band.

## Overview

Landsat Collection 2 provides a QA_PIXEL band with bit-packed quality flags. We decode these flags to create masks that exclude poor-quality pixels from analysis. The masking is applied per-date before temporal compositing to ensure consistent cloud removal across the time series.

## Algorithm

### Bit Flag Interpretation

The QA_PIXEL band uses 16-bit integers where each bit represents a quality flag:

| Bit | Flag | Description |
|-----|------|-------------|
| 0 | Fill | No data |
| 1 | Dilated Cloud | Buffered cloud pixels |
| 2 | Cirrus | High-altitude ice clouds |
| 3 | Cloud | Cloud pixels |
| 4 | Cloud Shadow | Shadow cast by clouds |
| 5 | Snow | Snow/ice pixels |
| 6 | Clear | Clear sky conditions |
| 7-8 | Water | Two-bit code: 00=land, 01=water, 10=reserved (snow/ice), 11=reserved |
| 9-10 | Cloud Confidence | 0=None, 1=Low, 2=Medium, 3=High |
| 11-12 | Cloud Shadow Confidence | Same as above |
| 13-14 | Snow/Ice Confidence | Same as above |
| 15 | Cirrus Confidence | 0=Low, 1=High |

### Masking Strategies

**Function**: `blackmarble.prepare.landsat_qa.create_landsat_cloud_mask()`

We provide three pre-configured strategies:

#### 1. Conservative Strategy
- **Behavior**: Accepts known clear values (21824, 21888) or pixels with CLEAR bit set
  - 21824 = 0b0101010101000000 (clear land, high confidence clear)
  - 21888 = 0b0101010110000000 (clear water, high confidence clear)
- **Keeps**: Maximum data retention, accepts lower quality pixels
- **Use Case**: Areas with persistent cloud cover

#### 2. Moderate Strategy (Default)
- **Masks**: Fill, dilated cloud, cirrus, cloud, shadow, snow, water
- **Behavior**: Rejects pixels if any problematic bits are set
- **Use Case**: Balance between quality and coverage

#### 3. Permissive Strategy  
- **Masks**: Only high-confidence (≥2) clouds, shadows, and cirrus
- **Keeps**: Low confidence artifacts
- **Use Case**: When some cloud contamination is acceptable

### Mask Dilation

**Purpose**: Expand masks to catch edge effects

**Parameter**: `dilate_pixels` (function default: 0, pipeline uses: 3)
- Creates buffer around detected features
- Uses square structuring element (2*dilate_pixels+1)
- Prevents cloud/shadow edges from contaminating analysis

## Pipeline Integration

The pipeline applies QA masking during the `process_landsat_date()` function:

1. **Multi-tile Handling**: Creates windowed mosaics for multiple Landsat tiles
2. **Grid Alignment**: Reprojects to locally-optimized Albers Equal Area projection
3. **Per-date Masking**: Applies masks before temporal compositing
4. **Diagnostic Output**: Optionally saves cloud masks as GeoTIFFs

The function:
- Processes QA_PIXEL first to create cloud mask
- Applies mask to all spectral bands (B3, B4, B5)
- Handles multiple tiles with windowed reads and mosaicking
- Outputs masked data ready for index calculation

## Quick Tuning

- Use `moderate` first (default).
- Switch to `conservative` when scenes are consistently cloudy.
- Increase `dilate_pixels` if cloud edges leak into output.

## Configuration

- `config.preparation.qa_cloud_strategy` (default `moderate`; options: `conservative`, `moderate`, `permissive`)
- `config.preparation.qa_dilation_radius` (default `3`)

## Notes

- QA masking runs before index calculation and temporal compositing.
- When diagnostics are enabled, mask outputs are written under `data/diagnostics/`.
- Related docs: [Landsat Search](./landsat-search.md), [Spectral Indices](./spectral-indices.md)