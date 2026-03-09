# Spectral Indices Calculation

This step computes NDVI and NDWI from Landsat bands to identify vegetation and water features.

## Overview

Spectral indices use ratios of different wavelengths to highlight specific land cover types. We calculate:
- **NDVI** (Normalized Difference Vegetation Index): Vegetation density
- **NDWI** (Normalized Difference Water Index): Water bodies and moisture

**Key Point**: Indices are calculated **per-date before temporal compositing** to ensure mathematical correctness. This happens automatically in the pipeline after QA masking.

## Algorithms

### NDVI Calculation

**Function**: `blackmarble.analyze.indices.calculate_ndvi()`

**Formula**: 
```
NDVI = (NIR - Red) / (NIR + Red)
```

**Bands Used**:
- NIR: Landsat Band 5 (0.85-0.88 μm)
- Red: Landsat Band 4 (0.64-0.67 μm)

**Value Interpretation**:
- `-1.0 to 0.0`: Water, bare soil, built-up areas
- `0.0 to 0.3`: Sparse vegetation
- `0.3 to 0.6`: Moderate vegetation
- `0.6 to 1.0`: Dense vegetation

### NDWI Calculation

**Function**: `blackmarble.analyze.indices.calculate_ndwi()`

**Formula**:
```
NDWI = (Green - NIR) / (Green + NIR)
```

**Bands Used**:
- Green: Landsat Band 3 (0.53-0.59 μm)
- NIR: Landsat Band 5 (0.85-0.88 μm)

**Value Interpretation**:
- `0.0 to 1.0`: Water bodies (threshold at 0.0 for water masking)
- `-0.2 to 0.0`: Moist soil, wetlands
- `-0.5 to -0.2`: Dry vegetation
- `-1.0 to -0.5`: Dry soil, built-up areas

## Pipeline Behavior

### Index-First Calculation

The pipeline computes indices per date immediately after QA masking:

- NDVI and NDWI are derived from masked reflectance inputs.
- NDVI is clamped to `>= 0` before NDUI-related downstream use.
- NDWI is not clamped and preserves negative values.
- Per-date index arrays are then stacked for temporal compositing.

### Mathematical Correctness

- Indices are computed per date first.
- Compositing happens on index stacks, not raw bands.
- This avoids ratio-of-medians artifacts.

## Epsilon Handling

- Pipeline uses `mask_low_denom=True` for both NDVI and NDWI.
- Low-denominator pixels become `NaN` and are excluded during temporal compositing.

## Default Parameters

The index functions use `LANDSAT_EPSILON = 0.02` (2% reflectance threshold), and pipeline processing enables low-denominator masking so unstable pixels are excluded from compositing.

## Notes

- Landsat scaling is applied before index calculation.
- NDVI is clamped to `>= 0` in the pipeline before NDUI.
- Related docs: [QA Masking](./landsat-qa-masking.md), [Temporal Compositing](./temporal-compositing.md)