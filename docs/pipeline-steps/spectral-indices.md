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

## Pipeline Implementation

### Index-First Calculation

The pipeline calculates indices per-date, immediately after QA masking:

```python
# For each Landsat date
for target_date in landsat_dates:
    # Process bands with QA masking
    processed_data = prepare.process_landsat_date(
        bands_data=bands_data,
        apply_qa_mask=True,
        cloud_strategy="moderate"
    )
    
    # Calculate indices immediately
    if all bands present:
        # Calculate NDVI
        ndvi_date = analyze.calculate_ndvi(
            nir=b5_data, 
            red=b4_data, 
            mask_low_denom=True  # Return NaN for low denominators
        )
        
        # Clamp negative NDVI to 0 (water/bare soil)
        # Note: Only NDVI is clamped, not NDWI
        ndvi_date = np.maximum(ndvi_date, 0.0)
        
        # Calculate NDWI (NOT clamped)
        ndwi_date = analyze.calculate_ndwi(
            green=b3_data, 
            nir=b5_data, 
            mask_low_denom=True
        )
        
        # Add to stacks for temporal compositing
        index_stacks["ndvi"].append(ndvi_date)
        index_stacks["ndwi"].append(ndwi_date)
```

### Mathematical Correctness

- Indices are computed per date first.
- Compositing happens on index stacks, not raw bands.
- This avoids ratio-of-medians artifacts.

## Epsilon Handling

- Pipeline uses `mask_low_denom=True` for both NDVI and NDWI.
- Low-denominator pixels become `NaN` and are excluded during temporal compositing.

## Default Parameters

The indices functions use these defaults:

```python
# From indices.py
LANDSAT_EPSILON = 0.02  # 2% reflectance threshold

# Function signatures (note: mask_low_denom defaults to False)
calculate_ndvi(nir, red, epsilon=LANDSAT_EPSILON, mask_low_denom=False)
calculate_ndwi(green, nir, epsilon=LANDSAT_EPSILON, mask_low_denom=False)

# But pipeline always passes mask_low_denom=True:
ndvi = calculate_ndvi(nir, red, mask_low_denom=True)
```

## Notes

- Landsat scaling is applied before index calculation.
- NDVI is clamped to `>= 0` in the pipeline before NDUI.
- Related docs: [QA Masking](./landsat-qa-masking.md), [Temporal Compositing](./temporal-compositing.md)