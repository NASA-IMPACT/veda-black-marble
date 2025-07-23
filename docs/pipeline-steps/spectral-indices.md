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

Calculating indices per-date is critical because:
- **median(NDVI_t) ≠ NDVI(median(NIR_t), median(RED_t))**
- The median of ratios is not the ratio of medians
- Band-first compositing produces incorrect results

### Memory Optimization

The pipeline includes memory management:
```python
# Optional: Delete bands after index calculation
if config.memory.delete_bands_after_indices:
    del b3_data, b4_data, b5_data
    gc.collect()
```

## Epsilon Handling

### The Problem

When band values are very similar (e.g., NIR ≈ Red), the denominator approaches zero, causing:
1. Division by zero → NaN
2. Near-zero division → Extreme values outside [-1, 1]

### The Solution

The implementation uses two strategies:

1. **mask_low_denom=True** (used in pipeline):
   - Returns NaN for denominators < epsilon
   - Preserves mathematical correctness for temporal compositing
   - NaN pixels are handled properly by `np.nanmedian()`

2. **Adaptive floor** (when mask_low_denom=False):
   - Uses both absolute and relative thresholds
   - Applies `np.copysign()` to preserve sign
   - Clamps output to [-1, 1]

```python
# Pipeline approach
ndvi = calculate_ndvi(nir, red, mask_low_denom=True)
# Low denominators → NaN → excluded from temporal composite

# Adaptive floor approach (for single-date analysis)
epsilon_rel = 0.05  # 5% of signal magnitude
signal_magnitude = (np.abs(nir) + np.abs(red)) / 2
adaptive_floor = np.maximum(epsilon, epsilon_rel * signal_magnitude)
```

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

## Test Coverage

**Unit Tests**: `tests/analyze/`
- Basic calculation verification
- Edge cases with near-zero denominators
- Range validation [-1, 1]
- NaN handling

**Integration**: Full pipeline tests verify index calculation in context

## Common Issues

1. **Bands Not Scaled**
   - Problem: Raw DN values instead of reflectance
   - Solution: Pipeline automatically applies Landsat scaling
   - Scaling: `reflectance = DN * 0.0000275 - 0.2`

2. **Missing Data After Compositing**
   - Problem: Too many NaN values from mask_low_denom
   - Solution: Temporal composite handles NaN properly
   - Check: Ensure enough valid observations

3. **Negative NDVI in NDUI**
   - Problem: Water/bare soil has negative NDVI
   - Solution: Pipeline clamps to 0 before NDUI calculation
   - `ndvi_date = np.maximum(ndvi_date, 0.0)`

## Configuration

```python
# Configuration is minimal - indices are always calculated
config.memory.delete_bands_after_indices = True  # Save memory
```

## Performance Notes

- Index calculation is fast (< 0.1s per date)
- Memory usage: ~4 bytes per pixel per index
- Temporal stacks grow with number of dates
- Memory optimization available via config

## Related Steps

- Previous: [QA Masking](./landsat-qa-masking.md) - Provides clean bands
- Next: [Temporal Compositing](./temporal-compositing.md) - Creates final indices
- Uses: QA-masked Landsat bands (already scaled)
- Outputs: Index stacks for temporal compositing