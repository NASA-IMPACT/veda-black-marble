# Spectral Indices Calculation

This step computes NDVI and NDWI from Landsat bands to identify vegetation and water features.

## Overview

Spectral indices use ratios of different wavelengths to highlight specific land cover types. We calculate:
- **NDVI** (Normalized Difference Vegetation Index): Vegetation density
- **NDWI** (Normalized Difference Water Index): Water bodies and moisture

**Critical Update**: As of the index-first calculation fix, indices are now calculated **per-date before temporal compositing** to ensure mathematical correctness.

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
- `0.3 to 1.0`: Open water
- `0.0 to 0.3`: Wet surfaces
- `-0.3 to 0.0`: Dry surfaces
- `-1.0 to -0.3`: Vegetation, built-up areas

## Index-First Temporal Compositing

### Mathematical Correctness

The pipeline now calculates indices **per-date before temporal compositing**. This is critical because:

```
median(NDVI_t) ≠ NDVI(median(NIR_t), median(RED_t))
```

The median of ratios is not equal to the ratio of medians. The old approach (band-first) produces mathematically incorrect results, particularly affecting agricultural areas with seasonal vegetation changes.

### Implementation

```python
# NEW APPROACH (Correct)
# 1. Calculate indices for each date
index_stacks = {"ndvi": [], "ndwi": []}
for date in landsat_dates:
    nir, red, green = load_bands(date)
    ndvi = calculate_ndvi(nir, red, mask_low_denom=True)
    ndwi = calculate_ndwi(green, nir, mask_low_denom=True)
    index_stacks["ndvi"].append(ndvi)
    index_stacks["ndwi"].append(ndwi)

# 2. Create temporal composites
ndvi_composite = create_index_temporal_composite(
    index_stacks["ndvi"],
    method="percentile",
    percentile=85.0  # Peak vegetation for NDUI
)
```

### Why 85th Percentile NDVI for NDUI?

NDUI uses NDVI to correct for vegetation's light-dimming effect:
```
NDUI = NTL / (1 - NDVI + epsilon)
```

Using the 85th percentile:
- Captures maximum vegetation light-blocking potential
- Prevents over-estimation of urban extent in agricultural areas
- Better represents the "greenest" state

### Configuration

```python
config = PipelineConfig(
    analysis=AnalysisConfig(
        # Index-first calculation (always enabled)
        calculate_indices_first=True,
        mask_low_denom=True
        
        # Temporal compositing is now hardcoded:
        # - NDVI: Always uses 85th percentile
        # - NDWI: Always uses median
    )
)
```

## Critical Implementation Detail: Epsilon Handling

### The Problem

When NIR ≈ Red (or Green ≈ NIR), the denominator approaches zero, causing:
1. Division by zero → NaN
2. Near-zero division → Extreme values (e.g., NDVI = 137.0)

### The Solution

**Key Fix**: Use `np.copysign()` to preserve sign when adding epsilon

```python
def calculate_ndvi(nir, red, epsilon=1e-8):
    numerator = nir - red
    denominator = nir + red
    
    # CRITICAL: Preserve sign of denominator when adding epsilon
    safe_denominator = np.where(
        np.abs(denominator) < epsilon,
        np.copysign(epsilon, denominator),  # ✅ Correct
        denominator
    )
    
    # NOT: np.sign(denominator) * epsilon  # ❌ Returns 0 when denominator=0
    
    return numerator / safe_denominator
```

**Why This Matters**:
- `np.sign(0)` returns 0, making safe_denominator = 0
- `np.copysign(epsilon, 0)` returns epsilon, preventing division by zero

## Implementation

```python
from blackmarble.analyze.indices import calculate_ndvi, calculate_ndwi

# Apply Landsat scaling first
nir = band5_data * 0.0000275 - 0.2
red = band4_data * 0.0000275 - 0.2
green = band3_data * 0.0000275 - 0.2

# Calculate indices
ndvi = calculate_ndvi(nir, red)
ndwi = calculate_ndwi(green, nir)

# Validate ranges
assert -1.0 <= np.nanmin(ndvi) <= np.nanmax(ndvi) <= 1.0
assert -1.0 <= np.nanmin(ndwi) <= np.nanmax(ndwi) <= 1.0
```

## Test Coverage

**Primary Test**: `tests/pipeline/test_pipeline_flow.py::test_09_calculate_indices`
- Basic calculation verification
- Range validation

**Critical Edge Case Tests**: `tests/analyze/test_ndvi_critical_edge_cases.py`
- Tests near-zero denominators
- Verifies epsilon handling
- Checks the 137.0 bug is fixed

**Additional Tests**: `tests/analyze/test_indices.py`
- Comprehensive value range tests
- NaN propagation
- Array shape handling

## Common Issues

1. **Extreme NDVI Values (>1.0 or <-1.0)**
   - Cause: Near-zero denominator
   - Fix: Epsilon handling with `np.copysign()`

2. **Unexpected NaN Values**
   - Cause: Masked/invalid input pixels
   - Fix: Use `np.nanmin/nanmax` for statistics

3. **Wrong Index Values**
   - Cause: Forgetting to apply Landsat scaling
   - Fix: Always scale before index calculation

## Performance Optimization

```python
# Efficient: Single pass calculation
ndvi = calculate_ndvi(nir, red)

# Inefficient: Multiple passes
numerator = nir - red
denominator = nir + red
ndvi = numerator / denominator  # Risk of division by zero!
```

## Configuration

```python
# In PipelineConfig
config.analysis.index_epsilon = 1e-8  # Epsilon for safe division
config.analysis.calculate_ndvi = True
config.analysis.calculate_ndwi = True
```

## Quality Checks

Always validate index ranges after calculation:

```python
def validate_index(data, name, expected_range=(-1, 1)):
    min_val, max_val = np.nanmin(data), np.nanmax(data)
    if not (expected_range[0] <= min_val <= max_val <= expected_range[1]):
        logger.warning(f"{name} outside expected range: [{min_val:.3f}, {max_val:.3f}]")
```

## Comparison Tool

A tool is provided to compare NDUI outputs between the legacy and new calculation methods:

```bash
python tools/compare_ndui.py old_ndui.tif new_ndui.tif -o comparison/
```

This generates:
- `comparison_stats.json`: Statistical analysis of differences
- `ndui_difference_raw.tif`: Raw difference values
- `ndui_difference_classes.tif`: Classified changes (decreased/unchanged/increased)

Expected differences:
- Agricultural areas: Higher NDUI with new method (better vegetation correction)
- Urban areas: Minimal change
- Overall: More accurate urban/rural discrimination

## Related Steps

- Previous: [Spatial Alignment](./spatial-alignment.md)
- Next: [NDUI Calculation](./ndui-calculation.md) 
- Uses: Scaled Landsat bands
- Required for: Urban area detection, water masking