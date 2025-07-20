# NDUI (Normalized Difference Urban Index) Calculation

This step combines nighttime lights with vegetation indices to identify urban areas.

## Overview

NDUI quantifies the "urbanness" of pixels by comparing nighttime light intensity with vegetation presence. High NDUI values indicate developed areas with artificial lighting and minimal vegetation.

## Algorithm

**Function**: `blackmarble.analyze.indices.calculate_ndui()`

**Formula**:
```
NDUI = (NTL - NDVI) / (NTL + NDVI)
```

Where:
- NTL: Nighttime lights (VIIRS radiance, 0-100+ nW/cm²/sr)
- NDVI: Vegetation index (-1 to 1)

**Value Interpretation**:
- `0.7 to 1.0`: Dense urban core
- `0.3 to 0.7`: Urban/suburban
- `0.0 to 0.3`: Rural with some lighting
- `-1.0 to 0.0`: Natural areas (vegetation, no lights)

## Key Features

### 1. Water Masking

Water bodies can produce misleading NDUI values (low NDVI + low NTL). We mask them using NDWI:

```python
def calculate_ndui(ntl, ndvi, ndwi=None, water_threshold=0.0):
    # Calculate base NDUI
    ndui = (ntl - ndvi) / (ntl + ndvi + epsilon)
    
    # Mask water if NDWI provided
    if ndwi is not None:
        water_mask = ndwi > water_threshold
        ndui[water_mask] = np.nan
    
    return ndui
```

### 2. Parameter Scaling

NDUI is sensitive to the scale difference between NTL (0-100+) and NDVI (-1 to 1):

**Options**:
1. **Scale NTL**: Divide by maximum expected value
2. **Apply floor/ceiling**: Clip extreme values
3. **Log transform**: Reduce NTL dynamic range

```python
# Example: Scale and clip
ntl_scaled = np.clip(ntl / 50.0, 0, 2)  # Scale to ~0-2 range
ndui = calculate_ndui(ntl_scaled, ndvi, ndwi)
```

### 3. Urban Threshold Selection

Convert NDUI to binary urban mask:

```python
urban_mask = ndui > 0.3  # Typical threshold

# Or data-driven threshold
threshold = np.percentile(ndui[ndui > 0], 70)  # Top 30% of positive values
urban_mask = ndui > threshold
```

## Implementation

```python
from blackmarble.analyze.indices import calculate_ndui

# Inputs from previous steps
ntl = viirs_aligned  # Nighttime lights
ndvi = calculate_ndvi(nir, red)  # Vegetation index
ndwi = calculate_ndwi(green, nir)  # Water index

# Calculate NDUI with water masking
ndui = calculate_ndui(
    ntl=ntl,
    ndvi=ndvi,
    ndwi=ndwi,
    water_threshold=0.0,  # NDWI > 0 indicates water
    floor=-1.0,
    ceiling=1.0
)

# Create urban mask
urban_mask = ndui > 0.3
```

## Test Coverage

**Primary Test**: `tests/pipeline/test_pipeline_flow.py::test_09_calculate_indices`
- Basic NDUI calculation
- Value range verification

**Detailed Tests**: `tests/pipeline/test_ndui_calculation.py`
- Water masking behavior
- Parameter effects (floor/ceiling)
- Edge cases: all zeros, all NaN
- Threshold sensitivity

## Decision Points

### 1. Water Masking Strategy

**Options**:
- `ndwi > 0.0`: Conservative, masks most water
- `ndwi > 0.1`: Permissive, only deep water
- `None`: No masking (not recommended)

**Recommendation**: Use NDWI > 0.0 for coastal areas

### 2. NTL Scaling

**Trade-offs**:
- No scaling: Preserves absolute radiance values
- Linear scaling: Better NDUI distribution
- Log scaling: Handles extreme bright spots

**Recommendation**: Scale by regional 95th percentile

### 3. Urban Threshold

**Methods**:
1. **Fixed**: 0.3 works globally
2. **Percentile**: Adapt to local conditions
3. **Otsu**: Automatic thresholding

**Recommendation**: Start with 0.3, validate with ground truth

## Common Issues

1. **Water Bodies Classified as Urban**
   - Cause: No water masking
   - Fix: Always provide NDWI for masking

2. **Rural Areas Over-classified**
   - Cause: Low but non-zero NTL
   - Fix: Increase threshold or apply NTL floor

3. **Bright Spots Dominate**
   - Cause: Industrial sites, airports
   - Fix: Clip extreme NTL values or use log transform

## Configuration

```python
# In PipelineConfig
config.analysis.ndui_water_threshold = 0.0
config.analysis.ndui_floor = -1.0
config.analysis.ndui_ceiling = 1.0
config.analysis.ndui_urban_threshold = 0.3
config.analysis.scale_ntl_for_ndui = True
```

## Validation

Compare NDUI-derived urban areas with:
- OpenStreetMap urban polygons
- Population density data
- Impervious surface datasets

Expected accuracy: 80-90% agreement with reference data

## Performance Notes

- Calculation is element-wise and fast
- Water masking adds ~10% overhead
- No significant memory requirements

## Related Steps

- Previous: [Spectral Indices](./spectral-indices.md) (NDVI, NDWI inputs)
- Next: [Urban Field Enhancement](./urban-field-enhancement.md)
- Uses: Aligned NTL, NDVI, NDWI
- Outputs: Urban mask for enhancement and statistics