# NDUI (Normalized Difference Urban Index) Calculation

This step combines nighttime lights with vegetation indices to identify urban areas using the Black Marble v3 algorithm.

## Overview

NDUI quantifies the "urbanness" of pixels by comparing nighttime light intensity with vegetation presence. High NDUI values indicate developed areas with artificial lighting and minimal vegetation.

## Algorithm

**Function**: `blackmarble.analyze.indices.calculate_ndui()`

**Processing Steps**:

### 1. Input Preparation

The function first prepares the NTL (nighttime lights) data:

- **DN to Radiance Conversion**: If max(NTL) > 100, assumes DN values and converts: `ntl = ntl / 10.0`
- **Ceiling Clipping**: `ntl = min(ntl, ntl_ceiling)` (default ceiling = 10.0 nW/cm²/sr)
- **Floor Clipping**: `ntl = 0 if ntl < ntl_floor else ntl` (default floor = 0.1)
- **Ceiling-Only Normalization**: `ntl_normalized = ntl / ntl_ceiling`

**Why ceiling-only normalization?** We divide by the fixed radiance ceiling instead of the (ceiling-floor) range so that an NDUI value of 1.0 always represents the same absolute radiance, independent of year-specific gain adjustments. The floor is used only to suppress sensor noise and is not part of the scaling.

### 2. Core Index Calculation

```
ndui_raw = (ntl_normalized - ndvi) / (ntl_normalized + ndvi + epsilon)
```

Where:
- `ntl_normalized`: Normalized nighttime lights (0-1)
- `ndvi`: Vegetation index (-1 to 1)
- `epsilon`: Small value (1e-6) to prevent division by zero

The raw NDUI is then clamped to [-1, 1].

### 3. Water Masking

Water bodies are masked using NDWI (Normalized Difference Water Index):

```
if ndwi >= 0:
    ndui_raw = -1  # Water pixels set to -1
```

### 4. Range Transformation

Convert from [-1, 1] to [0, 1] range for easier interpretation:

```
ndui = abs(ndui_raw + 1.0) / 2.0
```

This transformation ensures:
- Water pixels (-1) → 0
- Non-urban areas → low values
- Urban areas → high values

### 5. Floor Threshold

Remove noise and spurious low values:

```
if ndui < ndui_floor:
    ndui = 0  # Default floor = 0.02
```

## Value Interpretation

After all transformations, NDUI values represent:
- `0.0`: Water bodies or non-urban (below threshold)
- `0.0-0.2`: Natural areas with minimal lighting
- `0.2-0.4`: Rural areas with some development
- `0.4-0.6`: Suburban/mixed development
- `0.6-0.8`: Urban areas
- `0.8-1.0`: Dense urban cores/bright industrial

## Implementation

```python
from blackmarble.analyze.indices import calculate_ndui

# Inputs from previous steps
ntl = viirs_composite      # Nighttime lights (can be DN or radiance)
ndvi = ndvi_composite      # Vegetation index from Landsat
ndwi = ndwi_composite      # Water index from Landsat (optional)

# Calculate NDUI with default Black Marble v3 parameters
ndui = calculate_ndui(
    ntl=ntl,
    ndvi=ndvi,
    ndwi=ndwi,              # Optional water masking
    ntl_floor=0.1,          # Minimum NTL threshold (nW/cm²/sr)
    ntl_ceiling=10.0,       # Maximum NTL for normalization
    ndui_floor=0.02,        # Minimum NDUI threshold
    epsilon=1e-6            # Numerical stability
)

# Create urban mask
urban_mask = ndui > 0.3  # Typical threshold for urban areas
```

## Key Differences from Standard NDVI-like Indices

1. **Ceiling-only normalization**: Maintains temporal consistency
2. **Water masking**: Prevents false positives in water bodies
3. **Range transformation**: Converts to intuitive 0-1 scale
4. **Floor thresholding**: Removes sensor noise

## Configuration Parameters

```python
# In PipelineConfig
config.analysis.ntl_floor = 0.1      # Min NTL radiance (nW/cm²/sr)
config.analysis.ntl_ceiling = 10.0    # Max NTL for normalization
config.analysis.ndui_floor = 0.02     # Min NDUI value to keep
```

## Common Issues and Solutions

1. **Water Bodies Showing High NDUI**
   - Cause: Missing NDWI input for water masking
   - Solution: Always provide NDWI from Landsat processing

2. **Unexpected Zero Values**
   - Cause: NTL values below floor threshold
   - Solution: Check if ntl_floor is appropriate for your region

3. **Saturation in Bright Areas**
   - Cause: NTL values exceeding ceiling
   - Solution: Adjust ntl_ceiling based on regional statistics

4. **Different Results from Legacy Black Marble**
   - Note: Legacy bm.py included road boosting (1.3x for primary roads)
   - Current implementation does not include road enhancement in NDUI

## Test Coverage

**Primary Test**: `tests/pipeline/test_pipeline_flow.py::test_09_calculate_indices`
- Basic NDUI calculation
- Value range verification

**Detailed Tests**: `tests/pipeline/test_ndui_calculation.py`
- Water masking behavior
- Parameter effects (floor/ceiling)
- Edge cases: all zeros, all NaN
- DN to radiance conversion

## Performance Notes

- Calculation is element-wise and vectorized
- Water masking adds minimal overhead
- No significant memory requirements beyond input arrays

## Related Steps

- Previous: [Spectral Indices](./spectral-indices.md) (NDVI, NDWI inputs)
- Previous: [Temporal Composite](./temporal-composite.md) (NTL input)
- Next: [Urban Field Enhancement](./urban-field-enhancement.md)
- Outputs: NDUI array for final Black Marble product