# NDUI (Normalized Difference Urban Index) Calculation

This step combines nighttime lights with vegetation indices to identify urban areas using the Black Marble algorithm.

## Overview

NDUI quantifies the "urbanness" of pixels by comparing nighttime light intensity with vegetation presence. High NDUI values indicate developed areas with artificial lighting and minimal vegetation.

## Algorithm

**Function**: `blackmarble.analyze.indices.calculate_ndui()`

**Processing Steps**:

### 1. Input Preparation

The function prepares the NTL (nighttime lights) data:

- **Ceiling Clipping**: Values above `ntl_ceiling` are clipped (default = 10.0 nW/cm²/sr)
- **Floor Application**: Values below `ntl_floor` are set to 0 (default = 0.1)
- **Ceiling-Only Normalization**: `ntl_normalized = ntl_clipped / ntl_ceiling`

**Why ceiling-only normalization?** This keeps the 0-1 scale invariant when ceiling is tuned year-to-year, as physical zero is unambiguous unlike the floor.

### 2. Core Index Calculation

The original Black Marble formula:
```
ndui = (ntl_normalized - ndvi) / (ntl_normalized + ndvi + epsilon)
```

Where:
- `ntl_normalized`: Normalized nighttime lights (0-1)
- `ndvi`: Vegetation index from Landsat
- `epsilon`: Small value (default 0.00001) to prevent division by zero

The result is then clamped to [-1, 1].

### 3. Water Masking

Water bodies are masked using NDWI:

- Pixels with `NDWI >= 0.0` are treated as water and mapped to the minimum NDUI state.

### 4. Range Transformation

Convert from [-1, 1] to [0, 1] range using the bm.py formula:

`ndui = abs(ndui + 1.0) / 2.0`

This transformation:
- Maps water pixels (-1) → 0
- Maps the range [-1, 1] → [0, 1]
- Uses absolute value (matching legacy behavior)

### 5. Floor Threshold

Remove noise and spurious low values:

- Values below `ndui_floor` (default `0.02`) are set to `0.0`.

## Value Interpretation

After all transformations, NDUI values represent:
- `0.0`: Water bodies or non-urban (below threshold)
- `0.0-0.2`: Natural areas with minimal lighting
- `0.2-0.4`: Rural areas with some development
- `0.4-0.6`: Suburban/mixed development
- `0.6-0.8`: Urban areas
- `0.8-1.0`: Dense urban cores/bright industrial

## Pipeline Behavior

- NDUI is computed only after NTL and Landsat-derived indices are spatially aligned.
- The pipeline validates shape compatibility before NDUI calculation.
- Urban mask derivation for summary metrics uses the configured NDUI threshold.

## Key Differences from Standard NDVI-like Indices

1. **Ceiling-only normalization**: Maintains temporal consistency
2. **Water masking**: Prevents false positives in water bodies
3. **Range transformation**: Converts to intuitive 0-1 scale
4. **Floor thresholding**: Removes sensor noise

## Configuration Parameters

- `config.analysis.ntl_floor` (default `0.1`)
- `config.analysis.ntl_ceiling` (default `10.0`)
- `config.analysis.ndui_floor` (default `0.02`)

## Common Issues and Solutions

1. **Water Bodies Showing High NDUI**
   - Cause: Missing NDWI input for water masking
   - Solution: Always provide NDWI from Landsat processing
   - Note: Water pixels (NDWI >= 0) are set to -1, then mapped to 0

2. **Unexpected Zero Values**
   - Cause: NTL values below floor threshold (0.1 nW/cm²/sr)
   - Solution: Check if ntl_floor is appropriate for your region
   - Note: Floor is applied after conversion to radiance

3. **Saturation in Bright Areas**
   - Cause: NTL values exceeding ceiling (10.0 nW/cm²/sr)
   - Solution: Adjust ntl_ceiling based on regional statistics
   - Debug: Check logger output for clipping percentage

4. **Shape Mismatch Errors**
   - Cause: Inputs not aligned to same spatial grid
   - Solution: Pipeline handles alignment automatically
   - Debug: Check that all inputs go through reprojection

5. **NTL Values Seem Too Low**
   - Note: Input should be raw DN values, NOT pre-converted radiance
   - Function automatically divides by 10.0 for conversion
   - Check: Raw VIIRS DN values typically range 0-1000+

## Test Coverage

- Unit coverage exists in `tests/analyze/`.
- Pipeline-level behavior is validated in repository integration tests.

## Performance Notes

- Calculation is element-wise and vectorized (fast)
- Water masking adds minimal overhead
- Debug logging can be enabled for troubleshooting
- Memory usage: Same as input arrays (no additional copies)

## Diagnostic Outputs

When `config.save_diagnostics = True`:
- `data/diagnostics/ndui_raw.tif`: NDUI after calculation
- Logger outputs scaling info and clipping statistics
- Shape verification logs for debugging alignment issues

## Related Steps

- Previous: [Spectral Indices](./spectral-indices.md) - Provides NDVI, NDWI
- Previous: [Urban Field Enhancement](./urban-field-enhancement.md) - Enhanced NTL input
- Previous: [Spatial Alignment](./spatial-alignment.md) - Ensures matching grids
- Next: Final Black Marble product export
- Outputs: NDUI array (0-1 range) for urban area identification