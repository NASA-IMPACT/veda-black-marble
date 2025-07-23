# Temporal Compositing

This step creates cloud-free composites from pre-calculated spectral indices across multiple dates.

## Overview

Temporal compositing combines multiple observations to create a single, representative value for each pixel. The pipeline uses different compositing methods for different indices:
- **NDVI**: 85th percentile (captures peak vegetation)
- **NDWI**: Median (typical water presence)

**Key Point**: Indices are calculated per-date BEFORE compositing to ensure mathematical correctness. This is done automatically during the Landsat processing step.

## Algorithm

**Function**: `blackmarble.analyze.temporal.create_index_temporal_composite()`

### Processing Steps

1. **Stack Alignment**: 
   - All arrays must share the same CRS (handled by prior reprojection)
   - Different-sized arrays are padded with NaN to maximum size
   - Upper-left corners must be aligned

2. **Composite Calculation**:
   - NDVI: Uses 85th percentile via streaming algorithm (memory efficient)
   - NDWI: Uses median for typical water presence
   - Valid pixel counting for quality assessment

3. **Insufficient Data Handling**:
   - Pixels with < `min_valid_observations` are set to NaN
   - Default requires at least 1 valid observation

### Why Different Methods?

- **85th Percentile for NDVI**: Captures peak vegetation conditions which better represent the vegetation's potential to dim nighttime lights
- **Median for NDWI**: Provides typical water presence, filtering out temporary flooding or drought conditions

## Pipeline Implementation

```python
# From blackmarble/pipeline.py
if index_stacks["ndvi"]:
    logger.info("Creating NDVI composite from %d dates", len(index_stacks['ndvi']))
    ndvi_composite = analyze.create_index_temporal_composite(
        index_stacks["ndvi"],
        method="percentile",
        percentile=85.0,
        min_valid_observations=config.analysis.min_valid_observations,
    )

if index_stacks["ndwi"]:
    logger.info("Creating NDWI composite from %d dates", len(index_stacks['ndwi']))
    ndwi_composite = analyze.create_index_temporal_composite(
        index_stacks["ndwi"],
        method="median",
        min_valid_observations=config.analysis.min_valid_observations,
    )
```

## Test Coverage

**Unit Tests**: `tests/analyze/test_temporal.py`
- Composite generation with multiple dates
- Handling of different array sizes
- Edge cases: no data, single observation
- Percentile calculations

**Integration**: Pipeline tests verify end-to-end compositing

## Configuration

```python
# In PipelineConfig
config.analysis.min_valid_observations = 1  # Minimum observations per pixel
# Note: Composite methods are hardcoded (85th percentile for NDVI, median for NDWI)
```

## Common Issues

1. **Array Size Mismatches**
   - Cause: Different Landsat tiles have different extents
   - Solution: Arrays are automatically padded with NaN to maximum size
   - Note: Upper-left alignment is assumed

2. **Memory Usage with Many Dates**
   - Problem: Stacking 365 days of data can exhaust memory
   - Solution: 85th percentile uses streaming algorithm for NDVI
   - Alternative: Process in spatial chunks if needed

3. **All NaN Results**
   - Cause: All observations were masked by QA
   - Check: Cloud cover statistics from QA masking step
   - Solution: Use more permissive QA strategy or longer time window

4. **Unexpected Composite Values**
   - Remember: NDVI is clamped to [0, 1] before compositing
   - NDWI is NOT clamped (can be negative)
   - Check individual date values before compositing

## Performance Notes

- **Streaming 85th percentile**: Only keeps top 3 values per pixel (4x memory reduction)
  - Optimized for 4-23 observations (typical for 30-day windows)
  - Provides 4-5x speedup over standard numpy percentile
- **Median calculation**: Requires full stack in memory
- **NaN handling**: Minimal overhead with numpy's nan functions
- **Array padding**: Fast operation using np.full with NaN

## Related Steps

- Previous: [Spectral Indices](./spectral-indices.md) - Calculates indices per date
- Next: [Spatial Alignment](./spatial-alignment.md) - Aligns to common grid
- Uses: Pre-calculated NDVI/NDWI stacks with QA masking applied
- Outputs: Single composite arrays for NDUI calculation