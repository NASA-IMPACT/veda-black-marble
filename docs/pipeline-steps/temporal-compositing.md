# Temporal Compositing

This step creates cloud-free composites from multiple Landsat scenes.

## Overview

Temporal compositing combines multiple Landsat observations to create a single, cloud-free image. The pipeline uses median compositing after quality masking to produce robust surface reflectance values.

## Processing Steps

### TODO: Document the following sections
- Quality masking with QA_PIXEL band
- Median composite calculation
- Handling of partial coverage
- Temporal window selection

## Key Functions

**Primary Functions**:
- `blackmarble.prepare.landsat.process_landsat_date()` - Process individual scenes
- `blackmarble.analyze.temporal.create_landsat_median_composite()` - Create composite

**Supporting Functions**:
- Quality masking utilities
- Temporal statistics functions

## Test Coverage

**Primary Test**: `tests/pipeline/test_temporal_compositing.py`
- Tests composite generation with multiple dates
- Verifies cloud masking effectiveness
- Checks edge cases (no valid pixels, single scene)

## Configuration

```python
# In PipelineConfig
config.analysis.temporal_window_days = 30
config.analysis.composite_method = "median"  # or "mean", "max"
config.analysis.min_valid_observations = 1
```

## Common Issues

1. **No Valid Pixels**: All pixels masked as clouds
   - Solution: Expand temporal window or adjust QA thresholds

2. **Memory Usage**: Loading multiple scenes
   - Solution: Process in spatial tiles

## Related Steps

- Previous: [Landsat QA Masking](./landsat-qa-masking.md)
- Next: [Spectral Indices](./spectral-indices.md)
- Uses: Quality-masked Landsat scenes
- Outputs to: NDVI/NDWI calculations