# Statistics Calculation

This step computes summary statistics from the processed imagery.

## Overview

After all processing steps, the pipeline calculates key statistics to quantify urban development and nighttime lights intensity. These metrics are saved alongside the imagery outputs.

## Calculated Metrics

### TODO: Document the following metrics
- Urban area (km²) - based on NDUI threshold
- Mean radiance (nW/cm²/sr) - average NTL intensity
- Statistical summaries per administrative unit
- Time series comparisons

## Key Functions

**Location**: Statistics are calculated directly in `blackmarble.pipeline.black_marble_pipeline()`

**Key Calculations**:
```python
# Urban area calculation
urban_pixels = (ndui > threshold).sum()
urban_area_km2 = urban_pixels * pixel_area_km2

# Mean radiance
mean_radiance = np.nanmean(ntl_data[ntl_data > 0])
```

## Test Coverage

**Primary Test**: `tests/pipeline/test_statistics_calculation.py`
- Verifies area calculations with known inputs
- Tests handling of nodata values
- Checks statistical accuracy

## Configuration

```python
# In PipelineConfig
config.analysis.ndui_urban_threshold = 0.5
config.analysis.statistics_ignore_zeros = True
config.output.save_statistics = True
```

## Output Format

Statistics are saved as:
- JSON metadata files
- CSV summary tables
- GeoJSON with administrative summaries

## Related Steps

- Previous: [NDUI Calculation](./ndui-calculation.md)
- Uses: Final processed rasters (NDUI, NTL)
- Outputs: Summary statistics and reports