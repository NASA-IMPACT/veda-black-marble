# Statistics Calculation

This step computes summary statistics from the processed imagery.

## Overview

After all processing steps, the pipeline calculates key statistics to quantify urban development and nighttime lights intensity. These metrics are returned in the pipeline results dictionary.

## Calculated Metrics

### 1. Urban Area (km²)
- **Source**: Urban mask from NDUI threshold (default > 0.3)
- **Calculation**: Count of urban pixels × pixel area
- **Units**: Square kilometers

### 2. Mean Radiance
- **Source**: Enhanced NTL composite
- **Calculation**: Average of all non-NaN NTL values
- **Units**: Raw DN values (not converted to radiance)

### 3. Road Length (Optional)
- **Status**: Placeholder in results (always None)
- **Future**: Could calculate total road network length

## Implementation

**Location**: Statistics are calculated directly in `blackmarble.pipeline.black_marble_pipeline()`

**Actual Implementation**:
```python
# From blackmarble/pipeline.py
statistics = {
    "urban_area_km2": None,
    "mean_radiance": None,
    "road_length_km": None,
}

if urban_mask is not None and reference_transform is not None:
    # Calculate urban area in km²
    pixel_area_m2 = abs(reference_transform.a * reference_transform.e)
    urban_pixels = np.sum(urban_mask)
    statistics["urban_area_km2"] = float((urban_pixels * pixel_area_m2) / 1e6)

if ntl_composite is not None:
    # Calculate mean radiance (actually mean DN value)
    statistics["mean_radiance"] = float(np.nanmean(ntl_composite))
```

### Key Points

1. **Pixel Area Calculation**:
   - Uses transform parameters (a, e) which represent pixel width/height
   - Assumes square pixels in projected CRS
   - Converts from m² to km²

2. **Urban Mask**:
   - Binary mask where NDUI > 0.3
   - Simple pixel count × area calculation
   - No edge effects or partial pixel handling

3. **Mean Radiance**:
   - Actually mean DN value (not converted to physical radiance units)
   - Uses np.nanmean to ignore NaN values
   - Includes all pixels (no masking of low values)

## Test Coverage

**Integration Tests**: Pipeline tests verify statistics are calculated
- End-to-end tests check statistics in results dictionary
- No dedicated unit tests for statistics calculations

## Configuration

```python
# Urban threshold is hardcoded in pipeline:
urban_mask = ndui > 0.3  # Not configurable

# No configuration options for statistics calculations
```

## Output Format

Statistics are returned in the pipeline results dictionary:

```python
{
    "output_path": "path/to/output.tif",
    "bbox": (-3.8, 40.3, -3.6, 40.5),
    "date": datetime(2023, 6, 15),
    "metadata": {...},  # Processing metadata
    "validation": {...},  # COG validation results
    "statistics": {
        "urban_area_km2": 125.4,
        "mean_radiance": 45.2,
        "road_length_km": None
    }
}
```

## Common Issues

1. **Urban Area is Zero**
   - Cause: No pixels exceed NDUI threshold of 0.3
   - Check: NDUI value distribution
   - Solution: May need to adjust processing parameters

2. **Mean Radiance Very Low**
   - Remember: This is mean DN value, not physical radiance
   - DN values are typically 0-1000+ before enhancement
   - Enhanced values used for visualization, not statistics

3. **Missing Transform**
   - Statistics require reference_transform for pixel area
   - Should always be available if processing succeeded

## Future Enhancements

- Road network length calculation
- Per-admin-unit statistics
- Time series comparisons
- Percentile statistics for NTL
- Urban/rural breakdown

## Related Steps

- Previous: [NDUI Calculation](./ndui-calculation.md) - Provides urban mask
- Previous: [Urban Field Enhancement](./urban-field-enhancement.md) - Enhanced NTL
- Uses: Final processed rasters (NDUI, NTL composite, urban mask)
- Outputs: Statistics in results dictionary