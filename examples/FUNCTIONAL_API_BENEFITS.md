# Functional API Benefits

This document summarizes the key benefits of using the functional API from `blackmarble.processing` instead of the component-based approach.

## Quick Comparison

### Component Approach (OLD)
```python
from blackmarble.components.quality import CloudMasker

# Create component instance
masker = CloudMasker(strategy='conservative')
# Process returns MaskResult object
result = masker.process(qa_pixel)
# Extract actual mask from result
mask = result.mask
stats = result.statistics
```

### Functional Approach (NEW)
```python
from blackmarble.processing import create_cloud_mask, compute_statistics

# Direct function call returns mask
mask = create_cloud_mask(qa_pixel, conservative=True)
# Separate function for statistics
stats = compute_statistics(mask)
```

## Key Benefits

### 1. **Direct Return Values**
- Functions return actual data/masks, not wrapper objects
- No need to extract `.mask` from result objects
- Clear what each function returns

### 2. **Simplicity**
- No complex class hierarchies
- No inheritance chains to understand
- Just import and use functions

### 3. **Composability**
```python
# Easy to chain functions
enhanced = enhance_contrast(
    enhance_urban_features(ndui, roads, ntl),
    method='percentile'
)

# Or use intermediate variables for clarity
urban_enhanced = enhance_urban_features(ndui, roads, ntl)
final = enhance_contrast(urban_enhanced, method='percentile')
```

### 4. **Testability**
- Each function is independent
- Easy to test in isolation
- No complex setup required

### 5. **Performance**
- Less overhead from object creation
- Direct numpy operations
- No wrapper object allocation

### 6. **Pythonic Design**
- Follows "simple is better than complex"
- Functions do one thing well
- Clear parameter names

## Example: Quality Masking Pipeline

### Component Approach (Complex)
```python
from blackmarble.components.quality import (
    QualityCompositor, CloudMasker, AerosolMasker,
    RadianceSaturationMasker, MaskStatistics
)

# Create pipeline
compositor = QualityCompositor(
    components=[
        CloudMasker(name='qa_pixel', strategy='conservative'),
        AerosolMasker(name='qa_aerosol', threshold='standard'),
    ],
    combination_method='and'
)

# Process data
result = compositor.process({
    'qa_pixel': qa_pixel_data,
    'qa_aerosol': qa_aerosol_data
})

# Extract mask
final_mask = result.mask

# Track statistics separately
tracker = MaskStatistics()
tracker.process(final_mask)
stats = tracker.get_summary()
```

### Functional Approach (Simple)
```python
from blackmarble.processing import create_quality_mask, compute_statistics

# One function call for quality mask
mask = create_quality_mask(
    qa_bands={
        'qa_pixel': qa_pixel_data,
        'qa_aerosol': qa_aerosol_data
    },
    combine_method='and',
    cloud_strategy='conservative',
    aerosol_threshold='standard'
)

# Direct statistics computation
stats = compute_statistics(mask)
```

## Available Functions

### Quality Masking
- `create_cloud_mask()` - Cloud detection from QA_PIXEL
- `create_aerosol_mask()` - Aerosol detection from QA_AEROSOL
- `create_saturation_mask()` - Radiometric saturation detection
- `create_viirs_quality_mask()` - VIIRS quality flag processing
- `create_quality_mask()` - Combined quality masking
- `combine_masks()` - Combine multiple masks
- `dilate_mask()` - Expand masked regions

### Enhancement
- `enhance_contrast()` - Improve image contrast
- `enhance_roads()` - Enhance values along road networks
- `enhance_urban_features()` - Combined urban enhancement
- `normalize_data()` - Scale to specific range
- `scale_to_byte()` - Convert to 8-bit for visualization
- `apply_colormap()` - Create RGB visualizations

### Spatial Operations
- `resample_image()` - Change image resolution
- `reproject_image()` - Change coordinate system
- `crop_to_bounds()` - Crop to geographic bounds
- `align_rasters()` - Align multiple rasters
- `merge_scenes()` - Mosaic multiple scenes

### Temporal Processing
- `create_temporal_composite()` - Combine time series data
- `select_by_date_range()` - Filter scenes by date
- `composite_with_quality()` - Quality-weighted compositing

### Statistics
- `compute_statistics()` - Calculate data statistics

## Migration Guide

To migrate from components to functional API:

1. Replace component imports with function imports
2. Remove component instantiation
3. Call functions directly with data
4. Remove `.mask` extraction from results
5. Use separate functions for different concerns (e.g., statistics)

The functional API makes code cleaner, more testable, and easier to understand!