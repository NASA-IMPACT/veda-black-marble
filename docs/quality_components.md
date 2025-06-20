# Quality Control Components

This document describes the modular quality control components available in the Black Marble pipeline.

## Overview

The quality control components provide a modular, testable approach to satellite imagery quality assessment and masking. These components replace the monolithic QA masking logic with composable, reusable pieces.

## Key Features

- **Protocol-based design** - All components implement consistent interfaces
- **Pure functions** - Components are stateless for easy testing
- **Statistical tracking** - Comprehensive statistics about masking operations
- **Flexible composition** - Combine multiple masks using AND/OR logic
- **Edge case handling** - Graceful handling of NaN values and edge cases

## Components

### CloudMasker

Masks cloud-contaminated pixels in Landsat imagery using QA_PIXEL bands.

```python
from blackmarble.components.quality import CloudMasker

# Conservative strategy - only keep clear land/water
masker = CloudMasker(strategy="conservative", dilate_radius=2)

# Permissive strategy - keep more pixels
masker = CloudMasker(strategy="permissive")

result = masker.process(qa_pixel_array)
```

Strategies:
- **conservative**: Only pixels marked as clear land (21824) or clear water (21952)
- **permissive**: Excludes only high-confidence cloud pixels

### AerosolMasker

Masks aerosol-contaminated pixels using QA_AEROSOL bands.

```python
from blackmarble.components.quality import AerosolMasker

# Standard threshold
masker = AerosolMasker(threshold="standard")

# Strict threshold - masks more aerosol
masker = AerosolMasker(threshold="strict")

result = masker.process(qa_aerosol_array)
```

### RadianceSaturationMasker

Identifies radiometrically saturated pixels in Landsat bands 3, 4, and 5.

```python
from blackmarble.components.quality import RadianceSaturationMasker

masker = RadianceSaturationMasker()

# Process single band
result = masker.process(qa_radsat_array, band=3)

# Process all bands (returns 3D mask)
result = masker.process(qa_radsat_array)
```

### VIIRSQualityMasker

Masks VIIRS nighttime lights based on quality flags.

```python
from blackmarble.components.quality import VIIRSQualityMasker

masker = VIIRSQualityMasker()
result = masker.process(mandatory_quality_flag_array)
```

### QualityCompositor

Combines multiple quality masks using AND or OR logic.

```python
from blackmarble.components.quality import QualityCompositor

# Create compositor with AND logic
compositor = QualityCompositor(
    components=[cloud_masker, aerosol_masker],
    combination_method="and"
)

# Process multiple QA bands
result = compositor.process({
    "cloud_masker": qa_pixel_array,
    "aerosol_masker": qa_aerosol_array
})
```

### MaskStatistics

Tracks and reports comprehensive masking statistics.

```python
from blackmarble.components.quality import MaskStatistics

tracker = MaskStatistics()

# Track multiple masks
tracker.process(cloud_mask, mask_name="clouds")
tracker.process(aerosol_mask, mask_name="aerosol")

# Get summary report
summary = tracker.get_summary()
```

## Result Structure

All components return a `MaskResult` object containing:

```python
@dataclass
class MaskResult:
    mask: np.ndarray           # The binary mask (1=valid, 0=invalid)
    statistics: Dict[str, Any] # Statistical information
    metadata: Dict[str, Any]   # Additional metadata
```

Statistics typically include:
- `total_pixels`: Total number of pixels
- `valid_pixels`: Number of valid (unmasked) pixels
- `invalid_pixels`: Number of invalid (masked) pixels
- `valid_percentage`: Percentage of valid pixels
- Additional component-specific statistics

## Convenience Functions

### create_landsat_qa_pipeline

Creates a standard Landsat quality assessment pipeline:

```python
from blackmarble.components.quality import create_landsat_qa_pipeline

# Create standard pipeline
pipeline = create_landsat_qa_pipeline(
    strategy="conservative",
    dilate_clouds=2
)

# Process QA bands
result = pipeline.process({
    "qa_pixel": qa_pixel_array,
    "qa_aerosol": qa_aerosol_array
})
```

## Integration Examples

### With Existing Pipeline

```python
from blackmarble.components.quality import create_landsat_qa_pipeline
from blackmarble.components.base import apply_mask

# Create QA pipeline
qa_pipeline = create_landsat_qa_pipeline()

# Process QA bands
qa_result = qa_pipeline.process({
    "qa_pixel": qa_pixel_data,
    "qa_aerosol": qa_aerosol_data
})

# Apply mask to data bands
masked_b3 = apply_mask(band3_data, qa_result.mask)
masked_b4 = apply_mask(band4_data, qa_result.mask)
```

### With Adapters

See `examples/pipeline_with_quality_components.py` for a complete example of integrating quality components with the Phase 2 adapter architecture.

### Testing

The modular design makes testing straightforward:

```python
def test_cloud_masking():
    # Create test data
    qa_pixel = np.array([[21824, 54596], [21952, 55052]])
    
    # Create masker
    masker = CloudMasker(strategy="conservative")
    
    # Process
    result = masker.process(qa_pixel)
    
    # Verify
    assert result.statistics["valid_pixels"] == 2  # Only clear pixels
    assert np.array_equal(result.mask, [[1, 0], [1, 0]])
```

## Migration Guide

To replace existing QA masking logic:

1. Replace `mark_cloud()` with `CloudMasker`
2. Replace `mark_aerosol()` with `AerosolMasker`
3. Replace `mark_radsat()` with `RadianceSaturationMasker`
4. Use `QualityCompositor` to combine masks
5. Use `apply_mask()` to apply masks to data

Before:
```python
cloud_mask = mark_cloud(qa_pixel)
aerosol_mask = mark_aerosol(qa_aerosol)
combined_mask = cloud_mask & aerosol_mask
masked_data = data * combined_mask
```

After:
```python
qa_pipeline = create_landsat_qa_pipeline()
result = qa_pipeline.process({
    "qa_pixel": qa_pixel,
    "qa_aerosol": qa_aerosol
})
masked_data = apply_mask(data, result.mask)
```

## Performance Considerations

- Components process arrays in-place when possible
- Mask dilation is implemented efficiently without scipy
- Statistics are calculated lazily
- Memory usage scales with image size

## Future Enhancements

Potential improvements:
- GPU acceleration for large images
- Parallel processing for multi-band operations
- Additional masking strategies
- Machine learning-based quality assessment
- Integration with dask for out-of-core processing