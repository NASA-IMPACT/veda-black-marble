# Black Marble Component Usage Audit

This document provides a comprehensive audit of where component classes are imported and used throughout the Black Marble codebase. This will help guide the transition from the class-based component API to the functional API.

## Summary

The component system is used extensively throughout the codebase:
- **20 files** import components from `blackmarble.components`
- **13 files** instantiate component classes
- Components are used in production code, tests, and examples
- All components return `MaskResult` objects rather than simple arrays

## Component Import Locations

### Production Code

1. **`blackmarble/pipeline_refactored.py`**
   - Imports: `CloudMasker`, `QualityCompositor`, `MaskStatistics`, `ProjectionHandler`, `ResolutionResampler`, `BoundsCropper`, `SpatialAligner`, `TemporalCompositor`, `SceneMerger`, `DateRangeSelector`, `CompositeMethod`, `COGGenerator`, `MetadataGenerator`, `VisualizationEnhancer`, `OutputValidator`, `OutputMetadata`, `COGProfile`, `RoadEnhancer`, `ContrastEnhancer`, `ColorMapper`, `CompositeEnhancer`
   - Usage: Demonstrates refactored pipeline architecture using components
   - Instantiates components in `PipelineContext.create()` method

2. **`blackmarble/components/__init__.py`**
   - Exports all component classes for public API
   - Defines `__all__` with 25+ exported components

3. **`blackmarble/components/quality.py`**
   - Internal imports for component composition
   - Defines: `CloudMasker`, `AerosolMasker`, `RadianceSaturationMasker`, `VIIRSQualityMasker`, `QualityCompositor`, `MaskStatistics`

4. **`blackmarble/components/temporal.py`**
   - Internal imports from base components
   - Defines: `TemporalCompositor`, `SceneMerger`, `DateRangeSelector`

### Test Files

5. **`tests/test_components.py`**
   - Imports: `CloudMasker`, `QualityCompositor`, `MaskStatistics`, `MaskResult`, `QualityStatistics`
   - Also imports utility functions: `AerosolMasker`, `RadianceSaturationMasker`, `VIIRSQualityMasker`, `create_landsat_qa_pipeline`
   - Comprehensive unit tests for all quality components

6. **`tests/test_pipeline_integration.py`**
   - Imports: `TemporalCompositor`, `COGGenerator`, `OutputMetadata`
   - Integration tests using components in realistic scenarios

7. **`tests/test_quality_integration.py`**
   - Tests quality component integration

8. **`tests/test_temporal_behavior.py`**
   - Tests temporal processing components

9. **`tests/test_preprocessing_behavior.py`**
   - Tests preprocessing components

10. **`tests/test_output_behavior.py`**
    - Tests output generation components

11. **`tests/test_enhancement_behavior.py`**
    - Tests enhancement components

12. **`tests/test_scipy_features.py`**
    - Tests components with scipy dependencies

13. **`tests/test_parallel_processing.py`**
    - Tests parallel processing with components

14. **`tests/test_cog_validation.py`**
    - Tests COG generation components

### Example Files

15. **`examples/mask_generation_example.py`**
   - Shows comparison between old component system and new functional API
   - Demonstrates that components always return `MaskResult` objects

16. **`examples/pipeline_with_quality_components.py`**
   - Imports: `CloudMasker`, `AerosolMasker`, `RadianceSaturationMasker`, `VIIRSQualityMasker`, `QualityCompositor`, `MaskStatistics`, `create_landsat_qa_pipeline`
   - Demonstrates quality-enhanced data sources
   - Shows standalone component usage

17. **`examples/advanced_scipy_demo.py`**
   - Advanced component usage with scipy features

18. **`blackmarble/components/pipeline_integration_example.py`**
   - Internal example showing component integration

19. **`blackmarble/components/pipeline_example.py`**
   - Internal pipeline example

20. **`blackmarble/components/example_usage.py`**
   - Internal component usage examples

## Component Instantiation Patterns

### Quality Components

```python
# Conservative cloud masking with dilation
cloud_masker = CloudMasker(strategy="conservative", dilate_radius=2)

# Aerosol masking
aerosol_masker = AerosolMasker(threshold="standard")

# Radiance saturation masking
radsat_masker = RadianceSaturationMasker()

# VIIRS quality masking
viirs_masker = VIIRSQualityMasker()

# Quality compositor
compositor = QualityCompositor(
    components=[cloud_masker, aerosol_masker],
    combination_method="and"
)

# Pre-built pipeline
pipeline = create_landsat_qa_pipeline(
    strategy="conservative",
    dilate_clouds=1
)
```

### Preprocessing Components

```python
# Projection handling
projection_handler = ProjectionHandler()

# Resolution resampling
resampler = ResolutionResampler(method="bilinear")

# Bounds cropping
cropper = BoundsCropper()

# Spatial alignment
aligner = SpatialAligner()
```

### Temporal Components

```python
# Temporal compositing
temporal_compositor = TemporalCompositor(method=CompositeMethod.MEDIAN)

# Scene merging
scene_merger = SceneMerger()

# Date range selection
date_selector = DateRangeSelector()
```

### Output Components

```python
# COG generation
cog_generator = COGGenerator(profile=COGProfile(profile_name="deflate"))

# Metadata generation
metadata_generator = MetadataGenerator()

# Visualization enhancement
vis_enhancer = VisualizationEnhancer()

# Output validation
validator = OutputValidator()
```

### Enhancement Components

```python
# Road enhancement
road_enhancer = RoadEnhancer()

# Contrast enhancement
contrast_enhancer = ContrastEnhancer(method="percentile", clip_percentiles=(2, 98))

# Color mapping
color_mapper = ColorMapper(colormap_name="inferno")

# Composite enhancement
composite_enhancer = CompositeEnhancer()
```

## Key Issues with Current Component System

1. **Always Returns MaskResult**: All components return `MaskResult` objects even when a simple boolean array would suffice
   ```python
   result = cloud_masker.process(qa_pixel)
   mask = result.mask  # Extra step to get actual mask
   stats = result.statistics  # Often unused
   ```

2. **Overly Complex for Simple Operations**: Simple operations require class instantiation
   ```python
   # Current way
   masker = CloudMasker(strategy="conservative")
   result = masker.process(qa_pixel)
   mask = result.mask
   
   # Could be simpler
   mask = create_cloud_mask(qa_pixel, conservative=True)
   ```

3. **Stateful When Not Needed**: Most components don't need state but are implemented as classes

4. **Testing Complexity**: Tests must handle `MaskResult` objects instead of simple arrays

## Migration Strategy

1. **Priority Files for Migration**:
   - `blackmarble/pipeline_refactored.py` - Main refactored pipeline
   - Test files - Update to use functional API
   - Example files - Show new patterns

2. **Component Categories**:
   - **Quality components**: Most used, highest priority for functional conversion
   - **Preprocessing components**: Currently return masks instead of processed data
   - **Temporal components**: Need to return actual composited data
   - **Output components**: May benefit from remaining as classes due to complexity
   - **Enhancement components**: Simple operations that should be functions

3. **Backwards Compatibility**:
   - Keep component classes but deprecate them
   - Have components internally use new functional API
   - Provide migration guide for users

## Recommendations

1. **Implement functional API** in `blackmarble/processing.py` for all mask generation
2. **Update components** to use functional API internally
3. **Migrate tests** to use functional API directly
4. **Update examples** to showcase functional approach
5. **Add deprecation warnings** to component classes
6. **Document migration path** for existing users