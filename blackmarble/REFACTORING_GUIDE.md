# Black Marble Pipeline Refactoring Guide

This guide explains how the monolithic 850+ line `run_pipeline` function has been refactored into a modular, testable architecture using the new component system.

## Overview

The refactored pipeline (`pipeline_refactored.py`) demonstrates:
- **Separation of Concerns**: Each phase has clear responsibilities
- **Dependency Injection**: All dependencies are injected, making testing easy
- **Modular Components**: Small, focused components that do one thing well
- **Clear Data Flow**: Data passes through well-defined phases
- **Better Error Handling**: Each component can handle its own errors
- **Improved Testability**: Each component can be tested in isolation

## Architecture

### Pipeline Context

The `PipelineContext` class contains all dependencies and configuration:

```python
@dataclass
class PipelineContext:
    config: ProcessingConfig
    s3_adapter: S3Adapter
    raster_adapter: GDALRasterAdapter
    landsat_source: LandsatDataSource
    viirs_source: VIIRSDataSource
    osm_source: OSMDataSource
    # ... pipeline components
```

This allows for easy dependency injection and testing with mock objects.

### Pipeline Phases

The pipeline is broken down into 8 clear phases:

1. **Data Acquisition**: Fetch data from various sources
2. **Quality Assessment**: Apply masks and quality filters
3. **Spatial Processing**: Handle projections, cropping, and resampling
4. **Temporal Compositing**: Combine multi-date imagery
5. **Index Calculation**: Compute NDVI, NDWI, NDUI
6. **Enhancement**: Apply visualization enhancements
7. **Output Generation**: Create Cloud-Optimized GeoTIFF
8. **Upload**: Optional S3 upload

## Key Components Used

### Quality Components
- `QualityCompositor`: Handles all QA masking (cloud, aerosol, radsat)
- `MaskStatistics`: Provides statistics on mask coverage

### Spatial Components
- `ProjectionHandler`: Manages CRS transformations
- `BoundsCropper`: Crops data to specified bounds
- `ResolutionResampler`: Changes data resolution
- `SpatialAligner`: Ensures arrays have matching shapes

### Temporal Components
- `TemporalCompositor`: Creates median composites from multiple dates
- `DateRangeSelector`: Selects appropriate date ranges

### Enhancement Components
- `RoadEnhancer`: Processes OSM road data
- `ContrastEnhancer`: Improves image contrast
- `ColorMapper`: Applies color maps (e.g., inferno)

### Output Components
- `COGGenerator`: Creates Cloud-Optimized GeoTIFFs
- `OutputValidator`: Validates output files

## Usage Examples

### Basic Usage (Same Interface)

The refactored pipeline maintains the same interface as the original:

```python
from blackmarble.pipeline_refactored import run_refactored_pipeline

output_path = run_refactored_pipeline(
    lat1=40.7128,
    lon1=-74.0060,
    lat2=40.7614,
    lon2=-73.9776,
    year=2023,
    month=6,
    day=15,
    earthdata_token="your_token",
    output_dir="outputs",
    upload_bucket="my-bucket",
    upload_key="outputs/nyc_ndui.tif"
)
```

### Advanced Usage (Custom Components)

You can inject custom components for testing or special processing:

```python
from blackmarble.pipeline_refactored import PipelineContext, RefactoredPipeline
from blackmarble.config import ProcessingConfig

# Create custom components
custom_quality = CustomQualityCompositor()
custom_resampler = ResolutionResampler(method="cubic")

# Create context with custom components
config = ProcessingConfig.from_env()
context = PipelineContext.create(
    config=config,
    output_dir="outputs"
)

# Override specific components
context.quality_compositor = custom_quality
context.resolution_resampler = custom_resampler

# Run pipeline
pipeline = RefactoredPipeline(context)
output = pipeline.run(lat1, lon1, lat2, lon2, year, month, day, token)
```

## Testing Strategy

The refactored architecture enables comprehensive testing:

### Unit Tests

Test individual components:

```python
def test_quality_compositor():
    compositor = QualityCompositor(config=test_config)
    
    # Create test data
    data = {"B3": np.ones((100, 100))}
    qa_layers = {"cloud": np.zeros((100, 100))}
    
    # Test processing
    result = compositor.process(data, qa_layers)
    assert result.mask.shape == (100, 100)
    assert result.statistics.valid_pixels > 0
```

### Integration Tests

Test phase interactions:

```python
def test_spatial_processing_phase():
    pipeline = RefactoredPipeline(test_context)
    
    # Mock input data
    landsat_data = create_mock_landsat_data()
    viirs_data = create_mock_viirs_data()
    osm_data = create_mock_osm_data()
    
    # Test spatial processing
    result = pipeline._spatial_processing(
        landsat_data, viirs_data, osm_data,
        lat1, lon1, lat2, lon2
    )
    
    # Verify all outputs have same shape
    shapes = [v.shape for v in result.values()]
    assert all(s == shapes[0] for s in shapes)
```

### Mock Data Sources

Easy to mock for testing:

```python
class MockLandsatSource:
    def fetch(self, bbox, start_date, end_date, bands, **kwargs):
        return {
            "B3": np.random.rand(1000, 1000),
            "B4": np.random.rand(1000, 1000),
            "B5": np.random.rand(1000, 1000),
            "QA_PIXEL": np.zeros((1000, 1000)),
        }

# Use in tests
context.landsat_source = MockLandsatSource()
```

## Benefits of Refactoring

1. **Maintainability**: Each component has a single responsibility
2. **Testability**: Components can be tested in isolation
3. **Reusability**: Components can be used in other pipelines
4. **Extensibility**: Easy to add new processing steps
5. **Debugging**: Clear phase boundaries make debugging easier
6. **Performance**: Can optimize individual components
7. **Documentation**: Each component is self-documenting

## Migration Path

To migrate from the original pipeline:

1. **Drop-in Replacement**: Use `run_refactored_pipeline` with same parameters
2. **Gradual Migration**: Replace phases one at a time
3. **Custom Extensions**: Add new components for special processing
4. **Testing**: Add tests for each component before migrating

## Example: Adding a New Processing Step

To add a new processing step (e.g., atmospheric correction):

```python
from blackmarble.components.base import ProcessingComponent

class AtmosphericCorrector(ProcessingComponent):
    """Apply atmospheric correction to imagery."""
    
    def process(self, data: np.ndarray, metadata: Dict[str, Any]) -> ProcessingResult:
        # Implement correction algorithm
        corrected = self._apply_correction(data, metadata)
        
        return ProcessingResult(
            data=corrected,
            metadata={"correction_applied": True}
        )

# Add to pipeline
context.atmospheric_corrector = AtmosphericCorrector()

# Use in pipeline phase
def _atmospheric_correction(self, data):
    result = self.ctx.atmospheric_corrector.process(
        data["B3"], 
        metadata={"sensor": "landsat"}
    )
    return result.data
```

## Performance Considerations

The refactored pipeline maintains performance while improving structure:

- **Memory Efficiency**: Components process data in chunks when needed
- **Parallel Processing**: Components can be parallelized independently
- **Caching**: Intermediate results can be cached between phases
- **Lazy Loading**: Data sources can implement lazy loading

## Conclusion

The refactored pipeline demonstrates how a large, monolithic function can be broken down into manageable, testable components. This architecture provides a solid foundation for future enhancements while maintaining backward compatibility.