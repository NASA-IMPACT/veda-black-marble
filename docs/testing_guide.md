# Black Marble Testing Guide

This guide provides patterns and best practices for testing Black Marble components and pipelines.

## Component Testing

### Unit Testing Individual Components

Test components in isolation with mock data:

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

### Integration Testing Pipeline Phases

Test how components work together:

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

## Mock Data Sources

Create mock data sources for testing without external dependencies:

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

## Testing with Custom Components

Inject custom components for specialized testing:

```python
# Create custom components
custom_quality = CustomQualityCompositor()
custom_resampler = ResolutionResampler(method="cubic")

# Create context with custom components
context = PipelineContext.create(
    config=config,
    output_dir="outputs"
)

# Override specific components
context.quality_compositor = custom_quality
context.resolution_resampler = custom_resampler
```

## Type Safety in Tests

Always check for None before accessing optional attributes:

```python
def test_metadata_access():
    result = processor.process(data)
    
    # Type-safe metadata access
    assert result.metadata is not None
    assert "key" in result.metadata
    value = result.metadata["key"]
```

## Performance Testing

Test component performance with realistic data sizes:

```python
def test_component_performance():
    # Create large test dataset
    large_data = np.random.rand(10000, 10000)
    
    # Time the operation
    start_time = time.time()
    result = component.process(large_data)
    elapsed = time.time() - start_time
    
    # Assert reasonable performance
    assert elapsed < 60  # Should complete within 1 minute
```

## Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_components.py

# Run with coverage
uv run pytest --cov=blackmarble

# Run with type checking
uv run pre-commit run --all-files
```