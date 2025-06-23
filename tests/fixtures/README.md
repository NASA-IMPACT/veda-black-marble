# Black Marble Test Fixtures

This directory contains test data generators and fixtures for integration testing the Black Marble functional API pipeline.

## Overview

The test fixtures provide minimal but realistic test data that allows us to verify the pipeline executes correctly end-to-end without requiring actual satellite data downloads or external services.

## Key Components

### `data_generators.py`
Contains functions to generate realistic test data:
- **Landsat data**: Simulates multispectral bands with vegetation signatures and QA bands
- **VIIRS data**: Simulates nighttime lights with urban centers and quality flags
- **OSM data**: Simulates road networks with different road types

### `example_pipeline_test.py`
Demonstrates how to run a complete pipeline test using the functional API.

## Test Data Characteristics

### Landsat Test Data
- **Bands**: B3 (Green), B4 (Red), B5 (NIR)
- **QA Bands**: QA_PIXEL (clouds), QA_AEROSOL, QA_RADSAT
- **Realistic Features**:
  - Correlated band values representing vegetation
  - Cloud masks with shadows
  - Aerosol quality variations

### VIIRS Test Data
- **Layers**: NTL (nighttime lights), Quality Flag, Lunar Irradiance
- **Realistic Features**:
  - Urban centers with Gaussian light distributions
  - Road network illumination
  - Quality flag variations

### OSM Test Data
- **Road Types**: Motorway, Primary, Secondary, Residential
- **Realistic Features**:
  - Major road corridors
  - Minor road networks
  - Appropriate road widths

## Usage Example

```python
from tests.fixtures.data_generators import generate_complete_test_scene

# Generate a 100x100 test scene
scene = generate_complete_test_scene(shape=(100, 100))

# Access data
landsat_b3 = scene['landsat']['B3']
viirs_ntl = scene['viirs']['NTL']
osm_roads = scene['osm']['roads']
```

## Integration Test Pattern

The integration tests follow this pattern:

1. **Setup**: Create configuration and generate test data
2. **Acquire**: Mock data acquisition with test data
3. **Prepare**: Apply quality masks using functional API
4. **Analyze**: Calculate indices (NDVI, NDWI, NDUI)
5. **Enhance**: Apply visualization enhancements
6. **Export**: Validate output structure

## Key Test Scenarios

1. **Minimal Pipeline Execution**: Verify all phases complete successfully
2. **Configuration Testing**: Test both `for_bm_parity()` and `for_enhanced_quality()`
3. **Environment Variable Override**: Verify config can be modified via env vars
4. **Type Handling**: Verify pipeline handles both array data and file paths
5. **Temporal Compositing**: Test multi-date composite generation

## Running Tests

```bash
# Run all integration tests
pytest tests/test_pipeline_integration_functional.py -v

# Run specific test class
pytest tests/test_pipeline_integration_functional.py::TestMinimalPipelineExecution -v

# Run with coverage
pytest tests/test_pipeline_integration_functional.py --cov=blackmarble.processing
```

## Notes

- Test data is intentionally small (100x100 pixels) for fast execution
- Focus is on data flow validation, not numerical accuracy
- Mock objects are used for external dependencies (S3, GDAL)
- Tests are designed to catch integration issues between pipeline phases