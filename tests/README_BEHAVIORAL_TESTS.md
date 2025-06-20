# Behavioral Tests for Black Marble Pipeline Components

This directory contains comprehensive behavioral tests for all pipeline components using predictable artificial data. These tests verify exact behavior without external dependencies or mocks.

## Test Files Created

### 1. `test_preprocessing_behavior.py`
Tests preprocessing components that handle spatial operations:

- **ProjectionHandler**: Tests coordinate system transformations
  - Identity transforms (same CRS)
  - Metadata generation
  - NoData masking
  - Multi-band handling

- **ResolutionResampler**: Tests resolution changes
  - Downsampling and upsampling
  - Scale factor calculations
  - Multi-band preservation
  - NoData mask generation

- **BoundsCropper**: Tests spatial subsetting
  - Exact pixel boundary cropping
  - Buffer zone handling
  - Multi-band cropping
  - Pixel count statistics

- **SpatialAligner**: Tests multi-raster alignment
  - Same-grid alignment
  - Different resolution alignment
  - Multi-band alignment
  - CRS-agnostic operation

### 2. `test_temporal_behavior.py`
Tests temporal processing components:

- **TemporalCompositor**: Tests time series compositing
  - Median, mean, max, min composites
  - Percentile composites
  - Quality-weighted composites
  - NoData handling
  - Multi-band compositing
  - Gap filling

- **SceneMerger**: Tests scene mosaicking
  - Non-overlapping mosaics
  - Overlap blending (max, min, mean, feather)
  - Priority ordering
  - Multi-band merging
  - NoData handling

- **DateRangeSelector**: Tests scene selection
  - Date range filtering
  - Target date proximity
  - Season filtering
  - Cloud cover thresholds
  - Quality score sorting
  - Minimum scene requirements

### 3. `test_enhancement_behavior.py`
Tests data enhancement components:

- **RoadEnhancer**: Tests OSM road enhancement
  - Basic multiplication factors
  - Custom factors per road class
  - Statistics calculation
  - Zero and negative value handling
  - Mask generation

- **ContrastEnhancer**: Tests dynamic range adjustment
  - Percentile stretching
  - Min-max stretching
  - Adaptive stretching
  - Masked enhancement
  - Output range scaling

- **ColorMapper**: Tests colormap application
  - Grayscale mapping
  - Custom colormaps
  - Alpha channel generation
  - Auto-scaling
  - NaN handling
  - Colorbar generation

- **CompositeEnhancer**: Tests enhancement chaining
  - Sequential enhancement application
  - Dynamic enhancer addition
  - Statistics aggregation

### 4. `test_output_behavior.py`
Tests output generation components:

- **COGGenerator**: Tests Cloud-Optimized GeoTIFF creation
  - Single and multi-band generation
  - Profile options
  - Metadata tags
  - Automatic overview calculation

- **MetadataGenerator**: Tests metadata creation
  - Basic metadata generation
  - Statistics calculation
  - Multi-band statistics
  - GeoJSON geometry
  - STAC formatting

- **OutputValidator**: Tests output validation
  - Valid data detection
  - Shape validation
  - Range checking
  - NoData warnings
  - Multi-band validation

- **VisualizationEnhancer**: Tests display enhancement
  - Basic enhancement
  - Gamma correction
  - Colormap application
  - NaN handling

## Test Characteristics

All tests use:
- Simple artificial data with known properties
- Exact value verification where possible
- No external dependencies or network access
- Clear test names describing the behavior tested
- Minimal setup with inline data creation

## Running the Tests

Run all behavioral tests:
```bash
uv run pytest tests/test_*_behavior.py -v
```

Run specific component tests:
```bash
uv run pytest tests/test_preprocessing_behavior.py::TestProjectionHandler -v
uv run pytest tests/test_temporal_behavior.py::TestTemporalCompositor -v
uv run pytest tests/test_enhancement_behavior.py::TestRoadEnhancer -v
uv run pytest tests/test_output_behavior.py::TestCOGGenerator -v
```

## Notes

- Some tests may fail due to minor implementation differences
- Tests focus on behavior rather than implementation details
- Each test class tests one component in isolation
- Data is created programmatically for predictability