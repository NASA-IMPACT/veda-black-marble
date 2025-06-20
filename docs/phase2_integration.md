# Phase 2 Adapter Integration

This document describes how the Phase 2 adapters have been integrated into the Black Marble pipeline.

## Overview

The Phase 2 adapters provide abstraction layers for external services, enabling:
- Better testability through dependency injection
- Easier mocking of external dependencies
- Flexibility to swap implementations
- Backward compatibility with existing code

## Key Changes

### 1. Pipeline Function Signature

The `run_pipeline` function now accepts optional adapter parameters:

```python
def run_pipeline(
    # ... existing parameters ...
    s3_adapter: Optional[S3Adapter] = None,
    raster_adapter: Optional[GDALRasterAdapter] = None,
    landsat_source: Optional[LandsatDataSource] = None,
    viirs_source: Optional[VIIRSDataSource] = None,
) -> str:
```

### 2. Adapter Usage

The pipeline now uses adapters when provided, falling back to legacy methods when not:

- **S3 Operations**: Upload operations use `S3Adapter` if provided
- **Raster I/O**: File reading/writing uses `GDALRasterAdapter` if provided
- **Data Sources**: Landsat and VIIRS fetching uses data sources if provided

### 3. Backward Compatibility

The pipeline maintains full backward compatibility:
- All adapter parameters are optional
- Legacy functions are called when adapters are not provided
- Existing code continues to work without changes

## Usage Examples

### Using with Adapters (Recommended)

```python
from blackmarble.pipeline import run_pipeline
from blackmarble.adapters.s3 import S3Adapter
from blackmarble.adapters.raster_io import GDALRasterAdapter
from blackmarble.data_sources.landsat import LandsatDataSource
from blackmarble.data_sources.viirs import VIIRSDataSource

# Initialize adapters
s3_adapter = S3Adapter()
raster_adapter = GDALRasterAdapter()

# Initialize data sources
landsat_source = LandsatDataSource(
    storage_adapter=s3_adapter,
    raster_adapter=raster_adapter
)

viirs_source = VIIRSDataSource(
    raster_adapter=raster_adapter,
    earthdata_token=token
)

# Run pipeline with adapters
output = run_pipeline(
    lat1=37.7749, lon1=-122.4194,
    lat2=37.8049, lon2=-122.3894,
    year=2023, month=6, day=15,
    earthdata_token=token,
    s3_adapter=s3_adapter,
    raster_adapter=raster_adapter,
    landsat_source=landsat_source,
    viirs_source=viirs_source
)
```

### Legacy Usage (Still Supported)

```python
# Existing code continues to work
output = run_pipeline(
    lat1=37.7749, lon1=-122.4194,
    lat2=37.8049, lon2=-122.3894,
    year=2023, month=6, day=15,
    earthdata_token=token
)
```

## Testing Benefits

With adapters, testing becomes much easier:

```python
# Create mock adapters
mock_s3 = MockS3Adapter()
mock_raster = MockRasterAdapter()

# Run pipeline with mocks - no external calls!
output = run_pipeline(
    # ... parameters ...
    s3_adapter=mock_s3,
    raster_adapter=mock_raster,
    landsat_source=mock_landsat,
    viirs_source=mock_viirs
)

# Verify behavior
assert len(mock_s3.uploaded_files) == 1
assert len(mock_raster.written_files) > 0
```

## Implementation Details

### Conditional Adapter Usage

Throughout the pipeline, adapters are used conditionally:

```python
if raster_adapter is not None:
    try:
        data, metadata = raster_adapter.read(Path(file_path))
        # Process with adapter
    except Exception as e:
        # Fall back to legacy method
        data, gt, proj = get_geo(file_path, 1)
else:
    # Use legacy method directly
    data, gt, proj = get_geo(file_path, 1)
```

### Data Source Integration

When data sources are provided, they replace the legacy download logic:

```python
if landsat_source is not None:
    # Use new data source
    landsat_data = landsat_source.fetch(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        bands=["B3", "B4", "B5", "QA_PIXEL"]
    )
else:
    # Use legacy process_band_data function
    b3_out, b3_p, b3_r, b3_a = process_band_data(...)
```

## Future Improvements

1. **Complete Data Source Migration**: Fully implement mosaicking in data sources
2. **OSM Adapter**: Create an adapter for OpenStreetMap operations
3. **Configuration**: Allow adapter configuration through ProcessingConfig
4. **Async Support**: Add async versions of adapters for better performance
5. **Caching**: Implement caching in adapters to reduce redundant operations

## Migration Guide

To migrate existing code to use adapters:

1. Install the updated package
2. Optionally create adapter instances
3. Pass adapters to `run_pipeline`
4. No other changes needed!

The integration is designed to be gradual - you can adopt adapters at your own pace while maintaining existing functionality.