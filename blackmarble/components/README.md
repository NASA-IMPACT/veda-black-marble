# Black Marble Pipeline Components

This directory contains modular, reusable components for the Black Marble pipeline. These components are designed to replace the monolithic functions in `pipeline.py` with testable, composable units.

## Overview

The components are organized into five main categories:

### 1. Preprocessing Components (`preprocessing.py`)

These handle spatial operations on raster data:

- **ProjectionHandler**: Transforms between coordinate reference systems (CRS)
  - Replaces `convert_to_wgs84()` from `reproj.py`
  - Supports any CRS transformation via EPSG codes
  - Handles resampling during reprojection

- **ResolutionResampler**: Changes raster resolution
  - Replaces resolution changes in `subset_raster_normal()`
  - Supports multiple resampling methods
  - Maintains spatial alignment

- **BoundsCropper**: Crops rasters to specified bounds
  - Replaces `crop_to_bb_utm()` functionality
  - Supports buffer zones
  - Handles pixel-perfect alignment

- **SpatialAligner**: Aligns multiple rasters to common grid
  - New functionality for multi-source alignment
  - Ensures consistent spatial reference across datasets

### 2. Temporal Components (`temporal.py`)

These handle time series processing:

- **TemporalCompositor**: Creates composites from time series
  - Replaces the median compositing logic in pipeline
  - Supports multiple methods: median, mean, max, percentile
  - Handles cloud-masked data with NaN values

- **SceneMerger**: Merges overlapping satellite scenes
  - Handles scene mosaicking with various blend methods
  - Supports feathering for seamless transitions

- **DateRangeSelector**: Selects scenes based on temporal criteria
  - Filters by date range, season, cloud cover
  - Prioritizes scenes by quality or temporal proximity

### 3. Quality Components (`quality.py`)

These handle data quality and masking:

- **CloudMasker**: Identifies and masks clouds
- **QualityCompositor**: Combines multiple quality masks
- **MaskStatistics**: Generates statistics from masks

### 4. Output Components (`output.py`)

These handle output generation and metadata:

- **COGGenerator**: Creates Cloud-Optimized GeoTIFFs
  - Replaces the COG generation logic in pipeline
  - Configurable compression, tiling, and overviews
  - Supports single and multi-band outputs

- **MetadataGenerator**: Generates metadata in various formats
  - Creates STAC items for data discovery
  - Includes processing provenance
  - Supports custom metadata fields

- **VisualizationEnhancer**: Prepares data for visualization
  - Applies colormaps with proper scaling
  - Handles display range adjustments
  - Supports gamma correction

- **OutputValidator**: Validates outputs before saving
  - Checks valid pixel percentages
  - Verifies value ranges
  - Ensures metadata consistency

### 5. Enhancement Components (`enhancement.py`)

These provide data enhancement capabilities:

- **RoadEnhancer**: Enhances values along road networks
  - Replaces OSM road enhancement logic in NDUI calculation
  - Configurable enhancement factors by road type
  - Provides enhancement statistics

- **ContrastEnhancer**: Dynamic range adjustment
  - Multiple methods: percentile, min-max, adaptive
  - Preserves data distribution characteristics
  - Handles masked regions properly

- **ColorMapper**: Applies colormaps for visualization
  - Supports custom colormap arrays
  - Optional transparency/alpha channel
  - Creates colorbar images

- **CompositeEnhancer**: Chains multiple enhancements
  - Combines enhancement operations in sequence
  - Maintains processing metadata
  - Enables complex enhancement pipelines

## Usage Examples

### Basic Projection Transformation

```python
from blackmarble.components import ProjectionHandler

# Convert from UTM to WGS84
handler = ProjectionHandler(target_crs="EPSG:4326")

kwargs = {
    "source_crs": "EPSG:32618",  # UTM Zone 18N
    "source_transform": transform,
    "nodata": -9999
}

result = handler.process(utm_data, **kwargs)

# Access transformed data
wgs84_data = kwargs["transformed_data"]
wgs84_transform = kwargs["transform"]
```

### Temporal Compositing

```python
from blackmarble.components import TemporalCompositor, CompositeMethod

# Create median composite from time series
compositor = TemporalCompositor(
    method=CompositeMethod.MEDIAN,
    min_valid_observations=3
)

# Process stacked time series (time, height, width)
result = compositor.process(time_series_data, dates=scene_dates)

# Access composite
composite = kwargs["composite"]
```

### Output Generation Pipeline

```python
from blackmarble.components import (
    RoadEnhancer, ContrastEnhancer, COGGenerator,
    MetadataGenerator, OutputMetadata
)

# Enhance NDUI with roads
road_enhancer = RoadEnhancer(major_factor=1.5, minor_factor=1.2)
enhanced = road_enhancer.enhance(ndui, roads)

# Apply contrast enhancement
contrast = ContrastEnhancer(method="percentile")
final_data = contrast.enhance(enhanced.data)

# Create output metadata
metadata = OutputMetadata(
    crs=projection,
    transform=geotransform,
    bounds=bounds,
    shape=ndui.shape,
    dtype=np.float32
)

# Generate Cloud-Optimized GeoTIFF
cog_gen = COGGenerator()
cog_path = cog_gen.generate(final_data.data, metadata, "output.tif")

# Generate STAC metadata
meta_gen = MetadataGenerator(metadata_format="stac")
stac_item = meta_gen.generate(
    final_data.data,
    metadata,
    title="Black Marble NDUI",
    processing_info={"enhancements": ["road", "contrast"]}
)
```

### Visualization Pipeline

```python
from blackmarble.components import (
    VisualizationEnhancer, ColorMapper
)

# Load inferno colormap
colormap_data = {
    'r': np.load("inferno_r.npy"),
    'g': np.load("inferno_g.npy"),
    'b': np.load("inferno_b.npy")
}

# Create visualization
viz = VisualizationEnhancer(colormap_arrays=colormap_data)
rgb_image = viz.apply_colormap(
    ndui,
    display_min=0.3,
    display_max=0.8
)

# Or use ColorMapper directly for more control
mapper = ColorMapper(colormap_data=colormap_data, add_alpha=True)
rgba_image = mapper.apply_colormap(ndui, scale_range=(0, 1))
```

## Integration with Existing Pipeline

The components can be gradually integrated into the existing pipeline. See `pipeline_integration_example.py` for a complete example of refactoring the output generation section.

### Example Refactoring

Current pipeline code:
```python
# Current approach in pipeline.py (lines ~1194-1237)
r = np.load(os.path.join(data_dir, "inferno_r.npy"))
g = np.load(os.path.join(data_dir, "inferno_g.npy"))
b = np.load(os.path.join(data_dir, "inferno_b.npy"))
img_scaled = scale_image(ndui, config.display_min, config.display_max)
rr = r[img_scaled]
gg = g[img_scaled]
bb = b[img_scaled]
# ... COG generation ...
```

With components:
```python
# Component-based approach
from blackmarble.components.pipeline_integration_example import (
    refactored_pipeline_output_section
)

out_cog = refactored_pipeline_output_section(
    ndui=ndui,
    roads=img_road,
    geotransform=gt,
    projection=proj,
    config=config,
    final_dir=final_dir,
)
```

## Benefits

1. **Testability**: Each component can be tested independently
2. **Reusability**: Components can be used in different pipelines
3. **Maintainability**: Clear separation of concerns
4. **Flexibility**: Easy to add new processing methods
5. **Documentation**: Self-documenting through docstrings and type hints
6. **Validation**: Built-in validation and quality checks
7. **Metadata**: Automatic metadata generation and tracking

## Testing

Run the test suite:
```bash
# Simple tests without external dependencies
uv run python blackmarble/components/test_simple.py

# Full integration tests (requires rasterio)
uv run python blackmarble/components/test_preprocessing.py

# Test output components
uv run python -m pytest blackmarble/components/test_output.py -v
```

## Future Enhancements

1. Add GPU acceleration for large-scale processing
2. Implement lazy evaluation for memory efficiency
3. Add more compositing methods (e.g., greenest pixel)
4. Support for cloud-optimized processing workflows
5. Integration with Dask for distributed processing
6. Support for additional output formats (NetCDF, Zarr)
7. Advanced colormap management with matplotlib integration