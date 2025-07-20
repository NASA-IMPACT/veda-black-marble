# Black Marble Pipeline → Test Mapping Document

This document provides a detailed mapping between each processing step in the Black Marble pipeline (`blackmarble/pipeline.py`) and the corresponding test coverage. For each non-I/O processing step, we show:

1. The exact function used in the pipeline
2. The test method that exercises this function
3. Verification that both use the same imported function

## Pipeline Processing Steps

### 1. VIIRS Processing Phase

#### Step 1.1: Crop VIIRS to Bounds
- **Pipeline Location**: Line ~140 in `pipeline.py`
- **Function Used**: `prepare.crop_to_bounds()`
- **Test Coverage**: `test_pipeline_flow.py::test_03_viirs_processing()`
- **Import Match**: ✅ Both import from `blackmarble.prepare.spatial`

```python
# Pipeline code:
viirs_cropped, viirs_transform = prepare.crop_to_bounds(
    viirs_data, viirs_profile["transform"], bbox
)

# Test code:
viirs_cropped, viirs_transform, viirs_bounds = crop_to_bounds(
    data=viirs_data,
    src_transform=viirs_profile["transform"],
    bounds=test_bbox
)
```

#### Step 1.2: Resample VIIRS to 12m
- **Pipeline Location**: Line ~147 in `pipeline.py`
- **Function Used**: `prepare.resample_image()`
- **Test Coverage**: `test_viirs_processing.py::test_viirs_resampling_to_12m()`
- **Import Match**: ✅ Both import from `blackmarble.prepare.spatial`

#### Step 1.3: Reproject VIIRS to Target CRS
- **Pipeline Location**: Line ~154 in `pipeline.py`
- **Function Used**: `prepare.reproject_image()`
- **Test Coverage**: `test_pipeline_flow.py::test_03_viirs_processing()`
- **Import Match**: ✅ Both import from `blackmarble.prepare.spatial`

### 2. Landsat Processing Phase

#### Step 2.1: Create Cloud Mask from QA_PIXEL
- **Pipeline Location**: Line ~195 in `pipeline.py`
- **Function Used**: `prepare.create_landsat_cloud_mask()`
- **Test Coverage**: `test_pipeline_flow.py::test_04_landsat_processing()`
- **Additional Tests**: `test_qa_masking.py` (comprehensive bit flag testing)
- **Import Match**: ✅ Both import from `blackmarble.prepare.landsat_qa`

```python
# Pipeline code:
cloud_mask = prepare.create_landsat_cloud_mask(
    qa_pixel, strategy=qa_strategy, dilate_pixels=3
)

# Test code:
cloud_mask = create_landsat_cloud_mask(
    qa_pixel=qa_pixel,
    strategy="moderate",
    dilate_pixels=3
)
```

#### Step 2.2: Apply Cloud Mask to Bands
- **Pipeline Location**: Line ~200-205 in `pipeline.py`
- **Function Used**: NumPy masking operations
- **Test Coverage**: `test_pipeline_flow.py::test_04_landsat_processing()`
- **Import Match**: ✅ Both use numpy directly

#### Step 2.3: Create Temporal Median Composite
- **Pipeline Location**: Line ~220 in `pipeline.py`
- **Function Used**: `analyze.create_landsat_median_composite()`
- **Test Coverage**: `test_pipeline_flow.py::test_05_temporal_compositing()`
- **Additional Tests**: `test_temporal_compositing.py` (edge cases)
- **Import Match**: ✅ Both import from `blackmarble.analyze.temporal`

```python
# Pipeline code:
composite = analyze.create_landsat_median_composite(
    band_stack,
    scale_factor=0.0000275,
    add_offset=-0.2
)

# Test code:
composite = create_landsat_median_composite(
    stack,
    scale_factor=0.0000275,
    add_offset=-0.2
)
```

#### Step 2.4: Reproject Landsat to Target CRS
- **Pipeline Location**: Line ~240 in `pipeline.py`
- **Function Used**: `prepare.reproject_image()`
- **Test Coverage**: `test_pipeline_flow.py::test_06_landsat_reprojection()`
- **Import Match**: ✅ Same function as VIIRS reprojection

### 3. Spatial Alignment Phase

#### Step 3.1: Align All Rasters to Common Grid
- **Pipeline Location**: Line ~270 in `pipeline.py`
- **Function Used**: `prepare.align_rasters()`
- **Test Coverage**: `test_pipeline_flow.py::test_07_spatial_alignment()`
- **Additional Tests**: `test_alignment.py` (comprehensive alignment scenarios)
- **Import Match**: ✅ Both import from `blackmarble.prepare.spatial`

```python
# Pipeline code:
aligned_rasters = prepare.align_rasters(
    [viirs_resampled, landsat_b3, landsat_b4, landsat_b5],
    [viirs_transform, b3_transform, b4_transform, b5_transform],
    reference_idx=reference_idx
)

# Test code:
aligned = align_rasters(
    [self.viirs_reprojected, self.landsat_reprojected["B5"]],
    [self.viirs_transform, self.reference_transform],
    reference_idx=1  # Use Landsat as reference
)
```

### 4. Road Processing Phase

#### Step 4.1: Fractional Road Rasterization
- **Pipeline Location**: Line ~310 in `pipeline.py`
- **Function Used**: `acquire.buffer_and_rasterize_roads_fractional()`
- **Test Coverage**: `test_pipeline_flow.py::test_08_road_rasterization()`
- **Additional Tests**: `test_osm_fractional_crs.py` (CRS edge cases)
- **Import Match**: ✅ Both import from `blackmarble.acquire.osm_fractional`

```python
# Pipeline code:
roads_fractional = acquire.buffer_and_rasterize_roads_fractional(
    graph=G,
    bbox=bbox,
    transform=reference_transform,
    crs=target_crs,
    shape=reference_shape,
    buffer_meters=30.0,
    sub_pixels=5
)

# Test code:
roads_fractional = buffer_and_rasterize_roads_fractional(
    graph=G,
    bbox=test_bbox,
    transform=self.reference_transform,
    crs=self.target_crs,
    shape=self.reference_shape,
    buffer_meters=30.0,
    sub_pixels=5
)
```

### 5. Index Calculation Phase

#### Step 5.1: Calculate NDVI
- **Pipeline Location**: Line ~340 in `pipeline.py`
- **Function Used**: `analyze.calculate_ndvi()`
- **Test Coverage**: `test_pipeline_flow.py::test_09_calculate_indices()`
- **Additional Tests**: `test_ndvi_critical_edge_cases.py` (near-zero denominators)
- **Import Match**: ✅ Both import from `blackmarble.analyze.indices`

#### Step 5.2: Calculate NDWI
- **Pipeline Location**: Line ~345 in `pipeline.py`
- **Function Used**: `analyze.calculate_ndwi()`
- **Test Coverage**: `test_pipeline_flow.py::test_09_calculate_indices()`
- **Import Match**: ✅ Both import from `blackmarble.analyze.indices`

#### Step 5.3: Calculate NDUI
- **Pipeline Location**: Line ~350 in `pipeline.py`
- **Function Used**: `analyze.calculate_ndui()`
- **Test Coverage**: `test_pipeline_flow.py::test_09_calculate_indices()`
- **Additional Tests**: `test_ndui_calculation.py` (water masking, parameter effects)
- **Import Match**: ✅ Both import from `blackmarble.analyze.indices`

### 6. Urban Field Enhancement Phase

#### Step 6.1: Compute Urban Fields
- **Pipeline Location**: Line ~370 in `pipeline.py`
- **Function Used**: `analyze.compute_urban_fields()`
- **Test Coverage**: `test_pipeline_flow.py::test_10_urban_field_enhancement()`
- **Additional Tests**: `test_urban_field_enhancement.py` (multi-scale behavior)
- **Import Match**: ✅ Both import from `blackmarble.analyze.urban_fields`

#### Step 6.2: Enhance NTL with Urban Field
- **Pipeline Location**: Line ~380 in `pipeline.py`
- **Function Used**: `analyze.enhance_ntl_with_urban_field()`
- **Test Coverage**: `test_pipeline_flow.py::test_10_urban_field_enhancement()`
- **Import Match**: ✅ Both import from `blackmarble.analyze.urban_fields`

### 7. Visualization Enhancement Phase

#### Step 7.1: Contrast Enhancement
- **Pipeline Location**: Line ~410 in `pipeline.py`
- **Function Used**: `enhance.enhance_contrast()`
- **Test Coverage**: `test_pipeline_flow.py::test_11_visualization_enhancement()`
- **Additional Tests**: `test_enhancement.py` (percentile/minmax methods)
- **Import Match**: ✅ Both import from `blackmarble.enhance.contrast`

#### Step 7.2: Apply Inferno Colormap
- **Pipeline Location**: Line ~420 in `pipeline.py`
- **Function Used**: `enhance.apply_inferno_colormap()`
- **Test Coverage**: `test_pipeline_flow.py::test_11_visualization_enhancement()`
- **Import Match**: ✅ Both import from `blackmarble.enhance.visualize`

### 8. Export Phase

#### Step 8.1: Create Metadata
- **Pipeline Location**: Line ~440 in `pipeline.py`
- **Function Used**: `export.create_metadata()`
- **Test Coverage**: `test_pipeline_flow.py::test_12_cog_export()`
- **Import Match**: ✅ Both import from `blackmarble.export.metadata`

#### Step 8.2: Export as COG
- **Pipeline Location**: Line ~450 in `pipeline.py`
- **Function Used**: `export.create_cog()`
- **Test Coverage**: `test_pipeline_flow.py::test_12_cog_export()`
- **Additional Tests**: `test_cog_export.py` (compression options, RGB/multiband)
- **Import Match**: ✅ Both import from `blackmarble.export.cog`

## Coverage Summary

### ✅ Fully Covered Processing Steps (16/18)
1. VIIRS cropping, resampling, reprojection
2. Landsat cloud masking and temporal compositing
3. Spatial alignment of all rasters
4. Fractional road rasterization
5. NDVI, NDWI, NDUI calculation
6. Urban field computation and NTL enhancement
7. Contrast enhancement and colormap application
8. COG export with metadata

### ⚠️ Gaps in Test Coverage (2/18)
1. **Legacy binary road rasterization** - `acquire.buffer_and_rasterize_roads()`
   - Used when `config.analysis.osm_fractional = False`
   - No dedicated test in pipeline flow

2. **Auxiliary exports** - Urban mask and indices multiband files
   - Pipeline exports these optionally
   - `test_auxiliary_exports.py` exists but not integrated into flow test

## Test File Organization

```
tests/pipeline/
├── test_pipeline_flow.py          # Main integration test mirroring pipeline
├── test_ndvi_critical_edge_cases.py    # Edge case: near-zero denominators
├── test_temporal_compositing.py   # Edge case: all-NaN pixels
├── test_qa_masking.py            # Comprehensive QA bit testing
├── test_ndui_calculation.py      # NDUI water masking
├── test_urban_field_enhancement.py # Multi-scale urban fields
├── test_alignment.py             # Raster alignment scenarios
├── test_enhancement.py           # Contrast methods
├── test_cog_export.py           # COG validation
├── test_viirs_processing.py     # VIIRS-specific tests
├── test_auxiliary_exports.py    # Urban mask/indices export
└── test_coverage_summary.md     # Test coverage analysis
```

## Verification Commands

To verify function identity between pipeline and tests:

```python
# In pipeline.py
print(f"Pipeline uses: {id(prepare.crop_to_bounds)}")

# In test_pipeline_flow.py  
print(f"Test uses: {id(crop_to_bounds)}")

# These IDs should match, confirming same function object
```

## Recommendations

1. **Add test for legacy binary road rasterization** in `test_pipeline_flow.py`
2. **Integrate auxiliary export tests** into the main flow test
3. **Add assertions for statistics** returned by pipeline (urban area km², mean radiance)
4. **Consider property-based testing** for index calculations with hypothesis

This mapping ensures that every processing function used in the production pipeline is tested with the exact same function signature and behavior.