# Black Marble Pipeline Flow Diagram

This diagram shows the data flow through the Black Marble pipeline, with references to test coverage.

```mermaid
graph TD
    %% Input Data
    A[VIIRS HDF5<br/>VNP46A2] --> B[Load & Convert to GeoTIFF]
    C[Landsat Collection 2<br/>B3, B4, B5, QA_PIXEL] --> D[Load Bands]
    E[OSM Road Network<br/>GeoPackage/Graph] --> F[Load Roads]
    
    %% VIIRS Processing
    B --> G[crop_to_bounds<br/>✅ test_03_viirs_processing]
    G --> H[resample_image<br/>to 12m<br/>✅ test_viirs_processing]
    H --> I[reproject_image<br/>to Target CRS<br/>✅ test_03_viirs_processing]
    
    %% Landsat Processing
    D --> J[create_landsat_cloud_mask<br/>✅ test_04_landsat_processing]
    J --> K[Apply Mask to Bands<br/>✅ test_04_landsat_processing]
    K --> L[create_landsat_median_composite<br/>✅ test_05_temporal_compositing]
    L --> M[reproject_image<br/>to Target CRS<br/>✅ test_06_landsat_reprojection]
    
    %% Spatial Alignment
    I --> N[align_rasters<br/>✅ test_07_spatial_alignment]
    M --> N
    
    %% Road Processing
    F --> O{Fractional?}
    O -->|Yes| P[buffer_and_rasterize_roads_fractional<br/>✅ test_08_road_rasterization]
    O -->|No| Q[buffer_and_rasterize_roads<br/>⚠️ NO TEST]
    
    %% Index Calculation
    N --> R[calculate_ndvi<br/>✅ test_09_calculate_indices]
    N --> S[calculate_ndwi<br/>✅ test_09_calculate_indices]
    N --> T[VIIRS Aligned]
    R --> U[calculate_ndui<br/>✅ test_09_calculate_indices]
    S --> U
    T --> U
    
    %% Urban Field Enhancement
    P --> V[compute_urban_fields<br/>✅ test_10_urban_field_enhancement]
    Q --> V
    V --> W[enhance_ntl_with_urban_field<br/>✅ test_10_urban_field_enhancement]
    T --> W
    
    %% Visualization
    W --> X[enhance_contrast<br/>✅ test_11_visualization_enhancement]
    X --> Y[apply_inferno_colormap<br/>✅ test_11_visualization_enhancement]
    
    %% Export
    Y --> Z[create_metadata<br/>✅ test_12_cog_export]
    Z --> AA[create_cog<br/>Main Output<br/>✅ test_12_cog_export]
    
    %% Auxiliary Exports
    U --> AB[create_cog<br/>Urban Mask<br/>⚠️ test_auxiliary_exports]
    R --> AC[create_cog<br/>Indices Stack<br/>⚠️ test_auxiliary_exports]
    S --> AC
    U --> AC
    
    %% Statistics
    U --> AD[Calculate Statistics<br/>⚠️ NO TEST]
    W --> AD
    
    %% Legend
    style A fill:#e1f5fe
    style C fill:#e1f5fe
    style E fill:#e1f5fe
    style AA fill:#c8e6c9
    style AB fill:#fff9c4
    style AC fill:#fff9c4
```

## Legend

- 🔵 **Input Data** (Light Blue)
- 🟢 **Main Output** (Light Green) 
- 🟡 **Auxiliary Outputs** (Light Yellow)
- ✅ **Tested Function** - Function is tested in the specified test file
- ⚠️ **Gap in Testing** - Function needs test coverage

## Processing Phases

### Phase 1: Data Loading (I/O)
- Load VIIRS from HDF5 or GeoTIFF
- Load Landsat bands (B3, B4, B5, QA_PIXEL)
- Load OSM road network

### Phase 2: Preprocessing
- **VIIRS**: Crop → Resample → Reproject
- **Landsat**: QA Mask → Apply Mask → Temporal Composite → Reproject
- **Roads**: Buffer and Rasterize (fractional or binary)

### Phase 3: Analysis
- Calculate spectral indices (NDVI, NDWI)
- Calculate NDUI (urban index)
- Compute urban fields from roads
- Enhance NTL with urban fields

### Phase 4: Visualization
- Enhance contrast (percentile stretching)
- Apply colormap (Inferno)

### Phase 5: Export
- Create metadata
- Export main COG (RGB visualization)
- Export auxiliary products (urban mask, indices)
- Calculate statistics

## Test Coverage by Phase

| Phase | Coverage | Missing Tests |
|-------|----------|---------------|
| Data Loading | N/A (I/O) | - |
| Preprocessing | 90% | Binary road rasterization |
| Analysis | 100% | - |
| Visualization | 100% | - |
| Export | 67% | Auxiliary exports, Statistics |

## Key Test Files

1. **test_pipeline_flow.py** - Main integration test following the exact pipeline flow
2. **test_ndvi_critical_edge_cases.py** - Tests the NDVI near-zero denominator bug fix
3. **test_qa_masking.py** - Comprehensive Landsat QA bit interpretation
4. **test_alignment.py** - Multi-raster spatial alignment scenarios
5. **test_urban_field_enhancement.py** - Multi-scale urban field computation
6. **test_cog_export.py** - COG format validation

## Function Import Verification

All test files import the **exact same functions** used in the pipeline:

```python
# Pipeline imports
from blackmarble.prepare.spatial import crop_to_bounds, reproject_image, align_rasters
from blackmarble.analyze.indices import calculate_ndvi, calculate_ndwi, calculate_ndui

# Test imports (identical)
from blackmarble.prepare.spatial import crop_to_bounds, reproject_image, align_rasters
from blackmarble.analyze.indices import calculate_ndvi, calculate_ndwi, calculate_ndui
```

This ensures that tests exercise the actual production code, not mocked versions.