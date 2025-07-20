# Pipeline Test Coverage Summary

## Test Files Created/Updated

### 1. **test_pipeline_steps.py**
- Tests NDVI edge cases that caused 137.0 bug
- Tests road rasterization with CRS matching/mismatching
- Tests Landsat scaling that produces near-zero denominators
- Basic integration test

### 2. **test_temporal_compositing.py**
- Tests median compositing with multiple dates
- Handles all-NaN pixels correctly
- Tests different array shapes
- Tests scaling and offset application

### 3. **test_qa_masking.py**
- Tests QA pixel bit interpretation
- Tests conservative/moderate/permissive strategies
- Tests cloud mask dilation
- Tests consistent masking across bands

### 4. **test_ndui_calculation.py**
- Tests NDUI basic calculation
- Tests water masking with NDWI
- Tests floor/ceiling parameters
- Tests edge cases and NaN handling

### 5. **test_urban_field_enhancement.py**
- Tests multi-scale urban field computation
- Tests NTL enhancement modes (multiplicative, additive, hybrid)
- Tests handling of zero NTL values
- Tests preservation of spatial patterns

### 6. **test_alignment.py**
- Tests aligning rasters with same/different resolutions
- Tests nodata preservation
- Tests reference raster selection
- Tests VIIRS/Landsat/roads alignment

### 7. **test_enhancement.py**
- Tests contrast enhancement (percentile, minmax)
- Tests inferno colormap application
- Tests road enhancement with factors
- Tests urban feature enhancement

### 8. **test_cog_export.py**
- Tests basic COG creation
- Tests RGB and multiband COGs
- Tests overviews and metadata
- Tests compression options

### 9. **test_viirs_processing.py** (NEW)
- Tests VIIRS cropping to bounds
- Tests resampling to 12m resolution
- Tests non-zero pixel validation
- Tests fallback when no Landsat available

### 10. **test_legacy_burn_enhancement.py** (NEW)
- Tests legacy burn-in with binary roads
- Tests fractional road enhancement
- Tests preservation of zero values
- Tests enhancement statistics

### 11. **test_statistics_calculation.py** (NEW)
- Tests urban area calculation in km²
- Tests mean radiance calculation
- Tests handling of NaN values
- Tests edge cases (all zero, all NaN)

### 12. **test_auxiliary_exports.py** (NEW)
- Tests urban mask COG export
- Tests multi-band indices export
- Tests metadata preservation
- Tests compression constraints

## Coverage Analysis

### ✅ Covered Pipeline Calculations:
1. Landsat scaling and QA masking
2. Temporal median compositing
3. Raster reprojection and alignment
4. NDVI/NDWI calculation with epsilon fix
5. Road rasterization (fractional and binary)
6. NDUI calculation
7. Urban field computation and enhancement
8. Contrast enhancement and colormap
9. COG export with validation
10. VIIRS processing and fallback logic
11. Legacy burn enhancement
12. Statistics calculation
13. Auxiliary product exports

### ❌ Still Missing:
1. Diagnostic file saving (behind config flag)
2. Error handling for missing Earthdata token
3. Download failure handling
4. Full end-to-end async pipeline test

### 🔧 Test Fixes Applied:
1. Fixed temporal compositing to use correct function signature
2. Fixed QA masking tests to match actual bit logic
3. Fixed NDUI tests to match actual formula
4. Fixed urban field tests to include direct weight
5. Fixed alignment tests to use correct parameters and return types

## Running the Tests

```bash
# Run all pipeline tests
uv run pytest tests/pipeline/ -v

# Run specific test categories
uv run pytest tests/pipeline/test_ndvi_critical_edge_cases.py -v
uv run pytest tests/pipeline/test_statistics_calculation.py -v

# Run with coverage
uv run pytest tests/pipeline/ --cov=blackmarble --cov-report=html
```

## Key Test Patterns

1. **Edge Case Testing**: Each calculation has tests for edge cases (zeros, NaN, extreme values)
2. **Integration Testing**: Tests verify components work together (e.g., alignment → NDUI)
3. **Realistic Data**: Tests use realistic value ranges and data patterns
4. **Error Testing**: Tests verify proper error handling and messages

## Recommendations

1. Add async download failure tests
2. Add diagnostic file I/O tests  
3. Add full pipeline integration test with real downloaded data
4. Set up continuous integration to run these tests on every commit