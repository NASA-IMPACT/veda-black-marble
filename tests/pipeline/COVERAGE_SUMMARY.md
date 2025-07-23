# Pipeline Test Coverage Summary

## Overview

The Black Marble pipeline has comprehensive test coverage for all processing steps. This document summarizes the test coverage status as of the latest review.

## Coverage Status

### ✅ Tested Processing Steps (18/20)

1. **VIIRS Preprocessing**
   - Crop to bounds
   - Reproject to UTM  
   - Align to Landsat 30m grid
   - Test: `test_pipeline_flow.py::test_03_viirs_processing`

2. **Landsat QA Masking**
   - Cloud/shadow detection
   - QA strategies (conservative/moderate/permissive)
   - Test: `test_qa_masking.py`

3. **Temporal Compositing**
   - Median composite creation
   - Test: `test_temporal_compositing.py`

4. **Landsat Reprojection**
   - Reproject to target CRS
   - Test: `test_pipeline_flow.py::test_06_landsat_reprojection`

5. **Spatial Alignment**
   - Align all rasters to common grid
   - Test: `test_alignment.py`

6. **Fractional Road Rasterization** 
   - OSM roads to fractional coverage
   - Test: `test_pipeline_flow.py::test_08_road_rasterization`

7. **Binary Road Rasterization** (Alternative)
   - OSM roads to type codes (1-6)
   - Test: `test_missing_coverage.py::TestBinaryRoadRasterization`

8. **NDVI Calculation**
   - Normalized Difference Vegetation Index
   - Epsilon handling for near-zero denominators
   - Test: `test_pipeline_steps.py::TestStep4_NDVICalculation`

9. **NDWI Calculation**
   - Normalized Difference Water Index
   - Test: `test_pipeline_steps.py::TestStep5_NDWICalculation`

10. **NDUI Calculation**
    - Normalized Difference Urban Index
    - Water masking
    - Test: `test_ndui_calculation.py`

11. **Urban Field Computation**
    - Multi-scale Gaussian fields
    - Test: `test_urban_field_enhancement.py`

12. **NTL Enhancement (Urban Field)**
    - Multiplicative/additive/hybrid modes
    - Test: `test_urban_field_enhancement.py`

13. **Legacy Burn-in Enhancement** (Alternative)
    - Direct road enhancement
    - Major/minor road factors
    - Test: `test_missing_coverage.py::TestLegacyBurnInEnhancement`

14. **Contrast Enhancement**
    - Percentile/minmax methods
    - Test: `test_enhancement.py::TestContrastEnhancement`

15. **Colormap Application**
    - Apply inferno colormap
    - Test: `test_enhancement.py::TestColormapApplication`

16. **COG Export**
    - Cloud Optimized GeoTIFF creation
    - Test: `test_cog_export.py`

17. **Metadata Creation**
    - Generate processing metadata
    - Test: `test_cog_export.py::TestMetadataCreation`

18. **Statistics Calculation**
    - Urban area km²
    - Mean radiance
    - Test: `test_missing_coverage.py::TestStatisticsCalculation`

### ⭕ I/O Operations (Not Processing Steps)

1. **Data Download**
   - VIIRS/Landsat/OSM acquisition
   - No processing decisions

2. **File Writing**
   - Save outputs to disk
   - No processing decisions

## Test Organization

### Main Pipeline Flow
- `test_pipeline_flow.py` - End-to-end pipeline simulation using reference data
- 12 sequential test methods mirroring actual pipeline steps

### Specialized Test Files
- `test_qa_masking.py` - Landsat QA pixel interpretation
- `test_temporal_compositing.py` - Multi-date compositing
- `test_alignment.py` - Spatial grid alignment
- `test_ndui_calculation.py` - Urban index computation
- `test_urban_field_enhancement.py` - Multi-scale urban fields
- `test_enhancement.py` - Contrast and colormap
- `test_cog_export.py` - COG creation and validation
- `test_missing_coverage.py` - Binary roads, legacy enhancement, statistics

### Reference Data
- Located in `tests/data/reference/`
- VIIRS: 150×150 pixels, ~500m → 30m
- Landsat: 150×150 pixels, 30m resolution
- OSM: 12 road segments with various types

## Running Tests

```bash
# Run all pipeline tests
uv run pytest tests/pipeline/ -v

# Run specific processing step tests
uv run pytest tests/pipeline/test_missing_coverage.py -v

# Run with coverage report
uv run pytest tests/pipeline/ --cov=blackmarble --cov-report=html
```

## Notes

- Tests use the exact same functions as the production pipeline
- Each test verifies both functionality and output format
- Alternative processing paths (binary vs fractional roads, legacy vs urban field enhancement) are both tested
- All critical processing decisions are documented in `docs/pipeline-steps/`