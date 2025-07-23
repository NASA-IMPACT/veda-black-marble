# Pipeline Test Coverage Analysis

## Currently Tested Steps

### From test_pipeline_steps.py:
1. **Landsat Processing**
   - ✅ Landsat Collection 2 scaling edge cases
   - ✅ Values near -0.2 offset

2. **VIIRS Processing**
   - ✅ Basic reprojection from EPSG:4326 to UTM

3. **Road Rasterization**
   - ✅ CRS matching/mismatching cases
   - ✅ Fractional road coverage

4. **NDVI Calculation**
   - ✅ Edge cases with near-zero denominators
   - ✅ Landsat offset values (-0.2)
   - ✅ Critical denominator case (137.0 bug)

5. **NDWI Calculation**
   - ✅ Basic edge cases

6. **Integration**
   - ✅ Basic component integration

### From test_pipeline_ndvi_and_roads.py:
- ✅ Full pipeline test with mocked downloads
- ✅ VIIRS-only fallback path

## Missing Test Coverage (from pipeline.py analysis)

### 1. **Temporal Compositing** (Line ~350-400)
- ❌ `create_landsat_median_composite()` with multiple dates
- ❌ QA masking during compositing
- ❌ Handling missing dates/scenes

### 2. **QA/Quality Masking** (Line ~300-350)
- ❌ Cloud masking with QA_PIXEL bits
- ❌ Aerosol masking
- ❌ Radsat masking
- ❌ Mask statistics calculation

### 3. **Alignment** (Line ~450-500)
- ❌ `align_rasters()` with multiple rasters of different resolutions
- ❌ Handling shape mismatches
- ❌ Proper nodata handling during alignment

### 4. **NDUI Calculation** (Line ~835-892)
- ❌ NDUI formula with NTL and NDVI
- ❌ NTL floor/ceiling parameters
- ❌ NDUI floor parameter
- ❌ Handling when NDWI is provided vs None

### 5. **Urban Field Enhancement** (Line ~618-733)
- ❌ `compute_urban_fields()` with multi-scale sigmas
- ❌ `enhance_ntl_with_urban_field()` with different modes:
   - ❌ Multiplicative mode
   - ❌ Additive mode
   - ❌ Hybrid mode
- ❌ Binary roads converted to pseudo-fractional

### 6. **Legacy Burn Enhancement** (Line ~734-832)
- ❌ Legacy burn-in with major/minor road factors
- ❌ Fractional vs binary road handling

### 7. **Enhancement Phase** (Line ~898-1003)
- ❌ Contrast enhancement (percentile vs minmax)
- ❌ Road enhancement factors
- ❌ Urban feature enhancement
- ❌ False color composite creation
- ❌ Inferno colormap application

### 8. **COG Export** (Line ~1004-1140)
- ❌ COG creation with proper compression
- ❌ Metadata generation
- ❌ Validation checks
- ❌ Urban mask export
- ❌ Multi-band indices file export

### 9. **Diagnostics** (Throughout)
- ❌ Diagnostic file saving
- ❌ Statistics calculation (urban area, mean radiance)

### 10. **Error Handling**
- ❌ Missing data scenarios
- ❌ Shape mismatch errors
- ❌ CRS conversion failures
- ❌ Memory constraints

## Priority Test Additions

### High Priority (Core calculations):
1. **NDUI calculation** - Critical for final output
2. **Urban field enhancement** - Major processing step
3. **Temporal compositing** - Affects all Landsat processing
4. **QA masking** - Data quality critical

### Medium Priority (Enhancement/output):
5. **Contrast enhancement**
6. **COG export with validation**
7. **Alignment of multiple rasters**
8. **Legacy burn enhancement**

### Lower Priority (Optional features):
9. **Diagnostic saving**
10. **Statistics calculation**
11. **False color composite**
12. **Multi-band export**