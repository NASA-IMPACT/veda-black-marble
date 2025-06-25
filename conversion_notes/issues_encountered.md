# Issues Encountered in bm.py

This document catalogs issues found in the Black Marble pipeline implementation (`docker/bm.py`).

## Critical Bugs

### 1. Path Range Bug (Line 630)

```python
# Line 630 - INCORRECT:
for p in range(min(path1, path2), max(path1, path1) + 1 ):
                                      ^^^^^^^^^^^^
                                   Should be: max(path1, path2)
```

**Impact**: Only downloads Landsat data from one path when the bounding box spans multiple paths, resulting in missing data on the eastern side of the area.

**Example**: If path1=15 and path2=17, only path 15 is processed instead of paths 15, 16, and 17.

### 2. QA File Naming Mismatch (Lines 645, 652, 659)
**Severity**: MEDIUM - File management issues

```python
# Local file names (MISSING UNDERSCORES):
local_path = "outputs/temp/" + dir.split('/')[-2] + "QA_PIXEL.TIF"      # Line 645
local_path = "outputs/temp/" + dir.split('/')[-2] + "QA_RADSAT.TIF"     # Line 652
local_path = "outputs/temp/" + dir.split('/')[-2] + "QA_AEROSOL.TIF"    # Line 659

# But S3 files have underscores:
band_file = dir + dir.split('/')[-2] + "_QA_PIXEL.TIF"      # Line 362
band_file = dir + dir.split('/')[-2] + "_QA_RADSAT.TIF"     # Line 366
band_file = dir + dir.split('/')[-2] + "_SR_QA_AEROSOL.TIF" # Line 370
```

**Impact**: Files are saved locally without underscores but downloaded from S3 with underscores, potentially causing file tracking issues.

### 3A. VIIRS Quality Flag Boolean Logic Error (Line 1064)
**Severity**: HIGH - Incorrect data filtering

```python
# Line 1064 - BROKEN BOOLEAN LOGIC:
adjusted_ntl = img_ntl * ((mandatory_quality_flag == 0) & ~(mandatory_quality_flag == 2)).astype(float) - img_lunar_irradiance
```

**Issue**: The condition `(flag == 0) & ~(flag == 2)` is logically equivalent to just `(flag == 0)` because if flag equals 0, it cannot equal 2.

**Intended behavior** (based on VNP46A2 v2.0 documentation):
- Keep pixels with flag = 0 (high quality)
- Keep pixels with flag = 2 (twilight conditions, solar zenith angle 102-108°)

**VNP46A2 v2.0 Quality Flag Meanings**:
- 0 = High quality
- 1 = Poor quality
- 2 = Twilight (solar zenith angle 102-108°)
- 3 = Lunar eclipse
- 4 = Aurora
- 5 = Glint

### 3B. VIIRS Quality Masking Creates False Negative Values (Line 1064)
**Severity**: CRITICAL - Creates false data artifacts

```python
# Line 1064 - CREATES FALSE NEGATIVES:
adjusted_ntl = img_ntl * ((mandatory_quality_flag == 0) & ~(mandatory_quality_flag == 2)).astype(float) - img_lunar_irradiance
```

**Issue**: When quality flags are NOT 0 or 2 (i.e., poor quality pixels), the current implementation creates artificial negative values:

1. Quality mask evaluates to 0 (False)
2. `img_ntl * 0 = 0` (nighttime radiance is zeroed)
3. `0 - img_lunar_irradiance = negative value` (but lunar irradiance is still subtracted!)
4. Result: Artificial negative spikes wherever quality is poor

**Example**: A pixel with quality flag = 3 (lunar eclipse) and lunar_irradiance = 5:
- Current result: `0 - 5 = -5` (nonsensical negative radiance)
- Expected result: Should be masked out entirely or set to 0/NaN

**Correct implementations**:
```python
# Option 1: Apply mask to both values
quality_mask = ((mandatory_quality_flag == 0) | (mandatory_quality_flag == 2)).astype(float)
adjusted_ntl = (img_ntl - img_lunar_irradiance) * quality_mask

# Option 2: Set both to zero for bad pixels
quality_mask = ((mandatory_quality_flag == 0) | (mandatory_quality_flag == 2)).astype(float)
adjusted_ntl = (img_ntl * quality_mask) - (img_lunar_irradiance * quality_mask)

# Option 3: Use NaN for invalid pixels
adjusted_ntl = img_ntl - img_lunar_irradiance
adjusted_ntl = np.where((mandatory_quality_flag == 0) | (mandatory_quality_flag == 2), 
                       adjusted_ntl, np.nan)
```

**Impact**: False negative values propagate through NDUI calculation, potentially creating artifacts in the final urban index map.

### 4. Road Value Reference Error (Lines 1221-1222)
**Severity**: LOW - Logic error with no effect

```python
# Line 1222 - References non-existent value 6:
ndui = np.where((img_road == 2)|(img_road == 3)|(img_road == 5)|(img_road == 6), ndui*1.1, ndui)
                                                                   ^^^^^^^^^^^^^^
```

**Issue**: OSM road raster only contains values 1-5, but the code checks for value 6.

**Road value mapping**:
- 1 = primary/secondary/tertiary roads
- 2 = motorway
- 3 = residential
- 4 = trunk
- 5 = service/unclassified

## Design Issues

### 5. VIIRS Tile Calculation (Lines 517-525)
**Severity**: MEDIUM - May download wrong tiles

```python
def coordinates_to_tile_id(self, lat, longi):
    lat = -lat + 90.0
    longi = longi + 180.0
    
    tile_v = (lat / 180) * 18
    tile_h = (longi / 360) * 36
    
    return (int(tile_v), int(tile_h))
```

**Issue**: This appears to use MODIS sinusoidal grid (18x36) rather than the correct VIIRS grid system.

### 6. Arbitrary Resolution Upsampling (Line 1050)
**Severity**: LOW - Potential interpolation artifacts

```python
scale = 0.00011111111  # ~12.3m at equator
```

**Issue**: VIIRS native resolution is 500m. Upsampling to ~12m creates interpolated data that may give false impression of detail.

### 7. Final NDUI Scaling (Line 1224)
**Severity**: LOW - Reduces dynamic range

```python
ndui = ndui / 1.3
```

**Issue**: After applying road boosts up to 1.3x, dividing by 1.3 reduces the maximum possible NDUI value to ~0.77 instead of 1.0, compressing the output range unnecessarily.

## Error Handling Issues

### 8. Generic Exception Handling (Line 502)
**Severity**: MEDIUM - Hides errors

```python
except:
    print("No data for filter " + filter)
```

**Issue**: Catches all exceptions, potentially hiding serious errors like network failures or permission issues.

### 9. No Multi-Tile Support for VIIRS (Line 1040)
**Severity**: MEDIUM - Incomplete coverage for large areas

```python
vertical, horizontal = vnp46Util.coordinates_to_tile_id(lat1, long1)
```

**Issue**: Only uses the first coordinate to determine tile, missing data if bounding box spans multiple VIIRS tiles.

## Critical Calculation Errors

### 11. Date Selection Logic Inverted (Line 348)
**Severity**: CRITICAL - Selects wrong Landsat scenes

```python
# Line 348 - INVERTED LOGIC:
if selected_start_date < start_date and selected_end_date > end_date:
```

**Issue**: This selects a date range that is WIDER than the previously selected one, not narrower.

**Impact**: May select Landsat scenes that don't optimally cover the target date.

**Correct implementation**:
```python
if selected_start_date >= start_date and selected_end_date <= end_date:
```

### 12. NDUI Global Normalization Error (Line 1224)
**Severity**: CRITICAL - Incorrectly reduces all pixel values

```python
# Line 1224 - AFFECTS ALL PIXELS:
ndui = ndui / 1.3
```

**Issue**: After applying road boosts up to 1.3x, this divides ALL pixels by 1.3, not just the boosted ones.

**Impact**: Non-road urban areas are reduced to ~77% of their correct intensity.

**Correct approach**: Only normalize pixels that were boosted, or remove this line entirely.

### 13. Median Compositing Missing fill_gaps (Lines 1016-1018)
**Severity**: HIGH - Runtime error with different sized arrays

```python
# Lines 1016-1018 - WILL CRASH WITH DIFFERENT SHAPES:
b3 = np.nanmedian(np.where(np.stack(b3_pxs) == 0, np.nan, np.stack(b3_pxs)), axis=0)
b4 = np.nanmedian(np.where(np.stack(b4_pxs) == 0, np.nan, np.stack(b4_pxs)), axis=0)
b5 = np.nanmedian(np.where(np.stack(b5_pxs) == 0, np.nan, np.stack(b5_pxs)), axis=0)
```

**Issue**: np.stack requires all arrays to have the same shape. Different Landsat scenes may have slightly different dimensions.

**Impact**: ValueError crash when processing areas with mixed scene sizes.

**Fix**: Use the existing fill_gaps function before stacking.

### 14. rescale_bands Creates Invalid Negative Values (Lines 1090-1093)
**Severity**: HIGH - Creates physically impossible reflectance values

```python
def rescale_bands(band_data):
    band_data = np.where(band_data < 7273, 0, band_data)
    band_data = np.where(band_data > 43636, 0, band_data)
    return band_data * 0.0000275 - 0.2
```

**Issue**: Out-of-range values are set to 0, then transformed to -0.2 (negative reflectance).

**Impact**: Creates physically impossible negative reflectance values for masked/invalid pixels.

**Better approach**:
```python
band_data = band_data * 0.0000275 - 0.2
band_data = np.where((band_data < 0) | (band_data > 1), 0, band_data)
```

## Data Processing Issues

### 15. Inconsistent Epsilon Values
**Severity**: MEDIUM - Potential division issues

```python
# NDVI/NDWI use 6 zeros:
ndvi = (b5_final_px - b4_final_px) / (b5_final_px + b4_final_px + 0.000001)

# NDUI uses 5 zeros:
ndui = (img_ntl - img_ndvi) / (img_ntl + img_ndvi + 0.00001)
```

**Issue**: Inconsistent epsilon values suggest a typo. NDUI has 10x larger epsilon.

### 16. scale_image Division by Zero Risk (Lines 94-99)
**Severity**: HIGH - Potential crash

```python
new_arr = ((inputarry - inputarry.min()) * (1/(inputarry.max() - inputarry.min()) * 255)).astype('uint8')
```

**Issue**: If all values are identical, max - min = 0, causing division by zero.

**Additional issue**: Uses array min/max AFTER clamping, not the provided min_value/max_value parameters.

### 17. check_values Function Counterintuitive (Lines 942-943)
**Severity**: MEDIUM - Confusing code maintenance

```python
def check_values(values_to_check, array):
    return (~np.isin(values_to_check, array)).astype(float)
```

**Issue**: Returns 1.0 for values NOT in array, opposite of what the name suggests.

**Impact**: Makes code hard to understand and maintain.

### 18. Zero Values Indistinguishable from Masked (Lines 1003-1005)
**Severity**: MEDIUM - Data loss

```python
b3_px = b3_px * cloud_pixels * aerosol_pixels * b3_r_px
```

**Issue**: Multiplying by mask (0 or 1) makes legitimate zero values indistinguishable from masked values.

**Impact**: Valid pixels with value 0 are treated as invalid in median compositing.

### 19. Composite Download Off-by-One (Lines 673-677)
**Severity**: MEDIUM - Duplicate data download

```python
for comp in range(composite_history_months):
    if month <= comp:
        t = download_tiles_for_band(band, year - 1, 12 - comp + month, 1, ...)
```

**Issue**: When comp=0, downloads current month with day=1, duplicating the target date's month.

**Fix**: Start loop from `range(1, composite_history_months)`

### 20. Missing WRS Result Validation (Lines 617-623)
**Severity**: MEDIUM - Potential crash

```python
wrs_res1 = landsat.get_wrs(lat1, long1)
path1 = wrs_res1[0]['path']  # Assumes result exists
```

**Issue**: No check if get_wrs returns empty list (e.g., for ocean coordinates).

**Impact**: IndexError crash for invalid coordinates.

### 21. S3 Contents Key Missing Check (Line 385)
**Severity**: MEDIUM - Potential crash

```python
)['Contents']
```

**Issue**: S3 returns no 'Contents' key when no objects match the prefix.

**Impact**: KeyError when trying to download non-existent files.

## Performance Issues

### 22. Inefficient Road Mask Merging (Lines 497-500)
**Severity**: LOW - O(n²) complexity

```python
for row in range(height):
    for col in range(width):
        if mask[row][col] > base[row][col]:
            base[row][col] = mask[row][col]
```

**Better approach**:
```python
base = np.maximum(base, mask)
```

### 23. Unnecessary Array Copies (Lines 1007-1009)
**Severity**: LOW - Memory waste

```python
b3_pxs.append(np.array(b3_px))  # b3_px is already an array!
```

**Issue**: Creates redundant array copies when data is already an array.

### 24. Shapefile Loaded Multiple Times (Line 615)
**Severity**: LOW - I/O inefficiency

```python
landsat = LandsatUtil(shapefile = "./WRS2_descending_0/WRS2_descending.shp")
```

**Issue**: Shapefile is loaded from disk for each band (3+ times per run).

## Other Issues

### 25. Temporary File Race Conditions (Lines 175-187)
**Severity**: MEDIUM - Parallel execution issues

```python
ds = gdal.Translate('new1.tif', inputRaster, ...)
ds = gdal.Warp('new2.tif', ds, ...)
```

**Issue**: Fixed filenames could conflict if multiple instances run simultaneously.

### 26. UTM Zone Ignored (Lines 767-768)
**Severity**: LOW - Projection issues at boundaries

```python
x1, y1, _, _ = get_utm_zone(lat1,long1)  # Ignores zone and hemisphere
```

**Issue**: Could cause problems when bounding box spans UTM zones.

### 27. NDUI abs() Unnecessary (Line 1217)
**Severity**: TRIVIAL - Unnecessary operation

```python
ndui = abs((ndui+1.0))/2.0
```

**Issue**: For ndui in [-1,1], (ndui+1.0) is always >= 0, making abs() redundant.

## Recommendations

1. **Immediate fixes needed**:
   - Fix path range bug (line 630)
   - Fix quality flag logic (line 1064)
   - Fix QA file naming consistency

2. **Important corrections**:
   - Verify and fix VIIRS tile calculation
   - Add multi-tile support for VIIRS
   - Improve error handling with specific exceptions

3. **Quality improvements**:
   - Document quality flag meanings in code
   - Make hardcoded parameters configurable
   - Add validation for environment variables
   - Consider removing unnecessary NDUI scaling