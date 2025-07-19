# VIIRS South-Up Raster Bug Fix

## The Bug
VIIRS TIF files were being generated with **positive Y pixel size**, creating "south-up" rasters that gdalbuildvrt rejected with:
```
Warning 1: gdalbuildvrt does not support positive NS resolution. Skipping...
```

## Root Cause
In `blackmarble/acquire/viirs.py`, line 528 had incorrect parameter order:
```python
outputBounds=[west, south, east, north],  # WRONG!
```

GDAL's `outputBounds` expects `[ulx, uly, lrx, lry]` (Upper Left X/Y, Lower Right X/Y), but the code was passing the SOUTH latitude as the upper Y coordinate, causing GDAL to create south-up rasters.

## The Fix
Changed line 528 to:
```python
outputBounds=[west, north, east, south],  # CORRECT: [ulx, uly, lrx, lry]
```

Also added a guard rail to prevent future south-up rasters:
```python
if geotransform[5] > 0:  # Y pixel size should be negative
    os.remove(output_file)
    raise RuntimeError(
        f"Generated south-up raster with positive Y pixel size ({geotransform[5]}). "
        f"Refusing to write defective raster."
    )
```

## Why This Happened
The bug occurred because:
1. GDAL silently accepts the wrong bounds order
2. The resulting files appeared valid but had flipped orientation
3. The issue only manifested when gdalbuildvrt tried to create mosaics

## Prevention
1. **Unit Test**: Added `test_viirs_orientation.py` to verify negative Y pixel size
2. **Guard Rail**: Code now raises an error if it detects positive Y pixel size
3. **Documentation**: Clear comment showing correct parameter order

## Impact
- All VIIRS TIF files generated before this fix have incorrect orientation
- Pipeline would fail at VRT creation step with empty mosaic
- Files must be regenerated from HDF5 sources

## Verification
After fix, all VIIRS tiles show correct orientation:
```
Pixel Size = (0.004166666666667,-0.004166666666667)  ✓
Upper Left  (-110.0, 50.0)  ✓ (higher latitude)
Lower Left  (-110.0, 40.0)  ✓ (lower latitude)
```