# Urban Field Shape Mismatch Fix

## Problem
Pipeline was failing with:
```
ValueError: operands could not be broadcast together with shapes (22987,26599) (100,100)
```

The urban field was a tiny 100x100 array while NTL data was 22987x26599.

## Root Cause
1. All Landsat dates showed "Skipping - missing bands" 
2. `reference_shape` was only set from Landsat data
3. When no Landsat was available, `reference_shape` remained None
4. The road rasterization code had a fallback to create 100x100 arrays when `reference_shape` was None
5. This 100x100 urban field then failed when trying to enhance the full-size NTL data

## Fix Applied
1. **Added VIIRS fallback** (line 398-401):
   ```python
   # If no Landsat reference, use VIIRS as reference
   if reference_transform is None and viirs_transform is not None:
       reference_transform = viirs_transform
       reference_shape = viirs_reprojected.shape
       print("⚠️  No Landsat data available - using VIIRS as reference grid")
   ```

2. **Removed 100x100 fallback** (line 541-544):
   ```python
   else:
       # This should never happen if we have any data
       raise RuntimeError(
           "No reference shape available for road rasterization. "
           "This indicates no Landsat or VIIRS data was successfully processed."
       )
   ```

## Result
- When Landsat is available: Uses Landsat grid as reference (unchanged behavior)
- When only VIIRS is available: Uses VIIRS grid as reference
- When neither is available: Raises clear error instead of silently creating 100x100 array

This ensures all arrays (NTL, urban field, roads) share the same shape, preventing broadcast errors.