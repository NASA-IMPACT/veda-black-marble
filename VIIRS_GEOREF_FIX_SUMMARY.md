# VIIRS Georeferencing Fix Summary

## Problem
The Black Marble pipeline was producing all-zero/NaN outputs and black TIFF images. Investigation revealed:

1. **Incorrect tile selection**: VIIRS tile selection was using a simple 10°x10° grid formula instead of proper MODIS sinusoidal projection
2. **Wrong tiles for location**: The available VIIRS tiles (h17v04, h17v05) are for Western Europe, not West Africa
3. **Corrupted georeferencing**: VIIRS tiles had inverted N/S coordinates and wrong locations
4. **NEW: Positive NS resolution**: VIIRS tiles had positive Y-axis pixel size, causing gdalbuildvrt to reject them entirely

## Solution Implemented

### Phase 1: Immediate Fix
1. Created `viirs_spatial.py` with proper spatial intersection-based tile selection
2. Repaired existing VIIRS tiles using `gdal_edit.py` to fix sinusoidal projection metadata
3. Created WGS84 versions of all tiles for pipeline compatibility
4. Updated pipeline to prefer `_wgs84.tif` files when available

### Phase 2: Y-Axis Resolution Fix
Fixed the "positive NS resolution" error that caused gdalbuildvrt to skip all tiles:
```bash
# VIIRS tiles had positive Y pixel size: (0.004166666666667, 0.004166666666667)
# Fixed by swapping latitude bounds to create negative Y resolution
for tile in data/viirs/2023/01/24/VNP46A2_h0[78]v0[45]_*.tif; do
    gdal_edit.py -a_ullr $ul_lon $lr_lat $lr_lon $ul_lat "$tile"
done
# Result: Pixel Size = (0.004166666666667, -0.004166666666667)
```

### Code Changes
- Added `blackmarble/acquire/viirs_spatial.py` with:
  - `find_viirs_tiles_by_intersection()` - spatial overlap-based selection
  - `get_tile_footprint_wgs84()` - proper bounds extraction
  - `create_viirs_mosaic()` - multi-tile mosaicking
  - `lonlat_to_modis_sinusoidal_tile()` - correct tile ID calculation

- Updated `.gitignore` to exclude:
  - `*.bak` - backup files from georef repair
  - `*_wgs84.tif` - generated WGS84 versions

## Validation Results

All VIIRS tiles now have correct georeferencing:
- h17v04: Western Europe (~-15 to 0°E, 40-50°N) ✓
- h17v05: Southern Spain/Morocco (~-15 to 0°E, 30-40°N) ✓
- h07v04, h07v05: Central/Western US ✓
- h08v04, h08v05: Eastern US/Atlantic ✓

All tiles:
- Are in WGS84 (EPSG:4326) projection
- Have correct bounds for their tile IDs
- Have proper NoData value (65535) preserved
- Contain expected coverage for daily VNP46A2 (35-45% valid pixels)

## Understanding VNP46A2 Coverage

The ~40% valid pixel coverage is **normal** for daily VNP46A2 products. "Gap-filled" means:
- Temporal gap-filling from a ±30 day window
- NOT spatial interpolation to fill the entire tile

Coverage is limited by:
- Cloud masking
- Ocean masking (if enabled)
- Stray light filtering (Feb-Apr, Aug-Oct at mid-latitudes)
- Quality control filters
- Daylight/twilight exclusion

For 100% spatial coverage, use:
- Monthly composites (VNP46A3)
- Annual composites (VNP46A4)

## Data Availability Issue

For West Africa testing (WRS 181/40 at 15°E, 10°N), we need tile **h19v07** which is not available in the current dataset. The existing tiles are for different geographic regions.

## Next Steps

1. **For immediate testing**: Use Spain/Western Europe test locations where we have h17v04/h17v05
2. **For production**: Download the correct VIIRS tiles for West Africa (h18v07, h19v07, h20v07, h19v08)
3. **Pipeline improvements**: 
   - Add validation that selected VIIRS tiles actually overlap with target area
   - Document required VIIRS tiles for each geographic region