# Wrapper Function Inventory Summary

## Overview
This inventory identifies wrapper functions and unused code that could be removed or inlined to reduce code complexity.

## Summary Statistics
- **Total functions identified**: 12
- **Unused functions**: 6
- **Thin wrappers**: 2
- **Pure wrappers**: 2
- **Unused wrappers**: 1

## Categories

### 1. Unused Functions (Can be deleted)
These functions are not used anywhere in the codebase:
- `batch_export_cogs` - Not exported, not used
- `create_overviews` - Not exported, not used  
- `optimize_for_web` - Not exported, not used
- `enhance_roads` - Commented out in pipeline
- `enhance_urban_features` - Commented out in pipeline
- `create_false_color_composite` - Commented out in pipeline

### 2. Thin Wrappers (Consider inlining)
These functions add minimal value and could be inlined at call sites:
- `export_single_band` - Just adds band_name to metadata, then calls create_cog
- `export_multi_band` - Stacks arrays and adds band names to metadata, then calls create_cog

### 3. Pure Wrappers (Should be removed)
These functions are simple parameter reorderings or type conversions:
- `get_wrs_tiles_for_bbox` - Just unpacks bbox tuple and calls get_wrs_tiles_for_bbox_tuple
- `convert_to_wgs84` - Wrapper around reproject_image with dst_crs='EPSG:4326'

### 4. Unused Wrappers (Delete immediately)
- `coordinates_to_viirs_tile_string` - Formats tile coords as string, not used anywhere

## Recommendations

1. **Immediate deletions** (no risk):
   - All unused functions in export/cog.py
   - All commented-out functions in enhance module
   - `coordinates_to_viirs_tile_string`

2. **Consider inlining** (low risk):
   - `export_single_band` and `export_multi_band` - The pipeline could call `create_cog` directly
   - `get_wrs_tiles_for_bbox` - Just use `get_wrs_tiles_for_bbox_tuple` directly

3. **Review usage** (medium risk):
   - `convert_to_wgs84` - Only used in docker/bm.py, not in main pipeline

## Potential Code Reduction
- Removing unused functions: ~300 lines
- Inlining thin wrappers: ~50 lines
- **Total potential reduction**: ~350 lines