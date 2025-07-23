# Phase 3: Remove Pointless Wrappers - Implementation Plan

## Summary

Found 15 functions for review:
- **7 DELETE** - Unused functions (0 call sites)
- **6 INLINE** - Pointless wrappers 
- **2 KEEP** - Valid factory methods

**Estimated LOC reduction: ~287 lines**

## Priority Order

### P0 - Immediate Removal (117 LOC)
1. **blackmarble/export/cog.py** - 3 unused functions (110 LOC total)
   - `batch_export_cogs` - Not exported, never used
   - `create_overviews` - Not exported, never used  
   - `optimize_for_web` - Not exported, never used
   
2. **blackmarble/acquire/osm.py** - `get_osm_cache_dir` (3 LOC)
   - INLINE: Just returns `Path("data/osm_cache")`
   - 7 call sites to update
   
3. **blackmarble/acquire/viirs.py** - `coordinates_to_viirs_tile_string` (4 LOC)
   - DELETE: Not used anywhere

### P1 - Quick Wins (120 LOC)
1. **blackmarble/enhance/** - 3 unused functions (95 LOC)
   - `enhance_roads` - Commented out in pipeline
   - `enhance_urban_features` - Commented out in pipeline
   - `create_false_color_composite` - Commented out in pipeline

2. **blackmarble/export/cog.py** - `export_single_band` (16 LOC)
   - INLINE: Just adds metadata['band_name'] before calling create_cog
   - 2 call sites

3. **blackmarble/acquire/landsat.py** - `get_wrs_tiles_for_bbox` (9 LOC)
   - INLINE: Just unpacks bbox tuple
   - 3 call sites

### P2 - Lower Priority (50 LOC)
1. **blackmarble/export/cog.py** - `export_multi_band` (40 LOC)
   - INLINE: Stacks arrays and adds metadata
   - 1 call site

2. **blackmarble/prepare/spatial.py** - `convert_to_wgs84` (10 LOC)
   - INLINE: Just calls reproject_image with 'EPSG:4326'
   - Only used in docker/bm.py

## Implementation Steps

### Step 1: P0 Deletions
```bash
# Remove unused functions from export/cog.py
# Update __init__.py exports
# Remove coordinates_to_viirs_tile_string
```

### Step 2: P0 Inlining 
```bash
# Replace all calls to get_osm_cache_dir() with Path("data/osm_cache")
# 7 locations to update
```

### Step 3: P1 Deletions
```bash
# Remove commented-out enhancement functions
# These are already not being used
```

### Step 4: P1 Inlining
```bash
# Replace export_single_band calls
# Replace get_wrs_tiles_for_bbox calls
```

### Step 5: Verification
```bash
# Run tests
# Run static analysis (should stay at 0 issues)
# Update inventory.csv with results
```

## Risk Assessment
- **Low Risk**: All DELETE targets have 0 call sites
- **Medium Risk**: INLINE targets need careful parameter mapping
- **Mitigation**: Small PRs, one module at a time

## Success Metrics
- [ ] 287+ lines removed
- [ ] 0 ruff/vulture issues maintained
- [ ] All tests passing
- [ ] No new wrappers introduced