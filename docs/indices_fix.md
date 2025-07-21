# Spectral Indices Calculation Fix: Implementation Plan

## Executive Summary

This document outlines a critical fix to how spectral indices (NDVI, NDWI) are calculated from temporal stacks of satellite imagery. The current approach violates mathematical principles and produces biased results, particularly affecting NDUI calculations in seasonal vegetation areas.

**Key Changes:**
1. Calculate indices per-date BEFORE temporal aggregation (not after)
2. Use 85th percentile for NDVI in NDUI calculations (not median)
3. Return NaN for low denominators (not forced zero)
4. Implement streaming to reduce memory usage
5. Add fallback strategy for incomplete spatial coverage

## 1. Problem Statement

### 1.1 Mathematical Issue

The current pipeline incorrectly calculates spectral indices by taking the median of bands first, then calculating ratios:

```
Current (WRONG):
NDVI = (median(NIR) - median(RED)) / (median(NIR) + median(RED))

Correct:
NDVI = median((NIR[t] - RED[t]) / (NIR[t] + RED[t]))
```

These are mathematically different operations. The median of ratios ≠ ratio of medians.

### 1.2 Concrete Example

Consider two dates with different vegetation states:

| Date | RED (B4) | NIR (B5) | NDVI |
|------|----------|----------|------|
| Summer | 50 | 200 | 0.60 |
| Winter | 100 | 150 | 0.20 |

**Current approach:**
- median(RED) = 75
- median(NIR) = 175  
- NDVI = (175-75)/(175+75) = 0.40

**Correct approach:**
- NDVI stack = [0.60, 0.20]
- median(NDVI) = 0.40

In this simple case they match, but with real data including clouds, shadows, and seasonal variations, the differences can be substantial.

### 1.3 Impact on NDUI

NDUI uses NDVI as a vegetation correction factor:
```
NDUI = NTL / (1 - NDVI)
```

The issue is compounded because:
1. Using median NDVI underrepresents peak vegetation in seasonal areas
2. This makes agricultural regions appear more "urban" than they are
3. The 85th percentile better captures vegetation's light-dimming potential

## 2. Phased Implementation Plan

### Phase 0: Preparation (0.5 days)
- Create feature branch `feat/index-first-composite`
- Set up test data for validation
- Document current behavior for comparison

### Phase 1: Core Library Updates (1 day)

#### 1.1 Add Index Composite Function
**File:** `blackmarble/analyze/temporal.py`
```python
def create_index_temporal_composite(
    index_stack: list[ArrayLike],
    method: Literal["median", "mean", "percentile"] = "median",
    percentile: float = 50.0,
    mask_nan: bool = True
) -> ArrayLike:
    """Create temporal composite from pre-calculated index arrays.
    
    Unlike create_landsat_median_composite, this expects indices
    (NDVI, NDWI) already calculated per date.
    """
```

#### 1.2 Update Index Calculations
**File:** `blackmarble/analyze/indices.py`

Add parameter to control low-denominator behavior:
```python
def calculate_ndvi(
    nir: ArrayLike, 
    red: ArrayLike, 
    epsilon: float = LANDSAT_EPSILON,
    mask_low_denom: bool = False  # NEW
) -> ArrayLike:
    if mask_low_denom and denominator < epsilon:
        return np.nan  # Instead of forcing to 0
```

### Phase 2: Pipeline Refactor (1-2 days)

#### 2.1 Modify Temporal Processing Loop
**File:** `blackmarble/pipeline.py` (around line 400-500)

**Before the date loop:**
```python
# Initialize index stacks
index_stacks = {
    "ndvi": [],
    "ndwi": []
}
```

**Inside the date loop (after QA masking):**
```python
# Calculate indices for this date
if config.analysis.calculate_indices_first:
    ndvi_date = analyze.calculate_ndvi(
        nir=masked_bands["B5"],
        red=masked_bands["B4"],
        mask_low_denom=config.analysis.mask_low_denom
    )
    ndwi_date = analyze.calculate_ndwi(
        green=masked_bands["B3"],
        nir=masked_bands["B5"],
        mask_low_denom=config.analysis.mask_low_denom
    )
    
    index_stacks["ndvi"].append(ndvi_date)
    index_stacks["ndwi"].append(ndwi_date)
    
    # Optional: Delete bands to save memory
    if config.memory.delete_bands_after_indices:
        del masked_bands["B3"], masked_bands["B4"], masked_bands["B5"]
```

**After the date loop:**
```python
# Create index composites
ndvi_composite = analyze.create_index_temporal_composite(
    index_stacks["ndvi"],
    method=config.analysis.ndvi_composite_method,
    percentile=config.analysis.ndvi_composite_percentile
)

# For NDUI, use high percentile
ndvi_for_ndui = analyze.create_index_temporal_composite(
    index_stacks["ndvi"],
    method="percentile",
    percentile=85.0
)
```

### Phase 3: Date-Grouping Fallback (0.5 days)

**File:** `blackmarble/acquire/landsat_search_grouped.py`

Add fallback strategy:
```python
async def search_best_scenes_with_fallback(
    bbox: tuple[float, float, float, float],
    target_date: datetime,
    config: LandsatAcquisitionConfig,
    **kwargs
) -> tuple[list[SceneMeta], str]:
    """Try grouped search first, fall back to per-tile if needed."""
    
    # Try strict date grouping
    scenes = await search_best_scene_group(
        bbox, target_date, config, **kwargs
    )
    
    if scenes:
        return scenes, "date_grouped"
    
    # Fallback to per-tile
    logger.info("No complete date groups found, falling back to per-tile selection")
    from .landsat_search import search_best_scenes as search_per_tile
    
    scenes = await search_per_tile(
        bbox, target_date, config, **kwargs
    )
    return scenes, "per_tile_fallback"
```

### Phase 4: Configuration Updates (0.5 days)

**File:** `blackmarble/config.py`

Add to `AnalysisConfig`:
```python
@dataclass
class AnalysisConfig:
    # Existing fields...
    
    # Index calculation settings
    calculate_indices_first: bool = True
    mask_low_denom: bool = True
    
    # Composite methods
    ndvi_composite_method: Literal["median", "mean", "percentile"] = "percentile"
    ndvi_composite_percentile: float = 85.0
    ndwi_composite_method: Literal["median", "mean", "percentile"] = "median"
    ndwi_composite_percentile: float = 50.0
    
    # Legacy compatibility
    use_legacy_indices: bool = False
```

### Phase 5: Testing Strategy (1 day)

#### 5.1 Unit Tests
- `test_index_calculation_order()`: Verify median(ratios) ≠ ratio(medians)
- `test_low_denominator_nan()`: Ensure NaN propagation works correctly
- `test_percentile_composite()`: Validate 85th percentile calculation

#### 5.2 Integration Tests
- Kansas bbox with fallback: Verify output exists despite incomplete dates
- Small urban area: Verify strict grouping when possible
- Seasonal area: Compare NDUI with old vs new approach

#### 5.3 Performance Tests
- Memory usage: Confirm < 1GB for 12-date stack
- Runtime: Verify < 10% increase

### Phase 6: Documentation (0.5 days)
- Update `docs/pipeline-steps/spectral-indices.md`
- Add migration guide for users
- Update CHANGELOG

## 3. Risk Assessment

### Risk Matrix

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Memory overflow | High | Low | Stream processing, immediate deletion |
| Performance regression | Medium | Low | Vectorized operations, profiling |
| Backward compatibility | High | Medium | Legacy flag, clear documentation |
| Incorrect results | High | Low | Extensive testing, gradual rollout |
| User confusion | Medium | Medium | Clear documentation, examples |

### Specific Mitigations

1. **Memory Management**
   - Delete band arrays immediately after index calculation
   - Process dates in batches if needed
   - Add memory monitoring/warnings

2. **Backward Compatibility**
   - `--legacy-indices` CLI flag
   - `use_legacy_indices` config option
   - Deprecation warnings for 2 versions

3. **Quality Assurance**
   - Side-by-side comparison of outputs
   - Visual regression tests
   - Automated metrics comparison

## 4. Code Change Locations

### Primary Files to Modify
1. `blackmarble/analyze/temporal.py` - Add index composite function
2. `blackmarble/analyze/indices.py` - Add NaN option for low denominators
3. `blackmarble/pipeline.py` - Refactor temporal processing (~lines 400-600)
4. `blackmarble/config.py` - Add new configuration options
5. `blackmarble/acquire/landsat_search_grouped.py` - Add fallback strategy

### Test Files to Add/Update
1. `tests/analyze/test_temporal_indices.py` - New test file
2. `tests/analyze/test_indices.py` - Update for NaN behavior
3. `tests/pipeline/test_index_compositing.py` - New integration tests

### Documentation Updates
1. `docs/pipeline-steps/spectral-indices.md`
2. `docs/pipeline-steps/temporal-compositing.md`
3. `CHANGELOG.md`
4. `README.md` - Note about index calculation

## 5. Validation Criteria

### Correctness
- [ ] Unit tests pass for mathematical correctness
- [ ] NaN propagation works as expected
- [ ] 85th percentile NDVI used for NDUI

### Performance
- [ ] Memory usage < 1GB for typical 12-date stack
- [ ] Runtime increase < 10%
- [ ] No memory leaks in streaming mode

### Compatibility
- [ ] Legacy mode produces identical results to current version
- [ ] All existing tests pass with legacy flag
- [ ] Clear migration path documented

### Quality
- [ ] Visual inspection shows improved NDUI in agricultural areas
- [ ] No new artifacts introduced
- [ ] Metrics show expected changes (higher dynamic range)

## 6. Critical Implementation Details

### 6.1 Before/After Code Examples

**BEFORE (Current Implementation):**
```python
# In pipeline.py - current approach
band_stacks = {"B3": [], "B4": [], "B5": []}

# Collect all bands
for date in dates:
    bands = load_and_qa_mask(date)
    band_stacks["B3"].append(bands["B3"])
    band_stacks["B4"].append(bands["B4"]) 
    band_stacks["B5"].append(bands["B5"])

# Create composites THEN calculate indices
b3_composite = create_landsat_median_composite(band_stacks["B3"])
b4_composite = create_landsat_median_composite(band_stacks["B4"])
b5_composite = create_landsat_median_composite(band_stacks["B5"])

# Calculate indices from composites (WRONG!)
ndvi = calculate_ndvi(b5_composite, b4_composite)
```

**AFTER (Fixed Implementation):**
```python
# In pipeline.py - corrected approach
index_stacks = {"ndvi": [], "ndwi": []}

# Calculate indices per-date
for i, date in enumerate(dates):
    bands = load_and_qa_mask(date)
    
    # Calculate indices FIRST
    ndvi = calculate_ndvi(
        bands["B5"], bands["B4"], 
        mask_low_denom=True
    ).astype("float32")  # Control memory
    
    ndwi = calculate_ndwi(
        bands["B3"], bands["B5"],
        mask_low_denom=True
    ).astype("float32")
    
    index_stacks["ndvi"].append(ndvi)
    index_stacks["ndwi"].append(ndwi)
    
    # Free memory immediately
    del bands
    if i % 3 == 0:
        gc.collect()
        
    print(f"   Date {i+1}/{len(dates)} processed - RSS: {get_memory_gb():.1f} GB")

# Create composites from indices
ndvi_composite = create_index_temporal_composite(
    index_stacks["ndvi"], 
    method="percentile",
    percentile=85.0
)
```

### 6.2 Streaming Pitfalls and Safeguards

#### Shape Validation
```python
# Add after QA masking in date loop
assert bands["B3"].shape == bands["B4"].shape == bands["B5"].shape, \
    f"Band shape mismatch on {date}: B3={bands['B3'].shape}, " \
    f"B4={bands['B4'].shape}, B5={bands['B5'].shape}"
```

#### Memory Management
```python
# Force garbage collection periodically
if i % 3 == 0:
    before_gb = get_memory_usage_gb()
    gc.collect()
    after_gb = get_memory_usage_gb()
    if before_gb - after_gb > 0.5:
        logger.info(f"Freed {before_gb - after_gb:.1f} GB via gc.collect()")
```

#### Progress Monitoring
```python
def get_memory_usage_gb():
    """Get current process memory usage in GB."""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)
```

### 6.3 Metadata Enhancements

Add to pipeline output metadata:
```json
{
    "processing_metadata": {
        "ndvi_temporal_method": "percentile_85",
        "ndwi_temporal_method": "median",
        "indices_calculated_first": true,
        "landsat_selection_mode": "date_grouped",
        "date_grouped_attempted": true,
        "date_grouped_success": false,
        "fallback_reason": "no_complete_date_in_window",
        "fallback_coverage": 0.89
    }
}
```

### 6.4 NDUI Comparison Tool

Create `tools/compare_ndui.py`:
```python
#!/usr/bin/env python3
"""Compare NDUI outputs between old and new calculation methods."""

def compare_ndui(old_path: str, new_path: str, output_dir: str):
    """Generate comparison statistics and difference raster."""
    
    with rasterio.open(old_path) as old_src:
        old_ndui = old_src.read(1)
        profile = old_src.profile
        
    with rasterio.open(new_path) as new_src:
        new_ndui = new_src.read(1)
    
    # Calculate differences
    diff = new_ndui - old_ndui
    abs_diff = np.abs(diff)
    
    # Statistics
    stats = {
        "mean_difference": np.nanmean(diff),
        "median_difference": np.nanmedian(diff),
        "std_difference": np.nanstd(diff),
        "pixels_changed_5pct": np.sum(abs_diff > 0.05) / np.sum(~np.isnan(diff)),
        "pixels_changed_10pct": np.sum(abs_diff > 0.10) / np.sum(~np.isnan(diff)),
        "max_increase": np.nanmax(diff),
        "max_decrease": np.nanmin(diff)
    }
    
    # Create difference raster (3 classes)
    diff_classes = np.zeros_like(diff, dtype=np.uint8)
    diff_classes[diff < -0.05] = 1  # Decreased (darker)
    diff_classes[abs_diff <= 0.05] = 2  # Unchanged
    diff_classes[diff > 0.05] = 3  # Increased (brighter)
    
    # Save outputs
    with open(f"{output_dir}/comparison_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    profile.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(f"{output_dir}/ndui_difference_classes.tif", "w", **profile) as dst:
        dst.write(diff_classes, 1)
    
    return stats
```

### 6.5 Tropical Phenology Considerations

For areas with strong seasonal patterns, add configuration:
```python
@dataclass
class AnalysisConfig:
    # Minimum observations for percentile
    min_valid_observations_for_percentile: int = 3
    
    # Seasonal window (None = use all dates)
    ndvi_seasonal_filter: Callable[[datetime], bool] | None = None
    # Example: lambda d: d.month in [5,6,7,8,9]  # Growing season only
```

## 7. Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 0: Prep | 0.5 days | None |
| Phase 1: Libraries | 1 day | Phase 0 |
| Phase 2: Pipeline | 1-2 days | Phase 1 |
| Phase 3: Fallback | 0.5 days | Phase 2 |
| Phase 4: Config | 0.5 days | Phase 2 |
| Phase 5: Testing | 1 day | Phases 1-4 |
| Phase 6: Docs | 0.5 days | Phase 5 |
| Review & Deploy | 0.5 days | All phases |

**Total: 5-6 days**

## 7. Success Metrics

1. **Mathematical Correctness**: Verified through unit tests
2. **Performance**: < 10% runtime increase, < 1GB memory for 12 dates
3. **Quality**: Improved NDUI discrimination in agricultural areas
4. **Compatibility**: Zero breaking changes with legacy flag
5. **Coverage**: > 95% code coverage for new functions

## 8. Future Considerations

1. **Seasonal Composites**: Add option for growing-season-only composites
2. **Multiple Metrics**: Export both median and peak vegetation
3. **Phenology Alignment**: Match NDVI dates to NTL acquisition month
4. **Smart Compositing**: Weight by quality scores, not just median/percentile

## Appendix: Example Configuration

```python
# New default configuration (mathematically correct)
config = PipelineConfig(
    analysis=AnalysisConfig(
        calculate_indices_first=True,
        mask_low_denom=True,
        ndvi_composite_method="percentile",
        ndvi_composite_percentile=85.0,
    )
)

# Legacy configuration (backward compatible)
legacy_config = PipelineConfig(
    analysis=AnalysisConfig(
        use_legacy_indices=True  # Overrides all other settings
    )
)
```