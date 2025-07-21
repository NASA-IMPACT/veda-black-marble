# Landsat Tile Coverage Fix: Date-Grouped Selection

## Problem Statement

The Black Marble pipeline was producing composites with missing tiles (NaN regions) despite having abundant Landsat scenes available. Investigation revealed that the scene selection logic was optimizing each WRS tile independently, resulting in scenes from different dates being combined in a single composite.

### Root Cause

The `search_best_scenes()` function in `landsat_search.py`:
1. Searches for all scenes within ±45 days of target date
2. Groups scenes by (path, row) tile
3. Selects the single "best" scene for each tile independently
4. Returns a list where each tile's scene could be from a completely different date

Due to Landsat's orbital pattern where adjacent paths are captured ~7-8 days apart, this approach virtually guarantees temporal inconsistency.

## Solution: Two-Stage Filtering

We implemented a new date-grouped selection strategy that prioritizes temporal consistency:

### Stage 1: Coverage Filter
- Group all candidate scenes by acquisition date (tolerance = 0 days)
- Filter to only date groups with complete tile coverage (e.g., 9/9 tiles)
- This is a hard requirement - incomplete dates are rejected

### Stage 2: Quality Optimization  
- Among dates with complete coverage, select the one with lowest average cloud cover
- Quality is only considered after coverage requirements are met

## Implementation Details

### Configuration Changes

```python
# In blackmarble/config.py
@dataclass
class LandsatAcquisitionConfig:
    # New date-grouped selection settings
    selection_strategy: Literal["per_tile", "date_grouped"] = "date_grouped"
    tolerance_days: float = 0.0  # Same date only
    require_complete_coverage: bool = True  # All tiles required
```

### Key Design Decisions

1. **tolerance_days = 0.0**: All tiles must be from the exact same date
   - Ensures perfect temporal consistency
   - Eliminates atmospheric variation between tiles
   - Midnight-crossing edge case is extremely rare

2. **require_complete_coverage = True**: Only dates with all tiles are considered
   - No partial coverage accepted
   - With 12 dates × ±45 day windows = ~24 months of coverage opportunity
   - Sufficient redundancy to find complete dates

3. **Two-stage filtering**: Coverage first, quality second
   - Simplifies logic - coverage is binary, not weighted
   - Clear priority: consistency > completeness > quality

## Usage

### Default Behavior (New)
```python
# Automatically uses date-grouped selection
config = LandsatAcquisitionConfig()  
```

### Legacy Behavior (If Needed)
```python
# Revert to per-tile optimization
config = LandsatAcquisitionConfig(
    selection_strategy="per_tile"
)
```

### Relaxed Requirements
```python
# Allow incomplete coverage if necessary
config = LandsatAcquisitionConfig(
    require_complete_coverage=False,  # Use best available
    tolerance_days=1.0,  # Allow adjacent days
)
```

## Benefits

1. **Temporal Consistency**: All tiles from same satellite pass
2. **No NaN Artifacts**: Complete spatial coverage guaranteed
3. **Cleaner Composites**: No mixing of atmospheric conditions
4. **Sufficient Coverage**: 12 dates provide ample opportunities

## Trade-offs

1. **Stricter Requirements**: May occasionally find no suitable dates in very cloudy regions
2. **Higher Average Cloud Cover**: The selected date may have higher cloud cover than the independently-best tiles
3. **All-or-Nothing**: A single cloudy tile can disqualify an otherwise good date

However, with 12 target dates throughout the year, these trade-offs are minimal in practice.

## Verification

Run the test script to verify the new behavior:

```bash
uv run python test_grouped_selection.py
```

This will show:
- Whether complete coverage was found
- The selected date and its cloud cover
- Confirmation that all tiles are from the same date