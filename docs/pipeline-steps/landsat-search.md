# Landsat Scene Search and Selection

## Overview

The scene search strategy ensures complete spatial coverage and temporal consistency by finding all available Landsat scenes within a time window, then selecting the best date group that provides complete coverage. This date-grouped approach prioritizes getting all tiles from the same acquisition date when possible.

## Problem Solved

Searching for scenes on a single date often results in:
- Missing tiles due to satellite orbit patterns (16-day revisit cycle)
- High cloud cover on the target date
- Temporal inconsistency when composites mix different dates
- Manual intervention to find cloud-free alternatives

## How It Works

### 1. Time Window Search

The system searches within a configurable window (default ±7 days) around each target date:

```python
from blackmarble.acquire.landsat_search_grouped import search_best_scenes
from blackmarble.config import LandsatAcquisitionConfig

config = LandsatAcquisitionConfig(
    window_days=7,  # Search ±7 days from target
)

scenes = await search_best_scenes(
    bbox=(-73.0, 40.5, -72.5, 41.0),
    target_date=datetime(2023, 8, 10),
    config=config
)
```

### 2. Date Grouping with Tolerance

The key innovation is grouping scenes by acquisition date with a tolerance window:

1. **Get ALL scenes** in the time window (no per-tile selection yet)
2. **Group scenes by date** (just the date, ignoring time)
3. **Build date groups** with tolerance (default 12 days):
   - For each base date, include scenes within ±tolerance_days
   - This handles multi-day acquisitions for large areas
4. **Find complete groups** that cover all required tiles
5. **Select best group** by lowest average cloud cover

```python
config = LandsatAcquisitionConfig(
    window_days=7,        # Search ±7 days from each target date
    tolerance_days=12.0,  # Group scenes within 12 days together
)
```

### 3. Complete Coverage Priority

The algorithm prioritizes complete spatial coverage:
- Identifies all WRS-2 tiles needed for the bbox
- Checks each date group for complete tile coverage
- Only considers groups with 100% coverage
- Falls back to best partial coverage if `allow_incomplete=True`

### 4. Implementation Details

The grouped search is implemented in `landsat_search_grouped.py`:

```python
# Get all scenes in window
all_scenes = await get_all_scenes_in_window(params, find_scenes_func, s3_client)

# Group by date
scenes_by_date = {}
for scene in all_scenes:
    date_key = scene.date.date()  # Just the date, no time
    scenes_by_date[date_key].append(scene)

# Find date groups with complete coverage
complete_groups = []
for base_date in sorted(scenes_by_date.keys()):
    group_scenes = []
    for date in scenes_by_date:
        if abs((date - base_date).days) <= tolerance_days:
            group_scenes.extend(scenes_by_date[date])
    
    tiles = {(s.path, s.row) for s in group_scenes}
    if tiles >= required_tiles:
        complete_groups.append((base_date, group_scenes))

# Select group with lowest cloud cover
best_group = min(complete_groups, key=lambda x: avg_cloud_cover(x[1]))
```

## Configuration Options

### Basic Configuration

```python
config = LandsatAcquisitionConfig(
    window_days=7,               # Search ±7 days from target
    tolerance_days=12.0,         # Group scenes within 12 days
)
```

### Key Parameters

- **window_days**: How far to search from each target date (default: 7)
  - Smaller windows ensure temporal consistency
  - Larger windows find more cloud-free options
  
- **tolerance_days**: Maximum days between scenes in a group (default: 12)
  - Allows grouping scenes from multiple passes
  - Essential for large areas spanning multiple paths

### Handling Incomplete Coverage

```python
# Allow partial results instead of error
scenes = await search_best_scenes(
    bbox=bbox,
    target_date=target_date,
    config=config,
    allow_incomplete=True  # Return best partial coverage
)
```

## Scene Metadata

Each scene is represented with key attributes:

```python
@dataclass(frozen=True)
class SceneMeta:
    scene_id: str           # e.g., "LC08_L2SP_013032_20230809..."
    path: int               # WRS-2 path
    row: int                # WRS-2 row
    date: datetime          # Acquisition date
    cloud_cover: float      # 0-100 percentage
    s3_prefix: str          # S3 location
```

## Performance Optimizations

1. **Batch Processing**: Gets all scenes first, then groups
2. **S3 Client Reuse**: Single client for all operations
3. **Efficient Grouping**: Uses date tolerance to minimize groups
4. **Cloud-based Selection**: Simple average cloud cover metric

## Pipeline Integration

The date-grouped search is automatically used by the Black Marble pipeline:

```python
from blackmarble.pipeline import black_marble_pipeline_sync
from blackmarble.config import PipelineConfig

# Create pipeline config with 12 target dates throughout the year
config = PipelineConfig.for_bm_parity(
    bbox=bbox,
    date=date,
    output_path="output.tif"
)

# The pipeline automatically:
# 1. Creates 12 dates spanning the year
# 2. Searches ±7 days around each date
# 3. Groups scenes with 12-day tolerance
# 4. Selects best cloud-free groups

# Customize if needed
config.acquisition.landsat_acquisition.window_days = 10  # Search ±10 days
config.acquisition.landsat_acquisition.tolerance_days = 15.0  # Larger groups

# Run pipeline
result = black_marble_pipeline_sync(
    bbox=bbox,
    date=date,
    output_path="output.tif",
    config=config,
    earthdata_token=token
)
```

## How the Pipeline Creates Multiple Dates

The pipeline automatically generates multiple target dates for temporal compositing:

```python
# In AsyncDownloadOrchestrator.from_pipeline_config()
composite_months = config.acquisition.composite_months  # Default: 12

# Start with the target date
landsat_dates = [date]

# Add historical months (going back in time)
for i in range(1, composite_months):
    historical_date = date - timedelta(days=i * 30)
    landsat_dates.append(historical_date)
```

With `composite_months=12`, this creates 12 dates at ~30-day intervals:
- Target date (e.g., 2023-06-15)
- Target - 30 days (2023-05-16)
- Target - 60 days (2023-04-16)
- ... up to Target - 330 days

For each date, the grouped search finds the best available scenes within the time window.

## Benefits

1. **Temporal Consistency**: Prioritizes scenes from the same date
2. **Complete Coverage**: Ensures all tiles are covered
3. **Cloud Optimization**: Selects date groups with lowest average cloud cover
4. **Flexible Grouping**: Tolerance allows handling multi-day acquisitions
5. **Automatic Date Generation**: Pipeline creates optimal date spacing

## AWS Credentials

The Landsat search uses the AWS credential chain:
1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
2. AWS CLI configuration (~/.aws/credentials)
3. IAM roles (if running on AWS)

The USGS Landsat bucket requires authenticated requests with `RequestPayer='requester'`.

## Key Differences from Per-Tile Selection

The old per-tile "best scene" approach:
- Selected scenes independently for each tile
- Could mix dates within a single composite
- Used complex quality scoring weights

The new date-grouped approach:
- Gets all scenes first, then groups by date
- Prioritizes temporal consistency
- Uses simple cloud cover average for selection
- Better for change detection and time series analysis

## Related Steps

- Previous: Data acquisition setup
- Next: [QA Masking](./landsat-qa-masking.md) - Apply cloud masks
- Outputs to: All Landsat processing steps