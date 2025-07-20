# Landsat Scene Search and Selection

## Overview

The scene search strategy ensures complete spatial coverage of Landsat imagery by searching for the best available scenes within a time window rather than requiring exact date matches.

## Problem Solved

Searching for scenes on a single date often results in:
- Missing tiles due to satellite orbit patterns
- High cloud cover on the target date
- Edge cropping when temporal composites use different numbers of tiles
- Manual intervention to find cloud-free alternatives

## How It Works

### 1. Time Window Search

Instead of searching for scenes on a specific date, the system searches within a configurable window (default ±15 days):

```python
from blackmarble.acquire import search_best_scenes
from blackmarble.config import LandsatAcquisitionConfig

config = LandsatAcquisitionConfig(
    window_days=15,  # Search ±15 days from target
)

scenes = await search_best_scenes(
    bbox=(-73.0, 40.5, -72.5, 41.0),
    target_date=datetime(2023, 8, 10),
    config=config
)
```

### 2. Quality Scoring

Each candidate scene is scored based on weighted quality metrics:

```python
quality_weights = {
    "spatial_coverage": 0.5,   # Completeness of data
    "cloud_free": 0.3,         # 1 - (cloud_cover/100)
    "snow_free": 0.15,         # 1 - snow_fraction
    "off_nadir": 0.05,         # Viewing angle penalty
}
```

The scoring formula:
```
Quality Score = Σ(weight[i] × metric[i])
```

### 3. Best Scene Selection

For each WRS-2 path/row tile:
1. Find all scenes within the time window
2. Calculate quality scores
3. Select the highest scoring scene
4. Apply tie-breakers:
   - Prefer newer sensors (L8/L9 > L7 > L5)
   - Choose closest to target date

### 4. Complete Coverage Guarantee

The system ensures one scene per required tile:
- Identifies all WRS-2 tiles intersecting the bounding box
- Searches for scenes in each tile
- Raises an error if any tile has no acceptable scenes (unless `allow_incomplete=True`)

## Configuration Options

### Basic Configuration

```python
config = LandsatAcquisitionConfig(
    window_days=15,              # Search window size
    max_cloud_threshold=80.0,    # Pre-filter scenes
    quality_weights={...},       # Scoring weights
)
```

### Seasonal Filtering

Restrict scenes to specific seasons:

```python
def summer_only(date):
    return 5 <= date.month <= 9

config = LandsatAcquisitionConfig(
    seasonal_filter=summer_only
)
```

### Handling Incomplete Coverage

```python
# Allow partial results instead of error
scenes = await downloader.search_best_scenes(
    bbox=bbox,
    target_date=target_date,
    config=config,
    allow_incomplete=True  # Return available scenes
)
```

## Implementation Details

The implementation uses a functional approach with pure functions and explicit dependencies.

### Scene Metadata

Each scene is represented as an immutable data structure:

```python
@dataclass(frozen=True)
class SceneMeta:
    scene_id: str           # e.g., "LC08_L2SP_013032_20230809..."
    path: int               # WRS-2 path
    row: int                # WRS-2 row
    date: datetime          # Acquisition date
    cloud_cover: float      # 0-100 percentage
    spatial_coverage: float # 0-1 data completeness
    snow_cover: float       # 0-1 snow fraction
    off_nadir: bool         # Viewing angle flag
    sensor: str             # "OLI_TIRS", "ETM", etc.
    quality_score: float    # Calculated score
    s3_prefix: str          # S3 location
```

### Performance Optimizations

1. **Pre-filtering**: Initial cloud cover filter before MTL download
2. **S3 Client Reuse**: Single client for all MTL fetches
3. **Parallel Processing**: Concurrent scene metadata retrieval
4. **Duplicate Handling**: Automatic deduplication of re-indexed scenes

### Error Handling

The system handles various edge cases:
- Missing MTL metadata files (scene rejected)
- Duplicate scene IDs from S3 re-indexing
- Timezone-aware date handling
- Invalid weight configurations
- Seasonal filter eliminating all candidates

## Usage Example

Complete example with error handling:

```python
import asyncio
from datetime import datetime
from blackmarble.acquire import search_best_scenes
from blackmarble.config import LandsatAcquisitionConfig

async def acquire_landsat_scenes():
    # Configure acquisition
    config = LandsatAcquisitionConfig(
        window_days=20,
        quality_weights={
            "spatial_coverage": 0.4,
            "cloud_free": 0.4,
            "snow_free": 0.1,
            "off_nadir": 0.1,
        },
        max_cloud_threshold=70.0,
    )
    
    try:
        # Search for best scenes
        scenes = await search_best_scenes(
            bbox=(-74.0, 40.0, -73.0, 41.0),  # NYC area
            target_date=datetime(2023, 8, 15),
            config=config,
        )
        
        print(f"Found {len(scenes)} scenes:")
        for scene in scenes:
            print(f"  {scene.scene_id}: "
                  f"Path/Row {scene.path}/{scene.row}, "
                  f"Cloud {scene.cloud_cover:.1f}%, "
                  f"Score {scene.quality_score:.3f}")
        
        return scenes
        
    except ValueError as e:
        print(f"Failed to find complete coverage: {e}")
        
        # Try again allowing incomplete coverage
        scenes = await search_best_scenes(
            bbox=(-74.0, 40.0, -73.0, 41.0),
            target_date=datetime(2023, 8, 15),
            config=config,
            allow_incomplete=True,
        )
        
        print(f"Found {len(scenes)} scenes with incomplete coverage")
        return scenes

# Run the async function
scenes = asyncio.run(acquire_landsat_scenes())
```

## Benefits

1. **Automatic Coverage**: No manual searching for cloud-free dates
2. **Consistent Processing**: Same number of scenes per composite
3. **Quality Optimization**: Automatically selects clearest scenes
4. **Flexible Constraints**: Seasonal and quality thresholds
5. **Robust to Edge Cases**: Handles missing data, duplicates, timezones

## Testing

The implementation includes comprehensive tests covering:
- Happy path with multiple tiles
- Duplicate scene handling
- Missing metadata rejection
- Timezone boundary cases
- Seasonal filter edge cases
- Invalid configuration detection
- Quality score monotonicity (property-based)
- Zero coverage handling
- Multiple sensor parsing

Run tests with:
```bash
pytest tests/acquire/test_landsat_time_window_functional.py -v
```