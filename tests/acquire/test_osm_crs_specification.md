# OSM CRS-Aware Road Rasterization Test Specification

This document summarizes the test expectations for CRS-aware road rasterization in the Black Marble pipeline.

## Problem Statement

The current implementation buffers roads in degrees (WGS84), which creates inconsistent road widths at different latitudes:
- At 0° (equator): 0.0003 degrees = 33.4 meters
- At 30°: 0.0003 degrees = 28.9 meters  
- At 45°: 0.0003 degrees = 23.6 meters
- At 60°: 0.0003 degrees = 16.7 meters
- At 70°: 0.0003 degrees = 11.4 meters

This variation causes problems when trying to enhance roads consistently across different geographic regions.

## Expected Solution Workflow

1. **Download roads in WGS84** (from OpenStreetMap)
2. **Calculate appropriate UTM zone** for the area
3. **Project roads to UTM** for accurate meter-based operations
4. **Buffer roads by specified meters** in UTM projection
5. **Reproject buffered roads** to target CRS
6. **Rasterize** to the target grid

## Required Functions

### 1. `calculate_utm_zone(lon: float, lat: float) -> tuple[int, str]`
Calculate the UTM zone and hemisphere for a given coordinate.
- Returns: (zone_number, hemisphere) where hemisphere is 'north' or 'south'
- Example: `calculate_utm_zone(12.453, 41.902)` returns `(33, 'north')`

### 2. `buffer_roads_in_utm(geometry: dict, buffer_meters: float, lon_center: float, lat_center: float) -> dict`
Buffer a road geometry in meters using UTM projection.
- Takes GeoJSON-like geometry in WGS84
- Projects to appropriate UTM zone
- Buffers by specified meters
- Returns buffered geometry in WGS84

### 3. `create_road_raster_from_transform(...) -> np.ndarray`
Pipeline-friendly function that accepts transform/CRS directly instead of reference file.
Parameters:
- `bbox`: Bounding box in WGS84
- `transform`: Affine transform for output grid
- `crs`: Target CRS for output
- `shape`: Output shape (height, width)
- `buffer_meters`: Buffer distance in meters
- `road_type_values`: Dict mapping road types to raster values

### 4. Enhanced `rasterize_road_shapes(...)`
Add CRS support to existing function:
- `src_crs`: Source CRS of input shapes
- `dst_crs`: Target CRS for rasterization
- Handle reprojection internally

### 5. Enhanced `create_road_raster(...)`
Add meter-based buffering support:
- Detect when buffering should be in meters
- Use UTM internally for consistent road widths
- Maintain backward compatibility

## Test Cases

1. **test_create_road_raster_with_crs**: Verifies that roads are buffered consistently regardless of target CRS
2. **test_rasterize_road_shapes_with_reprojection**: Tests low-level reprojection capability
3. **test_create_road_raster_from_transform**: Tests pipeline-friendly function
4. **test_utm_buffer_workflow**: Verifies UTM buffering workflow
5. **test_utm_zone_calculation**: Tests UTM zone calculation for global locations
6. **test_current_vs_expected_behavior**: Demonstrates the problem and solution

## Integration Points

### Pipeline Usage (pipeline_target.py)
```python
# Current approach (lines 251-259)
if 'UTM' in target_crs or '326' in target_crs or '327' in target_crs:
    buffer_meters = 30.0
    buffer_degrees = buffer_meters / (111320 * np.cos(np.radians(center_lat)))
else:
    buffer_degrees = config.acquisition.osm_buffer if config else 0.0003
```

This should be replaced with direct meter-based buffering that works consistently.

### OSM Functions (osm_functions.py)
The pure functions in `osm_functions.py` handle the degree-based buffering. We need to either:
1. Add CRS-aware versions of these functions, or
2. Move CRS handling to a higher level

## Benefits

1. **Consistent road widths** across all latitudes
2. **Accurate buffer distances** in meters
3. **Better urban detection** in high-latitude cities
4. **Simplified pipeline logic** (no need for projection-specific buffer calculations)
5. **Future-proof** for different target projections