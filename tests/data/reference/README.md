# Reference Test Data

This directory contains synthetic reference data for testing the Black Marble pipeline.

## Geographic Coverage

- **Test Area**: San Francisco, CA (small ~1km x 1km area)
- **Bounding Box**: -122.42, 37.77, -122.41, 37.78 (WGS84)
- **Date**: 2023-10-15

## Data Inventory

### 1. VIIRS Nighttime Lights (`viirs/`)
- **File**: `VNP46A2_h08v05_20231015.tif`
- **Size**: 150x150 pixels, 103 KB
- **Resolution**: ~7.4 meters (0.000067 degrees)
- **CRS**: EPSG:4326 (WGS84)
- **Values**: 0-26.46 radiance units
- **Features**: 
  - Two urban centers with Gaussian intensity patterns
  - Linear features representing major roads
  - Realistic noise and variation

### 2. Landsat 8 Collection 2 (`landsat/`)
- **Bands**: B3 (Green), B4 (Red), B5 (NIR), QA_PIXEL
- **Size**: 150x150 pixels each, ~36 KB per band
- **Resolution**: 30 meters
- **CRS**: EPSG:32610 (UTM Zone 10N)
- **Features**:
  - Vegetation areas with realistic spectral signatures
  - Urban areas with lower vegetation indices
  - Water body with very low reflectance
  - QA_PIXEL includes clouds, shadows, and clear pixels

### 3. OpenStreetMap Roads (`osm/`)
- **File**: `test_roads.gpkg`
- **Features**: 12 road segments
- **Types**: residential (10), tertiary (1), motorway_link (1)
- **CRS**: EPSG:4326 (WGS84)
- **Pattern**: Grid pattern typical of San Francisco

## Pipeline Requirements Checklist

✅ **VIIRS Data**
- Nighttime lights radiance values
- Proper geographic coverage
- Float32 data type with realistic range

✅ **Landsat Data**
- B3 (Green) - for NDWI calculation
- B4 (Red) - for NDVI calculation
- B5 (NIR) - for NDVI/NDWI calculation
- QA_PIXEL - for cloud masking
- Proper scaling values (0.0000275, -0.2)

✅ **OSM Data**
- Road network for fractional rasterization
- Multiple road types for realistic patterns
- Proper CRS matching test area

## Data Characteristics

### Spectral Patterns (Landsat)
- **Vegetation**: High NIR (20000), Low Red (4000), creating positive NDVI
- **Urban**: Moderate NIR (10000), Higher Red (8000), creating lower NDVI
- **Water**: Very low all bands, creating negative NDVI/NDWI

### QA Pixel Values
- 21824: Clear land
- 21828: Clear water
- 21888: Cloud shadow
- 22080: Cloud

### Nighttime Lights Pattern
- Peak intensity ~26 in urban centers
- Road corridors with ~2-4 radiance
- Background noise ~0.5 radiance

## Usage in Tests

```python
# Load reference data paths
reference_data_paths = {
    "viirs": test_data_dir / "viirs" / "VNP46A2_h08v05_20231015.tif",
    "landsat": {
        "B3": test_data_dir / "landsat" / "LC08_L2SP_044034_20231015_B3.TIF",
        "B4": test_data_dir / "landsat" / "LC08_L2SP_044034_20231015_B4.TIF", 
        "B5": test_data_dir / "landsat" / "LC08_L2SP_044034_20231015_B5.TIF",
        "QA_PIXEL": test_data_dir / "landsat" / "LC08_L2SP_044034_20231015_QA_PIXEL.TIF",
    },
    "osm": test_data_dir / "osm" / "test_roads.gpkg"
}
```

## Limitations

1. **Geographic Alignment**: The Landsat data is slightly offset from the exact test bbox to simulate realistic conditions
2. **Temporal Coverage**: Only single date (no temporal compositing testing)
3. **Size**: Small area limits testing of large-scale patterns
4. **Synthetic Nature**: While realistic, may not capture all edge cases of real satellite data

## Regenerating Data

To regenerate this reference data:

```bash
python tools/fetch_reference_data.py
```

This will create new synthetic data with the same characteristics.