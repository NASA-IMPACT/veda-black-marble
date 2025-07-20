#!/usr/bin/env python3
"""Create reference data for pipeline tests.

This script creates realistic synthetic VIIRS, Landsat, and OSM data
for use in the test suite. The data is stored in tests/data/reference/
and committed to version control.

Note: For actual downloaded data, you would need to set up EARTHDATA
credentials and use the pipeline's download functionality.
"""

import shutil
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds, from_origin
from rasterio.enums import Resampling
from shapely.geometry import LineString

# Very small test area in San Francisco (about 1km x 1km)
TEST_BBOX = (-122.42, 37.77, -122.41, 37.78)  # (west, south, east, north)
TEST_DATE = datetime(2023, 10, 15)

# Output directory
REFERENCE_DIR = Path(__file__).parent.parent / "tests" / "data" / "reference"


def ensure_clean_directory(path: Path) -> None:
    """Ensure directory exists and is empty."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def create_viirs_data() -> None:
    """Create realistic VIIRS nighttime lights data."""
    print("\n📡 Creating VIIRS data...")
    
    viirs_dir = REFERENCE_DIR / "viirs"
    ensure_clean_directory(viirs_dir)
    create_synthetic_viirs(viirs_dir)


def create_landsat_data() -> None:
    """Create realistic Landsat data."""
    print("\n🛰️  Creating Landsat data...")
    
    landsat_dir = REFERENCE_DIR / "landsat"
    ensure_clean_directory(landsat_dir)
    create_synthetic_landsat(landsat_dir)


def create_osm_data() -> None:
    """Create realistic OSM road data."""
    print("\n🗺️  Creating OSM data...")
    
    osm_dir = REFERENCE_DIR / "osm"
    ensure_clean_directory(osm_dir)
    create_synthetic_osm(osm_dir)


def create_synthetic_viirs(output_dir: Path) -> None:
    """Create realistic synthetic VIIRS data."""
    # Smaller size for version control
    shape = (150, 150)
    
    # Create base nighttime lights pattern
    # Use exponential distribution to simulate light intensity
    data = np.random.exponential(0.5, shape).astype(np.float32)
    
    # Add urban centers with gaussian patterns
    y, x = np.ogrid[:shape[0], :shape[1]]
    
    # Main urban center
    center_y, center_x = 75, 75
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    urban_intensity = 20 * np.exp(-dist_from_center**2 / 500)
    
    # Secondary urban area
    center2_y, center2_x = 40, 100
    dist_from_center2 = np.sqrt((x - center2_x)**2 + (y - center2_y)**2)
    urban_intensity2 = 10 * np.exp(-dist_from_center2**2 / 300)
    
    # Combine patterns
    data += urban_intensity + urban_intensity2
    
    # Add some roads (linear features)
    data[70:80, :] += 2  # Horizontal road
    data[:, 70:80] += 2  # Vertical road
    
    # Add noise and clip
    data += np.random.normal(0, 0.2, shape)
    data = np.clip(data, 0, 50)
    
    # Create transform
    transform = from_bounds(*TEST_BBOX, shape[1], shape[0])
    
    profile = {
        'driver': 'GTiff',
        'height': shape[0],
        'width': shape[1],
        'count': 1,
        'dtype': 'float32',
        'crs': 'EPSG:4326',
        'transform': transform,
        'compress': 'deflate',
        'predictor': 3,  # Floating point predictor
        'tiled': False  # Smaller files don't need tiling
    }
    
    output_path = output_dir / f"VNP46A2_h08v05_{TEST_DATE.strftime('%Y%m%d')}.tif"
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data, 1)
        dst.build_overviews([2, 4], Resampling.average)
        dst.update_tags(
            PRODUCT='VNP46A2',
            LAYER='Gap_Filled_DNB_BRDF-Corrected_NTL',
            DATE=TEST_DATE.strftime('%Y-%m-%d')
        )
    
    print(f"  Created {output_path.name}: {shape} pixels, {output_path.stat().st_size / 1024:.1f} KB")


def create_synthetic_landsat(output_dir: Path) -> None:
    """Create realistic synthetic Landsat data."""
    # Smaller size for version control
    shape = (150, 150)
    
    # Create transform for UTM Zone 10N (San Francisco area)
    # Approximate UTM coordinates for the test bbox
    west_utm = 551650
    north_utm = 4181900
    pixel_size = 30
    transform = from_origin(west_utm, north_utm, pixel_size, pixel_size)
    
    # Create realistic spectral patterns
    # Base reflectance values (scaled to uint16)
    base_values = {
        "B3": 8000,   # Green - vegetation reflects well
        "B4": 6000,   # Red - absorbed by vegetation
        "B5": 15000,  # NIR - high reflectance from vegetation
    }
    
    # Create spatial patterns
    y, x = np.ogrid[:shape[0], :shape[1]]
    
    # Vegetation areas (parks)
    veg_mask = ((x - 50)**2 + (y - 50)**2) < 900  # Park area
    
    # Urban areas (lower vegetation)
    urban_mask = ~veg_mask
    
    # Water body (very low reflectance)
    water_mask = (x > 100) & (y < 30)
    
    for band_name, base_value in base_values.items():
        data = np.full(shape, base_value, dtype=np.uint16)
        
        if band_name == "B5":  # NIR
            data[veg_mask] = 20000  # High NIR for vegetation
            data[urban_mask] = 10000  # Lower for urban
            data[water_mask] = 2000  # Very low for water
        elif band_name == "B4":  # Red
            data[veg_mask] = 4000  # Low red for vegetation
            data[urban_mask] = 8000  # Higher for urban
            data[water_mask] = 3000  # Low for water
        elif band_name == "B3":  # Green
            data[veg_mask] = 9000  # Moderate green for vegetation
            data[urban_mask] = 7500  # Moderate for urban
            data[water_mask] = 4000  # Low for water
        
        # Add realistic noise
        noise = np.random.normal(0, 200, shape).astype(np.int16)
        data = np.clip(data.astype(np.int32) + noise, 0, 65535).astype(np.uint16)
        
        profile = {
            'driver': 'GTiff',
            'height': shape[0],
            'width': shape[1],
            'count': 1,
            'dtype': 'uint16',
            'crs': 'EPSG:32610',  # UTM Zone 10N
            'transform': transform,
            'compress': 'deflate',
            'predictor': 2,  # Horizontal differencing for integers
            'tiled': False
        }
        
        output_path = output_dir / f"LC08_L2SP_044034_{TEST_DATE.strftime('%Y%m%d')}_{band_name}.TIF"
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data, 1)
            dst.update_tags(
                SATELLITE='LANDSAT_8',
                SENSOR='OLI_TIRS',
                DATE_ACQUIRED=TEST_DATE.strftime('%Y-%m-%d'),
                COLLECTION='02',
                LEVEL='L2SP'
            )
        
        print(f"  Created {output_path.name}: {shape} pixels, {output_path.stat().st_size / 1024:.1f} KB")
    
    # Create QA_PIXEL band
    qa_data = np.full(shape, 21824, dtype=np.uint16)  # Clear land pixel
    
    # Add some clouds
    cloud_mask = ((x - 100)**2 + (y - 100)**2) < 400
    qa_data[cloud_mask] = 22080  # Cloud pixel
    
    # Add cloud shadows
    shadow_mask = ((x - 110)**2 + (y - 110)**2) < 400
    qa_data[shadow_mask] = 21888  # Cloud shadow
    
    # Water pixels
    qa_data[water_mask] = 21828  # Clear water
    
    qa_profile = profile.copy()
    qa_output = output_dir / f"LC08_L2SP_044034_{TEST_DATE.strftime('%Y%m%d')}_QA_PIXEL.TIF"
    
    with rasterio.open(qa_output, 'w', **qa_profile) as dst:
        dst.write(qa_data, 1)
        dst.update_tags(
            DESCRIPTION='Landsat Collection 2 QA Pixel Band',
            BIT_0='Fill',
            BIT_1='Dilated Cloud',
            BIT_2='Cirrus',
            BIT_3='Cloud',
            BIT_4='Cloud Shadow',
            BIT_5='Snow',
            BIT_6='Clear',
            BIT_7='Water'
        )
    
    print(f"  Created {qa_output.name}: {shape} pixels, {qa_output.stat().st_size / 1024:.1f} KB")


def create_synthetic_osm(output_dir: Path) -> None:
    """Create realistic synthetic OSM road network."""
    # Create a more realistic road network for San Francisco test area
    roads = []
    highway_types = []
    
    # Major streets (grid pattern typical of SF)
    # East-West streets
    for y in np.linspace(37.771, 37.779, 5):
        roads.append(LineString([
            (-122.419, y), (-122.411, y)
        ]))
        highway_types.append('secondary' if y == 37.775 else 'residential')
    
    # North-South streets
    for x in np.linspace(-122.419, -122.411, 5):
        roads.append(LineString([
            (x, 37.771), (x, 37.779)
        ]))
        highway_types.append('primary' if x == -122.415 else 'residential')
    
    # Add some diagonal streets (common in SF)
    roads.append(LineString([
        (-122.418, 37.772), (-122.412, 37.778)
    ]))
    highway_types.append('tertiary')
    
    # Add a curved road (simulate highway ramp)
    t = np.linspace(0, np.pi/2, 20)
    x = -122.420 + 0.008 * np.cos(t)
    y = 37.770 + 0.008 * np.sin(t)
    roads.append(LineString(list(zip(x, y))))
    highway_types.append('motorway_link')
    
    # Create GeoDataFrame with additional attributes
    gdf = gpd.GeoDataFrame({
        'osm_id': range(len(roads)),
        'highway': highway_types,
        'name': [f'Test Street {i}' if ht == 'residential' else 
                f'Test Avenue {i}' if ht == 'secondary' else
                f'Highway {i}' for i, ht in enumerate(highway_types)],
        'lanes': [4 if ht in ['primary', 'motorway_link'] else 
                 2 if ht == 'secondary' else 1 
                 for ht in highway_types],
        'maxspeed': [65 if ht == 'motorway_link' else
                    35 if ht == 'primary' else
                    25 for ht in highway_types],
        'geometry': roads
    }, crs='EPSG:4326')
    
    output_path = output_dir / "test_roads.gpkg"
    gdf.to_file(output_path, driver='GPKG', layer='roads')
    
    print(f"  Created {output_path.name}: {len(roads)} road segments, {output_path.stat().st_size / 1024:.1f} KB")


def check_total_size() -> float:
    """Check total size of reference data."""
    total_size = 0
    for path in REFERENCE_DIR.rglob("*"):
        if path.is_file():
            total_size += path.stat().st_size
    return total_size / (1024 * 1024)  # MB


def main():
    """Main function to create all reference data."""
    print("🚀 Creating reference data for Black Marble pipeline tests")
    print(f"   Test area: {TEST_BBOX}")
    print(f"   Test date: {TEST_DATE}")
    print(f"   Output dir: {REFERENCE_DIR}")
    
    # Create reference directory
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic data
    print("\n📝 Note: Creating realistic synthetic data for testing")
    print("   For actual satellite data, use the pipeline's download functionality")
    print("   with valid EARTHDATA credentials")
    
    # Create data
    create_viirs_data()
    create_landsat_data()
    create_osm_data()
    
    # Check total size
    total_mb = check_total_size()
    print(f"\n📊 Total reference data size: {total_mb:.2f} MB")
    
    if total_mb > 5:
        print("⚠️  Warning: Reference data exceeds 5 MB!")
        print("   Consider reducing the test area or image dimensions")
    else:
        print("✅ Reference data size is acceptable for version control")
    
    # Show file listing
    print("\n📁 Created files:")
    for path in sorted(REFERENCE_DIR.rglob("*")):
        if path.is_file():
            size_kb = path.stat().st_size / 1024
            print(f"   {path.relative_to(REFERENCE_DIR)}: {size_kb:.1f} KB")
    
    print("\n✨ Done! Reference data is ready for commit.")
    print("   Add these files to git:")
    print(f"   git add {REFERENCE_DIR}")


if __name__ == "__main__":
    main()