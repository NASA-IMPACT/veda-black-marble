"""Data generators for creating realistic test data.

This module provides functions to generate minimal but realistic test data
for Landsat, VIIRS, and OSM sources to support integration testing.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import json
from pathlib import Path


def generate_landsat_test_data(
    shape: Tuple[int, int] = (100, 100),
    n_dates: int = 1,
    cloud_coverage: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate realistic Landsat test data.
    
    Args:
        shape: Shape of the output arrays (height, width)
        n_dates: Number of dates to generate
        cloud_coverage: Fraction of pixels to mark as cloudy
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing Landsat bands and metadata
    """
    np.random.seed(seed)
    
    data_stack = {}
    
    for i in range(n_dates):
        # Create correlated bands representing vegetation
        base = np.random.beta(2, 5, shape)  # Natural texture
        
        # Band 3 (Green): 0.533-0.590 μm
        b3 = base * 0.15 + np.random.normal(0, 0.02, shape)
        b3 = np.clip(b3, 0.05, 0.25).astype(np.float32)
        
        # Band 4 (Red): 0.636-0.673 μm (absorbed by vegetation)
        b4 = base * 0.08 + np.random.normal(0, 0.01, shape)
        b4 = np.clip(b4, 0.02, 0.15).astype(np.float32)
        
        # Band 5 (NIR): 0.851-0.879 μm (reflected by vegetation)
        b5 = base * 0.4 + np.random.normal(0, 0.03, shape)
        b5 = np.clip(b5, 0.2, 0.6).astype(np.float32)
        
        # QA_PIXEL band (bit-packed quality flags)
        qa_pixel = np.full(shape, 21824, dtype=np.uint16)  # Clear land
        
        # Add clouds
        if cloud_coverage > 0:
            cloud_mask = np.random.rand(*shape) < cloud_coverage
            qa_pixel[cloud_mask] = 22280  # Cloud
            
            # Add cloud shadows nearby
            shadow_mask = np.roll(cloud_mask, 5, axis=0)
            qa_pixel[shadow_mask & (qa_pixel == 21824)] = 23888  # Cloud shadow
        
        # QA_AEROSOL band
        qa_aerosol = np.random.choice(
            [96, 224, 352],  # Low, medium, high aerosol
            size=shape,
            p=[0.7, 0.25, 0.05]
        ).astype(np.uint16)
        
        # QA_RADSAT band (radiometric saturation)
        qa_radsat = np.zeros(shape, dtype=np.uint16)
        # Add some saturation in bright areas
        bright_mask = (b5 > 0.5) | (b4 > 0.2)
        qa_radsat[bright_mask] = np.random.choice([0, 1, 3], size=bright_mask.sum())
        
        date_key = f"date_{i}"
        data_stack[date_key] = {
            'B3': b3,
            'B4': b4,
            'B5': b5,
            'QA_PIXEL': qa_pixel,
            'QA_AEROSOL': qa_aerosol,
            'QA_RADSAT': qa_radsat,
            'metadata': {
                'date': (datetime.now() - timedelta(days=i*5)).isoformat(),
                'cloud_cover': float(cloud_mask.mean() * 100) if cloud_coverage > 0 else 0,
                'scene_id': f'LC09_L2SP_013031_{datetime.now().strftime("%Y%m%d")}_02_T1',
            }
        }
    
    # If single date, return flattened structure
    if n_dates == 1:
        return data_stack['date_0']
    
    return data_stack


def generate_viirs_test_data(
    shape: Tuple[int, int] = (100, 100),
    urban_centers: int = 3,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate realistic VIIRS nighttime lights test data.
    
    Args:
        shape: Shape of the output arrays (height, width)
        urban_centers: Number of urban centers to simulate
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing VIIRS bands
    """
    np.random.seed(seed)
    
    # Initialize nighttime lights
    ntl = np.zeros(shape, dtype=np.float32)
    
    # Add urban centers
    for _ in range(urban_centers):
        # Random center location
        cx = np.random.randint(shape[0] // 4, 3 * shape[0] // 4)
        cy = np.random.randint(shape[1] // 4, 3 * shape[1] // 4)
        
        # Create 2D gaussian for urban light
        x, y = np.ogrid[:shape[0], :shape[1]]
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        
        # Urban parameters
        intensity = np.random.uniform(10, 50)
        sigma = np.random.uniform(5, 15)
        
        # Add gaussian blob
        urban_light = intensity * np.exp(-dist_sq / (2 * sigma ** 2))
        ntl += urban_light
    
    # Add rural/suburban background
    background = np.random.exponential(0.5, shape)
    ntl += background
    
    # Add road network effect (simplified)
    # Horizontal roads
    for i in range(0, shape[0], np.random.randint(10, 20)):
        width = np.random.randint(1, 3)
        brightness = np.random.uniform(1, 3)
        ntl[i:i+width, :] += brightness
    
    # Vertical roads
    for j in range(0, shape[1], np.random.randint(10, 20)):
        width = np.random.randint(1, 3)
        brightness = np.random.uniform(1, 3)
        ntl[:, j:j+width] += brightness
    
    # Clip to realistic range
    ntl = np.clip(ntl, 0, 100).astype(np.float32)
    
    # Quality flag (Mandatory_Quality_Flag)
    # 0 = High quality, 2 = Medium quality, 7 = Poor quality
    quality_flag = np.zeros(shape, dtype=np.uint8)
    
    # Most pixels are high quality
    quality_flag[:] = 0
    
    # Some medium quality areas (e.g., near bright lights)
    medium_qual_mask = (ntl > 30) & (np.random.rand(*shape) < 0.3)
    quality_flag[medium_qual_mask] = 2
    
    # Poor quality in very bright areas
    poor_qual_mask = ntl > 80
    quality_flag[poor_qual_mask] = 7
    
    # Lunar irradiance (simplified)
    lunar = np.full(shape, 10.0, dtype=np.float32)  # Standard lunar illumination
    
    # Add some variation
    lunar += np.random.normal(0, 0.5, shape)
    lunar = np.clip(lunar, 5, 15)
    
    return {
        'NTL': ntl,
        'DNB_At_Sensor_Radiance': ntl * 1.5,  # Related to NTL
        'QF': quality_flag,
        'Mandatory_Quality_Flag': quality_flag,
        'LUNAR': lunar,
        'Lunar_Irradiance': lunar,
        'metadata': {
            'acquisition_date': datetime.now().isoformat(),
            'satellite': 'SNPP',
            'product': 'VNP46A2',
        }
    }


def generate_osm_road_data(
    shape: Tuple[int, int] = (100, 100),
    n_major_roads: int = 2,
    n_minor_roads: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate simplified OSM road network data.
    
    Args:
        shape: Shape of the output array (height, width)
        n_major_roads: Number of major roads
        n_minor_roads: Number of minor roads
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing road raster and metadata
    """
    np.random.seed(seed)
    
    # Initialize road raster
    # 0 = no road, 1 = motorway, 2 = primary, 3 = secondary, 4 = residential
    roads = np.zeros(shape, dtype=np.uint8)
    
    # Add major roads (motorways and primary)
    for i in range(n_major_roads):
        if np.random.rand() > 0.5:
            # Horizontal major road
            y = np.random.randint(shape[0] // 4, 3 * shape[0] // 4)
            width = np.random.randint(2, 4)
            road_type = np.random.choice([1, 2])  # motorway or primary
            roads[y:y+width, :] = road_type
        else:
            # Vertical major road
            x = np.random.randint(shape[1] // 4, 3 * shape[1] // 4)
            width = np.random.randint(2, 4)
            road_type = np.random.choice([1, 2])
            roads[:, x:x+width] = road_type
    
    # Add minor roads
    for i in range(n_minor_roads):
        if np.random.rand() > 0.5:
            # Horizontal minor road
            y = np.random.randint(0, shape[0])
            road_type = np.random.choice([3, 4])  # secondary or residential
            roads[y, :] = np.where(roads[y, :] == 0, road_type, roads[y, :])
        else:
            # Vertical minor road
            x = np.random.randint(0, shape[1])
            road_type = np.random.choice([3, 4])
            roads[:, x] = np.where(roads[:, x] == 0, road_type, roads[:, x])
    
    # Road type mapping
    road_types = {
        0: 'none',
        1: 'motorway',
        2: 'primary',
        3: 'secondary',
        4: 'residential',
    }
    
    return {
        'roads': roads,
        'road_types': road_types,
        'metadata': {
            'source': 'OpenStreetMap',
            'extraction_date': datetime.now().isoformat(),
            'total_road_pixels': int((roads > 0).sum()),
            'road_density': float((roads > 0).mean()),
        }
    }


def generate_complete_test_scene(
    shape: Tuple[int, int] = (100, 100),
    include_clouds: bool = True,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Generate a complete test scene with all data sources.
    
    Args:
        shape: Shape of the output arrays (height, width)
        include_clouds: Whether to include clouds in Landsat data
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing all data sources
    """
    return {
        'landsat': generate_landsat_test_data(
            shape=shape,
            cloud_coverage=0.15 if include_clouds else 0.0,
            seed=seed
        ),
        'viirs': generate_viirs_test_data(
            shape=shape,
            urban_centers=3,
            seed=seed + 1
        ),
        'osm': generate_osm_road_data(
            shape=shape,
            n_major_roads=2,
            n_minor_roads=5,
            seed=seed + 2
        ),
    }


def save_test_scene_metadata(scene_data: Dict[str, Any], output_path: Path) -> None:
    """Save test scene metadata to JSON file.
    
    Args:
        scene_data: Complete scene data
        output_path: Path to save metadata JSON
    """
    metadata = {
        'landsat': scene_data['landsat'].get('metadata', {}),
        'viirs': scene_data['viirs'].get('metadata', {}),
        'osm': scene_data['osm'].get('metadata', {}),
        'generation_params': {
            'shape': scene_data['landsat']['B3'].shape,
            'timestamp': datetime.now().isoformat(),
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def create_mock_data_sources(test_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create mock data source objects that return test data.
    
    Args:
        test_data: Test data from generators
        
    Returns:
        Dictionary of mock data sources
    """
    from unittest.mock import Mock
    
    # Mock Landsat data source
    landsat_source = Mock()
    landsat_source.fetch.return_value = test_data['landsat']
    
    # Mock VIIRS data source
    viirs_source = Mock()
    viirs_source.fetch.return_value = test_data['viirs']
    
    # Mock OSM data source
    osm_source = Mock()
    osm_source.fetch.return_value = test_data['osm']
    
    return {
        'landsat_source': landsat_source,
        'viirs_source': viirs_source,
        'osm_source': osm_source,
    }


if __name__ == "__main__":
    # Example usage
    print("Generating test scene...")
    scene = generate_complete_test_scene(shape=(200, 200))
    
    print(f"Landsat B3 shape: {scene['landsat']['B3'].shape}")
    print(f"VIIRS NTL shape: {scene['viirs']['NTL'].shape}")
    print(f"OSM roads shape: {scene['osm']['roads'].shape}")
    
    print(f"\nLandsat cloud cover: {scene['landsat']['metadata']['cloud_cover']:.1f}%")
    print(f"VIIRS max radiance: {scene['viirs']['NTL'].max():.1f}")
    print(f"OSM road density: {scene['osm']['metadata']['road_density'] * 100:.1f}%")