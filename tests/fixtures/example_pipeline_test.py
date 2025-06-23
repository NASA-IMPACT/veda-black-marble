#!/usr/bin/env python3
"""Example of a complete pipeline integration test using the functional API.

This demonstrates how to test the pipeline end-to-end with minimal mocking,
focusing on data flow validation rather than exact numerical results.
"""

import numpy as np
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

# Import test data generators
from tests.fixtures.data_generators import (
    generate_complete_test_scene,
    create_mock_data_sources,
)

# Import pipeline config
from blackmarble.target_config import PipelineConfig

# Import functional API
from blackmarble.processing import (
    create_quality_mask,
    create_viirs_quality_mask,
    compute_statistics,
    enhance_contrast,
    enhance_urban_features,
    apply_colormap,
    create_temporal_composite,
)

# Import calculations
from blackmarble.calculations import (
    calculate_ndvi,
    calculate_ndwi,
    calculate_ndui,
)


def run_minimal_pipeline_test():
    """Run a minimal end-to-end pipeline test."""
    
    print("=== Minimal Pipeline Integration Test ===\n")
    
    # 1. Setup
    print("1. SETUP")
    print("-" * 50)
    
    bbox = (-74.05, 40.69, -73.95, 40.79)  # Small Manhattan area
    date = datetime(2023, 9, 15)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_output.tif"
        
        # Create pipeline config
        config = PipelineConfig.for_bm_parity(bbox, date, str(output_path))
        print(f"Config: {config.enhancement.quality_mode} mode")
        print(f"Bbox: {bbox}")
        print(f"Date: {date.strftime('%Y-%m-%d')}")
        
        # Generate test data
        test_scene = generate_complete_test_scene(shape=(100, 100))
        print(f"\nTest data shape: {test_scene['landsat']['B3'].shape}")
        
        # 2. Acquire Phase (mocked)
        print("\n2. ACQUIRE PHASE")
        print("-" * 50)
        
        # In real pipeline, this would fetch from data sources
        # For testing, we use the generated test data
        landsat_data = test_scene['landsat']
        viirs_data = test_scene['viirs']
        osm_data = test_scene['osm']
        
        print(f"Landsat bands: {list(landsat_data.keys())[:6]}")  # Exclude metadata
        print(f"VIIRS layers: {list(viirs_data.keys())[:5]}")
        print(f"OSM layers: {list(osm_data.keys())[:2]}")
        
        # 3. Prepare Phase - Quality Masking
        print("\n3. PREPARE PHASE - Quality Masking")
        print("-" * 50)
        
        # Create Landsat quality mask
        landsat_mask = create_quality_mask(
            qa_bands={
                'qa_pixel': landsat_data['QA_PIXEL'],
                'qa_aerosol': landsat_data['QA_AEROSOL'],
                'qa_radsat': landsat_data['QA_RADSAT'],
            },
            combine_method='and',
            cloud_strategy='conservative',
            cloud_dilate=config.preparation.qa_dilation_radius,
        )
        
        landsat_stats = compute_statistics(landsat_mask.astype(float))
        print(f"Landsat valid pixels: {landsat_stats['valid_fraction'] * 100:.1f}%")
        
        # Apply mask to bands
        for band in ['B3', 'B4', 'B5']:
            landsat_data[f'{band}_masked'] = np.where(
                landsat_mask, 
                landsat_data[band], 
                np.nan
            )
        
        # Create VIIRS quality mask
        viirs_mask = create_viirs_quality_mask(
            viirs_data['QF'],
            valid_flags=[0, 2],  # High and moderate quality
        )
        
        viirs_stats = compute_statistics(viirs_mask.astype(float))
        print(f"VIIRS valid pixels: {viirs_stats['valid_fraction'] * 100:.1f}%")
        
        # Apply mask to NTL
        viirs_data['NTL_masked'] = np.where(
            viirs_mask,
            viirs_data['NTL'],
            np.nan
        )
        
        # 4. Analyze Phase - Index Calculation
        print("\n4. ANALYZE PHASE - Index Calculation")
        print("-" * 50)
        
        # Calculate spectral indices
        ndvi = calculate_ndvi(
            landsat_data['B4_masked'],
            landsat_data['B5_masked']
        )
        
        ndwi = calculate_ndwi(
            landsat_data['B3_masked'],
            landsat_data['B5_masked']
        )
        
        # Calculate NDUI
        ndui_config = {
            'floor': config.analysis.ndui_floor,
            'water_threshold': 0.0,
            'road_enhancement_major': config.enhancement.road_major_factor,
            'road_enhancement_minor': config.enhancement.road_minor_factor,
            'epsilon': config.analysis.ndui_epsilon,
        }
        
        ndui = calculate_ndui(
            viirs_data['NTL_masked'],
            ndvi,
            ndwi,
            osm_data['roads'],
            ndui_config
        )
        
        # Print index statistics
        for name, index in [('NDVI', ndvi), ('NDWI', ndwi), ('NDUI', ndui)]:
            stats = compute_statistics(index)
            print(f"{name}: mean={stats['mean']:.3f}, "
                  f"range=[{stats['min']:.3f}, {stats['max']:.3f}]")
        
        # 5. Enhance Phase - Visualization
        print("\n5. ENHANCE PHASE - Visualization")
        print("-" * 50)
        
        # Apply urban enhancement
        enhanced = enhance_urban_features(
            ndui=ndui,
            roads=osm_data['roads'],
            ntl=viirs_data['NTL_masked'],
            enhancement_strength=0.3,
        )
        
        # Apply contrast enhancement
        contrast_enhanced = enhance_contrast(
            data=enhanced,
            method=config.enhancement.contrast_method,
            clip_percentiles=(2, 98),
            min_value=config.enhancement.colormap_range[0],
            max_value=config.enhancement.colormap_range[1],
        )
        
        # Apply colormap
        rgb_array = apply_colormap(
            data=contrast_enhanced,
            colormap=config.enhancement.colormap,
            vmin=0.0,
            vmax=1.0,
        )
        
        print(f"Enhancement method: {config.enhancement.contrast_method}")
        print(f"Colormap: {config.enhancement.colormap}")
        print(f"RGB output shape: {rgb_array.shape}")
        print(f"RGB dtype: {rgb_array.dtype}")
        
        # 6. Export Phase (simplified)
        print("\n6. EXPORT PHASE")
        print("-" * 50)
        
        # In real pipeline, this would generate COG
        # For testing, we just validate the output array
        assert rgb_array.shape == (3, 100, 100)
        assert rgb_array.dtype == np.uint8
        assert rgb_array.min() >= 0
        assert rgb_array.max() <= 255
        
        print(f"Output validation: PASSED")
        print(f"Would write to: {output_path}")
        
        # Summary
        print("\n=== PIPELINE TEST SUMMARY ===")
        print(f"✓ Data acquisition successful")
        print(f"✓ Quality masking applied")
        print(f"✓ Indices calculated")
        print(f"✓ Enhancement applied")
        print(f"✓ Output validated")
        print(f"✓ All phases completed successfully")
        
        return True


def test_configuration_modes():
    """Test different configuration modes."""
    
    print("\n\n=== Configuration Mode Tests ===\n")
    
    bbox = (-74.05, 40.69, -73.95, 40.79)
    date = datetime(2023, 9, 15)
    
    # Test 1: BM Parity Mode
    print("1. BM PARITY MODE")
    print("-" * 50)
    
    config_parity = PipelineConfig.for_bm_parity(bbox, date, "output_parity.tif")
    print(f"Quality mode: {config_parity.enhancement.quality_mode}")
    print(f"Contrast method: {config_parity.enhancement.contrast_method}")
    print(f"Use SciPy features: {config_parity.analysis.use_scipy_fill_gaps}")
    
    # Test 2: Enhanced Quality Mode
    print("\n2. ENHANCED QUALITY MODE")
    print("-" * 50)
    
    config_enhanced = PipelineConfig.for_enhanced_quality(bbox, date, "output_enhanced.tif")
    print(f"Quality mode: {config_enhanced.enhancement.quality_mode}")
    print(f"Contrast method: {config_enhanced.enhancement.contrast_method}")
    print(f"Use SciPy gap filling: {config_enhanced.analysis.use_scipy_fill_gaps}")
    print(f"Use SciPy feathering: {config_enhanced.analysis.use_scipy_feather}")
    print(f"Suppress blooming: {config_enhanced.enhancement.suppress_blooming}")
    print(f"Enhance small settlements: {config_enhanced.enhancement.enhance_small_settlements}")
    
    # Test 3: Environment Variable Override
    print("\n3. ENVIRONMENT VARIABLE OVERRIDE")
    print("-" * 50)
    
    import os
    os.environ['BLACKMARBLE_NTL_FLOOR'] = '0.5'
    os.environ['BLACKMARBLE_ROAD_ENHANCEMENT_MAJOR'] = '1.5'
    
    config_env = PipelineConfig.from_env(bbox, date, "output_env.tif")
    print(f"NTL floor (default=0.1): {config_env.analysis.ntl_floor}")
    print(f"Road enhancement major (default=1.3): {config_env.enhancement.road_major_factor}")
    
    # Clean up env vars
    del os.environ['BLACKMARBLE_NTL_FLOOR']
    del os.environ['BLACKMARBLE_ROAD_ENHANCEMENT_MAJOR']


def test_temporal_compositing():
    """Test temporal compositing with multiple dates."""
    
    print("\n\n=== Temporal Compositing Test ===\n")
    
    # Generate multi-date test data
    shape = (100, 100)
    n_dates = 5
    
    # Create time series
    data_stack = []
    timestamps = []
    
    for i in range(n_dates):
        # Generate data with increasing cloud cover
        cloud_frac = i * 0.1
        data = np.random.rand(*shape)
        
        # Add clouds
        cloud_mask = np.random.rand(*shape) < cloud_frac
        data[cloud_mask] = np.nan
        
        data_stack.append(data)
        timestamps.append(datetime(2023, 9, 1 + i * 5))
    
    data_stack = np.array(data_stack)
    
    print(f"Time series shape: {data_stack.shape}")
    print(f"Dates: {[d.strftime('%Y-%m-%d') for d in timestamps]}")
    
    # Test different composite methods
    methods = ['median', 'mean', 'percentile']
    
    for method in methods:
        if method == 'percentile':
            composite = create_temporal_composite(
                data_stack=data_stack,
                timestamps=timestamps,
                method=method,
                percentile=75,
                min_valid_observations=2,
            )
        else:
            composite = create_temporal_composite(
                data_stack=data_stack,
                timestamps=timestamps,
                method=method,
                min_valid_observations=2,
            )
        
        stats = compute_statistics(composite)
        print(f"\n{method.upper()} composite:")
        print(f"  Valid pixels: {stats['valid_fraction'] * 100:.1f}%")
        print(f"  Mean value: {stats['mean']:.3f}")
        print(f"  Value range: [{stats['min']:.3f}, {stats['max']:.3f}]")


if __name__ == "__main__":
    # Run all tests
    print("Running Black Marble Functional API Integration Tests\n")
    
    # Test 1: Minimal pipeline
    success = run_minimal_pipeline_test()
    
    # Test 2: Configuration modes
    test_configuration_modes()
    
    # Test 3: Temporal compositing
    test_temporal_compositing()
    
    print("\n\nAll tests completed successfully!")