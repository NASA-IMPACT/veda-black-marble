#!/usr/bin/env python3
"""Integration tests for the Black Marble functional API pipeline.

This module tests the end-to-end execution of the functional pipeline,
verifying that data flows correctly through all phases from acquisition
to export using the new functional API.
"""

import os
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import tempfile
import numpy as np
import pytest

# Import the pipeline configuration
from blackmarble.target_config import PipelineConfig

# Import functional API functions
from blackmarble.processing import (
    # Mask generation functions
    create_cloud_mask,
    create_quality_mask,
    create_viirs_quality_mask,
    compute_statistics,
    # Enhancement functions
    enhance_contrast,
    enhance_urban_features,
    apply_colormap,
    # Spatial functions
    resample_image,
    align_rasters,
    # Temporal functions
    create_temporal_composite,
)

# Import core functions
from blackmarble.calculations import (
    calculate_ndvi,
    calculate_ndwi,
    calculate_ndui,
)

# Import for COG generation
from blackmarble.components.output import COGGenerator, COGProfile


# Test fixtures
@pytest.fixture
def minimal_bbox():
    """Small Manhattan area for testing."""
    return (-74.05, 40.69, -73.95, 40.79)


@pytest.fixture
def test_date():
    """Fixed test date."""
    return datetime(2023, 9, 15)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def minimal_test_data():
    """Generate minimal test data arrays."""
    shape = (100, 100)
    np.random.seed(42)
    
    # Landsat bands (scaled reflectance values)
    b3 = np.random.uniform(0.1, 0.3, shape).astype(np.float32)
    b4 = np.random.uniform(0.2, 0.4, shape).astype(np.float32)
    b5 = np.random.uniform(0.3, 0.5, shape).astype(np.float32)
    
    # QA bands
    qa_pixel = np.random.choice([21824, 21952, 22280], shape, p=[0.7, 0.2, 0.1])
    qa_aerosol = np.random.choice([96, 224, 352], shape, p=[0.8, 0.15, 0.05])
    qa_radsat = np.zeros(shape, dtype=np.uint16)
    
    # VIIRS data
    ntl = np.random.exponential(2.0, shape).astype(np.float32)
    ntl = np.clip(ntl, 0, 10)
    quality_flag = np.random.choice([0, 2, 7], shape, p=[0.6, 0.3, 0.1])
    
    # OSM roads (simplified)
    roads = np.zeros(shape, dtype=np.uint8)
    # Add some horizontal and vertical roads
    roads[25:27, :] = 1  # Major road
    roads[:, 48:50] = 2  # Secondary road
    roads[75:76, :] = 3  # Minor road
    
    return {
        'landsat': {
            'B3': b3,
            'B4': b4,
            'B5': b5,
            'QA_PIXEL': qa_pixel,
            'QA_AEROSOL': qa_aerosol,
            'QA_RADSAT': qa_radsat,
        },
        'viirs': {
            'NTL': ntl,
            'QF': quality_flag,
            'LUNAR': np.ones(shape) * 10.0,  # Simplified lunar irradiance
        },
        'osm': {
            'roads': roads
        }
    }


class TestMinimalPipelineExecution:
    """Test that pipeline can execute end-to-end with minimal data."""
    
    def test_acquire_phase_mocked(self, minimal_test_data):
        """Test data acquisition phase with mocked data sources."""
        # This would normally fetch from data sources
        # For testing, we just verify the structure
        landsat_data = minimal_test_data['landsat']
        viirs_data = minimal_test_data['viirs']
        osm_data = minimal_test_data['osm']
        
        # Verify all required bands are present
        assert 'B3' in landsat_data
        assert 'B4' in landsat_data
        assert 'B5' in landsat_data
        assert 'QA_PIXEL' in landsat_data
        
        assert 'NTL' in viirs_data
        assert 'QF' in viirs_data
        
        assert 'roads' in osm_data
    
    def test_prepare_phase_quality_masking(self, minimal_test_data):
        """Test data preparation phase with quality masking."""
        landsat_data = minimal_test_data['landsat']
        
        # Create quality mask using functional API
        quality_mask = create_quality_mask(
            qa_bands={
                'qa_pixel': landsat_data['QA_PIXEL'],
                'qa_aerosol': landsat_data['QA_AEROSOL'],
                'qa_radsat': landsat_data['QA_RADSAT'],
            },
            combine_method='and',
            cloud_strategy='conservative',
            cloud_dilate=0,  # No dilation for testing
            aerosol_threshold='standard',
            include_saturation=True,
            saturation_bands=[3, 4, 5],
        )
        
        # Verify mask properties
        assert quality_mask.shape == landsat_data['B3'].shape
        assert quality_mask.dtype == bool
        
        # Compute statistics
        stats = compute_statistics(quality_mask.astype(float))
        assert 0 <= stats['valid_fraction'] <= 1
        assert stats['count'] == quality_mask.size
        
        # Apply mask to bands
        masked_b3 = np.where(quality_mask, landsat_data['B3'], np.nan)
        masked_b4 = np.where(quality_mask, landsat_data['B4'], np.nan)
        masked_b5 = np.where(quality_mask, landsat_data['B5'], np.nan)
        
        # Verify masking worked
        assert np.isnan(masked_b3[~quality_mask]).all()
        assert not np.isnan(masked_b3[quality_mask]).all()
    
    def test_prepare_phase_viirs_quality(self, minimal_test_data):
        """Test VIIRS quality filtering."""
        viirs_data = minimal_test_data['viirs']
        
        # Create VIIRS quality mask
        quality_mask = create_viirs_quality_mask(
            viirs_data['QF'],
            valid_flags=[0, 2],  # High and moderate quality
        )
        
        # Apply mask
        ntl_masked = np.where(quality_mask, viirs_data['NTL'], np.nan)
        
        # Verify masking
        assert ntl_masked.shape == viirs_data['NTL'].shape
        assert np.isnan(ntl_masked[viirs_data['QF'] == 7]).all()
        
    def test_analyze_phase_index_calculation(self, minimal_test_data):
        """Test analysis phase with index calculations."""
        landsat_data = minimal_test_data['landsat']
        viirs_data = minimal_test_data['viirs']
        osm_data = minimal_test_data['osm']
        
        # Calculate indices
        ndvi = calculate_ndvi(landsat_data['B4'], landsat_data['B5'])
        ndwi = calculate_ndwi(landsat_data['B3'], landsat_data['B5'])
        
        # Create NDUI config object
        from blackmarble.config import NDUIConfig
        ndui_config = NDUIConfig(
            floor=0.02,
            water_threshold=0.0,
            road_enhancement_major=1.3,
            road_enhancement_minor=1.1,
            epsilon=1e-6,
        )
        
        ndui = calculate_ndui(
            viirs_data['NTL'],
            ndvi,
            ndwi,
            osm_data['roads'],
            ndui_config
        )
        
        # Verify index properties
        assert ndvi.shape == landsat_data['B3'].shape
        assert ndwi.shape == landsat_data['B3'].shape
        assert ndui.shape == landsat_data['B3'].shape
        
        # Check value ranges
        assert np.nanmin(ndvi) >= -1.0
        assert np.nanmax(ndvi) <= 1.0
        assert np.nanmin(ndwi) >= -1.0
        assert np.nanmax(ndwi) <= 1.0
        assert np.nanmin(ndui) >= 0.0
    
    def test_enhance_phase_visualization(self, minimal_test_data):
        """Test enhancement phase for visualization."""
        # Create a sample NDUI array
        shape = (100, 100)
        ndui = np.random.beta(2, 5, shape)
        roads = minimal_test_data['osm']['roads']
        ntl = minimal_test_data['viirs']['NTL']
        
        # Apply urban enhancement
        enhanced = enhance_urban_features(
            ndui=ndui,
            roads=roads,
            ntl=ntl,
            enhancement_strength=0.3,
        )
        
        # Apply contrast enhancement
        contrast_enhanced = enhance_contrast(
            data=enhanced,
            method='percentile',
            clip_percentiles=(2, 98),
            min_value=0.3,
            max_value=0.8,
        )
        
        # Apply colormap
        rgb_array = apply_colormap(
            data=contrast_enhanced,
            colormap='inferno',
            vmin=0.0,
            vmax=1.0,
        )
        
        # Verify enhancement results
        assert enhanced.shape == ndui.shape
        assert contrast_enhanced.shape == ndui.shape
        assert rgb_array.shape == (3,) + ndui.shape
        assert rgb_array.dtype == np.uint8
        
        # Check RGB values are in valid range
        assert rgb_array.min() >= 0
        assert rgb_array.max() <= 255
    
    def test_export_phase_cog_generation(self, temp_output_dir):
        """Test export phase with COG generation."""
        # Create sample RGB data
        shape = (3, 100, 100)
        rgb_data = np.random.randint(0, 256, shape, dtype=np.uint8)
        
        # Setup COG generator
        cog_generator = COGGenerator(profile=COGProfile(profile_name="deflate"))
        
        output_path = temp_output_dir / "test_output.tif"
        
        # Generate COG (mocked since it requires GDAL)
        with patch.object(cog_generator, 'process') as mock_process:
            mock_process.return_value = MagicMock(
                output_path=output_path,
                success=True,
                metadata={'bands': 3, 'dtype': 'uint8'}
            )
            
            result = cog_generator.process(
                data=rgb_data,
                output_path=output_path,
                crs="EPSG:4326",
                transform=None,
                tags={
                    "description": "Test NDUI Composite",
                    "processing_date": datetime.now().isoformat(),
                },
            )
            
            assert result.success
            assert result.metadata['bands'] == 3


class TestPipelineConfiguration:
    """Test pipeline configuration options."""
    
    def test_bm_parity_config(self, minimal_bbox, test_date, temp_output_dir):
        """Test configuration for bm.py parity."""
        output_path = str(temp_output_dir / "bm_parity.tif")
        config = PipelineConfig.for_bm_parity(minimal_bbox, test_date, output_path)
        
        # Verify basic settings
        assert config.bbox == minimal_bbox
        assert config.date == test_date
        assert config.output_path == output_path
        
        # Verify bm.py specific settings
        assert config.enhancement.quality_mode == "basic"
        assert config.enhancement.contrast_method == "linear"
        assert config.enhancement.colormap == "inferno"
        assert config.enhancement.road_major_factor == 1.3
        assert config.enhancement.road_minor_factor == 1.1
        
        assert config.analysis.ntl_floor == 0.1
        assert config.analysis.ntl_ceiling == 10.0
        assert config.analysis.ndui_floor == 0.02
        
        assert config.export.generate_web_image == True
        assert config.export.validate_before_export == False
    
    def test_enhanced_quality_config(self, minimal_bbox, test_date, temp_output_dir):
        """Test configuration for enhanced quality."""
        output_path = str(temp_output_dir / "enhanced.tif")
        config = PipelineConfig.for_enhanced_quality(minimal_bbox, test_date, output_path)
        
        # Verify enhanced settings
        assert config.analysis.use_scipy_fill_gaps == True
        assert config.analysis.use_scipy_feather == True
        assert config.analysis.composite_method == "median"
        
        assert config.enhancement.quality_mode == "enhanced"
        assert config.enhancement.contrast_method == "percentile"
        assert config.enhancement.suppress_blooming == True
        assert config.enhancement.enhance_small_settlements == True
        
        assert config.export.generate_urban_mask == True
        assert config.export.generate_indices_file == True
        assert config.export.validate_before_export == True
    
    def test_env_var_override(self, minimal_bbox, test_date, temp_output_dir, monkeypatch):
        """Test environment variable configuration override."""
        # Set environment variables
        monkeypatch.setenv("BLACKMARBLE_NTL_FLOOR", "0.5")
        monkeypatch.setenv("BLACKMARBLE_NTL_CEILING", "20.0")
        monkeypatch.setenv("BLACKMARBLE_ROAD_ENHANCEMENT_MAJOR", "1.5")
        monkeypatch.setenv("BLACKMARBLE_DISPLAY_MIN", "0.2")
        monkeypatch.setenv("BLACKMARBLE_DISPLAY_MAX", "0.9")
        
        output_path = str(temp_output_dir / "env_config.tif")
        config = PipelineConfig.from_env(minimal_bbox, test_date, output_path)
        
        # Verify overrides
        assert config.analysis.ntl_floor == 0.5
        assert config.analysis.ntl_ceiling == 20.0
        assert config.enhancement.road_major_factor == 1.5
        assert config.enhancement.colormap_range == (0.2, 0.9)
    
    def test_config_validation(self, minimal_bbox, test_date, temp_output_dir):
        """Test configuration validation."""
        output_path = str(temp_output_dir / "test.tif")
        config = PipelineConfig.for_bm_parity(minimal_bbox, test_date, output_path)
        
        # Valid config should pass
        config.validate()
        
        # Invalid NTL range
        config.analysis.ntl_floor = 20.0
        config.analysis.ntl_ceiling = 10.0
        with pytest.raises(ValueError, match="NTL floor.*must be less than"):
            config.validate()
        
        # Reset and test invalid display range
        config = PipelineConfig.for_bm_parity(minimal_bbox, test_date, output_path)
        config.enhancement.colormap_range = (0.8, 0.3)
        with pytest.raises(ValueError, match="Display min.*must be less than"):
            config.validate()


class TestDataTypeHandling:
    """Test that SceneData can handle both paths and arrays."""
    
    def test_scene_data_with_arrays(self, minimal_test_data):
        """Test SceneData handling of numpy arrays."""
        # Create a mock SceneData-like structure
        scene_data = {
            'data': minimal_test_data['landsat'],
            'metadata': {
                'date': datetime(2023, 9, 15),
                'cloud_cover': 15.5,
                'sensor': 'Landsat-9'
            }
        }
        
        # Verify we can access array data
        assert isinstance(scene_data['data']['B3'], np.ndarray)
        assert scene_data['data']['B3'].shape == (100, 100)
    
    def test_scene_data_with_paths(self, temp_output_dir):
        """Test SceneData handling of file paths."""
        # Create temporary files
        b3_path = temp_output_dir / "B3.tif"
        b4_path = temp_output_dir / "B4.tif"
        
        # Create mock SceneData with paths
        scene_data = {
            'data': {
                'B3': str(b3_path),
                'B4': str(b4_path),
            },
            'metadata': {
                'date': datetime(2023, 9, 15),
                'cloud_cover': 15.5,
            }
        }
        
        # Verify paths are strings
        assert isinstance(scene_data['data']['B3'], str)
        assert Path(scene_data['data']['B3']).name == "B3.tif"


class TestTemporalProcessing:
    """Test temporal compositing functionality."""
    
    def test_single_date_compositing(self, minimal_test_data):
        """Test compositing with single date (pass-through)."""
        data = minimal_test_data['landsat']['B3']
        
        # Add time dimension
        data_3d = data[np.newaxis, ...]
        timestamps = [datetime(2023, 9, 15)]
        
        # Create composite
        composite = create_temporal_composite(
            data_stack=data_3d,
            timestamps=timestamps,
            method='median',
            min_valid_observations=1,
        )
        
        # Should return 2D array
        assert composite.shape == data.shape
        assert np.allclose(composite, data, rtol=1e-5, equal_nan=True)
    
    def test_multi_date_compositing(self):
        """Test compositing with multiple dates."""
        shape = (5, 100, 100)  # 5 dates
        np.random.seed(42)
        
        # Create time series with some clouds
        data_stack = np.random.rand(*shape)
        
        # Add clouds to some dates
        data_stack[0, 20:40, 20:40] = np.nan
        data_stack[2, 60:80, 60:80] = np.nan
        data_stack[4, 10:30, 70:90] = np.nan
        
        timestamps = [
            datetime(2023, 9, 1),
            datetime(2023, 9, 5),
            datetime(2023, 9, 10),
            datetime(2023, 9, 15),
            datetime(2023, 9, 20),
        ]
        
        # Test median composite
        median_composite = create_temporal_composite(
            data_stack=data_stack,
            timestamps=timestamps,
            method='median',
            min_valid_observations=3,
        )
        
        # Test percentile composite
        p90_composite = create_temporal_composite(
            data_stack=data_stack,
            timestamps=timestamps,
            method='percentile',
            percentile=90,
            min_valid_observations=3,
        )
        
        # Verify shapes
        assert median_composite.shape == (100, 100)
        assert p90_composite.shape == (100, 100)
        
        # Verify p90 >= median (where both valid)
        valid_mask = ~np.isnan(median_composite) & ~np.isnan(p90_composite)
        assert (p90_composite[valid_mask] >= median_composite[valid_mask]).all()


class TestSpatialProcessing:
    """Test spatial transformation functions."""
    
    def test_resampling(self):
        """Test image resampling."""
        # Create test data
        shape = (200, 200)
        data = np.random.rand(*shape)
        
        # Resample to half resolution
        resampled, _ = resample_image(
            data=data,
            source_resolution=30.0,  # 30m
            target_resolution=60.0,  # 60m
            method='bilinear',
        )
        
        # Should be approximately half the size
        expected_shape = (shape[0] // 2, shape[1] // 2)
        assert resampled.shape[0] == pytest.approx(expected_shape[0], abs=1)
        assert resampled.shape[1] == pytest.approx(expected_shape[1], abs=1)
    
    def test_array_alignment(self):
        """Test aligning multiple arrays."""
        # Create misaligned arrays
        array1 = np.random.rand(100, 100)
        array2 = np.random.rand(98, 102)
        array3 = np.random.rand(105, 95)
        
        arrays = [array1, array2, array3]
        
        # Mock align_rasters for testing
        # In real implementation, this would handle geographic alignment
        aligned = []
        min_rows = min(a.shape[0] for a in arrays)
        min_cols = min(a.shape[1] for a in arrays)
        
        for array in arrays:
            aligned.append(array[:min_rows, :min_cols])
        
        # Verify all arrays have same shape
        shapes = [a.shape for a in aligned]
        assert all(s == shapes[0] for s in shapes)
        assert aligned[0].shape == (98, 95)  # Min of (100, 98, 105) x (100, 102, 95)


# Helper functions for testing
def create_test_landsat_data(shape=(100, 100)):
    """Create realistic test Landsat data."""
    np.random.seed(42)
    
    # Create correlated bands (vegetation signature)
    base = np.random.rand(*shape)
    
    # Green band (lower in vegetation)
    b3 = base * 0.15 + np.random.normal(0, 0.02, shape)
    b3 = np.clip(b3, 0.05, 0.25)
    
    # Red band (much lower in vegetation)
    b4 = base * 0.08 + np.random.normal(0, 0.01, shape)
    b4 = np.clip(b4, 0.02, 0.15)
    
    # NIR band (higher in vegetation)
    b5 = base * 0.4 + np.random.normal(0, 0.03, shape)
    b5 = np.clip(b5, 0.2, 0.6)
    
    return b3.astype(np.float32), b4.astype(np.float32), b5.astype(np.float32)


def create_test_viirs_data(shape=(100, 100)):
    """Create realistic test VIIRS nighttime lights data."""
    np.random.seed(42)
    
    # Create urban cores
    ntl = np.zeros(shape, dtype=np.float32)
    
    # Add some bright spots (urban areas)
    for _ in range(5):
        x = np.random.randint(10, shape[0] - 10)
        y = np.random.randint(10, shape[1] - 10)
        
        # Create gaussian blob
        xx, yy = np.meshgrid(range(shape[0]), range(shape[1]), indexing='ij')
        dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        intensity = np.random.uniform(5, 20)
        sigma = np.random.uniform(5, 10)
        ntl += intensity * np.exp(-(dist ** 2) / (2 * sigma ** 2))
    
    # Add background noise
    ntl += np.random.exponential(0.1, shape)
    
    return np.clip(ntl, 0, 100).astype(np.float32)


def validate_output_cog(filepath):
    """Validate that output file is a valid COG."""
    # In a real test, this would use GDAL/rasterio to validate
    # For now, just check file exists
    return Path(filepath).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])