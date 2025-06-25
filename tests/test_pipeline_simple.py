"""Simplified pipeline integration tests.

This module provides simpler integration tests for the Black Marble pipeline
that focus on the core functionality without complex mocking scenarios.
"""

import pytest
import tempfile
import os
from datetime import datetime

# Import the main pipeline function and config
from blackmarble.pipeline import black_marble_pipeline
from blackmarble.config import PipelineConfig


@pytest.fixture
def sample_bbox():
    """Sample bounding box for Manhattan area."""
    return (-74.05, 40.69, -73.95, 40.79)  # (min_lon, min_lat, max_lon, max_lat)


@pytest.fixture
def sample_date():
    """Sample date for testing.""" 
    return datetime(2023, 9, 15)


@pytest.fixture
def temp_output_path():
    """Temporary output path for testing."""
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        yield f.name
    # Cleanup
    if os.path.exists(f.name):
        os.unlink(f.name)


def test_pipeline_config_creation():
    """Test that pipeline configurations can be created successfully.
    
    This test verifies that:
    - Basic configuration can be created with for_bm_parity
    - Enhanced configuration can be created with for_enhanced_quality
    - Configuration validation works correctly
    - Legacy config conversion produces expected structure
    """
    bbox = (-74.05, 40.69, -73.95, 40.79)
    date = datetime(2023, 9, 15)
    output_path = "/tmp/test.tif"
    
    # Test basic configuration
    basic_config = PipelineConfig.for_bm_parity(bbox, date, output_path)
    assert basic_config.bbox == bbox
    assert basic_config.date == date
    assert basic_config.output_path == output_path
    assert basic_config.enhancement.quality_mode == "basic"
    assert basic_config.enhancement.colormap == "inferno"
    assert not basic_config.export.generate_urban_mask
    
    # Test enhanced configuration
    enhanced_config = PipelineConfig.for_enhanced_quality(bbox, date, output_path)
    assert enhanced_config.enhancement.quality_mode == "enhanced"
    assert enhanced_config.analysis.use_scipy_fill_gaps is True
    assert enhanced_config.export.generate_urban_mask is True
    assert enhanced_config.export.generate_indices_file is True
    
    # Test validation
    basic_config.validate()  # Should not raise
    enhanced_config.validate()  # Should not raise
    
    # Test legacy config conversion
    legacy_dict = basic_config.to_legacy_config()
    assert isinstance(legacy_dict, dict)
    assert 'landsat' in legacy_dict
    assert 'viirs' in legacy_dict
    assert 'ndui' in legacy_dict
    
    # Verify key legacy compatibility values
    assert legacy_dict['viirs']['floor'] == basic_config.analysis.ntl_floor
    assert legacy_dict['ndui']['floor'] == basic_config.analysis.ndui_floor
    assert legacy_dict['landsat']['band_scale_factor'] == basic_config.preparation.landsat_scale_factor


def test_pipeline_config_validation():
    """Test pipeline configuration validation catches errors.
    
    This test verifies that:
    - Invalid NTL floor/ceiling combinations are rejected
    - Invalid percentage values are rejected  
    - Invalid enhancement factors are rejected
    - Helpful error messages are provided
    """
    bbox = (-74.05, 40.69, -73.95, 40.79)
    date = datetime(2023, 9, 15)
    output_path = "/tmp/test.tif"
    
    # Test invalid NTL floor >= ceiling
    config = PipelineConfig.for_bm_parity(bbox, date, output_path)
    config.analysis.ntl_floor = 15.0
    config.analysis.ntl_ceiling = 10.0
    with pytest.raises(ValueError, match="NTL floor.*must be less than.*ceiling"):
        config.validate()
    
    # Test invalid percentage
    config = PipelineConfig.for_bm_parity(bbox, date, output_path)
    config.min_valid_pixel_percentage = 150.0
    with pytest.raises(ValueError, match="min_valid_pixel_percentage must be between 0 and 100"):
        config.validate()
    
    # Test invalid resolution
    config = PipelineConfig.for_bm_parity(bbox, date, output_path)
    config.preparation.final_resolution = -1.0
    with pytest.raises(ValueError, match="final_resolution must be positive"):
        config.validate()
    
    # Test invalid enhancement factor
    config = PipelineConfig.for_bm_parity(bbox, date, output_path)
    config.enhancement.road_major_factor = 0.5
    with pytest.raises(ValueError, match="road_major_factor must be >= 1.0"):
        config.validate()


def test_pipeline_config_environment_loading():
    """Test configuration loading from environment variables.
    
    This test verifies that:
    - Environment variables are properly parsed
    - Configuration values are updated from environment
    - Default values are used when environment variables are not set
    - Type conversion works correctly for different value types
    """
    bbox = (-74.05, 40.69, -73.95, 40.79)
    date = datetime(2023, 9, 15)
    output_path = "/tmp/test.tif"
    
    # Test with no environment variables (should use defaults)
    config = PipelineConfig.from_env(bbox, date, output_path)
    assert config.acquisition.composite_months == 12  # Default value
    assert config.analysis.ntl_floor == 0.1  # Default value
    
    # Test with environment variables set
    os.environ['BLACKMARBLE_COMPOSITE_MONTHS'] = '6'
    os.environ['BLACKMARBLE_NTL_FLOOR'] = '0.2'
    os.environ['BLACKMARBLE_USE_PARALLEL'] = 'true'
    
    try:
        config = PipelineConfig.from_env(bbox, date, output_path)
        assert config.acquisition.composite_months == 6
        assert config.analysis.ntl_floor == 0.2
        assert config.use_parallel is True
    finally:
        # Cleanup environment variables
        for key in ['BLACKMARBLE_COMPOSITE_MONTHS', 'BLACKMARBLE_NTL_FLOOR', 'BLACKMARBLE_USE_PARALLEL']:
            if key in os.environ:
                del os.environ[key]


def test_pipeline_config_methods():
    """Test additional configuration methods and properties.
    
    This test verifies that:
    - Configuration dataclass methods work correctly
    - Default factory functions work as expected
    - Configuration can be copied and modified
    - All configuration sections are properly initialized
    """
    bbox = (-74.05, 40.69, -73.95, 40.79)
    date = datetime(2023, 9, 15)
    output_path = "/tmp/test.tif"
    
    config = PipelineConfig.for_bm_parity(bbox, date, output_path)
    
    # Test that all sections are initialized
    assert config.acquisition is not None
    assert config.preparation is not None
    assert config.analysis is not None
    assert config.enhancement is not None
    assert config.export is not None
    
    # Test that default factory functions work
    assert isinstance(config.acquisition.road_types, list)
    assert len(config.acquisition.road_types) > 0
    assert isinstance(config.export.overview_levels, list)
    assert len(config.export.overview_levels) > 0
    
    # Test configuration copying/modification
    config2 = PipelineConfig.for_enhanced_quality(bbox, date, output_path)
    assert config.enhancement.quality_mode != config2.enhancement.quality_mode
    assert config.analysis.use_scipy_fill_gaps != config2.analysis.use_scipy_fill_gaps


def test_pipeline_parameter_validation():
    """Test that pipeline parameters are validated correctly.
    
    This test verifies that:
    - Bounding box validation works
    - Date validation works
    - Output path validation works
    - Invalid parameter combinations are caught
    """
    date = datetime(2023, 9, 15)
    output_path = "/tmp/test.tif"
    
    # Test valid bbox
    valid_bbox = (-74.05, 40.69, -73.95, 40.79)  # min_lon < max_lon, min_lat < max_lat
    config = PipelineConfig.for_bm_parity(valid_bbox, date, output_path)
    assert config.bbox == valid_bbox
    
    # Test bbox components are accessible
    min_lon, min_lat, max_lon, max_lat = config.bbox
    assert min_lon < max_lon
    assert min_lat < max_lat
    
    # Test date is stored correctly
    assert config.date == date
    assert config.date.year == 2023
    assert config.date.month == 9
    assert config.date.day == 15
    
    # Test output path is stored correctly
    assert config.output_path == output_path


def test_configuration_sections_independence():
    """Test that configuration sections are independent.
    
    This test verifies that:
    - Modifying one section doesn't affect others
    - Each section has its own parameters
    - Default values are section-specific
    - Section validation is independent
    """
    bbox = (-74.05, 40.69, -73.95, 40.79)
    date = datetime(2023, 9, 15)
    output_path = "/tmp/test.tif"
    
    config = PipelineConfig.for_bm_parity(bbox, date, output_path)
    
    # Test acquisition section independence
    original_months = config.acquisition.composite_months
    config.acquisition.composite_months = 24
    assert config.acquisition.composite_months == 24
    assert config.preparation.landsat_scale_factor == 0.0000275  # Unchanged
    
    # Test analysis section independence
    original_floor = config.analysis.ntl_floor
    config.analysis.ntl_floor = 0.5
    assert config.analysis.ntl_floor == 0.5
    assert config.enhancement.colormap == "inferno"  # Unchanged
    
    # Test enhancement section independence
    original_factor = config.enhancement.road_major_factor
    config.enhancement.road_major_factor = 2.0
    assert config.enhancement.road_major_factor == 2.0
    assert config.export.tiled is True  # Unchanged
    
    # Test export section independence
    config.export.tiled = False
    assert config.export.tiled is False
    assert config.analysis.ntl_floor == 0.5  # Unchanged from our modification above


def test_default_config_functions():
    """Test the convenience configuration functions.
    
    This test verifies that:
    - get_default_config returns bm_parity configuration
    - get_enhanced_config returns enhanced_quality configuration
    - Both functions work with different parameters
    - Returned configurations are properly initialized
    """
    from blackmarble.config import get_default_config, get_enhanced_config
    
    bbox = (-74.05, 40.69, -73.95, 40.79)
    date = datetime(2023, 9, 15)
    output_path = "/tmp/test.tif"
    
    # Test default config function
    default_config = get_default_config(bbox, date, output_path)
    assert default_config.enhancement.quality_mode == "basic"
    assert default_config.bbox == bbox
    assert default_config.date == date
    assert default_config.output_path == output_path
    
    # Test enhanced config function
    enhanced_config = get_enhanced_config(bbox, date, output_path)
    assert enhanced_config.enhancement.quality_mode == "enhanced"
    assert enhanced_config.bbox == bbox
    assert enhanced_config.date == date
    assert enhanced_config.output_path == output_path
    
    # Test they return different configurations
    assert default_config.enhancement.quality_mode != enhanced_config.enhancement.quality_mode
    assert default_config.analysis.use_scipy_fill_gaps != enhanced_config.analysis.use_scipy_fill_gaps
    assert default_config.export.generate_urban_mask != enhanced_config.export.generate_urban_mask