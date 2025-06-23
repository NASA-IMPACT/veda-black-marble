"""Integration tests for the main Black Marble pipeline.

This module tests the pipeline configuration, imports, and structure
without requiring any mocking or external data sources.
"""

import pytest
import tempfile
import os
from datetime import datetime
import inspect

# Import the main pipeline function and config
from blackmarble.pipeline_target import black_marble_pipeline
from blackmarble.target_config import PipelineConfig


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


def test_pipeline_config_compatibility(sample_bbox, sample_date, temp_output_path):
    """Test pipeline configuration compatibility and validation.
    
    This test verifies that:
    - Different configuration modes produce expected behavior
    - Legacy config conversion maintains compatibility
    - Configuration validation catches common errors
    """
    # Test for_bm_parity configuration
    basic_config = PipelineConfig.for_bm_parity(sample_bbox, sample_date, temp_output_path)
    assert basic_config.enhancement.quality_mode == "basic"
    assert basic_config.enhancement.colormap == "inferno"
    assert not basic_config.export.generate_urban_mask
    
    # Test for_enhanced_quality configuration
    enhanced_config = PipelineConfig.for_enhanced_quality(sample_bbox, sample_date, temp_output_path)
    assert enhanced_config.enhancement.quality_mode == "enhanced"
    assert enhanced_config.analysis.use_scipy_fill_gaps is True
    assert enhanced_config.export.generate_urban_mask is True
    
    # Test legacy config conversion
    legacy_dict = basic_config.to_legacy_config()
    assert isinstance(legacy_dict, dict)
    assert 'landsat' in legacy_dict
    assert 'viirs' in legacy_dict
    assert 'ndui' in legacy_dict
    
    # Verify legacy compatibility values
    assert legacy_dict['viirs']['floor'] == basic_config.analysis.ntl_floor
    assert legacy_dict['ndui']['floor'] == basic_config.analysis.ndui_floor
    
    # Test configuration validation
    valid_config = PipelineConfig.for_bm_parity(sample_bbox, sample_date, temp_output_path)
    valid_config.validate()  # Should not raise
    
    # Test invalid configurations
    invalid_configs = [
        # NTL floor >= ceiling
        lambda c: setattr(c.analysis, 'ntl_floor', 15.0) or setattr(c.analysis, 'ntl_ceiling', 10.0),
        # Invalid percentage
        lambda c: setattr(c, 'min_valid_pixel_percentage', 150.0),
        # Invalid resolution
        lambda c: setattr(c.preparation, 'final_resolution', -1.0),
        # Invalid enhancement factor
        lambda c: setattr(c.enhancement, 'road_major_factor', 0.5),
    ]
    
    for config_modifier in invalid_configs:
        config = PipelineConfig.for_bm_parity(sample_bbox, sample_date, temp_output_path)
        config_modifier(config)
        with pytest.raises(ValueError):
            config.validate()


def test_pipeline_import_structure():
    """Test that pipeline imports work correctly.
    
    This test verifies that:
    - Main pipeline function can be imported
    - Configuration classes can be imported
    - All functional API modules are accessible
    - No import errors occur
    """
    # Test main pipeline import
    from blackmarble.pipeline_target import black_marble_pipeline
    assert callable(black_marble_pipeline)
    
    # Test configuration import
    from blackmarble.target_config import PipelineConfig
    assert PipelineConfig is not None
    
    # Test functional API imports
    from blackmarble import acquire, prepare, analyze, enhance, export
    assert acquire is not None
    assert prepare is not None
    assert analyze is not None
    assert enhance is not None
    assert export is not None


def test_pipeline_function_signature():
    """Test that the main pipeline function has the expected signature.
    
    This test verifies that:
    - Function accepts required parameters
    - Function accepts optional configuration
    - Parameter types are correct
    """
    from blackmarble.pipeline_target import black_marble_pipeline
    
    # Get function signature
    sig = inspect.signature(black_marble_pipeline)
    params = list(sig.parameters.keys())
    
    # Verify required parameters exist
    assert 'bbox' in params
    assert 'date' in params  
    assert 'output_path' in params
    
    # Verify optional config parameter
    assert 'config' in params
    
    # Verify parameter defaults
    config_param = sig.parameters['config']
    assert config_param.default is None


def test_config_module_structure():
    """Test that configuration module has expected structure.
    
    This test verifies that:
    - All configuration classes exist
    - Configuration factory methods work
    - Configuration validation methods exist
    - Expected configuration sections are present
    """
    from blackmarble.target_config import (
        PipelineConfig, 
        AcquisitionConfig,
        PreparationConfig, 
        AnalysisConfig,
        EnhancementConfig,
        ExportConfig
    )
    
    # Test that all config classes exist
    assert PipelineConfig is not None
    assert AcquisitionConfig is not None
    assert PreparationConfig is not None
    assert AnalysisConfig is not None
    assert EnhancementConfig is not None
    assert ExportConfig is not None
    
    # Test factory methods exist
    assert hasattr(PipelineConfig, 'for_bm_parity')
    assert hasattr(PipelineConfig, 'for_enhanced_quality')
    assert hasattr(PipelineConfig, 'from_env')
    
    # Test validation method exists
    assert hasattr(PipelineConfig, 'validate')
    
    # Test legacy conversion exists
    assert hasattr(PipelineConfig, 'to_legacy_config')


def test_pipeline_config_sections():
    """Test that pipeline configuration sections are properly structured.
    
    This test verifies that:
    - All configuration sections exist and are initialized
    - Section-specific parameters are accessible
    - Default values are reasonable
    """
    bbox = (-74.05, 40.69, -73.95, 40.79)
    date = datetime(2023, 9, 15)
    output_path = "/tmp/test.tif"
    
    config = PipelineConfig.for_bm_parity(bbox, date, output_path)
    
    # Test acquisition section
    assert hasattr(config.acquisition, 'landsat_collection')
    assert hasattr(config.acquisition, 'viirs_product')
    assert hasattr(config.acquisition, 'road_types')
    assert isinstance(config.acquisition.road_types, list)
    
    # Test preparation section
    assert hasattr(config.preparation, 'landsat_scale_factor')
    assert hasattr(config.preparation, 'target_crs')
    assert hasattr(config.preparation, 'target_resolution')
    
    # Test analysis section
    assert hasattr(config.analysis, 'ntl_floor')
    assert hasattr(config.analysis, 'ntl_ceiling')
    assert hasattr(config.analysis, 'ndui_floor')
    
    # Test enhancement section
    assert hasattr(config.enhancement, 'quality_mode')
    assert hasattr(config.enhancement, 'colormap')
    assert hasattr(config.enhancement, 'road_major_factor')
    
    # Test export section
    assert hasattr(config.export, 'compress')
    assert hasattr(config.export, 'tiled')
    assert hasattr(config.export, 'overview_levels')


def test_pipeline_config_environment_variables():
    """Test that environment variable configuration works.
    
    This test verifies that:
    - Environment variables are properly parsed
    - Type conversion works correctly
    - Default values are used when variables are not set
    """
    bbox = (-74.05, 40.69, -73.95, 40.79)
    date = datetime(2023, 9, 15)
    output_path = "/tmp/test.tif"
    
    # Test with no environment variables (should use defaults)
    config = PipelineConfig.from_env(bbox, date, output_path)
    assert config.acquisition.composite_months == 12  # Default value
    assert config.analysis.ntl_floor == 0.1  # Default value
    
    # Test with environment variables set
    original_env = {}
    test_vars = {
        'BLACKMARBLE_COMPOSITE_MONTHS': '6',
        'BLACKMARBLE_NTL_FLOOR': '0.2',
        'BLACKMARBLE_USE_PARALLEL': 'true'
    }
    
    # Save original values and set test values
    for key, value in test_vars.items():
        if key in os.environ:
            original_env[key] = os.environ[key]
        os.environ[key] = value
    
    try:
        config = PipelineConfig.from_env(bbox, date, output_path)
        assert config.acquisition.composite_months == 6
        assert config.analysis.ntl_floor == 0.2
        assert config.use_parallel is True
    finally:
        # Cleanup: restore original environment
        for key in test_vars:
            if key in original_env:
                os.environ[key] = original_env[key]
            elif key in os.environ:
                del os.environ[key]


def test_pipeline_type_annotations():
    """Test that pipeline function has proper type annotations.
    
    This test verifies that:
    - Function parameters have correct type hints
    - Return type is properly annotated
    - Type system is consistent
    """
    from blackmarble.pipeline_target import black_marble_pipeline
    
    # Get type hints
    hints = black_marble_pipeline.__annotations__
    
    # Verify parameter type hints exist
    assert 'bbox' in hints
    assert 'date' in hints
    assert 'output_path' in hints
    assert 'config' in hints
    assert 'return' in hints
    
    # Verify return type is dict
    return_type = hints['return']
    assert 'dict' in str(return_type).lower()