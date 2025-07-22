"""Test that NDVI for diagnostics and NDVI for NDUI are properly separated."""

import numpy as np
from unittest.mock import patch
from blackmarble.config import AnalysisConfig
from blackmarble.analyze import create_index_temporal_composite


def test_ndvi_separation():
    """Test that diagnostic NDVI and NDUI NDVI use different methods."""
    # Create test data
    ndvi_stack = [
        np.random.rand(100, 100).astype(np.float32) for _ in range(10)
    ]
    
    # Configure for median diagnostic but 90th percentile for NDUI
    config = AnalysisConfig(
        ndvi_composite_method="median",
        ndvi_percentile_for_ndui=90.0,
        min_valid_observations=1
    )
    
    # Create diagnostic NDVI (should use median)
    diagnostic_ndvi = create_index_temporal_composite(
        ndvi_stack,
        method=config.ndvi_composite_method,
        percentile=config.ndvi_composite_percentile,
        min_valid_observations=config.min_valid_observations
    )
    
    # Create NDUI NDVI (should use 90th percentile)
    ndui_ndvi = create_index_temporal_composite(
        ndvi_stack,
        method="percentile",
        percentile=config.ndvi_percentile_for_ndui,
        min_valid_observations=config.min_valid_observations
    )
    
    # They should be different
    assert not np.array_equal(diagnostic_ndvi, ndui_ndvi)
    
    # Verify the 90th percentile is higher than median (for most pixels)
    higher_count = np.sum(ndui_ndvi > diagnostic_ndvi)
    total_count = diagnostic_ndvi.size
    assert higher_count / total_count > 0.8  # Most pixels should be higher


def test_streaming_triggered_for_85():
    """Test that 85th percentile triggers streaming optimization."""
    # This test verifies the integration by checking that numpy.nanpercentile
    # is NOT called when we use exactly 85th percentile
    
    ndvi_stack = [
        np.ones((50, 50), dtype=np.float32) for _ in range(5)
    ]
    
    with patch('numpy.nanpercentile') as mock_percentile:
        # Call with 85th percentile
        result = create_index_temporal_composite(
            ndvi_stack,
            method="percentile",
            percentile=85.0
        )
        
        # numpy.nanpercentile should NOT have been called
        # because streaming implementation handles it
        mock_percentile.assert_not_called()
    
    # But for other percentiles it should be called
    with patch('numpy.nanpercentile', return_value=np.ones((50, 50))) as mock_percentile:
        result = create_index_temporal_composite(
            ndvi_stack,
            method="percentile", 
            percentile=50.0
        )
        
        # This time it should be called
        mock_percentile.assert_called()


def test_config_defaults():
    """Test that default config has sensible values."""
    config = AnalysisConfig()
    
    # Diagnostic NDVI defaults to median
    assert config.ndvi_composite_method == "median"
    
    # NDUI NDVI defaults to 85th percentile
    assert config.ndvi_percentile_for_ndui == 85.0
    
    # Index-first is enabled by default
    assert config.calculate_indices_first == True