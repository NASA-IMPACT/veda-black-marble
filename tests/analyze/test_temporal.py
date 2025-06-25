"""Test critical temporal compositing functions for bm.py parity.

This module tests only the essential temporal functionality needed for bm.py parity.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from blackmarble.analyze.temporal import (
    create_temporal_composite,
    compute_temporal_statistics,
)


def test_create_temporal_composite():
    """Test temporal composite creation - CRITICAL for multi-date processing."""
    # Create test data stack (3 dates, 3x3 pixels)
    data_stack = np.array([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],      # Date 1
        [[2, 3, 4], [5, 6, 7], [8, 9, 10]],    # Date 2
        [[3, 4, 5], [6, 7, 8], [9, 10, 11]]    # Date 3
    ])
    
    # Test median composite (default)
    median_composite, count, metadata = create_temporal_composite(
        data_stack,
        method='median'
    )
    
    # Verify median calculation
    assert median_composite.shape == (3, 3)
    assert median_composite[0, 0] == 2  # Median of [1, 2, 3]
    assert median_composite[1, 1] == 6  # Median of [5, 6, 7]
    
    # Test mean composite
    mean_composite, _, _ = create_temporal_composite(
        data_stack,
        method='mean'
    )
    
    assert mean_composite[0, 0] == 2  # Mean of [1, 2, 3]


def test_percentile_composite():
    """Test percentile composite via create_temporal_composite."""
    # Create test data
    data_stack = np.array([
        [[1, 5, 9], [2, 6, 10], [3, 7, 11]],
        [[2, 6, 10], [3, 7, 11], [4, 8, 12]],
        [[3, 7, 11], [4, 8, 12], [5, 9, 13]],
        [[4, 8, 12], [5, 9, 13], [6, 10, 14]],
        [[5, 9, 13], [6, 10, 14], [7, 11, 15]]
    ])
    
    # Test percentile via create_temporal_composite
    p50, _, _ = create_temporal_composite(data_stack, method='median')
    assert p50[0, 0] == 3  # Median of [1,2,3,4,5]


def test_compute_temporal_statistics():
    """Test temporal statistics calculation - needed for quality assessment."""
    # Create test data
    data_stack = np.array([
        [[1, 2, 3], [4, 5, 6]],
        [[2, 3, 4], [5, 6, 7]],
        [[3, 4, 5], [6, 7, 8]]
    ])
    
    # Calculate statistics
    stats = compute_temporal_statistics(data_stack)
    
    # Verify statistics
    assert 'mean' in stats
    assert 'std' in stats
    assert 'min' in stats
    assert 'max' in stats
    assert 'count' in stats
    
    # Check values
    assert stats['mean'][0, 0] == 2  # Mean of [1, 2, 3]
    assert stats['min'][0, 0] == 1
    assert stats['max'][0, 0] == 3
    assert stats['count'][0, 0] == 3  # 3 valid observations