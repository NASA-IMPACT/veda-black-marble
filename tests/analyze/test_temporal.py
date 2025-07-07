"""Test temporal analysis functions."""

import numpy as np
import pytest

from blackmarble.analyze.temporal import create_landsat_median_composite


def test_create_landsat_median_composite():
    """Test median composite creation."""
    # Create test data with 3 dates
    date1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    date2 = np.array([[2, 3, 4], [5, 6, 7]], dtype=np.float32)
    date3 = np.array([[3, 4, 5], [6, 7, 8]], dtype=np.float32)

    # Test with no scaling (scale_factor=1, add_offset=0)
    composite = create_landsat_median_composite(
        [date1, date2, date3], scale_factor=1.0, add_offset=0.0
    )

    # Should get median values
    expected = np.array([[2, 3, 4], [5, 6, 7]], dtype=np.float32)
    np.testing.assert_array_equal(composite, expected)


def test_create_landsat_median_composite_with_nan():
    """Test median composite with NaN values."""
    # Create test data with NaN (masked pixels)
    date1 = np.array([[1, np.nan, 3], [4, 5, np.nan]], dtype=np.float32)
    date2 = np.array([[2, 3, np.nan], [np.nan, 6, 7]], dtype=np.float32)
    date3 = np.array([[3, 4, 5], [6, np.nan, 8]], dtype=np.float32)

    # Test with no scaling
    composite = create_landsat_median_composite(
        [date1, date2, date3], scale_factor=1.0, add_offset=0.0
    )

    # Should handle NaN correctly
    assert composite[0, 0] == 2  # median of [1, 2, 3]
    assert composite[0, 1] == 3.5  # median of [nan, 3, 4]
    assert composite[0, 2] == 4  # median of [3, nan, 5]
    assert composite[1, 0] == 5  # median of [4, nan, 6]
    assert composite[1, 1] == 5.5  # median of [5, 6, nan]
    assert composite[1, 2] == 7.5  # median of [nan, 7, 8]


def test_create_landsat_median_composite_different_shapes():
    """Test median composite with different input shapes."""
    # Create test data with different shapes
    date1 = np.ones((10, 10))
    date2 = np.ones((12, 8)) * 2
    date3 = np.ones((9, 11)) * 3

    # Test with no scaling
    composite = create_landsat_median_composite(
        [date1, date2, date3], scale_factor=1.0, add_offset=0.0
    )

    # Should crop to minimum common size
    assert composite.shape == (9, 8)
    assert composite[0, 0] == 2  # median of [1, 2, 3]


def test_create_landsat_median_composite_empty_list():
    """Test error handling for empty input."""
    with pytest.raises(ValueError, match="No data provided"):
        create_landsat_median_composite([])
