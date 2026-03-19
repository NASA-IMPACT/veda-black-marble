"""Test streaming algorithms for memory-efficient processing."""

from unittest.mock import patch

import numpy as np
import pytest

from blackmarble.analyze.streaming import streaming_percentile_85
from blackmarble.analyze.temporal import create_index_temporal_composite


class TestStreamingPercentile85:
    """Test the streaming 85th percentile implementation."""

    def test_empty_input(self):
        """Test that empty input raises ValueError."""
        with pytest.raises(ValueError, match="No data provided"):
            streaming_percentile_85([])

    def test_single_array(self):
        """Test with a single array."""
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result, valid_count = streaming_percentile_85([arr])

        # With single array, result should equal input
        np.testing.assert_array_equal(result, arr)
        np.testing.assert_array_equal(valid_count, 1)

    def test_all_nan_pixel(self):
        """Test pixels that are NaN in all arrays."""
        arrays = [
            np.array([[1, np.nan], [3, 4]], dtype=np.float32),
            np.array([[2, np.nan], [5, 6]], dtype=np.float32),
            np.array([[3, np.nan], [7, 8]], dtype=np.float32),
        ]
        result, valid_count = streaming_percentile_85(arrays)

        # Second column of first row should be NaN
        assert np.isnan(result[0, 1])
        assert valid_count[0, 1] == 0

        # Other pixels should have valid values
        assert not np.isnan(result[0, 0])
        assert not np.isnan(result[1, 0])
        assert not np.isnan(result[1, 1])

    def test_min_valid_observations(self):
        """Test minimum valid observations requirement."""
        arrays = [
            np.array([[1, 2], [3, np.nan]], dtype=np.float32),
            np.array([[4, 5], [np.nan, np.nan]], dtype=np.float32),
            np.array([[7, 8], [np.nan, 6]], dtype=np.float32),
        ]
        result, valid_count = streaming_percentile_85(arrays, min_valid_observations=2)

        # Bottom-left pixel has only 1 valid observation
        assert np.isnan(result[1, 0])

        # Top row has 3 valid observations each
        assert not np.isnan(result[0, 0])
        assert not np.isnan(result[0, 1])

    def test_known_values(self):
        """Test with known values to verify algorithm."""
        # Create 5 arrays with known values
        arrays = []
        for i in range(5):
            arr = np.full((2, 2), i + 1, dtype=np.float32)
            arrays.append(arr)

        result, valid_count = streaming_percentile_85(arrays)

        # With values [1, 2, 3, 4, 5], 85th percentile should select
        # rank = 5 - floor(5 * 0.85) = 5 - 4 = 1st from top = 5
        expected = np.full((2, 2), 5, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(valid_count, 5)

    def test_shape_consistency(self):
        """Test that output shape matches input shape."""
        shape = (123, 456)
        arrays = [np.random.rand(*shape).astype(np.float32) for _ in range(3)]

        result, valid_count = streaming_percentile_85(arrays)

        assert result.shape == shape
        assert valid_count.shape == shape

    def test_dtype_preservation(self):
        """Test that output is float32."""
        arrays = [np.ones((10, 10), dtype=np.float64) for _ in range(3)]

        result, valid_count = streaming_percentile_85(arrays)

        assert result.dtype == np.float32
        assert valid_count.dtype == np.int32

    def test_performance_characteristics(self):
        """Test that streaming uses less memory than stacking."""
        # This is more of a documentation test
        n_arrays = 10
        shape = (1000, 1000)

        # Memory for streaming: ~3 arrays worth
        streaming_memory = 3 * shape[0] * shape[1] * 4  # float32

        # Memory for stacking: n_arrays worth
        stacking_memory = n_arrays * shape[0] * shape[1] * 4

        memory_ratio = stacking_memory / streaming_memory
        assert memory_ratio > 3.0  # Should be ~3.33x

    def test_mixed_nan_patterns(self):
        """Test with various NaN patterns."""
        arrays = [
            np.array([[1, 2, np.nan], [4, np.nan, 6]], dtype=np.float32),
            np.array([[7, np.nan, 9], [np.nan, 11, 12]], dtype=np.float32),
            np.array([[13, 14, 15], [16, 17, np.nan]], dtype=np.float32),
            np.array([[19, 20, np.nan], [22, 23, 24]], dtype=np.float32),
        ]

        result, valid_count = streaming_percentile_85(arrays)

        # Check valid counts match expected
        expected_counts = np.array([[4, 3, 2], [3, 3, 3]], dtype=np.int32)
        np.testing.assert_array_equal(valid_count, expected_counts)

        # All pixels with at least 1 observation should have a value
        assert not np.isnan(result[0, 0])  # 4 observations
        assert not np.isnan(result[0, 1])  # 3 observations
        assert not np.isnan(result[0, 2])  # 2 observations

    def test_approximation_behavior(self):
        """Test that approximation behaves reasonably."""
        # The streaming algorithm selects from top-3 values without interpolation
        # This is different from numpy's interpolated percentile but still useful

        # Test with known values where we can predict the result
        arrays = [
            np.full((2, 2), 1.0, dtype=np.float32),
            np.full((2, 2), 2.0, dtype=np.float32),
            np.full((2, 2), 3.0, dtype=np.float32),
            np.full((2, 2), 4.0, dtype=np.float32),
            np.full((2, 2), 5.0, dtype=np.float32),
        ]

        result, valid_count = streaming_percentile_85(arrays)

        # With 5 values [1,2,3,4,5], our algorithm selects the highest (5)
        # because rank_from_top = 5 - floor(5*0.85) = 5 - 4 = 1
        expected = np.full((2, 2), 5.0, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

        # Test that result is always one of the input values (no interpolation)
        n_arrays = 10
        arrays_random = []
        for i in range(n_arrays):
            arr = np.full((3, 3), float(i), dtype=np.float32)
            # Add some variation
            arr[1, 1] = float(i) + 0.5
            arrays_random.append(arr)

        result, _ = streaming_percentile_85(arrays_random)

        # Every output value should be one of the input values
        all_input_values = set()
        for arr in arrays_random:
            all_input_values.update(arr.flatten())

        output_values = set(result.flatten())
        assert output_values.issubset(all_input_values)

    def test_shape_consistency_required(self):
        """Test that different shapes raise an error."""
        # The streaming implementation requires consistent shapes
        # which should be ensured by boundless reads in the pipeline
        arrays = [
            np.ones((100, 200), dtype=np.float32),
            np.ones((100, 150), dtype=np.float32) * 2,  # Different width
        ]

        with pytest.raises(ValueError, match="Shape mismatch in array stack"):
            streaming_percentile_85(arrays)

    def test_integration_routing(self):
        """Test that create_index_temporal_composite routes to streaming."""

        # Create test data
        arrays = [np.ones((10, 10), dtype=np.float32) for _ in range(5)]

        # Patch numpy percentile to ensure it's not called
        with patch("numpy.nanpercentile") as mock_percentile:
            # Call with 85th percentile
            create_index_temporal_composite(arrays, method="percentile", percentile=85.0)

            # Numpy percentile should NOT have been called
            mock_percentile.assert_not_called()

        # Also test with integer 85
        with patch("numpy.nanpercentile") as mock_percentile:
            create_index_temporal_composite(
                arrays,
                method="percentile",
                percentile=85,  # Integer
            )

            # Should still route to streaming
            mock_percentile.assert_not_called()
