"""
Common test utilities for the Black Marble pipeline.

Provides assertion helpers with better error messages and data generation utilities.
"""

import numpy as np
import pytest
from typing import Dict, Any, Optional


def assert_array_close(actual: np.ndarray, expected: np.ndarray, 
                      rtol: float = 1e-6, atol: float = 1e-8, 
                      msg: str = "") -> None:
    """
    Assert arrays are close with detailed error reporting.
    
    Args:
        actual: Actual array
        expected: Expected array
        rtol: Relative tolerance
        atol: Absolute tolerance
        msg: Additional message
    
    Raises:
        AssertionError: With detailed information about mismatches
    """
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{msg}\nShape mismatch: {actual.shape} != {expected.shape}"
        )
    
    if not np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True):
        # Find worst mismatch
        with np.errstate(invalid='ignore'):
            diff = np.abs(actual - expected)
            rel_diff = diff / (np.abs(expected) + atol)
        
        # Handle NaN comparisons
        nan_mismatch = np.isnan(actual) != np.isnan(expected)
        if np.any(nan_mismatch):
            nan_loc = np.where(nan_mismatch)
            raise AssertionError(
                f"{msg}\nNaN mismatch at {nan_loc[0][0]}, {nan_loc[1][0]}\n"
                f"Actual: {actual[nan_loc][0]}, Expected: {expected[nan_loc][0]}"
            )
        
        # Find max difference
        max_diff_idx = np.unravel_index(np.nanargmax(diff), diff.shape)
        max_rel_idx = np.unravel_index(np.nanargmax(rel_diff), rel_diff.shape)
        
        raise AssertionError(
            f"{msg}\n"
            f"Max absolute difference: {diff[max_diff_idx]:.2e} at {max_diff_idx}\n"
            f"  Actual: {actual[max_diff_idx]}\n"
            f"  Expected: {expected[max_diff_idx]}\n"
            f"Max relative difference: {rel_diff[max_rel_idx]:.2e} at {max_rel_idx}\n"
            f"  Actual: {actual[max_rel_idx]}\n"
            f"  Expected: {expected[max_rel_idx]}"
        )


def assert_stats_close(actual_stats: Dict[str, Any], 
                      expected_stats: Dict[str, Any],
                      rtol: float = 0.01) -> None:
    """Assert statistics dictionaries are close."""
    for key in expected_stats:
        assert key in actual_stats, f"Missing statistic: {key}"
        
        actual = actual_stats[key]
        expected = expected_stats[key]
        
        if isinstance(expected, (int, float)):
            if isinstance(expected, float):
                assert np.isclose(actual, expected, rtol=rtol), \
                    f"{key}: {actual} != {expected} (tolerance={rtol})"
            else:
                assert actual == expected, f"{key}: {actual} != {expected}"
        elif isinstance(expected, dict):
            assert_stats_close(actual, expected, rtol)
        else:
            assert actual == expected, f"{key}: {actual} != {expected}"


def assert_valid_range(data: np.ndarray, min_val: float, max_val: float,
                      name: str = "data") -> None:
    """Assert all finite values are within range."""
    finite_mask = np.isfinite(data)
    if not np.any(finite_mask):
        return  # No finite values to check
    
    finite_data = data[finite_mask]
    actual_min = np.min(finite_data)
    actual_max = np.max(finite_data)
    
    assert actual_min >= min_val, \
        f"{name} minimum {actual_min} below {min_val}"
    assert actual_max <= max_val, \
        f"{name} maximum {actual_max} above {max_val}"


def assert_close(actual: float, expected: float, rtol: float = 1e-6, 
                atol: float = 1e-8, msg: str = "") -> None:
    """Assert scalar values are close."""
    if not np.isclose(actual, expected, rtol=rtol, atol=atol):
        diff = abs(actual - expected)
        rel_diff = diff / (abs(expected) + atol)
        raise AssertionError(
            f"{msg}\n"
            f"Values not close:\n"
            f"  Actual: {actual}\n"
            f"  Expected: {expected}\n"
            f"  Absolute diff: {diff:.2e}\n"
            f"  Relative diff: {rel_diff:.2e}"
        )