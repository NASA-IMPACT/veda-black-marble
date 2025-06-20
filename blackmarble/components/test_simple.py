"""Simple tests for preprocessing components without rasterio dependencies."""

import numpy as np
from datetime import datetime, timedelta

from blackmarble.components import (
    ResolutionResampler,
    BoundsCropper,
    TemporalCompositor,
    DateRangeSelector,
    CompositeMethod,
)


def test_bounds_cropper_simple():
    """Test bounds cropper with simple data."""
    print("Testing BoundsCropper (simple)...")
    
    # Create simple data and transform
    data = np.arange(100).reshape(10, 10)
    
    # Simple transform: origin at (0, 10), 1 unit per pixel
    from rasterio.transform import Affine
    transform = Affine(1, 0, 0, 0, -1, 10)
    
    # Crop to center region
    cropper = BoundsCropper(bounds=(2, 5, 7, 8), buffer=0)
    
    kwargs = {"transform": transform}
    result = cropper.process(data, **kwargs)
    
    print(f"  Original shape: {data.shape}")
    print(f"  Valid pixels: {result.statistics['valid_percentage']:.1f}%")
    
    cropped_data = kwargs.get("cropped_data")
    if cropped_data is not None:
        print(f"  Cropped shape: {cropped_data.shape}")
        print(f"  Cropped data:\n{cropped_data}")
    print()


def test_temporal_simple():
    """Test temporal compositing with simple data."""
    print("Testing TemporalCompositor (simple)...")
    
    # Create simple time series
    time_series = np.array([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[2, 3, 4], [5, 6, 7], [8, 9, 10]],
        [[3, 4, 5], [6, 7, 8], [9, 10, 11]],
    ], dtype=float)
    
    # Add some NaN values
    time_series[0, 0, 0] = np.nan
    time_series[1, 1, 1] = np.nan
    time_series[2, 2, 2] = np.nan
    
    compositor = TemporalCompositor(method=CompositeMethod.MEDIAN)
    
    kwargs = {}
    result = compositor.process(time_series, **kwargs)
    
    print(f"  Time steps: {time_series.shape[0]}")
    print(f"  Valid pixels: {result.statistics['valid_percentage']:.1f}%")
    
    composite = kwargs.get("composite")
    if composite is not None:
        print(f"  Composite shape: {composite.shape}")
        print(f"  Composite:\n{composite}")
    print()


def test_date_selector_simple():
    """Test date range selector."""
    print("Testing DateRangeSelector (simple)...")
    
    base_date = datetime(2023, 6, 15)
    scene_dates = [
        base_date - timedelta(days=10),
        base_date - timedelta(days=5),
        base_date,
        base_date + timedelta(days=5),
        base_date + timedelta(days=10),
    ]
    
    selector = DateRangeSelector(
        target_date=base_date,
        max_days_from_target=7
    )
    
    selected = selector.select_scenes(scene_dates)
    
    print(f"  Total scenes: {len(scene_dates)}")
    print(f"  Selected indices: {selected}")
    
    for idx in selected:
        days_diff = (scene_dates[idx] - base_date).days
        print(f"    Scene {idx}: {days_diff:+d} days from target")
    print()


def main():
    """Run simple tests."""
    print("="*50)
    print("Simple Component Tests")
    print("="*50)
    print()
    
    test_bounds_cropper_simple()
    test_temporal_simple()
    test_date_selector_simple()
    
    print("="*50)
    print("Simple tests completed!")
    print("="*50)


if __name__ == "__main__":
    main()