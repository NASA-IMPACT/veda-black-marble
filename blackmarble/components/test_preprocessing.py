"""Test and example usage for preprocessing and temporal components."""

import numpy as np
from datetime import datetime, timedelta
from rasterio.transform import Affine, from_bounds

from blackmarble.components import (
    ProjectionHandler,
    ResolutionResampler,
    BoundsCropper,
    SpatialAligner,
    TemporalCompositor,
    SceneMerger,
    DateRangeSelector,
    CompositeMethod,
)


def test_projection_handler():
    """Test projection handler component."""
    print("Testing ProjectionHandler...")
    
    # Create sample data
    data = np.random.rand(100, 100)
    
    # Define source CRS and transform (UTM Zone 18N)
    source_crs = "EPSG:32618"
    source_transform = Affine(30, 0, 500000, 0, -30, 4000000)
    
    # Create handler to transform to WGS84
    handler = ProjectionHandler(target_crs="EPSG:4326", resampling="bilinear")
    
    # Process
    kwargs = {
        "source_crs": source_crs,
        "source_transform": source_transform
    }
    result = handler.process(data, **kwargs)
    
    print(f"  Original CRS: {source_crs}")
    print(f"  Target CRS: {handler.target_crs}")
    print(f"  Valid pixels: {result.statistics['valid_percentage']:.1f}%")
    print(f"  Original shape: {result.metadata['original_shape']}")
    print(f"  Transformed shape: {result.metadata['transformed_shape']}")
    print(f"  Bounds: {result.metadata['bounds']}")
    
    # The transformed data is stored in kwargs
    transformed_data = kwargs.get("transformed_data")
    if transformed_data is not None:
        print(f"  Transformed data shape: {transformed_data.shape}")
    else:
        print("  Note: Transformed data would be available in kwargs after processing")
    print()


def test_resolution_resampler():
    """Test resolution resampler component."""
    print("Testing ResolutionResampler...")
    
    # Create sample data at 30m resolution
    data = np.random.rand(100, 100)
    transform = Affine(30, 0, 500000, 0, -30, 4000000)
    
    # Resample to 10m resolution
    resampler = ResolutionResampler(target_resolution=(10, 10), method="bilinear")
    
    kwargs = {"transform": transform}
    result = resampler.process(data, **kwargs)
    
    print(f"  Original resolution: {result.metadata['original_resolution']}")
    print(f"  Target resolution: {result.metadata['target_resolution']}")
    print(f"  Scale factors: {result.metadata['scale_factors']}")
    print(f"  Original shape: {result.metadata['original_shape']}")
    print(f"  Resampled shape: {result.metadata['resampled_shape']}")
    
    resampled_data = kwargs.get("resampled_data")
    if resampled_data is not None:
        print(f"  Resampled data shape: {resampled_data.shape}")
    print()


def test_bounds_cropper():
    """Test bounds cropper component."""
    print("Testing BoundsCropper...")
    
    # Create sample data
    data = np.random.rand(1000, 1000)
    transform = from_bounds(-120, 30, -119, 31, 1000, 1000)
    
    # Define crop bounds (smaller area)
    crop_bounds = (-119.7, 30.3, -119.3, 30.7)
    
    cropper = BoundsCropper(bounds=crop_bounds, buffer=0.01, buffer_units="crs")
    
    kwargs = {"transform": transform}
    result = cropper.process(data, **kwargs)
    
    print(f"  Original bounds: {result.metadata['original_bounds']}")
    print(f"  Target bounds: {result.metadata['target_bounds']}")
    print(f"  Applied bounds (with buffer): {result.metadata['applied_bounds']}")
    print(f"  Pixel window: {result.metadata['pixel_window']}")
    print(f"  Original shape: {result.metadata['original_shape']}")
    print(f"  Cropped shape: {result.metadata['cropped_shape']}")
    print(f"  Pixels removed: {result.statistics['pixels_removed']}")
    
    cropped_data = kwargs.get("cropped_data")
    if cropped_data is not None:
        print(f"  Cropped data shape: {cropped_data.shape}")
    print()


def test_spatial_aligner():
    """Test spatial aligner component."""
    print("Testing SpatialAligner...")
    
    # Create multiple rasters with different sizes/transforms
    raster1 = np.random.rand(100, 100)
    transform1 = Affine(30, 0, 500000, 0, -30, 4000000)
    
    raster2 = np.random.rand(120, 110)
    transform2 = Affine(30, 0, 500100, 0, -30, 4000050)
    
    raster3 = np.random.rand(90, 95)
    transform3 = Affine(30, 0, 499900, 0, -30, 3999950)
    
    # Align all rasters to the first one
    aligner = SpatialAligner(reference_idx=0, method="bilinear")
    
    results = aligner.align_rasters(
        [raster1, raster2, raster3],
        [transform1, transform2, transform3]
    )
    
    print(f"  Number of rasters: {len(results)}")
    for i, result in enumerate(results):
        print(f"  Raster {i}:")
        print(f"    Shape: {result.data.shape}")
        print(f"    Is reference: {result.metadata.get('is_reference', False)}")
        if not result.metadata.get('is_reference'):
            print(f"    Original shape: {result.metadata.get('original_shape')}")
    
    # All aligned rasters should have the same shape
    shapes = [r.data.shape for r in results]
    print(f"  All shapes equal: {all(s == shapes[0] for s in shapes)}")
    print()


def test_temporal_compositor():
    """Test temporal compositor component."""
    print("Testing TemporalCompositor...")
    
    # Create time series data (10 time steps, 100x100 pixels)
    time_series = np.random.rand(10, 100, 100)
    
    # Add some NaN values to simulate clouds
    for t in range(10):
        mask = np.random.rand(100, 100) > 0.8
        time_series[t][mask] = np.nan
    
    # Create dates
    base_date = datetime(2023, 6, 1)
    dates = [base_date + timedelta(days=i*5) for i in range(10)]
    
    # Create compositor
    compositor = TemporalCompositor(
        method=CompositeMethod.MEDIAN,
        min_valid_observations=3,
        fill_nodata=True
    )
    
    kwargs = {"dates": dates}
    result = compositor.process(time_series, **kwargs)
    
    print(f"  Method: {result.metadata['method']}")
    print(f"  Time steps: {result.metadata['time_steps']}")
    print(f"  Mean observations per pixel: {result.metadata['mean_observations']:.1f}")
    print(f"  Max observations: {result.metadata['max_observations']}")
    print(f"  Valid pixels: {result.statistics['valid_percentage']:.1f}%")
    print(f"  Temporal range: {result.metadata['coverage']['temporal_range_days']} days")
    
    composite = kwargs.get("composite")
    if composite is not None:
        print(f"  Composite shape: {composite.shape}")
        print(f"  Composite has NaN: {np.any(np.isnan(composite))}")
    print()


def test_scene_merger():
    """Test scene merger component."""
    print("Testing SceneMerger...")
    
    # Create overlapping scenes
    scene1 = np.ones((100, 100)) * 10
    transform1 = from_bounds(0, 0, 1, 1, 100, 100)
    bounds1 = (0, 0, 1, 1)
    
    scene2 = np.ones((100, 100)) * 20
    transform2 = from_bounds(0.5, 0, 1.5, 1, 100, 100)
    bounds2 = (0.5, 0, 1.5, 1)
    
    scene3 = np.ones((100, 100)) * 30
    transform3 = from_bounds(0.25, 0.5, 1.25, 1.5, 100, 100)
    bounds3 = (0.25, 0.5, 1.25, 1.5)
    
    # Create merger
    merger = SceneMerger(blend_method="feather", feather_distance=10)
    
    merged, merged_transform, merged_bounds = merger.merge_scenes(
        [scene1, scene2, scene3],
        [transform1, transform2, transform3],
        [bounds1, bounds2, bounds3]
    )
    
    print(f"  Number of input scenes: 3")
    print(f"  Blend method: {merger.blend_method}")
    print(f"  Merged shape: {merged.shape}")
    print(f"  Merged bounds: {merged_bounds}")
    print(f"  Value range: [{np.min(merged):.1f}, {np.max(merged):.1f}]")
    print()


def test_date_range_selector():
    """Test date range selector component."""
    print("Testing DateRangeSelector...")
    
    # Create scene dates
    base_date = datetime(2023, 6, 15)
    scene_dates = [
        base_date - timedelta(days=30),
        base_date - timedelta(days=20),
        base_date - timedelta(days=10),
        base_date,
        base_date + timedelta(days=10),
        base_date + timedelta(days=20),
        base_date + timedelta(days=30),
    ]
    
    # Create metadata with cloud cover
    scene_metadata = [
        {"cloud_cover": 80},
        {"cloud_cover": 20},
        {"cloud_cover": 10},
        {"cloud_cover": 5},
        {"cloud_cover": 15},
        {"cloud_cover": 60},
        {"cloud_cover": 90},
    ]
    
    # Create selector
    selector = DateRangeSelector(
        target_date=base_date,
        max_days_from_target=15,
        cloud_threshold=30,
        min_scenes=2,
        max_scenes=3
    )
    
    selected_indices = selector.select_scenes(scene_dates, scene_metadata)
    
    print(f"  Total scenes: {len(scene_dates)}")
    print(f"  Selected scenes: {len(selected_indices)}")
    print(f"  Selected indices: {selected_indices}")
    
    for idx in selected_indices:
        days_diff = (scene_dates[idx] - base_date).days
        cloud = scene_metadata[idx]["cloud_cover"]
        print(f"    Scene {idx}: {days_diff:+d} days from target, {cloud}% clouds")
    print()


def main():
    """Run all component tests."""
    print("="*60)
    print("Black Marble Preprocessing Components Test Suite")
    print("="*60)
    print()
    
    test_projection_handler()
    test_resolution_resampler()
    test_bounds_cropper()
    test_spatial_aligner()
    test_temporal_compositor()
    test_scene_merger()
    test_date_range_selector()
    
    print("="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()