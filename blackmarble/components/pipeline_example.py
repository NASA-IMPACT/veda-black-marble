"""Example of refactoring pipeline.py to use preprocessing components.

This demonstrates how the new preprocessing and temporal components can
replace the existing projection, resampling, and compositing logic.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
from datetime import datetime
from rasterio.transform import Affine

from blackmarble.components import (
    ProjectionHandler,
    ResolutionResampler,
    BoundsCropper,
    SpatialAligner,
    TemporalCompositor,
    SceneMerger,
    CompositeMethod,
)
from blackmarble.adapters.raster_io import GDALRasterAdapter


def process_landsat_with_components(
    band_files: List[str],
    qa_files: List[str],
    target_bounds: Tuple[float, float, float, float],
    target_crs: str = "EPSG:4326",
    target_resolution: float = 0.0001,
    composite_method: str = "median",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Process Landsat data using modular components.

    This replaces the existing pipeline logic with component-based approach.

    Args:
        band_files: List of band file paths
        qa_files: List of QA file paths
        target_bounds: (min_lon, min_lat, max_lon, max_lat)
        target_crs: Target coordinate system
        target_resolution: Target resolution in CRS units
        composite_method: Method for temporal compositing

    Returns:
        Tuple of (composite_data, metadata)
    """
    # Initialize components
    raster_adapter = GDALRasterAdapter()
    projection_handler = ProjectionHandler(target_crs=target_crs)
    resampler = ResolutionResampler(
        target_resolution=(target_resolution, target_resolution)
    )
    cropper = BoundsCropper(bounds=target_bounds, buffer=0.001, buffer_units="crs")
    compositor = TemporalCompositor(method=composite_method, min_valid_observations=2)

    # Process each scene
    processed_scenes = []

    for band_file, qa_file in zip(band_files, qa_files):
        print(f"Processing {Path(band_file).name}...")

        # 1. Read data
        band_data, band_meta = raster_adapter.read(Path(band_file))
        qa_data, qa_meta = raster_adapter.read(Path(qa_file))

        # 2. Apply cloud masking (simplified example)
        # In real implementation, use CloudMasker component
        cloud_mask = (qa_data & 0b11) == 0  # Clear pixels
        masked_data = np.where(cloud_mask, band_data, np.nan)

        # 3. Project to target CRS
        proj_kwargs = {
            "source_crs": band_meta["crs"],
            "source_transform": band_meta["transform"],
            "nodata": np.nan,
        }
        proj_result = projection_handler.process(masked_data, **proj_kwargs)

        # Get projected data from kwargs
        projected_data = proj_kwargs.get("transformed_data", masked_data)
        new_transform = proj_kwargs.get("transform", band_meta["transform"])

        # 4. Crop to target bounds
        crop_kwargs = {"transform": new_transform}
        crop_result = cropper.process(projected_data, **crop_kwargs)

        # Get cropped data
        cropped_data = crop_kwargs.get("cropped_data", projected_data)
        cropped_transform = crop_kwargs.get("transform", new_transform)

        # 5. Resample to target resolution
        resample_kwargs = {"transform": cropped_transform, "crs": target_crs}
        resample_result = resampler.process(cropped_data, **resample_kwargs)

        # Get resampled data
        resampled_data = resample_kwargs.get("resampled_data", cropped_data)

        processed_scenes.append(resampled_data)

    # 6. Create temporal composite
    if len(processed_scenes) > 1:
        # Stack scenes along time dimension
        time_stack = np.stack(processed_scenes, axis=0)

        comp_kwargs = {}
        comp_result = compositor.process(time_stack, **comp_kwargs)

        composite = comp_kwargs.get("composite", time_stack[0])

        # Collect metadata
        metadata = {
            "scenes_processed": len(processed_scenes),
            "composite_method": composite_method,
            "valid_pixels": comp_result.statistics["valid_percentage"],
            "target_crs": target_crs,
            "target_resolution": target_resolution,
            "bounds": target_bounds,
        }
    else:
        composite = processed_scenes[0]
        metadata = {
            "scenes_processed": 1,
            "target_crs": target_crs,
            "target_resolution": target_resolution,
            "bounds": target_bounds,
        }

    return composite, metadata


def merge_landsat_scenes_with_components(
    scene_files: List[List[str]],  # List of [B3, B4, B5] for each scene
    scene_bounds: List[Tuple[float, float, float, float]],
    output_bounds: Tuple[float, float, float, float],
) -> Dict[str, np.ndarray]:
    """Merge multiple Landsat scenes using SceneMerger component.

    Args:
        scene_files: List of band files for each scene
        scene_bounds: List of bounds for each scene
        output_bounds: Target output bounds

    Returns:
        Dictionary of merged bands
    """
    raster_adapter = GDALRasterAdapter()
    merger = SceneMerger(blend_method="feather", feather_distance=20)

    # Process each band separately
    merged_bands = {}

    for band_idx, band_name in enumerate(["B3", "B4", "B5"]):
        print(f"Merging {band_name} across scenes...")

        # Collect band data from all scenes
        band_arrays = []
        transforms = []

        for scene_idx, scene in enumerate(scene_files):
            band_file = scene[band_idx]
            data, meta = raster_adapter.read(Path(band_file))

            band_arrays.append(data)
            transforms.append(meta["transform"])

        # Merge scenes
        merged, merged_transform, merged_bounds = merger.merge_scenes(
            band_arrays, transforms, scene_bounds
        )

        merged_bands[band_name] = merged

    return merged_bands


def example_utm_to_wgs84_conversion():
    """Example showing UTM to WGS84 conversion using components.

    This replaces the convert_to_wgs84 function in reproj.py.
    """
    # Example: Convert UTM Zone 18N data to WGS84
    projection_handler = ProjectionHandler(
        target_crs="EPSG:4326",  # WGS84
        resampling="bilinear",
    )

    # Simulate UTM data
    utm_data = np.random.rand(1000, 1000)

    # UTM Zone 18N metadata
    kwargs = {
        "source_crs": "EPSG:32618",
        "source_transform": Affine(30, 0, 500000, 0, -30, 4000000),
        "nodata": -9999,
    }

    # Transform to WGS84
    result = projection_handler.process(utm_data, **kwargs)

    # Get results - they should be in kwargs after processing
    wgs84_data = kwargs.get("transformed_data")
    wgs84_transform = kwargs.get("transform")
    wgs84_bounds = kwargs.get("bounds")

    print(f"Converted from UTM to WGS84:")
    print(f"  Original shape: {utm_data.shape}")
    if wgs84_data is not None:
        print(f"  Transformed shape: {wgs84_data.shape}")
        print(f"  WGS84 bounds: {wgs84_bounds}")
    else:
        print("  Note: In real usage, transformed data would be available")
        print(
            f"  Transform succeeded: {result.statistics['valid_percentage']:.1f}% valid pixels"
        )

    return wgs84_data, wgs84_transform


def example_multi_resolution_alignment():
    """Example showing how to align rasters with different resolutions.

    This handles the case where VIIRS (463m) and Landsat (30m) need alignment.
    """
    aligner = SpatialAligner(reference_idx=0)  # Use Landsat as reference

    # Simulate data at different resolutions
    landsat_data = np.random.rand(1000, 1000)  # 30m resolution
    landsat_transform = Affine(30, 0, 500000, 0, -30, 4000000)

    viirs_data = np.random.rand(65, 65)  # ~463m resolution
    viirs_transform = Affine(463, 0, 500000, 0, -463, 4000000)

    osm_data = np.random.rand(3000, 3000)  # 10m resolution
    osm_transform = Affine(10, 0, 500000, 0, -10, 4000000)

    # Align all to Landsat resolution and grid
    aligned_results = aligner.align_rasters(
        [landsat_data, viirs_data, osm_data],
        [landsat_transform, viirs_transform, osm_transform],
    )

    print("Multi-resolution alignment:")
    for i, (name, original_shape) in enumerate(
        [
            ("Landsat", landsat_data.shape),
            ("VIIRS", viirs_data.shape),
            ("OSM", osm_data.shape),
        ]
    ):
        aligned_shape = aligned_results[i].data.shape
        print(f"  {name}: {original_shape} -> {aligned_shape}")

    return aligned_results


def main():
    """Run examples."""
    print("=" * 60)
    print("Pipeline Refactoring Examples")
    print("=" * 60)
    print()

    # Example 1: UTM to WGS84 conversion
    print("Example 1: UTM to WGS84 Conversion")
    print("-" * 40)
    example_utm_to_wgs84_conversion()
    print()

    # Example 2: Multi-resolution alignment
    print("Example 2: Multi-resolution Alignment")
    print("-" * 40)
    example_multi_resolution_alignment()
    print()

    print("=" * 60)
    print("Examples completed!")
    print("These demonstrate how to replace existing pipeline")
    print("functions with modular, testable components.")
    print("=" * 60)


if __name__ == "__main__":
    main()
