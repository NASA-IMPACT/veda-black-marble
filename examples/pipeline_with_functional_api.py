#!/usr/bin/env python3
"""
Example of using the functional API from blackmarble.processing.

This demonstrates the benefits of the functional approach over the component-based
system. The functional API is simpler, more composable, and returns what you expect!

Key Benefits of Functional API:
1. Direct return values - functions return data/masks, not wrapper objects
2. Simple composition - easily chain functions together
3. No complex inheritance hierarchies
4. Easier to test and debug
5. More Pythonic and intuitive
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from datetime import datetime

# Add parent directory to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from blackmarble.pipeline import run_pipeline
from blackmarble.adapters.s3 import S3Adapter
from blackmarble.adapters.raster_io import GDALRasterAdapter
from blackmarble.data_sources.landsat import LandsatDataSource
from blackmarble.data_sources.viirs import VIIRSDataSource
from blackmarble.config import ProcessingConfig

# Import the functional API
from blackmarble.processing import (
    # Quality masking functions
    create_cloud_mask,
    create_quality_mask,
    create_aerosol_mask,
    create_saturation_mask,
    create_viirs_quality_mask,
    combine_masks,
    dilate_mask,
    # Enhancement functions
    enhance_contrast,
    enhance_roads,
    enhance_urban_features,
    # Statistics functions
    compute_statistics,
    # Composite functions
    create_temporal_composite,
    # Scene selection
    select_by_date_range,
)


class FunctionalLandsatSource(LandsatDataSource):
    """Extended Landsat data source using the functional API.

    This demonstrates how much simpler the code becomes when using
    functions that return actual data instead of MaskResult objects.
    """

    def __init__(
        self,
        storage_adapter=None,
        raster_adapter=None,
        quality_strategy="conservative",
        dilate_clouds=0,
    ):
        super().__init__(storage_adapter, raster_adapter)
        self.quality_strategy = quality_strategy
        self.dilate_clouds = dilate_clouds

    def fetch_with_quality(
        self,
        bbox,
        start_date,
        end_date,
        bands=None,
        composite_months=1,
        return_quality_report=False,
    ):
        """Fetch Landsat data with quality control applied using functional API.

        Compare this to the component version - it's much cleaner!
        """
        # Fetch raw data using parent method
        raw_data = self.fetch(bbox, start_date, end_date, bands, composite_months)

        # Ensure we have the QA bands
        qa_bands = ["QA_PIXEL", "QA_AEROSOL", "QA_RADSAT"]
        missing_qa = [b for b in qa_bands if b not in raw_data]
        if missing_qa:
            raise ValueError(f"Missing required QA bands: {missing_qa}")

        # Create quality mask using functional API
        # Notice how simple this is compared to the component approach!
        quality_mask = create_quality_mask(
            qa_bands={
                "qa_pixel": raw_data["QA_PIXEL"],
                "qa_aerosol": raw_data["QA_AEROSOL"],
                "qa_radsat": raw_data["QA_RADSAT"],
            },
            combine_method="and",
            cloud_strategy=self.quality_strategy,
            cloud_dilate=self.dilate_clouds,
            aerosol_threshold="standard",
            include_saturation=True,
            saturation_bands=[3, 4, 5],
        )

        # Apply mask to data bands
        processed_data = {}
        for band_name, band_data in raw_data.items():
            if band_name in qa_bands:
                # Keep QA bands as-is
                processed_data[band_name] = band_data
            else:
                # Apply quality mask - simple boolean indexing!
                processed_data[band_name] = np.where(quality_mask, band_data, np.nan)

        if return_quality_report:
            # Generate quality report using functional API
            report = {
                "quality_summary": compute_statistics(
                    quality_mask.astype(float), percentiles=[25, 50, 75]
                ),
                "processing_config": {
                    "cloud_strategy": self.quality_strategy,
                    "cloud_dilate": self.dilate_clouds,
                    "combine_method": "and",
                },
            }
            return processed_data, report

        return processed_data


class FunctionalVIIRSSource(VIIRSDataSource):
    """Extended VIIRS data source using the functional API."""

    def __init__(self, api_adapter=None, raster_adapter=None, earthdata_token=None):
        super().__init__(api_adapter, raster_adapter, earthdata_token)

    def fetch_with_quality(
        self, bbox, start_date, end_date, layers=None, return_quality_report=False
    ):
        """Fetch VIIRS data with quality control using functional API."""
        # Fetch raw data
        raw_data = self.fetch(bbox, start_date, end_date, layers)

        # Ensure we have quality flag
        if "Mandatory_Quality_Flag" not in raw_data:
            raise ValueError("Missing Mandatory_Quality_Flag layer")

        # Create quality mask - one simple function call!
        quality_mask = create_viirs_quality_mask(
            raw_data["Mandatory_Quality_Flag"],
            valid_flags=[0, 2],  # High and moderate quality
        )

        # Apply mask to NTL data
        processed_data = {}
        for layer_name, layer_data in raw_data.items():
            if layer_name == "Mandatory_Quality_Flag":
                processed_data[layer_name] = layer_data
            elif layer_name in ["NTL", "DNB_At_Sensor_Radiance"]:
                # Apply quality mask
                processed_data[layer_name] = np.where(quality_mask, layer_data, np.nan)
            else:
                # Keep other layers as-is
                processed_data[layer_name] = layer_data

        if return_quality_report:
            report = {
                "quality_summary": compute_statistics(quality_mask.astype(float)),
                "viirs_quality_flags": "0=high, 2=moderate quality",
            }
            return processed_data, report

        return processed_data


def demonstrate_functional_vs_component():
    """Side-by-side comparison of functional API vs component approach."""

    print("=== FUNCTIONAL API vs COMPONENT APPROACH ===\n")

    # Create sample data
    np.random.seed(42)
    shape = (100, 100)
    qa_pixel = np.random.choice(
        [21824, 21952, 22280, 54596, 54852, 55052],
        size=shape,
        p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05],
    )

    print("1. CLOUD MASKING COMPARISON")
    print("-" * 50)

    # Component approach (OLD WAY)
    print("\nComponent Approach (OLD):")
    print("```python")
    print("# Must import and instantiate component class")
    print("from blackmarble.components.quality import CloudMasker")
    print("masker = CloudMasker(strategy='conservative')")
    print("result = masker.process(qa_pixel)")
    print("# Returns MaskResult object - must extract mask!")
    print("mask = result.mask")
    print("stats = result.statistics  # Buried in object")
    print("```")

    # Functional approach (NEW WAY)
    print("\nFunctional Approach (NEW):")
    print("```python")
    print("# Direct function import and use")
    print("from blackmarble.processing import create_cloud_mask")
    print("mask = create_cloud_mask(qa_pixel, conservative=True)")
    print("# Returns actual mask array - simple!")
    print("stats = compute_statistics(mask)  # Separate concern")
    print("```")

    # Show actual results
    mask = create_cloud_mask(qa_pixel, conservative=True)
    stats = compute_statistics(mask.astype(float))
    print(
        f"\nResult: {stats['count']} valid pixels ({stats['valid_fraction'] * 100:.1f}%)"
    )

    print("\n\n2. QUALITY COMPOSITING COMPARISON")
    print("-" * 50)

    # Component approach (OLD WAY)
    print("\nComponent Approach (OLD):")
    print("```python")
    print("# Complex setup with multiple components")
    print("from blackmarble.components.quality import (")
    print("    QualityCompositor, CloudMasker, AerosolMasker")
    print(")")
    print("compositor = QualityCompositor(")
    print("    components=[")
    print("        CloudMasker(name='qa_pixel'),")
    print("        AerosolMasker(name='qa_aerosol'),")
    print("    ],")
    print("    combination_method='and'")
    print(")")
    print("result = compositor.process({...})  # Returns MaskResult")
    print("mask = result.mask  # Extract actual mask")
    print("```")

    # Functional approach (NEW WAY)
    print("\nFunctional Approach (NEW):")
    print("```python")
    print("# Simple function call with clear parameters")
    print("from blackmarble.processing import create_quality_mask")
    print("mask = create_quality_mask(")
    print("    qa_bands={'qa_pixel': qa_pixel, 'qa_aerosol': qa_aerosol},")
    print("    combine_method='and',")
    print("    cloud_strategy='conservative'")
    print(")")
    print("# Direct mask return - no wrapper objects!")
    print("```")

    print("\n\n3. KEY BENEFITS OF FUNCTIONAL API")
    print("-" * 50)
    print("\n✓ Direct Returns: Functions return actual data/masks, not wrapper objects")
    print("✓ Simplicity: No complex class hierarchies or inheritance")
    print("✓ Composability: Easy to chain functions together")
    print("✓ Testability: Each function is independent and easy to test")
    print("✓ Pythonic: Follows Python's 'simple is better' philosophy")
    print("✓ Performance: Less overhead from object creation")

    print("\n\n4. REAL-WORLD EXAMPLE: MULTI-STEP PROCESSING")
    print("-" * 50)

    print("\nComponent Approach (OLD - verbose and complex):")
    print("```python")
    print("# Each step requires component instantiation")
    print("cloud_masker = CloudMasker(strategy='conservative')")
    print("cloud_result = cloud_masker.process(qa_pixel)")
    print("")
    print("dilater = MaskDilator(radius=2)")
    print("dilated_result = dilater.process(cloud_result.mask)")
    print("")
    print("enhancer = ContrastEnhancer(method='percentile')")
    print("enhanced_result = enhancer.process(data)")
    print("")
    print("# Extract actual values from results")
    print("final_mask = dilated_result.mask")
    print("final_data = enhanced_result.mask  # Confusing!")
    print("```")

    print("\nFunctional Approach (NEW - clean and clear):")
    print("```python")
    print("# Simple function composition")
    print("mask = create_cloud_mask(qa_pixel, conservative=True)")
    print("mask = dilate_mask(mask, radius=2)")
    print("enhanced = enhance_contrast(data, method='percentile')")
    print("")
    print("# Or even more concise:")
    print("mask = dilate_mask(")
    print("    create_cloud_mask(qa_pixel, conservative=True),")
    print("    radius=2")
    print(")")
    print("```")

    print("\n=== END COMPARISON ===")


def process_landsat_with_enhancements():
    """Example of processing Landsat data with urban enhancements."""

    print("\n\n=== LANDSAT PROCESSING WITH ENHANCEMENTS ===\n")

    # Create sample Landsat data
    np.random.seed(42)
    shape = (512, 512)

    # Simulate NDUI (Normalized Difference Urban Index)
    ndui = np.random.beta(2, 5, shape)  # Beta distribution for realistic values

    # Simulate road network (0=no road, 1-5=road types)
    roads = np.zeros(shape, dtype=int)
    # Add some random roads
    for _ in range(20):
        if np.random.rand() > 0.5:
            # Horizontal road
            y = np.random.randint(0, shape[0])
            roads[y, :] = np.random.choice([1, 2, 3, 4, 5])
        else:
            # Vertical road
            x = np.random.randint(0, shape[1])
            roads[:, x] = np.random.choice([1, 2, 3, 4, 5])

    # Simulate nighttime lights
    ntl = np.random.exponential(0.1, shape)
    ntl = np.clip(ntl, 0, 1)

    print("Processing steps using functional API:")
    print("1. Enhance urban features along roads")
    print("2. Apply contrast enhancement")
    print("3. Compute statistics")

    # Step 1: Enhance urban features
    enhanced_ndui = enhance_urban_features(ndui, roads, ntl, enhancement_strength=0.5)

    # Step 2: Apply contrast enhancement
    final_image = enhance_contrast(
        enhanced_ndui, method="percentile", clip_percentiles=(2, 98)
    )

    # Step 3: Compute statistics
    stats_before = compute_statistics(ndui, percentiles=[10, 50, 90])
    stats_after = compute_statistics(final_image, percentiles=[10, 50, 90])

    print("\nResults:")
    print(
        f"Original NDUI - Mean: {stats_before['mean']:.3f}, Std: {stats_before['std']:.3f}"
    )
    print(
        f"Enhanced NDUI - Mean: {stats_after['mean']:.3f}, Std: {stats_after['std']:.3f}"
    )
    print(f"Roads enhanced: {np.sum(roads > 0)} pixels")
    print(f"Bright NTL areas: {np.sum(ntl > np.percentile(ntl[ntl > 0], 75))} pixels")


def demonstrate_temporal_compositing():
    """Example of temporal compositing with quality weights."""

    print("\n\n=== TEMPORAL COMPOSITING EXAMPLE ===\n")

    # Create sample time series data
    np.random.seed(42)
    n_dates = 10
    shape = (100, 100)

    # Simulate satellite observations over time
    data_stack = np.random.rand(n_dates, *shape)

    # Add some clouds (NaN values) randomly
    for i in range(n_dates):
        cloud_mask = np.random.rand(*shape) > 0.8
        data_stack[i][cloud_mask] = np.nan

    # Create quality scores (higher = better)
    quality_scores = np.random.beta(5, 2, (n_dates, *shape))

    print("Comparing different compositing methods:")

    # Method 1: Simple median composite
    median_composite = create_temporal_composite(
        data_stack, method="median", min_valid_observations=3
    )

    # Method 2: Quality-weighted composite
    weighted_composite = create_temporal_composite(
        data_stack,
        method="quality_weighted",
        weights=quality_scores,
        min_valid_observations=3,
    )

    # Method 3: Percentile composite
    p90_composite = create_temporal_composite(
        data_stack, method="percentile", percentile=90, min_valid_observations=3
    )

    # Compare statistics
    print("\nComposite Statistics:")
    print("-" * 50)

    for name, composite in [
        ("Median", median_composite),
        ("Quality-Weighted", weighted_composite),
        ("90th Percentile", p90_composite),
    ]:
        stats = compute_statistics(composite)
        print(
            f"{name:20} - Mean: {stats['mean']:.3f}, "
            f"Valid: {stats['valid_fraction'] * 100:.1f}%"
        )


def demonstrate_scene_selection():
    """Example of intelligent scene selection."""

    print("\n\n=== SCENE SELECTION EXAMPLE ===\n")

    # Create sample scenes with dates and metadata
    n_scenes = 20
    scenes = [np.random.rand(100, 100) for _ in range(n_scenes)]

    # Generate dates over 2 months
    base_date = datetime(2023, 6, 1)
    dates = [base_date.replace(day=min(30, i + 1)) for i in range(n_scenes)]

    # Generate metadata with cloud cover
    metadata = [{"cloud_cover": np.random.uniform(0, 100)} for _ in range(n_scenes)]

    target_date = datetime(2023, 6, 15)

    print(f"Total scenes available: {n_scenes}")
    print(f"Target date: {target_date.strftime('%Y-%m-%d')}")
    print("\nApplying selection criteria:")
    print("- Maximum 7 days from target date")
    print("- Cloud cover < 30%")
    print("- Maximum 5 scenes")

    # Select scenes using functional API
    selected_scenes, selected_dates, indices = select_by_date_range(
        scenes,
        dates,
        metadata=metadata,
        target_date=target_date,
        max_days_from_target=7,
        cloud_threshold=30,
        max_scenes=5,
    )

    print(f"\nSelected {len(selected_scenes)} scenes:")
    for i, (date, idx) in enumerate(zip(selected_dates, indices)):
        cloud = metadata[idx]["cloud_cover"]
        days_diff = abs((date - target_date).days)
        print(
            f"  Scene {i + 1}: {date.strftime('%Y-%m-%d')} "
            f"(±{days_diff} days, {cloud:.1f}% clouds)"
        )


def run_functional_pipeline():
    """Run the pipeline with functional API data sources."""

    # Initialize adapters
    s3_adapter = S3Adapter(anonymous=False)
    raster_adapter = GDALRasterAdapter(use_rasterio=True)

    # Initialize functional data sources
    landsat_source = FunctionalLandsatSource(
        storage_adapter=s3_adapter,
        raster_adapter=raster_adapter,
        quality_strategy="conservative",
        dilate_clouds=2,
    )

    # Get Earthdata token
    earthdata_token = os.environ.get("EARTHDATA_TOKEN", "")

    viirs_source = FunctionalVIIRSSource(
        raster_adapter=raster_adapter, earthdata_token=earthdata_token
    )

    # Configure pipeline
    config = ProcessingConfig()
    config.landsat.composite_months = 6
    config.viirs.resolution = 0.00011111111
    config.final_resolution = 0.0001

    # Define area of interest
    lat1, lon1 = 37.7749, -122.4194  # San Francisco
    lat2, lon2 = 37.8049, -122.3894

    try:
        # Run pipeline
        output_path = run_pipeline(
            lat1=lat1,
            lon1=lon1,
            lat2=lat2,
            lon2=lon2,
            year=2023,
            month=6,
            day=15,
            earthdata_token=earthdata_token,
            output_dir="outputs_functional_api",
            config=config,
            s3_adapter=s3_adapter,
            raster_adapter=raster_adapter,
            landsat_source=landsat_source,
            viirs_source=viirs_source,
        )

        print(f"\nPipeline completed successfully!")
        print(f"Output: {output_path}")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Demonstrate Black Marble functional API"
    )
    parser.add_argument(
        "--mode",
        choices=["compare", "pipeline", "examples", "all"],
        default="compare",
        help="Demo mode: 'compare' shows API comparison, 'pipeline' runs full pipeline, "
        "'examples' shows usage examples, 'all' runs everything",
    )

    args = parser.parse_args()

    if args.mode == "compare" or args.mode == "all":
        demonstrate_functional_vs_component()

    if args.mode == "examples" or args.mode == "all":
        process_landsat_with_enhancements()
        demonstrate_temporal_compositing()
        demonstrate_scene_selection()

    if args.mode == "pipeline" or args.mode == "all":
        print("\n\nRunning functional pipeline...")
        run_functional_pipeline()
