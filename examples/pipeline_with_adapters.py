#!/usr/bin/env python3
"""
Example of using the Black Marble pipeline with dependency injection of adapters.

This demonstrates how to use the Phase 2 adapters for better testability
and flexibility in the pipeline.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from blackmarble.pipeline import run_pipeline
from blackmarble.adapters.s3 import S3Adapter
from blackmarble.adapters.raster_io import GDALRasterAdapter
from blackmarble.data_sources.landsat import LandsatDataSource
from blackmarble.data_sources.viirs import VIIRSDataSource
from blackmarble.config import ProcessingConfig


def run_pipeline_with_adapters():
    """Run the pipeline with custom adapters."""

    # Initialize adapters
    s3_adapter = S3Adapter(anonymous=False)  # Authenticated S3 access
    raster_adapter = GDALRasterAdapter(use_rasterio=True)

    # Initialize data sources with adapters
    landsat_source = LandsatDataSource(
        storage_adapter=s3_adapter, raster_adapter=raster_adapter
    )

    # Get Earthdata token from environment
    earthdata_token = os.environ.get("EARTH_DATA_TOKEN", "")

    viirs_source = VIIRSDataSource(
        raster_adapter=raster_adapter, earthdata_token=earthdata_token
    )

    # Configure pipeline
    config = ProcessingConfig()
    config.landsat.composite_months = 6  # Look back 6 months
    config.viirs.resolution = 0.00011111111  # ~500m at equator
    config.final_resolution = 0.0001  # ~10m at equator

    # Define area of interest (example: small area in California)
    lat1, lon1 = 37.7749, -122.4194  # San Francisco
    lat2, lon2 = 37.8049, -122.3894

    # Run pipeline with adapters
    try:
        output_path = run_pipeline(
            lat1=lat1,
            lon1=lon1,
            lat2=lat2,
            lon2=lon2,
            year=2023,
            month=6,
            day=15,
            earthdata_token=earthdata_token,
            output_dir="outputs_with_adapters",
            config=config,
            # Inject our custom adapters
            s3_adapter=s3_adapter,
            raster_adapter=raster_adapter,
            landsat_source=landsat_source,
            viirs_source=viirs_source,
        )

        print(f"Pipeline completed successfully!")
        print(f"Output: {output_path}")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise


def run_pipeline_legacy():
    """Run the pipeline without adapters (backward compatible)."""

    # Get required environment variables
    earthdata_token = os.environ.get("EARTH_DATA_TOKEN", "")

    # Define area of interest
    lat1, lon1 = 37.7749, -122.4194  # San Francisco
    lat2, lon2 = 37.8049, -122.3894

    # Run pipeline in legacy mode (no adapters)
    try:
        output_path = run_pipeline(
            lat1=lat1,
            lon1=lon1,
            lat2=lat2,
            lon2=lon2,
            year=2023,
            month=6,
            day=15,
            earthdata_token=earthdata_token,
            output_dir="outputs_legacy",
        )

        print(f"Legacy pipeline completed successfully!")
        print(f"Output: {output_path}")

    except Exception as e:
        print(f"Legacy pipeline failed: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Black Marble pipeline with or without adapters"
    )
    parser.add_argument(
        "--mode",
        choices=["adapters", "legacy"],
        default="adapters",
        help="Run mode: 'adapters' uses new dependency injection, 'legacy' uses old approach",
    )

    args = parser.parse_args()

    if args.mode == "adapters":
        print("Running pipeline with adapters...")
        run_pipeline_with_adapters()
    else:
        print("Running pipeline in legacy mode...")
        run_pipeline_legacy()
