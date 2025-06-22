#!/usr/bin/env python3
"""
Example of integrating the new quality control components with the Black Marble pipeline.

This demonstrates how to use the modular quality components with the Phase 2 adapters
for a cleaner, more testable pipeline implementation.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np

# Add parent directory to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from blackmarble.pipeline import run_pipeline
from blackmarble.adapters.s3 import S3Adapter
from blackmarble.adapters.raster_io import GDALRasterAdapter
from blackmarble.data_sources.landsat import LandsatDataSource
from blackmarble.data_sources.viirs import VIIRSDataSource
from blackmarble.config import ProcessingConfig

# Import the new quality components
from blackmarble.components.quality import (
    CloudMasker,
    AerosolMasker,
    RadianceSaturationMasker,
    VIIRSQualityMasker,
    QualityCompositor,
    MaskStatistics,
    create_landsat_qa_pipeline,
)
from blackmarble.components.base import apply_mask


class QualityEnhancedLandsatSource(LandsatDataSource):
    """Extended Landsat data source that uses the new quality components."""

    def __init__(
        self,
        storage_adapter=None,
        raster_adapter=None,
        quality_strategy="conservative",
        dilate_clouds=0,
    ):
        """Initialize with quality processing options.

        Args:
            storage_adapter: Storage adapter for S3/cloud access
            raster_adapter: Raster I/O adapter
            quality_strategy: Cloud masking strategy ("conservative" or "permissive")
            dilate_clouds: Dilation radius for cloud masks
        """
        super().__init__(storage_adapter, raster_adapter)

        # Create quality pipeline
        self.quality_pipeline = create_landsat_qa_pipeline(
            strategy=quality_strategy, dilate_clouds=dilate_clouds
        )

        # Add statistics tracker
        self.mask_stats = MaskStatistics(name="landsat_mask_stats")

        # Create radsat masker separately (applied per band)
        self.radsat_masker = RadianceSaturationMasker()

    def fetch_with_quality(
        self,
        bbox,
        start_date,
        end_date,
        bands=None,
        composite_months=1,
        return_quality_report=False,
    ):
        """Fetch Landsat data with quality control applied.

        Args:
            bbox: Bounding box (west, south, east, north)
            start_date: Start date for search
            end_date: End date for search
            bands: List of band names to fetch
            composite_months: Number of months to composite
            return_quality_report: Whether to return quality statistics

        Returns:
            Dictionary with processed bands and optionally quality report
        """
        # Fetch raw data using parent method
        raw_data = self.fetch(bbox, start_date, end_date, bands, composite_months)

        # Ensure we have the QA bands
        qa_bands = ["QA_PIXEL", "QA_AEROSOL", "QA_RADSAT"]
        missing_qa = [b for b in qa_bands if b not in raw_data]
        if missing_qa:
            raise ValueError(f"Missing required QA bands: {missing_qa}")

        # Process quality masks
        qa_data = {
            "qa_pixel": raw_data["QA_PIXEL"],
            "qa_aerosol": raw_data["QA_AEROSOL"],
        }

        # Get combined cloud/aerosol mask
        quality_result = self.quality_pipeline.process(qa_data)
        combined_mask = quality_result.mask

        # Track statistics
        self.mask_stats.process(combined_mask, mask_name="cloud_aerosol_combined")

        # Apply combined mask to all bands except QA bands
        processed_data = {}
        for band_name, band_data in raw_data.items():
            if band_name in qa_bands:
                # Keep QA bands as-is
                processed_data[band_name] = band_data
            else:
                # Apply quality mask to data bands
                masked_data = apply_mask(band_data, combined_mask)
                processed_data[band_name] = masked_data

                # For bands 3, 4, 5, also apply radsat mask
                if band_name in ["B3", "B4", "B5"]:
                    band_num = int(band_name[1])
                    radsat_result = self.radsat_masker.process(
                        raw_data["QA_RADSAT"], band=band_num
                    )
                    # Apply radsat mask
                    processed_data[band_name] = apply_mask(
                        processed_data[band_name], radsat_result.mask
                    )
                    # Track statistics
                    self.mask_stats.process(
                        radsat_result.mask, mask_name=f"radsat_band_{band_num}"
                    )

        if return_quality_report:
            # Generate comprehensive quality report
            report = {
                "quality_summary": self.mask_stats.get_summary(),
                "cloud_aerosol_details": quality_result.metadata,
                "processing_strategy": self.quality_pipeline.get_config(),
            }
            return processed_data, report

        return processed_data


class QualityEnhancedVIIRSSource(VIIRSDataSource):
    """Extended VIIRS data source that uses the new quality components."""

    def __init__(self, api_adapter=None, raster_adapter=None, earthdata_token=None):
        """Initialize with quality processing.

        Args:
            api_adapter: API adapter for NASA Earthdata
            raster_adapter: Raster I/O adapter
            earthdata_token: Earthdata authentication token
        """
        super().__init__(api_adapter, raster_adapter, earthdata_token)

        # Create VIIRS quality masker
        self.quality_masker = VIIRSQualityMasker()

        # Track statistics
        self.mask_stats = MaskStatistics(name="viirs_mask_stats")

    def fetch_with_quality(
        self, bbox, start_date, end_date, layers=None, return_quality_report=False
    ):
        """Fetch VIIRS data with quality control applied.

        Args:
            bbox: Bounding box (west, south, east, north)
            start_date: Start date for search
            end_date: End date for search
            layers: List of layers to fetch
            return_quality_report: Whether to return quality statistics

        Returns:
            Dictionary with processed layers and optionally quality report
        """
        # Fetch raw data
        raw_data = self.fetch(bbox, start_date, end_date, layers)

        # Ensure we have quality flag
        if "Mandatory_Quality_Flag" not in raw_data:
            raise ValueError("Missing Mandatory_Quality_Flag layer")

        # Process quality mask
        quality_result = self.quality_masker.process(raw_data["Mandatory_Quality_Flag"])
        quality_mask = quality_result.mask

        # Track statistics
        self.mask_stats.process(quality_mask, mask_name="viirs_quality")

        # Apply mask to NTL data
        processed_data = {}
        for layer_name, layer_data in raw_data.items():
            if layer_name == "Mandatory_Quality_Flag":
                processed_data[layer_name] = layer_data
            elif layer_name in ["NTL", "DNB_At_Sensor_Radiance"]:
                # Apply quality mask to nighttime lights
                processed_data[layer_name] = apply_mask(layer_data, quality_mask)
            else:
                # Keep other layers as-is
                processed_data[layer_name] = layer_data

        if return_quality_report:
            report = {
                "quality_summary": self.mask_stats.get_summary(),
                "viirs_quality_details": quality_result.metadata,
            }
            return processed_data, report

        return processed_data


def run_quality_enhanced_pipeline():
    """Run the pipeline with quality-enhanced data sources."""

    # Initialize adapters
    s3_adapter = S3Adapter(anonymous=False)
    raster_adapter = GDALRasterAdapter(use_rasterio=True)

    # Initialize quality-enhanced data sources
    landsat_source = QualityEnhancedLandsatSource(
        storage_adapter=s3_adapter,
        raster_adapter=raster_adapter,
        quality_strategy="conservative",  # Use conservative cloud masking
        dilate_clouds=2,  # Dilate cloud masks by 2 pixels
    )

    # Get Earthdata token
    earthdata_token = os.environ.get("EARTH_DATA_TOKEN", "")

    viirs_source = QualityEnhancedVIIRSSource(
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
            output_dir="outputs_quality_enhanced",
            config=config,
            s3_adapter=s3_adapter,
            raster_adapter=raster_adapter,
            landsat_source=landsat_source,
            viirs_source=viirs_source,
        )

        print(f"Pipeline completed successfully!")
        print(f"Output: {output_path}")

        # Print quality reports
        print("\nLandsat Quality Report:")
        print(landsat_source.mask_stats.get_summary())

        print("\nVIIRS Quality Report:")
        print(viirs_source.mask_stats.get_summary())

    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise


def demonstrate_quality_components_standalone():
    """Demonstrate using quality components standalone."""

    print("=== Standalone Quality Component Demo ===\n")

    # Create sample data
    np.random.seed(42)
    shape = (100, 100)

    # Simulate QA_PIXEL band with various cloud conditions
    qa_pixel = np.random.choice(
        [21824, 21952, 22280, 54596, 54852, 55052],  # Clear land, clear water, clouds
        size=shape,
        p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05],  # Probabilities
    )

    # Test conservative vs permissive strategies
    print("1. Cloud Masking Comparison:")

    conservative_masker = CloudMasker(strategy="conservative")
    permissive_masker = CloudMasker(strategy="permissive")

    cons_result = conservative_masker.process(qa_pixel)
    perm_result = permissive_masker.process(qa_pixel)

    print(
        f"   Conservative: {cons_result.statistics['valid_pixels']} valid pixels "
        f"({cons_result.statistics['valid_percentage']:.1f}%)"
    )
    print(
        f"   Permissive: {perm_result.statistics['valid_pixels']} valid pixels "
        f"({perm_result.statistics['valid_percentage']:.1f}%)"
    )

    # Test cloud dilation
    print("\n2. Cloud Dilation Effect:")

    dilated_masker = CloudMasker(strategy="conservative", dilate_radius=3)
    dilated_result = dilated_masker.process(qa_pixel)

    print(f"   No dilation: {cons_result.statistics['valid_pixels']} valid pixels")
    print(
        f"   With dilation (r=3): {dilated_result.statistics['valid_pixels']} valid pixels"
    )

    # Test quality compositor
    print("\n3. Quality Compositor:")

    # Simulate QA_AEROSOL band
    qa_aerosol = np.random.choice(
        [0, 192, 194, 196, 228], size=shape, p=[0.7, 0.1, 0.1, 0.05, 0.05]
    )

    # Create composite pipeline
    compositor = QualityCompositor(
        components=[
            CloudMasker(name="qa_pixel", strategy="conservative"),
            AerosolMasker(name="qa_aerosol", threshold="standard"),
        ],
        combination_method="and",
    )

    composite_result = compositor.process(
        {"qa_pixel": qa_pixel, "qa_aerosol": qa_aerosol}
    )

    print(
        f"   Combined mask: {composite_result.statistics['valid_pixels']} valid pixels "
        f"({composite_result.statistics['valid_percentage']:.1f}%)"
    )

    # Show component contributions
    if (
        composite_result.metadata is not None
        and "component_statistics" in composite_result.metadata
    ):
        comp_stats = composite_result.metadata["component_statistics"]
        print(
            f"   - Cloud mask contribution: {comp_stats['qa_pixel']['valid_percentage']:.1f}%"
        )
        print(
            f"   - Aerosol mask contribution: {comp_stats['qa_aerosol']['valid_percentage']:.1f}%"
        )
    else:
        print("   - Component statistics not available in metadata")

    # Test statistics tracking
    print("\n4. Statistics Tracking:")

    tracker = MaskStatistics()

    # Track multiple masks
    tracker.process(cons_result.mask, mask_name="conservative_clouds")
    tracker.process(perm_result.mask, mask_name="permissive_clouds")
    tracker.process(composite_result.mask, mask_name="final_composite")

    summary = tracker.get_summary()
    print(f"   Total masks tracked: {summary['total_masks']}")
    print(f"   Total pixels processed: {summary['total_pixels_processed']:,}")
    print(f"   Overall valid percentage: {summary['overall_valid_percentage']:.1f}%")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Black Marble pipeline with quality components"
    )
    parser.add_argument(
        "--mode",
        choices=["pipeline", "demo"],
        default="demo",
        help="Run mode: 'pipeline' runs full pipeline, 'demo' shows component usage",
    )

    args = parser.parse_args()

    if args.mode == "pipeline":
        print("Running quality-enhanced pipeline...")
        run_quality_enhanced_pipeline()
    else:
        demonstrate_quality_components_standalone()
