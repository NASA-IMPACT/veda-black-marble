#!/usr/bin/env python3
"""
Black Marble pipeline using functional API from blackmarble.processing.

This replaces the component-based pipeline with direct function calls,
making the code simpler and more straightforward while maintaining
the same functionality.
"""

import os
import sys
from pathlib import Path
from typing import Any
from datetime import datetime
import numpy as np
import boto3
from dataclasses import dataclass
import logging

# Functional API imports
from blackmarble.processing import (
    # Spatial transformation functions
    resample_image,
    reproject_image,
    crop_to_bounds,
    align_rasters,
    # Mask generation functions
    create_cloud_mask,
    create_quality_mask,
    create_aerosol_mask,
    create_saturation_mask,
    create_water_mask,
    create_viirs_quality_mask,
    combine_masks,
    dilate_mask,
    # Enhancement functions
    enhance_contrast,
    normalize_data,
    scale_to_byte,
    apply_colormap,
    enhance_roads,
    enhance_urban_features,
    # Temporal and composite functions
    create_temporal_composite,
    merge_scenes,
    select_by_date_range,
    composite_with_quality,
    compute_statistics,
)

# Core imports
from blackmarble.config import ProcessingConfig
from blackmarble.coords import get_utm_zone

# Data source and adapter imports
from blackmarble.adapters.s3 import S3Adapter
from blackmarble.adapters.raster_io import GDALRasterAdapter
from blackmarble.data_sources.landsat import LandsatDataSource
from blackmarble.data_sources.viirs import VIIRSDataSource

# Calculation imports
from blackmarble.calculations import (
    calculate_ndvi,
    calculate_ndwi,
    calculate_ndui,
)

# For COG generation, we still need the component
from blackmarble.components.output import COGGenerator, COGProfile

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Simplified context containing configuration and data sources."""

    config: ProcessingConfig
    s3_adapter: S3Adapter
    raster_adapter: GDALRasterAdapter
    landsat_source: LandsatDataSource
    viirs_source: VIIRSDataSource
    osm_source: Any | None  # OSM data source not yet implemented
    cog_generator: COGGenerator  # Still need this for COG output
    temp_dir: Path
    final_dir: Path

    @classmethod
    def create(
        cls,
        config: ProcessingConfig,
        output_dir: str = "outputs",
        s3_adapter: S3Adapter | None = None,
        raster_adapter: GDALRasterAdapter | None = None,
        landsat_source: LandsatDataSource | None = None,
        viirs_source: VIIRSDataSource | None = None,
        osm_source: Any | None = None,
    ) -> "PipelineContext":
        """Create a pipeline context with all dependencies."""
        # Initialize adapters if not provided
        if s3_adapter is None:
            s3_adapter = S3Adapter(anonymous=False)
        if raster_adapter is None:
            raster_adapter = GDALRasterAdapter()

        # Initialize data sources if not provided
        if landsat_source is None:
            landsat_source = LandsatDataSource(
                storage_adapter=s3_adapter,
                raster_adapter=raster_adapter,
            )
        if viirs_source is None:
            viirs_source = VIIRSDataSource(
                raster_adapter=raster_adapter,
            )

        # Create directories
        temp_dir = Path(output_dir) / "temp"
        final_dir = Path(output_dir) / "final"
        temp_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)

        # Still need COG generator for output
        cog_generator = COGGenerator(profile=COGProfile(profile_name="deflate"))

        return cls(
            config=config,
            s3_adapter=s3_adapter,
            raster_adapter=raster_adapter,
            landsat_source=landsat_source,
            viirs_source=viirs_source,
            osm_source=osm_source,
            cog_generator=cog_generator,
            temp_dir=temp_dir,
            final_dir=final_dir,
        )


class FunctionalPipeline:
    """Black Marble pipeline using functional API."""

    def __init__(self, context: PipelineContext):
        """Initialize pipeline with context containing dependencies."""
        self.ctx = context

    def run(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        year: int,
        month: int,
        day: int,
        earthdata_token: str,
        upload_bucket: str | None = None,
        upload_key: str | None = None,
    ) -> str:
        """
        Run the Black Marble processing pipeline using functional API.

        This demonstrates a cleaner approach using direct function calls
        instead of component classes.
        """
        logger.info(f"Starting pipeline for bbox: ({lat1}, {lon1}) to ({lat2}, {lon2})")
        logger.info(f"Target date: {year}-{month:02d}-{day:02d}")

        # Phase 1: Data Acquisition
        logger.info("Phase 1: Data Acquisition")
        landsat_data = self._acquire_landsat_data(
            lat1, lon1, lat2, lon2, year, month, day
        )
        viirs_data = self._acquire_viirs_data(
            lat1, lon1, lat2, lon2, year, month, day, earthdata_token
        )
        osm_data = self._acquire_osm_data(lat1, lon1, lat2, lon2)

        # Phase 2: Quality Assessment and Masking
        logger.info("Phase 2: Quality Assessment and Masking")
        masked_landsat = self._apply_quality_masks(landsat_data)
        masked_viirs = self._apply_viirs_quality(viirs_data)

        # Phase 3: Spatial Processing
        logger.info("Phase 3: Spatial Processing")
        aligned_data = self._spatial_processing(
            masked_landsat, masked_viirs, osm_data, lat1, lon1, lat2, lon2
        )

        # Phase 4: Temporal Compositing
        logger.info("Phase 4: Temporal Compositing")
        composite_data = self._temporal_compositing(aligned_data)

        # Phase 5: Index Calculation
        logger.info("Phase 5: Index Calculation")
        indices = self._calculate_indices(composite_data)

        # Phase 6: Enhancement and Visualization
        logger.info("Phase 6: Enhancement and Visualization")
        enhanced_data = self._enhance_visualization(indices)

        # Phase 7: Output Generation
        logger.info("Phase 7: Output Generation")
        output_path = self._generate_output(enhanced_data)

        # Phase 8: Upload (optional)
        if upload_bucket and upload_key:
            logger.info("Phase 8: Upload to S3")
            self._upload_to_s3(output_path, upload_bucket, upload_key)

        logger.info(f"Pipeline complete. Output: {output_path}")
        return output_path

    def _acquire_landsat_data(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        year: int,
        month: int,
        day: int,
    ) -> dict[str, Any]:
        """Phase 1a: Acquire Landsat data using data source."""
        logger.info("Acquiring Landsat data...")

        bbox = (min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2))
        start_date = datetime(year, month, day)

        # Use data source to fetch data
        data: dict[str, Any] = self.ctx.landsat_source.fetch(
            bbox=bbox,
            start_date=start_date,
            end_date=start_date,
            bands=["B3", "B4", "B5", "QA_PIXEL", "QA_AEROSOL", "QA_RADSAT"],
            composite_months=self.ctx.config.landsat.composite_months,
        )

        # Get UTM zone information
        e1, n1, zone1, letter1 = get_utm_zone(lat1, lon1)
        e2, n2, zone2, letter2 = get_utm_zone(lat2, lon2)

        if zone1 != zone2 or letter1 != letter2:
            logger.warning(
                f"Bounding box spans UTM zones ({zone1}{letter1} to {zone2}{letter2})"
            )

        data["utm_bounds"] = {
            "e1": e1,
            "n1": n1,
            "e2": e2,
            "n2": n2,
            "zone": zone1,
            "letter": letter1,
        }

        return data

    def _acquire_viirs_data(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        year: int,
        month: int,
        day: int,
        earthdata_token: str,
    ) -> dict[str, Any]:
        """Phase 1b: Acquire VIIRS nighttime lights data."""
        logger.info("Acquiring VIIRS data...")

        bbox = (min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2))
        start_date = datetime(year, month, day)

        # Fetch VIIRS data
        data = self.ctx.viirs_source.fetch(
            bbox=bbox,
            start_date=start_date,
            end_date=start_date,
            layers=["NTL", "LUNAR", "QF"],
        )

        return data

    def _acquire_osm_data(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> dict[str, Any]:
        """Phase 1c: Acquire OpenStreetMap road data."""
        logger.info("Acquiring OSM road data...")

        bbox = (min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2))

        # For now, return empty data since OSM source is not implemented
        if self.ctx.osm_source is None:
            logger.warning("OSM data source not configured, using empty roads data")
            return {"roads": {}}

        data = self.ctx.osm_source.fetch(
            bbox=bbox,
            features=["roads"],
        )

        return data

    def _apply_quality_masks(self, landsat_data: dict[str, Any]) -> dict[str, Any]:
        """Phase 2a: Apply quality masks to Landsat data using functional API."""
        logger.info("Applying quality masks to Landsat data...")

        # Extract quality bands
        qa_pixel = landsat_data.get("QA_PIXEL")
        qa_aerosol = landsat_data.get("QA_AEROSOL")
        qa_radsat = landsat_data.get("QA_RADSAT")

        # Extract data bands
        b3 = landsat_data.get("B3")
        b4 = landsat_data.get("B4")
        b5 = landsat_data.get("B5")

        if b3 is None or b4 is None or b5 is None:
            raise ValueError("Missing required Landsat bands")

        # Create quality mask using functional API
        qa_bands = {}
        if qa_pixel is not None:
            qa_bands["qa_pixel"] = qa_pixel
        if qa_aerosol is not None:
            qa_bands["qa_aerosol"] = qa_aerosol
        if qa_radsat is not None:
            qa_bands["qa_radsat"] = qa_radsat

        # Create combined quality mask
        quality_mask = create_quality_mask(
            qa_bands=qa_bands,
            combine_method="and",
            cloud_strategy="conservative",
            cloud_dilate=2,
            aerosol_threshold="standard",
            include_saturation=True,
            saturation_bands=[3, 4, 5],
        )

        # Compute statistics
        stats = compute_statistics(quality_mask)
        logger.info(
            f"Quality mask statistics: {stats['valid_fraction'] * 100:.1f}% valid pixels"
        )

        # Apply mask to the bands
        masked_b3 = np.where(quality_mask, b3, np.nan)
        masked_b4 = np.where(quality_mask, b4, np.nan)
        masked_b5 = np.where(quality_mask, b5, np.nan)

        return {
            "B3": masked_b3,
            "B4": masked_b4,
            "B5": masked_b5,
            "mask": quality_mask,
            "utm_bounds": landsat_data["utm_bounds"],
        }

    def _apply_viirs_quality(self, viirs_data: dict[str, Any]) -> dict[str, Any]:
        """Phase 2b: Apply quality filtering to VIIRS data using functional API."""
        logger.info("Applying quality masks to VIIRS data...")

        quality_flag = viirs_data.get("QF")
        ntl = viirs_data.get("NTL")
        lunar = viirs_data.get("LUNAR")

        if ntl is None or lunar is None:
            raise ValueError("Missing required VIIRS data layers")

        # Create VIIRS quality mask
        quality_mask = create_viirs_quality_mask(
            quality_flag,
            valid_flags=[0, 2],  # High and moderate quality
        )

        # Mask invalid values
        ntl_masked = np.where(quality_mask & (ntl <= 65530), ntl, np.nan)

        # Apply lunar correction where lunar data is valid
        corrected = np.where(
            (lunar > 0) & (lunar <= 65530),
            ntl_masked * (10.0 / lunar),  # Normalize by lunar irradiance
            ntl_masked,
        )

        # Normalize to 0-1 range
        corrected = normalize_data(
            corrected, output_min=0.0, output_max=1.0, input_min=0.0, input_max=100.0
        )

        return {
            "NTL": corrected,
            "quality_mask": quality_mask,
        }

    def _spatial_processing(
        self,
        landsat_data: dict[str, Any],
        viirs_data: dict[str, Any],
        osm_data: dict[str, Any],
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> dict[str, Any]:
        """Phase 3: Spatial processing using functional API."""
        logger.info("Performing spatial processing...")

        # Get UTM bounds from landsat data
        utm_bounds = landsat_data["utm_bounds"]

        # For this simplified example, we'll work with the data as-is
        # In a real implementation, you would need proper transforms from the data sources

        # Step 1: Ensure all data is in the same CRS (WGS84 for simplicity)
        # Assuming data is already in WGS84 from the sources

        # Step 2: Resample all data to final resolution
        logger.info(
            f"Resampling to final resolution: {self.ctx.config.final_resolution}m"
        )

        # Create target bounds
        target_bounds = (lon1, lat1, lon2, lat2)

        # For demonstration, we'll assume the data needs resampling
        # In reality, you'd get the actual resolution from the data source metadata
        source_resolution = 30.0  # Landsat resolution in meters
        target_resolution = self.ctx.config.final_resolution

        resampled_data = {}

        # Resample Landsat bands
        for band in ["B3", "B4", "B5"]:
            data = landsat_data[band]
            if data is not None:
                # Simplified resampling - in real use, you'd have proper transforms
                resampled, _ = resample_image(
                    data=data,
                    source_resolution=source_resolution,
                    target_resolution=target_resolution,
                    method="bilinear",
                )
                resampled_data[band] = resampled

        # Resample VIIRS (assuming 750m resolution)
        viirs_resolution = 750.0
        resampled_ntl, _ = resample_image(
            data=viirs_data["NTL"],
            source_resolution=viirs_resolution,
            target_resolution=target_resolution,
            method="bilinear",
        )
        resampled_data["NTL"] = resampled_ntl

        # Create placeholder for roads (since OSM processing isn't implemented)
        # In real implementation, you'd rasterize OSM vector data
        resampled_data["roads"] = np.zeros_like(resampled_data["NTL"])

        return resampled_data

    def _temporal_compositing(self, aligned_data: dict[str, Any]) -> dict[str, Any]:
        """Phase 4: Temporal compositing using functional API."""
        logger.info("Performing temporal compositing...")

        composite_data = {}

        # For this example, we assume aligned_data contains single-date imagery
        # In a real implementation, you would have multiple dates to composite

        # Apply median compositing to Landsat bands
        for band in ["B3", "B4", "B5"]:
            # For single date, just pass through
            band_data = aligned_data[band]

            # If we had multiple dates, we'd use create_temporal_composite
            # For now, simulate with single date
            if band_data.ndim == 2:
                # Add time dimension for API compatibility
                band_data_3d = band_data[np.newaxis, ...]
                composite = create_temporal_composite(
                    data_stack=band_data_3d,
                    timestamps=[datetime.now()],
                    method="median",
                    min_valid_observations=1,
                )
            else:
                composite = band_data

            composite_data[band] = composite

        # Pass through non-temporal data
        composite_data["NTL"] = aligned_data["NTL"]
        composite_data["roads"] = aligned_data["roads"]

        return composite_data

    def _calculate_indices(self, composite_data: dict[str, Any]) -> dict[str, Any]:
        """Phase 5: Calculate spectral indices."""
        logger.info("Calculating spectral indices...")

        # Extract bands
        b3 = composite_data["B3"]
        b4 = composite_data["B4"]
        b5 = composite_data["B5"]
        ntl = composite_data["NTL"]
        roads = composite_data["roads"]

        # Calculate indices
        ndvi = calculate_ndvi(b4, b5)
        ndwi = calculate_ndwi(b3, b5)
        ndui = calculate_ndui(ntl, ndvi, ndwi, roads, self.ctx.config.ndui)

        # Compute statistics for logging
        ndvi_stats = compute_statistics(ndvi)
        ndwi_stats = compute_statistics(ndwi)
        ndui_stats = compute_statistics(ndui)

        logger.info(f"NDVI range: {ndvi_stats['min']:.3f} to {ndvi_stats['max']:.3f}")
        logger.info(f"NDWI range: {ndwi_stats['min']:.3f} to {ndwi_stats['max']:.3f}")
        logger.info(f"NDUI range: {ndui_stats['min']:.3f} to {ndui_stats['max']:.3f}")

        return {
            "NDVI": ndvi,
            "NDWI": ndwi,
            "NDUI": ndui,
            "B3": b3,
            "B4": b4,
            "B5": b5,
            "NTL": ntl,
            "roads": roads,
        }

    def _enhance_visualization(self, indices: dict[str, Any]) -> np.ndarray:
        """Phase 6: Enhance and colorize the NDUI using functional API."""
        logger.info("Enhancing visualization...")

        ndui = indices["NDUI"]
        roads = indices["roads"]
        ntl = indices["NTL"]

        # Apply contrast enhancement
        enhanced = enhance_contrast(
            data=ndui,
            method="percentile",
            clip_percentiles=(2, 98),
            min_value=self.ctx.config.display_min,
            max_value=self.ctx.config.display_max,
        )

        # Apply urban feature enhancement
        enhanced = enhance_urban_features(
            ndui=enhanced,
            roads=roads,
            ntl=ntl,
            enhancement_strength=0.3,
        )

        # Apply colormap to create RGB visualization
        rgb_array = apply_colormap(
            data=enhanced,
            colormap="inferno",
            vmin=0.0,
            vmax=1.0,
        )

        return rgb_array

    def _generate_output(self, rgb_array: np.ndarray) -> str:
        """Phase 7: Generate Cloud-Optimized GeoTIFF output."""
        logger.info("Generating Cloud-Optimized GeoTIFF...")

        output_path = self.ctx.final_dir / "ndui-co.tif"

        # Generate COG using the component (since it handles the GDAL complexity)
        cog_result = self.ctx.cog_generator.process(
            data=rgb_array,
            output_path=output_path,
            crs="EPSG:4326",
            transform=None,  # Would be computed from bounds and resolution
            tags={
                "description": "Black Marble NDUI Composite",
                "processing_date": datetime.now().isoformat(),
                "software": "Black Marble Pipeline (Functional)",
            },
        )

        # Compute output statistics
        stats = compute_statistics(rgb_array[0])  # Stats for first band
        logger.info(f"Output statistics: {stats}")

        logger.info(f"Generated COG: {output_path}")
        return str(output_path)

    def _upload_to_s3(self, file_path: str, bucket: str, key: str) -> None:
        """Phase 8: Upload output to S3."""
        logger.info(f"Uploading to s3://{bucket}/{key}")

        try:
            self.ctx.s3_adapter.upload_file(Path(file_path), bucket, key)
            logger.info("Upload complete")
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise


def run_functional_pipeline(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    year: int,
    month: int,
    day: int,
    earthdata_token: str,
    output_dir: str = "outputs",
    upload_bucket: str | None = None,
    upload_key: str | None = None,
    config: ProcessingConfig | None = None,
    **kwargs: Any,
) -> str:
    """
    Run the Black Marble pipeline using functional API.

    This function maintains the same interface as the original run_pipeline
    but uses functional programming style internally.

    Key improvements over component-based approach:
    - Direct function calls instead of component.process()
    - Functions return actual data instead of MaskResult objects
    - Simpler, more readable code
    - Easier to test individual functions
    - Less abstraction overhead
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize configuration
    if config is None:
        config = ProcessingConfig.from_env()

    # Create pipeline context with dependencies
    context = PipelineContext.create(
        config=config,
        output_dir=output_dir,
        **kwargs,  # Allow injection of custom adapters/sources
    )

    # Create and run pipeline
    pipeline = FunctionalPipeline(context)

    return pipeline.run(
        lat1=lat1,
        lon1=lon1,
        lat2=lat2,
        lon2=lon2,
        year=year,
        month=month,
        day=day,
        earthdata_token=earthdata_token,
        upload_bucket=upload_bucket,
        upload_key=upload_key,
    )


def main():
    """Entry point demonstrating the functional pipeline."""
    try:
        earthdata_token = os.environ["EARTH_DATA_TOKEN"]
        lat1 = float(os.environ["LAT1"])
        lon1 = float(os.environ["LONG1"])
        lat2 = float(os.environ["LAT2"])
        lon2 = float(os.environ["LONG2"])
        year = int(os.environ["YEAR"])
        month = int(os.environ["MONTH"])
        day = int(os.environ["DAY"])
    except KeyError as e:
        print(f"Error: Missing required environment variable: {e}", file=sys.stderr)
        sys.exit(1)

    # Optional S3 upload parameters
    upload_bucket = os.environ.get("UPLOAD_BUCKET")
    upload_key = os.environ.get("UPLOAD_FILE")

    try:
        final_path = run_functional_pipeline(
            lat1=lat1,
            lon1=lon1,
            lat2=lat2,
            lon2=lon2,
            year=year,
            month=month,
            day=day,
            earthdata_token=earthdata_token,
            upload_bucket=upload_bucket,
            upload_key=upload_key,
        )
        print(f"Pipeline complete. Final output: {final_path}")
    except Exception as e:
        print(f"Pipeline failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
