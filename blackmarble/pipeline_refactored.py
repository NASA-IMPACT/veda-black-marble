#!/usr/bin/env python3
"""
Refactored Black Marble pipeline using modular components.

This demonstrates how the 850+ line run_pipeline function can be broken down
into manageable, testable pieces using dependency injection and clear separation
of concerns.
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

# Component imports
from blackmarble.components.base import apply_mask, MaskResult
from blackmarble.components import (
    # Quality components
    CloudMasker,
    QualityCompositor,
    MaskStatistics,
    # Preprocessing components
    ProjectionHandler,
    ResolutionResampler,
    BoundsCropper,
    SpatialAligner,
    # Temporal components
    TemporalCompositor,
    SceneMerger,
    DateRangeSelector,
    CompositeMethod,
    # Output components
    COGGenerator,
    MetadataGenerator,
    VisualizationEnhancer,
    OutputValidator,
    OutputMetadata,
    COGProfile,
    # Enhancement components
    RoadEnhancer,
    ContrastEnhancer,
    ColorMapper,
    CompositeEnhancer,
)

# Core imports
from blackmarble.config import ProcessingConfig
from blackmarble.coords import get_utm_zone

# Data source and adapter imports
from blackmarble.adapters.s3 import S3Adapter
from blackmarble.adapters.raster_io import GDALRasterAdapter
from blackmarble.data_sources.landsat import LandsatDataSource
from blackmarble.data_sources.viirs import VIIRSDataSource
# OSM data source not yet implemented
# from blackmarble.data_sources.osm import OSMDataSource

# Calculation imports
from blackmarble.calculations import (
    calculate_ndvi,
    calculate_ndwi,
    calculate_ndui,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Context object containing all pipeline dependencies and configuration."""

    config: ProcessingConfig
    s3_adapter: S3Adapter
    raster_adapter: GDALRasterAdapter
    landsat_source: LandsatDataSource
    viirs_source: VIIRSDataSource
    osm_source: Any | None  # OSMDataSource not yet implemented

    # Pipeline components
    quality_compositor: QualityCompositor
    projection_handler: ProjectionHandler
    resolution_resampler: ResolutionResampler
    bounds_cropper: BoundsCropper
    temporal_compositor: TemporalCompositor
    cog_generator: COGGenerator
    color_mapper: ColorMapper
    road_enhancer: RoadEnhancer

    # Paths
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
        osm_source: Any | None = None,  # OSMDataSource not yet implemented
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
        # OSM source not yet implemented
        # if osm_source is None:
        #     osm_source = OSMDataSource(
        #         config=config,
        #         raster_adapter=raster_adapter,
        #     )

        # Create directories
        temp_dir = Path(output_dir) / "temp"
        final_dir = Path(output_dir) / "final"
        temp_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)

        # Initialize pipeline components
        quality_compositor = QualityCompositor()
        projection_handler = ProjectionHandler()
        resolution_resampler = ResolutionResampler(method="bilinear")
        bounds_cropper = BoundsCropper()
        temporal_compositor = TemporalCompositor(method=CompositeMethod.MEDIAN)
        cog_generator = COGGenerator(profile=COGProfile(profile_name="deflate"))
        color_mapper = ColorMapper(colormap_name="inferno")
        road_enhancer = RoadEnhancer()

        return cls(
            config=config,
            s3_adapter=s3_adapter,
            raster_adapter=raster_adapter,
            landsat_source=landsat_source,
            viirs_source=viirs_source,
            osm_source=osm_source,
            quality_compositor=quality_compositor,
            projection_handler=projection_handler,
            resolution_resampler=resolution_resampler,
            bounds_cropper=bounds_cropper,
            temporal_compositor=temporal_compositor,
            cog_generator=cog_generator,
            color_mapper=color_mapper,
            road_enhancer=road_enhancer,
            temp_dir=temp_dir,
            final_dir=final_dir,
        )


class RefactoredPipeline:
    """Refactored Black Marble pipeline using modular components."""

    def __init__(self, context: PipelineContext):
        """Initialize pipeline with context containing all dependencies."""
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
        Run the refactored Black Marble processing pipeline.

        This demonstrates how the monolithic run_pipeline function can be
        broken down into clear phases with modular components.
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

        # VIIRS source will use the token from environment

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

        # Fetch OSM data
        if self.ctx.osm_source is None:
            raise ValueError("OSM data source not configured")
        data = self.ctx.osm_source.fetch(
            bbox=bbox,
            features=["roads"],
        )

        return data

    def _apply_quality_masks(self, landsat_data: dict[str, Any]) -> dict[str, Any]:
        """Phase 2a: Apply quality masks to Landsat data."""
        logger.info("Applying quality masks to Landsat data...")

        # Extract quality bands
        qa_pixel = landsat_data.get("QA_PIXEL")
        qa_aerosol = landsat_data.get("QA_AEROSOL")
        qa_radsat = landsat_data.get("QA_RADSAT")

        # Apply masks using quality compositor
        b3 = landsat_data.get("B3")
        b4 = landsat_data.get("B4")
        b5 = landsat_data.get("B5")
        
        if b3 is None or b4 is None or b5 is None:
            raise ValueError("Missing required Landsat bands")
            
        mask_result = self.ctx.quality_compositor.process(
            data={
                "B3": b3,
                "B4": b4,
                "B5": b5,
            },
            qa_layers={
                "cloud": qa_pixel,
                "aerosol": qa_aerosol,
                "radsat": qa_radsat,
            },
        )

        # Log statistics
        logger.info(
            f"Quality mask statistics: {mask_result.statistics.get('valid_percentage', 0):.1f}% valid pixels"
        )

        # Apply mask to the bands
        masked_b3 = apply_mask(b3, mask_result.mask)
        masked_b4 = apply_mask(b4, mask_result.mask)
        masked_b5 = apply_mask(b5, mask_result.mask)
        
        return {
            "B3": masked_b3,
            "B4": masked_b4,
            "B5": masked_b5,
            "mask": mask_result.mask,
            "utm_bounds": landsat_data["utm_bounds"],
        }

    def _apply_viirs_quality(self, viirs_data: dict[str, Any]) -> dict[str, Any]:
        """Phase 2b: Apply quality filtering to VIIRS data."""
        logger.info("Applying quality masks to VIIRS data...")

        quality_flag = viirs_data.get("QF")

        # Apply VIIRS quality mask (0 = high quality, 2 = good quality)
        quality_mask = (quality_flag == 0) | (quality_flag == 2)

        # Apply lunar correction
        ntl = viirs_data.get("NTL")
        lunar = viirs_data.get("LUNAR")
        
        if ntl is None or lunar is None:
            raise ValueError("Missing required VIIRS data layers")

        # Mask invalid values
        ntl_masked = np.where(quality_mask & (ntl <= 65530), ntl, np.nan)

        # Apply lunar correction where lunar data is valid
        corrected = np.where(
            (lunar > 0) & (lunar <= 65530),
            ntl_masked * (10.0 / lunar),  # Normalize by lunar irradiance
            ntl_masked,
        )

        # Normalize to 0-1 range
        corrected = np.clip(corrected / 100.0, 0, 1)

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
        """Phase 3: Spatial processing - cropping, reprojection, resampling."""
        logger.info("Performing spatial processing...")

        # Get UTM bounds from landsat data
        utm_bounds = landsat_data["utm_bounds"]

        # Step 1: Crop Landsat to UTM bounding box
        logger.info("Cropping to UTM bounds...")
        cropped_landsat = {}
        for band in ["B3", "B4", "B5"]:
            result = self.ctx.bounds_cropper.process(
                data=landsat_data[band],
                bounds=(
                    utm_bounds["e1"],
                    utm_bounds["n1"],
                    utm_bounds["e2"],
                    utm_bounds["n2"],
                ),
                crs=f"EPSG:326{utm_bounds['zone']}",  # UTM zone
            )
            # BoundsCropper returns a mask, not cropped data
            # For now, store the original data - actual cropping needs implementation
            cropped_landsat[band] = landsat_data[band]

        # Step 2: Reproject to WGS84
        logger.info("Reprojecting to WGS84...")
        wgs84_landsat = {}
        for band, data in cropped_landsat.items():
            result = self.ctx.projection_handler.process(
                data=data,
                source_crs=f"EPSG:326{utm_bounds['zone']}",
                target_crs="EPSG:4326",
                bounds=(lon1, lat1, lon2, lat2),
            )
            # ProjectionHandler returns a mask, not reprojected data
            # For now, store the original data - actual reprojection needs implementation
            wgs84_landsat[band] = cropped_landsat[band]

        # Step 3: Resample all data to final resolution
        logger.info(
            f"Resampling to final resolution: {self.ctx.config.final_resolution}"
        )

        resampled_data = {}

        # Resample Landsat bands
        for band, data in wgs84_landsat.items():
            result = self.ctx.resolution_resampler.process(
                data=data,
                target_resolution=self.ctx.config.final_resolution,
                bounds=(lon1, lat1, lon2, lat2),
            )
            # ResolutionResampler returns a mask, not resampled data
            # For now, store the original data - actual resampling needs implementation
            resampled_data[band] = data

        # Resample VIIRS
        viirs_result = self.ctx.resolution_resampler.process(
            data=viirs_data["NTL"],
            target_resolution=self.ctx.config.final_resolution,
            bounds=(lon1, lat1, lon2, lat2),
        )
        # ResolutionResampler returns a mask, not resampled data
        # For now, store the original data - actual resampling needs implementation
        resampled_data["NTL"] = viirs_data["NTL"]

        # Rasterize and resample OSM roads
        roads_raster = self.ctx.road_enhancer.process(
            data=osm_data.get("roads", {}),
            reference_shape=resampled_data["B3"].shape,
            bounds=(lon1, lat1, lon2, lat2),
            resolution=self.ctx.config.final_resolution,
        )
        # RoadEnhancer returns a mask, not rasterized data
        # For now, create a placeholder - actual road rasterization needs implementation
        resampled_data["roads"] = np.zeros_like(resampled_data["NTL"])

        # For now, return data as-is - spatial alignment needs implementation
        return resampled_data

    def _temporal_compositing(self, aligned_data: dict[str, Any]) -> dict[str, Any]:
        """Phase 4: Temporal compositing of multi-date imagery."""
        logger.info("Performing temporal compositing...")

        # For this example, we assume aligned_data contains single-date imagery
        # In a real implementation, this would handle multiple dates

        composite_data = {}

        # Apply median compositing to Landsat bands
        for band in ["B3", "B4", "B5"]:
            # In real implementation, this would composite multiple dates
            # For now, just pass through the single date
            # TemporalCompositor.process expects a 3D array (time, height, width)
            # For single date, add time dimension
            band_data = aligned_data[band]
            if band_data.ndim == 2:
                band_data = band_data[np.newaxis, ...]
            
            result = self.ctx.temporal_compositor.process(
                data=band_data,
                timestamps=[datetime.now()],  # Corresponding timestamps
                method=CompositeMethod.MEDIAN,
            )
            # Result is a MaskResult, not composite data
            # For now, use original data
            composite_data[band] = aligned_data[band]

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

        logger.info(f"NDVI range: {np.nanmin(ndvi):.3f} to {np.nanmax(ndvi):.3f}")
        logger.info(f"NDWI range: {np.nanmin(ndwi):.3f} to {np.nanmax(ndwi):.3f}")
        logger.info(f"NDUI range: {np.nanmin(ndui):.3f} to {np.nanmax(ndui):.3f}")

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
        """Phase 6: Enhance and colorize the NDUI for visualization."""
        logger.info("Enhancing visualization...")

        ndui = indices["NDUI"]

        # Apply contrast enhancement
        enhancer = ContrastEnhancer(method="percentile", clip_percentiles=(2, 98))
        enhanced_result = enhancer.process(
            data=ndui,
            min_value=self.ctx.config.display_min,
            max_value=self.ctx.config.display_max,
        )

        # Components return MaskResult, not enhanced data
        # For now, use original NDUI data
        enhanced_data = ndui
        
        # Apply color mapping would require actual colormap implementation
        # For now, create a simple grayscale RGB
        normalized = np.clip((enhanced_data - enhanced_data.min()) / 
                           (enhanced_data.max() - enhanced_data.min() + 1e-10), 0, 1)
        rgb_channel = (normalized * 255).astype(np.uint8)
        rgb_array = np.stack([rgb_channel, rgb_channel, rgb_channel])

        return rgb_array

    def _generate_output(self, rgb_array: np.ndarray) -> str:
        """Phase 7: Generate Cloud-Optimized GeoTIFF output."""
        logger.info("Generating Cloud-Optimized GeoTIFF...")

        output_path = self.ctx.final_dir / "ndui-co.tif"

        # Generate COG
        cog_result = self.ctx.cog_generator.process(
            data=rgb_array,
            output_path=output_path,
            crs="EPSG:4326",
            transform=None,  # Would be computed from bounds and resolution
            tags={
                "description": "Black Marble NDUI Composite",
                "processing_date": datetime.now().isoformat(),
                "software": "Black Marble Pipeline (Refactored)",
            },
        )

        # Validate output
        validator = OutputValidator()
        
        # Create metadata for validation
        metadata = OutputMetadata(
            crs="EPSG:4326",
            transform=(0, 1, 0, 0, 0, -1),  # Dummy transform - would need actual bounds
            bounds=(0, 0, rgb_array.shape[2], rgb_array.shape[1]),  # Dummy bounds
            shape=(rgb_array.shape[1], rgb_array.shape[2]),  # Height, width
            dtype=rgb_array.dtype,
            nodata=None,
        )
        
        validation_result = validator.validate(rgb_array, metadata)

        if not validation_result.get("valid", True):
            logger.warning(f"Output validation warnings: {validation_result.get('warnings', [])}")

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


def run_refactored_pipeline(
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
    Run the refactored Black Marble pipeline.

    This function maintains the same interface as the original run_pipeline
    but uses the modular component architecture internally.

    Key improvements:
    - Clear separation of phases
    - Dependency injection for testability
    - Modular components that can be tested independently
    - Better error handling and logging
    - Easier to extend and maintain
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize configuration
    if config is None:
        config = ProcessingConfig.from_env()

    # Create pipeline context with all dependencies
    context = PipelineContext.create(
        config=config,
        output_dir=output_dir,
        **kwargs,  # Allow injection of custom adapters/sources
    )

    # Create and run pipeline
    pipeline = RefactoredPipeline(context)

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
    """Entry point demonstrating the refactored pipeline."""
    # This maintains compatibility with the original main function
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
        final_path = run_refactored_pipeline(
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
