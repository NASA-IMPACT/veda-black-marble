"""Target pipeline structure showing the ideal flow of Black Marble processing.

This file demonstrates the clear progression through:
1. Acquire - Get the data we need
2. Prepare - Process and quality check the data
3. Analyze - Calculate indices and composites
4. Enhance - Improve visual quality and add features
5. Export - Generate final outputs

Each phase builds on the previous one in a clear, readable flow.
"""

import logging
import os
from datetime import datetime
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import rasterio
from osgeo import gdal

from . import acquire, analyze, enhance, export, prepare
from .config import PipelineConfig
from .pipeline_async import black_marble_pipeline_async

# Type aliases
ArrayLike = npt.NDArray[Any]
BBox = tuple[float, float, float, float]

logger = logging.getLogger(__name__)


def black_marble_pipeline(
    bbox: BBox,
    date: datetime,
    output_path: str,
    config: PipelineConfig | None = None,
    use_async: bool = True,
    earthdata_token: str | None = None,
) -> dict[str, Any]:
    """Main Black Marble processing pipeline.

    By default, uses async downloads for maximum performance. Set use_async=False
    to use the original sequential download approach.

    Args:
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
        date: Target date for processing
        output_path: Where to save the final output
        config: Optional configuration overrides
        use_async: Use async downloads (default: True)
        earthdata_token: NASA Earthdata token (or set EARTHDATA_TOKEN env var)

    Returns:
        Dictionary with results and metadata
    """
    # Default to async pipeline for better performance
    if use_async:
        logger.info("Using async pipeline with front-loaded downloads")
        return black_marble_pipeline_async(
            bbox=bbox,
            date=date,
            output_path=output_path,
            config=config,
            earthdata_token=earthdata_token,
        )

    # Original sequential pipeline
    logger.info("Using sequential pipeline (legacy mode)")

    # ==================== ACQUIRE PHASE ====================
    # Get all the data we need from various sources

    # 1. Setup for Landsat acquisition
    # Get number of months for compositing from config
    composite_months = getattr(config, "composite_history_months", 12) if config else 12

    # Get cache directory from config if enabled
    cache_dir = (
        getattr(config, "cache_dir", None)
        if config and getattr(config, "cache_downloads", False)
        else None
    )

    # 2. Acquire VIIRS nighttime lights data
    viirs_data = acquire.download_vnp46a2_tiles(
        date=date,
        bbox=bbox,
        output_dir="./data/viirs",
        skip_existing=True,
        cache_dir=cache_dir,
    )

    # 3. Acquire OpenStreetMap road network
    # Note: fetch_road_network uses custom_filter for road types
    road_types = ["motorway", "trunk", "primary", "secondary"]
    custom_filter = f'["highway"~"{"_link|".join(road_types)}"]' if road_types else None

    road_network = acquire.fetch_road_network(
        bbox=bbox, network_type="all", custom_filter=custom_filter
    )

    # ==================== PREPARE PHASE ====================
    # Process raw data into analysis-ready format

    # Ensure temp directory exists for VRTs
    os.makedirs("data/temp", exist_ok=True)

    # 1. Process Landsat data date-by-date with QA masking
    # This ensures QA masks are applied before temporal compositing
    print("Processing Landsat data with date-by-date QA masking")

    # Get scale factors from config
    scale_factor = (
        config.preparation.landsat_scale_factor
        if config and hasattr(config, "preparation")
        else 0.0000275
    )
    add_offset = (
        config.preparation.landsat_add_offset if config and hasattr(config, "preparation") else -0.2
    )
    dilate_pixels = (
        config.preparation.qa_dilation_radius if config and hasattr(config, "preparation") else 3
    )

    # Initialize stacks for each band
    band_stacks = {"B2": [], "B3": [], "B4": [], "B5": []}
    reference_profile = None

    # Track downloaded files for metadata
    all_landsat_files = {"B2": [], "B3": [], "B4": [], "B5": []}

    # Calculate total number of dates to process
    num_dates = composite_months + 1  # Primary date + historical dates

    # Process each date independently
    for date_idx in range(num_dates):
        if date_idx == 0:
            # Primary date
            target_date = date
            print(f"\nProcessing primary date: {target_date.strftime('%Y-%m-%d')}")
        else:
            # Historical date
            from datetime import timedelta

            target_date = date - timedelta(days=date_idx * 30)
            print(f"\nProcessing historical date {date_idx}: {target_date.strftime('%Y-%m-%d')}")

        # Download all bands for this specific date
        try:
            bands_data = acquire.download_all_bands_for_date(
                year=target_date.year,
                month=target_date.month,
                day=target_date.day,
                bbox=bbox,
                bands=[2, 3, 4, 5, "QA_PIXEL"],
                cache_dir=cache_dir,
                ignore_missing=(date_idx > 0),  # Only ignore missing for historical dates
            )

            # Process this date's data with QA masking
            processed_data = prepare.process_landsat_date(
                bands_data=bands_data,
                bbox=bbox,
                apply_qa_mask=True,
                dilate_pixels=dilate_pixels,
                cloud_strategy="moderate",  # Use bit-based moderate strategy
            )

            # Add masked data to stacks and track files
            for band_name in ["B2", "B3", "B4", "B5"]:
                if band_name in processed_data:
                    masked_data, profile = processed_data[band_name]
                    band_stacks[band_name].append(masked_data)

                    # Store first profile as reference
                    if reference_profile is None:
                        reference_profile = profile

                # Track downloaded files for metadata
                if band_name in bands_data:
                    band_files, _, _, _ = bands_data[band_name]
                    all_landsat_files[band_name].extend(band_files)

        except Exception as e:
            if date_idx > 0:  # Historical date
                print(f"  Warning: Could not process date {target_date.strftime('%Y-%m-%d')}: {e}")
                continue
            else:  # Primary date
                raise

    # Create temporal composites from the masked stacks
    landsat_bands = {}
    landsat_profiles = {}

    for band_name, stack in band_stacks.items():
        if stack:
            print(f"\nCreating temporal composite for {band_name} from {len(stack)} dates")

            # Create median composite from pre-masked data
            composite_data = analyze.create_landsat_median_composite(
                stack, scale_factor=scale_factor, add_offset=add_offset
            )

            landsat_bands[band_name] = (composite_data, reference_profile)
            landsat_profiles[band_name] = reference_profile

            print(
                f"  Created composite for {band_name}: shape={composite_data.shape}, "
                f"coverage={(composite_data > 0).sum() / composite_data.size * 100:.1f}%"
            )
        else:
            print(f"  Warning: No data found for {band_name}")

    # Note: QA masking has already been applied during date-by-date processing
    # No need for separate QA compositing or post-processing masking

    # 3. Process VIIRS nighttime lights
    # Load VIIRS gap-filled BRDF-corrected NTL data
    viirs_corrected = None
    viirs_profile = None
    # Use gap-filled NTL if available, fall back to radiance if not
    if viirs_data and viirs_data.get("gap_filled_ntl"):
        # Create VRT mosaic if we have multiple VIIRS tiles
        if len(viirs_data["gap_filled_ntl"]) > 1:
            print(
                f"Creating VRT mosaic for {len(viirs_data['gap_filled_ntl'])} VIIRS gap-filled NTL tiles"
            )
            print("VIIRS tiles in mosaic:")
            for i, tile_path in enumerate(viirs_data["gap_filled_ntl"]):
                print(f"  {i + 1}. {os.path.basename(tile_path)}")
            viirs_ntl_vrt = "data/temp/viirs_gap_filled_ntl_mosaic.vrt"
            os.makedirs("data/temp", exist_ok=True)
            vrt_ds = gdal.BuildVRT(viirs_ntl_vrt, viirs_data["gap_filled_ntl"])
            vrt_ds = None  # Close to write to disk

            with rasterio.open(viirs_ntl_vrt) as src:
                viirs_ntl = src.read(1)
                viirs_profile = src.profile
        else:
            with rasterio.open(viirs_data["gap_filled_ntl"][0]) as src:
                viirs_ntl = src.read(1)
                viirs_profile = src.profile

        # CRITICAL: Crop VIIRS to bounding box like bm.py does
        # This ensures we only process the area of interest, not entire tiles
        print(f"\nCropping VIIRS data to bbox: {bbox}")
        print(f"  VIIRS data shape before crop: {viirs_ntl.shape}")
        viirs_ntl, viirs_transform, _ = prepare.crop_to_bounds(
            data=viirs_ntl, src_transform=viirs_profile["transform"], bounds=bbox
        )
        print(f"  VIIRS data shape after crop: {viirs_ntl.shape}")

        # CRITICAL: Resample to finer resolution like bm.py (scale = 0.00011111111)
        # This fills gaps through bilinear interpolation
        target_resolution = 0.00011111111  # ~12 meters, matching bm.py
        viirs_ntl, viirs_transform = prepare.resample_image(
            data=viirs_ntl,
            src_transform=viirs_transform,
            target_resolution=target_resolution,
            resampling="bilinear",
        )

        # Update profile with new shape and transform
        viirs_profile["transform"] = viirs_transform
        viirs_profile["height"] = viirs_ntl.shape[0]
        viirs_profile["width"] = viirs_ntl.shape[1]

        # Gap-filled NTL is already BRDF-corrected by NASA
        # No need for nodata handling or lunar correction
        # Values are in nW/cm²/sr units

        # The gap-filled product already incorporates quality filtering
        # by using the best available pixels from recent observations
        viirs_corrected = viirs_ntl

        # OLD CODE (commented out):
        # viirs_corrected = prepare.apply_viirs_floor_ceiling(
        #     ntl_data=viirs_scaled,
        #     floor_value=config.analysis.ntl_floor if hasattr(config, 'analysis') else (config.viirs.floor if hasattr(config, 'viirs') else 0.1),
        #     ceiling_value=config.analysis.ntl_ceiling if hasattr(config, 'analysis') else (config.viirs.ceiling if hasattr(config, 'viirs') else 10.0)
        # )

        # Save diagnostic output if requested
        if config and getattr(config, "save_diagnostics", False):
            prepare.save_diagnostic(
                viirs_corrected,
                viirs_profile,
                "data/diagnostics/viirs_final.tif",
                "VIIRS final (gap-filled NTL)",
            )
    elif viirs_data and viirs_data.get("radiance"):
        # Fallback: Use raw radiance if gap-filled NTL not available
        print("Warning: Gap-filled NTL not available, falling back to raw radiance")
        print("Consider updating VIIRS processing to extract gap-filled data")
        # Process raw radiance data (simplified - no lunar correction)
        if len(viirs_data["radiance"]) > 1:
            viirs_radiance_vrt = "data/temp/viirs_radiance_mosaic.vrt"
            os.makedirs("data/temp", exist_ok=True)
            vrt_ds = gdal.BuildVRT(viirs_radiance_vrt, viirs_data["radiance"])
            vrt_ds = None  # Close to write to disk

            with rasterio.open(viirs_radiance_vrt) as src:
                viirs_radiance = src.read(1)
                viirs_profile = src.profile
        else:
            with rasterio.open(viirs_data["radiance"][0]) as src:
                viirs_radiance = src.read(1)
                viirs_profile = src.profile

        # Crop and resample
        viirs_radiance, viirs_transform, _ = prepare.crop_to_bounds(
            data=viirs_radiance, src_transform=viirs_profile["transform"], bounds=bbox
        )
        viirs_radiance, viirs_transform = prepare.resample_image(
            data=viirs_radiance,
            src_transform=viirs_transform,
            target_resolution=0.00011111111,
            resampling="bilinear",
        )
        viirs_profile["transform"] = viirs_transform
        viirs_profile["height"] = viirs_radiance.shape[0]
        viirs_profile["width"] = viirs_radiance.shape[1]

        # Basic nodata handling
        viirs_radiance = np.where(viirs_radiance > 65530, 0, viirs_radiance)
        viirs_corrected = viirs_radiance

    # 4. Reproject everything to common grid
    # Determine appropriate UTM zone for the area
    center_lon = (bbox[0] + bbox[2]) / 2
    center_lat = (bbox[1] + bbox[3]) / 2

    # Calculate UTM zone (simplified - assumes northern hemisphere)
    utm_zone = int((center_lon + 180) / 6) + 1
    hemisphere = "north" if center_lat >= 0 else "south"
    epsg_code = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone

    # Use configured CRS or default to UTM for accurate local analysis
    target_crs = f"EPSG:{epsg_code}"  # Use calculated UTM zone
    target_resolution = 30.0  # 30 meters for Landsat

    # Print projection info clearly
    if config and config.verbose:
        print(
            f"Using projection: {target_crs} (UTM Zone {utm_zone}{'N' if hemisphere == 'north' else 'S'})"
        )

    # Reproject Landsat bands
    landsat_reprojected = {}
    reference_transform = None
    reference_shape = None
    reference_bbox = None

    for band_name, (data, profile) in landsat_bands.items():
        reprojected, new_transform, new_bbox = prepare.reproject_image(
            data=data,
            src_transform=profile["transform"],
            src_crs=profile["crs"],
            dst_crs=target_crs,
            resampling="bilinear",  # Use bilinear for smoother results
        )
        landsat_reprojected[band_name] = reprojected

        # Store reference transform and shape from first band
        if reference_transform is None:
            reference_transform = new_transform
            reference_shape = reprojected.shape
            reference_bbox = new_bbox

    # Reproject VIIRS to target CRS first
    viirs_reprojected = None
    viirs_transform = None
    if viirs_corrected is not None and viirs_profile is not None:
        viirs_reprojected, viirs_transform, _ = prepare.reproject_image(
            data=viirs_corrected,
            src_transform=viirs_profile["transform"],
            src_crs=viirs_profile["crs"],
            dst_crs=target_crs,
            resampling="bilinear",  # Use bilinear like bm.py for smoother results
        )

        # Align VIIRS to Landsat grid for consistent processing
        if reference_transform is not None and reference_shape is not None:
            print(f"VIIRS shape before alignment: {viirs_reprojected.shape}")
            print(f"Reference Landsat shape: {reference_shape}")

            # Save pre-alignment for diagnostics
            if config and getattr(config, "save_diagnostics", False):
                prepare.save_diagnostic(
                    viirs_reprojected,
                    viirs_transform,
                    "data/diagnostics/viirs_before_align.tif",
                    "VIIRS before alignment",
                    crs=target_crs,
                )

            # Use the first Landsat band as reference
            reference_band = landsat_reprojected[list(landsat_reprojected.keys())[0]]

            # Align VIIRS to Landsat grid
            aligned_rasters = prepare.align_rasters(
                rasters=[reference_band, viirs_reprojected],
                transforms=[reference_transform, viirs_transform],
                reference_idx=0,  # Use Landsat as reference
                resampling="bilinear",  # Use bilinear for VIIRS
            )

            # Extract the aligned VIIRS data
            viirs_reprojected = aligned_rasters[1][0]
            viirs_transform = aligned_rasters[1][1]

            print(f"Aligned VIIRS shape: {viirs_reprojected.shape}")
            print(
                f"Aligned VIIRS non-zero pixels: {(viirs_reprojected > 0).sum()} ({(viirs_reprojected > 0).sum() / viirs_reprojected.size * 100:.1f}%)"
            )

            # Save diagnostic output if requested
            if config and getattr(config, "save_diagnostics", False):
                prepare.save_diagnostic(
                    viirs_reprojected,
                    reference_transform,
                    "data/diagnostics/viirs_aligned.tif",
                    "VIIRS aligned to Landsat grid",
                    crs=target_crs,
                )

    # Rasterize roads to match grid
    roads_rasterized = None
    if road_network is not None and reference_shape is not None and reference_transform is not None:
        # Use the new CRS-aware road rasterization
        # This handles meter-based buffering via UTM internally
        buffer_meters = 30.0  # Buffer roads by 30 meters (default in osm.py)

        print(f"Road buffering: {buffer_meters:.1f} meters (applied in UTM)")
        print(f"Rasterizing to: {target_crs}")
        print(f"Road network has {len(road_network.nodes)} nodes, {len(road_network.edges)} edges")

        # Use the new function that buffers and rasterizes with CRS support
        roads_rasterized = acquire.buffer_and_rasterize_roads(
            graph=road_network,
            bbox=bbox,
            transform=reference_transform,
            crs=target_crs,
            shape=(reference_shape[0], reference_shape[1]),
            buffer_meters=buffer_meters,
            hierarchical=True,  # Use hierarchical road classification
        )

        # Print statistics
        road_pixels = np.sum(roads_rasterized > 0)
        print(f"Rasterized {road_pixels} pixels ({road_pixels / roads_rasterized.size * 100:.2f}%)")

        # Save diagnostic output if requested
        if config and getattr(config, "save_diagnostics", False):
            # Convert road raster to float for saving
            road_diagnostic = roads_rasterized.astype(np.float32)
            prepare.save_diagnostic(
                road_diagnostic,
                reference_transform,
                "data/diagnostics/roads_rasterized.tif",
                "Rasterized road network",
                crs=target_crs,
            )

    # ==================== ANALYZE PHASE ====================
    # Calculate indices and create composites

    # 1. Calculate spectral indices
    ndvi = None
    ndwi = None

    if "B5" in landsat_reprojected and "B4" in landsat_reprojected:
        ndvi = analyze.calculate_ndvi(
            nir=landsat_reprojected["B5"],
            red=landsat_reprojected["B4"],
            # Use default epsilon=1e-4 for better stability
        )

        # Save diagnostic output if requested
        if config and getattr(config, "save_diagnostics", False) and ndvi is not None:
            prepare.save_diagnostic(
                ndvi,
                reference_transform,
                "data/diagnostics/ndvi.tif",
                "NDVI (Normalized Difference Vegetation Index)",
                crs=target_crs,
            )

    if "B3" in landsat_reprojected and "B5" in landsat_reprojected:
        ndwi = analyze.calculate_ndwi(
            green=landsat_reprojected["B3"],
            nir=landsat_reprojected["B5"],
            # Use default epsilon=1e-4 for better stability
        )

        # Save diagnostic output if requested
        if config and getattr(config, "save_diagnostics", False) and ndwi is not None:
            prepare.save_diagnostic(
                ndwi,
                reference_transform,
                "data/diagnostics/ndwi.tif",
                "NDWI (Normalized Difference Water Index)",
                crs=target_crs,
            )

    # 2. For now, we only have single date VIIRS, so no temporal compositing
    # In a full implementation, we'd download multiple dates and composite them
    ntl_composite = viirs_reprojected if viirs_reprojected is not None else None

    # Save fine-resolution VIIRS as diagnostic output
    if config and getattr(config, "save_diagnostics", False) and ntl_composite is not None:
        prepare.save_diagnostic(
            ntl_composite,
            viirs_transform,
            "data/diagnostics/viirs_final.tif",
            "Final processed VIIRS at fine resolution",
            crs=target_crs,
        )

    # ENHANCE NTL WITH ROADS (before NDUI calculation)
    # This is the correct place to apply road enhancement - to the NTL data itself,
    # not to the derived NDUI index
    if ntl_composite is not None and roads_rasterized is not None:
        print("\n🛣️  Enhancing NTL with road data")

        # Legacy binary enhancement
        # Major roads (1, 4): factor 1.3
        # Minor roads (2, 3, 5, 6): factor 1.1
        ntl_enhanced = ntl_composite.copy()

        # Apply enhancement based on road type
        major_roads_mask = (roads_rasterized == 1) | (roads_rasterized == 4)
        minor_roads_mask = (
            (roads_rasterized == 2)
            | (roads_rasterized == 3)
            | (roads_rasterized == 5)
            | (roads_rasterized == 6)
        )

        ntl_enhanced = np.where(major_roads_mask, ntl_composite * 1.3, ntl_enhanced)
        ntl_enhanced = np.where(minor_roads_mask, ntl_composite * 1.1, ntl_enhanced)

        # Log statistics
        enhancement = ntl_enhanced - ntl_composite
        enhanced_pixels = np.sum(enhancement > 0)
        if enhanced_pixels > 0:
            print(
                f"  Enhanced {enhanced_pixels} pixels ({enhanced_pixels / ntl_composite.size * 100:.1f}%)"
            )
            print(f"  Major roads (1.3x): {np.sum(major_roads_mask)} pixels")
            print(f"  Minor roads (1.1x): {np.sum(minor_roads_mask)} pixels")
            if enhanced_pixels > 0:
                print(f"  Mean enhancement: {np.mean(enhancement[enhancement > 0]):.3f}")

        # Save enhancement diagnostic
        if config and getattr(config, "save_diagnostics", False):
            prepare.save_diagnostic(
                enhancement,
                reference_transform,
                "data/diagnostics/ntl_road_enhancement.tif",
                "NTL enhancement from roads (binary)",
                crs=target_crs,
            )

        ntl_composite = ntl_enhanced

    # 3. Calculate urban index combining NTL and vegetation
    ndui = None
    if ntl_composite is not None and ndvi is not None:
        # Verify shapes match before calculating NDUI
        if ntl_composite.shape != ndvi.shape:
            print("WARNING: Shape mismatch after alignment!")
            print(f"  NTL shape: {ntl_composite.shape}")
            print(f"  NDVI shape: {ndvi.shape}")
            raise ValueError(
                f"NTL and NDVI shapes still don't match after alignment: "
                f"{ntl_composite.shape} vs {ndvi.shape}. This indicates a problem "
                f"with the spatial alignment step."
            )

        # DIAGNOSTIC: Print data statistics before NDUI
        print("\nBefore NDUI Calculation:")
        print(f"  NTL composite shape: {ntl_composite.shape}")
        print(
            f"  NTL non-zero pixels: {(ntl_composite > 0).sum()} ({(ntl_composite > 0).sum() / ntl_composite.size * 100:.1f}%)"
        )
        print(f"  NTL range: [{np.nanmin(ntl_composite):.3f}, {np.nanmax(ntl_composite):.3f}]")
        print(
            f"  NDVI non-NaN pixels: {(~np.isnan(ndvi)).sum()} ({(~np.isnan(ndvi)).sum() / ndvi.size * 100:.1f}%)"
        )
        print(f"  NDVI range: [{np.nanmin(ndvi):.3f}, {np.nanmax(ndvi):.3f}]")

        ndui = analyze.calculate_ndui(
            ntl=ntl_composite,
            ndvi=ndvi,
            ndwi=ndwi,  # Optional, can be None
            ntl_floor=config.analysis.ntl_floor if config and hasattr(config, "analysis") else 0.1,
            ntl_ceiling=config.analysis.ntl_ceiling
            if config and hasattr(config, "analysis")
            else 10.0,
            ndui_floor=config.analysis.ndui_floor
            if config and hasattr(config, "analysis")
            else 0.02,
            epsilon=1e-6,
        )

        # DIAGNOSTIC: Print NDUI statistics
        print("\nAfter NDUI Calculation:")
        print(
            f"  NDUI non-zero pixels: {(ndui > 0).sum()} ({(ndui > 0).sum() / ndui.size * 100:.1f}%)"
        )
        print(f"  NDUI range: [{np.nanmin(ndui):.3f}, {np.nanmax(ndui):.3f}]")

        # Save diagnostic output if requested
        if config and getattr(config, "save_diagnostics", False) and ndui is not None:
            prepare.save_diagnostic(
                ndui,
                reference_transform,
                "data/diagnostics/ndui_raw.tif",
                "NDUI (calculated with enhanced NTL)",
                crs=target_crs,
            )

    # 4. Skip composite index - we'll use NDUI directly like bm.py
    # The urban mask is essentially where NDUI is high
    urban_mask = None
    if ndui is not None:
        # Create a simple threshold mask for other processing needs
        urban_mask = ndui > 0.3  # Reasonable threshold for urban areas

    # ==================== ENHANCE PHASE ====================
    # Improve visual quality and highlight features

    # 1. Enhance contrast for better visualization
    ntl_enhanced = None
    if ntl_composite is not None:
        # Map config contrast method to valid options
        method_mapping: dict[str, Literal["percentile", "minmax"]] = {
            "linear": "minmax",
            "percentile": "percentile",
        }
        config_method = config.enhancement.contrast_method if config else "linear"
        method: Literal["percentile", "minmax"] = method_mapping.get(config_method, "percentile")

        ntl_enhanced = enhance.enhance_contrast(
            data=ntl_composite,
            method=method,
            percentiles=config.enhancement.contrast_percentiles if config else (2.0, 98.0),
            output_range=config.enhancement.contrast_output_range if config else (0.0, 1.0),
        )

        # Save diagnostic output if requested
        if config and getattr(config, "save_diagnostics", False):
            prepare.save_diagnostic(
                ntl_enhanced,
                reference_transform,
                "data/diagnostics/ntl_contrast_enhanced.tif",
                "NTL after contrast enhancement",
                crs=target_crs,
            )

    # 2. Enhance road networks in the NTL data
    ntl_roads_enhanced = None
    if ntl_enhanced is not None and roads_rasterized is not None:
        ntl_roads_enhanced = enhance.enhance_roads(
            data=ntl_enhanced,
            road_mask=roads_rasterized,
            major_factor=config.enhancement.road_major_factor if config else 1.3,
            minor_factor=config.enhancement.road_minor_factor if config else 1.1,
            custom_factors=config.enhancement.custom_road_factors if config else None,
        )
    else:
        ntl_roads_enhanced = ntl_enhanced

    # 3. Enhance urban features
    ntl_urban_enhanced = None
    if ntl_roads_enhanced is not None and urban_mask is not None:
        ntl_urban_enhanced = enhance.enhance_urban_features(
            ntl_data=ntl_roads_enhanced,
            urban_mask=urban_mask,
            road_mask=roads_rasterized,
            enhancement_factor=1.2,
            road_factor=1.1,
        )
    else:
        ntl_urban_enhanced = ntl_roads_enhanced

    # 4. Apply color mapping for visualization
    # CRITICAL FIX: Use NDUI for colormap like bm.py does, not NTL
    # The old bm.py applies colormap to NDUI (urban development index)
    # which combines nighttime lights with vegetation info for better results
    ntl_colored = None
    if ndui is not None:
        # DIAGNOSTIC: Print data before colormap
        print("\nBefore Colormap Application (USING NDUI):")
        print(f"  Data shape: {ndui.shape}")
        print(f"  Non-zero pixels: {(ndui > 0).sum()} ({(ndui > 0).sum() / ndui.size * 100:.1f}%)")
        print(f"  Data range: [{np.nanmin(ndui):.3f}, {np.nanmax(ndui):.3f}]")
        print(f"  Data type: {ndui.dtype}")

        # Apply inferno colormap to NDUI with appropriate scale range
        # Use 0.3-0.8 range to focus on urban areas (matching bm.py behavior)
        ntl_colored = enhance.apply_inferno_colormap(
            data=ndui,
            scale_range=(0.3, 0.8),  # Focus colormap on urban range for better contrast
        )

        # DIAGNOSTIC: Print data after colormap
        print("\nAfter Colormap Application:")
        print(f"  Output shape: {ntl_colored.shape}")
        print(f"  Output channels: {ntl_colored.shape[0] if len(ntl_colored.shape) == 3 else 1}")
        print(f"  Output range: [{np.nanmin(ntl_colored):.3f}, {np.nanmax(ntl_colored):.3f}]")
        print(f"  Output type: {ntl_colored.dtype}")

        # Save diagnostic output if requested
        if config and getattr(config, "save_diagnostics", False):
            # Save pre-colormap data (now NDUI)
            prepare.save_diagnostic(
                ndui,
                reference_transform,
                "data/diagnostics/ndui_final_before_colormap.tif",
                "Final NDUI before colormap",
                crs=target_crs,
            )

    # 5. Create false color composite for context (optional)
    landsat_viz = None
    if "B5" in landsat_reprojected and "B4" in landsat_reprojected and "B3" in landsat_reprojected:
        landsat_viz = enhance.create_false_color_composite(
            red_channel=landsat_reprojected["B5"],  # NIR in red
            green_channel=landsat_reprojected["B4"],  # Red in green
            blue_channel=landsat_reprojected["B3"],  # Green in blue
            contrast_enhance=True,
            gamma=None,  # Use default gamma
        )

    # ==================== EXPORT PHASE ====================
    # Generate final outputs in various formats

    # 1. Validate output before saving
    validation_results = {"valid": True, "errors": []}
    if ntl_colored is not None and reference_transform is not None:
        validation_results = export.validate_for_cog(
            data=ntl_colored,
            transform=reference_transform,
            crs=target_crs,
            check_tiling=True,
            check_compression=True,
            check_overviews=True,
        )

        if not validation_results["valid"]:
            print(f"Warning: Output validation failed: {validation_results['errors']}")

    # 2. Generate metadata
    metadata = export.create_metadata(
        data_sources={
            "landsat": {
                "files": all_landsat_files["B5"]
                + all_landsat_files["B4"]
                + all_landsat_files["B3"]
                + all_landsat_files["B2"],
                "date_range": f"{date.strftime('%Y-%m-%d')}",
                "sensor": "Landsat 8/9",
                "processing_level": "L2SP",
            },
            "viirs": {
                "files": viirs_data.get("radiance", []) if viirs_data else [],
                "date_range": f"{date.strftime('%Y-%m-%d')}",
                "sensor": "VIIRS",
                "product": "VNP46A2",
            },
            "osm": {
                "date_accessed": datetime.now().strftime("%Y-%m-%d"),
                "road_types": road_types,
            },
        },
        processing_steps=[
            "cloud_masking",
            "spatial_reprojection",
            "index_calculation",
            "urban_enhancement",
            "road_enhancement",
            "contrast_enhancement",
            "colormap_application",
        ],
        indices_calculated=["ndvi", "ndwi", "ndui"] if ndui is not None else [],
        date_processed=datetime.now(),
        bbox=bbox,
        crs=target_crs,
        resolution=target_resolution,
    )

    # 3. Create Cloud-Optimized GeoTIFF
    if ntl_colored is not None and reference_transform is not None:
        # Get compression setting with proper type casting
        compress_setting = config.export.compress if config else "deflate"
        valid_compressions: set[str] = {"deflate", "lzw", "jpeg", "webp", "zstd"}
        if compress_setting not in valid_compressions:
            compress_setting = "deflate"

        export.create_cog(
            data=ntl_colored,
            output_path=output_path,
            transform=reference_transform,
            crs=target_crs,
            metadata=metadata,
            compress=cast(Literal["deflate", "lzw", "jpeg", "webp", "zstd"], compress_setting),
            tiled=config.export.tiled if config else True,
            overviews=config.export.overview_levels if config else [2, 4, 8, 16],
        )

    # 4. Generate auxiliary outputs if requested
    if (
        config
        and config.export.generate_urban_mask
        and urban_mask is not None
        and reference_transform is not None
    ):
        # Export urban mask
        compress_setting = config.export.compress if config else "deflate"
        valid_single_compressions: set[str] = {"deflate", "lzw"}
        if compress_setting not in valid_single_compressions:
            compress_setting = "deflate"

        export.export_single_band(
            data=urban_mask.astype(np.uint8),
            output_path=output_path.replace(".tif", "_urban_mask.tif"),
            transform=reference_transform,
            crs=target_crs,
            band_name="urban_mask",
            compress=cast(Literal["deflate", "lzw"], compress_setting),
        )

    if config and config.export.generate_indices_file and reference_transform is not None:
        # Export indices as multi-band file
        indices_dict = {}
        if ndvi is not None:
            indices_dict["ndvi"] = ndvi
        if ndwi is not None:
            indices_dict["ndwi"] = ndwi
        if ndui is not None:
            indices_dict["ndui"] = ndui

        if indices_dict:
            compress_setting = config.export.compress if config else "deflate"
            valid_multi_compressions: set[str] = {"deflate", "lzw"}
            if compress_setting not in valid_multi_compressions:
                compress_setting = "deflate"

            export.export_multi_band(
                bands=indices_dict,
                output_path=output_path.replace(".tif", "_indices.tif"),
                transform=reference_transform,
                crs=target_crs,
                compress=cast(Literal["deflate", "lzw"], compress_setting),
            )

    # Calculate basic statistics for the results
    statistics: dict[str, float | None] = {
        "urban_area_km2": None,
        "mean_radiance": None,
        "road_length_km": None,
    }

    if urban_mask is not None and reference_transform is not None:
        # Calculate urban area in km²
        pixel_area_m2 = abs(reference_transform.a * reference_transform.e)  # pixel area in m²
        urban_pixels = np.sum(urban_mask)
        statistics["urban_area_km2"] = float((urban_pixels * pixel_area_m2) / 1e6)

    if ntl_composite is not None:
        # Calculate mean radiance
        statistics["mean_radiance"] = float(np.nanmean(ntl_composite))

    # Return results summary
    return {
        "output_path": output_path,
        "bbox": bbox,
        "date": date,
        "metadata": metadata,
        "validation": validation_results,
        "statistics": statistics,
    }


if __name__ == "__main__":
    # Example usage showing the complete flow
    from .config import PipelineConfig

    # Create config using for_bm_parity method
    config = PipelineConfig.for_bm_parity(
        bbox=(-74.05, 40.69, -73.95, 40.79),  # Manhattan
        date=datetime(2023, 9, 15),
        output_path="data/outputs/manhattan_black_marble.tif",
    )

    result = black_marble_pipeline(
        bbox=config.bbox,
        date=config.date,
        output_path=config.output_path,
        config=config,
    )

    print(f"Processing complete: {result['output_path']}")
    print(f"Urban area: {result['statistics']['urban_area_km2']} km²")
