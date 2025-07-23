"""Pipeline for Black Marble processing."""

import logging
import os
from datetime import datetime
from pathlib import Path
import psutil
from typing import Any

import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.transform import array_bounds
from osgeo import gdal

from . import acquire, analyze, enhance, export, prepare
from .config import PipelineConfig

# Create logger for spatial debugging
log = logging.getLogger("bm.spatial")

# Enable debug logging if environment variable is set
if os.environ.get("BM_SPATIAL_DEBUG", "").lower() in ("1", "true", "yes"):
    log.setLevel(logging.DEBUG)

# Type aliases
ArrayLike = npt.NDArray[Any]
BBox = tuple[float, float, float, float]

logger = logging.getLogger(__name__)


def quick_stats(data: ArrayLike, label: str) -> None:
    """Log array statistics for debugging."""
    if data is None:
        logger.debug(f"{label}: None")
        return

    logger.debug(f"{label}:")
    logger.debug(f"  Shape: {data.shape}, Dtype: {data.dtype}")
    logger.debug(f"  Min: {np.nanmin(data):.6f}, Max: {np.nanmax(data):.6f}")
    logger.debug(f"  Non-zero: {(data != 0).sum()} ({(data != 0).sum() / data.size * 100:.1f}%)")
    logger.debug(f"  NaN count: {np.isnan(data).sum()}")


async def black_marble_pipeline(
    bbox: BBox,
    date: datetime,
    output_path: str,
    config: PipelineConfig | None = None,
    earthdata_token: str | None = None,
) -> dict[str, Any]:
    """Main Black Marble processing pipeline using async I/O.

    This pipeline uses async I/O for all data acquisition operations
    to maximize download performance.

    Args:
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
        date: Target date for processing
        output_path: Where to save the final output
        config: Optional configuration overrides
        earthdata_token: NASA Earthdata token (or set EARTHDATA_TOKEN env var)

    Returns:
        Dictionary with results and metadata
    """

    print("🚀 Starting Black Marble pipeline")

    if earthdata_token is None:
        earthdata_token = os.environ.get("EARTHDATA_TOKEN")
        if not earthdata_token:
            logger.warning(
                "No Earthdata token provided. Set EARTHDATA_TOKEN environment variable "
                "or pass earthdata_token parameter. VIIRS downloads will be skipped."
            )

    composite_months = config.acquisition.composite_months if config else 12
    data_dir = "./data"

    print(f"\n📥 Downloading all data for {composite_months} dates...")
    print(f"   Target date: {date.strftime('%Y-%m-%d')}")
    print(f"   Bounding box: {bbox}")

    from .acquire.async_orchestrator import AsyncDownloadOrchestrator, DownloadPlan

    plan = DownloadPlan.from_pipeline_config(bbox, date, config)
    orchestrator = AsyncDownloadOrchestrator(earthdata_token=earthdata_token, show_progress=True)

    download_results = await orchestrator.download_all(plan)

    print("\n✅ Downloads complete!")
    print(download_results.summary())

    if not download_results.viirs_data and earthdata_token:
        logger.error("VIIRS downloads failed - cannot continue without nighttime lights data")
        print("❌ Pipeline failed: VIIRS data is required but download failed")
        print("   Please check your EARTHDATA_TOKEN and network connection")
        viirs_errors = [err for err in download_results.failed_downloads if "VIIRS" in err]
        if viirs_errors:
            print(f"   Error details: {viirs_errors[0]}")
        raise RuntimeError(
            "VIIRS download failed - pipeline cannot continue without nighttime lights data"
        )

    if not download_results.landsat_data:
        logger.warning(
            "No Landsat data downloaded - pipeline will create a minimal output with OSM data only"
        )
        print("⚠️  No Landsat data available - creating minimal output with available data")

    print("\n🔧 Processing downloaded data...")

    os.makedirs(f"{data_dir}/temp", exist_ok=True)

    viirs_files = download_results.viirs_data
    landsat_files_by_date = download_results.landsat_data
    road_network = download_results.osm_data

    # Log OSM road network info
    if road_network is not None:
        log.debug("OSM road network loaded:")
        log.debug(f"  - Nodes: {len(road_network.nodes)}")
        log.debug(f"  - Edges: {len(road_network.edges)}")
        if hasattr(road_network, "graph") and "crs" in road_network.graph:
            log.debug(f"  - CRS: {road_network.graph.get('crs', 'Unknown')}")

        # Try to get bounds from the graph
        try:
            import osmnx as ox

            # Use edges instead of nodes to avoid MultiIndex issues with tuple node IDs
            edges_gdf = ox.graph_to_gdfs(road_network, nodes=False)
            if len(edges_gdf) > 0:
                bounds = edges_gdf.total_bounds
                log.debug(f"  - Bounds (from edges): {bounds}")
                centroid = edges_gdf.unary_union.centroid
                log.debug(f"  - Centroid: ({centroid.x:.6f}, {centroid.y:.6f})")
        except Exception as e:
            log.debug(f"  - Could not extract bounds: {e}")

    viirs_corrected = None
    viirs_profile = None
    viirs_transform = None

    if viirs_files:
        print("\n📊 Processing VIIRS nighttime lights...")
        print(f"VIIRS files keys: {list(viirs_files.keys())}")
        for key, files in viirs_files.items():
            print(f"  {key}: {len(files)} files")

        viirs_ntl_files = viirs_files.get("gap_filled_ntl", [])
        viirs_quality_files = viirs_files.get("quality", [])

        if not viirs_ntl_files:
            print("⚠️  No VIIRS NTL files found - skipping VIIRS processing")
        else:
            if len(viirs_ntl_files) > 1:
                print(f"Creating VRT mosaic for {len(viirs_ntl_files)} VIIRS tiles")
                viirs_ntl_vrt = f"{data_dir}/temp/viirs_gap_filled_ntl_mosaic.vrt"
                os.makedirs(f"{data_dir}/temp", exist_ok=True)
                # VIIRS uses fill_value of 65530 based on config
                vrt_options = gdal.BuildVRTOptions(
                    separate=False,  # Mosaic tiles together
                    resolution='highest'  # Use highest resolution from input files
                )
                vrt_ds = gdal.BuildVRT(viirs_ntl_vrt, [str(f) for f in viirs_ntl_files], options=vrt_options)
                vrt_ds = None  # Close to write to disk

                # Save VRT to diagnostics if requested
                if config and getattr(config, "save_diagnostics", False):
                    os.makedirs(f"{data_dir}/diagnostics/vrts", exist_ok=True)
                    import shutil

                    shutil.copy(
                        viirs_ntl_vrt,
                        f"{data_dir}/diagnostics/vrts/viirs_gap_filled_ntl_mosaic.vrt",
                    )
                    print(f"  Saved VRT to: data/diagnostics/vrts/viirs_gap_filled_ntl_mosaic.vrt")

                with rasterio.open(viirs_ntl_vrt) as src:
                    viirs_ntl = src.read(1)
                    viirs_profile = src.profile
            else:
                with rasterio.open(str(viirs_ntl_files[0])) as src:
                    viirs_ntl = src.read(1)
                    viirs_profile = src.profile

            logger.debug(f"Cropping VIIRS data to bbox: {bbox}")
            logger.debug(f"Original VIIRS shape: {viirs_ntl.shape}")
            logger.debug(f"Original VIIRS CRS: {viirs_profile['crs']}")
            orig_bounds = rasterio.transform.array_bounds(
                viirs_ntl.shape[0], viirs_ntl.shape[1], viirs_profile["transform"]
            )
            logger.debug(f"Original VIIRS bounds: {orig_bounds}")

            viirs_ntl, viirs_transform, viirs_bounds = prepare.crop_to_bounds(
                data=viirs_ntl, src_transform=viirs_profile["transform"], bounds=bbox
            )

            logger.debug(f"Cropped VIIRS shape: {viirs_ntl.shape}")
            logger.debug(f"Cropped VIIRS bounds: {viirs_bounds}")
            logger.debug(
                f"Non-zero pixels after crop: {(viirs_ntl > 0).sum()} ({(viirs_ntl > 0).sum() / viirs_ntl.size * 100:.1f}%)"
            )

            viirs_profile["transform"] = viirs_transform
            viirs_profile["height"] = viirs_ntl.shape[0]
            viirs_profile["width"] = viirs_ntl.shape[1]

            if config and getattr(config, "save_diagnostics", False):
                prepare.save_diagnostic(
                    viirs_ntl,
                    viirs_profile,
                    "data/diagnostics/viirs_resampled.tif",
                    "VIIRS resampled to target resolution",
                )

            # The gap-filled product already incorporates quality filtering
            # by using the best available pixels from recent observations
            viirs_corrected = viirs_ntl

        if config and getattr(config, "save_diagnostics", False) and viirs_profile is not None:
            if viirs_corrected is not None:
                prepare.save_diagnostic(
                    viirs_corrected,
                    viirs_profile,
                    "data/diagnostics/viirs_final.tif",
                    "VIIRS final (gap-filled NTL)",
                )

    if viirs_corrected is None:
        print("⚠️  No VIIRS data - using Landsat as reference")

    from .crs import get_processing_crs
    
    processing_crs = get_processing_crs(bbox)
    target_resolution_landsat = 30.0  # 30 meters for all projections

    print(
        f"Using locally-optimized Albers Equal Area projection "
        f"with {target_resolution_landsat}m resolution"
    )

    log.debug(f"Processing CRS determined from bbox:")
    log.debug(f"  - BBox: {bbox}")
    log.debug(f"  - CRS WKT: {processing_crs.to_wkt()}")

    print("\n📐 Computing target grid for spatial alignment...")
    from .crs import create_processing_grid
    
    processing_crs, reference_transform, reference_shape = create_processing_grid(bbox, target_resolution_landsat)
    
    bounds = array_bounds(reference_shape[0], reference_shape[1], reference_transform)
    
    print(f"  Target grid: {reference_shape} pixels")
    print(f"  Bounds in processing CRS: ({bounds[0]:.1f}, {bounds[1]:.1f}, {bounds[2]:.1f}, {bounds[3]:.1f})")
    print(f"  Pixel-aligned to {target_resolution_landsat}m grid")

    print("\n🛰️ Processing Landsat data...")

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

    band_stacks = {"B3": [], "B4": [], "B5": []}
    band_dates = {"B3": [], "B4": [], "B5": []}

    index_stacks = {"ndvi": [], "ndwi": []}
    index_dates = []

    ndvi_composite = None
    ndwi_composite = None
    reference_profile = None

    all_landsat_files = {"B3": [], "B4": [], "B5": []}

    tiles_with_data = set()

    process = psutil.Process()
    monitor_memory = True

    print(f"\n💾 Processing {len(landsat_files_by_date)} Landsat dates")

    # Debug configuration
    if config:
        print(f"  Mask low denominator: {getattr(config.analysis, 'mask_low_denom', False)}")

    # Debug: Print structure of landsat_files_by_date
    if not landsat_files_by_date:
        print("WARNING: landsat_files_by_date is empty!")
    else:
        sample_date = next(iter(landsat_files_by_date.keys()))
        print(f"Sample Landsat data structure for {sample_date}:")
        print(f"  Keys: {list(landsat_files_by_date[sample_date].keys())}")
        for band, paths in landsat_files_by_date[sample_date].items():
            print(f"  {band}: {len(paths)} files")

    print("\n📊 Analyzing tile coverage across dates:")
    all_tiles_by_date = {}
    for date, date_files in landsat_files_by_date.items():
        date_tiles = set()
        for band, paths in date_files.items():
            for path in paths:
                filename = Path(path).stem
                parts = filename.split("_")
                if len(parts) >= 3:
                    pathrow = parts[2]
                    if len(pathrow) == 6 and pathrow.isdigit():
                        date_tiles.add(pathrow)
        all_tiles_by_date[date] = date_tiles
    all_possible_tiles = set()
    for tiles in all_tiles_by_date.values():
        all_possible_tiles.update(tiles)

    print(f"  Total unique tiles across all dates: {sorted(all_possible_tiles)}")
    for date, tiles in sorted(all_tiles_by_date.items()):
        missing = all_possible_tiles - tiles
        if missing:
            print(f"  {date.strftime('%Y-%m-%d')}: Missing tiles {sorted(missing)}")

    # Process each date's Landsat data
    for idx, target_date in enumerate(sorted(landsat_files_by_date.keys())):
        # Log memory usage
        print(
            f"\nProcessing Landsat for {target_date.strftime('%Y-%m-%d')} "
            f"(date {idx + 1}/{len(landsat_files_by_date)})"
        )
        if monitor_memory and process is not None:
            memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
            print(f"  Current memory usage: {memory_gb:.1f} GB")

        date_files = landsat_files_by_date[target_date]

        # Skip if we don't have all bands with actual files
        REQUIRED_BANDS = ["B3", "B4", "B5", "QA_PIXEL"]
        missing_bands = [b for b in REQUIRED_BANDS if b not in date_files or not date_files.get(b)]
        if missing_bands:
            print(f"  Skipping - missing bands: {missing_bands}")
            print(
                f"  Available: {[b for b in REQUIRED_BANDS if b in date_files and date_files.get(b)]}"
            )
            continue

        # Track which tiles this date has
        date_tiles = set()
        for band, paths in date_files.items():
            for path in paths:
                # Extract path/row from filename
                filename = Path(path).stem
                parts = filename.split("_")
                if len(parts) >= 3:
                    pathrow = parts[2]
                    if len(pathrow) == 6 and pathrow.isdigit():
                        date_tiles.add(pathrow)
                        tiles_with_data.add(pathrow)

        if date_tiles:
            print(f"  Tiles for this date: {sorted(date_tiles)}")

        logger.info(
            f"Processing {target_date.strftime('%Y-%m-%d')} with tiles: {sorted(date_tiles)}"
        )
        for band, paths in date_files.items():
            if paths:
                logger.info(f"  {band}: {len(paths)} files")
                for path in paths[:3]:
                    logger.debug(f"    - {Path(path).name}")
                if len(paths) > 3:
                    logger.debug(f"    ... and {len(paths) - 3} more")

        # Convert paths to tuple format expected by process_landsat_date
        # Format is: (band_files, qa_pixel_files, qa_radsat_files, qa_aerosol_files)
        bands_data = {}
        for band, paths in date_files.items():
            if paths:
                # All bands use the same format - files go in first position
                bands_data[band] = ([str(p) for p in paths], [], [], [])

        # Process this date's data
        # Use less aggressive masking for compositing if configured
        composite_strategy = (
            getattr(config.preparation, "qa_composite_strategy", None) if config else None
        )
        composite_dilation = (
            getattr(config.preparation, "qa_composite_dilation", None) if config else None
        )

        actual_strategy = (
            composite_strategy or getattr(config.preparation, "qa_cloud_strategy", "moderate")
            if config
            else "moderate"
        )
        actual_dilation = composite_dilation if composite_dilation is not None else dilate_pixels

        processed_data = prepare.process_landsat_date(
            bands_data=bands_data,
            bbox=bbox,
            apply_qa_mask=True,
            dilate_pixels=actual_dilation,
            cloud_strategy=actual_strategy,
            save_diagnostics=config and getattr(config, "save_diagnostics", False),
            date_str=target_date.strftime('%Y-%m-%d'),
        )
        
        # DEBUG: Log the transform from processed data
        if processed_data and "B5" in processed_data:
            _, profile = processed_data["B5"]
            print(f"  📍 Date {target_date.strftime('%Y-%m-%d')} transform:")
            print(f"     CRS: {profile['crs']}")
            print(f"     Transform.c (x-origin): {profile['transform'].c}")
            print(f"     Transform.f (y-origin): {profile['transform'].f}")
            print(f"     Reference transform.c: {reference_transform.c}")
            print(f"     Reference transform.f: {reference_transform.f}")
            if abs(profile['transform'].c - reference_transform.c) > 0.1:
                print(f"     ⚠️  X-origin differs by {profile['transform'].c - reference_transform.c:.1f}m")

        # Calculate indices for this date BEFORE temporal compositing
        print(f"  Calculating indices for {target_date.strftime('%Y-%m-%d')}")
        if "B5" in processed_data and "B4" in processed_data and "B3" in processed_data:
                # Get the masked data
                b3_data, _ = processed_data["B3"]
                b4_data, _ = processed_data["B4"]
                b5_data, profile = processed_data["B5"]
                
                # DEBUG: Log shapes
                print(f"  📐 Band shapes - B3: {b3_data.shape}, B4: {b4_data.shape}, B5: {b5_data.shape}")
                print(f"  📐 Expected reference shape: {reference_shape}")

                if reference_profile is None:
                    reference_profile = profile
                    logger.info(
                        f"Set reference profile from B5: shape={profile['height']}x{profile['width']}, crs={profile['crs']}"
                    )
                    # DEBUG: Check if reference CRS matches processing CRS
                    print(f"  ⚠️  Reference profile CRS: {profile['crs']}")
                    print(f"  ⚠️  Processing CRS: {processing_crs}")
                    if str(profile['crs']) != str(processing_crs):
                        print(f"  🚨 WARNING: Reference CRS does not match processing CRS!")
                elif (
                    profile["width"] * profile["height"]
                    > reference_profile["width"] * reference_profile["height"]
                ):
                    logger.info(
                        f"Updating reference profile to larger area: {reference_profile['height']}x{reference_profile['width']} -> {profile['height']}x{profile['width']}"
                    )
                    reference_profile = profile

                # Calculate NDVI for this date
                ndvi_date = analyze.calculate_ndvi(
                    nir=b5_data, red=b4_data, mask_low_denom=config.analysis.mask_low_denom
                ).astype(np.float32)  # Control memory
                
                # Clamp negative NDVI values to 0 for NDUI calculation
                # Negative NDVI indicates water/bare soil/artificial surfaces
                ndvi_date = np.maximum(ndvi_date, 0.0)

                # Calculate NDWI for this date
                ndwi_date = analyze.calculate_ndwi(
                    green=b3_data, nir=b5_data, mask_low_denom=config.analysis.mask_low_denom
                ).astype(np.float32)  # Control memory
                
                # Add to index stacks
                index_stacks["ndvi"].append(ndvi_date)
                index_stacks["ndwi"].append(ndwi_date)
                index_dates.append(target_date)

                # Optional: Delete bands to save memory
                if config.memory.delete_bands_after_indices:
                    del b3_data, b4_data, b5_data
                    import gc

                    gc.collect()

                # Track files for metadata
                for band_name in ["B3", "B4", "B5"]:
                    if band_name in bands_data:
                        band_files, _, _, _ = bands_data[band_name]
                        all_landsat_files[band_name].extend(band_files)
        else:
            print(f"  Skipping index calculation - missing required bands")

        # Log memory after processing date
        if monitor_memory and process is not None:
            memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
            total_arrays = sum(len(stack) for stack in index_stacks.values())
            print(
                f"  Memory after date: {memory_gb:.1f} GB (index arrays in memory: {total_arrays})"
            )

    # Check which tiles we got vs expected
    if tiles_with_data:
        from .acquire.wrs_utils import get_wrs_tiles_for_bbox_tuple

        expected_tiles = get_wrs_tiles_for_bbox_tuple(bbox)
        expected_pathrows = {f"{t['path']:03d}{t['row']:03d}" for t in expected_tiles}

        print(f"\n📊 Tile coverage summary:")
        print(f"  Expected tiles for bbox: {sorted(expected_pathrows)}")
        print(f"  Tiles with data: {sorted(tiles_with_data)}")

        missing_tiles = expected_pathrows - tiles_with_data
        if missing_tiles:
            print(f"  ⚠️  MISSING TILES: {sorted(missing_tiles)}")
            for pr in sorted(missing_tiles):
                path = int(pr[:3])
                row = int(pr[3:])
                print(f"     - Path {path}, Row {row}")
        else:
            print(f"  ✅ All expected tiles have data!")

    # Create temporal composites
    print("\n⏳ Creating Landsat temporal composites...")

    # Debug what we have
    print(f"  Index stacks: NDVI={len(index_stacks['ndvi'])}, NDWI={len(index_stacks['ndwi'])}")
    print(
        f"  Band stacks: B3={len(band_stacks['B3'])}, B4={len(band_stacks['B4'])}, B5={len(band_stacks['B5'])}"
    )

    if any(index_stacks.values()):
        # Create index composites (indices already calculated per-date)
        print("  Creating index composites from per-date indices")

        if index_stacks["ndvi"]:
            print(f"  Creating NDVI composite from {len(index_stacks['ndvi'])} dates")
            ndvi_composite = analyze.create_index_temporal_composite(
                index_stacks["ndvi"],
                method=config.analysis.ndvi_composite_method,
                percentile=config.analysis.ndvi_composite_percentile,
                min_valid_observations=config.analysis.min_valid_observations,
            )
            print(
                f"    NDVI composite range: [{np.nanmin(ndvi_composite):.3f}, {np.nanmax(ndvi_composite):.3f}]"
            )
            print(
                f"    Valid pixels: {(~np.isnan(ndvi_composite)).sum() / ndvi_composite.size * 100:.1f}%"
            )
        else:
            ndvi_composite = None

        if index_stacks["ndwi"]:
            print(f"  Creating NDWI composite from {len(index_stacks['ndwi'])} dates")
            ndwi_composite = analyze.create_index_temporal_composite(
                index_stacks["ndwi"],
                method=config.analysis.ndwi_composite_method,
                percentile=config.analysis.ndwi_composite_percentile,
                min_valid_observations=config.analysis.min_valid_observations,
            )
            print(
                f"    NDWI composite range (clamped to [0,1]): [{np.nanmin(ndwi_composite):.3f}, {np.nanmax(ndwi_composite):.3f}]"
            )
            print(
                f"    Valid pixels: {(~np.isnan(ndwi_composite)).sum() / ndwi_composite.size * 100:.1f}%"
            )
        else:
            ndwi_composite = None


        # Skip saving original CRS versions - we'll save the reprojected versions later

        # Validate that we have a reference profile for reprojection
        if reference_profile is None and (ndvi_composite is not None or ndwi_composite is not None):
            print("⚠️  WARNING: No reference profile available for index reprojection")
            print("    This can happen if all Landsat scenes were filtered out")
            # Try to get profile from the first index array
            if index_stacks["ndvi"] and hasattr(index_stacks["ndvi"][0], "shape"):
                # We need to construct a minimal profile for reprojection
                # This is a fallback - ideally we should have saved profiles with indices
                print("    Creating fallback reference profile from index shape")

        landsat_reprojected = {}


    else:
        print("⚠️  No valid Landsat data found - skipping index calculations")
        landsat_reprojected = {}

    # Reproject Landsat bands to target CRS - common for both VRT and in-memory
    # Note: reference_transform and reference_shape are already set from create_processing_grid()
    reference_bbox = None

    landsat_bands = locals().get("landsat_bands", {})
    if landsat_bands:
        print("\n🗺️  Reprojecting Landsat to target CRS...")
        landsat_reprojected = {}

        for band_name, (data, profile) in landsat_bands.items():
            reprojected, new_transform, new_bbox = prepare.reproject_image(
                data=data,
                src_transform=profile["transform"],
                src_crs=profile["crs"],
                dst_crs=processing_crs,
                dst_transform=reference_transform,
                dst_shape=reference_shape,
                resampling="bilinear",  # Use bilinear for smoother results
            )
            landsat_reprojected[band_name] = reprojected

            # Store reference bbox from first band (but keep using pre-ordained transform/shape)
            if reference_bbox is None:
                reference_bbox = new_bbox
    else:
        # No Landsat data available
        landsat_reprojected = {}
        # This is expected in index-first mode

    # Reproject VIIRS to target CRS first (like sync pipeline)
    viirs_reprojected = None
    viirs_transform = None
    if viirs_corrected is not None and viirs_profile is not None:
        logger.debug(f"VIIRS Reprojection:")
        logger.debug(f"  Original VIIRS shape: {viirs_corrected.shape}")
        logger.debug(f"  Original VIIRS CRS: {viirs_profile['crs']}")
        logger.debug(
            f"  Original VIIRS bounds: {rasterio.transform.array_bounds(viirs_corrected.shape[0], viirs_corrected.shape[1], viirs_profile['transform'])}"
        )
        logger.debug(f"  Target CRS: {processing_crs}")

        viirs_reprojected, viirs_transform, viirs_bounds = prepare.reproject_image(
            data=viirs_corrected,
            src_transform=viirs_profile["transform"],
            src_crs=viirs_profile["crs"],
            dst_crs=processing_crs,
            dst_transform=reference_transform,
            dst_shape=reference_shape,
            resampling="bilinear",  # Use bilinear like sync pipeline
        )

        logger.debug(f"  Reprojected VIIRS shape: {viirs_reprojected.shape}")
        logger.debug(f"  Reprojected VIIRS bounds: {viirs_bounds}")

        # We should NEVER use VIIRS as reference - we have a pre-ordained grid
        # from create_processing_grid()
        if reference_transform is None:
            raise RuntimeError(
                "Reference transform is None! This should never happen as we create "
                "the processing grid at the start of the pipeline."
            )

        # Align VIIRS to Landsat grid for consistent processing
        if reference_transform is not None and reference_shape is not None and landsat_reprojected:
            # We have Landsat data to align to
            print(f"VIIRS shape before alignment: {viirs_reprojected.shape}")
            print(f"Reference Landsat shape: {reference_shape}")

            # Debug stats before alignment
            quick_stats(viirs_reprojected, "VIIRS before alignment")

            # Use the first Landsat band as reference
            reference_band = landsat_reprojected[list(landsat_reprojected.keys())[0]]
            quick_stats(reference_band, "Landsat reference band")

            # Align VIIRS to Landsat grid
            logger.debug(f"Aligning VIIRS to Landsat")
            logger.debug(f"  Reference transform: {reference_transform}")
            logger.debug(f"  VIIRS transform: {viirs_transform}")
            ref_bounds = rasterio.transform.array_bounds(
                reference_shape[0], reference_shape[1], reference_transform
            )
            viirs_bounds = rasterio.transform.array_bounds(
                viirs_reprojected.shape[0], viirs_reprojected.shape[1], viirs_transform
            )
            logger.debug(f"  Reference bounds: {ref_bounds}")
            logger.debug(f"  VIIRS bounds: {viirs_bounds}")

            # Check coverage
            x_coverage = min(viirs_bounds[2], ref_bounds[2]) - max(viirs_bounds[0], ref_bounds[0])
            y_coverage = min(viirs_bounds[3], ref_bounds[3]) - max(viirs_bounds[1], ref_bounds[1])
            x_total = ref_bounds[2] - ref_bounds[0]
            y_total = ref_bounds[3] - ref_bounds[1]

            logger.debug(f"  Coverage Analysis:")
            logger.debug(
                f"    X-axis coverage: {x_coverage/1000:.1f} km of {x_total/1000:.1f} km ({x_coverage/x_total*100:.1f}%)"
            )
            logger.debug(
                f"    Y-axis coverage: {y_coverage/1000:.1f} km of {y_total/1000:.1f} km ({y_coverage/y_total*100:.1f}%)"
            )

            if x_coverage < x_total * 0.95:
                logger.warning(
                    f"VIIRS is missing {(x_total-x_coverage)/1000:.1f} km on the east side!"
                )
            if y_coverage < y_total * 0.95:
                logger.warning(f"VIIRS has incomplete Y coverage!")

            aligned_rasters = prepare.align_rasters(
                rasters=[reference_band, viirs_reprojected],
                transforms=[reference_transform, viirs_transform],
                reference_idx=0,  # Use Landsat as reference
                resampling="bilinear",  # Use bilinear for VIIRS
            )

            # Extract the aligned VIIRS data
            viirs_reprojected = aligned_rasters[1][0]
            viirs_transform = aligned_rasters[1][1]

            # Debug stats after alignment
            quick_stats(viirs_reprojected, "VIIRS after alignment")

            print(f"Aligned VIIRS shape: {viirs_reprojected.shape}")
            nonzero_count = (viirs_reprojected > 0).sum()
            nonzero_pct = nonzero_count / viirs_reprojected.size * 100
            print(f"Aligned VIIRS non-zero pixels: {nonzero_count} " f"({nonzero_pct:.1f}%)")

            # Critical assertion - stop if alignment produced all zeros
            if nonzero_pct < 10.0:  # Less than 10% non-zero is suspicious
                logger.error(
                    f"CRITICAL: VIIRS alignment produced only {nonzero_pct:.1f}% non-zero pixels!"
                )
                logger.error("This likely indicates incorrect tile selection or alignment failure.")
                raise RuntimeError(
                    f"VIIRS alignment failed - only {nonzero_pct:.1f}% non-zero pixels. "
                    "Check tile selection and spatial footprints."
                )
        else:
            # No reprojected Landsat bands to align to
            # This is expected in indices-first mode where we calculate NDVI/NDWI before reprojection
            if not landsat_reprojected:
                logger.debug("No reprojected Landsat bands for VIIRS alignment (expected in indices-first mode)")

    # Rasterize roads to match grid (fractional only)
    roads_fractional = None
    if road_network is not None and reference_shape is not None and reference_transform is not None:
        # Use the new CRS-aware road rasterization (like sync pipeline)
        # Use buffer sizes from config or default variable sizes
        from blackmarble.acquire.osm_fractional import DEFAULT_BUFFER_BY_TYPE

        if config and config.acquisition.osm_buffer_meters is not None:
            buffer_meters = config.acquisition.osm_buffer_meters
        else:
            buffer_meters = DEFAULT_BUFFER_BY_TYPE  # Variable buffer by road type

        if isinstance(buffer_meters, dict):
            print(
                f"Road buffering: Variable by type (motorway: {buffer_meters.get('motorway', 'N/A')}m, residential: {buffer_meters.get('residential', 'N/A')}m)"
            )
        else:
            print(f"Road buffering: Uniform {buffer_meters}m")
        print(f"Rasterizing to: {processing_crs}")
        print(f"Road network has {len(road_network.nodes)} nodes, {len(road_network.edges)} edges")
        print(f"Reference transform: {reference_transform}")
        print(f"Reference shape: {reference_shape}")

        # Use fractional road rasterization
        print("  Using fractional road rasterization (5x5 sub-pixels)")
        roads_fractional = acquire.buffer_and_rasterize_roads_fractional(
            graph=road_network,
            bbox=bbox,
            transform=reference_transform,
            crs=processing_crs,
            shape=(reference_shape[0], reference_shape[1]),
            buffer_meters=buffer_meters,
            sub_pixels=config.analysis.osm_sub_pixels if config else 5,
            hierarchical=False,  # Not supported for fractional
        )

        # Print statistics
        road_pixels = np.sum(roads_fractional > 0)
        print(
            f"Rasterized {road_pixels} pixels with roads "
            f"({road_pixels / roads_fractional.size * 100:.2f}%)"
        )
        if road_pixels > 0:
            print(
                f"Fractional coverage range: "
                f"{roads_fractional[roads_fractional > 0].min():.3f} to "
                f"{roads_fractional.max():.3f}"
            )
            print(
                f"Mean fractional coverage: {roads_fractional[roads_fractional > 0].mean():.3f}"
            )
        else:
            print("⚠️  WARNING: No roads were rasterized! Check buffer size and CRS.")

        # Save diagnostic
        if config and getattr(config, "save_diagnostics", False):
            prepare.save_diagnostic(
                roads_fractional,
                reference_transform,
                "data/diagnostics/roads_fractional.tif",
                "Fractional road coverage (0.0-1.0)",
                crs=processing_crs,
            )
    else:
        # No road network data
        print("  No road network data available")
        if reference_shape is not None:
            roads_fractional = np.zeros(reference_shape, dtype="float32")
        else:
            # This should never happen since we create the processing grid at the start
            raise RuntimeError(
                "No reference shape available for road rasterization. "
                "This should never happen as we create the processing grid at the start of the pipeline."
            )

    # Save diagnostic composite bands if requested
    if config and getattr(config, "save_diagnostics", False) and landsat_reprojected:
        diagnostic_bands = getattr(config, "diagnostic_bands", ["B3", "B4", "B5"])
        print(f"\n💾 Saving diagnostic composite bands: {diagnostic_bands}")

        for band_name in diagnostic_bands:
            if band_name in landsat_reprojected:
                band_data = landsat_reprojected[band_name]
                output_path = f"data/diagnostics/composite_{band_name}.tif"

                prepare.save_diagnostic(
                    band_data,
                    reference_transform,
                    output_path,
                    f"Landsat {band_name} Temporal Composite",
                    crs=processing_crs,
                )

                # Log statistics
                valid_pixels = ~np.isnan(band_data)
                if valid_pixels.any():
                    print(
                        f"  {band_name}: min={np.nanmin(band_data):.3f}, max={np.nanmax(band_data):.3f}, "
                        f"coverage={valid_pixels.sum() / band_data.size * 100:.1f}%"
                    )
                else:
                    print(f"  {band_name}: No valid pixels")

    print("\n📈 Calculating indices...")

    ndvi = None
    ndwi = None

    if ndvi_composite is not None and reference_profile is not None:
        print("  Reprojecting pre-calculated NDVI composite...")
        print(f"    Using reference grid: {reference_shape}")
        ndvi, ndvi_transform, _ = prepare.reproject_image(
            data=ndvi_composite,
            src_transform=reference_profile["transform"],
            src_crs=reference_profile["crs"],
            dst_crs=processing_crs,
            dst_transform=reference_transform,
            dst_shape=reference_shape,
            resampling="bilinear",
        )
        print(f"    Reprojected NDVI shape: {ndvi.shape}")
        
        # Save NDVI diagnostic
        if config and getattr(config, "save_diagnostics", False) and ndvi is not None:
            prepare.save_diagnostic(
                ndvi,
                reference_transform,
                "data/diagnostics/ndvi.tif",
                f"NDVI {config.analysis.ndvi_composite_percentile}th percentile composite",
                crs=processing_crs,
            )

    if ndwi_composite is not None and reference_profile is not None:
        print("  Reprojecting pre-calculated NDWI composite...")
        print(f"    Using reference grid: {reference_shape}")
        ndwi, ndwi_transform, _ = prepare.reproject_image(
            data=ndwi_composite,
            src_transform=reference_profile["transform"],
            src_crs=reference_profile["crs"],
            dst_crs=processing_crs,
            dst_transform=reference_transform,
            dst_shape=reference_shape,
            resampling="bilinear",
        )
        print(f"    Reprojected NDWI shape: {ndwi.shape}")

        # Save diagnostic output if requested
        if config and getattr(config, "save_diagnostics", False) and ndwi is not None:
            prepare.save_diagnostic(
                ndwi,
                reference_transform,
                "data/diagnostics/ndwi.tif",
                f"NDWI temporal composite ({config.analysis.ndwi_composite_method})",
                crs=processing_crs,
            )

    ntl_composite = viirs_reprojected if viirs_reprojected is not None else None

    if config and getattr(config, "save_diagnostics", False) and ntl_composite is not None:
        prepare.save_diagnostic(
            ntl_composite,
            reference_transform,
            "data/diagnostics/ntl_raw.tif",
            "Raw NTL values (before enhancement)",
            crs=processing_crs,
        )

    if ntl_composite is not None:
        # NTL enhancement now always uses urban field approach

        if (
            config
            and config.analysis.osm_fractional
            and roads_fractional is not None
        ):
            # URBAN FIELD ENHANCEMENT with fractional roads
            print("\n🌃 Enhancing NTL with urban fields (fractional roads)")

            # Compute multi-scale urban fields
            urban_field = analyze.compute_urban_fields(
                road_fraction=roads_fractional,
                sigmas=config.analysis.urban_field_sigmas,
                weights=config.analysis.urban_field_weights,
            )

            print("\n🏙️ URBAN FIELD STATS:")
            print(f"  - Max urban field: {urban_field.max():.3f}")
            print(f"  - Mean urban field (where > 0): {urban_field[urban_field > 0].mean():.3f}")
            print(
                f"  - Urban pixels: {np.sum(urban_field > 0)} "
                f"({np.sum(urban_field > 0) / urban_field.size * 100:.1f}%)"
            )

            # Save urban field diagnostic
            if getattr(config, "save_diagnostics", False):
                prepare.save_diagnostic(
                    urban_field,
                    reference_transform,
                    "data/diagnostics/urban_field.tif",
                    "Multi-scale urban density field",
                    crs=processing_crs,
                )

            # Enhance NTL
            ntl_enhanced = analyze.enhance_ntl_with_urban_field(
                ntl=ntl_composite,
                urban_field=urban_field,
                enhancement_mode=config.analysis.ntl_enhancement_mode,
                multiplicative_factor=config.analysis.enhancement_multiplicative_factor,
                additive_factor=config.analysis.enhancement_additive_factor,
                hybrid_expected_scale=config.analysis.enhancement_hybrid_expected_scale,
                hybrid_expected_sqrt_scale=config.analysis.enhancement_hybrid_expected_sqrt_scale,
                hybrid_application_factor=config.analysis.enhancement_hybrid_application_factor,
            )

            # Calculate enhancement
            enhancement = ntl_enhanced - ntl_composite

            # Save before/after comparison
            if getattr(config, "save_diagnostics", False):
                prepare.save_diagnostic(
                    enhancement,
                    reference_transform,
                    "data/diagnostics/ntl_enhancement.tif",
                    "NTL enhancement from urban fields",
                    crs=processing_crs,
                )

            ntl_composite = ntl_enhanced
            enhanced_pixels = np.sum(enhancement > 0)
            if enhanced_pixels > 0:
                print(f"  Enhanced {enhanced_pixels} pixels")
                print(f"  Mean enhancement: {np.mean(enhancement[enhancement > 0]):.3f}")
            else:
                print("  No pixels enhanced (no roads with sufficient density)")



    if reference_transform is not None and reference_shape is not None:
        print(f"\\n✅ Verifying spatial alignment to reference grid {reference_shape}:")

        if ndvi is not None and ndvi.shape != reference_shape:
            raise ValueError(
                f"NDVI shape {ndvi.shape} does not match reference {reference_shape}. "
                "This indicates a reprojection error."
            )

        if ndwi is not None and ndwi.shape != reference_shape:
            raise ValueError(
                f"NDWI shape {ndwi.shape} does not match reference {reference_shape}. "
                "This indicates a reprojection error."
            )


        print("  All arrays properly aligned!")

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

        print("\nBefore NDUI Calculation:")
        print(f"  NTL composite shape: {ntl_composite.shape}")
        print(
            f"  NTL non-zero pixels: {(ntl_composite > 0).sum()} "
            f"({(ntl_composite > 0).sum() / ntl_composite.size * 100:.1f}%)"
        )
        print(f"  NTL range: [{np.nanmin(ntl_composite):.3f}, {np.nanmax(ntl_composite):.3f}]")
        print(
            f"  NDVI non-NaN pixels: {(~np.isnan(ndvi)).sum()} "
            f"({(~np.isnan(ndvi)).sum() / ndvi.size * 100:.1f}%)"
        )
        print(
            f"  NDVI range (clamped to [0,1]): [{np.nanmin(ndvi):.3f}, {np.nanmax(ndvi):.3f}]"
        )

        ceiling_value = config.analysis.ntl_ceiling if config and hasattr(config, "analysis") else 10.0
        print(f"\n🔍 DEBUG: Using ntl_ceiling = {ceiling_value}")
        
        ndui = analyze.calculate_ndui(
            ntl=ntl_composite,
            ndvi=ndvi,
            ndwi=ndwi,  # Optional, can be None
            ntl_floor=config.analysis.ntl_floor if config and hasattr(config, "analysis") else 0.1,
            ntl_ceiling=ceiling_value,
            ndui_floor=config.analysis.ndui_floor
            if config and hasattr(config, "analysis")
            else 0.02,
            epsilon=1e-6,
        )

        print("\nAfter NDUI Calculation:")
        print(
            f"  NDUI non-zero pixels: {(ndui > 0).sum()} "
            f"({(ndui > 0).sum() / ndui.size * 100:.1f}%)"
        )
        print(f"  NDUI range: [{np.nanmin(ndui):.3f}, {np.nanmax(ndui):.3f}]")

        # Save diagnostic output if requested
        if config and getattr(config, "save_diagnostics", False) and ndui is not None:
            prepare.save_diagnostic(
                ndui,
                reference_transform,
                "data/diagnostics/ndui_raw.tif",
                "NDUI (calculated with enhanced NTL)",
                crs=processing_crs,
            )

    urban_mask = None
    if ndui is not None:
        urban_mask = ndui > 0.3

    print("\n✨ Enhancing results...")

    ntl_enhanced = None
    if ntl_composite is not None:
        # Map config contrast method to valid options
        from typing import Literal

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
                crs=processing_crs,
            )


    ntl_colored = None
    if ndui is not None:
        print("\nBefore Colormap Application (USING NDUI):")
        print(f"  Data shape: {ndui.shape}")
        print(f"  Non-zero pixels: {(ndui > 0).sum()} ({(ndui > 0).sum() / ndui.size * 100:.1f}%)")
        print(f"  Data range: [{np.nanmin(ndui):.3f}, {np.nanmax(ndui):.3f}]")
        print(f"  Data type: {ndui.dtype}")

        # Apply inferno colormap to NDUI with appropriate scale range
        ntl_colored = enhance.apply_inferno_colormap(
            data=ndui,
            scale_range=(0.3, 1.0),
        )

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
                crs=processing_crs,
            )

    landsat_viz = None
    if "B5" in landsat_reprojected and "B4" in landsat_reprojected and "B3" in landsat_reprojected:
        landsat_viz = enhance.create_false_color_composite(
            red_channel=landsat_reprojected["B5"],  # NIR in red
            green_channel=landsat_reprojected["B4"],  # Red in green
            blue_channel=landsat_reprojected["B3"],  # Green in blue
            contrast_enhance=True,
            gamma=None,  # Use default gamma
        )

    print("\n💾 Exporting results...")

    validation_results = {"valid": True, "errors": []}
    if ntl_colored is not None and reference_transform is not None:
        validation_results = export.validate_for_cog(
            data=ntl_colored,
            transform=reference_transform,
            crs=processing_crs,
            check_tiling=True,
            check_compression=True,
            check_overviews=True,
        )

        if not validation_results["valid"]:
            print(f"Warning: Output validation failed: {validation_results['errors']}")

    road_types = ["motorway", "trunk", "primary", "secondary"]
    metadata = export.create_metadata(
        data_sources={
            "landsat": {
                "files": all_landsat_files["B5"]
                + all_landsat_files["B4"]
                + all_landsat_files["B3"],
                "date_range": f"{date.strftime('%Y-%m-%d')}",
                "sensor": "Landsat 8/9",
                "processing_level": "L2SP",
            },
            "viirs": {
                "files": viirs_files.get("gap_filled_ntl", []) if viirs_files else [],
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
        crs=processing_crs,
        resolution=target_resolution_landsat,
    )

    if ntl_colored is not None and reference_transform is not None:
        compress_setting = config.export.compress if config else "deflate"
        valid_compressions: set[str] = {"deflate", "lzw", "jpeg", "webp", "zstd"}
        if compress_setting not in valid_compressions:
            compress_setting = "deflate"

        # Note: ntl_colored is RGB uint8 data, can't use NaN as nodata
        # The colormap application already handles NaN -> black mapping
        export.create_cog(
            data=ntl_colored,
            output_path=output_path,
            transform=reference_transform,
            crs=processing_crs,
            metadata=metadata,
            compress=compress_setting,  # type: ignore[arg-type]
            tiled=config.export.tiled if config else True,
            overviews=config.export.overview_levels if config else [2, 4, 8, 16],
            # No nodata for RGB imagery
        )

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

        export.create_cog(
            data=urban_mask.astype(np.uint8),
            output_path=output_path.replace(".tif", "_urban_mask.tif"),
            transform=reference_transform,
            crs=processing_crs,
            metadata={"band_name": "urban_mask"},
            compress=compress_setting,  # type: ignore[arg-type]
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

            # Stack bands and create metadata
            band_names = list(indices_dict.keys())
            bands_list = list(indices_dict.values())
            data = np.stack(bands_list, axis=0)

            metadata = {}
            for i, name in enumerate(band_names):
                metadata[f"band_{i + 1}_name"] = name

            export.create_cog(
                data=data,
                output_path=output_path.replace(".tif", "_indices.tif"),
                transform=reference_transform,
                crs=processing_crs,
                metadata=metadata,
                compress=compress_setting,  # type: ignore[arg-type]
                nodata=np.nan,  # Properly handle NaN values from masking
            )

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


# Sync wrapper for CLI convenience
def black_marble_pipeline_sync(**kwargs):
    """Synchronous wrapper for the async pipeline."""
    import asyncio

    return asyncio.run(black_marble_pipeline(**kwargs))
