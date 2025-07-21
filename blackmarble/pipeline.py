"""Pipeline for Black Marble processing using async I/O.

This pipeline uses async I/O for all data acquisition operations
for maximum performance.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import rasterio
from osgeo import gdal

from . import acquire, analyze, enhance, export, prepare
from .config import PipelineConfig
from .prepare.diagnostic import save_temporal_stack

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

    # ==================== SETUP PHASE ====================
    print("🚀 Starting Black Marble pipeline with async downloads")

    # Get Earthdata token from parameter or environment
    if earthdata_token is None:
        earthdata_token = os.environ.get("EARTHDATA_TOKEN")
        if not earthdata_token:
            logger.warning(
                "No Earthdata token provided. Set EARTHDATA_TOKEN environment variable "
                "or pass earthdata_token parameter. VIIRS downloads will be skipped."
            )

    # Get configuration values
    composite_months = config.acquisition.composite_months if config else 12
    data_dir = "./data"  # Use default data directory

    # ==================== ASYNC DOWNLOAD PHASE ====================
    # Download ALL data upfront using async I/O

    print(f"\n📥 Downloading all data for {composite_months + 1} dates...")
    print(f"   Target date: {date.strftime('%Y-%m-%d')}")
    print(f"   Bounding box: {bbox}")

    # Create download plan and orchestrator
    from .acquire.async_orchestrator import AsyncDownloadOrchestrator, DownloadPlan

    plan = DownloadPlan.from_pipeline_config(bbox, date, config)
    orchestrator = AsyncDownloadOrchestrator(earthdata_token=earthdata_token, show_progress=True)

    # Run async downloads
    download_results = await orchestrator.download_all(plan)

    print("\n✅ Downloads complete!")
    print(download_results.summary())

    # Check for critical failures
    if not download_results.viirs_data and earthdata_token:
        logger.error("VIIRS downloads failed - cannot continue without nighttime lights data")
        print("❌ Pipeline failed: VIIRS data is required but download failed")
        print("   Please check your EARTHDATA_TOKEN and network connection")
        # Check if there were specific download errors
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

    # ==================== PREPARE PHASE ====================
    # Process raw data into analysis-ready format

    print("\n🔧 Processing downloaded data...")

    # Ensure temp directory exists for VRTs
    os.makedirs(f"{data_dir}/temp", exist_ok=True)

    # Extract downloaded data
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

    # Process VIIRS data (if available)
    viirs_corrected = None
    viirs_profile = None
    viirs_transform = None

    if viirs_files:
        print("\n📊 Processing VIIRS nighttime lights...")
        print(f"VIIRS files keys: {list(viirs_files.keys())}")
        for key, files in viirs_files.items():
            print(f"  {key}: {len(files)} files")

        # Check if we have the expected layers - match sync pipeline names
        viirs_ntl_files = viirs_files.get("gap_filled_ntl", [])
        viirs_quality_files = viirs_files.get("quality", [])

        if not viirs_ntl_files:
            print("⚠️  No VIIRS NTL files found - skipping VIIRS processing")
        else:
            # Create VRT mosaic if multiple tiles
            if len(viirs_ntl_files) > 1:
                print(f"Creating VRT mosaic for {len(viirs_ntl_files)} VIIRS tiles")
                viirs_ntl_vrt = f"{data_dir}/temp/viirs_gap_filled_ntl_mosaic.vrt"
                os.makedirs(f"{data_dir}/temp", exist_ok=True)
                vrt_ds = gdal.BuildVRT(viirs_ntl_vrt, [str(f) for f in viirs_ntl_files])
                vrt_ds = None  # Close to write to disk
                
                # Save VRT to diagnostics if requested
                if config and getattr(config, "save_diagnostics", False):
                    os.makedirs(f"{data_dir}/diagnostics/vrts", exist_ok=True)
                    import shutil
                    shutil.copy(viirs_ntl_vrt, f"{data_dir}/diagnostics/vrts/viirs_gap_filled_ntl_mosaic.vrt")
                    print(f"  Saved VRT to: data/diagnostics/vrts/viirs_gap_filled_ntl_mosaic.vrt")

                with rasterio.open(viirs_ntl_vrt) as src:
                    viirs_ntl = src.read(1)
                    viirs_profile = src.profile
            else:
                with rasterio.open(str(viirs_ntl_files[0])) as src:
                    viirs_ntl = src.read(1)
                    viirs_profile = src.profile

            # Crop VIIRS to bounding box
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

            # Resample to finer resolution
            target_resolution = 0.00011111111  # ~12 meters
            viirs_ntl, viirs_transform = prepare.resample_image(
                data=viirs_ntl,
                src_transform=viirs_transform,
                target_resolution=target_resolution,
                resampling="bilinear",
            )

            # Update profile
            viirs_profile["transform"] = viirs_transform
            viirs_profile["height"] = viirs_ntl.shape[0]
            viirs_profile["width"] = viirs_ntl.shape[1]

            # Save diagnostic output if requested
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

        # Save final VIIRS data
        if config and getattr(config, "save_diagnostics", False) and viirs_profile is not None:
            if viirs_corrected is not None:
                prepare.save_diagnostic(
                    viirs_corrected,
                    viirs_profile,
                    "data/diagnostics/viirs_final.tif",
                    "VIIRS final (gap-filled NTL)",
                )

    # Set default resolution and profile for case when VIIRS is not available
    target_resolution = 0.00011111111  # ~12 meters
    if viirs_corrected is None:
        print("⚠️  No VIIRS data - using Landsat as reference")
        # We'll set the reference profile from the first Landsat scene

    # Determine appropriate UTM zone for the area (like sync pipeline)
    center_lon = (bbox[0] + bbox[2]) / 2
    center_lat = (bbox[1] + bbox[3]) / 2

    # Calculate UTM zone
    utm_zone = int((center_lon + 180) / 6) + 1
    hemisphere = "north" if center_lat >= 0 else "south"
    epsg_code = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone

    # Use configured CRS or default to UTM for accurate local analysis
    target_crs = f"EPSG:{epsg_code}"  # Use calculated UTM zone
    target_resolution_landsat = 30.0  # 30 meters for Landsat

    print(
        f"Using projection: {target_crs} "
        f"(UTM Zone {utm_zone}{'N' if hemisphere == 'north' else 'S'})"
    )

    log.debug(f"Target CRS determined from bbox center:")
    log.debug(f"  - BBox: {bbox}")
    log.debug(f"  - Center: ({center_lon:.6f}, {center_lat:.6f})")
    log.debug(f"  - UTM Zone: {utm_zone}{'N' if hemisphere == 'north' else 'S'}")
    log.debug(f"  - Target CRS: {target_crs}")

    # Process Landsat data
    print("\n🛰️ Processing Landsat data...")

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

    # Initialize band stacks and index stacks
    band_stacks = {"B3": [], "B4": [], "B5": []}
    band_dates = {"B3": [], "B4": [], "B5": []}  # Track dates for diagnostics
    
    # Initialize index stacks for index-first calculation
    index_stacks = {"ndvi": [], "ndwi": []}
    index_dates = []  # Dates for index composites
    
    # Initialize composite variables (for index-first calculation)
    ndvi_composite = None
    ndwi_composite = None
    ndvi_for_ndui = None  # High-percentile NDVI for NDUI
    ndvi_for_ndui_reprojected = None  # Reprojected version
    reference_profile = None

    # Track all landsat files for metadata
    all_landsat_files = {"B3": [], "B4": [], "B5": []}
    
    # Track which tiles we actually have data for
    tiles_with_data = set()

    # Add memory monitoring if available
    monitor_memory = False
    process = None
    try:
        import psutil

        process = psutil.Process()
        monitor_memory = True
    except ImportError:
        monitor_memory = False
        print("Note: Install psutil for memory monitoring (pip install psutil)")

    # Original in-memory processing
    print(f"\n💾 Processing {len(landsat_files_by_date)} Landsat dates")
    
    # Debug configuration
    if config:
        print(f"  Index-first calculation: {getattr(config.analysis, 'calculate_indices_first', False)}")
        print(f"  Use legacy indices: {getattr(config.analysis, 'use_legacy_indices', False)}")
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
                parts = filename.split('_')
                if len(parts) >= 3:
                    pathrow = parts[2]
                    if len(pathrow) == 6 and pathrow.isdigit():
                        date_tiles.add(pathrow)
                        tiles_with_data.add(pathrow)
        
        if date_tiles:
            print(f"  Tiles for this date: {sorted(date_tiles)}")

        # Convert paths to tuple format expected by process_landsat_date
        # Format is: (band_files, qa_pixel_files, qa_radsat_files, qa_aerosol_files)
        bands_data = {}
        for band, paths in date_files.items():
            if paths:
                # All bands use the same format - files go in first position
                bands_data[band] = ([str(p) for p in paths], [], [], [])

        # Process this date's data
        # Use less aggressive masking for compositing if configured
        composite_strategy = getattr(config.preparation, "qa_composite_strategy", None) if config else None
        composite_dilation = getattr(config.preparation, "qa_composite_dilation", None) if config else None
        
        actual_strategy = composite_strategy or getattr(config.preparation, "qa_cloud_strategy", "moderate") if config else "moderate"
        actual_dilation = composite_dilation if composite_dilation is not None else dilate_pixels
        
        processed_data = prepare.process_landsat_date(
            bands_data=bands_data,
            bbox=bbox,
            apply_qa_mask=True,
            dilate_pixels=actual_dilation,
            cloud_strategy=actual_strategy,
        )

        # Check if we should calculate indices first (new approach)
        if config and config.analysis.calculate_indices_first and not config.analysis.use_legacy_indices:
            # Calculate indices for this date BEFORE temporal compositing
            print(f"  Using index-first calculation for {target_date.strftime('%Y-%m-%d')}")
            if "B5" in processed_data and "B4" in processed_data and "B3" in processed_data:
                # Get the masked data
                b3_data, _ = processed_data["B3"]
                b4_data, _ = processed_data["B4"]
                b5_data, profile = processed_data["B5"]
                
                if reference_profile is None:
                    reference_profile = profile
                
                # Calculate NDVI for this date
                ndvi_date = analyze.calculate_ndvi(
                    nir=b5_data,
                    red=b4_data,
                    mask_low_denom=config.analysis.mask_low_denom
                ).astype(np.float32)  # Control memory
                
                # Calculate NDWI for this date
                ndwi_date = analyze.calculate_ndwi(
                    green=b3_data,
                    nir=b5_data,
                    mask_low_denom=config.analysis.mask_low_denom
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
        else:
            # Legacy approach: Store bands for later
            for band_name in ["B3", "B4", "B5"]:
                if band_name in processed_data:
                    masked_data, profile = processed_data[band_name]
                    band_stacks[band_name].append(masked_data)

                    if reference_profile is None:
                        reference_profile = profile

                    # Track downloaded files for metadata
                    if band_name in bands_data:
                        band_files, _, _, _ = bands_data[band_name]
                        all_landsat_files[band_name].extend(band_files)

                    band_dates[band_name].append(target_date)  # NEW: track date

        # Log memory after processing date
        if monitor_memory and process is not None:
            memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
            if config and config.analysis.calculate_indices_first and not config.analysis.use_legacy_indices:
                total_arrays = sum(len(stack) for stack in index_stacks.values())
                print(
                    f"  Memory after date: {memory_gb:.1f} GB (index arrays in memory: {total_arrays})"
                )
            else:
                total_arrays = sum(len(stack) for stack in band_stacks.values())
                print(
                    f"  Memory after date: {memory_gb:.1f} GB (band arrays in memory: {total_arrays})"
                )

    # Persist per-date arrays for diagnostics
    if config and getattr(config, "save_temporal_stack", False) and reference_profile is not None:
        for bn, stack in band_stacks.items():
            if stack:
                save_temporal_stack(
                    stack=stack,
                    band_name=bn,
                    dates=band_dates[bn],
                    reference_profile=reference_profile,
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
    print(f"  Band stacks: B3={len(band_stacks['B3'])}, B4={len(band_stacks['B4'])}, B5={len(band_stacks['B5'])}")

    if config and config.analysis.calculate_indices_first and not config.analysis.use_legacy_indices and any(index_stacks.values()):
        # NEW APPROACH: Create index composites (indices already calculated per-date)
        print("  Using index-first calculation (mathematically correct)")
        
        # Create NDVI composite
        if index_stacks["ndvi"]:
            print(f"  Creating NDVI composite from {len(index_stacks['ndvi'])} dates")
            ndvi_composite = analyze.create_index_temporal_composite(
                index_stacks["ndvi"],
                method=config.analysis.ndvi_composite_method,
                percentile=config.analysis.ndvi_composite_percentile,
                min_valid_observations=config.analysis.min_valid_observations
            )
            print(f"    NDVI composite range: [{np.nanmin(ndvi_composite):.3f}, {np.nanmax(ndvi_composite):.3f}]")
            print(f"    Valid pixels: {(~np.isnan(ndvi_composite)).sum() / ndvi_composite.size * 100:.1f}%")
        else:
            ndvi_composite = None
            
        # Create NDWI composite  
        if index_stacks["ndwi"]:
            print(f"  Creating NDWI composite from {len(index_stacks['ndwi'])} dates")
            ndwi_composite = analyze.create_index_temporal_composite(
                index_stacks["ndwi"],
                method=config.analysis.ndwi_composite_method,
                percentile=config.analysis.ndwi_composite_percentile,
                min_valid_observations=config.analysis.min_valid_observations
            )
            print(f"    NDWI composite range: [{np.nanmin(ndwi_composite):.3f}, {np.nanmax(ndwi_composite):.3f}]")
            print(f"    Valid pixels: {(~np.isnan(ndwi_composite)).sum() / ndwi_composite.size * 100:.1f}%")
        else:
            ndwi_composite = None
            
        # For NDUI, we also want a high-percentile NDVI
        if index_stacks["ndvi"] and config.analysis.ndvi_composite_method == "percentile":
            print(f"  Creating 85th percentile NDVI for NDUI vegetation correction")
            ndvi_for_ndui = analyze.create_index_temporal_composite(
                index_stacks["ndvi"],
                method="percentile",
                percentile=85.0,
                min_valid_observations=config.analysis.min_valid_observations
            )
        else:
            ndvi_for_ndui = ndvi_composite  # Use the regular composite
            
        # Validate that we have a reference profile for reprojection
        if reference_profile is None and (ndvi_composite is not None or ndwi_composite is not None):
            print("⚠️  WARNING: No reference profile available for index reprojection")
            print("    This can happen if all Landsat scenes were filtered out")
            # Try to get profile from the first index array
            if index_stacks["ndvi"] and hasattr(index_stacks["ndvi"][0], 'shape'):
                # We need to construct a minimal profile for reprojection
                # This is a fallback - ideally we should have saved profiles with indices
                print("    Creating fallback reference profile from index shape")
                # For now, we'll have to skip reprojection
                reference_profile = None
        
        # We still need to reproject indices to target CRS
        # Create a dummy "landsat_reprojected" dict for compatibility
        landsat_reprojected = {}
        
        # Skip band compositing entirely
        
    elif not any(band_stacks.values()):
        print("⚠️  No valid Landsat data found - skipping index calculations")
        landsat_reprojected = {}
    else:
        # Create median composites from the masked stacks
        landsat_bands = {}
        landsat_profiles = {}

        for band_name, stack in band_stacks.items():
            if stack:
                print(f"  Creating temporal composite for {band_name} from {len(stack)} dates")

                # Create composite from pre-masked data with scaling
                # Use mean instead of median to avoid hard edges at scene boundaries
                composite_method = getattr(config, "landsat_composite_method", "mean")
                logger.debug(
                    f"Creating composite with scale_factor={scale_factor}, add_offset={add_offset}, method={composite_method}"
                )
                # Check if we should log coverage
                log_coverage = getattr(config, "log_composite_coverage", False) if config else False
                
                composite = analyze.create_landsat_median_composite(
                    stack, 
                    scale_factor=scale_factor, 
                    add_offset=add_offset, 
                    method=composite_method,
                    log_coverage=log_coverage
                )

                logger.debug(f"Composite stats for {band_name}:")
                logger.debug(f"  Min: {np.nanmin(composite):.3f}, Max: {np.nanmax(composite):.3f}")
                logger.debug(
                    f"  NaN pixels: {np.isnan(composite).sum()} ({np.isnan(composite).sum() / composite.size * 100:.1f}%)"
                )
                logger.debug(
                    f"  Non-zero pixels: {(composite != 0).sum()} ({(composite != 0).sum() / composite.size * 100:.1f}%)"
                )

                landsat_bands[band_name] = (composite, reference_profile)
                landsat_profiles[band_name] = reference_profile

                print(
                    f"  Created composite for {band_name}: shape={composite.shape}, "
                    f"coverage={(composite > 0).sum() / composite.size * 100:.1f}%"
                )
                # DEBUG: Check if composite is scaled properly
                print(
                    f"    Composite range: [{np.nanmin(composite):.3f}, {np.nanmax(composite):.3f}]"
                )
            else:
                print(f"  Warning: No data found for {band_name}")

    # Reproject Landsat bands to target CRS (UTM) - common for both VRT and in-memory
    reference_transform = None
    reference_shape = None
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
                dst_crs=target_crs,
                resampling="bilinear",  # Use bilinear for smoother results
            )
            landsat_reprojected[band_name] = reprojected

            # Store reference transform and shape from first band
            if reference_transform is None:
                reference_transform = new_transform
                reference_shape = reprojected.shape
                reference_bbox = new_bbox
    else:
        # No Landsat data available
        landsat_reprojected = {}
        if config and config.analysis.calculate_indices_first and not config.analysis.use_legacy_indices:
            # This is expected in index-first mode
            pass
        else:
            print("⚠️  No Landsat bands available after temporal compositing")

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
        logger.debug(f"  Target CRS: {target_crs}")

        viirs_reprojected, viirs_transform, viirs_bounds = prepare.reproject_image(
            data=viirs_corrected,
            src_transform=viirs_profile["transform"],
            src_crs=viirs_profile["crs"],
            dst_crs=target_crs,
            resampling="bilinear",  # Use bilinear like sync pipeline
        )

        logger.debug(f"  Reprojected VIIRS shape: {viirs_reprojected.shape}")
        logger.debug(f"  Reprojected VIIRS bounds: {viirs_bounds}")

        # If no reference from Landsat, use VIIRS as reference
        if reference_transform is None and viirs_transform is not None:
            reference_transform = viirs_transform
            reference_shape = viirs_reprojected.shape
            # Keep the target_crs as UTM - VIIRS has already been reprojected to it
            print(f"📍 Using VIIRS as reference grid (no Landsat reference available)")
            print(f"   Reference CRS: {target_crs} (UTM)")
            print(f"   Reference shape: {reference_shape}")

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
            # No Landsat data to align to
            if not landsat_reprojected:
                print("⚠️  WARNING: No Landsat data available for NDVI calculation!")
                print("    Pipeline will continue but NDUI calculation will be limited")

    # Rasterize roads to match grid
    roads_rasterized = None
    if road_network is not None and reference_shape is not None and reference_transform is not None:
        # Use the new CRS-aware road rasterization (like sync pipeline)
        # Use 30m buffer to match default in osm.py - 5.5m was too small
        buffer_meters = 30.0  # Buffer roads by 30 meters

        print(f"Road buffering: {buffer_meters:.1f} meters (applied in UTM)")
        print(f"Rasterizing to: {target_crs}")
        print(f"Road network has {len(road_network.nodes)} nodes, {len(road_network.edges)} edges")
        print(f"Reference transform: {reference_transform}")
        print(f"Reference shape: {reference_shape}")

        # Use fractional or binary road rasterization based on config
        if config and config.analysis.osm_fractional:
            # NEW: Fractional rasterization
            print("  Using fractional road rasterization (5x5 sub-pixels)")
            roads_fractional = acquire.buffer_and_rasterize_roads_fractional(
                graph=road_network,
                bbox=bbox,
                transform=reference_transform,
                crs=target_crs,
                shape=(reference_shape[0], reference_shape[1]),
                buffer_meters=buffer_meters,
                sub_pixels=config.analysis.osm_sub_pixels,
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
            if getattr(config, "save_diagnostics", False):
                prepare.save_diagnostic(
                    roads_fractional,
                    reference_transform,
                    "data/diagnostics/roads_fractional.tif",
                    "Fractional road coverage (0.0-1.0)",
                    crs=target_crs,
                )

            roads_rasterized = None
        else:
            # LEGACY: Binary hierarchical rasterization
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
            print(
                f"Rasterized {road_pixels} pixels "
                f"({road_pixels / roads_rasterized.size * 100:.2f}%)"
            )

            # Save diagnostic output if requested
            if config and getattr(config, "save_diagnostics", False):
                road_diagnostic = roads_rasterized.astype(np.float32)
                prepare.save_diagnostic(
                    road_diagnostic,
                    reference_transform,
                    "data/diagnostics/roads_rasterized.tif",
                    "Rasterized road network",
                    crs=target_crs,
                )

            roads_fractional = None
    else:
        # No road network data
        print("  No road network data available")
        if reference_shape is not None:
            roads_rasterized = np.zeros(reference_shape, dtype="uint8")
            roads_fractional = np.zeros(reference_shape, dtype="float32")
        else:
            # This should never happen if we have any data
            raise RuntimeError(
                "No reference shape available for road rasterization. "
                "This indicates no Landsat or VIIRS data was successfully processed."
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
                    crs=target_crs,
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

    # ==================== ANALYZE PHASE ====================
    print("\n📈 Calculating indices...")

    # Calculate spectral indices
    ndvi = None
    ndwi = None
    
    # Check if we already have indices from index-first calculation
    if config and config.analysis.calculate_indices_first and not config.analysis.use_legacy_indices:
        # We already have composited indices - just need to reproject them
        if ndvi_composite is not None and reference_profile is not None:
            print("  Reprojecting pre-calculated NDVI composite...")
            ndvi, ndvi_transform, _ = prepare.reproject_image(
                data=ndvi_composite,
                src_transform=reference_profile["transform"],
                src_crs=reference_profile["crs"],
                dst_crs=target_crs,
                resampling="bilinear",
            )
            print(f"    Reprojected NDVI shape: {ndvi.shape}")
            
            # Also reproject the high-percentile NDVI for NDUI if different
            if ndvi_for_ndui is not None and ndvi_for_ndui is not ndvi_composite:
                ndvi_for_ndui_reprojected, _, _ = prepare.reproject_image(
                    data=ndvi_for_ndui,
                    src_transform=reference_profile["transform"],
                    src_crs=reference_profile["crs"],
                    dst_crs=target_crs,
                    resampling="bilinear",
                )
            else:
                ndvi_for_ndui_reprojected = ndvi
                
        if ndwi_composite is not None and reference_profile is not None:
            print("  Reprojecting pre-calculated NDWI composite...")
            ndwi, ndwi_transform, _ = prepare.reproject_image(
                data=ndwi_composite,
                src_transform=reference_profile["transform"],
                src_crs=reference_profile["crs"],
                dst_crs=target_crs,
                resampling="bilinear",
            )
            print(f"    Reprojected NDWI shape: {ndwi.shape}")
            
        # Update reference transform if we have it from NDVI
        if ndvi is not None and reference_transform is None:
            # Get the transform from the reprojected NDVI
            # Since we used prepare.reproject_image, the transform was returned
            if 'ndvi_transform' in locals():
                reference_transform = ndvi_transform
                reference_shape = ndvi.shape
            
    # Legacy approach: calculate from band composites
    elif "B5" in landsat_reprojected and "B4" in landsat_reprojected:
        # DEBUG: Check Landsat values before NDVI
        logger.debug("Landsat Values Before NDVI:")
        logger.debug(
            f"  B4 (red): min={np.nanmin(landsat_reprojected['B4']):.3f}, max={np.nanmax(landsat_reprojected['B4']):.3f}"
        )
        logger.debug(
            f"  B4 negative pixels: {(landsat_reprojected['B4'] < 0).sum()} ({(landsat_reprojected['B4'] < 0).sum() / landsat_reprojected['B4'].size * 100:.1f}%)"
        )
        logger.debug(
            f"  B5 (NIR): min={np.nanmin(landsat_reprojected['B5']):.3f}, max={np.nanmax(landsat_reprojected['B5']):.3f}"
        )
        logger.debug(
            f"  B5 negative pixels: {(landsat_reprojected['B5'] < 0).sum()} ({(landsat_reprojected['B5'] < 0).sum() / landsat_reprojected['B5'].size * 100:.1f}%)"
        )

        ndvi = analyze.calculate_ndvi(nir=landsat_reprojected["B5"], red=landsat_reprojected["B4"])

        # Save diagnostic output if requested
        if config and getattr(config, "save_diagnostics", False) and ndvi is not None:
            prepare.save_diagnostic(
                ndvi,
                reference_transform,
                "data/diagnostics/ndvi.tif",
                "NDVI (Normalized Difference Vegetation Index)",
                crs=target_crs,
            )

    # Legacy NDWI calculation
    if ndwi is None and "B3" in landsat_reprojected and "B5" in landsat_reprojected:
        ndwi = analyze.calculate_ndwi(
            green=landsat_reprojected["B3"], nir=landsat_reprojected["B5"]
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

    # For now, we only have single date VIIRS, so no temporal compositing
    ntl_composite = viirs_reprojected if viirs_reprojected is not None else None

    # ENHANCE NTL WITH ROADS (before NDUI calculation)
    # This is the correct place to apply road enhancement - to the NTL data itself,
    # not to the derived NDUI index
    if ntl_composite is not None:
        enhancement_approach = config.analysis.ntl_enhancement_approach if config else "urban_field"

        if (
            enhancement_approach == "urban_field"
            and config
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

            # DEBUG: Print urban field statistics
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
                    crs=target_crs,
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
                    crs=target_crs,
                )

            ntl_composite = ntl_enhanced
            enhanced_pixels = np.sum(enhancement > 0)
            if enhanced_pixels > 0:
                print(f"  Enhanced {enhanced_pixels} pixels")
                print(f"  Mean enhancement: {np.mean(enhancement[enhancement > 0]):.3f}")
            else:
                print("  No pixels enhanced (no roads with sufficient density)")

        elif (
            enhancement_approach == "urban_field"
            and roads_rasterized is not None
            and not (config and config.analysis.osm_fractional)
        ):
            # Urban field approach but only binary roads available - convert to pseudo-fractional
            print("\n🌃 Enhancing NTL with urban fields (binary roads converted to fractional)")

            # Convert binary road mask to fractional (0 or 1)
            roads_fractional_pseudo = roads_rasterized.astype(np.float32)
            roads_fractional_pseudo[roads_fractional_pseudo > 0] = 1.0

            # Compute urban fields from pseudo-fractional roads
            urban_field = analyze.compute_urban_fields(
                road_fraction=roads_fractional_pseudo,
                sigmas=config.analysis.urban_field_sigmas if config else [2.0, 5.0, 10.0],
                weights=config.analysis.urban_field_weights if config else [0.5, 0.3, 0.2],
            )

            # Enhance NTL
            ntl_enhanced = analyze.enhance_ntl_with_urban_field(
                ntl=ntl_composite,
                urban_field=urban_field,
                enhancement_mode=config.analysis.ntl_enhancement_mode if config else "hybrid",
                multiplicative_factor=config.analysis.enhancement_multiplicative_factor
                if config
                else 0.3,
                additive_factor=config.analysis.enhancement_additive_factor if config else 0.02,
                hybrid_expected_scale=config.analysis.enhancement_hybrid_expected_scale
                if config
                else 0.2,
                hybrid_expected_sqrt_scale=config.analysis.enhancement_hybrid_expected_sqrt_scale
                if config
                else 0.1,
                hybrid_application_factor=config.analysis.enhancement_hybrid_application_factor
                if config
                else 1.0,
            )

            enhancement = ntl_enhanced - ntl_composite
            ntl_composite = ntl_enhanced
            enhanced_pixels = np.sum(enhancement > 0)
            if enhanced_pixels > 0:
                print(f"  Enhanced {enhanced_pixels} pixels")
                print(f"  Mean enhancement: {np.mean(enhancement[enhancement > 0]):.3f}")

        elif enhancement_approach == "legacy_burn" and (
            roads_rasterized is not None or roads_fractional is not None
        ):
            # LEGACY BURN-IN ENHANCEMENT
            print("\n🛣️  Enhancing NTL with road data (legacy burn-in)")

            major_factor = config.analysis.legacy_major_road_factor if config else 1.3
            minor_factor = config.analysis.legacy_minor_road_factor if config else 1.1

            if roads_fractional is not None and config and config.analysis.osm_fractional:
                # Use fractional roads for burn-in
                print("  Using fractional road coverage for burn-in")
                ntl_enhanced = ntl_composite.copy()

                # The fractional value represents the percentage of pixel covered by roads
                # Apply enhancement proportionally
                # For pixels with roads, enhance by: 1 + (factor - 1) * coverage
                # This gives smooth transitions: 0% coverage = 1x, 100% coverage = factor
                # roads_fractional is always a numpy array with float values
                # Simple fractional coverage (0-1), use average of factors
                avg_factor = (major_factor + minor_factor) / 2
                enhancement_factor = 1 + (avg_factor - 1) * roads_fractional
                ntl_enhanced = ntl_composite * enhancement_factor

            else:
                # Use binary roads
                print("  Using binary road mask for burn-in")
                ntl_enhanced = ntl_composite.copy()

                # Apply enhancement based on road type
                major_roads_mask = (roads_rasterized == 1) | (roads_rasterized == 4)
                minor_roads_mask = (
                    (roads_rasterized == 2)
                    | (roads_rasterized == 3)
                    | (roads_rasterized == 5)
                    | (roads_rasterized == 6)
                )

                ntl_enhanced = np.where(
                    major_roads_mask, ntl_composite * major_factor, ntl_enhanced
                )
                ntl_enhanced = np.where(
                    minor_roads_mask, ntl_composite * minor_factor, ntl_enhanced
                )

            # Log statistics
            enhancement = ntl_enhanced - ntl_composite
            enhanced_pixels = np.sum(enhancement > 0)
            if enhanced_pixels > 0:
                print(
                    f"  Enhanced {enhanced_pixels} pixels "
                    f"({enhanced_pixels / ntl_composite.size * 100:.1f}%)"
                )
                # Show appropriate statistics based on road type
                if roads_rasterized is not None and not (
                    roads_fractional is not None and config and config.analysis.osm_fractional
                ):
                    # Binary roads - show road type breakdown
                    # Calculate masks for statistics
                    major_roads_mask_stats = (roads_rasterized == 1) | (roads_rasterized == 4)
                    minor_roads_mask_stats = (
                        (roads_rasterized == 2)
                        | (roads_rasterized == 3)
                        | (roads_rasterized == 5)
                        | (roads_rasterized == 6)
                    )
                    print(
                        f"  Major roads ({major_factor}x): {np.sum(major_roads_mask_stats)} pixels"
                    )
                    print(
                        f"  Minor roads ({minor_factor}x): {np.sum(minor_roads_mask_stats)} pixels"
                    )
                else:
                    # Fractional roads - show coverage statistics
                    if roads_fractional is not None:
                        coverage_pixels = np.sum(roads_fractional > 0)
                        avg_coverage = (
                            np.mean(roads_fractional[roads_fractional > 0])
                            if coverage_pixels > 0
                            else 0
                        )
                        print(
                            f"  Road coverage: {coverage_pixels} pixels "
                            f"with avg {avg_coverage:.1%} coverage"
                        )
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

    # Align NDVI/NDWI to reference grid if needed (especially important when VIIRS is the reference)
    if reference_transform is not None and reference_shape is not None:
        # Align NDVI if it exists and hasn't been aligned yet
        if ndvi is not None and hasattr(ndvi, 'shape'):
            # Check if NDVI needs alignment (different shape from reference)
            if ndvi.shape != reference_shape:
                print(f"\\n📐 Aligning NDVI to reference grid...")
                print(f"  NDVI shape: {ndvi.shape}")
                print(f"  Reference shape: {reference_shape}")
                
                # We need the NDVI transform - it should have been set during reprojection
                if 'ndvi_transform' in locals() and ndvi_transform is not None:
                    # Create a dummy reference array with the correct shape
                    ref_array = np.zeros(reference_shape, dtype=np.float32)
                    aligned_rasters = prepare.align_rasters(
                        rasters=[ref_array, ndvi],
                        transforms=[reference_transform, ndvi_transform],
                        reference_idx=0,
                        resampling="bilinear",
                    )
                    ndvi = aligned_rasters[1][0]
                    ndvi_transform = aligned_rasters[1][1]
                    print(f"  Aligned NDVI shape: {ndvi.shape}")
                else:
                    print("  ⚠️  WARNING: Cannot align NDVI - no transform available")
        
        # Align NDWI similarly
        if ndwi is not None and hasattr(ndwi, 'shape'):
            if ndwi.shape != reference_shape:
                print(f"\\n📐 Aligning NDWI to reference grid...")
                print(f"  NDWI shape: {ndwi.shape}")
                print(f"  Reference shape: {reference_shape}")
                
                if 'ndwi_transform' in locals() and ndwi_transform is not None:
                    # Create a dummy reference array with the correct shape
                    ref_array = np.zeros(reference_shape, dtype=np.float32)
                    aligned_rasters = prepare.align_rasters(
                        rasters=[ref_array, ndwi],
                        transforms=[reference_transform, ndwi_transform],
                        reference_idx=0,
                        resampling="bilinear",
                    )
                    ndwi = aligned_rasters[1][0]
                    ndwi_transform = aligned_rasters[1][1]
                    print(f"  Aligned NDWI shape: {ndwi.shape}")
                else:
                    print("  ⚠️  WARNING: Cannot align NDWI - no transform available")
        
        # Align high-percentile NDVI for NDUI if it exists
        if ndvi_for_ndui_reprojected is not None and hasattr(ndvi_for_ndui_reprojected, 'shape'):
            if ndvi_for_ndui_reprojected.shape != reference_shape:
                print(f"\\n📐 Aligning high-percentile NDVI to reference grid...")
                # Use the same transform as regular NDVI
                if 'ndvi_transform' in locals() and ndvi_transform is not None:
                    # Create a dummy reference array with correct shape
                    ref_array = np.zeros(reference_shape, dtype=np.float32)
                    aligned_rasters = prepare.align_rasters(
                        rasters=[ref_array, ndvi_for_ndui_reprojected],
                        transforms=[reference_transform, ndvi_transform],
                        reference_idx=0,
                        resampling="bilinear",
                    )
                    ndvi_for_ndui_reprojected = aligned_rasters[1][0]
                    print(f"  Aligned high-percentile NDVI shape: {ndvi_for_ndui_reprojected.shape}")

    # Calculate urban index combining NTL and vegetation
    ndui = None
    if ntl_composite is not None and ndvi is not None:
        # Use the appropriate NDVI for NDUI calculation
        ndvi_for_ndui_final = ndvi  # Default to regular NDVI
        
        # Check if we have a special high-percentile NDVI for NDUI
        if ndvi_for_ndui_reprojected is not None:
            ndvi_for_ndui_final = ndvi_for_ndui_reprojected
            print("  Using 85th percentile NDVI for NDUI vegetation correction")
        
        # Verify shapes match before calculating NDUI
        if ntl_composite.shape != ndvi_for_ndui_final.shape:
            print("WARNING: Shape mismatch after alignment!")
            print(f"  NTL shape: {ntl_composite.shape}")
            print(f"  NDVI shape: {ndvi_for_ndui_final.shape}")
            raise ValueError(
                f"NTL and NDVI shapes still don't match after alignment: "
                f"{ntl_composite.shape} vs {ndvi_for_ndui_final.shape}. This indicates a problem "
                f"with the spatial alignment step."
            )

        # DIAGNOSTIC: Print data statistics before NDUI
        print("\nBefore NDUI Calculation:")
        print(f"  NTL composite shape: {ntl_composite.shape}")
        print(
            f"  NTL non-zero pixels: {(ntl_composite > 0).sum()} "
            f"({(ntl_composite > 0).sum() / ntl_composite.size * 100:.1f}%)"
        )
        print(f"  NTL range: [{np.nanmin(ntl_composite):.3f}, {np.nanmax(ntl_composite):.3f}]")
        print(
            f"  NDVI non-NaN pixels: {(~np.isnan(ndvi_for_ndui_final)).sum()} "
            f"({(~np.isnan(ndvi_for_ndui_final)).sum() / ndvi_for_ndui_final.size * 100:.1f}%)"
        )
        print(f"  NDVI range: [{np.nanmin(ndvi_for_ndui_final):.3f}, {np.nanmax(ndvi_for_ndui_final):.3f}]")

        ndui = analyze.calculate_ndui(
            ntl=ntl_composite,
            ndvi=ndvi_for_ndui_final,
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
                crs=target_crs,
            )

    # Create urban mask
    urban_mask = None
    if ndui is not None:
        urban_mask = ndui > 0.3  # Reasonable threshold for urban areas

    # ==================== ENHANCE PHASE ====================
    print("\n✨ Enhancing results...")

    # Enhance contrast for better visualization
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
                crs=target_crs,
            )

    # Enhance road networks in the NTL data
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

    # Enhance urban features
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

    # Apply color mapping for visualization
    # CRITICAL FIX: Use NDUI for colormap like sync pipeline, not NTL
    ntl_colored = None
    if ndui is not None:
        # DIAGNOSTIC: Print data before colormap
        print("\nBefore Colormap Application (USING NDUI):")
        print(f"  Data shape: {ndui.shape}")
        print(f"  Non-zero pixels: {(ndui > 0).sum()} ({(ndui > 0).sum() / ndui.size * 100:.1f}%)")
        print(f"  Data range: [{np.nanmin(ndui):.3f}, {np.nanmax(ndui):.3f}]")
        print(f"  Data type: {ndui.dtype}")

        # Apply inferno colormap to NDUI with appropriate scale range
        # Use 0.3-0.8 range to focus on urban areas (matching sync pipeline)
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

    # Create false color composite for context (optional)
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
    print("\n💾 Exporting results...")

    # Validate output before saving
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

    # Generate metadata
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
        crs=target_crs,
        resolution=target_resolution_landsat,
    )

    # Create Cloud-Optimized GeoTIFF
    if ntl_colored is not None and reference_transform is not None:
        # Get compression setting with proper type casting

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
            crs=target_crs,
            metadata=metadata,
            compress=compress_setting,  # type: ignore[arg-type]
            tiled=config.export.tiled if config else True,
            overviews=config.export.overview_levels if config else [2, 4, 8, 16],
            # No nodata for RGB imagery
        )

    # Generate auxiliary outputs if requested
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
            crs=target_crs,
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
                crs=target_crs,
                metadata=metadata,
                compress=compress_setting,  # type: ignore[arg-type]
                nodata=np.nan,  # Properly handle NaN values from masking
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


# Sync wrapper for CLI convenience
def black_marble_pipeline_sync(**kwargs):
    """Synchronous wrapper for the async pipeline."""
    import asyncio

    return asyncio.run(black_marble_pipeline(**kwargs))
