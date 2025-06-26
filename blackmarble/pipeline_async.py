"""Async-enabled pipeline for Black Marble processing with front-loaded downloads.

This version of the pipeline downloads all data upfront using async I/O for
maximum performance, then processes everything from local files.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import os
import logging
import numpy as np
import numpy.typing as npt

from .config import PipelineConfig
from . import acquire, prepare, analyze, enhance, export

# Type aliases
ArrayLike = npt.NDArray[Any]
BBox = tuple[float, float, float, float]

logger = logging.getLogger(__name__)


def black_marble_pipeline_async(
    bbox: BBox,
    date: datetime,
    output_path: str,
    config: PipelineConfig | None = None,
    earthdata_token: str | None = None
) -> dict[str, Any]:
    """Main Black Marble processing pipeline with async downloads.
    
    This version front-loads all downloads using async I/O for maximum
    performance, then processes everything from local files.
    
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
    data_dir = './data'  # Use default data directory
    
    # ==================== ASYNC DOWNLOAD PHASE ====================
    # Download ALL data upfront using async I/O
    
    print(f"\n📥 Downloading all data for {composite_months + 1} dates...")
    print(f"   Target date: {date.strftime('%Y-%m-%d')}")
    print(f"   Bounding box: {bbox}")
    
    # Use the async download orchestrator
    download_results = acquire.download_all_data_sync(
        bbox=bbox,
        date=date,
        config=config,
        earthdata_token=earthdata_token
    )
    
    print(f"\n✅ Downloads complete!")
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
        raise RuntimeError("VIIRS download failed - pipeline cannot continue without nighttime lights data")
    
    if not download_results.landsat_data:
        logger.warning("No Landsat data downloaded - pipeline will create a minimal output with OSM data only")
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
        viirs_ntl_files = viirs_files.get('gap_filled_ntl', [])
        viirs_quality_files = viirs_files.get('quality', [])
        
        if not viirs_ntl_files:
            print("⚠️  No VIIRS NTL files found - skipping VIIRS processing")
        else:
            # Create VRT mosaic if multiple tiles
            if len(viirs_ntl_files) > 1:
                print(f"Creating VRT mosaic for {len(viirs_ntl_files)} VIIRS tiles")
                viirs_ntl_vrt = prepare.create_virtual_raster(
                    file_paths=[str(f) for f in viirs_ntl_files],
                    output_vrt=f"{data_dir}/temp/viirs_gap_filled_ntl_mosaic.vrt"
                )
                viirs_ntl, viirs_profile = prepare.load_with_rasterio(viirs_ntl_vrt)
            else:
                viirs_ntl, viirs_profile = prepare.load_with_rasterio(str(viirs_ntl_files[0]))
            
            # Crop VIIRS to bounding box
            print(f"Cropping VIIRS data to bbox: {bbox}")
            viirs_ntl, viirs_transform, _ = prepare.crop_to_bounds(
                data=viirs_ntl,
                src_transform=viirs_profile['transform'],
                bounds=bbox
            )
            
            # Resample to finer resolution
            target_resolution = 0.00011111111  # ~12 meters
            viirs_ntl, viirs_transform = prepare.resample_image(
                data=viirs_ntl,
                src_transform=viirs_transform,
                target_resolution=target_resolution,
                resampling='bilinear'
            )
            
            # Update profile
            viirs_profile['transform'] = viirs_transform
            viirs_profile['height'] = viirs_ntl.shape[0]
            viirs_profile['width'] = viirs_ntl.shape[1]
            
            # Save diagnostic output if requested
            if config and getattr(config, 'save_diagnostics', False):
                prepare.save_diagnostic(viirs_ntl, viirs_profile,
                                      "data/diagnostics/viirs_resampled.tif",
                                      "VIIRS resampled to target resolution")
            
            # The gap-filled product already incorporates quality filtering
            # by using the best available pixels from recent observations
            viirs_corrected = viirs_ntl
            
            # OLD CODE - No longer needed with gap-filled product:
            # Apply quality masking if available
            if False and viirs_quality_files:
                print("Applying VIIRS quality masking...")
                
                # Process quality files similarly
                if len(viirs_quality_files) > 1:
                    quality_vrt = prepare.create_virtual_raster(
                        file_paths=[str(f) for f in viirs_quality_files],
                        output_vrt=f"{data_dir}/temp/viirs_quality_mosaic.vrt"
                    )
                    quality_flag, _ = prepare.load_with_rasterio(quality_vrt)
                else:
                    quality_flag, _ = prepare.load_with_rasterio(str(viirs_quality_files[0]))
                
                # Crop and resample to match NTL
                quality_flag, _, _ = prepare.crop_to_bounds(
                    data=quality_flag,
                    src_transform=viirs_profile['transform'],
                    bounds=bbox
                )
                quality_flag, _ = prepare.resample_image(
                    data=quality_flag,
                    src_transform=viirs_transform,
                    target_resolution=target_resolution,
                    resampling='nearest'
                )
                
                # Apply quality mask (0 and 1 are good quality)
                quality_mask = (quality_flag <= 1).astype(float)
                viirs_corrected = viirs_ntl * quality_mask
                
                # Print quality mask statistics like sync pipeline
                print(f"\nVIIRS Quality Mask Statistics (Latest_High_Quality_Retrieval):")
                print(f"  Total pixels: {quality_flag.size}")
                print(f"  Quality flag range: [{quality_flag.min()}, {quality_flag.max()}]")
                print(f"  Pixels with quality<=1 (high/good): {quality_mask.sum()} ({quality_mask.sum() / quality_mask.size * 100:.1f}%)")
                print(f"  Non-zero VIIRS values before mask: {(viirs_ntl > 0).sum()}")
                print(f"  Non-zero VIIRS values after mask: {(viirs_corrected > 0).sum()}")
                
                # Save diagnostic output if requested
                if config and getattr(config, 'save_diagnostics', False):
                    prepare.save_diagnostic(viirs_corrected, viirs_profile,
                                          "data/diagnostics/viirs_quality_masked.tif",
                                          "VIIRS gap-filled NTL after quality masking")
            else:
                # No quality masking available
                viirs_corrected = viirs_ntl
                print("Warning: No quality data available, using gap-filled NTL without masking")
        
        # Save final VIIRS data
        if config and getattr(config, 'save_diagnostics', False) and viirs_profile is not None:
            prepare.save_diagnostic(viirs_corrected, viirs_profile,
                                  "data/diagnostics/viirs_final.tif",
                                  "VIIRS final (gap-filled NTL)")
    
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
    hemisphere = 'north' if center_lat >= 0 else 'south'
    epsg_code = 32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone
    
    # Use configured CRS or default to UTM for accurate local analysis
    target_crs = f"EPSG:{epsg_code}"  # Use calculated UTM zone
    target_resolution_landsat = 30.0  # 30 meters for Landsat
    
    print(f"Using projection: {target_crs} (UTM Zone {utm_zone}{'N' if hemisphere == 'north' else 'S'})")
    
    # Process Landsat data
    print("\n🛰️ Processing Landsat data...")
    
    # Get scale factors from config
    scale_factor = config.preparation.landsat_scale_factor if config and hasattr(config, 'preparation') else 0.0000275
    add_offset = config.preparation.landsat_add_offset if config and hasattr(config, 'preparation') else -0.2
    dilate_pixels = config.preparation.qa_dilation_radius if config and hasattr(config, 'preparation') else 3
    
    # Initialize band stacks
    band_stacks = {
        'B2': [],
        'B3': [],
        'B4': [],
        'B5': []
    }
    reference_profile = None
    
    # Track all landsat files for metadata
    all_landsat_files = {
        'B2': [],
        'B3': [],
        'B4': [],
        'B5': []
    }
    
    # Process each date's Landsat data
    for target_date in sorted(landsat_files_by_date.keys()):
        print(f"\nProcessing Landsat for {target_date.strftime('%Y-%m-%d')}")
        
        date_files = landsat_files_by_date[target_date]
        
        # Skip if we don't have all bands
        if not all(band in date_files for band in ['B2', 'B3', 'B4', 'B5', 'QA_PIXEL']):
            print(f"  Skipping - missing bands")
            continue
        
        # Convert paths to tuple format expected by process_landsat_date
        # Format is: (band_files, qa_pixel_files, qa_radsat_files, qa_aerosol_files)
        bands_data = {}
        for band, paths in date_files.items():
            if paths:
                # All bands use the same format - files go in first position
                bands_data[band] = ([str(p) for p in paths], [], [], [])
        
        # Process this date's data
        processed_data = prepare.process_landsat_date(
            bands_data=bands_data,
            bbox=bbox,
            apply_qa_mask=True,
            dilate_pixels=dilate_pixels,
            cloud_strategy="moderate"
        )
        
        # Add to stacks
        for band_name in ['B2', 'B3', 'B4', 'B5']:
            if band_name in processed_data:
                masked_data, profile = processed_data[band_name]
                band_stacks[band_name].append(masked_data)
                
                if reference_profile is None:
                    reference_profile = profile
                
                # Track downloaded files for metadata
                if band_name in bands_data:
                    band_files, _, _, _ = bands_data[band_name]
                    all_landsat_files[band_name].extend(band_files)
    
    # Create temporal composites  
    print("\n⏳ Creating Landsat temporal composites...")
    
    if not any(band_stacks.values()):
        print("⚠️  No valid Landsat data found - skipping index calculations")
        landsat_reprojected = {}
    else:
        # Create median composites from the masked stacks
        landsat_bands = {}
        landsat_profiles = {}
        
        for band_name, stack in band_stacks.items():
            if stack:
                print(f"  Creating temporal composite for {band_name} from {len(stack)} dates")
                
                # Create median composite from pre-masked data with scaling
                composite = analyze.create_landsat_median_composite(
                    stack,
                    scale_factor=scale_factor,
                    add_offset=add_offset
                )
                
                landsat_bands[band_name] = (composite, reference_profile)
                landsat_profiles[band_name] = reference_profile
                
                print(f"  Created composite for {band_name}: shape={composite.shape}, "
                      f"coverage={(composite > 0).sum()/composite.size*100:.1f}%")
            else:
                print(f"  Warning: No data found for {band_name}")
        
        # Reproject Landsat bands to target CRS (UTM)
        landsat_reprojected = {}
        reference_transform = None
        reference_shape = None
        reference_bbox = None
        
        for band_name, (data, profile) in landsat_bands.items():
            reprojected, new_transform, new_bbox = prepare.reproject_image(
                data=data,
                src_transform=profile['transform'],
                src_crs=profile['crs'],
                dst_crs=target_crs,
                resampling="bilinear"  # Use bilinear for smoother results
            )
            landsat_reprojected[band_name] = reprojected
            
            # Store reference transform and shape from first band
            if reference_transform is None:
                reference_transform = new_transform
                reference_shape = reprojected.shape
                reference_bbox = new_bbox
    
    # Reproject VIIRS to target CRS first (like sync pipeline)
    viirs_reprojected = None
    viirs_transform = None
    if viirs_corrected is not None and viirs_profile is not None:
        viirs_reprojected, viirs_transform, _ = prepare.reproject_image(
            data=viirs_corrected,
            src_transform=viirs_profile['transform'],
            src_crs=viirs_profile['crs'],
            dst_crs=target_crs,
            resampling="bilinear"  # Use bilinear like sync pipeline
        )
        
        # Align VIIRS to Landsat grid for consistent processing
        if reference_transform is not None and reference_shape is not None:
            print(f"VIIRS shape before alignment: {viirs_reprojected.shape}")
            print(f"Reference Landsat shape: {reference_shape}")
            
            # Use the first Landsat band as reference
            reference_band = landsat_reprojected[list(landsat_reprojected.keys())[0]]
            
            # Align VIIRS to Landsat grid
            aligned_rasters = prepare.align_rasters(
                rasters=[reference_band, viirs_reprojected],
                transforms=[reference_transform, viirs_transform],
                reference_idx=0,  # Use Landsat as reference
                resampling="bilinear"  # Use bilinear for VIIRS
            )
            
            # Extract the aligned VIIRS data
            viirs_reprojected = aligned_rasters[1][0]
            viirs_transform = aligned_rasters[1][1]
            
            print(f"Aligned VIIRS shape: {viirs_reprojected.shape}")
            print(f"Aligned VIIRS non-zero pixels: {(viirs_reprojected > 0).sum()} ({(viirs_reprojected > 0).sum()/viirs_reprojected.size*100:.1f}%)")
    
    # Rasterize roads to match grid
    roads_rasterized = None
    if road_network is not None and reference_shape is not None and reference_transform is not None:
        # Use the new CRS-aware road rasterization (like sync pipeline)
        buffer_meters = 5.5  # Buffer roads by 5.5 meters
        
        print(f"Road buffering: {buffer_meters:.1f} meters (applied in UTM)")
        print(f"Rasterizing to: {target_crs}")
        print(f"Road network has {len(road_network.nodes)} nodes, {len(road_network.edges)} edges")
        
        # Use buffer_and_rasterize_roads like sync pipeline
        roads_rasterized = acquire.buffer_and_rasterize_roads(
            graph=road_network,
            bbox=bbox,
            transform=reference_transform,
            crs=target_crs,
            shape=(reference_shape[0], reference_shape[1]),
            buffer_meters=buffer_meters,
            hierarchical=True  # Use hierarchical road classification
        )
        
        # Print statistics
        road_pixels = np.sum(roads_rasterized > 0)
        print(f"Rasterized {road_pixels} pixels ({road_pixels / roads_rasterized.size * 100:.2f}%)")
        
        # Save diagnostic output if requested
        if config and getattr(config, 'save_diagnostics', False):
            road_diagnostic = roads_rasterized.astype(np.float32)
            prepare.save_diagnostic(road_diagnostic, reference_transform,
                                  "data/diagnostics/roads_rasterized.tif",
                                  "Rasterized road network",
                                  crs=target_crs)
    else:
        # No road network data
        print("  No road network data available")
        if reference_shape is not None:
            roads_rasterized = np.zeros(reference_shape, dtype='uint8')
        else:
            roads_rasterized = np.zeros((100, 100), dtype='uint8')
    
    
    # ==================== ANALYZE PHASE ====================
    print("\n📈 Calculating indices...")
    
    # Calculate spectral indices
    ndvi = None
    ndwi = None
    
    # Use the reprojected data (already in UTM) 
    if "B5" in landsat_reprojected and "B4" in landsat_reprojected:
        ndvi = analyze.calculate_ndvi(
            nir=landsat_reprojected["B5"],
            red=landsat_reprojected["B4"]
        )
        
        # Save diagnostic output if requested
        if config and getattr(config, 'save_diagnostics', False) and ndvi is not None:
            prepare.save_diagnostic(ndvi, reference_transform,
                                  "data/diagnostics/ndvi.tif",
                                  "NDVI (Normalized Difference Vegetation Index)",
                                  crs=target_crs)
    
    if "B3" in landsat_reprojected and "B5" in landsat_reprojected:
        ndwi = analyze.calculate_ndwi(
            green=landsat_reprojected["B3"],
            nir=landsat_reprojected["B5"]
        )
        
        # Save diagnostic output if requested
        if config and getattr(config, 'save_diagnostics', False) and ndwi is not None:
            prepare.save_diagnostic(ndwi, reference_transform,
                                  "data/diagnostics/ndwi.tif",
                                  "NDWI (Normalized Difference Water Index)",
                                  crs=target_crs)
    
    # For now, we only have single date VIIRS, so no temporal compositing
    ntl_composite = viirs_reprojected if viirs_reprojected is not None else None
    
    # Calculate urban index combining NTL and vegetation
    ndui = None
    if ntl_composite is not None and ndvi is not None:
        # Verify shapes match before calculating NDUI
        if ntl_composite.shape != ndvi.shape:
            print(f"WARNING: Shape mismatch after alignment!")
            print(f"  NTL shape: {ntl_composite.shape}")
            print(f"  NDVI shape: {ndvi.shape}")
            raise ValueError(
                f"NTL and NDVI shapes still don't match after alignment: "
                f"{ntl_composite.shape} vs {ndvi.shape}. This indicates a problem "
                f"with the spatial alignment step."
            )
        
        # DIAGNOSTIC: Print data statistics before NDUI
        print(f"\nBefore NDUI Calculation:")
        print(f"  NTL composite shape: {ntl_composite.shape}")
        print(f"  NTL non-zero pixels: {(ntl_composite > 0).sum()} ({(ntl_composite > 0).sum() / ntl_composite.size * 100:.1f}%)")
        print(f"  NTL range: [{np.nanmin(ntl_composite):.3f}, {np.nanmax(ntl_composite):.3f}]")
        print(f"  NDVI non-NaN pixels: {(~np.isnan(ndvi)).sum()} ({(~np.isnan(ndvi)).sum() / ndvi.size * 100:.1f}%)")
        print(f"  NDVI range: [{np.nanmin(ndvi):.3f}, {np.nanmax(ndvi):.3f}]")
        
        ndui = analyze.calculate_ndui(
            ntl=ntl_composite,
            ndvi=ndvi,
            ndwi=ndwi,  # Optional, can be None
            ntl_floor=config.analysis.ntl_floor if config and hasattr(config, 'analysis') else 0.1,
            ntl_ceiling=config.analysis.ntl_ceiling if config and hasattr(config, 'analysis') else 10.0,
            ndui_floor=config.analysis.ndui_floor if config and hasattr(config, 'analysis') else 0.02,
            epsilon=1e-6
        )
        
        # DIAGNOSTIC: Print NDUI statistics
        print(f"\nAfter NDUI Calculation:")
        print(f"  NDUI non-zero pixels: {(ndui > 0).sum()} ({(ndui > 0).sum() / ndui.size * 100:.1f}%)")
        print(f"  NDUI range: [{np.nanmin(ndui):.3f}, {np.nanmax(ndui):.3f}]")
        
        # Save diagnostic output if requested
        if config and getattr(config, 'save_diagnostics', False) and ndui is not None:
            prepare.save_diagnostic(ndui, reference_transform,
                                  "data/diagnostics/ndui_raw.tif",
                                  "NDUI before road enhancement",
                                  crs=target_crs)
        
        # Enhance NDUI with roads (following sync pipeline approach)
        if ndui is not None and roads_rasterized is not None:
            # Apply road enhancement directly to NDUI
            # Major roads (1, 4): factor 1.3
            # Minor roads (2, 3, 5, 6): factor 1.1
            ndui_enhanced = ndui.copy()
            ndui_enhanced = np.where((roads_rasterized == 1) | (roads_rasterized == 4), 
                                   ndui * 1.3, ndui_enhanced)
            ndui_enhanced = np.where((roads_rasterized == 2) | (roads_rasterized == 3) | 
                                   (roads_rasterized == 5) | (roads_rasterized == 6),
                                   ndui * 1.1, ndui_enhanced)
            # Normalize back
            ndui = ndui_enhanced / 1.3
            
            # Save diagnostic output if requested
            if config and getattr(config, 'save_diagnostics', False):
                prepare.save_diagnostic(ndui, reference_transform,
                                      "data/diagnostics/ndui_road_enhanced.tif",
                                      "NDUI after road enhancement",
                                      crs=target_crs)
    
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
        method_mapping: dict[str, Literal["percentile", "minmax", "histogram", "adaptive"]] = {
            "linear": "minmax",
            "percentile": "percentile", 
            "adaptive": "adaptive",
            "retinex": "histogram"  # Closest approximation
        }
        config_method = config.enhancement.contrast_method if config else "linear"
        method: Literal["percentile", "minmax", "histogram", "adaptive"] = method_mapping.get(config_method, "percentile")
        
        ntl_enhanced = enhance.enhance_contrast(
            data=ntl_composite,
            method=method,
            percentiles=config.enhancement.contrast_percentiles if config else (2.0, 98.0),
            output_range=config.enhancement.contrast_output_range if config else (0.0, 1.0)
        )
        
        # Save diagnostic output if requested
        if config and getattr(config, 'save_diagnostics', False):
            prepare.save_diagnostic(ntl_enhanced, reference_transform,
                                  "data/diagnostics/ntl_contrast_enhanced.tif",
                                  "NTL after contrast enhancement",
                                  crs=target_crs)
    
    # Enhance road networks in the NTL data
    ntl_roads_enhanced = None
    if ntl_enhanced is not None and roads_rasterized is not None:
        ntl_roads_enhanced = enhance.enhance_roads(
            data=ntl_enhanced,
            road_mask=roads_rasterized,
            major_factor=config.enhancement.road_major_factor if config else 1.3,
            minor_factor=config.enhancement.road_minor_factor if config else 1.1,
            custom_factors=config.enhancement.custom_road_factors if config else None
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
            road_factor=1.1
        )
    else:
        ntl_urban_enhanced = ntl_roads_enhanced
    
    # Apply color mapping for visualization
    # CRITICAL FIX: Use NDUI for colormap like sync pipeline, not NTL
    ntl_colored = None
    if ndui is not None:
        # DIAGNOSTIC: Print data before colormap
        print(f"\nBefore Colormap Application (USING NDUI):")
        print(f"  Data shape: {ndui.shape}")
        print(f"  Non-zero pixels: {(ndui > 0).sum()} ({(ndui > 0).sum() / ndui.size * 100:.1f}%)")
        print(f"  Data range: [{np.nanmin(ndui):.3f}, {np.nanmax(ndui):.3f}]")
        print(f"  Data type: {ndui.dtype}")
        
        # Apply inferno colormap to NDUI with appropriate scale range
        # Use 0.3-0.8 range to focus on urban areas (matching sync pipeline)
        ntl_colored = enhance.apply_inferno_colormap(
            data=ndui,
            scale_range=(0.3, 0.8)  # Focus colormap on urban range for better contrast
        )
        
        # DIAGNOSTIC: Print data after colormap
        print(f"\nAfter Colormap Application:")
        print(f"  Output shape: {ntl_colored.shape}")
        print(f"  Output channels: {ntl_colored.shape[0] if len(ntl_colored.shape) == 3 else 1}")
        print(f"  Output range: [{np.nanmin(ntl_colored):.3f}, {np.nanmax(ntl_colored):.3f}]")
        print(f"  Output type: {ntl_colored.dtype}")
        
        # Save diagnostic output if requested
        if config and getattr(config, 'save_diagnostics', False):
            # Save pre-colormap data (now NDUI)
            prepare.save_diagnostic(ndui, reference_transform,
                                  "data/diagnostics/ndui_final_before_colormap.tif",
                                  "Final NDUI before colormap",
                                  crs=target_crs)
    
    # Create false color composite for context (optional)
    landsat_viz = None
    if "B5" in landsat_reprojected and "B4" in landsat_reprojected and "B3" in landsat_reprojected:
        landsat_viz = enhance.create_false_color_composite(
            red_channel=landsat_reprojected["B5"],   # NIR in red
            green_channel=landsat_reprojected["B4"], # Red in green
            blue_channel=landsat_reprojected["B3"],  # Green in blue
            contrast_enhance=True,
            gamma=None  # Use default gamma
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
            check_overviews=True
        )
        
        if not validation_results["valid"]:
            print(f"Warning: Output validation failed: {validation_results['errors']}")
    
    # Generate metadata
    road_types = ["motorway", "trunk", "primary", "secondary"]
    metadata = export.create_metadata(
        data_sources={
            "landsat": {
                "files": all_landsat_files['B5'] + all_landsat_files['B4'] + all_landsat_files['B3'] + all_landsat_files['B2'],
                "date_range": f"{date.strftime('%Y-%m-%d')}",
                "sensor": "Landsat 8/9",
                "processing_level": "L2SP"
            },
            "viirs": {
                "files": viirs_files.get('gap_filled_ntl', []) if viirs_files else [],
                "date_range": f"{date.strftime('%Y-%m-%d')}",
                "sensor": "VIIRS",
                "product": "VNP46A2"
            },
            "osm": {
                "date_accessed": datetime.now().strftime('%Y-%m-%d'),
                "road_types": road_types
            }
        },
        processing_steps=[
            "cloud_masking", 
            "spatial_reprojection",
            "index_calculation",
            "urban_enhancement",
            "road_enhancement",
            "contrast_enhancement",
            "colormap_application"
        ],
        indices_calculated=["ndvi", "ndwi", "ndui"] if ndui is not None else [],
        date_processed=datetime.now(),
        bbox=bbox,
        crs=target_crs,
        resolution=target_resolution_landsat
    )
    
    # Create Cloud-Optimized GeoTIFF
    if ntl_colored is not None and reference_transform is not None:
        # Get compression setting with proper type casting
        from typing import cast
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
            overviews=config.export.overview_levels if config else [2, 4, 8, 16]
        )
    
    # Generate auxiliary outputs if requested
    if config and config.export.generate_urban_mask and urban_mask is not None and reference_transform is not None:
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
            compress=cast(Literal["deflate", "lzw"], compress_setting)
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
                compress=cast(Literal["deflate", "lzw"], compress_setting)
            )
    
    # Calculate basic statistics for the results
    statistics: dict[str, float | None] = {
        "urban_area_km2": None,
        "mean_radiance": None,
        "road_length_km": None
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
        "statistics": statistics
    }