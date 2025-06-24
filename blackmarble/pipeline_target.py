"""Target pipeline structure showing the ideal flow of Black Marble processing.

This file demonstrates the clear progression through:
1. Acquire - Get the data we need
2. Prepare - Process and quality check the data
3. Analyze - Calculate indices and composites
4. Enhance - Improve visual quality and add features
5. Export - Generate final outputs

Each phase builds on the previous one in a clear, readable flow.
"""

from datetime import datetime
from typing import Any, Literal, cast
import os
import numpy as np
import numpy.typing as npt

from .target_config import PipelineConfig
from . import acquire, prepare, analyze, enhance, export

# Type aliases
ArrayLike = npt.NDArray[Any]
BBox = tuple[float, float, float, float]


def black_marble_pipeline(
    bbox: BBox,
    date: datetime,
    output_path: str,
    config: PipelineConfig | None = None
) -> dict[str, Any]:
    """Main Black Marble processing pipeline.
    
    Args:
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
        date: Target date for processing
        output_path: Where to save the final output
        config: Optional configuration overrides
        
    Returns:
        Dictionary with results and metadata
    """
    
    # ==================== ACQUIRE PHASE ====================
    # Get all the data we need from various sources
    
    # 1. Acquire Landsat data for the area and date
    # Download NIR band for NDVI calculation
    print(f"Downloading Landsat data for bbox: {bbox}")
    nir_files, nir_files_georef, _, _ = acquire.download_tiles_for_band(
        band=5,  # NIR band
        year=date.year,
        month=date.month,
        day=date.day,
        bbox=bbox
    )
    
    # Download RGB bands for visualization
    red_files, red_files_georef, _, _ = acquire.download_tiles_for_band(
        band=4,  # Red band
        year=date.year,
        month=date.month,
        day=date.day,
        bbox=bbox
    )
    
    green_files, green_files_georef, _, _ = acquire.download_tiles_for_band(
        band=3,  # Green band
        year=date.year,
        month=date.month,
        day=date.day,
        bbox=bbox
    )
    
    blue_files, blue_files_georef, _, _ = acquire.download_tiles_for_band(
        band=2,  # Blue band
        year=date.year,
        month=date.month,
        day=date.day,
        bbox=bbox
    )
    
    # Download quality bands
    qa_files, qa_files_georef, _, _ = acquire.download_tiles_for_band(
        band="QA_PIXEL",
        year=date.year,
        month=date.month,
        day=date.day,
        bbox=bbox
    )
    
    # 2. Acquire VIIRS nighttime lights data
    viirs_data = acquire.download_vnp46a2_tiles(
        date=date,
        bbox=bbox,
        output_dir="./data/viirs",
        skip_existing=True
    )
    
    # 3. Acquire OpenStreetMap road network
    # Note: fetch_road_network uses custom_filter for road types
    road_types = ["motorway", "trunk", "primary", "secondary"]
    custom_filter = f'["highway"~"{"_link|".join(road_types)}"]' if road_types else None
    
    road_network = acquire.fetch_road_network(
        bbox=bbox,
        network_type="all",
        custom_filter=custom_filter
    )
    
    
    # ==================== PREPARE PHASE ====================
    # Process raw data into analysis-ready format
    
    # Ensure temp directory exists for VRTs
    os.makedirs("data/temp", exist_ok=True)
    
    # 1. Load and scale Landsat data to reflectance
    # Use VRT for efficient lazy loading and mosaicking
    landsat_bands = {}
    landsat_vrts = {}
    
    for band_name, files, georef_files in [
        ("B2", blue_files, blue_files_georef),
        ("B3", green_files, green_files_georef),
        ("B4", red_files, red_files_georef),
        ("B5", nir_files, nir_files_georef),
    ]:
        if files:
            print(f"Creating VRT for {band_name} with {len(files)} scene(s)")
            
            # Create VRT for all scenes of this band
            if len(files) > 1:
                # Create VRT mosaic for multiple scenes
                vrt_path = prepare.create_virtual_raster(
                    file_paths=files,
                    output_vrt=f"data/temp/{band_name}_mosaic.vrt"
                )
                print(f"Created VRT mosaic: {vrt_path}")
            else:
                # Single scene - use file directly
                vrt_path = files[0]
            
            # Load the VRT/file and scale to reflectance
            data, profile = prepare.load_with_rasterio(vrt_path)
            
            # Scale to reflectance
            scaled_data = prepare.scale_landsat_bands(
                data,
                scale_factor=config.preparation.landsat_scale_factor if hasattr(config, 'preparation') else (config.landsat.band_scale_factor if hasattr(config, 'landsat') else 0.0000275),
                add_offset=config.preparation.landsat_add_offset if hasattr(config, 'preparation') else (config.landsat.band_offset if hasattr(config, 'landsat') else -0.2)
            )
            
            landsat_bands[band_name] = (scaled_data, profile)
            landsat_vrts[band_name] = vrt_path
            
            # TODO: For very large areas, consider using windowed processing:
            # from blackmarble.prepare.vrt_utils import process_vrt_by_chunks
            # This would process the VRT in chunks without loading all into memory
    
    # Load QA band - also use VRT for multiple scenes
    qa_data = None
    if qa_files:
        print(f"Creating VRT for QA with {len(qa_files)} scene(s)")
        
        if len(qa_files) > 1:
            # Create VRT for QA files
            # Note: For QA, we might want different merge strategy
            # but VRT will use "first" method by default
            qa_vrt_path = prepare.create_virtual_raster(
                file_paths=qa_files,
                output_vrt="data/temp/QA_mosaic.vrt"
            )
            print(f"Created QA VRT mosaic: {qa_vrt_path}")
        else:
            qa_vrt_path = qa_files[0]
        
        # Load QA data from VRT
        qa_data, qa_profile = prepare.load_with_rasterio(qa_vrt_path)
        
        # Note: VRT uses "first" method by default, which takes the first valid pixel
        # This is actually appropriate for QA bands in most cases
        # If we need bitwise OR, we'd need to load all scenes and process in memory
    
    # 2. Apply quality masks to remove clouds/shadows
    cloud_mask = None
    if qa_data is not None:
        cloud_mask = prepare.create_cloud_mask(
            qa_pixel=qa_data,
            conservative=True,  # Use conservative cloud masking
            dilate_pixels=config.preparation.qa_dilation_radius if hasattr(config, 'preparation') else (config.landsat.qa_dilation_radius if hasattr(config, 'landsat') else 3)
        )
    
    # Apply masks to all bands
    if cloud_mask is not None:
        for band_name in landsat_bands:
            data, profile = landsat_bands[band_name]
            landsat_bands[band_name] = (data * cloud_mask, profile)  # cloud_mask is True for clear pixels
            
            # Save diagnostic output if requested
            if config and getattr(config, 'save_diagnostics', False):
                prepare.save_diagnostic(data * cloud_mask, profile,
                                      f"data/diagnostics/landsat_{band_name}_masked.tif",
                                      f"Landsat {band_name} after cloud masking")
    
    # 3. Process VIIRS nighttime lights
    # Load VIIRS radiance data
    viirs_corrected = None
    viirs_profile = None
    if viirs_data and viirs_data['radiance']:
        # Create VRT mosaic if we have multiple VIIRS tiles
        if len(viirs_data['radiance']) > 1:
            print(f"Creating VRT mosaic for {len(viirs_data['radiance'])} VIIRS radiance tiles")
            viirs_radiance_vrt = prepare.create_virtual_raster(
                file_paths=viirs_data['radiance'],
                output_vrt="data/temp/viirs_radiance_mosaic.vrt"
            )
            viirs_radiance, viirs_profile = prepare.load_with_rasterio(viirs_radiance_vrt)
        else:
            viirs_radiance, viirs_profile = prepare.load_with_rasterio(viirs_data['radiance'][0])
        
        # CRITICAL: Crop VIIRS to bounding box like bm.py does
        # This ensures we only process the area of interest, not entire tiles
        viirs_radiance, viirs_transform, _ = prepare.crop_to_bounds(
            data=viirs_radiance,
            src_transform=viirs_profile['transform'],
            bounds=bbox
        )
        
        # CRITICAL: Resample to finer resolution like bm.py (scale = 0.00011111111)
        # This fills gaps through bilinear interpolation
        target_resolution = 0.00011111111  # ~12 meters, matching bm.py
        viirs_radiance, viirs_transform = prepare.resample_image(
            data=viirs_radiance,
            src_transform=viirs_transform,
            target_resolution=target_resolution,
            resampling='bilinear'
        )
        
        # Update profile with new shape and transform
        viirs_profile['transform'] = viirs_transform
        viirs_profile['height'] = viirs_radiance.shape[0]
        viirs_profile['width'] = viirs_radiance.shape[1]
        
        # CRITICAL: Handle nodata values like bm.py (line 1442)
        # Set values > 65530 to 0 BEFORE scaling
        viirs_radiance = np.where(viirs_radiance > 65530, 0, viirs_radiance)
        
        # Apply lunar correction BEFORE scaling (like bm.py does)
        if viirs_data['lunar'] and viirs_data.get('quality'):
            # Create VRT mosaics for lunar and quality if we have multiple tiles
            if len(viirs_data['lunar']) > 1:
                print(f"Creating VRT mosaic for {len(viirs_data['lunar'])} VIIRS lunar tiles")
                viirs_lunar_vrt = prepare.create_virtual_raster(
                    file_paths=viirs_data['lunar'],
                    output_vrt="data/temp/viirs_lunar_mosaic.vrt"
                )
                lunar_irradiance, lunar_profile = prepare.load_with_rasterio(viirs_lunar_vrt)
            else:
                lunar_irradiance, lunar_profile = prepare.load_with_rasterio(viirs_data['lunar'][0])
            
            # Crop to bounds
            lunar_irradiance, lunar_transform, _ = prepare.crop_to_bounds(
                data=lunar_irradiance,
                src_transform=lunar_profile['transform'],
                bounds=bbox
            )
            # Resample to match radiance resolution
            lunar_irradiance, lunar_transform = prepare.resample_image(
                data=lunar_irradiance,
                src_transform=lunar_transform,
                target_resolution=target_resolution,
                resampling='bilinear'
            )
            # Update profile
            lunar_profile['transform'] = lunar_transform
            lunar_profile['height'] = lunar_irradiance.shape[0]
            lunar_profile['width'] = lunar_irradiance.shape[1]
            
            if len(viirs_data['quality']) > 1:
                print(f"Creating VRT mosaic for {len(viirs_data['quality'])} VIIRS quality tiles")
                viirs_quality_vrt = prepare.create_virtual_raster(
                    file_paths=viirs_data['quality'],
                    output_vrt="data/temp/viirs_quality_mosaic.vrt"
                )
                quality_flag, quality_profile = prepare.load_with_rasterio(viirs_quality_vrt)
            else:
                quality_flag, quality_profile = prepare.load_with_rasterio(viirs_data['quality'][0])
            
            # Crop to bounds
            quality_flag, quality_transform, _ = prepare.crop_to_bounds(
                data=quality_flag,
                src_transform=quality_profile['transform'],
                bounds=bbox
            )
            # Resample to match radiance resolution
            quality_flag, quality_transform = prepare.resample_image(
                data=quality_flag,
                src_transform=quality_transform,
                target_resolution=target_resolution,
                resampling='nearest'  # Use nearest for quality flags
            )
            # Update profile
            quality_profile['transform'] = quality_transform
            quality_profile['height'] = quality_flag.shape[0]
            quality_profile['width'] = quality_flag.shape[1]
            
            # Handle lunar nodata like bm.py (line 1270)
            lunar_irradiance = np.where(lunar_irradiance > 65530, 0, lunar_irradiance)
            
            # CRITICAL QUALITY FLAG FIX - USING CORRECT LAYER
            # =================================================
            # 
            # BM.PY USES Latest_High_Quality_Retrieval (HDF5 layer 3)
            # --------------------------------------------------------
            # We discovered that bm.py uses HDF5 subdataset index 3, which is
            # "Latest_High_Quality_Retrieval", NOT "Mandatory_Quality_Flag" (index 4).
            # 
            # This layer has simple 0-255 values that become binary 0/1 after
            # bilinear resampling during the subset_raster_normal process.
            #
            # The mask logic is simple: quality == 0 means GOOD quality
            # This gives us ~50-70% coverage depending on the data.
            #
            # No bit extraction needed - just use quality == 0
            quality_mask = (quality_flag == 0).astype(float)
            
            # DIAGNOSTIC: Print quality mask statistics
            print(f"\nVIIRS Quality Mask Statistics (Latest_High_Quality_Retrieval):")
            print(f"  Total pixels: {quality_flag.size}")
            print(f"  Quality flag range: [{quality_flag.min()}, {quality_flag.max()}]")
            print(f"  Unique quality values: {len(np.unique(quality_flag))}")
            if len(np.unique(quality_flag)) <= 10:
                print(f"  Quality values: {np.unique(quality_flag)}")
            print(f"  Pixels with quality==0 (good): {quality_mask.sum()} ({quality_mask.sum() / quality_mask.size * 100:.1f}%)")
            print(f"  Non-zero VIIRS values before mask: {(viirs_radiance > 0).sum()}")
            print(f"  Non-zero VIIRS values after mask: {(viirs_radiance * quality_mask > 0).sum()}")
            
            viirs_adjusted = viirs_radiance * quality_mask - lunar_irradiance
            viirs_adjusted = np.where(viirs_adjusted < 0, 0, viirs_adjusted)
            
            # Save diagnostic output if requested
            if config and getattr(config, 'save_diagnostics', False):
                prepare.save_diagnostic(viirs_adjusted, viirs_profile, 
                                      "data/diagnostics/viirs_lunar_corrected.tif",
                                      "VIIRS after lunar correction")
        elif viirs_data['lunar']:
            # No quality flag available, simple subtraction
            if len(viirs_data['lunar']) > 1:
                print(f"Creating VRT mosaic for {len(viirs_data['lunar'])} VIIRS lunar tiles")
                viirs_lunar_vrt = prepare.create_virtual_raster(
                    file_paths=viirs_data['lunar'],
                    output_vrt="data/temp/viirs_lunar_mosaic.vrt"
                )
                lunar_irradiance, lunar_profile = prepare.load_with_rasterio(viirs_lunar_vrt)
            else:
                lunar_irradiance, lunar_profile = prepare.load_with_rasterio(viirs_data['lunar'][0])
            
            # Crop to bounds
            lunar_irradiance, lunar_transform, _ = prepare.crop_to_bounds(
                data=lunar_irradiance,
                src_transform=lunar_profile['transform'],
                bounds=bbox
            )
            # Resample to match radiance resolution
            lunar_irradiance, lunar_transform = prepare.resample_image(
                data=lunar_irradiance,
                src_transform=lunar_transform,
                target_resolution=target_resolution,
                resampling='bilinear'
            )
            # Update profile
            lunar_profile['transform'] = lunar_transform
            lunar_profile['height'] = lunar_irradiance.shape[0]
            lunar_profile['width'] = lunar_irradiance.shape[1]
            lunar_irradiance = np.where(lunar_irradiance > 65530, 0, lunar_irradiance)
            viirs_adjusted = viirs_radiance - lunar_irradiance
            viirs_adjusted = np.where(viirs_adjusted < 0, 0, viirs_adjusted)
        else:
            viirs_adjusted = viirs_radiance
        
        # NOW divide by 10 AFTER lunar correction (like bm.py line 1446)
        viirs_scaled = viirs_adjusted / 10.0
        
        # Apply VIIRS floor and ceiling
        viirs_corrected = prepare.apply_viirs_floor_ceiling(
            ntl_data=viirs_scaled,
            floor_value=config.analysis.ntl_floor if hasattr(config, 'analysis') else (config.viirs.floor if hasattr(config, 'viirs') else 0.1),
            ceiling_value=config.analysis.ntl_ceiling if hasattr(config, 'analysis') else (config.viirs.ceiling if hasattr(config, 'viirs') else 10.0)
        )
        
        # Save diagnostic output if requested
        if config and getattr(config, 'save_diagnostics', False):
            prepare.save_diagnostic(viirs_corrected, viirs_profile,
                                  "data/diagnostics/viirs_final.tif", 
                                  "VIIRS final (scaled and clipped)")
    
    # 4. Reproject everything to common grid
    # Determine appropriate UTM zone for the area
    center_lon = (bbox[0] + bbox[2]) / 2
    center_lat = (bbox[1] + bbox[3]) / 2
    
    # Calculate UTM zone (simplified - assumes northern hemisphere)
    utm_zone = int((center_lon + 180) / 6) + 1
    hemisphere = 'north' if center_lat >= 0 else 'south'
    epsg_code = 32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone
    
    # Use configured CRS or default to UTM for accurate local analysis
    target_crs = f"EPSG:{epsg_code}"  # Use calculated UTM zone
    target_resolution = 30.0  # 30 meters for Landsat
    
    # Print projection info clearly
    if False:
        print(f"Target projection: {target_crs}")
        print(f"Area center is in UTM Zone {utm_zone}{'N' if hemisphere == 'north' else 'S'} (used for road buffering)")
    else:
        print(f"Using projection: {target_crs} (UTM Zone {utm_zone}{'N' if hemisphere == 'north' else 'S'})")
    
    # Reproject Landsat bands
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
    
    # Reproject VIIRS to target CRS first
    viirs_reprojected = None
    viirs_transform = None
    if viirs_corrected is not None and viirs_profile is not None:
        viirs_reprojected, viirs_transform, _ = prepare.reproject_image(
            data=viirs_corrected,
            src_transform=viirs_profile['transform'],
            src_crs=viirs_profile['crs'],
            dst_crs=target_crs,
            resampling="bilinear"  # Use bilinear like bm.py for smoother results
        )
        
        # Align VIIRS to Landsat grid for consistent processing
        if reference_transform is not None and reference_shape is not None:
            print(f"VIIRS shape before alignment: {viirs_reprojected.shape}")
            print(f"Reference Landsat shape: {reference_shape}")
            
            # Save pre-alignment for diagnostics
            if config and getattr(config, 'save_diagnostics', False):
                prepare.save_diagnostic(viirs_reprojected, viirs_transform,
                                      "data/diagnostics/viirs_before_align.tif",
                                      "VIIRS before alignment",
                                      crs=target_crs)
            
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
            
            # Save diagnostic output if requested
            if config and getattr(config, 'save_diagnostics', False):
                prepare.save_diagnostic(viirs_reprojected, reference_transform,
                                      "data/diagnostics/viirs_aligned.tif",
                                      "VIIRS aligned to Landsat grid",
                                      crs=target_crs)
    
    # Rasterize roads to match grid
    roads_rasterized = None
    if road_network is not None and reference_shape is not None and reference_transform is not None:
        # Use the new CRS-aware road rasterization
        # This handles meter-based buffering via UTM internally
        buffer_meters = 30.0  # Buffer roads by 30 meters
        
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
            hierarchical=True  # Use hierarchical road classification
        )
        
        # Print statistics
        road_pixels = np.sum(roads_rasterized > 0)
        print(f"Rasterized {road_pixels} pixels ({road_pixels / roads_rasterized.size * 100:.2f}%)")
        
        # Save diagnostic output if requested
        if config and getattr(config, 'save_diagnostics', False):
            # Convert road raster to float for saving
            road_diagnostic = roads_rasterized.astype(np.float32)
            prepare.save_diagnostic(road_diagnostic, reference_transform,
                                  "data/diagnostics/roads_rasterized.tif",
                                  "Rasterized road network",
                                  crs=target_crs)
    
    
    # ==================== ANALYZE PHASE ====================
    # Calculate indices and create composites
    
    # 1. Calculate spectral indices
    ndvi = None
    ndwi = None
    
    if "B5" in landsat_reprojected and "B4" in landsat_reprojected:
        ndvi = analyze.calculate_ndvi(
            nir=landsat_reprojected["B5"],
            red=landsat_reprojected["B4"],
            epsilon=1e-6
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
            nir=landsat_reprojected["B5"],
            epsilon=1e-6
        )
        
        # Save diagnostic output if requested
        if config and getattr(config, 'save_diagnostics', False) and ndwi is not None:
            prepare.save_diagnostic(ndwi, reference_transform,
                                  "data/diagnostics/ndwi.tif",
                                  "NDWI (Normalized Difference Water Index)",
                                  crs=target_crs)
    
    # 2. For now, we only have single date VIIRS, so no temporal compositing
    # In a full implementation, we'd download multiple dates and composite them
    ntl_composite = viirs_reprojected if viirs_reprojected is not None else None
    
    # Save fine-resolution VIIRS as diagnostic output
    if config and getattr(config, 'save_diagnostics', False) and ntl_composite is not None:
        prepare.save_diagnostic(ntl_composite, viirs_transform,
                              "data/diagnostics/viirs_final.tif",
                              "Final processed VIIRS at fine resolution",
                              crs=target_crs)
    
    # 3. Calculate urban index combining NTL and vegetation
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
            ntl_floor=config.analysis.ntl_floor if hasattr(config, 'analysis') else (config.viirs.floor if hasattr(config, 'viirs') else 0.1),
            ntl_ceiling=config.analysis.ntl_ceiling if hasattr(config, 'analysis') else (config.viirs.ceiling if hasattr(config, 'viirs') else 10.0),
            ndui_floor=config.analysis.ndui_floor if hasattr(config, 'analysis') else (config.ndui.floor if hasattr(config, 'ndui') else 0.02),
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
        
        # Enhance NDUI with roads (following bm.py approach)
        if ndui is not None and roads_rasterized is not None:
            # Apply road enhancement directly to NDUI like bm.py does
            # Major roads (1, 4): factor 1.3
            # Minor roads (2, 3, 5, 6): factor 1.1
            ndui_enhanced = ndui.copy()
            ndui_enhanced = np.where((roads_rasterized == 1) | (roads_rasterized == 4), 
                                   ndui * 1.3, ndui_enhanced)
            ndui_enhanced = np.where((roads_rasterized == 2) | (roads_rasterized == 3) | 
                                   (roads_rasterized == 5) | (roads_rasterized == 6),
                                   ndui * 1.1, ndui_enhanced)
            # Normalize back like bm.py
            ndui = ndui_enhanced / 1.3
            
            # Save diagnostic output if requested
            if config and getattr(config, 'save_diagnostics', False):
                prepare.save_diagnostic(ndui, reference_transform,
                                      "data/diagnostics/ndui_road_enhanced.tif",
                                      "NDUI after road enhancement",
                                      crs=target_crs)
    
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
    
    # 2. Enhance road networks in the NTL data
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
    
    # 3. Enhance urban features
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
    
    # 4. Apply color mapping for visualization
    # CRITICAL FIX: Use NDUI for colormap like bm.py does, not NTL
    # The old bm.py applies colormap to NDUI (urban development index)
    # which combines nighttime lights with vegetation info for better results
    ntl_colored = None
    if ndui is not None:
        # DIAGNOSTIC: Print data before colormap
        print(f"\nBefore Colormap Application (USING NDUI):")
        print(f"  Data shape: {ndui.shape}")
        print(f"  Non-zero pixels: {(ndui > 0).sum()} ({(ndui > 0).sum() / ndui.size * 100:.1f}%)")
        print(f"  Data range: [{np.nanmin(ndui):.3f}, {np.nanmax(ndui):.3f}]")
        print(f"  Data type: {ndui.dtype}")
        
        # Apply inferno colormap to NDUI with appropriate scale range
        # NDUI is normalized to [0, 1] so use full range
        ntl_colored = enhance.apply_inferno_colormap(
            data=ndui,
            scale_range=(0.0, 1.0)  # NDUI is 0-1 range, not 0.3-0.8
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
    
    # 5. Create false color composite for context (optional)
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
            check_overviews=True
        )
        
        if not validation_results["valid"]:
            print(f"Warning: Output validation failed: {validation_results['errors']}")
    
    # 2. Generate metadata
    metadata = export.create_metadata(
        data_sources={
            "landsat": {
                "files": nir_files + red_files + green_files + blue_files,
                "date_range": f"{date.strftime('%Y-%m-%d')}",
                "sensor": "Landsat 8/9",
                "processing_level": "L2SP"
            },
            "viirs": {
                "files": viirs_data.get('radiance', []) if viirs_data else [],
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
            "lunar_correction",
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
        resolution=target_resolution
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
            overviews=config.export.overview_levels if config else [2, 4, 8, 16]
        )
    
    # 4. Generate auxiliary outputs if requested
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


if __name__ == "__main__":
    # Example usage showing the complete flow
    from .target_config import PipelineConfig
    
    # Create config using for_bm_parity method
    config = PipelineConfig.for_bm_parity(
        bbox=(-74.05, 40.69, -73.95, 40.79),  # Manhattan
        date=datetime(2023, 9, 15),
        output_path="data/outputs/manhattan_black_marble.tif"
    )
    
    result = black_marble_pipeline(
        bbox=config.bbox,
        date=config.date,
        output_path=config.output_path,
        config=config
    )
    
    print(f"Processing complete: {result['output_path']}")
    print(f"Urban area: {result['statistics']['urban_area_km2']} km²")