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
    road_types = config.acquisition.road_types if config else ["motorway", "trunk", "primary", "secondary"]
    custom_filter = f'["highway"~"{"_link|".join(road_types)}"]' if road_types else None
    
    road_network = acquire.fetch_road_network(
        bbox=bbox,
        network_type="all",
        custom_filter=custom_filter
    )
    
    
    # ==================== PREPARE PHASE ====================
    # Process raw data into analysis-ready format
    
    # 1. Load and scale Landsat data to reflectance
    # First, we need to load the downloaded Landsat files
    landsat_bands = {}
    for band_name, files, georef_files in [
        ("B2", blue_files, blue_files_georef),
        ("B3", green_files, green_files_georef),
        ("B4", red_files, red_files_georef),
        ("B5", nir_files, nir_files_georef),
    ]:
        if files:
            # Load with rasterio for proper handling
            data, profile = prepare.load_with_rasterio(files[0])
            # Scale to reflectance
            scaled_data = prepare.scale_landsat_bands(
                data,
                scale_factor=config.preparation.landsat_scale_factor if config else 0.0000275,
                add_offset=config.preparation.landsat_add_offset if config else -0.2
            )
            landsat_bands[band_name] = (scaled_data, profile)
    
    # Load QA band
    qa_data = None
    if qa_files:
        qa_data, qa_profile = prepare.load_with_rasterio(qa_files[0])
    
    # 2. Apply quality masks to remove clouds/shadows
    cloud_mask = None
    if qa_data is not None:
        cloud_mask = prepare.create_cloud_mask(
            qa_pixel=qa_data,
            conservative=True,  # Use conservative cloud masking
            dilate_pixels=config.preparation.qa_dilation_radius if config else 3
        )
    
    # Apply masks to all bands
    if cloud_mask is not None:
        for band_name in landsat_bands:
            data, profile = landsat_bands[band_name]
            landsat_bands[band_name] = (data * cloud_mask, profile)  # cloud_mask is True for clear pixels
    
    # 3. Process VIIRS nighttime lights
    # Load VIIRS radiance data
    viirs_corrected = None
    viirs_profile = None
    if viirs_data and viirs_data['radiance']:
        viirs_radiance, viirs_profile = prepare.load_with_rasterio(viirs_data['radiance'][0])
        
        # Apply VIIRS floor and ceiling
        viirs_clipped = prepare.apply_viirs_floor_ceiling(
            ntl_data=viirs_radiance,
            floor_value=config.analysis.ntl_floor if config else 0.1,
            ceiling_value=config.analysis.ntl_ceiling if config else 10.0
        )
        
        # Apply lunar correction if available
        if viirs_data['lunar']:
            lunar_irradiance, _ = prepare.load_with_rasterio(viirs_data['lunar'][0])
            viirs_corrected = prepare.apply_lunar_correction(
                vnp_data=viirs_clipped,
                lunar_irradiance=lunar_irradiance,
                moon_illumination_fraction=None  # Let function use default
            )
        else:
            viirs_corrected = viirs_clipped
    
    # 4. Reproject everything to common grid
    target_crs = config.preparation.target_crs if config else "EPSG:3857"
    target_resolution = config.preparation.target_resolution if config else 30.0
    
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
            dst_crs=target_crs
        )
        landsat_reprojected[band_name] = reprojected
        
        # Store reference transform and shape from first band
        if reference_transform is None:
            reference_transform = new_transform
            reference_shape = reprojected.shape
            reference_bbox = new_bbox
    
    # Reproject VIIRS to match Landsat grid
    viirs_reprojected = None
    if viirs_corrected is not None and viirs_profile is not None:
        viirs_reprojected, _, _ = prepare.reproject_image(
            data=viirs_corrected,
            src_transform=viirs_profile['transform'],
            src_crs=viirs_profile['crs'],
            dst_crs=target_crs,
            dst_transform=reference_transform,
            dst_shape=(reference_shape[0], reference_shape[1]) if reference_shape is not None else None
        )
    
    # Rasterize roads to match grid
    roads_rasterized = None
    if road_network is not None and reference_shape is not None and reference_transform is not None:
        # Convert road network graph to shapes for rasterization
        road_shapes = acquire.roads_graph_to_shapes(
            graph=road_network,
            buffer_distance=config.acquisition.osm_buffer if config else 0.00005
        )
        
        # Rasterize the road shapes to match the Landsat grid
        if road_shapes:
            roads_rasterized = acquire.rasterize_road_shapes(
                shapes=road_shapes,
                reference_transform=reference_transform,
                shape=(reference_shape[0], reference_shape[1]),
                all_touched=True  # Include all pixels touched by roads
            )
        else:
            # No roads found in the area
            roads_rasterized = np.zeros(reference_shape, dtype=np.uint8)
    
    
    # ==================== ANALYZE PHASE ====================
    # Calculate indices and create composites
    
    # 1. Calculate spectral indices
    ndvi = None
    ndwi = None
    
    if "B5" in landsat_reprojected and "B4" in landsat_reprojected:
        ndvi = analyze.calculate_ndvi(
            nir=landsat_reprojected["B5"],
            red=landsat_reprojected["B4"],
            epsilon=config.analysis.ndvi_epsilon if config else 1e-6
        )
    
    if "B3" in landsat_reprojected and "B5" in landsat_reprojected:
        ndwi = analyze.calculate_ndwi(
            green=landsat_reprojected["B3"],
            nir=landsat_reprojected["B5"],
            epsilon=config.analysis.ndwi_epsilon if config else 1e-6
        )
    
    # 2. For now, we only have single date VIIRS, so no temporal compositing
    # In a full implementation, we'd download multiple dates and composite them
    ntl_composite = viirs_reprojected if viirs_reprojected is not None else None
    
    # 3. Calculate urban index combining NTL and vegetation
    ndui = None
    if ntl_composite is not None and ndvi is not None:
        ndui = analyze.calculate_ndui(
            ntl=ntl_composite,
            ndvi=ndvi,
            ndwi=ndwi,  # Optional, can be None
            ntl_floor=config.analysis.ntl_floor if config else 0.1,
            ntl_ceiling=config.analysis.ntl_ceiling if config else 10.0,
            ndui_floor=config.analysis.ndui_floor if config else 0.02,
            epsilon=config.analysis.ndui_epsilon if config else 1e-6
        )
    
    # 4. Create urban mask from indices
    urban_mask = None
    if ndvi is not None and ndui is not None:
        # Use combine_indices_for_classification to create composite index
        # Then threshold to create urban mask
        composite_index = analyze.combine_indices_for_classification(
            ndvi=ndvi,
            ndwi=ndwi if ndwi is not None else np.zeros_like(ndvi),
            ndui=ndui,
            weights={
                "ndvi": -0.3,  # Negative weight - urban areas have low NDVI
                "ndwi": -0.2,  # Negative weight - urban areas have low NDWI
                "ndui": 0.5    # Positive weight - urban areas have high NDUI
            }
        )
        # Create binary urban mask by thresholding the composite index
        urban_mask = composite_index > 0.3
    
    
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
    ntl_colored = None
    if ntl_urban_enhanced is not None:
        # First apply inferno colormap specifically
        ntl_colored = enhance.apply_inferno_colormap(
            data=ntl_urban_enhanced,
            scale_range=config.enhancement.colormap_range if config else (0.3, 0.8)
        )
    
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
                "road_types": config.acquisition.road_types if config else []
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
        output_path="/tmp/manhattan_black_marble.tif"
    )
    
    result = black_marble_pipeline(
        bbox=config.bbox,
        date=config.date,
        output_path=config.output_path,
        config=config
    )
    
    print(f"Processing complete: {result['output_path']}")
    print(f"Urban area: {result['statistics']['urban_area_km2']} km²")