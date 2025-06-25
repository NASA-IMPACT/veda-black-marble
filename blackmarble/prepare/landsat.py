"""Landsat-specific data preparation functions.

This module provides functions for processing Landsat data in the pipeline,
including date-by-date processing with QA masking applied before temporal compositing.
"""

import os
from typing import Any, Literal
import numpy as np
import numpy.typing as npt

from .io import load_with_rasterio, create_virtual_raster
from .spatial import crop_to_bounds
from .landsat_qa import create_landsat_cloud_mask, print_qa_pixel_summary

# Type aliases
ArrayLike = npt.NDArray[Any]
BBox = tuple[float, float, float, float]


def process_landsat_date(
    bands_data: dict[str, tuple[list[str], list[str], list[str], list[str]]],
    bbox: BBox | None = None,
    apply_qa_mask: bool = True,
    dilate_pixels: int = 3,
    cloud_strategy: Literal["conservative", "moderate", "permissive"] = "moderate"
) -> dict[str, tuple[ArrayLike, dict[str, Any]]]:
    """Process all Landsat bands for a single date with QA masking.
    
    This function loads and processes multiple Landsat bands from the same
    acquisition date, applying QA masks before the data is used for temporal
    compositing. This ensures temporal consistency in cloud masking.
    
    Args:
        bands_data: Dictionary from download_all_bands_for_date containing
                   band data for a single date
        bbox: Optional bounding box for cropping (min_lon, min_lat, max_lon, max_lat)
        apply_qa_mask: Whether to apply QA-based cloud masking
        dilate_pixels: Number of pixels to dilate cloud mask
        cloud_strategy: Cloud masking strategy ("conservative", "moderate", or "permissive")
        
    Returns:
        Dictionary mapping band names to tuples of (masked_data, profile)
        
    Example:
        >>> bands_data = download_all_bands_for_date(2023, 9, 15, bbox)
        >>> processed = process_landsat_date(bands_data, bbox)
        >>> b2_masked = processed['B2'][0]  # Masked Band 2 data
    """
    result = {}
    
    # First, process QA_PIXEL if available and masking is requested
    cloud_mask = None
    qa_profile = None
    
    if apply_qa_mask and 'QA_PIXEL' in bands_data:
        qa_files, _, _, _ = bands_data['QA_PIXEL']
        
        if qa_files:
            print(f"Processing QA_PIXEL from {len(qa_files)} tiles")
            
            # Create VRT mosaic if multiple tiles
            if len(qa_files) > 1:
                qa_vrt = create_virtual_raster(
                    file_paths=qa_files,
                    output_vrt="data/temp/qa_pixel_mosaic.vrt"
                )
                qa_data, qa_profile = load_with_rasterio(qa_vrt)
            else:
                qa_data, qa_profile = load_with_rasterio(qa_files[0])
            
            # Debug: check data before cropping
            print(f"  QA data shape before crop: {qa_data.shape}")
            print(f"  QA data dtype: {qa_data.dtype}")
            
            # Debug: check bounds overlap
            if bbox:
                # Get the CRS from the profile
                src_crs = qa_profile.get('crs')
                print(f"  Data CRS: {src_crs}")
                
                # Convert bbox to data CRS if needed
                if src_crs and hasattr(src_crs, 'to_epsg'):
                    src_epsg = src_crs.to_epsg()
                    if src_epsg != 4326:  # Data is not in lat/lon
                        print(f"  Converting bbox from EPSG:4326 to EPSG:{src_epsg}")
                        # Use pyproj to transform bbox
                        try:
                            from pyproj import Transformer
                            transformer = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
                            # Transform corners
                            minx, miny = transformer.transform(bbox[0], bbox[1])
                            maxx, maxy = transformer.transform(bbox[2], bbox[3])
                            bbox_transformed = (minx, miny, maxx, maxy)
                            print(f"  Transformed bbox: {bbox_transformed}")
                        except Exception as e:
                            print(f"  WARNING: Could not transform bbox: {e}")
                            bbox_transformed = bbox
                    else:
                        bbox_transformed = bbox
                else:
                    bbox_transformed = bbox
                
                # Calculate tile bounds from transform and shape
                tile_bounds = (
                    qa_profile['transform'].c,  # west
                    qa_profile['transform'].f + qa_profile['transform'].e * qa_data.shape[0],  # south
                    qa_profile['transform'].c + qa_profile['transform'].a * qa_data.shape[1],  # east
                    qa_profile['transform'].f   # north
                )
                print(f"  Tile bounds: {tile_bounds}")
                
                # Check if bounds overlap
                overlap = (
                    bbox_transformed[0] < tile_bounds[2] and bbox_transformed[2] > tile_bounds[0] and
                    bbox_transformed[1] < tile_bounds[3] and bbox_transformed[3] > tile_bounds[1]
                )
                print(f"  Bounds overlap: {overlap}")
            
            # Crop to bbox if specified
            if bbox:
                qa_data, qa_transform, _ = crop_to_bounds(
                    data=qa_data,
                    src_transform=qa_profile['transform'],
                    bounds=bbox_transformed
                )
                qa_profile['transform'] = qa_transform
                qa_profile['height'] = qa_data.shape[0]
                qa_profile['width'] = qa_data.shape[1]
                print(f"  QA data shape after crop: {qa_data.shape}")
            
            # Print QA pixel summary for debugging
            print_qa_pixel_summary(qa_data)
            
            # Check if QA data is empty
            if qa_data.size == 0:
                print("  WARNING: QA_PIXEL data is empty after loading/cropping!")
                cloud_mask = None
            else:
                # Create cloud mask using bit-based approach
                cloud_mask = create_landsat_cloud_mask(
                    qa_pixel=qa_data,
                    strategy=cloud_strategy,
                    dilate_pixels=dilate_pixels
                )
                
                # Debug cloud mask
                mask_clear = cloud_mask.sum()
                mask_total = cloud_mask.size
                
                if mask_total > 0:
                    print(f"  Cloud mask coverage: {mask_clear / mask_total * 100:.1f}% clear")
                else:
                    print(f"  WARNING: Cloud mask is empty!")
    
    # Process each spectral band
    for band_name, (band_files, _, _, _) in bands_data.items():
        if band_name == 'QA_PIXEL' or not band_files:
            continue
            
        print(f"Processing {band_name} from {len(band_files)} tiles")
        
        # Create VRT mosaic if multiple tiles
        if len(band_files) > 1:
            band_vrt = create_virtual_raster(
                file_paths=band_files,
                output_vrt=f"data/temp/{band_name.lower()}_mosaic.vrt"
            )
            band_data, band_profile = load_with_rasterio(band_vrt)
        else:
            band_data, band_profile = load_with_rasterio(band_files[0])
        
        # Crop to bbox if specified
        if bbox:
            # Convert bbox to data CRS if needed (same as for QA)
            src_crs = band_profile.get('crs')
            if src_crs and hasattr(src_crs, 'to_epsg'):
                src_epsg = src_crs.to_epsg()
                if src_epsg != 4326:  # Data is not in lat/lon
                    try:
                        from pyproj import Transformer
                        transformer = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
                        # Transform corners
                        minx, miny = transformer.transform(bbox[0], bbox[1])
                        maxx, maxy = transformer.transform(bbox[2], bbox[3])
                        bbox_for_crop = (minx, miny, maxx, maxy)
                    except Exception as e:
                        print(f"  WARNING: Could not transform bbox for {band_name}: {e}")
                        bbox_for_crop = bbox
                else:
                    bbox_for_crop = bbox
            else:
                bbox_for_crop = bbox
            
            band_data, band_transform, _ = crop_to_bounds(
                data=band_data,
                src_transform=band_profile['transform'],
                bounds=bbox_for_crop
            )
            band_profile['transform'] = band_transform
            band_profile['height'] = band_data.shape[0]
            band_profile['width'] = band_data.shape[1]
        
        # Check if band data is empty
        if band_data.size == 0:
            print(f"  WARNING: {band_name} data is empty after loading/cropping!")
            masked_data = band_data
        elif cloud_mask is not None and apply_qa_mask:
            # Apply cloud mask if available
            # Ensure mask and data have same shape
            if cloud_mask.shape != band_data.shape:
                print(f"  Warning: Shape mismatch - band: {band_data.shape}, mask: {cloud_mask.shape}")
                # Crop to common size
                min_height = min(cloud_mask.shape[0], band_data.shape[0])
                min_width = min(cloud_mask.shape[1], band_data.shape[1])
                band_data = band_data[:min_height, :min_width]
                cloud_mask_cropped = cloud_mask[:min_height, :min_width]
            else:
                cloud_mask_cropped = cloud_mask
            
            # Apply mask: set cloudy pixels to NaN directly
            masked_data = band_data.astype(np.float32)
            masked_data[~cloud_mask_cropped] = np.nan
            clear_pixels = cloud_mask_cropped.sum()
            total_pixels = cloud_mask_cropped.size
            
            # Debug info
            if total_pixels == 0:
                print(f"  WARNING: {band_name} has zero pixels after cropping!")
                print(f"    Band shape: {band_data.shape}, Mask shape: {cloud_mask.shape}")
                print(f"    Cropped mask shape: {cloud_mask_cropped.shape}")
            else:
                print(f"  {band_name} after masking: {clear_pixels / total_pixels * 100:.1f}% valid pixels")
        else:
            masked_data = band_data
        
        result[band_name] = (masked_data, band_profile)
    
    return result