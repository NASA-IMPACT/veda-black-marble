#!/usr/bin/env python3
# Allow direct execution: ensure project root is on PYTHONPATH
import os, sys
# If run as script, ensure project root is on PYTHONPATH and set package name
if __name__ == '__main__' and __package__ is None:
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, pkg_root)
    __package__ = 'blackmarble'
    # Enable GDAL/OGR exceptions globally to avoid FutureWarning
    try:
        from osgeo import ogr
        ogr.UseExceptions()
    except ImportError:
        pass
"""
Pipeline orchestration for Black Marble processing.
Imports modules in this package to perform the full workflow:
  1. Download & merge Landsat bands + QA
  2. Download VIIRS nightlights + convert to TIFF
  3. Rasterize OSM roads
  4. Mask clouds, aerosol, radsat
  5. Stack, compute NDVI/NDWI/NDUI, colorize
  6. Export cloud-optimized GeoTIFF & upload
"""
import argparse
import importlib
import numpy as np
import boto3
from osgeo import gdal
import rasterio
from rasterio.io import MemoryFile
from rio_cogeo.profiles import cog_profiles
from rio_cogeo.cogeo import cog_translate
import os.path
from typing import List, Tuple, Optional, Dict, Any
from mypy_boto3_s3.client import S3Client

# Package imports
from blackmarble.coords import get_utm_zone
from blackmarble.landsat import process_band_data
from blackmarble.reproj import crop_to_bb_utm, convert_to_wgs84, subset_raster_normal
from blackmarble.io import get_geo, save_geotiff_single, scale_image
from blackmarble.masks import mark_cloud, mark_aerosol, mark_radsat
from blackmarble.utils import fill_gaps, rescale_bands
from blackmarble.vnp import VNP46A2Util
from blackmarble.osm import OSMUtil

NTL_FLOOR = 0.1
NTL_CEILING = 10
NDUI_FLOOR = 0.02

def run_pipeline(
    lat1: float, lon1: float, lat2: float, lon2: float,
    year: int, month: int, day: int,
    earthdata_token: str,
    output_dir: str = "outputs",
    upload_bucket: Optional[str] = None,
    upload_key: Optional[str] = None
) -> str:
    """
    Runs the Black Marble processing pipeline for a given area and date.

    Args:
        lat1 (float): Latitude of the first corner.
        lon1 (float): Longitude of the first corner.
        lat2 (float): Latitude of the second corner.
        lon2 (float): Longitude of the second corner.
        year (int): Target year.
        month (int): Target month.
        day (int): Target day.
        earthdata_token (str): NASA Earthdata login token.
        output_dir (str): Base directory for all outputs and temporary files. Defaults to "outputs".
        upload_bucket (Optional[str]): S3 bucket name for final upload. Defaults to None.
        upload_key (Optional[str]): S3 key (path/filename) for final upload. Defaults to None.

    Returns:
        str: The path to the final generated Cloud-Optimized GeoTIFF.

    Raises:
        RuntimeError: If critical steps like data download or processing fail.
    """
    # Define output subdirectories
    temp_dir = os.path.join(output_dir, 'temp')
    final_dir = os.path.join(output_dir, 'final')

    # Prepare output dirs
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    # Compute UTM coords
    e1: float; n1: float; zone1: int; letter1: str
    e1, n1, zone1, letter1 = get_utm_zone(lat1, lon1)
    e2: float; n2: float; zone2: int; letter2: str
    e2, n2, zone2, letter2 = get_utm_zone(lat2, lon2)
    
    # Warn if bounding box spans UTM zones
    if zone1 != zone2 or letter1 != letter2:
        print(f"Warning: Bounding box spans UTM zones ({zone1}{letter1} to {zone2}{letter2})")
        print("This may cause projection issues at zone boundaries.")
        # Continue processing - the current implementation can handle this, but results may be less accurate

    # 1) Landsat bands
    print("Downloading and processing Landsat bands...")
    b3_out: List[str]; b3_p: List[str]; b3_r: List[str]; b3_a: List[str]
    b3_out, b3_p, b3_r, b3_a = process_band_data(3, year, month, day, lat1, lon1, lat2, lon2, os.path.join(output_dir, 'B3_Merged_'), temp_dir)
    b4_out: List[str]; b4_p: List[str]; b4_r: List[str]; b4_a: List[str]
    b4_out, b4_p, b4_r, b4_a = process_band_data(4, year, month, day, lat1, lon1, lat2, lon2, os.path.join(output_dir, 'B4_Merged_'), temp_dir)
    b5_out: List[str]; b5_p: List[str]; b5_r: List[str]; b5_a: List[str]
    b5_out, b5_p, b5_r, b5_a = process_band_data(5, year, month, day, lat1, lon1, lat2, lon2, os.path.join(output_dir, 'B5_Merged_'), temp_dir)

    # 2) Crop to UTM bbox
    print("Cropping Landsat bands to UTM bounding box...")
    b3_final: List[str] = []
    b4_final: List[str] = []
    b5_final: List[str] = []
    p_final: List[str] = []
    r_final: List[str] = []
    a_final: List[str] = []
    for i, fn in enumerate(b3_out):
        out: str = os.path.join(output_dir, f'B3_{i}.TIFF')
        crop_to_bb_utm(fn, out, e1, n1, e2, n2)
        b3_final.append(out)
    for i, fn in enumerate(b4_out):
        out = os.path.join(output_dir, f'B4_{i}.TIFF')
        crop_to_bb_utm(fn, out, e1, n1, e2, n2)
        b4_final.append(out)
    for i, fn in enumerate(b5_out):
        out = os.path.join(output_dir, f'B5_{i}.TIFF')
        crop_to_bb_utm(fn, out, e1, n1, e2, n2)
        b5_final.append(out)
    for i, fn in enumerate(b3_p):
        out = os.path.join(output_dir, f'P_{i}.TIFF')
        crop_to_bb_utm(fn, out, e1, n1, e2, n2)
        p_final.append(out)
    for i, fn in enumerate(b3_r):
        out = os.path.join(output_dir, f'R_{i}.TIFF')
        crop_to_bb_utm(fn, out, e1, n1, e2, n2)
        r_final.append(out)
    for i, fn in enumerate(b3_a):
        out = os.path.join(output_dir, f'A_{i}.TIFF')
        crop_to_bb_utm(fn, out, e1, n1, e2, n2)
        a_final.append(out)

    # 3) Mask & Stack
    print("Masking and stacking bands...")
    b3_pxs: List[np.ndarray] = []
    b4_pxs: List[np.ndarray] = []
    b5_pxs: List[np.ndarray] = []
    for i in range(len(b3_final)):
        p_px: np.ndarray; _: Tuple[float, ...]; __: str
        p_px, _, _ = get_geo(p_final[i], 1)
        a_px: np.ndarray
        a_px, _, _ = get_geo(a_final[i], 1)
        r_px: np.ndarray
        r_px, _, _ = get_geo(r_final[i], 1)
        c_mask: np.ndarray = mark_cloud(p_px)
        a_mask: np.ndarray = mark_aerosol(a_px)
        b3_r_px: np.ndarray; b4_r_px: np.ndarray; b5_r_px: np.ndarray
        b3_r_px, b4_r_px, b5_r_px = mark_radsat(r_px)
        b3_px: np.ndarray
        b3_px, _, _ = get_geo(b3_final[i], 1)
        b4_px: np.ndarray
        b4_px, _, _ = get_geo(b4_final[i], 1)
        b5_px: np.ndarray
        b5_px, _, _ = get_geo(b5_final[i], 1)
        b3_pxs.append(b3_px * c_mask * a_mask * b3_r_px)
        b4_pxs.append(b4_px * c_mask * a_mask * b4_r_px)
        b5_pxs.append(b5_px * c_mask * a_mask * b5_r_px)

    # Median composite - use fill_gaps to handle different array shapes
    stack3: np.ndarray = fill_gaps(b3_pxs)
    stack4: np.ndarray = fill_gaps(b4_pxs)
    stack5: np.ndarray = fill_gaps(b5_pxs)
    b3_med: np.ndarray = np.nanmedian(np.where(stack3 == 0, np.nan, stack3), axis=0)
    b4_med: np.ndarray = np.nanmedian(np.where(stack4 == 0, np.nan, stack4), axis=0)
    b5_med: np.ndarray = np.nanmedian(np.where(stack5 == 0, np.nan, stack5), axis=0)

    # 4) Save merged bands
    print("Saving merged Landsat bands...")
    # get georef from first band
    _: np.ndarray; gt: Tuple[float, ...]; proj: str
    _, gt, proj = get_geo(b3_final[0], 1)
    b3_merged_path = os.path.join(output_dir, 'B3.TIFF')
    b4_merged_path = os.path.join(output_dir, 'B4.TIFF')
    b5_merged_path = os.path.join(output_dir, 'B5.TIFF')
    save_geotiff_single(b3_merged_path, b3_med, proj, gt)
    save_geotiff_single(b4_merged_path, b4_med, proj, gt)
    save_geotiff_single(b5_merged_path, b5_med, proj, gt)

    # 5) Reproject to WGS84
    print("Reprojecting to WGS84...")
    b3_wgs84_path = os.path.join(output_dir, 'B3_WGS84.TIFF')
    b4_wgs84_path = os.path.join(output_dir, 'B4_WGS84.TIFF')
    b5_wgs84_path = os.path.join(output_dir, 'B5_WGS84.TIFF')
    convert_to_wgs84(b3_merged_path, b3_wgs84_path, lon1, lat1, lon2, lat2, output_dir)
    convert_to_wgs84(b4_merged_path, b4_wgs84_path, lon1, lat1, lon2, lat2, output_dir)
    convert_to_wgs84(b5_merged_path, b5_wgs84_path, lon1, lat1, lon2, lat2, output_dir)

    # 6) Download & subset Nightlights (VIIRS)
    print("Downloading nightlights (VIIRS)...")
    vnp = VNP46A2Util()
    
    # Get all tiles that intersect the bounding box
    tiles = vnp.get_tiles_for_bbox(lat1, lon1, lat2, lon2)
    print(f"Found {len(tiles)} VIIRS tiles to download: {tiles}")
    
    # Download and process each tile
    ntl_tiles: List[str] = []
    lunar_tiles: List[str] = []
    quality_tiles: List[str] = []
    
    for i, (vert, horz) in enumerate(tiles):
        print(f"Processing VIIRS tile {i+1}/{len(tiles)}: h{horz:02d}v{vert:02d}")
        download_file: str = os.path.join(temp_dir, f'VNP46A2_h{horz:02d}v{vert:02d}.h5')
        
        try:
            vnp.download_h5(earthdata_token, year, month, day, vert, horz, target_filepath=download_file)
            
            # Extract all three layers from VNP46A2
            ntl_tile = os.path.join(temp_dir, f'ntl_raw_h{horz:02d}v{vert:02d}.tif')
            vnp.convert_hdf_to_tiff(download_file, ntl_tile, layer=0)  # DNB_At_Sensor_Radiance_500m
            ntl_tiles.append(ntl_tile)
            
            lunar_tile = os.path.join(temp_dir, f'lunar_irradiance_h{horz:02d}v{vert:02d}.tif')
            vnp.convert_hdf_to_tiff(download_file, lunar_tile, layer=1)  # Lunar_Irradiance
            lunar_tiles.append(lunar_tile)
            
            quality_tile = os.path.join(temp_dir, f'mandatory_quality_flag_h{horz:02d}v{vert:02d}.tif')
            vnp.convert_hdf_to_tiff(download_file, quality_tile, layer=3)  # Mandatory_Quality_Flag
            quality_tiles.append(quality_tile)
            
        except RuntimeError as e:
            print(f"Warning: Failed to download tile h{horz:02d}v{vert:02d}: {e}")
            # Continue with other tiles
    
    if not ntl_tiles:
        raise RuntimeError("No VIIRS tiles could be downloaded for the specified area and date")
    
    # Mosaic tiles if multiple, or use single tile
    if len(ntl_tiles) > 1:
        print("Mosaicing VIIRS tiles...")
        ntl_temp_tif = os.path.join(temp_dir, 'ntl_raw_mosaic.tif')
        gdal.Warp(ntl_temp_tif, ntl_tiles, format="GTiff", resampleAlg='bilinear')
        
        lunar_temp_tif = os.path.join(temp_dir, 'lunar_irradiance_mosaic.tif')
        gdal.Warp(lunar_temp_tif, lunar_tiles, format="GTiff", resampleAlg='bilinear')
        
        quality_temp_tif = os.path.join(temp_dir, 'mandatory_quality_flag_mosaic.tif')
        gdal.Warp(quality_temp_tif, quality_tiles, format="GTiff", resampleAlg='nearest')  # Use nearest for quality flags
    else:
        # Single tile, use directly
        ntl_temp_tif = ntl_tiles[0]
        lunar_temp_tif = lunar_tiles[0]
        quality_temp_tif = quality_tiles[0]
    
    # subset to area + scale
    minX: float; maxX: float
    minX, maxX = sorted([lon1, lon2])
    minY: float; maxY: float
    minY, maxY = sorted([lat1, lat2])
    scale: float = 0.00011111111
    
    # Subset all three layers
    ntl_subset_path = os.path.join(output_dir, 'ntl.tif')
    subset_raster_normal(ntl_temp_tif, ntl_subset_path, minX, maxY, maxX, minY, scale, temp_dir)
    
    lunar_subset_path = os.path.join(output_dir, 'lunar_irradiance.tif')
    subset_raster_normal(lunar_temp_tif, lunar_subset_path, minX, maxY, maxX, minY, scale, temp_dir)
    
    quality_subset_path = os.path.join(output_dir, 'mandatory_quality_flag.tif')
    subset_raster_normal(quality_temp_tif, quality_subset_path, minX, maxY, maxX, minY, scale, temp_dir)

    # 7) Rasterize OSM roads
    print("Rasterizing OpenStreetMap roads...")
    osm = OSMUtil()
    osm_path = os.path.join(output_dir, 'osm.tif')
    osm.create_road_raster(lat1, lon1, lat2, lon2,
                           reference_raster=ntl_subset_path,
                           output_path=osm_path,
                           output_dir=output_dir)

    # 8) Resample all to final resolution
    print("Resampling to final resolution...")
    desired_res: Tuple[float, float] = (0.0001, -0.0001)
    inputs: List[str] = [
        b3_wgs84_path, b4_wgs84_path, b5_wgs84_path,
        ntl_subset_path, lunar_subset_path, quality_subset_path, osm_path
    ]
    outputs_final: List[str] = [
        os.path.join(final_dir, 'B3.TIFF'), os.path.join(final_dir, 'B4.TIFF'),
        os.path.join(final_dir, 'B5.TIFF'), os.path.join(final_dir, 'ntl.tif'),
        os.path.join(final_dir, 'lunar_irradiance.tif'), os.path.join(final_dir, 'mandatory_quality_flag.tif'),
        os.path.join(final_dir, 'osm.tif')
    ]
    for inp, outp in zip(inputs, outputs_final):
        # Resample to final resolution
        warp_opts = gdal.WarpOptions(
            format='GTiff',
            xRes=desired_res[0],
            yRes=desired_res[1]
        )
        # Execute warp
        gdal.Warp(outp, inp, options=warp_opts)

    # 9) Load final rasters and stack
    print("Loading final rasters and stacking...")
    b3_px: np.ndarray; gt: Tuple[float, ...]; proj: str
    b3_px, gt, proj = get_geo(outputs_final[0], 1)
    b4_px: np.ndarray
    b4_px, _, _ = get_geo(outputs_final[1], 1)
    b5_px: np.ndarray
    b5_px, _, _ = get_geo(outputs_final[2], 1)
    ntl_px: np.ndarray
    ntl_px, _, _ = get_geo(outputs_final[3], 1)
    lunar_px: np.ndarray
    lunar_px, _, _ = get_geo(outputs_final[4], 1)
    quality_px: np.ndarray
    quality_px, _, _ = get_geo(outputs_final[5], 1)
    osm_px: np.ndarray
    osm_px, _, _ = get_geo(outputs_final[6], 1)
    
    # Apply fill_gaps to ensure all arrays have same shape
    stack: np.ndarray = fill_gaps([b3_px, b4_px, b5_px, ntl_px, lunar_px, quality_px, osm_px])
    b3_px = rescale_bands(stack[0])
    b4_px = rescale_bands(stack[1])
    b5_px = rescale_bands(stack[2])
    ntl_px = stack[3]
    lunar_px = stack[4]
    quality_px = stack[5]
    osm_px = stack[6]

    # 10) Compute indices
    print("Computing NDVI, NDWI, NDUI...")
    ndvi: np.ndarray = (b5_px - b4_px) / (b5_px + b4_px + 1e-6)
    ndvi = np.clip(ndvi, -1.0, 1.0)
    ndwi: np.ndarray = (b3_px - b5_px) / (b3_px + b5_px + 1e-6)
    ndwi = np.clip(ndwi, -1.0, 1.0)

    # NTL processing with lunar irradiance correction and quality filtering
    # Handle fill values (>65530 becomes 0)
    img_ntl: np.ndarray = np.where(ntl_px > 65530, 0.0, ntl_px).astype(float)
    img_lunar: np.ndarray = np.where(lunar_px > 65530, 0.0, lunar_px).astype(float)
    
    # Apply quality mask (keep pixels with flag == 0 (high quality) or flag == 2 (twilight))
    # Fix the original boolean logic error
    quality_mask: np.ndarray = ((quality_px == 0) | (quality_px == 2)).astype(float)
    
    # Apply lunar irradiance correction properly to avoid false negatives
    # Subtract lunar irradiance first, then apply quality mask
    adjusted_ntl: np.ndarray = (img_ntl - img_lunar) * quality_mask
    
    # Convert units (nW/cm²/sr) and apply scaling
    adjusted_ntl = adjusted_ntl / 10.0
    adjusted_ntl = np.where(adjusted_ntl > NTL_CEILING, NTL_CEILING, adjusted_ntl)
    adjusted_ntl = np.where(adjusted_ntl < NTL_FLOOR, 0.0, adjusted_ntl)
    
    # Use adjusted NTL for NDUI calculation
    img_ntl = adjusted_ntl / NTL_CEILING  # Normalize to 0-1
    img_ndvi: np.ndarray = ndvi
    img_ndwi: np.ndarray = ndwi
    img_road: np.ndarray = osm_px

    # NDUI
    ndui: np.ndarray = (img_ntl - img_ndvi) / (img_ntl + img_ndvi + 1e-6)
    ndui = np.clip(ndui, -1.0, 1.0)
    
    # Apply water mask (Task 1.3) - set water pixels to -1
    ndui = np.where(img_ndwi >= 0.0, -1.0, ndui)
    
    # Transform to 0-1 range (simplified without unnecessary abs)
    ndui = (ndui + 1.0) / 2.0
    ndui = np.where(ndui < NDUI_FLOOR, 0.0, ndui)
    
    # Road enhancement (OSM produces values 1-5 only)
    ndui = np.where((img_road == 1) | (img_road == 4), ndui * 1.3, ndui)
    ndui = np.where((img_road == 2) | (img_road == 3) | (img_road == 5), ndui * 1.1, ndui)
    
    # Task 1.2 Fix: Remove the division by 1.3 that affects all pixels
    # ndui = ndui / 1.3  # REMOVED - This was the normalization bug

    # 11) Colorize & create COG
    print("Colorizing and creating cloud-optimized GeoTIFF...")
    # load bundled colormap arrays
    data_dir: str = str(importlib.resources.files('blackmarble').joinpath('data'))
    r: np.ndarray = np.load(os.path.join(data_dir, 'inferno_r.npy'))
    g: np.ndarray = np.load(os.path.join(data_dir, 'inferno_g.npy'))
    b: np.ndarray = np.load(os.path.join(data_dir, 'inferno_b.npy'))
    r[0] = 0; g[0] = 0; b[0] = 0
    # scale
    img_scaled: np.ndarray = scale_image(ndui, 0.3, 0.8)
    rr: np.ndarray = r[img_scaled]; gg: np.ndarray = g[img_scaled]; bb: np.ndarray = b[img_scaled]

    # build 3-band array
    with rasterio.open(outputs_final[0]) as src: # Use one of the final resampled files for metadata
        meta: Dict[str, Any] = src.meta.copy()
    meta.update(dtype='uint8', count=3, driver='GTiff', predictor=2)
    arr: np.ndarray = np.stack([rr, gg, bb]).astype('uint8')
    out_cog: str = os.path.join(final_dir, 'ndui-co.tif')
    with MemoryFile() as memfile:
        with memfile.open(**meta) as mem:
            mem.write(arr)
            profile: Dict[str, Any] = cog_profiles.get('deflate')
            cog_translate(mem, out_cog, profile, in_memory=False)

    # 12) Upload (skip if no bucket/key provided)
    if upload_bucket and upload_key:
        print(f"Uploading {out_cog} to s3://{upload_bucket}/{upload_key}...")
        s3: S3Client = boto3.client('s3')
        with open(out_cog, 'rb') as f:
            s3.upload_fileobj(f, upload_bucket, upload_key)
        print("Upload complete.")
    else:
        print("UPLOAD_BUCKET/UPLOAD_FILE not set; skipping S3 upload.")

    # Clean up large intermediate files? (Optional)
    # Consider removing files in output_dir (but not final_dir or temp_dir if needed)

    return out_cog


def main() -> None:
    """Entry point for the Black Marble pipeline, reads env vars."""

    # Environment vars for S3 upload (optional)
    final_bucket: Optional[str] = os.environ.get('UPLOAD_BUCKET')
    final_key: Optional[str] = os.environ.get('UPLOAD_FILE')

    # AWS web identity (optional) - Write token to file if both env vars are set
    # This supports AWS IAM roles for service accounts (IRSA) in Kubernetes environments
    if 'AWS_WEB_IDENTITY_TOKEN_FILE' in os.environ and 'AWS_WEB_IDENTITY_TOKEN' in os.environ:
        token_file: str = os.environ['AWS_WEB_IDENTITY_TOKEN_FILE']
        token: str = os.environ['AWS_WEB_IDENTITY_TOKEN']
        # Ensure directory exists
        os.makedirs(os.path.dirname(token_file), exist_ok=True)
        with open(token_file, 'w+') as f:
            f.write(token)

    # Read required parameters from environment
    try:
        earthdata_token: str = os.environ['EARTH_DATA_TOKEN']
        lat1: float = float(os.environ['LAT1']); lon1: float = float(os.environ['LONG1'])
        lat2: float = float(os.environ['LAT2']); lon2: float = float(os.environ['LONG2'])
        year: int = int(os.environ['YEAR']); month: int = int(os.environ['MONTH']); day: int = int(os.environ['DAY'])
    except KeyError as e:
        print(f"Error: Missing required environment variable: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid format for environment variable: {e}", file=sys.stderr)
        sys.exit(1)

    # Define base output directory
    output_dir = "outputs" # Keep default for script execution

    # Call the core pipeline function
    try:
        final_cog_path = run_pipeline(
            lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2,
            year=year, month=month, day=day,
            earthdata_token=earthdata_token,
            output_dir=output_dir,
            upload_bucket=final_bucket,
            upload_key=final_key
        )
        print(f"Pipeline complete. Final output: {final_cog_path}")
    except Exception as e:
        print(f"Pipeline failed: {e}", file=sys.stderr)
        # Optionally re-raise or log traceback
        # import traceback
        # traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()