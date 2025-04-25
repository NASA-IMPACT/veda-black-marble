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

# Package imports
from blackmarble.coords import get_utm_zone
from blackmarble.landsat import process_band_data
from blackmarble.reproj import crop_to_bb_utm, convert_to_wgs84, subset_raster_normal
from blackmarble.io import get_geo, save_geotiff_single, scale_image
from blackmarble.masks import mark_cloud, mark_aerosol, mark_radsat
from blackmarble.utils import fill_gaps, rescale_bands
from blackmarble.vnp import VNP46A2Util
from blackmarble.osm import OSMUtil

def main():
    """Entry point for the Black Marble pipeline."""

    # Environment vars for S3 upload (optional)
    final_bucket = os.environ.get('UPLOAD_BUCKET')
    final_key = os.environ.get('UPLOAD_FILE')
    # AWS web identity (optional)
    if 'AWS_WEB_IDENTITY_TOKEN_FILE' in os.environ and 'AWS_WEB_IDENTITY_TOKEN' in os.environ:
        token_file = os.environ['AWS_WEB_IDENTITY_TOKEN_FILE']
        token = os.environ['AWS_WEB_IDENTITY_TOKEN']
        with open(token_file, 'w+') as f:
            f.write(token)

    earthdata_token = os.environ['EARTH_DATA_TOKEN']
    lat1 = float(os.environ['LAT1']); lon1 = float(os.environ['LONG1'])
    lat2 = float(os.environ['LAT2']); lon2 = float(os.environ['LONG2'])
    year = int(os.environ['YEAR']); month = int(os.environ['MONTH']); day = int(os.environ['DAY'])

    # Compute UTM coords
    e1, n1, _, _ = get_utm_zone(lat1, lon1)
    e2, n2, _, _ = get_utm_zone(lat2, lon2)

    # Prepare output dirs
    os.makedirs('outputs/temp', exist_ok=True)
    os.makedirs('outputs/final', exist_ok=True)

    # 1) Landsat bands
    print("Downloading and processing Landsat bands...")
    b3_out, b3_p, b3_r, b3_a = process_band_data(3, year, month, day, lat1, lon1, lat2, lon2, 'outputs/B3_Merged_')
    b4_out, b4_p, b4_r, b4_a = process_band_data(4, year, month, day, lat1, lon1, lat2, lon2, 'outputs/B4_Merged_')
    b5_out, b5_p, b5_r, b5_a = process_band_data(5, year, month, day, lat1, lon1, lat2, lon2, 'outputs/B5_Merged_')

    # 2) Crop to UTM bbox
    print("Cropping Landsat bands to UTM bounding box...")
    b3_final = []
    b4_final = []
    b5_final = []
    p_final = []
    r_final = []
    a_final = []
    for i, fn in enumerate(b3_out):
        out = f'outputs/B3_{i}.TIFF'
        crop_to_bb_utm(fn, out, e1, n1, e2, n2)
        b3_final.append(out)
    for i, fn in enumerate(b4_out):
        out = f'outputs/B4_{i}.TIFF'
        crop_to_bb_utm(fn, out, e1, n1, e2, n2)
        b4_final.append(out)
    for i, fn in enumerate(b5_out):
        out = f'outputs/B5_{i}.TIFF'
        crop_to_bb_utm(fn, out, e1, n1, e2, n2)
        b5_final.append(out)
    for i, fn in enumerate(b3_p):
        out = f'outputs/P_{i}.TIFF'
        crop_to_bb_utm(fn, out, e1, n1, e2, n2)
        p_final.append(out)
    for i, fn in enumerate(b3_r):
        out = f'outputs/R_{i}.TIFF'
        crop_to_bb_utm(fn, out, e1, n1, e2, n2)
        r_final.append(out)
    for i, fn in enumerate(b3_a):
        out = f'outputs/A_{i}.TIFF'
        crop_to_bb_utm(fn, out, e1, n1, e2, n2)
        a_final.append(out)

    # 3) Mask & Stack
    print("Masking and stacking bands...")
    b3_pxs = []
    b4_pxs = []
    b5_pxs = []
    for i in range(len(b3_final)):
        p_px, _, _ = get_geo(p_final[i], 1)
        a_px, _, _ = get_geo(a_final[i], 1)
        r_px, _, _ = get_geo(r_final[i], 1)
        c_mask = mark_cloud(p_px)
        a_mask = mark_aerosol(a_px)
        b3_r_px, b4_r_px, b5_r_px = mark_radsat(r_px)
        b3_px, _, _ = get_geo(b3_final[i], 1)
        b4_px, _, _ = get_geo(b4_final[i], 1)
        b5_px, _, _ = get_geo(b5_final[i], 1)
        b3_pxs.append(b3_px * c_mask * a_mask * b3_r_px)
        b4_pxs.append(b4_px * c_mask * a_mask * b4_r_px)
        b5_pxs.append(b5_px * c_mask * a_mask * b5_r_px)

    # Median composite
    stack3 = np.stack(b3_pxs)
    stack4 = np.stack(b4_pxs)
    stack5 = np.stack(b5_pxs)
    b3_med = np.nanmedian(np.where(stack3 == 0, np.nan, stack3), axis=0)
    b4_med = np.nanmedian(np.where(stack4 == 0, np.nan, stack4), axis=0)
    b5_med = np.nanmedian(np.where(stack5 == 0, np.nan, stack5), axis=0)

    # 4) Save merged bands
    print("Saving merged Landsat bands...")
    # get georef from first band
    _, gt, proj = get_geo(b3_final[0], 1)
    save_geotiff_single('outputs/B3.TIFF', b3_med, proj, gt)
    save_geotiff_single('outputs/B4.TIFF', b4_med, proj, gt)
    save_geotiff_single('outputs/B5.TIFF', b5_med, proj, gt)

    # 5) Reproject to WGS84
    print("Reprojecting to WGS84...")
    convert_to_wgs84('outputs/B3.TIFF', 'outputs/B3_WGS84.TIFF', lon1, lat1, lon2, lat2)
    convert_to_wgs84('outputs/B4.TIFF', 'outputs/B4_WGS84.TIFF', lon1, lat1, lon2, lat2)
    convert_to_wgs84('outputs/B5.TIFF', 'outputs/B5_WGS84.TIFF', lon1, lat1, lon2, lat2)

    # 6) Download & subset Nightlights (VIIRS)
    print("Downloading nightlights (VIIRS)...")
    vnp = VNP46A2Util()
    vert, horz = vnp.coordinates_to_tile_id(lat1, lon1)
    download_file = 'VNP46A2.h5'
    vnp.download_h5(earthdata_token, year, month, day, vert, horz, download_file)
    vnp.convert_hdf_to_tiff(download_file, 'ntl.tif', layer=0)
    # subset to area + scale
    minX, maxX = sorted([lon1, lon2])
    minY, maxY = sorted([lat1, lat2])
    scale = 0.00011111111
    subset_raster_normal('ntl.tif', 'outputs/ntl.tif', minX, maxY, maxX, minY, scale)

    # 7) Rasterize OSM roads
    print("Rasterizing OpenStreetMap roads...")
    osm = OSMUtil()
    osm.create_road_raster(lat1, lon1, lat2, lon2,
                           raster_value=1,
                           reference_raster='outputs/ntl.tif',
                           output_path='outputs/osm.tif')

    # 8) Resample all to final resolution
    print("Resampling to final resolution...")
    desired_res = (0.0001, -0.0001)
    inputs = [
        'outputs/B3_WGS84.TIFF', 'outputs/B4_WGS84.TIFF',
        'outputs/B5_WGS84.TIFF', 'outputs/ntl.tif', 'outputs/osm.tif'
    ]
    outputs = [
        'outputs/final/B3.TIFF', 'outputs/final/B4.TIFF',
        'outputs/final/B5.TIFF', 'outputs/final/ntl.tif', 'outputs/final/osm.tif'
    ]
    for inp, outp in zip(inputs, outputs):
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
    b3_px, gt, proj = get_geo(outputs[0], 1)
    b4_px, _, _ = get_geo(outputs[1], 1)
    b5_px, _, _ = get_geo(outputs[2], 1)
    osm_px, _, _ = get_geo(outputs[4], 1)
    ntl_px, _, _ = get_geo(outputs[3], 1)
    stack = fill_gaps([b3_px, b4_px, b5_px, osm_px, ntl_px])
    b3_px = rescale_bands(stack[0])
    b4_px = rescale_bands(stack[1])
    b5_px = rescale_bands(stack[2])
    osm_px = stack[3]
    ntl_px = stack[4]

    # 10) Compute indices
    print("Computing NDVI, NDWI, NDUI...")
    ndvi = (b5_px - b4_px) / (b5_px + b4_px + 1e-6)
    ndvi = np.clip(ndvi, -1.0, 1.0)
    ndwi = (b3_px - b5_px) / (b3_px + b5_px + 1e-6)
    ndwi = np.clip(ndwi, -1.0, 1.0)

    # NTL scaling
    ntl_floor = 0.1; ntl_ceiling = 10; ndui_floor = 0.02
    img_ntl = np.where(ntl_px > 65530, 0.0, ntl_px).astype(float) / 10.0
    img_ntl = np.where(img_ntl > ntl_ceiling, ntl_ceiling, img_ntl)
    img_ntl = np.where(img_ntl < ntl_floor, 0.0, img_ntl)
    img_ntl = img_ntl / ntl_ceiling

    img_ndvi = ndvi
    img_ndwi = ndwi
    img_road = osm_px

    # NDUI
    ndui = (img_ntl - img_ndvi) / (img_ntl + img_ndvi + 1e-5)
    ndui = np.clip(ndui, -1.0, 1.0)
    ndui = np.abs((ndui + 1.0)) / 2.0
    ndui = np.where(ndui < ndui_floor, 0.0, ndui)
    # road enhancement
    ndui = np.where((img_road == 1) | (img_road == 4), ndui * 1.3, ndui)
    ndui = np.where((img_road == 2) | (img_road == 3) |
                     (img_road == 5) | (img_road == 6), ndui * 1.1, ndui)
    ndui = ndui / 1.3

    # 11) Colorize & create COG
    print("Colorizing and creating cloud-optimized GeoTIFF...")
    # load bundled colormap arrays
    data_dir = str(importlib.resources.files('blackmarble').joinpath('data'))
    r = np.load(os.path.join(data_dir, 'inferno_r.npy'))
    g = np.load(os.path.join(data_dir, 'inferno_g.npy'))
    b = np.load(os.path.join(data_dir, 'inferno_b.npy'))
    r[0] = 0; g[0] = 0; b[0] = 0
    # scale
    img_scaled = scale_image(ndui, 0.3, 0.8)
    rr = r[img_scaled]; gg = g[img_scaled]; bb = b[img_scaled]

    # build 3-band array
    with rasterio.open(outputs[0]) as src:
        meta = src.meta.copy()
    meta.update(dtype='uint8', count=3, driver='GTiff', predictor=2)
    arr = np.stack([rr, gg, bb]).astype('uint8')
    out_cog = 'outputs/final/ndui-co.tif'
    with MemoryFile() as memfile:
        with memfile.open(**meta) as mem:
            mem.write(arr)
            profile = cog_profiles.get('deflate')
            cog_translate(mem, out_cog, profile, in_memory=False)

    # 12) Upload (skip if no bucket/key provided)
    if final_bucket and final_key:
        print(f"Uploading {out_cog} to s3://{final_bucket}/{final_key}...")
        s3 = boto3.client('s3')
        with open(out_cog, 'rb') as f:
            s3.upload_fileobj(f, final_bucket, final_key)
        print("Upload complete.")
    else:
        print("UPLOAD_BUCKET/UPLOAD_FILE not set; skipping S3 upload.")


if __name__ == "__main__":
    main()