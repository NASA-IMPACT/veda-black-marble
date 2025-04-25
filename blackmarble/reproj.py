"""
Reprojection & subsetting utilities:
  - subset_raster_normal: crop + resample via interim files
  - subset_raster: crop + warp directly to EPSG:4326
  - convert_to_wgs84: reproject to WGS84 and crop to lon/lat box
  - crop_to_bb_utm: crop a UTM-projected raster with margin
"""
# Imports
import os
from osgeo import gdal
from pyproj import Proj
from .coords import wkt_to_json

def subset_raster_normal(input_raster, output_raster, minX, maxY, maxX, minY, scale):
    """Crop and resample via two interim files (new1.tif, new2.tif)."""
    # Extended BBox
    # Step 1: initial crop
    ds = gdal.Translate('new1.tif', input_raster,
                       projWin=[minX, maxY, maxX, minY],
                       resampleAlg='bilinear')
    ds = None

    # Step 2: resample
    ds = gdal.Open('new1.tif')
    # Resample to target resolution
    warp_opts = gdal.WarpOptions(
        xRes=scale,
        yRes=scale,
        resampleAlg='bilinear'
    )
    ds = gdal.Warp('new2.tif', ds, options=warp_opts)
    ds = None

    # Step 3: final crop
    ds = gdal.Open('new2.tif')
    ds = gdal.Translate(output_raster, ds,
                       projWin=[minX, maxY, maxX, minY],
                       resampleAlg='bilinear')
    ds = None

    # Clean up
    os.remove('new1.tif')
    os.remove('new2.tif')

def subset_raster(input_raster, output_raster, minX, maxY, maxX, minY, scale):
    """Crop and warp directly to EPSG:4326 at given scale."""
    # Initial crop
    ds1 = gdal.Open(input_raster)
    ds2 = gdal.Translate('new1.tif', ds1,
                        projWin=[minX, maxY, maxX, minY],
                        resampleAlg='bilinear')
    ds1 = None
    ds2 = None

    # Warp to target SRS and scale
    ds = gdal.Open('new1.tif')
    # Warp to EPSG:4326 with target resolution
    warp_opts = gdal.WarpOptions(
        xRes=scale,
        yRes=scale,
        resampleAlg='bilinear',
        warpMemoryLimit=4000,
        multithread=True,
        dstSRS='EPSG:4326'
    )
    ds = gdal.Warp(output_raster, ds, options=warp_opts)
    ds = None
    os.remove('new1.tif')

def convert_to_wgs84(input_path, output_path, lon1, lat1, lon2, lat2):
    """Reproject raster to WGS84 and crop to lon/lat bounds."""
    ds = gdal.Open(input_path)
    srs_wkt = ds.GetProjection()

    json_data = wkt_to_json(srs_wkt)
    src_srs = json_data['id']['code']

    # Reproject to WGS84
    # Reproject to WGS84
    warp_opts = gdal.WarpOptions(
        resampleAlg='bilinear',
        warpMemoryLimit=4000,
        multithread=True,
        srcSRS=f'EPSG:{src_srs}',
        dstSRS='EPSG:4326'
    )
    ds = gdal.Warp('outputs/conv.tiff', ds, options=warp_opts)
    ds = None

    # Crop to provided lon/lat box
    min_x, max_x = sorted([lon1, lon2])
    min_y, max_y = sorted([lat1, lat2])
    ds = gdal.Open('outputs/conv.tiff')
    ds = gdal.Translate(output_path, ds,
                       projWin=[min_x, max_y, max_x, min_y],
                       resampleAlg='bilinear')
    ds = None

def crop_to_bb_utm(input_path, output_path, x1, y1, x2, y2):
    """Crop a UTM-projected raster to a bounding box plus margin (1000m)."""
    ds = gdal.Open(input_path)
    min_x, max_x = sorted([x1, x2])
    min_y, max_y = sorted([y1, y2])
    # add margin
    min_x -= 1000
    min_y -= 1000
    max_x += 1000
    max_y += 1000

    ds = gdal.Translate(output_path, ds,
                       projWin=[min_x, max_y, max_x, min_y],
                       resampleAlg='bilinear')
    ds = None