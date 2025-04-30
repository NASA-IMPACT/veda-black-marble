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
from typing import Tuple, Dict, Any, Optional
from .coords import wkt_to_json

def subset_raster_normal(input_raster: str, output_raster: str, minX: float, maxY: float, maxX: float, minY: float, scale: float, temp_dir: str) -> None:
    """Crop and resample via two interim files within temp_dir."""
    # Define intermediate file paths within temp_dir
    temp_file1 = os.path.join(temp_dir, 'new1_subset.tif')
    temp_file2 = os.path.join(temp_dir, 'new2_subset.tif')

    # Extended BBox
    # Step 1: initial crop
    ds = gdal.Translate(temp_file1, input_raster,
                       projWin=[minX, maxY, maxX, minY],
                       resampleAlg='bilinear')
    ds = None

    # Step 2: resample
    ds = gdal.Open(temp_file1)
    # Resample to target resolution
    warp_opts = gdal.WarpOptions(
        xRes=scale,
        yRes=scale,
        resampleAlg='bilinear'
    )
    ds = gdal.Warp(temp_file2, ds, options=warp_opts)
    ds = None

    # Step 3: final crop
    ds = gdal.Open(temp_file2)
    ds = gdal.Translate(output_raster, ds,
                       projWin=[minX, maxY, maxX, minY],
                       resampleAlg='bilinear')
    ds = None

    # Clean up
    try:
        os.remove(temp_file1)
        os.remove(temp_file2)
    except OSError as e:
        print(f"Warning: Could not remove temporary file: {e}")


def subset_raster(input_raster: str, output_raster: str, minX: float, maxY: float, maxX: float, minY: float, scale: float, temp_dir: str) -> None:
    """Crop and warp directly to EPSG:4326 at given scale, using temp_dir."""
    # Define intermediate file path within temp_dir
    temp_file1 = os.path.join(temp_dir, 'new1_warp.tif')

    # Initial crop
    ds1 = gdal.Open(input_raster)
    ds2 = gdal.Translate(temp_file1, ds1,
                        projWin=[minX, maxY, maxX, minY],
                        resampleAlg='bilinear')
    ds1 = None
    ds2 = None

    # Warp to target SRS and scale
    ds = gdal.Open(temp_file1)
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
    try:
        os.remove(temp_file1)
    except OSError as e:
        print(f"Warning: Could not remove temporary file: {e}")


def convert_to_wgs84(input_path: str, output_path: str, lon1: float, lat1: float, lon2: float, lat2: float, output_dir: str) -> None:
    """Reproject raster to WGS84 and crop to lon/lat bounds, using output_dir for temp."""
    ds = gdal.Open(input_path)
    srs_wkt: str = ds.GetProjection()

    json_data: Dict[str, Any] = wkt_to_json(srs_wkt)
    src_srs_code: Optional[int] = None # Initialize
    if 'id' in json_data and 'code' in json_data['id']:
        src_srs_code = json_data['id']['code']
    else:
        # Attempt to parse from authority string if available
        # This part might need refinement based on actual WKT variations
        print(f"Warning: Could not directly find EPSG code in WKT for {input_path}. Using WKT directly.")
        # Fallback or raise error depending on requirements

    # Define intermediate file path within output_dir (or its temp subdir)
    temp_conv_path = os.path.join(output_dir, 'conv.tiff') # Place in main output dir for simplicity

    # Reproject to WGS84
    warp_opts_list = [
        '-r', 'bilinear',
        '-wm', '4000',
        '-multi',
        '-t_srs', 'EPSG:4326'
    ]
    if src_srs_code:
         warp_opts_list.extend(['-s_srs', f'EPSG:{src_srs_code}'])
    else:
         # If no code, GDAL might infer from WKT, or you might need to pass WKT explicitly
         # For simplicity, assume GDAL handles it or rely on the WKT embedded in the source file
         pass

    warp_opts = gdal.WarpOptions(gdal.ParseCommandLine(' '.join(warp_opts_list)))

    ds = gdal.Warp(temp_conv_path, ds, options=warp_opts)
    ds = None

    # Crop to provided lon/lat box
    min_x, max_x = sorted([lon1, lon2])
    min_y, max_y = sorted([lat1, lat2])
    ds = gdal.Open(temp_conv_path)
    ds = gdal.Translate(output_path, ds,
                       projWin=[min_x, max_y, max_x, min_y],
                       resampleAlg='bilinear')
    ds = None
    # Clean up intermediate file
    try:
        os.remove(temp_conv_path)
    except OSError as e:
        print(f"Warning: Could not remove temporary file: {e}")


def crop_to_bb_utm(input_path: str, output_path: str, x1: float, y1: float, x2: float, y2: float) -> None:
    """Crop a UTM-projected raster to a bounding box plus margin (1000m)."""
    ds = gdal.Open(input_path)
    min_x: float; max_x: float
    min_x, max_x = sorted([x1, x2])
    min_y: float; max_y: float
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