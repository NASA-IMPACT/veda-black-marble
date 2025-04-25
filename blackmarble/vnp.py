"""
VIIRS (VNP46A2) utilities:
  - conv0: simple NTL HDF→TIFF converter with clipping
  - VNP46A2Util: tile ID, H5 download & HDF→TIFF conversion
"""
# VIIRS (VNP46A2) utilities
from osgeo import gdal
import os
import requests
from datetime import date

from .io import get_geo, save_image

def conv0(input_ntl, output_ntl, gdal_type):
    """Clip extreme values in an NTL TIFF and write out as a GeoTIFF."""
    img, geotransform, proj = get_geo(input_ntl, 1)
    img = img.copy()
    img[img > 65530] = 0
    save_image(img, geotransform, proj, gdal_type, output_ntl)

def convert_VNP46A2_HDF2TIFF(h5_file, output_file):
    """Convert the third subdataset of VNP46A2 HDF to a georeferenced GeoTIFF."""
    hdflayer = gdal.Open(h5_file, gdal.GA_ReadOnly)
    subdataset = hdflayer.GetSubDatasets()[2][0]
    rlayer = gdal.Open(subdataset, gdal.GA_ReadOnly)

    meta = rlayer.GetMetadata_Dict()
    hNum = int(meta.get('HorizontalTileNumber', 0))
    vNum = int(meta.get('VerticalTileNumber', 0))

    west = (10 * hNum) - 180
    north = 90 - (10 * vNum)
    east = west + 10
    south = north - 10

    # WGS84 and georef
    opts = gdal.TranslateOptions(
        gdal.ParseCommandLine(
            f'-a_srs EPSG:4326 -a_ullr {west} {north} {east} {south}'
        )
    )
    gdal.Translate(output_file, rlayer, options=opts)

class VNP46A2Util:
    """Download and convert VNP46A2 (VIIRS) HDF files."""

    def coordinates_to_tile_id(self, lat, lon):
        """Return (vertical, horizontal) tile indices for lat/lon."""
        # convert to 0–360/0–180 range
        v = int((( -lat + 90.0) / 180.0) * 18)
        h = int((( lon + 180.0) / 360.0) * 36)
        return v, h

    def download_h5(self, earthdata_token, year, month, day,
                    vertical, horizontal, download_file='VNP46A2.h5'):
        """Download the .h5 VNP46A2 file for given date and tile via NASA API."""
        tile = f'h{horizontal:02d}v{vertical:02d}'
        # Monthly product lookup: use two-digit month as subdirectory
        api = (
            f'https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/details/allData/5000/'
            f'VNP46A2/{year}/{month:02d}'
        )
        # Query available products
        resp = requests.get(api)
        resp.raise_for_status()
        items = resp.json().get('content', [])

        link = None
        for item in items:
            # match by tile code in filename (e.g., 'h18v08')
            if item.get('name', '').split('.')[2] == tile:
                link = item.get('downloadsLink')
                break
        if not link:
            raise RuntimeError(f'No VNP46A2 product for tile {tile}')

        # Prepare HTTP session with Bearer token authentication
        session = requests.Session()
        if earthdata_token:
            session.headers.update({'Authorization': f'Bearer {earthdata_token}'})
        else:
            raise RuntimeError(
                'Earthdata token not found. Set EARTH_DATA_TOKEN environment variable.'
            )
        # Download the HDF5 file
        r = session.get(link, stream=True)
        if r.status_code != 200:
            raise RuntimeError(f'Download failed: HTTP {r.status_code}')
        with open(download_file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    def convert_hdf_to_tiff(self, h5_file, output_file, layer=0):
        """Convert a specified subdataset layer of HDF to GeoTIFF."""
        # Try GDAL-based HDF to GeoTIFF conversion via subdatasets
        hdflayer = gdal.Open(h5_file, gdal.GA_ReadOnly)
        if hdflayer is not None:
            try:
                sub = hdflayer.GetSubDatasets()[layer][0]
                rlayer = gdal.Open(sub, gdal.GA_ReadOnly)

                meta = rlayer.GetMetadata_Dict()
                hNum = int(meta.get('HorizontalTileNumber', 0))
                vNum = int(meta.get('VerticalTileNumber', 0))
                west = (10 * hNum) - 180
                north = 90 - (10 * vNum)
                east = west + 10
                south = north - 10

                opts = gdal.TranslateOptions(
                    gdal.ParseCommandLine(
                        f'-a_srs EPSG:4326 -a_ullr {west} {north} {east} {south}'
                    )
                )
                gdal.Translate(output_file, rlayer, options=opts)
                return
            except Exception:
                # Failed to use GDAL HDF driver; fall back
                pass

        # Fallback: read HDF5 file directly with h5py + rasterio
        try:
            import h5py
            import numpy as np
            import rasterio
            from rasterio.transform import from_bounds
        except ImportError as e:
            raise RuntimeError("h5py and rasterio are required for HDF5 fallback") from e

        with h5py.File(h5_file, 'r') as hf:
            all_data = hf.get('All_Data')
            if all_data is None:
                raise RuntimeError("'All_Data' group not found in HDF file")
            data_fields = all_data.get('Data_Fields')
            if data_fields is None:
                raise RuntimeError("'Data_Fields' group not found in HDF file")
            # Pick the specified layer
            field_names = list(data_fields.keys())
            if layer < 0 or layer >= len(field_names):
                raise IndexError(f"Layer index {layer} out of range")
            ds_name = field_names[layer]
            arr = data_fields[ds_name][()]
            # mask out extreme fill values
            arr = np.where(arr > 65530, 0, arr)

            hNum = int(all_data.attrs.get('HorizontalTileNumber', 0))
            vNum = int(all_data.attrs.get('VerticalTileNumber', 0))
            west = hNum * 10 - 180
            north = 90 - vNum * 10
            east = west + 10
            south = north - 10
            transform = from_bounds(west, south, east, north,
                                    arr.shape[1], arr.shape[0])
            profile = {
                'driver': 'GTiff',
                'height': arr.shape[0],
                'width': arr.shape[1],
                'count': 1,
                'dtype': arr.dtype,
                'crs': 'EPSG:4326',
                'transform': transform
            }
            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(arr, 1)