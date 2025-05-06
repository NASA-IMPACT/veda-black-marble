"""
VIIRS (VNP46A2) utilities:
  - conv0: simple NTL HDF→TIFF converter with clipping
  - VNP46A2Util: tile ID, H5 download & HDF→TIFF conversion
"""
# VIIRS (VNP46A2) utilities
from osgeo import gdal
import os
import requests
import shutil # Add shutil for rmtree
from datetime import date
from typing import Tuple # Add typing import

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

    def coordinates_to_tile_id(self, lat, lon) -> Tuple[int, int]: # Add return type hint
        """Return (vertical, horizontal) tile indices for lat/lon."""
        # convert to 0–360/0–180 range
        v = int((( -lat + 90.0) / 180.0) * 18)
        h = int((( lon + 180.0) / 360.0) * 36)
        return v, h

    def download_h5(self, earthdata_token: str, year: int, month: int, day: int,
                    vertical: int, horizontal: int, target_filepath: str) -> None:
        """Download the .h5 VNP46A2 file for given date and tile via NASA API
        and save it to the specified target_filepath."""
        tile = f'h{horizontal:02d}v{vertical:02d}'
        day_of_year = date(year, month, day).timetuple().tm_yday

        api = (
            f'https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/details/allData/5000/'
            f'VNP46A2/{year}/{day_of_year}'
        )
        print(f"Querying VNP46A2 API: {api}")
        resp = requests.get(api)
        resp.raise_for_status()
        items = resp.json().get('content', [])
        print(f"Found {len(items)} items for {year}-{day_of_year}")

        link = None
        target_filename_part = f'VNP46A2.A{year}{day_of_year:03d}.{tile}'
        print(f"Searching for filename containing: {target_filename_part}")
        for item in items:
            if item.get('name', '').startswith(target_filename_part):
                link = item.get('downloadsLink')
                print(f"Found matching file: {item.get('name')}")
                print(f"Download link: {link}")
                break
        if not link:
            raise RuntimeError(f'No VNP46A2 product found for tile {tile} on {year}-{day_of_year:03d}')

        session = requests.Session()
        if earthdata_token:
            session.headers.update({'Authorization': f'Bearer {earthdata_token}'})
        else:
            raise RuntimeError(
                'Earthdata token not found. Set EARTH_DATA_TOKEN environment variable.'
            )

        print(f"Downloading VNP46A2 HDF5 to {target_filepath}...")
        try:
            r = session.get(link, stream=True, timeout=300)
            r.raise_for_status()

            # Ensure the parent directory for the target file exists
            parent_dir = os.path.dirname(target_filepath)
            if parent_dir: # Only call makedirs if there's an actual parent directory path
                os.makedirs(parent_dir, exist_ok=True)

            # If target_filepath already exists (e.g., as a directory or an old file), remove it
            if os.path.exists(target_filepath):
                if os.path.isdir(target_filepath):
                    print(f"Warning: Target path {target_filepath} is a directory. Removing it to save the file.")
                    shutil.rmtree(target_filepath)
                else:
                    print(f"Warning: Target file {target_filepath} already exists. Removing it to overwrite.")
                    os.remove(target_filepath)
            
            with open(target_filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("File downloaded successfully")
        except requests.exceptions.RequestException as e:
             raise RuntimeError(f"Download failed: {e}")
        except OSError as e: # Catch file operation errors
            raise RuntimeError(f"File operation failed for {target_filepath}: {e}")

    def convert_hdf_to_tiff(self, h5_file: str, output_file: str, layer: int = 0) -> None: # Add type hints
        """Convert a specified subdataset layer of HDF to GeoTIFF."""
        # Try GDAL-based HDF to GeoTIFF conversion via subdatasets
        print(f"Converting HDF5 {h5_file} (layer {layer}) to TIFF {output_file}...")
        hdflayer = gdal.Open(h5_file, gdal.GA_ReadOnly)
        if hdflayer is not None:
            try:
                subdatasets = hdflayer.GetSubDatasets()
                if layer < 0 or layer >= len(subdatasets):
                     raise IndexError(f"Layer index {layer} out of range for subdatasets in {h5_file}")
                sub = subdatasets[layer][0]
                print(f"Using subdataset: {sub}")
                rlayer = gdal.Open(sub, gdal.GA_ReadOnly)

                # Attempt to get metadata for georeferencing
                meta = rlayer.GetMetadata_Dict()
                hNum_str = meta.get('HorizontalTileNumber')
                vNum_str = meta.get('VerticalTileNumber')

                if hNum_str is None or vNum_str is None:
                    print("Warning: Could not read tile numbers from HDF metadata. Georeferencing might be incorrect.")
                    # Fallback or default values if needed, or rely on GDAL's internal handling
                    hNum = 0
                    vNum = 0
                else:
                    hNum = int(hNum_str)
                    vNum = int(vNum_str)

                west = (10 * hNum) - 180
                north = 90 - (10 * vNum)
                east = west + 10
                south = north - 10

                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                opts = gdal.TranslateOptions(
                    gdal.ParseCommandLine(
                        f'-a_srs EPSG:4326 -a_ullr {west} {north} {east} {south}'
                    )
                )
                gdal.Translate(output_file, rlayer, options=opts)
                print("Conversion using GDAL subdatasets successful.")
                # Close datasets
                rlayer = None
                hdflayer = None
                return
            except Exception as e:
                print(f"GDAL subdataset conversion failed: {e}. Falling back to h5py/rasterio.")
                # Ensure datasets are closed on failure
                rlayer = None
                hdflayer = None
                # Fall through to h5py/rasterio fallback

        # Fallback: read HDF5 file directly with h5py + rasterio
        try:
            import h5py
            import numpy as np
            import rasterio
            from rasterio.transform import from_bounds
        except ImportError as e:
            raise RuntimeError("h5py and rasterio are required for HDF5 fallback") from e

        print("Attempting conversion using h5py and rasterio...")
        with h5py.File(h5_file, 'r') as hf:
            # Navigate the typical VNP46A2 structure
            grid_group = hf.get('HDFEOS/GRIDS/VNP_Grid_DNB')
            if grid_group is None:
                 raise RuntimeError("'HDFEOS/GRIDS/VNP_Grid_DNB' group not found in HDF file")
            data_fields = grid_group.get('Data Fields') # Note space in 'Data Fields'
            if data_fields is None:
                raise RuntimeError("'Data Fields' group not found in HDF file")

            # Determine dataset name based on layer index (common layers)
            # Layer 0: DNB_BRDF-Corrected_NTL
            # Layer 1: DNB_Lunar_Irradiance
            # Layer 2: Latest_High_Quality_Retrieval (sometimes different name/location)
            # Layer 3: Mandatory_Quality_Flag
            # Layer 4: Snow_Flag
            # ... and others
            # This mapping might need adjustment based on exact product version
            common_layers = [
                'DNB_BRDF-Corrected_NTL',
                'DNB_Lunar_Irradiance',
                'Latest_High_Quality_Retrieval', # Check exact name if needed
                'Mandatory_Quality_Flag',
                'Snow_Flag'
            ]

            if layer < 0 or layer >= len(common_layers): # Adjust if more layers are expected
                 raise IndexError(f"Layer index {layer} out of range for common VNP46A2 layers")

            ds_name = common_layers[layer]
            dataset = data_fields.get(ds_name)
            if dataset is None:
                 raise RuntimeError(f"Dataset '{ds_name}' (layer {layer}) not found in HDF file")

            arr = dataset[()]
            print(f"Read array for {ds_name}, shape: {arr.shape}, dtype: {arr.dtype}")
            # mask out extreme fill values if applicable (check _FillValue attribute)
            fill_value = dataset.attrs.get('_FillValue')
            if fill_value is not None:
                 print(f"Applying fill value: {fill_value}")
                 arr = np.where(arr == fill_value, 0, arr) # Replace fill with 0, adjust if needed
            else:
                 # Default masking for common NTL fill values if attribute missing
                 arr = np.where(arr > 65530, 0, arr)

            # Get georeferencing info from HDF attributes
            hNum_attr = hf['HDFEOS INFORMATION'].get('StructMetadata.0')
            # Need to parse this string metadata to get tile numbers reliably
            # Example parsing (may need adjustment):
            hNum = 0
            vNum = 0
            if hNum_attr:
                 meta_str = hNum_attr[()].decode('utf-8')
                 try:
                     hNum = int(meta_str.split('HorizontalTileNumber=')[1].split('\n')[0])
                     vNum = int(meta_str.split('VerticalTileNumber=')[1].split('\n')[0])
                     print(f"Read tile numbers from metadata string: H={hNum}, V={vNum}")
                 except (IndexError, ValueError):
                     print("Warning: Could not parse tile numbers from HDF metadata string. Georeferencing might be incorrect.")


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
                'transform': transform,
                'nodata': 0 # Assuming 0 is nodata after fill value replacement
            }

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(arr, 1)
            print("Conversion using h5py/rasterio successful.")