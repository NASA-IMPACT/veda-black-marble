"""
VIIRS (VNP46A2) utilities:
  - conv0: simple NTL HDF→TIFF converter with clipping
  - VNP46A2Util: tile ID, H5 download & HDF→TIFF conversion
"""

# VIIRS (VNP46A2) utilities
from osgeo import gdal
import os
import requests
import shutil  # Add shutil for rmtree
from datetime import date
# no typing import needed

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
    hNum = int(meta.get("HorizontalTileNumber", 0))
    vNum = int(meta.get("VerticalTileNumber", 0))

    west = (10 * hNum) - 180
    north = 90 - (10 * vNum)
    east = west + 10
    south = north - 10

    # WGS84 and georef
    opts = gdal.TranslateOptions(
        gdal.ParseCommandLine(f"-a_srs EPSG:4326 -a_ullr {west} {north} {east} {south}")
    )
    gdal.Translate(output_file, rlayer, options=opts)


class VNP46A2Util:
    """Download and convert VNP46A2 (VIIRS) HDF files."""

    def coordinates_to_tile_id(self, lat, lon) -> tuple[int, int]:
        """Return (vertical, horizontal) tile indices for lat/lon."""
        # convert to 0–360/0–180 range
        v = int(((-lat + 90.0) / 180.0) * 18)
        h = int(((lon + 180.0) / 360.0) * 36)
        return v, h

    def get_tiles_for_bbox(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> list[tuple[int, int]]:
        """Return all (vertical, horizontal) tile pairs that intersect the bounding box.

        Args:
            lat1, lon1: First corner coordinates
            lat2, lon2: Second corner coordinates

        Returns:
            List of (vertical, horizontal) tile index tuples
        """
        # Get min/max bounds
        min_lat = min(lat1, lat2)
        max_lat = max(lat1, lat2)
        min_lon = min(lon1, lon2)
        max_lon = max(lon1, lon2)

        # Get tile indices for all corners
        # Each tile covers 10 degrees, so we need all tiles that intersect
        v1, h1 = self.coordinates_to_tile_id(max_lat, min_lon)  # NW corner
        v2, h2 = self.coordinates_to_tile_id(min_lat, max_lon)  # SE corner

        # Generate all tiles in the rectangular region
        tiles = []
        for v in range(min(v1, v2), max(v1, v2) + 1):
            for h in range(min(h1, h2), max(h1, h2) + 1):
                # Handle wraparound for horizontal tiles (0-35)
                h_wrapped = h % 36
                tiles.append((v, h_wrapped))

        return tiles

    def download_h5(
        self,
        earthdata_token: str,
        year: int,
        month: int,
        day: int,
        vertical: int,
        horizontal: int,
        target_filepath: str,
    ) -> None:
        """Download the .h5 VNP46A2 file for given date and tile via NASA API
        and save it to the specified target_filepath."""
        tile = f"h{horizontal:02d}v{vertical:02d}"
        day_of_year = date(year, month, day).timetuple().tm_yday

        api = (
            f"https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/details/allData/5000/"
            f"VNP46A2/{year}/{day_of_year}"
        )
        print(f"Querying VNP46A2 API: {api}")
        resp = requests.get(api)
        resp.raise_for_status()
        items = resp.json().get("content", [])
        print(f"Found {len(items)} items for {year}-{day_of_year}")

        link = None
        target_filename_part = f"VNP46A2.A{year}{day_of_year:03d}.{tile}"
        print(f"Searching for filename containing: {target_filename_part}")
        for item in items:
            if item.get("name", "").startswith(target_filename_part):
                link = item.get("downloadsLink")
                print(f"Found matching file: {item.get('name')}")
                print(f"Download link: {link}")
                break
        if not link:
            raise RuntimeError(
                f"No VNP46A2 product found for tile {tile} on {year}-{day_of_year:03d}"
            )

        session = requests.Session()
        if earthdata_token:
            session.headers.update({"Authorization": f"Bearer {earthdata_token}"})
        else:
            raise RuntimeError(
                "Earthdata token not found. Set EARTHDATA_TOKEN environment variable."
            )

        print(f"Downloading VNP46A2 HDF5 to {target_filepath}...")
        try:
            r = session.get(link, stream=True, timeout=300)
            r.raise_for_status()

            # Ensure the parent directory for the target file exists
            parent_dir = os.path.dirname(target_filepath)
            if (
                parent_dir
            ):  # Only call makedirs if there's an actual parent directory path
                os.makedirs(parent_dir, exist_ok=True)

            # If target_filepath already exists (e.g., as a directory or an old file), remove it
            if os.path.exists(target_filepath):
                if os.path.isdir(target_filepath):
                    print(
                        f"Warning: Target path {target_filepath} is a directory. Removing it to save the file."
                    )
                    shutil.rmtree(target_filepath)
                else:
                    print(
                        f"Warning: Target file {target_filepath} already exists. Removing it to overwrite."
                    )
                    os.remove(target_filepath)

            with open(target_filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("File downloaded successfully")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Download failed: {e}")
        except OSError as e:  # Catch file operation errors
            raise RuntimeError(f"File operation failed for {target_filepath}: {e}")

    def convert_hdf_to_tiff(
        self, h5_file: str, output_file: str, layer: int = 0
    ) -> None:
        hdflayer = gdal.Open(h5_file, gdal.GA_ReadOnly)
        subdatasets = hdflayer.GetSubDatasets()
        sub = subdatasets[layer][0]
        rlayer = gdal.Open(sub, gdal.GA_ReadOnly)
        meta = rlayer.GetMetadata_Dict()
        hNum = int(meta.get("HorizontalTileNumber", 0))
        vNum = int(meta.get("VerticalTileNumber", 0))
        west = (10 * hNum) - 180
        north = 90 - (10 * vNum)
        east = west + 10
        south = north - 10
        opts = gdal.TranslateOptions(
            gdal.ParseCommandLine(
                f"-a_srs EPSG:4326 -a_ullr {west} {north} {east} {south}"
            )
        )
        gdal.Translate(output_file, rlayer, options=opts)


# Use rasterio implementation
print("INFO: Switching to rasterio-based VIIRS HDF5 handling")
try:
    from .vnp_rasterio import (
        convert_VNP46A2_HDF2TIFF as _rio_convert_VNP46A2_HDF2TIFF,
        VNP46A2Util as _rio_VNP46A2Util,
    )
    
    # Replace GDAL implementations
    convert_VNP46A2_HDF2TIFF = _rio_convert_VNP46A2_HDF2TIFF
    VNP46A2Util = _rio_VNP46A2Util
    
    print("INFO: Successfully loaded rasterio VIIRS implementations")
except ImportError as e:
    print(f"WARNING: Could not import rasterio VIIRS implementations: {e}")
    print("INFO: Using GDAL implementations")
