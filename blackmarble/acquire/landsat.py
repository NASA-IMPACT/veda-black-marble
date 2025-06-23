"""Landsat data acquisition and tile selection.

This module handles downloading Landsat data from AWS, selecting appropriate
tiles based on geographic bounds, and managing the acquisition workflow.
"""

from typing import Any
from datetime import datetime
import numpy.typing as npt

# Import existing implementations
from blackmarble.landsat import download_tiles_for_band as _download_tiles_for_band
from blackmarble.landsat import download_tiles_for_band_with_composites as _download_tiles_for_band_with_composites

# For class methods, we need to handle the Fiona/OGR split
# Import the module to access the actual class being used
import blackmarble.landsat as landsat_module

# Type aliases
BBox = tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
ArrayLike = npt.NDArray[Any]


def get_wrs_path_row(lat: float, lon: float) -> list[dict[str, int]]:
    """Get Landsat WRS-2 path and row for given coordinates.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        List of dicts with 'path' and 'row' keys (can be multiple for overlap areas)
    """
    # Delegate to whichever LandsatUtil implementation is loaded
    util = landsat_module.LandsatUtil()
    return util.get_wrs(lat, lon)


def filter_landsat_scenes(
    year: int,
    month: int, 
    day: int,
    path: int,
    row: int,
    collection: str = "collection02"
) -> str | None:
    """Find Landsat scene directory for given date and path/row.
    
    Args:
        year: Target year
        month: Target month
        day: Target day
        path: WRS path
        row: WRS row
        collection: Landsat collection name
        
    Returns:
        S3 prefix for scene directory or None if not found
    """
    # This method is only available in the original OGR-based class
    # We need to create an instance of the original class directly
    # Import the base class definition
    import collections
    import boto3
    from osgeo import ogr
    import shapely.geometry
    from datetime import datetime as dt
    
    class _LandsatUtilOriginal:
        """Minimal implementation of filter_landsat_data_dir method."""
        dir_cache: collections.OrderedDict[str, list[dict[str, str]]] = (
            collections.OrderedDict()
        )
        MAX_CACHE_SIZE = 100
        
        def filter_landsat_data_dir(
            self, year: int, month: int, day: int, path: str, row: str
        ) -> str | None:
            yr: str = str(year)
            mo: str = str(month).zfill(2)
            da: str = str(day).zfill(2)
            prefix: str = f"collection02/level-2/standard/oli-tirs/{yr}/{path}/{row}/"
            directories: list[dict[str, str]]
            if prefix in self.dir_cache:
                directories = self.dir_cache[prefix]
            else:
                session = boto3.Session()
                s3 = session.client("s3")
                resp = s3.list_objects_v2(
                    Bucket="usgs-landsat",
                    Prefix=prefix,
                    RequestPayer="requester",
                    Delimiter="/",
                )
                directories = []
                common_prefixes = resp.get("CommonPrefixes", [])
                for prefix_entry in common_prefixes:
                    if isinstance(prefix_entry, dict) and "Prefix" in prefix_entry:
                        directories.append({"Prefix": prefix_entry["Prefix"]})
                if len(self.dir_cache) >= self.MAX_CACHE_SIZE:
                    self.dir_cache.popitem(last=False)
                self.dir_cache[prefix] = directories
            target: dt = dt.strptime(yr + mo + da, "%Y%m%d")
            selected: str | None = None
            for entry in directories:
                dr: list[str] = entry["Prefix"].split("/")[-2].split("_")[3:5]
                start: dt = dt.strptime(dr[0], "%Y%m%d")
                end: dt = dt.strptime(dr[1], "%Y%m%d")
                if start <= target <= end:
                    selected = entry["Prefix"]
                    break
            return selected
    
    util = _LandsatUtilOriginal()
    return util.filter_landsat_data_dir(
        year, month, day, 
        str(path).zfill(3),  # Convert to 3-digit string format
        str(row).zfill(3)    # Convert to 3-digit string format
    )


def download_landsat_band(
    scene_dir: str,
    band: int | str,
    output_path: str,
    aws_credentials: dict[str, str] | None = None
) -> str:
    """Download a specific Landsat band from AWS.
    
    Args:
        scene_dir: S3 directory containing scene
        band: Band number (1-7) or name ('QA_PIXEL', 'QA_RADSAT', 'QA_AEROSOL')
        output_path: Local path to save file
        aws_credentials: Optional AWS credentials dict
        
    Returns:
        Path to downloaded file
    """
    # This method is also only in the original class
    # We'll use a minimal implementation
    import os
    import boto3
    from boto3.session import Session
    
    scene: str = scene_dir.split("/")[-2]
    key: str
    if band in ("QA_PIXEL", "QA_RADSAT", "QA_AEROSOL"):
        suffix_map: dict[str, str] = {
            "QA_PIXEL": "_QA_PIXEL.TIF",
            "QA_RADSAT": "_QA_RADSAT.TIF",
            "QA_AEROSOL": "_SR_QA_AEROSOL.TIF",
        }
        suffix = suffix_map[str(band)]
        key = scene_dir + scene + suffix
    else:
        key = f"{scene_dir}{scene}_SR_B{band}.TIF"
    
    # Use provided AWS keys or default credential chain
    session: Session = boto3.Session()
    s3 = session.client("s3")
    
    # Ensure the directory for the output file exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    s3.download_file(
        "usgs-landsat", key, output_path, ExtraArgs={"RequestPayer": "requester"}
    )
    
    return output_path


def download_tiles_for_band(
    band: int | str,
    year: int,
    month: int,
    day: int,
    bbox: BBox,
    ignore_missing: bool = False
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Download all tiles for a band covering the given bounding box.
    
    Args:
        band: Band identifier
        year: Target year
        month: Target month  
        day: Target day
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
        ignore_missing: Whether to ignore missing tiles
        
    Returns:
        Tuple of (band_files, qa_pixel_files, qa_radsat_files, qa_aerosol_files)
    """
    # Extract bbox components and create a temp directory
    import tempfile
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # Create a temporary directory for downloads
    temp_dir = tempfile.mkdtemp(prefix="landsat_")
    
    # Delegate to existing implementation
    # Note: existing function expects lat1, lon1, lat2, lon2 format
    return _download_tiles_for_band(
        band, year, month, day,
        min_lat, min_lon,  # lat1, lon1
        max_lat, max_lon,  # lat2, lon2
        temp_dir,
        ignore_missing
    )


def download_tiles_with_composites(
    band: int | str,
    year: int,
    month: int,
    day: int,
    bbox: BBox,
    composite_history_months: int = 12
) -> list[tuple[list[str], list[str], list[str], list[str]]]:
    """Download tiles including historical composites for gap filling.
    
    Args:
        band: Band identifier
        year: Target year
        month: Target month
        day: Target day
        bbox: Bounding box
        composite_history_months: Number of months of history to include
        
    Returns:
        List of tuples, each containing (band_files, qa_pixel, qa_radsat, qa_aerosol)
    """
    # Extract bbox components and create a temp directory
    import tempfile
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # Create a temporary directory for downloads
    temp_dir = tempfile.mkdtemp(prefix="landsat_composite_")
    
    # Delegate to existing implementation
    return _download_tiles_for_band_with_composites(
        band, year, month, day,
        min_lat, min_lon,  # lat1, lon1
        max_lat, max_lon,  # lat2, lon2
        temp_dir,
        composite_history_months
    )


def get_landsat_metadata(scene_path: str) -> dict[str, Any]:
    """Extract metadata from Landsat scene.
    
    Args:
        scene_path: Path to scene file or directory
        
    Returns:
        Dictionary containing scale factors, offsets, acquisition date, etc.
    """
    # Extract metadata using GDAL
    from osgeo import gdal
    import os
    
    # Open the dataset
    ds = gdal.Open(scene_path)
    if not ds:
        raise ValueError(f"Could not open scene file: {scene_path}")
    
    # Extract basic metadata
    metadata = {
        "width": ds.RasterXSize,
        "height": ds.RasterYSize,
        "bands": ds.RasterCount,
        "projection": ds.GetProjection(),
        "geotransform": ds.GetGeoTransform(),
    }
    
    # Try to extract Landsat-specific metadata
    # These are standard for Landsat Collection 2 Level-2 products
    landsat_metadata = ds.GetMetadata()
    
    # Common Landsat scaling factors for Collection 2
    metadata.update({
        "scale_factor": 0.0000275,  # Standard for surface reflectance
        "add_offset": -0.2,         # Standard for surface reflectance
        "valid_range": [7273, 43636],  # After scaling: ~0.0 to 1.0
        "fill_value": 0,
    })
    
    # Extract acquisition date from filename if possible
    basename = os.path.basename(scene_path)
    if "_" in basename and len(basename) > 20:
        # Landsat naming convention: LC08_L2SP_PPPRRR_YYYYMMDD_...
        parts = basename.split("_")
        if len(parts) >= 4:
            date_str = parts[3]
            if len(date_str) == 8:
                metadata["acquisition_date"] = date_str
    
    # Add any additional metadata from GDAL
    if landsat_metadata:
        metadata["gdal_metadata"] = landsat_metadata
    
    ds = None  # Close dataset
    return metadata