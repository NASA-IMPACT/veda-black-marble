"""Pure functions for Landsat data processing - testable components.

This module contains stateless functions for Landsat data acquisition and processing,
extracted from the LandsatUtil class for better testability and functional design.
"""

import os
import boto3
import collections
from datetime import datetime
from typing import Any
import shapely.geometry
from shapely.geometry import shape, Point


# Global cache for S3 directory listings to improve performance
_DIR_CACHE: collections.OrderedDict[str, list[dict[str, str]]] = collections.OrderedDict()
_MAX_CACHE_SIZE = 100


def load_wrs_shapefile(shapefile_path: str | None = None) -> list[tuple[shapely.geometry.Polygon, int, int]]:
    """Load WRS-2 path/row polygons from shapefile.
    
    This function provides the core WRS-2 spatial lookup capability needed for
    determining which Landsat tiles cover a given geographic area. It's essential
    for the acquisition pipeline to identify appropriate data granules.
    
    Args:
        shapefile_path: Path to WRS-2 shapefile, or None for bundled default
        
    Returns:
        List of tuples: (polygon_geometry, path, row)
        
    Raises:
        ImportError: If Fiona is not available
        FileNotFoundError: If shapefile doesn't exist
    """
    try:
        import fiona
    except ImportError:
        raise ImportError("Fiona is required for WRS shapefile processing. Install with: uv add fiona")
    
    # Use bundled default if none provided
    if shapefile_path is None:
        from importlib.resources import files
        shapefile_path = str(files("blackmarble").joinpath("data", "WRS2_descending_0", "WRS2_descending.shp"))
    
    polygons = []
    
    # Read all features from the shapefile using Fiona for robust vector I/O
    with fiona.open(shapefile_path, "r") as collection:
        for feature in collection:
            # Extract WRS path and row from feature attributes
            properties = feature["properties"]
            path = int(properties["PATH"])
            row = int(properties["ROW"])
            
            # Convert GeoJSON geometry to shapely for spatial operations
            geom = shape(feature["geometry"])
            
            # Only store valid polygon geometries (skip any malformed features)
            if isinstance(geom, shapely.geometry.Polygon):
                polygons.append((geom, path, row))
    
    print(f"Loaded {len(polygons)} WRS-2 path/row polygons from shapefile")
    return polygons


def get_wrs_path_row(
    lat: float, 
    lon: float, 
    wrs_polygons: list[tuple[shapely.geometry.Polygon, int, int]] | None = None
) -> list[dict[str, int]]:
    """Find WRS-2 path/row for given coordinates.
    
    Determines which Landsat WRS-2 path/row tiles contain the specified point.
    This is crucial for the acquisition pipeline to identify the correct Landsat
    granules to download for a given geographic location.
    
    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        wrs_polygons: Pre-loaded WRS polygons, or None to load from default shapefile
        
    Returns:
        List of dicts with 'path' and 'row' keys (multiple for overlap areas)
    """
    # Load WRS polygons if not provided (allows caching by caller)
    if wrs_polygons is None:
        wrs_polygons = load_wrs_shapefile()
    
    # Create point geometry for spatial query
    point = Point(lon, lat)
    
    # Find all WRS tiles that contain this point
    results = []
    for polygon, path, row in wrs_polygons:
        if polygon.contains(point):
            results.append({"path": path, "row": row})
    
    return results


def get_wrs_tiles_for_bbox(
    min_lat: float,
    max_lat: float, 
    min_lon: float,
    max_lon: float,
    wrs_polygons: list[tuple[shapely.geometry.Polygon, int, int]] | None = None
) -> list[dict[str, int]]:
    """Find all WRS-2 path/row tiles intersecting a bounding box.
    
    Identifies all Landsat tiles needed to cover the specified geographic area.
    This function is essential for regional analysis where multiple Landsat tiles
    are required to achieve complete spatial coverage.
    
    Args:
        min_lat: Minimum latitude of bounding box
        max_lat: Maximum latitude of bounding box
        min_lon: Minimum longitude of bounding box
        max_lon: Maximum longitude of bounding box
        wrs_polygons: Pre-loaded WRS polygons, or None to load from default shapefile
        
    Returns:
        List of unique path/row dictionaries covering the bounding box
    """
    # Load WRS polygons if not provided
    if wrs_polygons is None:
        wrs_polygons = load_wrs_shapefile()
    
    # Create bounding box geometry
    bbox = shapely.geometry.box(min_lon, min_lat, max_lon, max_lat)
    
    # Find all intersecting tiles, avoiding duplicates
    results = []
    seen = set()
    
    for polygon, path, row in wrs_polygons:
        if polygon.intersects(bbox):
            key = (path, row)
            if key not in seen:
                seen.add(key)
                results.append({"path": path, "row": row})
    
    print(f"Found {len(results)} WRS tiles covering bounding box")
    return results


def find_landsat_scene_directory(
    year: int,
    month: int,
    day: int,
    path: int,
    row: int,
    collection: str = "collection02"
) -> str | None:
    """Find the S3 directory for a Landsat scene on the specified date.
    
    Searches the USGS Landsat S3 bucket for scenes matching the date and location.
    This function handles the complexity of Landsat's multi-day acquisition windows
    and returns the appropriate scene directory for download operations.
    
    Args:
        year: Target acquisition year
        month: Target acquisition month  
        day: Target acquisition day
        path: WRS-2 path
        row: WRS-2 row
        collection: Landsat collection identifier (default: collection02)
        
    Returns:
        S3 prefix for scene directory, or None if no scene found
    """
    global _DIR_CACHE
    
    # Format inputs for S3 path construction
    yr = str(year)
    mo = str(month).zfill(2)
    da = str(day).zfill(2)
    path_str = str(path).zfill(3)
    row_str = str(row).zfill(3)
    
    # Construct S3 prefix for this path/row/year combination
    prefix = f"{collection}/level-2/standard/oli-tirs/{yr}/{path_str}/{row_str}/"
    
    # Check cache first to avoid redundant S3 API calls
    if prefix in _DIR_CACHE:
        directories = _DIR_CACHE[prefix]
        print(f"Using cached directory listing for {prefix}")
    else:
        # Query S3 for available scene directories
        print(f"Querying S3 for scenes in {prefix}")
        session = boto3.Session()
        s3 = session.client("s3")
        
        try:
            resp = s3.list_objects_v2(
                Bucket="usgs-landsat",
                Prefix=prefix,
                RequestPayer="requester",
                Delimiter="/",
            )
            
            # Extract directory prefixes from S3 response
            directories = []
            common_prefixes = resp.get("CommonPrefixes", [])
            for prefix_entry in common_prefixes:
                if isinstance(prefix_entry, dict) and "Prefix" in prefix_entry:
                    directories.append({"Prefix": prefix_entry["Prefix"]})
                    
            print(f"Found {len(directories)} scene directories")
            
            # Cache the result, managing cache size
            if len(_DIR_CACHE) >= _MAX_CACHE_SIZE:
                _DIR_CACHE.popitem(last=False)  # Remove oldest entry
            _DIR_CACHE[prefix] = directories
            
        except Exception as e:
            print(f"Error querying S3: {e}")
            return None
    
    # Find scene that covers the target date
    target_date = datetime.strptime(yr + mo + da, "%Y%m%d")
    
    for entry in directories:
        try:
            # Parse scene name to extract acquisition date range
            # Format: LC08_L2SP_PPPRRR_YYYYMMDD_YYYYMMDD_02_T1/
            scene_name = entry["Prefix"].split("/")[-2]
            date_parts = scene_name.split("_")[3:5]  # YYYYMMDD start and end dates
            
            start_date = datetime.strptime(date_parts[0], "%Y%m%d")
            end_date = datetime.strptime(date_parts[1], "%Y%m%d")
            
            # Check if target date falls within scene's acquisition window
            if start_date <= target_date <= end_date:
                print(f"Selected scene: {scene_name}")
                return entry["Prefix"]
                
        except (IndexError, ValueError) as e:
            print(f"Error parsing scene name {entry.get('Prefix', 'unknown')}: {e}")
            continue
    
    print(f"No scene found for {year}-{month:02d}-{day:02d} at path {path}, row {row}")
    return None


def download_landsat_file(
    scene_directory: str,
    band_identifier: int | str,
    output_path: str,
    aws_credentials: dict[str, str] | None = None
) -> str:
    """Download a Landsat band or QA file from USGS S3 bucket.
    
    Handles the download of individual Landsat bands (1-7) or quality assessment
    files (QA_PIXEL, QA_RADSAT, QA_AEROSOL) from the AWS Open Data Program.
    This function manages the S3 key construction and download mechanics.
    
    Args:
        scene_directory: S3 prefix for the scene (from find_landsat_scene_directory)
        band_identifier: Band number (1-7) or QA layer name  
        output_path: Local file path for downloaded data
        aws_credentials: Optional AWS credentials dict (uses default chain if None)
        
    Returns:
        Path to downloaded file
        
    Raises:
        Exception: If download fails or S3 object doesn't exist
    """
    # Extract scene name from directory path
    scene_name = scene_directory.rstrip("/").split("/")[-1]
    
    # Construct S3 object key based on band type
    if band_identifier in ("QA_PIXEL", "QA_RADSAT", "QA_AEROSOL"):
        # Quality assessment layers have specific suffixes
        suffix_map = {
            "QA_PIXEL": "_QA_PIXEL.TIF",
            "QA_RADSAT": "_QA_RADSAT.TIF", 
            "QA_AEROSOL": "_SR_QA_AEROSOL.TIF",
        }
        suffix = suffix_map[str(band_identifier)]
        s3_key = scene_directory + scene_name + suffix
    else:
        # Spectral bands follow standard naming convention
        s3_key = f"{scene_directory}{scene_name}_SR_B{band_identifier}.TIF"
    
    print(f"Downloading {s3_key} to {output_path}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize S3 client with optional credentials
    session = boto3.Session()
    s3 = session.client("s3")
    
    # Download file with requester pays configuration
    try:
        s3.download_file(
            "usgs-landsat", 
            s3_key, 
            output_path, 
            ExtraArgs={"RequestPayer": "requester"}
        )
        print(f"Successfully downloaded to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Download failed: {e}")
        raise


def get_landsat_band_metadata(file_path: str) -> dict[str, Any]:
    """Extract metadata from a Landsat band file.
    
    Reads Landsat Collection 2 metadata including scaling factors, spatial reference,
    and acquisition information. This metadata is essential for proper radiometric
    processing and coordinate transformations in the pipeline.
    
    Args:
        file_path: Path to Landsat band TIFF file
        
    Returns:
        Dictionary containing scale factors, offsets, spatial reference, etc.
        
    Raises:
        ValueError: If file cannot be opened or lacks required metadata
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio is required for metadata extraction. Install with: uv add rasterio")
    
    with rasterio.open(file_path) as src:
        # Extract basic raster properties
        metadata = {
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": str(src.dtype),
            "crs": str(src.crs) if src.crs else None,
            "bounds": src.bounds,
            "transform": src.transform,
            "nodata": src.nodata,
        }
        
        # Extract Landsat Collection 2 scaling parameters
        # These are standardized for surface reflectance products
        if "SR_B" in os.path.basename(file_path):
            # Surface reflectance bands
            metadata.update({
                "scale_factor": 0.0000275,  # Collection 2 surface reflectance
                "add_offset": -0.2,
                "valid_range": [7273, 43636],  # Post-scaling: ~0.0 to 1.0
                "units": "reflectance",
                "product_type": "surface_reflectance"
            })
        elif "QA_" in os.path.basename(file_path):
            # Quality assessment layers (no scaling)
            metadata.update({
                "scale_factor": 1.0,
                "add_offset": 0.0,
                "product_type": "quality_assessment"
            })
        
        # Extract GDAL tags if available
        tags = src.tags()
        if tags:
            metadata["gdal_tags"] = tags
        
        # Parse acquisition date from filename
        basename = os.path.basename(file_path)
        if "_" in basename:
            parts = basename.split("_")
            if len(parts) >= 4:
                date_str = parts[3]
                if len(date_str) == 8 and date_str.isdigit():
                    metadata["acquisition_date"] = date_str
    
    return metadata


def calculate_landsat_cloud_cover(qa_pixel_file: str) -> float:
    """Calculate cloud cover percentage from QA_PIXEL band.
    
    Analyzes the Landsat QA_PIXEL band to determine cloud coverage percentage.
    This metric is crucial for scene selection and quality assessment in the
    pipeline, helping users choose the clearest available imagery.
    
    Args:
        qa_pixel_file: Path to QA_PIXEL.TIF file
        
    Returns:
        Cloud cover percentage (0.0 to 100.0)
    """
    try:
        import rasterio
        import numpy as np
    except ImportError:
        raise ImportError("rasterio and numpy are required. Install with: uv add rasterio numpy")
    
    with rasterio.open(qa_pixel_file) as src:
        qa_data = src.read(1)
        
        # Landsat Collection 2 QA_PIXEL bit definitions
        # Bit 3: Cloud
        # Bit 4: Cloud shadow
        cloud_bit = 3
        cloud_shadow_bit = 4
        
        # Extract cloud and cloud shadow pixels using bitwise operations
        cloud_mask = (qa_data >> cloud_bit) & 1
        cloud_shadow_mask = (qa_data >> cloud_shadow_bit) & 1
        
        # Combine cloud and shadow masks
        cloudy_pixels = (cloud_mask == 1) | (cloud_shadow_mask == 1)
        
        # Calculate valid pixel count (exclude fill/nodata)
        valid_pixels = qa_data != src.nodata if src.nodata is not None else qa_data > 0
        valid_count = np.sum(valid_pixels)
        
        if valid_count == 0:
            return 100.0  # All pixels invalid
        
        # Calculate cloud cover percentage
        cloudy_count = np.sum(cloudy_pixels & valid_pixels)
        cloud_cover = (cloudy_count / valid_count) * 100.0
        
        print(f"Cloud cover: {cloud_cover:.1f}%")
        return cloud_cover