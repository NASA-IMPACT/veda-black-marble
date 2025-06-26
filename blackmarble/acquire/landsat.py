"""Landsat data acquisition and tile selection.

This module provides a pure functional API for Landsat data acquisition,
handling WRS-2 tile selection, S3 downloads, and metadata extraction.
All functions are stateless and testable, replacing the legacy LandsatUtil class.
"""

import os
import tempfile
from typing import Any
import numpy.typing as npt
from datetime import datetime, timedelta
from importlib.resources import files

import boto3
import collections
import shapely.geometry
from shapely.geometry import shape, Point
import fiona
import rasterio

# Global cache for S3 directory listings to improve performance
_DIR_CACHE: collections.OrderedDict[str, list[dict[str, str]]] = collections.OrderedDict()
_MAX_CACHE_SIZE = 100

# Type aliases
BBox = tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
ArrayLike = npt.NDArray[Any]


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
    # Fiona import handled at module level
    
    # Use bundled default if none provided
    if shapefile_path is None:
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


def get_wrs_tiles_for_point(
    lat: float, 
    lon: float, 
    wrs_polygons: list[tuple[shapely.geometry.Polygon, int, int]] | None = None
) -> list[dict[str, int]]:
    """Get Landsat WRS-2 path and row for given coordinates.
    
    Determines which Landsat WRS-2 tiles contain the specified geographic point.
    This is essential for identifying the correct Landsat data granules to acquire
    for any given location. Multiple tiles may be returned for overlap areas.
    
    Args:
        lat: Latitude in decimal degrees (-90 to 90)
        lon: Longitude in decimal degrees (-180 to 180)
        wrs_polygons: Pre-loaded WRS polygons, or None to load from default shapefile
        
    Returns:
        List of dicts with 'path' and 'row' keys (can be multiple for overlap areas)
        
    Example:
        >>> get_wrs_tiles_for_point(40.7, -74.0)  # Manhattan, NY
        [{'path': 14, 'row': 32}]
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


def get_wrs_tiles_for_bbox_tuple(
    bbox: BBox,
    wrs_polygons: list[tuple[shapely.geometry.Polygon, int, int]] | None = None
) -> list[dict[str, int]]:
    """Get all Landsat WRS-2 tiles that intersect with a bounding box.
    
    This function finds ALL tiles that have any overlap with the specified
    bounding box, not just tiles that contain specific points. This ensures
    complete coverage of the area of interest.
    
    Args:
        bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
        wrs_polygons: Pre-loaded WRS polygons, or None to load from default shapefile
        
    Returns:
        List of unique dicts with 'path' and 'row' keys
        
    Example:
        >>> get_wrs_tiles_for_bbox((-74.1, 40.6, -73.9, 40.8))  # NYC area
        [{'path': 13, 'row': 32}, {'path': 14, 'row': 32}, {'path': 14, 'row': 33}]
    """
    from shapely.geometry import box
    
    # Load WRS polygons if not provided
    if wrs_polygons is None:
        wrs_polygons = load_wrs_shapefile()
    
    # Create bounding box polygon
    min_lon, min_lat, max_lon, max_lat = bbox
    bbox_polygon = box(min_lon, min_lat, max_lon, max_lat)
    
    # Find all WRS tiles that intersect with the bounding box
    results = []
    seen_tiles = set()  # Track unique path/row combinations
    
    for polygon, path, row in wrs_polygons:
        if polygon.intersects(bbox_polygon):
            tile_key = (path, row)
            if tile_key not in seen_tiles:
                seen_tiles.add(tile_key)
                results.append({"path": path, "row": row})
    
    # Sort by path then row for consistent ordering
    results.sort(key=lambda x: (x['path'], x['row']))
    
    return results


def get_wrs_path_row(lat: float, lon: float) -> tuple[int, int]:
    """Get WRS-2 path and row for given coordinates.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Tuple of (path, row)
    """
    # Use the proper shapefile-based function
    results = get_wrs_tiles_for_point(lat, lon, None)
    
    if results:
        # Return first result as tuple
        return results[0]['path'], results[0]['row']
    else:
        # Fallback calculation for areas not covered
        # Fixed calculation based on actual WRS-2 grid
        # WRS-2 paths increase westward, rows increase southward
        path = int((184.5 - lon) % 233) + 1
        row = int((90 - lat) / 1.42) + 1
        
        # Clamp to valid ranges
        path = max(1, min(233, path))
        row = max(1, min(248, row))
        
        return path, row


def filter_landsat_scenes(
    year: int,
    month: int, 
    day: int,
    path: int,
    row: int,
    collection: str = "collection02"
) -> str | None:
    """Find Landsat scene directory for given date and path/row.
    
    Searches the USGS Landsat S3 bucket for scenes matching the specified
    acquisition date and WRS-2 coordinates. This function handles Landsat's
    multi-day acquisition windows and returns the appropriate scene directory.
    
    Args:
        year: Target acquisition year
        month: Target acquisition month (1-12)
        day: Target acquisition day (1-31)
        path: WRS-2 path (1-233)
        row: WRS-2 row (1-248)
        collection: Landsat collection identifier (default: collection02)
        
    Returns:
        S3 prefix for scene directory or None if not found
        
    Example:
        >>> filter_landsat_scenes(2023, 6, 15, 14, 32)
        'collection02/level-2/standard/oli-tirs/2023/014/032/LC08_L2SP_014032_20230615_20230621_02_T1/'
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


def download_landsat_band(
    scene_dir: str,
    band: int | str,
    output_path: str,
    aws_credentials: dict[str, str] | None = None
) -> str:
    """Download a specific Landsat band from AWS.
    
    Downloads individual Landsat bands or quality assessment files from the
    USGS Landsat S3 bucket. Supports both spectral bands (1-7) and QA layers
    for comprehensive data acquisition in the pipeline.
    
    Args:
        scene_dir: S3 directory containing scene (from filter_landsat_scenes)
        band: Band number (1-7) or QA layer ('QA_PIXEL', 'QA_RADSAT', 'QA_AEROSOL')
        output_path: Local file path for downloaded data
        aws_credentials: Optional AWS credentials dict (uses default chain if None)
        
    Returns:
        Path to downloaded file
        
    Example:
        >>> scene = filter_landsat_scenes(2023, 6, 15, 14, 32)
        >>> download_landsat_band(scene, 4, '/tmp/landsat_b4.tif')
        '/tmp/landsat_b4.tif'
    """
    # Extract scene name from directory path
    scene_name = scene_dir.rstrip("/").split("/")[-1]
    
    # Construct S3 object key based on band type
    if band in ("QA_PIXEL", "QA_RADSAT", "QA_AEROSOL"):
        # Quality assessment layers have specific suffixes
        suffix_map = {
            "QA_PIXEL": "_QA_PIXEL.TIF",
            "QA_RADSAT": "_QA_RADSAT.TIF", 
            "QA_AEROSOL": "_SR_QA_AEROSOL.TIF",
        }
        suffix = suffix_map[str(band)]
        s3_key = scene_dir + scene_name + suffix
    else:
        # Spectral bands follow standard naming convention
        s3_key = f"{scene_dir}{scene_name}_SR_B{band}.TIF"
    
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
    # Rasterio import handled at module level
    
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


def download_tiles_for_band(
    band: int | str,
    year: int,
    month: int,
    day: int,
    bbox: BBox,
    ignore_missing: bool = False,
    cache_dir: str | None = None
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Download all tiles for a band covering the given bounding box.
    
    Args:
        band: Band identifier
        year: Target year
        month: Target month  
        day: Target day
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
        ignore_missing: Whether to ignore missing tiles
        cache_dir: Optional cache directory for storing downloads
        
    Returns:
        Tuple of (band_files, qa_pixel_files, qa_radsat_files, qa_aerosol_files)
    """
    # Extract bbox components
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # Use cache directory if provided, otherwise create temporary directory
    if cache_dir:
        # Create structured cache directory: cache/landsat/YYYY/MM/DD/
        download_dir = os.path.join(cache_dir, "landsat", f"{year:04d}", f"{month:02d}", f"{day:02d}")
        os.makedirs(download_dir, exist_ok=True)
    else:
        # Use temporary directory
        download_dir = tempfile.mkdtemp(prefix="landsat_")
    
    # Get WRS path/row with validation
    wrs1_results = get_wrs_path_row(min_lat, min_lon)
    wrs2_results = get_wrs_path_row(max_lat, max_lon)
    
    if not wrs1_results:
        raise RuntimeError(
            f"No WRS path/row found for coordinates: ({min_lat}, {min_lon}). "
            "This location may be over water or outside Landsat coverage."
        )
    if not wrs2_results:
        raise RuntimeError(
            f"No WRS path/row found for coordinates: ({max_lat}, {max_lon}). "
            "This location may be over water or outside Landsat coverage."
        )
    
    wrs1 = wrs1_results[0]
    wrs2 = wrs2_results[0]
    
    # Debug: print tile range
    print(f"  Tile selection for band {band} on {year}-{month:02d}-{day:02d}:")
    print(f"    SW corner ({min_lat}, {min_lon}) -> Path {wrs1['path']}, Row {wrs1['row']}")
    print(f"    NE corner ({max_lat}, {max_lon}) -> Path {wrs2['path']}, Row {wrs2['row']}")
    
    paths = range(min(wrs1["path"], wrs2["path"]), max(wrs1["path"], wrs2["path"]) + 1)
    rows = range(min(wrs1["row"], wrs2["row"]), max(wrs1["row"], wrs2["row"]) + 1)
    
    print(f"    Path range: {list(paths)}")
    print(f"    Row range: {list(rows)}")
    print(f"    Total tiles: {len(paths) * len(rows)}")
    
    downloads: list[str] = []
    pixels: list[str] = []
    radsat: list[str] = []
    aerosol: list[str] = []
    
    for p in paths:
        for r in rows:
            # Find scene directory
            scene_dir = filter_landsat_scenes(year, month, day, p, r)
            
            if not scene_dir:
                if ignore_missing:
                    print(
                        f"Warning: No data found for tile P{p:03d}R{r:03d} on {year}-{month}-{day}. Skipping."
                    )
                    continue
                else:
                    raise RuntimeError(
                        f"No data for tile P{p:03d}R{r:03d} on {year}-{month}-{day}"
                    )
            
            base = scene_dir.split("/")[-2]
            
            # Download band file
            out_src = os.path.join(download_dir, f"{base}_SR_B{band}.TIF")
            if not os.path.exists(out_src):
                print(f"Downloading {out_src}...")
                download_landsat_band(scene_dir, band, out_src)
            else:
                print(f"Using cached file: {out_src}")
            downloads.append(out_src)
            
            # Download QA files
            qa_types = {
                "QA_PIXEL": ("_QA_PIXEL.TIF", pixels),
                "QA_RADSAT": ("_QA_RADSAT.TIF", radsat),
                "QA_AEROSOL": ("_SR_QA_AEROSOL.TIF", aerosol),
            }
            
            for qa_key, (suffix, target_list) in qa_types.items():
                out_qa = os.path.join(download_dir, f"{base}{suffix}")
                if not os.path.exists(out_qa):
                    print(f"Downloading {out_qa}...")
                    download_landsat_band(scene_dir, qa_key, out_qa)
                else:
                    print(f"Using cached file: {out_qa}")
                target_list.append(out_qa)
    
    return downloads, pixels, radsat, aerosol




def download_all_bands_for_date(
    year: int,
    month: int,
    day: int,
    bbox: BBox,
    bands: list[int | str] | None = None,
    cache_dir: str | None = None,
    ignore_missing: bool = False
) -> dict[str, tuple[list[str], list[str], list[str], list[str]]]:
    """Download all specified bands for a single date.
    
    Downloads multiple Landsat bands for the same acquisition date, ensuring
    temporal consistency when applying QA masks. This function is essential
    for the date-by-date processing approach where QA masks are applied
    before temporal compositing.
    
    Args:
        year: Target year
        month: Target month
        day: Target day
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
        bands: List of bands to download (default: [2,3,4,5,'QA_PIXEL'])
        cache_dir: Optional cache directory for storing downloads
        ignore_missing: Whether to ignore missing tiles
        
    Returns:
        Dictionary mapping band names to tuples of (band_files, qa_pixel, qa_radsat, qa_aerosol)
        
    Example:
        >>> bands_data = download_all_bands_for_date(2023, 9, 15, bbox, bands=[2,3,4,5,'QA_PIXEL'])
        >>> b2_files = bands_data['B2'][0]
        >>> qa_files = bands_data['QA_PIXEL'][0]
    """
    if bands is None:
        bands = [2, 3, 4, 5, 'QA_PIXEL']
    
    result = {}
    
    for band in bands:
        band_key = f'B{band}' if isinstance(band, int) else band
        print(f"Downloading {band_key} for {year}-{month:02d}-{day:02d}")
        
        try:
            # Download this band for the specified date
            band_data = download_tiles_for_band(
                band=band,
                year=year,
                month=month,
                day=day,
                bbox=bbox,
                ignore_missing=ignore_missing,
                cache_dir=cache_dir
            )
            result[band_key] = band_data
        except Exception as e:
            if ignore_missing:
                print(f"Warning: Could not download {band_key} for {year}-{month:02d}-{day:02d}: {e}")
                result[band_key] = ([], [], [], [])
            else:
                raise
    
    return result




