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
    """Load WRS-2 path/row polygons from shapefile."""
    if shapefile_path is None:
        shapefile_path = str(files("blackmarble").joinpath("data", "WRS2_descending_0", "WRS2_descending.shp"))
    
    polygons = []
    with fiona.open(shapefile_path, "r") as collection:
        for feature in collection:
            properties = feature["properties"]
            path = int(properties["PATH"])
            row = int(properties["ROW"])
            geom = shape(feature["geometry"])
            
            if isinstance(geom, shapely.geometry.Polygon):
                polygons.append((geom, path, row))
    
    print(f"Loaded {len(polygons)} WRS-2 path/row polygons from shapefile")
    return polygons


def get_wrs_tiles_for_point(
    lat: float, 
    lon: float, 
    wrs_polygons: list[tuple[shapely.geometry.Polygon, int, int]] | None = None
) -> list[dict[str, int]]:
    """Get Landsat WRS-2 path and row for given coordinates."""
    if wrs_polygons is None:
        wrs_polygons = load_wrs_shapefile()
    
    point = Point(lon, lat)
    results = []
    for polygon, path, row in wrs_polygons:
        if polygon.contains(point):
            results.append({"path": path, "row": row})
    
    return results


def get_wrs_tiles_for_bbox_tuple(
    bbox: BBox,
    wrs_polygons: list[tuple[shapely.geometry.Polygon, int, int]] | None = None
) -> list[dict[str, int]]:
    """Get all Landsat WRS-2 tiles that intersect with a bounding box."""
    from shapely.geometry import box
    
    if wrs_polygons is None:
        wrs_polygons = load_wrs_shapefile()
    
    min_lon, min_lat, max_lon, max_lat = bbox
    bbox_polygon = box(min_lon, min_lat, max_lon, max_lat)
    
    results = []
    seen_tiles = set()
    
    for polygon, path, row in wrs_polygons:
        if polygon.intersects(bbox_polygon):
            tile_key = (path, row)
            if tile_key not in seen_tiles:
                seen_tiles.add(tile_key)
                results.append({"path": path, "row": row})
    
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
    """Find Landsat scene directory for given date and path/row."""
    global _DIR_CACHE
    
    yr = str(year)
    mo = str(month).zfill(2)
    da = str(day).zfill(2)
    path_str = str(path).zfill(3)
    row_str = str(row).zfill(3)
    
    prefix = f"{collection}/level-2/standard/oli-tirs/{yr}/{path_str}/{row_str}/"
    
    if prefix in _DIR_CACHE:
        directories = _DIR_CACHE[prefix]
        print(f"Using cached directory listing for {prefix}")
    else:
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
            
            directories = []
            common_prefixes = resp.get("CommonPrefixes", [])
            for prefix_entry in common_prefixes:
                if isinstance(prefix_entry, dict) and "Prefix" in prefix_entry:
                    directories.append({"Prefix": prefix_entry["Prefix"]})
                    
            print(f"Found {len(directories)} scene directories")
            
            if len(_DIR_CACHE) >= _MAX_CACHE_SIZE:
                _DIR_CACHE.popitem(last=False)
            _DIR_CACHE[prefix] = directories
            
        except Exception as e:
            print(f"Error querying S3: {e}")
            return None
    
    target_date = datetime.strptime(yr + mo + da, "%Y%m%d")
    
    for entry in directories:
        try:
            scene_name = entry["Prefix"].split("/")[-2]
            date_parts = scene_name.split("_")[3:5]
            
            start_date = datetime.strptime(date_parts[0], "%Y%m%d")
            end_date = datetime.strptime(date_parts[1], "%Y%m%d")
            
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
    """Download a specific Landsat band from AWS."""
    scene_name = scene_dir.rstrip("/").split("/")[-1]
    
    if band in ("QA_PIXEL", "QA_RADSAT", "QA_AEROSOL"):
        suffix_map = {
            "QA_PIXEL": "_QA_PIXEL.TIF",
            "QA_RADSAT": "_QA_RADSAT.TIF", 
            "QA_AEROSOL": "_SR_QA_AEROSOL.TIF",
        }
        suffix = suffix_map[str(band)]
        s3_key = scene_dir + scene_name + suffix
    else:
        s3_key = f"{scene_dir}{scene_name}_SR_B{band}.TIF"
    
    print(f"Downloading {s3_key} to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    session = boto3.Session()
    s3 = session.client("s3")
    
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
    """Find all WRS-2 path/row tiles intersecting a bounding box."""
    if wrs_polygons is None:
        wrs_polygons = load_wrs_shapefile()
    
    bbox = shapely.geometry.box(min_lon, min_lat, max_lon, max_lat)
    
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
    """Extract metadata from a Landsat band file."""
    with rasterio.open(file_path) as src:
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
        
        if "SR_B" in os.path.basename(file_path):
            metadata.update({
                "scale_factor": 0.0000275,
                "add_offset": -0.2,
                "valid_range": [7273, 43636],
                "units": "reflectance",
                "product_type": "surface_reflectance"
            })
        elif "QA_" in os.path.basename(file_path):
            metadata.update({
                "scale_factor": 1.0,
                "add_offset": 0.0,
                "product_type": "quality_assessment"
            })
        
        tags = src.tags()
        if tags:
            metadata["gdal_tags"] = tags
        
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
    """Download all tiles for a band covering the given bounding box."""
    min_lon, min_lat, max_lon, max_lat = bbox
    
    if cache_dir:
        download_dir = os.path.join(cache_dir, "landsat", f"{year:04d}", f"{month:02d}", f"{day:02d}")
        os.makedirs(download_dir, exist_ok=True)
    else:
        download_dir = tempfile.mkdtemp(prefix="landsat_")
    
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
    """Download all specified bands for a single date."""
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




