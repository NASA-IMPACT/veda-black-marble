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

# Import pure functional implementations
from blackmarble.landsat_functions import (
    get_wrs_path_row as _get_wrs_path_row,
    get_wrs_tiles_for_bbox,
    find_landsat_scene_directory,
    download_landsat_file,
    get_landsat_band_metadata,
    calculate_landsat_cloud_cover
)

# Type aliases
BBox = tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
ArrayLike = npt.NDArray[Any]


def get_wrs_path_row(lat: float, lon: float) -> list[dict[str, int]]:
    """Get Landsat WRS-2 path and row for given coordinates.
    
    Determines which Landsat WRS-2 tiles contain the specified geographic point.
    This is essential for identifying the correct Landsat data granules to acquire
    for any given location. Multiple tiles may be returned for overlap areas.
    
    Args:
        lat: Latitude in decimal degrees (-90 to 90)
        lon: Longitude in decimal degrees (-180 to 180)
        
    Returns:
        List of dicts with 'path' and 'row' keys (can be multiple for overlap areas)
        
    Example:
        >>> get_wrs_path_row(40.7, -74.0)  # Manhattan, NY
        [{'path': 14, 'row': 32}]
    """
    # Use pure functional implementation - no class instantiation needed
    return _get_wrs_path_row(lat, lon)


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
    # Use pure functional implementation
    return find_landsat_scene_directory(year, month, day, path, row, collection)


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
    # Use pure functional implementation
    return download_landsat_file(scene_dir, band, output_path, aws_credentials)


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
    paths = range(min(wrs1["path"], wrs2["path"]), max(wrs1["path"], wrs2["path"]) + 1)
    rows = range(min(wrs1["row"], wrs2["row"]), max(wrs1["row"], wrs2["row"]) + 1)
    
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


def download_tiles_with_composites(
    band: int | str,
    year: int,
    month: int,
    day: int,
    bbox: BBox,
    composite_history_months: int = 12,
    cache_dir: str | None = None
) -> list[tuple[list[str], list[str], list[str], list[str]]]:
    """Download tiles including historical composites for gap filling.
    
    Args:
        band: Band identifier
        year: Target year
        month: Target month
        day: Target day
        bbox: Bounding box
        composite_history_months: Number of months of history to include
        cache_dir: Optional cache directory for storing downloads
        
    Returns:
        List of tuples, each containing (band_files, qa_pixel, qa_radsat, qa_aerosol)
    """
    # Download primary tiles
    tiles = [download_tiles_for_band(band, year, month, day, bbox, ignore_missing=False, cache_dir=cache_dir)]
    
    # Download historical composites
    for months_back in range(1, composite_history_months + 1):
        # Calculate date for historical composite
        target_date = datetime(year, month, day) - timedelta(days=months_back * 30)
        hist_year = target_date.year
        hist_month = target_date.month
        hist_day = target_date.day
        
        try:
            # Try to download historical data, ignoring missing tiles
            hist_tiles = download_tiles_for_band(
                band, hist_year, hist_month, hist_day, bbox, ignore_missing=True, cache_dir=cache_dir
            )
            if any(hist_tiles):  # Only add if we got some data
                tiles.append(hist_tiles)
        except Exception as e:
            print(f"Warning: Could not download composite for {hist_year}-{hist_month}-{hist_day}: {e}")
            continue
    
    return tiles


def get_tiles_for_bbox(bbox: BBox) -> list[dict[str, int]]:
    """Get all WRS-2 path/row tiles needed to cover a bounding box.
    
    Identifies all Landsat tiles required to achieve complete coverage of the
    specified geographic area. Essential for regional analysis requiring multiple
    Landsat scenes to cover the area of interest.
    
    Args:
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
        
    Returns:
        List of dicts with 'path' and 'row' keys covering the bounding box
        
    Example:
        >>> get_tiles_for_bbox((-75.0, 40.0, -73.0, 42.0))  # NYC region
        [{'path': 13, 'row': 32}, {'path': 14, 'row': 32}]
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    return get_wrs_tiles_for_bbox(min_lat, max_lat, min_lon, max_lon)


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


def get_landsat_metadata(scene_path: str) -> dict[str, Any]:
    """Extract metadata from Landsat scene.
    
    Extracts comprehensive metadata from Landsat band files including scaling
    factors, spatial reference, and acquisition information. This metadata is
    essential for proper radiometric processing in the pipeline.
    
    Args:
        scene_path: Path to Landsat band TIFF file
        
    Returns:
        Dictionary containing scale factors, offsets, spatial reference, etc.
        
    Example:
        >>> metadata = get_landsat_metadata('/path/to/LC08_L2SP_014032_B4.TIF')
        >>> print(metadata['scale_factor'])
        0.0000275
    """
    # Use pure functional implementation  
    return get_landsat_band_metadata(scene_path)


def get_cloud_cover(qa_pixel_path: str) -> float:
    """Calculate cloud cover percentage from QA_PIXEL band.
    
    Analyzes the QA_PIXEL band to determine scene cloud coverage. This metric
    is crucial for scene selection and quality assessment, helping users
    choose the clearest available imagery for analysis.
    
    Args:
        qa_pixel_path: Path to QA_PIXEL.TIF file
        
    Returns:
        Cloud cover percentage (0.0 to 100.0)
        
    Example:
        >>> cloud_cover = get_cloud_cover('/path/to/LC08_QA_PIXEL.TIF')
        >>> print(f"Scene has {cloud_cover:.1f}% cloud cover")
    """
    # Use pure functional implementation
    return calculate_landsat_cloud_cover(qa_pixel_path)