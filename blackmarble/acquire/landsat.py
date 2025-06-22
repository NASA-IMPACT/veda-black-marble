"""Landsat data acquisition and tile selection.

This module handles downloading Landsat data from AWS, selecting appropriate
tiles based on geographic bounds, and managing the acquisition workflow.
"""

from typing import Any
from datetime import datetime
import numpy.typing as npt

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
    # Currently in landsat.py LandsatUtil.get_wrs()
    ...


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
    # Currently in landsat.py LandsatUtil.filter_landsat_data_dir()
    ...


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
    # Currently in landsat.py LandsatUtil.download_landsat_band()
    ...


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
    # Currently in bm.py download_tiles_for_band()
    ...


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
    # Currently in bm.py download_tiles_for_band_with_composites()
    ...


def get_landsat_metadata(scene_path: str) -> dict[str, Any]:
    """Extract metadata from Landsat scene.
    
    Args:
        scene_path: Path to scene file or directory
        
    Returns:
        Dictionary containing scale factors, offsets, acquisition date, etc.
    """
    # New function - metadata extraction for scaling
    ...