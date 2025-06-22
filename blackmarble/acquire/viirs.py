"""VIIRS nighttime lights data acquisition.

This module handles downloading VIIRS Black Marble (VNP46A2) data from NASA
LAADS DAAC, including authentication and tile selection.
"""

from typing import Any
from datetime import datetime
import numpy.typing as npt

# Type aliases
ArrayLike = npt.NDArray[Any]


def coordinates_to_viirs_tile(lat: float, lon: float) -> tuple[int, int]:
    """Convert geographic coordinates to VIIRS tile indices.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Tuple of (vertical_tile, horizontal_tile)
    """
    # Currently in vnp.py VNP46A2Util.coordinates_to_tile_id()
    ...


def download_viirs_h5(
    earthdata_token: str,
    year: int,
    month: int,
    day: int,
    vertical: int,
    horizontal: int,
    output_path: str = "VNP46A2.h5"
) -> str:
    """Download VIIRS VNP46A2 HDF5 file from NASA LAADS.
    
    Args:
        earthdata_token: NASA Earthdata authentication token
        year: Target year
        month: Target month
        day: Target day
        vertical: Vertical tile index
        horizontal: Horizontal tile index
        output_path: Path to save downloaded file
        
    Returns:
        Path to downloaded file
    """
    # Currently in vnp.py VNP46A2Util.download_h5()
    ...


def convert_viirs_hdf_to_tiff(
    hdf_file: str,
    output_file: str,
    layer: int = 0
) -> str:
    """Convert VIIRS HDF5 layer to GeoTIFF.
    
    Args:
        hdf_file: Path to HDF5 file
        output_file: Path for output GeoTIFF
        layer: HDF5 layer index (0=DNB_At_Sensor_Radiance, 1=Lunar_Irradiance, etc.)
        
    Returns:
        Path to created GeoTIFF
    """
    # Currently in vnp.py VNP46A2Util.convert_VNP46A2_HDF2TIFF()
    ...


def extract_viirs_layers(
    hdf_file: str,
    output_dir: str = "."
) -> dict[str, str]:
    """Extract all relevant layers from VIIRS HDF5 file.
    
    Args:
        hdf_file: Path to HDF5 file
        output_dir: Directory for output files
        
    Returns:
        Dictionary mapping layer names to file paths:
        - 'radiance': DNB at-sensor radiance
        - 'lunar_irradiance': Lunar irradiance
        - 'quality_flag': Mandatory quality flag
        - 'cloud_mask': Cloud mask (if available)
    """
    # Wrapper around convert_viirs_hdf_to_tiff for all layers
    ...


def read_viirs_with_rasterio(
    hdf_file: str,
    layer_name: str
) -> tuple[ArrayLike, dict[str, Any]]:
    """Read VIIRS HDF5 data using rasterio.
    
    Args:
        hdf_file: Path to HDF5 file
        layer_name: Name of layer to read
        
    Returns:
        Tuple of (data_array, metadata_dict)
    """
    # Currently in vnp_rasterio.py read_viirs_hdf5()
    ...


def get_viirs_metadata(hdf_file: str) -> dict[str, Any]:
    """Extract metadata from VIIRS HDF5 file.
    
    Args:
        hdf_file: Path to HDF5 file
        
    Returns:
        Dictionary containing acquisition date, tile info, quality stats, etc.
    """
    # Metadata extraction for processing
    ...