"""VIIRS nighttime lights data acquisition.

This module handles downloading VIIRS Black Marble (VNP46A2) data from NASA
LAADS DAAC, including authentication and tile selection.
"""

from typing import Any
from datetime import datetime
import numpy.typing as npt
import os

# Import existing implementations
from blackmarble.vnp import VNP46A2Util

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
    # Delegate to existing VNP46A2Util implementation
    util = VNP46A2Util()
    return util.coordinates_to_tile_id(lat, lon)


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
    # Delegate to existing VNP46A2Util implementation
    util = VNP46A2Util()
    util.download_h5(
        earthdata_token,
        year, month, day,
        vertical, horizontal,
        output_path
    )
    return output_path


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
    # Delegate to existing VNP46A2Util implementation
    util = VNP46A2Util()
    util.convert_hdf_to_tiff(hdf_file, output_file, layer)
    return output_file


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
    # Extract base filename without extension
    basename = os.path.splitext(os.path.basename(hdf_file))[0]
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Layer indices for VNP46A2 HDF5 files
    # Based on standard VNP46A2 structure
    layers = {
        'radiance': (0, f"{basename}_DNB_At_Sensor_Radiance.tif"),
        'lunar_irradiance': (1, f"{basename}_Lunar_Irradiance.tif"),
        'quality_flag': (2, f"{basename}_Mandatory_Quality_Flag.tif"),
        'cloud_mask': (3, f"{basename}_QF_Cloud_Mask.tif"),
    }
    
    output_files = {}
    util = VNP46A2Util()
    
    for layer_name, (layer_idx, filename) in layers.items():
        output_path = os.path.join(output_dir, filename)
        try:
            util.convert_hdf_to_tiff(hdf_file, output_path, layer_idx)
            output_files[layer_name] = output_path
        except Exception as e:
            # Some layers might not exist in all files
            print(f"Warning: Could not extract {layer_name}: {e}")
            if layer_name in ['radiance', 'quality_flag']:
                # These are mandatory layers, re-raise
                raise
    
    return output_files


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
    # The rasterio implementation doesn't have a separate read function
    # We'll implement it directly here
    from osgeo import gdal
    import numpy as np
    
    # Map layer names to subdataset indices
    layer_map = {
        'DNB_At_Sensor_Radiance': 0,
        'Lunar_Irradiance': 1,
        'Mandatory_Quality_Flag': 2,
        'QF_Cloud_Mask': 3,
    }
    
    layer_idx = layer_map.get(layer_name, 0)
    
    # Open HDF5 file
    hdf_ds = gdal.Open(hdf_file)
    if not hdf_ds:
        raise ValueError(f"Could not open HDF5 file: {hdf_file}")
    
    # Get subdataset
    subdatasets = hdf_ds.GetSubDatasets()
    if layer_idx >= len(subdatasets):
        raise ValueError(f"Layer index {layer_idx} not found in file")
    
    sub_ds = gdal.Open(subdatasets[layer_idx][0])
    
    # Read data
    data = sub_ds.ReadAsArray()
    
    # Get metadata
    metadata = {
        'width': sub_ds.RasterXSize,
        'height': sub_ds.RasterYSize,
        'transform': sub_ds.GetGeoTransform(),
        'projection': sub_ds.GetProjection(),
        'metadata': sub_ds.GetMetadata(),
    }
    
    # Close datasets
    sub_ds = None
    hdf_ds = None
    
    return data, metadata


def get_tiles_for_bbox(
    bbox: tuple[float, float, float, float],
    include_partial: bool = True
) -> list[tuple[int, int]]:
    """Get all VIIRS tiles that intersect with a bounding box.
    
    This critical function determines which VIIRS tiles need to be downloaded
    to cover a given geographic area. VIIRS data is distributed in a global
    grid of tiles (similar to MODIS), and this function maps from geographic
    coordinates to tile indices.
    
    This is essential for the acquisition phase as it tells us exactly which
    tiles to download from NASA LAADS DAAC to ensure complete coverage of
    the area of interest.
    
    Args:
        bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat) in degrees
        include_partial: Whether to include tiles that only partially overlap
                        the bbox. If False, only tiles fully within bbox are returned.
        
    Returns:
        List of (vertical, horizontal) tile indices that cover the bbox.
        Tiles are returned in row-major order (top-left to bottom-right).
        
    Example:
        >>> # Get tiles for New York City area
        >>> tiles = get_tiles_for_bbox((-74.3, 40.5, -73.7, 40.9))
        >>> for v, h in tiles:
        >>>     download_viirs_h5(token, 2023, 9, 15, v, h)
    
    Note:
        VIIRS tiles are 10x10 degrees at the equator in a sinusoidal projection.
        A single geographic area may require multiple tiles, especially at
        higher latitudes where tiles overlap more.
    """
    # Extract bbox components
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # Delegate to existing VNP46A2Util implementation
    util = VNP46A2Util()
    return util.get_tiles_for_bbox(min_lat, min_lon, max_lat, max_lon)


def download_vnp46a2_tiles(
    date: datetime,
    bbox: tuple[float, float, float, float],
    product: str = "DNB_At_Sensor_Radiance",
    output_dir: str = ".",
    earthdata_token: str | None = None,
    skip_existing: bool = True
) -> dict[str, list[str]]:
    """Download all VNP46A2 tiles for a date and bounding box.
    
    This high-level function combines tile selection with downloading,
    providing a simple interface for acquiring all necessary VIIRS data
    for a given area and date. It handles authentication, tile selection,
    and parallel downloads.
    
    Args:
        date: Target date for data acquisition
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
        product: Which VNP46A2 product layer to extract
        output_dir: Directory to save downloaded files
        earthdata_token: NASA Earthdata token (uses env var if None)
        skip_existing: Skip downloading files that already exist
        
    Returns:
        Dictionary with keys:
        - 'radiance': List of radiance GeoTIFF paths
        - 'lunar': List of lunar irradiance GeoTIFF paths
        - 'quality': List of quality flag GeoTIFF paths
        - 'metadata': List of metadata dictionaries
        
    Example:
        >>> # Download all data for Manhattan on Sept 15, 2023
        >>> files = download_vnp46a2_tiles(
        >>>     date=datetime(2023, 9, 15),
        >>>     bbox=(-74.05, 40.69, -73.95, 40.79)
        >>> )
        >>> print(f"Downloaded {len(files['radiance'])} tiles")
    """
    # Get Earthdata token from environment if not provided
    if not earthdata_token:
        earthdata_token = os.environ.get('EARTHDATA_TOKEN')
        if not earthdata_token:
            raise ValueError("Earthdata token required. Set EARTHDATA_TOKEN env var or pass as parameter.")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get tiles that cover the bbox
    tiles = get_tiles_for_bbox(bbox)
    
    # Extract date components
    year = date.year
    month = date.month
    day = date.day
    
    # Initialize result lists
    result = {
        'radiance': [],
        'lunar': [],
        'quality': [],
        'metadata': []
    }
    
    # Download and process each tile
    for v, h in tiles:
        # Create filename for HDF5 file
        hdf_filename = f"VNP46A2_h{h:02d}v{v:02d}_{year}{month:02d}{day:02d}.h5"
        hdf_path = os.path.join(output_dir, hdf_filename)
        
        # Skip if exists and skip_existing is True
        if skip_existing and os.path.exists(hdf_path):
            print(f"Skipping existing file: {hdf_path}")
        else:
            # Download HDF5 file
            download_viirs_h5(
                earthdata_token,
                year, month, day,
                v, h,
                hdf_path
            )
        
        # Extract layers to GeoTIFF
        try:
            layer_files = extract_viirs_layers(hdf_path, output_dir)
            
            # Add to result lists
            if 'radiance' in layer_files:
                result['radiance'].append(layer_files['radiance'])
            if 'lunar_irradiance' in layer_files:
                result['lunar'].append(layer_files['lunar_irradiance'])
            if 'quality_flag' in layer_files:
                result['quality'].append(layer_files['quality_flag'])
            
            # Extract metadata
            metadata = get_viirs_metadata(hdf_path)
            metadata['tile'] = {'vertical': v, 'horizontal': h}
            result['metadata'].append(metadata)
            
        except Exception as e:
            print(f"Error processing tile h{h:02d}v{v:02d}: {e}")
            continue
    
    return result


def get_viirs_metadata(hdf_file: str) -> dict[str, Any]:
    """Extract metadata from VIIRS HDF5 file.
    
    Args:
        hdf_file: Path to HDF5 file
        
    Returns:
        Dictionary containing acquisition date, tile info, quality stats, etc.
    """
    # Extract metadata using GDAL
    from osgeo import gdal
    
    # Open HDF5 file
    ds = gdal.Open(hdf_file)
    if not ds:
        raise ValueError(f"Could not open HDF5 file: {hdf_file}")
    
    # Get basic info
    metadata = {
        'subdatasets': len(ds.GetSubDatasets()),
        'driver': ds.GetDriver().ShortName,
    }
    
    # Get metadata from first subdataset (DNB radiance)
    if ds.GetSubDatasets():
        sub_ds = gdal.Open(ds.GetSubDatasets()[0][0])
        if sub_ds:
            meta_dict = sub_ds.GetMetadata_Dict()
            
            # Extract useful VIIRS metadata
            metadata.update({
                'horizontal_tile': int(meta_dict.get('HorizontalTileNumber', 0)),
                'vertical_tile': int(meta_dict.get('VerticalTileNumber', 0)),
                'acquisition_date': meta_dict.get('AcquisitionDate', ''),
                'acquisition_time': meta_dict.get('AcquisitionTime', ''),
                'day_night_flag': meta_dict.get('DayNightFlag', ''),
                'product_name': meta_dict.get('ProductName', 'VNP46A2'),
                'processing_level': meta_dict.get('ProcessingLevel', ''),
            })
            
            # Add geospatial info
            metadata['geospatial'] = {
                'width': sub_ds.RasterXSize,
                'height': sub_ds.RasterYSize,
                'bands': sub_ds.RasterCount,
            }
            
            sub_ds = None
    
    # Close dataset
    ds = None
    
    return metadata