"""VIIRS nighttime lights data acquisition.

This module handles downloading VIIRS Black Marble (VNP46A2) data from NASA
LAADS DAAC, including authentication and tile selection.
"""

from typing import Any
from datetime import datetime, date
import numpy.typing as npt
import os
import requests
import shutil

# Type aliases
ArrayLike = npt.NDArray[Any]


def coordinates_to_viirs_tile(lat: float, lon: float) -> tuple[int, int]:
    """Convert geographic coordinates to VIIRS tile indices.
    
    VIIRS data is organized in a global sinusoidal projection grid with 36 horizontal
    and 18 vertical tiles, each covering 10 degrees at the equator. This function
    maps any geographic coordinate to its corresponding tile indices, which is
    essential for determining which data files to download from NASA LAADS DAAC.
    
    Args:
        lat: Latitude in decimal degrees (-90 to 90)
        lon: Longitude in decimal degrees (-180 to 180)
        
    Returns:
        Tuple of (vertical_tile, horizontal_tile) where:
        - vertical_tile ranges from 0 to 17 (0 = northernmost)
        - horizontal_tile ranges from 0 to 35 (0 = westernmost)
        
    Example:
        >>> coordinates_to_viirs_tile(40.7, -74.0)  # Manhattan, NY
        (5, 6)
        
    Note:
        The VIIRS tiling system uses a 10-degree grid starting from (-180, 90).
        Each tile covers exactly 10x10 degrees in geographic coordinates.
    """
    # Convert latitude to vertical tile index (0-17, where 0 is northernmost)
    # Each tile is 10 degrees tall
    vertical_tile = int((90 - lat) / 10)
    
    # Convert longitude to horizontal tile index (0-35, where 0 is westernmost)  
    # Each tile is 10 degrees wide
    horizontal_tile = int((lon + 180) / 10)
    
    # Ensure tiles are within valid ranges
    vertical_tile = max(0, min(17, vertical_tile))
    horizontal_tile = max(0, min(35, horizontal_tile))
    
    return vertical_tile, horizontal_tile


def download_viirs_h5(
    earthdata_token: str,
    year: int,
    month: int,
    day: int,
    vertical: int,
    horizontal: int,
    output_path: str = "VNP46A2.h5"
) -> str:
    """Download VIIRS VNP46A2 HDF5 file from NASA LAADS DAAC.
    
    Downloads Black Marble VNP46A2 data for a specific date and tile from NASA's
    Level-1 and Atmosphere Archive & Distribution System (LAADS) DAAC. This function
    handles the NASA API authentication, file discovery, and HTTP download process.
    
    The VNP46A2 product provides daily nighttime lights data at 500m resolution,
    essential for urban analysis, disaster response, and economic monitoring.
    
    Args:
        earthdata_token: NASA Earthdata authentication bearer token
        year: Target acquisition year (2012-present for VNP46A2)
        month: Target acquisition month (1-12)
        day: Target acquisition day (1-31)
        vertical: VIIRS vertical tile index (0-17)
        horizontal: VIIRS horizontal tile index (0-35)
        output_path: Local file path to save the downloaded HDF5 file
        
    Returns:
        Path to the downloaded HDF5 file (same as output_path)
        
    Raises:
        RuntimeError: If no data found for the specified date/tile or download fails
        requests.exceptions.RequestException: If API request or download fails
        OSError: If file system operations fail
        
    Example:
        >>> # Download Manhattan tile for September 15, 2023
        >>> token = "your_earthdata_token_here"
        >>> file_path = download_viirs_h5(token, 2023, 9, 15, 5, 6, "manhattan.h5")
        >>> print(f"Downloaded: {file_path}")
        
    Note:
        Requires a valid NASA Earthdata account and bearer token. Files are typically
        50-100MB each. The function will overwrite existing files at output_path.
    """
    # Format tile identifier as required by NASA LAADS API
    tile_id = f"h{horizontal:02d}v{vertical:02d}"
    
    # Convert date to day-of-year format required by NASA API
    target_date = date(year, month, day)
    day_of_year = target_date.timetuple().tm_yday
    
    # Construct NASA LAADS DAAC API URL for VNP46A2 product
    # API endpoint follows: /api/v2/content/details/allData/{collection}/{product}/{year}/{day_of_year}
    api_url = (
        f"https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/details/allData/5000/"
        f"VNP46A2/{year}/{day_of_year}"
    )
    
    print(f"Querying NASA LAADS API: {api_url}")
    
    # Query the API to get available files for the specified date
    try:
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' in content_type:
            # This is an HTML error page, not JSON
            html_preview = response.text[:500]
            raise RuntimeError(
                f"NASA LAADS API returned HTML instead of JSON. "
                f"This usually indicates an authentication or endpoint issue. "
                f"URL: {api_url}\n"
                f"Content preview: {html_preview}"
            )
        
        # Try to parse JSON
        try:
            api_data = response.json()
        except ValueError as e:
            # Not valid JSON - show what we got
            content_preview = response.text[:500]
            raise RuntimeError(
                f"NASA LAADS API returned invalid JSON. "
                f"Content-Type: {content_type}\n"
                f"Content preview: {content_preview}"
            )
            
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to query NASA LAADS API: {e}")
    
    # Extract file list from API response
    if not isinstance(api_data, dict):
        raise RuntimeError(
            f"NASA LAADS API returned unexpected data type: {type(api_data).__name__}. "
            f"Expected dict with 'content' field."
        )
    
    items = api_data.get("content", [])
    if not isinstance(items, list):
        # Check if we got an error response
        if "error" in api_data:
            error_msg = api_data.get("error", {}).get("message", "Unknown error")
            raise RuntimeError(f"NASA LAADS API error: {error_msg}")
        else:
            raise RuntimeError(
                f"NASA LAADS API returned unexpected structure. "
                f"Expected 'content' to be a list, got: {type(items).__name__}"
            )
    
    print(f"Found {len(items)} VNP46A2 files for {year}-{day_of_year:03d}")
    
    # Search for the specific tile we need
    # NASA file naming: VNP46A2.A{year}{day_of_year}.{tile}.{version}.{timestamp}.h5
    target_filename_pattern = f"VNP46A2.A{year}{day_of_year:03d}.{tile_id}"
    print(f"Searching for file matching pattern: {target_filename_pattern}")
    
    download_link = None
    matching_filename = None
    
    for item in items:
        filename = item.get("name", "")
        if filename.startswith(target_filename_pattern):
            download_link = item.get("downloadsLink")
            matching_filename = filename
            print(f"Found matching file: {filename}")
            print(f"Download link: {download_link}")
            break
    
    if not download_link:
        raise RuntimeError(
            f"No VNP46A2 file found for tile {tile_id} on {year}-{day_of_year:03d}. "
            f"The tile may not have data for this date, or the date may be outside "
            f"the VNP46A2 temporal coverage (2012-present)."
        )
    
    # Set up authenticated session for download
    session = requests.Session()
    if not earthdata_token:
        raise RuntimeError(
            "NASA Earthdata token is required for LAADS downloads. "
            "Obtain a token from https://urs.earthdata.nasa.gov/"
        )
    
    # NASA LAADS uses Bearer token authentication
    session.headers.update({"Authorization": f"Bearer {earthdata_token}"})
    
    print(f"Downloading {matching_filename} to {output_path}...")
    
    try:
        # Ensure output directory exists
        parent_dir = os.path.dirname(output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        
        # Remove existing file/directory at target path if it exists
        if os.path.exists(output_path):
            if os.path.isdir(output_path):
                print(f"Warning: Removing existing directory at {output_path}")
                shutil.rmtree(output_path)
            else:
                print(f"Warning: Overwriting existing file at {output_path}")
                os.remove(output_path)
        
        # Download file with streaming to handle large files efficiently
        with session.get(download_link, stream=True, timeout=300) as download_response:
            download_response.raise_for_status()
            
            # Check content type to detect HTML error pages
            content_type = download_response.headers.get('Content-Type', '').lower()
            if 'text/html' in content_type:
                # Read the HTML content to get error details
                html_content = download_response.text[:1000]  # First 1000 chars
                raise RuntimeError(
                    f"Received HTML error page instead of HDF5 file. "
                    f"This usually indicates an authentication issue. "
                    f"Content preview: {html_content[:200]}..."
                )
            
            # Write to temporary file first
            temp_path = output_path + '.tmp'
            with open(temp_path, "wb") as output_file:
                # Download in 8KB chunks to manage memory usage
                first_chunk = True
                for chunk in download_response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        if first_chunk:
                            # Check if first chunk looks like HTML
                            if chunk.startswith(b'<!DOCTYPE') or chunk.startswith(b'<html'):
                                os.remove(temp_path)
                                raise RuntimeError(
                                    f"Downloaded file appears to be HTML, not HDF5. "
                                    f"This usually indicates an authentication or access issue. "
                                    f"First bytes: {chunk[:100]}"
                                )
                            # Check for HDF5 magic number
                            if not chunk.startswith(b'\x89HDF'):
                                print(f"Warning: File does not start with HDF5 magic number")
                            first_chunk = False
                        output_file.write(chunk)
            
            # Verify file size is reasonable (VNP46A2 files are typically 2-5 MB)
            file_size = os.path.getsize(temp_path)
            if file_size < 1000:  # Less than 1KB is definitely wrong
                with open(temp_path, 'rb') as f:
                    content = f.read(500)
                os.remove(temp_path)
                raise RuntimeError(
                    f"Downloaded file is too small ({file_size} bytes). "
                    f"Content preview: {content[:200]}"
                )
            
            # Move temp file to final location
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_path, output_path)
        
        print(f"Successfully downloaded {matching_filename} ({os.path.getsize(output_path):,} bytes)")
        return output_path
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Download failed for {matching_filename}: {e}")
    except OSError as e:
        raise RuntimeError(f"File operation failed for {output_path}: {e}")


def convert_viirs_hdf_to_tiff(
    hdf_file: str,
    output_file: str,
    layer: int = 0
) -> str:
    """Convert VIIRS HDF5 layer to a georeferenced GeoTIFF.
    
    Extracts a specific science dataset from a VIIRS VNP46A2 HDF5 file and converts
    it to a georeferenced GeoTIFF using the tile's geographic bounds. This is essential
    for integrating VIIRS data into GIS workflows and spatial analysis pipelines.
    
    VIIRS tiles are 10x10 degree squares in geographic coordinates (WGS84), with
    tile positions encoded in the HDF5 metadata. This function reads the tile
    position and applies the appropriate geographic transformation.
    
    Args:
        hdf_file: Path to the input VNP46A2 HDF5 file
        output_file: Path for the output GeoTIFF file
        layer: HDF5 subdataset index to extract:
               - 0: DNB_At_Sensor_Radiance (primary nighttime lights)
               - 1: Lunar_Irradiance (moon brightness correction)
               - 2: Gap_Filled_DNB_BRDF-Corrected_NTL (gap-filled version)
               - 3: Latest_High_Quality_Retrieval (quality flag used by bm.py)
               - 4: Mandatory_Quality_Flag (complex bit-packed quality)
               - 5: QF_Cloud_Mask (cloud detection flags)
        
    Returns:
        Path to the created GeoTIFF file (same as output_file)
        
    Raises:
        ValueError: If HDF5 file cannot be opened or layer doesn't exist
        RuntimeError: If GDAL operations fail
        
    Example:
        >>> # Convert DNB radiance layer to GeoTIFF
        >>> tiff_path = convert_viirs_hdf_to_tiff("VNP46A2.h5", "radiance.tif", 0)
        >>> print(f"Created: {tiff_path}")
        
    Note:
        Requires GDAL/OGR for HDF5 and GeoTIFF support. The output uses WGS84
        geographic coordinates (EPSG:4326) with pixel registration.
    """
    # First validate the file is actually HDF5
    if not os.path.exists(hdf_file):
        raise ValueError(f"HDF5 file not found: {hdf_file}")
    
    # Check file size
    file_size = os.path.getsize(hdf_file)
    if file_size < 1000:
        raise ValueError(
            f"HDF5 file is too small ({file_size} bytes). "
            f"This may be an HTML error page. Check authentication."
        )
    
    # Check file header for HDF5 magic number or HTML
    with open(hdf_file, 'rb') as f:
        header = f.read(100)
        if header.startswith(b'<!DOCTYPE') or header.startswith(b'<html'):
            # Try to read more to get error details
            f.seek(0)
            content = f.read(1000).decode('utf-8', errors='ignore')
            raise ValueError(
                f"File {hdf_file} is an HTML error page, not HDF5. "
                f"Content preview: {content[:200]}..."
            )
        if not header.startswith(b'\x89HDF'):
            print(f"Warning: {hdf_file} does not have HDF5 magic number. Header: {header[:20]}")
    
    try:
        from osgeo import gdal
    except ImportError:
        raise ImportError("GDAL is required for HDF5 to GeoTIFF conversion. Install with: uv add gdal")
    
    # Open the HDF5 file using GDAL
    hdf_dataset = gdal.Open(hdf_file, gdal.GA_ReadOnly)
    if not hdf_dataset:
        raise ValueError(f"Could not open HDF5 file: {hdf_file}")
    
    try:
        # Get list of available subdatasets
        subdatasets = hdf_dataset.GetSubDatasets()
        if not subdatasets:
            raise ValueError(f"No subdatasets found in HDF5 file: {hdf_file}")
        
        if layer >= len(subdatasets):
            raise ValueError(
                f"Layer index {layer} not found. File has {len(subdatasets)} layers: "
                f"{[sd[1] for sd in subdatasets]}"
            )
        
        # Open the specific subdataset (layer) we want to convert
        subdataset_name = subdatasets[layer][0]
        layer_dataset = gdal.Open(subdataset_name, gdal.GA_ReadOnly)
        if not layer_dataset:
            raise RuntimeError(f"Could not open subdataset {layer}: {subdataset_name}")
        
        try:
            # Extract tile position from metadata to determine geographic bounds
            metadata = layer_dataset.GetMetadata_Dict()
            
            # Get horizontal and vertical tile numbers from HDF5 metadata
            # These determine the 10x10 degree tile's position in the global grid
            horizontal_tile = int(metadata.get("HorizontalTileNumber", 0))
            vertical_tile = int(metadata.get("VerticalTileNumber", 0))
            
            # Calculate geographic bounds for this tile
            # VIIRS global grid: 36 horizontal × 18 vertical tiles
            # Each tile covers exactly 10 degrees
            # Grid origin: (-180°, 90°) for tile h00v00
            west_bound = (10 * horizontal_tile) - 180    # Western edge
            north_bound = 90 - (10 * vertical_tile)      # Northern edge  
            east_bound = west_bound + 10                 # Eastern edge
            south_bound = north_bound - 10               # Southern edge
            
            print(f"Converting VIIRS tile h{horizontal_tile:02d}v{vertical_tile:02d}")
            print(f"Geographic bounds: ({west_bound}, {south_bound}, {east_bound}, {north_bound})")
            
            # Create GDAL translate options for georeferencing
            # -a_srs: Set spatial reference system to WGS84 (EPSG:4326)
            # -a_ullr: Set upper-left and lower-right coordinates
            translate_options = gdal.TranslateOptions(
                gdal.ParseCommandLine(
                    f"-a_srs EPSG:4326 -a_ullr {west_bound} {north_bound} {east_bound} {south_bound}"
                )
            )
            
            # Perform the conversion with georeferencing
            result_dataset = gdal.Translate(output_file, layer_dataset, options=translate_options)
            if not result_dataset:
                raise RuntimeError(f"GDAL Translate failed for output: {output_file}")
            
            # Ensure the file is written and closed properly
            result_dataset.FlushCache()
            result_dataset = None
            
            print(f"Successfully converted HDF5 layer {layer} to GeoTIFF: {output_file}")
            return output_file
            
        finally:
            # Clean up layer dataset
            layer_dataset = None
            
    finally:
        # Clean up main HDF dataset
        hdf_dataset = None


def extract_viirs_layers(
    hdf_file: str,
    output_dir: str = "."
) -> dict[str, str]:
    """Extract all relevant layers from VIIRS HDF5 file to separate GeoTIFFs.
    
    Processes a VNP46A2 HDF5 file and extracts the primary science datasets as
    individual georeferenced GeoTIFF files. This is essential for the pipeline
    as it converts the complex HDF5 structure into standard raster formats that
    can be processed by geospatial analysis tools.
    
    The function handles the standard VNP46A2 layer structure and gracefully
    manages missing optional layers while ensuring mandatory layers are present.
    
    Args:
        hdf_file: Path to the input VNP46A2 HDF5 file
        output_dir: Directory to save the extracted GeoTIFF files
        
    Returns:
        Dictionary mapping layer names to their output file paths:
        - 'radiance': DNB at-sensor radiance (nW cm⁻² sr⁻¹) - primary nighttime lights
        - 'quality_flag': Mandatory quality assessment flags
        - 'cloud_mask': Cloud detection mask (if available)
        
    Raises:
        RuntimeError: If mandatory layers (radiance, quality_flag) cannot be extracted
        ValueError: If HDF5 file is invalid or corrupted
        
    Example:
        >>> layers = extract_viirs_layers("VNP46A2.h5", "output/")
        >>> print(f"Radiance layer: {layers['radiance']}")
        >>> print(f"Quality flags: {layers['quality_flag']}")
        
    Note:
        Creates output directory if it doesn't exist. Some layers (like cloud_mask)
        may not be present in all VNP46A2 files depending on processing version.
    """
    # Extract base filename without extension for consistent naming
    basename = os.path.splitext(os.path.basename(hdf_file))[0]
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define VNP46A2 layer structure with indices and output filenames
    # Based on NASA's standard VNP46A2 subdataset organization
    layer_config = {
        'radiance': {
            'index': 0,
            'filename': f"{basename}_DNB_At_Sensor_Radiance.tif",
            'mandatory': True,
            'description': "Day/Night Band at-sensor radiance"
        },
        'gap_filled_ntl': {
            'index': 2,
            'filename': f"{basename}_Gap_Filled_DNB_BRDF-Corrected_NTL.tif",
            'mandatory': True,
            'description': "Gap-filled BRDF-corrected nighttime lights (NASA pre-processed)"
        },
        'quality_flag': {
            'index': 3,  # Use Latest_High_Quality_Retrieval like bm.py
            'filename': f"{basename}_Latest_High_Quality_Retrieval.tif", 
            'mandatory': True,
            'description': "Latest high quality retrieval flag (becomes 0/1 after resampling)"
        },
        'cloud_mask': {
            'index': 5,  # QF_Cloud_Mask is actually at index 5
            'filename': f"{basename}_QF_Cloud_Mask.tif",
            'mandatory': False,
            'description': "Cloud detection quality flags"
        },
    }
    
    extracted_files = {}
    extraction_errors = []
    
    # Process each layer using the pure functional implementation
    for layer_name, layer_info in layer_config.items():
        layer_index = layer_info['index']
        output_filename = layer_info['filename']
        is_mandatory = layer_info['mandatory']
        
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            # Use the pure functional conversion implementation
            convert_viirs_hdf_to_tiff(hdf_file, output_path, layer_index)
            extracted_files[layer_name] = output_path
            print(f"Extracted {layer_name}: {output_path}")
            
        except Exception as e:
            error_message = f"Failed to extract {layer_name} (layer {layer_index}): {e}"
            extraction_errors.append(error_message)
            
            if is_mandatory:
                # Mandatory layers must be successfully extracted
                raise RuntimeError(
                    f"Could not extract mandatory layer '{layer_name}': {e}. "
                    f"This layer is required for VIIRS data processing."
                )
            else:
                # Optional layers can fail without stopping the process
                print(f"Warning: {error_message}")
                print(f"Continuing without optional layer: {layer_name}")
    
    # Log summary of extraction results
    if extraction_errors:
        print(f"Extraction completed with {len(extraction_errors)} warnings")
    else:
        print(f"Successfully extracted all {len(extracted_files)} layers")
    
    return extracted_files


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
    
    # Map layer names to subdataset indices
    layer_map = {
        'DNB_At_Sensor_Radiance': 0,
        'Lunar_Irradiance': 1,
        'Gap_Filled_DNB_BRDF-Corrected_NTL': 2,
        'Latest_High_Quality_Retrieval': 3,  # This is what bm.py uses
        'Mandatory_Quality_Flag': 4,
        'QF_Cloud_Mask': 5,
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
    # Extract and validate bounding box components
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # Ensure longitude is within valid range (-180 to 180)
    min_lon = max(-180.0, min(180.0, min_lon))
    max_lon = max(-180.0, min(180.0, max_lon))
    
    # Ensure latitude is within valid range (-90 to 90)
    min_lat = max(-90.0, min(90.0, min_lat))
    max_lat = max(-90.0, min(90.0, max_lat))
    
    # Handle longitude wrapping around the international date line
    if min_lon > max_lon:
        # Bounding box crosses date line - split into two regions
        west_tiles = get_tiles_for_bbox((min_lon, min_lat, 180.0, max_lat), include_partial)
        east_tiles = get_tiles_for_bbox((-180.0, min_lat, max_lon, max_lat), include_partial)
        
        # Combine and deduplicate tiles
        all_tiles = west_tiles + east_tiles
        unique_tiles = list(set(all_tiles))  # Remove duplicates
        
        # Sort by vertical then horizontal for consistent ordering
        return sorted(unique_tiles, key=lambda tile: (tile[0], tile[1]))
    
    # Get tile indices for key corners of the bounding box
    # We need to check multiple points because VIIRS tiles can have complex coverage patterns
    
    # Northwest corner (top-left)
    nw_vertical, nw_horizontal = coordinates_to_viirs_tile(max_lat, min_lon)
    
    # Southeast corner (bottom-right)  
    se_vertical, se_horizontal = coordinates_to_viirs_tile(min_lat, max_lon)
    
    # Calculate the range of tiles needed to cover the entire bounding box
    # VIIRS uses a regular 10-degree grid, so we need all tiles in the rectangular region
    min_vertical = min(nw_vertical, se_vertical)
    max_vertical = max(nw_vertical, se_vertical)
    min_horizontal = min(nw_horizontal, se_horizontal)
    max_horizontal = max(nw_horizontal, se_horizontal)
    
    # Generate all tile combinations in the range
    tiles = []
    for vertical in range(min_vertical, max_vertical + 1):
        # Ensure vertical tile is within valid range (0-17)
        if not (0 <= vertical <= 17):
            continue
            
        for horizontal in range(min_horizontal, max_horizontal + 1):
            # Handle horizontal wraparound (tiles 0-35 represent full 360° coverage)
            horizontal_wrapped = horizontal % 36
            
            # Only include tiles that actually intersect the bounding box
            if include_partial or _tile_fully_contains_bbox(
                vertical, horizontal_wrapped, min_lat, min_lon, max_lat, max_lon
            ):
                tiles.append((vertical, horizontal_wrapped))
    
    # Remove any duplicate tiles that might occur due to edge cases
    unique_tiles = list(set(tiles))
    
    # Sort tiles in row-major order (top-left to bottom-right) for consistent processing
    sorted_tiles = sorted(unique_tiles, key=lambda tile: (tile[0], tile[1]))
    
    print(f"Found {len(sorted_tiles)} VIIRS tiles covering bounding box {bbox}")
    return sorted_tiles


def _tile_fully_contains_bbox(
    vertical: int, 
    horizontal: int, 
    min_lat: float, 
    min_lon: float, 
    max_lat: float, 
    max_lon: float
) -> bool:
    """Check if a VIIRS tile fully contains the given bounding box.
    
    Helper function to determine if the entire bounding box falls within
    a single VIIRS tile, which can be useful for optimization when
    include_partial=False.
    
    Args:
        vertical: VIIRS vertical tile index (0-17)
        horizontal: VIIRS horizontal tile index (0-35)
        min_lat, min_lon, max_lat, max_lon: Bounding box coordinates
        
    Returns:
        True if the tile completely contains the bounding box
    """
    # Calculate tile boundaries (each tile is exactly 10x10 degrees)
    tile_west = (10 * horizontal) - 180
    tile_east = tile_west + 10
    tile_north = 90 - (10 * vertical)  
    tile_south = tile_north - 10
    
    # Check if bounding box is entirely within tile bounds
    return (
        tile_west <= min_lon and max_lon <= tile_east and
        tile_south <= min_lat and max_lat <= tile_north
    )


def download_vnp46a2_tiles(
    date: datetime,
    bbox: tuple[float, float, float, float],
    product: str = "DNB_At_Sensor_Radiance",
    output_dir: str = ".",
    earthdata_token: str | None = None,
    skip_existing: bool = True,
    cache_dir: str | None = None
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
        cache_dir: Optional cache directory for storing downloads
        
    Returns:
        Dictionary with keys:
        - 'radiance': List of radiance GeoTIFF paths
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
    
    # Extract date components
    year = date.year
    month = date.month
    day = date.day
    
    # Use cache directory if provided, otherwise use output_dir
    if cache_dir:
        # Create structured cache directory: cache/viirs/YYYY/MM/DD/
        effective_output_dir = os.path.join(cache_dir, "viirs", f"{year:04d}", f"{month:02d}", f"{day:02d}")
        os.makedirs(effective_output_dir, exist_ok=True)
    else:
        effective_output_dir = output_dir
        os.makedirs(effective_output_dir, exist_ok=True)
    
    # Get tiles that cover the bbox
    tiles = get_tiles_for_bbox(bbox)
    
    # Debug: Print tile selection
    print(f"\nVIIRS tile selection for bbox {bbox}:")
    print(f"  Date: {year}-{month:02d}-{day:02d}")
    print(f"  Number of tiles: {len(tiles)}")
    for i, (v, h) in enumerate(tiles):
        print(f"  Tile {i+1}: h{h:02d}v{v:02d}")
        # Calculate tile bounds for debugging
        west = (10 * h) - 180
        east = west + 10
        north = 90 - (10 * v)
        south = north - 10
        print(f"    Bounds: ({west}, {south}, {east}, {north})")
    
    # Initialize result lists
    result = {
        'radiance': [],
        'quality': [],
        'gap_filled_ntl': [],
        'metadata': []
    }
    
    # Download and process each tile
    for v, h in tiles:
        # Create filename for HDF5 file
        hdf_filename = f"VNP46A2_h{h:02d}v{v:02d}_{year}{month:02d}{day:02d}.h5"
        hdf_path = os.path.join(effective_output_dir, hdf_filename)
        
        # Skip if exists and skip_existing is True
        if skip_existing and os.path.exists(hdf_path):
            print(f"Using cached HDF5 file: {hdf_path}")
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
            layer_files = extract_viirs_layers(hdf_path, effective_output_dir)
            
            # Add to result lists
            if 'radiance' in layer_files:
                result['radiance'].append(layer_files['radiance'])
            if 'quality_flag' in layer_files:
                result['quality'].append(layer_files['quality_flag'])
            if 'gap_filled_ntl' in layer_files:
                result['gap_filled_ntl'].append(layer_files['gap_filled_ntl'])
            
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