"""VIIRS nighttime lights data acquisition - SLIM VERSION."""

from typing import Any
from datetime import datetime, date
import numpy.typing as npt
import os
import requests
import shutil
from osgeo import gdal

# Type aliases
ArrayLike = npt.NDArray[Any]


def coordinates_to_viirs_tile(lat: float, lon: float) -> tuple[int, int]:
    """Convert lat/lon to VIIRS tile indices."""
    vertical_tile = max(0, min(17, int((90 - lat) / 10)))
    horizontal_tile = max(0, min(35, int((lon + 180) / 10)))
    return vertical_tile, horizontal_tile


def download_viirs_h5(
    earthdata_token: str,
    year: int,
    month: int,
    day: int,
    vertical: int,
    horizontal: int,
    output_path: str = "VNP46A2.h5",
) -> str:
    """Download VIIRS VNP46A2 HDF5 file from NASA."""
    tile_id = f"h{horizontal:02d}v{vertical:02d}"
    target_date = date(year, month, day)
    day_of_year = target_date.timetuple().tm_yday

    # NASA API URL
    api_url = (
        f"https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/details/allData/5000/"
        f"VNP46A2/{year}/{day_of_year}"
    )

    print(f"Querying NASA LAADS API: {api_url}")

    # Query API
    response = requests.get(api_url, timeout=30)
    response.raise_for_status()

    api_data = response.json()

    # Find matching file
    matching_filename = None
    for entry in api_data.get("content", []):
        if isinstance(entry, dict) and "name" in entry:
            filename = entry["name"]
            if tile_id in filename and filename.endswith(".h5"):
                matching_filename = filename
                break

    if not matching_filename:
        raise RuntimeError(
            f"No VNP46A2 file found for tile {tile_id} on {year}-{month:02d}-{day:02d}"
        )

    # Download file
    download_url = f"https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/archives/{matching_filename}"
    headers = {"Authorization": f"Bearer {earthdata_token}"}

    print(f"Downloading {matching_filename}...")

    response = requests.get(download_url, headers=headers, stream=True, timeout=300)
    response.raise_for_status()

    # Save to disk
    temp_path = output_path + ".tmp"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(response.raw, f)

    os.rename(temp_path, output_path)
    print(f"Downloaded {matching_filename} ({os.path.getsize(output_path):,} bytes)")
    return output_path


def convert_viirs_hdf_to_tiff(hdf_file: str, output_file: str, layer: int = 0) -> str:
    """Convert VIIRS HDF5 layer to GeoTIFF."""
    # Open HDF
    hdf_dataset = gdal.Open(hdf_file, gdal.GA_ReadOnly)
    if not hdf_dataset:
        raise ValueError(f"Could not open HDF5 file: {hdf_file}")

    # Get subdatasets
    subdatasets = hdf_dataset.GetSubDatasets()
    if layer >= len(subdatasets):
        raise ValueError(f"Layer {layer} not found. File has {len(subdatasets)} layers")

    # Open subdataset
    subdataset_name = subdatasets[layer][0]
    layer_dataset = gdal.Open(subdataset_name, gdal.GA_ReadOnly)

    # Get tile metadata
    metadata = layer_dataset.GetMetadata_Dict()
    horizontal_tile = int(metadata.get("HorizontalTileNumber", 0))
    vertical_tile = int(metadata.get("VerticalTileNumber", 0))

    # Calculate bounds (10x10 degree tiles)
    west = (10 * horizontal_tile) - 180
    north = 90 - (10 * vertical_tile)
    east = west + 10
    south = north - 10

    print(f"Converting tile h{horizontal_tile:02d}v{vertical_tile:02d}")

    # Create output with proper georeferencing
    translate_options = gdal.TranslateOptions(
        outputSRS="EPSG:4326",
        outputBounds=[west, north, east, south],  # ulx, uly, lrx, lry
    )
    gdal.Translate(output_file, layer_dataset, options=translate_options)

    return output_file


def extract_viirs_layers(hdf_file: str, output_dir: str = ".") -> dict[str, str]:
    """Extract gap-filled NTL layer from VIIRS HDF5."""
    basename = os.path.splitext(os.path.basename(hdf_file))[0]
    os.makedirs(output_dir, exist_ok=True)

    # We only need gap-filled NTL (layer 2)
    output_path = os.path.join(
        output_dir, f"{basename}_Gap_Filled_DNB_BRDF-Corrected_NTL.tif"
    )

    try:
        convert_viirs_hdf_to_tiff(hdf_file, output_path, layer=2)
        print(f"Extracted gap_filled_ntl: {output_path}")
        return {"gap_filled_ntl": output_path}
    except Exception as e:
        raise RuntimeError(f"Could not extract gap-filled NTL layer: {e}")


def get_tiles_for_bbox(
    bbox: tuple[float, float, float, float], overlap: float = 0.0
) -> list[tuple[int, int]]:
    """Get VIIRS tiles covering a bounding box."""
    min_lon, min_lat, max_lon, max_lat = bbox

    # Expand bounds if overlap requested
    if overlap > 0:
        min_lon -= overlap
        min_lat -= overlap
        max_lon += overlap
        max_lat += overlap

    # Get corner tiles
    v_min, h_min = coordinates_to_viirs_tile(max_lat, min_lon)
    v_max, h_max = coordinates_to_viirs_tile(min_lat, max_lon)

    # Generate all tiles in range
    tiles = []
    for v in range(v_min, v_max + 1):
        for h in range(h_min, h_max + 1):
            if 0 <= v <= 17 and 0 <= h <= 35:
                tiles.append((v, h))

    return tiles


def download_vnp46a2_tiles(
    date: datetime,
    bbox: tuple[float, float, float, float],
    product: str = "DNB_At_Sensor_Radiance",
    output_dir: str = ".",
    earthdata_token: str | None = None,
    skip_existing: bool = True,
    cache_dir: str | None = None,
) -> dict[str, list[str]]:
    """Download all VNP46A2 tiles for a date and bounding box."""
    # Product parameter reserved for future multi-product support
    _ = product  # Currently unused, kept for API compatibility
    
    # Get token
    if not earthdata_token:
        earthdata_token = os.environ.get("EARTHDATA_TOKEN")
        if not earthdata_token:
            raise ValueError("Earthdata token required. Set EARTHDATA_TOKEN env var.")

    # Setup directories
    if cache_dir:
        download_dir = os.path.join(
            cache_dir,
            "viirs",
            f"{date.year:04d}",
            f"{date.month:02d}",
            f"{date.day:02d}",
        )
        os.makedirs(download_dir, exist_ok=True)
    else:
        download_dir = output_dir

    # Get tiles
    tiles = get_tiles_for_bbox(bbox)
    print(f"Found {len(tiles)} tiles for bbox")

    downloaded_files = {"gap_filled_ntl": [], "metadata": []}

    # Download each tile
    for v, h in tiles:
        h5_filename = (
            f"VNP46A2.A{date.year}{date.timetuple().tm_yday:03d}.h{h:02d}v{v:02d}.h5"
        )
        h5_path = os.path.join(download_dir, h5_filename)

        # Download HDF5 if needed
        if not os.path.exists(h5_path) or not skip_existing:
            try:
                download_viirs_h5(
                    earthdata_token, date.year, date.month, date.day, v, h, h5_path
                )
            except Exception as e:
                print(f"Failed to download tile h{h:02d}v{v:02d}: {e}")
                continue
        else:
            print(f"Using cached: {h5_path}")

        # Extract layers
        try:
            layers = extract_viirs_layers(h5_path, output_dir)
            if "gap_filled_ntl" in layers:
                downloaded_files["gap_filled_ntl"].append(layers["gap_filled_ntl"])
                downloaded_files["metadata"].append(
                    {
                        "tile": f"h{h:02d}v{v:02d}",
                        "date": date.isoformat(),
                        "source_file": h5_path,
                    }
                )
        except Exception as e:
            print(f"Failed to extract layers from {h5_filename}: {e}")

    print(f"Successfully processed {len(downloaded_files['gap_filled_ntl'])} tiles")
    return downloaded_files


def coordinates_to_viirs_tile_string(lat: float, lon: float) -> str:
    """Convert coordinates to VIIRS tile string (e.g., 'h06v05')."""
    v, h = coordinates_to_viirs_tile(lat, lon)
    return f"h{h:02d}v{v:02d}"
