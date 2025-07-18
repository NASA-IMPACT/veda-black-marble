"""Functions for downloading VIIRS nighttime lights data from NASA LAADS.

This module provides async download functions for VIIRS data acquisition.
"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path

import aiofiles
import aiohttp

logger = logging.getLogger(__name__)


async def download_viirs_h5_async(
    session: aiohttp.ClientSession,
    earthdata_token: str,
    year: int,
    month: int,
    day: int,
    vertical: int,
    horizontal: int,
    output_path: str,
) -> str:
    """Download a single VIIRS VNP46A2 HDF5 file asynchronously.

    This is the async version of download_viirs_h5 from viirs.py.

    Args:
        session: Aiohttp session for making requests
        earthdata_token: NASA Earthdata authentication bearer token
        year: Target acquisition year
        month: Target acquisition month (1-12)
        day: Target acquisition day (1-31)
        vertical: VIIRS vertical tile index (0-17)
        horizontal: VIIRS horizontal tile index (0-35)
        output_path: Local file path to save the downloaded HDF5 file

    Returns:
        Path to the downloaded HDF5 file

    Raises:
        RuntimeError: If no data found or download fails
    """
    # Format tile identifier
    tile_id = f"h{horizontal:02d}v{vertical:02d}"

    # Convert date to day-of-year format
    from datetime import date

    target_date = date(year, month, day)
    day_of_year = target_date.timetuple().tm_yday

    # Use the same API endpoint as synchronous version
    api_url = f"https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/details/allData/5000/VNP46A2/{year}/{day_of_year}"

    logger.debug(f"Querying NASA LAADS API: {api_url}")
    logger.debug(f"Looking for tile: {tile_id}")

    try:
        # Query API to get available files
        async with session.get(api_url) as response:
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                html_content = await response.text()
                logger.error("NASA LAADS API returned HTML instead of JSON")
                logger.debug(f"HTML preview: {html_content[:500]}")
                raise RuntimeError(
                    "NASA LAADS API returned HTML instead of JSON. "
                    "This usually indicates an authentication or endpoint issue."
                )

            api_data = await response.json()

    except aiohttp.ClientError as e:
        logger.error(f"Failed to query NASA LAADS API: {e}")
        raise RuntimeError(f"Failed to query NASA LAADS API: {e}")

    # Find the file for our tile
    items = api_data.get("content", [])
    logger.debug(f"Found {len(items)} files for day {day_of_year}")

    download_link = None
    matching_filename = None

    for item in items:
        filename = item.get("name", "")
        if f".{tile_id}." in filename:
            download_link = item.get("downloadsLink")
            matching_filename = filename
            logger.debug(f"Found matching file: {filename}")
            logger.debug(f"Download link: {download_link}")
            break

    if not download_link:
        logger.error(f"No VNP46A2 file found for tile {tile_id} on {year}-{day_of_year:03d}")
        raise RuntimeError(f"No VNP46A2 file found for tile {tile_id} on {year}-{day_of_year:03d}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Download the file
    logger.info(f"Downloading {matching_filename} to {output_path}")

    headers = {"Authorization": f"Bearer {earthdata_token}"}

    try:
        async with session.get(download_link, headers=headers) as download_response:
            download_response.raise_for_status()

            # Check for HTML error page
            content_type = download_response.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                html_content = await download_response.text()
                logger.error("Received HTML error page instead of HDF5 file")
                logger.debug(f"HTML content: {html_content[:500]}")
                raise RuntimeError(
                    "Received HTML error page instead of HDF5 file. " "Check your Earthdata token."
                )

            # Download in chunks
            total_size = int(download_response.headers.get("content-length", 0))
            logger.debug(f"File size: {total_size:,} bytes")

            async with aiofiles.open(output_path, "wb") as f:
                downloaded = 0
                async for chunk in download_response.content.iter_chunked(8192):
                    await f.write(chunk)
                    downloaded += len(chunk)

                    # Log progress every 10MB
                    if downloaded % (10 * 1024 * 1024) < 8192:
                        logger.debug(f"Downloaded {downloaded:,} / {total_size:,} bytes")

            logger.info(f"Successfully downloaded {output_path} ({total_size:,} bytes)")
            return output_path

    except aiohttp.ClientError as e:
        logger.error(f"Download failed for {matching_filename}: {e}")
        raise RuntimeError(f"Download failed for {matching_filename}: {e}")


async def download_viirs_tiles_async(
    date: datetime,
    bbox: tuple[float, float, float, float],
    earthdata_token: str,
    output_dir: str = "./data/viirs",
    max_concurrent: int = 3,
) -> dict[str, list[str]]:
    """Download all VIIRS tiles for a date and bounding box using async I/O.

    This is the async version of download_vnp46a2_tiles from the synchronous API.

    Args:
        date: Target date for data acquisition
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
        earthdata_token: NASA Earthdata token
        output_dir: Directory to save downloaded files
        max_concurrent: Maximum concurrent downloads (NASA rate limits)

    Returns:
        Dictionary with keys:
        - 'gap_filled_ntl': List of gap-filled NTL GeoTIFF paths
        - 'quality': List of quality flag GeoTIFF paths
        - 'radiance': List of radiance GeoTIFF paths (if extracted)
    """
    logger.info(">>> download_viirs_tiles_async called")
    logger.info(f"    date: {date}")
    logger.info(f"    bbox: {bbox}")
    logger.info(
        f"    earthdata_token: {'***' + earthdata_token[-4:] if earthdata_token else 'None'}"
    )

    try:
        # Validate bbox unpacking
        min_lon, min_lat, max_lon, max_lat = bbox
        logger.debug(
            f"Bbox unpacked: min_lon={min_lon}, min_lat={min_lat}, "
            f"max_lon={max_lon}, max_lat={max_lat}"
        )
    except ValueError as e:
        logger.error(f"FOUND UNPACKING ERROR IN BBOX: {e}")
        logger.error(f"bbox type: {type(bbox)}, content: {bbox}")
        raise

    from .viirs import extract_viirs_layers, get_tiles_for_bbox

    # Get tiles that cover the bbox
    logger.debug("Calling get_tiles_for_bbox...")
    try:
        tiles = get_tiles_for_bbox(bbox)
        logger.debug(f"get_tiles_for_bbox returned: {tiles}")
        logger.debug(f"tiles type: {type(tiles)}, length: {len(tiles)}")
        if tiles:
            logger.debug(f"First tile: {tiles[0]}, type: {type(tiles[0])}")
    except Exception as e:
        logger.error(f"Error in get_tiles_for_bbox: {e}")
        import traceback

        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise

    logger.info(f"VIIRS tile selection for bbox {bbox}:")
    logger.info(f"  Date: {date.strftime('%Y-%m-%d')}")
    logger.info(f"  Number of tiles: {len(tiles)}")

    try:
        for i, tile in enumerate(tiles):
            if isinstance(tile, tuple) and len(tile) == 2:
                v, h = tile
                logger.debug(f"  Tile {i + 1}: h{h:02d}v{v:02d}")
            else:
                logger.error(f"  Invalid tile at index {i}: {tile} (type: {type(tile)})")
                raise ValueError(f"Invalid tile format: expected (v, h) tuple, got {tile}")
    except ValueError as e:
        logger.error(f"Error unpacking tiles: {e}")
        logger.error(f"tiles content: {tiles}")
        raise

    # Create output directory structure
    year = date.year
    month = date.month
    day = date.day
    effective_output_dir = os.path.join(output_dir, f"{year:04d}", f"{month:02d}", f"{day:02d}")
    os.makedirs(effective_output_dir, exist_ok=True)

    # Initialize result lists
    result = {"gap_filled_ntl": []}

    # Create session with rate limiting
    connector = aiohttp.TCPConnector(limit_per_host=max_concurrent)
    timeout = aiohttp.ClientTimeout(total=600)  # 10 minute timeout

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Create download tasks for each tile
        download_tasks = []

        for tile in tiles:
            try:
                v, h = tile
            except ValueError as e:
                if "too many values to unpack" in str(e):
                    logger.error("FOUND UNPACKING ERROR AT LINE 238!")
                logger.error(f"Error unpacking tile {tile}: {e}")
                logger.error(f"Tile type: {type(tile)}, content: {tile}")
                raise ValueError(f"Cannot unpack tile: {tile}. Expected (v, h) tuple. Error: {e}")
            hdf_filename = f"VNP46A2_h{h:02d}v{v:02d}_{year}{month:02d}{day:02d}.h5"
            hdf_path = os.path.join(effective_output_dir, hdf_filename)

            # Skip if file already exists
            if os.path.exists(hdf_path) and os.path.getsize(hdf_path) > 1000:
                logger.info(f"Using cached HDF5 file: {hdf_path}")
                # Extract layers from existing file
                try:
                    layer_files = extract_viirs_layers(hdf_path, effective_output_dir)
                    _add_layer_files_to_result(result, layer_files)
                except Exception as e:
                    logger.error(f"Error processing cached tile h{h:02d}v{v:02d}: {e}")
                continue

            # Create download task
            task = download_viirs_h5_async(
                session=session,
                earthdata_token=earthdata_token,
                year=year,
                month=month,
                day=day,
                vertical=v,
                horizontal=h,
                output_path=hdf_path,
            )
            download_tasks.append((task, v, h, hdf_path))

        # Download all tiles concurrently
        if download_tasks:
            logger.info(f"Downloading {len(download_tasks)} VIIRS tiles...")
            logger.debug(f"download_tasks length: {len(download_tasks)}")

            # Use gather with return_exceptions to handle individual failures
            try:
                task_futures = [task for task, _, _, _ in download_tasks]
            except ValueError as e:
                logger.error(f"Error unpacking download_tasks: {e}")
                logger.error(f"download_tasks: {download_tasks}")
                raise ValueError(f"Failed to unpack download_tasks: {e}")
            download_results = await asyncio.gather(*task_futures, return_exceptions=True)

            # Process results
            for i, result_or_error in enumerate(download_results):
                try:
                    _, v, h, hdf_path = download_tasks[i]
                except ValueError as e:
                    logger.error(f"Error unpacking download_tasks[{i}]: {e}")
                    logger.error(f"download_tasks[{i}] = {download_tasks[i]}")
                    has_len = hasattr(download_tasks[i], '__len__')
                    length = len(download_tasks[i]) if has_len else 'N/A'
                    logger.error(
                        f"Type: {type(download_tasks[i])}, "
                        f"Length: {length}"
                    )
                    raise ValueError(f"Failed to unpack download_tasks[{i}]: {e}")

                if isinstance(result_or_error, Exception):
                    logger.error(f"Failed to download tile h{h:02d}v{v:02d}: {result_or_error}")
                    continue

                # Extract layers from downloaded HDF5
                try:
                    logger.debug(f"Extracting layers from {hdf_path}")
                    layer_files = extract_viirs_layers(hdf_path, effective_output_dir)
                    _add_layer_files_to_result(result, layer_files)
                    logger.debug(f"Successfully processed tile h{h:02d}v{v:02d}")
                except Exception as e:
                    logger.error(f"Error extracting layers from tile h{h:02d}v{v:02d}: {e}")

    logger.info("VIIRS download complete. Files by layer:")
    for layer, files in result.items():
        if files:
            logger.info(f"  {layer}: {len(files)} files")

    return result


def _add_layer_files_to_result(result: dict[str, list[str]], layer_files: dict[str, str]) -> None:
    """Helper to add extracted layer files to result dictionary."""
    if "gap_filled_ntl" in layer_files:
        result["gap_filled_ntl"].append(layer_files["gap_filled_ntl"])


async def download_viirs_for_pipeline(
    date: datetime,
    bbox: tuple[float, float, float, float],
    earthdata_token: str,
    output_dir: str = "./data/viirs",
    show_progress: bool = True,
) -> dict[str, list[Path]]:
    """Download VIIRS data for the async pipeline.

    This is the main entry point from the async pipeline, returning
    Path objects as expected by the pipeline.

    Args:
        date: Target date
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
        earthdata_token: NASA Earthdata token
        output_dir: Output directory
        show_progress: Show progress (for compatibility)

    Returns:
        Dictionary mapping layer names to lists of Path objects
    """
    logger.info(f"Starting VIIRS download for {date.strftime('%Y-%m-%d')}")
    logger.info(f">>> FUNCTIONAL API CALLED - bbox: {bbox}")

    try:
        # Download tiles
        file_dict = await download_viirs_tiles_async(
            date=date, bbox=bbox, earthdata_token=earthdata_token, output_dir=output_dir
        )
        logger.info(f"download_viirs_tiles_async returned keys: {list(file_dict.keys())}")
        for key, files in file_dict.items():
            logger.info(f"  {key}: {len(files)} files")
    except ValueError as e:
        if "too many values to unpack" in str(e):
            logger.error("FOUND THE UNPACKING ERROR!")
            logger.error(f"Error details: {e}")
            import traceback

            logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise
    except Exception as e:
        logger.error(f"Error in download_viirs_tiles_async: {e}")
        import traceback

        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise

    # Convert string paths to Path objects
    result = {}

    # We only need gap_filled_ntl now
    if "gap_filled_ntl" in file_dict and file_dict["gap_filled_ntl"]:
        result["gap_filled_ntl"] = [Path(f) for f in file_dict["gap_filled_ntl"]]
        logger.debug(f"gap_filled_ntl: {len(result['gap_filled_ntl'])} files")
    else:
        result["gap_filled_ntl"] = []

    return result

# Legacy name aliases for compatibility
download_vnp46a2_tiles = download_viirs_for_pipeline
download_viirs_h5 = download_viirs_h5_async
