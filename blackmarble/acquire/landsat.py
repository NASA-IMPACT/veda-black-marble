"""Landsat data acquisition from AWS S3 using async I/O."""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aioboto3
from botocore.config import Config

from .async_base import (
    AsyncDownloaderBase,
    DownloadConfig,
    DownloadProgressTracker,
    DownloadResult,
    DownloadStatus,
    RateLimiter,
)

logger = logging.getLogger(__name__)


class AsyncLandsatDownloader(AsyncDownloaderBase):
    """Async downloader for Landsat Collection 2 data from AWS S3"""

    # S3 bucket and structure
    BUCKET = "usgs-landsat"
    COLLECTION = "collection02"
    LEVEL = "level-2"

    # Band mapping
    BAND_MAPPING = {
        "B1": "SR_B1.TIF",
        "B2": "SR_B2.TIF",
        "B3": "SR_B3.TIF",
        "B4": "SR_B4.TIF",
        "B5": "SR_B5.TIF",
        "B6": "SR_B6.TIF",
        "B7": "SR_B7.TIF",
        "QA_PIXEL": "QA_PIXEL.TIF",
    }

    def __init__(
        self,
        aws_unsigned: bool = True,
        config: DownloadConfig | None = None,
        rate_limiter: RateLimiter | None = None,
        progress_callback=None,
    ):
        """
        Args:
            aws_unsigned: Use unsigned S3 access (public data)
            config: Download configuration
            rate_limiter: Rate limiter
            progress_callback: Progress callback function
        """
        # Set up default config with S3-specific settings
        if config is None:
            config = DownloadConfig(
                max_concurrent=10,  # S3 can handle more
                timeout_seconds=300,
                chunk_size=1024 * 1024,  # 1MB chunks
                max_retries=3,
            )

        super().__init__(config, rate_limiter, progress_callback)

        # S3 configuration
        self.aws_unsigned = aws_unsigned
        # Don't use UNSIGNED config - just use default boto3 client
        self.s3_config = Config(retries={"max_attempts": 0})  # We handle retries ourselves

        # Cache for S3 directory listings
        self._listing_cache = {}

    async def list_s3_objects(self, prefix: str, delimiter: str = "/") -> dict[str, Any]:
        """
        List S3 objects with caching.

        Args:
            prefix: S3 prefix to list
            delimiter: Delimiter for grouping

        Returns:
            S3 list response
        """
        cache_key = f"{prefix}:{delimiter}"

        if cache_key in self._listing_cache:
            return self._listing_cache[cache_key]

        # Create default session (will use AWS credentials if available, or anonymous)
        session = aioboto3.Session()
        async with session.client("s3", config=self.s3_config) as s3:  # type: ignore
            try:
                response = await s3.list_objects_v2(
                    Bucket=self.BUCKET,
                    Prefix=prefix,
                    Delimiter=delimiter,
                    RequestPayer="requester",  # Required for USGS Landsat bucket
                )
                self._listing_cache[cache_key] = response
                return response
            except Exception as e:
                logger.error(f"Failed to list S3 objects at {prefix}: {e}")
                return {"CommonPrefixes": [], "Contents": []}

    async def get_scene_cloud_cover(self, scene_prefix: str, scene_id: str) -> float:
        """
        Get cloud cover percentage from scene MTL metadata.

        Args:
            scene_prefix: S3 prefix for the scene
            scene_id: Scene identifier

        Returns:
            Cloud cover percentage (0-100)
        """
        # MTL file S3 key
        mtl_key = f"{scene_prefix}{scene_id}_MTL.txt"

        try:
            # Download MTL metadata to memory
            session = aioboto3.Session()
            async with session.client("s3", config=self.s3_config) as s3:  # type: ignore
                response = await s3.get_object(
                    Bucket=self.BUCKET, Key=mtl_key, RequestPayer="requester"
                )

                # Read MTL content
                mtl_content = await response["Body"].read()
                mtl_text = mtl_content.decode("utf-8")

                # Parse cloud cover from MTL
                # Look for IMAGE_ATTRIBUTES_GROUP and CLOUD_COVER_LAND
                cloud_cover = 100.0  # Default to high if not found

                for line in mtl_text.split("\n"):
                    if "CLOUD_COVER_LAND" in line:
                        # Format: CLOUD_COVER_LAND = 12.34
                        parts = line.split("=")
                        if len(parts) == 2:
                            try:
                                cloud_cover = float(parts[1].strip())
                                break
                            except ValueError:
                                pass
                    elif "CLOUD_COVER" in line and "CLOUD_COVER_LAND" not in line:
                        # Fallback to general cloud cover if land-specific not found
                        parts = line.split("=")
                        if len(parts) == 2:
                            try:
                                cloud_cover = float(parts[1].strip())
                            except ValueError:
                                pass

                return cloud_cover

        except Exception as e:
            logger.warning(f"Failed to get cloud cover for {scene_id}: {e}")
            # Return a conservative estimate if we can't get metadata
            return 50.0

    async def find_landsat_scenes(
        self,
        year: int,
        bbox: tuple[float, float, float, float],
        date_range: tuple[datetime, datetime] | None = None,
        cloud_cover_max: float = 20.0,
    ) -> list[dict[str, Any]]:
        """
        Find Landsat scenes for given criteria.

        Args:
            year: Year to search
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            date_range: Optional date range filter
            cloud_cover_max: Maximum cloud cover percentage

        Returns:
            List of scene metadata
        """
        from .wrs_utils import get_wrs_tiles_for_bbox_tuple

        # Get ALL WRS tiles that intersect with the bbox
        tile_dicts = get_wrs_tiles_for_bbox_tuple(bbox)
        tiles = [f"{t['path']:03d}{t['row']:03d}" for t in tile_dicts]

        logger.info(f"Searching Landsat scenes for tiles: {tiles}")

        # Search for scenes
        scenes = []
        scenes_checked = 0
        scenes_filtered_by_cloud = 0
        sensor_prefix = f"{self.COLLECTION}/{self.LEVEL}/standard/oli-tirs/{year}/"

        # List all paths for the year
        response = await self.list_s3_objects(sensor_prefix)

        for prefix_info in response.get("CommonPrefixes", []):
            path_prefix = prefix_info["Prefix"]
            path_num = int(path_prefix.split("/")[-2])

            # Check if any of our tiles are in this path
            relevant_tiles = [t for t in tiles if int(t[:3]) == path_num]
            if not relevant_tiles:
                continue

            # List rows in this path
            row_response = await self.list_s3_objects(path_prefix)

            for row_info in row_response.get("CommonPrefixes", []):
                row_prefix = row_info["Prefix"]
                row_num = int(row_prefix.split("/")[-2])

                # Check if this path/row matches our tiles
                current_tile = f"{path_num:03d}{row_num:03d}"
                if current_tile not in relevant_tiles:
                    continue

                # List scenes in this path/row
                scene_response = await self.list_s3_objects(row_prefix)

                # Process scenes with cloud cover check
                for scene_info in scene_response.get("CommonPrefixes", []):
                    scene_prefix = scene_info["Prefix"]
                    scene_id = scene_prefix.split("/")[-2]

                    # Parse scene metadata from ID
                    scene_date = datetime.strptime(scene_id.split("_")[3], "%Y%m%d")

                    # Apply date filter
                    if date_range:
                        if scene_date < date_range[0] or scene_date > date_range[1]:
                            continue

                    scenes_checked += 1

                    # Get actual cloud cover from MTL metadata
                    cloud_cover = await self.get_scene_cloud_cover(scene_prefix, scene_id)

                    # Only add scene if it passes cloud cover filter
                    if cloud_cover <= cloud_cover_max:
                        scenes.append(
                            {
                                "scene_id": scene_id,
                                "path": path_num,
                                "row": row_num,
                                "date": scene_date,
                                "s3_prefix": scene_prefix,
                                "cloud_cover": cloud_cover,
                            }
                        )
                    else:
                        scenes_filtered_by_cloud += 1
                        logger.debug(
                            f"Skipping scene {scene_id} with cloud cover {cloud_cover:.1f}%"
                        )

        logger.info(
            f"Found {len(scenes)} Landsat scenes matching criteria "
            f"(cloud cover <= {cloud_cover_max}%). "
            f"Checked {scenes_checked} scenes, "
            f"filtered {scenes_filtered_by_cloud} due to cloud cover."
        )
        return scenes

    async def prepare_download(self, item: dict[str, Any], output_dir: Path) -> tuple[str, Path]:
        """
        Prepare download URL and output path for Landsat band.

        Args:
            item: Download item with scene and band info
            output_dir: Base output directory

        Returns:
            Tuple of (s3_url, local_path)
        """
        scene_id = item["scene_id"]
        band = item["band"]
        date = item["date"]

        # S3 key
        s3_key = f"{item['s3_prefix']}{scene_id}_{self.BAND_MAPPING[band]}"
        s3_url = f"s3://{self.BUCKET}/{s3_key}"

        # Local path organized by date
        date_dir = output_dir / date.strftime("%Y/%m/%d") / scene_id
        local_path = date_dir / f"{scene_id}_{self.BAND_MAPPING[band]}"

        return s3_url, local_path

    async def download_file(self, url: str, output_path: Path, **kwargs) -> DownloadResult:
        """Override to use S3 download instead of HTTP"""
        result = DownloadResult(url=url, local_path=output_path, status=DownloadStatus.PENDING)

        # Check if file exists
        if output_path.exists() and output_path.stat().st_size > 0:
            result.status = DownloadStatus.COMPLETED
            result.size_bytes = output_path.stat().st_size
            return result

        # Parse S3 URL
        if not url.startswith("s3://"):
            raise ValueError(f"Expected S3 URL, got: {url}")

        parts = url[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1]

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Download from S3
        start_time = asyncio.get_event_loop().time()

        # Mark as in progress and notify callback
        result.status = DownloadStatus.IN_PROGRESS
        if self.progress_callback:
            await self.progress_callback(result)

        try:
            async with self._semaphore:
                # Create default session (will use AWS credentials if available, or anonymous)
                session = aioboto3.Session()
                async with session.client("s3", config=self.s3_config) as s3:  # type: ignore
                    # Use temporary file
                    temp_path = output_path.with_suffix(".tmp")

                    try:
                        await s3.download_file(
                            bucket,
                            key,
                            str(temp_path),
                            ExtraArgs={
                                "RequestPayer": "requester"
                            },  # Required for USGS Landsat bucket
                        )

                        # Atomic rename
                        temp_path.rename(output_path)

                        result.status = DownloadStatus.COMPLETED
                        result.size_bytes = output_path.stat().st_size
                        result.duration_seconds = asyncio.get_event_loop().time() - start_time

                        # Notify callback of completion
                        if self.progress_callback:
                            await self.progress_callback(result)

                    except Exception:
                        if temp_path.exists():
                            temp_path.unlink()
                        raise

        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            result.status = DownloadStatus.FAILED
            result.error = str(e)

            # Notify callback of failure
            if self.progress_callback:
                await self.progress_callback(result)

            # Retry logic would go here
            raise

        return result

    async def download_scene_bands(
        self, scene: dict[str, Any], bands: list[str], output_dir: Path
    ) -> dict[str, Path]:
        """
        Download multiple bands for a scene.

        Args:
            scene: Scene metadata
            bands: List of band names
            output_dir: Output directory

        Returns:
            Dictionary mapping band names to file paths
        """
        # Log scene download details
        logger.info(
            f"Downloading scene {scene['scene_id']} "
            f"(path={scene['path']:03d}, row={scene['row']:03d}, "
            f"date={scene['date'].strftime('%Y-%m-%d')}, "
            f"cloud_cover={scene['cloud_cover']:.1f}%)"
        )

        # Create download items
        items = []
        for band in bands:
            items.append({**scene, "band": band})

        # Download all bands concurrently
        results = await self.download_batch(items, output_dir)

        # Map results to band names
        band_paths = {}
        for item, result in zip(items, results):
            if result.status == DownloadStatus.COMPLETED:
                band_paths[item["band"]] = result.local_path
            else:
                logger.warning(f"Failed to download {item['band']}: {result.error}")
                band_paths[item["band"]] = None

        return band_paths


async def download_landsat(
    dates: list[datetime],
    bbox: tuple[float, float, float, float],
    bands: list[str],
    output_dir: str = "./data/landsat",
    cloud_cover_max: float = 20.0,
    show_progress: bool = True,
) -> dict[datetime, dict[str, list[Path]]]:
    """
    High-level async function to download Landsat data for multiple dates.

    Args:
        dates: List of target dates
        bbox: Bounding box
        bands: List of band names to download
        output_dir: Output directory
        cloud_cover_max: Maximum cloud cover
        show_progress: Show progress bar

    Returns:
        Dictionary mapping dates to band paths
    """
    logger.info(f"Downloading Landsat data for {len(dates)} dates")
    logger.info(f"Bounding box: {bbox}")
    logger.info(f"Cloud cover max: {cloud_cover_max}%")

    # Group dates by year for efficient searching
    dates_by_year = {}
    for date in dates:
        if date.year not in dates_by_year:
            dates_by_year[date.year] = []
        dates_by_year[date.year].append(date)

    async with AsyncLandsatDownloader() as downloader:
        all_downloads = []
        tiles_found = set()  # Track which tiles we found scenes for
        tiles_needed = set()  # Track all tiles we need

        # Find scenes for each year
        for year, year_dates in dates_by_year.items():
            # Date range for the year
            date_range = (
                min(year_dates) - timedelta(days=8),
                max(year_dates) + timedelta(days=8),
            )

            # Find scenes
            scenes = await downloader.find_landsat_scenes(
                year=year,
                bbox=bbox,
                date_range=date_range,
                cloud_cover_max=cloud_cover_max,
            )

            # Track tiles from found scenes
            for scene in scenes:
                tile_id = f"{scene['path']:03d}{scene['row']:03d}"
                tiles_found.add(tile_id)

            # Match scenes to requested dates
            for target_date in year_dates:
                # Group scenes by tile for this date
                scenes_by_tile = {}
                for scene in scenes:
                    tile_id = f"{scene['path']:03d}{scene['row']:03d}"
                    tiles_needed.add(tile_id)

                    diff = abs(scene["date"] - target_date)
                    if diff <= timedelta(days=8):
                        if tile_id not in scenes_by_tile or diff < abs(
                            scenes_by_tile[tile_id]["date"] - target_date
                        ):
                            scenes_by_tile[tile_id] = scene

                # Add downloads for all tiles found
                for tile_id, scene in scenes_by_tile.items():
                    all_downloads.append(
                        {
                            "date": target_date,
                            "scene": scene,
                            "bands": bands,
                            "tile_id": tile_id,
                        }
                    )

                # Log tiles not found for this date
                missing_tiles = tiles_needed - set(scenes_by_tile.keys())
                if missing_tiles:
                    logger.warning(
                        f"Missing tiles for {target_date.strftime('%Y-%m-%d')}: "
                        f"{sorted(missing_tiles)}"
                    )

        # Summary of tile coverage
        logger.info("\nTile coverage summary:")
        logger.info(f"  Total tiles needed: {len(tiles_needed)}")
        logger.info(f"  Tiles with data: {len(tiles_found)}")
        logger.info(f"  Tiles always missing: {sorted(tiles_needed - tiles_found)}")
        logger.info(f"  Total scene downloads: {len(all_downloads)}")

        # Set up progress tracking
        total_files = len(all_downloads) * len(bands)
        progress = None
        if show_progress:
            progress = DownloadProgressTracker(
                total_files=total_files, description="Downloading Landsat"
            )
            downloader.progress_callback = progress.update  # type: ignore

        # Download all scenes/bands concurrently
        download_tasks = []
        for download in all_downloads:
            task = downloader.download_scene_bands(
                scene=download["scene"],
                bands=download["bands"],
                output_dir=Path(output_dir),
            )
            download_tasks.append((download["date"], download["tile_id"], task))

        # Execute all downloads
        results = {}
        tiles_downloaded_by_date = {}

        for date, tile_id, task in download_tasks:
            band_paths = await task
            if date not in results:
                results[date] = {}
                tiles_downloaded_by_date[date] = set()

            # Track successful downloads
            if any(path is not None for path in band_paths.values()):
                tiles_downloaded_by_date[date].add(tile_id)

            # Organize by band
            for band, path in band_paths.items():
                if band not in results[date]:
                    results[date][band] = []
                if path:
                    results[date][band].append(path)

        # Log final summary
        logger.info("\nDownload summary by date:")
        for date in sorted(tiles_downloaded_by_date.keys()):
            logger.info(
                f"  {date.strftime('%Y-%m-%d')}: "
                f"{len(tiles_downloaded_by_date[date])} tiles downloaded"
            )

        if progress:
            progress.finish()

    return results

# Legacy sync function needed by pipeline
def filter_landsat_scenes(
    year: int,
    month: int,
    day: int,
    path: int,
    row: int,
    collection: str = "collection02",
) -> str | None:
    """Find Landsat scene directory for given date and path/row.

    This is a sync wrapper for compatibility.
    """
    import boto3

    yr = str(year)
    mo = str(month).zfill(2)
    da = str(day).zfill(2)
    path_str = str(path).zfill(3)
    row_str = str(row).zfill(3)

    prefix = f"{collection}/level-2/standard/oli-tirs/{yr}/{path_str}/{row_str}/"

    # Simple sync implementation
    s3 = boto3.client("s3", config=boto3.session.Config(signature_version=boto3.session.UNSIGNED))

    result = s3.list_objects_v2(
        Bucket="usgs-landsat",
        Prefix=prefix,
        Delimiter="/",
    )

    if "CommonPrefixes" not in result:
        return None

    directories = [x["Prefix"] for x in result["CommonPrefixes"]]

    # Filter directories by date
    yyyymmdd = f"{yr}{mo}{da}"

    for directory in directories:
        parts = directory.split("/")
        if len(parts) > 0:
            base_name = parts[-2] if parts[-1] == "" else parts[-1]
            if yyyymmdd in base_name:
                return directory

    return None

