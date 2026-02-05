"""Orchestrator for front-loading all downloads asynchronously."""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# from .async_base import DownloadProgressTracker  # Currently unused
from .landsat_search_grouped import search_best_scenes as search_best_scenes_grouped
from .landsat import AsyncLandsatDownloader
from .viirs import download_viirs

logger = logging.getLogger(__name__)


@dataclass
class DownloadPlan:
    """Plan for all downloads needed by the pipeline"""

    # Target parameters
    bbox: tuple[float, float, float, float]
    primary_date: datetime

    # Landsat parameters
    landsat_dates: list[datetime] = field(default_factory=lambda: [])
    landsat_bands: list[str] = field(default_factory=lambda: ["B3", "B4", "B5", "QA_PIXEL"])
    landsat_cloud_max: float = 30.0
    landsat_acquisition_config: Any | None = None  # LandsatAcquisitionConfig

    # VIIRS parameters
    viirs_layers: list[str] = field(
        default_factory=lambda: [
            "Gap_Filled_DNB_BRDF-Corrected_NTL",
            "Latest_High_Quality_Retrieval",
        ]
    )

    # OSM parameters
    osm_road_types: list[str] = field(
        default_factory=lambda: [
            "motorway",
            "trunk",
            "primary",
            "secondary",
            "tertiary",
            "residential",
            "service",
            "unclassified",
            "road",
            "busway",
        ]
    )

    # Output configuration
    output_dir: Path = Path("./data")
    cache_downloads: bool = True

    @classmethod
    def from_pipeline_config(
        cls,
        bbox: tuple[float, float, float, float],
        date: datetime,
        config: Any = None,  # PipelineConfig
        data_dir: str | Path | None = None,
    ) -> "DownloadPlan":
        """Create download plan from pipeline configuration"""
        # Calculate Landsat dates based on compositing period
        composite_months = 12  # Default
        if config and hasattr(config, "acquisition"):
            composite_months = getattr(config.acquisition, "composite_months", 12)

        # Start with the target date
        landsat_dates = [date]

        # Add historical months (composite_months includes the target month)
        for i in range(1, composite_months):
            historical_date = date - timedelta(days=i * 30)
            landsat_dates.append(historical_date)

        logger.debug(
            f"Generated {len(landsat_dates)} Landsat dates for composite_months={composite_months}"
        )

        # Get output directory
        output_dir = Path(data_dir) if data_dir is not None else Path("./data")
        cache_enabled = True
        cloud_cover_max = 30.0  # Default matching config.py
        landsat_acquisition_config = None

        if config and data_dir is None:
            output_dir = Path(getattr(config, "data_dir", "./data"))
            cache_enabled = getattr(config, "cache_downloads", True)
            # Try to get cloud_cover_max from acquisition config
            if hasattr(config, "acquisition") and hasattr(config.acquisition, "cloud_cover_max"):
                cloud_cover_max = config.acquisition.cloud_cover_max
            # Get landsat acquisition config
            if hasattr(config, "acquisition") and hasattr(
                config.acquisition, "landsat_acquisition"
            ):
                landsat_acquisition_config = config.acquisition.landsat_acquisition

        return cls(
            bbox=bbox,
            primary_date=date,
            landsat_dates=landsat_dates,
            landsat_cloud_max=cloud_cover_max,
            landsat_acquisition_config=landsat_acquisition_config,
            output_dir=output_dir,
            cache_downloads=cache_enabled,
        )


@dataclass
class DownloadResults:
    """Results from all downloads"""

    viirs_data: dict[str, list[Path]] = field(default_factory=lambda: {})
    landsat_data: dict[datetime, dict[str, list[Path]]] = field(default_factory=lambda: {})
    osm_data: Any | None = None  # OSM graph data

    download_time: float = 0.0
    total_files: int = 0
    failed_downloads: list[str] = field(default_factory=lambda: [])

    def summary(self) -> str:
        """Generate summary of download results"""
        viirs_files = sum(len(paths) for paths in self.viirs_data.values())
        landsat_files = sum(
            len(paths) for date_data in self.landsat_data.values() for paths in date_data.values()
        )

        return (
            f"Download Summary:\n"
            f"  Total time: {self.download_time:.1f} seconds\n"
            f"  VIIRS files: {viirs_files}\n"
            f"  Landsat files: {landsat_files}\n"
            f"  OSM data: {'Yes' if self.osm_data else 'No'}\n"
            f"  Failed downloads: {len(self.failed_downloads)}"
        )


class AsyncDownloadOrchestrator:
    """Orchestrates concurrent downloads from multiple sources"""

    def __init__(
        self,
        earthdata_token: str | None = None,
        show_progress: bool = True,
        max_concurrent_total: int = 20,
    ):
        """
        Args:
            earthdata_token: NASA Earthdata token for VIIRS
            show_progress: Show progress bars
            max_concurrent_total: Maximum total concurrent downloads
        """
        self.earthdata_token = earthdata_token
        self.show_progress = show_progress
        self.max_concurrent_total = max_concurrent_total

        # Semaphore for total concurrency
        self._total_semaphore = asyncio.Semaphore(max_concurrent_total)

    async def download_all(
        self,
        plan: DownloadPlan,
        progress_callback: Callable[[str], None] | None = None,
    ) -> DownloadResults:
        """
        Execute all downloads according to plan.

        Args:
            plan: Download plan
            progress_callback: Optional callback for status updates

        Returns:
            DownloadResults with all downloaded data
        """
        start_time = time.time()
        results = DownloadResults()

        # Estimate total files for progress (currently unused)
        # estimated_files = (
        #     3  # VIIRS tiles (estimate)
        #     + len(plan.landsat_dates) * len(plan.landsat_bands) * 2  # Landsat
        #     + 1  # OSM
        # )

        if progress_callback:
            progress_callback(f"Starting downloads for {len(plan.landsat_dates)} dates...")

        # Create all download tasks
        tasks: list[asyncio.Task[Any]] = []
        task_names: list[str] = []

        # VIIRS download task
        if self.earthdata_token:
            logger.info("Creating VIIRS download task")
            viirs_task = asyncio.create_task(self._download_viirs(plan))
            tasks.append(viirs_task)
            task_names.append("VIIRS")
        else:
            logger.warning("No Earthdata token provided, skipping VIIRS downloads")

        # Landsat download task
        landsat_task = asyncio.create_task(self._download_landsat(plan))
        tasks.append(landsat_task)
        task_names.append("Landsat")

        # OSM download task
        osm_task = asyncio.create_task(self._download_osm(plan))
        tasks.append(osm_task)
        task_names.append("OSM")

        # Execute all downloads concurrently
        logger.info(f"Starting concurrent downloads: {', '.join(task_names)}")

        # Run all tasks
        task_results: list[Any | BaseException] = await asyncio.gather(
            *tasks, return_exceptions=True
        )

        # Process results
        for name, result in zip(task_names, task_results):
            if isinstance(result, Exception):
                logger.error(f"{name} download failed: {result}")
                # Include full traceback for debugging
                import traceback

                tb_str = "".join(
                    traceback.format_exception(type(result), result, result.__traceback__)
                )
                logger.error(f"{name} traceback:\n{tb_str}")
                results.failed_downloads.append(f"{name}: {str(result)}")
            else:
                if name == "VIIRS":
                    # Only set VIIRS data if we actually got results
                    if result and isinstance(result, dict):
                        # Type narrow result to dict[str, list[Path]]
                        has_files = False
                        for files in result.values():
                            if isinstance(files, list) and len(files) > 0:
                                has_files = True
                                break
                        if has_files:
                            results.viirs_data = result
                    else:
                        logger.warning("VIIRS download returned empty results")
                elif name == "Landsat":
                    if isinstance(result, dict):
                        results.landsat_data = result
                elif name == "OSM":
                    results.osm_data = result

        # Calculate final statistics
        results.download_time = time.time() - start_time
        results.total_files = sum(len(paths) for paths in results.viirs_data.values()) + sum(
            len(paths)
            for date_data in results.landsat_data.values()
            for paths in date_data.values()
        )

        logger.info(results.summary())

        if progress_callback:
            progress_callback(
                f"Downloads complete: {results.total_files} files in {results.download_time:.1f}s"
            )

        return results

    async def _download_viirs(self, plan: DownloadPlan) -> dict[str, list[Path]]:
        """Download VIIRS data"""
        logger.info(f"Downloading VIIRS data for {plan.primary_date}")
        logger.debug(f"Plan bbox: {plan.bbox}")
        token_info = "***" + self.earthdata_token[-4:] if self.earthdata_token else "None"
        logger.debug(f"Earthdata token: {token_info}")

        try:
            result = await download_viirs(
                date=plan.primary_date,
                bbox=plan.bbox,
                earthdata_token=self.earthdata_token or "",  # Provide empty string if None
                output_dir=str(plan.output_dir / "viirs"),
                show_progress=self.show_progress,
            )
            logger.debug(f"VIIRS download result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in _download_viirs: {e}")
            import traceback

            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise

    async def _download_landsat(self, plan: DownloadPlan) -> dict[datetime, dict[str, list[Path]]]:
        """
        Download Landsat data using the new time-window search strategy.

        Steps
        1. For each target date in the download plan, call `search_best_scenes_grouped`
           to obtain the optimal scene per WRS tile inside the configured time
           window (the window and quality weights are defined in
           landsat_search.LandsatAcquisitionConfig, imported internally by the
           helper).
        2. Use a single AsyncLandsatDownloader context to fetch the requested
           bands for every selected scene.  All I/O happens concurrently and is
           constrained by the downloader's internal semaphore.
        3. Return the data structure expected by the downstream pipeline:
           results[date][band] = list[Path]
        """
        logger.info(
            "Searching for optimal Landsat scenes across %d target dates (time-window)",
            len(plan.landsat_dates),
        )
        logger.debug("Plan bands: %s", plan.landsat_bands)
        logger.info(f"Target dates: {[d.strftime('%Y-%m-%d') for d in plan.landsat_dates]}")
        logger.info(f"Search bbox: {plan.bbox}")

        # Print WRS tile info once
        from .wrs_utils import get_wrs_tiles_for_bbox_tuple

        try:
            tile_dicts = get_wrs_tiles_for_bbox_tuple(plan.bbox)
            tiles = [f"{t['path']:03d}{t['row']:03d}" for t in tile_dicts]
            logger.debug("📍 WRS tiles that should cover bbox %s:", plan.bbox)
            logger.debug("   Total tiles: %d", len(tiles))
            logger.debug("   Tiles: %s", sorted(tiles))
            paths = sorted(set(t["path"] for t in tile_dicts))
            logger.debug("   Paths: %s", paths)
            # Empty line removed - not needed for logging
        except Exception as e:
            logger.warning(f"Could not determine WRS tiles: {e}")

        # Phase 1 ───────────────────────────────────────────────────────────
        # Discover best scenes (one coroutine per target date)
        # Always use date-grouped selection for spatial consistency
        search_func = search_best_scenes_grouped
        logger.info("Using date-grouped scene selection for spatial consistency")

        search_tasks: list[asyncio.Task[list[Any]]] = [
            asyncio.create_task(
                search_func(
                    bbox=plan.bbox,
                    target_date=target_date,
                    config=plan.landsat_acquisition_config,  # Pass the functional config
                    allow_incomplete=True,  # Always allow partial coverage
                )
            )
            for target_date in plan.landsat_dates
        ]

        scene_lists: list[list[Any]] = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Map target date → scenes
        date_to_scenes: dict[datetime, list[Any]] = {}
        for target_date, result in zip(plan.landsat_dates, scene_lists):
            if isinstance(result, Exception):
                logger.error("Scene search failed for %s: %s", target_date.date(), result)
                logger.error("Exception details: %s", str(result))
                # Check if it's an AttributeError about max_cloud_threshold
                if isinstance(result, AttributeError) and "max_cloud_threshold" in str(result):
                    logger.error(
                        "Config is missing max_cloud_threshold - this was removed in recent cleanup"
                    )
                raise result
            logger.info(f"Search result for {target_date.date()}: {len(result)} scenes found")
            if not result:
                logger.warning(f"No scenes found for {target_date.date()}")
            date_to_scenes[target_date] = result

        # Phase 2 ───────────────────────────────────────────────────────────
        # Download all requested bands for every scene
        results: dict[datetime, dict[str, list[Path]]] = {}
        async with AsyncLandsatDownloader() as downloader:
            download_coroutines: list[tuple[datetime, Any, asyncio.Future]] = []

            for date_key, scenes in date_to_scenes.items():
                for scene in scenes:
                    # Convert SceneMeta → dict expected by downloader
                    scene_dict = scene.to_scene_dict()
                    coro = downloader.download_scene_bands(
                        scene=scene_dict,
                        bands=plan.landsat_bands,
                        output_dir=plan.output_dir / "landsat",
                    )
                    download_coroutines.append((date_key, scene, asyncio.create_task(coro)))

            # Gather with error capture
            for date_key, _scene, task in download_coroutines:
                try:
                    band_paths = await task
                except Exception as exc:
                    logger.error("Download failed for scene %s: %s", _scene.scene_id, exc)
                    raise

                if date_key not in results:
                    results[date_key] = {}

                for band, path in band_paths.items():
                    if band not in results[date_key]:
                        results[date_key][band] = []
                    if path:
                        results[date_key][band].append(path)

        # Log summary
        for date_key, band_dict in results.items():
            logger.info(
                "Landsat %s: downloaded %d bands across %d tiles",
                date_key.strftime("%Y-%m-%d"),
                sum(len(v) for v in band_dict.values()),
                len({p.parent for v in band_dict.values() for p in v}),
            )

        return results

    async def _download_osm(self, plan: DownloadPlan) -> Any:
        """Download OSM data"""
        logger.info("Downloading OSM road network")

        # For now, use synchronous OSM download in thread pool
        # since osmnx doesn't have async support
        loop = asyncio.get_event_loop()

        try:
            from ..acquire import fetch_road_network

            # Prepare custom filter for road types
            custom_filter = None
            if plan.osm_road_types:
                # Include both main roads and their links (e.g., "motorway|motorway_link")
                road_parts: list[str] = []
                for rt in plan.osm_road_types:
                    road_parts.extend([rt, f"{rt}_link"])
                road_filter = "|".join(road_parts)
                custom_filter = f'["highway"~"{road_filter}"]'

            # Run in thread pool
            road_network = await loop.run_in_executor(
                None,
                fetch_road_network,
                plan.bbox,
                "all",  # network_type
                custom_filter,
                True,  # use_cache
                plan.output_dir / "osm_cache",
            )

            # TODO: Save to file for caching
            # For now just return the graph
            return road_network

        except Exception as e:
            logger.error(f"Failed to download OSM data: {e}")
            raise


def download_all_data_sync(
    bbox: tuple[float, float, float, float],
    date: datetime,
    config: Any,
    earthdata_token: str | None = None,
    data_dir: str | Path | None = None,
) -> DownloadResults:
    """
    Synchronous wrapper for async downloads.

    This is the main entry point from the synchronous pipeline.

    Args:
        bbox: Bounding box
        date: Primary date
        config: Pipeline configuration
        earthdata_token: NASA Earthdata token

    Returns:
        DownloadResults with all downloaded data
    """
    # Create download plan from config
    plan = DownloadPlan.from_pipeline_config(bbox, date, config, data_dir=data_dir)

    # Create orchestrator
    orchestrator = AsyncDownloadOrchestrator(earthdata_token=earthdata_token, show_progress=True)

    # Run async downloads
    return asyncio.run(orchestrator.download_all(plan))


# Convenience function for testing
async def test_async_downloads():
    """Test async downloads with a small area"""

    # Small test area (Madrid)
    bbox = (-3.8, 40.3, -3.6, 40.5)
    date = datetime(2023, 6, 15)

    # Create simple plan
    plan = DownloadPlan(
        bbox=bbox,
        primary_date=date,
        landsat_dates=[date],  # Just one date for testing
        output_dir=Path("./data/test"),
    )

    # Create orchestrator (would need real token)
    orchestrator = AsyncDownloadOrchestrator(earthdata_token="YOUR_TOKEN_HERE", show_progress=True)

    # Run downloads
    results = await orchestrator.download_all(plan)

    logger.info(results.summary())
    return results


if __name__ == "__main__":
    # Test the async downloads
    asyncio.run(test_async_downloads())
