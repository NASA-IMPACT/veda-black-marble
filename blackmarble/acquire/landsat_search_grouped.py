"""Date-grouped Landsat scene selection.

Simple approach: Get all scenes in time window, group by date, pick the date with best coverage.
"""

import logging
from datetime import date, datetime
from typing import Any

from ..config import LandsatAcquisitionConfig
from .landsat_search import (
    S3Client,
    SceneMeta,
    SearchParams,
    get_all_scenes_in_window,
    with_s3_client,
)

logger = logging.getLogger(__name__)


async def search_best_scenes(
    bbox: tuple[float, float, float, float],
    target_date: datetime,
    config: LandsatAcquisitionConfig | None = None,
    allow_incomplete: bool = False,
    find_scenes_func: Any = None,
) -> list[SceneMeta]:
    """Search for best Landsat scenes, preferring same-date coverage.

    This is the ONLY function you need. It finds scenes and tries to get
    them all from the same date for temporal consistency.
    """
    if config is None:
        config = LandsatAcquisitionConfig()

    # Get required tiles
    from .wrs_utils import get_wrs_tiles_for_bbox_tuple

    tile_dicts = get_wrs_tiles_for_bbox_tuple(bbox)
    required_tiles = {(t["path"], t["row"]) for t in tile_dicts}

    logger.info(f"Need {len(required_tiles)} tiles for bbox {bbox}")
    logger.info(f"Required tiles (path/row): {sorted(required_tiles)}")

    # Setup search params
    params = SearchParams(
        bbox=bbox,
        target_date=target_date,
        config=config,
        allow_incomplete=True,  # We'll check completeness ourselves
    )

    # Get the find_scenes function
    if find_scenes_func is None:
        from .landsat import AsyncLandsatDownloader

        downloader = AsyncLandsatDownloader(aws_unsigned=False)
        find_scenes_func = downloader.find_landsat_scenes

    # Run the search
    async def search_with_client(s3_client: S3Client) -> list[SceneMeta]:
        scenes_func = find_scenes_func

        # Get ALL scenes in time window
        all_scenes = await get_all_scenes_in_window(
            params=params,
            find_scenes_func=scenes_func,
            s3_client=s3_client,
        )

        if not all_scenes:
            logger.warning("No scenes found")
            return []

        logger.info(f"Found {len(all_scenes)} total scenes in time window")

        tolerance_days = config.tolerance_days
        scenes_by_date: dict[date, list[SceneMeta]] = {}
        for scene in all_scenes:
            date_key = scene.date.date()  # Just the date, no time
            if date_key not in scenes_by_date:
                scenes_by_date[date_key] = []
            scenes_by_date[date_key].append(scene)

        logger.info(f"Scenes grouped into {len(scenes_by_date)} unique dates")
        logger.info(f"Using tolerance of {tolerance_days} days for grouping")

        # Find date groups with complete coverage (considering tolerance)
        complete_groups = []
        processed_dates = set()

        for base_date in sorted(scenes_by_date.keys()):
            if base_date in processed_dates:
                continue

            group_scenes = []
            group_dates = []

            for d in scenes_by_date:
                if abs((d - base_date).days) <= tolerance_days:
                    group_scenes.extend(scenes_by_date[d])
                    group_dates.append(d)
                    processed_dates.add(d)
            tiles = {(s.path, s.row) for s in group_scenes}
            coverage_pct = len(tiles) / len(required_tiles) * 100

            date_range = (
                f"{min(group_dates)} to {max(group_dates)}"
                if len(group_dates) > 1
                else str(base_date)
            )
            logger.debug(f"  Group {date_range}: {len(tiles)} tiles ({coverage_pct:.0f}% coverage)")

            if tiles >= required_tiles:
                avg_cloud = sum(s.cloud_cover for s in group_scenes) / len(group_scenes)
                complete_groups.append((base_date, group_scenes, avg_cloud, group_dates))
                logger.debug(f"    -> Complete coverage, avg cloud: {avg_cloud:.1f}%")

        if not complete_groups:
            logger.warning(
                f"No date groups with complete coverage even with {tolerance_days} day tolerance"
            )

            best_coverage = 0
            for base_date in sorted(scenes_by_date.keys()):
                group_scenes = []
                for d in scenes_by_date:
                    if abs((d - base_date).days) <= tolerance_days:
                        group_scenes.extend(scenes_by_date[d])
                tiles = {(s.path, s.row) for s in group_scenes}
                coverage = len(tiles)
                if coverage > best_coverage:
                    best_coverage = coverage

            logger.warning(
                f"Best coverage found: {best_coverage} tiles out of {len(required_tiles)} required"
            )

            if not allow_incomplete:
                return []
            best_date = max(scenes_by_date.items(), key=lambda x: len(x[1]))
            return best_date[1]

        best_group = min(complete_groups, key=lambda x: x[2])
        base_date, group_scenes, avg_cloud, group_dates = best_group

        date_desc = (
            f"{min(group_dates)} to {max(group_dates)}" if len(group_dates) > 1 else str(base_date)
        )
        logger.info(
            f"Selected group {date_desc} with {len(group_scenes)} scenes from {len(group_dates)} dates, avg cloud {avg_cloud:.1f}%"
        )

        from .landsat_search import group_by_pathrow

        tiles = group_by_pathrow(group_scenes)
        result = []
        for tile_scenes in tiles.values():
            best = min(tile_scenes, key=lambda s: s.cloud_cover)
            result.append(best)

        return result

    return await with_s3_client(search_with_client)
