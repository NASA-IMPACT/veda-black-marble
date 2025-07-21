"""Date-grouped Landsat scene selection.

Simple approach: Get all scenes in time window, group by date, pick the date with best coverage.
"""

import logging
from datetime import datetime
from typing import Any

from .landsat_search import (
    SceneMeta, 
    SearchParams,
    search_best_scenes_functional,
    with_s3_client,
    S3Client,
)
from blackmarble.config import LandsatAcquisitionConfig

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
    required_tiles = {(t['path'], t['row']) for t in tile_dicts}
    
    logger.info(f"Need {len(required_tiles)} tiles for bbox {bbox}")
    
    # Setup search params
    params = SearchParams(
        bbox=bbox,
        target_date=target_date, 
        config=config,
        allow_incomplete=True,  # We'll check completeness ourselves
    )
    
    # Run the search
    async def search_with_client(s3_client: S3Client) -> list[SceneMeta]:
        # Get the find_scenes function
        scenes_func = find_scenes_func
        if scenes_func is None:
            from .landsat import AsyncLandsatDownloader
            downloader = AsyncLandsatDownloader(aws_unsigned=False)
            scenes_func = downloader.find_landsat_scenes
        
        # Get ALL scenes in time window
        all_scenes = await search_best_scenes_functional(
            params=params,
            find_scenes_func=scenes_func,
            s3_client=s3_client,
        )
        
        if not all_scenes:
            logger.warning("No scenes found")
            return []
        
        # Group by date
        from datetime import date as DateType
        scenes_by_date: dict[DateType, list[SceneMeta]] = {}
        for scene in all_scenes:
            date_key = scene.date.date()  # Just the date, no time
            if date_key not in scenes_by_date:
                scenes_by_date[date_key] = []
            scenes_by_date[date_key].append(scene)
        
        # Find dates with complete coverage
        complete_dates = []
        for date, scenes in scenes_by_date.items():
            tiles = {(s.path, s.row) for s in scenes}
            if tiles >= required_tiles:  # Has all required tiles
                avg_cloud = sum(s.cloud_cover for s in scenes) / len(scenes)
                complete_dates.append((date, scenes, avg_cloud))
        
        if not complete_dates:
            logger.warning(f"No dates with complete coverage. Best coverage: {max(len(scenes) for scenes in scenes_by_date.values())} tiles")
            if not allow_incomplete:
                return []
            # Fall back to best coverage we can get
            best_date = max(scenes_by_date.items(), key=lambda x: len(x[1]))
            return best_date[1]
        
        # Pick date with lowest average cloud cover
        best_date = min(complete_dates, key=lambda x: x[2])
        logger.info(f"Selected {best_date[0]} with {len(best_date[1])} tiles, avg cloud {best_date[2]:.1f}%")
        
        # Return best scene per tile from that date
        from .landsat_search import group_by_pathrow
        tiles = group_by_pathrow(best_date[1])
        result = []
        for tile_scenes in tiles.values():
            # Pick best scene for this tile
            best = min(tile_scenes, key=lambda s: s.cloud_cover)
            result.append(best)
        
        return result
    
    return await with_s3_client(search_with_client)