"""Grouped date-based Landsat scene selection.

This module implements scene selection that prioritizes spatial consistency
by selecting scenes from the same satellite overpass or adjacent passes.

Design Philosophy:
-----------------
1. Temporal consistency is paramount - all tiles should be from the exact same date
2. Complete spatial coverage is required - no missing tiles accepted
3. Quality optimization is secondary - only applied after coverage requirements are met

Implementation:
--------------
- Stage 1: Filter to only dates with complete tile coverage (e.g., 9/9 tiles)
- Stage 2: Among complete dates, select the one with best average quality (lowest cloud cover)

Rationale:
----------
With 12 target dates throughout the year and ±45 day search windows, we have
~24 months of potential coverage. This redundancy allows us to be strict about
requiring complete coverage from a single date, eliminating temporal artifacts
while rarely missing data.
"""

import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Any

from .landsat_search import (
    SceneMeta, 
    SearchParams,
    search_best_scenes_functional,
    group_by_pathrow,
    with_s3_client,
    S3Client,
)
from blackmarble.config import LandsatAcquisitionConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DateGroup:
    """A group of scenes from the same or adjacent acquisition dates."""
    center_date: datetime
    scenes: list[SceneMeta]
    tile_coverage: set[tuple[int, int]]  # Set of (path, row) tuples
    
    @property
    def coverage_count(self) -> int:
        """Number of unique tiles covered."""
        return len(self.tile_coverage)
    
    @property
    def average_cloud_cover(self) -> float:
        """Average cloud cover across all scenes."""
        if not self.scenes:
            return 100.0
        return sum(s.cloud_cover for s in self.scenes) / len(self.scenes)
    
    @property
    def average_quality_score(self) -> float:
        """Average quality score across all scenes."""
        if not self.scenes:
            return 0.0
        return sum(s.quality_score for s in self.scenes) / len(self.scenes)
    
    @property
    def date_range_days(self) -> float:
        """Range of dates in this group (in days)."""
        if not self.scenes:
            return 0.0
        dates = [s.date for s in self.scenes]
        return (max(dates) - min(dates)).total_seconds() / 86400.0


def group_scenes_by_date(
    scenes: list[SceneMeta], 
    tolerance_days: float = 0.0
) -> list[DateGroup]:
    """Group scenes by acquisition date with tolerance.
    
    Args:
        scenes: List of scenes to group
        tolerance_days: Maximum days between scenes to be in same group.
                       0 means exact same date only (recommended for temporal consistency)
        
    Returns:
        List of date groups, sorted by center date
    """
    if not scenes:
        return []
    
    # Sort scenes by date
    sorted_scenes = sorted(scenes, key=lambda s: s.date)
    
    # Group scenes
    groups: list[list[SceneMeta]] = []
    current_group: list[SceneMeta] = [sorted_scenes[0]]
    
    for scene in sorted_scenes[1:]:
        # Check if scene belongs to current group
        days_diff = (scene.date - current_group[-1].date).total_seconds() / 86400.0
        
        if days_diff <= tolerance_days:
            current_group.append(scene)
        else:
            # Start new group
            groups.append(current_group)
            current_group = [scene]
    
    # Add final group
    if current_group:
        groups.append(current_group)
    
    # Convert to DateGroup objects
    date_groups = []
    for group in groups:
        # Calculate center date
        dates = [s.date for s in group]
        center_date = min(dates) + (max(dates) - min(dates)) / 2
        
        # Get tile coverage
        tile_coverage = {(s.path, s.row) for s in group}
        
        date_groups.append(DateGroup(
            center_date=center_date,
            scenes=group,
            tile_coverage=tile_coverage
        ))
    
    return date_groups


def select_best_date_group(
    date_groups: list[DateGroup],
    required_tiles: set[tuple[int, int]],
    weights: dict[str, float] | None = None,
    require_complete: bool = True
) -> DateGroup | None:
    """Select the best date group based on coverage and quality.
    
    Two-stage selection:
    1. Filter to only groups with complete coverage (if require_complete=True)
    2. Among complete groups, select based on quality metrics
    
    Args:
        date_groups: List of date groups to evaluate
        required_tiles: Set of (path, row) tuples that we need
        weights: Weights for scoring (quality, consistency only - coverage is binary)
        require_complete: If True, only consider groups with 100% tile coverage
        
    Returns:
        Best date group, or None if no suitable groups found
    """
    if not date_groups:
        return None
    
    if weights is None:
        weights = {
            "quality": 0.7,       # Average scene quality (cloud-free, etc)
            "consistency": 0.3,   # How close in time the scenes are (usually 0 with tolerance=0)
        }
    
    # Stage 1: Filter by coverage
    candidate_groups: list[DateGroup] = []
    incomplete_groups: list[tuple[float, DateGroup]] = []  # Track incomplete groups for fallback
    
    for group in date_groups:
        coverage_fraction = len(group.tile_coverage & required_tiles) / len(required_tiles)
        
        if coverage_fraction == 1.0:
            # Complete coverage - add to candidates
            candidate_groups.append(group)
            logger.debug(
                f"Date group {group.center_date.strftime('%Y-%m-%d')}: "
                f"COMPLETE coverage ({len(group.tile_coverage & required_tiles)}/{len(required_tiles)} tiles)"
            )
        else:
            # Incomplete coverage - track for potential fallback
            incomplete_groups.append((coverage_fraction, group))
            logger.debug(
                f"Date group {group.center_date.strftime('%Y-%m-%d')}: "
                f"INCOMPLETE coverage ({len(group.tile_coverage & required_tiles)}/{len(required_tiles)} tiles, "
                f"{coverage_fraction:.1%})"
            )
    
    # Check if we should use incomplete groups
    if not candidate_groups:
        if require_complete:
            # Log what we found
            if incomplete_groups:
                best_incomplete = max(incomplete_groups, key=lambda x: x[0])
                logger.warning(
                    f"No date groups with complete coverage found. "
                    f"Best coverage was {best_incomplete[0]:.1%} on {best_incomplete[1].center_date.strftime('%Y-%m-%d')}. "
                    f"Consider setting require_complete_coverage=False in config."
                )
            else:
                logger.warning("No date groups found at all.")
            return None
        else:
            # Use incomplete groups as candidates
            logger.info("No complete coverage found, using incomplete groups")
            candidate_groups = [g for _, g in incomplete_groups]
    
    # Stage 2: Score remaining candidates by quality
    scored_groups: list[tuple[float, DateGroup]] = []
    
    for group in candidate_groups:
        # Quality score (0-1) 
        quality_score = group.average_quality_score
        
        # Date consistency score (1 for same day, decreasing with spread)
        # With tolerance_days=0, this should always be 1.0
        import math
        consistency_score = math.exp(-group.date_range_days / 2.0) if group.date_range_days > 0 else 1.0
        
        # Combined score (no coverage component - that's already filtered)
        total_score = (
            weights["quality"] * quality_score +
            weights["consistency"] * consistency_score
        )
        
        scored_groups.append((total_score, group))
        
        logger.debug(
            f"Scoring date group {group.center_date.strftime('%Y-%m-%d')}: "
            f"quality={quality_score:.2f}, "
            f"consistency={consistency_score:.2f}, "
            f"total={total_score:.2f}"
        )
    
    # Sort by score (descending) and return best
    scored_groups.sort(key=lambda x: x[0], reverse=True)
    return scored_groups[0][1] if scored_groups else None


def deduplicate_scenes_by_tile(scenes: list[SceneMeta]) -> list[SceneMeta]:
    """Keep only the best scene per tile."""
    scenes_by_tile = group_by_pathrow(scenes)
    deduped = []
    
    for tile_scenes in scenes_by_tile.values():
        if tile_scenes:
            # Sort by quality score (descending) and take first
            sorted_scenes = sorted(tile_scenes, key=lambda s: s.quality_score, reverse=True)
            deduped.append(sorted_scenes[0])
    
    return deduped


async def search_best_scene_group(
    bbox: tuple[float, float, float, float],
    target_date: datetime,
    config: LandsatAcquisitionConfig | None = None,
    tolerance_days: float | None = None,
    find_scenes_func: Any = None,
) -> list[SceneMeta]:
    """Search for best group of Landsat scenes from same/adjacent overpasses.
    
    This function prioritizes spatial consistency by selecting scenes from
    the same satellite overpass or adjacent passes, rather than picking
    the best individual scene for each tile.
    
    With the default tolerance_days=0, all selected scenes will be from the
    exact same date, ensuring perfect temporal consistency. This may occasionally
    result in incomplete coverage, but with 12 target dates throughout the year,
    this is rarely an issue in practice.
    
    Args:
        bbox: Bounding box
        target_date: Target date
        config: Acquisition configuration  
        tolerance_days: Max days between scenes to group together
        find_scenes_func: Function to find scenes (for testing)
        
    Returns:
        List of scenes from the best date group
    """
    if config is None:
        config = LandsatAcquisitionConfig()
    
    # Use tolerance_days from config if not explicitly provided
    if tolerance_days is None:
        tolerance_days = getattr(config, 'tolerance_days', 2.0)
    
    # First, get all candidate scenes using existing search
    params = SearchParams(
        bbox=bbox,
        target_date=target_date, 
        config=config,
        allow_incomplete=True,  # We'll handle completeness ourselves
    )
    
    # Get required tiles
    from .wrs_utils import get_wrs_tiles_for_bbox_tuple
    tile_dicts = get_wrs_tiles_for_bbox_tuple(bbox)
    required_tiles = {(t['path'], t['row']) for t in tile_dicts}
    
    logger.info(
        f"Searching for scene groups covering {len(required_tiles)} tiles "
        f"near {target_date.strftime('%Y-%m-%d')}"
    )
    
    # Run the search
    async def search_with_client(s3_client: S3Client) -> list[SceneMeta]:
        # Get all scenes in time window (don't select best per tile yet)
        
        # Use the provided find_scenes_func or create one if needed
        # Note: This follows the same pattern as landsat_search.py
        scenes_func = find_scenes_func
        if scenes_func is None:
            # Import here to avoid circular dependency
            from .landsat import AsyncLandsatDownloader
            downloader = AsyncLandsatDownloader(aws_unsigned=False)
            scenes_func = downloader.find_landsat_scenes
        
        all_scenes = await search_best_scenes_functional(
            params=params,
            find_scenes_func=scenes_func,
            s3_client=s3_client,
        )
        
        # Group scenes by date
        date_groups = group_scenes_by_date(all_scenes, tolerance_days or 0.0)
        
        logger.info(f"Found {len(date_groups)} date groups from {len(all_scenes)} scenes")
        
        # Select best date group
        require_complete = getattr(config, 'require_complete_coverage', True)
        best_group = select_best_date_group(date_groups, required_tiles, require_complete=require_complete)
        
        if not best_group:
            logger.warning("No suitable date groups found")
            return []
        
        logger.info(
            f"Selected date group centered on {best_group.center_date.strftime('%Y-%m-%d')} "
            f"with {best_group.coverage_count} tiles, "
            f"avg cloud cover {best_group.average_cloud_cover:.1f}%"
        )
        
        # Deduplicate (keep best scene per tile) and return
        return deduplicate_scenes_by_tile(best_group.scenes)
    
    return await with_s3_client(search_with_client)


# Override the main search function to use grouped selection
async def search_best_scenes(
    bbox: tuple[float, float, float, float],
    target_date: datetime,
    config: LandsatAcquisitionConfig | None = None,
    allow_incomplete: bool = False,
    find_scenes_func: Any = None,
) -> list[SceneMeta]:
    """Search for best Landsat scenes using date-grouped selection.
    
    This overrides the default search to use grouped selection for better
    spatial consistency. Scenes will be from the same or adjacent satellite
    overpasses rather than the independently-best scene per tile.
    """
    return await search_best_scene_group(
        bbox=bbox,
        target_date=target_date,
        config=config,
        tolerance_days=None,  # Will use config value
        find_scenes_func=find_scenes_func,
    )


async def search_best_scenes_with_fallback(
    bbox: tuple[float, float, float, float],
    target_date: datetime,
    config: LandsatAcquisitionConfig | None = None,
    allow_incomplete: bool = False,
    find_scenes_func: Any = None,
) -> tuple[list[SceneMeta], str]:
    """Try grouped search first, fall back to per-tile if needed.
    
    This function implements a two-stage strategy:
    1. Try strict date grouping for temporal consistency
    2. If no complete groups found, fall back to per-tile selection
    
    Args:
        bbox: Bounding box
        target_date: Target date
        config: Acquisition configuration
        allow_incomplete: Whether to allow incomplete coverage
        find_scenes_func: Function to find scenes (for testing)
        
    Returns:
        Tuple of (scenes, strategy_used) where strategy_used is one of:
        - "date_grouped": Successful grouped selection
        - "per_tile_fallback": Fell back to per-tile selection
    """
    if config is None:
        config = LandsatAcquisitionConfig()
    
    # Try strict date grouping first
    scenes = await search_best_scene_group(
        bbox=bbox,
        target_date=target_date,
        config=config,
        tolerance_days=None,
        find_scenes_func=find_scenes_func,
    )
    
    if scenes:
        return scenes, "date_grouped"
    
    # Fallback to per-tile selection
    logger.info("No complete date groups found, falling back to per-tile selection")
    
    # Import here to avoid circular dependency
    from .landsat_search import search_best_scenes as search_per_tile
    
    scenes = await search_per_tile(
        bbox=bbox,
        target_date=target_date,
        config=config,
        allow_incomplete=allow_incomplete,
        find_scenes_func=find_scenes_func,
    )
    
    return scenes, "per_tile_fallback"