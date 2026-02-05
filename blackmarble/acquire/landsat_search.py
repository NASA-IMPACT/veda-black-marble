"""Functional implementation of time-window based Landsat acquisition.

This module provides pure functions for Landsat scene selection without OOP complexity.
All I/O and stateful operations are explicitly passed as dependencies.
"""

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from ..config import LandsatAcquisitionConfig


logger = logging.getLogger(__name__)


# Pure Data Structures (Immutable)


@dataclass(frozen=True)
class SceneMeta:
    """Immutable metadata for a Landsat scene."""

    scene_id: str
    path: int
    row: int
    date: datetime
    cloud_cover: float  # 0-100
    spatial_coverage: float  # 0-1
    snow_cover: float  # 0-1
    off_nadir: bool
    sensor: str  # ETM, OLI_TIRS, etc.
    quality_score: float = 0.0  # Calculated
    s3_prefix: str = ""  # S3 location

    def to_scene_dict(self) -> dict[str, Any]:
        """
        Convert to the mutable dict structure consumed by
        AsyncLandsatDownloader.download_scene_bands().
        """
        return {
            "scene_id": self.scene_id,
            "path": self.path,
            "row": self.row,
            "date": self.date,
            "cloud_cover": self.cloud_cover,
            "s3_prefix": self.s3_prefix,
        }


@dataclass(frozen=True)
class SearchParams:
    """Immutable search parameters."""

    bbox: tuple[float, float, float, float]
    target_date: datetime
    config: LandsatAcquisitionConfig
    allow_incomplete: bool = False


# Pure Functions (No Side Effects)


def group_by_pathrow(scenes: list[SceneMeta]) -> dict[tuple[int, int], list[SceneMeta]]:
    """Group scenes by path/row (pure function)."""
    grouped = {}
    for scene in scenes:
        key = (scene.path, scene.row)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(scene)
    return grouped


def normalize_target_date(target_date: datetime) -> datetime:
    """Convert timezone-aware date to UTC naive (pure function)."""
    if target_date.tzinfo is not None:
        # Convert to UTC, preserving the actual time instant
        target_date_utc = target_date.astimezone(UTC)
        # Remove timezone info for datetime arithmetic
        return target_date_utc.replace(tzinfo=None)
    return target_date


def calculate_search_years(start_date: datetime, end_date: datetime) -> list[int]:
    """Calculate which years to search (pure function)."""
    years: set[int] = set()
    current = datetime(start_date.year, 1, 1)
    while current.year <= end_date.year:
        years.add(current.year)
        # Step exactly one calendar year ahead
        current = datetime(current.year + 1, 1, 1)
    return sorted(years)


def filter_duplicate_scenes(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter duplicate scenes including T1/T2 variants (pure function)."""
    seen_ids = set()
    filtered: list[dict[str, Any]] = []

    for scene_dict in scenes:
        scene_id = scene_dict.get("scene_id", "")

        # Remove only the final tier segment (T1/T2) to form base identifier.
        # This retains collection number (01/02) so variants are treated as different.
        parts = scene_id.split("_")
        base_id = "_".join(parts[:-1]) if parts else ""

        if base_id in seen_ids:
            logger.debug(f"Skipping duplicate/variant scene: {scene_id}")
            continue

        seen_ids.add(base_id)
        filtered.append(scene_dict)

    return filtered


def parse_scene_id(scene_id: str) -> dict[str, Any]:
    """Parse scene ID components (pure function)."""
    parts = scene_id.split("_")

    # Validate format first ─ exactly 7 segments for Collection-2 IDs
    # e.g. LC08_L2SP_013032_20230809_20230812_02_T1
    if len(parts) < 7:
        raise ValueError(
            f"Invalid Landsat scene ID '{scene_id}': "
            "expected at least 7 underscore-separated fields"
        )

    sensor_code = parts[0]  # LC08, LE07, LC09, etc.
    path_row_segment = parts[2]
    if len(path_row_segment) != 6 or not path_row_segment.isdigit():
        raise ValueError(f"Invalid path/row segment '{path_row_segment}' in scene ID '{scene_id}'")

    path = int(path_row_segment[:3])
    row = int(path_row_segment[3:])
    date_str = parts[3]
    if len(date_str) != 8 or not date_str.isdigit():
        raise ValueError(f"Invalid acquisition date '{date_str}' in scene ID '{scene_id}'")

    # Determine sensor type
    sensor_map = {
        "LC08": "OLI_TIRS",  # Landsat 8
        "LC09": "OLI_TIRS",  # Landsat 9
        "LC09R": "OLI_TIRS",  # Landsat 9 RMAP (prefix LC09_RMAP → sliced to LC09R)
        "LE07": "ETM",  # Landsat 7
        "LT05": "TM",  # Landsat 5
        "LT04": "TM",  # Landsat 4
    }
    sensor = sensor_map.get(sensor_code, sensor_map.get(sensor_code[:4], "UNKNOWN"))

    # Parse acquisition date
    scene_date = datetime.strptime(date_str, "%Y%m%d")

    return {
        "path": path,
        "row": row,
        "date": scene_date,
        "sensor": sensor,
    }


# MTL Parsing Functions (Pure)


def parse_mtl_text(text: str) -> dict[str, Any]:
    """Parse MTL text format into dictionary (pure function)."""
    result = {"LANDSAT_METADATA_FILE": {}}
    current_group = result["LANDSAT_METADATA_FILE"]
    group_stack: list[tuple[str, dict[str, Any]]] = [("LANDSAT_METADATA_FILE", current_group)]

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Handle group start
        if line.startswith("GROUP = "):
            group_name = line.replace("GROUP = ", "")
            new_group: dict[str, Any] = {}
            current_group[group_name] = new_group
            group_stack.append((group_name, current_group))
            current_group = new_group

        # Handle labelled END_GROUP or bare END_GROUP
        elif line.startswith("END_GROUP"):
            parts = line.split("=", 1)
            label = parts[1].strip() if len(parts) == 2 else None
            if len(group_stack) > 1:
                # Pop until the matching label or just the most recent
                while group_stack:
                    last_label, parent = group_stack.pop()
                    current_group = parent
                    if label is None or label == last_label:
                        break

        # Handle key-value pairs
        elif "=" in line and not line.startswith("END"):
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"')

            # Try to convert to appropriate type
            converted_value: str | float | int = value
            with contextlib.suppress(ValueError):
                converted_value = float(value) if "." in value else int(value)

            current_group[key] = converted_value

    return result


def extract_cloud_cover(mtl_data: dict[str, Any]) -> float:
    """Extract cloud cover from MTL data (pure function)."""
    try:
        metadata = mtl_data.get("LANDSAT_METADATA_FILE", {})
        attrs = metadata.get("IMAGE_ATTRIBUTES", {})

        # Prefer CLOUD_COVER_LAND if available
        cloud = attrs.get("CLOUD_COVER_LAND", attrs.get("CLOUD_COVER", 100.0))
        return float(cloud)
    except (TypeError, ValueError) as e:
        logger.debug(f"Error extracting cloud cover: {e}")
        return 100.0  # Worst case default


def extract_snow_cover(mtl_data: dict[str, Any]) -> float:
    """Extract snow cover from MTL data (pure function)."""
    try:
        metadata = mtl_data.get("LANDSAT_METADATA_FILE", {})
        attrs = metadata.get("IMAGE_ATTRIBUTES", {})

        # Look for snow/ice percentage
        snow_pixels = float(attrs.get("SNOW_ICE_PIXELS", 0.0))
        total_pixels = float(attrs.get("TOTAL_PIXELS", 1.0))

        if total_pixels > 0:
            return snow_pixels / total_pixels
        return 0.0
    except (TypeError, ValueError, ZeroDivisionError) as e:
        logger.debug(f"Error extracting snow cover: {e}")
        return 0.0


def extract_off_nadir(mtl_data: dict[str, Any]) -> bool:
    """Extract off-nadir flag from MTL data (pure function)."""
    try:
        metadata = mtl_data.get("LANDSAT_METADATA_FILE", {})
        proj = metadata.get("PROJECTION_ATTRIBUTES", {})

        nadir_flag = proj.get("NADIR_OFFNADIR", "NADIR")
        return nadir_flag.upper() != "NADIR"
    except (TypeError, KeyError) as e:
        logger.debug(f"Error extracting off-nadir flag: {e}")
        return False


def extract_spatial_coverage(mtl_data: dict[str, Any]) -> float:
    """Extract spatial coverage from MTL data (pure function)."""
    try:
        metadata = mtl_data.get("LANDSAT_METADATA_FILE", {})
        attrs = metadata.get("IMAGE_ATTRIBUTES", {})

        # Check for data coverage metrics
        data_pixels = float(attrs.get("DATA_PIXELS", 0))
        total_pixels = float(attrs.get("TOTAL_PIXELS", 1))

        if total_pixels > 0 and data_pixels >= 0:
            return min(1.0, data_pixels / total_pixels)
        elif total_pixels == 0:
            logger.warning("TOTAL_PIXELS is 0 in MTL metadata")
            return 0.0  # No data

        # Default to zero coverage if metrics unavailable (conservative)
        logger.warning("Spatial coverage metrics unavailable, defaulting to 0.0")
        return 0.0
    except (TypeError, ValueError, ZeroDivisionError) as e:
        logger.debug(f"Error extracting spatial coverage: {e}")
        return 0.0


def parse_scene_metadata(
    scene_dict: dict[str, Any], mtl_data: dict[str, Any] | None
) -> SceneMeta | None:
    """Parse scene dictionary and MTL data into SceneMeta (pure function)."""
    try:
        scene_id = scene_dict["scene_id"]
        id_parts = parse_scene_id(scene_id)

        cloud_cover = scene_dict.get("cloud_cover", 100.0)
        s3_prefix = scene_dict.get("s3_prefix", "")

        if not mtl_data:
            logger.debug("No MTL metadata available for %s", scene_id)
            return None

        cloud_cover = extract_cloud_cover(mtl_data)
        snow_cover = extract_snow_cover(mtl_data)
        off_nadir = extract_off_nadir(mtl_data)
        spatial_coverage = extract_spatial_coverage(mtl_data)

        return SceneMeta(
            scene_id=scene_id,
            path=id_parts["path"],
            row=id_parts["row"],
            date=id_parts["date"],
            cloud_cover=cloud_cover,
            spatial_coverage=spatial_coverage,
            snow_cover=snow_cover,
            off_nadir=off_nadir,
            sensor=id_parts["sensor"],
            s3_prefix=s3_prefix,
        )

    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("Failed to parse scene metadata for %s: %s", scene_dict.get("scene_id"), exc)
        return None
    # Other exceptions (e.g. programming errors) propagate.


# I/O Functions (Side Effects Isolated)


class S3Client(Protocol):
    """Protocol for S3 client operations."""

    async def get_object(self, *, Bucket: str, Key: str, **kwargs) -> dict[str, Any]:  # noqa: N803
        """Get object from S3 bucket."""
        ...

    @property
    def exceptions(self) -> Any: ...


async def fetch_mtl_metadata(
    s3: S3Client,
    bucket: str,
    scene_prefix: str,
    scene_id: str,
) -> dict[str, Any] | None:
    """Fetch MTL metadata for a scene (I/O function)."""
    expected_bucket = "usgs-landsat"
    if bucket != expected_bucket:
        logger.error(
            "Refusing to fetch MTL for %s from unexpected bucket '%s' (expected '%s')",
            scene_id,
            bucket,
            expected_bucket,
        )
        return None

    mtl_key = f"{scene_prefix}{scene_id}_MTL.json"
    try:
        response = await s3.get_object(Bucket=bucket, Key=mtl_key, RequestPayer="requester")
        async with response["Body"] as body:
            content = await body.read()

        if not content:
            logger.warning("Empty MTL JSON file for %s", scene_id)
            return None

        try:
            return json.loads(content.decode("utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Corrupted JSON MTL for %s: %s", scene_id, exc)
            return None

    except Exception as e:
        # Check if it's NoSuchKey error
        error_code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
        if error_code == "NoSuchKey":
            # Fall back to text format .....................................
            txt_key = mtl_key.replace(".json", ".txt")
            try:
                response = await s3.get_object(Bucket=bucket, Key=txt_key, RequestPayer="requester")
                async with response["Body"] as body:
                    content = await body.read()

                if not content:
                    logger.warning("Empty MTL text file for %s", scene_id)
                    return None

                return parse_mtl_text(content.decode("utf-8"))
            except Exception as e2:
                error_code2 = getattr(e2, "response", {}).get("Error", {}).get("Code", "")
                if error_code2 == "NoSuchKey":
                    logger.debug("No MTL file found for %s", scene_id)
                else:
                    logger.debug("Failed to retrieve MTL text for %s: %s", scene_id, e2)
                return None
        elif error_code == "AccessDenied":
            logger.debug("Access denied for MTL file %s (scene may not exist)", scene_id)
            logger.debug("Tried key: %s", mtl_key)
            return None
        else:
            logger.exception("Failed to retrieve MTL for %s: %s", scene_id, str(e))
            logger.debug("Tried key: %s", mtl_key)
            return None


async def fetch_mtl_with_semaphore(
    s3: S3Client,
    bucket: str,
    scene_dict: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> SceneMeta | None:
    """Fetch and parse scene with MTL metadata (I/O with concurrency control)."""
    async with semaphore:
        scene_id = scene_dict.get("scene_id", "")
        scene_prefix = scene_dict.get("s3_prefix", "")

        logger.debug(f"Fetching MTL for scene_id={scene_id}, prefix={scene_prefix}")

        mtl_data = await fetch_mtl_metadata(s3, bucket, scene_prefix, scene_id)
        return parse_scene_metadata(scene_dict, mtl_data)


# Resource Management (Bracket Pattern)


async def with_s3_client(
    func: Callable[[S3Client], Any],
    *,
    region_name: str = "us-west-2",
) -> Any:
    """Execute function with managed S3 client (bracket pattern)."""
    # USGS Landsat data requires signed requests with RequestPayer='requester'
    # Import here to avoid circular dependency
    from .landsat import create_s3_session

    session = create_s3_session()

    async with session.client(
        "s3",
        region_name=region_name,
        config=None,  # Use default (signed) configuration
    ) as s3_client:
        return await func(s3_client)


# Main Pipeline (Composition of Pure Functions)


async def get_all_scenes_in_window(
    params: SearchParams,
    find_scenes_func: Any,
    s3_client: S3Client,
    bucket: str = "usgs-landsat",
    max_concurrent_mtl: int = 20,
) -> list[SceneMeta]:
    """Get ALL scenes in time window without any selection.

    This is used by the grouped search to get all scenes before
    it does its own date-based grouping and selection.
    """
    # Normalize timezone
    target_date_utc = normalize_target_date(params.target_date)

    # Calculate search window
    window_start = target_date_utc - timedelta(days=params.config.window_days)
    window_end = target_date_utc + timedelta(days=params.config.window_days)

    logger.info(
        f"Getting all Landsat scenes between {window_start.date()} "
        f"and {window_end.date()} UTC (±{params.config.window_days} days)"
    )

    # Calculate years to search
    years = calculate_search_years(window_start, window_end)

    # Collect all candidates (I/O operation)
    all_scene_dicts = []
    for year in years:
        scenes = await find_scenes_func(
            year=year,
            bbox=params.bbox,
            date_range=(window_start, window_end),
            cloud_cover_max=95.0,  # Very permissive to get Tier 2 scenes
        )
        all_scene_dicts.extend(scenes)

    # Filter duplicates (pure)
    unique_scene_dicts = filter_duplicate_scenes(all_scene_dicts)

    # Fetch MTL metadata concurrently (I/O operation)
    semaphore = asyncio.Semaphore(max_concurrent_mtl)
    tasks = [
        fetch_mtl_with_semaphore(s3_client, bucket, scene_dict, semaphore)
        for scene_dict in unique_scene_dicts
    ]
    scene_metas = await asyncio.gather(*tasks)

    # Filter out failed fetches (pure)
    valid_scenes = [s for s in scene_metas if s is not None]

    return valid_scenes


# Public API
