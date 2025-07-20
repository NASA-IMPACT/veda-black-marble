"""Tests for functional Landsat time window implementation."""

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from blackmarble.acquire.landsat_search import (
    SceneMeta,
    calculate_quality_score,
    group_by_pathrow,
    select_best_scene,
    normalize_target_date,
    calculate_search_years,
    filter_duplicate_scenes,
    parse_scene_id,
    extract_spatial_coverage,
    search_best_scenes,
    fetch_mtl_metadata,
    with_s3_client,
)
from blackmarble.config import LandsatAcquisitionConfig


class TestPureFunctions:
    """Test pure functions with no side effects."""
    
    def test_calculate_quality_score(self):
        """Test quality score calculation."""
        scene = SceneMeta(
            scene_id="LC08_L2SP_013032_20230809_20230812_02_T1",
            path=13,
            row=32,
            date=datetime(2023, 8, 9),
            cloud_cover=10.0,
            spatial_coverage=0.95,
            snow_cover=0.02,
            off_nadir=False,
            sensor="OLI_TIRS",
        )
        
        weights = {
            "spatial_coverage": 0.5,
            "cloud_free": 0.3,
            "snow_free": 0.15,
            "off_nadir": 0.05,
        }
        
        score = calculate_quality_score(scene, weights)
        
        # Manual calculation
        expected = (
            0.5 * 0.95 +  # spatial
            0.3 * 0.9 +   # cloud-free (1 - 10/100)
            0.15 * 0.98 + # snow-free (1 - 0.02)
            0.05          # nadir bonus
        )
        
        assert abs(score - expected) < 0.001
        assert 0.0 <= score <= 1.0
    
    def test_group_by_pathrow(self):
        """Test grouping scenes by path/row."""
        scenes = [
            SceneMeta("scene1", 13, 32, datetime(2023, 8, 1), 5.0, 1.0, 0.0, False, "OLI_TIRS"),
            SceneMeta("scene2", 13, 32, datetime(2023, 8, 9), 10.0, 1.0, 0.0, False, "OLI_TIRS"),
            SceneMeta("scene3", 13, 33, datetime(2023, 8, 1), 15.0, 1.0, 0.0, False, "OLI_TIRS"),
        ]
        
        grouped = group_by_pathrow(scenes)
        
        assert len(grouped) == 2
        assert (13, 32) in grouped
        assert (13, 33) in grouped
        assert len(grouped[(13, 32)]) == 2
        assert len(grouped[(13, 33)]) == 1
    
    def test_normalize_target_date(self):
        """Test timezone normalization."""
        # Timezone-aware date
        est_date = datetime(2023, 8, 10, 23, 30, tzinfo=timezone(timedelta(hours=-5)))
        utc_date = normalize_target_date(est_date)
        
        # Should convert to UTC
        assert utc_date.tzinfo is None
        assert utc_date == datetime(2023, 8, 11, 4, 30)  # +5 hours
        
        # Naive date should pass through
        naive_date = datetime(2023, 8, 10, 12, 0)
        assert normalize_target_date(naive_date) == naive_date
    
    def test_calculate_search_years(self):
        """Test year calculation including leap years."""
        # Single year
        years = calculate_search_years(
            datetime(2023, 6, 1),
            datetime(2023, 8, 31)
        )
        assert years == [2023]
        
        # Span multiple years
        years = calculate_search_years(
            datetime(2023, 12, 15),
            datetime(2024, 2, 15)
        )
        assert years == [2023, 2024]
        
        # Leap year edge case
        years = calculate_search_years(
            datetime(2024, 2, 29),  # Leap day
            datetime(2024, 3, 15)
        )
        assert years == [2024]
    
    def test_filter_duplicate_scenes(self):
        """Test duplicate filtering including T1/T2 variants."""
        scenes = [
            {"scene_id": "LC08_L2SP_013032_20230809_20230812_02_T1", "cloud_cover": 5.0},
            {"scene_id": "LC08_L2SP_013032_20230809_20230812_02_T2", "cloud_cover": 5.0},  # Variant
            {"scene_id": "LC08_L2SP_013032_20230809_20230812_02_T1", "cloud_cover": 5.0},  # Duplicate
            {"scene_id": "LC08_L2SP_013031_20230809_20230812_02_T1", "cloud_cover": 8.0},
        ]
        
        filtered = filter_duplicate_scenes(scenes)
        
        assert len(filtered) == 2
        # Should keep first occurrence
        assert filtered[0]["scene_id"].endswith("T1")
        assert "013032" in filtered[0]["scene_id"]
        assert "013031" in filtered[1]["scene_id"]
    
    def test_parse_scene_id(self):
        """Test scene ID parsing."""
        # Landsat 8
        parts = parse_scene_id("LC08_L2SP_013032_20230809_20230812_02_T1")
        assert parts["path"] == 13
        assert parts["row"] == 32
        assert parts["date"] == datetime(2023, 8, 9)
        assert parts["sensor"] == "OLI_TIRS"
        
        # Landsat 9 RMAP
        parts = parse_scene_id("LC09_RMAP_013032_20230809_20230812_02_T1")
        assert parts["sensor"] == "OLI_TIRS"
        
        # Landsat 7
        parts = parse_scene_id("LE07_L2SP_013032_20230809_20230812_02_T1")
        assert parts["sensor"] == "ETM"
    
    def test_parse_scene_id_invalid(self):
        """Test that invalid scene IDs raise ValueError."""
        bad_ids = [
            "LC08_013032_20230809_T1",      # too few parts
            "LC08_L2SP_01303_20230809_02_T1",  # path/row wrong length
            "LC08_L2SP_01303A_20230809_02_T1", # non-digit path/row
            "LC08_L2SP_013032_202308_02_T1",   # bad date
        ]
        for bad_id in bad_ids:
            with pytest.raises(ValueError):
                parse_scene_id(bad_id)
    
    def test_extract_spatial_coverage_edge_cases(self):
        """Test spatial coverage extraction with edge cases."""
        # Normal case
        mtl = {
            "LANDSAT_METADATA_FILE": {
                "IMAGE_ATTRIBUTES": {
                    "DATA_PIXELS": 7000000,
                    "TOTAL_PIXELS": 7800000,
                }
            }
        }
        coverage = extract_spatial_coverage(mtl)
        assert abs(coverage - 0.897) < 0.001
        
        # Zero total pixels
        mtl["LANDSAT_METADATA_FILE"]["IMAGE_ATTRIBUTES"]["TOTAL_PIXELS"] = 0
        coverage = extract_spatial_coverage(mtl)
        assert coverage == 0.0
        
        # Missing metrics
        mtl = {"LANDSAT_METADATA_FILE": {}}
        coverage = extract_spatial_coverage(mtl)
        assert coverage == 0.0  # Conservative default
    
    def test_select_best_scene_quality_precedence(self):
        """Test that quality score takes precedence in selection."""
        weights = {
            "spatial_coverage": 0.5,
            "cloud_free": 0.3,
            "snow_free": 0.15,
            "off_nadir": 0.05,
        }
        
        # High quality but old
        scene1 = SceneMeta(
            "scene1", 13, 32, datetime(2023, 7, 1),
            cloud_cover=5.0, spatial_coverage=1.0,
            snow_cover=0.0, off_nadir=False, sensor="OLI_TIRS"
        )
        
        # Lower quality but recent
        scene2 = SceneMeta(
            "scene2", 13, 32, datetime(2023, 8, 9),
            cloud_cover=40.0, spatial_coverage=0.8,
            snow_cover=0.1, off_nadir=True, sensor="OLI_TIRS"
        )
        
        best = select_best_scene([scene1, scene2], datetime(2023, 8, 10), weights)
        
        # Should prefer scene1 due to higher quality
        assert best.scene_id == "scene1"


class TestIOFunctions:
    """Test I/O functions with mocked dependencies."""
    
    @pytest.mark.asyncio
    async def test_fetch_mtl_metadata_json(self):
        """Test fetching JSON MTL."""
        s3_mock = AsyncMock()
        
        # Mock successful JSON response
        body_mock = AsyncMock()
        body_mock.__aenter__ = AsyncMock(return_value=body_mock)
        body_mock.__aexit__ = AsyncMock()
        body_mock.read = AsyncMock(return_value=json.dumps({
            "LANDSAT_METADATA_FILE": {
                "IMAGE_ATTRIBUTES": {"CLOUD_COVER": 10.0}
            }
        }).encode())
        body_mock.close = AsyncMock()
        
        s3_mock.get_object = AsyncMock(return_value={"Body": body_mock})
        
        mtl = await fetch_mtl_metadata(
            s3_mock,
            "usgs-landsat",
            "collection02/level-2/",
            "LC08_test"
        )
        
        assert mtl is not None
        assert mtl["LANDSAT_METADATA_FILE"]["IMAGE_ATTRIBUTES"]["CLOUD_COVER"] == 10.0
    
    @pytest.mark.asyncio
    async def test_fetch_mtl_metadata_missing(self):
        """Test handling missing MTL."""
        s3_mock = AsyncMock()
        s3_mock.exceptions.NoSuchKey = Exception
        s3_mock.get_object = AsyncMock(side_effect=Exception("NoSuchKey"))
        
        mtl = await fetch_mtl_metadata(
            s3_mock,
            "usgs-landsat",
            "collection02/level-2/",
            "LC08_test"
        )
        
        assert mtl is None
    
    @pytest.mark.asyncio
    async def test_fetch_mtl_metadata_invalid_bucket(self):
        """Should refuse to fetch from an unexpected bucket."""
        s3_mock = AsyncMock()
        # get_object should never be called because bucket is rejected
        mtl = await fetch_mtl_metadata(
            s3_mock,
            bucket="evil-bucket",
            scene_prefix="x/",
            scene_id="LC08_fake"
        )
        assert mtl is None
        s3_mock.get_object.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetch_mtl_metadata_corrupted_json(self):
        """Return None when JSON MTL is corrupted."""
        s3_mock = AsyncMock()
        bad_body = AsyncMock()
        bad_body.__aenter__ = AsyncMock(return_value=bad_body)
        bad_body.__aexit__ = AsyncMock()
        bad_body.read = AsyncMock(return_value=b"{not json")
        bad_body.close = AsyncMock()
        s3_mock.get_object = AsyncMock(return_value={"Body": bad_body})

        # passing expected bucket to avoid early exit
        mtl = await fetch_mtl_metadata(
            s3_mock,
            bucket="usgs-landsat",
            scene_prefix="x/",
            scene_id="LC08_corrupt"
        )
        assert mtl is None
    
    @pytest.mark.asyncio
    async def test_with_s3_client_bracket(self):
        """Test resource bracket pattern."""
        # Track if function was called
        called = False
        
        async def test_func(s3_client):
            nonlocal called
            called = True
            # Verify we got a client
            assert s3_client is not None
            return "result"
        
        # Mock aioboto3
        with patch('blackmarble.acquire.landsat_search.aioboto3.Session') as mock_session:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            
            mock_session.return_value.client.return_value = mock_client
            
            result = await with_s3_client(test_func, unsigned=True)
            
            assert called
            assert result == "result"
            # Verify cleanup was called
            mock_client.__aexit__.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_with_s3_client_session_caching(self):
        """Global aioboto3.Session object should be reused between calls."""
        # Ensure cache is clear before we start
        import importlib
        ls_mod = importlib.import_module("blackmarble.acquire.landsat_search")
        ls_mod._session_cache = None

        async def dummy(_):
            return "ok"

        with patch('blackmarble.acquire.landsat_search.aioboto3.Session') as mock_session:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_session.return_value.client.return_value = mock_client

            await with_s3_client(dummy)
            await with_s3_client(dummy)

            # aioboto3.Session() should be called exactly once due to cache
            assert mock_session.call_count == 1


class TestMainPipeline:
    """Test the main functional pipeline."""
    
    @pytest.mark.asyncio
    async def test_search_best_scenes_integration(self):
        """Test full pipeline with mocked dependencies."""
        # Clear global session cache
        import importlib
        ls_mod = importlib.import_module("blackmarble.acquire.landsat_search")
        ls_mod._session_cache = None
        
        bbox = (-73.0, 40.5, -72.5, 41.0)
        target_date = datetime(2023, 8, 10)
        config = LandsatAcquisitionConfig(window_days=5)
        
        # Mock find_scenes function
        async def mock_find_scenes(**kwargs):
            return [
                {
                    "scene_id": "LC08_L2SP_013032_20230809_20230812_02_T1",
                    "scene_prefix": "collection02/level-2/standard/oli-tirs/2023/013/032/",
                    "cloud_cover": 10.0,
                },
                {
                    "scene_id": "LC08_L2SP_013032_20230807_20230810_02_T1",
                    "scene_prefix": "collection02/level-2/standard/oli-tirs/2023/013/032/",
                    "cloud_cover": 25.0,
                },
            ]
        
        # Mock S3 client for MTL fetches
        with patch('blackmarble.acquire.landsat_search.aioboto3.Session') as mock_session:
            s3_mock = AsyncMock()
            s3_mock.__aenter__ = AsyncMock(return_value=s3_mock)
            s3_mock.__aexit__ = AsyncMock()
            
            # Mock MTL responses
            mtl_data = {
                "LANDSAT_METADATA_FILE": {
                    "IMAGE_ATTRIBUTES": {
                        "CLOUD_COVER": 10.0,
                        "DATA_PIXELS": 7650000,
                        "TOTAL_PIXELS": 7800000,
                        "SNOW_ICE_PIXELS": 0,
                    },
                    "PROJECTION_ATTRIBUTES": {
                        "NADIR_OFFNADIR": "NADIR",
                    },
                }
            }
            
            body_mock = AsyncMock()
            body_mock.__aenter__ = AsyncMock(return_value=body_mock)
            body_mock.__aexit__ = AsyncMock()
            body_mock.read = AsyncMock(return_value=json.dumps(mtl_data).encode())
            body_mock.close = AsyncMock()
            
            s3_mock.get_object = AsyncMock(return_value={"Body": body_mock})
            s3_mock.exceptions.NoSuchKey = Exception
            
            mock_session.return_value.client.return_value = s3_mock
            
            # Run search
            scenes = await search_best_scenes(
                bbox=bbox,
                target_date=target_date,
                config=config,
                find_scenes_func=mock_find_scenes,
            )
            
            # Verify results
            assert len(scenes) == 1  # Should select best of the two
            assert scenes[0].cloud_cover == 10.0  # Lower cloud cover
            assert scenes[0].path == 13
            assert scenes[0].row == 32
            assert 0.0 <= scenes[0].quality_score <= 1.0