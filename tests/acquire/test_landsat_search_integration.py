"""Integration tests for functional Landsat acquisition with real S3."""

import pytest
from datetime import datetime

from blackmarble.acquire.landsat_search import (
    search_best_scenes,
    SceneMeta,
)
from blackmarble.config import LandsatAcquisitionConfig


@pytest.mark.integration
@pytest.mark.slow
class TestFunctionalIntegration:
    """Integration tests that hit real S3 (no mocks)."""
    
    @pytest.mark.asyncio
    async def test_search_real_s3_tiny_area(self):
        """Test against real S3 with a tiny area and short time window.
        
        This test uses unsigned S3 access to the public USGS Landsat bucket.
        We use a very small area and time window to minimize data transfer.
        """
        # Tiny area in NYC (should be 1 path/row)
        bbox = (-73.99, 40.75, -73.98, 40.76)
        
        # Recent date with small window
        target_date = datetime(2023, 8, 15)
        config = LandsatAcquisitionConfig(
            window_days=3,  # Very small window
            max_cloud_threshold=90.0,  # Accept cloudy scenes for test
        )
        
        try:
            # This actually hits S3!
            scenes = await search_best_scenes(
                bbox=bbox,
                target_date=target_date,
                config=config,
                allow_incomplete=True,  # OK if no scenes found
            )
            
            # Verify we got valid SceneMeta objects
            assert isinstance(scenes, list)
            for scene in scenes:
                assert isinstance(scene, SceneMeta)
                assert scene.scene_id.startswith(("LC08", "LC09", "LE07"))
                assert 0 <= scene.cloud_cover <= 100
                assert 0 <= scene.quality_score <= 1
                assert scene.path > 0
                assert scene.row > 0
                
        except Exception as e:
            # Network issues are OK in CI
            if "Network" in str(e) or "Connection" in str(e):
                pytest.skip(f"Network error in CI: {e}")
            raise
    
    @pytest.mark.asyncio
    async def test_quality_scoring_real_data(self):
        """Test quality scoring with real scene metadata."""
        # Use a known good area/date
        bbox = (-122.5, 37.7, -122.4, 37.8)  # San Francisco
        target_date = datetime(2023, 7, 1)  # Summer (less clouds)
        
        config = LandsatAcquisitionConfig(
            window_days=15,
            quality_weights={
                "spatial_coverage": 0.5,
                "cloud_free": 0.3,
                "snow_free": 0.15,
                "off_nadir": 0.05,
            }
        )
        
        try:
            scenes = await search_best_scenes(
                bbox=bbox,
                target_date=target_date,
                config=config,
                allow_incomplete=True,
            )
            
            if scenes:
                # Verify quality scores make sense
                for scene in scenes:
                    # Summer in SF should have low snow
                    assert scene.snow_cover < 0.1
                    
                    # Quality score should reflect the weights
                    if scene.cloud_cover < 20:
                        assert scene.quality_score > 0.7  # Good scene
                    
        except Exception as e:
            if "Network" in str(e) or "Connection" in str(e):
                pytest.skip(f"Network error in CI: {e}")
            raise