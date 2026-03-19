#!/usr/bin/env python3
"""Integration test for the synchronous pipeline wrapper and WGS84 export."""

import asyncio
import os
from datetime import datetime

import pytest
import rasterio

from blackmarble.config import PipelineConfig
from blackmarble.pipeline import pipeline as black_marble_pipeline_sync


@pytest.mark.integration
@pytest.mark.skip(reason="Integration test - run manually with pytest -m integration")
def test_sync_pipeline_with_wgs84_export():
    """Test the synchronous pipeline wrapper with WGS84 export enabled."""
    # Small test area (part of Madrid)
    bbox = (-3.75, 40.4, -3.65, 40.45)
    date = datetime(2023, 6, 15)
    output_path = "test_madrid_sync.tif"

    # Create a basic config with WGS84 output enabled
    config = PipelineConfig(bbox=bbox, date=date, output_path=output_path)
    config.export.generate_wgs84 = True

    # Run the pipeline
    result = black_marble_pipeline_sync(
        bbox=bbox, date=date, output_path=output_path, config=config
    )

    # Verify main output exists
    assert os.path.exists(output_path)
    assert "output_path" in result
    assert result["output_path"] == output_path

    # Verify WGS84 output exists
    wgs84_path = output_path.replace(".tif", "_wgs84.tif")
    assert os.path.exists(wgs84_path), "WGS84 output file should be created"

    # Verify both are valid COGs
    # Check main output
    with rasterio.open(output_path) as src:
        assert src.profile.get("tiled"), "Main output should be tiled (COG)"
        assert hasattr(src, "overviews") and src.overviews(1), (
            "Main output should have overviews (COG)"
        )  # type: ignore[attr-defined]
        assert src.profile.get("compress"), "Main output should be compressed"

    # Check WGS84 output
    with rasterio.open(wgs84_path) as src:
        assert src.profile.get("tiled"), "WGS84 output should be tiled (COG)"
        assert hasattr(src, "overviews") and src.overviews(1), (
            "WGS84 output should have overviews (COG)"
        )  # type: ignore[attr-defined]
        assert src.profile.get("compress"), "WGS84 output should be compressed"
        assert src.crs and src.crs.to_epsg() == 4326, "WGS84 output should be in EPSG:4326"

    # Cleanup
    if os.path.exists(output_path):
        os.remove(output_path)
    if os.path.exists(wgs84_path):
        os.remove(wgs84_path)


@pytest.mark.integration
@pytest.mark.skip(reason="Integration test - run manually with pytest -m integration")
def test_sync_pipeline_in_notebook_context():
    """Test the sync pipeline wrapper simulating a notebook context with existing event loop."""

    # Create and set a running event loop (simulating notebook environment)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Small test area
    bbox = (-3.75, 40.4, -3.65, 40.45)
    date = datetime(2023, 6, 15)
    output_path = "test_notebook_sync.tif"

    try:
        # Run in the context of an existing loop
        async def run_test():
            # The sync wrapper should detect the running loop and use nest_asyncio
            result = black_marble_pipeline_sync(bbox=bbox, date=date, output_path=output_path)
            return result

        result = loop.run_until_complete(run_test())

        # Verify output exists
        assert os.path.exists(output_path)
        assert "output_path" in result

        # Cleanup
        if os.path.exists(output_path):
            os.remove(output_path)

    finally:
        loop.close()


if __name__ == "__main__":
    # Allow running directly for quick testing
    print("Running integration tests manually...")
    print("\nTest 1: Sync pipeline with WGS84 export")
    test_sync_pipeline_with_wgs84_export()
    print("✓ Test 1 passed")

    print("\nTest 2: Sync pipeline in notebook context")
    test_sync_pipeline_in_notebook_context()
    print("✓ Test 2 passed")

    print("\nAll tests passed!")
