#!/usr/bin/env python3
"""Quick test of Landsat download to diagnose issues."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_download():
    """Test downloading a single Landsat scene."""
    from blackmarble.acquire.landsat import download_landsat
    
    # Small test area
    bbox = (-110.01, 39.99, -109.99, 40.01)
    dates = [datetime(2023, 1, 24)]
    bands = ["B3", "B4", "B5", "QA_PIXEL"]
    output_dir = "./test_landsat_output"
    
    print(f"Testing Landsat download for {dates[0]}")
    print(f"BBox: {bbox}")
    print(f"Bands: {bands}")
    
    try:
        results = await download_landsat(
            dates=dates,
            bbox=bbox,
            bands=bands,
            output_dir=output_dir,
            cloud_cover_max=50.0,
            show_progress=True
        )
        
        print("\nResults:")
        for date, bands_data in results.items():
            print(f"\nDate: {date}")
            for band, paths in bands_data.items():
                print(f"  {band}: {paths}")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_download())