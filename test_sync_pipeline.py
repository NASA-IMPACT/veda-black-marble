#!/usr/bin/env python3
"""Test the synchronous pipeline wrapper."""

from datetime import datetime
from blackmarble.pipeline import black_marble_pipeline_sync
from blackmarble.config import PipelineConfig

# Small test area (part of Madrid)
bbox = (-3.75, 40.4, -3.65, 40.45)
date = datetime(2023, 6, 15)
output_path = "test_madrid_sync.tif"

# Create a basic config with WGS84 output enabled
config = PipelineConfig(
    bbox=bbox,
    date=date,
    output_path=output_path
)
config.export.generate_wgs84 = True

print(f"Testing sync pipeline for bbox: {bbox}")
print(f"Date: {date}")
print(f"Output: {output_path}")

try:
    result = black_marble_pipeline_sync(
        bbox=bbox,
        date=date,
        output_path=output_path,
        config=config
    )
    
    print("\nPipeline completed successfully!")
    print(f"Output saved to: {result['output_path']}")
    print(f"Statistics: {result['statistics']}")
    
except Exception as e:
    print(f"\nError running pipeline: {e}")
    import traceback
    traceback.print_exc()