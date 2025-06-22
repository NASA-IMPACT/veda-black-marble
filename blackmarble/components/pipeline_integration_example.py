"""Example of integrating output components into the Black Marble pipeline.

This shows how to refactor the pipeline.py to use the new modular components
for output generation and enhancement.
"""

from pathlib import Path
import numpy as np
import os
from typing import Any

from blackmarble.components import (
    COGGenerator,
    MetadataGenerator,
    VisualizationEnhancer,
    OutputValidator,
    OutputMetadata,
    COGProfile,
    RoadEnhancer,
    ContrastEnhancer,
    ColorMapper,
)
from blackmarble.config import ProcessingConfig
from blackmarble.io import scale_image


def generate_black_marble_outputs(
    ndui: np.ndarray,
    roads: np.ndarray,
    georef: tuple[Any, Any],  # (geotransform, projection)
    config: ProcessingConfig,
    output_dir: Path,
    processing_metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Generate all Black Marble output products using components.

    This function replaces the output generation section of the pipeline
    with modular, testable components.

    Args:
        ndui: Computed NDUI array
        roads: OSM road classification array
        georef: Tuple of (geotransform, projection)
        config: Processing configuration
        output_dir: Output directory path
        processing_metadata: Optional metadata about processing

    Returns:
        Dictionary of output paths
    """
    geotransform, projection = georef
    height, width = ndui.shape

    # Create output metadata
    metadata = OutputMetadata(
        crs=projection,
        transform=geotransform,
        bounds=_get_bounds_from_transform(geotransform, width, height),
        shape=(height, width),
        dtype=np.dtype(np.float32),
        nodata=np.nan,
        tags=processing_metadata,
    )

    # 1. Apply road enhancement
    road_enhancer = RoadEnhancer(
        major_factor=config.ndui.road_enhancement_major,
        minor_factor=config.ndui.road_enhancement_minor,
    )
    enhanced_result = road_enhancer.enhance(ndui, roads)

    print(
        f"Road enhancement applied: {enhanced_result.statistics['road_pixels']} pixels enhanced"
    )

    # 2. Validate the enhanced data
    validator = OutputValidator(min_valid_percentage=config.min_valid_pixel_percentage)
    validation = None

    try:
        validation = validator.validate(
            enhanced_result.data, metadata, expected_range=(0.0, 1.0)
        )
        print(
            f"Validation passed: {validation['statistics']['valid_percentage']:.1f}% valid pixels"
        )
    except ValueError as e:
        print(f"Warning: Validation failed - {e}")
        # Could decide to proceed anyway or return early

    # 3. Generate data COG
    cog_generator = COGGenerator(
        profile=COGProfile(profile_name="deflate", blocksize=512, web_optimized=True)
    )

    data_cog_path = output_dir / "ndui_enhanced.tif"
    cog_generator.generate(enhanced_result.data, metadata, data_cog_path)
    print(f"Generated data COG: {data_cog_path}")

    # 4. Create visualization
    # Load colormap arrays (in real pipeline, these would be loaded once)
    colormap_arrays = _load_inferno_colormap()

    # Apply contrast enhancement for visualization
    contrast_enhancer = ContrastEnhancer(method="percentile", clip_percentiles=(2, 98))
    contrast_result = contrast_enhancer.enhance(
        enhanced_result.data, output_range=(config.display_min, config.display_max)
    )

    # Create RGB visualization
    viz_enhancer = VisualizationEnhancer(colormap_arrays=colormap_arrays)
    rgb_data = viz_enhancer.apply_colormap(
        contrast_result.data,
        display_min=config.display_min,
        display_max=config.display_max,
    )

    # 5. Generate RGB COG
    rgb_metadata = OutputMetadata(
        crs=projection,
        transform=geotransform,
        bounds=metadata.bounds,
        shape=(height, width),
        dtype=np.dtype(np.uint8),
        nodata=None,  # No nodata for RGB
        tags={**(processing_metadata or {}), "visualization": "inferno_colormap"},
    )

    rgb_cog_path = output_dir / "ndui_rgb.tif"
    cog_generator.generate(rgb_data, rgb_metadata, rgb_cog_path)
    print(f"Generated RGB COG: {rgb_cog_path}")

    # 6. Generate metadata
    metadata_generator = MetadataGenerator(metadata_format="stac")

    stac_item = metadata_generator.generate(
        enhanced_result.data,
        metadata,
        processing_info={
            "pipeline_version": "2.0",
            "road_enhancement": enhanced_result.metadata,
            "contrast_enhancement": contrast_result.metadata,
            "validation": validation["statistics"] if validation else None,
        },
        title="Black Marble NDUI Product",
        description="Normalized Difference Urban Index with road network enhancement",
        datetime=processing_metadata.get("acquisition_date") if processing_metadata else None,
        id=f"black-marble-ndui-{processing_metadata.get('acquisition_date', 'unknown') if processing_metadata else 'unknown'}",
        assets={
            "visual": {
                "href": str(rgb_cog_path.name),
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                "roles": ["visual"],
                "title": "RGB visualization",
            },
            "data": {
                "href": str(data_cog_path.name),
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                "roles": ["data"],
                "title": "NDUI data",
            },
        },
    )

    # Save STAC metadata
    import json

    stac_path = output_dir / "ndui_stac.json"
    with open(stac_path, "w") as f:
        json.dump(stac_item, f, indent=2)
    print(f"Generated STAC metadata: {stac_path}")

    return {
        "data_cog": data_cog_path,
        "rgb_cog": rgb_cog_path,
        "stac_metadata": stac_path,
    }


def refactored_pipeline_output_section(
    ndui: np.ndarray,
    roads: np.ndarray,
    geotransform: tuple[float, ...],
    projection: str,
    config: ProcessingConfig,
    final_dir: str,
) -> str:
    """Refactored version of the pipeline output section.

    This shows how to replace the existing output logic in pipeline.py
    with the new component-based approach.

    Args:
        ndui: Computed NDUI array
        roads: OSM road classification array
        geotransform: GDAL geotransform
        projection: Projection WKT string
        config: Processing configuration
        final_dir: Final output directory

    Returns:
        Path to final COG file
    """
    # The old code did this inline - now we use components
    output_dir = Path(final_dir)

    # Prepare processing metadata
    processing_metadata = {
        "product": "black_marble",
        "version": "2.0",
        "processing_date": "2024-01-15",  # Would be datetime.now() in real code
    }

    # Generate all outputs using components
    outputs = generate_black_marble_outputs(
        ndui=ndui,
        roads=roads,
        georef=(geotransform, projection),
        config=config,
        output_dir=output_dir,
        processing_metadata=processing_metadata,
    )

    # Return the RGB COG path (matching original pipeline behavior)
    return str(outputs["rgb_cog"])


def _get_bounds_from_transform(
    transform: tuple[float, ...], width: int, height: int
) -> tuple[float, float, float, float]:
    """Calculate bounds from GDAL transform."""
    # GDAL transform: (x_origin, pixel_width, 0, y_origin, 0, -pixel_height)
    minx = transform[0]
    maxx = transform[0] + width * transform[1]
    maxy = transform[3]
    miny = transform[3] + height * transform[5]
    return (minx, miny, maxx, maxy)


def _load_inferno_colormap() -> dict[str, np.ndarray]:
    """Load inferno colormap arrays.

    In the real pipeline, this loads from the bundled .npy files.
    """
    # This would load from blackmarble/data/inferno_*.npy
    # For example purposes, create a simple colormap
    r = np.linspace(0, 255, 256).astype(np.uint8)
    g = np.linspace(0, 200, 256).astype(np.uint8)
    b = np.linspace(100, 50, 256).astype(np.uint8)
    r[0] = g[0] = b[0] = 0  # Black for zero

    return {"r": r, "g": g, "b": b}


# Example of how to update the main pipeline.py:
"""
# In pipeline.py, replace lines ~1194-1237 (colorization and COG creation) with:

from blackmarble.components.pipeline_integration_example import refactored_pipeline_output_section

# ... earlier pipeline code ...

# After NDUI calculation (line ~1155)
out_cog = refactored_pipeline_output_section(
    ndui=ndui,
    roads=img_road,
    geotransform=gt,
    projection=proj,
    config=config,
    final_dir=final_dir,
)

# Continue with upload section...
"""
