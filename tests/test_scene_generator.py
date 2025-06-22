"""Extended test scene generator for comprehensive pipeline tests."""

import numpy as np
from typing import Dict, Tuple

from .test_data_generators import (
    create_test_scene as create_basic_scene,
    create_urban_lights_pattern,
    create_cloud_pattern,
    create_water_mask,
)


def create_test_scene(
    shape: Tuple[int, int] = (512, 512),
    include_clouds: bool = True,
    include_water: bool = True,
    include_urban: bool = True,
    include_forest: bool = True,
    cloud_coverage: float = 0.2,
    water_fraction: float = 0.1,
) -> Dict[str, np.ndarray]:
    """
    Create an extended test scene with land cover and spectral characteristics.

    This maintains the rigorous testing requirements by creating realistic
    spectral signatures for different land cover types.
    """
    # Start with basic scene
    scene = create_basic_scene(
        shape=shape,
        include_clouds=include_clouds,
        include_water=include_water,
        cloud_coverage=cloud_coverage,
        water_fraction=water_fraction,
    )

    # Create land cover classification
    # 1 = water, 2 = urban, 3 = agriculture, 4 = forest
    land_cover = np.ones(shape, dtype=np.uint8)  # Default to water

    # Add water from water mask (land_cover = 1)
    if include_water:
        water_mask = create_water_mask(shape, water_fraction=water_fraction)
        land_cover[water_mask] = 1
    else:
        land_cover[:] = 3  # Default to agriculture if no water

    # Add urban areas (land_cover = 2)
    if include_urban:
        urban_lights = create_urban_lights_pattern(shape, num_cities=3)
        urban_mask = urban_lights > 0.3
        land_cover[urban_mask] = 2

    # Add forest areas (land_cover = 4)
    if include_forest:
        # Create forest patches
        forest_mask = create_cloud_pattern(shape, coverage=0.25, num_clusters=6)
        # Don't overwrite water or urban
        available_mask = (land_cover != 1) & (land_cover != 2)
        land_cover[forest_mask & available_mask] = 4

    # Remaining areas are agriculture (land_cover = 3)
    available_mask = (land_cover != 1) & (land_cover != 2) & (land_cover != 4)
    land_cover[available_mask] = 3

    scene["land_cover"] = land_cover

    # Add missing data that process_scene expects
    if "LUNAR" not in scene:
        # Create synthetic lunar irradiance (0.0 to 1.0)
        lunar = np.full(shape, 0.1, dtype=np.float32)
        lunar += np.random.rand(*shape) * 0.05
        scene["LUNAR"] = lunar

    if "QF" not in scene:
        # Create quality flag (0=good, 1=poor, 2=no retrieval)
        qf = np.zeros(shape, dtype=np.uint8)
        if include_clouds:
            cloud_mask = create_cloud_pattern(shape, coverage=cloud_coverage * 0.3)
            qf[cloud_mask] = 1
        scene["QF"] = qf

    # Set realistic spectral signatures for each land cover type
    # Water (land_cover = 1): Low reflectance, high NDWI
    water_mask = land_cover == 1
    if np.any(water_mask):
        scene["B3"][water_mask] = 8500  # Higher green (for positive NDWI)
        scene["B4"][water_mask] = 7500  # Low red
        scene["B5"][water_mask] = 7800  # Lower NIR (for positive NDWI)

    # Urban (land_cover = 2): Low NDVI, negative NDWI, bright NTL
    urban_mask = land_cover == 2
    if np.any(urban_mask):
        scene["B3"][urban_mask] = 8000  # Lower green (for negative NDWI)
        scene["B4"][urban_mask] = 12000  # Higher red (for low NDVI)
        scene["B5"][urban_mask] = 10000  # Moderate NIR
        # Ensure bright NTL in urban areas
        if "NTL" in scene:
            scene["NTL"][urban_mask] = np.maximum(scene["NTL"][urban_mask], 50.0)

    # Agriculture (land_cover = 3): Moderate NDVI and NDWI
    ag_mask = land_cover == 3
    if np.any(ag_mask):
        scene["B3"][ag_mask] = 9000  # Moderate green
        scene["B4"][ag_mask] = 10000  # Moderate red
        scene["B5"][ag_mask] = 12000  # Moderate NIR

    # Forest (land_cover = 4): High NDVI, negative NDWI
    forest_mask = land_cover == 4
    if np.any(forest_mask):
        scene["B3"][forest_mask] = 8500  # Lower green (for negative NDWI)
        scene["B4"][forest_mask] = 8000  # Lower red (for high NDVI)
        scene["B5"][forest_mask] = 20000  # Much higher NIR (for high NDVI)

    # Ensure roads are present in urban areas
    if "roads" in scene and np.any(urban_mask):
        # Add roads in urban areas
        urban_indices = np.where(urban_mask)
        if len(urban_indices[0]) > 0:
            num_road_pixels = min(len(urban_indices[0]) // 10, 100)
            road_indices = np.random.choice(
                len(urban_indices[0]), num_road_pixels, replace=False
            )
            scene["roads"][
                urban_indices[0][road_indices], urban_indices[1][road_indices]
            ] = 1.0

    # Dim NTL in non-urban areas to enhance urban contrast
    if "NTL" in scene:
        non_urban_mask = land_cover != 2
        if np.any(non_urban_mask):
            scene["NTL"][non_urban_mask] = scene["NTL"][non_urban_mask] * 0.1

    return scene
