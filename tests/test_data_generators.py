"""Test data generators for Black Marble processing tests.

This module provides functions to generate realistic synthetic data for testing
the Black Marble pipeline without requiring actual satellite data or I/O operations.
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from scipy import ndimage
from tests.qa_constants import (
    QA_CLEAR_LAND,
    QA_CLEAR_WATER,
    QA_HIGH_CLOUD,
    QA_AEROSOL_VALID,
    QA_RADSAT_NONE,
    VIIRS_QF_GOOD,
    VIIRS_QF_POOR,
)


def add_spatial_noise(
    data: np.ndarray, noise_level: float = 0.05, smoothing: float = 1.0
) -> np.ndarray:
    """Add spatially correlated noise to data.

    Args:
        data: Base data array
        noise_level: Standard deviation of noise relative to data range
        smoothing: Gaussian smoothing sigma for spatial correlation

    Returns:
        Data with added spatially correlated noise
    """
    noise = np.random.randn(*data.shape) * noise_level
    if smoothing > 0:
        noise = ndimage.gaussian_filter(noise, sigma=smoothing)
    return data + noise


def create_cloud_pattern(
    shape: Tuple[int, int], coverage: float = 0.2, cluster_size: int = 20
) -> np.ndarray:
    """Create realistic cloud patterns.

    Args:
        shape: Output array shape
        coverage: Fraction of pixels covered by clouds
        cluster_size: Average size of cloud clusters

    Returns:
        Binary cloud mask
    """
    # Handle full coverage case
    if coverage >= 1.0:
        return np.ones(shape, dtype=bool)

    if coverage <= 0.0:
        return np.zeros(shape, dtype=bool)

    # Create random seed points for clouds
    n_seeds = max(1, int(shape[0] * shape[1] * coverage / (cluster_size**2)))
    seeds = np.zeros(shape, dtype=bool)

    y_coords = np.random.randint(0, shape[0], n_seeds)
    x_coords = np.random.randint(0, shape[1], n_seeds)
    seeds[y_coords, x_coords] = True

    # Grow clouds from seeds
    structure = np.ones((3, 3))
    clouds = ndimage.binary_dilation(seeds, structure, iterations=cluster_size // 2)

    # Add some edge softness
    clouds = ndimage.gaussian_filter(clouds.astype(float), sigma=2) > 0.3

    return clouds


def create_urban_pattern(
    shape: Tuple[int, int],
    center: Optional[Tuple[float, float]] = None,
    radius: float = 0.3,
    density: float = 0.8,
) -> np.ndarray:
    """Create urban density pattern with center and falloff.

    Args:
        shape: Output array shape
        center: Urban center location (relative 0-1), defaults to center
        radius: Urban area radius (relative to image size)
        density: Peak urban density (0-1)

    Returns:
        Urban density map (0-1)
    """
    if center is None:
        center = (0.5, 0.5)

    y, x = np.ogrid[0 : shape[0], 0 : shape[1]]
    y = y / shape[0]
    x = x / shape[1]

    # Distance from center
    dist = np.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)

    # Urban density with smooth falloff
    urban = density * np.exp(-((dist / radius) ** 2))

    # Add some radial roads
    angle = np.arctan2(y - center[0], x - center[1])
    roads = np.sin(angle * 8) ** 2 * 0.3
    urban = np.clip(urban + roads * urban, 0, 1)

    # Add noise for realism
    urban = add_spatial_noise(urban, noise_level=0.1, smoothing=2)

    return np.clip(urban, 0, 1)


def create_test_landsat_data(
    shape: Tuple[int, int] = (100, 100),
    scenario: str = "mixed",
    cloud_coverage: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Generate realistic Landsat test data for different scenarios.

    Args:
        shape: Output array shape
        scenario: One of 'urban', 'forest', 'water', 'desert', 'mixed'
        cloud_coverage: Fraction of pixels covered by clouds

    Returns:
        Dictionary with band data and QA arrays
    """
    scenarios = {
        "urban": {
            "red_mean": 0.15,
            "red_std": 0.05,
            "nir_mean": 0.10,
            "nir_std": 0.05,
            "green_mean": 0.12,
            "green_std": 0.04,
        },
        "forest": {
            "red_mean": 0.05,
            "red_std": 0.02,
            "nir_mean": 0.40,
            "nir_std": 0.10,
            "green_mean": 0.08,
            "green_std": 0.03,
        },
        "water": {
            "red_mean": 0.10,
            "red_std": 0.03,
            "nir_mean": 0.02,
            "nir_std": 0.01,
            "green_mean": 0.08,
            "green_std": 0.02,
        },
        "desert": {
            "red_mean": 0.25,
            "red_std": 0.05,
            "nir_mean": 0.30,
            "nir_std": 0.05,
            "green_mean": 0.22,
            "green_std": 0.04,
        },
    }

    if scenario == "mixed":
        # Create mixed landscape with different regions
        urban_map = create_urban_pattern(shape, radius=0.2)
        water_map = np.zeros(shape)
        water_map[shape[0] // 2 :, : shape[1] // 3] = 1  # Water in lower left

        forest_map = 1 - urban_map - water_map
        forest_map = np.clip(forest_map, 0, 1)

        # Weighted combination of scenarios
        red = (
            urban_map * np.random.normal(0.15, 0.05, shape)
            + water_map * np.random.normal(0.10, 0.03, shape)
            + forest_map * np.random.normal(0.05, 0.02, shape)
        )
        nir = (
            urban_map * np.random.normal(0.10, 0.05, shape)
            + water_map * np.random.normal(0.02, 0.01, shape)
            + forest_map * np.random.normal(0.40, 0.10, shape)
        )
        green = (
            urban_map * np.random.normal(0.12, 0.04, shape)
            + water_map * np.random.normal(0.08, 0.02, shape)
            + forest_map * np.random.normal(0.08, 0.03, shape)
        )
    else:
        params = scenarios.get(scenario, scenarios["urban"])  # Default to urban
        red = np.random.normal(params["red_mean"], params["red_std"], shape)
        nir = np.random.normal(params["nir_mean"], params["nir_std"], shape)
        green = np.random.normal(params["green_mean"], params["green_std"], shape)

    # Add spatial correlation
    red = add_spatial_noise(red, noise_level=0.02, smoothing=1.5)
    nir = add_spatial_noise(nir, noise_level=0.02, smoothing=1.5)
    green = add_spatial_noise(green, noise_level=0.02, smoothing=1.5)

    # Clip to valid reflectance range
    red = np.clip(red, 0, 1)
    nir = np.clip(nir, 0, 1)
    green = np.clip(green, 0, 1)

    # Create QA_PIXEL array
    qa_pixel = np.full(shape, QA_CLEAR_LAND, dtype=np.uint16)  # Clear land

    # Add water pixels
    if scenario in ["water", "mixed"]:
        if scenario == "water":
            water_mask = np.ones(shape, dtype=bool)
        else:
            water_mask = water_map > 0.5
        qa_pixel[water_mask] = QA_CLEAR_WATER  # Clear water

    # Add clouds if requested
    if cloud_coverage > 0:
        cloud_mask = create_cloud_pattern(shape, cloud_coverage)
        qa_pixel[cloud_mask] = QA_HIGH_CLOUD  # High confidence cloud

        # Set reflectance to high values under clouds
        red[cloud_mask] = 0.8 + np.random.normal(0, 0.1, np.sum(cloud_mask))
        nir[cloud_mask] = 0.8 + np.random.normal(0, 0.1, np.sum(cloud_mask))
        green[cloud_mask] = 0.8 + np.random.normal(0, 0.1, np.sum(cloud_mask))

    # Create QA_RADSAT (no saturation by default)
    qa_radsat = np.full(shape, QA_RADSAT_NONE, dtype=np.uint16)

    # Create QA_AEROSOL (low aerosol by default)
    qa_aerosol = np.full(shape, QA_AEROSOL_VALID, dtype=np.uint8)

    return {
        "B3": green.astype(np.float32),
        "B4": red.astype(np.float32),
        "B5": nir.astype(np.float32),
        "QA_PIXEL": qa_pixel,
        "QA_RADSAT": qa_radsat,
        "QA_AEROSOL": qa_aerosol,
    }


def create_test_viirs_data(
    shape: Tuple[int, int] = (100, 100),
    scenario: str = "mixed",
    lunar_illumination: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Generate realistic VIIRS nighttime lights test data.

    Args:
        shape: Output array shape
        scenario: One of 'urban', 'rural', 'dark', 'mixed'
        lunar_illumination: Lunar illumination level (0-1)

    Returns:
        Dictionary with NTL, lunar, and quality arrays
    """
    if scenario == "urban":
        # High nighttime lights
        ntl = np.random.normal(50, 20, shape)
        ntl = add_spatial_noise(ntl, noise_level=5, smoothing=2)
    elif scenario == "rural":
        # Low nighttime lights
        ntl = np.random.normal(5, 2, shape)
        ntl = add_spatial_noise(ntl, noise_level=1, smoothing=2)
    elif scenario == "dark":
        # No lights
        ntl = np.random.normal(0.1, 0.1, shape)
    else:  # mixed
        urban_map = create_urban_pattern(shape)
        ntl = urban_map * np.random.normal(50, 20, shape) + (
            1 - urban_map
        ) * np.random.normal(2, 1, shape)
        ntl = add_spatial_noise(ntl, noise_level=2, smoothing=2)

    # Ensure non-negative
    ntl = np.clip(ntl, 0, None).astype(np.float32)

    # Lunar irradiance
    lunar = np.full(shape, lunar_illumination * 10, dtype=np.float32)
    lunar = add_spatial_noise(lunar, noise_level=0.5, smoothing=3)
    lunar = np.clip(lunar, 0, None)

    # Quality flags (0 = good quality)
    quality = np.full(shape, VIIRS_QF_GOOD, dtype=np.uint8)

    # Add some poor quality pixels
    poor_quality_mask = np.random.random(shape) < 0.05
    quality[poor_quality_mask] = VIIRS_QF_POOR  # Poor quality

    return {"NTL": ntl, "LUNAR": lunar, "QF": quality}


def create_test_road_data(
    shape: Tuple[int, int] = (100, 100), scenario: str = "mixed", density: float = 0.1
) -> np.ndarray:
    """Generate realistic road network test data.

    Args:
        shape: Output array shape
        scenario: One of 'urban', 'rural', 'highway', 'mixed'
        density: Road density (fraction of pixels with roads)

    Returns:
        Road classification array (0=no road, 1-5=road types)
    """
    roads = np.zeros(shape, dtype=np.uint8)

    if scenario == "urban":
        # Dense urban grid
        grid_spacing = max(5, int(shape[0] * (1 - density) / 10))
        # Vertical streets
        roads[::grid_spacing, :] = 3  # Secondary roads
        # Horizontal streets
        roads[:, ::grid_spacing] = 3
        # Main roads
        roads[shape[0] // 2, :] = 1  # Major road
        roads[:, shape[1] // 2] = 1

    elif scenario == "rural":
        # Sparse rural roads
        n_roads = int(density * shape[0] * shape[1] / 100)
        for _ in range(n_roads):
            if np.random.random() < 0.5:
                # Horizontal road
                y = np.random.randint(0, shape[0])
                roads[y, :] = np.random.choice([2, 3, 5])
            else:
                # Vertical road
                x = np.random.randint(0, shape[1])
                roads[:, x] = np.random.choice([2, 3, 5])

    elif scenario == "highway":
        # Highway corridor
        center = shape[0] // 2
        width = 3
        roads[center - width : center + width, :] = 1  # Motorway
        # Some exits
        for x in range(0, shape[1], 20):
            roads[:, x] = 2

    else:  # mixed
        # Combination of patterns
        urban_center = create_urban_pattern(shape, radius=0.2)

        # Major roads radiating from center
        center_y, center_x = shape[0] // 2, shape[1] // 2
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)

        for angle in angles:
            for r in range(max(shape)):
                y = int(center_y + r * np.sin(angle))
                x = int(center_x + r * np.cos(angle))
                if 0 <= y < shape[0] and 0 <= x < shape[1]:
                    roads[y, x] = 1 if r < shape[0] // 3 else 2

        # Urban grid in center
        mask = urban_center > 0.5
        y_coords, x_coords = np.where(mask)
        if len(y_coords) > 0:
            # Add grid roads in urban area
            urban_roads = np.random.random(np.sum(mask)) < density
            roads[y_coords[urban_roads], x_coords[urban_roads]] = 3

    return roads


def create_test_scene(
    shape: Tuple[int, int] = (256, 256),
    include_clouds: bool = True,
    include_water: bool = True,
    include_urban: bool = True,
    include_forest: bool = True,
) -> Dict[str, np.ndarray]:
    """Create a complete test scene with all data types.

    Args:
        shape: Output array shape
        include_clouds: Whether to include cloud cover
        include_water: Whether to include water bodies
        include_urban: Whether to include urban areas
        include_forest: Whether to include forested areas

    Returns:
        Dictionary with all band data, QA, VIIRS, and road arrays
    """
    # Create base land cover map
    land_cover = np.zeros(shape)

    if include_water:
        # Add a river or lake
        water_mask = np.zeros(shape, dtype=bool)
        # Simple river
        river_width = shape[0] // 20
        river_y = shape[0] // 3
        water_mask[river_y : river_y + river_width, :] = True
        # Add lake
        lake_center = (shape[0] * 2 // 3, shape[1] * 2 // 3)
        y, x = np.ogrid[0 : shape[0], 0 : shape[1]]
        lake_mask = (y - lake_center[0]) ** 2 + (x - lake_center[1]) ** 2 < (
            shape[0] // 6
        ) ** 2
        water_mask |= lake_mask
        land_cover[water_mask] = 1  # Water

    if include_urban:
        urban_mask = create_urban_pattern(shape, center=(0.3, 0.3), radius=0.15) > 0.5
        land_cover[urban_mask & (land_cover == 0)] = 2  # Urban

    if include_forest:
        # Everything else is forest
        land_cover[land_cover == 0] = 3  # Forest

    # Generate Landsat data based on land cover
    red = np.zeros(shape, dtype=np.float32)
    nir = np.zeros(shape, dtype=np.float32)
    green = np.zeros(shape, dtype=np.float32)

    # Water
    water_pixels = land_cover == 1
    red[water_pixels] = np.random.normal(0.10, 0.03, np.sum(water_pixels))
    nir[water_pixels] = np.random.normal(0.02, 0.01, np.sum(water_pixels))
    green[water_pixels] = np.random.normal(0.08, 0.02, np.sum(water_pixels))

    # Urban
    urban_pixels = land_cover == 2
    red[urban_pixels] = np.random.normal(0.15, 0.05, np.sum(urban_pixels))
    nir[urban_pixels] = np.random.normal(0.10, 0.05, np.sum(urban_pixels))
    green[urban_pixels] = np.random.normal(0.12, 0.04, np.sum(urban_pixels))

    # Forest
    forest_pixels = land_cover == 3
    red[forest_pixels] = np.random.normal(0.05, 0.02, np.sum(forest_pixels))
    nir[forest_pixels] = np.random.normal(0.40, 0.10, np.sum(forest_pixels))
    green[forest_pixels] = np.random.normal(0.08, 0.03, np.sum(forest_pixels))

    # Add spatial smoothing
    red = ndimage.gaussian_filter(red, sigma=1)
    nir = ndimage.gaussian_filter(nir, sigma=1)
    green = ndimage.gaussian_filter(green, sigma=1)

    # Clip values
    red = np.clip(red, 0, 1)
    nir = np.clip(nir, 0, 1)
    green = np.clip(green, 0, 1)

    # QA_PIXEL
    qa_pixel = np.full(shape, QA_CLEAR_LAND, dtype=np.uint16)  # Clear land
    qa_pixel[water_pixels] = 21952  # Clear water

    if include_clouds:
        cloud_mask = create_cloud_pattern(shape, coverage=0.15)
        qa_pixel[cloud_mask] = QA_HIGH_CLOUD  # High confidence cloud
        # Brighten cloudy pixels
        red[cloud_mask] = 0.8
        nir[cloud_mask] = 0.8
        green[cloud_mask] = 0.8

    # VIIRS data - bright in urban areas
    ntl = np.zeros(shape, dtype=np.float32)
    ntl[urban_pixels] = np.random.normal(
        150, 50, np.sum(urban_pixels)
    )  # Brighter urban lights
    ntl[forest_pixels] = np.random.normal(0.5, 0.5, np.sum(forest_pixels))
    ntl = ndimage.gaussian_filter(ntl, sigma=3)
    ntl = np.clip(ntl, 0, None)

    # Roads - concentrated in urban areas
    roads = np.zeros(shape, dtype=np.uint8)
    if include_urban:
        # Major roads connecting urban areas
        roads[shape[0] // 2, :] = 1
        roads[:, shape[1] // 2] = 1
        # Urban roads
        urban_road_mask = urban_pixels & (np.random.random(shape) < 0.2)
        roads[urban_road_mask] = np.random.choice([2, 3], size=np.sum(urban_road_mask))

    return {
        "B3": green,
        "B4": red,
        "B5": nir,
        "QA_PIXEL": qa_pixel,
        "QA_RADSAT": np.full(shape, QA_RADSAT_NONE, dtype=np.uint16),
        "QA_AEROSOL": np.full(shape, QA_AEROSOL_VALID, dtype=np.uint8),
        "NTL": ntl,
        "LUNAR": np.full(shape, 5.0, dtype=np.float32),
        "QF": np.full(shape, VIIRS_QF_GOOD, dtype=np.uint8),
        "roads": roads,
        "land_cover": land_cover,  # For validation
    }
