"""Pure data processing functions for Black Marble pipeline.

This module contains the core processing logic extracted from the pipeline,
implemented as pure functions that work with numpy arrays. This separation
enables better testing, profiling, and parallelization.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings

from .calculations import (
    calculate_ndvi,
    calculate_ndwi,
    calculate_ndui,
    apply_lunar_correction,
    scale_landsat_to_reflectance,
)
from .config import ProcessingConfig, NDUIConfig, VIIRSConfig, LandsatConfig


def extract_cloud_mask(
    qa_pixel: np.ndarray, strategy: str = "conservative"
) -> np.ndarray:
    """Extract cloud mask from Landsat QA_PIXEL band.

    Args:
        qa_pixel: QA_PIXEL array with bit-packed quality flags
        strategy: 'conservative' (more clouds) or 'permissive' (fewer clouds)

    Returns:
        Boolean mask where True indicates clouds
    """
    if strategy == "conservative":
        # Flag as cloud if any cloud confidence > low
        cloud_values = [
            22280,  # Low confidence cloud
            54596,  # Medium confidence cloud
            54852,  # High confidence cloud
            55052,  # High confidence cloud + shadow
            # Add more conservative values
            22344,  # Low confidence cloud + shadow
            54660,  # Medium confidence cloud + shadow
        ]
    else:  # permissive
        # Only flag high confidence clouds
        cloud_values = [
            54852,  # High confidence cloud only
            55052,  # High confidence cloud + shadow
        ]

    return np.isin(qa_pixel, cloud_values)


def extract_water_mask(qa_pixel: np.ndarray) -> np.ndarray:
    """Extract water mask from Landsat QA_PIXEL band.

    Args:
        qa_pixel: QA_PIXEL array

    Returns:
        Boolean mask where True indicates water
    """
    # Bit 3 indicates water
    water_bit = 3
    return ((qa_pixel >> water_bit) & 1).astype(bool)


def process_landsat_bands(
    bands: Dict[str, np.ndarray], config: Optional[LandsatConfig] = None
) -> Dict[str, np.ndarray]:
    """Process Landsat bands to calculate indices and masks.

    Args:
        bands: Dictionary with B3, B4, B5, QA_PIXEL arrays
        config: Landsat configuration

    Returns:
        Dictionary with processed results
    """
    if config is None:
        config = LandsatConfig()

    # Scale to reflectance if needed
    green = scale_landsat_to_reflectance(bands["B3"], config)
    red = scale_landsat_to_reflectance(bands["B4"], config)
    nir = scale_landsat_to_reflectance(bands["B5"], config)

    # Calculate indices
    ndvi = calculate_ndvi(red, nir)
    ndwi = calculate_ndwi(green, nir)

    # Extract masks
    cloud_mask = extract_cloud_mask(bands["QA_PIXEL"])
    water_mask = extract_water_mask(bands["QA_PIXEL"])

    # Create combined valid pixel mask
    valid_mask = ~cloud_mask & np.isfinite(ndvi) & np.isfinite(ndwi)

    return {
        "green": green,
        "red": red,
        "nir": nir,
        "ndvi": ndvi,
        "ndwi": ndwi,
        "cloud_mask": cloud_mask,
        "water_mask": water_mask,
        "valid_mask": valid_mask,
    }


def process_viirs_ntl(
    ntl: np.ndarray,
    lunar: np.ndarray,
    quality: np.ndarray,
    config: Optional[VIIRSConfig] = None,
) -> np.ndarray:
    """Process VIIRS nighttime lights with lunar correction.

    Args:
        ntl: Raw NTL radiance values
        lunar: Lunar irradiance values
        quality: Quality flag array
        config: VIIRS configuration

    Returns:
        Corrected and normalized NTL (0-1)
    """
    return apply_lunar_correction(ntl, lunar, quality, config)


def compute_ndui_enhanced(
    ntl: np.ndarray,
    ndvi: np.ndarray,
    ndwi: np.ndarray,
    roads: np.ndarray,
    config: Optional[NDUIConfig] = None,
) -> np.ndarray:
    """Compute NDUI with all enhancements.

    Args:
        ntl: Normalized nighttime lights (0-1)
        ndvi: NDVI values
        ndwi: NDWI values
        roads: Road classification array
        config: NDUI configuration

    Returns:
        Enhanced NDUI values (0-1)
    """
    return calculate_ndui(ntl, ndvi, ndwi, roads, config)


def create_quality_composite(
    masks: Dict[str, np.ndarray], combination: str = "and"
) -> np.ndarray:
    """Combine multiple quality masks.

    Args:
        masks: Dictionary of boolean masks
        combination: 'and' (all must be valid) or 'or' (any can be valid)

    Returns:
        Combined mask
    """
    mask_arrays = list(masks.values())

    if not mask_arrays:
        raise ValueError("No masks provided")

    if combination == "and":
        result = mask_arrays[0].copy()
        for mask in mask_arrays[1:]:
            result &= mask
    else:  # 'or'
        result = mask_arrays[0].copy()
        for mask in mask_arrays[1:]:
            result |= mask

    return result


def process_scene_data(
    landsat_bands: Dict[str, np.ndarray],
    viirs_data: Dict[str, np.ndarray],
    roads: np.ndarray,
    config: Optional[ProcessingConfig] = None,
) -> Dict[str, np.ndarray]:
    """Process complete scene data through the pipeline.

    This is the main processing function that combines all components.

    Args:
        landsat_bands: Dictionary with B3, B4, B5, QA_* arrays
        viirs_data: Dictionary with NTL, LUNAR, QF arrays
        roads: OSM road classification array
        config: Processing configuration

    Returns:
        Dictionary with all processed outputs
    """
    if config is None:
        config = ProcessingConfig()

    # Process Landsat
    landsat_results = process_landsat_bands(landsat_bands, config.landsat)

    # Process VIIRS
    ntl_corrected = process_viirs_ntl(
        viirs_data["NTL"], viirs_data["LUNAR"], viirs_data["QF"], config.viirs
    )

    # Compute NDUI
    ndui = compute_ndui_enhanced(
        ntl_corrected,
        landsat_results["ndvi"],
        landsat_results["ndwi"],
        roads,
        config.ndui,
    )

    # Apply quality masking
    quality_mask = create_quality_composite(
        {
            "not_cloud": ~landsat_results["cloud_mask"],
            "valid_data": landsat_results["valid_mask"],
        },
        combination="and",
    )

    ndui_masked = np.where(quality_mask, ndui, 0)

    # Compile results
    return {
        # Indices
        "ndvi": landsat_results["ndvi"],
        "ndwi": landsat_results["ndwi"],
        "ndui": ndui,
        "ndui_masked": ndui_masked,
        # Processed data
        "ntl_corrected": ntl_corrected,
        # Masks
        "cloud_mask": landsat_results["cloud_mask"],
        "water_mask": landsat_results["water_mask"],
        "quality_mask": quality_mask,
        # Reflectance bands
        "green": landsat_results["green"],
        "red": landsat_results["red"],
        "nir": landsat_results["nir"],
    }


# Parallel processing utilities


def process_tiles_parallel(
    data: np.ndarray,
    process_func,
    tile_size: int = 256,
    overlap: int = 0,
    n_workers: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """Process large array in parallel tiles.

    Args:
        data: Input array to process
        process_func: Function to apply to each tile
        tile_size: Size of each tile
        overlap: Overlap between tiles (for edge effects)
        n_workers: Number of parallel workers
        **kwargs: Additional arguments for process_func

    Returns:
        Processed array
    """
    height, width = data.shape[:2]

    # Calculate tile grid
    tiles = []
    for y in range(0, height, tile_size - overlap):
        for x in range(0, width, tile_size - overlap):
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)

            tiles.append(
                {
                    "data": data[y:y_end, x:x_end],
                    "y": y,
                    "x": x,
                    "y_end": y_end,
                    "x_end": x_end,
                }
            )

    # Process tiles in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tiles
        futures = []
        for tile in tiles:
            future = executor.submit(process_func, tile["data"], **kwargs)
            futures.append((future, tile))

        # Collect results
        result = np.zeros_like(data)
        for future, tile in futures:
            processed_tile = future.result()

            # Handle overlap by taking maximum
            if overlap > 0:
                y, x = tile["y"], tile["x"]
                y_end, x_end = tile["y_end"], tile["x_end"]

                # Crop overlap from tile (except at edges)
                if y > 0:
                    processed_tile = processed_tile[overlap // 2 :, :]
                    y += overlap // 2
                if x > 0:
                    processed_tile = processed_tile[:, overlap // 2 :]
                    x += overlap // 2
                if y_end < height:
                    processed_tile = processed_tile[: -(overlap // 2), :]
                if x_end < width:
                    processed_tile = processed_tile[:, : -(overlap // 2)]

                result[y:y_end, x:x_end] = processed_tile
            else:
                result[tile["y"] : tile["y_end"], tile["x"] : tile["x_end"]] = (
                    processed_tile
                )

    return result


def process_temporal_stack(
    temporal_data: List[Dict[str, np.ndarray]],
    method: str = "median",
    min_valid: int = 3,
) -> Dict[str, np.ndarray]:
    """Process temporal stack of scenes.

    Args:
        temporal_data: List of scene dictionaries over time
        method: Compositing method ('median', 'mean', 'max', 'min')
        min_valid: Minimum valid observations required

    Returns:
        Composite scene dictionary
    """
    if not temporal_data:
        raise ValueError("No temporal data provided")

    # Get keys from first scene
    keys = list(temporal_data[0].keys())
    result = {}

    for key in keys:
        # Stack arrays along time dimension
        stack = np.stack([scene[key] for scene in temporal_data], axis=0)

        # Handle masked/invalid data
        if "mask" in key or key in ["cloud_mask", "water_mask"]:
            # For masks, use logical operations
            if method == "median":
                # For boolean, use mode
                result[key] = np.apply_along_axis(
                    lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=stack
                ).astype(bool)
            else:
                # Default to any/all for masks
                result[key] = np.any(stack, axis=0)
        else:
            # For data arrays, compute composite
            # Count valid pixels
            valid_mask = np.isfinite(stack)
            valid_count = np.sum(valid_mask, axis=0)

            # Mask out pixels with insufficient observations
            insufficient = valid_count < min_valid

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)

                if method == "median":
                    composite = np.nanmedian(stack, axis=0)
                elif method == "mean":
                    composite = np.nanmean(stack, axis=0)
                elif method == "max":
                    composite = np.nanmax(stack, axis=0)
                elif method == "min":
                    composite = np.nanmin(stack, axis=0)
                else:
                    raise ValueError(f"Unknown method: {method}")

            # Set insufficient pixels to NaN
            composite[insufficient] = np.nan
            result[key] = composite

    return result
