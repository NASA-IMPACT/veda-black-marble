from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.transform import Affine

from blackmarble.acquire.landsat import download_landsat
from blackmarble.acquire.osm import fetch_road_network
from blackmarble.acquire.osm_fractional import (
    DEFAULT_BUFFER_BY_TYPE,
    buffer_and_rasterize_roads_fractional,
)
from blackmarble.acquire.viirs import download_viirs
from blackmarble.typing import ArrayLike, BBox

from . import analyze, enhance, export, prepare
from .config import PipelineConfig
from .crs import create_processing_grid
from .logging_utils import get_logger


DEFAULT_OSM_ROAD_TYPES: list[str] = [
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "residential",
    "service",
    "unclassified",
    "road",
    "busway",
]


logger = get_logger(__name__)


# mosaic viirs data with rasterio merge to array
def mosaic_viirs(viirs_data: list[Path]) -> tuple[ArrayLike, Affine, CRS]:
    """
    Mosaics multiple VIIRS data files into a single array.

    Args:
        viirs_data (list[Path]): List of VIIRS data file paths.

    Returns:
        np.ndarray: Mosaicked VIIRS data array.
        Affine: Affine transformation for the mosaicked array.
        CRS: Coordinate reference system of the mosaicked array.
    """
    assert viirs_data, "No VIIRS data files provided for mosaicking."
    datasets = [rasterio.open(fp) for fp in viirs_data]
    try:
        crs = datasets[0].crs
        if crs is None:
            raise ValueError("VIIRS input CRS is undefined")
        mosaic, out_trans = merge(datasets, method="max")
        return mosaic, out_trans, crs
    finally:
        for dataset in datasets:
            dataset.close()


def download_data(
    bbox: BBox,
    date: datetime,
    data_dir: str = "./data",
    config: PipelineConfig | None = None,
) -> dict[str, Any]:
    logger.info("Starting data download for date %s and bbox %s", date.date(), bbox)
    data_path = Path(data_dir)
    # Download all data
    # Download VIIRS data
    logger.info("Downloading VIIRS data")
    viirs_data_dir = data_path / "viirs"
    viirs_data_dir.mkdir(parents=True, exist_ok=True)
    viirs_data = download_viirs(date, bbox, viirs_data_dir)
    viirs_file_counts = {key: len(value) for key, value in viirs_data.items()}
    logger.info("VIIRS download complete: %s", viirs_file_counts)
    # Download Landsat data
    logger.info("Downloading Landsat data")
    landsat_data_dir = data_path / "landsat"
    landsat_data_dir.mkdir(parents=True, exist_ok=True)
    landsat_data = download_landsat(date, bbox, str(landsat_data_dir))
    logger.info("Landsat download complete for %d composite date(s)", len(landsat_data))
    # Download OSM data
    logger.info("Downloading OSM road network data")
    road_types = config.acquisition.road_types if config and config.acquisition.road_types else None
    if not road_types:
        road_types = DEFAULT_OSM_ROAD_TYPES

    road_parts: list[str] = []
    for road_type in road_types:
        road_parts.extend([road_type, f"{road_type}_link"])
    custom_filter = f'["highway"~"{"|".join(road_parts)}"]'

    road_network = fetch_road_network(
        bbox,
        network_type="all",
        custom_filter=custom_filter,
        use_cache=True,
        cache_dir=data_path / "osm_cache",
    )

    logger.info(
        "OSM road network downloaded: %d nodes, %d edges",
        len(road_network.nodes),
        len(road_network.edges),
    )

    return {
        "viirs": viirs_data,
        "landsat": landsat_data,
        "roads": road_network,
    }


def process_viirs(
    viirs_data: list[Path],
    bbox: BBox,
    processing_crs: Any,
    reference_transform: Affine,
    reference_shape: tuple[int, int],
) -> ArrayLike:

    viirs_ntl, viirs_transform, viirs_crs = mosaic_viirs(viirs_data)
    if viirs_ntl.ndim == 3:
        viirs_ntl = viirs_ntl[0]

    viirs_ntl, viirs_transform, _ = prepare.crop_to_bounds(
        data=viirs_ntl,
        src_transform=viirs_transform,
        bounds=bbox,
    )

    viirs_reprojected, viirs_transform, _ = prepare.reproject_image(
        data=viirs_ntl,
        src_transform=viirs_transform,
        src_crs=viirs_crs,
        dst_crs=processing_crs,
        dst_transform=reference_transform,
        dst_shape=reference_shape,
        resampling="bilinear",  # Use bilinear like sync pipeline
    )

    reference_raster = np.zeros(reference_shape, dtype=viirs_reprojected.dtype)
    aligned_rasters = prepare.align_rasters(
        rasters=[reference_raster, viirs_reprojected],
        transforms=[reference_transform, viirs_transform],
        reference_idx=0,
        resampling="bilinear",
    )
    viirs_reprojected = aligned_rasters[1][0]

    return viirs_reprojected.astype(np.float32)


def _group_landsat_files_by_band(
    paths: Sequence[str | Path],
) -> dict[str, tuple[list[str], list[str], list[str], list[str]]]:
    bands: dict[str, list[str]] = {
        "B3": [],
        "B4": [],
        "B5": [],
        "QA_PIXEL": [],
    }

    for path in paths:
        path_str = str(path)
        name_upper = Path(path_str).name.upper()
        if "_SR_B3" in name_upper:
            bands["B3"].append(path_str)
        elif "_SR_B4" in name_upper:
            bands["B4"].append(path_str)
        elif "_SR_B5" in name_upper:
            bands["B5"].append(path_str)
        elif "_QA_PIXEL" in name_upper:
            bands["QA_PIXEL"].append(path_str)

    grouped: dict[str, tuple[list[str], list[str], list[str], list[str]]] = {}
    for band_name, band_paths in bands.items():
        if band_paths:
            grouped[band_name] = (band_paths, [], [], [])

    return grouped


def process_landsat_composite(
    landsat_data: dict[datetime, list[str]],
    bbox: BBox,
    temp_dir: str | Path,
) -> tuple[ArrayLike, ArrayLike, dict[str, Any]]:
    diagnostics_dir = Path(temp_dir) / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    index_stacks: dict[str, list[ArrayLike]] = {"ndvi": [], "ndwi": []}
    reference_profile: dict[str, Any] | None = None

    for dt, date_files in landsat_data.items():
        bands_data = _group_landsat_files_by_band(date_files)
        if not {"B3", "B4", "B5"}.issubset(set(bands_data.keys())):
            logger.warning(
                "Skipping Landsat date %s: missing required bands",
                dt.strftime("%Y-%m-%d"),
            )
            continue

        processed_data = prepare.process_landsat_date(
            bands_data=bands_data,
            bbox=bbox,
            apply_qa_mask=True,
            dilate_pixels=3,
            cloud_strategy="moderate",
            save_diagnostics=False,
            date_str=dt.strftime("%Y-%m-%d"),
            diagnostics_dir=str(diagnostics_dir),
        )

        if not {"B3", "B4", "B5"}.issubset(set(processed_data.keys())):
            logger.warning(
                "Skipping processed Landsat date %s: required bands missing after preprocessing",
                dt.strftime("%Y-%m-%d"),
            )
            continue

        b3_data, _ = processed_data["B3"]
        b4_data, _ = processed_data["B4"]
        b5_data, profile = processed_data["B5"]

        if reference_profile is None:
            reference_profile = profile

        ndvi_date = analyze.calculate_ndvi(
            nir=b5_data,
            red=b4_data,
            mask_low_denom=True,
        ).astype(np.float32)
        # Clamp negative NDVI values to 0 for NDUI calculation
        # Negative NDVI indicates water/bare soil/artificial surfaces
        ndvi_date = np.maximum(ndvi_date, 0.0)

        ndwi_date = analyze.calculate_ndwi(
            green=b3_data,
            nir=b5_data,
            mask_low_denom=True,
        ).astype(np.float32)

        index_stacks["ndvi"].append(ndvi_date)
        index_stacks["ndwi"].append(ndwi_date)

    if not index_stacks["ndvi"] or not index_stacks["ndwi"] or reference_profile is None:
        raise ValueError("No valid Landsat data available for temporal compositing")

    ndvi_composite = analyze.create_index_temporal_composite(
        index_stacks["ndvi"],
        method="percentile",
        percentile=85.0,
        min_valid_observations=1,
    )
    ndwi_composite = analyze.create_index_temporal_composite(
        index_stacks["ndwi"],
        method="median",
        min_valid_observations=1,
    )

    return ndvi_composite, ndwi_composite, reference_profile


def process_landsat(
    landsat_data: dict[datetime, list[str]],
    bbox: BBox,
    temp_dir: str | Path,
    processing_crs: Any,
    reference_transform: Affine,
    reference_shape: tuple[int, int],
) -> tuple[ArrayLike, ArrayLike]:

    ndvi_composite, ndwi_composite, reference_profile = process_landsat_composite(
        landsat_data,
        bbox,
        temp_dir,
    )

    ndvi, _, _ = prepare.reproject_image(
        data=ndvi_composite,
        src_transform=reference_profile["transform"],
        src_crs=reference_profile["crs"],
        dst_crs=processing_crs,
        dst_transform=reference_transform,
        dst_shape=reference_shape,
        resampling="bilinear",
    )

    ndwi, _, _ = prepare.reproject_image(
        data=ndwi_composite,
        src_transform=reference_profile["transform"],
        src_crs=reference_profile["crs"],
        dst_crs=processing_crs,
        dst_transform=reference_transform,
        dst_shape=reference_shape,
        resampling="bilinear",
    )

    return ndvi, ndwi


def process_roads(
    road_network: Any,
    bbox: BBox,
    reference_transform: Affine,
    processing_crs: Any,
    reference_shape: tuple[int, int],
    buffer_meters: float | dict[str, float],
    config: PipelineConfig | None,
) -> ArrayLike:
    sub_pixels = config.analysis.osm_sub_pixels if config else 5
    sigmas = config.analysis.urban_field_sigmas if config else None
    weights = config.analysis.urban_field_weights if config else None

    logger.info(
        "Processing roads: %d nodes, %d edges",
        len(road_network.nodes),
        len(road_network.edges),
    )
    roads_fractional = buffer_and_rasterize_roads_fractional(
        graph=road_network,
        bbox=bbox,
        transform=reference_transform,
        crs=processing_crs,
        shape=(reference_shape[0], reference_shape[1]),
        buffer_meters=buffer_meters,
        sub_pixels=sub_pixels,
        hierarchical=False,  # Not supported for fractional
    )

    urban_field = analyze.compute_urban_fields(
        road_fraction=roads_fractional,
        sigmas=sigmas,
        weights=weights,
    )
    return urban_field


def calculate_ndui(
    ntl_composite: ArrayLike,
    ndvi: ArrayLike,
    ndwi: ArrayLike,
    urban_field: ArrayLike,
    ceiling_value: float | None,
    config: PipelineConfig | None,
) -> tuple[ArrayLike, ArrayLike]:
    # configuration values
    enhancement_factor = config.analysis.enhancement_multiplicative_factor if config else 0.3
    ntl_ceiling = ceiling_value
    if ntl_ceiling is None:
        ntl_ceiling = config.analysis.ntl_ceiling if config else 10.0
    ntl_floor = config.analysis.ntl_floor if config else 0.1
    ndui_floor = config.analysis.ndui_floor if config else 0.02

    ntl_with_urban = analyze.enhance_ntl_with_urban_field(
        ntl=ntl_composite,
        urban_field=urban_field,
        enhancement_mode="multiplicative",
        multiplicative_factor=enhancement_factor,
    )

    ndui = analyze.calculate_ndui(
        ntl=ntl_with_urban,
        ndvi=ndvi,
        ndwi=ndwi,
        ntl_floor=ntl_floor,
        ntl_ceiling=ntl_ceiling,
        ndui_floor=ndui_floor,
        epsilon=1e-6,
    )

    ntl_enhanced = enhance.enhance_contrast(
        data=ntl_with_urban,
        method=(
            "minmax" if config and config.enhancement.contrast_method == "linear" else "percentile"
        ),
        percentiles=config.enhancement.contrast_percentiles if config else (2.0, 98.0),
        output_range=config.enhancement.contrast_output_range if config else (0.0, 1.0),
    )

    return ndui, ntl_enhanced


def generate_grid(
    bbox: BBox,
    target_resolution: float,
) -> tuple[Any, Affine, tuple[int, int]]:
    processing_crs, reference_transform, reference_shape = create_processing_grid(
        bbox, target_resolution
    )
    return processing_crs, reference_transform, reference_shape


def pipeline(
    bbox: BBox,
    date: datetime,
    output_path: str,
    data_dir: str = "./data",
    config: PipelineConfig | None = None,
) -> dict[str, Any]:
    logger.info("Starting pipeline for date %s and bbox %s", date.date(), bbox)
    logger.debug(
        "Pipeline config snapshot: target_resolution=%s, road_types=%s, osm_buffer=%s",
        30,
        (config.acquisition.road_types if config and config.acquisition.road_types else "default"),
        (config.acquisition.osm_buffer_meters if config else "default"),
    )

    logger.info("Downloading data")
    try:
        data = download_data(bbox, date, data_dir, config)
    except Exception:
        logger.exception("Data download failed")
        raise

    logger.info("Creating processing grid")
    try:
        processing_crs, reference_transform, reference_shape = generate_grid(bbox, 30)
    except Exception:
        logger.exception("Grid creation failed")
        raise

    viirs_files = data["viirs"].get("gap_filled_ntl", [])
    if not viirs_files:
        raise ValueError("No VIIRS gap-filled NTL files available")
    if len(viirs_files) < 2:
        logger.warning("Only %d VIIRS file(s) available for mosaicking", len(viirs_files))

    logger.info("Processing VIIRS")
    try:
        viirs_ntl = process_viirs(
            viirs_files,
            bbox,
            processing_crs,
            reference_transform,
            reference_shape,
        )
    except Exception:
        logger.exception("VIIRS processing failed")
        raise

    logger.info("Processing Landsat")
    if len(data["landsat"]) == 0:
        logger.warning("No Landsat dates available before processing")
    try:
        ndvi, ndwi = process_landsat(
            data["landsat"],
            bbox,
            data_dir,
            processing_crs,
            reference_transform,
            reference_shape,
        )
    except Exception:
        logger.exception("Landsat processing failed")
        raise

    if config and config.acquisition.osm_buffer_meters is not None:
        buffer_meters = config.acquisition.osm_buffer_meters
    else:
        buffer_meters = DEFAULT_BUFFER_BY_TYPE

    if isinstance(buffer_meters, dict):
        buffer_meters = {key: float(value) for key, value in buffer_meters.items()}

    logger.info("Processing roads")
    try:
        urban_field = process_roads(
            data["roads"],
            bbox,
            reference_transform,
            processing_crs,
            reference_shape,
            buffer_meters,
            config,
        )
    except Exception:
        logger.exception("Road processing failed")
        raise

    ceiling_value = config.analysis.ntl_ceiling if config else 10.0

    logger.info("Calculating NDUI")
    try:
        ndui, ntl_enhanced = calculate_ndui(
            viirs_ntl,
            ndvi,
            ndwi,
            urban_field,
            ceiling_value,
            config,
        )
    except Exception:
        logger.exception("NDUI calculation failed")
        raise

    ntl_colored = enhance.apply_inferno_colormap(
        data=ndui,
        scale_range=(0.3, 1.0),
    )

    logger.info(
        "NDUI summary: shape=%s dtype=%s min=%.4f max=%.4f",
        ndui.shape,
        ndui.dtype,
        float(np.nanmin(ndui)),
        float(np.nanmax(ndui)),
    )

    logger.info("Exporting output COGs")
    try:
        export.create_cog_local(
            data=ndui.astype(np.float32),
            output_path=output_path,
            transform=reference_transform,
            crs=processing_crs,
        )

        export.create_cog_local(
            data=ntl_colored,
            output_path=output_path.replace(".tif", "-colored.tif"),
            transform=reference_transform,
            crs=processing_crs,
        )
    except Exception:
        logger.exception("COG export failed")
        raise

    logger.info("Pipeline complete: outputs written to %s", output_path)

    return {
        "ndui": ndui,
        "ntl_enhanced": ntl_enhanced,
        "ndvi": ndvi,
        "ndwi": ndwi,
        "urban_field": urban_field,
        "output_path": output_path,
    }
