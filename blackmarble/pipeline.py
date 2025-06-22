#!/usr/bin/env python3
"""
Pipeline orchestration for Black Marble processing.
Imports modules in this package to perform the full workflow:
  1. Download & merge Landsat bands + QA
  2. Download VIIRS nightlights + convert to TIFF
  3. Rasterize OSM roads
  4. Mask clouds, aerosol, radsat
  5. Stack, compute NDVI/NDWI/NDUI, colorize
  6. Export cloud-optimized GeoTIFF & upload
"""

# Allow direct execution: ensure project root is on PYTHONPATH
import os
import sys
import numpy as np
import boto3
from osgeo import gdal
import rasterio
from rasterio.io import MemoryFile
from rio_cogeo.profiles import cog_profiles
from rio_cogeo.cogeo import cog_translate
import os.path
from typing import Any
from pathlib import Path
from datetime import datetime

# Package imports
from blackmarble.coords import get_utm_zone
from blackmarble.landsat import process_band_data
from blackmarble.reproj import crop_to_bb_utm, convert_to_wgs84, subset_raster_normal
from blackmarble.io import get_geo, save_geotiff_single, scale_image
from blackmarble.masks import mark_cloud, mark_aerosol, mark_radsat
from blackmarble.utils import fill_gaps, rescale_bands
from blackmarble.vnp import VNP46A2Util
from blackmarble.osm import OSMUtil
from blackmarble.config import ProcessingConfig
from blackmarble.calculations import (
    calculate_ndvi,
    calculate_ndwi,
    calculate_ndui,
    apply_lunar_correction,
    scale_landsat_to_reflectance,
)
from blackmarble.validation import (
    check_data_coverage,
    validate_reflectance_range,
    generate_quality_report,
)

# Phase 2 adapter and data source imports
from blackmarble.adapters.s3 import S3Adapter
from blackmarble.adapters.raster_io import GDALRasterAdapter
from blackmarble.data_sources.landsat import LandsatDataSource
from blackmarble.data_sources.viirs import VIIRSDataSource

# If run as script, ensure project root is on PYTHONPATH and set package name
if __name__ == "__main__" and __package__ is None:
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, pkg_root)
    __package__ = "blackmarble"
    # Enable GDAL/OGR exceptions globally to avoid FutureWarning
    try:
        from osgeo import ogr

        ogr.UseExceptions()
    except ImportError:
        pass

# Constants are now in ProcessingConfig


def run_pipeline(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    year: int,
    month: int,
    day: int,
    earthdata_token: str,
    output_dir: str = "outputs",
    upload_bucket: str | None = None,
    upload_key: str | None = None,
    # New optional configuration parameters
    composite_months: int = 12,
    ntl_resolution: float = 0.00011111111,
    final_resolution: float = 0.0001,
    osm_buffer: float = 0.00005,
    display_min: float = 0.3,
    display_max: float = 0.8,
    config: ProcessingConfig | None = None,
    # Optional adapters for dependency injection
    s3_adapter: S3Adapter | None = None,
    raster_adapter: GDALRasterAdapter | None = None,
    landsat_source: LandsatDataSource | None = None,
    viirs_source: VIIRSDataSource | None = None,
) -> str:
    """
    Runs the Black Marble processing pipeline for a given area and date.

    Args:
        lat1 (float): Latitude of the first corner.
        lon1 (float): Longitude of the first corner.
        lat2 (float): Latitude of the second corner.
        lon2 (float): Longitude of the second corner.
        year (int): Target year.
        month (int): Target month.
        day (int): Target day.
        earthdata_token (str): NASA Earthdata login token.
        output_dir (str): Base directory for all outputs and temporary files. Defaults to "outputs".
        upload_bucket (str | None): S3 bucket name for final upload. Defaults to None.
        upload_key (str | None): S3 key (path/filename) for final upload. Defaults to None.
        composite_months (int): Number of months to look back for composite imagery. Defaults to 12.
        ntl_resolution (float): Resolution for VIIRS nighttime lights resampling in degrees. Defaults to 0.00011111111.
        final_resolution (float): Final output resolution in degrees. Defaults to 0.0001.
        osm_buffer (float): Buffer distance for OSM road rasterization in degrees. Defaults to 0.00005.
        display_min (float): Minimum value for display scaling (0-1). Defaults to 0.3.
        display_max (float): Maximum value for display scaling (0-1). Defaults to 0.8.
        config (ProcessingConfig | None): Configuration object. If None, uses defaults or environment.
        s3_adapter (S3Adapter | None): S3 storage adapter for dependency injection.
        raster_adapter (GDALRasterAdapter | None): Raster I/O adapter for dependency injection.
        landsat_source (LandsatDataSource | None): Landsat data source for dependency injection.
        viirs_source (VIIRSDataSource | None): VIIRS data source for dependency injection.

    Returns:
        str: The path to the final generated Cloud-Optimized GeoTIFF.

    Raises:
        RuntimeError: If critical steps like data download or processing fail.
    """
    # Initialize configuration
    if config is None:
        config = ProcessingConfig.from_env()
        # Override with any explicitly passed parameters
        config.landsat.composite_months = composite_months
        config.viirs.resolution = ntl_resolution
        config.final_resolution = final_resolution
        config.osm_buffer = osm_buffer
        config.display_min = display_min
        config.display_max = display_max

    # Validate configuration
    config.validate()

    # Initialize adapters if not provided (dependency injection)
    if s3_adapter is None:
        s3_adapter = S3Adapter(anonymous=False)
    if raster_adapter is None:
        raster_adapter = GDALRasterAdapter()
    # Define output subdirectories
    temp_dir = os.path.join(output_dir, "temp")
    final_dir = os.path.join(output_dir, "final")

    # Prepare output dirs
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    # Compute UTM coords
    e1: float
    n1: float
    zone1: int
    letter1: str | None
    e1, n1, zone1, letter1 = get_utm_zone(lat1, lon1)
    e2: float
    n2: float
    zone2: int
    letter2: str | None
    e2, n2, zone2, letter2 = get_utm_zone(lat2, lon2)

    # Warn if bounding box spans UTM zones
    if zone1 != zone2 or letter1 != letter2:
        print(
            f"Warning: Bounding box spans UTM zones ({zone1}{letter1} to {zone2}{letter2})"
        )
        print("This may cause projection issues at zone boundaries.")
        # Continue processing - the current implementation can handle this, but results may be less accurate

    # 1) Landsat bands
    print("Downloading and processing Landsat bands...")
    
    # Initialize band variables
    b3_out: list[str] = []
    b4_out: list[str] = []
    b5_out: list[str] = []
    b3_p: list[str] = []
    b4_p: list[str] = []
    b5_p: list[str] = []
    b3_r: list[str] = []
    b4_r: list[str] = []
    b5_r: list[str] = []
    b3_a: list[str] = []
    b4_a: list[str] = []
    b5_a: list[str] = []

    # Check if we should use the new data source approach
    if landsat_source is not None:
        # Use the new Landsat data source with adapters
        print("Using LandsatDataSource with adapters...")
        bbox = (min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2))
        start_date = datetime(year, month, day)
        end_date = start_date

        try:
            # Fetch Landsat data using the data source
            landsat_data = landsat_source.fetch(
                bbox=bbox,
                start_date=start_date,
                end_date=end_date,
                bands=["B3", "B4", "B5", "QA_PIXEL"],
                composite_months=config.landsat.composite_months,
            )

            # Save fetched data to temporary files for compatibility
            b3_out = [os.path.join(output_dir, "B3_Merged_0.TIFF")]
            b4_out = [os.path.join(output_dir, "B4_Merged_0.TIFF")]
            b5_out = [os.path.join(output_dir, "B5_Merged_0.TIFF")]
            qa_out = [os.path.join(output_dir, "QA_PIXEL_0.TIFF")]

            # Write data using raster adapter
            for path, data, band_name in [
                (b3_out[0], landsat_data.get("B3"), "B3"),
                (b4_out[0], landsat_data.get("B4"), "B4"),
                (b5_out[0], landsat_data.get("B5"), "B5"),
                (qa_out[0], landsat_data.get("QA_PIXEL"), "QA_PIXEL"),
            ]:
                if data is not None:
                    # Create minimal metadata for writing
                    metadata = {
                        "width": data.shape[-1],
                        "height": data.shape[-2],
                        "count": 1,
                        "dtype": data.dtype,
                        "driver": "GTiff",
                    }
                    raster_adapter.write(Path(path), data, metadata)

            # For compatibility, create empty lists for QA bands (will process QA_PIXEL later)
            b3_p = b3_r = b3_a = qa_out
            b4_p = b4_r = b4_a = qa_out
            b5_p = b5_r = b5_a = qa_out

        except Exception as e:
            print(f"LandsatDataSource failed: {e}. Falling back to legacy method...")
            # Fall back to legacy method
            landsat_source = None

    if landsat_source is None:
        # Use legacy process_band_data function
        b3_out, b3_p, b3_r, b3_a = process_band_data(
            3,
            year,
            month,
            day,
            lat1,
            lon1,
            lat2,
            lon2,
            os.path.join(output_dir, "B3_Merged_"),
            temp_dir,
            config.landsat.composite_months,
        )
        b4_out, b4_p, b4_r, b4_a = process_band_data(
            4,
            year,
            month,
            day,
            lat1,
            lon1,
            lat2,
            lon2,
            os.path.join(output_dir, "B4_Merged_"),
            temp_dir,
            config.landsat.composite_months,
        )
        b5_out, b5_p, b5_r, b5_a = process_band_data(
            5,
            year,
            month,
            day,
            lat1,
            lon1,
            lat2,
            lon2,
            os.path.join(output_dir, "B5_Merged_"),
            temp_dir,
            config.landsat.composite_months,
        )

    # 2) Crop to UTM bbox
    print("Cropping Landsat bands to UTM bounding box...")
    b3_final: list[str] = []
    b4_final: list[str] = []
    b5_final: list[str] = []
    p_final: list[str] = []
    r_final: list[str] = []
    a_final: list[str] = []
    
    
    for i, fn in enumerate(b3_out):
        out: str = os.path.join(output_dir, f"B3_{i}.TIFF")
        crop_to_bb_utm(fn, out, e1, n1, e2, n2)
        b3_final.append(out)
    for i, fn in enumerate(b4_out):
        out = os.path.join(output_dir, f"B4_{i}.TIFF")
        crop_to_bb_utm(fn, out, e1, n1, e2, n2)
        b4_final.append(out)
    for i, fn in enumerate(b5_out):
        out = os.path.join(output_dir, f"B5_{i}.TIFF")
        crop_to_bb_utm(fn, out, e1, n1, e2, n2)
        b5_final.append(out)
    for i, fn in enumerate(b3_p):
        out = os.path.join(output_dir, f"P_{i}.TIFF")
        crop_to_bb_utm(fn, out, e1, n1, e2, n2)
        p_final.append(out)
    for i, fn in enumerate(b3_r):
        out = os.path.join(output_dir, f"R_{i}.TIFF")
        crop_to_bb_utm(fn, out, e1, n1, e2, n2)
        r_final.append(out)
    for i, fn in enumerate(b3_a):
        out = os.path.join(output_dir, f"A_{i}.TIFF")
        crop_to_bb_utm(fn, out, e1, n1, e2, n2)
        a_final.append(out)

    # 3) Mask & Stack
    print("Masking and stacking bands...")
    print(f"DEBUG: Processing {len(b3_final)} scenes")
    b3_pxs: list[np.ndarray] = []
    b4_pxs: list[np.ndarray] = []
    b5_pxs: list[np.ndarray] = []
    for i in range(len(b3_final)):
        print(f"\nDEBUG: Processing scene {i + 1}/{len(b3_final)}")
        p_px: np.ndarray
        _: Any  # Can be dict or tuple depending on adapter
        __: Any  # Can be str or tuple depending on adapter

        # Use raster adapter if available, otherwise fall back to get_geo
        if raster_adapter is not None:
            try:
                p_data, _ = raster_adapter.read(Path(p_final[i]), band=1)
                p_px = p_data
                a_data, _ = raster_adapter.read(Path(a_final[i]), band=1)
                a_px = a_data
                r_data, _ = raster_adapter.read(Path(r_final[i]), band=1)
                r_px = r_data
            except Exception as e:
                print(f"RasterAdapter failed: {e}. Using get_geo...")
                p_px, _, __ = get_geo(p_final[i], 1)
                a_px, _, __ = get_geo(a_final[i], 1)
                r_px, _, __ = get_geo(r_final[i], 1)
        else:
            p_px, _, __ = get_geo(p_final[i], 1)
            a_px, _, __ = get_geo(a_final[i], 1)
            r_px, _, __ = get_geo(r_final[i], 1)

        # Debug QA values
        print(
            f"DEBUG: QA_PIXEL unique values: {np.unique(p_px)[:10]}... (showing first 10)"
        )
        print(
            f"DEBUG: QA_AEROSOL unique values: {np.unique(a_px)[:10]}... (showing first 10)"
        )
        print(
            f"DEBUG: QA_RADSAT unique values: {np.unique(r_px)[:10]}... (showing first 10)"
        )

        c_mask: np.ndarray = mark_cloud(p_px)
        a_mask: np.ndarray = mark_aerosol(a_px)
        b3_r_px: np.ndarray
        b4_r_px: np.ndarray
        b5_r_px: np.ndarray
        # Check if r_px has valid data before calling mark_radsat
        r_valid = np.sum(~np.isnan(r_px))
        print(
            f"DEBUG: QA_RADSAT has {r_valid} valid pixels ({100.0 * r_valid / r_px.size:.1f}%)"
        )

        if r_valid == 0:
            print("WARNING: QA_RADSAT has no valid data - assuming no saturation")
            # If no radsat data, assume no saturation (all pixels good)
            b3_r_px = np.ones_like(r_px)
            b4_r_px = np.ones_like(r_px)
            b5_r_px = np.ones_like(r_px)
        else:
            b3_r_px, b4_r_px, b5_r_px = mark_radsat(r_px)

        # Debug radsat masks
        print(
            f"DEBUG: Radsat B3 keep: {np.sum(b3_r_px > 0)} ({100.0 * np.sum(b3_r_px > 0) / b3_r_px.size:.1f}%)"
        )
        print(
            f"DEBUG: Radsat B4 keep: {np.sum(b4_r_px > 0)} ({100.0 * np.sum(b4_r_px > 0) / b4_r_px.size:.1f}%)"
        )
        print(
            f"DEBUG: Radsat B5 keep: {np.sum(b5_r_px > 0)} ({100.0 * np.sum(b5_r_px > 0) / b5_r_px.size:.1f}%)"
        )
        b3_px: np.ndarray
        b4_px: np.ndarray
        b5_px: np.ndarray

        # Use raster adapter if available, otherwise fall back to get_geo
        if raster_adapter is not None:
            try:
                b3_data, _ = raster_adapter.read(Path(b3_final[i]), band=1)
                b3_px = b3_data
                b4_data, _ = raster_adapter.read(Path(b4_final[i]), band=1)
                b4_px = b4_data
                b5_data, _ = raster_adapter.read(Path(b5_final[i]), band=1)
                b5_px = b5_data
            except Exception as e:
                print(f"RasterAdapter failed for bands: {e}. Using get_geo...")
                b3_px, _, __ = get_geo(b3_final[i], 1)
                b4_px, _, __ = get_geo(b4_final[i], 1)
                b5_px, _, __ = get_geo(b5_final[i], 1)
        else:
            b3_px, _, __ = get_geo(b3_final[i], 1)
            b4_px, _, __ = get_geo(b4_final[i], 1)
            b5_px, _, __ = get_geo(b5_final[i], 1)

        # Check band data ranges
        print(f"\nDEBUG: Band value ranges:")
        print(f"DEBUG: B3 range: {np.min(b3_px)} - {np.max(b3_px)}")
        print(f"DEBUG: B4 range: {np.min(b4_px)} - {np.max(b4_px)}")
        print(f"DEBUG: B5 range: {np.min(b5_px)} - {np.max(b5_px)}")
        # Debug masking statistics
        total_pixels = p_px.size
        cloud_keep = np.sum(c_mask > 0)
        aerosol_keep = np.sum(a_mask > 0)
        b3_radsat_keep = np.sum(b3_r_px > 0)

        print(f"DEBUG: Total pixels in scene: {total_pixels}")
        print(
            f"DEBUG: Pixels kept after cloud mask: {cloud_keep} ({100.0 * cloud_keep / total_pixels:.1f}%)"
        )
        print(
            f"DEBUG: Pixels kept after aerosol mask: {aerosol_keep} ({100.0 * aerosol_keep / total_pixels:.1f}%)"
        )
        print(
            f"DEBUG: B3 pixels kept after radsat mask: {b3_radsat_keep} ({100.0 * b3_radsat_keep / total_pixels:.1f}%)"
        )

        # Use NaN for masked pixels instead of 0 to distinguish from valid zeros
        combined_mask = c_mask * a_mask
        combined_keep = np.sum(combined_mask > 0)
        print(
            f"DEBUG: Pixels kept after combined cloud+aerosol mask: {combined_keep} ({100.0 * combined_keep / total_pixels:.1f}%)"
        )

        b3_masked = np.where(combined_mask * b3_r_px == 0, np.nan, b3_px)
        b4_masked = np.where(combined_mask * b4_r_px == 0, np.nan, b4_px)
        b5_masked = np.where(combined_mask * b5_r_px == 0, np.nan, b5_px)

        # Check valid pixels after all masking
        b3_valid = np.sum(~np.isnan(b3_masked))
        b4_valid = np.sum(~np.isnan(b4_masked))
        b5_valid = np.sum(~np.isnan(b5_masked))
        print(
            f"DEBUG: Valid pixels after all masks - B3: {b3_valid} ({100.0 * b3_valid / total_pixels:.1f}%), "
            f"B4: {b4_valid} ({100.0 * b4_valid / total_pixels:.1f}%), "
            f"B5: {b5_valid} ({100.0 * b5_valid / total_pixels:.1f}%)"
        )

        # Check if band values are in expected range
        b3_range = (np.nanmin(b3_px), np.nanmax(b3_px))
        print(f"DEBUG: B3 value range: {b3_range}")

        # IMPORTANT: Check if values need rescaling (Landsat Collection 2 DN values)
        if (
            np.nanmax(b3_px) > 1.5
        ):  # If max > 1.5, it's likely DN values not reflectance
            print(
                f"WARNING: Band values appear to be DN values (max={np.nanmax(b3_px)}), not reflectance!"
            )
            print("         These should be scaled with rescale_bands() before use.")

        b3_pxs.append(b3_masked)
        b4_pxs.append(b4_masked)
        b5_pxs.append(b5_masked)

    # Median composite - use fill_gaps to handle different array shapes
    print("\nDEBUG: Creating median composite...")
    stack3: np.ndarray = fill_gaps(b3_pxs)
    stack4: np.ndarray = fill_gaps(b4_pxs)
    stack5: np.ndarray = fill_gaps(b5_pxs)

    print(
        f"DEBUG: Stack shapes - B3: {stack3.shape}, B4: {stack4.shape}, B5: {stack5.shape}"
    )
    print(
        f"DEBUG: Valid pixels in stacks - B3: {np.sum(~np.isnan(stack3))}, "
        f"B4: {np.sum(~np.isnan(stack4))}, B5: {np.sum(~np.isnan(stack5))}"
    )

    # Already using NaN for invalid pixels, so just use nanmedian directly
    # Note: "All-NaN slice" warnings are expected for pixels with no valid data across all time periods
    with np.errstate(all="ignore"):  # Suppress warnings temporarily
        b3_med: np.ndarray = np.nanmedian(stack3, axis=0)
        b4_med: np.ndarray = np.nanmedian(stack4, axis=0)
        b5_med: np.ndarray = np.nanmedian(stack5, axis=0)

    # For pixels with no valid data across time, nanmedian returns NaN
    print(f"\nDEBUG: Pixels with no valid data across time (NaN in median):")
    print(
        f"DEBUG: B3: {np.sum(np.isnan(b3_med))} pixels ({100.0 * np.sum(np.isnan(b3_med)) / b3_med.size:.1f}%)"
    )
    print(
        f"DEBUG: B4: {np.sum(np.isnan(b4_med))} pixels ({100.0 * np.sum(np.isnan(b4_med)) / b4_med.size:.1f}%)"
    )
    print(
        f"DEBUG: B5: {np.sum(np.isnan(b5_med))} pixels ({100.0 * np.sum(np.isnan(b5_med)) / b5_med.size:.1f}%)"
    )

    # Check results
    b3_all_nan = np.all(np.isnan(b3_med))
    b4_all_nan = np.all(np.isnan(b4_med))
    b5_all_nan = np.all(np.isnan(b5_med))

    print(f"\nDEBUG: Median composite results:")
    print(f"DEBUG: B3 all NaN: {b3_all_nan}, valid pixels: {np.sum(~np.isnan(b3_med))}")
    print(f"DEBUG: B4 all NaN: {b4_all_nan}, valid pixels: {np.sum(~np.isnan(b4_med))}")
    print(f"DEBUG: B5 all NaN: {b5_all_nan}, valid pixels: {np.sum(~np.isnan(b5_med))}")

    if b3_all_nan or b4_all_nan or b5_all_nan:
        print("\nERROR: One or more bands have NO valid pixels after masking!")
        print("This typically means:")
        print("  1. The QA masks are too aggressive (marking all pixels as bad)")
        print("  2. The input data is corrupted or missing")
        print("  3. The mask logic might be inverted")
        # Don't raise error yet, let's see what happens

    # 4) Save merged bands
    print("\nDEBUG: Saving merged Landsat bands...")
    print(f"DEBUG: Reading georef from {b3_final[0]}")

    # CRITICAL: The band values are raw DN, not reflectance! Need to scale them first
    print(f"\nDEBUG: Band value ranges BEFORE rescaling:")
    print(f"DEBUG: B3 median range: {np.nanmin(b3_med)} - {np.nanmax(b3_med)}")
    print(f"DEBUG: B4 median range: {np.nanmin(b4_med)} - {np.nanmax(b4_med)}")
    print(f"DEBUG: B5 median range: {np.nanmin(b5_med)} - {np.nanmax(b5_med)}")

    # Check if data is already scaled (reflectance) or needs scaling (DN)
    b3_needs_scaling = np.nanmax(b3_med) > 1.5
    b4_needs_scaling = np.nanmax(b4_med) > 1.5
    b5_needs_scaling = np.nanmax(b5_med) > 1.5

    print(f"\nDEBUG: Checking if bands need scaling:")
    print(
        f"DEBUG: B3 needs scaling: {b3_needs_scaling} (max value: {np.nanmax(b3_med)})"
    )
    print(
        f"DEBUG: B4 needs scaling: {b4_needs_scaling} (max value: {np.nanmax(b4_med)})"
    )
    print(
        f"DEBUG: B5 needs scaling: {b5_needs_scaling} (max value: {np.nanmax(b5_med)})"
    )

    # Apply rescaling to convert DN to reflectance (only if needed)
    # The scale_landsat_to_reflectance function handles the check internally
    nodata_value = -9999.0  # Define nodata value
    b3_med_scaled = scale_landsat_to_reflectance(
        b3_med, config.landsat, nodata_value=nodata_value
    )
    b4_med_scaled = scale_landsat_to_reflectance(
        b4_med, config.landsat, nodata_value=nodata_value
    )
    b5_med_scaled = scale_landsat_to_reflectance(
        b5_med, config.landsat, nodata_value=nodata_value
    )

    print(f"\nDEBUG: Band value ranges AFTER rescaling:")
    print(
        f"DEBUG: B3 scaled range: {np.nanmin(b3_med_scaled)} - {np.nanmax(b3_med_scaled)}"
    )
    print(
        f"DEBUG: B4 scaled range: {np.nanmin(b4_med_scaled)} - {np.nanmax(b4_med_scaled)}"
    )
    print(
        f"DEBUG: B5 scaled range: {np.nanmin(b5_med_scaled)} - {np.nanmax(b5_med_scaled)}"
    )

    # Validate reflectance ranges
    b3_validation = validate_reflectance_range(b3_med_scaled, "B3")
    b4_validation = validate_reflectance_range(b4_med_scaled, "B4")
    b5_validation = validate_reflectance_range(b5_med_scaled, "B5")

    print(f"\nDEBUG: Reflectance validation:")
    print(
        f"DEBUG: B3: {b3_validation['valid_percentage']:.1f}% valid, {b3_validation['out_of_range']} out of range"
    )
    print(
        f"DEBUG: B4: {b4_validation['valid_percentage']:.1f}% valid, {b4_validation['out_of_range']} out of range"
    )
    print(
        f"DEBUG: B5: {b5_validation['valid_percentage']:.1f}% valid, {b5_validation['out_of_range']} out of range"
    )

    # Check valid pixel counts
    print(f"\nDEBUG: Valid pixel counts after processing:")
    print(f"DEBUG: B3: {np.sum(~np.isnan(b3_med_scaled))} valid pixels")
    print(f"DEBUG: B4: {np.sum(~np.isnan(b4_med_scaled))} valid pixels")
    print(f"DEBUG: B5: {np.sum(~np.isnan(b5_med_scaled))} valid pixels")

    # get georef from first band
    if raster_adapter is not None:
        try:
            _, metadata = raster_adapter.read(Path(b3_final[0]), band=1)
            # Extract geotransform and projection from metadata
            if "gdal_transform" in metadata:
                gt_save = metadata["gdal_transform"]
            elif "transform" in metadata:
                # Convert Affine to GDAL transform tuple
                t = metadata["transform"]
                gt_save = (t.c, t.a, t.b, t.f, t.d, t.e)
            else:
                raise ValueError("No geotransform in metadata")

            proj_save = metadata.get("crs", "")
            if not isinstance(proj_save, str):
                proj_save = str(proj_save)

            print(f"DEBUG: Using RasterAdapter - Geotransform: {gt_save}")
            print(f"DEBUG: Using RasterAdapter - Projection: {proj_save[:100]}...")
        except Exception as e:
            print(f"RasterAdapter failed for georef: {e}. Using get_geo...")
            _unused, gt_temp, proj_temp = get_geo(b3_final[0], 1)
            gt_save = gt_temp
            proj_save = proj_temp
    else:
        _unused, gt_temp, proj_temp = get_geo(b3_final[0], 1)

        print(f"DEBUG: Geotransform type: {type(gt_temp)}, value: {gt_temp}")
        print(
            f"DEBUG: Projection type: {type(proj_temp)}, length: {len(str(proj_temp)) if proj_temp else 0}"
        )

        # Direct assignment - get_geo always returns the correct types
        gt_save: tuple[float, ...] = gt_temp
        proj_save: str = proj_temp
    b3_merged_path = os.path.join(output_dir, "B3.TIFF")
    b4_merged_path = os.path.join(output_dir, "B4.TIFF")
    b5_merged_path = os.path.join(output_dir, "B5.TIFF")

    print(f"\nDEBUG: About to save B3 to {b3_merged_path}")
    print(
        f"DEBUG: save_geotiff_single params - shape: {b3_med_scaled.shape}, proj type: {type(proj_save)}, gt type: {type(gt_save)}"
    )

    # Save using raster adapter if available
    if raster_adapter is not None:
        try:
            # Prepare metadata for saving
            save_metadata = {
                "crs": proj_save,
                "gdal_transform": gt_save,
                "width": b3_med_scaled.shape[1],
                "height": b3_med_scaled.shape[0],
                "count": 1,
                "dtype": b3_med_scaled.dtype,
                "driver": "GTiff",
                "nodata": np.nan,
            }

            raster_adapter.write(Path(b3_merged_path), b3_med_scaled, save_metadata)
            print("DEBUG: B3 saved successfully using RasterAdapter")

            raster_adapter.write(Path(b4_merged_path), b4_med_scaled, save_metadata)
            raster_adapter.write(Path(b5_merged_path), b5_med_scaled, save_metadata)

        except Exception as e:
            print(f"RasterAdapter write failed: {e}. Using save_geotiff_single...")
            save_geotiff_single(b3_merged_path, b3_med_scaled, proj_save, gt_save)
            save_geotiff_single(b4_merged_path, b4_med_scaled, proj_save, gt_save)
            save_geotiff_single(b5_merged_path, b5_med_scaled, proj_save, gt_save)
    else:
        try:
            save_geotiff_single(b3_merged_path, b3_med_scaled, proj_save, gt_save)
            print("DEBUG: B3 saved successfully")
        except Exception as e:
            print(f"ERROR: Failed to save B3: {type(e).__name__}: {e}")
            print(f"DEBUG: proj_save = {repr(proj_save)}")
            print(f"DEBUG: gt_save = {repr(gt_save)}")
            import traceback

            traceback.print_exc()
            raise

        save_geotiff_single(b4_merged_path, b4_med_scaled, proj_save, gt_save)
        save_geotiff_single(b5_merged_path, b5_med_scaled, proj_save, gt_save)

    # 5) Reproject to WGS84
    print("Reprojecting to WGS84...")
    b3_wgs84_path = os.path.join(output_dir, "B3_WGS84.TIFF")
    b4_wgs84_path = os.path.join(output_dir, "B4_WGS84.TIFF")
    b5_wgs84_path = os.path.join(output_dir, "B5_WGS84.TIFF")
    convert_to_wgs84(b3_merged_path, b3_wgs84_path, lon1, lat1, lon2, lat2, output_dir)
    convert_to_wgs84(b4_merged_path, b4_wgs84_path, lon1, lat1, lon2, lat2, output_dir)
    convert_to_wgs84(b5_merged_path, b5_wgs84_path, lon1, lat1, lon2, lat2, output_dir)

    # 6) Download & subset Nightlights (VIIRS)
    print("\nDEBUG: Downloading nightlights (VIIRS)...")
    
    # Initialize VIIRS temp file paths
    ntl_temp_tif: str = ""
    lunar_temp_tif: str = ""
    quality_temp_tif: str = ""

    # Check if we should use the new data source approach
    if viirs_source is not None:
        # Use the new VIIRS data source with adapters
        print("Using VIIRSDataSource with adapters...")
        bbox = (min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2))
        start_date = datetime(year, month, day)
        end_date = start_date

        try:
            # Fetch VIIRS data using the data source
            viirs_data = viirs_source.fetch(
                bbox=bbox,
                start_date=start_date,
                end_date=end_date,
                layers=["NTL", "LUNAR", "QF"],
            )

            # Save fetched data to temporary files for compatibility
            ntl_temp_tif = os.path.join(temp_dir, "ntl_raw_mosaic.tif")
            lunar_temp_tif = os.path.join(temp_dir, "lunar_irradiance_mosaic.tif")
            quality_temp_tif = os.path.join(
                temp_dir, "mandatory_quality_flag_mosaic.tif"
            )

            # Write data using raster adapter
            for path, data, layer_name in [
                (ntl_temp_tif, viirs_data.get("NTL"), "NTL"),
                (lunar_temp_tif, viirs_data.get("LUNAR"), "LUNAR"),
                (quality_temp_tif, viirs_data.get("QF"), "QF"),
            ]:
                if data is not None:
                    # Create minimal metadata for writing
                    metadata = {
                        "width": data.shape[-1],
                        "height": data.shape[-2],
                        "count": 1,
                        "dtype": data.dtype,
                        "driver": "GTiff",
                    }
                    raster_adapter.write(Path(path), data, metadata)

        except Exception as e:
            print(f"VIIRSDataSource failed: {e}. Falling back to legacy method...")
            # Fall back to legacy method
            viirs_source = None

    if viirs_source is None:
        # Use legacy VNP46A2Util
        vnp = VNP46A2Util()

        # Get all tiles that intersect the bounding box
        tiles = vnp.get_tiles_for_bbox(lat1, lon1, lat2, lon2)
        print(f"Found {len(tiles)} VIIRS tiles to download: {tiles}")

        # Download and process each tile
        ntl_tiles: list[str] = []
        lunar_tiles: list[str] = []
        quality_tiles: list[str] = []

        for i, (vert, horz) in enumerate(tiles):
            print(f"Processing VIIRS tile {i + 1}/{len(tiles)}: h{horz:02d}v{vert:02d}")
            download_file: str = os.path.join(
                temp_dir, f"VNP46A2_h{horz:02d}v{vert:02d}.h5"
            )

            try:
                vnp.download_h5(
                    earthdata_token,
                    year,
                    month,
                    day,
                    vert,
                    horz,
                    target_filepath=download_file,
                )

                # Extract all three layers from VNP46A2
                ntl_tile = os.path.join(temp_dir, f"ntl_raw_h{horz:02d}v{vert:02d}.tif")
                vnp.convert_hdf_to_tiff(
                    download_file, ntl_tile, layer=0
                )  # DNB_At_Sensor_Radiance_500m
                ntl_tiles.append(ntl_tile)

                lunar_tile = os.path.join(
                    temp_dir, f"lunar_irradiance_h{horz:02d}v{vert:02d}.tif"
                )
                vnp.convert_hdf_to_tiff(
                    download_file, lunar_tile, layer=1
                )  # Lunar_Irradiance
                lunar_tiles.append(lunar_tile)

                quality_tile = os.path.join(
                    temp_dir, f"mandatory_quality_flag_h{horz:02d}v{vert:02d}.tif"
                )
                vnp.convert_hdf_to_tiff(
                    download_file, quality_tile, layer=3
                )  # Mandatory_Quality_Flag
                quality_tiles.append(quality_tile)

            except RuntimeError as e:
                print(f"Warning: Failed to download tile h{horz:02d}v{vert:02d}: {e}")
                # Continue with other tiles

        if not ntl_tiles:
            raise RuntimeError(
                "No VIIRS tiles could be downloaded for the specified area and date"
            )

        # Mosaic tiles if multiple, or use single tile
        if len(ntl_tiles) > 1:
            from .pipeline_ops import mosaic_viirs_tiles

            ntl_temp_tif, lunar_temp_tif, quality_temp_tif = mosaic_viirs_tiles(
                ntl_tiles, lunar_tiles, quality_tiles, temp_dir
            )
        else:
            # Single tile, use directly
            ntl_temp_tif = ntl_tiles[0]
            lunar_temp_tif = lunar_tiles[0]
            quality_temp_tif = quality_tiles[0]

    # subset to area + scale
    minX: float
    maxX: float
    minX, maxX = sorted([lon1, lon2])
    minY: float
    maxY: float
    minY, maxY = sorted([lat1, lat2])
    scale: float = config.viirs.resolution

    # Subset all three layers
    ntl_subset_path = os.path.join(output_dir, "ntl.tif")
    subset_raster_normal(
        ntl_temp_tif, ntl_subset_path, minX, maxY, maxX, minY, scale, temp_dir
    )

    lunar_subset_path = os.path.join(output_dir, "lunar_irradiance.tif")
    subset_raster_normal(
        lunar_temp_tif, lunar_subset_path, minX, maxY, maxX, minY, scale, temp_dir
    )

    quality_subset_path = os.path.join(output_dir, "mandatory_quality_flag.tif")
    subset_raster_normal(
        quality_temp_tif, quality_subset_path, minX, maxY, maxX, minY, scale, temp_dir
    )

    # 7) Rasterize OSM roads
    print("Rasterizing OpenStreetMap roads...")
    osm = OSMUtil()
    osm_path = os.path.join(output_dir, "osm.tif")
    osm.create_road_raster(
        lat1,
        lon1,
        lat2,
        lon2,
        reference_raster=ntl_subset_path,
        output_path=osm_path,
        output_dir=output_dir,
        path_thickness=config.osm_buffer,
    )

    # 8) Resample all to final resolution
    print("\nDEBUG: Resampling to final resolution...")
    desired_res: tuple[float, float] = (
        config.final_resolution,
        -config.final_resolution,
    )
    inputs: list[str] = [
        b3_wgs84_path,
        b4_wgs84_path,
        b5_wgs84_path,
        ntl_subset_path,
        lunar_subset_path,
        quality_subset_path,
        osm_path,
    ]
    outputs_final: list[str] = [
        os.path.join(final_dir, "B3.TIFF"),
        os.path.join(final_dir, "B4.TIFF"),
        os.path.join(final_dir, "B5.TIFF"),
        os.path.join(final_dir, "ntl.tif"),
        os.path.join(final_dir, "lunar_irradiance.tif"),
        os.path.join(final_dir, "mandatory_quality_flag.tif"),
        os.path.join(final_dir, "osm.tif"),
    ]

    print(f"DEBUG: Resampling {len(inputs)} files to resolution {desired_res}")

    for i, (inp, outp) in enumerate(zip(inputs, outputs_final)):
        print(
            f"\nDEBUG: Resampling file {i + 1}/{len(inputs)}: {os.path.basename(inp)}"
        )

        # Check input file
        try:
            with rasterio.open(inp) as src:
                print(f"  Input shape: {src.shape}, CRS: {src.crs}")
                print(f"  Input bounds: {src.bounds}")
                data = src.read(1)
                print(f"  Input data range: {np.nanmin(data)} to {np.nanmax(data)}")
                print(f"  Input NoData value: {src.nodata}")
        except Exception as e:
            print(f"  ERROR reading input: {e}")

        # Resample to final resolution
        from .pipeline_ops import resample_to_resolution

        success = resample_to_resolution(inp, outp, desired_res)

        if not success:
            print(f"  ERROR: Failed to resample {inp}!")
        else:
            print(f"  Successfully resampled to {outp}")
            # Check output
            try:
                with rasterio.open(outp) as dst:
                    print(f"  Output shape: {dst.shape}")
                    data = dst.read(1)
                    print(
                        f"  Output data range: {np.nanmin(data)} to {np.nanmax(data)}"
                    )
            except Exception as e:
                print(f"  ERROR reading output: {e}")

    # 9) Load final rasters and stack
    print("\nDEBUG: Loading final rasters and stacking...")
    b3_px_final: np.ndarray
    gt_temp3: Any
    proj_temp3: Any

    # Use raster adapter if available
    if raster_adapter is not None:
        try:
            b3_px_final, metadata = raster_adapter.read(Path(outputs_final[0]), band=1)
            # Extract geotransform and projection
            if "gdal_transform" in metadata:
                gt_temp3 = metadata["gdal_transform"]
            elif "transform" in metadata:
                t = metadata["transform"]
                gt_temp3 = (t.c, t.a, t.b, t.f, t.d, t.e)
            else:
                raise ValueError("No geotransform in metadata")

            proj_temp3 = metadata.get("crs", "")
            if not isinstance(proj_temp3, str):
                proj_temp3 = str(proj_temp3)
        except Exception as e:
            print(f"RasterAdapter failed for final read: {e}. Using get_geo...")
            b3_px_final, gt_temp3, proj_temp3 = get_geo(outputs_final[0], 1)
    else:
        b3_px_final, gt_temp3, proj_temp3 = get_geo(outputs_final[0], 1)

    print(
        f"DEBUG: Final georef - geotransform type: {type(gt_temp3)}, value: {gt_temp3}"
    )
    print(f"DEBUG: Final georef - projection type: {type(proj_temp3)}")

    # Ensure correct types
    if not isinstance(gt_temp3, tuple):
        raise RuntimeError(
            f"Expected tuple for final geotransform, got {type(gt_temp3)}: {gt_temp3}"
        )
    if not isinstance(proj_temp3, str):
        raise RuntimeError(
            f"Expected string for final projection, got {type(proj_temp3)}: {proj_temp3}"
        )

    gt: tuple[float, ...] = gt_temp3
    proj: str = proj_temp3
    # Read remaining bands
    b4_px_final: np.ndarray
    b5_px_final: np.ndarray
    ntl_px: np.ndarray
    lunar_px: np.ndarray
    quality_px: np.ndarray
    osm_px: np.ndarray

    if raster_adapter is not None:
        try:
            b4_px_final, _ = raster_adapter.read(Path(outputs_final[1]), band=1)
            b5_px_final, _ = raster_adapter.read(Path(outputs_final[2]), band=1)
            ntl_px, _ = raster_adapter.read(Path(outputs_final[3]), band=1)
            lunar_px, _ = raster_adapter.read(Path(outputs_final[4]), band=1)
            quality_px, _ = raster_adapter.read(Path(outputs_final[5]), band=1)
            osm_px, _ = raster_adapter.read(Path(outputs_final[6]), band=1)
        except Exception as e:
            print(f"RasterAdapter failed for other bands: {e}. Using get_geo...")
            # Fall back to get_geo
            _gt4: tuple[float, ...] | str
            _proj4: tuple[float, ...] | str
            b4_px_final, _gt4, _proj4 = get_geo(outputs_final[1], 1)
            _gt5: tuple[float, ...] | str
            _proj5: tuple[float, ...] | str
            b5_px_final, _gt5, _proj5 = get_geo(outputs_final[2], 1)
            _gt_ntl: tuple[float, ...] | str
            _proj_ntl: tuple[float, ...] | str
            ntl_px, _gt_ntl, _proj_ntl = get_geo(outputs_final[3], 1)
            _gt_lunar: tuple[float, ...] | str
            _proj_lunar: tuple[float, ...] | str
            lunar_px, _gt_lunar, _proj_lunar = get_geo(outputs_final[4], 1)
            _gt_quality: tuple[float, ...] | str
            _proj_quality: tuple[float, ...] | str
            quality_px, _gt_quality, _proj_quality = get_geo(outputs_final[5], 1)
            _gt_osm: tuple[float, ...] | str
            _proj_osm: tuple[float, ...] | str
            osm_px, _gt_osm, _proj_osm = get_geo(outputs_final[6], 1)
    else:
        _gt4: tuple[float, ...] | str
        _proj4: tuple[float, ...] | str
        b4_px_final, _gt4, _proj4 = get_geo(outputs_final[1], 1)
        _gt5: tuple[float, ...] | str
        _proj5: tuple[float, ...] | str
        b5_px_final, _gt5, _proj5 = get_geo(outputs_final[2], 1)
        _gt_ntl: tuple[float, ...] | str
        _proj_ntl: tuple[float, ...] | str
        ntl_px, _gt_ntl, _proj_ntl = get_geo(outputs_final[3], 1)
        _gt_lunar: tuple[float, ...] | str
        _proj_lunar: tuple[float, ...] | str
        lunar_px, _gt_lunar, _proj_lunar = get_geo(outputs_final[4], 1)
        _gt_quality: tuple[float, ...] | str
        _proj_quality: tuple[float, ...] | str
        quality_px, _gt_quality, _proj_quality = get_geo(outputs_final[5], 1)
        _gt_osm: tuple[float, ...] | str
        _proj_osm: tuple[float, ...] | str
        osm_px, _gt_osm, _proj_osm = get_geo(outputs_final[6], 1)

    # Apply fill_gaps to ensure all arrays have same shape
    stack: np.ndarray = fill_gaps(
        [b3_px_final, b4_px_final, b5_px_final, ntl_px, lunar_px, quality_px, osm_px]
    )

    print(f"\nDEBUG: Final raster value ranges BEFORE rescaling:")
    print(
        f"DEBUG: B3: min={np.nanmin(stack[0])}, max={np.nanmax(stack[0])}, shape={stack[0].shape}"
    )
    print(
        f"DEBUG: B4: min={np.nanmin(stack[1])}, max={np.nanmax(stack[1])}, shape={stack[1].shape}"
    )
    print(
        f"DEBUG: B5: min={np.nanmin(stack[2])}, max={np.nanmax(stack[2])}, shape={stack[2].shape}"
    )

    # Check for NoData values
    b3_nodata_count = np.sum(stack[0] == -9999.0)
    b4_nodata_count = np.sum(stack[1] == -9999.0)
    b5_nodata_count = np.sum(stack[2] == -9999.0)
    print(f"\nDEBUG: NoData value (-9999) counts:")
    print(
        f"DEBUG: B3: {b3_nodata_count} pixels ({100.0 * b3_nodata_count / stack[0].size:.1f}%)"
    )
    print(
        f"DEBUG: B4: {b4_nodata_count} pixels ({100.0 * b4_nodata_count / stack[1].size:.1f}%)"
    )
    print(
        f"DEBUG: B5: {b5_nodata_count} pixels ({100.0 * b5_nodata_count / stack[2].size:.1f}%)"
    )

    # Check actual valid data range (excluding NoData)
    b3_valid = stack[0][stack[0] > -1000]
    b4_valid = stack[1][stack[1] > -1000]
    b5_valid = stack[2][stack[2] > -1000]
    if len(b3_valid) > 0:
        print(f"\nDEBUG: Valid data ranges (excluding NoData):")
        print(
            f"DEBUG: B3: {np.min(b3_valid):.4f} to {np.max(b3_valid):.4f} ({len(b3_valid)} valid pixels)"
        )
        print(
            f"DEBUG: B4: {np.min(b4_valid):.4f} to {np.max(b4_valid):.4f} ({len(b4_valid)} valid pixels)"
        )
        print(
            f"DEBUG: B5: {np.min(b5_valid):.4f} to {np.max(b5_valid):.4f} ({len(b5_valid)} valid pixels)"
        )

    # Scale bands to reflectance if needed
    b3_px = scale_landsat_to_reflectance(stack[0], config.landsat, nodata_value=-9999.0)
    b4_px = scale_landsat_to_reflectance(stack[1], config.landsat, nodata_value=-9999.0)
    b5_px = scale_landsat_to_reflectance(stack[2], config.landsat, nodata_value=-9999.0)
    ntl_px = stack[3]
    lunar_px = stack[4]
    quality_px = stack[5]
    osm_px = stack[6]

    print(f"\nDEBUG: Final raster value ranges AFTER rescaling:")
    print(f"DEBUG: B3: {np.nanmin(b3_px)} - {np.nanmax(b3_px)}")
    print(f"DEBUG: B4: {np.nanmin(b4_px)} - {np.nanmax(b4_px)}")
    print(f"DEBUG: B5: {np.nanmin(b5_px)} - {np.nanmax(b5_px)}")

    # 10) Compute indices
    print("\nDEBUG: Computing NDVI, NDWI, NDUI...")

    # Check for valid data before computing indices
    b3_valid = np.sum(~np.isnan(b3_px))
    b4_valid = np.sum(~np.isnan(b4_px))
    b5_valid = np.sum(~np.isnan(b5_px))
    print(f"DEBUG: Valid pixels - B3: {b3_valid}, B4: {b4_valid}, B5: {b5_valid}")

    if b3_valid == 0 or b4_valid == 0 or b5_valid == 0:
        raise RuntimeError("No valid pixels in one or more bands after rescaling!")

    # Calculate spectral indices using refactored functions
    ndvi = calculate_ndvi(b4_px, b5_px)
    ndwi = calculate_ndwi(b3_px, b5_px)

    # Apply lunar correction using refactored function
    print(f"\nDEBUG: VIIRS quality flag unique values: {np.unique(quality_px)}")
    print(
        f"\nDEBUG: NTL value range before correction: {np.min(ntl_px[ntl_px <= 65530])} - {np.max(ntl_px[ntl_px <= 65530])}"
    )
    print(
        f"DEBUG: Lunar irradiance range: {np.min(lunar_px[lunar_px <= 65530])} - {np.max(lunar_px[lunar_px <= 65530])}"
    )

    # Apply lunar correction and normalization
    img_ntl = apply_lunar_correction(ntl_px, lunar_px, quality_px, config.viirs)

    # Debug output
    quality_mask = (quality_px == 0) | (quality_px == 2)
    quality_keep = np.sum(quality_mask)
    print(
        f"DEBUG: VIIRS pixels kept after quality mask: {quality_keep} ({100.0 * quality_keep / quality_px.size:.1f}%)"
    )
    print(
        f"DEBUG: NTL normalized range after correction: {np.min(img_ntl)} - {np.max(img_ntl)}"
    )
    img_ndvi: np.ndarray = ndvi
    img_ndwi: np.ndarray = ndwi
    img_road: np.ndarray = osm_px

    # Debug intermediate values
    print(f"\nDEBUG: Before NDUI calculation:")
    print(
        f"DEBUG: NTL pixels > 0: {np.sum(img_ntl > 0)} ({100.0 * np.sum(img_ntl > 0) / img_ntl.size:.1f}%)"
    )
    print(
        f"DEBUG: NDVI pixels > 0: {np.sum(img_ndvi > 0)} ({100.0 * np.sum(img_ndvi > 0) / img_ndvi.size:.1f}%)"
    )
    print(
        f"DEBUG: OSM road pixels > 0: {np.sum(img_road > 0)} ({100.0 * np.sum(img_road > 0) / img_road.size:.1f}%)"
    )

    # Calculate NDUI using refactored function
    print(f"\nDEBUG: Computing NDUI...")
    print(f"DEBUG: NTL normalized range: {np.nanmin(img_ntl)} - {np.nanmax(img_ntl)}")
    print(f"DEBUG: NDVI range: {np.nanmin(img_ndvi)} - {np.nanmax(img_ndvi)}")

    ndui = calculate_ndui(img_ntl, img_ndvi, img_ndwi, img_road, config.ndui)

    # Debug output
    water_pixels = np.sum(img_ndwi >= 0.0)
    print(
        f"\nDEBUG: Water pixels detected (NDWI >= 0): {water_pixels} ({100.0 * water_pixels / img_ndwi.size:.1f}%)"
    )
    print(f"DEBUG: Final NDUI range: {np.nanmin(ndui)} - {np.nanmax(ndui)}")
    print(
        f"DEBUG: Pixels > NDUI_FLOOR ({config.ndui.floor}): {np.sum(ndui > config.ndui.floor)} ({100.0 * np.sum(ndui > config.ndui.floor) / ndui.size:.1f}%)"
    )

    # Generate quality report for debugging
    quality_data = {
        "B3": b3_px,
        "B4": b4_px,
        "B5": b5_px,
        "NDVI": ndvi,
        "NDWI": ndwi,
        "NTL": img_ntl,
        "NDUI": ndui,
        "Roads": img_road,
    }
    quality_report = generate_quality_report(quality_data)

    print("\nDEBUG: Quality Report Summary:")
    for name, stats in quality_report.items():
        if stats.get("status") != "missing":
            print(
                f"  {name}: {stats['valid_percentage']:.1f}% valid, "
                f"range [{stats.get('min', 'N/A'):.3f}, {stats.get('max', 'N/A'):.3f}]"
            )

    # 11) Colorize & create COG
    print("\nDEBUG: Colorizing and creating cloud-optimized GeoTIFF...")
    print(f"DEBUG: Final NDUI range: {np.nanmin(ndui)} - {np.nanmax(ndui)}")
    print(
        f"DEBUG: Non-zero NDUI pixels: {np.sum(ndui > 0)} ({100.0 * np.sum(ndui > 0) / ndui.size:.1f}%)"
    )
    # load bundled colormap arrays
    # Use importlib.resources for Python 3.9+ compatibility
    import importlib.resources as resources

    data_dir: str = str(resources.files("blackmarble").joinpath("data"))
    r: np.ndarray = np.load(os.path.join(data_dir, "inferno_r.npy"))
    g: np.ndarray = np.load(os.path.join(data_dir, "inferno_g.npy"))
    b: np.ndarray = np.load(os.path.join(data_dir, "inferno_b.npy"))
    r[0] = 0
    g[0] = 0
    b[0] = 0
    # scale
    img_scaled: np.ndarray = scale_image(ndui, config.display_min, config.display_max)
    rr: np.ndarray = r[img_scaled]
    gg: np.ndarray = g[img_scaled]
    bb: np.ndarray = b[img_scaled]

    # build 3-band array
    with rasterio.open(
        outputs_final[0]
    ) as src:  # Use one of the final resampled files for metadata
        meta: dict[str, Any] = src.meta.copy()

    # Update metadata for uint8 RGB output
    meta.update(
        dtype="uint8",
        count=3,
        driver="GTiff",
        predictor=2,
        nodata=None,  # Remove nodata since we're creating an RGB image
    )

    print(
        f"\nDEBUG: Creating COG with metadata: dtype={meta['dtype']}, count={meta['count']}, nodata={meta.get('nodata')}"
    )

    arr: np.ndarray = np.stack([rr, gg, bb]).astype("uint8")
    out_cog: str = os.path.join(final_dir, "ndui-co.tif")
    with MemoryFile() as memfile:
        with memfile.open(**meta) as mem:
            mem.write(arr)
            profile: dict[str, Any] = cog_profiles.get("deflate")
            profile["nodata"] = None  # Ensure COG profile doesn't have nodata
            cog_translate(mem, out_cog, profile, in_memory=False)

    # 12) Upload (skip if no bucket/key provided)
    if upload_bucket and upload_key:
        print(f"Uploading {out_cog} to s3://{upload_bucket}/{upload_key}...")

        # Use S3 adapter if available, otherwise fall back to boto3
        if s3_adapter is not None:
            try:
                # Use the adapter's upload_file method
                s3_adapter.upload_file(Path(out_cog), upload_bucket, upload_key)
                print("Upload complete using S3Adapter.")
            except Exception as e:
                print(f"S3Adapter upload failed: {e}. Falling back to boto3...")
                # Fall back to direct boto3
                s3 = boto3.client("s3")
                with open(out_cog, "rb") as f:
                    s3.upload_fileobj(f, upload_bucket, upload_key)
                print("Upload complete using boto3.")
        else:
            # Use direct boto3
            s3 = boto3.client("s3")
            with open(out_cog, "rb") as f:
                s3.upload_fileobj(f, upload_bucket, upload_key)
            print("Upload complete.")
    else:
        print("UPLOAD_BUCKET/UPLOAD_FILE not set; skipping S3 upload.")

    # Clean up large intermediate files? (Optional)
    # Consider removing files in output_dir (but not final_dir or temp_dir if needed)

    return out_cog


def main() -> None:
    """Entry point for the Black Marble pipeline, reads env vars."""

    # Environment vars for S3 upload (optional)
    final_bucket: str | None = os.environ.get("UPLOAD_BUCKET")
    final_key: str | None = os.environ.get("UPLOAD_FILE")

    # AWS web identity (optional) - Write token to file if both env vars are set
    # This supports AWS IAM roles for service accounts (IRSA) in Kubernetes environments
    if (
        "AWS_WEB_IDENTITY_TOKEN_FILE" in os.environ
        and "AWS_WEB_IDENTITY_TOKEN" in os.environ
    ):
        token_file: str = os.environ["AWS_WEB_IDENTITY_TOKEN_FILE"]
        token: str = os.environ["AWS_WEB_IDENTITY_TOKEN"]
        # Ensure directory exists
        os.makedirs(os.path.dirname(token_file), exist_ok=True)
        with open(token_file, "w+") as f:
            f.write(token)

    # Read required parameters from environment
    try:
        earthdata_token: str = os.environ["EARTH_DATA_TOKEN"]
        lat1: float = float(os.environ["LAT1"])
        lon1: float = float(os.environ["LONG1"])
        lat2: float = float(os.environ["LAT2"])
        lon2: float = float(os.environ["LONG2"])
        year: int = int(os.environ["YEAR"])
        month: int = int(os.environ["MONTH"])
        day: int = int(os.environ["DAY"])
    except KeyError as e:
        print(f"Error: Missing required environment variable: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid format for environment variable: {e}", file=sys.stderr)
        sys.exit(1)

    # Define base output directory
    output_dir = "outputs"  # Keep default for script execution

    # Call the core pipeline function
    try:
        final_cog_path = run_pipeline(
            lat1=lat1,
            lon1=lon1,
            lat2=lat2,
            lon2=lon2,
            year=year,
            month=month,
            day=day,
            earthdata_token=earthdata_token,
            output_dir=output_dir,
            upload_bucket=final_bucket,
            upload_key=final_key,
        )
        print(f"Pipeline complete. Final output: {final_cog_path}")
    except Exception as e:
        print(f"Pipeline failed: {e}", file=sys.stderr)
        # Optionally re-raise or log traceback
        # import traceback
        # traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
