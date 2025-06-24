"""Data transformation and masking functions for Black Marble processing.

This module provides simple, focused functions for spatial data transformations
and mask generation, replacing the overly complex component system that always
returns MaskResult. These functions return what they claim to return - resampled
data or boolean/float masks!
"""

import numpy as np
import os
from datetime import datetime
from typing import Any
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio import MemoryFile
import rasterio


def resample_image(
    data: np.ndarray,
    source_resolution: float | tuple[float, float],
    target_resolution: float | tuple[float, float],
    method: str = "bilinear",
    transform: Affine | None = None,
    nodata: float | None = None,
) -> tuple[np.ndarray, Affine]:
    """Resample image data to target resolution.

    Args:
        data: Input image array (2D or 3D with bands as first dimension)
        source_resolution: Current resolution in meters (single value or (x_res, y_res))
        target_resolution: Desired resolution in meters (single value or (x_res, y_res))
        method: Resampling method ('bilinear', 'cubic', 'nearest', 'average')
        transform: Optional affine transform for the source data
        nodata: Optional nodata value

    Returns:
        Tuple of (resampled_data, new_transform)
    """
    # Normalize resolutions to tuples
    if isinstance(source_resolution, (int, float)):
        source_resolution = (source_resolution, source_resolution)
    if isinstance(target_resolution, (int, float)):
        target_resolution = (target_resolution, target_resolution)

    # Map resampling methods
    resample_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
    }
    resample_alg = resample_map.get(method, Resampling.bilinear)

    # Calculate scaling factors
    scale_x = source_resolution[0] / target_resolution[0]
    scale_y = source_resolution[1] / target_resolution[1]

    # Get dimensions
    if data.ndim == 2:
        height, width = data.shape
        bands = 1
        data_3d = data[np.newaxis, ...]
    else:
        bands, height, width = data.shape
        data_3d = data

    # Calculate new dimensions
    new_width = int(width * scale_x)
    new_height = int(height * scale_y)

    # Use identity transform if none provided
    if transform is None:
        transform = Affine.identity()

    # Create new transform with target resolution
    new_transform = Affine(
        target_resolution[0] * (1 if transform.a > 0 else -1),
        transform.b,
        transform.c,
        transform.d,
        target_resolution[1] * (1 if transform.e > 0 else -1),
        transform.f,
    )

    # Create output array
    resampled = np.empty((bands, new_height, new_width), dtype=data.dtype)

    # Use MemoryFile for in-memory resampling
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": bands,
        "dtype": data.dtype,
        "transform": transform,
        "nodata": nodata,
    }

    with MemoryFile() as memfile:
        with memfile.open(**profile) as src:
            # Write all bands
            for i in range(bands):
                src.write(data_3d[i], i + 1)

            # Read resampled data
            out_shape = (bands, new_height, new_width)
            resampled = src.read(out_shape=out_shape, resampling=resample_alg)

    # Return 2D array if input was 2D
    if data.ndim == 2:
        resampled = resampled[0]

    return resampled, new_transform


def reproject_image(
    data: np.ndarray,
    source_crs: str,
    target_crs: str,
    source_transform: Affine | None = None,
    source_bounds: tuple[float, float, float, float] | None = None,
    method: str = "bilinear",
    nodata: float | None = None,
) -> tuple[np.ndarray, Affine, tuple[float, float, float, float]]:
    """Reproject image data to target coordinate reference system.

    Args:
        data: Input image array (2D or 3D with bands as first dimension)
        source_crs: Source coordinate reference system (e.g., "EPSG:4326")
        target_crs: Target coordinate reference system
        source_transform: Affine transform for source data
        source_bounds: Optional source bounds (min_x, min_y, max_x, max_y)
        method: Resampling method ('bilinear', 'cubic', 'nearest', 'average')
        nodata: Optional nodata value

    Returns:
        Tuple of (reprojected_data, new_transform, new_bounds)
    """
    # Map resampling methods
    resample_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
    }
    resample_alg = resample_map.get(method, Resampling.bilinear)

    # Get dimensions
    if data.ndim == 2:
        height, width = data.shape
        bands = 1
        data_3d = data[np.newaxis, ...]
    else:
        bands, height, width = data.shape
        data_3d = data

    # Use identity transform if none provided
    if source_transform is None:
        source_transform = Affine.identity()

    # Calculate bounds from transform if not provided
    if source_bounds is None:
        west = source_transform.c
        north = source_transform.f
        east = west + source_transform.a * width
        south = north + source_transform.e * height
        source_bounds = (west, south, east, north)

    # Calculate transform for target CRS
    dst_transform, dst_width, dst_height = calculate_default_transform(
        source_crs, target_crs, width, height, *source_bounds
    )

    # Create output array
    reprojected = np.empty((bands, dst_height, dst_width), dtype=data.dtype)

    # Reproject each band
    for i in range(bands):
        reproject(
            source=data_3d[i],
            destination=reprojected[i],
            src_transform=source_transform,
            src_crs=source_crs,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            resampling=resample_alg,
            src_nodata=nodata,
            dst_nodata=nodata,
        )

    # Calculate new bounds
    west = dst_transform.c
    north = dst_transform.f
    east = west + dst_transform.a * dst_width
    south = north + dst_transform.e * dst_height
    new_bounds = (west, south, east, north)

    # Return 2D array if input was 2D
    if data.ndim == 2:
        reprojected = reprojected[0]

    return reprojected, dst_transform, new_bounds


def crop_to_bounds(
    data: np.ndarray,
    transform: Affine,
    bounds: tuple[float, float, float, float],
    buffer: float = 0.0,
    buffer_units: str = "crs",
) -> tuple[np.ndarray, Affine, tuple[float, float, float, float]]:
    """Crop raster data to specified spatial bounds.

    Args:
        data: Input image array (2D or 3D with bands as first dimension)
        transform: Affine transform for the data
        bounds: Target bounds (min_x, min_y, max_x, max_y) in CRS units
        buffer: Optional buffer distance to add around bounds
        buffer_units: Units for buffer ("crs" for coordinate units or "pixels")

    Returns:
        Tuple of (cropped_data, new_transform, actual_bounds)
    """
    try:
        # Use rasterio for exact cropping
        import rasterio
        from rasterio.windows import from_bounds
        
        # Apply buffer to bounds
        min_x, min_y, max_x, max_y = bounds
        if buffer > 0:
            if buffer_units == "crs":
                min_x -= buffer
                min_y -= buffer
                max_x += buffer
                max_y += buffer
            else:  # pixels
                pixel_buffer_x = buffer * abs(transform.a)
                pixel_buffer_y = buffer * abs(transform.e)
                min_x -= pixel_buffer_x
                min_y -= pixel_buffer_y
                max_x += pixel_buffer_x
                max_y += pixel_buffer_y
        
        # Get window from bounds - rasterio handles fractional pixels
        window = from_bounds(min_x, min_y, max_x, max_y, transform)
        
        # Round to integer pixels to match GDAL's projWin behavior
        col_off = round(window.col_off)
        row_off = round(window.row_off)
        width = round(window.width)
        height = round(window.height)
        
        # Get data dimensions
        if data.ndim == 2:
            data_height, data_width = data.shape
        else:
            _, data_height, data_width = data.shape
        
        # Ensure we don't exceed data bounds
        col_off = max(0, col_off)
        row_off = max(0, row_off)
        col_end = min(data_width, col_off + width)
        row_end = min(data_height, row_off + height)
        
        # Crop the data
        if data.ndim == 2:
            cropped = data[row_off:row_end, col_off:col_end]
        else:
            cropped = data[:, row_off:row_end, col_off:col_end]
        
        # Calculate the new transform
        new_transform: Affine = transform * Affine.translation(col_off, row_off)
        
        # Calculate actual bounds of the cropped data
        actual_height, actual_width = cropped.shape[-2:]
        actual_bounds = (
            new_transform.c,  # west
            new_transform.f + new_transform.e * actual_height,  # south
            new_transform.c + new_transform.a * actual_width,  # east
            new_transform.f   # north
        )
        
        return cropped, new_transform, actual_bounds
        
    except ImportError:
        # Fallback if rasterio not available
        raise ImportError("Rasterio is required for crop_to_bounds")


def crop_and_resample_to_bounds(
    data: np.ndarray,
    transform: Affine,
    bounds: tuple[float, float, float, float],
    target_resolution: float,
    resampling: str = "bilinear",
    crs: str = "EPSG:4326",
) -> tuple[np.ndarray, Affine, tuple[float, float, float, float]]:
    """Crop and resample to exact bounds at specified resolution.
    
    This replicates bm.py's behavior of resampling to a target resolution
    which creates a pixel-aligned grid that can exactly match the bounds.
    
    Args:
        data: Input image array (2D or 3D)
        transform: Affine transform for the data
        bounds: Target bounds (min_x, min_y, max_x, max_y)
        target_resolution: Target pixel size in CRS units
        resampling: Resampling method (default: "bilinear")
        crs: Coordinate reference system (default: "EPSG:4326")
        
    Returns:
        Tuple of (resampled_data, new_transform, actual_bounds)
    """
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds
    
    min_x, min_y, max_x, max_y = bounds
    
    # Calculate output dimensions for exact bounds
    out_width = int(round((max_x - min_x) / target_resolution))
    out_height = int(round((max_y - min_y) / target_resolution))
    
    # Create transform for exact bounds
    out_transform = from_bounds(min_x, min_y, max_x, max_y, out_width, out_height)
    
    # Map resampling string to enum
    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
    }
    resampling_enum = resampling_map.get(resampling, Resampling.bilinear)
    
    # Create output array
    if data.ndim == 2:
        out_shape = (out_height, out_width)
        out_data = np.empty(out_shape, dtype=data.dtype)
        
        # Reproject to exact bounds
        reproject(
            source=data,
            destination=out_data,
            src_transform=transform,
            dst_transform=out_transform,
            src_crs=crs,
            dst_crs=crs,
            resampling=resampling_enum
        )
    else:
        # Handle multi-band
        n_bands = data.shape[0]
        out_shape = (n_bands, out_height, out_width)
        out_data = np.empty(out_shape, dtype=data.dtype)
        
        for i in range(n_bands):
            reproject(
                source=data[i],
                destination=out_data[i],
                src_transform=transform,
                dst_transform=out_transform,
                src_crs=crs,
                dst_crs=crs,
                resampling=resampling_enum
            )
    
    # The output bounds are exactly what was requested
    actual_bounds = (min_x, min_y, max_x, max_y)
    
    return out_data, out_transform, actual_bounds


def align_rasters(
    rasters: list[np.ndarray],
    transforms: list[Affine],
    crs_list: list[str] | None = None,
    reference_idx: int = 0,
    method: str = "nearest",
    nodata: float | None = None,
) -> list[tuple[np.ndarray, Affine]]:
    """Align multiple rasters to a common grid and extent.

    Args:
        rasters: List of raster arrays to align
        transforms: List of affine transforms for each raster
        crs_list: Optional list of CRS strings for each raster
        reference_idx: Index of reference raster to align others to
        method: Resampling method for alignment
        nodata: Optional nodata value

    Returns:
        List of tuples (aligned_data, transform) for each raster
    """
    if not rasters:
        raise ValueError("No rasters provided for alignment")

    if len(rasters) != len(transforms):
        raise ValueError("Number of rasters must match number of transforms")

    if crs_list and len(crs_list) != len(rasters):
        raise ValueError("Number of CRS strings must match number of rasters")

    # Use reference raster for alignment target
    ref_raster = rasters[reference_idx]
    ref_transform = transforms[reference_idx]
    ref_crs = crs_list[reference_idx] if crs_list else "EPSG:4326"

    # Get reference dimensions
    if ref_raster.ndim == 2:
        ref_height, ref_width = ref_raster.shape
    else:
        _, ref_height, ref_width = ref_raster.shape

    # Map resampling methods
    resample_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
    }
    resample_alg = resample_map.get(method, Resampling.nearest)

    # Align each raster
    results = []
    for i, (raster, transform) in enumerate(zip(rasters, transforms)):
        if i == reference_idx:
            # Reference raster stays unchanged
            results.append((raster, transform))
        else:
            # Get source CRS
            src_crs = crs_list[i] if crs_list else ref_crs

            # Handle 2D or 3D data
            if raster.ndim == 2:
                bands = 1
                height, width = raster.shape
                raster_3d = raster[np.newaxis, ...]
            else:
                bands, height, width = raster.shape
                raster_3d = raster

            # Create aligned array
            aligned = np.empty((bands, ref_height, ref_width), dtype=raster.dtype)

            # Reproject each band to reference grid
            for b in range(bands):
                reproject(
                    source=raster_3d[b],
                    destination=aligned[b],
                    src_transform=transform,
                    src_crs=src_crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=resample_alg,
                    src_nodata=nodata,
                    dst_nodata=nodata,
                )

            # Return 2D if input was 2D
            if raster.ndim == 2:
                aligned = aligned[0]

            results.append((aligned, ref_transform))

    return results


# Mask generation functions


def create_cloud_mask(
    qa_pixel: np.ndarray, conservative: bool = True, dilate_pixels: int = 0
) -> np.ndarray:
    """Create cloud mask from Landsat QA_PIXEL band.

    Args:
        qa_pixel: QA_PIXEL band from Landsat Collection 2
        conservative: If True, use more aggressive cloud detection (only clear pixels)
        dilate_pixels: Number of pixels to dilate cloud mask (0 = no dilation)

    Returns:
        Boolean mask where True = clear (valid), False = cloudy (masked)
    """
    if conservative:
        # Conservative strategy: only keep clear land (21824) and clear water (21952)
        valid_values = {21824, 21952}
        mask = np.isin(qa_pixel, list(valid_values))
    else:
        # Permissive strategy: mask only high confidence clouds
        # These are the QA values for different cloud confidence levels
        invalid_values = {
            55052,
            54852,
            54596,
            22280,
        }  # High/medium/low confidence clouds
        mask = ~np.isin(qa_pixel, list(invalid_values))

    # Apply dilation if requested
    if dilate_pixels > 0:
        mask = dilate_mask(mask.astype(float), dilate_pixels) > 0

    return mask


def create_quality_mask(
    qa_bands: dict[str, np.ndarray],
    combine_method: str = "and",
    cloud_strategy: str = "conservative",
    cloud_dilate: int = 0,
    aerosol_threshold: str = "standard",
    include_saturation: bool = True,
    saturation_bands: list[int] | None = None,
) -> np.ndarray:
    """Create combined quality mask from multiple QA bands.

    This function replaces the QualityCompositor component with a simpler approach
    that directly returns a mask array instead of MaskResult objects.

    Args:
        qa_bands: Dictionary of QA band arrays. Supported keys:
            - "qa_pixel": Landsat QA_PIXEL band (required for cloud masking)
            - "qa_aerosol": Landsat QA_AEROSOL band (optional)
            - "qa_radsat": Landsat QA_RADSAT band (optional)
        combine_method: How to combine masks - "and" (all must be valid) or "or" (any valid)
        cloud_strategy: Cloud masking strategy - "conservative" or "permissive"
        cloud_dilate: Pixels to dilate cloud mask
        aerosol_threshold: Aerosol masking threshold - "standard" or "strict"
        include_saturation: Whether to include radiometric saturation masking
        saturation_bands: List of band numbers (3, 4, 5) to check for saturation.
                         If None and include_saturation=True, checks all bands.

    Returns:
        Combined quality mask where True = valid pixel, False = masked pixel
    """
    masks = []

    # Cloud mask from QA_PIXEL
    if "qa_pixel" in qa_bands:
        cloud_mask = create_cloud_mask(
            qa_bands["qa_pixel"],
            conservative=(cloud_strategy == "conservative"),
            dilate_pixels=cloud_dilate,
        )
        masks.append(cloud_mask)

    # Aerosol mask from QA_AEROSOL
    if "qa_aerosol" in qa_bands:
        aerosol_mask = create_aerosol_mask(
            qa_bands["qa_aerosol"], threshold=aerosol_threshold
        )
        masks.append(aerosol_mask)

    # Radiometric saturation mask from QA_RADSAT
    if include_saturation and "qa_radsat" in qa_bands:
        if saturation_bands:
            # Check specific bands
            for band in saturation_bands:
                sat_mask = create_saturation_mask(qa_bands["qa_radsat"], band=band)
                masks.append(sat_mask)
        else:
            # Check all bands (3, 4, 5)
            sat_mask = create_saturation_mask(qa_bands["qa_radsat"], band=None)
            masks.append(sat_mask)

    # Combine all masks
    if not masks:
        raise ValueError("No QA bands provided for quality masking")

    return combine_masks(masks, method=combine_method)


def create_aerosol_mask(
    qa_aerosol: np.ndarray, threshold: str = "standard"
) -> np.ndarray:
    """Create aerosol mask from Landsat QA_AEROSOL band.

    Args:
        qa_aerosol: QA_AEROSOL band from Landsat
        threshold: Threshold level - "standard" or "strict"

    Returns:
        Boolean mask where True = low aerosol (valid), False = high aerosol (masked)
    """
    if threshold == "standard":
        # Standard threshold masks medium and high aerosol
        invalid_values = {192, 194, 196, 228}
    else:  # strict
        # Strict threshold masks more aerosol values
        invalid_values = {192, 194, 196, 224, 228}

    # Create mask - pixels NOT in invalid set are kept
    mask = ~np.isin(qa_aerosol, list(invalid_values))

    return mask


def create_saturation_mask(
    qa_radsat: np.ndarray, band: int | None = None
) -> np.ndarray:
    """Create radiometric saturation mask from Landsat QA_RADSAT band.

    Args:
        qa_radsat: QA_RADSAT band from Landsat
        band: Specific band number (3, 4, or 5) or None to check all bands

    Returns:
        Boolean mask where True = not saturated (valid), False = saturated (masked)
    """
    # Handle NaN values by treating them as invalid
    qa_radsat_safe = np.where(np.isnan(qa_radsat), -1, qa_radsat).astype(np.int32)

    if band is not None:
        # Single band mask
        if band not in {3, 4, 5}:
            raise ValueError(f"Band must be 3, 4, or 5, got {band}")

        # Band 3 = bit 2, Band 4 = bit 3, Band 5 = bit 4
        bit_position = band - 1
        bit_value = 1 << bit_position

        # Pixel is valid if bit is NOT set
        mask = (qa_radsat_safe & bit_value) == 0

        # Also mask out original NaN pixels
        mask = np.where(np.isnan(qa_radsat), False, mask)

    else:
        # Check all bands - pixel is valid only if NO bands are saturated
        # Bits 2-4 represent bands 3-5
        any_saturated_mask = 0b11100  # Bits 2, 3, 4

        # Pixel is valid if none of the saturation bits are set
        mask = (qa_radsat_safe & any_saturated_mask) == 0

        # Also mask out original NaN pixels
        mask = np.where(np.isnan(qa_radsat), False, mask)

    return mask


def create_water_mask(ndwi: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Create water mask from Normalized Difference Water Index (NDWI).

    Args:
        ndwi: NDWI array calculated as (Green - NIR) / (Green + NIR)
        threshold: NDWI threshold for water detection (default 0.0)

    Returns:
        Boolean mask where True = land (valid), False = water (masked)
    """
    # Water has positive NDWI values
    # Land pixels are those with NDWI <= threshold
    mask = ndwi <= threshold

    # Handle NaN values
    mask = np.where(np.isnan(ndwi), False, mask)

    return mask


def create_viirs_quality_mask(
    quality_flag: np.ndarray, valid_flags: list[int] | None = None
) -> np.ndarray:
    """Create quality mask from VIIRS Mandatory_Quality_Flag band.

    Args:
        quality_flag: Mandatory_Quality_Flag array from VIIRS
        valid_flags: List of quality flag values to consider valid.
                    If None, uses default [0, 2] (high and moderate quality)

    Returns:
        Boolean mask where True = good quality (valid), False = poor quality (masked)
    """
    if valid_flags is None:
        valid_flags = [0, 2]  # Default: high quality (0) and moderate quality (2)

    # Create mask for valid quality flags
    mask = np.isin(quality_flag, valid_flags)

    return mask


def combine_masks(masks: list[np.ndarray], method: str = "and") -> np.ndarray:
    """Combine multiple masks using the specified method.

    Args:
        masks: List of mask arrays to combine (boolean or float 0/1)
        method: Combination method - "and" (all must be valid) or "or" (any can be valid)

    Returns:
        Combined mask array as boolean

    Raises:
        ValueError: If method is not recognized or masks have different shapes
    """
    if not masks:
        raise ValueError("No masks provided to combine")

    # Ensure all masks have the same shape
    shape = masks[0].shape
    for i, mask in enumerate(masks[1:], 1):
        if mask.shape != shape:
            raise ValueError(f"Mask {i} has shape {mask.shape}, expected {shape}")

    # Convert to boolean if needed
    bool_masks = [mask.astype(bool) if mask.dtype != bool else mask for mask in masks]

    if method == "and":
        # All masks must be True for pixel to be valid
        result = bool_masks[0].copy()
        for mask in bool_masks[1:]:
            result = result & mask
    elif method == "or":
        # Any mask being True means pixel is valid
        result = bool_masks[0].copy()
        for mask in bool_masks[1:]:
            result = result | mask
    else:
        raise ValueError(f"Unknown combination method: {method}")

    return result


def dilate_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Dilate a mask to expand masked regions.

    This is useful for conservative masking around clouds and other
    artifacts that may have edge effects.

    Args:
        mask: Input mask where False/0 indicates pixels to mask
        radius: Dilation radius in pixels

    Returns:
        Dilated mask with same convention as input
    """
    if radius == 0:
        return mask.copy()

    # Convert to float for processing
    result = mask.astype(float) if mask.dtype != float else mask.copy()

    # Simple dilation implementation without scipy
    # This is a basic box filter approach
    for _ in range(radius):
        # Create a temporary mask for this iteration
        temp_mask = result.copy()

        # For each pixel, if any neighbor is masked (0), mask this pixel too
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                # Check if any neighbor is masked
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ni, nj = i + di, j + dj
                        # Check bounds
                        if 0 <= ni < mask.shape[0] and 0 <= nj < mask.shape[1]:
                            if result[ni, nj] == 0:
                                temp_mask[i, j] = 0
                                break
                    if temp_mask[i, j] == 0:
                        break

        result = temp_mask

    # Return same type as input
    if mask.dtype == bool:
        return result.astype(bool)
    else:
        return result


# Enhancement functions


def enhance_contrast(
    data: np.ndarray,
    method: str = "percentile",
    clip_percentiles: tuple[float, float] = (2, 98),
    min_value: float | None = None,
    max_value: float | None = None,
) -> np.ndarray:
    """Enhance image contrast using specified method.

    This function improves the visual quality of imagery by adjusting
    the dynamic range. Unlike the component system, it returns enhanced
    data directly, not a mask!

    Args:
        data: Input image data (2D array)
        method: Enhancement method ('percentile', 'minmax', 'histogram')
        clip_percentiles: Percentiles for clipping if method='percentile'
        min_value: Optional manual minimum value for scaling
        max_value: Optional manual maximum value for scaling

    Returns:
        Contrast-enhanced image data (same shape as input)

    Example:
        >>> data = np.array([[0.1, 0.2], [0.3, 0.9]])
        >>> enhanced = enhance_contrast(data, method='percentile')
        >>> # Enhanced data will have better contrast
    """
    # Handle NaN values
    valid_mask = ~np.isnan(data)
    if not np.any(valid_mask):
        # No valid data
        return np.zeros_like(data)

    valid_data = data[valid_mask]

    if method == "percentile":
        # Calculate percentiles from valid data
        vmin = np.percentile(valid_data, clip_percentiles[0])
        vmax = np.percentile(valid_data, clip_percentiles[1])
    elif method == "minmax":
        # Use full range of valid data
        vmin = np.min(valid_data)
        vmax = np.max(valid_data)
    elif method == "histogram":
        # Histogram equalization approximation
        # For simplicity, we'll use percentile-based approach
        vmin = np.percentile(valid_data, 1)
        vmax = np.percentile(valid_data, 99)
    else:
        raise ValueError(f"Unknown enhancement method: {method}")

    # Override with manual values if provided
    if min_value is not None:
        vmin = min_value
    if max_value is not None:
        vmax = max_value

    # Avoid division by zero
    if vmax == vmin:
        return np.full_like(data, 0.5)

    # Apply linear scaling
    enhanced = (data - vmin) / (vmax - vmin)

    # Clip to valid range
    enhanced = np.clip(enhanced, 0, 1)

    # Preserve NaN values
    enhanced = np.where(valid_mask, enhanced, np.nan)

    return enhanced


def normalize_data(
    data: np.ndarray,
    output_min: float = 0.0,
    output_max: float = 1.0,
    input_min: float | None = None,
    input_max: float | None = None,
) -> np.ndarray:
    """Normalize array values to specified range.

    Linear scaling from input range to output range. This is useful
    for preparing data for visualization or analysis.

    Args:
        data: Input array to normalize
        output_min: Minimum value of output range
        output_max: Maximum value of output range
        input_min: Minimum value of input range (auto-detected if None)
        input_max: Maximum value of input range (auto-detected if None)

    Returns:
        Normalized array with values in [output_min, output_max]

    Example:
        >>> data = np.array([0, 50, 100])
        >>> normalized = normalize_data(data, 0, 1)
        >>> normalized
        array([0. , 0.5, 1. ])
    """
    # Handle NaN values
    valid_mask = ~np.isnan(data)
    if not np.any(valid_mask):
        return np.full_like(data, (output_min + output_max) / 2)

    # Auto-detect input range if not provided
    if input_min is None:
        input_min = float(np.nanmin(data))
    if input_max is None:
        input_max = float(np.nanmax(data))

    # Handle case where all values are the same
    if input_max == input_min:
        return np.full_like(data, (output_min + output_max) / 2)

    # Linear scaling
    scaled = (data - input_min) / (input_max - input_min)
    scaled = scaled * (output_max - output_min) + output_min

    return scaled


def scale_to_byte(
    data: np.ndarray,
    min_value: float | None = None,
    max_value: float | None = None,
    nodata_value: int = 0,
) -> np.ndarray:
    """Convert float data to byte values for visualization.

    Scales floating point data to 0-255 byte range, which is
    required for many image formats and visualization tools.

    Args:
        data: Input data array (float)
        min_value: Minimum value for scaling (auto-detected if None)
        max_value: Maximum value for scaling (auto-detected if None)
        nodata_value: Byte value to use for NaN/invalid pixels

    Returns:
        Byte array (uint8) with values 0-255

    Example:
        >>> data = np.array([[0.0, 0.5], [1.0, np.nan]])
        >>> byte_data = scale_to_byte(data)
        >>> byte_data
        array([[  0, 127],
               [255,   0]], dtype=uint8)
    """
    # Handle NaN values
    valid_mask = ~np.isnan(data)

    if not np.any(valid_mask):
        # No valid data
        return np.full(data.shape, nodata_value, dtype=np.uint8)

    # Auto-detect range if not provided
    if min_value is None:
        min_value = float(np.nanmin(data))
    if max_value is None:
        max_value = float(np.nanmax(data))

    # Normalize to 0-1
    if max_value > min_value:
        normalized = (data - min_value) / (max_value - min_value)
    else:
        normalized = np.zeros_like(data)

    # Scale to 0-255
    scaled = normalized * 255

    # Convert to uint8
    result = np.zeros(data.shape, dtype=np.uint8)
    result[valid_mask] = np.clip(np.round(scaled[valid_mask]), 0, 255).astype(np.uint8)
    result[~valid_mask] = nodata_value

    return result


def apply_colormap(
    data: np.ndarray,
    colormap: str = "grayscale",
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a colormap to create RGB visualization.

    Converts single-band data to RGB using a colormap. This is
    essential for creating visually appealing outputs from
    scientific data.

    Args:
        data: Input data array (2D)
        colormap: Colormap name ('grayscale', 'viridis', 'plasma', 'inferno', 'turbo')
        vmin: Minimum value for colormap scaling
        vmax: Maximum value for colormap scaling
        alpha: Optional alpha channel array (2D, same shape as data)

    Returns:
        RGB(A) array with shape (3 or 4, height, width) as uint8

    Example:
        >>> data = np.array([[0.0, 0.5], [0.7, 1.0]])
        >>> rgb = apply_colormap(data, colormap='viridis')
        >>> rgb.shape
        (3, 2, 2)
    """
    # First scale data to byte range
    byte_data = scale_to_byte(data, min_value=vmin, max_value=vmax)

    # Define some basic colormaps
    if colormap == "grayscale":
        r = g = b = byte_data
    elif colormap == "viridis":
        # Simplified viridis colormap
        t = byte_data / 255.0
        r = (68 + 163 * t).astype(np.uint8)
        g = (1 + 196 * t).astype(np.uint8)
        b = (84 + 41 * (1 - t)).astype(np.uint8)
    elif colormap == "plasma":
        # Simplified plasma colormap
        t = byte_data / 255.0
        r = (13 + 242 * t).astype(np.uint8)
        g = (8 + 104 * t * (1 - t) * 4).astype(np.uint8)
        b = (135 + 120 * (1 - t)).astype(np.uint8)
    elif colormap == "inferno":
        # Simplified inferno colormap
        t = byte_data / 255.0
        r = np.clip(255 * (t + 0.03), 0, 255).astype(np.uint8)
        g = np.clip(255 * (t - 0.3) * 1.5, 0, 255).astype(np.uint8)
        b = np.clip(255 * (1 - t) * 0.6, 0, 255).astype(np.uint8)
    elif colormap == "turbo":
        # Simplified turbo colormap
        t = byte_data / 255.0
        r = np.clip(
            34.61
            + t
            * (
                1172.33
                - t * (10793.56 - t * (33300.12 - t * (38394.49 - t * 14825.05)))
            ),
            0,
            255,
        ).astype(np.uint8)
        g = np.clip(
            23.31
            + t * (557.33 + t * (1225.33 - t * (3574.96 - t * (1073.77 + t * 707.56)))),
            0,
            255,
        ).astype(np.uint8)
        b = np.clip(
            27.2
            + t
            * (3211.1 - t * (15327.97 - t * (27814 - t * (22569.18 - t * 6838.66)))),
            0,
            255,
        ).astype(np.uint8)
    else:
        # Default to grayscale for unknown colormaps
        r = g = b = byte_data

    # Stack into RGB array
    if alpha is not None:
        # Include alpha channel
        alpha_byte = scale_to_byte(alpha, min_value=0.0, max_value=1.0)
        return np.stack([r, g, b, alpha_byte], axis=0)
    else:
        return np.stack([r, g, b], axis=0)


def enhance_roads(
    data: np.ndarray,
    roads: np.ndarray,
    major_factor: float = 1.3,
    minor_factor: float = 1.1,
    custom_factors: dict[int, float] | None = None,
) -> np.ndarray:
    """Enhance data values along road networks.

    This function enhances urban signatures along transportation corridors
    by multiplying values where roads are present. Unlike the component
    system, it returns enhanced data, not a mask!

    Road classification:
    - 0: No road
    - 1: Motorway (major)
    - 2: Primary road (minor)
    - 3: Secondary road (minor)
    - 4: Trunk road (major)
    - 5: Tertiary road (minor)

    Args:
        data: Base data layer to enhance
        roads: Road classification array (0=no road, 1-5=road types)
        major_factor: Enhancement factor for major roads (types 1, 4)
        minor_factor: Enhancement factor for minor roads (types 2, 3, 5)
        custom_factors: Optional dict mapping road type to enhancement factor

    Returns:
        Enhanced data array (same shape as input)

    Example:
        >>> data = np.array([[0.5, 0.6], [0.4, 0.7]])
        >>> roads = np.array([[1, 0], [2, 0]])
        >>> enhanced = enhance_roads(data, roads)
        >>> enhanced[0, 0]  # Major road: 0.5 * 1.3
        0.65
    """
    # Create a copy to avoid modifying input
    enhanced = data.copy()

    if custom_factors:
        # Apply custom factors for each road type
        for road_type, factor in custom_factors.items():
            mask = roads == road_type
            enhanced = np.where(mask, enhanced * factor, enhanced)
    else:
        # Apply default factors
        # Major roads (motorway and trunk)
        major_mask = (roads == 1) | (roads == 4)
        enhanced = np.where(major_mask, enhanced * major_factor, enhanced)

        # Minor roads (primary, secondary, tertiary)
        minor_mask = (roads == 2) | (roads == 3) | (roads == 5)
        enhanced = np.where(minor_mask, enhanced * minor_factor, enhanced)

    return enhanced


def enhance_urban_features(
    ndui: np.ndarray,
    roads: np.ndarray,
    ntl: np.ndarray | None = None,
    enhancement_strength: float = 0.3,
) -> np.ndarray:
    """Enhance urban features in NDUI using road and nightlight data.

    This function applies targeted enhancements to better highlight
    urban areas, transportation corridors, and lit infrastructure.

    Args:
        ndui: Normalized Difference Urban Index data
        roads: Road classification array
        ntl: Optional nighttime lights data for additional enhancement
        enhancement_strength: Strength of enhancement (0-1)

    Returns:
        Enhanced NDUI with improved urban feature visibility

    Example:
        >>> ndui = np.array([[0.3, 0.5], [0.6, 0.8]])
        >>> roads = np.array([[1, 0], [0, 2]])
        >>> enhanced = enhance_urban_features(ndui, roads)
    """
    # Start with base NDUI
    enhanced = ndui.copy()

    # Enhance along roads with graduated factors
    road_factors = {
        1: 1.0 + 0.4 * enhancement_strength,  # Motorway
        2: 1.0 + 0.3 * enhancement_strength,  # Primary
        3: 1.0 + 0.2 * enhancement_strength,  # Secondary
        4: 1.0 + 0.35 * enhancement_strength,  # Trunk
        5: 1.0 + 0.15 * enhancement_strength,  # Tertiary
    }

    enhanced = enhance_roads(enhanced, roads, custom_factors=road_factors)

    # If nighttime lights are provided, use them for additional enhancement
    if ntl is not None:
        # Enhance bright areas (high NTL values)
        bright_mask = ntl > np.percentile(ntl[ntl > 0], 75)
        ntl_factor = 1.0 + 0.2 * enhancement_strength
        enhanced = np.where(bright_mask, enhanced * ntl_factor, enhanced)

        # Slightly enhance moderately lit areas
        moderate_mask = (ntl > np.percentile(ntl[ntl > 0], 50)) & (~bright_mask)
        moderate_factor = 1.0 + 0.1 * enhancement_strength
        enhanced = np.where(moderate_mask, enhanced * moderate_factor, enhanced)

    # Ensure values stay in valid range
    enhanced = np.clip(enhanced, 0, 1)

    return enhanced


# Temporal and composite functions


def create_temporal_composite(
    data_stack: np.ndarray,
    timestamps: list[datetime] | None = None,
    method: str = "median",
    weights: np.ndarray | None = None,
    percentile: float | None = None,
    min_valid_observations: int = 1,
    fill_gaps: bool = False,
) -> np.ndarray:
    """Create temporal composite from time series data.

    This function combines multiple temporal observations into a single
    composite image using various statistical methods. It handles missing
    data and can optionally fill gaps using nearest neighbor interpolation.

    Args:
        data_stack: 3D array with shape (time, height, width) or
                   4D array with shape (time, bands, height, width)
        timestamps: Optional list of timestamps for each layer
        method: Compositing method ('median', 'mean', 'max', 'min', 'percentile', 'quality_weighted')
        weights: Optional quality weights for each pixel/time (same shape as data_stack)
        percentile: Percentile value if method='percentile' (0-100)
        min_valid_observations: Minimum observations required for a valid composite pixel
        fill_gaps: Whether to fill gaps using nearest neighbor interpolation

    Returns:
        2D composite array (height, width) or 3D (bands, height, width) for multi-band data

    Example:
        >>> # Create median composite from 5 dates
        >>> data = np.random.rand(5, 100, 100)  # 5 dates, 100x100 pixels
        >>> composite = create_temporal_composite(data, method='median')
        >>> composite.shape
        (100, 100)
    """
    if data_stack.ndim < 3:
        raise ValueError(f"Expected 3D or 4D array, got shape {data_stack.shape}")

    # Handle NaN values to create valid mask
    valid_mask = ~np.isnan(data_stack)

    # Count valid observations per pixel
    if data_stack.ndim == 3:
        valid_count = np.sum(valid_mask, axis=0)
    else:
        # For multi-band, use minimum valid count across bands
        valid_count = np.min(np.sum(valid_mask, axis=0), axis=0)

    # Create composite based on method
    if method == "median":
        composite = _median_composite(data_stack, valid_mask)
    elif method == "mean":
        composite = _mean_composite(data_stack, valid_mask)
    elif method == "max":
        composite = _max_composite(data_stack, valid_mask)
    elif method == "min":
        composite = _min_composite(data_stack, valid_mask)
    elif method == "percentile":
        if percentile is None:
            raise ValueError("Percentile value required for 'percentile' method")
        composite = _percentile_composite(data_stack, valid_mask, percentile)
    elif method == "quality_weighted":
        if weights is None:
            raise ValueError("Weights required for 'quality_weighted' method")
        composite = _quality_weighted_composite(data_stack, valid_mask, weights)
    else:
        raise ValueError(f"Unknown composite method: {method}")

    # Mark invalid pixels (insufficient observations)
    if data_stack.ndim == 3:
        composite[valid_count < min_valid_observations] = np.nan
    else:
        # For multi-band data
        for i in range(composite.shape[0]):
            composite[i][valid_count < min_valid_observations] = np.nan

    # Fill gaps if requested
    if fill_gaps:
        composite = _fill_gaps_nearest(composite)

    return composite


def _median_composite(data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Create median composite."""
    masked_data = np.ma.masked_where(~valid_mask, data)
    return np.ma.median(masked_data, axis=0).filled(np.nan)


def _mean_composite(data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Create mean composite."""
    masked_data = np.ma.masked_where(~valid_mask, data)
    return np.ma.mean(masked_data, axis=0).filled(np.nan)


def _max_composite(data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Create maximum value composite."""
    masked_data = np.ma.masked_where(~valid_mask, data)
    return np.ma.max(masked_data, axis=0).filled(np.nan)


def _min_composite(data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Create minimum value composite."""
    masked_data = np.ma.masked_where(~valid_mask, data)
    return np.ma.min(masked_data, axis=0).filled(np.nan)


def _percentile_composite(
    data: np.ndarray, valid_mask: np.ndarray, percentile: float
) -> np.ndarray:
    """Create percentile composite."""
    # For percentile, we need to handle each pixel separately
    # due to varying number of valid observations
    output_shape = data.shape[1:]
    composite = np.full(output_shape, np.nan)

    # Process each pixel
    if data.ndim == 3:
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                valid_data = data[valid_mask[:, i, j], i, j]
                if len(valid_data) > 0:
                    composite[i, j] = np.percentile(valid_data, percentile)
    else:
        # Multi-band case
        for b in range(output_shape[0]):
            for i in range(output_shape[1]):
                for j in range(output_shape[2]):
                    valid_data = data[valid_mask[:, b, i, j], b, i, j]
                    if len(valid_data) > 0:
                        composite[b, i, j] = np.percentile(valid_data, percentile)

    return composite


def _quality_weighted_composite(
    data: np.ndarray, valid_mask: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Create quality-weighted composite."""
    # Ensure weights match data shape
    if weights.shape[0] != data.shape[0]:
        raise ValueError("Weights must match time dimension")

    # Expand weights to match data dimensions if needed
    if data.ndim == 4 and weights.ndim == 3:
        weights = weights[:, np.newaxis, :, :]
    elif data.ndim == 3 and weights.ndim == 1:
        weights = weights[:, np.newaxis, np.newaxis]

    # Apply valid mask to weights
    masked_weights = weights * valid_mask

    # Normalize weights
    weight_sum = np.sum(masked_weights, axis=0)
    weight_sum[weight_sum == 0] = 1  # Avoid division by zero

    # Compute weighted average
    weighted_data = data * masked_weights
    composite = np.sum(weighted_data, axis=0) / weight_sum

    return composite


def _fill_gaps_nearest(data: np.ndarray) -> np.ndarray:
    """Fill gaps using nearest neighbor approach.

    This is a simplified version that doesn't require scipy.
    For production use, consider using scipy's distance_transform_edt.
    """
    # Find indices of valid and invalid pixels
    if data.ndim == 2:
        valid_mask = ~np.isnan(data)
        if not np.any(~valid_mask):
            return data  # No gaps to fill

        # Simple iterative filling
        filled = data.copy()
        max_iterations = max(data.shape)

        for _ in range(max_iterations):
            old_filled = filled.copy()

            # For each NaN pixel, check neighbors
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if np.isnan(filled[i, j]):
                        # Collect valid neighbors
                        neighbors = []
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di == 0 and dj == 0:
                                    continue
                                ni, nj = i + di, j + dj
                                if 0 <= ni < data.shape[0] and 0 <= nj < data.shape[1]:
                                    if not np.isnan(filled[ni, nj]):
                                        neighbors.append(filled[ni, nj])

                        # Fill with mean of valid neighbors
                        if neighbors:
                            filled[i, j] = np.mean(neighbors)

            # Check if we've filled all gaps
            if np.allclose(filled, old_filled, equal_nan=True):
                break

        return filled
    else:
        # Multi-band case - fill each band separately
        filled = np.empty_like(data)
        for b in range(data.shape[0]):
            filled[b] = _fill_gaps_nearest(data[b])
        return filled


def merge_scenes(
    scenes: list[np.ndarray],
    transforms: list[Affine],
    bounds: list[tuple[float, float, float, float]],
    method: str = "feather",
    feather_distance: int = 10,
    nodata: float | None = None,
) -> tuple[np.ndarray, Affine, tuple[float, float, float, float]]:
    """Merge multiple scenes into a single mosaic.

    Handles overlapping scenes with various blending methods to create
    seamless mosaics. Unlike the component system, this returns actual
    merged data!

    Args:
        scenes: List of scene arrays to merge
        transforms: List of affine transforms for each scene
        bounds: List of (min_x, min_y, max_x, max_y) tuples for each scene
        method: Blending method ('feather', 'max', 'min', 'mean', 'first')
        feather_distance: Distance for feathering in pixels (only for 'feather')
        nodata: Value representing no data

    Returns:
        Tuple of (merged_data, merged_transform, merged_bounds)

    Example:
        >>> # Merge two overlapping scenes
        >>> scene1 = np.ones((100, 100)) * 0.5
        >>> scene2 = np.ones((100, 100)) * 0.7
        >>> transform1 = Affine.identity()
        >>> transform2 = Affine.translation(50, 0)
        >>> bounds1 = (0, 0, 100, 100)
        >>> bounds2 = (50, 0, 150, 100)
        >>> merged, transform, bounds = merge_scenes(
        ...     [scene1, scene2], [transform1, transform2], [bounds1, bounds2]
        ... )
    """
    if not scenes:
        raise ValueError("No scenes provided for merging")

    if len(scenes) != len(transforms) or len(scenes) != len(bounds):
        raise ValueError("Number of scenes, transforms, and bounds must match")

    # Calculate output bounds
    min_x = min(b[0] for b in bounds)
    min_y = min(b[1] for b in bounds)
    max_x = max(b[2] for b in bounds)
    max_y = max(b[3] for b in bounds)
    merged_bounds = (min_x, min_y, max_x, max_y)

    # Use first transform's resolution
    res_x = abs(transforms[0].a)
    res_y = abs(transforms[0].e)

    # Calculate output dimensions
    width = int(np.ceil((max_x - min_x) / res_x))
    height = int(np.ceil((max_y - min_y) / res_y))

    # Create output transform
    # Use Affine directly to maintain same orientation as input
    merged_transform = Affine(
        transforms[0].a,  # Keep x resolution with sign
        0,  # No rotation
        min_x,  # Left edge
        0,  # No rotation
        transforms[0].e,  # Keep y resolution with sign
        max_y if transforms[0].e < 0 else min_y,  # Top edge if y is negative
    )

    # Initialize output array
    shape = scenes[0].shape
    if nodata is None:
        nodata = np.nan

    if len(shape) == 2:
        merged = np.full((height, width), nodata, dtype=scenes[0].dtype)
    else:
        merged = np.full((shape[0], height, width), nodata, dtype=scenes[0].dtype)

    # Merge based on method
    if method == "feather":
        merged = _feather_blend_simple(
            merged, scenes, transforms, merged_transform, feather_distance, nodata
        )
    elif method == "first":
        merged = _first_valid_merge(
            merged, scenes, transforms, merged_transform, nodata
        )
    elif method == "max":
        merged = _max_merge(merged, scenes, transforms, merged_transform, nodata)
    elif method == "min":
        merged = _min_merge(merged, scenes, transforms, merged_transform, nodata)
    elif method == "mean":
        merged = _mean_merge(merged, scenes, transforms, merged_transform, nodata)
    else:
        raise ValueError(f"Unknown blend method: {method}")

    return merged, merged_transform, merged_bounds


def _feather_blend_simple(
    output: np.ndarray,
    scenes: list[np.ndarray],
    transforms: list[Affine],
    out_transform: Affine,
    feather_distance: int,
    nodata: float,
) -> np.ndarray:
    """Simple feather blending without scipy."""
    # Process each scene
    for scene, transform in zip(scenes, transforms):
        # Calculate offset in output array
        col_off = int((transform.c - out_transform.c) / out_transform.a)
        row_off = int((transform.f - out_transform.f) / out_transform.e)

        # Get valid data mask
        if np.isnan(nodata):
            valid_mask = ~np.isnan(scene)
        else:
            valid_mask = scene != nodata

        if scene.ndim > 2:
            valid_mask = np.all(valid_mask, axis=0)

        # Simple distance-based weights (approximation)
        # Create weight array that's 1 in center and fades at edges
        weights = np.ones_like(valid_mask, dtype=float)

        # Fade edges
        for i in range(feather_distance):
            fade = (i + 1) / feather_distance
            # Top and bottom edges
            if i < valid_mask.shape[0]:
                weights[i, :] = np.minimum(weights[i, :], fade)
                weights[-i - 1, :] = np.minimum(weights[-i - 1, :], fade)
            # Left and right edges
            if i < valid_mask.shape[1]:
                weights[:, i] = np.minimum(weights[:, i], fade)
                weights[:, -i - 1] = np.minimum(weights[:, -i - 1], fade)

        weights = weights * valid_mask

        # Apply to output
        if scene.ndim == 2:
            # Calculate valid slice bounds
            row_start = max(0, row_off)
            row_end = min(output.shape[0], row_off + scene.shape[0])
            col_start = max(0, col_off)
            col_end = min(output.shape[1], col_off + scene.shape[1])

            # Calculate corresponding scene slice
            scene_row_start = max(0, -row_off)
            scene_row_end = scene_row_start + (row_end - row_start)
            scene_col_start = max(0, -col_off)
            scene_col_end = scene_col_start + (col_end - col_start)

            out_slice = output[row_start:row_end, col_start:col_end]
            scene_slice = scene[
                scene_row_start:scene_row_end, scene_col_start:scene_col_end
            ]
            valid_slice = valid_mask[
                scene_row_start:scene_row_end, scene_col_start:scene_col_end
            ]
            weights_slice = weights[
                scene_row_start:scene_row_end, scene_col_start:scene_col_end
            ]

            # Check for existing data
            existing_mask = (
                ~np.isnan(out_slice) if np.isnan(nodata) else out_slice != nodata
            )

            # Blend where we have overlap
            overlap = valid_slice & existing_mask
            if np.any(overlap):
                total_weight = weights_slice + existing_mask.astype(float)
                total_weight[total_weight == 0] = 1
                out_slice[overlap] = (
                    weights_slice[overlap] * scene_slice[overlap]
                    + existing_mask[overlap] * out_slice[overlap]
                ) / total_weight[overlap]

            # Copy where we only have new data
            new_only = valid_slice & ~existing_mask
            out_slice[new_only] = scene_slice[new_only]
        else:
            # Multi-band case
            for b in range(scene.shape[0]):
                # Calculate valid slice bounds
                row_start = max(0, row_off)
                row_end = min(output.shape[1], row_off + scene.shape[1])
                col_start = max(0, col_off)
                col_end = min(output.shape[2], col_off + scene.shape[2])

                # Calculate corresponding scene slice
                scene_row_start = max(0, -row_off)
                scene_row_end = scene_row_start + (row_end - row_start)
                scene_col_start = max(0, -col_off)
                scene_col_end = scene_col_start + (col_end - col_start)

                out_slice = output[b, row_start:row_end, col_start:col_end]
                scene_slice = scene[
                    b, scene_row_start:scene_row_end, scene_col_start:scene_col_end
                ]
                valid_slice = valid_mask[
                    scene_row_start:scene_row_end, scene_col_start:scene_col_end
                ]
                weights_slice = weights[
                    scene_row_start:scene_row_end, scene_col_start:scene_col_end
                ]

                existing_mask = (
                    ~np.isnan(out_slice) if np.isnan(nodata) else out_slice != nodata
                )

                overlap = valid_slice & existing_mask
                if np.any(overlap):
                    total_weight = weights_slice + existing_mask.astype(float)
                    total_weight[total_weight == 0] = 1
                    out_slice[overlap] = (
                        weights_slice[overlap] * scene_slice[overlap]
                        + existing_mask[overlap] * out_slice[overlap]
                    ) / total_weight[overlap]

                new_only = valid_slice & ~existing_mask
                out_slice[new_only] = scene_slice[new_only]

    return output


def _first_valid_merge(
    output: np.ndarray,
    scenes: list[np.ndarray],
    transforms: list[Affine],
    out_transform: Affine,
    nodata: float,
) -> np.ndarray:
    """Merge using first valid pixel."""
    for scene, transform in zip(scenes, transforms):
        col_off = int((transform.c - out_transform.c) / out_transform.a)
        row_off = int((transform.f - out_transform.f) / out_transform.e)

        if np.isnan(nodata):
            valid_mask = ~np.isnan(scene)
        else:
            valid_mask = scene != nodata

        if scene.ndim == 2:
            # Calculate valid slice bounds
            row_start = max(0, row_off)
            row_end = min(output.shape[0], row_off + scene.shape[0])
            col_start = max(0, col_off)
            col_end = min(output.shape[1], col_off + scene.shape[1])

            # Calculate corresponding scene slice
            scene_row_start = max(0, -row_off)
            scene_row_end = scene_row_start + (row_end - row_start)
            scene_col_start = max(0, -col_off)
            scene_col_end = scene_col_start + (col_end - col_start)

            out_slice = output[row_start:row_end, col_start:col_end]
            scene_slice = scene[
                scene_row_start:scene_row_end, scene_col_start:scene_col_end
            ]
            valid_slice = valid_mask[
                scene_row_start:scene_row_end, scene_col_start:scene_col_end
            ]

            # Only update where output is still nodata
            update_mask = valid_slice & (
                np.isnan(out_slice) if np.isnan(nodata) else out_slice == nodata
            )
            out_slice[update_mask] = scene_slice[update_mask]
        else:
            for b in range(scene.shape[0]):
                # Calculate valid slice bounds
                row_start = max(0, row_off)
                row_end = min(output.shape[1], row_off + scene.shape[1])
                col_start = max(0, col_off)
                col_end = min(output.shape[2], col_off + scene.shape[2])

                # Calculate corresponding scene slice
                scene_row_start = max(0, -row_off)
                scene_row_end = scene_row_start + (row_end - row_start)
                scene_col_start = max(0, -col_off)
                scene_col_end = scene_col_start + (col_end - col_start)

                out_slice = output[b, row_start:row_end, col_start:col_end]
                scene_slice = scene[
                    b, scene_row_start:scene_row_end, scene_col_start:scene_col_end
                ]
                band_valid = valid_mask if valid_mask.ndim == 2 else valid_mask[b]
                valid_slice = band_valid[
                    scene_row_start:scene_row_end, scene_col_start:scene_col_end
                ]

                update_mask = valid_slice & (
                    np.isnan(out_slice) if np.isnan(nodata) else out_slice == nodata
                )
                out_slice[update_mask] = scene_slice[update_mask]

    return output


def _max_merge(
    output: np.ndarray,
    scenes: list[np.ndarray],
    transforms: list[Affine],
    out_transform: Affine,
    nodata: float,
) -> np.ndarray:
    """Merge using maximum values."""
    # Initialize with -inf for max comparison
    output.fill(-np.inf)

    for scene, transform in zip(scenes, transforms):
        col_off = int((transform.c - out_transform.c) / out_transform.a)
        row_off = int((transform.f - out_transform.f) / out_transform.e)

        if np.isnan(nodata):
            valid_mask = ~np.isnan(scene)
        else:
            valid_mask = scene != nodata

        if scene.ndim == 2:
            # Calculate valid slice bounds
            row_start = max(0, row_off)
            row_end = min(output.shape[0], row_off + scene.shape[0])
            col_start = max(0, col_off)
            col_end = min(output.shape[1], col_off + scene.shape[1])

            # Calculate corresponding scene slice
            scene_row_start = max(0, -row_off)
            scene_row_end = scene_row_start + (row_end - row_start)
            scene_col_start = max(0, -col_off)
            scene_col_end = scene_col_start + (col_end - col_start)

            out_slice = output[row_start:row_end, col_start:col_end]
            scene_slice = scene[
                scene_row_start:scene_row_end, scene_col_start:scene_col_end
            ]
            valid_slice = valid_mask[
                scene_row_start:scene_row_end, scene_col_start:scene_col_end
            ]

            np.maximum(out_slice, scene_slice, out=out_slice, where=valid_slice)
        else:
            for b in range(scene.shape[0]):
                # Calculate valid slice bounds
                row_start = max(0, row_off)
                row_end = min(output.shape[1], row_off + scene.shape[1])
                col_start = max(0, col_off)
                col_end = min(output.shape[2], col_off + scene.shape[2])

                # Calculate corresponding scene slice
                scene_row_start = max(0, -row_off)
                scene_row_end = scene_row_start + (row_end - row_start)
                scene_col_start = max(0, -col_off)
                scene_col_end = scene_col_start + (col_end - col_start)

                out_slice = output[b, row_start:row_end, col_start:col_end]
                scene_slice = scene[
                    b, scene_row_start:scene_row_end, scene_col_start:scene_col_end
                ]
                band_valid = valid_mask if valid_mask.ndim == 2 else valid_mask[b]
                valid_slice = band_valid[
                    scene_row_start:scene_row_end, scene_col_start:scene_col_end
                ]

                np.maximum(out_slice, scene_slice, out=out_slice, where=valid_slice)

    # Replace -inf with nodata
    output[output == -np.inf] = nodata
    return output


def _min_merge(
    output: np.ndarray,
    scenes: list[np.ndarray],
    transforms: list[Affine],
    out_transform: Affine,
    nodata: float,
) -> np.ndarray:
    """Merge using minimum values."""
    # Initialize with inf for min comparison
    output.fill(np.inf)

    for scene, transform in zip(scenes, transforms):
        col_off = int((transform.c - out_transform.c) / out_transform.a)
        row_off = int((transform.f - out_transform.f) / out_transform.e)

        if np.isnan(nodata):
            valid_mask = ~np.isnan(scene)
        else:
            valid_mask = scene != nodata

        if scene.ndim == 2:
            # Calculate valid slice bounds
            row_start = max(0, row_off)
            row_end = min(output.shape[0], row_off + scene.shape[0])
            col_start = max(0, col_off)
            col_end = min(output.shape[1], col_off + scene.shape[1])

            # Calculate corresponding scene slice
            scene_row_start = max(0, -row_off)
            scene_row_end = scene_row_start + (row_end - row_start)
            scene_col_start = max(0, -col_off)
            scene_col_end = scene_col_start + (col_end - col_start)

            out_slice = output[row_start:row_end, col_start:col_end]
            scene_slice = scene[
                scene_row_start:scene_row_end, scene_col_start:scene_col_end
            ]
            valid_slice = valid_mask[
                scene_row_start:scene_row_end, scene_col_start:scene_col_end
            ]

            np.minimum(out_slice, scene_slice, out=out_slice, where=valid_slice)
        else:
            for b in range(scene.shape[0]):
                # Calculate valid slice bounds
                row_start = max(0, row_off)
                row_end = min(output.shape[1], row_off + scene.shape[1])
                col_start = max(0, col_off)
                col_end = min(output.shape[2], col_off + scene.shape[2])

                # Calculate corresponding scene slice
                scene_row_start = max(0, -row_off)
                scene_row_end = scene_row_start + (row_end - row_start)
                scene_col_start = max(0, -col_off)
                scene_col_end = scene_col_start + (col_end - col_start)

                out_slice = output[b, row_start:row_end, col_start:col_end]
                scene_slice = scene[
                    b, scene_row_start:scene_row_end, scene_col_start:scene_col_end
                ]
                band_valid = valid_mask if valid_mask.ndim == 2 else valid_mask[b]
                valid_slice = band_valid[
                    scene_row_start:scene_row_end, scene_col_start:scene_col_end
                ]

                np.minimum(out_slice, scene_slice, out=out_slice, where=valid_slice)

    # Replace inf with nodata
    output[output == np.inf] = nodata
    return output


def _mean_merge(
    output: np.ndarray,
    scenes: list[np.ndarray],
    transforms: list[Affine],
    out_transform: Affine,
    nodata: float,
) -> np.ndarray:
    """Merge using mean values."""
    # Initialize sum and count arrays
    if output.ndim == 2:
        sum_array = np.zeros_like(output)
        count_array = np.zeros_like(output, dtype=int)
    else:
        sum_array = np.zeros_like(output)
        count_array = np.zeros(output.shape[1:], dtype=int)

    # Accumulate values
    for scene, transform in zip(scenes, transforms):
        col_off = int((transform.c - out_transform.c) / out_transform.a)
        row_off = int((transform.f - out_transform.f) / out_transform.e)

        if np.isnan(nodata):
            valid_mask = ~np.isnan(scene)
        else:
            valid_mask = scene != nodata

        if scene.ndim == 2:
            # Calculate valid slice bounds
            row_start = max(0, row_off)
            row_end = min(output.shape[0], row_off + scene.shape[0])
            col_start = max(0, col_off)
            col_end = min(output.shape[1], col_off + scene.shape[1])

            # Calculate corresponding scene slice
            scene_row_start = max(0, -row_off)
            scene_row_end = scene_row_start + (row_end - row_start)
            scene_col_start = max(0, -col_off)
            scene_col_end = scene_col_start + (col_end - col_start)

            sum_slice = sum_array[row_start:row_end, col_start:col_end]
            count_slice = count_array[row_start:row_end, col_start:col_end]
            scene_slice = scene[
                scene_row_start:scene_row_end, scene_col_start:scene_col_end
            ]
            valid_slice = valid_mask[
                scene_row_start:scene_row_end, scene_col_start:scene_col_end
            ]

            sum_slice[valid_slice] += scene_slice[valid_slice]
            count_slice[valid_slice] += 1
        else:
            # Calculate valid slice bounds
            row_start = max(0, row_off)
            row_end = min(output.shape[1], row_off + scene.shape[1])
            col_start = max(0, col_off)
            col_end = min(output.shape[2], col_off + scene.shape[2])

            # Calculate corresponding scene slice
            scene_row_start = max(0, -row_off)
            scene_row_end = scene_row_start + (row_end - row_start)
            scene_col_start = max(0, -col_off)
            scene_col_end = scene_col_start + (col_end - col_start)

            for b in range(scene.shape[0]):
                sum_slice = sum_array[b, row_start:row_end, col_start:col_end]
                scene_slice = scene[
                    b, scene_row_start:scene_row_end, scene_col_start:scene_col_end
                ]
                band_valid = valid_mask if valid_mask.ndim == 2 else valid_mask[b]
                valid_slice = band_valid[
                    scene_row_start:scene_row_end, scene_col_start:scene_col_end
                ]
                sum_slice[valid_slice] += scene_slice[valid_slice]

            count_slice = count_array[row_start:row_end, col_start:col_end]
            if valid_mask.ndim == 2:
                valid_slice = valid_mask[
                    scene_row_start:scene_row_end, scene_col_start:scene_col_end
                ]
                count_slice[valid_slice] += 1
            else:
                valid_any = np.any(
                    valid_mask[
                        :, scene_row_start:scene_row_end, scene_col_start:scene_col_end
                    ],
                    axis=0,
                )
                count_slice[valid_any] += 1

    # Calculate mean
    if output.ndim == 2:
        output = np.where(count_array > 0, sum_array / count_array, nodata)
    else:
        for b in range(output.shape[0]):
            output[b] = np.where(count_array > 0, sum_array[b] / count_array, nodata)

    return output


def select_by_date_range(
    scenes: list[np.ndarray],
    dates: list[datetime],
    metadata: list[dict[str, Any]] | None = None,
    target_date: datetime | None = None,
    date_range: tuple[datetime, datetime] | None = None,
    max_days_from_target: int | None = None,
    cloud_threshold: float | None = None,
    max_scenes: int | None = None,
) -> tuple[list[np.ndarray], list[datetime], list[int]]:
    """Select scenes based on date and quality criteria.

    This function filters and selects satellite scenes based on temporal
    criteria and quality metrics. It replaces the DateRangeSelector
    component with a simpler functional approach.

    Args:
        scenes: List of scene arrays
        dates: List of acquisition dates for each scene
        metadata: Optional list of metadata dicts containing 'cloud_cover' etc.
        target_date: Target acquisition date for sorting by proximity
        date_range: Explicit (start, end) date range to filter scenes
        max_days_from_target: Maximum days from target date
        cloud_threshold: Maximum cloud cover percentage (0-100)
        max_scenes: Maximum number of scenes to return

    Returns:
        Tuple of (selected_scenes, selected_dates, selected_indices)

    Example:
        >>> scenes = [np.ones((100, 100)) for _ in range(10)]
        >>> dates = [datetime(2023, 1, i) for i in range(1, 11)]
        >>> target = datetime(2023, 1, 5)
        >>> selected, sel_dates, indices = select_by_date_range(
        ...     scenes, dates, target_date=target, max_days_from_target=2
        ... )
        >>> len(selected)  # Should select days 3-7
        5
    """
    if not scenes or not dates:
        return [], [], []

    if len(scenes) != len(dates):
        raise ValueError("Number of scenes must match number of dates")

    # Start with all scene indices
    candidates = list(range(len(scenes)))

    # Apply date range filter
    if date_range:
        candidates = [
            i for i in candidates if date_range[0] <= dates[i] <= date_range[1]
        ]

    # Apply target date filter
    if target_date and max_days_from_target:
        candidates = [
            i
            for i in candidates
            if abs((dates[i] - target_date).days) <= max_days_from_target
        ]

    # Apply cloud cover filter
    if cloud_threshold is not None and metadata:
        candidates = [
            i
            for i in candidates
            if i < len(metadata)
            and metadata[i].get("cloud_cover", 100) <= cloud_threshold
        ]

    # Sort by proximity to target date or by date
    if target_date:
        candidates.sort(key=lambda i: abs((dates[i] - target_date).days))
    else:
        candidates.sort(key=lambda i: dates[i])

    # Apply maximum scenes limit
    if max_scenes and len(candidates) > max_scenes:
        candidates = candidates[:max_scenes]

    # Return selected scenes, dates, and indices
    selected_scenes = [scenes[i] for i in candidates]
    selected_dates = [dates[i] for i in candidates]

    return selected_scenes, selected_dates, candidates


def composite_with_quality(
    data_stack: np.ndarray,
    quality_scores: np.ndarray,
    method: str = "weighted_mean",
    quality_threshold: float = 0.0,
) -> np.ndarray:
    """Create composite using quality scores for weighting or selection.

    This function creates high-quality composites by using quality metrics
    to weight or select the best pixels from multiple observations.

    Args:
        data_stack: 3D array with shape (time, height, width)
        quality_scores: Quality scores with same shape as data_stack (0=worst, 1=best)
        method: Compositing method ('weighted_mean', 'best_quality', 'threshold_mean')
        quality_threshold: Minimum quality score to include pixel (for 'threshold_mean')

    Returns:
        2D composite array (height, width)

    Example:
        >>> data = np.random.rand(5, 100, 100)
        >>> quality = np.random.rand(5, 100, 100)
        >>> composite = composite_with_quality(data, quality, method='best_quality')
    """
    if data_stack.shape != quality_scores.shape:
        raise ValueError("Data and quality scores must have the same shape")

    if method == "weighted_mean":
        # Weight by quality scores
        valid_mask = ~np.isnan(data_stack) & (quality_scores > 0)
        weights = quality_scores * valid_mask
        weight_sum = np.sum(weights, axis=0)
        weight_sum[weight_sum == 0] = 1  # Avoid division by zero

        weighted_data = data_stack * weights
        composite = np.sum(weighted_data, axis=0) / weight_sum
        composite[weight_sum == 0] = np.nan

    elif method == "best_quality":
        # Select pixel from date with highest quality
        composite = np.full(data_stack.shape[1:], np.nan)

        for i in range(data_stack.shape[1]):
            for j in range(data_stack.shape[2]):
                valid_mask = ~np.isnan(data_stack[:, i, j])
                if np.any(valid_mask):
                    valid_quality = quality_scores[:, i, j].copy()
                    valid_quality[~valid_mask] = -np.inf
                    best_idx = np.argmax(valid_quality)
                    if valid_quality[best_idx] > -np.inf:
                        composite[i, j] = data_stack[best_idx, i, j]

    elif method == "threshold_mean":
        # Mean of pixels above quality threshold
        mask = (quality_scores >= quality_threshold) & ~np.isnan(data_stack)
        masked_data = np.ma.masked_where(~mask, data_stack)
        composite = np.ma.mean(masked_data, axis=0).filled(np.nan)

    else:
        raise ValueError(f"Unknown method: {method}")

    return composite


def compute_statistics(
    data: np.ndarray,
    mask: np.ndarray | None = None,
    percentiles: list[float] | None = None,
) -> dict[str, float]:
    """Compute statistical summary of data.

    This replaces the MaskStatistics component with a simple function
    that computes common statistics for data arrays.

    Args:
        data: Input data array (2D or 3D)
        mask: Optional boolean mask where True indicates valid pixels
        percentiles: Optional list of percentiles to compute (0-100)

    Returns:
        Dictionary of statistics including min, max, mean, std, etc.

    Example:
        >>> data = np.random.rand(100, 100)
        >>> stats = compute_statistics(data, percentiles=[25, 50, 75])
        >>> print(f"Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
    """
    # Apply mask if provided
    if mask is not None:
        valid_data = data[mask]
    else:
        # Use non-NaN values
        valid_data = data[~np.isnan(data)]

    if len(valid_data) == 0:
        # No valid data
        stats = {
            "count": 0,
            "valid_fraction": 0.0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "variance": np.nan,
        }
        if percentiles:
            for p in percentiles:
                stats[f"p{int(p)}"] = np.nan
        return stats

    # Compute statistics
    stats = {
        "count": len(valid_data),
        "valid_fraction": len(valid_data) / data.size,
        "min": float(np.min(valid_data)),
        "max": float(np.max(valid_data)),
        "mean": float(np.mean(valid_data)),
        "median": float(np.median(valid_data)),
        "std": float(np.std(valid_data)),
        "variance": float(np.var(valid_data)),
    }

    # Add percentiles if requested
    if percentiles:
        for p in percentiles:
            stats[f"p{int(p)}"] = float(np.percentile(valid_data, p))

    return stats


def apply_mask(
    data: np.ndarray, mask: np.ndarray, fill_value: float = np.nan
) -> np.ndarray:
    """Apply a mask to data, filling masked pixels with fill_value.

    Args:
        data: Data array to mask
        mask: Mask array where True indicates valid pixels, False indicates pixels to mask
        fill_value: Value to use for masked pixels

    Returns:
        Masked data array

    Example:
        >>> data = np.array([[1, 2], [3, 4]])
        >>> mask = np.array([[True, False], [True, True]])
        >>> masked = apply_mask(data, mask, fill_value=0)
        >>> masked
        array([[1, 0],
               [3, 4]])
    """
    result = data.copy()
    result[~mask] = fill_value
    return result


# VNP46A2 (VIIRS) tile utilities
def coordinates_to_tile_id(lat: float, lon: float) -> tuple[int, int]:
    """Convert lat/lon coordinates to VIIRS tile indices.
    
    Args:
        lat: Latitude (-90 to 90)
        lon: Longitude (-180 to 180)
        
    Returns:
        Tuple of (vertical, horizontal) tile indices
    """
    # Convert to 0–360/0–180 range
    v = int(((-lat + 90.0) / 180.0) * 18)
    h = int(((lon + 180.0) / 360.0) * 36)
    return v, h


def get_tiles_for_bbox(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> list[tuple[int, int]]:
    """Get all VIIRS tile indices that intersect a bounding box.
    
    Args:
        lat1, lon1: First corner coordinates
        lat2, lon2: Second corner coordinates
        
    Returns:
        List of (vertical, horizontal) tile index tuples
    """
    # Get min/max bounds
    min_lat = min(lat1, lat2)
    max_lat = max(lat1, lat2)
    min_lon = min(lon1, lon2)
    max_lon = max(lon1, lon2)
    
    # Get tile indices for all corners
    v1, h1 = coordinates_to_tile_id(max_lat, min_lon)  # NW corner
    v2, h2 = coordinates_to_tile_id(min_lat, max_lon)  # SE corner
    
    # Generate all tiles in the rectangular region
    tiles = []
    for v in range(min(v1, v2), max(v1, v2) + 1):
        for h in range(min(h1, h2), max(h1, h2) + 1):
            # Handle wraparound for horizontal tiles (0-35)
            h_wrapped = h % 36
            tiles.append((v, h_wrapped))
    
    return tiles


def download_vnp46a2_h5(
    earthdata_token: str,
    year: int,
    month: int,
    day: int,
    vertical: int,
    horizontal: int,
    target_filepath: str,
) -> None:
    """Download VNP46A2 HDF5 file from NASA for given date and tile.
    
    Args:
        earthdata_token: NASA Earthdata authentication token
        year: Year (e.g., 2023)
        month: Month (1-12)
        day: Day of month
        vertical: Vertical tile index
        horizontal: Horizontal tile index
        target_filepath: Where to save the downloaded file
        
    Raises:
        RuntimeError: If download fails or product not found
    """
    import requests
    import shutil
    from datetime import date as dt_date
    
    tile = f"h{horizontal:02d}v{vertical:02d}"
    day_of_year = dt_date(year, month, day).timetuple().tm_yday
    
    api = (
        f"https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/details/allData/5000/"
        f"VNP46A2/{year}/{day_of_year}"
    )
    
    resp = requests.get(api)
    resp.raise_for_status()
    items = resp.json().get("content", [])
    
    link = None
    target_filename_part = f"VNP46A2.A{year}{day_of_year:03d}.{tile}"
    for item in items:
        if item.get("name", "").startswith(target_filename_part):
            link = item.get("downloadsLink")
            break
    
    if not link:
        raise RuntimeError(
            f"No VNP46A2 product found for tile {tile} on {year}-{day_of_year:03d}"
        )
    
    session = requests.Session()
    if earthdata_token:
        session.headers.update({"Authorization": f"Bearer {earthdata_token}"})
    else:
        raise RuntimeError(
            "Earthdata token not found. Set EARTHDATA_TOKEN environment variable."
        )
    
    r = session.get(link, stream=True, timeout=300)
    r.raise_for_status()
    
    # Ensure parent directory exists
    parent_dir = os.path.dirname(target_filepath)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    
    # Remove existing file/directory if present
    if os.path.exists(target_filepath):
        if os.path.isdir(target_filepath):
            shutil.rmtree(target_filepath)
        else:
            os.remove(target_filepath)
    
    with open(target_filepath, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def convert_vnp46a2_hdf_to_tiff(
    h5_file: str, output_file: str, layer: int = 2
) -> None:
    """Convert VNP46A2 HDF5 file to georeferenced GeoTIFF.
    
    Args:
        h5_file: Path to input HDF5 file
        output_file: Path for output GeoTIFF
        layer: HDF5 layer index (default 2 for DNB_At_Sensor_Radiance_500m)
        
    Note:
        This function wraps the existing convert_VNP46A2_HDF2TIFF function
        which uses layer index 2 by default.
    """
    # Import the existing implementation
    try:
        from .vnp_rasterio import convert_VNP46A2_HDF2TIFF
    except ImportError:
        from .vnp import convert_VNP46A2_HDF2TIFF
    
    # The existing function uses layer 2 by default, which matches our default
    convert_VNP46A2_HDF2TIFF(h5_file, output_file)


# Landsat WRS-2 utilities
_wrs_polygons_cache = None

def load_wrs_shapefile(shapefile_path: str | None = None) -> list[tuple[Any, int, int]]:
    """Load WRS-2 shapefile and return list of (polygon, path, row) tuples.
    
    Args:
        shapefile_path: Path to WRS2 descending shapefile (uses default if None)
        
    Returns:
        List of (polygon, path, row) tuples for all WRS-2 tiles
        
    Note:
        This function caches the result to avoid reloading the shapefile
    """
    global _wrs_polygons_cache
    
    if _wrs_polygons_cache is not None:
        return _wrs_polygons_cache
    
    # Try Fiona first, fall back to OGR
    try:
        import fiona
        import shapely.geometry
        from importlib.resources import files
        
        if shapefile_path is None:
            shapefile_path = str(
                files("blackmarble").joinpath("data", "WRS2_descending_0", "WRS2_descending.shp")
            )
        
        polygons = []
        with fiona.open(shapefile_path) as src:
            for feature in src:
                path = feature["properties"]["PATH"]
                row = feature["properties"]["ROW"]
                shape = shapely.geometry.shape(feature["geometry"])
                if isinstance(shape, shapely.geometry.Polygon):
                    polygons.append((shape, path, row))
        
        _wrs_polygons_cache = polygons
        return polygons
        
    except ImportError:
        # Fall back to OGR
        try:
            from osgeo import ogr
            import shapely.geometry
            from importlib.resources import files
            
            if shapefile_path is None:
                shapefile_path = str(
                    files("blackmarble").joinpath("data", "WRS2_descending_0", "WRS2_descending.shp")
                )
            
            shapefile = ogr.Open(shapefile_path)
            layer = shapefile.GetLayer(0)
            polygons = []
            
            for i in range(layer.GetFeatureCount()):
                feat = layer.GetFeature(i)
                path = feat["PATH"]
                row = feat["ROW"]
                geom = feat.GetGeometryRef()
                from shapely import wkt
                shape = wkt.loads(geom.ExportToWkt())
                if isinstance(shape, shapely.geometry.Polygon):
                    polygons.append((shape, path, row))
            
            _wrs_polygons_cache = polygons
            return polygons
            
        except ImportError:
            raise ImportError("Neither Fiona nor GDAL/OGR is available for reading shapefiles")


def get_wrs_path_row(lat: float, lon: float) -> list[dict[str, int]]:
    """Get Landsat WRS-2 path/row for given coordinates.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        List of dicts with 'path' and 'row' keys for tiles containing the point
    """
    import shapely.geometry
    
    polygons = load_wrs_shapefile()
    pt = shapely.geometry.Point(lon, lat)
    res = []
    
    for poly, path, row in polygons:
        if pt.within(poly):
            res.append({"path": path, "row": row})
    
    return res


# Directory cache for Landsat S3 lookups
_landsat_dir_cache = {}
_LANDSAT_MAX_CACHE_SIZE = 100

def find_landsat_directory(
    year: int, month: int, day: int, path: str, row: str
) -> str | None:
    """Find Landsat directory on USGS S3 for given date and WRS tile.
    
    Args:
        year: Year
        month: Month (1-12)
        day: Day of month
        path: WRS path (e.g., "044")
        row: WRS row (e.g., "033")
        
    Returns:
        S3 prefix for the matching Landsat scene, or None if not found
    """
    import boto3
    from datetime import datetime
    
    yr = str(year)
    mo = str(month).zfill(2)
    da = str(day).zfill(2)
    prefix = f"collection02/level-2/standard/oli-tirs/{yr}/{path}/{row}/"
    
    # Check cache first
    if prefix in _landsat_dir_cache:
        directories = _landsat_dir_cache[prefix]
    else:
        # List directories from S3
        session = boto3.Session()
        s3 = session.client("s3")
        resp = s3.list_objects_v2(
            Bucket="usgs-landsat",
            Prefix=prefix,
            RequestPayer="requester",
            Delimiter="/",
        )
        
        directories = []
        common_prefixes = resp.get("CommonPrefixes", [])
        for prefix_entry in common_prefixes:
            if isinstance(prefix_entry, dict) and "Prefix" in prefix_entry:
                directories.append({"Prefix": prefix_entry["Prefix"]})
        
        # Update cache with size limit
        if len(_landsat_dir_cache) >= _LANDSAT_MAX_CACHE_SIZE:
            # Remove oldest entry
            _landsat_dir_cache.pop(next(iter(_landsat_dir_cache)))
        _landsat_dir_cache[prefix] = directories
    
    # Find directory containing the target date
    target = datetime.strptime(yr + mo + da, "%Y%m%d")
    selected = None
    
    for entry in directories:
        dr = entry["Prefix"].split("/")[-2].split("_")[3:5]
        start = datetime.strptime(dr[0], "%Y%m%d")
        end = datetime.strptime(dr[1], "%Y%m%d")
        if start <= target <= end:
            selected = entry["Prefix"]
            break
    
    return selected


def download_landsat_band(
    directory: str, band: int | str, output_path: str
) -> None:
    """Download a Landsat band or QA file from USGS S3.
    
    Args:
        directory: S3 directory prefix for the Landsat scene
        band: Band number (1-11) or QA type ("QA_PIXEL", "QA_RADSAT", "QA_AEROSOL")
        output_path: Local path to save the downloaded file
    """
    import boto3
    
    scene = directory.split("/")[-2]
    
    # Determine S3 key based on band type
    if band in ("QA_PIXEL", "QA_RADSAT", "QA_AEROSOL"):
        suffix_map = {
            "QA_PIXEL": "_QA_PIXEL.TIF",
            "QA_RADSAT": "_QA_RADSAT.TIF",
            "QA_AEROSOL": "_SR_QA_AEROSOL.TIF",
        }
        suffix = suffix_map[str(band)]
        key = directory + scene + suffix
    else:
        key = f"{directory}{scene}_SR_B{band}.TIF"
    
    # Download from S3
    session = boto3.Session()
    s3 = session.client("s3")
    
    # Ensure output directory exists
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    
    s3.download_file(
        "usgs-landsat", key, output_path, 
        ExtraArgs={"RequestPayer": "requester"}
    )


# Cloud-Optimized GeoTIFF (COG) output functions
def create_cog(
    data: np.ndarray,
    output_path: str,
    crs: str | Any,
    transform: Affine | tuple[float, ...],
    nodata: float | None = None,
    profile_name: str = "deflate",
    blocksize: int = 512,
    overview_resampling: str = "bilinear",
    tags: dict[str, Any] | None = None,
) -> str:
    """Create a Cloud-Optimized GeoTIFF from array data.
    
    Args:
        data: Input array (2D for single band, 3D for multi-band with band dim first)
        output_path: Path for output COG file
        crs: Coordinate reference system (e.g., "EPSG:4326")
        transform: Affine transform or GDAL geotransform tuple
        nodata: NoData value
        profile_name: COG profile ('deflate', 'lzw', 'webp', 'zstd')
        blocksize: Internal tile size
        overview_resampling: Method for overview generation
        tags: Optional metadata tags
        
    Returns:
        Path to created COG file
    """
    from rio_cogeo.profiles import cog_profiles
    from rio_cogeo.cogeo import cog_translate
    from pathlib import Path
    
    output_path = str(Path(output_path))
    
    # Convert transform if needed
    if isinstance(transform, tuple):
        transform = Affine.from_gdal(*transform)
    
    # Determine band count
    if data.ndim == 2:
        count = 1
        height, width = data.shape
    elif data.ndim == 3:
        count, height, width = data.shape
    else:
        raise ValueError(f"Data must be 2D or 3D, got shape {data.shape}")
    
    # Create profile
    profile = {
        "driver": "GTiff",
        "dtype": data.dtype.name,
        "width": width,
        "height": height,
        "count": count,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    
    # Get COG profile
    cog_profile = cog_profiles.get(profile_name, cog_profiles["deflate"])
    
    # COG configuration
    config = {
        "GDAL_NUM_THREADS": "ALL_CPUS",
        "GDAL_TIFF_INTERNAL_MASK": "YES",
        "GDAL_TIFF_OVR_BLOCKSIZE": str(blocksize),
    }
    
    # Calculate overview levels
    max_dim = max(height, width)
    overview_level = 0
    dim = max_dim
    while dim > blocksize:
        overview_level += 1
        dim //= 2
    
    # Create COG
    with MemoryFile() as memfile:
        with memfile.open(**profile) as mem:
            if count == 1:
                mem.write(data, 1)
            else:
                for i in range(count):
                    mem.write(data[i], i + 1)
            
            if tags:
                mem.update_tags(**tags)
        
        # Translate to COG
        with memfile.open() as src:
            cog_translate(
                src,
                output_path,
                cog_profile,
                config=config,
                overview_level=overview_level,
                overview_resampling=overview_resampling,  # type: ignore
                web_optimized=True,
                in_memory=True,
            )
    
    return output_path


def save_geotiff(
    data: np.ndarray,
    output_path: str,
    crs: str | Any,
    transform: Affine | tuple[float, ...],
    nodata: float | None = None,
    compress: str = "deflate",
    tiled: bool = True,
    blockxsize: int = 256,
    blockysize: int = 256,
) -> str:
    """Save array as a standard GeoTIFF (not COG).
    
    Args:
        data: Input array (2D for single band, 3D for multi-band)
        output_path: Output file path
        crs: Coordinate reference system
        transform: Affine transform or GDAL geotransform tuple
        nodata: NoData value
        compress: Compression method
        tiled: Use internal tiling
        blockxsize: Tile width
        blockysize: Tile height
        
    Returns:
        Path to saved file
    """
    from pathlib import Path
    
    output_path = str(Path(output_path))
    
    # Convert transform if needed
    if isinstance(transform, tuple):
        transform = Affine.from_gdal(*transform)
    
    # Determine dimensions
    if data.ndim == 2:
        count = 1
        height, width = data.shape
    elif data.ndim == 3:
        count, height, width = data.shape
    else:
        raise ValueError(f"Data must be 2D or 3D, got shape {data.shape}")
    
    # Create profile
    profile = {
        "driver": "GTiff",
        "dtype": data.dtype.name,
        "width": width,
        "height": height,
        "count": count,
        "crs": crs,
        "transform": transform,
        "compress": compress,
        "nodata": nodata,
    }
    
    if tiled:
        profile.update({
            "tiled": True,
            "blockxsize": blockxsize,
            "blockysize": blockysize,
        })
    
    # Write file
    with rasterio.open(output_path, "w", **profile) as dst:
        if count == 1:
            dst.write(data, 1)
        else:
            for i in range(count):
                dst.write(data[i], i + 1)
    
    return output_path


def generate_metadata(
    data: np.ndarray,
    bounds: tuple[float, float, float, float],
    crs: str,
    datetime_str: str | None = None,
    title: str = "Black Marble Output",
    description: str = "",
) -> dict[str, Any]:
    """Generate metadata for output products.
    
    Args:
        data: Input data array
        bounds: Bounding box (minx, miny, maxx, maxy)
        crs: Coordinate reference system
        datetime_str: ISO datetime string
        title: Product title
        description: Product description
        
    Returns:
        STAC-like metadata dictionary
    """
    # Calculate statistics
    valid_data = data[~np.isnan(data)]
    stats = {}
    
    if len(valid_data) > 0:
        stats = {
            "min": float(np.min(valid_data)),
            "max": float(np.max(valid_data)),
            "mean": float(np.mean(valid_data)),
            "stdev": float(np.std(valid_data)),
            "valid_percent": 100.0 * len(valid_data) / data.size,
        }
    
    # Create GeoJSON geometry
    minx, miny, maxx, maxy = bounds
    geometry = {
        "type": "Polygon",
        "coordinates": [[
            [minx, miny],
            [maxx, miny],
            [maxx, maxy],
            [minx, maxy],
            [minx, miny],
        ]],
    }
    
    # Create feature
    metadata = {
        "type": "Feature",
        "geometry": geometry,
        "properties": {
            "datetime": datetime_str,
            "title": title,
            "description": description,
            "proj:epsg": int(crs.split(":")[1]) if ":" in str(crs) else None,
            "statistics": stats,
        },
    }
    
    return metadata
