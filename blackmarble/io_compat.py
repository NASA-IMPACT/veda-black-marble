"""
Compatibility module for transitioning from GDAL to rasterio-based save_geotiff_single.
This module provides a controlled way to switch between implementations.
"""

import logging
import os
from typing import Callable

logger = logging.getLogger(__name__)

# Feature flag for using rasterio implementation
USE_RASTERIO_WRITER = (
    os.environ.get("BM_USE_RASTERIO_WRITER", "false").lower() == "true"
)


def get_save_geotiff_single_impl() -> Callable:
    """
    Returns the appropriate save_geotiff_single implementation based on configuration.

    Returns:
        Callable: Either the GDAL or rasterio implementation of save_geotiff_single
    """
    if USE_RASTERIO_WRITER:
        try:
            from .io_rasterio import save_geotiff_single

            logger.info("Using rasterio-based save_geotiff_single implementation")
            return save_geotiff_single
        except ImportError as e:
            logger.warning(f"Failed to import rasterio implementation: {e}")
            logger.warning("Falling back to GDAL implementation")

    # Default to GDAL implementation
    from .io import save_geotiff_single

    logger.info("Using GDAL-based save_geotiff_single implementation")
    return save_geotiff_single


# For backwards compatibility, provide a direct function
def save_geotiff_single_compat(out_path, array, projection, geotransform):
    """
    Compatibility wrapper that uses the configured implementation.

    This function has the same signature as the original save_geotiff_single
    and will use either GDAL or rasterio based on configuration.
    """
    impl = get_save_geotiff_single_impl()
    return impl(out_path, array, projection, geotransform)
