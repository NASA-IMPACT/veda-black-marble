"""I/O adapters for external services.

This package provides abstraction layers for external services,
enabling better testability and flexibility in the Black Marble pipeline.
"""

from .base import StorageAdapter, APIAdapter, RasterAdapter
from .s3 import S3Adapter
from .nasa_api import NASAEarthdataAdapter
from .raster_io import GDALRasterAdapter

__all__ = [
    "StorageAdapter",
    "APIAdapter",
    "RasterAdapter",
    "S3Adapter",
    "NASAEarthdataAdapter",
    "GDALRasterAdapter",
]
