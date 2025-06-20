"""Data source implementations using adapters.

This package provides high-level data source abstractions that use
the I/O adapters to fetch and process data from various sources.
"""

from .base import DataSource, DataSourceError
from .landsat import LandsatDataSource
from .viirs import VIIRSDataSource

__all__ = [
    "DataSource",
    "DataSourceError",
    "LandsatDataSource",
    "VIIRSDataSource",
]
