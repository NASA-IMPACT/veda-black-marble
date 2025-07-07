"""Export module for Black Marble pipeline.

This module provides functions for generating various output formats,
including Cloud-Optimized GeoTIFFs, metadata, and validation utilities.
"""

from .cog import (
    create_cog,
    export_multi_band,
    export_single_band,
)

from .metadata import (
    create_metadata,
)

from .validate import (
    validate_for_cog,
)

__all__ = [
    "create_cog",
    "export_multi_band",
    "export_single_band",
    "create_metadata",
    "validate_for_cog",
]
