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
    # COG functions (used by pipeline)
    "create_cog",
    "export_multi_band",
    "export_single_band",
    # Metadata functions (used by pipeline)
    "create_metadata",
    # Validation functions (used by pipeline)
    "validate_for_cog",
]