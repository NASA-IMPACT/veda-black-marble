"""Export module for Black Marble pipeline.

This module provides functions for generating various output formats,
including Cloud-Optimized GeoTIFFs, metadata, and validation utilities.
"""

from .cog import (
    batch_export_cogs,
    create_cog,
    create_overviews,
    export_multi_band,
    export_single_band,
    optimize_for_web,
)

from .metadata import (
    bundle_outputs,
    create_metadata,
    create_processing_report,
    create_stac_item,
    export_metadata_sidecar,
    export_processing_lineage,
    generate_statistics,
)

from .validate import (
    check_output_quality,
    compare_with_reference,
    generate_validation_report,
    validate_cog_structure,
    validate_data_range,
    validate_export_config,
    validate_for_cog,
    validate_spatial_alignment,
)

__all__ = [
    # COG functions
    "batch_export_cogs",
    "create_cog",
    "create_overviews",
    "export_multi_band",
    "export_single_band",
    "optimize_for_web",
    # Metadata functions
    "bundle_outputs",
    "create_metadata",
    "create_processing_report",
    "create_stac_item",
    "export_metadata_sidecar",
    "export_processing_lineage",
    "generate_statistics",
    # Validation functions
    "check_output_quality",
    "compare_with_reference",
    "generate_validation_report",
    "validate_cog_structure",
    "validate_data_range",
    "validate_export_config",
    "validate_for_cog",
    "validate_spatial_alignment",
]