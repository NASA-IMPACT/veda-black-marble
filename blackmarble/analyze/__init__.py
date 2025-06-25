"""Analysis module for Black Marble pipeline.

This module provides analytical functions for:
- Spectral indices calculation (NDVI, NDWI, NDUI)
- Temporal compositing and time series analysis
- Quality assessment and validation
- Statistical analysis and metrics
"""

from .indices import (
    calculate_ndui,
    calculate_ndvi,
    calculate_ndwi,
)

from .quality import (
    assess_data_quality,
    calculate_quality_weighted_composite,
    compute_confidence_scores,
    compute_quality_metrics,
    create_composite_quality_mask,
    flag_suspicious_values,
    generate_quality_report,
    identify_processing_artifacts,
)

from .statistics import (
    analyze_spatial_autocorrelation,
    calculate_coverage_statistics,
    calculate_temporal_statistics,
    calculate_uncertainty_metrics,
    compute_comparison_metrics,
    compute_zonal_statistics,
    generate_histogram_analysis,
    generate_quality_metrics_summary,
)

from .temporal import (
    composite_with_quality_weighting,
    compute_phenology_metrics,
    compute_temporal_statistics,
    create_landsat_median_composite,
    create_temporal_composite,
    detect_change,
    fill_temporal_gaps,
    merge_scenes,
    select_by_date_range,
)

__all__ = [
    # Indices functions
    "calculate_ndui",
    "calculate_ndvi",
    "calculate_ndwi",
    # Quality functions
    "assess_data_quality",
    "calculate_quality_weighted_composite",
    "compute_confidence_scores",
    "compute_quality_metrics",
    "create_composite_quality_mask",
    "flag_suspicious_values",
    "generate_quality_report",
    "identify_processing_artifacts",
    # Statistics functions
    "analyze_spatial_autocorrelation",
    "calculate_coverage_statistics",
    "calculate_temporal_statistics",
    "calculate_uncertainty_metrics",
    "compute_comparison_metrics",
    "compute_zonal_statistics",
    "generate_histogram_analysis",
    "generate_quality_metrics_summary",
    # Temporal functions
    "composite_with_quality_weighting",
    "compute_phenology_metrics",
    "compute_temporal_statistics",
    "create_landsat_median_composite",
    "create_temporal_composite",
    "detect_change",
    "fill_temporal_gaps",
    "merge_scenes",
    "select_by_date_range",
]