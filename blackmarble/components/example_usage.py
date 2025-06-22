"""Example usage of quality control components.

This module demonstrates how to use the quality control components
to replace the masking logic in the main pipeline.
"""

import numpy as np
from typing import Any

from .quality import (
    CloudMasker,
    AerosolMasker,
    RadianceSaturationMasker,
    VIIRSQualityMasker,
    QualityCompositor,
    MaskStatistics,
    create_landsat_qa_pipeline,
)
from .base import apply_mask
from ..config import ProcessingConfig


def process_landsat_qa_traditional(
    qa_pixel: np.ndarray, qa_aerosol: np.ndarray, qa_radsat: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Traditional approach using the old mask functions.

    This shows what the current pipeline does.
    """
    from ..masks import mark_cloud, mark_aerosol, mark_radsat

    # Create masks
    cloud_mask = mark_cloud(qa_pixel)
    aerosol_mask = mark_aerosol(qa_aerosol)
    b3_mask, b4_mask, b5_mask = mark_radsat(qa_radsat)

    # Combine cloud and aerosol masks
    combined_mask = cloud_mask * aerosol_mask

    return cloud_mask, combined_mask, (b3_mask, b4_mask, b5_mask)


def process_landsat_qa_components(
    qa_pixel: np.ndarray,
    qa_aerosol: np.ndarray,
    qa_radsat: np.ndarray,
    config: ProcessingConfig | None = None,
) -> dict[str, Any]:
    """New approach using quality control components.

    This demonstrates the cleaner, more modular approach.
    """
    if config is None:
        config = ProcessingConfig()

    # Create individual maskers
    cloud_masker = CloudMasker(
        strategy="conservative",
        dilate_radius=config.landsat.qa_dilation_radius,
        config=config.landsat,
    )
    aerosol_masker = AerosolMasker(threshold="standard", config=config.landsat)
    radsat_masker = RadianceSaturationMasker()

    # Process each QA band
    cloud_result = cloud_masker.process(qa_pixel)
    aerosol_result = aerosol_masker.process(qa_aerosol)
    radsat_result = radsat_masker.process(qa_radsat)  # Returns all 3 bands

    # Create compositor for cloud + aerosol
    compositor = QualityCompositor(
        components=[cloud_masker, aerosol_masker], combination_method="and"
    )

    # Process combined mask
    combined_result = compositor.process(
        {"cloud_masker": qa_pixel, "aerosol_masker": qa_aerosol}
    )

    # Extract individual band masks from radsat
    b3_mask = radsat_result.mask[0]
    b4_mask = radsat_result.mask[1]
    b5_mask = radsat_result.mask[2]

    return {
        "cloud_mask": cloud_result.mask,
        "aerosol_mask": aerosol_result.mask,
        "combined_mask": combined_result.mask,
        "radsat_masks": (b3_mask, b4_mask, b5_mask),
        "statistics": {
            "cloud": cloud_result.statistics,
            "aerosol": aerosol_result.statistics,
            "radsat": radsat_result.statistics,
            "combined": combined_result.statistics,
        },
    }


def apply_masks_to_bands(
    bands: list[np.ndarray],
    qa_masks: dict[str, np.ndarray],
    config: ProcessingConfig | None = None,
) -> list[np.ndarray]:
    """Apply quality masks to Landsat bands.

    This replaces the masking logic in the pipeline.
    """
    if config is None:
        config = ProcessingConfig()

    # Get masks
    combined_mask = qa_masks["combined_mask"]
    b3_radsat, b4_radsat, b5_radsat = qa_masks["radsat_masks"]

    # Apply masks to each band
    b3_masked = apply_mask(bands[0], combined_mask * b3_radsat, fill_value=np.nan)
    b4_masked = apply_mask(bands[1], combined_mask * b4_radsat, fill_value=np.nan)
    b5_masked = apply_mask(bands[2], combined_mask * b5_radsat, fill_value=np.nan)

    return [b3_masked, b4_masked, b5_masked]


def process_viirs_quality(
    quality_flags: np.ndarray, config: ProcessingConfig | None = None
) -> dict[str, Any]:
    """Process VIIRS quality flags using components.

    This demonstrates quality control for nighttime lights.
    """
    if config is None:
        config = ProcessingConfig()

    # Create VIIRS quality masker
    viirs_masker = VIIRSQualityMasker(config=config.viirs)

    # Process quality flags
    result = viirs_masker.process(quality_flags)

    return {
        "mask": result.mask,
        "statistics": result.statistics,
        "metadata": result.metadata,
    }


def create_quality_report(
    qa_results: dict[str, Any], output_path: str | None = None
) -> str:
    """Generate a quality control report.

    This demonstrates the reporting capabilities of the components.
    """
    report_lines = ["Quality Control Report", "=" * 50, ""]

    # Cloud masking statistics
    if "cloud" in qa_results.get("statistics", {}):
        cloud_stats = qa_results["statistics"]["cloud"]
        report_lines.extend(
            [
                "Cloud Masking:",
                f"  Total pixels: {cloud_stats['total_pixels']:,}",
                f"  Valid pixels: {cloud_stats['valid_pixels']:,} ({cloud_stats['valid_percentage']:.1f}%)",
                f"  Categories:",
            ]
        )

        if "categories" in cloud_stats:
            for category, count in cloud_stats["categories"].items():
                report_lines.append(f"    {category}: {count:,}")
        report_lines.append("")

    # Aerosol masking statistics
    if "aerosol" in qa_results.get("statistics", {}):
        aerosol_stats = qa_results["statistics"]["aerosol"]
        report_lines.extend(
            [
                "Aerosol Masking:",
                f"  Total pixels: {aerosol_stats['total_pixels']:,}",
                f"  Valid pixels: {aerosol_stats['valid_pixels']:,} ({aerosol_stats['valid_percentage']:.1f}%)",
                "",
            ]
        )

    # Combined masking statistics
    if "combined" in qa_results.get("statistics", {}):
        combined_stats = qa_results["statistics"]["combined"]
        report_lines.extend(
            [
                "Combined Cloud + Aerosol Masking:",
                f"  Valid pixels: {combined_stats['valid_pixels']:,} ({combined_stats['valid_percentage']:.1f}%)",
                "",
            ]
        )

    # Radiance saturation statistics
    if "radsat" in qa_results.get("statistics", {}):
        radsat_stats = qa_results["statistics"]["radsat"]
        if "band_statistics" in radsat_stats:
            report_lines.append("Radiance Saturation by Band:")
            for band, stats in radsat_stats["band_statistics"].items():
                report_lines.append(
                    f"  {band}: {stats['saturated_pixels']:,} saturated "
                    f"({100 - stats['valid_percentage']:.1f}%)"
                )
            report_lines.append("")

    report_text = "\n".join(report_lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report_text)

    return report_text


# Example integration with the main pipeline
def create_landsat_mask(
    qa_pixel: np.ndarray,
    qa_aerosol: np.ndarray,
    qa_radsat: np.ndarray,
    config: ProcessingConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Drop-in replacement for the masking logic in pipeline.py.

    This function can replace the mark_cloud, mark_aerosol, and mark_radsat
    calls in the main pipeline with a cleaner, more informative approach.

    Returns:
        Tuple of (combined_mask, qa_statistics)
    """
    if config is None:
        config = ProcessingConfig()

    # Use the standard Landsat QA pipeline
    pipeline = create_landsat_qa_pipeline(
        strategy="conservative",
        dilate_clouds=0,  # No dilation by default
        config=config.landsat,
    )

    # Process cloud and aerosol together
    qa_data = {"qa_pixel": qa_pixel, "qa_aerosol": qa_aerosol}

    combined_result = pipeline.process(qa_data)

    # Process radsat separately (it's per-band)
    radsat_masker = RadianceSaturationMasker()
    radsat_result = radsat_masker.process(qa_radsat)

    # Prepare statistics
    statistics = {
        "combined": combined_result.statistics,
        "radsat": radsat_result.statistics,
        "component_stats": combined_result.metadata.get("component_statistics", {})
        if combined_result.metadata
        else {},
    }

    return combined_result.mask, statistics


if __name__ == "__main__":
    # Example usage
    print("Quality Control Components Example")
    print("-" * 50)

    # Create some synthetic QA data
    qa_pixel = np.random.choice([21824, 21952, 54852, 22280], size=(100, 100))
    qa_aerosol = np.random.choice([64, 192, 194, 196, 228], size=(100, 100))
    qa_radsat = np.random.choice([0, 4, 8, 12, 16, 20, 24, 28], size=(100, 100))

    # Process using components
    results = process_landsat_qa_components(qa_pixel, qa_aerosol, qa_radsat)

    # Generate report
    report = create_quality_report(results)
    print(report)

    # Show how to use the drop-in replacement
    print("\nDrop-in Replacement Example:")
    print("-" * 50)

    mask, stats = create_landsat_mask(qa_pixel, qa_aerosol, qa_radsat)
    print(f"Combined mask shape: {mask.shape}")
    print(
        f"Valid pixels: {stats['combined']['valid_pixels']:,} "
        f"({stats['combined']['valid_percentage']:.1f}%)"
    )

    # Show component statistics
    for component, component_stats in stats["component_stats"].items():
        print(f"\n{component}:")
        print(
            f"  Valid: {component_stats['valid_pixels']:,} "
            f"({component_stats['valid_percentage']:.1f}%)"
        )
