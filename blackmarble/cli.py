#!/usr/bin/env python3
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path

import typer

from blackmarble.config import PipelineConfig
from blackmarble.pipeline import black_marble_pipeline_sync

app = typer.Typer(help="Command-line interface for the VEDA Black Marble processing pipeline.")


@app.command()
def run(
    lat1: float = typer.Option(..., "--lat1", help="Latitude of the first corner (e.g., North)."),
    lon1: float = typer.Option(..., "--lon1", help="Longitude of the first corner (e.g., West)."),
    lat2: float = typer.Option(..., "--lat2", help="Latitude of the second corner (e.g., South)."),
    lon2: float = typer.Option(..., "--lon2", help="Longitude of the second corner (e.g., East)."),
    year: int = typer.Option(..., "--year", help="Target year for data."),
    month: int = typer.Option(..., "--month", help="Target month for data."),
    day: int = typer.Option(..., "--day", help="Target day for data."),
    earthdata_token: str | None = typer.Option(
        None,
        "--earthdata-token",
        "-t",
        help="NASA Earthdata token for VIIRS downloads. "
        "If not provided, reads from EARTHDATA_TOKEN environment variable.",
        envvar="EARTHDATA_TOKEN",
    ),
    output_path: str = typer.Option(
        "black_marble_output.tif",
        "--output-path",
        "-o",
        help="Output path for the final COG file.",
    ),
    config_preset: str = typer.Option(
        "default",
        "--config",
        "-c",
        help="Configuration preset: 'default', 'high_quality', or 'fast'.",
    ),
    save_diagnostics: bool = typer.Option(
        False,
        "--save-diagnostics",
        "-d",
        help="Save diagnostic intermediate outputs to data/diagnostics/",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level: DEBUG, INFO, WARNING, ERROR",
    ),
):
    """
    Runs the Black Marble processing pipeline for a given area and date.

    This uses the functional API pipeline with features like:
    - Temporal compositing of Landsat scenes
    - VIIRS nighttime lights integration
    - OSM road enhancement
    - Urban index calculation (NDUI)
    - Cloud-optimized GeoTIFF output
    """
    
    # Configure logging
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        typer.echo(f"Error: Invalid log level '{log_level}'", err=True)
        raise typer.Exit(code=1)
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Suppress noisy INFO messages from AWS libraries
    logging.getLogger('aiobotocore').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('boto3').setLevel(logging.WARNING)

    # Handle Earthdata token
    if not earthdata_token:
        typer.echo(
            "Warning: No Earthdata token provided. VIIRS downloads will be skipped.\n"
            "Provide token via --earthdata-token or set EARTHDATA_TOKEN environment variable.",
            err=True,
        )

    # Set as environment variable for compatibility
    if earthdata_token:
        os.environ["EARTHDATA_TOKEN"] = earthdata_token

    # Convert lat/lon corners to bbox (min_lon, min_lat, max_lon, max_lat)
    min_lat = min(lat1, lat2)
    max_lat = max(lat1, lat2)
    min_lon = min(lon1, lon2)
    max_lon = max(lon1, lon2)
    bbox = (min_lon, min_lat, max_lon, max_lat)

    # Create datetime
    try:
        date = datetime(year, month, day)
    except ValueError as e:
        typer.echo(f"Error: Invalid date {year}-{month:02d}-{day:02d}: {e}", err=True)
        raise typer.Exit(code=1)

    # Ensure output directory exists
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Create configuration based on preset
    config = None
    if config_preset == "high_quality":
        config = PipelineConfig.for_enhanced_quality(
            bbox=bbox, date=date, output_path=str(output_path_obj)
        )
    elif config_preset == "fast":
        # For fast processing, use basic mode with minimal enhancements
        config = PipelineConfig.for_bm_parity(
            bbox=bbox, date=date, output_path=str(output_path_obj)
        )
        # Disable optional outputs for speed
        config.export.generate_urban_mask = False
        config.export.generate_indices_file = False
        config.export.validate_before_export = False
    elif config_preset == "default":
        config = PipelineConfig.for_bm_parity(
            bbox=bbox, date=date, output_path=str(output_path_obj)
        )
    else:
        typer.echo(
            f"Error: Unknown config preset '{config_preset}'. "
            f"Use 'default', 'high_quality', or 'fast'.",
            err=True,
        )
        raise typer.Exit(code=1)

    # Enable diagnostics if requested
    if save_diagnostics:
        config.save_diagnostics = True
        typer.echo("📊 Diagnostic outputs will be saved to data/diagnostics/")
        # Create diagnostics directory
        os.makedirs("data/diagnostics", exist_ok=True)



    # Configure NTL enhancement - always use multiplicative mode
    config.analysis.ntl_enhancement_mode = "multiplicative"
    typer.echo("🌃 NTL enhancement: Urban field (multiplicative mode)")
    typer.echo(f"   Urban field scales: {config.analysis.urban_field_sigmas} pixels")

    # Display processing info
    typer.echo("\nStarting Black Marble pipeline:")
    typer.echo(f"  Bounding Box: {bbox}")
    typer.echo(f"  Date: {date.strftime('%Y-%m-%d')}")
    typer.echo(f"  Config Preset: {config_preset}")
    typer.echo(f"  Road Rasterization: Fractional (5x5)")
    typer.echo(f"  NTL Enhancement: multiplicative")
    typer.echo(f"  Output Path: {output_path_obj}")

    try:
        typer.echo("\nRunning pipeline...")

        result = black_marble_pipeline_sync(
            bbox=bbox,
            date=date,
            output_path=str(output_path_obj),
            config=config,
            earthdata_token=earthdata_token,
        )

        typer.echo("\n✅ Pipeline completed successfully!")
        typer.echo(f"📁 Output: {result['output_path']}")

        # Display statistics if available
        stats = result.get("statistics", {})
        if stats:
            typer.echo("\n📊 Statistics:")
            if stats.get("urban_area_km2") is not None:
                typer.echo(f"  Urban area: {stats['urban_area_km2']:.2f} km²")
            if stats.get("mean_radiance") is not None:
                typer.echo(f"  Mean radiance: {stats['mean_radiance']:.4f} nW/cm²/sr")

        # Display validation info
        validation = result.get("validation", {})
        if validation and not validation.get("valid", True):
            typer.echo(f"\n⚠️  Validation warnings: {validation.get('errors', [])}")

    except Exception as e:
        typer.echo(f"\n❌ Pipeline failed: {str(e)}", err=True)
        typer.echo(f"Traceback:\n{traceback.format_exc()}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
