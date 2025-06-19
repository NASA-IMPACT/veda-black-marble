#!/usr/bin/env python3
import typer
import os
import sys

app = typer.Typer(
    help="Command-line interface for the VEDA Black Marble processing pipeline."
)


@app.command()
def run(
    lat1: float = typer.Option(
        ..., "--lat1", help="Latitude of the first corner (e.g., North)."
    ),
    lon1: float = typer.Option(
        ..., "--lon1", help="Longitude of the first corner (e.g., West)."
    ),
    lat2: float = typer.Option(
        ..., "--lat2", help="Latitude of the second corner (e.g., South)."
    ),
    lon2: float = typer.Option(
        ..., "--lon2", help="Longitude of the second corner (e.g., East)."
    ),
    year: int = typer.Option(..., "--year", help="Target year for data."),
    month: int = typer.Option(..., "--month", help="Target month for data."),
    day: int = typer.Option(..., "--day", help="Target day for data."),
    earthdata_token: str | None = typer.Option(
        None,
        "--earthdata-token",
        "-t",
        help="NASA Earthdata login token. If not provided, reads from EARTHDATA_TOKEN environment variable.",
        envvar="EARTHDATA_TOKEN",  # Automatically read from env var if not provided
    ),
    output_dir: str = typer.Option(
        "outputs",
        "--output-dir",
        "-o",
        help="Base directory for all output and temporary files.",
    ),
    upload_bucket: str | None = typer.Option(
        None,
        "--upload-bucket",
        "-b",
        help="S3 bucket name for uploading the final COG output.",
    ),
    upload_key: str | None = typer.Option(
        None,
        "--upload-key",
        "-k",
        help="S3 key (path/filename) for uploading the final COG output.",
    ),
):
    """
    Runs the Black Marble processing pipeline for a given area and date.
    """
    from blackmarble.pipeline import run_pipeline

    if not earthdata_token:
        typer.echo(
            "Error: Earthdata token is required. Provide via --earthdata-token or EARTHDATA_TOKEN environment variable.",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(f"Starting Black Marble pipeline for bounding box:")
    typer.echo(f"  Lat1: {lat1}, Lon1: {lon1}")
    typer.echo(f"  Lat2: {lat2}, Lon2: {lon2}")
    typer.echo(f"Date: {year}-{month:02d}-{day:02d}")
    typer.echo(f"Output Directory: {output_dir}")
    if upload_bucket and upload_key:
        typer.echo(f"Upload Target: s3://{upload_bucket}/{upload_key}")
    else:
        typer.echo("S3 Upload: Disabled")

    try:
        final_output_path = run_pipeline(
            lat1=lat1,
            lon1=lon1,
            lat2=lat2,
            lon2=lon2,
            year=year,
            month=month,
            day=day,
            earthdata_token=earthdata_token,
            output_dir=output_dir,
            upload_bucket=upload_bucket,
            upload_key=upload_key,
        )
        typer.echo(f"\nPipeline completed successfully!")
        typer.echo(f"Final output generated at: {final_output_path}")
    except Exception as e:
        typer.echo(
            f"Pipeline failed with error: {str(e)} - Please check inputs and logs.",
            err=True,
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
