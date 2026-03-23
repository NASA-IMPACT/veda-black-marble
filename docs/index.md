# VEDA Black Marble

Nighttime lights processing pipeline for NASA VEDA, combining VIIRS nighttime lights with Landsat data to create urban-focused imagery.

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/NASA-IMPACT/veda-black-marble.git
cd veda-black-marble
pip install -e .
```

## Quick Start

```bash
export EARTHDATA_TOKEN="your-token-here"

blackmarble \
  --bbox "-122.55,37.69,-122.32,37.81" \
  --date 2023-06-15 \
  --output-path san_francisco_lights.tif
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--bbox` | required | Bounding box for processing (WGS84) |
| `--date` | required | Target date for processing |
| `--output-path`, `-o` | `black_marble_output.tif` | Output COG file path |
| `--data-dir` | `./data` | Directory to store/cache data |
| `--config`, `-c` | `default` | Preset: `default`, `high_quality`, or `fast` |
| `--save-diagnostics`, `-d` | False | Save intermediate outputs to `{data-dir}/diagnostics/` |
| `--wgs84`, `-w` | False | Also export EPSG:4326 version |
| `--earthdata-token`, `-t` | ENV_VAR | NASA Earthdata token (or use EARTHDATA_TOKEN) |
| `--log-level`, `-l` | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

Run `blackmarble --help` for complete options.

## Documentation Map

- Pipeline overview: [Pipeline Steps](pipeline-steps/README.md)
- Algorithm release notes: [Release Notes](RELEASE_NOTES.md)
- API docs: [API Reference](api-reference.md)
- Source repository: [NASA-IMPACT/veda-black-marble](https://github.com/NASA-IMPACT/veda-black-marble)
