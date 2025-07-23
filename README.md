# VEDA Black Marble

Nighttime lights processing pipeline for NASA VEDA, combining VIIRS nighttime lights with Landsat data to create urban-focused imagery.

## Installation

Requires Python 3.11+ and GDAL 3.6+.

```bash
# Clone the repository
git clone https://github.com/NASA-IMPACT/veda-black-marble.git
cd veda-black-marble

# Install in editable mode
pip install -e .
# or with uv
uv pip install -e .
```

## Quick Start

```bash
# Set your NASA Earthdata token (required for VIIRS data)
export EARTHDATA_TOKEN="your-token-here"

# Process nighttime lights for San Francisco
blackmarble \
  --lat1 37.81 --lon1 -122.55 \
  --lat2 37.69 --lon2 -122.32 \
  --year 2023 --month 6 --day 15 \
  --output-path san_francisco_lights.tif
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--lat1`, `--lon1` | required | First corner coordinates |
| `--lat2`, `--lon2` | required | Second corner coordinates |
| `--year`, `--month`, `--day` | required | Target date for processing |
| `--output-path`, `-o` | `black_marble_output.tif` | Output COG file path |
| `--config`, `-c` | `default` | Preset: `default`, `high_quality`, or `fast` |
| `--save-diagnostics`, `-d` | off | Save intermediate outputs to `data/diagnostics/` |
| `--wgs84`, `-w` | off | Also export EPSG:4326 version |
| `--earthdata-token`, `-t` | env var | NASA Earthdata token (or use EARTHDATA_TOKEN) |
| `--log-level`, `-l` | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

Run `blackmarble --help` for complete options.

## Examples

### High-Quality Urban Analysis
```bash
# Paris with high quality settings and WGS84 export
blackmarble \
  --lat1 48.92 --lon1 2.08 \
  --lat2 48.80 --lon2 2.42 \
  --year 2023 --month 8 --day 1 \
  --config high_quality \
  --wgs84 \
  --output-path paris_lights_hq.tif
```

### Quick Diagnostic Run
```bash
# Kansas City with diagnostics for QA
blackmarble \
  --lat1 39.20 --lon1 -94.74 \
  --lat2 38.97 --lon2 -94.42 \
  --year 2023 --month 5 --day 15 \
  --config fast \
  --save-diagnostics \
  --output-path kansas_test.tif
```

## Documentation

For detailed algorithm documentation (QA masking, temporal compositing, urban field enhancement), see [`docs/pipeline-steps/`](docs/pipeline-steps/). Start with [`SUMMARY.md`](docs/pipeline-steps/SUMMARY.md) for an overview.

## Python API

```python
from blackmarble.pipeline import black_marble_pipeline_sync
from datetime import datetime

# Process a region
result = black_marble_pipeline_sync(
    bbox=(-122.55, 37.69, -122.32, 37.81),  # (min_lon, min_lat, max_lon, max_lat)
    date=datetime(2023, 6, 15),
    output_path="san_francisco.tif"
)
```

## Module Organization

The pipeline follows a clear data flow from acquisition through export:

```
blackmarble/
├── acquire/          # Async downloads: Landsat, VIIRS, OSM roads
│   ├── landsat.py      # Scene selection with date-grouped approach
│   ├── viirs.py        # VIIRS nighttime lights (VNP46A2)
│   └── osm.py          # OpenStreetMap road networks
├── prepare/          # Data preparation and quality control
│   ├── landsat_qa.py   # Cloud/shadow masking with QA bands
│   └── spatial.py      # Reprojection to common grid
├── analyze/          # Core algorithms
│   ├── indices.py      # NDVI, NDWI calculation
│   ├── temporal.py     # Temporal compositing (85th percentile NDVI)
│   └── urban_fields.py # Multi-scale urban enhancement
├── enhance/          # Visualization
│   ├── contrast.py     # Dynamic range adjustment
│   └── visualize.py    # Colormap application
└── export/           # Output generation
    ├── cog.py          # Cloud-Optimized GeoTIFF creation
    └── metadata.py     # Processing metadata
```

## Output

The pipeline produces:
- **Cloud-Optimized GeoTIFF** with embedded metadata
- **RGB visualization** using the inferno colormap
- **Optional EPSG:4326 version** for web mapping (use `--wgs84`)
- **Optional diagnostics** including intermediate processing steps

## Requirements

- Python 3.11+
- GDAL 3.6+
- NASA Earthdata account (free) for VIIRS data
- ~8GB RAM for typical 100x100km regions

## License

[Apache License 2.0](LICENSE)

## Contributing

Issues and pull requests welcome! Report bugs and request features at https://github.com/NASA-IMPACT/veda-black-marble/issues