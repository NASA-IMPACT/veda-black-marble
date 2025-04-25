<!--
Project expectations and next steps captured by Codex.
-->
# Black Marble Refactor: Vision & Next Steps

This document captures the agreed requirements and implementation plan for the `blackmarble` package, based on our recent design discussion.

## 1. CLI Interface
- Provide a command‑line entry point (e.g. `blackmarble run`) that accepts arguments for:
  - Geographic bounds (`--lat1`, `--long1`, `--lat2`, `--long2`)
  - Date (`--year`, `--month`, `--day`)
  - Output / upload settings (`--upload-bucket`, `--upload-key`)
  - Optional flags (e.g. `--verbose`, `--dry-run`)
- Automatically generate a `console_scripts` entry in `setup.py` / `pyproject.toml`.

## 2. Configuration Mechanisms
- Support environment variables (current approach) for credentials and defaults.
- Add optional YAML/JSON config file (e.g. `blackmarble.yml`) that can be passed via `--config`.
- Expose a Python API:
  ```python
  from blackmarble import pipeline
  config = pipeline.Config.from_file('myconfig.yml')
  pipeline.run(config)
  ```

## 3. Testing & Validation
- Build a suite of unit tests targeting each module (`io`, `reproj`, `coords`, `landsat`, `vnp`, `osm`, `masks`, `utils`).
- Include minimal sample data (or mocks) for:
  - GeoTIFF I/O (small synthetic rasters)
  - WRS‑2 shapefile lookups (tiny test shapefile)
  - HDF subdataset parsing (mock or stub)
  - OSM rasterization (small bbox with a few roads)
- Add integration tests for the full `pipeline.run()` using a small bounding box and fixed date. Validate outputs (e.g. correct array shapes, no errors).
- Integrate tests into CI (e.g. GitHub Actions).

## 4. Packaging & Deployment
- Distribute as a standard Python package (via PyPI):
  - `pyproject.toml` / `setup.cfg`
  - Console entry point
- Provide a Dockerfile for containerized execution, pre‑installing all GDAL, Rasterio, OSMnx, etc.
  - Keep the existing `docker/` folder up to date
  - Document usage in `README.md`

## 5. Documentation & Examples
- Expand `README.md` with:
  - Quickstart CLI examples
  - Python API snippets
  - Environment variable reference table
  - Docker usage examples
- Create a `docs/` directory for extended documentation (e.g. Sphinx or MkDocs):
  - Module reference
  - Configuration guide
  - Troubleshooting tips

## 6. Performance & Robustness (Future)
- Introduce optional parallelism or tiled processing for large bounding boxes.
- Add logging and progress reporting (e.g. via `tqdm` or `logging` module).
- Graceful error handling and retry logic for network operations (S3, Earthdata, OSMnx).

---
_Document last updated by Codex on: `date +'%Y-%m-%d'`_