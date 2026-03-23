# Testing and Reference Data

This page centralizes developer-focused testing commands and test dataset details.

## Running Tests

Run tests from the repository root:

```bash
# Run all tests
uv run pytest

# Run pipeline statistics tests
uv run pytest tests/pipeline/test_statistics.py -v
```

## Reference Data

Test data is committed to the repository in `tests/data/reference/` including:
- VIIRS nighttime lights (150x150 px, ~7.4m resolution)
- Landsat 8 bands B3/B4/B5/QA_PIXEL (150x150 px, 30m resolution)
- OSM road network (12 segments)

### Data Characteristics

- Synthetic but realistic patterns (deterministic generation)
- Small size suitable for version control (~500 KB total)
- Includes test scenarios: clouds, water, vegetation, urban areas
- SHA-256 checksums in `manifest.json` for integrity verification

### Verification

```bash
# Verify data integrity (default mode)
uv run python tools/fetch_reference_data.py

# Force regeneration (uses fixed seeds)
uv run python tools/fetch_reference_data.py --force
```

See [Reference Data README](https://github.com/NASA-IMPACT/veda-black-marble/blob/main/tests/data/reference/README.md) for details.
