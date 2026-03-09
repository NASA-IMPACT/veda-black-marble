# Statistics Calculation

Reference formulas for summary metrics used in tests and optional reporting.

## Metrics

- **Urban area (km²)**: `urban_pixels * pixel_area_m2 / 1e6`
- **Mean radiance-like value**: `np.nanmean(ntl_array)`

## Formula Reference

- `pixel_area_m2 = abs(transform.a * transform.e)`
- `urban_area_km2 = (urban_pixels * pixel_area_m2) / 1e6`
- `mean_value = nanmean(ntl_data)`

## Current Behavior

- Statistics are not guaranteed in all pipeline return payloads.
- CLI code handles statistics as optional fields.

## Tests

- See `tests/pipeline/test_statistics.py` for calculation examples.

## Related Steps

- [NDUI Calculation](./ndui-calculation.md)
- [Urban Field Enhancement](./urban-field-enhancement.md)