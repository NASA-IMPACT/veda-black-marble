# Release Notes

## Bug Fixes

- Fixed bug where Landsat scene paths were not covering entire area of interest
- Use gap-filled NTL data that has already been corrected for lunar luminance rather than attempting to correct values ourselves
- No longer derive NDVI from composited reflectance bands; NDVI is calculated per date, then composited using the 85th percentile (median of ratios ≠ ratio of medians)

## Performance

- Use asynchronous I/O wherever possible
- Chunked reading and writing of rasters to avoid memory pressure
- Downloads may be optionally cached if reuse is expected or desirable

## Algorithms

- Moved to VNP46A2 v2 product as v1 is no longer published
- Remove scaling from ntl product, as data now is published as uint32 
- Binary road rasterization has been replaced with fractional road rasterization
- OSM data no longer is burnt in in a binary fashion - instead the fraction of a 30 meter pixel covered by buffered road segments determines the degree of augmentation applied to NTL values
- Replaced simple burn-in augmentation with field-based approach that uses road density and proximity to determine NTL augmentation levels
- Date-grouped Landsat scene selection - instead of selecting the "best" scene per tile independently (which mixed dates), now groups scenes by date and prioritizes temporal consistency

## Infrastructure/Development

- AWS credential chain improvements for better default credential handling in cloud deployments
- Enhanced logging by converting print statements to proper logging throughout codebase
- Addition of a CLI to exercise the pipeline
