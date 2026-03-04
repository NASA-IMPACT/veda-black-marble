# Urban Field Enhancement

This step creates multi-scale urban influence fields from road networks and uses them to enhance nighttime lights in urban areas.

## Overview

Urban areas have characteristic spatial patterns - roads create "fields" of urban influence. By computing these fields at multiple scales and using them to enhance nighttime lights, we can better highlight urban corridors and suburban areas that may have lower direct lighting.

## Algorithm

### Step 1: Compute Multi-Scale Urban Fields

**Function**: `blackmarble.analyze.urban_fields.compute_urban_fields()`

**Process**:
1. Start with fractional road raster (0-1 values)
2. Apply Gaussian blur at multiple scales
3. Combine with weighted average

```python
def compute_urban_fields(
    road_fraction: ArrayLike,
    sigmas: list[float] = [2.0, 5.0, 10.0],
    weights: list[float] = [0.2, 0.3, 0.3, 0.2],
) -> ArrayLike:
    """
    Create multi-scale urban influence field from roads.
    
    Args:
        road_fraction: Fractional road coverage (0.0-1.0) per pixel
        sigmas: Gaussian blur scales in pixels
        weights: Weights for [original, scale1, scale2, ...]
    
    Note: len(weights) must equal len(sigmas) + 1
    """
    # Direct coverage (unblurred)
    fields = [road_fraction.astype(np.float32)]
    
    # Add blurred versions at each scale
    for sigma in sigmas:
        density = gaussian_filter(
            road_fraction.astype(np.float32), 
            sigma=sigma, 
            mode="reflect"  # Handle edges properly
        )
        fields.append(density)
    
    # Weighted combination
    urban_field = np.zeros_like(road_fraction, dtype=np.float32)
    for field, weight in zip(fields, weights):
        urban_field += weight * field
    
    return np.clip(urban_field, 0.0, 1.0)
```

**Default Parameters (from config)**:
- `sigmas`: [2.0, 5.0, 10.0] pixels
  - 2 pixels (~60m): Immediate road vicinity
  - 5 pixels (~150m): Neighborhood scale
  - 10 pixels (~300m): District scale
- `weights`: [0.2, 0.3, 0.3, 0.2]
  - 20% original (sharp roads)
  - 30% each for 2 and 5 pixel scales
  - 20% for 10 pixel scale
  - Balanced approach between sharp roads and smooth fields

### Step 2: Enhance NTL with Urban Field

**Function**: `blackmarble.analyze.urban_fields.enhance_ntl_with_urban_field()`

**How it works**:

```python
# Calculate enhancement factor based on urban field density
# Areas with roads get up to 30% brighter (factor of 1.3)
enhancement_factor = 1.0 + (0.3 * urban_field)

# Apply enhancement to nighttime lights
enhanced = ntl * enhancement_factor

# Rescale to preserve original dynamic range
# This prevents the whole image from getting brighter
max_factor = enhancement_factor.max()
result = enhanced / max_factor
```

**Why this approach**:
- Roads indicate urban activity, even if lights are dim
- Enhancement is proportional to road density
- Original brightness relationships are preserved
- No artificial brightening of the entire image

## Pipeline Implementation

The pipeline applies urban field enhancement after VIIRS reprojection:

```python
# From blackmarble/pipeline.py
if config and roads_fractional is not None:
    logger.info("🌃 Enhancing NTL with urban fields (fractional roads)")
    
    # Step 1: Compute multi-scale urban fields
    urban_field = analyze.compute_urban_fields(
        road_fraction=roads_fractional,
        sigmas=config.analysis.urban_field_sigmas,      # [2.0, 5.0, 10.0]
        weights=config.analysis.urban_field_weights,    # [0.2, 0.3, 0.3, 0.2]
    )
    
    # Step 2: Enhance NTL using multiplicative mode
    ntl_enhanced = analyze.enhance_ntl_with_urban_field(
        ntl=ntl_composite,
        urban_field=urban_field,
        enhancement_mode="multiplicative",
        multiplicative_factor=config.analysis.enhancement_multiplicative_factor,  # 0.3
    )
    
    ntl_composite = ntl_enhanced
```

## Test Coverage

- Unit coverage exists in `tests/analyze/`.
- Pipeline-level behavior is validated in repository integration tests.

## Configuration

```python
# In PipelineConfig
config.analysis.urban_field_sigmas = [2.0, 5.0, 10.0]
config.analysis.urban_field_weights = [0.2, 0.3, 0.3, 0.2]
config.analysis.enhancement_multiplicative_factor = 0.3
```

## Related Steps

- Previous: [VIIRS Preprocessing](./viirs-preprocessing.md) - Provides NTL data
- Next: [NDUI Calculation](./ndui-calculation.md) - Uses enhanced NTL
- Outputs: Enhanced NTL array for NDUI calculation