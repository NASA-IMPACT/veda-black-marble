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
def compute_urban_fields(road_fraction, sigmas=[2, 5, 10], weights=[0.5, 0.3, 0.2]):
    """
    Create multi-scale urban influence field from roads.
    
    Args:
        road_fraction: Road pixels (0-1)
        sigmas: Gaussian blur scales in pixels
        weights: Weight for each scale (must sum to ~1)
    
    Note: len(weights) must equal len(sigmas) + 1 (for original)
    """
    fields = [road_fraction]  # Original roads
    
    # Add blurred versions at each scale
    for sigma in sigmas:
        blurred = gaussian_filter(road_fraction, sigma=sigma)
        fields.append(blurred)
    
    # Weighted combination
    urban_field = np.zeros_like(road_fraction)
    for field, weight in zip(fields, weights):
        urban_field += field * weight
    
    return np.clip(urban_field, 0, 1)
```

**Key Parameters**:
- `sigmas`: Controls influence distance
  - 2 pixels (~60m): Immediate road vicinity
  - 5 pixels (~150m): Neighborhood scale
  - 10 pixels (~300m): District scale
- `weights`: Relative importance of each scale
  - Higher weight on original → sharper roads
  - Higher weight on blurred → smoother fields

### Step 2: Enhance NTL with Urban Field {#ntl-enhancement}

**Function**: `blackmarble.analyze.urban_fields.enhance_ntl_with_urban_field()`

**Enhancement Modes**:

#### 1. Multiplicative Enhancement
```python
enhanced = ntl * (1 + urban_field * enhancement_factor)
```
- Scales existing lights
- Preserves relative brightness
- Good for areas with some lighting

#### 2. Additive Enhancement  
```python
enhanced = ntl + (urban_field * enhancement_factor)
```
- Adds light to dark areas
- Can reveal unlit roads
- Risk of over-brightening

#### 3. Hybrid Enhancement (Recommended)
```python
multiplicative = ntl * (1 + urban_field * mult_factor)
additive = ntl + (urban_field * add_factor)
enhanced = mult_weight * multiplicative + (1 - mult_weight) * additive
```
- Best of both approaches
- Configurable blend

## Implementation

```python
from blackmarble.analyze.urban_fields import (
    compute_urban_fields, 
    enhance_ntl_with_urban_field
)

# Step 1: Create urban field from roads
urban_field = compute_urban_fields(
    road_fraction=roads_fractional,  # From road rasterization
    sigmas=[2.0, 5.0, 10.0],        # Multi-scale blur
    weights=[0.5, 0.3, 0.15, 0.05]  # Original + 3 scales
)

# Step 2: Enhance nighttime lights
ntl_enhanced = enhance_ntl_with_urban_field(
    ntl=viirs_aligned,
    urban_field=urban_field,
    enhancement_mode="hybrid",
    enhancement_factor=0.5,      # Overall strength
    mult_factor=0.3,            # Multiplicative component
    add_factor=2.0,             # Additive component  
    mult_weight=0.7             # 70% multiplicative, 30% additive
)
```

## Test Coverage

**Primary Test**: `tests/pipeline/test_pipeline_flow.py::test_10_urban_field_enhancement`
- End-to-end urban field creation and enhancement
- Verifies enhancement increases values

**Detailed Tests**: `tests/pipeline/test_urban_field_enhancement.py`
- Multi-scale field computation
- Each enhancement mode
- Weight array validation
- Edge cases: no roads, all roads

## Key Decisions

### 1. Scale Selection

**Considerations**:
- Pixel size: 30m Landsat → different sigmas than 12m data
- Urban morphology: Dense cities need smaller scales
- Regional patterns: Sprawl needs larger scales

**Recommendations**:
- Dense urban: [1, 3, 5] pixels
- Suburban: [2, 5, 10] pixels  
- Rural roads: [5, 10, 20] pixels

### 2. Weight Distribution

**Strategies**:
1. **Sharp**: [0.7, 0.2, 0.1] - Emphasize actual roads
2. **Balanced**: [0.4, 0.3, 0.2, 0.1] - Even influence
3. **Smooth**: [0.1, 0.3, 0.4, 0.2] - Emphasize fields

**Validation**: Compare with ground truth urban extent

### 3. Enhancement Parameters

**Factors**:
- Base NTL brightness: Dimmer areas need more enhancement
- Urban density: Dense areas need less enhancement
- Output goals: Visualization vs analysis

**Typical Values**:
- Enhancement factor: 0.3-0.7
- Multiplicative weight: 0.6-0.8
- Additive factor: 1.0-5.0 (in radiance units)

## Common Issues

1. **Over-enhancement in Suburbs**
   - Cause: Additive component too strong
   - Fix: Reduce add_factor or increase mult_weight

2. **Roads Not Visible**
   - Cause: Weights favor blurred scales
   - Fix: Increase weight on original (index 0)

3. **Blocky Artifacts**
   - Cause: Integer sigma values
   - Fix: Use float sigmas (2.5, 5.5, etc.)

## Performance Optimization

```python
# Efficient: Single-pass enhancement
enhanced = enhance_ntl_with_urban_field(ntl, field, **params)

# Inefficient: Multiple temporary arrays
temp1 = ntl * (1 + field * 0.3)
temp2 = ntl + field * 2.0
enhanced = 0.7 * temp1 + 0.3 * temp2  # 3x memory
```

## Configuration

```python
# In PipelineConfig
config.analysis.urban_field_sigmas = [2.0, 5.0, 10.0]
config.analysis.urban_field_weights = [0.5, 0.3, 0.15, 0.05]
config.enhancement.ntl_enhancement_mode = "hybrid"
config.enhancement.ntl_enhancement_factor = 0.5
```

## Visual Examples

Before enhancement:
- Dark suburban roads
- Isolated bright spots
- Poor urban continuity

After enhancement:
- Visible road networks
- Connected urban areas
- Smooth brightness gradients

## Related Steps

- Previous: [Road Rasterization](./road-rasterization.md) (provides road fraction)
- Previous: [NDUI Calculation](./ndui-calculation.md) (urban context)
- Next: [Contrast Enhancement](./contrast-enhancement.md)
- Alternative: Legacy road burn-in (when fractional=False)