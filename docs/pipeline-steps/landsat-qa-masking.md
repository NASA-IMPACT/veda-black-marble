# Landsat QA Masking

This step identifies and masks clouds, shadows, and other artifacts in Landsat imagery using the QA_PIXEL band.

## Overview

Landsat Collection 2 provides a QA_PIXEL band with bit-packed quality flags. We decode these flags to create masks that exclude poor-quality pixels from analysis.

## Algorithm

### Bit Flag Interpretation

The QA_PIXEL band uses 16-bit integers where each bit represents a quality flag:

| Bit | Flag | Description |
|-----|------|-------------|
| 0 | Fill | No data |
| 1 | Dilated Cloud | Buffered cloud pixels |
| 2 | Cirrus | High-altitude ice clouds |
| 3 | Cloud | Cloud pixels |
| 4 | Cloud Shadow | Shadow cast by clouds |
| 5 | Snow | Snow/ice pixels |
| 6 | Clear | Clear sky conditions |
| 7 | Water | Water body |
| 8-9 | Cloud Confidence | 0=None, 1=Low, 2=Medium, 3=High |
| 10-11 | Cloud Shadow Confidence | Same as above |
| 12-13 | Snow/Ice Confidence | Same as above |
| 14-15 | Cirrus Confidence | Same as above |

### Masking Strategies

**Function**: `blackmarble.prepare.landsat_qa.create_landsat_cloud_mask()`

We provide three pre-configured strategies:

#### 1. Conservative Strategy
- **Masks**: Cloud (high confidence only), shadow (high confidence)
- **Keeps**: Medium/low confidence clouds, all snow, cirrus
- **Use Case**: Maximize data retention, some clouds acceptable

#### 2. Moderate Strategy (Default)
- **Masks**: Cloud (medium & high), shadow (medium & high), snow
- **Keeps**: Low confidence artifacts, clear cirrus
- **Use Case**: Balance between quality and coverage

#### 3. Permissive Strategy  
- **Masks**: All clouds, all shadows, cirrus, snow
- **Keeps**: Only highest quality pixels
- **Use Case**: Highest quality requirements

### Mask Dilation

**Purpose**: Expand masks to catch edge effects

**Parameter**: `dilate_pixels` (default: 3)
- Creates buffer around detected features
- Uses circular structuring element
- Prevents cloud/shadow edges from contaminating analysis

## Implementation

```python
from blackmarble.prepare.landsat_qa import create_landsat_cloud_mask

# Load QA_PIXEL band
with rasterio.open("LC08_QA_PIXEL.TIF") as src:
    qa_pixel = src.read(1)

# Create mask with moderate strategy
cloud_mask = create_landsat_cloud_mask(
    qa_pixel=qa_pixel,
    strategy="moderate",  # or "conservative", "permissive"
    dilate_pixels=3
)

# Apply to spectral bands
for band in ["B3", "B4", "B5"]:
    data[band][~cloud_mask] = np.nan
```

## Key Decisions

### Strategy Selection

**Factors to Consider**:
1. **Scene cloud cover**: High cloud cover → use conservative
2. **Analysis requirements**: Change detection → use permissive
3. **Region characteristics**: 
   - Tropical (frequent clouds) → conservative
   - Arid (few clouds) → permissive

**Recommendation**: Start with moderate, adjust based on results

### Dilation Size

**Trade-offs**:
- Larger dilation (5-10): Better edge removal, more data loss
- Smaller dilation (1-2): More data retained, risk of edge artifacts
- Default (3): Good balance for 30m Landsat pixels

## Test Coverage

**Primary Test**: `tests/pipeline/test_pipeline_flow.py::test_04_landsat_processing`
- Verifies mask creation and application
- Checks clear pixel percentage

**Comprehensive Tests**: `tests/pipeline/test_qa_masking.py`
- Tests all three strategies
- Validates bit flag interpretation  
- Tests mask dilation
- Edge cases: All clear, all cloudy scenes

## Common Issues

1. **Over-masking in Snowy Regions**
   - Solution: Use conservative strategy or custom bit flags
   
2. **Under-masking Cloud Edges**
   - Solution: Increase dilation to 5-7 pixels

3. **Missing Thin Clouds**
   - Solution: Include cirrus bit in custom strategy

## Custom Strategies

For specific requirements, create custom bit flag combinations:

```python
# Custom: Mask only thick clouds
custom_mask = (
    (qa_pixel & (1 << 3)) |  # Cloud bit
    ((qa_pixel >> 8) & 3) == 3  # High confidence only
)
```

## Configuration

```python
# In PipelineConfig
config.preparation.qa_strategy = "moderate"
config.preparation.qa_dilate_pixels = 3
config.preparation.qa_custom_bits = None  # Or list of bits to mask
```

## Performance Notes

- Bit operations are very fast (~0.1s for 5000×5000 pixels)
- Dilation is the slowest part (~1s for same size)
- Strategy choice has minimal performance impact

## Related Steps

- Previous: Landsat data loading
- Next: [Temporal Compositing](./temporal-compositing.md)
- Affects: All Landsat-based indices (NDVI, NDWI)