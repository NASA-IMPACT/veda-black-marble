# Landsat QA Masking

This step identifies and masks clouds, shadows, and other artifacts in Landsat imagery using the QA_PIXEL band.

## Overview

Landsat Collection 2 provides a QA_PIXEL band with bit-packed quality flags. We decode these flags to create masks that exclude poor-quality pixels from analysis. The masking is applied per-date before temporal compositing to ensure consistent cloud removal across the time series.

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
| 7-8 | Water | Two-bit code: 00=land, 01=water, 10=reserved (snow/ice), 11=reserved |
| 9-10 | Cloud Confidence | 0=None, 1=Low, 2=Medium, 3=High |
| 11-12 | Cloud Shadow Confidence | Same as above |
| 13-14 | Snow/Ice Confidence | Same as above |
| 15 | Cirrus Confidence | 0=Low, 1=High |

### Masking Strategies

**Function**: `blackmarble.prepare.landsat_qa.create_landsat_cloud_mask()`

We provide three pre-configured strategies:

#### 1. Conservative Strategy
- **Implementation**: Only accepts known clear values (21824, 21888) OR pixels with CLEAR_BIT set
  - 21824 = 0b0101010101000000 (clear land, high confidence clear)
  - 21888 = 0b0101010110000000 (clear water, high confidence clear)
- **Keeps**: Maximum data retention, accepts lower quality pixels
- **Use Case**: Areas with persistent cloud cover

#### 2. Moderate Strategy (Default)
- **Masks**: Fill, dilated cloud, cirrus, cloud, shadow, snow, water
- **Implementation**: Rejects pixels if ANY problematic bits are set
- **Use Case**: Balance between quality and coverage

#### 3. Permissive Strategy  
- **Masks**: Only high-confidence (≥2) clouds, shadows, and cirrus
- **Keeps**: Low confidence artifacts
- **Use Case**: When some cloud contamination is acceptable

### Mask Dilation

**Purpose**: Expand masks to catch edge effects

**Parameter**: `dilate_pixels` (default: 3)
- Creates buffer around detected features
- Uses square structuring element (2*dilate_pixels+1)
- Prevents cloud/shadow edges from contaminating analysis

## Pipeline Integration

The pipeline applies QA masking during the `process_landsat_date()` function:

1. **Multi-tile Handling**: Creates windowed mosaics for multiple Landsat tiles
2. **Grid Alignment**: Reprojects to locally-optimized Albers Equal Area projection
3. **Per-date Masking**: Applies masks before temporal compositing
4. **Diagnostic Output**: Optionally saves cloud masks as GeoTIFFs

```python
# In process_landsat_date() - actual pipeline code
processed_data = prepare.process_landsat_date(
    bands_data=bands_data,
    bbox=bbox,
    apply_qa_mask=True,
    dilate_pixels=3,
    cloud_strategy="moderate",
    save_diagnostics=True,
    date_str=target_date.strftime('%Y-%m-%d'),
)
```

The function:
- Processes QA_PIXEL first to create cloud mask
- Applies mask to all spectral bands (B3, B4, B5)
- Handles multiple tiles with windowed reads and mosaicking
- Outputs masked data ready for index calculation

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

**Unit Tests**: Located in `blackmarble/prepare/landsat_qa.py`
- `qa_mask()`: Tests all three strategies with bit manipulation
- `create_landsat_cloud_mask()`: Tests with dilation
- `decode_qa_pixel()`: Validates bit field extraction

**Integration**: Full pipeline tests verify end-to-end masking

## Common Issues

1. **Over-masking in Snowy Regions**
   - Solution: Use conservative strategy (keeps snow pixels)
   - Note: Moderate strategy masks snow by default
   
2. **Under-masking Cloud Edges**
   - Solution: Increase dilation to 5-7 pixels
   - Trade-off: More data loss vs better edge removal

3. **Water Bodies Masked**
   - Solution: Moderate strategy excludes definite water (code 01)
   - For water analysis: Use custom strategy without water bit

4. **Multi-tile Boundaries**
   - Solution: Pipeline uses 120-pixel margin for seamless mosaics
   - Prevents gaps at UTM zone boundaries

## Configuration

```python
# In PipelineConfig
config.preparation.qa_cloud_strategy = "moderate"  # or "conservative", "permissive"
config.preparation.qa_dilation_radius = 3          # pixels to dilate

# Per-composite settings (optional)
config.preparation.qa_composite_strategy = "conservative"  # Less aggressive for compositing
config.preparation.qa_composite_dilation = 1              # Less dilation for compositing
```

## Performance Notes

- Bit operations are very fast (~0.1s for 5000×5000 pixels)
- Dilation is the slowest part (~1s for same size)
- Strategy choice has minimal performance impact
- Multi-tile mosaicking uses windowed reads to minimize memory

## Diagnostic Outputs

When `save_diagnostics=True`:
- Cloud masks saved as: `data/diagnostics/cloud_masks/cloud_mask_YYYY-MM-DD.tif`
- QA pixel summary printed to console
- Shows percentage of clear pixels per date

## Related Steps

- Previous: [Landsat Search](./landsat-search.md) - Scene selection
- Next: [Spectral Indices](./spectral-indices.md) - Calculate NDVI/NDWI
- Affects: All Landsat-based analysis (indices, temporal composites)