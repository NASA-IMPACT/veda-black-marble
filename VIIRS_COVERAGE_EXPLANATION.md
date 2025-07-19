# VIIRS VNP46A2 Coverage Explanation

## Why Only ~40% Coverage in "Gap-Filled" Data?

The user correctly asked: if VNP46A2 does temporal gap-filling over ±30 days, shouldn't we see near 100% coverage?

The answer is **no**, due to multiple compounding factors:

### 1. The ≥3 Clear Samples Rule
- Pixels need at least 3 valid observations in the 30-day window
- Areas with persistent clouds may not meet this threshold

### 2. Ocean Masking
For h17v04 (Western Europe) coverage analysis:
- Western half (Atlantic Ocean): 19% valid
- Eastern half (European land): 58% valid
- The tile is ~50% ocean, which is permanently masked

### 3. Multiple Quality Filters
Every potential observation must pass ALL of these:
- Nighttime only (solar zenith > 101°)
- Lunar illumination < 0.5 × full moon
- No atmospheric stray light
- Cloud-free
- No snow/ice (or snow removal flag)
- Scan angle < 60°
- Terrain slope < 10°
- Good ancillary QC

### 4. Temporal Exclusions
- Full moon periods: 5-7 nights out of 29 excluded
- Stray light seasons: 6-8 weeks/year at mid-latitudes (40-60°N)
- These can overlap, creating multi-week data gaps

### 5. Daily Window Reset
- The 30-day window is recalculated fresh each day
- Previous gap-fills are NOT used to fill current gaps
- If the last clear view was 35 days ago, today remains unfilled

## Expected Coverage by Region

NASA's own documentation states VNP46A2 is "not expected to yield complete coverage."

Typical coverage for daily products:
- Low-latitude land, dry season: 50-90%
- Low-latitude land, wet season: 20-60%
- Mid-latitude land (40-60°N): 0-30% during stray light, 50-85% otherwise
- Tiles ≥50% water: 5-40%

## For Complete Coverage

Use:
- **Monthly composites (VNP46A3)**: Aggregates all clear observations over a month
- **Annual composites (VNP46A4)**: Maximum temporal aggregation
- **Custom rolling composite**: Maintain your own "last clear value" cache

## Conclusion

The ~40% coverage we see is **completely normal** for daily VNP46A2 products over partially oceanic tiles in winter. The gap-filling works as designed - it just has strict quality requirements that often can't be met within a 30-day window.