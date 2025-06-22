# Temporal Compositor Gap Filling Test - Step by Step Walkthrough

## Test Overview
This test verifies that the TemporalCompositor can fill gaps (NaN values) in temporal composite data using spatial interpolation.

## Step 1: Create the Compositor

```python
compositor = TemporalCompositor(
    method=CompositeMethod.MEDIAN, 
    min_valid_observations=1, 
    fill_nodata=True,
    use_scipy_fill_gaps=True
)
```

**Parameters explained:**
- `method=CompositeMethod.MEDIAN`: Use median value across time for each pixel
- `min_valid_observations=1`: Only need 1 valid observation to create a composite value
- `fill_nodata=True`: Enable gap filling functionality
- `use_scipy_fill_gaps=True`: Use scipy's spatial interpolation to fill remaining gaps

## Step 2: Create Test Data

```python
data = np.array([
    [[1, np.nan, np.nan], [np.nan, 5, np.nan], [np.nan, np.nan, 9]],
    [[np.nan, np.nan, 3], [4, np.nan, np.nan], [np.nan, 8, np.nan]],
], dtype=np.float32)
```

This creates a 3D array with shape `(2, 3, 3)`:
- 2 time steps
- 3x3 spatial grid

### Data Visualization:

**Time Step 0:**
```
[1    NaN  NaN]
[NaN  5    NaN]
[NaN  NaN  9  ]
```

**Time Step 1:**
```
[NaN  NaN  3  ]
[4    NaN  NaN]
[NaN  8    NaN]
```

## Step 3: Temporal Compositing Process

When `compositor.process(data)` is called:

### 3a. Median Calculation
For each pixel position, take the median of valid values across time:

| Position | Time 0 | Time 1 | Median |
|----------|--------|--------|--------|
| [0,0]    | 1      | NaN    | 1      |
| [0,1]    | NaN    | NaN    | NaN    |
| [0,2]    | NaN    | 3      | 3      |
| [1,0]    | NaN    | 4      | 4      |
| [1,1]    | 5      | NaN    | 5      |
| [1,2]    | NaN    | NaN    | NaN    |
| [2,0]    | NaN    | NaN    | NaN    |
| [2,1]    | NaN    | 8      | 8      |
| [2,2]    | 9      | NaN    | 9      |

**After temporal median:**
```
[1    NaN  3  ]
[4    5    NaN]
[NaN  8    9  ]
```

### 3b. Gap Identification
Three positions still have NaN after temporal compositing:
- [0,1] - No valid observations across time
- [1,2] - No valid observations across time  
- [2,0] - No valid observations across time

### 3c. Spatial Gap Filling (scipy)
When `use_scipy_fill_gaps=True`, the `_fill_gaps_scipy` method uses scipy's distance transform to find the nearest valid pixel for each gap:

1. **Distance Transform**: For each NaN pixel, find the nearest valid pixel
2. **Nearest Neighbor Fill**: Copy the value from the nearest valid pixel

For position [0,1]:
- Nearest valid pixels are [0,0]=1 and [1,1]=5 (both distance 1)
- Algorithm picks one (typically [0,0]=1)

For position [1,2]:
- Nearest valid pixels are [0,2]=3 and [2,2]=9 (both distance 1)
- Algorithm picks one (typically [0,2]=3)

For position [2,0]:
- Nearest valid pixels are [1,0]=4 and [2,1]=8 (both distance 1)
- Algorithm picks one (typically [1,0]=4)

**After spatial filling:**
```
[1  1  3]
[4  5  3]
[4  8  9]
```

## Step 4: Verification

```python
# Check that gaps are filled
assert not np.any(np.isnan(composite))

# Check specific filled values
assert composite[0, 0] == 1  # Original value
assert composite[1, 1] == 5  # Original value
assert composite[2, 2] == 9  # Original value
```

The test verifies:
1. No NaN values remain in the composite
2. Original valid values are preserved at positions [0,0], [1,1], and [2,2]

## Key Insights

1. **Two-Stage Process**: 
   - First: Temporal compositing (median across time)
   - Second: Spatial filling (nearest neighbor for remaining gaps)

2. **Parameter Importance**:
   - `fill_nodata=True` alone doesn't fill gaps
   - `use_scipy_fill_gaps=True` is required for actual spatial interpolation

3. **Test Design**: The sparse data pattern ensures:
   - Some pixels have values in both time steps (none)
   - Some pixels have values in only one time step (6 pixels)
   - Some pixels have no values in any time step (3 pixels)
   - This tests both temporal compositing and spatial gap filling

## Why the Test Initially Failed

Without `use_scipy_fill_gaps=True`, the compositor would:
1. Perform temporal median compositing
2. Skip spatial gap filling (just `pass` in the code)
3. Return composite with NaN values still present
4. Cause the assertion `assert not np.any(np.isnan(composite))` to fail