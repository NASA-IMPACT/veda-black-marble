# Test Simplification Example

## Before: Component-based Test

```python
def test_quality_compositing(self):
    """Test combining multiple quality masks."""
    compositor = QualityCompositor(
        cloud_strategy="conservative",
        aerosol_threshold="strict",
        include_saturation=True
    )
    
    # Create test data
    data = {"B3": np.ones((10, 10)), "B4": np.ones((10, 10))}
    qa_layers = {
        "cloud": qa_pixel_array,
        "aerosol": qa_aerosol_array,
        "saturation": qa_radsat_array
    }
    
    # Process
    result = compositor.process(data, qa_layers)
    
    # Extract mask from result
    mask = result.mask
    stats = result.statistics
    
    # Assertions
    assert isinstance(result, MaskResult)
    assert mask.shape == (10, 10)
    assert stats["valid_percentage"] > 0
```

## After: Functional Test

```python
def test_quality_compositing(self):
    """Test combining multiple quality masks."""
    # Create combined mask directly
    mask = create_quality_mask(
        qa_bands={
            "qa_pixel": qa_pixel_array,
            "qa_aerosol": qa_aerosol_array,
            "qa_radsat": qa_radsat_array
        },
        cloud_strategy="conservative",
        aerosol_threshold="strict",
        include_saturation=True
    )
    
    # Direct assertions on mask
    assert mask.shape == (10, 10)
    assert np.sum(mask) > 0  # Some valid pixels
```

## Code Reduction Analysis

### Lines Saved Per Test Pattern:

1. **Component instantiation**: -3 to -5 lines
2. **Result extraction**: -2 to -3 lines  
3. **Wrapper checks**: -1 to -2 lines
4. **Direct data access**: -1 to -2 lines

### Typical Reduction: 30-40% fewer lines per test

### Real Example from test_quality_integration.py:

**Before**: 
```python
# 12 lines
compositor = QualityCompositor()
result = compositor.process(data, qa_layers)
mask = result.mask
stats = result.statistics
assert isinstance(result, MaskResult)
assert mask.shape == expected_shape
assert stats["valid_pixels"] > 0
```

**After**:
```python
# 5 lines
mask = create_quality_mask(qa_bands, combine_method="and")
assert mask.shape == expected_shape
assert np.sum(mask) > 0
```

## Why More Tests Overall?

The functional API made it so easy to write tests that we added more:
- Easier to test edge cases
- Simpler to test function composition
- Less boilerplate = more time for actual test cases
- Better test coverage with less effort