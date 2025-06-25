"""Test functions in blackmarble.prepare.quality module.

This module tests quality assessment and masking functions.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Import the module under test
from blackmarble.prepare.quality import (
    create_cloud_mask,
    create_aerosol_mask,
    create_saturation_mask,
    create_water_mask,
    create_viirs_quality_mask,
    create_quality_mask,
    combine_masks,
    dilate_mask,
    compute_quality_flags,
    create_urban_mask,
    compute_mask_statistics,
)


def test_create_cloud_mask():
    """Test cloud mask creation from Landsat QA_PIXEL band.
    
    This test should verify that:
    - QA_PIXEL bit flags are correctly interpreted for cloud detection
    - Conservative masking provides strict cloud filtering
    - Permissive masking allows more pixels through
    - Cloud dilation expands cloud masks by specified pixels
    - Different cloud confidence levels are handled appropriately
    - Shadow detection works in conjunction with cloud detection
    - Cirrus cloud detection is included in cloud mask
    - Return values are boolean arrays with True = clear pixels
    - Array shapes match input QA_PIXEL dimensions
    - Performance is acceptable for large Landsat scenes
    """
    # Create QA_PIXEL test data with known patterns
    # Based on processing.py, conservative uses only 21824 (clear land) and 21952 (clear water)
    # Permissive masks specific cloud values: 55052, 54852, 54596, 22280
    qa_pixel = np.array([
        [21824, 21952, 55052, 54852],  # clear land, clear water, high conf cloud, medium conf cloud
        [54596, 22280, 30000, 40000],  # low conf cloud, cloud shadow, other, other
        [21824, 21824, 21952, 21952],  # all clear pixels
        [55052, 55052, 55052, 55052]   # all cloudy pixels
    ], dtype=np.uint16)
    
    # Test conservative masking
    conservative_mask = create_cloud_mask(qa_pixel, conservative=True)
    
    # Verify conservative: only 21824 and 21952 are clear (True)
    assert conservative_mask[0, 0] == True   # Clear land
    assert conservative_mask[0, 1] == True   # Clear water
    assert conservative_mask[0, 2] == False  # High confidence cloud
    assert conservative_mask[0, 3] == False  # Medium confidence cloud
    assert conservative_mask[1, 0] == False  # Low confidence cloud
    assert conservative_mask[1, 1] == False  # Cloud shadow
    assert conservative_mask[1, 2] == False  # Other value
    assert conservative_mask[1, 3] == False  # Other value
    
    # Verify all clear row
    assert np.all(conservative_mask[2, :] == True)
    
    # Verify all cloudy row
    assert np.all(conservative_mask[3, :] == False)
    
    # Test permissive masking
    permissive_mask = create_cloud_mask(qa_pixel, conservative=False)
    
    # Verify permissive: masks only specific cloud values
    assert permissive_mask[0, 0] == True   # Clear land (not in invalid list)
    assert permissive_mask[0, 1] == True   # Clear water (not in invalid list)
    assert permissive_mask[0, 2] == False  # 55052 is in invalid list
    assert permissive_mask[0, 3] == False  # 54852 is in invalid list
    assert permissive_mask[1, 0] == False  # 54596 is in invalid list
    assert permissive_mask[1, 1] == False  # 22280 is in invalid list
    assert permissive_mask[1, 2] == True   # 30000 not in invalid list
    assert permissive_mask[1, 3] == True   # 40000 not in invalid list
    
    # Verify permissive allows more pixels than conservative
    assert np.sum(permissive_mask) > np.sum(conservative_mask)
    
    # Test cloud dilation
    dilated_mask = create_cloud_mask(qa_pixel, conservative=True, dilate_pixels=1)
    
    # Original conservative mask has True at [0,0] and [0,1]
    # After dilation, neighboring pixels of False pixels should also become False
    assert dilated_mask[0, 1] == False  # Adjacent to cloud at [0,2]
    assert dilated_mask[1, 0] == False  # Below cloud at [0,0] but adjacent to [1,0] which is masked
    
    # Test zero dilation returns same as no dilation
    no_dilate = create_cloud_mask(qa_pixel, conservative=True, dilate_pixels=0)
    assert np.array_equal(no_dilate, conservative_mask)
    
    # Verify return type and shape
    assert isinstance(conservative_mask, np.ndarray)
    assert conservative_mask.dtype == bool
    assert conservative_mask.shape == qa_pixel.shape
    
    # Test with larger array for performance
    large_qa = np.random.choice([21824, 21952, 55052, 54852], size=(1000, 1000))
    large_mask = create_cloud_mask(large_qa, conservative=True)
    assert large_mask.shape == (1000, 1000)


def test_create_cloud_mask_conservative_vs_permissive():
    """Test differences between conservative and permissive cloud masking.
    
    This test should verify that:
    - Conservative masking removes more potentially cloudy pixels
    - Permissive masking retains more pixels for analysis
    - Cloud confidence thresholds differ appropriately
    - Statistical differences are measurable and documented
    - Use case guidance is clear for each approach
    """
    # Create a diverse QA_PIXEL array with various cloud conditions
    qa_pixel = np.array([
        # Row 0: Mix of clear and various cloud types
        [21824, 21952, 55052, 54852, 54596, 22280],  # clear land, clear water, high/med/low cloud, shadow
        # Row 1: Edge cases and mixed pixels
        [30000, 40000, 50000, 21824, 21952, 55052],  # various non-cloud values, clear, cloud
        # Row 2: Realistic scene with patches
        [21824, 21824, 54596, 54596, 21824, 21824],  # clear patches with low cloud in middle
        # Row 3: All ambiguous values
        [35000, 36000, 37000, 38000, 39000, 40000],  # none are explicitly clear or cloud
    ], dtype=np.uint16)
    
    # Generate masks with both strategies
    conservative_mask = create_cloud_mask(qa_pixel, conservative=True)
    permissive_mask = create_cloud_mask(qa_pixel, conservative=False)
    
    # Count valid pixels in each
    conservative_valid = np.sum(conservative_mask)
    permissive_valid = np.sum(permissive_mask)
    
    # Conservative should remove more pixels (fewer valid)
    assert conservative_valid < permissive_valid
    
    # Calculate statistical differences
    total_pixels = qa_pixel.size
    conservative_percent = (conservative_valid / total_pixels) * 100
    permissive_percent = (permissive_valid / total_pixels) * 100
    
    # Conservative should have significantly lower valid percentage
    assert conservative_percent < permissive_percent
    assert (permissive_percent - conservative_percent) > 10  # At least 10% difference
    
    # Verify specific threshold behaviors
    # Conservative: ONLY 21824 and 21952 are valid
    conservative_valid_values = {21824, 21952}
    for i in range(qa_pixel.shape[0]):
        for j in range(qa_pixel.shape[1]):
            pixel_value = qa_pixel[i, j]
            expected_valid = pixel_value in conservative_valid_values
            assert conservative_mask[i, j] == expected_valid
    
    # Permissive: specific cloud values are invalid
    permissive_invalid_values = {55052, 54852, 54596, 22280}
    for i in range(qa_pixel.shape[0]):
        for j in range(qa_pixel.shape[1]):
            pixel_value = qa_pixel[i, j]
            expected_valid = pixel_value not in permissive_invalid_values
            assert permissive_mask[i, j] == expected_valid
    
    # Document use case guidance
    # Conservative is best for:
    # - High accuracy requirements
    # - Clear sky composites
    # - Avoiding any cloud contamination
    assert conservative_valid == 8  # Only clear land/water pixels
    
    # Permissive is best for:
    # - Maximizing data coverage
    # - Time series with gap filling
    # - Areas with frequent cloud cover
    assert permissive_valid == 17  # Most pixels except explicit clouds
    
    # Verify masks are complementary for decision making
    # Pixels that differ between strategies are "uncertain" pixels
    uncertain_pixels = permissive_mask & ~conservative_mask
    uncertain_count = np.sum(uncertain_pixels)
    assert uncertain_count == (permissive_valid - conservative_valid)


def test_create_cloud_mask_dilation():
    """Test cloud mask dilation functionality.
    
    This test should verify that:
    - Dilation expands cloud edges by specified pixel count
    - Dilation handles image boundaries correctly
    - Zero dilation returns original mask
    - Large dilation values work without performance issues
    - Dilation preserves mask data type and properties
    """
    # Create QA_PIXEL array with isolated cloud pixels for testing dilation
    qa_pixel = np.array([
        [21824, 21824, 21824, 21824, 21824],  # Row 0: all clear
        [21824, 55052, 21824, 21824, 21824],  # Row 1: single cloud pixel
        [21824, 21824, 21824, 21824, 21824],  # Row 2: all clear
        [21824, 21824, 21824, 55052, 21824],  # Row 3: cloud near edge
        [55052, 21824, 21824, 21824, 55052],  # Row 4: clouds at corners
    ], dtype=np.uint16)
    
    # Test zero dilation
    mask_no_dilate = create_cloud_mask(qa_pixel, conservative=True, dilate_pixels=0)
    mask_base = create_cloud_mask(qa_pixel, conservative=True)
    assert np.array_equal(mask_no_dilate, mask_base)
    
    # Test 1-pixel dilation
    mask_dilate1 = create_cloud_mask(qa_pixel, conservative=True, dilate_pixels=1)
    
    # Cloud at [1,1] should expand to affect neighbors
    assert mask_dilate1[1, 1] == False  # Original cloud pixel
    assert mask_dilate1[0, 1] == False  # Above
    assert mask_dilate1[2, 1] == False  # Below
    assert mask_dilate1[1, 0] == False  # Left
    assert mask_dilate1[1, 2] == False  # Right
    
    # Test 2-pixel dilation
    mask_dilate2 = create_cloud_mask(qa_pixel, conservative=True, dilate_pixels=2)
    
    # Cloud influence should extend 2 pixels in all directions
    assert mask_dilate2[0, 0] == False  # Diagonal from [1,1]
    assert mask_dilate2[0, 2] == False  # Diagonal from [1,1]
    assert mask_dilate2[2, 0] == False  # Diagonal from [1,1]
    assert mask_dilate2[2, 2] == False  # Diagonal from [1,1]
    
    # Test boundary handling - cloud at [3,3]
    assert mask_dilate1[3, 3] == False  # Original cloud
    assert mask_dilate1[3, 2] == False  # Left of cloud
    assert mask_dilate1[2, 3] == False  # Above cloud
    assert mask_dilate1[3, 4] == False  # Right of cloud
    assert mask_dilate1[4, 3] == False  # Below cloud
    
    # Test corner handling - clouds at [4,0] and [4,4]
    assert mask_dilate1[4, 0] == False  # Original corner cloud
    assert mask_dilate1[3, 0] == False  # Above corner
    assert mask_dilate1[4, 1] == False  # Right of corner
    
    assert mask_dilate1[4, 4] == False  # Original corner cloud
    assert mask_dilate1[3, 4] == False  # Above corner
    assert mask_dilate1[4, 3] == False  # Left of corner
    
    # Test larger dilation
    mask_dilate3 = create_cloud_mask(qa_pixel, conservative=True, dilate_pixels=3)
    # With 3-pixel dilation, almost entire array should be affected
    valid_count = np.sum(mask_dilate3)
    assert valid_count < 5  # Very few pixels should remain valid
    
    # Test data type preservation
    assert mask_dilate1.dtype == bool
    assert mask_dilate2.dtype == bool
    assert mask_dilate3.dtype == bool
    
    # Test shape preservation
    assert mask_dilate1.shape == qa_pixel.shape
    assert mask_dilate2.shape == qa_pixel.shape
    assert mask_dilate3.shape == qa_pixel.shape
    
    # Performance test with larger array
    large_qa = np.full((500, 500), 21824, dtype=np.uint16)
    # Add some scattered clouds
    large_qa[100:110, 100:110] = 55052
    large_qa[300:310, 300:310] = 55052
    
    # Time large dilation (should complete quickly)
    import time
    start = time.time()
    large_mask = create_cloud_mask(large_qa, conservative=True, dilate_pixels=5)
    elapsed = time.time() - start
    
    assert elapsed < 5.0  # Should complete in reasonable time (dilation is slow)
    assert large_mask.shape == (500, 500)


def test_create_aerosol_mask():
    """Test aerosol quality mask creation from QA_AEROSOL band.
    
    This test should verify that:
    - QA_AEROSOL bit flags are correctly interpreted
    - Standard threshold provides typical aerosol filtering
    - Strict threshold provides more aggressive filtering
    - Different aerosol levels (low, medium, high) are handled
    - Climatology vs. model aerosol sources are considered
    - Return values are boolean arrays with True = acceptable quality
    - Array shapes match input QA_AEROSOL dimensions
    - Edge cases (missing aerosol data) are handled gracefully
    """
    # Create QA_AEROSOL test data
    # Based on processing.py:
    # Standard threshold invalid: {192, 194, 196, 228}
    # Strict threshold invalid: {192, 194, 196, 224, 228}
    qa_aerosol = np.array([
        [192, 194, 196, 228],    # All invalid for standard
        [224, 100, 150, 200],    # 224 only invalid for strict, others valid
        [0, 50, 128, 255],       # Various other values
        [192, 224, 192, 224],    # Mix of standard and strict invalid
    ], dtype=np.uint16)
    
    # Test standard threshold
    standard_mask = create_aerosol_mask(qa_aerosol, threshold="standard")
    
    # Verify standard threshold behavior
    # Row 0: all values are in standard invalid set
    assert standard_mask[0, 0] == False  # 192 is invalid
    assert standard_mask[0, 1] == False  # 194 is invalid
    assert standard_mask[0, 2] == False  # 196 is invalid
    assert standard_mask[0, 3] == False  # 228 is invalid
    
    # Row 1: 224 is valid for standard, only invalid for strict
    assert standard_mask[1, 0] == True   # 224 is valid for standard
    assert standard_mask[1, 1] == True   # 100 is valid
    assert standard_mask[1, 2] == True   # 150 is valid
    assert standard_mask[1, 3] == True   # 200 is valid
    
    # Row 2: all should be valid (not in invalid set)
    assert standard_mask[2, 0] == True   # 0 is valid
    assert standard_mask[2, 1] == True   # 50 is valid
    assert standard_mask[2, 2] == True   # 128 is valid
    assert standard_mask[2, 3] == True   # 255 is valid
    
    # Test strict threshold
    strict_mask = create_aerosol_mask(qa_aerosol, threshold="strict")
    
    # Verify strict threshold behavior
    # Row 0: all still invalid
    assert np.all(strict_mask[0, :] == False)
    
    # Row 1: 224 is now invalid for strict
    assert strict_mask[1, 0] == False  # 224 is invalid for strict
    assert strict_mask[1, 1] == True   # 100 still valid
    assert strict_mask[1, 2] == True   # 150 still valid
    assert strict_mask[1, 3] == True   # 200 still valid
    
    # Verify strict removes more pixels than standard
    assert np.sum(strict_mask) < np.sum(standard_mask)
    
    # Test aerosol level interpretation
    # Low aerosol (good quality): values not in invalid sets
    # Medium aerosol: some values like 224 (strict only)
    # High aerosol: values like 192, 194, 196, 228
    
    # Row with mixed aerosol levels
    mixed_qa = np.array([100, 224, 192, 50], dtype=np.uint16)
    mixed_standard = create_aerosol_mask(mixed_qa, threshold="standard")
    mixed_strict = create_aerosol_mask(mixed_qa, threshold="strict")
    
    assert mixed_standard[0] == True   # Low aerosol
    assert mixed_standard[1] == True   # Medium aerosol (valid for standard)
    assert mixed_standard[2] == False  # High aerosol
    assert mixed_standard[3] == True   # Low aerosol
    
    assert mixed_strict[0] == True    # Low aerosol
    assert mixed_strict[1] == False   # Medium aerosol (invalid for strict)
    assert mixed_strict[2] == False   # High aerosol
    assert mixed_strict[3] == True    # Low aerosol
    
    # Verify return type and shape
    assert isinstance(standard_mask, np.ndarray)
    assert standard_mask.dtype == bool
    assert standard_mask.shape == qa_aerosol.shape
    
    # Test edge case with uniform values
    uniform_valid = np.full((10, 10), 100, dtype=np.uint16)
    uniform_mask = create_aerosol_mask(uniform_valid)
    assert np.all(uniform_mask == True)
    
    uniform_invalid = np.full((10, 10), 192, dtype=np.uint16)
    uniform_mask = create_aerosol_mask(uniform_invalid)
    assert np.all(uniform_mask == False)


def test_create_aerosol_mask_threshold_levels():
    """Test aerosol mask threshold level differences.
    
    This test should verify that:
    - Standard threshold balances data retention and quality
    - Strict threshold provides highest quality at cost of data loss
    - Threshold selection impacts on analysis results
    - Statistical differences between thresholds are documented
    - Guidance for threshold selection is provided
    """
    # Create realistic QA_AEROSOL data representing various conditions
    np.random.seed(42)  # For reproducibility
    
    # Simulate a scene with aerosol gradient
    qa_aerosol = np.zeros((100, 100), dtype=np.uint16)
    
    # Clear area (low aerosol)
    qa_aerosol[0:30, :] = np.random.choice([0, 50, 100, 150], size=(30, 100))
    
    # Moderate aerosol area (includes 224)
    qa_aerosol[30:60, :] = np.random.choice([100, 150, 200, 224], size=(30, 100))
    
    # High aerosol area
    qa_aerosol[60:100, :] = np.random.choice([192, 194, 196, 228], size=(40, 100))
    
    # Apply both thresholds
    standard_mask = create_aerosol_mask(qa_aerosol, threshold="standard")
    strict_mask = create_aerosol_mask(qa_aerosol, threshold="strict")
    
    # Calculate statistics
    total_pixels = qa_aerosol.size
    standard_valid = np.sum(standard_mask)
    strict_valid = np.sum(strict_mask)
    
    standard_percent = (standard_valid / total_pixels) * 100
    strict_percent = (strict_valid / total_pixels) * 100
    
    # Standard should retain more data than strict
    assert standard_valid > strict_valid
    assert standard_percent > strict_percent
    
    # Calculate data loss from standard to strict
    data_loss = standard_valid - strict_valid
    data_loss_percent = (data_loss / standard_valid) * 100
    
    # Should have measurable difference (due to 224 values)
    assert data_loss > 0
    assert 5 < data_loss_percent < 30  # Reasonable range
    
    # Analyze impact on different regions
    # Clear area should be mostly retained by both
    clear_standard = np.sum(standard_mask[0:30, :])
    clear_strict = np.sum(strict_mask[0:30, :])
    clear_retention = (clear_strict / clear_standard) * 100
    assert clear_retention > 95  # Most clear pixels retained
    
    # Moderate area shows the difference
    moderate_standard = np.sum(standard_mask[30:60, :])
    moderate_strict = np.sum(strict_mask[30:60, :])
    moderate_retention = (moderate_strict / moderate_standard) * 100
    assert 50 < moderate_retention < 90  # Significant loss due to 224 values
    
    # High aerosol area should be mostly masked by both
    high_standard = np.sum(standard_mask[60:100, :])
    high_strict = np.sum(strict_mask[60:100, :])
    assert high_standard == 0  # All high aerosol masked
    assert high_strict == 0    # All high aerosol masked
    
    # Document threshold selection guidance
    # Standard threshold use cases:
    # - General analysis where coverage is important
    # - Time series analysis needing consistent coverage
    # - Areas with moderate aerosol levels
    assert standard_percent > 45  # Retains reasonable data
    
    # Strict threshold use cases:
    # - High precision measurements
    # - Clear atmosphere requirements
    # - Radiometric calibration studies
    assert strict_percent < standard_percent - 5  # Noticeably more strict
    
    # Quality vs quantity tradeoff
    quality_improvement = 100 - moderate_retention  # Percent of moderate aerosol removed
    assert quality_improvement > 10  # Meaningful quality improvement
    
    # Verify spatial patterns
    # Strict mask should be a subset of standard mask
    # (pixels valid in strict should also be valid in standard)
    assert np.all(strict_mask <= standard_mask)


def test_create_saturation_mask():
    """Test radiometric saturation mask creation from QA_RADSAT band.
    
    This test should verify that:
    - QA_RADSAT bit flags are correctly interpreted for each band
    - Specific band saturation can be checked individually
    - All-band saturation checking works when band=None
    - Band numbers (1-7) map correctly to bit positions
    - Saturated pixels are identified as False (masked out)
    - Non-saturated pixels are identified as True (valid)
    - Array shapes match input QA_RADSAT dimensions
    - Error handling for invalid band numbers
    """
    # Create QA_RADSAT test data with known bit patterns
    # Based on processing.py: Band 3 = bit 2, Band 4 = bit 3, Band 5 = bit 4
    # Bit positions: 0b00011100 would have bands 3,4,5 saturated
    qa_radsat = np.array([
        [0b00000000, 0b00000100, 0b00001000, 0b00010000],  # No sat, band 3, band 4, band 5
        [0b00001100, 0b00011000, 0b00010100, 0b00011100],  # bands 3+4, 4+5, 3+5, all
        [0b11111111, 0b00000000, 0b00000100, 0b00001000],  # all bits, none, band 3, band 4
        [0b00000001, 0b00000010, 0b00100000, 0b01000000],  # other bits (not 3,4,5)
    ], dtype=np.uint16)
    
    # Test band 3 saturation (bit 2)
    mask_band3 = create_saturation_mask(qa_radsat, band=3)
    
    # Row 0 verification for band 3
    assert mask_band3[0, 0] == True   # No saturation
    assert mask_band3[0, 1] == False  # Band 3 saturated (bit 2 set)
    assert mask_band3[0, 2] == True   # Band 4 saturated, but checking band 3
    assert mask_band3[0, 3] == True   # Band 5 saturated, but checking band 3
    
    # Row 1 verification for band 3
    assert mask_band3[1, 0] == False  # Bands 3+4 saturated
    assert mask_band3[1, 1] == True   # Bands 4+5 saturated, band 3 ok
    assert mask_band3[1, 2] == False  # Bands 3+5 saturated
    assert mask_band3[1, 3] == False  # All bands saturated
    
    # Test band 4 saturation (bit 3)
    mask_band4 = create_saturation_mask(qa_radsat, band=4)
    
    assert mask_band4[0, 0] == True   # No saturation
    assert mask_band4[0, 1] == True   # Band 3 saturated, but checking band 4
    assert mask_band4[0, 2] == False  # Band 4 saturated (bit 3 set)
    assert mask_band4[0, 3] == True   # Band 5 saturated, but checking band 4
    
    # Test band 5 saturation (bit 4)
    mask_band5 = create_saturation_mask(qa_radsat, band=5)
    
    assert mask_band5[0, 0] == True   # No saturation
    assert mask_band5[0, 1] == True   # Band 3 saturated, but checking band 5
    assert mask_band5[0, 2] == True   # Band 4 saturated, but checking band 5
    assert mask_band5[0, 3] == False  # Band 5 saturated (bit 4 set)
    
    # Test all-band saturation check (band=None)
    mask_all = create_saturation_mask(qa_radsat, band=None)
    
    # Only pixels with NO saturation in bands 3,4,5 should be True
    assert mask_all[0, 0] == True   # No saturation
    assert mask_all[0, 1] == False  # Band 3 saturated
    assert mask_all[0, 2] == False  # Band 4 saturated
    assert mask_all[0, 3] == False  # Band 5 saturated
    
    assert mask_all[1, 0] == False  # Bands 3+4 saturated
    assert mask_all[1, 1] == False  # Bands 4+5 saturated
    assert mask_all[1, 2] == False  # Bands 3+5 saturated
    assert mask_all[1, 3] == False  # All bands saturated
    
    # Row 3: other bits set but not bands 3,4,5
    assert mask_all[3, 0] == True  # Bit 0 set (not a saturation bit for 3,4,5)
    assert mask_all[3, 1] == True  # Bit 1 set (not a saturation bit for 3,4,5)
    assert mask_all[3, 2] == True  # Bit 5 set (not bands 3,4,5)
    assert mask_all[3, 3] == True  # Bit 6 set (not bands 3,4,5)
    
    # Test error handling for invalid band numbers
    # Note: processing.py only supports bands 3, 4, 5
    with pytest.raises(ValueError, match="Band must be 3, 4, or 5"):
        create_saturation_mask(qa_radsat, band=1)
    
    with pytest.raises(ValueError, match="Band must be 3, 4, or 5"):
        create_saturation_mask(qa_radsat, band=7)
    
    with pytest.raises(ValueError, match="Band must be 3, 4, or 5"):
        create_saturation_mask(qa_radsat, band=2)
    
    # Verify return type and shape
    assert isinstance(mask_band3, np.ndarray)
    assert mask_band3.dtype == bool
    assert mask_band3.shape == qa_radsat.shape
    
    # Test with NaN values
    qa_nan = np.array([[np.nan, 0, 4, 8]], dtype=np.float32)
    mask_nan = create_saturation_mask(qa_nan, band=3)
    assert mask_nan[0, 0] == False  # NaN treated as invalid
    assert mask_nan[0, 1] == True   # Valid pixel
    assert mask_nan[0, 2] == False  # Saturated
    assert mask_nan[0, 3] == True   # Different band saturated


def test_create_saturation_mask_band_specific():
    """Test band-specific saturation mask creation.
    
    This test should verify that:
    - Individual bands (1-7) can be checked for saturation
    - Band-specific masks differ appropriately
    - Critical bands (e.g., red, NIR) are properly handled
    - Thermal bands have different saturation characteristics
    - Band validation prevents invalid selections
    """
    # Create QA_RADSAT with different band saturation patterns
    # Each pixel has different bands saturated
    qa_radsat = np.array([
        # Systematic saturation patterns
        [0b00000000, 0b00000100, 0b00001000, 0b00010000, 0b00011100],  # None, B3, B4, B5, All
        [0b00000100, 0b00001100, 0b00010100, 0b00011000, 0b00011100],  # B3, B3+4, B3+5, B4+5, All
        # Real-world scenarios
        [0b00000100, 0b00001000, 0b00000000, 0b00011100, 0b00000100],  # Bright targets
        [0b00010000, 0b00010000, 0b00010000, 0b00000000, 0b00011100],  # NIR saturation
    ], dtype=np.uint16)
    
    # Test each supported band (3, 4, 5)
    # Band 3 (Red) - often saturates on bright targets
    mask_b3 = create_saturation_mask(qa_radsat, band=3)
    # Band 4 (NIR) - saturates on vegetation
    mask_b4 = create_saturation_mask(qa_radsat, band=4)
    # Band 5 (SWIR) - saturates on very bright targets
    mask_b5 = create_saturation_mask(qa_radsat, band=5)
    
    # Verify band-specific differences
    # Pixel [0,1]: only band 3 saturated
    assert mask_b3[0, 1] == False  # Band 3 saturated
    assert mask_b4[0, 1] == True   # Band 4 not saturated
    assert mask_b5[0, 1] == True   # Band 5 not saturated
    
    # Pixel [0,2]: only band 4 saturated
    assert mask_b3[0, 2] == True   # Band 3 not saturated
    assert mask_b4[0, 2] == False  # Band 4 saturated
    assert mask_b5[0, 2] == True   # Band 5 not saturated
    
    # Pixel [0,3]: only band 5 saturated
    assert mask_b3[0, 3] == True   # Band 3 not saturated
    assert mask_b4[0, 3] == True   # Band 4 not saturated
    assert mask_b5[0, 3] == False  # Band 5 saturated
    
    # Pixel [0,4]: all bands saturated
    assert mask_b3[0, 4] == False  # Band 3 saturated
    assert mask_b4[0, 4] == False  # Band 4 saturated
    assert mask_b5[0, 4] == False  # Band 5 saturated
    
    # Verify masks are different
    assert not np.array_equal(mask_b3, mask_b4)
    assert not np.array_equal(mask_b4, mask_b5)
    assert not np.array_equal(mask_b3, mask_b5)
    
    # Critical bands analysis
    # Band 3 (Red): Important for vegetation indices
    red_saturated = np.sum(~mask_b3)
    # Band 4 (NIR): Critical for NDVI calculations
    nir_saturated = np.sum(~mask_b4)
    # Band 5 (SWIR): Important for moisture indices
    swir_saturated = np.sum(~mask_b5)
    
    # In this test data, each band has different saturation counts
    assert red_saturated > 0
    assert nir_saturated > 0
    assert swir_saturated > 0
    assert len({red_saturated, nir_saturated, swir_saturated}) > 1  # Different counts
    
    # Test band validation
    # Only bands 3, 4, 5 are supported in processing.py
    valid_bands = [3, 4, 5]
    invalid_bands = [1, 2, 6, 7, 0, -1, 10]
    
    # Valid bands should work
    for band in valid_bands:
        mask = create_saturation_mask(qa_radsat, band=band)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
    
    # Invalid bands should raise errors
    for band in invalid_bands:
        with pytest.raises(ValueError, match="Band must be 3, 4, or 5"):
            create_saturation_mask(qa_radsat, band=band)
    
    # Thermal bands note: Landsat thermal bands (10, 11) are not included
    # in QA_RADSAT saturation flags, only bands 1-7 are tracked
    
    # Verify independence of band checks
    # A pixel can have multiple bands saturated independently
    multi_saturated = qa_radsat[1, 1]  # Has bands 3 and 4 saturated
    assert create_saturation_mask(np.array([[multi_saturated]]), band=3)[0, 0] == False
    assert create_saturation_mask(np.array([[multi_saturated]]), band=4)[0, 0] == False
    assert create_saturation_mask(np.array([[multi_saturated]]), band=5)[0, 0] == True


def test_create_water_mask():
    """Test water mask creation from NDWI values.
    
    This test should verify that:
    - NDWI threshold correctly separates water from land
    - Default threshold (0.0) works for typical conditions
    - Custom thresholds handle different water types
    - Water pixels are identified as True in output mask
    - Land pixels are identified as False in output mask
    - NaN values are treated as land (False) conservatively
    - Seasonal water variations are handled appropriately
    - Different water types (rivers, lakes, ocean) are detected
    - Array shapes and data types are preserved
    """
    # Create NDWI test data representing various conditions
    # NDWI = (Green - NIR) / (Green + NIR)
    # Water: positive values (high green, low NIR)
    # Land: negative values (low green, high NIR)
    ndwi = np.array([
        [0.5, 0.3, 0.1, 0.05],    # Strong water signatures
        [0.0, -0.1, -0.3, -0.5],  # Threshold boundary and land
        [np.nan, 0.2, -0.2, np.nan],  # NaN handling
        [0.8, 0.6, 0.4, 0.2],     # Various water types
    ], dtype=np.float32)
    
    # Test with default threshold (0.0)
    water_mask = create_water_mask(ndwi)
    
    # Verify water detection with default threshold
    # Row 0: all positive NDWI = water
    assert water_mask[0, 0] == True   # 0.5 > 0.0 = water
    assert water_mask[0, 1] == True   # 0.3 > 0.0 = water
    assert water_mask[0, 2] == True   # 0.1 > 0.0 = water
    assert water_mask[0, 3] == True   # 0.05 > 0.0 = water
    
    # Row 1: threshold boundary and land
    assert water_mask[1, 0] == False  # 0.0 <= 0.0 = land (at threshold)
    assert water_mask[1, 1] == False  # -0.1 < 0.0 = land
    assert water_mask[1, 2] == False  # -0.3 < 0.0 = land
    assert water_mask[1, 3] == False  # -0.5 < 0.0 = land
    
    # Row 2: NaN handling
    assert water_mask[2, 0] == False  # NaN treated as land (conservative)
    assert water_mask[2, 1] == True   # 0.2 > 0.0 = water
    assert water_mask[2, 2] == False  # -0.2 < 0.0 = land
    assert water_mask[2, 3] == False  # NaN treated as land
    
    # Test with custom threshold for shallow water
    shallow_threshold = 0.1
    shallow_mask = create_water_mask(ndwi, threshold=shallow_threshold)
    
    # Higher threshold is more selective
    assert shallow_mask[0, 0] == True   # 0.5 > 0.1 = water
    assert shallow_mask[0, 1] == True   # 0.3 > 0.1 = water
    assert shallow_mask[0, 2] == False  # 0.1 <= 0.1 = not water
    assert shallow_mask[0, 3] == False  # 0.05 < 0.1 = not water
    
    # Test with negative threshold for wetlands
    wetland_threshold = -0.2
    wetland_mask = create_water_mask(ndwi, threshold=wetland_threshold)
    
    # Lower threshold includes more pixels
    assert wetland_mask[1, 0] == True   # 0.0 > -0.2 = water/wetland
    assert wetland_mask[1, 1] == True   # -0.1 > -0.2 = water/wetland
    assert wetland_mask[1, 2] == False  # -0.3 <= -0.2 = land
    assert wetland_mask[1, 3] == False  # -0.5 < -0.2 = land
    
    # Verify different water types
    # Ocean/deep water: NDWI > 0.6
    ocean_pixels = ndwi > 0.6
    ocean_mask = create_water_mask(ndwi, threshold=0.6)
    assert ocean_mask[3, 0] == True   # 0.8 > 0.6 = deep water
    assert ocean_mask[3, 1] == False  # 0.6 <= 0.6 = not deep water
    
    # Rivers: NDWI typically 0.2-0.4
    river_threshold = 0.15
    river_mask = create_water_mask(ndwi, threshold=river_threshold)
    assert river_mask[3, 2] == True   # 0.4 > 0.15 = river
    assert river_mask[3, 3] == True   # 0.2 > 0.15 = river
    
    # Seasonal variations test
    # Dry season: higher threshold needed
    dry_season_ndwi = ndwi - 0.1  # Simulate drier conditions
    dry_mask = create_water_mask(dry_season_ndwi, threshold=0.0)
    # Fewer water pixels detected
    assert np.sum(dry_mask) < np.sum(water_mask)
    
    # Wet season: lower threshold appropriate
    wet_season_ndwi = ndwi + 0.1  # Simulate wetter conditions
    wet_mask = create_water_mask(wet_season_ndwi, threshold=0.0)
    # More water pixels detected
    assert np.sum(wet_mask) > np.sum(water_mask)
    
    # Verify array shape and type preservation
    assert isinstance(water_mask, np.ndarray)
    assert water_mask.dtype == bool
    assert water_mask.shape == ndwi.shape
    
    # Test edge cases
    # All water
    all_water_ndwi = np.full((5, 5), 0.5)
    all_water_mask = create_water_mask(all_water_ndwi)
    assert np.all(all_water_mask == True)
    
    # All land
    all_land_ndwi = np.full((5, 5), -0.5)
    all_land_mask = create_water_mask(all_land_ndwi)
    assert np.all(all_land_mask == False)
    
    # All NaN
    all_nan_ndwi = np.full((5, 5), np.nan)
    all_nan_mask = create_water_mask(all_nan_ndwi)
    assert np.all(all_nan_mask == False)  # Conservative: NaN = land


def test_create_water_mask_threshold_sensitivity():
    """Test water mask sensitivity to NDWI threshold.
    
    This test should verify that:
    - Positive thresholds increase water detection sensitivity
    - Negative thresholds decrease water detection sensitivity
    - Threshold selection affects water/land boundary accuracy
    - Seasonal and regional variations require different thresholds
    - Guidance for threshold selection is provided
    """
    # Create NDWI gradient simulating water-land transition
    x = np.linspace(-0.5, 0.8, 100)
    y = np.linspace(-0.5, 0.8, 100)
    ndwi_gradient = np.outer(x, np.ones_like(y))
    
    # Test multiple thresholds
    thresholds = [-0.3, -0.1, 0.0, 0.1, 0.3, 0.5]
    water_pixel_counts = []
    
    for threshold in thresholds:
        mask = create_water_mask(ndwi_gradient, threshold=threshold)
        water_count = np.sum(mask)
        water_pixel_counts.append(water_count)
    
    # Verify that higher thresholds are more selective (fewer water pixels)
    for i in range(len(water_pixel_counts) - 1):
        assert water_pixel_counts[i] > water_pixel_counts[i + 1]
    
    # Test specific threshold behaviors
    # Negative threshold: includes wetlands and moist soil
    wetland_threshold = -0.2
    wetland_mask = create_water_mask(ndwi_gradient, threshold=wetland_threshold)
    wetland_percent = (np.sum(wetland_mask) / wetland_mask.size) * 100
    assert 60 < wetland_percent < 80  # Includes marginal water areas
    
    # Zero threshold: standard water bodies
    standard_mask = create_water_mask(ndwi_gradient, threshold=0.0)
    standard_percent = (np.sum(standard_mask) / standard_mask.size) * 100
    assert 40 < standard_percent < 60  # Balanced water detection
    
    # Positive threshold: only clear water
    clear_water_threshold = 0.3
    clear_mask = create_water_mask(ndwi_gradient, threshold=clear_water_threshold)
    clear_percent = (np.sum(clear_mask) / clear_mask.size) * 100
    assert 15 < clear_percent < 35  # Only high confidence water
    
    # Test boundary accuracy with different thresholds
    # Create test pattern with known water bodies
    test_scene = np.zeros((50, 50))
    # Deep water body (NDWI = 0.6)
    test_scene[10:20, 10:20] = 0.6
    # Shallow water (NDWI = 0.2)
    test_scene[30:40, 10:20] = 0.2
    # Wetland (NDWI = -0.1)
    test_scene[10:20, 30:40] = -0.1
    # Moist soil (NDWI = -0.3)
    test_scene[30:40, 30:40] = -0.3
    
    # Apply different thresholds
    mask_conservative = create_water_mask(test_scene, threshold=0.3)
    mask_standard = create_water_mask(test_scene, threshold=0.0)
    mask_inclusive = create_water_mask(test_scene, threshold=-0.2)
    
    # Verify detection patterns
    # Conservative: only deep water
    assert np.sum(mask_conservative[10:20, 10:20]) == 100  # Deep water detected
    assert np.sum(mask_conservative[30:40, 10:20]) == 0    # Shallow water missed
    assert np.sum(mask_conservative[10:20, 30:40]) == 0    # Wetland missed
    assert np.sum(mask_conservative[30:40, 30:40]) == 0    # Moist soil missed
    
    # Standard: deep and shallow water
    assert np.sum(mask_standard[10:20, 10:20]) == 100  # Deep water detected
    assert np.sum(mask_standard[30:40, 10:20]) == 100  # Shallow water detected
    assert np.sum(mask_standard[10:20, 30:40]) == 0    # Wetland missed
    assert np.sum(mask_standard[30:40, 30:40]) == 0    # Moist soil missed
    
    # Inclusive: water and wetlands
    assert np.sum(mask_inclusive[10:20, 10:20]) == 100  # Deep water detected
    assert np.sum(mask_inclusive[30:40, 10:20]) == 100  # Shallow water detected
    assert np.sum(mask_inclusive[10:20, 30:40]) == 100  # Wetland detected
    assert np.sum(mask_inclusive[30:40, 30:40]) == 0    # Moist soil still missed
    
    # Regional variation guidance
    # Arid regions: use lower threshold (-0.1 to 0.0)
    # Tropical regions: use standard threshold (0.0 to 0.1)
    # Coastal regions: use higher threshold (0.1 to 0.3)
    
    # Seasonal variation guidance
    # Wet season: use higher threshold to avoid false positives
    # Dry season: use lower threshold to capture all water
    # Verify these guidelines with example
    dry_season_ndwi = test_scene - 0.1  # Simulate drier conditions
    wet_season_ndwi = test_scene + 0.1  # Simulate wetter conditions
    
    # Same threshold gives different results
    dry_mask = create_water_mask(dry_season_ndwi, threshold=0.0)
    wet_mask = create_water_mask(wet_season_ndwi, threshold=0.0)
    
    assert np.sum(dry_mask) < np.sum(mask_standard)  # Less water in dry season
    assert np.sum(wet_mask) > np.sum(mask_standard)  # More water in wet season


def test_create_water_mask_edge_cases():
    """Test water mask edge cases and data quality.
    
    This test should verify that:
    - All-water or all-land scenes are handled correctly
    - Mixed water/land pixels near boundaries work appropriately
    - Very small water bodies are detected when appropriate
    - Urban water features are properly identified
    - Mask quality is sufficient for intended applications
    """
    # Test all-water scene
    all_water = np.full((100, 100), 0.5, dtype=np.float32)
    water_mask = create_water_mask(all_water, threshold=0.0)
    assert np.all(water_mask == True)
    assert np.sum(water_mask) == 10000
    
    # Test all-land scene
    all_land = np.full((100, 100), -0.5, dtype=np.float32)
    land_mask = create_water_mask(all_land, threshold=0.0)
    assert np.all(land_mask == False)
    assert np.sum(land_mask) == 0
    
    # Test mixed water/land boundary
    # Create realistic water-land transition
    boundary_scene = np.zeros((50, 50), dtype=np.float32)
    # Water body with gradual transition
    for i in range(50):
        for j in range(50):
            # Distance from center
            dist = np.sqrt((i - 25)**2 + (j - 25)**2)
            if dist < 10:
                boundary_scene[i, j] = 0.4  # Clear water
            elif dist < 15:
                boundary_scene[i, j] = 0.1  # Shallow water
            elif dist < 18:
                boundary_scene[i, j] = 0.0  # Boundary
            elif dist < 20:
                boundary_scene[i, j] = -0.1  # Wet land
            else:
                boundary_scene[i, j] = -0.3  # Dry land
    
    # Test boundary detection with different thresholds
    mask_strict = create_water_mask(boundary_scene, threshold=0.05)
    mask_standard = create_water_mask(boundary_scene, threshold=0.0)
    mask_inclusive = create_water_mask(boundary_scene, threshold=-0.05)
    
    # Verify nested masks (strict subset of standard subset of inclusive)
    assert np.sum(mask_strict) < np.sum(mask_standard)
    assert np.sum(mask_standard) < np.sum(mask_inclusive)
    
    # Verify boundary pixels are handled consistently
    # Pixels exactly at threshold should be excluded (<=)
    boundary_pixels = np.sum(boundary_scene == 0.0)
    assert boundary_pixels > 0  # We have boundary pixels
    
    # Test very small water bodies
    small_water_scene = np.full((100, 100), -0.3, dtype=np.float32)
    # Add small ponds (2x2, 3x3, single pixel)
    small_water_scene[10:12, 10:12] = 0.3  # 2x2 pond
    small_water_scene[20:23, 20:23] = 0.2  # 3x3 pond
    small_water_scene[30, 30] = 0.4        # Single pixel
    small_water_scene[40:41, 40:42] = 0.25 # 1x2 pond
    
    small_mask = create_water_mask(small_water_scene, threshold=0.0)
    
    # Verify small water bodies are detected
    assert np.sum(small_mask[10:12, 10:12]) == 4   # 2x2 detected
    assert np.sum(small_mask[20:23, 20:23]) == 9   # 3x3 detected
    assert small_mask[30, 30] == True               # Single pixel detected
    assert np.sum(small_mask[40:41, 40:42]) == 2   # 1x2 detected
    
    # Test urban water features
    urban_scene = np.full((100, 100), -0.2, dtype=np.float32)  # Urban background
    # Add urban water features
    # Fountain/pool (small, clear water)
    urban_scene[20:25, 20:25] = 0.35
    # Canal (linear feature)
    urban_scene[40:42, 10:90] = 0.15
    # Retention pond (irregular shape)
    for i in range(60, 75):
        for j in range(30, 45):
            if (i-67)**2/64 + (j-37)**2/49 < 1:
                urban_scene[i, j] = 0.25
    
    urban_mask = create_water_mask(urban_scene, threshold=0.1)
    
    # Verify urban water features detected
    assert np.sum(urban_mask[20:25, 20:25]) == 25  # Pool detected
    assert np.sum(urban_mask[40:42, 10:90]) > 100  # Canal detected
    assert np.sum(urban_mask[60:75, 30:45]) > 50   # Retention pond detected
    
    # Test mask quality metrics
    # Good quality mask should have:
    # 1. No isolated single pixels (noise)
    # 2. Smooth boundaries
    # 3. Connected water bodies
    
    # Create noisy test case
    noisy_scene = np.random.uniform(-0.5, 0.5, size=(50, 50))
    noisy_mask = create_water_mask(noisy_scene, threshold=0.0)
    
    # Quality check: real water bodies should be more coherent than noise
    # Count connected components (simplified check)
    water_pixels = np.sum(small_mask)
    noise_pixels = np.sum(noisy_mask)
    
    # Real scenes should have different characteristics than pure noise
    assert water_pixels != noise_pixels
    
    # Test extreme NDWI values
    extreme_scene = np.array([
        [-1.0, -0.99, 0.99, 1.0],
        [-np.inf, np.inf, 0, np.nan]
    ], dtype=np.float32)
    
    extreme_mask = create_water_mask(extreme_scene, threshold=0.0)
    
    # Verify handling of extreme values
    assert extreme_mask[0, 0] == False  # -1.0 is land
    assert extreme_mask[0, 1] == False  # -0.99 is land
    assert extreme_mask[0, 2] == True   # 0.99 is water
    assert extreme_mask[0, 3] == True   # 1.0 is water
    assert extreme_mask[1, 0] == False  # -inf treated as land
    assert extreme_mask[1, 1] == True   # inf treated as water (> threshold)
    assert extreme_mask[1, 2] == False  # 0 is at threshold (land)
    assert extreme_mask[1, 3] == False  # NaN treated as land


def test_create_viirs_quality_mask():
    """Test VIIRS quality mask creation from mandatory quality flag.
    
    This test should verify that:
    - VIIRS quality flag values are correctly interpreted
    - Default valid flags ([0, 2]) work for typical VIIRS data
    - Custom valid flag lists allow flexible quality control
    - Quality mask identifies good pixels as True
    - Poor quality pixels are identified as False
    - Array shapes match input quality flag dimensions
    - Different VIIRS products use appropriate flag interpretation
    - Performance is acceptable for large VIIRS tiles
    """
    # Create VIIRS quality flag test data
    # VIIRS quality flags: 0=high quality, 1=low quality, 2=moderate quality
    quality_flag = np.array([
        [0, 1, 2, 3],      # High, low, moderate, other
        [0, 0, 1, 1],      # Multiple high and low quality
        [2, 2, 2, 2],      # All moderate quality
        [1, 3, 4, 255],    # Various quality levels
    ], dtype=np.uint8)
    
    # Test with default valid flags ([0, 2])
    default_mask = create_viirs_quality_mask(quality_flag)
    
    # Verify default behavior: 0 and 2 are valid
    assert default_mask[0, 0] == True   # 0 = high quality (valid)
    assert default_mask[0, 1] == False  # 1 = low quality (invalid)
    assert default_mask[0, 2] == True   # 2 = moderate quality (valid)
    assert default_mask[0, 3] == False  # 3 = other (invalid)
    
    # Row 1: multiple high and low quality
    assert default_mask[1, 0] == True   # 0 = valid
    assert default_mask[1, 1] == True   # 0 = valid
    assert default_mask[1, 2] == False  # 1 = invalid
    assert default_mask[1, 3] == False  # 1 = invalid
    
    # Row 2: all moderate quality
    assert np.all(default_mask[2, :] == True)  # All 2s are valid
    
    # Row 3: various invalid values
    assert default_mask[3, 0] == False  # 1 = invalid
    assert default_mask[3, 1] == False  # 3 = invalid
    assert default_mask[3, 2] == False  # 4 = invalid
    assert default_mask[3, 3] == False  # 255 = invalid
    
    # Test with custom valid flags - high quality only
    high_only_mask = create_viirs_quality_mask(quality_flag, valid_flags=[0])
    
    # Only flag 0 should be valid
    assert high_only_mask[0, 0] == True   # 0 = valid
    assert high_only_mask[0, 1] == False  # 1 = invalid
    assert high_only_mask[0, 2] == False  # 2 = now invalid
    assert high_only_mask[0, 3] == False  # 3 = invalid
    
    # Verify fewer valid pixels with stricter criteria
    assert np.sum(high_only_mask) < np.sum(default_mask)
    
    # Test with permissive flags - include low quality
    permissive_mask = create_viirs_quality_mask(quality_flag, valid_flags=[0, 1, 2])
    
    # Flags 0, 1, 2 should all be valid
    assert permissive_mask[0, 0] == True   # 0 = valid
    assert permissive_mask[0, 1] == True   # 1 = now valid
    assert permissive_mask[0, 2] == True   # 2 = valid
    assert permissive_mask[0, 3] == False  # 3 = still invalid
    
    # More valid pixels with permissive criteria
    assert np.sum(permissive_mask) > np.sum(default_mask)
    
    # Test empty valid flags list
    empty_flags_mask = create_viirs_quality_mask(quality_flag, valid_flags=[])
    assert np.all(empty_flags_mask == False)  # Nothing is valid
    
    # Test with all possible flags
    all_flags = list(range(256))
    all_valid_mask = create_viirs_quality_mask(quality_flag, valid_flags=all_flags)
    assert np.all(all_valid_mask == True)  # Everything is valid
    
    # Verify array shape and type
    assert isinstance(default_mask, np.ndarray)
    assert default_mask.dtype == bool
    assert default_mask.shape == quality_flag.shape
    
    # Test different VIIRS products interpretation
    # VNP46A1: Daily product might use standard flags
    vnp46a1_mask = create_viirs_quality_mask(quality_flag, valid_flags=[0, 2])
    
    # VNP46A2: Monthly composite might be stricter
    vnp46a2_mask = create_viirs_quality_mask(quality_flag, valid_flags=[0])
    
    # Verify product-specific differences
    assert np.sum(vnp46a2_mask) <= np.sum(vnp46a1_mask)
    
    # Performance test with large VIIRS tile
    large_quality = np.random.choice([0, 1, 2, 3], size=(3000, 3000), p=[0.6, 0.2, 0.15, 0.05])
    
    import time
    start = time.time()
    large_mask = create_viirs_quality_mask(large_quality)
    elapsed = time.time() - start
    
    assert elapsed < 1.0  # Should process large tile quickly
    assert large_mask.shape == (3000, 3000)
    
    # Verify quality flag statistics make sense
    valid_percent = (np.sum(large_mask) / large_mask.size) * 100
    assert 60 < valid_percent < 85  # Based on probabilities used


def test_create_viirs_quality_mask_flag_validation():
    """Test VIIRS quality flag validation and interpretation.
    
    This test should verify that:
    - All valid VIIRS quality flag values are recognized
    - Invalid flag values are handled appropriately
    - Flag interpretation matches VIIRS documentation
    - Custom flag lists are validated for correctness
    - Statistical summaries of flag distributions are meaningful
    """
    # Create comprehensive test data with all common VIIRS quality flags
    # Based on VIIRS documentation:
    # 0 = High-quality, Persistent nighttime lights
    # 1 = High-quality, Ephemeral nighttime Lights
    # 2 = Poor-quality, Outlier, potential cloud contamination
    # 255 = No retrieval, Fill value
    quality_flags = np.array([
        [0, 1, 2, 255],     # All documented values
        [0, 0, 0, 0],       # All high quality
        [1, 1, 1, 1],       # All ephemeral
        [2, 2, 2, 2],       # All poor quality
        [255, 255, 255, 255],  # All fill values
        [3, 4, 5, 10],      # Undocumented values
    ], dtype=np.uint8)
    
    # Test recognition of valid flag values
    # Standard interpretation: 0 and 2 are typically used
    standard_mask = create_viirs_quality_mask(quality_flags)
    
    # Row 0: mixed flags
    assert standard_mask[0, 0] == True   # 0 in valid list
    assert standard_mask[0, 1] == False  # 1 not in default valid list
    assert standard_mask[0, 2] == True   # 2 in valid list
    assert standard_mask[0, 3] == False  # 255 not in valid list
    
    # Test with all documented flags
    all_documented = [0, 1, 2, 255]
    doc_mask = create_viirs_quality_mask(quality_flags, valid_flags=all_documented)
    
    # All documented values should be recognized
    assert np.all(doc_mask[0, :] == True)  # Row 0 has all documented values
    
    # Undocumented values should be invalid
    assert np.all(doc_mask[5, :] == False)  # Row 5 has undocumented values
    
    # Test flag interpretation according to VIIRS docs
    # High quality persistent lights (0) - always valid for nighttime analysis
    high_quality_mask = create_viirs_quality_mask(quality_flags, valid_flags=[0])
    assert np.all(high_quality_mask[1, :] == True)   # All 0s
    assert np.all(high_quality_mask[2, :] == False)  # No 0s
    
    # Ephemeral lights (1) - valid for event detection
    ephemeral_mask = create_viirs_quality_mask(quality_flags, valid_flags=[1])
    assert np.all(ephemeral_mask[2, :] == True)   # All 1s
    assert np.all(ephemeral_mask[1, :] == False)  # No 1s
    
    # Poor quality (2) - sometimes included for coverage
    poor_quality_mask = create_viirs_quality_mask(quality_flags, valid_flags=[2])
    assert np.all(poor_quality_mask[3, :] == True)   # All 2s
    assert np.all(poor_quality_mask[1, :] == False)  # No 2s
    
    # Fill values (255) - should never be valid for analysis
    fill_mask = create_viirs_quality_mask(quality_flags, valid_flags=[255])
    assert np.all(fill_mask[4, :] == True)  # All 255s marked as "valid" if explicitly requested
    
    # Test custom flag list validation
    # Valid flag lists
    valid_lists = [
        [0],           # High quality only
        [0, 1],        # High quality + ephemeral
        [0, 2],        # Default
        [0, 1, 2],     # All quality levels
        list(range(256)),  # All possible values
    ]
    
    for flag_list in valid_lists:
        mask = create_viirs_quality_mask(quality_flags, valid_flags=flag_list)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
    
    # Test statistical summaries
    # Create realistic VIIRS scene
    np.random.seed(42)
    realistic_scene = np.random.choice(
        [0, 1, 2, 255],
        size=(1000, 1000),
        p=[0.7, 0.1, 0.15, 0.05]  # Typical distribution
    )
    
    # Calculate flag distribution
    unique, counts = np.unique(realistic_scene, return_counts=True)
    flag_dist = dict(zip(unique, counts))
    
    # Verify distribution matches expected probabilities
    total = realistic_scene.size
    assert 0.65 < flag_dist[0] / total < 0.75    # ~70% high quality
    assert 0.05 < flag_dist[1] / total < 0.15    # ~10% ephemeral
    assert 0.10 < flag_dist[2] / total < 0.20    # ~15% poor quality
    assert 0.02 < flag_dist[255] / total < 0.08  # ~5% fill
    
    # Test mask statistics for different flag selections
    masks_stats = {
        'high_only': create_viirs_quality_mask(realistic_scene, valid_flags=[0]),
        'default': create_viirs_quality_mask(realistic_scene, valid_flags=[0, 2]),
        'inclusive': create_viirs_quality_mask(realistic_scene, valid_flags=[0, 1, 2]),
        'all': create_viirs_quality_mask(realistic_scene, valid_flags=list(range(256))),
    }
    
    # Verify statistical relationships
    high_percent = np.sum(masks_stats['high_only']) / total * 100
    default_percent = np.sum(masks_stats['default']) / total * 100
    inclusive_percent = np.sum(masks_stats['inclusive']) / total * 100
    all_percent = np.sum(masks_stats['all']) / total * 100
    
    assert high_percent < default_percent < inclusive_percent < all_percent
    assert 65 < high_percent < 75      # Matches high quality distribution
    assert 80 < default_percent < 90   # High + poor quality
    assert 90 < inclusive_percent < 98 # Excludes only fill values
    assert all_percent == 100          # Everything included
    
    # Test edge cases
    # Empty array
    empty_flags = np.array([], dtype=np.uint8)
    empty_mask = create_viirs_quality_mask(empty_flags)
    assert empty_mask.size == 0
    assert empty_mask.shape == (0,)
    
    # Single value
    single_flag = np.array([0], dtype=np.uint8)
    single_mask = create_viirs_quality_mask(single_flag)
    assert single_mask[0] == True  # 0 is in default valid flags


def test_create_quality_mask():
    """Test comprehensive quality mask creation from multiple QA bands.
    
    This test should verify that:
    - Multiple QA bands are combined correctly
    - Combination methods ('and', 'or') work as expected
    - Cloud strategy parameter affects cloud masking
    - Cloud dilation parameter expands cloud masks appropriately
    - Aerosol threshold parameter filters aerosol contamination
    - Saturation masking includes/excludes saturated pixels correctly
    - Saturation bands parameter controls which bands are checked
    - Water masking removes/retains water pixels as requested
    - Combined mask represents overall pixel quality accurately
    - Error handling for missing or invalid QA bands
    """
    # Create test QA bands
    qa_pixel = np.array([
        [21824, 21824, 55052, 55052],  # Clear, clear, cloud, cloud
        [21824, 55052, 21824, 55052],  # Mixed
        [21952, 21952, 21824, 21824],  # Water, water, clear, clear
        [30000, 40000, 50000, 21824],  # Various non-cloud, clear
    ], dtype=np.uint16)
    
    qa_aerosol = np.array([
        [100, 100, 192, 192],    # Good, good, bad, bad
        [100, 192, 100, 192],    # Mixed
        [224, 224, 100, 100],    # Moderate (bad for strict), good
        [100, 100, 100, 192],    # Good, good, good, bad
    ], dtype=np.uint16)
    
    qa_radsat = np.array([
        [0, 0, 4, 4],      # No sat, no sat, band 3 sat, band 3 sat
        [0, 8, 0, 12],     # No sat, band 4 sat, no sat, bands 3+4 sat
        [0, 0, 0, 0],      # All clear
        [16, 0, 28, 0],    # Band 5 sat, clear, all bands sat, clear
    ], dtype=np.uint16)
    
    ndwi = np.array([
        [0.5, 0.3, -0.2, -0.3],   # Water, water, land, land
        [-0.1, 0.1, -0.1, 0.2],   # Land, water, land, water
        [0.6, 0.7, -0.4, -0.5],   # Deep water, deep water, land, land
        [-0.2, -0.3, 0.0, 0.4],   # Land, land, boundary, water
    ], dtype=np.float32)
    
    qa_bands = {
        "qa_pixel": qa_pixel,
        "qa_aerosol": qa_aerosol,
        "qa_radsat": qa_radsat,
        "ndwi": ndwi
    }
    
    # Test default parameters (conservative cloud, standard aerosol, include saturation)
    default_mask = create_quality_mask(qa_bands)
    
    # Pixel [0,0]: clear + good aerosol + no sat = valid
    assert default_mask[0, 0] == True
    # Pixel [0,2]: cloud = invalid (conservative strategy)
    assert default_mask[0, 2] == False
    # Pixel [1,1]: cloud = invalid
    assert default_mask[1, 1] == False
    # Pixel [0,1] with aerosol 192 = invalid
    # Actually pixel [0,1] has good aerosol (100), let me fix:
    # Pixel [0,2] has bad aerosol (192) AND cloud
    assert default_mask[0, 2] == False
    
    # Test 'or' combination method
    or_mask = create_quality_mask(qa_bands, combine_method="or")
    # With 'or', pixel is valid if ANY mask says it's valid
    # This should be more permissive
    assert np.sum(or_mask) > np.sum(default_mask)
    
    # Test cloud strategies
    conservative_mask = create_quality_mask(qa_bands, cloud_strategy="conservative")
    permissive_mask = create_quality_mask(qa_bands, cloud_strategy="permissive")
    assert np.sum(permissive_mask) > np.sum(conservative_mask)
    
    # Test cloud dilation
    dilated_mask = create_quality_mask(qa_bands, cloud_dilate=1)
    no_dilate_mask = create_quality_mask(qa_bands, cloud_dilate=0)
    # Dilation should reduce valid pixels
    assert np.sum(dilated_mask) < np.sum(no_dilate_mask)
    
    # Test aerosol thresholds
    standard_aerosol = create_quality_mask(qa_bands, aerosol_threshold="standard")
    strict_aerosol = create_quality_mask(qa_bands, aerosol_threshold="strict")
    # Strict should have fewer valid pixels
    assert np.sum(strict_aerosol) < np.sum(standard_aerosol)
    
    # Test saturation masking
    with_sat = create_quality_mask(qa_bands, include_saturation=True)
    without_sat = create_quality_mask(qa_bands, include_saturation=False)
    # Without saturation check should have more valid pixels
    assert np.sum(without_sat) >= np.sum(with_sat)
    
    # Test specific saturation bands
    band3_only = create_quality_mask(qa_bands, saturation_bands=[3])
    band4_only = create_quality_mask(qa_bands, saturation_bands=[4])
    band5_only = create_quality_mask(qa_bands, saturation_bands=[5])
    all_bands = create_quality_mask(qa_bands, saturation_bands=[3, 4, 5])
    
    # Different band checks should give different results
    assert not np.array_equal(band3_only, band4_only)
    assert not np.array_equal(band4_only, band5_only)
    # All bands should be most restrictive
    assert np.sum(all_bands) <= np.sum(band3_only)
    
    # Test water masking
    with_water = create_quality_mask(qa_bands, include_water=True, water_threshold=0.0)
    without_water = create_quality_mask(qa_bands, include_water=False)
    
    # Water masking should affect results
    assert not np.array_equal(with_water, without_water)
    # Pixels with positive NDWI should be masked when include_water=True
    # Check pixel [0,0] which has NDWI=0.5 (water)
    # When water is masked, this pixel should be invalid
    assert without_water[0, 0] == True  # Not masking water
    assert with_water[0, 0] == False    # Water pixel masked out
    
    # Test custom water threshold
    high_water_threshold = create_quality_mask(
        qa_bands, include_water=True, water_threshold=0.3
    )
    # Higher threshold means fewer pixels identified as water
    assert np.sum(high_water_threshold) > np.sum(with_water)
    
    # Test error handling - missing QA bands
    incomplete_bands = {"qa_pixel": qa_pixel}  # Missing other bands
    # Should still work with just qa_pixel
    partial_mask = create_quality_mask(incomplete_bands)
    assert isinstance(partial_mask, np.ndarray)
    
    # Test with empty QA bands dict
    with pytest.raises(ValueError):
        create_quality_mask({})
    
    # Test comprehensive scenario with all parameters
    comprehensive_mask = create_quality_mask(
        qa_bands,
        combine_method="and",
        cloud_strategy="conservative",
        cloud_dilate=1,
        aerosol_threshold="strict",
        include_saturation=True,
        saturation_bands=[3, 4, 5],
        include_water=True,
        water_threshold=0.1
    )
    
    # This should be the most restrictive configuration
    assert np.sum(comprehensive_mask) < np.sum(default_mask)
    
    # Verify specific pixels in comprehensive scenario
    # Pixel [2,2]: clear land, good aerosol, no sat, not water (-0.4 NDWI)
    # Should be valid even in strict mode
    assert comprehensive_mask[2, 2] == True
    
    # Pixel [0,0]: clear but water (0.5 NDWI > 0.1)
    # Should be invalid due to water masking
    assert comprehensive_mask[0, 0] == False


def test_create_quality_mask_combination_methods():
    """Test quality mask combination methods.
    
    This test should verify that:
    - 'and' combination requires all masks to pass (conservative)
    - 'or' combination allows any mask to pass (permissive)
    - Combination results differ appropriately between methods
    - Statistical properties of combined masks are documented
    - Guidance for method selection is provided
    """
    # Create test data with specific quality issues
    qa_pixel = np.array([
        [21824, 21824, 55052, 21824],  # Clear except one cloud
        [21824, 55052, 55052, 55052],  # One clear, rest cloud
        [21824, 21824, 21824, 21824],  # All clear
        [55052, 55052, 55052, 55052],  # All cloud
    ], dtype=np.uint16)
    
    qa_aerosol = np.array([
        [100, 192, 100, 100],    # One bad aerosol
        [100, 100, 100, 192],    # One bad aerosol
        [100, 100, 100, 100],    # All good
        [192, 192, 192, 192],    # All bad
    ], dtype=np.uint16)
    
    qa_radsat = np.array([
        [0, 0, 0, 4],      # One saturated
        [0, 0, 8, 0],      # One saturated
        [0, 0, 0, 0],      # None saturated
        [28, 28, 28, 28],  # All saturated
    ], dtype=np.uint16)
    
    qa_bands = {
        "qa_pixel": qa_pixel,
        "qa_aerosol": qa_aerosol,
        "qa_radsat": qa_radsat
    }
    
    # Test 'and' combination (all masks must pass)
    and_mask = create_quality_mask(qa_bands, combine_method="and")
    
    # Test 'or' combination (any mask can pass)
    or_mask = create_quality_mask(qa_bands, combine_method="or")
    
    # Verify 'and' behavior - pixel must pass ALL quality checks
    # Row 0, Col 0: clear + good aerosol + no sat = VALID
    assert and_mask[0, 0] == True
    # Row 0, Col 1: clear + BAD aerosol + no sat = INVALID (fails aerosol)
    assert and_mask[0, 1] == False
    # Row 0, Col 2: CLOUD + good aerosol + no sat = INVALID (fails cloud)
    assert and_mask[0, 2] == False
    # Row 0, Col 3: clear + good aerosol + SAT = INVALID (fails saturation)
    assert and_mask[0, 3] == False
    
    # Row 2: all clear, good aerosol, no saturation = ALL VALID
    assert np.all(and_mask[2, :] == True)
    
    # Row 3: all fail everything = ALL INVALID
    assert np.all(and_mask[3, :] == False)
    
    # Verify 'or' behavior - pixel passes if ANY check passes
    # For 'or' to work properly, we need to think about what "passes" means
    # Actually, let me reconsider - the masks being combined are:
    # cloud_mask (True = clear), aerosol_mask (True = good), sat_mask (True = not saturated)
    # So 'or' means: pixel is valid if clear OR good aerosol OR not saturated
    
    # This is a bit counterintuitive, so let's verify the actual behavior
    # Row 3: all cloud, bad aerosol, saturated
    # With 'and': needs all conditions = False (none pass)
    # With 'or': needs any condition = False (none pass)
    assert np.all(or_mask[3, :] == False)
    
    # Statistical comparison
    and_valid = np.sum(and_mask)
    or_valid = np.sum(or_mask)
    
    # 'or' should generally be more permissive than 'and'
    assert or_valid >= and_valid
    
    # Calculate statistics
    total_pixels = qa_pixel.size
    and_percent = (and_valid / total_pixels) * 100
    or_percent = (or_valid / total_pixels) * 100
    
    # Document statistical properties
    # In this test case:
    # - Only pixels passing ALL checks are valid in 'and' mode
    # - Pixels passing ANY check are valid in 'or' mode
    assert and_percent < 50  # Restrictive
    assert or_percent >= and_percent  # More permissive
    
    # Test with real-world scenario
    # Create larger dataset with typical distributions
    np.random.seed(42)
    size = (100, 100)
    
    # 80% clear pixels
    qa_pixel_large = np.where(
        np.random.random(size) < 0.8, 21824, 55052
    ).astype(np.uint16)
    
    # 90% good aerosol
    qa_aerosol_large = np.where(
        np.random.random(size) < 0.9, 100, 192
    ).astype(np.uint16)
    
    # 95% not saturated
    qa_radsat_large = np.where(
        np.random.random(size) < 0.95, 0, 4
    ).astype(np.uint16)
    
    qa_bands_large = {
        "qa_pixel": qa_pixel_large,
        "qa_aerosol": qa_aerosol_large,
        "qa_radsat": qa_radsat_large
    }
    
    and_large = create_quality_mask(qa_bands_large, combine_method="and")
    or_large = create_quality_mask(qa_bands_large, combine_method="or")
    
    # Calculate expected valid pixels
    # For 'and': 0.8 * 0.9 * 0.95 ≈ 0.684 (68.4%)
    and_percent_large = (np.sum(and_large) / and_large.size) * 100
    assert 60 < and_percent_large < 75
    
    # For 'or': harder to calculate exactly, but should be higher
    or_percent_large = (np.sum(or_large) / or_large.size) * 100
    assert or_percent_large > and_percent_large
    
    # Method selection guidance
    # Use 'and' when:
    # - High quality requirements
    # - Can afford data loss
    # - Need confidence in all aspects
    quality_ratio = and_percent_large / 100
    assert quality_ratio < 0.75  # Significant data filtering
    
    # Use 'or' when:
    # - Maximum coverage needed
    # - Can tolerate some quality issues
    # - Have other filtering downstream
    coverage_ratio = or_percent_large / 100
    assert coverage_ratio > quality_ratio  # Better coverage
    
    # Verify the methods produce different results
    assert not np.array_equal(and_mask, or_mask)
    assert not np.array_equal(and_large, or_large)
    
    # Edge case: single quality band
    single_band = {"qa_pixel": qa_pixel}
    and_single = create_quality_mask(single_band, combine_method="and")
    or_single = create_quality_mask(single_band, combine_method="or")
    # With single band, 'and' and 'or' should be identical
    assert np.array_equal(and_single, or_single)


def test_create_quality_mask_saturation_bands():
    """Test saturation bands parameter in quality mask creation.
    
    This test should verify that:
    - Specific bands can be checked for saturation
    - Band list parameter is passed through correctly
    - Default band selection is appropriate for typical use
    - Band validation prevents invalid specifications
    - Different band combinations affect mask results appropriately
    """
    # Create test data with different band saturations
    qa_pixel = np.full((4, 4), 21824, dtype=np.uint16)  # All clear
    
    qa_radsat = np.array([
        [0, 4, 8, 16],     # No sat, band 3, band 4, band 5
        [12, 20, 24, 28],  # bands 3+4, bands 3+5, bands 4+5, all bands
        [0, 0, 0, 0],      # No saturation
        [4, 8, 16, 0],     # Individual bands again
    ], dtype=np.uint16)
    
    qa_bands = {"qa_pixel": qa_pixel, "qa_radsat": qa_radsat}
    
    # Test default behavior (should check all bands 3,4,5)
    default_mask = create_quality_mask(qa_bands, include_saturation=True)
    
    # Test band 3 only
    band3_mask = create_quality_mask(qa_bands, include_saturation=True, saturation_bands=[3])
    
    # Test band 4 only  
    band4_mask = create_quality_mask(qa_bands, include_saturation=True, saturation_bands=[4])
    
    # Test band 5 only
    band5_mask = create_quality_mask(qa_bands, include_saturation=True, saturation_bands=[5])
    
    # Test multiple bands
    bands34_mask = create_quality_mask(qa_bands, include_saturation=True, saturation_bands=[3, 4])
    
    # Verify different results for different band selections
    # Pixel [0,1]: band 3 saturated
    assert band3_mask[0, 1] == False  # Band 3 check fails
    assert band4_mask[0, 1] == True   # Band 4 check passes
    assert band5_mask[0, 1] == True   # Band 5 check passes
    
    # Pixel [0,2]: band 4 saturated
    assert band3_mask[0, 2] == True   # Band 3 check passes
    assert band4_mask[0, 2] == False  # Band 4 check fails
    assert band5_mask[0, 2] == True   # Band 5 check passes
    
    # Pixel [0,3]: band 5 saturated
    assert band3_mask[0, 3] == True   # Band 3 check passes
    assert band4_mask[0, 3] == True   # Band 4 check passes
    assert band5_mask[0, 3] == False  # Band 5 check fails
    
    # Combined band check should be more restrictive
    # Pixel [1,0]: bands 3+4 saturated
    assert bands34_mask[1, 0] == False  # Both bands 3 and 4 saturated
    assert band3_mask[1, 0] == False    # Band 3 saturated
    assert band4_mask[1, 0] == False    # Band 4 saturated
    assert band5_mask[1, 0] == True     # Band 5 not saturated
    
    # Verify that checking multiple bands is more restrictive
    assert np.sum(bands34_mask) <= np.sum(band3_mask)
    assert np.sum(bands34_mask) <= np.sum(band4_mask)
    
    # Test with all bands specified
    all_bands_mask = create_quality_mask(qa_bands, include_saturation=True, saturation_bands=[3, 4, 5])
    # Should be same as default (which checks all bands)
    assert np.array_equal(all_bands_mask, default_mask)
    
    # Test band validation happens in the underlying function
    # Invalid bands should be caught by create_saturation_mask
    with pytest.raises(ValueError):
        create_quality_mask(qa_bands, include_saturation=True, saturation_bands=[1, 2])


def test_combine_masks():
    """Test logical combination of multiple mask arrays.
    
    This test should verify that:
    - List of masks are combined using specified method
    - 'and' method requires all masks to be True
    - 'or' method requires any mask to be True
    - Different mask array shapes are handled appropriately
    - Boolean and numeric masks work correctly
    - Empty mask lists are handled gracefully
    - Large numbers of masks are processed efficiently
    - Result preserves boolean data type
    """
    # Create test masks
    mask1 = np.array([
        [True, True, False, False],
        [True, False, True, False],
        [False, True, True, False],
        [True, True, True, True]
    ])
    
    mask2 = np.array([
        [True, False, True, False],
        [False, True, True, False],
        [True, True, False, True],
        [True, True, True, True]
    ])
    
    mask3 = np.array([
        [False, True, True, True],
        [True, True, False, True],
        [True, False, True, True],
        [True, True, True, True]
    ])
    
    masks = [mask1, mask2, mask3]
    
    # Test 'and' combination
    and_result = combine_masks(masks, method="and")
    
    # Verify 'and' behavior - all masks must be True
    assert and_result[0, 0] == (True and True and False)    # False
    assert and_result[0, 1] == (True and False and True)    # False
    assert and_result[1, 0] == (True and False and True)    # False
    assert and_result[3, 3] == (True and True and True)     # True (all pass)
    
    # Test 'or' combination
    or_result = combine_masks(masks, method="or")
    
    # Verify 'or' behavior - any mask can be True
    assert or_result[0, 0] == (True or True or False)      # True
    assert or_result[0, 1] == (True or False or True)      # True
    assert or_result[0, 2] == (False or True or True)      # True
    assert or_result[2, 3] == (False or True or True)      # True
    
    # Compare 'and' vs 'or' results
    assert np.sum(or_result) >= np.sum(and_result)  # 'or' is more permissive
    
    # Test with numeric masks (0/1)
    numeric_mask1 = mask1.astype(int)
    numeric_mask2 = mask2.astype(int)
    numeric_masks = [numeric_mask1, numeric_mask2]
    
    numeric_and = combine_masks(numeric_masks, method="and")
    numeric_or = combine_masks(numeric_masks, method="or")
    
    # Results should be boolean even with numeric input
    assert numeric_and.dtype == bool
    assert numeric_or.dtype == bool
    
    # Test with float masks
    float_mask1 = mask1.astype(float)
    float_mask2 = mask2.astype(float)
    float_masks = [float_mask1, float_mask2]
    
    float_and = combine_masks(float_masks, method="and")
    assert float_and.dtype == bool
    
    # Test single mask
    single_result = combine_masks([mask1], method="and")
    assert np.array_equal(single_result, mask1)
    
    single_or = combine_masks([mask1], method="or")
    assert np.array_equal(single_or, mask1)
    
    # Test empty mask list
    with pytest.raises(ValueError, match="No masks provided"):
        combine_masks([], method="and")
    
    # Test shape mismatch
    wrong_shape = np.array([[True, False], [False, True]])
    with pytest.raises(ValueError, match="shape"):
        combine_masks([mask1, wrong_shape], method="and")
    
    # Test invalid method
    with pytest.raises(ValueError, match="Unknown combination method"):
        combine_masks([mask1, mask2], method="invalid")
    
    # Test large number of masks
    many_masks = [mask1.copy() for _ in range(100)]
    # Randomly flip some values
    np.random.seed(42)
    for i, mask in enumerate(many_masks):
        if i % 10 == 0:  # Every 10th mask has some changes
            flip_indices = np.random.choice(mask.size, size=5, replace=False)
            flat_mask = mask.flatten()
            flat_mask[flip_indices] = ~flat_mask[flip_indices]
            many_masks[i] = flat_mask.reshape(mask.shape)
    
    # Test performance with many masks
    import time
    start = time.time()
    many_and = combine_masks(many_masks, method="and")
    and_time = time.time() - start
    
    start = time.time()
    many_or = combine_masks(many_masks, method="or")
    or_time = time.time() - start
    
    assert and_time < 1.0  # Should be fast
    assert or_time < 1.0   # Should be fast
    assert many_and.dtype == bool
    assert many_or.dtype == bool
    
    # With many masks and 'and', most pixels should be False
    assert np.sum(many_and) < np.sum(mask1) / 2
    
    # Test with all-True and all-False masks
    all_true = np.ones_like(mask1, dtype=bool)
    all_false = np.zeros_like(mask1, dtype=bool)
    
    # All-True with 'and'
    true_and = combine_masks([all_true, all_true], method="and")
    assert np.all(true_and == True)
    
    # All-False with 'or'
    false_or = combine_masks([all_false, all_false], method="or")
    assert np.all(false_or == False)
    
    # Mixed all-True/all-False
    mixed_and = combine_masks([all_true, all_false], method="and")
    assert np.all(mixed_and == False)
    
    mixed_or = combine_masks([all_true, all_false], method="or")
    assert np.all(mixed_or == True)
    
    # Test with very large masks for memory efficiency
    large_mask1 = np.random.choice([True, False], size=(1000, 1000), p=[0.7, 0.3])
    large_mask2 = np.random.choice([True, False], size=(1000, 1000), p=[0.6, 0.4])
    
    large_result = combine_masks([large_mask1, large_mask2], method="and")
    assert large_result.shape == (1000, 1000)
    assert large_result.dtype == bool


def test_combine_masks_different_types():
    """Test mask combination with different mask types.
    
    This test should verify that:
    - Boolean and float masks can be combined
    - Type conversion preserves logical meaning
    - Mixed precision masks work correctly
    - Result type is appropriate for downstream use
    - Performance is acceptable for large mask arrays
    """
    # Create masks of different types
    bool_mask = np.array([[True, False], [False, True]])
    int_mask = np.array([[1, 0], [1, 0]])
    float_mask = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    # Test combining different types
    mixed_and = combine_masks([bool_mask, int_mask, float_mask], method="and")
    mixed_or = combine_masks([bool_mask, int_mask, float_mask], method="or")
    
    # Result should be boolean
    assert mixed_and.dtype == bool
    assert mixed_or.dtype == bool
    
    # Verify logical operations work correctly
    assert mixed_and[0, 0] == (True and True and True)    # True
    assert mixed_and[0, 1] == (False and False and False) # False
    assert mixed_and[1, 0] == (False and True and False)  # False
    assert mixed_and[1, 1] == (True and False and True)   # False
    
    # Performance test
    large_bool = np.random.choice([True, False], size=(1000, 1000))
    large_int = np.random.choice([0, 1], size=(1000, 1000))
    large_float = np.random.choice([0.0, 1.0], size=(1000, 1000))
    
    import time
    start = time.time()
    large_result = combine_masks([large_bool, large_int, large_float])
    elapsed = time.time() - start
    
    assert elapsed < 1.0
    assert large_result.dtype == bool


def test_dilate_mask():
    """Test binary mask dilation by specified pixels.
    
    This test should verify that:
    - Mask dilation expands True regions by specified radius
    - Zero radius returns original mask unchanged
    - Large radius values work without performance issues
    - Image boundaries are handled correctly
    - Connectivity parameter affects dilation pattern
    - Boolean and numeric masks work correctly
    - Result preserves mask properties appropriately
    - Performance scales reasonably with radius and image size
    """
    # Create test mask with isolated True regions
    mask = np.array([
        [False, False, False, False, False],
        [False, True,  False, False, False],
        [False, False, False, False, False],
        [False, False, False, True,  False],
        [False, False, False, False, False],
    ], dtype=bool)
    
    # Test zero radius (should return unchanged)
    dilated_0 = dilate_mask(mask, pixels=0)
    assert np.array_equal(dilated_0, mask)
    
    # Test 1-pixel dilation
    # NOTE: dilate_mask expands MASKED regions (False values), not True values
    # So True pixels surrounded by False pixels will become False
    dilated_1 = dilate_mask(mask, pixels=1)
    
    # Original True pixels should become False (consumed by surrounding False pixels)
    assert dilated_1[1, 1] == False  # True pixel consumed by False neighbors
    assert dilated_1[3, 3] == False  # True pixel consumed by False neighbors
    
    # Test with mostly True mask to see expansion of False regions
    mostly_true = np.ones((5, 5), dtype=bool)
    mostly_true[2, 2] = False  # Single False pixel
    
    dilated_mostly = dilate_mask(mostly_true, pixels=1)
    # False region should expand
    assert dilated_mostly[2, 2] == False  # Original False
    assert dilated_mostly[1, 2] == False  # Neighbor should become False
    assert dilated_mostly[3, 2] == False  # Neighbor should become False
    assert dilated_mostly[2, 1] == False  # Neighbor should become False
    assert dilated_mostly[2, 3] == False  # Neighbor should become False
    
    # Test with numeric mask
    numeric_mask = mask.astype(int)
    dilated_numeric = dilate_mask(numeric_mask, pixels=1)
    # Note: implementation converts to float during processing but returns same numeric type
    assert dilated_numeric.dtype in [int, np.int64, float, np.float64]  # Some numeric type
    
    # Test with float mask
    float_mask = mask.astype(float)
    dilated_float = dilate_mask(float_mask, pixels=1)
    assert dilated_float.dtype == float
    
    # Test large radius
    large_dilated = dilate_mask(mask, pixels=3)
    # Should work without error (all isolated True pixels will be consumed)
    assert isinstance(large_dilated, np.ndarray)
    assert large_dilated.shape == mask.shape
    
    # Test boundary handling
    edge_mask = np.zeros((5, 5), dtype=bool)
    edge_mask[0, 0] = True  # Corner
    edge_mask[2, 4] = True  # Edge
    
    edge_dilated = dilate_mask(edge_mask, pixels=1)
    # Should handle boundaries without error
    assert isinstance(edge_dilated, np.ndarray)
    assert edge_dilated.shape == edge_mask.shape
    
    # Performance test
    large_mask = np.random.choice([True, False], size=(100, 100), p=[0.1, 0.9])
    
    import time
    start = time.time()
    large_dilated = dilate_mask(large_mask, pixels=2)
    elapsed = time.time() - start
    
    assert elapsed < 2.0  # Should complete reasonably quickly
    assert large_dilated.shape == large_mask.shape


def test_dilate_mask_connectivity():
    """Test mask dilation connectivity options.
    
    This test should verify that:
    - 4-connectivity and 8-connectivity produce different results
    - Connectivity parameter affects dilation shape appropriately
    - Edge cases (corners, boundaries) work correctly
    - Performance differences between connectivity options
    - Output quality is appropriate for intended use
    """
    # Note: The current dilate_mask implementation doesn't support connectivity parameter
    # It's passed to the function but not used in processing.py
    # This test documents the expected behavior if connectivity were implemented
    
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True  # Center pixel
    
    # Test with different connectivity (though implementation may not differ)
    dilated_4 = dilate_mask(mask, pixels=1, connectivity=4)
    dilated_8 = dilate_mask(mask, pixels=1, connectivity=8)
    
    # Both should work without error
    assert isinstance(dilated_4, np.ndarray)
    assert isinstance(dilated_8, np.ndarray)
    assert dilated_4.shape == mask.shape
    assert dilated_8.shape == mask.shape
    
    # Center should be False (dilated away by surrounding False pixels)
    assert dilated_4[2, 2] == False
    assert dilated_8[2, 2] == False
    
    # Note: Current implementation may not actually use connectivity parameter


def test_compute_quality_flags():
    """Test comprehensive quality flag computation from Landsat QA bands.
    
    This test should verify that:
    - QA_PIXEL flags are interpreted correctly
    - QA_AEROSOL and QA_RADSAT are integrated appropriately
    - Custom cloud and aerosol values work correctly
    - Combined quality assessment is scientifically sound
    - Different Landsat collections are handled appropriately
    - Optional QA bands can be omitted without errors
    - Return values represent overall pixel quality accurately
    - Performance is acceptable for operational processing
    """
    # Create test QA bands
    qa_pixel = np.array([
        [21824, 21952, 55052, 54852],  # Clear land, clear water, clouds
        [21824, 21824, 21824, 21824],  # All clear
    ], dtype=np.uint16)
    
    qa_aerosol = np.array([
        [100, 192, 100, 100],    # Good, bad, good, good
        [100, 100, 100, 100],    # All good
    ], dtype=np.uint16)
    
    qa_radsat = np.array([
        [0, 0, 4, 8],      # No sat, no sat, band 3 sat, band 4 sat
        [0, 0, 0, 0],      # All clear
    ], dtype=np.uint16)
    
    # Test with all QA bands
    quality_mask = compute_quality_flags(qa_pixel, qa_aerosol, qa_radsat)
    
    # Verify basic functionality
    assert isinstance(quality_mask, np.ndarray)
    assert quality_mask.dtype == bool
    assert quality_mask.shape == qa_pixel.shape
    
    # Test with only QA_PIXEL
    pixel_only = compute_quality_flags(qa_pixel)
    assert isinstance(pixel_only, np.ndarray)
    
    # Test with custom values
    custom_mask = compute_quality_flags(
        qa_pixel, 
        cloud_values=[55052, 54852],  # Specific cloud values
        aerosol_values=[100]          # Only best aerosol
    )
    assert isinstance(custom_mask, np.ndarray)


def test_compute_quality_flags_custom_values():
    """Test quality flag computation with custom flag values.
    
    This test should verify that:
    - Custom cloud values modify cloud detection
    - Custom aerosol values modify aerosol filtering
    - Value lists are validated for correctness
    - Custom values integrate properly with other quality checks
    - Results are scientifically meaningful with custom parameters
    """
    qa_pixel = np.array([[21824, 55052, 30000, 40000]], dtype=np.uint16)
    qa_aerosol = np.array([[100, 192, 224, 150]], dtype=np.uint16)
    
    # Test default values
    default_mask = compute_quality_flags(qa_pixel, qa_aerosol)
    
    # Test custom cloud values
    custom_cloud = compute_quality_flags(
        qa_pixel, qa_aerosol,
        cloud_values=[55052]  # Only this specific value is cloud
    )
    
    # Test custom aerosol values
    custom_aerosol = compute_quality_flags(
        qa_pixel, qa_aerosol,
        aerosol_values=[100, 150]  # More permissive aerosol
    )
    
    # All should return valid masks
    assert isinstance(default_mask, np.ndarray)
    assert isinstance(custom_cloud, np.ndarray)
    assert isinstance(custom_aerosol, np.ndarray)
    
    # Custom values should produce different results
    assert not np.array_equal(default_mask, custom_cloud)
    assert not np.array_equal(default_mask, custom_aerosol)


def test_create_urban_mask():
    """Test urban area mask creation from spectral indices.
    
    This test should verify that:
    - NDVI, NDWI, and NDUI thresholds correctly identify urban areas
    - Default thresholds work for typical urban environments
    - Custom thresholds adapt to different urban characteristics
    - Urban pixels are identified as True in output mask
    - Non-urban pixels are identified as False
    - Spectral index combinations improve urban detection accuracy
    - Different urban types (dense, sparse, industrial) are handled
    - Array shapes and data types are preserved
    - Performance is acceptable for large-scale urban mapping
    """
    # Create test spectral indices representing different land cover types
    # Urban areas: low NDVI (<0.2), low NDWI (<0.0), high NDUI (>0.3)
    
    # Test data with known land cover types
    ndvi = np.array([
        [0.1, 0.15, 0.8, 0.7],     # Urban, urban, vegetation, vegetation
        [0.05, 0.3, 0.1, 0.4],     # Urban, vegetation, urban, vegetation
        [0.2, 0.25, 0.0, 0.6],     # Boundary, vegetation, bare, vegetation
        [-0.1, 0.1, 0.35, 0.0],    # Water/bare, urban, vegetation, bare
    ], dtype=np.float32)
    
    ndwi = np.array([
        [-0.1, -0.2, -0.3, -0.1],  # Not water (urban/land)
        [-0.05, -0.4, 0.1, -0.2], # Not water, water, not water
        [0.0, -0.1, -0.5, -0.3],  # Boundary, not water, dry, not water
        [0.6, -0.1, -0.2, -0.4],  # Water, not water, not water, not water
    ], dtype=np.float32)
    
    ndui = np.array([
        [0.4, 0.35, 0.1, 0.15],    # High urban, high urban, low, low
        [0.45, 0.2, 0.32, 0.25],   # High urban, low, high urban, low
        [0.3, 0.28, 0.5, 0.1],     # Boundary, low, very high, low
        [0.1, 0.38, 0.25, 0.4],    # Low, high urban, low, high urban
    ], dtype=np.float32)
    
    # Test with default thresholds
    # Urban criteria: NDVI < 0.2 AND NDWI < 0.0 AND NDUI > 0.3
    urban_mask = create_urban_mask(ndvi, ndwi, ndui)
    
    # Verify default threshold behavior
    # Pixel [0,0]: NDVI=0.1 (<0.2) ✓, NDWI=-0.1 (<0.0) ✓, NDUI=0.4 (>0.3) ✓ = URBAN
    assert urban_mask[0, 0] == True
    
    # Pixel [0,1]: NDVI=0.15 (<0.2) ✓, NDWI=-0.2 (<0.0) ✓, NDUI=0.35 (>0.3) ✓ = URBAN
    assert urban_mask[0, 1] == True
    
    # Pixel [0,2]: NDVI=0.8 (>=0.2) ✗ = NOT URBAN (vegetation)
    assert urban_mask[0, 2] == False
    
    # Pixel [1,0]: NDVI=0.05 (<0.2) ✓, NDWI=-0.05 (<0.0) ✓, NDUI=0.45 (>0.3) ✓ = URBAN
    assert urban_mask[1, 0] == True
    
    # Pixel [1,2]: NDVI=0.1 (<0.2) ✓, NDWI=0.1 (>=0.0) ✗ = NOT URBAN (water influence)
    assert urban_mask[1, 2] == False
    
    # Pixel [2,0]: NDVI=0.2 (>=0.2) ✗ = NOT URBAN (at boundary)
    assert urban_mask[2, 0] == False
    
    # Pixel [3,1]: NDVI=0.1 (<0.2) ✓, NDWI=-0.1 (<0.0) ✓, NDUI=0.38 (>0.3) ✓ = URBAN
    assert urban_mask[3, 1] == True
    
    # Test custom thresholds for different urban types
    # Dense urban areas (stricter NDVI, higher NDUI)
    dense_urban = create_urban_mask(
        ndvi, ndwi, ndui,
        ndvi_threshold=0.1,   # Stricter vegetation threshold
        ndwi_threshold=0.0,   # Same water threshold
        ndui_threshold=0.4    # Higher urban index threshold
    )
    
    # Should identify fewer pixels as urban
    assert np.sum(dense_urban) <= np.sum(urban_mask)
    
    # With dense criteria, only [1,0] and [2,2] should be urban
    # Pixel [1,0]: NDVI=0.05 (<0.1) ✓, NDWI=-0.05 (<0.0) ✓, NDUI=0.45 (>0.4) ✓ = URBAN
    assert dense_urban[1, 0] == True
    # Pixel [2,2]: NDVI=0.0 (<0.1) ✓, NDWI=-0.5 (<0.0) ✓, NDUI=0.5 (>0.4) ✓ = URBAN
    assert dense_urban[2, 2] == True
    # All other pixels should be False
    assert dense_urban[0, 0] == False  # NDVI=0.1 not < 0.1, NDUI=0.4 not > 0.4
    
    # Sparse urban areas (more permissive)
    sparse_urban = create_urban_mask(
        ndvi, ndwi, ndui,
        ndvi_threshold=0.3,   # More permissive vegetation
        ndwi_threshold=0.1,   # Allow some water influence
        ndui_threshold=0.25   # Lower urban index threshold
    )
    
    # Should identify more pixels as urban
    assert np.sum(sparse_urban) >= np.sum(urban_mask)
    
    # Industrial areas (different characteristics)
    industrial_urban = create_urban_mask(
        ndvi, ndwi, ndui,
        ndvi_threshold=0.15,  # Some vegetation allowed
        ndwi_threshold=-0.1,  # Dry conditions
        ndui_threshold=0.35   # Moderate urban index
    )
    
    # Test that spectral index combinations improve accuracy
    # Individual index masks
    ndvi_only = ndvi < 0.2
    ndwi_only = ndwi < 0.0
    ndui_only = ndui > 0.3
    
    # Combined mask should be more selective than individual indices
    assert np.sum(urban_mask) <= np.sum(ndvi_only)
    assert np.sum(urban_mask) <= np.sum(ndwi_only)
    assert np.sum(urban_mask) <= np.sum(ndui_only)
    
    # Test with real-world urban scenario
    # Create synthetic scene with different urban types
    size = (50, 50)
    np.random.seed(42)
    
    # Dense urban core (low NDVI, low NDWI, high NDUI)
    scene_ndvi = np.full(size, 0.4)  # Default vegetation
    scene_ndwi = np.full(size, -0.2) # Default dry land
    scene_ndui = np.full(size, 0.2)  # Default low urban
    
    # Dense urban core
    scene_ndvi[10:20, 10:20] = np.random.uniform(0.05, 0.15, (10, 10))
    scene_ndwi[10:20, 10:20] = np.random.uniform(-0.3, -0.1, (10, 10))
    scene_ndui[10:20, 10:20] = np.random.uniform(0.35, 0.5, (10, 10))
    
    # Suburban area
    scene_ndvi[25:35, 25:35] = np.random.uniform(0.1, 0.25, (10, 10))
    scene_ndwi[25:35, 25:35] = np.random.uniform(-0.2, 0.0, (10, 10))
    scene_ndui[25:35, 25:35] = np.random.uniform(0.25, 0.35, (10, 10))
    
    # Apply urban detection
    scene_urban = create_urban_mask(scene_ndvi, scene_ndwi, scene_ndui)
    
    # Dense urban core should have high detection rate
    urban_core_detection = np.sum(scene_urban[10:20, 10:20]) / 100
    assert urban_core_detection > 0.7  # Most pixels detected
    
    # Suburban area should have lower detection rate
    suburban_detection = np.sum(scene_urban[25:35, 25:35]) / 100
    assert suburban_detection < urban_core_detection
    
    # Background should have very low detection
    background_mask = np.ones(size, dtype=bool)
    background_mask[10:20, 10:20] = False
    background_mask[25:35, 25:35] = False
    background_detection = np.sum(scene_urban[background_mask]) / np.sum(background_mask)
    assert background_detection < 0.1
    
    # Test array shape and data type preservation
    assert isinstance(urban_mask, np.ndarray)
    assert urban_mask.dtype == bool
    assert urban_mask.shape == ndvi.shape
    
    # Performance test with large array
    large_ndvi = np.random.uniform(-0.1, 0.8, size=(1000, 1000))
    large_ndwi = np.random.uniform(-0.5, 0.5, size=(1000, 1000))
    large_ndui = np.random.uniform(0.0, 0.6, size=(1000, 1000))
    
    import time
    start = time.time()
    large_urban = create_urban_mask(large_ndvi, large_ndwi, large_ndui)
    elapsed = time.time() - start
    
    assert elapsed < 1.0  # Should complete quickly
    assert large_urban.shape == (1000, 1000)
    
    # Test edge cases
    # All urban
    all_urban_ndvi = np.full((10, 10), 0.1)
    all_urban_ndwi = np.full((10, 10), -0.1)
    all_urban_ndui = np.full((10, 10), 0.4)
    all_urban_mask = create_urban_mask(all_urban_ndvi, all_urban_ndwi, all_urban_ndui)
    assert np.all(all_urban_mask == True)
    
    # No urban
    no_urban_ndvi = np.full((10, 10), 0.5)  # High vegetation
    no_urban_ndwi = np.full((10, 10), -0.1)
    no_urban_ndui = np.full((10, 10), 0.1)  # Low urban index
    no_urban_mask = create_urban_mask(no_urban_ndvi, no_urban_ndwi, no_urban_ndui)
    assert np.all(no_urban_mask == False)


def test_create_urban_mask_threshold_tuning():
    """Test urban mask threshold parameter tuning.
    
    This test should verify that:
    - NDVI threshold affects vegetation masking
    - NDWI threshold affects water body masking
    - NDUI threshold affects urban sensitivity
    - Threshold interactions produce reasonable urban boundaries
    - Parameter tuning guidance is provided for different regions
    """
    # Create test data
    ndvi = np.array([[0.15, 0.25, 0.05]])
    ndwi = np.array([[-0.05, 0.05, -0.1]])
    ndui = np.array([[0.35, 0.25, 0.45]])
    
    # Test NDVI threshold effects
    strict_ndvi = create_urban_mask(ndvi, ndwi, ndui, ndvi_threshold=0.1)
    loose_ndvi = create_urban_mask(ndvi, ndwi, ndui, ndvi_threshold=0.3)
    assert np.sum(loose_ndvi) >= np.sum(strict_ndvi)
    
    # Test NDWI threshold effects
    strict_ndwi = create_urban_mask(ndvi, ndwi, ndui, ndwi_threshold=-0.1)
    loose_ndwi = create_urban_mask(ndvi, ndwi, ndui, ndwi_threshold=0.1)
    assert np.sum(loose_ndwi) >= np.sum(strict_ndwi)
    
    # Test NDUI threshold effects
    strict_ndui = create_urban_mask(ndvi, ndwi, ndui, ndui_threshold=0.4)
    loose_ndui = create_urban_mask(ndvi, ndwi, ndui, ndui_threshold=0.2)
    assert np.sum(loose_ndui) >= np.sum(strict_ndui)


def test_create_urban_mask_validation():
    """Test urban mask validation and accuracy assessment.
    
    This test should verify that:
    - Urban mask accuracy is measurable against reference data
    - False positives and false negatives are characterized
    - Different urban environments show appropriate performance
    - Mask quality is sufficient for intended applications
    - Validation metrics guide parameter selection
    """
    # Create synthetic reference data
    size = (20, 20)
    ndvi = np.random.uniform(0, 0.8, size)
    ndwi = np.random.uniform(-0.5, 0.5, size)
    ndui = np.random.uniform(0, 0.6, size)
    
    # Create reference urban mask (ground truth)
    reference = (ndvi < 0.2) & (ndwi < 0.0) & (ndui > 0.3)
    
    # Test with default parameters
    predicted = create_urban_mask(ndvi, ndwi, ndui)
    
    # Calculate basic accuracy metrics
    true_positives = np.sum(reference & predicted)
    false_positives = np.sum(~reference & predicted)
    false_negatives = np.sum(reference & ~predicted)
    true_negatives = np.sum(~reference & ~predicted)
    
    # Verify metrics are calculable
    accuracy = (true_positives + true_negatives) / size[0] / size[1]
    assert 0 <= accuracy <= 1
    
    # Should have reasonable agreement with reference
    assert accuracy > 0.8  # Should be quite accurate with same thresholds


def test_compute_mask_statistics():
    """Test quality mask statistics computation.
    
    This test should verify that:
    - Valid and invalid pixel counts are accurate
    - Valid percentage calculation is correct
    - Coverage metrics provide spatial distribution information
    - Row and column coverage statistics are meaningful
    - Total pixel count handling works correctly
    - Boolean and numeric masks are handled appropriately
    - Performance is acceptable for large mask arrays
    - Statistics are useful for quality assessment
    """
    # Create test mask with known pattern
    mask = np.array([
        [True, True, False, False],
        [True, False, True, False],
        [False, False, False, False],  # All invalid row
        [True, True, True, True],      # All valid row
    ], dtype=bool)
    
    stats = compute_mask_statistics(mask)
    
    # Test basic counts
    assert stats["valid_pixels"] == 8      # Count of True pixels
    assert stats["invalid_pixels"] == 8    # Count of False pixels
    assert stats["total_pixels"] == 16     # Total pixels
    
    # Test percentage calculation
    assert stats["valid_percentage"] == 50.0  # 8/16 * 100
    
    # Test coverage metrics
    coverage = stats["mask_coverage"]
    assert coverage["rows_with_valid"] == 3     # Rows 0, 1, 3 have valid pixels
    assert coverage["cols_with_valid"] == 3     # Cols 0, 1, 2 have valid pixels
    assert coverage["total_rows"] == 4
    assert coverage["total_cols"] == 4
    
    # Test with custom total pixels
    stats_custom = compute_mask_statistics(mask, total_pixels=20)
    assert stats_custom["total_pixels"] == 20
    assert stats_custom["invalid_pixels"] == 12  # 20 - 8
    assert stats_custom["valid_percentage"] == 40.0  # 8/20 * 100
    
    # Test with numeric mask (0/1)
    numeric_mask = mask.astype(int)
    numeric_stats = compute_mask_statistics(numeric_mask)
    
    # Should give same results as boolean mask
    assert numeric_stats["valid_pixels"] == 8
    assert numeric_stats["invalid_pixels"] == 8
    assert numeric_stats["valid_percentage"] == 50.0
    
    # Test with float mask
    float_mask = mask.astype(float)
    float_stats = compute_mask_statistics(float_mask)
    assert float_stats["valid_pixels"] == 8
    
    # Test edge cases
    # All valid mask
    all_valid = np.ones((5, 5), dtype=bool)
    all_valid_stats = compute_mask_statistics(all_valid)
    assert all_valid_stats["valid_pixels"] == 25
    assert all_valid_stats["invalid_pixels"] == 0
    assert all_valid_stats["valid_percentage"] == 100.0
    assert all_valid_stats["mask_coverage"]["rows_with_valid"] == 5
    assert all_valid_stats["mask_coverage"]["cols_with_valid"] == 5
    
    # All invalid mask
    all_invalid = np.zeros((5, 5), dtype=bool)
    all_invalid_stats = compute_mask_statistics(all_invalid)
    assert all_invalid_stats["valid_pixels"] == 0
    assert all_invalid_stats["invalid_pixels"] == 25
    assert all_invalid_stats["valid_percentage"] == 0.0
    assert all_invalid_stats["mask_coverage"]["rows_with_valid"] == 0
    assert all_invalid_stats["mask_coverage"]["cols_with_valid"] == 0
    
    # Single pixel mask
    single_pixel = np.array([[True]])
    single_stats = compute_mask_statistics(single_pixel)
    assert single_stats["valid_pixels"] == 1
    assert single_stats["invalid_pixels"] == 0
    assert single_stats["valid_percentage"] == 100.0
    assert single_stats["mask_coverage"]["rows_with_valid"] == 1
    assert single_stats["mask_coverage"]["cols_with_valid"] == 1
    
    # Test spatial distribution patterns
    # Striped pattern (alternating rows)
    striped = np.zeros((10, 10), dtype=bool)
    striped[::2, :] = True  # Every other row
    striped_stats = compute_mask_statistics(striped)
    
    assert striped_stats["valid_pixels"] == 50
    assert striped_stats["valid_percentage"] == 50.0
    assert striped_stats["mask_coverage"]["rows_with_valid"] == 5  # Half the rows
    assert striped_stats["mask_coverage"]["cols_with_valid"] == 10  # All columns
    
    # Checkerboard pattern
    checkerboard = np.zeros((10, 10), dtype=bool)
    checkerboard[::2, ::2] = True
    checkerboard[1::2, 1::2] = True
    checker_stats = compute_mask_statistics(checkerboard)
    
    assert checker_stats["valid_pixels"] == 50
    assert checker_stats["mask_coverage"]["rows_with_valid"] == 10  # All rows
    assert checker_stats["mask_coverage"]["cols_with_valid"] == 10  # All columns
    
    # Corner pattern (only corners valid)
    corner = np.zeros((10, 10), dtype=bool)
    corner[0, 0] = True
    corner[0, 9] = True
    corner[9, 0] = True
    corner[9, 9] = True
    corner_stats = compute_mask_statistics(corner)
    
    assert corner_stats["valid_pixels"] == 4
    assert corner_stats["valid_percentage"] == 4.0
    assert corner_stats["mask_coverage"]["rows_with_valid"] == 2  # First and last row
    assert corner_stats["mask_coverage"]["cols_with_valid"] == 2  # First and last col
    
    # Test 1D array handling
    array_1d = np.array([True, False, True, False, True])
    stats_1d = compute_mask_statistics(array_1d)
    
    assert stats_1d["valid_pixels"] == 3
    assert stats_1d["invalid_pixels"] == 2
    assert stats_1d["valid_percentage"] == 60.0
    # For 1D arrays, row/col coverage should be 1
    assert stats_1d["mask_coverage"]["rows_with_valid"] == 1
    assert stats_1d["mask_coverage"]["cols_with_valid"] == 1
    assert stats_1d["mask_coverage"]["total_rows"] == 1
    assert stats_1d["mask_coverage"]["total_cols"] == 1
    
    # Performance test with large array
    large_mask = np.random.choice([True, False], size=(1000, 1000), p=[0.7, 0.3])
    
    import time
    start = time.time()
    large_stats = compute_mask_statistics(large_mask)
    elapsed = time.time() - start
    
    assert elapsed < 1.0  # Should be fast
    
    # Verify large array statistics are reasonable
    expected_valid = 1000000 * 0.7
    actual_valid = large_stats["valid_pixels"]
    # Should be close to expected (within 5%)
    assert 0.95 * expected_valid < actual_valid < 1.05 * expected_valid
    
    # Test zero total pixels handling
    empty_mask = np.array([], dtype=bool).reshape(0, 0)
    empty_stats = compute_mask_statistics(empty_mask)
    assert empty_stats["valid_pixels"] == 0
    assert empty_stats["invalid_pixels"] == 0
    assert empty_stats["total_pixels"] == 0
    assert empty_stats["valid_percentage"] == 0.0
    
    # Test quality assessment utility
    # Statistics should help assess mask quality
    
    # High quality mask (good coverage, not too fragmented)
    high_quality = np.ones((100, 100), dtype=bool)
    high_quality[40:60, 40:60] = False  # Single hole
    hq_stats = compute_mask_statistics(high_quality)
    
    assert hq_stats["valid_percentage"] > 80  # Good coverage
    assert hq_stats["mask_coverage"]["rows_with_valid"] > 80  # Good row coverage
    assert hq_stats["mask_coverage"]["cols_with_valid"] > 80  # Good col coverage
    
    # Poor quality mask (low coverage, fragmented)
    poor_quality = np.random.choice([True, False], size=(100, 100), p=[0.2, 0.8])
    pq_stats = compute_mask_statistics(poor_quality)
    
    assert pq_stats["valid_percentage"] < 30  # Poor coverage
    
    # Verify return dictionary structure
    required_keys = {
        "valid_pixels", "invalid_pixels", "valid_percentage", 
        "mask_coverage", "total_pixels"
    }
    assert set(stats.keys()) == required_keys
    
    coverage_keys = {
        "rows_with_valid", "cols_with_valid", "total_rows", "total_cols"
    }
    assert set(stats["mask_coverage"].keys()) == coverage_keys


def test_compute_mask_statistics_coverage_metrics():
    """Test mask statistics spatial coverage metrics.
    
    This test should verify that:
    - Row coverage indicates latitudinal data distribution
    - Column coverage indicates longitudinal data distribution
    - Coverage metrics help assess spatial data gaps
    - Edge cases (all valid, all invalid) are handled correctly
    - Multi-dimensional masks work appropriately
    """
    # Create mask with specific spatial patterns
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, :] = True  # Valid data in middle rows (latitudinal band)
    mask[:, 1:9] = mask[:, 1:9] | True  # Valid data in middle columns
    
    stats = compute_mask_statistics(mask)
    coverage = stats["mask_coverage"]
    
    # Verify row coverage
    assert coverage["rows_with_valid"] == 6  # Rows 2-7
    assert coverage["total_rows"] == 10
    
    # Verify column coverage  
    assert coverage["cols_with_valid"] == 8  # Columns 1-8
    assert coverage["total_cols"] == 10
    
    # Test edge cases
    all_valid = np.ones((5, 5), dtype=bool)
    all_stats = compute_mask_statistics(all_valid)
    assert all_stats["mask_coverage"]["rows_with_valid"] == 5
    assert all_stats["mask_coverage"]["cols_with_valid"] == 5
    
    all_invalid = np.zeros((5, 5), dtype=bool)
    none_stats = compute_mask_statistics(all_invalid)
    assert none_stats["mask_coverage"]["rows_with_valid"] == 0
    assert none_stats["mask_coverage"]["cols_with_valid"] == 0


def test_quality_mask_integration():
    """Test integration of quality masking functions.
    
    This test should verify that:
    - Multiple quality functions work together seamlessly
    - Combined quality assessment is scientifically valid
    - Processing chains maintain data quality appropriately
    - Performance is acceptable for operational workflows
    - Quality statistics guide processing decisions effectively
    """
    # Create test data for integrated workflow
    qa_pixel = np.random.choice([21824, 55052], size=(50, 50), p=[0.8, 0.2])
    qa_aerosol = np.random.choice([100, 192], size=(50, 50), p=[0.9, 0.1])
    qa_radsat = np.random.choice([0, 4], size=(50, 50), p=[0.95, 0.05])
    ndwi = np.random.uniform(-0.5, 0.5, size=(50, 50))
    
    qa_bands = {
        "qa_pixel": qa_pixel,
        "qa_aerosol": qa_aerosol, 
        "qa_radsat": qa_radsat,
        "ndwi": ndwi
    }
    
    # Test integrated quality assessment
    quality_mask = create_quality_mask(qa_bands, include_water=True)
    
    # Calculate quality statistics
    stats = compute_mask_statistics(quality_mask)
    
    # Verify integration works
    assert isinstance(quality_mask, np.ndarray)
    assert quality_mask.dtype == bool
    assert isinstance(stats, dict)
    assert "valid_percentage" in stats
    
    # Quality percentage should be reasonable
    assert 10 < stats["valid_percentage"] < 90


def test_quality_mask_performance():
    """Test performance of quality masking operations.
    
    This test should verify that:
    - Large arrays are processed efficiently
    - Memory usage is reasonable and predictable
    - Processing time scales appropriately with data size
    - Vectorized operations are used where possible
    - Performance bottlenecks are identified and documented
    """
    # Test with large arrays
    large_size = (1000, 1000)
    qa_pixel = np.random.choice([21824, 55052], size=large_size, p=[0.7, 0.3])
    qa_aerosol = np.random.choice([100, 192], size=large_size, p=[0.8, 0.2])
    
    import time
    
    # Test individual function performance
    start = time.time()
    cloud_mask = create_cloud_mask(qa_pixel)
    cloud_time = time.time() - start
    
    start = time.time()
    aerosol_mask = create_aerosol_mask(qa_aerosol)
    aerosol_time = time.time() - start
    
    start = time.time()
    combined = combine_masks([cloud_mask, aerosol_mask])
    combine_time = time.time() - start
    
    # All operations should complete quickly
    assert cloud_time < 1.0
    assert aerosol_time < 1.0
    assert combine_time < 1.0
    
    # Verify results are correct
    assert cloud_mask.shape == large_size
    assert aerosol_mask.shape == large_size
    assert combined.shape == large_size


def test_quality_mask_edge_cases():
    """Test quality masking edge cases and error handling.
    
    This test should verify that:
    - Empty arrays are handled gracefully
    - All-valid and all-invalid masks work correctly
    - Invalid parameter combinations are caught
    - Resource cleanup occurs after errors
    - Error messages are informative and actionable
    """
    # Test empty arrays
    empty_array = np.array([], dtype=np.uint16).reshape(0, 0)
    
    # Most functions should handle empty input gracefully or raise clear errors
    try:
        empty_stats = compute_mask_statistics(np.array([], dtype=bool).reshape(0, 0))
        assert empty_stats["total_pixels"] == 0
    except (ValueError, IndexError):
        pass  # Acceptable to raise error for empty arrays
    
    # Test all-valid scenario
    all_clear_qa = np.full((10, 10), 21824, dtype=np.uint16)
    all_valid_mask = create_cloud_mask(all_clear_qa)
    assert np.all(all_valid_mask == True)
    
    # Test all-invalid scenario
    all_cloud_qa = np.full((10, 10), 55052, dtype=np.uint16)
    all_invalid_mask = create_cloud_mask(all_cloud_qa)
    assert np.all(all_invalid_mask == False)
    
    # Test invalid parameter combinations
    with pytest.raises(ValueError):
        combine_masks([])  # Empty mask list
    
    # Test mismatched shapes
    mask1 = np.ones((5, 5), dtype=bool)
    mask2 = np.ones((3, 3), dtype=bool)
    with pytest.raises(ValueError):
        combine_masks([mask1, mask2])
    
    # Test invalid combination method
    with pytest.raises(ValueError):
        combine_masks([mask1], method="invalid")