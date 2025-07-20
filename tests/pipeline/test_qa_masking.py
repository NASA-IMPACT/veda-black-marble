"""Test QA masking for Landsat data quality control.

The pipeline uses QA_PIXEL band to mask clouds, shadows, and other
artifacts to ensure high-quality data for analysis.
"""

import numpy as np
import pytest

from blackmarble.prepare.landsat_qa import (
    create_landsat_cloud_mask,
    print_qa_pixel_summary
)


class TestQAMasking:
    """Test QA masking functionality used in the pipeline."""
    
    def test_qa_pixel_bit_interpretation(self):
        """Test correct interpretation of QA_PIXEL bits."""
        # Landsat Collection 2 QA_PIXEL bit definitions:
        # Bit 0: Fill
        # Bit 1: Dilated Cloud  
        # Bit 2: Cirrus (high-altitude clouds)
        # Bit 3: Cloud
        # Bit 4: Cloud Shadow
        # Bit 5: Snow
        # Bit 6-7: Clear confidence (0=No, 1=Low, 2=Reserved, 3=High)
        # Bit 8-9: Cloud confidence
        # Bit 10-11: Cloud shadow confidence
        # Bit 12-13: Snow/Ice confidence
        # Bit 14-15: Cirrus confidence
        
        shape = (10, 10)
        qa = np.zeros(shape, dtype=np.uint16)
        
        # Set specific bits for testing
        qa[0, 0] = 1 << 0  # Fill (bit 0)
        qa[1, 0] = 1 << 1  # Dilated cloud (bit 1)
        qa[2, 0] = 1 << 2  # Cirrus (bit 2)
        qa[3, 0] = 1 << 3  # Cloud (bit 3)
        qa[4, 0] = 1 << 4  # Cloud shadow (bit 4)
        qa[5, 0] = 1 << 5  # Snow (bit 5)
        qa[6, 0] = 21824  # Clear land (typical value)
        qa[7, 0] = 22080  # Cloud (21824 + 256)
        
        # Get mask using moderate strategy (default)
        mask = create_landsat_cloud_mask(qa, strategy="moderate")
        
        # Check masking decisions (True = clear, False = masked)
        assert not mask[0, 0], "Fill pixels should be masked"
        assert not mask[1, 0], "Dilated cloud should be masked"
        assert not mask[2, 0], "Cirrus should be masked in moderate strategy"
        assert not mask[3, 0], "Cloud should be masked"
        assert not mask[4, 0], "Cloud shadow should be masked"
        assert mask[5, 0], "Snow should NOT be masked (valid data)"
        assert mask[6, 0], "Clear land should NOT be masked"
        assert not mask[7, 0], "Cloudy pixel should be masked"
    
    def test_qa_masking_strategies(self):
        """Test different QA masking strategies."""
        shape = (20, 20)
        
        # Create QA with various conditions
        qa = np.full(shape, 21824, dtype=np.uint16)  # Clear land base
        
        # Add different cloud confidence levels
        qa[0:5, :] |= (1 << 3)  # Cloud bit
        qa[5:10, :] |= (1 << 2)  # Cirrus bit
        qa[10:15, :] |= (1 << 9)  # Low cloud confidence
        qa[15:20, :] |= (3 << 8)  # High cloud confidence
        
        # Test conservative strategy (most aggressive masking)
        mask_conservative = create_landsat_cloud_mask(qa, strategy="conservative")
        assert not np.any(mask_conservative[0:5, :]), "Clouds masked"
        assert not np.any(mask_conservative[5:10, :]), "Cirrus masked in conservative"
        assert not np.any(mask_conservative[10:15, :]), "Low confidence masked in conservative"
        assert not np.any(mask_conservative[15:20, :]), "High confidence masked"
        
        # Test moderate strategy (balanced)
        mask_moderate = create_landsat_cloud_mask(qa, strategy="moderate")
        assert not np.any(mask_moderate[0:5, :]), "Clouds masked"
        assert not np.any(mask_moderate[5:10, :]), "Cirrus masked in moderate"
        assert np.all(mask_moderate[10:15, 0] == True), "Low confidence NOT masked in moderate"
        assert not np.any(mask_moderate[15:20, :]), "High confidence masked"
        
        # Test permissive strategy (least aggressive)
        mask_permissive = create_landsat_cloud_mask(qa, strategy="permissive")
        assert not np.any(mask_permissive[0:5, :]), "Clouds still masked"
        assert np.all(mask_permissive[5:10, :]), "Cirrus NOT masked in permissive"
        assert np.all(mask_permissive[10:15, :]), "Low confidence NOT masked"
        assert not np.any(mask_permissive[15:20, :]), "High confidence still masked"
    
    def test_cloud_mask_dilation(self):
        """Test cloud mask dilation to handle cloud edges."""
        shape = (30, 30)
        qa = np.full(shape, 21824, dtype=np.uint16)  # Clear
        
        # Add single cloud pixel
        qa[15, 15] = 21824 | (1 << 3)  # Cloud bit
        
        # Test without dilation
        mask_no_dilate = create_landsat_cloud_mask(qa, dilate_pixels=0)
        masked_count_no_dilate = np.sum(~mask_no_dilate)
        assert masked_count_no_dilate == 1, "Only cloud pixel masked"
        
        # Test with dilation
        mask_dilated = create_landsat_cloud_mask(qa, dilate_pixels=3)
        masked_count_dilated = np.sum(~mask_dilated)
        assert masked_count_dilated > 1, "Dilation should mask more pixels"
        assert masked_count_dilated <= 49, "Dilation shouldn't mask too many pixels"
        
        # Check dilation pattern (should be roughly circular)
        assert not mask_dilated[14, 15], "Adjacent pixels should be masked"
        assert not mask_dilated[15, 14], "Adjacent pixels should be masked"
        assert not mask_dilated[16, 15], "Adjacent pixels should be masked"
        assert not mask_dilated[15, 16], "Adjacent pixels should be masked"
    
    def test_qa_summary_output(self, capsys):
        """Test QA pixel summary statistics output."""
        shape = (100, 100)
        qa = np.full(shape, 21824, dtype=np.uint16)  # Clear
        
        # Add various conditions
        qa[0:20, :] = 21824 | (1 << 3)   # 20% clouds
        qa[20:30, :] = 21824 | (1 << 4)  # 10% shadows
        qa[30:35, :] = 21824 | (1 << 2)  # 5% cirrus
        qa[35:36, :] = 1                  # 1% fill
        
        # Print summary
        print_qa_pixel_summary(qa)
        
        # Check output
        captured = capsys.readouterr()
        assert "Clear pixels:" in captured.out
        assert "Cloud pixels:" in captured.out
        assert "Shadow pixels:" in captured.out
        assert "20.0%" in captured.out  # Cloud percentage
        assert "10.0%" in captured.out  # Shadow percentage
    
    def test_combined_qa_conditions(self):
        """Test QA pixels with multiple conditions."""
        shape = (5, 5)
        qa = np.full(shape, 21824, dtype=np.uint16)  # Clear
        
        # Pixel with cloud AND shadow
        qa[0, 0] = 21824 | (1 << 3) | (1 << 4)
        
        # Pixel with cirrus AND high cloud confidence
        qa[1, 0] = 21824 | (1 << 2) | (3 << 8)
        
        # Pixel with snow AND shadow (snow is valid but shadow masks it)
        qa[2, 0] = 21824 | (1 << 5) | (1 << 4)
        
        mask = create_landsat_cloud_mask(qa, strategy="moderate")
        
        assert not mask[0, 0], "Cloud+shadow should be masked"
        assert not mask[1, 0], "Cirrus with high cloud confidence should be masked"
        assert not mask[2, 0], "Shadow masks even valid snow pixels"
    
    def test_qa_mask_preserves_shape(self):
        """Test that QA mask preserves data shape."""
        shapes = [(10, 10), (20, 30), (100, 100), (7, 13)]
        
        for shape in shapes:
            qa = np.full(shape, 21824, dtype=np.uint16)
            mask = create_landsat_cloud_mask(qa)
            
            assert mask.shape == shape, f"Mask shape {mask.shape} != input shape {shape}"
            assert mask.dtype == bool, "Mask should be boolean"
    
    def test_qa_mask_edge_values(self):
        """Test QA masking with edge case QA values."""
        shape = (10, 10)
        
        # Test with all zeros (should mask everything)
        qa_zeros = np.zeros(shape, dtype=np.uint16)
        mask_zeros = create_landsat_cloud_mask(qa_zeros)
        assert not np.any(mask_zeros), "All zeros should be masked"
        
        # Test with max uint16 (all bits set)
        qa_max = np.full(shape, 65535, dtype=np.uint16)
        mask_max = create_landsat_cloud_mask(qa_max)
        assert not np.any(mask_max), "All bits set should be masked"
        
        # Test with typical clear value
        qa_clear = np.full(shape, 21824, dtype=np.uint16)
        mask_clear = create_landsat_cloud_mask(qa_clear)
        assert np.all(mask_clear), "Clear land should not be masked"


class TestQAMaskingIntegration:
    """Test QA masking in pipeline context."""
    
    def test_qa_masking_for_temporal_composite(self):
        """Test QA masking prepares data for temporal compositing."""
        shape = (30, 30)
        
        # Create QA for multiple dates
        qa_dates = []
        
        # Date 1: Clouds in upper left
        qa1 = np.full(shape, 21824, dtype=np.uint16)
        qa1[0:10, 0:10] = 21824 | (1 << 3)  # Cloud
        qa_dates.append(qa1)
        
        # Date 2: Clouds in lower right
        qa2 = np.full(shape, 21824, dtype=np.uint16)
        qa2[20:30, 20:30] = 21824 | (1 << 3)  # Cloud
        qa_dates.append(qa2)
        
        # Date 3: Some shadows
        qa3 = np.full(shape, 21824, dtype=np.uint16)
        qa3[10:15, 10:15] = 21824 | (1 << 4)  # Shadow
        qa_dates.append(qa3)
        
        # Create masks for each date
        masks = [create_landsat_cloud_mask(qa) for qa in qa_dates]
        
        # Simulate band data and masking
        band_data = np.full(shape, 10000, dtype=np.float32)
        masked_bands = []
        
        for mask in masks:
            masked = band_data.copy()
            masked[~mask] = np.nan  # Set masked pixels to NaN
            masked_bands.append(masked)
        
        # Stack for compositing
        stack = np.stack(masked_bands, axis=0)
        
        # Compute median (simulating temporal composite)
        composite = np.nanmedian(stack, axis=0)
        
        # Check that different areas have different numbers of valid observations
        # Upper left had clouds in date 1
        assert np.sum(~np.isnan(stack[:, 5, 5])) == 2
        # Lower right had clouds in date 2
        assert np.sum(~np.isnan(stack[:, 25, 25])) == 2
        # Center had shadows in date 3
        assert np.sum(~np.isnan(stack[:, 12, 12])) == 2
        # Clear area has all dates
        assert np.sum(~np.isnan(stack[:, 0, 25])) == 3
    
    def test_consistent_masking_across_bands(self):
        """Test that QA mask is applied consistently to all bands."""
        shape = (20, 20)
        
        # Create QA with mixed conditions
        qa = np.full(shape, 21824, dtype=np.uint16)
        qa[5:10, 5:10] = 21824 | (1 << 3)  # Cloud
        qa[10:15, 10:15] = 21824 | (1 << 4)  # Shadow
        
        # Create mask once
        mask = create_landsat_cloud_mask(qa)
        
        # Apply to multiple bands (simulating pipeline)
        bands = ['B3', 'B4', 'B5']
        masked_bands = {}
        
        for band in bands:
            # Simulate band data
            data = np.random.randint(8000, 12000, shape, dtype=np.uint16)
            # Apply mask
            masked = data.astype(np.float32)
            masked[~mask] = np.nan
            masked_bands[band] = masked
        
        # Verify all bands have same NaN pattern
        for i in range(1, len(bands)):
            nan_mask_1 = np.isnan(masked_bands[bands[i-1]])
            nan_mask_2 = np.isnan(masked_bands[bands[i]])
            assert np.array_equal(nan_mask_1, nan_mask_2), \
                f"NaN pattern differs between {bands[i-1]} and {bands[i]}"