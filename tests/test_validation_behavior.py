"""Behavioral tests for validation logic with controlled inputs.

These tests verify the validation functions work correctly with
predictable data, testing range checking, quality flag interpretation,
and threshold behaviors.
"""

import numpy as np
import pytest
from blackmarble.validation import (
    validate_reflectance_range,
    check_data_coverage,
    validate_index_range,
    validate_nighttime_lights,
    validate_road_coverage,
    validate_processing_inputs,
    generate_quality_report
)


class TestReflectanceRangeValidation:
    """Test reflectance range validation behavior."""
    
    def test_perfect_valid_data(self):
        """Test validation with all valid reflectance values."""
        # All values in perfect range [0.1, 0.9]
        data = np.array([[0.1, 0.3, 0.5], 
                        [0.7, 0.9, 0.2]])
        
        result = validate_reflectance_range(data, "B4", (0.0, 1.0))
        
        assert result["band"] == "B4"
        assert result["total_pixels"] == 6
        assert result["valid_pixels"] == 6
        assert result["valid_percentage"] == 100.0
        assert result["min_value"] == 0.1
        assert result["max_value"] == 0.9
        assert pytest.approx(result["mean_value"]) == 0.45
        assert result["out_of_range"] == 0
    
    def test_out_of_range_detection(self):
        """Test detection of out-of-range values."""
        # Mix of valid and invalid values
        data = np.array([[-0.1, 0.5, 1.2],  # 2 out of range
                        [0.3, 1.5, 0.8]])    # 1 out of range
        
        result = validate_reflectance_range(data, "B3", (0.0, 1.0))
        
        assert result["out_of_range"] == 3
        assert result["valid_pixels"] == 6  # All non-NaN
        assert result["min_value"] == -0.1
        assert result["max_value"] == 1.5
    
    def test_nan_handling(self):
        """Test validation with NaN values."""
        data = np.array([[0.5, np.nan, 0.3],
                        [np.nan, 0.7, np.nan]])
        
        result = validate_reflectance_range(data, "B5")
        
        assert result["total_pixels"] == 6
        assert result["valid_pixels"] == 3  # Only non-NaN
        assert result["valid_percentage"] == 50.0
        assert result["min_value"] == 0.3
        assert result["max_value"] == 0.7
        assert pytest.approx(result["mean_value"]) == 0.5
    
    def test_all_nan_data(self):
        """Test validation with all NaN values."""
        data = np.full((3, 3), np.nan)
        
        result = validate_reflectance_range(data, "B2")
        
        assert result["total_pixels"] == 9
        assert result["valid_pixels"] == 0
        assert result["valid_percentage"] == 0.0
        assert np.isnan(result["min_value"])
        assert np.isnan(result["max_value"])
        assert np.isnan(result["mean_value"])
        assert result["out_of_range"] == 0
    
    def test_custom_valid_range(self):
        """Test validation with custom valid range."""
        # Data with some high reflectance (snow/clouds)
        data = np.array([[0.8, 0.9, 1.1, 1.3, 1.5]])
        
        # Standard range [0, 1]
        result_standard = validate_reflectance_range(data, "B4", (0.0, 1.0))
        assert result_standard["out_of_range"] == 3  # 1.1, 1.3, 1.5
        
        # Extended range for snow/clouds
        result_extended = validate_reflectance_range(data, "B4", (0.0, 1.5))
        assert result_extended["out_of_range"] == 0  # All valid


class TestDataCoverageCheck:
    """Test data coverage validation behavior."""
    
    def test_full_coverage(self):
        """Test with complete data coverage."""
        data = np.ones((10, 10))
        
        is_valid, coverage = check_data_coverage(data, min_valid_percentage=50.0)
        
        assert is_valid == True
        assert coverage == 100.0
    
    def test_exact_threshold(self):
        """Test coverage at exact threshold."""
        # Create 60% valid data
        data = np.ones((10, 10))
        data[:4, :] = np.nan  # 40 NaN values
        
        # Should pass with 60% threshold
        is_valid, coverage = check_data_coverage(data, min_valid_percentage=60.0)
        assert is_valid == True
        assert coverage == 60.0
        
        # Should fail with 61% threshold
        is_valid, coverage = check_data_coverage(data, min_valid_percentage=61.0)
        assert is_valid == False
        assert coverage == 60.0
    
    def test_mask_based_coverage(self):
        """Test coverage calculation with explicit mask."""
        data = np.ones((5, 5))  # All valid data
        mask = np.array([[True, True, False, False, False],
                        [True, True, False, False, False],
                        [True, True, False, False, False],
                        [True, True, False, False, False],
                        [True, True, False, False, False]])
        
        is_valid, coverage = check_data_coverage(data, mask=mask)
        
        # 10 True values out of 25
        assert coverage == 40.0
    
    def test_sparse_data(self):
        """Test with very sparse data."""
        data = np.full((100, 100), np.nan)
        data[50, 50] = 1.0  # Single valid pixel
        
        is_valid, coverage = check_data_coverage(data, min_valid_percentage=0.01)
        
        assert is_valid == True
        assert coverage == 0.01  # 1/10000
        
        # Should fail with higher threshold
        is_valid, coverage = check_data_coverage(data, min_valid_percentage=0.02)
        assert is_valid == False


class TestIndexRangeValidation:
    """Test spectral index range validation."""
    
    def test_valid_ndvi_range(self):
        """Test validation of NDVI in correct range."""
        ndvi = np.array([[-1.0, -0.5, 0.0, 0.5, 1.0]])
        
        result = validate_index_range(ndvi, "NDVI", -1.0, 1.0)
        
        assert result["index"] == "NDVI"
        assert result["valid_pixels"] == 5
        assert result["min_value"] == -1.0
        assert result["max_value"] == 1.0
        assert result["out_of_range_count"] == 0
        assert result["out_of_range_percentage"] == 0.0
    
    def test_out_of_range_indices(self):
        """Test detection of indices outside theoretical range."""
        # NDVI with computational errors
        ndvi = np.array([[-1.1, -0.8, 0.0, 0.8, 1.1]])
        
        result = validate_index_range(ndvi, "NDVI", -1.0, 1.0)
        
        assert result["out_of_range_count"] == 2  # -1.1 and 1.1
        assert result["out_of_range_percentage"] == 40.0  # 2/5
        assert result["min_value"] == -1.1
        assert result["max_value"] == 1.1
    
    def test_mixed_valid_invalid(self):
        """Test with mix of valid, invalid, and NaN values."""
        data = np.array([[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, np.nan]])
        
        result = validate_index_range(data, "NDWI", -1.0, 1.0)
        
        assert result["valid_pixels"] == 7  # Excluding NaN
        assert result["total_pixels"] == 8
        assert result["out_of_range_count"] == 2  # -1.5 and 1.5
        assert result["out_of_range_percentage"] == pytest.approx(28.57, 0.01)  # 2/7
    
    def test_custom_index_range(self):
        """Test validation with non-standard index range."""
        # Some custom index with range [0, 2]
        custom_index = np.array([[0.0, 0.5, 1.0, 1.5, 2.0, 2.5]])
        
        result = validate_index_range(custom_index, "Custom", 0.0, 2.0)
        
        assert result["out_of_range_count"] == 1  # Only 2.5
        assert result["min_value"] == 0.0
        assert result["max_value"] == 2.5


class TestNighttimeLightsValidation:
    """Test nighttime lights validation behavior."""
    
    def test_urban_area_lights(self):
        """Test validation of typical urban area."""
        # Urban pattern: mix of dark and bright
        ntl = np.array([[0.0, 0.05, 0.1, 0.3],
                       [0.0, 0.2, 0.5, 0.7],
                       [0.05, 0.4, 0.8, 0.9],
                       [0.1, 0.3, 0.6, 1.0]])
        
        result = validate_nighttime_lights(ntl, expected_max=1.0, zero_threshold=0.1)
        
        assert result["total_pixels"] == 16
        assert result["valid_pixels"] == 16
        assert result["dark_pixels"] == 4  # Values < 0.1 (0.0, 0.05, 0.0, 0.05)
        assert result["dark_percentage"] == 25.0  # 4/16
        assert result["bright_pixels"] == 5  # Values > 0.5 (0.7, 0.8, 0.9, 0.6, 1.0)
        assert result["max_value"] == 1.0
        assert result["mean_brightness"] > 0.4  # Mean of non-dark pixels
    
    def test_rural_area_lights(self):
        """Test validation of rural/dark area."""
        # Mostly dark with few lights
        ntl = np.zeros((10, 10))
        ntl[5, 5] = 0.2  # Single light source
        ntl[7, 7] = 0.15
        
        result = validate_nighttime_lights(ntl, zero_threshold=0.1)
        
        assert result["dark_pixels"] == 98  # All but 2
        assert result["dark_percentage"] == 98.0
        assert result["bright_pixels"] == 0  # None > 0.5
        assert result["mean_brightness"] == 0.175  # (0.2 + 0.15) / 2
    
    def test_saturated_lights(self):
        """Test detection of saturated light values."""
        # Some pixels at maximum
        ntl = np.array([[0.8, 0.9, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 0.95]])
        
        result = validate_nighttime_lights(ntl, expected_max=1.0)
        
        assert result["max_value"] == 1.0
        assert result["bright_pixels"] == 8  # All > 0.5
        assert result["dark_pixels"] == 0
        
        # Count how many are at maximum
        at_max = np.sum(ntl == 1.0)
        assert at_max == 5
    
    def test_all_dark_area(self):
        """Test completely dark area."""
        ntl = np.full((5, 5), 0.05)  # All below threshold
        
        result = validate_nighttime_lights(ntl, zero_threshold=0.1)
        
        assert result["dark_pixels"] == 25
        assert result["dark_percentage"] == 100.0
        assert result["mean_brightness"] == 0.0  # No non-dark pixels
    
    def test_nan_in_lights(self):
        """Test NaN handling in nighttime lights."""
        ntl = np.array([[0.5, np.nan, 0.7],
                       [np.nan, 0.0, np.nan]])
        
        result = validate_nighttime_lights(ntl)
        
        assert result["total_pixels"] == 6
        assert result["valid_pixels"] == 3
        assert result["dark_pixels"] == 1  # The 0.0 value
        assert result["mean_brightness"] == 0.6  # (0.5 + 0.7) / 2


class TestRoadCoverageValidation:
    """Test road network coverage validation."""
    
    def test_typical_urban_roads(self):
        """Test validation of typical urban road coverage."""
        # Create realistic urban pattern (5-15% roads)
        roads = np.zeros((100, 100), dtype=np.uint8)
        # Add some roads
        roads[45:55, :] = 1  # Major road across
        roads[:, 45:55] = 2  # Another major road
        roads[20:25, :] = 3  # Secondary road
        roads[:, 75:78] = 4  # Minor road
        
        result = validate_road_coverage(roads, min_road_percentage=5.0, max_road_percentage=30.0)
        
        assert result["total_pixels"] == 10000
        assert result["is_valid"] == True
        
        # Check road types are counted correctly
        assert "type_1" in result["road_types"]
        assert "type_2" in result["road_types"]
        assert "type_3" in result["road_types"]
        assert "type_4" in result["road_types"]
    
    def test_no_roads(self):
        """Test validation with no roads."""
        roads = np.zeros((50, 50), dtype=np.uint8)
        
        result = validate_road_coverage(roads, min_road_percentage=0.1)
        
        assert result["road_pixels"] == 0
        assert result["road_percentage"] == 0.0
        assert result["is_valid"] == False  # Below minimum
        assert result["road_types"] == {}
    
    def test_excessive_roads(self):
        """Test detection of excessive road coverage."""
        # Unrealistic: 60% roads
        roads = np.ones((10, 10), dtype=np.uint8)
        roads[:4, :] = 0  # 40% no roads
        
        result = validate_road_coverage(roads, max_road_percentage=50.0)
        
        assert result["road_percentage"] == 60.0
        assert result["is_valid"] == False  # Above maximum
    
    def test_road_type_distribution(self):
        """Test counting of different road types."""
        roads = np.array([[1, 1, 2, 2, 3],
                         [1, 1, 2, 2, 3],
                         [4, 4, 5, 5, 0],
                         [4, 4, 5, 5, 0]])
        
        result = validate_road_coverage(roads)
        
        assert result["road_types"]["type_1"] == 4
        assert result["road_types"]["type_2"] == 4
        assert result["road_types"]["type_3"] == 2
        assert result["road_types"]["type_4"] == 4
        assert result["road_types"]["type_5"] == 4
        assert result["road_pixels"] == 18  # Total non-zero
    
    def test_boundary_thresholds(self):
        """Test exact threshold boundaries."""
        # Create exactly 10% road coverage
        roads = np.zeros((10, 10), dtype=np.uint8)
        roads[:1, :] = 1  # 10 pixels = 10%
        
        # Should pass at 10% threshold
        result = validate_road_coverage(roads, min_road_percentage=10.0, max_road_percentage=10.0)
        assert result["is_valid"] == True
        assert result["road_percentage"] == 10.0
        
        # Should fail just above/below
        result = validate_road_coverage(roads, min_road_percentage=10.1)
        assert result["is_valid"] == False
        
        result = validate_road_coverage(roads, max_road_percentage=9.9)
        assert result["is_valid"] == False


class TestProcessingInputsValidation:
    """Test validation of processing inputs."""
    
    def test_all_inputs_valid(self):
        """Test when all required inputs are present and valid."""
        bands = {
            "B3": np.random.rand(100, 100) * 0.3,  # Green
            "B4": np.random.rand(100, 100) * 0.2,  # Red  
            "B5": np.random.rand(100, 100) * 0.5,  # NIR
        }
        
        is_valid, errors = validate_processing_inputs(bands)
        
        assert is_valid == True
        assert errors == []
    
    def test_missing_bands(self):
        """Test detection of missing required bands."""
        bands = {
            "B3": np.ones((10, 10)),
            "B4": np.ones((10, 10)),
            # B5 is missing
        }
        
        is_valid, errors = validate_processing_inputs(bands)
        
        assert is_valid == False
        assert len(errors) == 1
        assert "Missing required band: B5" in errors[0]
    
    def test_insufficient_coverage(self):
        """Test detection of insufficient data coverage."""
        # B4 has only 5% valid data
        bands = {
            "B3": np.ones((10, 10)),
            "B4": np.full((10, 10), np.nan),
            "B5": np.ones((10, 10)),
        }
        bands["B4"][0, :5] = 0.5  # Only 5 valid pixels
        
        is_valid, errors = validate_processing_inputs(bands, min_coverage=10.0)
        
        assert is_valid == False
        assert len(errors) == 1
        assert "B4 has insufficient coverage" in errors[0]
        assert "5.0%" in errors[0]
    
    def test_all_nan_band(self):
        """Test detection of completely invalid band."""
        bands = {
            "B3": np.ones((5, 5)),
            "B4": np.full((5, 5), np.nan),  # All NaN
            "B5": np.ones((5, 5)),
        }
        
        is_valid, errors = validate_processing_inputs(bands)
        
        assert is_valid == False
        assert len(errors) == 2  # Insufficient coverage AND all NaN
        assert any("B4 contains only NaN values" in e for e in errors)
    
    def test_multiple_issues(self):
        """Test detection of multiple validation issues."""
        bands = {
            "B3": np.full((10, 10), np.nan),  # All NaN
            # B4 missing
            "B5": np.ones((10, 10)),
        }
        bands["B3"][:1, :] = 0.5  # Only 10% valid
        
        is_valid, errors = validate_processing_inputs(bands, min_coverage=20.0)
        
        assert is_valid == False
        assert len(errors) >= 2
        assert any("Missing required band: B4" in e for e in errors)
        assert any("B3 has insufficient coverage" in e for e in errors)
    
    def test_custom_required_bands(self):
        """Test validation with custom band requirements."""
        bands = {
            "B2": np.ones((10, 10)),
            "B3": np.ones((10, 10)),
            "B4": np.ones((10, 10)),
        }
        
        # Default requirements should fail (missing B5)
        is_valid, _ = validate_processing_inputs(bands)
        assert is_valid == False
        
        # Custom requirements should pass
        is_valid, errors = validate_processing_inputs(
            bands, 
            required_bands=["B2", "B3", "B4"]
        )
        assert is_valid == True


class TestQualityReportGeneration:
    """Test comprehensive quality report generation."""
    
    def test_basic_report_structure(self):
        """Test basic quality report structure."""
        data = {
            "ndvi": np.array([[0.1, 0.5], [0.7, 0.3]]),
            "ntl": np.array([[0.0, 0.2], [0.8, 1.0]]),
        }
        
        report = generate_quality_report(data)
        
        assert "ndvi" in report
        assert "ntl" in report
        
        # Check NDVI statistics
        ndvi_stats = report["ndvi"]
        assert ndvi_stats["shape"] == (2, 2)
        assert ndvi_stats["dtype"] == "float64"
        assert ndvi_stats["total_pixels"] == 4
        assert ndvi_stats["valid_pixels"] == 4
        assert ndvi_stats["valid_percentage"] == 100.0
        assert ndvi_stats["min"] == 0.1
        assert ndvi_stats["max"] == 0.7
        assert pytest.approx(ndvi_stats["mean"]) == 0.4
    
    def test_report_with_nan_values(self):
        """Test report generation with NaN values."""
        data = {
            "data": np.array([[1.0, np.nan], [3.0, np.nan]])
        }
        
        report = generate_quality_report(data)
        
        stats = report["data"]
        assert stats["valid_pixels"] == 2
        assert stats["valid_percentage"] == 50.0
        assert stats["min"] == 1.0
        assert stats["max"] == 3.0
        assert stats["mean"] == 2.0
    
    def test_report_with_missing_data(self):
        """Test report with None/missing data."""
        data = {
            "present": np.ones((3, 3)),
            "missing": None,
        }
        
        report = generate_quality_report(data)
        
        assert report["present"]["valid_pixels"] == 9
        assert report["missing"]["status"] == "missing"
    
    def test_report_unique_values(self):
        """Test counting of unique values."""
        data = {
            "categorical": np.array([[1, 1, 2], [2, 3, 3]]),
            "continuous": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        }
        
        report = generate_quality_report(data)
        
        assert report["categorical"]["unique_values"] == 3  # 1, 2, 3
        assert report["continuous"]["unique_values"] == 6  # All different
    
    def test_empty_array_report(self):
        """Test report generation for empty arrays."""
        data = {
            "empty": np.array([]),
        }
        
        report = generate_quality_report(data)
        
        stats = report["empty"]
        assert stats["total_pixels"] == 0
        assert stats["valid_pixels"] == 0
        # With empty array, percentage calculation gives NaN (0/0)
        assert np.isnan(stats["valid_percentage"])