"""Tests for validation functions."""

import numpy as np
import pytest
from blackmarble.validation import (
    validate_reflectance_range,
    check_data_coverage,
    validate_index_range,
    validate_nighttime_lights,
    validate_road_coverage,
    validate_processing_inputs,
    generate_quality_report,
)


class TestValidateReflectanceRange:
    """Test reflectance validation function."""

    def test_valid_reflectance(self):
        """Test validation of valid reflectance data."""
        data = np.array([[0.1, 0.3], [0.5, 0.8]])

        result = validate_reflectance_range(data, "B4")

        assert result["band"] == "B4"
        assert result["total_pixels"] == 4
        assert result["valid_pixels"] == 4
        assert result["valid_percentage"] == 100.0
        assert result["min_value"] == 0.1
        assert result["max_value"] == 0.8
        assert result["out_of_range"] == 0

    def test_reflectance_with_nan(self):
        """Test validation with NaN values."""
        data = np.array([[0.1, np.nan], [0.5, 0.8]])

        result = validate_reflectance_range(data, "B3")

        assert result["valid_pixels"] == 3
        assert result["valid_percentage"] == 75.0
        assert result["min_value"] == 0.1
        assert result["max_value"] == 0.8

    def test_out_of_range_values(self):
        """Test detection of out-of-range values."""
        data = np.array([[0.1, 1.2], [-0.1, 0.5]])

        result = validate_reflectance_range(data, "B5", valid_range=(0.0, 1.0))

        assert result["out_of_range"] == 2  # 1.2 and -0.1
        assert result["min_value"] == -0.1
        assert result["max_value"] == 1.2

    def test_empty_array(self):
        """Test validation of all-NaN array."""
        data = np.array([[np.nan, np.nan]])

        result = validate_reflectance_range(data, "B4")

        assert result["valid_pixels"] == 0
        assert result["valid_percentage"] == 0.0
        assert np.isnan(result["min_value"])
        assert np.isnan(result["max_value"])


class TestCheckDataCoverage:
    """Test data coverage checking function."""

    def test_sufficient_coverage(self):
        """Test data with sufficient coverage."""
        data = np.array([[1, 2, np.nan], [4, 5, 6]])

        is_valid, percentage = check_data_coverage(data, min_valid_percentage=50.0)

        assert is_valid == True
        assert percentage == pytest.approx(83.33, 0.01)

    def test_insufficient_coverage(self):
        """Test data with insufficient coverage."""
        data = np.array([[np.nan, np.nan], [1, np.nan]])

        is_valid, percentage = check_data_coverage(data, min_valid_percentage=50.0)

        assert is_valid == False
        assert percentage == 25.0

    def test_with_mask(self):
        """Test coverage check with explicit mask."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        mask = np.array([[True, True, False], [True, False, False]])

        is_valid, percentage = check_data_coverage(data, mask=mask)

        assert percentage == 50.0

    def test_all_valid(self):
        """Test data with 100% coverage."""
        data = np.array([[1, 2], [3, 4]])

        is_valid, percentage = check_data_coverage(data)

        assert is_valid == True
        assert percentage == 100.0


class TestValidateIndexRange:
    """Test spectral index validation function."""

    def test_valid_ndvi(self):
        """Test validation of valid NDVI values."""
        ndvi = np.array([[-0.5, 0.0], [0.4, 0.8]])

        result = validate_index_range(ndvi, "NDVI")

        assert result["index"] == "NDVI"
        assert result["valid_pixels"] == 4
        assert result["min_value"] == -0.5
        assert result["max_value"] == 0.8
        assert result["out_of_range_count"] == 0
        assert result["out_of_range_percentage"] == 0.0

    def test_out_of_range_index(self):
        """Test detection of out-of-range index values."""
        # NDVI should be in [-1, 1] but we have 1.2
        ndvi = np.array([[-0.5, 0.8], [1.2, np.nan]])

        result = validate_index_range(ndvi, "NDVI", -1.0, 1.0)

        assert result["valid_pixels"] == 3
        assert result["out_of_range_count"] == 1
        assert result["out_of_range_percentage"] == pytest.approx(33.33, 0.01)

    def test_custom_range(self):
        """Test validation with custom expected range."""
        data = np.array([[0.0, 0.5], [0.8, 1.0]])

        result = validate_index_range(data, "NDUI", 0.0, 1.0)

        assert result["out_of_range_count"] == 0


class TestValidateNighttimeLights:
    """Test nighttime lights validation function."""

    def test_normal_ntl(self):
        """Test validation of normal NTL data."""
        ntl = np.array([[0.0, 0.05], [0.3, 0.8]])

        result = validate_nighttime_lights(ntl)

        assert result["total_pixels"] == 4
        assert result["valid_pixels"] == 4
        assert result["dark_pixels"] == 2  # 0.0 and 0.05
        assert result["dark_percentage"] == 50.0
        assert result["bright_pixels"] == 1  # 0.8
        assert result["max_value"] == 0.8

    def test_all_dark(self):
        """Test validation of all-dark NTL data."""
        ntl = np.array([[0.0, 0.01], [0.05, 0.08]])

        result = validate_nighttime_lights(ntl, zero_threshold=0.1)

        assert result["dark_pixels"] == 4
        assert result["dark_percentage"] == 100.0
        assert result["mean_brightness"] == 0.0  # No non-dark pixels

    def test_with_nan(self):
        """Test NTL validation with NaN values."""
        ntl = np.array([[0.5, np.nan], [0.8, 0.2]])

        result = validate_nighttime_lights(ntl)

        assert result["valid_pixels"] == 3
        assert result["dark_pixels"] == 0
        assert result["mean_brightness"] > 0


class TestValidateRoadCoverage:
    """Test road coverage validation function."""

    def test_normal_road_coverage(self):
        """Test validation of normal road coverage."""
        roads = np.array([[0, 1, 0], [2, 0, 3], [0, 0, 0]])

        result = validate_road_coverage(roads)

        assert result["total_pixels"] == 9
        assert result["road_pixels"] == 3
        assert result["road_percentage"] == pytest.approx(33.33, 0.01)
        assert result["is_valid"] == True
        assert result["road_types"]["type_1"] == 1
        assert result["road_types"]["type_2"] == 1
        assert result["road_types"]["type_3"] == 1

    def test_no_roads(self):
        """Test validation when no roads present."""
        roads = np.zeros((10, 10), dtype=int)

        result = validate_road_coverage(roads)

        assert result["road_pixels"] == 0
        assert result["road_percentage"] == 0.0
        assert result["is_valid"] == False  # Below minimum
        assert len(result["road_types"]) == 0

    def test_too_many_roads(self):
        """Test validation when too many roads."""
        roads = np.ones((10, 10), dtype=int)  # All pixels are roads

        result = validate_road_coverage(roads, max_road_percentage=50.0)

        assert result["road_percentage"] == 100.0
        assert result["is_valid"] == False  # Above maximum


class TestValidateProcessingInputs:
    """Test processing input validation function."""

    def test_all_inputs_valid(self):
        """Test validation when all inputs are valid."""
        bands = {
            "B3": np.random.rand(100, 100),
            "B4": np.random.rand(100, 100),
            "B5": np.random.rand(100, 100),
        }

        is_valid, errors = validate_processing_inputs(bands)

        assert is_valid == True
        assert len(errors) == 0

    def test_missing_band(self):
        """Test validation with missing band."""
        bands = {
            "B3": np.random.rand(100, 100),
            "B4": np.random.rand(100, 100),
            # B5 missing
        }

        is_valid, errors = validate_processing_inputs(bands)

        assert is_valid == False
        assert len(errors) == 1
        assert "Missing required band: B5" in errors[0]

    def test_insufficient_coverage(self):
        """Test validation with insufficient coverage."""
        # Create band with mostly NaN
        bad_band = np.full((100, 100), np.nan)
        bad_band[:5, :5] = 0.5  # Only 25/10000 = 0.25% valid

        bands = {
            "B3": np.random.rand(100, 100),
            "B4": bad_band,
            "B5": np.random.rand(100, 100),
        }

        is_valid, errors = validate_processing_inputs(bands, min_coverage=10.0)

        assert is_valid == False
        assert any("B4 has insufficient coverage" in e for e in errors)

    def test_all_nan_band(self):
        """Test validation with all-NaN band."""
        bands = {
            "B3": np.full((10, 10), np.nan),
            "B4": np.random.rand(10, 10),
            "B5": np.random.rand(10, 10),
        }

        is_valid, errors = validate_processing_inputs(bands)

        assert is_valid == False
        assert any("B3 contains only NaN values" in e for e in errors)


class TestGenerateQualityReport:
    """Test quality report generation function."""

    def test_basic_report(self):
        """Test basic quality report generation."""
        data = {
            "ndvi": np.array([[0.2, 0.4], [0.6, 0.8]]),
            "ntl": np.array([[0.1, 0.0], [0.5, 0.9]]),
        }

        report = generate_quality_report(data)

        # Check NDVI report
        assert report["ndvi"]["shape"] == (2, 2)
        assert report["ndvi"]["total_pixels"] == 4
        assert report["ndvi"]["valid_pixels"] == 4
        assert report["ndvi"]["min"] == 0.2
        assert report["ndvi"]["max"] == 0.8
        assert report["ndvi"]["mean"] == 0.5

        # Check NTL report
        assert report["ntl"]["min"] == 0.0
        assert report["ntl"]["max"] == 0.9

    def test_report_with_nan(self):
        """Test report with NaN values."""
        data = {"test": np.array([[1.0, np.nan], [3.0, 4.0]])}

        report = generate_quality_report(data)

        assert report["test"]["valid_pixels"] == 3
        assert report["test"]["valid_percentage"] == 75.0
        assert report["test"]["mean"] == pytest.approx(2.67, 0.01)

    def test_report_missing_data(self):
        """Test report with missing data."""
        data = {"present": np.array([1, 2, 3]), "missing": None}

        report = generate_quality_report(data)

        assert report["present"]["total_pixels"] == 3
        assert report["missing"]["status"] == "missing"
