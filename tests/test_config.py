"""Tests for configuration management."""

import os
import pytest
from blackmarble.config import ProcessingConfig, LandsatConfig, VIIRSConfig, NDUIConfig


class TestProcessingConfig:
    """Test ProcessingConfig class."""

    def test_default_values(self):
        """Test that default configuration values are set correctly."""
        config = ProcessingConfig()

        # Landsat defaults
        assert config.landsat.composite_months == 12
        assert config.landsat.band_scale_factor == 0.0000275
        assert config.landsat.band_offset == -0.2
        assert config.landsat.valid_dn_min == 7273
        assert config.landsat.valid_dn_max == 43636

        # VIIRS defaults
        assert config.viirs.floor == 0.1
        assert config.viirs.ceiling == 10.0
        assert config.viirs.resolution == 0.00011111111
        assert config.viirs.fill_value == 65530

        # NDUI defaults
        assert config.ndui.floor == 0.02
        assert config.ndui.water_threshold == 0.0
        assert config.ndui.road_enhancement_major == 1.3
        assert config.ndui.road_enhancement_minor == 1.1

        # Output defaults
        assert config.final_resolution == 0.0001
        assert config.osm_buffer == 0.00005
        assert config.display_min == 0.3
        assert config.display_max == 0.8

    def test_from_env(self, monkeypatch):
        """Test loading configuration from environment variables."""
        # Set test environment variables
        monkeypatch.setenv("BLACKMARBLE_COMPOSITE_MONTHS", "6")
        monkeypatch.setenv("BLACKMARBLE_NTL_FLOOR", "0.2")
        monkeypatch.setenv("BLACKMARBLE_NTL_CEILING", "15.0")
        monkeypatch.setenv("BLACKMARBLE_NDUI_FLOOR", "0.05")
        monkeypatch.setenv("BLACKMARBLE_FINAL_RESOLUTION", "0.0002")
        monkeypatch.setenv("BLACKMARBLE_VERBOSE", "true")
        monkeypatch.setenv("BLACKMARBLE_USE_PARALLEL", "false")

        config = ProcessingConfig.from_env()

        # Check values were loaded from environment
        assert config.landsat.composite_months == 6
        assert config.viirs.floor == 0.2
        assert config.viirs.ceiling == 15.0
        assert config.ndui.floor == 0.05
        assert config.final_resolution == 0.0002
        assert config.verbose is True
        assert config.use_parallel is False

    def test_validate_valid_config(self):
        """Test validation with valid configuration."""
        config = ProcessingConfig()
        config.validate()  # Should not raise

    def test_validate_invalid_viirs(self):
        """Test validation catches invalid VIIRS configuration."""
        config = ProcessingConfig()
        config.viirs.floor = 20.0  # Floor > ceiling

        with pytest.raises(ValueError, match="VIIRS floor.*must be less than ceiling"):
            config.validate()

    def test_validate_invalid_display(self):
        """Test validation catches invalid display range."""
        config = ProcessingConfig()
        config.display_min = 0.9
        config.display_max = 0.8  # Min > max

        with pytest.raises(ValueError, match="Display min.*must be less than max"):
            config.validate()

    def test_validate_invalid_percentage(self):
        """Test validation catches invalid percentage values."""
        config = ProcessingConfig()
        config.min_valid_pixel_percentage = 150.0  # > 100%

        with pytest.raises(
            ValueError, match="min_valid_pixel_percentage must be between"
        ):
            config.validate()

    def test_validate_invalid_resolution(self):
        """Test validation catches invalid resolution."""
        config = ProcessingConfig()
        config.final_resolution = -0.001  # Negative

        with pytest.raises(ValueError, match="final_resolution must be positive"):
            config.validate()

    def test_validate_invalid_enhancement(self):
        """Test validation catches invalid enhancement factors."""
        config = ProcessingConfig()
        config.ndui.road_enhancement_major = 0.5  # < 1.0

        with pytest.raises(ValueError, match="road_enhancement_major must be >= 1.0"):
            config.validate()

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ProcessingConfig()
        config_dict = config.to_dict()

        # Check structure
        assert "landsat" in config_dict
        assert "viirs" in config_dict
        assert "ndui" in config_dict
        assert "output" in config_dict
        assert "processing" in config_dict

        # Check some values
        assert config_dict["landsat"]["composite_months"] == 12
        assert config_dict["viirs"]["floor"] == 0.1
        assert config_dict["output"]["final_resolution"] == 0.0001


class TestLandsatConfig:
    """Test LandsatConfig dataclass."""

    def test_default_values(self):
        """Test Landsat configuration defaults."""
        config = LandsatConfig()
        assert config.composite_months == 12
        assert config.band_scale_factor == 0.0000275
        assert config.band_offset == -0.2
        assert config.valid_dn_min == 7273
        assert config.valid_dn_max == 43636
        assert config.qa_dilation_radius == 3

    def test_custom_values(self):
        """Test creating Landsat config with custom values."""
        config = LandsatConfig(
            composite_months=6, band_scale_factor=0.00003, qa_dilation_radius=5
        )
        assert config.composite_months == 6
        assert config.band_scale_factor == 0.00003
        assert config.qa_dilation_radius == 5
        # Other values should still be defaults
        assert config.band_offset == -0.2


class TestVIIRSConfig:
    """Test VIIRSConfig dataclass."""

    def test_default_values(self):
        """Test VIIRS configuration defaults."""
        config = VIIRSConfig()
        assert config.resolution == 0.00011111111
        assert config.floor == 0.1
        assert config.ceiling == 10.0
        assert config.fill_value == 65530
        assert config.lunar_unit_conversion == 10.0
        assert config.quality_valid_flags == [0, 2]

    def test_quality_flags_mutable(self):
        """Test that quality_valid_flags list is mutable but independent."""
        config1 = VIIRSConfig()
        config2 = VIIRSConfig()

        # Modify one config's flags
        config1.quality_valid_flags.append(1)

        # Other config should be unaffected
        assert config1.quality_valid_flags == [0, 2, 1]
        assert config2.quality_valid_flags == [0, 2]


class TestNDUIConfig:
    """Test NDUIConfig dataclass."""

    def test_default_values(self):
        """Test NDUI configuration defaults."""
        config = NDUIConfig()
        assert config.floor == 0.02
        assert config.water_threshold == 0.0
        assert config.road_enhancement_major == 1.3
        assert config.road_enhancement_minor == 1.1
        assert config.epsilon == 1e-6

    def test_epsilon_precision(self):
        """Test that epsilon value maintains precision."""
        config = NDUIConfig(epsilon=1e-10)
        assert config.epsilon == 1e-10
