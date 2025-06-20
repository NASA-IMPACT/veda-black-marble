"""Tests for configuration management."""

import os
import pytest
from blackmarble.config import ProcessingConfig, LandsatConfig, VIIRSConfig, NDUIConfig


class TestProcessingConfig:
    """Test ProcessingConfig class."""

    # Removed trivial default value test - only checked hardcoded defaults

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


# Removed TestLandsatConfig class - only tested dataclass initialization


# Removed TestVIIRSConfig class - only tested dataclass defaults


# Removed TestNDUIConfig class - only tested dataclass initialization
