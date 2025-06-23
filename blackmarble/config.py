"""Configuration management for Black Marble processing.

This module provides centralized configuration for all Black Marble processing
parameters, supporting both default values and environment variable overrides.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
import os

# Type aliases  
BBox = tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)


@dataclass
class LandsatConfig:
    """Configuration for Landsat data processing.

    Attributes:
        composite_months: Number of months to include in temporal composite
        band_scale_factor: Scale factor for DN to reflectance conversion
        band_offset: Offset for DN to reflectance conversion
        valid_dn_min: Minimum valid digital number value
        valid_dn_max: Maximum valid digital number value
        qa_dilation_radius: Radius for dilating QA masks (pixels)
    """

    composite_months: int = 12
    band_scale_factor: float = 0.0000275
    band_offset: float = -0.2
    valid_dn_min: int = 7273
    valid_dn_max: int = 43636
    qa_dilation_radius: int = 3


@dataclass
class VIIRSConfig:
    """Configuration for VIIRS nighttime lights processing.

    Attributes:
        resolution: Spatial resolution in decimal degrees
        floor: Minimum valid radiance value (nW/cm²/sr)
        ceiling: Maximum valid radiance value (nW/cm²/sr)
        fill_value: Value indicating no data in VIIRS products
        lunar_unit_conversion: Factor to convert to standard units
        quality_valid_flags: List of quality flag values to consider valid
    """

    resolution: float = 0.00011111111
    floor: float = 0.1
    ceiling: float = 10.0
    fill_value: float = 65530
    lunar_unit_conversion: float = 10.0
    quality_valid_flags: list[int] = field(default_factory=lambda: [0, 2])


@dataclass
class NDUIConfig:
    """Configuration for NDUI (Normalized Difference Urban Index) calculation.

    Attributes:
        floor: Minimum NDUI value threshold
        water_threshold: NDWI threshold for water masking
        road_enhancement_major: Enhancement factor for major roads
        road_enhancement_minor: Enhancement factor for minor roads
        epsilon: Small value to prevent division by zero
    """

    floor: float = 0.02
    water_threshold: float = 0.0
    road_enhancement_major: float = 1.3
    road_enhancement_minor: float = 1.1
    epsilon: float = 1e-6


@dataclass
class ProcessingConfig:
    """Master configuration for Black Marble processing pipeline.

    This class aggregates all configuration subsections and provides
    methods for loading from environment variables and validation.

    Attributes:
        landsat: Landsat-specific configuration
        viirs: VIIRS-specific configuration
        ndui: NDUI calculation configuration
        final_resolution: Output resolution in decimal degrees
        osm_buffer: Buffer distance for OSM roads in decimal degrees
        display_min: Minimum value for visualization scaling
        display_max: Maximum value for visualization scaling
        min_valid_pixel_percentage: Minimum valid pixels required (%)
        use_parallel: Enable parallel processing where possible
        verbose: Enable verbose output
    """

    landsat: LandsatConfig = field(default_factory=LandsatConfig)
    viirs: VIIRSConfig = field(default_factory=VIIRSConfig)
    ndui: NDUIConfig = field(default_factory=NDUIConfig)

    # Output configuration
    final_resolution: float = 0.0001
    osm_buffer: float = 0.00005
    display_min: float = 0.3
    display_max: float = 0.8

    # Processing options
    min_valid_pixel_percentage: float = 10.0
    use_parallel: bool = True
    verbose: bool = True

    @classmethod
    def from_env(cls) -> "ProcessingConfig":
        """Create configuration from environment variables.

        Environment variables are prefixed with BLACKMARBLE_ and follow
        the pattern BLACKMARBLE_SECTION_PARAMETER. For example:
        - BLACKMARBLE_COMPOSITE_MONTHS -> landsat.composite_months
        - BLACKMARBLE_NTL_FLOOR -> viirs.floor
        - BLACKMARBLE_NDUI_FLOOR -> ndui.floor

        Returns:
            ProcessingConfig with values updated from environment
        """
        config = cls()

        # Landsat configuration
        if val := os.getenv("BLACKMARBLE_COMPOSITE_MONTHS"):
            config.landsat.composite_months = int(val)
        if val := os.getenv("BLACKMARBLE_BAND_SCALE_FACTOR"):
            config.landsat.band_scale_factor = float(val)
        if val := os.getenv("BLACKMARBLE_BAND_OFFSET"):
            config.landsat.band_offset = float(val)

        # VIIRS configuration
        if val := os.getenv("BLACKMARBLE_NTL_FLOOR"):
            config.viirs.floor = float(val)
        if val := os.getenv("BLACKMARBLE_NTL_CEILING"):
            config.viirs.ceiling = float(val)
        if val := os.getenv("BLACKMARBLE_NTL_RESOLUTION"):
            config.viirs.resolution = float(val)

        # NDUI configuration
        if val := os.getenv("BLACKMARBLE_NDUI_FLOOR"):
            config.ndui.floor = float(val)
        if val := os.getenv("BLACKMARBLE_WATER_THRESHOLD"):
            config.ndui.water_threshold = float(val)
        if val := os.getenv("BLACKMARBLE_ROAD_ENHANCEMENT_MAJOR"):
            config.ndui.road_enhancement_major = float(val)
        if val := os.getenv("BLACKMARBLE_ROAD_ENHANCEMENT_MINOR"):
            config.ndui.road_enhancement_minor = float(val)

        # Output configuration
        if val := os.getenv("BLACKMARBLE_FINAL_RESOLUTION"):
            config.final_resolution = float(val)
        if val := os.getenv("BLACKMARBLE_OSM_BUFFER"):
            config.osm_buffer = float(val)
        if val := os.getenv("BLACKMARBLE_DISPLAY_MIN"):
            config.display_min = float(val)
        if val := os.getenv("BLACKMARBLE_DISPLAY_MAX"):
            config.display_max = float(val)

        # Processing options
        if val := os.getenv("BLACKMARBLE_MIN_VALID_PIXELS"):
            config.min_valid_pixel_percentage = float(val)
        if val := os.getenv("BLACKMARBLE_USE_PARALLEL"):
            config.use_parallel = val.lower() in ("true", "1", "yes")
        if val := os.getenv("BLACKMARBLE_VERBOSE"):
            config.verbose = val.lower() in ("true", "1", "yes")

        return config

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any configuration values are invalid
        """
        # VIIRS validation
        if self.viirs.floor >= self.viirs.ceiling:
            raise ValueError(
                f"VIIRS floor ({self.viirs.floor}) must be less than "
                f"ceiling ({self.viirs.ceiling})"
            )

        # Display range validation
        if self.display_min >= self.display_max:
            raise ValueError(
                f"Display min ({self.display_min}) must be less than "
                f"max ({self.display_max})"
            )

        # Percentage validation
        if not 0 <= self.min_valid_pixel_percentage <= 100:
            raise ValueError(
                f"min_valid_pixel_percentage must be between 0 and 100, "
                f"got {self.min_valid_pixel_percentage}"
            )

        # Resolution validation
        if self.final_resolution <= 0:
            raise ValueError(
                f"final_resolution must be positive, got {self.final_resolution}"
            )

        # Enhancement factor validation
        if self.ndui.road_enhancement_major < 1.0:
            raise ValueError(
                f"road_enhancement_major must be >= 1.0, "
                f"got {self.ndui.road_enhancement_major}"
            )
        if self.ndui.road_enhancement_minor < 1.0:
            raise ValueError(
                f"road_enhancement_minor must be >= 1.0, "
                f"got {self.ndui.road_enhancement_minor}"
            )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "landsat": {
                "composite_months": self.landsat.composite_months,
                "band_scale_factor": self.landsat.band_scale_factor,
                "band_offset": self.landsat.band_offset,
                "valid_dn_min": self.landsat.valid_dn_min,
                "valid_dn_max": self.landsat.valid_dn_max,
                "qa_dilation_radius": self.landsat.qa_dilation_radius,
            },
            "viirs": {
                "resolution": self.viirs.resolution,
                "floor": self.viirs.floor,
                "ceiling": self.viirs.ceiling,
                "fill_value": self.viirs.fill_value,
                "lunar_unit_conversion": self.viirs.lunar_unit_conversion,
                "quality_valid_flags": self.viirs.quality_valid_flags,
            },
            "ndui": {
                "floor": self.ndui.floor,
                "water_threshold": self.ndui.water_threshold,
                "road_enhancement_major": self.ndui.road_enhancement_major,
                "road_enhancement_minor": self.ndui.road_enhancement_minor,
                "epsilon": self.ndui.epsilon,
            },
            "output": {
                "final_resolution": self.final_resolution,
                "osm_buffer": self.osm_buffer,
                "display_min": self.display_min,
                "display_max": self.display_max,
            },
            "processing": {
                "min_valid_pixel_percentage": self.min_valid_pixel_percentage,
                "use_parallel": self.use_parallel,
                "verbose": self.verbose,
            },
        }
