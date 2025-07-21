"""Target configuration structure for the new functional pipeline.

This module shows the ideal configuration design for the reorganized
Black Marble pipeline, with clear separation of concerns across phases.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Literal

import numpy.typing as npt

# Type aliases
BBox = tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
ArrayLike = npt.NDArray[Any]


@dataclass
class LandsatAcquisitionConfig:
    """Configuration for time-window based Landsat acquisition."""
    
    # Time window settings
    window_days: int = 15  # Search ±15 days from target date
    
    # Quality scoring weights (must sum to 1.0)
    quality_weights: dict[str, float] = field(default_factory=lambda: {
        "spatial_coverage": 0.5,   # Weight for spatial coverage
        "cloud_free": 0.3,         # Weight for cloud-free percentage
        "snow_free": 0.15,         # Weight for snow-free percentage
        "off_nadir": 0.05,         # Weight for nadir viewing
    })
    
    # Minimum quality thresholds
    min_coverage_threshold: float = 0.7  # Minimum acceptable coverage
    max_cloud_threshold: float = 80.0    # Maximum cloud cover percentage
    
    # Seasonal filter (optional)
    seasonal_filter: Callable[[datetime], bool] | None = None
    
    # Scene selection preferences
    prefer_newer_sensors: bool = True  # Prefer L8/L9 over L7
    allow_incomplete_coverage: bool = False  # Allow partial tile coverage


@dataclass
class AcquisitionConfig:
    """Configuration for data acquisition phase."""

    # Landsat settings
    landsat_collection: int = 2
    landsat_level: str = "L2SP"
    cloud_cover_max: float = 30.0
    composite_months: int = 12  # Number of months for temporal composite
    
    # Landsat time-window acquisition configuration
    landsat_acquisition: LandsatAcquisitionConfig = field(default_factory=LandsatAcquisitionConfig)

    # VIIRS settings
    viirs_product: str = "VNP46A2"
    viirs_band: str = "DNB_At_Sensor_Radiance"

    # OSM settings
    road_types: list[str] = field(
        default_factory=lambda: ["motorway", "trunk", "primary", "secondary"]
    )
    osm_buffer: float = 0.00005  # Buffer distance for OSM roads in decimal degrees

    # AWS settings
    aws_credentials: dict[str, str] | None = None
    ignore_missing_tiles: bool = False


@dataclass
class LandsatAcquisitionConfig:
    """Configuration for time-window based Landsat acquisition."""
    
    # Time window settings
    window_days: int = 15  # Search ±15 days from target date
    
    # Quality scoring weights (must sum to 1.0)
    quality_weights: dict[str, float] = field(default_factory=lambda: {
        "spatial_coverage": 0.5,   # Weight for spatial coverage
        "cloud_free": 0.3,         # Weight for cloud-free percentage
        "snow_free": 0.15,         # Weight for snow-free percentage
        "off_nadir": 0.05,         # Weight for nadir viewing
    })
    
    # Minimum quality thresholds
    min_coverage_threshold: float = 0.7  # Minimum acceptable coverage
    max_cloud_threshold: float = 80.0    # Maximum cloud cover percentage
    
    # Seasonal filter (optional)
    seasonal_filter: Callable[[datetime], bool] | None = None
    
    # Scene selection preferences
    prefer_newer_sensors: bool = True  # Prefer L8/L9 over L7
    allow_incomplete_coverage: bool = False  # Allow partial tile coverage


@dataclass
class PreparationConfig:
    """Configuration for data preparation phase."""

    # Landsat scaling (matches existing config.py)
    landsat_scale_factor: float = 0.0000275
    landsat_add_offset: float = -0.2
    valid_dn_min: int = 7273
    valid_dn_max: int = 43636

    # Quality masking
    mask_clouds: bool = True
    mask_shadows: bool = True
    mask_snow: bool = True
    mask_water: bool = False
    qa_dilation_radius: int = 3

    # VIIRS corrections
    viirs_fill_value: float = 65530
    quality_valid_flags: list[int] = field(default_factory=lambda: [0, 2])

    # Spatial processing
    target_crs: str = "EPSG:3857"  # Web Mercator
    target_resolution: float = 30.0  # meters
    final_resolution: float = 0.0001  # decimal degrees
    resampling_method: str = "bilinear"


@dataclass
class AnalysisConfig:
    """Configuration for analysis phase."""

    # Index calculations (matches existing config.py)
    ndvi_epsilon: float = 1e-6
    ndwi_epsilon: float = 1e-6

    # NDUI parameters (matches existing config.py)
    ntl_floor: float = 0.1
    ntl_ceiling: float = 10.0
    ndui_floor: float = 0.02
    ndui_epsilon: float = 1e-6

    # Temporal compositing
    composite_method: Literal["median", "mean", "max", "percentile"] = "median"
    composite_percentile: int | None = None
    min_valid_observations: int = 1

    # OSM fractional road rasterization
    osm_fractional: bool = False  # Use fractional road rasterization
    osm_sub_pixels: int = 5  # Sub-pixels per dimension (5x5 = 25 total)
    osm_all_touched: bool = True  # For legacy binary rasterization

    # NTL enhancement approach
    ntl_enhancement_approach: Literal["legacy_burn", "urban_field"] = "urban_field"
    # - "legacy_burn": Direct multiplication at road pixels (1.3x major, 1.1x minor)
    # - "urban_field": Multi-scale Gaussian field-based enhancement

    # Legacy burn-in factors (used when approach="legacy_burn")
    legacy_major_road_factor: float = 1.3  # Factor for major roads
    legacy_minor_road_factor: float = 1.1  # Factor for minor roads

    # Urban field computation (used when approach="urban_field")
    urban_field_sigmas: list[float] = field(
        default_factory=lambda: [2.0, 5.0, 10.0]
    )  # Gaussian scales in pixels
    urban_field_weights: list[float] = field(
        default_factory=lambda: [0.2, 0.3, 0.3, 0.2]
    )  # Direct, local, block, district
    ntl_enhancement_mode: Literal["multiplicative", "additive", "hybrid"] = "hybrid"

    # Enhancement strength parameters (conservative defaults)
    enhancement_multiplicative_factor: float = (
        0.3  # Multiplicative mode: max enhancement factor - 1 (e.g., 0.3 = up to 1.3x)
    )
    enhancement_additive_factor: float = 0.02  # Additive mode: NTL + (factor * urban_field)
    enhancement_hybrid_expected_scale: float = 0.2  # Hybrid mode: linear term coefficient
    enhancement_hybrid_expected_sqrt_scale: float = 0.1  # Hybrid mode: sqrt term coefficient
    enhancement_hybrid_application_factor: float = (
        1.0  # Hybrid mode: apply full calculated enhancement
    )


@dataclass
class EnhancementConfig:
    """Configuration for enhancement phase."""

    # Enhancement mode
    quality_mode: Literal["basic", "enhanced"] = "basic"

    # Road enhancement (matches existing config.py for bm.py parity)
    road_major_factor: float = 1.3
    road_minor_factor: float = 1.1
    custom_road_factors: dict[int, float] | None = None

    # Contrast enhancement
    contrast_method: Literal["linear", "percentile"] = "linear"
    contrast_percentiles: tuple[float, float] = (2.0, 98.0)
    contrast_output_range: tuple[float, float] = (0.0, 1.0)

    # Advanced features (only used if quality_mode="enhanced")
    suppress_blooming: bool = False
    enhance_small_settlements: bool = False
    detect_urban_corridors: bool = False

    # Colormap settings (matches existing config.py display range)
    colormap: Literal["inferno", "viridis", "plasma", "grayscale"] = "inferno"
    colormap_range: tuple[float, float] = (0.3, 0.8)
    zero_as_black: bool = True

    # Visualization
    add_hillshade: bool = False
    add_cartographic_elements: bool = False


@dataclass
class ExportConfig:
    """Configuration for export phase."""

    # COG settings
    compress: str = "deflate"
    tiled: bool = True
    overview_levels: list[int] = field(default_factory=lambda: [2, 4, 8, 16])

    # Output formats
    generate_urban_mask: bool = False
    generate_indices_file: bool = False
    generate_web_image: bool = True
    web_image_quality: int = 85

    # Validation
    validate_before_export: bool = True
    generate_metadata: bool = True


@dataclass
class PipelineConfig:
    """Main configuration class for the target pipeline."""

    # Processing parameters
    bbox: BBox
    date: datetime
    output_path: str

    # Phase configurations
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    preparation: PreparationConfig = field(default_factory=PreparationConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    enhancement: EnhancementConfig = field(default_factory=EnhancementConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    # Global settings
    verbose: bool = False
    debug: bool = False
    max_workers: int = 4
    min_valid_pixel_percentage: float = 10.0
    use_parallel: bool = True

    # Caching and diagnostics
    cache_downloads: bool = False
    cache_dir: str | None = None
    save_diagnostics: bool = False

    @classmethod
    def for_bm_parity(cls, bbox: BBox, date: datetime, output_path: str) -> "PipelineConfig":
        """Create configuration that exactly matches legacy bm.py behavior.

        This uses the parameters that ensure parity with the legacy docker/bm.py implementation.

        Args:
            bbox: Bounding box for processing
            date: Target date
            output_path: Output file path

        Returns:
            PipelineConfig configured for bm.py parity
        """
        return cls(
            bbox=bbox,
            date=date,
            output_path=output_path,
            enhancement=EnhancementConfig(
                quality_mode="basic",
                contrast_method="linear",
                colormap="inferno",
                road_major_factor=1.3,
                road_minor_factor=1.1,
            ),
            analysis=AnalysisConfig(ntl_floor=0.1, ntl_ceiling=10.0, ndui_floor=0.02),
            export=ExportConfig(generate_web_image=True, validate_before_export=False),
        )

    @classmethod
    def for_enhanced_quality(cls, bbox: BBox, date: datetime, output_path: str) -> "PipelineConfig":
        """Create configuration for enhanced quality output.

        This enables advanced features that improve output quality beyond
        what bm.py produces, suitable for publication or high-quality visualization.

        Args:
            bbox: Bounding box for processing
            date: Target date
            output_path: Output file path

        Returns:
            PipelineConfig configured for enhanced quality
        """
        return cls(
            bbox=bbox,
            date=date,
            output_path=output_path,
            analysis=AnalysisConfig(
                composite_method="median",
            ),
            enhancement=EnhancementConfig(
                quality_mode="enhanced",
                contrast_method="percentile",
                suppress_blooming=True,
                enhance_small_settlements=True,
                add_cartographic_elements=True,
            ),
            export=ExportConfig(
                generate_urban_mask=True,
                generate_indices_file=True,
                validate_before_export=True,
            ),
        )
