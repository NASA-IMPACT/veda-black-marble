"""Target configuration structure for the new functional pipeline.

This module shows the ideal configuration design for the reorganized
Black Marble pipeline, with clear separation of concerns across phases.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
import numpy.typing as npt
import os

# Type aliases
BBox = tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
ArrayLike = npt.NDArray[Any]


@dataclass
class AcquisitionConfig:
    """Configuration for data acquisition phase."""
    
    # Landsat settings
    landsat_collection: int = 2
    landsat_level: str = "L2SP"
    cloud_cover_max: float = 30.0
    composite_months: int = 12  # Number of months for temporal composite
    
    # VIIRS settings
    viirs_product: str = "VNP46A2"
    viirs_band: str = "DNB_At_Sensor_Radiance"
    
    # OSM settings
    road_types: list[str] = field(default_factory=lambda: [
        "motorway", "trunk", "primary", "secondary"
    ])
    osm_buffer: float = 0.00005  # Buffer distance for OSM roads in decimal degrees
    
    # AWS settings
    aws_credentials: dict[str, str] | None = None
    ignore_missing_tiles: bool = False


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
    apply_lunar_correction: bool = True
    lunar_correction_threshold: float = 0.0005
    viirs_fill_value: float = 65530
    lunar_unit_conversion: float = 10.0
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
    use_scipy_fill_gaps: bool = False
    
    # Scene merging
    merge_blend_method: Literal["feather", "max", "min", "mean"] = "feather"
    feather_distance: int = 10
    use_scipy_feather: bool = False


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
    contrast_method: Literal["linear", "percentile", "adaptive", "retinex"] = "linear"
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
                road_minor_factor=1.1
            ),
            analysis=AnalysisConfig(
                ntl_floor=0.1,
                ntl_ceiling=10.0,
                ndui_floor=0.02
            ),
            export=ExportConfig(
                generate_web_image=True,
                validate_before_export=False
            )
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
                use_scipy_fill_gaps=True,
                use_scipy_feather=True,
                composite_method="median"
            ),
            enhancement=EnhancementConfig(
                quality_mode="enhanced",
                contrast_method="percentile",
                suppress_blooming=True,
                enhance_small_settlements=True,
                add_cartographic_elements=True
            ),
            export=ExportConfig(
                generate_urban_mask=True,
                generate_indices_file=True,
                validate_before_export=True
            )
        )
    
    @classmethod
    def from_env(cls, bbox: BBox, date: datetime, output_path: str) -> "PipelineConfig":
        """Create configuration from environment variables.
        
        Environment variables are prefixed with BLACKMARBLE_ and follow
        the pattern BLACKMARBLE_SECTION_PARAMETER. For example:
        - BLACKMARBLE_COMPOSITE_MONTHS -> acquisition.composite_months
        - BLACKMARBLE_NTL_FLOOR -> analysis.ntl_floor
        - BLACKMARBLE_NDUI_FLOOR -> analysis.ndui_floor
        
        Args:
            bbox: Bounding box for processing
            date: Target date
            output_path: Output file path
            
        Returns:
            PipelineConfig with values updated from environment
        """
        config = cls.for_bm_parity(bbox, date, output_path)
        
        # Acquisition configuration
        if val := os.getenv("BLACKMARBLE_COMPOSITE_MONTHS"):
            config.acquisition.composite_months = int(val)
        if val := os.getenv("BLACKMARBLE_CLOUD_COVER_MAX"):
            config.acquisition.cloud_cover_max = float(val)
            
        # Preparation configuration
        if val := os.getenv("BLACKMARBLE_BAND_SCALE_FACTOR"):
            config.preparation.landsat_scale_factor = float(val)
        if val := os.getenv("BLACKMARBLE_BAND_OFFSET"):
            config.preparation.landsat_add_offset = float(val)
        if val := os.getenv("BLACKMARBLE_QA_DILATION_RADIUS"):
            config.preparation.qa_dilation_radius = int(val)
        if val := os.getenv("BLACKMARBLE_FINAL_RESOLUTION"):
            config.preparation.final_resolution = float(val)
            
        # Analysis configuration
        if val := os.getenv("BLACKMARBLE_NTL_FLOOR"):
            config.analysis.ntl_floor = float(val)
        if val := os.getenv("BLACKMARBLE_NTL_CEILING"):
            config.analysis.ntl_ceiling = float(val)
        if val := os.getenv("BLACKMARBLE_NDUI_FLOOR"):
            config.analysis.ndui_floor = float(val)
            
        # Enhancement configuration
        if val := os.getenv("BLACKMARBLE_ROAD_ENHANCEMENT_MAJOR"):
            config.enhancement.road_major_factor = float(val)
        if val := os.getenv("BLACKMARBLE_ROAD_ENHANCEMENT_MINOR"):
            config.enhancement.road_minor_factor = float(val)
        if val := os.getenv("BLACKMARBLE_DISPLAY_MIN"):
            config.enhancement.colormap_range = (
                float(val),
                config.enhancement.colormap_range[1]
            )
        if val := os.getenv("BLACKMARBLE_DISPLAY_MAX"):
            config.enhancement.colormap_range = (
                config.enhancement.colormap_range[0],
                float(val)
            )
            
        # Global settings
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
        if self.analysis.ntl_floor >= self.analysis.ntl_ceiling:
            raise ValueError(
                f"NTL floor ({self.analysis.ntl_floor}) must be less than "
                f"ceiling ({self.analysis.ntl_ceiling})"
            )
            
        # Display range validation
        display_min, display_max = self.enhancement.colormap_range
        if display_min >= display_max:
            raise ValueError(
                f"Display min ({display_min}) must be less than "
                f"max ({display_max})"
            )
            
        # Percentage validation
        if not 0 <= self.min_valid_pixel_percentage <= 100:
            raise ValueError(
                f"min_valid_pixel_percentage must be between 0 and 100, "
                f"got {self.min_valid_pixel_percentage}"
            )
            
        # Resolution validation
        if self.preparation.final_resolution <= 0:
            raise ValueError(
                f"final_resolution must be positive, got {self.preparation.final_resolution}"
            )
            
        # Enhancement factor validation
        if self.enhancement.road_major_factor < 1.0:
            raise ValueError(
                f"road_major_factor must be >= 1.0, "
                f"got {self.enhancement.road_major_factor}"
            )
        if self.enhancement.road_minor_factor < 1.0:
            raise ValueError(
                f"road_minor_factor must be >= 1.0, "
                f"got {self.enhancement.road_minor_factor}"
            )
    
    def to_legacy_config(self) -> dict[str, Any]:
        """Convert to format compatible with existing config.py structure.
        
        This allows gradual migration by converting the new config format
        to the existing ProcessingConfig structure.
        
        Returns:
            Dictionary compatible with existing config.py
        """
        return {
            "landsat": {
                "band_scale_factor": self.preparation.landsat_scale_factor,
                "band_offset": self.preparation.landsat_add_offset,
                "composite_months": self.acquisition.composite_months,
                "valid_dn_min": self.preparation.valid_dn_min,
                "valid_dn_max": self.preparation.valid_dn_max,
                "qa_dilation_radius": self.preparation.qa_dilation_radius,
            },
            "viirs": {
                "floor": self.analysis.ntl_floor,
                "ceiling": self.analysis.ntl_ceiling,
                "resolution": 0.00011111111,  # Default from existing config
                "fill_value": self.preparation.viirs_fill_value,
                "lunar_unit_conversion": self.preparation.lunar_unit_conversion,
                "quality_valid_flags": self.preparation.quality_valid_flags,
            },
            "ndui": {
                "floor": self.analysis.ndui_floor,
                "road_enhancement_major": self.enhancement.road_major_factor,
                "road_enhancement_minor": self.enhancement.road_minor_factor,
                "epsilon": self.analysis.ndui_epsilon,
            },
            "final_resolution": self.preparation.final_resolution,
            "display_min": self.enhancement.colormap_range[0],
            "display_max": self.enhancement.colormap_range[1],
            "min_valid_pixel_percentage": self.min_valid_pixel_percentage,
            "use_parallel": self.use_parallel,
            "verbose": self.verbose,
        }


# Pre-defined configurations for common use cases
def get_default_config(bbox: BBox, date: datetime, output_path: str) -> PipelineConfig:
    """Get default configuration that matches bm.py behavior."""
    return PipelineConfig.for_bm_parity(bbox, date, output_path)


def get_enhanced_config(bbox: BBox, date: datetime, output_path: str) -> PipelineConfig:
    """Get enhanced configuration with advanced features."""
    return PipelineConfig.for_enhanced_quality(bbox, date, output_path)
    
