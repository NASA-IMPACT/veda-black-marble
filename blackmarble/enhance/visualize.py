"""Visualization enhancements and colormap application.

This module provides functions for preparing data for display,
including colormap application and visualization-specific enhancements.
"""

from typing import Any, Literal
import numpy.typing as npt

# Type aliases
ArrayLike = npt.NDArray[Any]


def apply_colormap(
    data: ArrayLike,
    colormap: Literal["grayscale", "viridis", "plasma", "inferno", "turbo"] = "grayscale",
    colormap_data: dict[str, ArrayLike] | None = None,
    scale_range: tuple[float, float] | None = None,
    alpha: float | ArrayLike | None = None
) -> ArrayLike:
    """Apply colormap to data for visualization.
    
    Args:
        data: Input data array (2D)
        colormap: Name of standard colormap
        colormap_data: Custom colormap as dict with 'r', 'g', 'b' arrays
        scale_range: Range to scale data before colormap (min, max)
        alpha: Optional alpha channel (scalar or array)
        
    Returns:
        RGB(A) array with shape (3 or 4, height, width) as uint8
    """
    # Currently in ported_functionality.py
    ...


def create_false_color_composite(
    red_channel: ArrayLike,
    green_channel: ArrayLike,
    blue_channel: ArrayLike,
    contrast_enhance: bool = True,
    gamma: tuple[float, float, float] | None = None
) -> ArrayLike:
    """Create false color composite from three bands.
    
    Args:
        red_channel: Data for red channel
        green_channel: Data for green channel
        blue_channel: Data for blue channel
        contrast_enhance: Whether to apply contrast enhancement
        gamma: Optional gamma correction for each channel
        
    Returns:
        RGB array with shape (3, height, width)
    """
    # For multi-band visualization
    ...


def apply_inferno_colormap(
    data: ArrayLike,
    scale_range: tuple[float, float] = (0.3, 0.8),
    zero_as_black: bool = True
) -> ArrayLike:
    """Apply inferno colormap optimized for nighttime lights.
    
    Args:
        data: Nighttime lights data (should be in [0, 1])
        scale_range: Range for optimal display contrast
        zero_as_black: Whether to force zero values to black
        
    Returns:
        RGB array with inferno colormap applied
    """
    # Specific implementation for Black Marble visualization
    ...


def create_hillshade(
    elevation: ArrayLike,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0
) -> ArrayLike:
    """Create hillshade for terrain visualization.
    
    Args:
        elevation: Elevation data array
        azimuth: Light source azimuth in degrees
        altitude: Light source altitude in degrees
        z_factor: Vertical exaggeration factor
        
    Returns:
        Hillshade array in [0, 1] range
    """
    # For terrain context
    ...


def blend_with_hillshade(
    data: ArrayLike,
    hillshade: ArrayLike,
    blend_mode: Literal["multiply", "overlay", "soft_light"] = "multiply",
    opacity: float = 0.5
) -> ArrayLike:
    """Blend data with hillshade for topographic context.
    
    Args:
        data: Primary data array (e.g., colored NTL)
        hillshade: Hillshade array
        blend_mode: Blending method
        opacity: Hillshade opacity (0-1)
        
    Returns:
        Blended array
    """
    # Add terrain context to visualization
    ...


def generate_colorbar(
    colormap: str | dict[str, ArrayLike],
    value_range: tuple[float, float],
    size: tuple[int, int] = (256, 32),
    orientation: Literal["horizontal", "vertical"] = "vertical",
    labels: dict[float, str] | None = None
) -> ArrayLike:
    """Generate colorbar for legend.
    
    Args:
        colormap: Colormap name or custom colormap data
        value_range: Range of values (min, max)
        size: Size of colorbar (height, width)
        orientation: Colorbar orientation
        labels: Optional value labels
        
    Returns:
        Colorbar array
    """
    # For map legends
    ...


def prepare_for_web_display(
    data: ArrayLike,
    output_size: tuple[int, int] | None = None,
    jpeg_quality: int = 85,
    add_watermark: bool = False
) -> ArrayLike:
    """Prepare imagery for web display.
    
    Args:
        data: Input RGB data
        output_size: Optional resize dimensions
        jpeg_quality: JPEG compression quality
        add_watermark: Whether to add attribution
        
    Returns:
        Web-optimized array
    """
    # Optimize for web viewing
    ...


def create_comparison_view(
    before: ArrayLike,
    after: ArrayLike,
    layout: Literal["side_by_side", "slider", "checkerboard"] = "side_by_side",
    divider_width: int = 2
) -> ArrayLike:
    """Create comparison visualization of two images.
    
    Args:
        before: First image array
        after: Second image array
        layout: Comparison layout style
        divider_width: Width of divider in pixels
        
    Returns:
        Combined comparison array
    """
    # For before/after comparisons
    ...


def add_scale_bar(
    image: ArrayLike,
    pixel_size: float,
    length_km: float = 10.0,
    position: Literal["bottom_left", "bottom_right"] = "bottom_right",
    color: tuple[int, int, int] = (255, 255, 255)
) -> ArrayLike:
    """Add scale bar to image.
    
    Args:
        image: Input image array
        pixel_size: Size of pixel in meters
        length_km: Length of scale bar in kilometers
        position: Position of scale bar
        color: RGB color for scale bar
        
    Returns:
        Image with scale bar added
    """
    # Add cartographic element
    ...


def add_north_arrow(
    image: ArrayLike,
    position: Literal["top_left", "top_right"] = "top_right",
    size: int = 50,
    color: tuple[int, int, int] = (255, 255, 255)
) -> ArrayLike:
    """Add north arrow to image.
    
    Args:
        image: Input image array
        position: Position of north arrow
        size: Size of arrow in pixels
        color: RGB color for arrow
        
    Returns:
        Image with north arrow added
    """
    # Add cartographic element
    ...