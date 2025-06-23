"""Visualization enhancements and colormap application.

This module provides functions for preparing data for display,
including colormap application and visualization-specific enhancements.
"""

from typing import Any, Literal
import numpy as np
import numpy.typing as npt
from pathlib import Path

# Import existing implementations
from blackmarble.ported_functionality import apply_colormap as _apply_colormap
from blackmarble.io_rasterio import scale_image

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
    # Delegate to existing implementation in ported_functionality.py
    return _apply_colormap(
        data=data,
        colormap=colormap,
        colormap_data=colormap_data,
        scale_range=scale_range,
        alpha=alpha
    )


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
    # Import contrast enhancement
    from blackmarble.ported_functionality import enhance_contrast as _enhance_contrast
    
    channels = [red_channel, green_channel, blue_channel]
    result_channels = []
    
    for i, channel in enumerate(channels):
        # Apply contrast enhancement if requested
        if contrast_enhance:
            enhanced = _enhance_contrast(
                channel,
                method="percentile",
                percentiles=(2, 98),
                output_range=(0, 1)
            )
        else:
            enhanced = channel
        
        # Apply gamma correction if provided
        if gamma is not None and gamma[i] != 1.0:
            enhanced = np.power(np.clip(enhanced, 0, 1), gamma[i])
        
        # Scale to byte
        scaled = scale_image(enhanced, 0, 1)
        result_channels.append(scaled)
    
    # Stack into RGB array
    return np.stack(result_channels, axis=0)


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
    # Load exact inferno colormap from data files
    data_dir = Path(__file__).parent.parent / "data"
    
    try:
        # Load the exact inferno colormap arrays
        inferno_r = np.load(data_dir / "inferno_r.npy")
        inferno_g = np.load(data_dir / "inferno_g.npy")
        inferno_b = np.load(data_dir / "inferno_b.npy")
        
        colormap_data = {
            "r": inferno_r,
            "g": inferno_g,
            "b": inferno_b
        }
    except FileNotFoundError:
        # Fallback to approximate inferno
        colormap_data = None
    
    # Apply colormap with specified scale range
    result = _apply_colormap(
        data=data,
        colormap="inferno" if colormap_data is None else "grayscale",
        colormap_data=colormap_data,
        scale_range=scale_range
    )
    
    # Force zero values to black if requested
    if zero_as_black and result.ndim >= 3:
        zero_mask = data == 0
        for i in range(3):  # RGB channels
            result[i][zero_mask] = 0
    
    return result


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
    # Convert angles to radians
    azimuth_rad = np.radians(azimuth)
    altitude_rad = np.radians(altitude)
    
    # Calculate gradient
    dy, dx = np.gradient(elevation * z_factor)
    
    # Calculate slope and aspect
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)
    
    # Calculate hillshade
    hillshade = (
        np.cos(altitude_rad) * np.cos(slope) + 
        np.sin(altitude_rad) * np.sin(slope) * 
        np.cos(azimuth_rad - aspect)
    )
    
    # Normalize to [0, 1]
    hillshade = np.clip(hillshade, 0, 1)
    
    return hillshade


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
    # Ensure hillshade is normalized
    hs_norm = np.clip(hillshade, 0, 1)
    
    if blend_mode == "multiply":
        # Simple multiply blend
        if data.ndim == 3:  # RGB data
            blended = data * hs_norm[np.newaxis, :, :]
        else:
            blended = data * hs_norm
    
    elif blend_mode == "overlay":
        # Overlay blend: multiply dark areas, screen light areas
        if data.ndim == 3:
            blended = np.zeros_like(data)
            for i in range(data.shape[0]):
                mask = data[i] < 0.5
                blended[i][mask] = 2 * data[i][mask] * hs_norm[mask]
                blended[i][~mask] = 1 - 2 * (1 - data[i][~mask]) * (1 - hs_norm[~mask])
        else:
            mask = data < 0.5
            blended = np.zeros_like(data)
            blended[mask] = 2 * data[mask] * hs_norm[mask]
            blended[~mask] = 1 - 2 * (1 - data[~mask]) * (1 - hs_norm[~mask])
    
    else:  # soft_light
        # Soft light blend
        if data.ndim == 3:
            blended = np.zeros_like(data)
            for i in range(data.shape[0]):
                blended[i] = (1 - 2 * hs_norm) * data[i]**2 + 2 * hs_norm * data[i]
        else:
            blended = (1 - 2 * hs_norm) * data**2 + 2 * hs_norm * data
    
    # Apply opacity
    if opacity < 1.0:
        blended = data * (1 - opacity) + blended * opacity
    
    return np.clip(blended, 0, 1)


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
    height, width = size
    
    if orientation == "vertical":
        # Create gradient from top to bottom
        gradient = np.linspace(value_range[1], value_range[0], height)
        gradient = np.tile(gradient[:, np.newaxis], (1, width))
    else:
        # Create gradient from left to right
        gradient = np.linspace(value_range[0], value_range[1], width)
        gradient = np.tile(gradient[np.newaxis, :], (height, 1))
    
    # Apply colormap
    if isinstance(colormap, str):
        # Cast to proper literal type
        from typing import cast
        colormap_literal = cast(Literal["grayscale", "viridis", "plasma", "inferno", "turbo"], colormap)
        colorbar = _apply_colormap(gradient, colormap=colormap_literal)
    else:
        colorbar = _apply_colormap(gradient, colormap_data=colormap)
    
    # Note: Label rendering would require PIL or matplotlib
    # This returns just the colorbar image
    
    return colorbar


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
    # Ensure data is uint8
    if data.dtype != np.uint8:
        if data.max() <= 1.0:
            data = (data * 255).astype(np.uint8)
        else:
            data = np.clip(data, 0, 255).astype(np.uint8)
    
    # Simple resize if needed (nearest neighbor)
    if output_size is not None:
        target_h, target_w = output_size
        if data.ndim == 3:
            current_h, current_w = data.shape[1:]
            scale_h = target_h / current_h
            scale_w = target_w / current_w
            
            # Create output array
            resized = np.zeros((data.shape[0], target_h, target_w), dtype=np.uint8)
            
            # Simple nearest neighbor resize
            for i in range(target_h):
                for j in range(target_w):
                    src_i = int(i / scale_h)
                    src_j = int(j / scale_w)
                    resized[:, i, j] = data[:, src_i, src_j]
            
            data = resized
    
    # Note: JPEG compression and watermarking would require PIL
    # This returns the prepared array
    
    return data


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
    # Ensure same shape
    assert before.shape == after.shape, "Images must have same shape"
    
    if layout == "side_by_side":
        # Create side by side view
        if before.ndim == 3:
            _, h, w = before.shape
            combined = np.zeros((before.shape[0], h, w * 2 + divider_width), dtype=before.dtype)
            combined[:, :, :w] = before
            combined[:, :, w + divider_width:] = after
            # Add divider
            combined[:, :, w:w + divider_width] = 255 if before.dtype == np.uint8 else 1.0
        else:
            h, w = before.shape
            combined = np.zeros((h, w * 2 + divider_width), dtype=before.dtype)
            combined[:, :w] = before
            combined[:, w + divider_width:] = after
            combined[:, w:w + divider_width] = 255 if before.dtype == np.uint8 else 1.0
    
    elif layout == "checkerboard":
        # Create checkerboard pattern
        check_size = 50  # Size of checkerboard squares
        combined = before.copy()
        
        if before.ndim == 3:
            h, w = before.shape[1:]
        else:
            h, w = before.shape
        
        for i in range(0, h, check_size):
            for j in range(0, w, check_size):
                if (i // check_size + j // check_size) % 2 == 1:
                    if before.ndim == 3:
                        combined[:, i:i+check_size, j:j+check_size] = after[:, i:i+check_size, j:j+check_size]
                    else:
                        combined[i:i+check_size, j:j+check_size] = after[i:i+check_size, j:j+check_size]
    
    else:  # slider
        # Create vertical slider view (split at middle)
        combined = before.copy()
        if before.ndim == 3:
            _, h, w = before.shape
            combined[:, :, w//2:] = after[:, :, w//2:]
            # Add divider
            combined[:, :, w//2-divider_width//2:w//2+divider_width//2] = 255 if before.dtype == np.uint8 else 1.0
        else:
            h, w = before.shape
            combined[:, w//2:] = after[:, w//2:]
            combined[:, w//2-divider_width//2:w//2+divider_width//2] = 255 if before.dtype == np.uint8 else 1.0
    
    return combined


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
    # Calculate scale bar length in pixels
    length_m = length_km * 1000
    bar_length_px = int(length_m / pixel_size)
    bar_height_px = 10
    
    # Get image dimensions
    if image.ndim == 3:
        _, h, w = image.shape
    else:
        h, w = image.shape
    
    # Position calculations
    margin = 20
    if position == "bottom_right":
        x_start = w - margin - bar_length_px
        y_start = h - margin - bar_height_px
    else:  # bottom_left
        x_start = margin
        y_start = h - margin - bar_height_px
    
    # Create copy and add scale bar
    result = image.copy()
    
    # Draw scale bar
    if image.ndim == 3:
        for i, c in enumerate(color):
            result[i, y_start:y_start+bar_height_px, x_start:x_start+bar_length_px] = c
    else:
        # Grayscale - use average of color
        gray_value = sum(color) // 3
        result[y_start:y_start+bar_height_px, x_start:x_start+bar_length_px] = gray_value
    
    return result


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
    # Get image dimensions
    if image.ndim == 3:
        _, h, w = image.shape
    else:
        h, w = image.shape
    
    # Position calculations
    margin = 20
    if position == "top_right":
        x_center = w - margin - size // 2
        y_center = margin + size // 2
    else:  # top_left
        x_center = margin + size // 2
        y_center = margin + size // 2
    
    # Create simple arrow shape (triangle pointing up)
    result = image.copy()
    
    # Define arrow points (simple triangle)
    arrow_points = [
        (x_center, y_center - size // 2),  # Top
        (x_center - size // 4, y_center + size // 4),  # Bottom left
        (x_center + size // 4, y_center + size // 4),  # Bottom right
    ]
    
    # Simple rasterization of triangle
    for y in range(max(0, y_center - size // 2), min(h, y_center + size // 2)):
        for x in range(max(0, x_center - size // 2), min(w, x_center + size // 2)):
            # Check if point is inside triangle (simple approximation)
            rel_y = y - y_center
            rel_x = x - x_center
            
            # Simple triangle test
            if rel_y < 0 and abs(rel_x) < (size // 4) * (1 + rel_y / (size // 2)):
                if image.ndim == 3:
                    for i, c in enumerate(color):
                        result[i, y, x] = c
                else:
                    gray_value = sum(color) // 3
                    result[y, x] = gray_value
    
    return result