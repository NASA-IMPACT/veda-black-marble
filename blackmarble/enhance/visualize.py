"""Visualization - ONLY WHAT'S USED."""

from typing import Any
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

ArrayLike = npt.NDArray[Any]


def apply_inferno_colormap(
    data: ArrayLike,
    scale_range: tuple[float, float] = (0.0, 1.0)
) -> ArrayLike:
    """Apply inferno colormap to single-band data."""
    # Normalize to scale range
    vmin, vmax = scale_range
    normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    
    # Apply inferno colormap
    cmap = plt.cm.inferno
    colored = cmap(normalized)
    
    # Return RGB channels (drop alpha)
    # matplotlib returns (height, width, 4) with RGBA
    # We need (3, height, width) for rasterio RGB
    rgb = colored[..., :3]  # Drop alpha channel, shape is (height, width, 3)
    
    # Convert to uint8 (0-255) for proper RGB imagery
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    
    # Transpose to (bands, height, width) for rasterio
    return np.transpose(rgb_uint8, (2, 0, 1))


def create_false_color_composite(
    red_channel: ArrayLike,
    green_channel: ArrayLike,
    blue_channel: ArrayLike,
    contrast_enhance: bool = True,
    gamma: float | None = None
) -> ArrayLike:
    """Create false color composite from three bands.
    
    Args:
        red_channel: Data for red channel
        green_channel: Data for green channel  
        blue_channel: Data for blue channel
        contrast_enhance: Whether to apply contrast enhancement
        gamma: Optional gamma correction value
        
    Returns:
        RGB composite array
    """
    # Stack channels
    composite = np.stack([red_channel, green_channel, blue_channel], axis=0)
    
    if contrast_enhance:
        # Simple percentile stretch per band
        for i in range(3):
            band = composite[i]
            valid_data = band[~np.isnan(band)]
            if len(valid_data) > 0:
                vmin, vmax = np.percentile(valid_data, [2, 98])
                if vmax > vmin:
                    composite[i] = np.clip((band - vmin) / (vmax - vmin), 0, 1)
    
    # Apply gamma correction if specified
    if gamma is not None and gamma != 1.0:
        composite = np.power(composite, 1.0 / gamma)
    
    return composite