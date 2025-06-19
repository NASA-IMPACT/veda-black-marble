"""
Miscellaneous utilities:
  - readVariables: parse simple KEY=VALUE files
  - removeFiles: delete all files in a folder
  - fill_gaps: stack arrays of differing shapes by padding
  - rescale_bands: threshold & scale reflectance/radiance for imagery
"""
# Miscellaneous utilities
import os
import glob
import numpy as np
# no typing import needed

def readVariables(file_path: str) -> dict[str, str]:
    """Parse a simple KEY=VALUE file into a dict."""
    variables: dict[str, str] = {}
    with open(file_path) as f:
        for line in f:
            if '=' not in line:
                continue
            key, val = line.split('=', 1)
            variables[key.strip()] = val.strip()
    return variables

def removeFiles(folder: str) -> None:
    for f in glob.glob(os.path.join(folder, '*')):
        try:
            os.remove(f)
        except OSError as e:
            print(f"Error removing file {f}: {e}")

def fill_gaps(pxs: list[np.ndarray]) -> np.ndarray:
    """Stack arrays of differing shapes by zero-padding to the max shape."""
    max_shape = np.max([m.shape for m in pxs], axis=0)
    stacked: np.ndarray = np.zeros((len(pxs), *max_shape), dtype=np.float64)
    for idx, m in enumerate(pxs):
        h, w = m.shape
        stacked[idx, :h, :w] = m
    return stacked

def rescale_bands(band_data: np.ndarray) -> np.ndarray:
    """Threshold and scale band data to reflectance range.
    
    Applies Landsat Collection 2 Level-2 scaling to convert from
    scaled DN values to surface reflectance.
    
    Args:
        band_data: Raw band data from Landsat
        
    Returns:
        Surface reflectance values (0 for invalid pixels)
    """
    # Apply scaling formula first to get reflectance values
    scaled = band_data * 0.0000275 - 0.2
    
    # Mark out-of-range pixels as invalid (0)
    # Valid range for scaled DN is 7273-43636, which corresponds to 0-1 reflectance
    scaled = np.where((band_data < 7273) | (band_data > 43636), 0, scaled)
    
    # Ensure no negative reflectance values
    scaled = np.where(scaled < 0, 0, scaled)
    
    return scaled