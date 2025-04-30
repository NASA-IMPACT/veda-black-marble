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
from typing import Dict, List  # Add typing imports

def readVariables(file_path: str) -> Dict[str, str]:  # Add type hints
    """Parse a simple KEY=VALUE file into a dict."""
    variables: Dict[str, str] = {}  # Add type hint
    with open(file_path) as f:
        for line in f:
            if '=' not in line:
                continue
            key, val = line.split('=', 1)
            variables[key.strip()] = val.strip()
    return variables

def removeFiles(folder: str) -> None:  # Add type hint
    """Remove all files in the given folder."""
    for f in glob.glob(os.path.join(folder, '*')):
        try:
            os.remove(f)
        except OSError:
            pass

def fill_gaps(pxs: List[np.ndarray]) -> np.ndarray:  # Add type hint
    """Stack arrays of differing shapes by zero-padding to the max shape."""
    max_shape = np.max([m.shape for m in pxs], axis=0)
    stacked: np.ndarray = np.zeros((len(pxs), *max_shape), dtype=np.float64)  # Add type hint
    for idx, m in enumerate(pxs):
        h, w = m.shape
        stacked[idx, :h, :w] = m
    return stacked

def rescale_bands(band_data: np.ndarray) -> np.ndarray:  # Add type hint
    """Threshold and scale band data to reflectance range."""
    arr: np.ndarray = np.where(band_data < 7273, 0, band_data)  # Add type hint
    arr = np.where(arr > 43636, 0, arr)
    return arr * 0.0000275 - 0.2