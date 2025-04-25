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

def readVariables(file_path):
    """Parse a simple KEY=VALUE file into a dict."""
    variables = {}
    with open(file_path) as f:
        for line in f:
            if '=' not in line:
                continue
            key, val = line.split('=', 1)
            variables[key.strip()] = val.strip()
    return variables

def removeFiles(folder):
    """Remove all files in the given folder."""
    for f in glob.glob(os.path.join(folder, '*')):
        try:
            os.remove(f)
        except OSError:
            pass

def fill_gaps(pxs):
    """Stack arrays of differing shapes by zero-padding to the max shape."""
    max_shape = np.max([m.shape for m in pxs], axis=0)
    stacked = np.zeros((len(pxs), *max_shape), dtype=np.float64)
    for idx, m in enumerate(pxs):
        h, w = m.shape
        stacked[idx, :h, :w] = m
    return stacked

def rescale_bands(band_data):
    """Threshold and scale band data to reflectance range."""
    arr = np.where(band_data < 7273, 0, band_data)
    arr = np.where(arr > 43636, 0, arr)
    return arr * 0.0000275 - 0.2