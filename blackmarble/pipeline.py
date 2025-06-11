        upload_bucket (Optional[str]): S3 bucket name for final upload. Defaults to None.
Miscellaneous utilities:
  - readVariables: parse simple KEY=VALUE files
  - removeFiles: delete all files in a folder
  - fill_gaps: stack arrays of differing shapes by padding
  - rescale_bands: threshold & scale reflectance/radiance for imagery
    # Define output subdirectories
# Miscellaneous utilities
import os
import glob

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
    # 2) Crop to UTM bbox
def removeFiles(folder: str) -> None:
    for f in glob.glob(os.path.join(folder, '*')):
        try:
            os.remove(f)
        except OSError as e:
            print(f"Error removing file {f}: {e}")
    b5_final: List[str] = []
def fill_gaps(pxs: list[np.ndarray]) -> np.ndarray:
    """Stack arrays of differing shapes by zero-padding to the max shape."""
    max_shape = np.max([m.shape for m in pxs], axis=0)
    stacked: np.ndarray = np.zeros((len(pxs), *max_shape), dtype=np.float64)
    for idx, m in enumerate(pxs):
        h, w = m.shape
        stacked[idx, :h, :w] = m
    return stacked
        b4_final.append(out)