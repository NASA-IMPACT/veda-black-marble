"""Diagnostic output functions for debugging pipeline issues."""

import json
import os
from datetime import datetime
from typing import Any

import numpy.typing as npt
import rasterio

ArrayLike = npt.NDArray[Any]


def save_diagnostic(
    data: ArrayLike,
    profile: dict[str, Any] | Any,
    output_path: str,
    description: str = "",
    crs: Any = None,
) -> None:
    """Save diagnostic output for debugging.

    Args:
        data: Array data to save
        profile: Rasterio profile dict or transform
        output_path: Output file path
        description: Description of what this data represents
        crs: Optional CRS to use (if profile is just a transform)
    """
    import numpy as np

    # Ensure data is a numpy array
    data = np.asarray(data)

    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Handle case where profile is just a transform
    if hasattr(profile, "a"):  # It's a transform
        transform = profile
        profile = {
            "driver": "GTiff",
            "height": data.shape[0],
            "width": data.shape[1],
            "count": 1,
            "dtype": data.dtype,
            "transform": transform,
            "compress": "lzw",
        }
        if crs is not None:
            profile["crs"] = crs
    else:
        # Handle case where profile is None
        if profile is None:
            profile = {
                "driver": "GTiff",
                "height": data.shape[0],
                "width": data.shape[1],
                "count": 1,
                "dtype": data.dtype,
                "compress": "lzw",
            }
            if crs is not None:
                profile["crs"] = crs
        else:
            # Update profile for single band
            profile = profile.copy()
            profile.update({"driver": "GTiff", "count": 1, "dtype": data.dtype, "compress": "lzw"})

        # Ensure height and width are set
        if "height" not in profile or profile["height"] is None:
            profile["height"] = data.shape[0]
        if "width" not in profile or profile["width"] is None:
            profile["width"] = data.shape[1]

        # Ensure transform is set (use identity if missing)
        if "transform" not in profile or profile["transform"] is None:
            from rasterio.transform import Affine

            profile["transform"] = Affine.identity()

    # Validate required fields
    required_fields = ["driver", "height", "width", "count", "dtype", "transform"]
    missing_fields = [f for f in required_fields if f not in profile or profile[f] is None]
    if missing_fields:
        raise ValueError(f"Missing required fields in profile: {missing_fields}")

    # Write file
    try:
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data, 1)
            if description:
                dst.set_band_description(1, description)
    except Exception as e:
        print(f"Error saving diagnostic {output_path}: {e}")
        print(f"Profile: {profile}")
        raise

    print(f"Saved diagnostic: {output_path}")
    print(f"  Description: {description}")
    print(f"  Shape: {data.shape}")
    print(f"  Range: {data.min():.2f} to {data.max():.2f}")
    print(f"  Non-zero pixels: {(data > 0).sum()}")


def save_temporal_stack(
    stack: list[npt.NDArray[Any]],
    band_name: str,
    dates: list[datetime],
    reference_profile: dict[str, Any],
    root_dir: str = "data/diagnostics/landsat_temporal",
) -> None:
    """
    Persist every 2-D Landsat array that will feed temporal compositing.

    Args:
        stack: List of 2-D numpy arrays, already cloud-masked / cropped.
        band_name: e.g. "B4".
        dates: List of datetime objects in **the same order** as stack.
        reference_profile: Rasterio profile dict obtained from the first
            successfully read band – used as template for GeoTIFFs.
        root_dir: Base directory for diagnostics (created if missing).

    Side-effects:
        • Writes one GeoTIFF per array using numbered filenames:
          {root_dir}/{band_name}_date_01.tif, …, _date_NN.tif
        • Writes a JSON sidecar `{root_dir}/{band_name}_stack_metadata.json`
          mapping 1-based layer numbers to ISO-8601 dates.
    """
    import numpy as np
    import os
    import rasterio
    from rasterio.enums import Compression

    if not stack or not dates:
        return  # nothing to do

    if len(stack) != len(dates):
        raise ValueError("stack and dates must have equal length")

    os.makedirs(root_dir, exist_ok=True)

    # Prepare template profile (copy to avoid mutating caller)
    base_profile = reference_profile.copy()
    base_profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        compress=Compression.deflate.value,  # type: ignore[attr-defined]
        nodata=np.nan,
    )

    # Write each slice
    metadata_map: dict[str, str] = {}
    for idx, (arr, dt) in enumerate(zip(stack, dates), start=1):
        filename = f"{band_name}_date_{idx:02d}.tif"
        out_path = os.path.join(root_dir, filename)

        # Ensure profile matches current array shape
        profile = base_profile.copy()
        profile.update(height=arr.shape[0], width=arr.shape[1])

        with rasterio.open(out_path, "w", **profile) as dst:  # type: ignore[arg-type]
            dst.write(arr.astype("float32"), 1)

        metadata_map[str(idx)] = dt.isoformat()

    # Write JSON metadata
    meta_path = os.path.join(root_dir, f"{band_name}_stack_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(metadata_map, fp, indent=2)

    print(
        f"📝 Saved {len(stack)} layers for {band_name} temporal stack "
        f"to {root_dir} with metadata."
    )
    
    # Create a VRT that references all the temporal layers
    vrt_path = os.path.join(root_dir, f"{band_name}_temporal_stack.vrt")
    
    # Get list of all temporal layer files
    layer_files = []
    for idx in range(1, len(stack) + 1):
        layer_file = os.path.join(root_dir, f"{band_name}_date_{idx:02d}.tif")
        if os.path.exists(layer_file):
            layer_files.append(layer_file)
    
    if layer_files:
        from osgeo import gdal
        vrt_ds = gdal.BuildVRT(vrt_path, layer_files, separate=True)
        vrt_ds = None  # Close to write
        print(f"   Created temporal stack VRT: {vrt_path}")
