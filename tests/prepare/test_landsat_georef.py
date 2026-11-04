"""Landsat multi-tile mosaics must place pixels where they belong.

Regression test for the band-path reprojection in ``process_landsat_date``, which used
to pass the full-mosaic transform as ``dst_transform`` while reprojecting into a
window-sized array. The window origin (driven by ``MARGIN_PIXELS``) was then applied a
second time when pasting into the mosaic, shifting band content ~3.6 km north-west.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_bounds

from blackmarble.crs import create_processing_grid
from blackmarble.prepare import process_landsat_date


BBOX = (-122.55, 37.69, -122.32, 37.81)
UTM = "EPSG:32610"
CLEAR_QA = 21824  # Landsat C2 QA_PIXEL "clear, low-confidence everything"

# Each tile only carries data on its own side of SPLIT_LON (0 = nodata elsewhere), so the
# mosaic has to take content from both. Markers sit well inside the bbox so a ~3.6 km
# shift moves them visibly.
SPLIT_LON = -122.457
MARKERS = {
    "tile_a": {"FerryBuilding": (-122.3937, 37.7955), "TwinPeaks": (-122.4477, 37.7544)},
    "tile_b": {"OceanBeach": (-122.5107, 37.7600), "Presidio": (-122.4662, 37.7989)},
}
# Placement tolerance in metres: nearest-neighbour through two reprojections of a 30 m
# grid. The defect this guards against is ~3600 m.
TOLERANCE_M = 90.0


def _write_tile(
    directory: Path,
    name: str,
    markers: dict[str, tuple[float, float]],
    value0: int,
    east: bool,
):
    fwd = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
    x0, y1 = fwd.transform(-122.80, 38.00)
    x1, y0 = fwd.transform(-122.10, 37.50)
    width, height = int((x1 - x0) / 30), int((y1 - y0) / 30)
    transform = from_bounds(x0, y0, x1, y1, width, height)

    band = np.full((height, width), 10000, dtype=np.uint16)
    split_col = int((~transform * fwd.transform(SPLIT_LON, 37.75))[0])
    if east:
        band[:, :split_col] = 0
    else:
        band[:, split_col:] = 0
    labels: dict[float, tuple[float, float]] = {}
    for i, (_, (lon, lat)) in enumerate(markers.items()):
        value = value0 + 1000 * i
        col, row = (~transform) * fwd.transform(lon, lat)
        band[int(row) - 2 : int(row) + 3, int(col) - 2 : int(col) + 3] = value
        labels[float(value)] = (lon, lat)

    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "width": width,
        "height": height,
        "count": 1,
        "crs": UTM,
        "transform": transform,
    }
    band_path = directory / f"{name}_SR_B5.TIF"
    qa_path = directory / f"{name}_QA_PIXEL.TIF"
    with rasterio.open(band_path, "w", **profile) as dst:
        dst.write(band, 1)
    with rasterio.open(qa_path, "w", **profile) as dst:
        dst.write(np.full((height, width), CLEAR_QA, dtype=np.uint16), 1)
    return str(band_path), str(qa_path), labels


def test_multi_tile_band_mosaic_is_correctly_registered(tmp_path: Path):
    band_a, qa_a, labels_a = _write_tile(tmp_path, "tile_a", MARKERS["tile_a"], 20000, east=True)
    band_b, qa_b, labels_b = _write_tile(tmp_path, "tile_b", MARKERS["tile_b"], 30000, east=False)
    labels = {**labels_a, **labels_b}

    crs, transform, shape = create_processing_grid(BBOX, 30.0)
    result = process_landsat_date(
        bands_data={
            "QA_PIXEL": ([qa_a, qa_b], [], [], []),
            "B5": ([band_a, band_b], [], [], []),
        },
        bbox=BBOX,
        processing_crs=crs,
        reference_transform=transform,
        reference_shape=shape,
        apply_qa_mask=True,
        dilate_pixels=0,
    )
    mosaic, profile = result["B5"]
    assert profile["transform"] == transform

    back = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    for value, (true_lon, true_lat) in labels.items():
        hits = np.argwhere(np.isclose(mosaic, value))
        assert hits.size, f"marker {value} missing from mosaic (shifted off the grid?)"
        row, col = hits.mean(axis=0)
        lon, lat = back.transform(*(transform * (col + 0.5, row + 0.5)))
        dx = (lon - true_lon) * 111_320.0 * np.cos(np.radians(true_lat))
        dy = (lat - true_lat) * 110_900.0
        error = float(np.hypot(dx, dy))
        assert error < TOLERANCE_M, f"marker {value} misplaced by {error:.0f} m"


if __name__ == "__main__":
    pytest.main([__file__])
