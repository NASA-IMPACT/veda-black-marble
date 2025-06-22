"""
Unit tests for rasterio-based mosaic operations.
"""

import os
import tempfile
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from pathlib import Path

from blackmarble.mosaic_rasterio import (
    mosaic_tiles,
    mosaic_viirs_tiles,
    resample_to_resolution,
)


@pytest.fixture
def sample_tiles(tmp_path):
    """Create sample raster tiles for testing."""
    tiles = []

    # Create 2x2 grid of tiles
    for i in range(2):
        for j in range(2):
            # Create a simple raster with values based on position
            data = np.full((100, 100), i * 2 + j, dtype=np.float32)

            # Add some variation
            data[25:75, 25:75] = (i * 2 + j) * 10

            # Create transform (tiles are adjacent)
            transform = from_origin(i * 1.0, (1 - j) * 1.0, 0.01, 0.01)

            # Write tile
            tile_path = tmp_path / f"tile_{i}_{j}.tif"
            with rasterio.open(
                tile_path,
                "w",
                driver="GTiff",
                height=100,
                width=100,
                count=1,
                dtype=data.dtype,
                crs="EPSG:4326",
                transform=transform,
                nodata=-9999,
            ) as dst:
                dst.write(data, 1)

            tiles.append(str(tile_path))

    return tiles


@pytest.fixture
def viirs_tiles(tmp_path):
    """Create sample VIIRS-like tiles with appropriate data ranges."""
    ntl_tiles = []
    lunar_tiles = []
    quality_tiles = []

    for i in range(2):
        # NTL data (0-65530 range, with 65535 as nodata)
        ntl_data = np.random.randint(0, 1000, size=(100, 100), dtype=np.uint16)
        ntl_data[0:10, 0:10] = 65535  # Add some nodata

        # Lunar irradiance (similar range)
        lunar_data = np.random.randint(100, 5000, size=(100, 100), dtype=np.uint16)
        lunar_data[0:10, 0:10] = 65535

        # Quality flags (0-255 range)
        quality_data = np.random.choice([0, 1, 2, 255], size=(100, 100)).astype(
            np.uint16
        )

        transform = from_origin(i * 0.5, 1.0, 0.005, 0.005)

        # Write tiles
        for data, prefix, tiles_list in [
            (ntl_data, "ntl", ntl_tiles),
            (lunar_data, "lunar", lunar_tiles),
            (quality_data, "quality", quality_tiles),
        ]:
            tile_path = tmp_path / f"{prefix}_{i}.tif"
            with rasterio.open(
                tile_path,
                "w",
                driver="GTiff",
                height=100,
                width=100,
                count=1,
                dtype=data.dtype,
                crs="EPSG:4326",
                transform=transform,
                nodata=65535 if prefix != "quality" else None,
            ) as dst:
                dst.write(data, 1)

            tiles_list.append(str(tile_path))

    return ntl_tiles, lunar_tiles, quality_tiles


class TestMosaicTiles:
    """Test the basic mosaic_tiles function."""

    def test_mosaic_basic(self, sample_tiles, tmp_path):
        """Test basic mosaic operation."""
        output = tmp_path / "mosaic.tif"

        mosaic_tiles(sample_tiles, output)

        # Check output exists
        assert output.exists()

        # Check output properties
        with rasterio.open(output) as src:
            assert src.count == 1
            assert src.crs.to_string() == "EPSG:4326"

            # Check approximate dimensions (should be ~200x200)
            assert 180 <= src.width <= 220
            assert 180 <= src.height <= 220

            # Check data
            data = src.read(1)
            assert not np.all(data == src.nodata)

    def test_mosaic_resampling_methods(self, sample_tiles, tmp_path):
        """Test different resampling methods."""
        for method in ["nearest", "bilinear", "cubic"]:
            output = tmp_path / f"mosaic_{method}.tif"

            mosaic_tiles(sample_tiles, output, resampling=method)

            assert output.exists()

            # For nearest neighbor, check that values are preserved
            if method == "nearest":
                with rasterio.open(output) as src:
                    data = src.read(1)
                    unique_values = np.unique(data[data != src.nodata])
                    # Should only have the original tile values
                    assert all(v in [0, 1, 2, 3, 10, 20, 30] for v in unique_values)

    def test_mosaic_with_bounds(self, sample_tiles, tmp_path):
        """Test mosaic with custom bounds."""
        output = tmp_path / "mosaic_bounded.tif"
        bounds = (0.2, 0.2, 0.8, 0.8)  # Crop to center region

        mosaic_tiles(sample_tiles, output, bounds=bounds)

        with rasterio.open(output) as src:
            assert src.bounds.left >= bounds[0]
            assert src.bounds.bottom >= bounds[1]
            assert src.bounds.right <= bounds[2]
            assert src.bounds.top <= bounds[3]

    def test_mosaic_empty_list(self, tmp_path):
        """Test error handling for empty input list."""
        output = tmp_path / "mosaic.tif"

        with pytest.raises(ValueError, match="No input tiles provided"):
            mosaic_tiles([], output)

    def test_mosaic_invalid_resampling(self, sample_tiles, tmp_path):
        """Test error handling for invalid resampling method."""
        output = tmp_path / "mosaic.tif"

        with pytest.raises(ValueError, match="Invalid resampling method"):
            mosaic_tiles(sample_tiles, output, resampling="invalid")


class TestMosaicVIIRSTiles:
    """Test the specialized VIIRS mosaic function."""

    def test_viirs_mosaic(self, viirs_tiles, tmp_path):
        """Test VIIRS-specific mosaic operation."""
        ntl_tiles, lunar_tiles, quality_tiles = viirs_tiles
        output_dir = tmp_path / "viirs_output"

        ntl_out, lunar_out, quality_out = mosaic_viirs_tiles(
            ntl_tiles, lunar_tiles, quality_tiles, output_dir
        )

        # Check all outputs exist
        assert os.path.exists(ntl_out)
        assert os.path.exists(lunar_out)
        assert os.path.exists(quality_out)

        # Check NTL properties
        with rasterio.open(ntl_out) as src:
            assert src.nodata == 65535
            data = src.read(1)
            # Check that we have valid data
            assert np.sum(data != 65535) > 0

        # Check quality flags preserved exact values
        with rasterio.open(quality_out) as src:
            data = src.read(1)
            unique_values = np.unique(data)
            # Should only have the original flag values
            assert all(v in [0, 1, 2, 255] for v in unique_values)


class TestResampleToResolution:
    """Test the resample_to_resolution function."""

    def test_resample_basic(self, sample_tiles, tmp_path):
        """Test basic resampling operation."""
        input_file = sample_tiles[0]
        output = tmp_path / "resampled.tif"

        # Resample to coarser resolution
        resample_to_resolution(input_file, output, resolution=0.02)

        assert output.exists()

        # Check output resolution
        with rasterio.open(output) as src:
            assert abs(src.res[0] - 0.02) < 0.0001
            assert abs(abs(src.res[1]) - 0.02) < 0.0001

            # Should have ~50x50 pixels
            assert 45 <= src.width <= 55
            assert 45 <= src.height <= 55

    def test_resample_tuple_resolution(self, sample_tiles, tmp_path):
        """Test resampling with different x/y resolutions."""
        input_file = sample_tiles[0]
        output = tmp_path / "resampled_xy.tif"

        resample_to_resolution(input_file, output, resolution=(0.02, 0.01))

        with rasterio.open(output) as src:
            assert abs(src.res[0] - 0.02) < 0.0001
            assert abs(abs(src.res[1]) - 0.01) < 0.0001

    def test_resample_preserves_data_type(self, viirs_tiles, tmp_path):
        """Test that resampling preserves data type."""
        ntl_tiles, _, _ = viirs_tiles
        input_file = ntl_tiles[0]
        output = tmp_path / "resampled_uint16.tif"

        resample_to_resolution(input_file, output, resolution=0.01)

        with rasterio.open(input_file) as src_orig:
            with rasterio.open(output) as src_resamp:
                assert src_orig.dtypes[0] == src_resamp.dtypes[0]


@pytest.mark.integration
class TestIntegration:
    """Integration tests comparing with GDAL results."""

    def test_compare_with_gdal(self, sample_tiles, tmp_path):
        """Compare rasterio mosaic with GDAL mosaic results."""
        try:
            from osgeo import gdal
        except ImportError:
            pytest.skip("GDAL not available for comparison")

        # Create mosaics with both methods
        rio_output = tmp_path / "mosaic_rasterio.tif"
        gdal_output = tmp_path / "mosaic_gdal.tif"

        # Rasterio mosaic
        mosaic_tiles(sample_tiles, rio_output)

        # GDAL mosaic
        gdal.Warp(
            str(gdal_output), sample_tiles, format="GTiff", resampleAlg="bilinear"
        )

        # Compare results
        with rasterio.open(rio_output) as rio_src:
            with rasterio.open(gdal_output) as gdal_src:
                # Check dimensions are similar
                assert abs(rio_src.width - gdal_src.width) <= 2
                assert abs(rio_src.height - gdal_src.height) <= 2

                # Check bounds are similar
                for i in range(4):
                    assert abs(rio_src.bounds[i] - gdal_src.bounds[i]) < 0.01
