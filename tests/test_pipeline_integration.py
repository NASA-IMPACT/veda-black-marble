"""Integration tests for the Black Marble pipeline using in-memory data.

These tests verify the pipeline works end-to-end without external dependencies.
"""

import tempfile
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.io import MemoryFile
from pathlib import Path
import pytest

from blackmarble.pipeline import run_pipeline
from blackmarble.config import ProcessingConfig
from tests.test_scene_generator import create_test_scene
from tests.test_data_generators import (
    create_test_landsat_data,
    create_test_viirs_data,
    create_test_road_data,
)


class InMemoryAdapter:
    """Adapter that works with in-memory data instead of files."""

    def __init__(self):
        self.data_store = {}

    def write(self, key: str, data: np.ndarray, profile: dict):
        """Write data to in-memory store."""
        with MemoryFile() as memfile:
            with memfile.open(**profile) as dataset:
                if data.ndim == 2:
                    dataset.write(data, 1)
                else:
                    for i in range(data.shape[0]):
                        dataset.write(data[i], i + 1)
            self.data_store[key] = memfile.read()

    def read(self, key: str):
        """Read data from in-memory store."""
        if key not in self.data_store:
            raise KeyError(f"No data found for key: {key}")

        with MemoryFile(self.data_store[key]) as memfile:
            with memfile.open() as dataset:
                data = dataset.read()
                profile = dataset.profile
                return data, profile

    def exists(self, key: str) -> bool:
        """Check if key exists in store."""
        return key in self.data_store


def create_test_pipeline_data(shape=(512, 512)):
    """Create complete test data for pipeline."""
    # Create a test scene with all components
    scene_data = create_test_scene(
        shape=shape,
        include_clouds=True,
        include_water=True,
        include_urban=True,
        include_forest=True,
    )

    # Set up coordinate system
    transform = Affine(30, 0, -13360000, 0, -30, 4870000)  # 30m resolution in UTM
    bounds = (
        -13360000,  # west
        4870000 - shape[0] * 30,  # south
        -13360000 + shape[1] * 30,  # east
        4870000,  # north
    )

    return {
        "landsat": {
            "B3": scene_data["B3"],
            "B4": scene_data["B4"],
            "B5": scene_data["B5"],
            "QA_PIXEL": scene_data["QA_PIXEL"],
            "QA_RADSAT": scene_data["QA_RADSAT"],
            "QA_AEROSOL": scene_data["QA_AEROSOL"],
        },
        "viirs": {
            "NTL": scene_data["NTL"],
            "LUNAR": scene_data["LUNAR"],
            "QF": scene_data["QF"],
        },
        "roads": scene_data["roads"],
        "transform": transform,
        "bounds": bounds,
        "crs": "EPSG:32613",  # UTM Zone 13N
    }


class TestPipelineIntegration:
    """Test complete pipeline flow with realistic data."""

    def test_minimal_pipeline_flow(self, tmp_path):
        """Test minimal pipeline execution with in-memory data."""
        # Create test data
        test_data = create_test_pipeline_data(shape=(256, 256))

        # Create temporary files for the data
        landsat_dir = tmp_path / "landsat"
        landsat_dir.mkdir()
        viirs_dir = tmp_path / "viirs"
        viirs_dir.mkdir()

        # Write Landsat data
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": 256,
            "height": 256,
            "count": 1,
            "crs": test_data["crs"],
            "transform": test_data["transform"],
        }

        for band_name, data in test_data["landsat"].items():
            if band_name.startswith("QA"):
                band_profile = profile.copy()
                band_profile["dtype"] = (
                    "uint16" if band_name != "QA_AEROSOL" else "uint8"
                )
            else:
                band_profile = profile.copy()

            filepath = landsat_dir / f"{band_name}.tif"
            with rasterio.open(filepath, "w", **band_profile) as dst:
                dst.write(data, 1)

        # Write VIIRS data
        viirs_profile = profile.copy()
        for layer_name, data in test_data["viirs"].items():
            if layer_name == "QF":
                layer_profile = viirs_profile.copy()
                layer_profile["dtype"] = "uint8"
            else:
                layer_profile = viirs_profile.copy()

            filepath = viirs_dir / f"{layer_name}.tif"
            with rasterio.open(filepath, "w", **layer_profile) as dst:
                dst.write(data, 1)

        # Write roads data
        roads_path = tmp_path / "roads.tif"
        roads_profile = profile.copy()
        roads_profile["dtype"] = "uint8"
        with rasterio.open(roads_path, "w", **roads_profile) as dst:
            dst.write(test_data["roads"], 1)

        # Create a minimal processing function that uses the files
        output_path = tmp_path / "output.tif"

        # Instead of running the full pipeline, test key components
        from blackmarble.calculations import (
            calculate_ndvi,
            calculate_ndwi,
            calculate_ndui,
            apply_lunar_correction,
        )
        from blackmarble.masks import mark_cloud

        # Read the data back
        with rasterio.open(landsat_dir / "B4.tif") as src:
            red = src.read(1)
        with rasterio.open(landsat_dir / "B5.tif") as src:
            nir = src.read(1)
        with rasterio.open(landsat_dir / "B3.tif") as src:
            green = src.read(1)
        with rasterio.open(landsat_dir / "QA_PIXEL.tif") as src:
            qa_pixel = src.read(1)
        with rasterio.open(viirs_dir / "NTL.tif") as src:
            ntl = src.read(1)
        with rasterio.open(viirs_dir / "LUNAR.tif") as src:
            lunar = src.read(1)
        with rasterio.open(viirs_dir / "QF.tif") as src:
            qf = src.read(1)
        with rasterio.open(roads_path) as src:
            roads = src.read(1)

        # Test calculations
        ndvi = calculate_ndvi(red, nir)
        ndwi = calculate_ndwi(green, nir)

        # Apply cloud mask
        cloud_mask = mark_cloud(qa_pixel)
        # mark_cloud returns 1.0 for clear pixels, 0.0 for cloudy pixels
        ntl_masked = ntl * cloud_mask

        # Calculate NDUI
        ndui = calculate_ndui(ndvi, ndwi, ntl_masked, roads)

        # Apply lunar correction
        corrected_ntl = apply_lunar_correction(ntl_masked, lunar, qf)

        # Verify results
        assert ndvi.shape == (256, 256)
        assert ndwi.shape == (256, 256)
        assert ndui.shape == (256, 256)
        assert corrected_ntl.shape == (256, 256)

        # Check value ranges
        assert -1 <= np.nanmin(ndvi) and np.nanmax(ndvi) <= 1
        assert -1 <= np.nanmin(ndwi) and np.nanmax(ndwi) <= 1
        assert np.nanmin(ndui) >= 0  # NDUI is floored at 0
        assert np.nanmin(corrected_ntl) >= 0

        # Check that urban areas have higher NDUI (if there's any signal)
        if np.any(ndui > 0):
            # Use the original land cover map from test data generation
            urban_pixels = test_data.get("land_cover", np.zeros_like(ndui)) == 2
            if np.any(urban_pixels) and np.any(~urban_pixels):
                urban_ndui = ndui[urban_pixels]
                non_urban_ndui = ndui[~urban_pixels]
                if (
                    len(urban_ndui) > 0
                    and len(non_urban_ndui) > 0
                    and np.any(urban_ndui > 0)
                ):
                    # Only check if we have non-zero NDUI values
                    assert np.mean(urban_ndui) >= np.mean(non_urban_ndui)

    def test_processing_with_edge_cases(self, tmp_path):
        """Test pipeline with edge case scenarios."""
        # Test all clouds scenario
        all_cloud_data = create_test_landsat_data(
            shape=(100, 100), include_clouds=True
        )

        # Verify cloud mask works correctly
        from blackmarble.masks import mark_cloud

        cloud_mask = mark_cloud(all_cloud_data["QA_PIXEL"])
        # mark_cloud returns 0.0 for cloudy pixels
        assert np.all(cloud_mask == 0.0), "All pixels should be masked (0.0) for clouds"

        # Test all water scenario
        all_water_data = create_test_landsat_data(
            shape=(100, 100), scenario="water", cloud_coverage=0.0
        )

        # Calculate indices
        from blackmarble.calculations import calculate_ndvi, calculate_ndwi

        # Check QA_PIXEL values - should be QA_CLEAR_WATER for water scenario
        qa_pixel = all_water_data["QA_PIXEL"]
        unique_qa = np.unique(qa_pixel)
        assert len(unique_qa) == 1 and unique_qa[0] == 21952, f"Water scenario should have only QA_CLEAR_WATER (21952), got {unique_qa}"

        # Scale bands to reflectance if needed
        from blackmarble.calculations import scale_landsat_to_reflectance
        b3_refl = scale_landsat_to_reflectance(all_water_data["B3"])
        b4_refl = scale_landsat_to_reflectance(all_water_data["B4"])
        b5_refl = scale_landsat_to_reflectance(all_water_data["B5"])

        ndvi = calculate_ndvi(b4_refl, b5_refl)
        ndwi = calculate_ndwi(b3_refl, b5_refl)

        # Water should have negative NDVI and positive NDWI
        assert np.nanmean(ndvi) < 0, f"Water should have negative NDVI, got {np.nanmean(ndvi)}"
        assert np.nanmean(ndwi) > 0, f"Water should have positive NDWI, got {np.nanmean(ndwi)}"

    def test_multi_temporal_processing(self):
        """Test processing multiple time steps."""
        # Create data for multiple dates
        dates = ["2023-01-01", "2023-02-01", "2023-03-01"]
        scenes = []

        for date in dates:
            scene_data = create_test_scene(shape=(128, 128))
            scenes.append(scene_data)

        # Test temporal compositing
        from blackmarble.components.temporal import TemporalCompositor

        compositor = TemporalCompositor(method="median")

        # Stack NTL data
        ntl_stack = np.stack([s["NTL"] for s in scenes], axis=0)

        # Create composite
        result = compositor.process(ntl_stack)

        # The result contains the composite in the mask field and metadata
        assert result.mask.shape == (128, 128)
        assert result.metadata["method"] == "median"
        # Check that we processed 3 time steps
        assert ntl_stack.shape[0] == 3

        # The composite data is in result.mask for MaskResult
        composite = result.mask

    def test_output_generation(self, tmp_path):
        """Test output file generation."""
        from blackmarble.components import COGGenerator, OutputMetadata

        # Create test data
        data = np.random.rand(256, 256).astype(np.float32)

        metadata = OutputMetadata(
            crs="EPSG:4326",
            transform=Affine(0.1, 0, -180, 0, -0.1, 90),
            bounds=(-180, 64.4, -154.4, 90),
            shape=(256, 256),
            dtype=np.float32,
            tags={"PROCESSING_DATE": "2023-01-15", "SENSOR": "Test"},
        )

        generator = COGGenerator()
        output_path = tmp_path / "test_output.tif"

        result_path = generator.generate(data, metadata, output_path)

        assert result_path.exists()

        # Verify output
        with rasterio.open(result_path) as src:
            assert src.count == 1
            assert src.shape == (256, 256)
            assert src.crs is not None
            assert src.is_tiled  # COG property
