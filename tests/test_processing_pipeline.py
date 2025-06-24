"""Comprehensive tests for Black Marble data processing pipeline.

These tests focus on the actual data processing logic using realistic synthetic data,
without any I/O operations or external dependencies.
"""

import numpy as np
import pytest
from typing import Dict
from tests.qa_constants import (
    QA_CLEAR_LAND,
    QA_CLEAR_WATER,
    QA_LOW_CLOUD,
    QA_MEDIUM_CLOUD,
    QA_HIGH_CLOUD,
    QA_DILATED_CLOUD,
)

from blackmarble.calculations import (
    calculate_ndvi,
    calculate_ndwi,
    calculate_ndui,
    apply_lunar_correction,
    scale_landsat_to_reflectance,
    enhance_with_roads,
)
from blackmarble.config import NDUIConfig, VIIRSConfig, LandsatConfig

from .test_data_generators import (
    create_test_landsat_data,
    create_test_viirs_data,
    create_test_road_data,
)
from .test_scene_generator import create_test_scene


class TestCompleteProcessingPipeline:
    """Test complete processing pipeline with realistic data."""

    def process_scene(self, scene_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Process a complete scene through the pipeline.

        This mimics the actual pipeline processing without I/O.
        """
        # Extract bands
        green = scene_data["B3"]
        red = scene_data["B4"]
        nir = scene_data["B5"]
        qa_pixel = scene_data["QA_PIXEL"]
        ntl = scene_data["NTL"]
        lunar = scene_data["LUNAR"]
        quality = scene_data["QF"]
        roads = scene_data["roads"]

        # Scale Landsat bands to reflectance (0-1)
        green_reflectance = scale_landsat_to_reflectance(green.astype(np.float32))
        red_reflectance = scale_landsat_to_reflectance(red.astype(np.float32))
        nir_reflectance = scale_landsat_to_reflectance(nir.astype(np.float32))

        # Calculate indices
        ndvi = calculate_ndvi(red_reflectance, nir_reflectance)
        ndwi = calculate_ndwi(green_reflectance, nir_reflectance)

        # Apply lunar correction to NTL
        ntl_corrected = apply_lunar_correction(ntl, lunar, quality)

        # Calculate NDUI
        ndui = calculate_ndui(ntl_corrected, ndvi, ndwi, roads)

        # Create cloud mask from QA
        cloud_values = [
            QA_MEDIUM_CLOUD,
            QA_HIGH_CLOUD,
            QA_DILATED_CLOUD,
        ]  # Various cloud confidence levels
        cloud_mask = np.isin(qa_pixel, cloud_values)

        # Apply cloud mask
        ndui_masked = np.where(cloud_mask, 0, ndui)

        return {
            "ndvi": ndvi,
            "ndwi": ndwi,
            "ndui": ndui,
            "ndui_masked": ndui_masked,
            "cloud_mask": cloud_mask,
            "ntl_corrected": ntl_corrected,
        }

    def test_urban_scene_processing(self):
        """Test processing of urban scene."""
        # Create urban scene
        scene = create_test_scene(
            shape=(256, 256),
            include_clouds=True,
            include_water=False,
            include_urban=True,
            include_forest=True,
        )

        # Process scene
        results = self.process_scene(scene)

        # Verify urban characteristics
        urban_mask = scene["land_cover"] == 2

        # Check if we have urban pixels
        assert np.any(urban_mask), (
            f"No urban pixels found in land cover: {np.unique(scene['land_cover'])}"
        )

        # Urban areas should have:
        # - Low NDVI (less vegetation)
        urban_ndvi = results["ndvi"][urban_mask]
        urban_ndvi_valid = urban_ndvi[np.isfinite(urban_ndvi)]
        assert len(urban_ndvi_valid) > 0, "No valid urban NDVI values found"
        assert np.mean(urban_ndvi_valid) < 0.2, (
            f"Urban NDVI should be low, got {np.mean(urban_ndvi_valid)}"
        )

        # - High NDUI (urban index)
        urban_ndui = results["ndui"][urban_mask]
        # Filter out masked pixels (clouds, water)
        urban_ndui_valid = urban_ndui[urban_ndui > 0]
        assert len(urban_ndui_valid) > 0, "No valid urban pixels"
        assert np.mean(urban_ndui_valid) > 0.4

        # - Bright NTL
        urban_ntl = results["ntl_corrected"][urban_mask]
        assert np.mean(urban_ntl) > 0.3

        # Forest areas should have opposite characteristics
        forest_mask = scene["land_cover"] == 4
        if np.any(forest_mask):
            forest_ndvi = results["ndvi"][forest_mask]
            forest_ndvi_valid = forest_ndvi[np.isfinite(forest_ndvi)]
            assert len(forest_ndvi_valid) > 0, "No valid forest NDVI values"
            assert np.mean(forest_ndvi_valid) > 0.6, (
                f"Forest NDVI should be high, got {np.mean(forest_ndvi_valid)}"
            )

            forest_ndui = results["ndui"][forest_mask]
            forest_ndui_valid = forest_ndui[
                (forest_ndui > 0) & np.isfinite(forest_ndui)
            ]
            if len(forest_ndui_valid) > 0:
                assert np.mean(forest_ndui_valid) < 0.3, (
                    f"Forest NDUI should be low, got {np.mean(forest_ndui_valid)}"
                )

    def test_water_detection(self):
        """Test water body detection through NDWI."""
        # Create scene with water
        scene = create_test_scene(
            shape=(200, 200),
            include_clouds=False,
            include_water=True,
            include_urban=False,
            include_forest=True,
        )

        results = self.process_scene(scene)

        # Water pixels should have positive NDWI
        water_mask = scene["land_cover"] == 1
        water_ndwi = results["ndwi"][water_mask]
        assert np.mean(water_ndwi) > 0.3

        # Non-water pixels should have negative NDWI
        land_mask = scene["land_cover"] != 1
        if np.any(land_mask):  # Only test if we have land pixels
            land_ndwi = results["ndwi"][land_mask]
            land_ndwi_valid = land_ndwi[np.isfinite(land_ndwi)]
            if len(land_ndwi_valid) > 0:
                assert np.mean(land_ndwi_valid) < 0, (
                    f"Land should have negative NDWI, got {np.mean(land_ndwi_valid)}"
                )

        # NDUI should be zero in water areas (masked out)
        water_ndui = results["ndui"][water_mask]
        assert np.mean(water_ndui) < 0.1

    def test_cloud_masking(self):
        """Test cloud masking effectiveness."""
        # Create scene with ~30% cloud cover
        from .test_data_generators import create_test_scene

        scene_base = create_test_scene(
            shape=(150, 150),
            include_clouds=True,
            cloud_coverage=0.3,
            include_water=False,
        )
        # Extract just Landsat bands
        landsat = {
            k: v
            for k, v in scene_base.items()
            if k.startswith("B") or k.startswith("QA")
        }
        viirs = create_test_viirs_data(shape=(150, 150))
        roads = create_test_road_data(shape=(150, 150))

        # Merge scene data and ensure all required keys are present
        scene = {**landsat, **viirs, "roads": roads}

        # Add missing LUNAR and QF if not present
        if "LUNAR" not in scene:
            scene["LUNAR"] = viirs.get(
                "lunar_irradiance", np.full((150, 150), 0.1, dtype=np.float32)
            )
        if "QF" not in scene:
            scene["QF"] = viirs.get(
                "quality_flag", np.ones((150, 150), dtype=np.uint8)  # Quality = 1 (valid)
            )
        if "NTL" not in scene:
            scene["NTL"] = viirs.get(
                "ntl", np.random.uniform(0, 100, (150, 150)).astype(np.float32)
            )

        results = self.process_scene(scene)

        # Verify cloud mask
        cloud_mask = results["cloud_mask"]
        cloud_percentage = np.mean(cloud_mask)
        assert 0.20 < cloud_percentage < 0.35, (
            f"Expected ~30% clouds, got {cloud_percentage:.1%}"
        )

        # Cloudy areas should be masked in final NDUI
        assert np.all(results["ndui_masked"][cloud_mask] == 0)

        # Non-cloudy areas should retain values
        clear_pixels = ~cloud_mask
        assert np.mean(results["ndui_masked"][clear_pixels]) > 0

    def test_road_enhancement(self):
        """Test road network enhancement of urban signals."""
        # Create simple test case
        shape = (100, 100)

        # Uniform moderate urban signal
        ntl = np.full(shape, 30.0, dtype=np.float32)
        ndvi = np.full(shape, 0.3, dtype=np.float32)
        ndwi = np.full(shape, -0.2, dtype=np.float32)

        # Add roads
        roads = np.zeros(shape, dtype=np.uint8)
        roads[50, :] = 1  # Major road (horizontal)
        roads[:, 50] = 2  # Minor road (vertical)

        # Calculate NDUI
        config = NDUIConfig()
        ndui = calculate_ndui(ntl / 100, ndvi, ndwi, roads, config)

        # Check enhancement
        base_value = ndui[0, 0]  # No road
        major_road_value = ndui[50, 25]  # Major road
        minor_road_value = ndui[25, 50]  # Minor road

        # Major roads should be enhanced more
        assert major_road_value > base_value * 1.2
        assert major_road_value > minor_road_value

        # Minor roads should also be enhanced
        assert minor_road_value > base_value * 1.05

    def test_lunar_correction(self):
        """Test lunar irradiance correction."""
        # Create test data
        shape = (100, 100)

        # NTL with varying brightness
        ntl = np.linspace(0, 200, shape[0])[:, np.newaxis] * np.ones(shape)

        # Uniform lunar illumination
        lunar = np.full(shape, 20.0, dtype=np.float32)

        # All valid quality (not 0 or 2)
        quality = np.ones(shape, dtype=np.uint8)  # Quality = 1

        # Apply correction
        corrected = apply_lunar_correction(ntl, lunar, quality)

        # Verify correction
        # Dark areas with lunar should become darker
        assert corrected[0, 0] < 0.1  # (0-20)/10 < 0

        # Bright areas should remain bright but normalized
        assert 0.8 < corrected[-1, 0] <= 1.0  # (200-20)/10/10, capped at 1.0

        # Test quality masking
        quality[50:, :] = 0  # Quality 0 is now masked out
        corrected_masked = apply_lunar_correction(ntl, lunar, quality)

        # Quality 0 pixels should be zero
        assert np.all(corrected_masked[50:, :] == 0)
        # Valid quality pixels should be non-zero where NTL > lunar
        # First few rows have NTL < lunar, so they'll be 0
        assert np.all(corrected_masked[30:50, :] > 0)  # Where NTL > lunar


class TestProcessingEdgeCases:
    """Test edge cases and error conditions in processing."""

    def test_all_cloud_scene(self):
        """Test processing when entire scene is cloudy."""
        scene = create_test_landsat_data(
            shape=(100, 100),
            include_clouds=True,  # Will create 100% cloud coverage
        )

        # Check that QA_PIXEL contains cloud values
        qa_pixel = scene["QA_PIXEL"]
        cloud_values = [QA_MEDIUM_CLOUD, QA_HIGH_CLOUD]
        cloud_mask = np.isin(qa_pixel, cloud_values)
        assert np.all(cloud_mask), (
            f"Expected all pixels to be clouds, got {np.sum(cloud_mask)}/{qa_pixel.size}"
        )

        # NDVI should still be calculable but meaningless
        ndvi = calculate_ndvi(scene["B4"], scene["B5"])
        assert np.all(np.isfinite(ndvi))

    def test_all_water_scene(self):
        """Test processing of complete water body."""
        scene = create_test_landsat_data(
            shape=(100, 100), scenario="water", cloud_coverage=0.0
        )

        # Scale bands to reflectance before calculating indices
        b3_refl = scale_landsat_to_reflectance(scene["B3"])
        b4_refl = scale_landsat_to_reflectance(scene["B4"])
        b5_refl = scale_landsat_to_reflectance(scene["B5"])

        ndvi = calculate_ndvi(b4_refl, b5_refl)
        ndwi = calculate_ndwi(b3_refl, b5_refl)

        # Water characteristics
        assert np.nanmean(ndvi) < 0, (
            f"Water should have negative NDVI, got {np.nanmean(ndvi)}"
        )
        assert np.nanmean(ndwi) > 0.3, (
            f"Water should have positive NDWI, got {np.nanmean(ndwi)}"
        )

    def test_no_lights_scene(self):
        """Test NDUI calculation with no nighttime lights."""
        shape = (100, 100)

        # No lights
        ntl = np.zeros(shape, dtype=np.float32)
        ndvi = np.full(shape, 0.7, dtype=np.float32)  # Forest
        ndwi = np.full(shape, -0.3, dtype=np.float32)
        roads = np.zeros(shape, dtype=np.uint8)

        ndui = calculate_ndui(ntl, ndvi, ndwi, roads)

        # Should produce very low NDUI values
        assert np.mean(ndui) < 0.1

    def test_saturated_lights(self):
        """Test handling of saturated NTL values."""
        shape = (50, 50)

        # Saturated NTL
        ntl = np.full(shape, 1000.0, dtype=np.float32)
        lunar = np.zeros(shape, dtype=np.float32)
        quality = np.ones(shape, dtype=np.uint8)  # Quality = 1 (valid)

        corrected = apply_lunar_correction(ntl, lunar, quality)

        # Should be capped at 1.0
        assert np.all(corrected == 1.0)

    def test_missing_bands(self):
        """Test handling of missing band data."""
        # Create scene with NaN values
        shape = (100, 100)

        red = np.full(shape, np.nan, dtype=np.float32)
        nir = np.random.rand(*shape).astype(np.float32)

        ndvi = calculate_ndvi(red, nir)

        # NDVI should be NaN where input is NaN
        assert np.all(np.isnan(ndvi))

    def test_mixed_quality_viirs(self):
        """Test VIIRS processing with mixed quality flags."""
        shape = (100, 100)

        ntl = np.random.uniform(10, 100, shape).astype(np.float32)
        lunar = np.full(shape, 5.0, dtype=np.float32)

        # Mixed quality flags
        quality = np.random.choice([0, 1, 2, 3], size=shape).astype(np.uint8)

        corrected = apply_lunar_correction(ntl, lunar, quality)

        # Based on analysis: quality 0 and 2 should be EXCLUDED (they have low max values)
        # Only quality NOT in [0, 2] should have values
        valid_mask = ~((quality == 0) | (quality == 2))
        assert np.all(corrected[~valid_mask] == 0)
        assert np.all(corrected[valid_mask] > 0)


class TestProcessingPerformance:
    """Test performance characteristics of processing functions."""

    def test_large_scene_processing(self):
        """Test processing of large scenes."""
        # Create large scene (1000x1000)
        large_scene = create_test_scene(shape=(1000, 1000))

        # Time critical operations
        import time

        # NDVI calculation
        start = time.time()
        ndvi = calculate_ndvi(large_scene["B4"], large_scene["B5"])
        ndvi_time = time.time() - start

        # Should complete quickly (< 100ms for 1M pixels)
        assert ndvi_time < 0.1
        assert ndvi.shape == (1000, 1000)

        # NDUI calculation
        start = time.time()
        ndui = calculate_ndui(
            large_scene["NTL"] / 100,
            ndvi,
            calculate_ndwi(large_scene["B3"], large_scene["B5"]),
            large_scene["roads"],
        )
        ndui_time = time.time() - start

        # Should complete reasonably fast
        assert ndui_time < 0.2

    def test_memory_efficiency(self):
        """Test memory usage of processing functions."""
        # Create scene
        scene = create_test_scene(shape=(500, 500))

        # Check that functions don't create excessive copies
        red = scene["B4"]
        nir = scene["B5"]

        # Get initial memory usage
        initial_id = id(red)

        # Calculate NDVI
        ndvi = calculate_ndvi(red, nir)

        # Original arrays should be unchanged
        assert id(red) == initial_id
        assert not np.shares_memory(ndvi, red)
        assert not np.shares_memory(ndvi, nir)


class TestDataValidation:
    """Test data validation and quality checks."""

    def test_reflectance_scaling_detection(self):
        """Test automatic detection of reflectance vs DN values."""
        shape = (100, 100)

        # Already scaled reflectance
        reflectance = np.random.uniform(0, 0.8, shape).astype(np.float32)
        scaled = scale_landsat_to_reflectance(reflectance)

        # Should be unchanged
        np.testing.assert_array_equal(scaled, reflectance)

        # DN values needing scaling
        dn_values = np.random.uniform(7273, 43636, shape).astype(np.float32)
        scaled_dn = scale_landsat_to_reflectance(dn_values)

        # Should be different and in reflectance range
        assert not np.array_equal(scaled_dn, dn_values)
        assert np.all(scaled_dn >= 0)
        assert np.all(scaled_dn <= 1.2)

    def test_index_range_validation(self):
        """Test that indices are in expected ranges."""
        scene = create_test_scene(shape=(200, 200))

        # Calculate indices
        ndvi = calculate_ndvi(scene["B4"], scene["B5"])
        ndwi = calculate_ndwi(scene["B3"], scene["B5"])

        # Verify ranges
        assert np.all(ndvi >= -1.0)
        assert np.all(ndvi <= 1.0)
        assert np.all(ndwi >= -1.0)
        assert np.all(ndwi <= 1.0)

        # NDUI should be 0-1
        ndui = calculate_ndui(scene["NTL"] / 100, ndvi, ndwi, scene["roads"])
        assert np.all(ndui >= 0.0)
        assert np.all(ndui <= 1.0)

    def test_qa_pixel_interpretation(self):
        """Test correct interpretation of QA_PIXEL values."""
        # Known QA values
        qa_values = {
            QA_CLEAR_LAND: "clear_land",
            QA_CLEAR_WATER: "clear_water",
            QA_LOW_CLOUD: "cloud_low",
            QA_MEDIUM_CLOUD: "cloud_medium",
            QA_HIGH_CLOUD: "cloud_high",
            55052: "cloud_shadow",
        }

        # Create test array with known values
        qa_pixel = np.array(list(qa_values.keys()), dtype=np.uint16)

        # Test cloud detection
        cloud_values = [QA_LOW_CLOUD, QA_MEDIUM_CLOUD, QA_HIGH_CLOUD, QA_DILATED_CLOUD]
        cloud_mask = np.isin(qa_pixel, cloud_values)

        # Should identify correct cloud pixels
        expected_cloud_indices = [2, 3, 4, 5]
        assert np.array_equal(np.where(cloud_mask)[0], expected_cloud_indices)
