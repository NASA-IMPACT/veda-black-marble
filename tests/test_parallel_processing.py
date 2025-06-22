"""Tests for parallel processing capabilities in the Black Marble pipeline.

These tests verify that components can safely process data in parallel
without race conditions or data corruption.
"""

import tempfile
import numpy as np
import pytest
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing
from pathlib import Path

from blackmarble.calculations import (
    calculate_ndvi,
    calculate_ndwi,
    calculate_ndui,
    apply_lunar_correction,
)
from blackmarble.masks import mark_cloud, mark_aerosol, mark_radsat
from blackmarble.components import COGGenerator, OutputMetadata
from tests.test_data_generators import create_test_scene


class TestParallelCalculations:
    """Test that calculation functions are thread-safe."""

    def test_parallel_ndvi_calculation(self):
        """Test NDVI calculation with multiple threads."""
        # Create test data
        n_scenes = 10
        shape = (100, 100)

        red_bands = [
            np.random.rand(*shape).astype(np.float32) * 0.3 for _ in range(n_scenes)
        ]
        nir_bands = [
            np.random.rand(*shape).astype(np.float32) * 0.5 for _ in range(n_scenes)
        ]

        # Sequential calculation for reference
        sequential_results = []
        for red, nir in zip(red_bands, nir_bands):
            ndvi = calculate_ndvi(red, nir)
            sequential_results.append(ndvi)

        # Parallel calculation
        with ThreadPoolExecutor(max_workers=4) as executor:
            parallel_results = list(executor.map(calculate_ndvi, red_bands, nir_bands))

        # Compare results
        for seq, par in zip(sequential_results, parallel_results):
            np.testing.assert_allclose(seq, par, rtol=1e-7)

    def test_parallel_mask_operations(self):
        """Test mask operations with multiple threads."""
        # Create test QA data
        n_scenes = 8
        shape = (200, 200)

        qa_pixel_arrays = []
        for _ in range(n_scenes):
            # Mix of clear and cloudy pixels
            qa = np.random.choice([21824, 21952, 54852], size=shape).astype(np.uint16)
            qa_pixel_arrays.append(qa)

        # Sequential masking
        sequential_masks = []
        for qa in qa_pixel_arrays:
            mask = mark_cloud(qa)
            sequential_masks.append(mask)

        # Parallel masking
        with ThreadPoolExecutor(max_workers=4) as executor:
            parallel_masks = list(executor.map(mark_cloud, qa_pixel_arrays))

        # Compare results
        for seq, par in zip(sequential_masks, parallel_masks):
            np.testing.assert_array_equal(seq, par)

    def test_thread_local_state(self):
        """Test that calculations don't share state between threads."""
        # This tests that there are no global variables being modified

        def process_scene(scene_id):
            """Process a single scene."""
            # Use thread-local random generator
            rng = np.random.RandomState(scene_id)

            # Create scene-specific data
            red = rng.rand(50, 50).astype(np.float32)
            nir = rng.rand(50, 50).astype(np.float32)
            green = rng.rand(50, 50).astype(np.float32)

            # Calculate indices
            ndvi = calculate_ndvi(red, nir)
            ndwi = calculate_ndwi(green, nir)

            # Return checksum to verify correctness
            return {
                "scene_id": scene_id,
                "ndvi_sum": float(np.sum(ndvi)),
                "ndwi_sum": float(np.sum(ndwi)),
            }

        # Process scenes in parallel
        scene_ids = list(range(20))

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(process_scene, scene_ids))

        # Verify each scene has unique results
        checksums = [(r["ndvi_sum"], r["ndwi_sum"]) for r in results]
        assert len(set(checksums)) == len(checksums), (
            "Each scene should have unique results"
        )

        # Verify deterministic results by re-processing some scenes
        for scene_id in [0, 5, 10, 15]:
            reprocessed = process_scene(scene_id)
            original = next(r for r in results if r["scene_id"] == scene_id)
            assert abs(reprocessed["ndvi_sum"] - original["ndvi_sum"]) < 1e-5
            assert abs(reprocessed["ndwi_sum"] - original["ndwi_sum"]) < 1e-5


class TestParallelIO:
    """Test parallel I/O operations."""

    def test_parallel_cog_generation(self, tmp_path):
        """Test generating multiple COGs in parallel."""
        # Create test data
        n_outputs = 6
        shape = (256, 256)

        datasets = []
        for i in range(n_outputs):
            data = np.random.rand(*shape).astype(np.float32)
            metadata = OutputMetadata(
                crs="EPSG:4326",
                transform=(0.1, 0.0, -180.0, 0.0, -0.1, 90.0),
                bounds=(-180, 64.4, -154.4, 90),
                shape=shape,
                dtype=np.dtype(np.float32),
                tags={"scene_id": str(i)},
            )
            output_path = tmp_path / f"output_{i}.tif"
            datasets.append((data, metadata, output_path))

        # Sequential generation for reference
        generator = COGGenerator()
        sequential_paths = []
        for data, metadata, output_path in datasets:
            path = generator.generate(data, metadata, output_path)
            sequential_paths.append(path)

        # Verify all files exist
        assert all(p.exists() for p in sequential_paths)

        # Clean up for parallel test
        for p in sequential_paths:
            p.unlink()

        # Parallel generation
        def generate_cog(args):
            data, metadata, output_path = args
            gen = COGGenerator()  # Create new instance per thread
            return gen.generate(data, metadata, output_path)

        with ThreadPoolExecutor(max_workers=3) as executor:
            parallel_paths = list(executor.map(generate_cog, datasets))

        # Verify all files exist and have correct properties
        assert len(parallel_paths) == n_outputs
        for i, path in enumerate(parallel_paths):
            assert path.exists()

            # Verify COG properties
            import rasterio

            with rasterio.open(path) as src:
                assert src.count == 1
                # COG might be reprojected, so just check it's reasonable
                assert src.width > 0 and src.height > 0
                assert src.tags()["scene_id"] == str(i)
                assert src.is_tiled  # Verify it's actually a COG


def _process_large_scene(scene_id):
    """CPU-intensive scene processing - module level for pickling."""
    shape = (500, 500)
    rng = np.random.RandomState(scene_id)

    # Generate data
    red = rng.rand(*shape).astype(np.float32)
    nir = rng.rand(*shape).astype(np.float32)
    green = rng.rand(*shape).astype(np.float32)
    ntl = rng.rand(*shape).astype(np.float32) * 10
    lunar = rng.rand(*shape).astype(np.float32) * 5
    quality = np.zeros(shape, dtype=np.uint8)
    roads = rng.randint(0, 6, shape, dtype=np.uint8)

    # Perform calculations
    ndvi = calculate_ndvi(red, nir)
    ndwi = calculate_ndwi(green, nir)
    ndui = calculate_ndui(ndvi, ndwi, ntl, roads)
    corrected = apply_lunar_correction(ntl, lunar, quality)

    # Return statistics instead of full arrays
    return {
        "scene_id": scene_id,
        "ndvi_mean": float(np.mean(ndvi)),
        "ndwi_mean": float(np.mean(ndwi)),
        "ndui_mean": float(np.mean(ndui)),
        "corrected_mean": float(np.mean(corrected)),
    }


class TestProcessPooling:
    """Test multiprocessing with process pools."""

    def test_process_pool_calculations(self):
        """Test calculations using process pool for CPU-bound operations."""
        # Create larger datasets for CPU-bound processing
        n_scenes = min(multiprocessing.cpu_count(), 4)  # Limit to 4 for testing

        # Process with multiprocessing
        with ProcessPoolExecutor(max_workers=n_scenes) as executor:
            results = list(executor.map(_process_large_scene, range(n_scenes)))

        # Verify we got results for all scenes
        assert len(results) == n_scenes
        assert all("scene_id" in r for r in results)

        # Verify results are reasonable
        for r in results:
            assert -1 <= r["ndvi_mean"] <= 1
            assert -1 <= r["ndwi_mean"] <= 1
            assert r["ndui_mean"] >= 0
            assert r["corrected_mean"] >= 0


class TestRaceConditions:
    """Test for potential race conditions in shared resources."""

    def test_concurrent_file_writes(self, tmp_path):
        """Test that concurrent file operations don't interfere."""
        # This simulates multiple workers writing to different files
        # in the same directory

        def write_test_file(worker_id):
            """Write a test file with worker-specific content."""
            import time
            import random

            # Add small random delay to increase chance of race conditions
            time.sleep(random.uniform(0, 0.1))

            filepath = tmp_path / f"worker_{worker_id}.txt"
            content = f"Worker {worker_id} data: " + "x" * 1000

            # Write file
            with open(filepath, "w") as f:
                f.write(content)

            # Read back to verify
            with open(filepath, "r") as f:
                read_content = f.read()

            assert read_content == content
            return worker_id

        # Run multiple workers concurrently
        n_workers = 20
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(write_test_file, range(n_workers)))

        # Verify all workers completed
        assert sorted(results) == list(range(n_workers))

        # Verify all files exist with correct content
        for i in range(n_workers):
            filepath = tmp_path / f"worker_{i}.txt"
            assert filepath.exists()
            with open(filepath, "r") as f:
                content = f.read()
                assert content.startswith(f"Worker {i} data:")


class TestParallelPipelineComponents:
    """Test parallel execution of pipeline components."""

    def test_parallel_scene_processing(self):
        """Test processing multiple scenes in parallel."""
        # Create test scenes
        n_scenes = 6
        scenes = []

        for i in range(n_scenes):
            scene_data = create_test_scene(shape=(128, 128))
            scenes.append((i, scene_data))

        def process_single_scene(args):
            """Process a single scene through the pipeline."""
            scene_id, scene_data = args

            # Extract bands
            red = scene_data["B4"]
            nir = scene_data["B5"]
            green = scene_data["B3"]
            qa_pixel = scene_data["QA_PIXEL"]
            ntl = scene_data["NTL"]
            roads = scene_data["roads"]

            # Apply cloud mask
            cloud_mask = mark_cloud(qa_pixel)

            # Calculate indices
            ndvi = calculate_ndvi(red, nir)
            ndwi = calculate_ndwi(green, nir)

            # Apply cloud mask to NTL
            ntl_masked = ntl * cloud_mask

            # Calculate NDUI
            ndui = calculate_ndui(ndvi, ndwi, ntl_masked, roads)

            # Return summary statistics
            return {
                "scene_id": scene_id,
                "cloud_percentage": float(100 * (1 - np.mean(cloud_mask))),
                "ndvi_mean": float(np.mean(ndvi)),
                "ndui_max": float(np.max(ndui)),
            }

        # Process scenes in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(process_single_scene, scenes))

        # Verify results
        assert len(results) == n_scenes
        for r in results:
            assert 0 <= r["cloud_percentage"] <= 100
            assert -1 <= r["ndvi_mean"] <= 1
            assert r["ndui_max"] >= 0


def test_thread_safety_imports():
    """Test that module imports are thread-safe."""
    # This tests that lazy imports or module-level initialization
    # doesn't cause issues when multiple threads import simultaneously

    def import_and_use(thread_id):
        """Import modules and use them."""
        # Import inside thread
        from blackmarble.calculations import calculate_ndvi
        from blackmarble.masks import mark_cloud

        # Use the functions
        data = np.random.rand(50, 50).astype(np.float32)
        qa = np.full((50, 50), 21824, dtype=np.uint16)

        ndvi = calculate_ndvi(data, data * 1.5)
        mask = mark_cloud(qa)

        return thread_id, ndvi.shape, mask.shape

    # Run imports in multiple threads simultaneously
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(import_and_use, range(10)))

    # Verify all threads completed successfully
    assert len(results) == 10
    for thread_id, ndvi_shape, mask_shape in results:
        assert ndvi_shape == (50, 50)
        assert mask_shape == (50, 50)
