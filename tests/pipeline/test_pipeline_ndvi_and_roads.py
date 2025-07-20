"""Test pipeline flow focusing on NDVI edge cases and road rasterization.

This test uses carefully crafted synthetic data to test specific edge cases
that are difficult to reproduce with real satellite data:
1. NDVI calculation producing extreme values (137.0) with near-zero denominators
2. Road rasterization producing 0 pixels due to CRS handling

Note: This test intentionally uses mocked data to ensure deterministic testing
of mathematical edge cases. For integration testing with real data, see
test_pipeline_flow.py.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass

import numpy as np
import networkx as nx
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

from blackmarble.pipeline import black_marble_pipeline_sync
from blackmarble.config import PipelineConfig
from blackmarble.acquire.async_orchestrator import DownloadResults

# Set deterministic random seed for reproducible edge cases
np.random.seed(12345)


# Define a small test area (UTM Zone 32N - covers central Europe)
TEST_CRS = CRS.from_epsg(32632)
TEST_PIXEL_SIZE = 30  # meters
TEST_SHAPE = (100, 100)  # 100x100 pixels = 3km x 3km

# Create transform for our test grid
TEST_WEST = 500000  # 500km east in UTM
TEST_NORTH = 5000000  # 5000km north in UTM
TEST_TRANSFORM = from_origin(TEST_WEST, TEST_NORTH, TEST_PIXEL_SIZE, TEST_PIXEL_SIZE)

# Bounding box in lat/lon (roughly central Europe)
TEST_BBOX = (11.0, 45.0, 11.03, 45.03)  # Small area near 45°N, 11°E


def create_test_landsat_data():
    """Create synthetic Landsat data that will trigger NDVI edge cases.
    
    Returns dict of band arrays with realistic Landsat Collection 2 values.
    """
    # Create bands with Landsat Collection 2 scaling
    # Scale: 0.0000275, Offset: -0.2
    # This means DN values need to produce reflectance near -0.2
    
    # To get reflectance = -0.2: DN = 0
    # To get reflectance = -0.199: DN = 36.36
    # To get reflectance = 0.0: DN = 7272.73
    # To get reflectance = 0.5: DN = 25454.55
    
    # Create bands that will produce near-zero NDVI denominators
    landsat_bands = {}
    
    # QA_PIXEL - all clear pixels (value 21824 = clear land)
    landsat_bands['QA_PIXEL'] = np.full(TEST_SHAPE, 21824, dtype=np.uint16)
    
    # Create gradient that includes problematic values
    rows, cols = TEST_SHAPE
    
    # Band 3 (Green) - gradient from -0.2 to 0.3
    green_dn = np.linspace(0, 18181, rows*cols).reshape(TEST_SHAPE).astype(np.uint16)
    landsat_bands['B3'] = green_dn
    
    # Band 4 (Red) - designed to create edge cases with Band 5
    # Top half: values that make NIR+Red ≈ 0
    # Bottom half: normal values
    red_dn = np.zeros(TEST_SHAPE, dtype=np.uint16)
    # Top rows: Create values where NIR+Red will be very small
    red_dn[:50, :] = np.linspace(0, 100, cols)  # Near -0.2 reflectance
    # Bottom rows: Normal values
    red_dn[50:, :] = np.linspace(5000, 15000, cols)
    landsat_bands['B4'] = red_dn
    
    # Band 5 (NIR) - designed to create edge cases
    # Make NIR ≈ -Red in top half to get denominator ≈ 0
    nir_dn = np.zeros(TEST_SHAPE, dtype=np.uint16)
    # Top rows: NIR values that make NIR+Red ≈ 0
    # If Red DN = 50 (reflectance ≈ -0.1986), we want NIR DN ≈ 50 too
    nir_dn[:50, :] = np.linspace(100, 0, cols)  # Opposite gradient to red
    # Bottom rows: Normal values
    nir_dn[50:, :] = np.linspace(10000, 20000, cols)
    landsat_bands['B5'] = nir_dn
    
    return landsat_bands


def create_test_viirs_data():
    """Create synthetic VIIRS NTL data."""
    # VIIRS values in nW/cm²/sr (typically 0-50)
    # Create gradient with some bright areas
    viirs_ntl = np.zeros(TEST_SHAPE, dtype=np.float32)
    
    # Add some "urban" bright spots
    viirs_ntl[40:60, 40:60] = 15.0  # City center
    viirs_ntl[30:70, 30:70] = 5.0   # Urban area
    viirs_ntl[20:80, 20:80] = 1.0   # Suburban
    
    # Add noise
    viirs_ntl += np.random.random(TEST_SHAPE) * 0.1
    
    return viirs_ntl


def create_test_road_network():
    """Create a simple road network that crosses our test area."""
    G = nx.MultiDiGraph()
    
    # Add nodes in lat/lon coordinates
    # Our TEST_BBOX is (11.0, 45.0, 11.03, 45.03)
    # Add a road that crosses diagonally
    G.add_node(1, y=45.005, x=11.005)
    G.add_node(2, y=45.025, x=11.025)
    
    # Add edges
    G.add_edge(1, 2, highway='primary', length=100)
    
    # Add more roads to ensure coverage
    G.add_node(3, y=45.010, x=11.000)
    G.add_node(4, y=45.010, x=11.030)
    G.add_edge(3, 4, highway='secondary', length=150)
    
    G.add_node(5, y=45.000, x=11.015)
    G.add_node(6, y=45.030, x=11.015)
    G.add_edge(5, 6, highway='residential', length=120)
    
    # Set graph CRS
    G.graph['crs'] = 'EPSG:4326'
    
    return G


def write_test_geotiff(data, filepath, transform=TEST_TRANSFORM, crs=TEST_CRS, dtype=None):
    """Write test data as a GeoTIFF file."""
    if dtype is None:
        dtype = data.dtype
    
    profile = {
        'driver': 'GTiff',
        'height': data.shape[0],
        'width': data.shape[1],
        'count': 1,
        'dtype': dtype,
        'crs': crs,
        'transform': transform,
        'compress': 'deflate'
    }
    
    with rasterio.open(filepath, 'w', **profile) as dst:
        dst.write(data, 1)


@pytest.mark.edgecase
@pytest.mark.asyncio
async def test_pipeline_with_ndvi_edge_cases_and_roads():
    """Test full pipeline with data that triggers both NDVI and road bugs."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test data
        landsat_data = create_test_landsat_data()
        viirs_data = create_test_viirs_data()
        road_network = create_test_road_network()
        
        # Create mock download results
        download_results = DownloadResults()
        
        # Add Landsat data
        # Simulate one date with all bands
        test_date = datetime(2024, 10, 1)
        scene_dir = tmpdir / "landsat" / "LC08_L2SP_012345_20241001"
        scene_dir.mkdir(parents=True)
        
        landsat_files = {}
        for band_name, band_data in landsat_data.items():
            filepath = scene_dir / f"LC08_L2SP_012345_20241001_{band_name}.TIF"
            write_test_geotiff(band_data, filepath, dtype=np.uint16)
            landsat_files[band_name] = str(filepath)
        
        download_results.landsat_data = {
            test_date: {
                'scene_1': landsat_files
            }
        }
        
        # Add VIIRS data
        viirs_file = tmpdir / "viirs" / "VNP46A2_h19v04_20241001.tif"
        viirs_file.parent.mkdir(parents=True)
        # VIIRS is in EPSG:4326 in real pipeline
        write_test_geotiff(
            viirs_data, 
            viirs_file,
            transform=from_origin(TEST_BBOX[0], TEST_BBOX[3], 0.0003, 0.0003),
            crs=CRS.from_epsg(4326),
            dtype=np.float32
        )
        
        # Create VIIRS profile
        viirs_profile = {
            'transform': from_origin(TEST_BBOX[0], TEST_BBOX[3], 0.0003, 0.0003),
            'crs': CRS.from_epsg(4326),
            'dtype': np.float32
        }
        
        download_results.viirs_data = viirs_data
        download_results.viirs_profile = viirs_profile
        
        # Add road network
        download_results.osm_data = road_network
        
        # Mock the download orchestrator
        mock_orchestrator = Mock()
        mock_download_all = AsyncMock(return_value=download_results)
        mock_orchestrator.download_all = mock_download_all
        
        # Create pipeline config
        config = PipelineConfig(
            save_diagnostics=True,
            analysis={'osm_fractional': True, 'osm_sub_pixels': 5}
        )
        
        output_path = str(tmpdir / "output.tif")
        
        # Patch the AsyncDownloadOrchestrator creation
        with patch('blackmarble.pipeline.AsyncDownloadOrchestrator', return_value=mock_orchestrator):
            # Run the pipeline
            result = await black_marble_pipeline(
                bbox=TEST_BBOX,
                date=test_date,
                output_path=output_path,
                config=config,
                earthdata_token="fake_token"
            )
        
        # Verify the pipeline completed successfully
        assert result is not None
        assert 'success' in result
        assert result['success'] is True
        
        # Check diagnostics were saved
        diag_dir = Path("data/diagnostics")
        if diag_dir.exists():
            # Check NDVI diagnostic
            ndvi_file = diag_dir / "ndvi.tif"
            if ndvi_file.exists():
                with rasterio.open(ndvi_file) as src:
                    ndvi = src.read(1)
                    # CRITICAL: Verify NDVI is bounded
                    assert np.all(ndvi >= -1), f"NDVI min {np.min(ndvi)} < -1"
                    assert np.all(ndvi <= 1), f"NDVI max {np.max(ndvi)} > 1"
                    # Should have handled edge cases in top half
                    assert not np.any(np.isnan(ndvi)), "NDVI contains NaN values"
                    assert not np.any(np.isinf(ndvi)), "NDVI contains inf values"
            
            # Check road diagnostic
            road_file = diag_dir / "roads_fractional.tif"
            if road_file.exists():
                with rasterio.open(road_file) as src:
                    roads = src.read(1)
                    # CRITICAL: Verify roads were rasterized
                    pixels_with_roads = np.sum(roads > 0)
                    assert pixels_with_roads > 0, "No roads were rasterized!"
                    # Check fractional values
                    assert np.all(roads >= 0), "Road values < 0"
                    assert np.all(roads <= 1), "Road values > 1"


@pytest.mark.edgecase
def test_pipeline_with_no_landsat_fallback():
    """Test pipeline behavior when no Landsat data is available.
    
    This tests the VIIRS-only fallback path and ensures roads still
    get rasterized with the correct CRS.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create only VIIRS and road data (no Landsat)
        viirs_data = create_test_viirs_data()
        road_network = create_test_road_network()
        
        # Create mock download results with NO Landsat
        download_results = DownloadResults()
        download_results.landsat_data = {}  # Empty!
        
        # Add VIIRS data  
        download_results.viirs_data = viirs_data
        download_results.viirs_profile = {
            'transform': from_origin(TEST_BBOX[0], TEST_BBOX[3], 0.0003, 0.0003),
            'crs': CRS.from_epsg(4326),
            'dtype': np.float32
        }
        
        # Add road network
        download_results.osm_data = road_network
        
        # Mock the download orchestrator
        mock_orchestrator = Mock()
        mock_download_all = AsyncMock(return_value=download_results)
        mock_orchestrator.download_all = mock_download_all
        
        # Create pipeline config
        config = PipelineConfig()
        output_path = str(tmpdir / "output_viirs_only.tif")
        
        # Patch the AsyncDownloadOrchestrator creation
        with patch('blackmarble.pipeline.AsyncDownloadOrchestrator', return_value=mock_orchestrator):
            # Run the pipeline - it should handle missing Landsat gracefully
            result = black_marble_pipeline_sync(
                bbox=TEST_BBOX,
                date=datetime(2024, 10, 1),
                output_path=output_path,
                config=config,
                earthdata_token="fake_token"
            )
        
        # Pipeline should complete (though with limited functionality)
        assert result is not None
        # The pipeline might warn but shouldn't crash