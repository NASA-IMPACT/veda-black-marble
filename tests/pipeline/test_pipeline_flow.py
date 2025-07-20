"""Test infrastructure that mirrors the exact pipeline processing flow.

This test suite follows the same logical steps as blackmarble/pipeline.py,
using reference data stored in the test directory.
"""

import os
import shutil
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

from blackmarble import acquire, analyze, enhance, export, prepare
from blackmarble.config import PipelineConfig
from blackmarble.prepare.spatial import crop_to_bounds, reproject_image, align_rasters
from blackmarble.prepare.landsat_qa import create_landsat_cloud_mask
from blackmarble.analyze.temporal import create_landsat_median_composite
from blackmarble.analyze.indices import calculate_ndvi, calculate_ndwi, calculate_ndui
from blackmarble.analyze.urban_fields import compute_urban_fields, enhance_ntl_with_urban_field
from blackmarble.enhance.contrast import enhance_contrast
from blackmarble.enhance.visualize import apply_inferno_colormap
from blackmarble.export.cog import create_cog
from blackmarble.export.metadata import create_metadata
from blackmarble.acquire.osm_fractional import buffer_and_rasterize_roads_fractional


class TestPipelineFlow:
    """Test each processing step in the exact order of the pipeline."""
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_and_cleanup(self):
        """Set up test environment and clean up after all tests."""
        # Create temp directories
        self.temp_dir = tempfile.mkdtemp(prefix="blackmarble_test_")
        self.data_dir = Path(self.temp_dir) / "data"
        self.output_dir = Path(self.temp_dir) / "output"
        self.data_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)
        
        # Clean any existing test outputs
        self._cleanup_outputs()
        
        yield
        
        # Clean up after tests
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _cleanup_outputs(self):
        """Clean up any test outputs from previous runs."""
        # Clean diagnostics if they exist
        diag_dir = Path("data/diagnostics")
        if diag_dir.exists():
            shutil.rmtree(diag_dir, ignore_errors=True)
        
        # Clean temp VRTs
        temp_dir = Path("data/temp")
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def reference_data_paths(self):
        """Get paths to reference test data."""
        test_data_dir = Path(__file__).parent.parent / "data" / "reference"
        
        return {
            "viirs": test_data_dir / "viirs" / "VNP46A2_h08v05_20231015.tif",
            "landsat": {
                "B3": test_data_dir / "landsat" / "LC08_L2SP_044034_20231015_B3.TIF",
                "B4": test_data_dir / "landsat" / "LC08_L2SP_044034_20231015_B4.TIF", 
                "B5": test_data_dir / "landsat" / "LC08_L2SP_044034_20231015_B5.TIF",
                "QA_PIXEL": test_data_dir / "landsat" / "LC08_L2SP_044034_20231015_QA_PIXEL.TIF",
            },
            "osm": test_data_dir / "osm" / "test_roads.gpkg"
        }
    
    @pytest.fixture
    def pipeline_config(self, test_bbox, test_date, setup_and_cleanup):
        """Create pipeline configuration for testing."""
        config = PipelineConfig(
            bbox=test_bbox,
            date=test_date,
            output_path=str(self.output_dir)
        )
        config.acquisition.composite_months = 1  # Simple for testing
        config.analysis.osm_fractional = True
        config.analysis.ntl_enhancement_approach = "urban_field"
        return config
    
    @pytest.fixture
    def test_bbox(self):
        """Define test bounding box."""
        # Small area for fast testing (somewhere in US)
        return (-122.5, 37.7, -122.4, 37.8)  # San Francisco area
    
    @pytest.fixture
    def test_date(self):
        """Test date matching reference data."""
        return datetime(2023, 10, 15)
    
    # ==================== SETUP PHASE ====================
    
    def test_01_setup_phase(self, pipeline_config, test_bbox, test_date):
        """Test pipeline setup and configuration."""
        print("🚀 TEST: Starting Black Marble pipeline")
        
        # Verify config
        assert pipeline_config is not None
        assert pipeline_config.save_diagnostics is True
        
        # Verify inputs
        assert len(test_bbox) == 4
        assert test_date.year == 2023
    
    # ==================== DATA LOADING PHASE ====================
    
    def test_02_load_reference_data(self, reference_data_paths):
        """Test loading reference data files."""
        print("\n📥 TEST: Loading reference data")
        
        # Check if reference data exists (we'll create it if not)
        viirs_path = reference_data_paths["viirs"]
        if not viirs_path.exists():
            print(f"  Creating reference VIIRS data at {viirs_path}")
            self._create_reference_viirs(viirs_path)
        
        for band, path in reference_data_paths["landsat"].items():
            if not path.exists():
                print(f"  Creating reference Landsat {band} at {path}")
                self._create_reference_landsat(path, band)
        
        if not reference_data_paths["osm"].exists():
            print(f"  Creating reference OSM data at {reference_data_paths['osm']}")
            self._create_reference_osm(reference_data_paths["osm"])
    
    # ==================== VIIRS PROCESSING PHASE ====================
    
    def test_03_viirs_processing(self, reference_data_paths, test_bbox):
        """Test VIIRS nighttime lights processing."""
        print("\n🌃 TEST: Processing VIIRS nighttime lights")
        
        # Load VIIRS
        with rasterio.open(reference_data_paths["viirs"]) as src:
            viirs_data = src.read(1)
            viirs_profile = src.profile
        
        # Crop to bbox
        viirs_cropped, viirs_transform, viirs_bounds = crop_to_bounds(
            data=viirs_data,
            src_transform=viirs_profile["transform"],
            bounds=test_bbox
        )
        
        print(f"  VIIRS shape: {viirs_data.shape} -> {viirs_cropped.shape}")
        
        # Reproject to target CRS (UTM)
        target_crs = CRS.from_epsg(32610)  # UTM Zone 10N for West Coast
        viirs_reprojected, viirs_new_transform, viirs_new_bounds = reproject_image(
            data=viirs_cropped,
            src_transform=viirs_transform,
            src_crs=viirs_profile["crs"],
            dst_crs=target_crs,
            resampling="bilinear"
        )
        
        print(f"  VIIRS reprojected to {target_crs}: shape {viirs_reprojected.shape}")
        
        # Quick stats (matching pipeline)
        self._quick_stats(viirs_reprojected, "VIIRS reprojected")
        
        # Store for later steps
        self.viirs_reprojected = viirs_reprojected
        self.viirs_transform = viirs_new_transform
        self.target_crs = target_crs
    
    # ==================== LANDSAT PROCESSING PHASE ====================
    
    def test_04_landsat_processing(self, reference_data_paths):
        """Test Landsat data processing with QA masking."""
        print("\n🛰️  TEST: Processing Landsat imagery")
        
        # Load all bands
        landsat_data = {}
        landsat_profiles = {}
        
        for band, path in reference_data_paths["landsat"].items():
            with rasterio.open(path) as src:
                landsat_data[band] = src.read(1)
                landsat_profiles[band] = src.profile
        
        # Apply QA mask
        qa_pixel = landsat_data["QA_PIXEL"]
        cloud_mask = create_landsat_cloud_mask(
            qa_pixel=qa_pixel,
            strategy="moderate",
            dilate_pixels=3
        )
        
        print(f"  Cloud mask: {np.sum(cloud_mask) / cloud_mask.size * 100:.1f}% clear")
        
        # Apply mask to bands
        for band in ["B3", "B4", "B5"]:
            masked = landsat_data[band].astype(np.float32)
            masked[~cloud_mask] = np.nan
            landsat_data[band] = masked
        
        # Store for temporal compositing
        self.landsat_stack = {band: [landsat_data[band]] for band in ["B3", "B4", "B5"]}
        self.landsat_profiles = landsat_profiles
    
    # ==================== TEMPORAL COMPOSITING PHASE ====================
    
    def test_05_temporal_compositing(self):
        """Test creating temporal composite from Landsat stack."""
        print("\n⏳ TEST: Creating Landsat temporal composites")
        
        # Create median composites
        landsat_composites = {}
        
        for band, stack in self.landsat_stack.items():
            print(f"  Creating composite for {band} from {len(stack)} dates")
            composite = create_landsat_median_composite(
                stack,
                scale_factor=0.0000275,
                add_offset=-0.2
            )
            landsat_composites[band] = composite
        
        self.landsat_composites = landsat_composites
    
    # ==================== REPROJECTION PHASE ====================
    
    def test_06_landsat_reprojection(self):
        """Test reprojecting Landsat to target CRS."""
        print("\n🗺️  TEST: Reprojecting Landsat to target CRS")
        
        landsat_reprojected = {}
        
        for band, data in self.landsat_composites.items():
            profile = self.landsat_profiles[band]
            
            reprojected, new_transform, new_bounds = reproject_image(
                data=data,
                src_transform=profile["transform"],
                src_crs=profile["crs"],
                dst_crs=self.target_crs,
                resampling="bilinear"
            )
            
            landsat_reprojected[band] = reprojected
            
            if band == "B5":  # Use NIR as reference
                self.reference_transform = new_transform
                self.reference_shape = reprojected.shape
        
        self.landsat_reprojected = landsat_reprojected
    
    # ==================== ALIGNMENT PHASE ====================
    
    def test_07_spatial_alignment(self):
        """Test aligning all rasters to common grid."""
        print("\n🎯 TEST: Aligning rasters to common grid")
        
        # Check if we need to align VIIRS
        if self.viirs_reprojected.shape != self.reference_shape:
            print(f"  Aligning VIIRS: {self.viirs_reprojected.shape} -> {self.reference_shape}")
            
            aligned = align_rasters(
                [self.viirs_reprojected, self.landsat_reprojected["B5"]],
                [self.viirs_transform, self.reference_transform],
                reference_idx=1  # Use Landsat as reference
            )
            
            self.viirs_aligned, _ = aligned[0]
        else:
            self.viirs_aligned = self.viirs_reprojected
    
    # ==================== ROAD PROCESSING PHASE ====================
    
    def test_08_road_rasterization(self, reference_data_paths, test_bbox):
        """Test OpenStreetMap road rasterization."""
        print("\n🛣️  TEST: Processing OpenStreetMap roads")
        
        # Load road network
        import geopandas as gpd
        if reference_data_paths["osm"].exists():
            roads_gdf = gpd.read_file(reference_data_paths["osm"])
            # Convert to graph (simplified for test)
            import networkx as nx
            G = nx.MultiDiGraph()
            # Add nodes and edges from GeoDataFrame
            # (This is simplified - real implementation would use osmnx)
        else:
            # Create empty graph for testing
            import networkx as nx
            G = nx.MultiDiGraph()
        
        # Rasterize roads (fractional)
        roads_fractional = buffer_and_rasterize_roads_fractional(
            graph=G,
            bbox=test_bbox,
            transform=self.reference_transform,
            crs=self.target_crs,
            shape=self.reference_shape,
            buffer_meters=30.0,
            sub_pixels=5
        )
        
        print(f"  Fractional roads: {np.sum(roads_fractional > 0)} pixels with roads")
        
        self.roads_fractional = roads_fractional
    
    # ==================== INDEX CALCULATION PHASE ====================
    
    def test_09_calculate_indices(self):
        """Test calculating NDVI, NDWI, and NDUI."""
        print("\n📊 TEST: Calculating indices")
        
        # NDVI
        nir = self.landsat_reprojected["B5"]
        red = self.landsat_reprojected["B4"]
        self.ndvi = calculate_ndvi(nir, red)
        print(f"  NDVI range: [{np.nanmin(self.ndvi):.3f}, {np.nanmax(self.ndvi):.3f}]")
        
        # NDWI
        green = self.landsat_reprojected["B3"]
        self.ndwi = calculate_ndwi(green, nir)
        print(f"  NDWI range: [{np.nanmin(self.ndwi):.3f}, {np.nanmax(self.ndwi):.3f}]")
        
        # NDUI
        self.ndui = calculate_ndui(
            ntl=self.viirs_aligned,
            ndvi=self.ndvi,
            ndwi=self.ndwi
        )
        print(f"  NDUI range: [{np.nanmin(self.ndui):.3f}, {np.nanmax(self.ndui):.3f}]")
    
    # ==================== ENHANCEMENT PHASE ====================
    
    def test_10_urban_field_enhancement(self):
        """Test urban field computation and NTL enhancement."""
        print("\n🌃 TEST: Enhancing NTL with urban fields")
        
        # Compute urban fields
        urban_field = compute_urban_fields(
            road_fraction=self.roads_fractional,
            sigmas=[2.0, 5.0, 10.0],
            weights=[0.5, 0.3, 0.2]
        )
        
        # Enhance NTL
        self.ntl_enhanced = enhance_ntl_with_urban_field(
            ntl=self.viirs_aligned,
            urban_field=urban_field,
            enhancement_mode="hybrid"
        )
        
        enhancement = self.ntl_enhanced - self.viirs_aligned
        print(f"  Enhanced {np.sum(enhancement > 0)} pixels")
    
    # ==================== VISUALIZATION PHASE ====================
    
    def test_11_visualization_enhancement(self):
        """Test contrast enhancement and colormap."""
        print("\n✨ TEST: Enhancing for visualization")
        
        # Contrast enhancement
        self.ntl_contrast = enhance_contrast(
            data=self.ntl_enhanced,
            method="percentile",
            percentiles=(2.0, 98.0),
            output_range=(0.0, 1.0)
        )
        
        # Apply colormap to NDUI
        self.ntl_colored = apply_inferno_colormap(
            data=self.ndui,
            scale_range=(0.3, 0.8)
        )
        
        print(f"  Output shape: {self.ntl_colored.shape} (RGB)")
    
    # ==================== EXPORT PHASE ====================
    
    def test_12_cog_export(self):
        """Test exporting results as Cloud Optimized GeoTIFF."""
        print("\n💾 TEST: Exporting results")
        
        output_path = self.output_dir / "blackmarble_test.tif"
        
        # Create metadata
        metadata = create_metadata(
            data_sources={
                "landsat": {"sensor": "Landsat 8", "date_range": "2023-10-15"},
                "viirs": {"sensor": "VIIRS", "product": "VNP46A2"},
                "osm": {"date_accessed": "2023-10-15"}
            },
            processing_steps=["All steps tested"],
            indices_calculated=["ndvi", "ndwi", "ndui"],
            date_processed=datetime.now(),
            bbox=(-122.5, 37.7, -122.4, 37.8),
            crs=self.target_crs,
            resolution=30
        )
        
        # Export COG
        create_cog(
            data=self.ntl_colored,
            output_path=str(output_path),
            transform=self.reference_transform,
            crs=self.target_crs,
            metadata=metadata,
            compress="deflate"
        )
        
        assert output_path.exists()
        print(f"  Exported to: {output_path}")
    
    # ==================== HELPER METHODS ====================
    
    def _quick_stats(self, data, label):
        """Log array statistics (matching pipeline)."""
        print(f"\n  {label}:")
        print(f"    Shape: {data.shape}, Dtype: {data.dtype}")
        print(f"    Min: {np.nanmin(data):.6f}, Max: {np.nanmax(data):.6f}")
        print(f"    Non-zero: {(data != 0).sum()} ({(data != 0).sum() / data.size * 100:.1f}%)")
        print(f"    NaN count: {np.isnan(data).sum()}")
    
    def _create_reference_viirs(self, path):
        """Create synthetic VIIRS reference data."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create realistic VIIRS data
        shape = (500, 500)
        data = np.random.exponential(2, shape).astype(np.float32)
        data = np.clip(data, 0, 50)  # Realistic range
        
        # Add some bright spots (cities)
        y, x = np.ogrid[:shape[0], :shape[1]]
        mask = ((x - 250)**2 + (y - 250)**2) < 2500
        data[mask] = data[mask] * 5
        
        # Save as GeoTIFF
        transform = from_origin(-122.5, 37.8, 0.005, 0.005)  # ~500m pixels
        profile = {
            'driver': 'GTiff',
            'height': shape[0],
            'width': shape[1],
            'count': 1,
            'dtype': 'float32',
            'crs': 'EPSG:4326',
            'transform': transform,
            'compress': 'deflate'
        }
        
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(data, 1)
    
    def _create_reference_landsat(self, path, band):
        """Create synthetic Landsat reference data."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        shape = (300, 300)
        
        if band == "QA_PIXEL":
            # Create QA data with some clouds
            data = np.full(shape, 21824, dtype=np.uint16)  # Clear land
            # Add cloud patches
            data[50:100, 50:100] = 22080  # Cloud
            data[200:220, 150:200] = 21824 | (1 << 4)  # Shadow
        else:
            # Create spectral band data
            if band == "B3":  # Green
                base = 9000
            elif band == "B4":  # Red
                base = 8000
            elif band == "B5":  # NIR
                base = 12000
            
            data = np.full(shape, base, dtype=np.uint16)
            # Add variation
            data += np.random.randint(-500, 500, shape)
            data = np.clip(data, 0, 65535)
        
        # Save as GeoTIFF (UTM projection)
        transform = from_origin(500000, 4200000, 30, 30)  # 30m pixels
        profile = {
            'driver': 'GTiff',
            'height': shape[0],
            'width': shape[1],
            'count': 1,
            'dtype': 'uint16',
            'crs': 'EPSG:32610',  # UTM Zone 10N
            'transform': transform,
            'compress': 'deflate'
        }
        
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(data, 1)
    
    def _create_reference_osm(self, path):
        """Create synthetic OSM road data."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        import geopandas as gpd
        from shapely.geometry import LineString
        
        # Create simple road network
        roads = [
            LineString([(-122.45, 37.75), (-122.42, 37.77)]),  # Main road
            LineString([(-122.44, 37.73), (-122.44, 37.78)]),  # Cross road
            LineString([(-122.46, 37.76), (-122.41, 37.76)]),  # Another road
        ]
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame({
            'highway': ['primary', 'secondary', 'residential'],
            'geometry': roads
        }, crs='EPSG:4326')
        
        # Save as GeoPackage
        gdf.to_file(path, driver='GPKG')