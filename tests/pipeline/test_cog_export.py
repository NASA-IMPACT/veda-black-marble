"""Test Cloud Optimized GeoTIFF (COG) export functionality.

The pipeline exports final results as COGs for efficient cloud-based access.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

from blackmarble.export import (
    create_cog,
    validate_for_cog,
    create_metadata
)


class TestCOGCreation:
    """Test COG creation functionality."""
    
    def test_basic_cog_creation(self):
        """Test creating a basic COG file."""
        # Create test data
        data = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        transform = from_origin(0, 1000, 10, 10)
        crs = CRS.from_epsg(32633)  # UTM Zone 33N
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.tif"
            
            # Create COG
            create_cog(
                data=data,
                output_path=str(output_path),
                transform=transform,
                crs=crs,
                compress="deflate"
            )
            
            # Verify output
            assert output_path.exists()
            
            # Check COG properties
            with rasterio.open(output_path) as src:
                assert src.driver == "GTiff"
                assert src.compression.lower() == "deflate"
                assert src.is_tiled
                assert src.crs == crs
                assert src.transform == transform
                assert src.count == 1
                assert src.dtype == data.dtype
    
    def test_rgb_cog_creation(self):
        """Test creating RGB COG (3-band)."""
        # Create RGB data (3, height, width)
        data = np.random.randint(0, 255, (3, 256, 256), dtype=np.uint8)
        transform = from_origin(100, 200, 30, 30)
        crs = CRS.from_epsg(4326)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rgb.tif"
            
            create_cog(
                data=data,
                output_path=str(output_path),
                transform=transform,
                crs=crs,
                compress="lzw"
            )
            
            with rasterio.open(output_path) as src:
                assert src.count == 3
                assert src.compression.lower() == "lzw"
                # Check band data
                for i in range(3):
                    band = src.read(i + 1)
                    assert np.array_equal(band, data[i])
    
    def test_cog_with_overviews(self):
        """Test COG creation with overviews (pyramids)."""
        data = np.ones((1024, 1024), dtype=np.float32) * 100
        transform = from_origin(0, 0, 1, 1)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "overviews.tif"
            
            create_cog(
                data=data,
                output_path=str(output_path),
                transform=transform,
                crs=CRS.from_epsg(3857),
                overviews=[2, 4, 8]
            )
            
            with rasterio.open(output_path) as src:
                # Check overviews exist
                assert src.overviews(1) == [2, 4, 8]
    
    def test_cog_with_metadata(self):
        """Test COG with metadata."""
        data = np.ones((100, 100), dtype=np.uint16)
        transform = from_origin(0, 0, 10, 10)
        
        metadata = {
            "PROCESSED_BY": "blackmarble-pipeline",
            "PROCESSING_DATE": "2024-01-15",
            "INDEX_TYPE": "NDUI",
            "NTL_SOURCE": "VIIRS"
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metadata.tif"
            
            create_cog(
                data=data,
                output_path=str(output_path),
                transform=transform,
                crs=CRS.from_epsg(32633),
                metadata=metadata
            )
            
            with rasterio.open(output_path) as src:
                # Check metadata
                for key, value in metadata.items():
                    assert src.tags()[key] == value
    
    def test_cog_with_nodata(self):
        """Test COG with nodata value."""
        data = np.ones((200, 200), dtype=np.float32)
        data[50:100, 50:100] = np.nan
        transform = from_origin(0, 0, 30, 30)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nodata.tif"
            
            create_cog(
                data=data,
                output_path=str(output_path),
                transform=transform,
                crs=CRS.from_epsg(32633),
                nodata=np.nan
            )
            
            with rasterio.open(output_path) as src:
                assert src.nodata is not None
                # Read and check NaN handling
                read_data = src.read(1, masked=True)
                assert read_data.mask[50:100, 50:100].all()


class TestCOGValidation:
    """Test COG validation functionality."""
    
    def test_validate_good_data(self):
        """Test validation of data suitable for COG."""
        data = np.random.rand(512, 512).astype(np.float32)
        transform = from_origin(0, 0, 10, 10)
        crs = CRS.from_epsg(32633)
        
        result = validate_for_cog(
            data=data,
            transform=transform,
            crs=crs
        )
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0
    
    def test_validate_bad_transform(self):
        """Test validation catches invalid transform."""
        data = np.ones((100, 100))
        transform = None  # Invalid
        crs = CRS.from_epsg(32633)
        
        result = validate_for_cog(
            data=data,
            transform=transform,
            crs=crs
        )
        
        assert result["valid"] is False
        assert any("transform" in str(e).lower() for e in result["errors"])
    
    def test_validate_tile_alignment(self):
        """Test validation of tile alignment."""
        # Non-tile-aligned size (not multiple of 256)
        data = np.ones((333, 333))
        transform = from_origin(0, 0, 10, 10)
        crs = CRS.from_epsg(32633)
        
        result = validate_for_cog(
            data=data,
            transform=transform,
            crs=crs,
            check_tiling=True
        )
        
        # Should warn but not fail
        assert result["valid"] is True
        assert any("tile" in str(w).lower() for w in result.get("warnings", []))


class TestMetadataCreation:
    """Test metadata creation for COGs."""
    
    def test_create_metadata(self):
        """Test creating comprehensive metadata."""
        from datetime import datetime
        
        metadata = create_metadata(
            data_sources={
                "landsat": {
                    "files": ["LC08_L2SP_123_045_20240101_B4.TIF"],
                    "date_range": "2024-01-01",
                    "sensor": "Landsat 8",
                    "processing_level": "L2SP"
                },
                "viirs": {
                    "files": ["VNP46A2_h19v04_20240101.tif"],
                    "date_range": "2024-01-01",
                    "sensor": "VIIRS",
                    "product": "VNP46A2"
                }
            },
            processing_steps=[
                "cloud_masking",
                "reprojection",
                "index_calculation",
                "enhancement"
            ],
            indices_calculated=["ndvi", "ndui"],
            date_processed=datetime(2024, 1, 15),
            bbox=(-100, 30, -99, 31),
            crs=CRS.from_epsg(32614),
            resolution=30
        )
        
        # Check metadata structure
        assert "PROCESSING_SOFTWARE" in metadata
        assert "blackmarble" in metadata["PROCESSING_SOFTWARE"].lower()
        assert metadata["PROCESSING_DATE"] == "2024-01-15"
        assert "LANDSAT_FILES" in metadata
        assert metadata["INDICES"] == "ndvi,ndui"
        assert metadata["RESOLUTION_M"] == "30"
        assert "32614" in metadata["CRS"]


class TestCOGExportIntegration:
    """Test COG export in pipeline context."""
    
    def test_export_pipeline_result(self):
        """Test exporting typical pipeline output as COG."""
        # Simulate pipeline output
        # RGB data from inferno colormap
        shape = (3, 1024, 1024)
        rgb_data = np.random.randint(0, 255, shape, dtype=np.uint8)
        
        # Typical UTM transform
        transform = from_origin(500000, 4000000, 30, 30)
        crs = CRS.from_epsg(32633)
        
        # Comprehensive metadata
        metadata = {
            "DESCRIPTION": "Black Marble urban mapping output",
            "PROCESSING_DATE": "2024-01-15",
            "NTL_SOURCE": "VIIRS VNP46A2",
            "OPTICAL_SOURCE": "Landsat 8/9 Collection 2",
            "ROAD_SOURCE": "OpenStreetMap",
            "INDEX": "NDUI",
            "ENHANCEMENT": "urban_field_multiplicative",
            "COLORMAP": "inferno",
            "RESOLUTION_M": "30",
            "CRS": "EPSG:32633"
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "blackmarble_result.tif"
            
            create_cog(
                data=rgb_data,
                output_path=str(output_path),
                transform=transform,
                crs=crs,
                metadata=metadata,
                compress="deflate",
                tiled=True,
                overviews=[2, 4, 8, 16]
            )
            
            # Verify comprehensive output
            with rasterio.open(output_path) as src:
                # Check structure
                assert src.count == 3  # RGB
                assert src.is_tiled
                assert src.compression.lower() == "deflate"
                assert src.overviews(1) == [2, 4, 8, 16]
                
                # Check metadata preserved
                tags = src.tags()
                assert tags["INDEX"] == "NDUI"
                assert tags["COLORMAP"] == "inferno"
                
                # Check data integrity
                for i in range(3):
                    band = src.read(i + 1)
                    assert band.shape == (1024, 1024)
                    assert band.dtype == np.uint8
    
    def test_export_indices_multiband(self):
        """Test exporting multiple indices as multiband COG."""
        # NDVI, NDWI, NDUI bands
        shape = (3, 500, 500)
        indices = np.random.rand(*shape).astype(np.float32)
        indices[0] = (indices[0] - 0.5) * 2  # NDVI [-1, 1]
        indices[1] = (indices[1] - 0.5) * 2  # NDWI [-1, 1]
        indices[2] = indices[2]  # NDUI [0, 1]
        
        # Add some NaN values
        indices[:, 100:150, 100:150] = np.nan
        
        transform = from_origin(0, 0, 30, 30)
        crs = CRS.from_epsg(32633)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "indices.tif"
            
            create_cog(
                data=indices,
                output_path=str(output_path),
                transform=transform,
                crs=crs,
                metadata={
                    "band_1_name": "ndvi",
                    "band_2_name": "ndwi", 
                    "band_3_name": "ndui"
                },
                compress="deflate",
                nodata=np.nan
            )
            
            with rasterio.open(output_path) as src:
                assert src.count == 3
                assert src.nodata is not None
                
                # Check band names in metadata
                assert src.tags()["band_1_name"] == "ndvi"
                assert src.tags()["band_2_name"] == "ndwi"
                assert src.tags()["band_3_name"] == "ndui"