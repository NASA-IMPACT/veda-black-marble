"""Behavioral tests for output generation components.

These tests use predictable artificial data to verify exact behavior
of output components without external dependencies.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import rasterio
from rasterio.transform import Affine

from blackmarble.components.output import (
    COGGenerator,
    COGProfile,
    MetadataGenerator,
    OutputMetadata,
    OutputValidator,
    VisualizationEnhancer,
)


class TestCOGGenerator:
    """Test COGGenerator creates valid profiles."""
    
    def test_single_band_cog_generation(self):
        """Test generating single-band COG."""
        generator = COGGenerator()
        
        # Create test data
        data = np.arange(100).reshape(10, 10).astype(np.float32)
        metadata = OutputMetadata(
            crs="EPSG:4326",
            transform=Affine(1.0, 0, 0, 0, -1.0, 10),
            bounds=(0, 0, 10, 10),
            shape=(10, 10),
            dtype=np.float32,
            nodata=-999
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.tif"
            
            # Generate COG
            result_path = generator.generate(data, metadata, output_path)
            
            assert result_path.exists()
            
            # Verify COG properties
            with rasterio.open(result_path) as src:
                assert src.count == 1
                assert src.shape == (10, 10)
                assert src.crs.to_string() == "EPSG:4326"
                assert src.nodata == -999
                assert src.driver == "GTiff"
                # Check data is preserved
                read_data = src.read(1)
                assert np.array_equal(read_data, data)
                
    def test_multi_band_cog_generation(self):
        """Test generating multi-band COG."""
        generator = COGGenerator()
        
        # Create 3-band test data
        data = np.array([
            np.ones((5, 5)) * 100,
            np.ones((5, 5)) * 200,
            np.ones((5, 5)) * 250  # Changed from 300 to 250 to fit in uint8
        ], dtype=np.uint8)
        
        metadata = OutputMetadata(
            crs="EPSG:3857",
            transform=Affine(10.0, 0, 1000, 0, -10.0, 2000),
            bounds=(1000, 1950, 1050, 2000),
            shape=(5, 5),
            dtype=np.uint8
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "multiband.tif"
            
            result_path = generator.generate(data, metadata, output_path)
            
            with rasterio.open(result_path) as src:
                assert src.count == 3
                assert src.read(1)[0, 0] == 100
                assert src.read(2)[0, 0] == 200
                assert src.read(3)[0, 0] == 250  # Changed from 300 to 250
                
    def test_cog_profile_options(self):
        """Test different COG profile options."""
        # Test with custom profile
        profile = COGProfile(
            profile_name="lzw",
            blocksize=256,
            overview_level=3,
            overview_resampling="average"
        )
        generator = COGGenerator(profile=profile)
        
        data = np.random.rand(512, 512).astype(np.float32)
        metadata = OutputMetadata(
            crs="EPSG:4326",
            transform=Affine(0.1, 0, 0, 0, -0.1, 50),
            bounds=(0, 0, 51.2, 51.2),
            shape=(512, 512),
            dtype=np.float32
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "custom_profile.tif"
            
            result_path = generator.generate(data, metadata, output_path)
            
            with rasterio.open(result_path) as src:
                # Check that file was created with compression
                assert src.compression is not None
                
    def test_metadata_tags(self):
        """Test adding metadata tags to COG."""
        generator = COGGenerator()
        
        data = np.ones((10, 10), dtype=np.float32)
        metadata = OutputMetadata(
            crs="EPSG:4326",
            transform=Affine(1.0, 0, 0, 0, -1.0, 10),
            bounds=(0, 0, 10, 10),
            shape=(10, 10),
            dtype=np.float32,
            tags={
                "PROCESSING_DATE": "2023-01-15",
                "SENSOR": "VIIRS",
                "VERSION": "1.0"
            }
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "tagged.tif"
            
            result_path = generator.generate(data, metadata, output_path)
            
            with rasterio.open(result_path) as src:
                tags = src.tags()
                assert tags["PROCESSING_DATE"] == "2023-01-15"
                assert tags["SENSOR"] == "VIIRS"
                assert tags["VERSION"] == "1.0"
                
    def test_automatic_overview_calculation(self):
        """Test automatic overview level calculation."""
        generator = COGGenerator()
        
        # Large image should get multiple overview levels
        data = np.ones((2048, 2048), dtype=np.uint8)
        metadata = OutputMetadata(
            crs="EPSG:4326",
            transform=Affine(0.01, 0, 0, 0, -0.01, 20),
            bounds=(0, 0, 20.48, 20.48),
            shape=(2048, 2048),
            dtype=np.uint8
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "overviews.tif"
            
            # Should calculate appropriate overview levels
            result_path = generator.generate(data, metadata, output_path)
            
            assert result_path.exists()
            # COG should have been created with overviews


class TestMetadataGenerator:
    """Test MetadataGenerator produces correct metadata."""
    
    def test_basic_metadata_generation(self):
        """Test generating basic metadata."""
        generator = MetadataGenerator(metadata_format="json")
        
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        metadata = OutputMetadata(
            crs="EPSG:4326",
            transform=Affine(1.0, 0, 0, 0, -1.0, 2),
            bounds=(0, 0, 3, 2),
            shape=(2, 3),
            dtype=np.float32,
            nodata=-999
        )
        
        result = generator.generate(
            data,
            metadata,
            title="Test Data",
            description="Test description"
        )
        
        assert result["type"] == "Feature"
        assert result["properties"]["title"] == "Test Data"
        assert result["properties"]["description"] == "Test description"
        assert result["properties"]["shape"] == (2, 3)
        assert result["properties"]["dtype"] == "float32"
        assert result["properties"]["nodata"] == -999
        assert result["bbox"] == [0, 0, 3, 2]
        
    def test_statistics_calculation(self):
        """Test statistics are calculated correctly."""
        generator = MetadataGenerator()
        
        data = np.array([
            [1, 2, 3, np.nan],
            [4, 5, 6, 7],
            [8, 9, np.nan, 10]
        ], dtype=np.float32)
        
        metadata = OutputMetadata(
            crs="EPSG:4326",
            transform=Affine(1.0, 0, 0, 0, -1.0, 3),
            bounds=(0, 0, 4, 3),
            shape=(3, 4),
            dtype=np.float32
        )
        
        result = generator.generate(data, metadata)
        stats = result["properties"]["statistics"]
        
        assert stats["min"] == 1.0
        assert stats["max"] == 10.0
        assert stats["mean"] == pytest.approx(5.5)
        assert stats["count"] == 10  # 12 total - 2 NaN
        assert stats["valid_percentage"] == pytest.approx(83.33, rel=0.01)
        
    def test_multiband_statistics(self):
        """Test statistics for multi-band data."""
        generator = MetadataGenerator()
        
        data = np.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ], dtype=np.float32)
        
        metadata = OutputMetadata(
            crs="EPSG:4326",
            transform=Affine(1.0, 0, 0, 0, -1.0, 2),
            bounds=(0, 0, 2, 2),
            shape=(2, 2),
            dtype=np.float32
        )
        
        result = generator.generate(data, metadata)
        stats = result["properties"]["statistics"]
        
        assert "band_1" in stats
        assert "band_2" in stats
        assert stats["band_1"]["min"] == 1.0
        assert stats["band_1"]["max"] == 4.0
        assert stats["band_2"]["min"] == 5.0
        assert stats["band_2"]["max"] == 8.0
        
    def test_geometry_generation(self):
        """Test GeoJSON geometry generation."""
        generator = MetadataGenerator()
        
        data = np.ones((5, 5), dtype=np.float32)
        metadata = OutputMetadata(
            crs="EPSG:4326",
            transform=Affine(0.1, 0, -180, 0, -0.1, 90),
            bounds=(-180, 89.5, -179.5, 90),
            shape=(5, 5),
            dtype=np.float32
        )
        
        result = generator.generate(data, metadata)
        
        # Check geometry
        geom = result["geometry"]
        assert geom["type"] == "Polygon"
        coords = geom["coordinates"][0]
        assert len(coords) == 5  # Closed polygon
        assert coords[0] == [-180, 89.5]
        assert coords[1] == [-179.5, 89.5]
        assert coords[2] == [-179.5, 90]
        assert coords[3] == [-180, 90]
        assert coords[4] == coords[0]  # Closed
        
    def test_processing_info_inclusion(self):
        """Test including processing information."""
        generator = MetadataGenerator()
        
        data = np.ones((10, 10), dtype=np.float32)
        metadata = OutputMetadata(
            crs="EPSG:4326",
            transform=Affine(1.0, 0, 0, 0, -1.0, 10),
            bounds=(0, 0, 10, 10),
            shape=(10, 10),
            dtype=np.float32
        )
        
        processing_info = {
            "algorithm": "median_composite",
            "input_scenes": 5,
            "quality_threshold": 0.8
        }
        
        result = generator.generate(
            data,
            metadata,
            processing_info=processing_info
        )
        
        assert "processing" in result["properties"]
        assert result["properties"]["processing"]["algorithm"] == "median_composite"
        assert result["properties"]["processing"]["input_scenes"] == 5
        
    def test_stac_format(self):
        """Test STAC-specific formatting."""
        generator = MetadataGenerator(metadata_format="stac")
        
        data = np.ones((10, 10), dtype=np.float32)
        metadata = OutputMetadata(
            crs="EPSG:4326",
            transform=Affine(1.0, 0, 0, 0, -1.0, 10),
            bounds=(0, 0, 10, 10),
            shape=(10, 10),
            dtype=np.float32
        )
        
        result = generator.generate(
            data,
            metadata,
            id="test-item",
            collection="test-collection",
            datetime="2023-01-15T00:00:00Z"
        )
        
        assert result["stac_version"] == "1.0.0"
        assert result["id"] == "test-item"
        assert result["collection"] == "test-collection"
        assert result["properties"]["datetime"] == "2023-01-15T00:00:00Z"


class TestOutputValidator:
    """Test OutputValidator catches invalid data."""
    
    def test_valid_data_passes(self):
        """Test that valid data passes validation."""
        validator = OutputValidator(
            min_valid_percentage=50.0,
            check_range=True,
            check_nodata=True
        )
        
        data = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ], dtype=np.float32)
        
        metadata = OutputMetadata(
            crs="EPSG:4326",
            transform=Affine(1.0, 0, 0, 0, -1.0, 3),
            bounds=(0, 0, 3, 3),
            shape=(3, 3),
            dtype=np.float32
        )
        
        result = validator.validate(data, metadata, expected_range=(0.0, 1.0))
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert result["statistics"]["valid_percentage"] == 100.0
        
    def test_insufficient_valid_pixels(self):
        """Test detection of insufficient valid pixels."""
        validator = OutputValidator(min_valid_percentage=80.0)
        
        # Create data with 50% NaN
        data = np.array([
            [1, 2, np.nan, np.nan],
            [3, 4, np.nan, np.nan],
            [5, 6, np.nan, np.nan],
            [7, 8, np.nan, np.nan]
        ], dtype=np.float32)
        
        metadata = OutputMetadata(
            crs="EPSG:4326",
            transform=Affine(1.0, 0, 0, 0, -1.0, 4),
            bounds=(0, 0, 4, 4),
            shape=(4, 4),
            dtype=np.float32
        )
        
        with pytest.raises(ValueError, match="Valid pixel percentage"):
            validator.validate(data, metadata)
            
    def test_shape_mismatch(self):
        """Test detection of shape mismatch."""
        validator = OutputValidator()
        
        data = np.ones((5, 5), dtype=np.float32)
        metadata = OutputMetadata(
            crs="EPSG:4326",
            transform=Affine(1.0, 0, 0, 0, -1.0, 3),
            bounds=(0, 0, 3, 3),
            shape=(3, 3),  # Mismatch!
            dtype=np.float32
        )
        
        with pytest.raises(ValueError, match="shape"):
            validator.validate(data, metadata)
            
    def test_range_validation(self):
        """Test value range validation."""
        validator = OutputValidator(check_range=True)
        
        data = np.array([
            [0.1, 0.5, 1.0],
            [1.5, 2.0, 2.5],  # Outside expected range
            [0.2, 0.6, 0.9]
        ], dtype=np.float32)
        
        metadata = OutputMetadata(
            crs="EPSG:4326",
            transform=Affine(1.0, 0, 0, 0, -1.0, 3),
            bounds=(0, 0, 3, 3),
            shape=(3, 3),
            dtype=np.float32
        )
        
        result = validator.validate(data, metadata, expected_range=(0.0, 1.0))
        
        assert result["valid"] is True  # Warnings don't fail validation
        assert len(result["warnings"]) > 0
        assert "range" in result["warnings"][0]
        assert result["statistics"]["actual_range"] == (0.1, 2.5)
        
    def test_nodata_warning(self):
        """Test warning about nodata in float data."""
        validator = OutputValidator(check_nodata=True)
        
        data = np.array([
            [1.0, 2.0, -999],
            [3.0, -999, 4.0],
            [5.0, 6.0, 7.0]
        ], dtype=np.float32)
        
        metadata = OutputMetadata(
            crs="EPSG:4326",
            transform=Affine(1.0, 0, 0, 0, -1.0, 3),
            bounds=(0, 0, 3, 3),
            shape=(3, 3),
            dtype=np.float32,
            nodata=-999
        )
        
        result = validator.validate(data, metadata)
        
        assert result["valid"] is True
        assert len(result["warnings"]) > 0
        assert "NoData" in result["warnings"][0]
        assert "NaN" in result["warnings"][0]
        
    def test_multiband_validation(self):
        """Test validation of multi-band data."""
        validator = OutputValidator(min_valid_percentage=75.0)
        
        # Band 1 has 100% valid, Band 2 has 50% valid
        data = np.array([
            [[1, 2], [3, 4]],
            [[5, np.nan], [np.nan, 6]]
        ], dtype=np.float32)
        
        metadata = OutputMetadata(
            crs="EPSG:4326",
            transform=Affine(1.0, 0, 0, 0, -1.0, 2),
            bounds=(0, 0, 2, 2),
            shape=(2, 2),
            dtype=np.float32
        )
        
        # Should fail because band 2 has only 50% valid
        with pytest.raises(ValueError):
            validator.validate(data, metadata)


class TestVisualizationEnhancer:
    """Test VisualizationEnhancer display methods."""
    
    def test_basic_enhancement(self):
        """Test basic display enhancement."""
        enhancer = VisualizationEnhancer()
        
        # Create data with values 0-10
        data = np.linspace(0, 10, 100).reshape(10, 10).astype(np.float32)
        
        # Enhance to display range 2-8
        enhanced = enhancer.enhance(data, display_min=2.0, display_max=8.0)
        
        assert enhanced.dtype == np.uint8
        assert enhanced.min() == 0  # Values below 2 -> 0
        assert enhanced.max() == 255  # Values at 8 -> 255
        # Check middle value (data[5,0] ≈ 5.05, scaled: (5.05-2)/(8-2)*255 ≈ 130)
        assert enhanced[5, 0] == pytest.approx(130, abs=2)
        
    def test_gamma_correction(self):
        """Test gamma correction application."""
        enhancer = VisualizationEnhancer()
        
        # Create linear gradient
        data = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)
        
        # Apply gamma < 1 (brighten)
        enhanced_bright = enhancer.enhance(data, gamma=0.5)
        # Apply gamma > 1 (darken)
        enhanced_dark = enhancer.enhance(data, gamma=2.0)
        
        # Middle value should be different
        mid_linear = enhancer.enhance(data, gamma=1.0)[5, 0]
        mid_bright = enhanced_bright[5, 0]
        mid_dark = enhanced_dark[5, 0]
        
        assert mid_bright > mid_linear  # Brightened
        assert mid_dark < mid_linear    # Darkened
        
    def test_colormap_application(self):
        """Test applying colormap."""
        # Create custom colormap (red to blue gradient)
        colormap_arrays = {
            "r": np.linspace(255, 0, 256).astype(np.uint8),
            "g": np.zeros(256, dtype=np.uint8),
            "b": np.linspace(0, 255, 256).astype(np.uint8)
        }
        
        enhancer = VisualizationEnhancer(colormap_arrays=colormap_arrays)
        
        # Create test data
        data = np.array([
            [0.0, 0.5, 1.0],
            [0.25, 0.75, 0.5],
            [1.0, 0.0, 0.5]
        ], dtype=np.float32)
        
        rgb = enhancer.apply_colormap(data)
        
        assert rgb.shape == (3, 3, 3)  # RGB channels
        assert rgb.dtype == np.uint8
        
        # Check specific colors
        # Value 0.0 -> Red (255, 0, 0) at position [2,1]
        assert rgb[0, 2, 1] == 255  # R
        assert rgb[1, 2, 1] == 0    # G
        assert rgb[2, 2, 1] == 0    # B
        
        # Value 1.0 -> Blue (0, 0, 255) at position [2,0]
        assert rgb[0, 2, 0] == 0    # R
        assert rgb[1, 2, 0] == 0    # G
        assert rgb[2, 2, 0] == 255  # B
        
    def test_grayscale_fallback(self):
        """Test grayscale output when no colormap specified."""
        enhancer = VisualizationEnhancer()
        
        data = np.array([[0.0, 0.5], [0.75, 1.0]], dtype=np.float32)
        
        rgb = enhancer.apply_colormap(data)
        
        # Should produce grayscale (all channels equal)
        assert np.array_equal(rgb[0], rgb[1])
        assert np.array_equal(rgb[1], rgb[2])
        
    def test_nan_handling(self):
        """Test handling of NaN values in enhancement."""
        enhancer = VisualizationEnhancer()
        
        data = np.array([
            [1.0, np.nan, 3.0],
            [np.nan, 5.0, np.nan],
            [7.0, 8.0, 9.0]
        ], dtype=np.float32)
        
        enhanced = enhancer.enhance(data, display_min=0.0, display_max=10.0)
        
        # NaN should become 0
        assert enhanced[0, 1] == 0
        assert enhanced[1, 0] == 0
        assert enhanced[1, 2] == 0
        
    def test_edge_case_display_range(self):
        """Test edge case where min equals max."""
        enhancer = VisualizationEnhancer()
        
        data = np.ones((5, 5), dtype=np.float32) * 0.5
        
        # When min equals max, should return zeros
        enhanced = enhancer.enhance(data, display_min=0.5, display_max=0.5)
        
        assert np.all(enhanced == 0)