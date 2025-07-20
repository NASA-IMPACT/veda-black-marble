"""Test auxiliary export products from the pipeline.

The pipeline can optionally export urban masks and multi-band
index files in addition to the main visualization output.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

from blackmarble.export import create_cog


class TestUrbanMaskExport:
    """Test urban mask export functionality."""
    
    def test_urban_mask_cog_creation(self):
        """Test exporting urban mask as single-band COG."""
        # Create urban mask (boolean -> uint8)
        shape = (512, 512)
        urban_mask = np.zeros(shape, dtype=bool)
        # Create urban areas
        urban_mask[100:200, 100:200] = True  # City center
        urban_mask[300:350, 300:400] = True  # Suburban area
        
        # Convert to uint8 for export
        urban_mask_uint8 = urban_mask.astype(np.uint8)
        
        transform = from_origin(500000, 4000000, 30, 30)
        crs = CRS.from_epsg(32633)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate pipeline export path manipulation
            main_output = Path(tmpdir) / "blackmarble_output.tif"
            urban_output = Path(tmpdir) / "blackmarble_output_urban_mask.tif"
            
            # Export urban mask
            create_cog(
                data=urban_mask_uint8,
                output_path=str(urban_output),
                transform=transform,
                crs=crs,
                metadata={"band_name": "urban_mask"},
                compress="deflate"
            )
            
            # Verify output
            assert urban_output.exists()
            
            with rasterio.open(urban_output) as src:
                # Check properties
                assert src.count == 1
                assert src.dtype == np.uint8
                assert src.compression.lower() == "deflate"
                
                # Check data
                data = src.read(1)
                assert np.array_equal(data, urban_mask_uint8)
                
                # Check values are binary
                unique_values = np.unique(data)
                assert set(unique_values) <= {0, 1}
                
                # Check metadata
                assert src.tags()["band_name"] == "urban_mask"
    
    def test_urban_mask_with_compression_options(self):
        """Test urban mask export respects compression constraints."""
        urban_mask = np.random.choice([0, 1], size=(256, 256), p=[0.7, 0.3]).astype(np.uint8)
        transform = from_origin(0, 0, 30, 30)
        crs = CRS.from_epsg(32633)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test valid compressions for single-band
            for compress in ["deflate", "lzw"]:
                output_path = Path(tmpdir) / f"urban_{compress}.tif"
                
                create_cog(
                    data=urban_mask,
                    output_path=str(output_path),
                    transform=transform,
                    crs=crs,
                    compress=compress
                )
                
                with rasterio.open(output_path) as src:
                    assert src.compression.lower() == compress


class TestIndicesMultibandExport:
    """Test multi-band indices file export."""
    
    def test_indices_multiband_cog(self):
        """Test exporting NDVI, NDWI, NDUI as multi-band COG."""
        shape = (400, 400)
        
        # Create realistic index values
        ndvi = np.random.uniform(-0.5, 0.8, shape).astype(np.float32)
        ndwi = np.random.uniform(-0.3, 0.5, shape).astype(np.float32)
        ndui = np.random.uniform(0, 0.9, shape).astype(np.float32)
        
        # Add some NaN values (masked areas)
        mask = np.random.random(shape) < 0.1
        ndvi[mask] = np.nan
        ndwi[mask] = np.nan
        ndui[mask] = np.nan
        
        # Stack bands
        indices = np.stack([ndvi, ndwi, ndui], axis=0)
        
        transform = from_origin(0, 0, 30, 30)
        crs = CRS.from_epsg(32633)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "indices.tif"
            
            # Create metadata for bands
            metadata = {
                "band_1_name": "ndvi",
                "band_2_name": "ndwi",
                "band_3_name": "ndui"
            }
            
            create_cog(
                data=indices,
                output_path=str(output_path),
                transform=transform,
                crs=crs,
                metadata=metadata,
                compress="deflate",
                nodata=np.nan
            )
            
            # Verify output
            with rasterio.open(output_path) as src:
                assert src.count == 3
                assert src.dtype == np.float32
                assert src.nodata is not None
                
                # Check band names
                tags = src.tags()
                assert tags["band_1_name"] == "ndvi"
                assert tags["band_2_name"] == "ndwi"
                assert tags["band_3_name"] == "ndui"
                
                # Check data integrity
                for i, expected in enumerate([ndvi, ndwi, ndui]):
                    band_data = src.read(i + 1)
                    # Compare non-NaN values
                    valid_mask = ~np.isnan(expected)
                    assert np.allclose(
                        band_data[valid_mask], 
                        expected[valid_mask],
                        rtol=1e-5
                    )
                    # Check NaN preservation
                    assert np.all(np.isnan(band_data[mask]))
    
    def test_indices_value_ranges(self):
        """Test that exported indices maintain valid ranges."""
        shape = (200, 200)
        
        # Create indices with full valid ranges
        ndvi = np.linspace(-1, 1, shape[0] * shape[1]).reshape(shape).astype(np.float32)
        ndwi = np.linspace(-1, 1, shape[0] * shape[1]).reshape(shape).astype(np.float32)
        ndui = np.linspace(0, 1, shape[0] * shape[1]).reshape(shape).astype(np.float32)
        
        indices = np.stack([ndvi, ndwi, ndui], axis=0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "indices_ranges.tif"
            
            create_cog(
                data=indices,
                output_path=str(output_path),
                transform=from_origin(0, 0, 30, 30),
                crs=CRS.from_epsg(32633),
                compress="lzw"
            )
            
            with rasterio.open(output_path) as src:
                # Check each band maintains its range
                ndvi_read = src.read(1)
                ndwi_read = src.read(2)
                ndui_read = src.read(3)
                
                # NDVI and NDWI: [-1, 1]
                assert -1 <= np.min(ndvi_read) <= np.max(ndvi_read) <= 1
                assert -1 <= np.min(ndwi_read) <= np.max(ndwi_read) <= 1
                
                # NDUI: [0, 1]
                assert 0 <= np.min(ndui_read) <= np.max(ndui_read) <= 1


class TestAuxiliaryExportIntegration:
    """Test auxiliary exports in pipeline context."""
    
    def test_complete_auxiliary_export_set(self):
        """Test exporting all auxiliary products together."""
        shape = (300, 300)
        
        # Main RGB output (from colormap)
        rgb_data = np.random.randint(0, 255, (3, *shape), dtype=np.uint8)
        
        # Urban mask from NDUI threshold
        ndui = np.random.uniform(0, 0.8, shape).astype(np.float32)
        urban_mask = (ndui > 0.3).astype(np.uint8)
        
        # Individual indices
        ndvi = np.random.uniform(-0.3, 0.7, shape).astype(np.float32)
        ndwi = np.random.uniform(-0.2, 0.4, shape).astype(np.float32)
        
        transform = from_origin(500000, 4000000, 30, 30)
        crs = CRS.from_epsg(32633)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "blackmarble_20240115"
            
            # Export main output
            main_output = f"{base_path}.tif"
            create_cog(
                data=rgb_data,
                output_path=main_output,
                transform=transform,
                crs=crs,
                compress="deflate"
            )
            
            # Export urban mask
            urban_output = f"{base_path}_urban_mask.tif"
            create_cog(
                data=urban_mask,
                output_path=urban_output,
                transform=transform,
                crs=crs,
                metadata={"band_name": "urban_mask"},
                compress="deflate"
            )
            
            # Export indices
            indices_output = f"{base_path}_indices.tif"
            indices_stack = np.stack([ndvi, ndwi, ndui], axis=0)
            create_cog(
                data=indices_stack,
                output_path=indices_output,
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
            
            # Verify all outputs exist
            assert Path(main_output).exists()
            assert Path(urban_output).exists()
            assert Path(indices_output).exists()
            
            # Verify they share spatial reference
            outputs = [main_output, urban_output, indices_output]
            transforms = []
            crs_list = []
            
            for output in outputs:
                with rasterio.open(output) as src:
                    transforms.append(src.transform)
                    crs_list.append(src.crs)
            
            # All should have same transform and CRS
            assert all(t == transforms[0] for t in transforms)
            assert all(c == crs_list[0] for c in crs_list)