"""
Integration tests for quality components with the adapter pattern.

These tests demonstrate how the quality components integrate with
the Phase 2 adapter architecture for better testability.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from blackmarble.components.quality import (
    CloudMasker,
    AerosolMasker,
    RadianceSaturationMasker,
    VIIRSQualityMasker,
    QualityCompositor,
    MaskStatistics,
    create_landsat_qa_pipeline
)
from blackmarble.components.base import apply_mask, combine_masks
from blackmarble.adapters.base import RasterAdapter
from blackmarble.data_sources.landsat import LandsatDataSource
from blackmarble.data_sources.viirs import VIIRSDataSource


class MockQualityRasterAdapter:
    """Mock raster adapter that provides quality band data."""
    
    def __init__(self, shape=(100, 100)):
        self.shape = shape
        self.written_files = {}
        
    def read(self, path: Path, band: int = None):
        """Return mock raster data based on file type."""
        # Determine data type from path
        path_str = str(path)
        
        if "QA_PIXEL" in path_str:
            # Mix of clear and cloudy pixels
            data = np.random.choice(
                [21824, 21952, 22280, 54596, 54852, 55052],
                size=self.shape,
                p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05]
            ).astype(np.uint16)
        elif "QA_AEROSOL" in path_str:
            # Mix of aerosol levels
            data = np.random.choice(
                [0, 192, 194, 196, 228],
                size=self.shape,
                p=[0.7, 0.1, 0.1, 0.05, 0.05]
            ).astype(np.uint16)
        elif "QA_RADSAT" in path_str:
            # Radiance saturation flags
            data = np.random.choice(
                [0, 4, 8, 12, 16, 28],  # Various bit combinations
                size=self.shape,
                p=[0.8, 0.05, 0.05, 0.05, 0.025, 0.025]
            ).astype(np.uint16)
        elif "Mandatory_Quality_Flag" in path_str:
            # VIIRS quality flags
            data = np.random.choice(
                [0, 2, 1, 3],  # Good, moderate, poor, bad
                size=self.shape,
                p=[0.7, 0.2, 0.05, 0.05]
            ).astype(np.uint8)
        else:
            # Regular data band
            data = np.random.rand(*self.shape).astype(np.float32) * 0.3 + 0.1
            
        metadata = {
            "crs": "EPSG:4326",
            "transform": (0.0001, 0.0, -122.4194, 0.0, -0.0001, 37.8049),
            "width": self.shape[1],
            "height": self.shape[0],
            "count": 1,
            "dtype": str(data.dtype),
            "nodata": -9999.0,
            "driver": "GTiff",
        }
        
        return data, metadata
    
    def write(self, path: Path, data: np.ndarray, metadata: dict, compress: bool = True):
        """Mock write - track what was written."""
        self.written_files[str(path)] = {
            "data": data,
            "metadata": metadata,
            "compress": compress,
        }
    
    def get_info(self, path: Path) -> dict:
        """Get mock raster info."""
        return {
            "driver": "GTiff",
            "width": self.shape[1],
            "height": self.shape[0],
            "count": 1,
            "crs": "EPSG:4326",
            "bounds": (-122.4194, 37.7749, -122.3894, 37.8049),
            "dtype": ["uint16"] if "QA" in str(path) else ["float32"],
            "nodata": -9999.0,
            "shape": (1, self.shape[0], self.shape[1]),
        }


@pytest.fixture
def mock_quality_adapter():
    """Provide a mock quality raster adapter."""
    return MockQualityRasterAdapter()


def test_quality_components_with_mock_data(mock_quality_adapter):
    """Test quality components with mock adapter data."""
    
    # Read mock QA bands
    qa_pixel, _ = mock_quality_adapter.read(Path("QA_PIXEL.tif"))
    qa_aerosol, _ = mock_quality_adapter.read(Path("QA_AEROSOL.tif"))
    qa_radsat, _ = mock_quality_adapter.read(Path("QA_RADSAT.tif"))
    
    # Create quality pipeline
    pipeline = create_landsat_qa_pipeline(strategy="conservative", dilate_clouds=1)
    
    # Process QA data
    result = pipeline.process({
        "qa_pixel": qa_pixel,
        "qa_aerosol": qa_aerosol
    })
    
    # Verify result structure
    assert "mask" in result.__dict__
    assert "statistics" in result.__dict__
    assert "metadata" in result.__dict__
    
    # Check statistics
    stats = result.statistics
    assert stats["total_pixels"] == qa_pixel.size
    assert 0 <= stats["valid_pixels"] <= stats["total_pixels"]
    assert 0 <= stats["valid_percentage"] <= 100
    
    # Check metadata
    metadata = result.metadata
    assert metadata["combination_method"] == "and"
    assert "component_statistics" in metadata
    assert "qa_pixel" in metadata["component_statistics"]
    assert "qa_aerosol" in metadata["component_statistics"]


def test_quality_enhanced_landsat_source(mock_quality_adapter):
    """Test quality-enhanced Landsat source with mocks."""
    
    # Import the enhanced source from the example
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
    from pipeline_with_quality_components import QualityEnhancedLandsatSource
    
    # Create mock storage adapter
    mock_storage = Mock()
    mock_storage.list_objects.return_value = [
        "B3.tif", "B4.tif", "B5.tif", 
        "QA_PIXEL.tif", "QA_AEROSOL.tif", "QA_RADSAT.tif"
    ]
    
    # Create enhanced source
    source = QualityEnhancedLandsatSource(
        storage_adapter=mock_storage,
        raster_adapter=mock_quality_adapter,
        quality_strategy="conservative",
        dilate_clouds=1
    )
    
    # Mock the parent fetch method
    with patch.object(LandsatDataSource, 'fetch') as mock_fetch:
        # Prepare mock data
        shape = mock_quality_adapter.shape
        mock_data = {
            "B3": np.random.rand(*shape).astype(np.float32) * 0.3 + 0.1,
            "B4": np.random.rand(*shape).astype(np.float32) * 0.3 + 0.1,
            "B5": np.random.rand(*shape).astype(np.float32) * 0.3 + 0.1,
            "QA_PIXEL": np.random.choice([21824, 21952, 22280, 54596], size=shape),
            "QA_AEROSOL": np.random.choice([0, 192, 194], size=shape),
            "QA_RADSAT": np.random.choice([0, 4, 8, 12], size=shape),
        }
        mock_fetch.return_value = mock_data
        
        # Fetch with quality processing
        result, report = source.fetch_with_quality(
            bbox=(-122.4194, 37.7749, -122.3894, 37.8049),
            start_date="2023-01-01",
            end_date="2023-06-30",
            bands=["B3", "B4", "B5"],
            return_quality_report=True
        )
    
    # Verify data was processed
    assert all(band in result for band in ["B3", "B4", "B5"])
    assert all(qa in result for qa in ["QA_PIXEL", "QA_AEROSOL", "QA_RADSAT"])
    
    # Verify quality report
    assert "quality_summary" in report
    assert "cloud_aerosol_details" in report
    assert "processing_strategy" in report
    
    # Check that masks were applied
    summary = report["quality_summary"]
    assert summary["total_masks"] > 0
    assert "cloud_aerosol_combined" in summary["mask_names"]


def test_quality_component_composition():
    """Test composing multiple quality components."""
    
    # Create sample data
    shape = (50, 50)
    qa_pixel = np.full(shape, 21824, dtype=np.uint16)  # All clear land
    qa_pixel[10:20, 10:20] = 54596  # Add cloud patch
    
    qa_aerosol = np.zeros(shape, dtype=np.uint16)  # All clear
    qa_aerosol[15:25, 15:25] = 196  # Add high aerosol patch
    
    # Create individual components
    cloud_masker = CloudMasker(strategy="conservative")
    aerosol_masker = AerosolMasker(threshold="standard")
    
    # Test AND composition
    and_compositor = QualityCompositor(
        components=[cloud_masker, aerosol_masker],
        combination_method="and"
    )
    
    and_result = and_compositor.process({
        "cloud_masker": qa_pixel,
        "aerosol_masker": qa_aerosol
    })
    
    # Test OR composition
    or_compositor = QualityCompositor(
        components=[cloud_masker, aerosol_masker],
        combination_method="or"
    )
    
    or_result = or_compositor.process({
        "cloud_masker": qa_pixel,
        "aerosol_masker": qa_aerosol
    })
    
    # AND should be more restrictive than OR
    assert and_result.statistics["valid_pixels"] <= or_result.statistics["valid_pixels"]
    
    # Check that cloud and aerosol patches were detected
    cloud_stats = and_result.metadata["component_statistics"]["cloud_masker"]
    aerosol_stats = and_result.metadata["component_statistics"]["aerosol_masker"]
    
    assert cloud_stats["valid_pixels"] < shape[0] * shape[1]  # Some clouds masked
    assert aerosol_stats["valid_pixels"] < shape[0] * shape[1]  # Some aerosol masked


def test_mask_statistics_tracking():
    """Test mask statistics tracking across multiple operations."""
    
    tracker = MaskStatistics()
    
    # Create series of masks with decreasing valid pixels
    shape = (100, 100)
    masks = [
        np.ones(shape),  # All valid
        np.random.rand(*shape) > 0.2,  # ~80% valid
        np.random.rand(*shape) > 0.5,  # ~50% valid
        np.random.rand(*shape) > 0.8,  # ~20% valid
    ]
    
    # Track each mask
    for i, mask in enumerate(masks):
        tracker.process(mask, mask_name=f"mask_{i+1}")
    
    # Get summary
    summary = tracker.get_summary()
    
    assert summary["total_masks"] == 4
    assert summary["total_pixels_processed"] == 4 * shape[0] * shape[1]
    assert len(summary["mask_names"]) == 4
    assert all(f"mask_{i+1}" in summary["individual_stats"] for i in range(4))
    
    # Verify decreasing valid percentages
    stats = summary["individual_stats"]
    assert stats["mask_1"]["valid_percentage"] > stats["mask_2"]["valid_percentage"]
    assert stats["mask_2"]["valid_percentage"] > stats["mask_3"]["valid_percentage"]
    assert stats["mask_3"]["valid_percentage"] > stats["mask_4"]["valid_percentage"]


def test_viirs_quality_masking():
    """Test VIIRS quality masking component."""
    
    # Create sample VIIRS quality flags
    shape = (100, 100)
    quality_flags = np.random.choice(
        [0, 2, 1, 3],  # Good, moderate, poor quality, bad quality
        size=shape,
        p=[0.6, 0.25, 0.1, 0.05]
    ).astype(np.uint8)
    
    # Create VIIRS quality masker
    masker = VIIRSQualityMasker()
    
    # Process quality flags
    result = masker.process(quality_flags)
    
    # Check statistics
    stats = result.statistics
    assert stats["total_pixels"] == shape[0] * shape[1]
    assert stats["valid_pixels"] == np.sum(np.isin(quality_flags, [0, 2]))
    
    # Check metadata
    metadata = result.metadata
    assert "qa_value_distribution" in metadata
    assert metadata["valid_flags"] == [0, 2]
    
    # Verify value distribution exists
    qa_dist = metadata["qa_value_distribution"]
    assert 0 in qa_dist  # High quality value
    assert 2 in qa_dist  # Moderate quality value


def test_radiance_saturation_masking():
    """Test radiance saturation masking for individual bands."""
    
    # Create sample QA_RADSAT data with known bit patterns
    shape = (50, 50)
    qa_radsat = np.zeros(shape, dtype=np.uint16)
    
    # Set specific saturation patterns
    qa_radsat[10:20, :] = 4   # Band 3 saturated (bit 2)
    qa_radsat[20:30, :] = 8   # Band 4 saturated (bit 3)
    qa_radsat[30:40, :] = 16  # Band 5 saturated (bit 4)
    qa_radsat[40:50, :] = 28  # All bands saturated (bits 2, 3, 4)
    
    masker = RadianceSaturationMasker()
    
    # Test individual band masking
    for band_num in [3, 4, 5]:
        result = masker.process(qa_radsat, band=band_num)
        
        stats = result.statistics
        bit_position = band_num - 1
        
        # Calculate expected saturated pixels
        if band_num == 3:
            expected_saturated = 20 * 50  # rows 10-20 and 40-50
        elif band_num == 4:
            expected_saturated = 20 * 50  # rows 20-30 and 40-50
        else:  # band 5
            expected_saturated = 20 * 50  # rows 30-40 and 40-50
            
        assert result.metadata["saturated_pixels"] == expected_saturated
        assert stats["valid_pixels"] == shape[0] * shape[1] - expected_saturated
    
    # Test multi-band masking
    multi_result = masker.process(qa_radsat)
    
    assert multi_result.mask.shape == (3, *shape)  # 3 bands
    assert "band_statistics" in multi_result.statistics
    
    # Verify band statistics
    band_stats = multi_result.statistics["band_statistics"]
    assert all(f"band_{b}" in band_stats for b in [3, 4, 5])


if __name__ == "__main__":
    # Run specific test
    import sys
    
    adapter = MockQualityRasterAdapter()
    test_quality_components_with_mock_data(adapter)
    print("✓ Quality components with mock data test passed")
    
    test_quality_component_composition()
    print("✓ Quality component composition test passed")
    
    test_mask_statistics_tracking()
    print("✓ Mask statistics tracking test passed")
    
    test_viirs_quality_masking()
    print("✓ VIIRS quality masking test passed")
    
    test_radiance_saturation_masking()
    print("✓ Radiance saturation masking test passed")
    
    print("\n✅ All integration tests passed!")