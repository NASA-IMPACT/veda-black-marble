"""Test advanced scipy features in temporal processing.

This module tests the scipy-based features:
- Advanced gap filling with nearest neighbor interpolation
- Feather blending for seamless scene merging
"""

import numpy as np
import pytest

# Test if scipy is available
try:
    import scipy.ndimage

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from blackmarble.components.temporal import TemporalCompositor, SceneMerger


class TestScipyGapFilling:
    """Test scipy-based gap filling feature."""

    @pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not installed")
    def test_gap_filling_enabled(self):
        """Test gap filling with scipy enabled."""
        # Create test data with gaps
        data = np.array(
            [
                [[1, 2, np.nan, 4], [5, np.nan, np.nan, 8], [9, 10, 11, np.nan]],
                [[np.nan, 2, 3, np.nan], [5, 6, 7, 8], [np.nan, 10, np.nan, 12]],
                [[1, np.nan, 3, 4], [np.nan, 6, np.nan, 8], [9, np.nan, 11, 12]],
            ]
        )

        # Create compositor with scipy gap filling
        compositor = TemporalCompositor(method="median", use_scipy_fill_gaps=True)

        # Process data
        result = compositor.process(data)
        assert result.metadata is not None
        assert "composite" in result.metadata
        composite = result.metadata["composite"]

        # Check that gaps are filled
        assert not np.any(np.isnan(composite))

        # Check that filled values are reasonable
        # The gap at position [0,2] should be filled with nearby values
        assert 1 <= composite[0, 2] <= 3

    def test_gap_filling_disabled(self):
        """Test with gap filling disabled (default)."""
        # Create test data with gaps
        data = np.array(
            [
                [[1, 2, np.nan, 4], [5, np.nan, np.nan, 8], [9, 10, 11, np.nan]],
                [[np.nan, 2, 3, np.nan], [5, 6, 7, 8], [np.nan, 10, np.nan, 12]],
                [[1, np.nan, 3, 4], [np.nan, 6, np.nan, 8], [9, np.nan, 11, 12]],
            ]
        )

        # Create compositor without scipy gap filling
        compositor = TemporalCompositor(
            method="median", use_scipy_fill_gaps=False, min_valid_observations=2
        )

        # Process data
        result = compositor.process(data)
        assert result.metadata is not None
        assert "composite" in result.metadata
        composite = result.metadata["composite"]

        # Some gaps should remain where we have < 2 valid observations
        # Position [0,2] has only 1 valid observation (value 3 from time step 2)
        # Position [1,1] has 2 valid observations (values 6 from time steps 1,2)
        assert composite[1, 1] == 6  # Median of [6, 6]

    def test_scipy_import(self):
        """Test that scipy can be imported."""
        import scipy.ndimage

        assert hasattr(scipy.ndimage, "distance_transform_edt")

    @pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not installed")
    def test_multiband_gap_filling(self):
        """Test gap filling with multi-band data."""
        # Create 4D test data (time, bands, height, width)
        data = np.random.rand(3, 2, 10, 10)
        # Add some gaps
        data[0, :, 0:3, 0:3] = np.nan
        data[1, :, 7:10, 7:10] = np.nan
        data[2, :, 4:6, 4:6] = np.nan

        compositor = TemporalCompositor(method="median", use_scipy_fill_gaps=True)

        result = compositor.process(data)
        assert result.metadata is not None
        composite = result.metadata["composite"]

        # Check that all gaps are filled
        assert not np.any(np.isnan(composite))
        assert composite.shape == (2, 10, 10)


class TestScipyFeatherBlending:
    """Test scipy-based feather blending feature."""

    @pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not installed")
    def test_feather_blending(self):
        """Test feather blending creates smooth transitions."""
        # Create two overlapping scenes
        scene1 = np.ones((10, 10)) * 100
        scene2 = np.ones((10, 10)) * 200

        # Create transforms (identity transforms with offsets)
        from rasterio.transform import Affine

        transform1 = Affine(1, 0, 0, 0, -1, 10)
        transform2 = Affine(1, 0, 5, 0, -1, 10)  # Offset by 5 pixels in x

        bounds1 = (0, 0, 10, 10)
        bounds2 = (5, 0, 15, 10)

        # Create merger with feather blending
        merger = SceneMerger(blend_method="feather", feather_distance=3)

        # Merge scenes
        merged, merged_transform, merged_bounds = merger.merge_scenes(
            [scene1, scene2], [transform1, transform2], [bounds1, bounds2]
        )

        # Check overlap region has smooth transition
        # The merged array shape should be (10, 15) to cover both scenes
        assert merged.shape == (10, 15)

        # The overlap is from x=5 to x=10 in the merged output
        # This corresponds to the region where both scenes have data
        overlap_region = merged[:, 5:10]

        # Values should be between 100 and 200
        assert np.all(overlap_region >= 100)
        assert np.all(overlap_region <= 200)

        # Should have some intermediate values (not just 100 or 200)
        unique_values = np.unique(overlap_region)
        # With feathering, we should see blended values
        assert len(unique_values) > 2, f"Expected blended values, got {unique_values}"

    def test_non_feather_blending(self):
        """Test non-feather blending methods don't require scipy."""
        # Create two overlapping scenes
        scene1 = np.ones((10, 10)) * 100
        scene2 = np.ones((10, 10)) * 200

        from rasterio.transform import Affine

        transform1 = Affine(1, 0, 0, 0, -1, 10)
        transform2 = Affine(1, 0, 5, 0, -1, 10)

        bounds1 = (0, 0, 10, 10)
        bounds2 = (5, 0, 15, 10)

        # Test max blending (doesn't need scipy)
        merger = SceneMerger(blend_method="max")
        merged, _, _ = merger.merge_scenes(
            [scene1, scene2], [transform1, transform2], [bounds1, bounds2]
        )

        # Overlap region should have max value (200)
        assert merged[5, 5] == 200

        # Test mean blending (doesn't need scipy)
        merger = SceneMerger(blend_method="mean")
        merged, _, _ = merger.merge_scenes(
            [scene1, scene2], [transform1, transform2], [bounds1, bounds2]
        )

        # Overlap region should have mean value (150)
        assert merged[5, 5] == 150

    def test_scipy_feather_import(self):
        """Test that scipy can be imported for feather blending."""
        import scipy.ndimage

        merger = SceneMerger(blend_method="feather")
        assert merger.blend_method == "feather"


class TestDocumentation:
    """Test that documentation is comprehensive."""

    def test_module_docstring(self):
        """Test module has advanced features documented."""
        from blackmarble.components import temporal

        assert temporal.__doc__ is not None
        assert "Advanced Features" in temporal.__doc__
        assert "scipy" in temporal.__doc__
        assert "Gap Filling" in temporal.__doc__
        assert "Feather Blending" in temporal.__doc__

    def test_compositor_docstring(self):
        """Test TemporalCompositor documents scipy parameter."""
        assert TemporalCompositor.__init__.__doc__ is not None
        assert "use_scipy_fill_gaps" in TemporalCompositor.__init__.__doc__
        assert (
            "advanced" in TemporalCompositor.__init__.__doc__
            or "scipy" in TemporalCompositor.__init__.__doc__
        )
        assert "scipy" in TemporalCompositor.__init__.__doc__

    def test_merger_docstring(self):
        """Test SceneMerger documents feather blending."""
        assert SceneMerger.__init__.__doc__ is not None
        assert "feather" in SceneMerger.__init__.__doc__
        assert (
            "smooth" in SceneMerger.__init__.__doc__
            or "seamless" in SceneMerger.__init__.__doc__
        )

    def test_gap_filling_method_docstring(self):
        """Test gap filling method is well documented."""
        # Check the renamed method exists
        assert hasattr(TemporalCompositor, "_fill_gaps_scipy")
        method = getattr(TemporalCompositor, "_fill_gaps_scipy")
        assert "Euclidean distance transform" in method.__doc__
        assert "nearest valid pixel" in method.__doc__

    def test_feather_blend_method_docstring(self):
        """Test feather blend method is well documented."""
        assert hasattr(SceneMerger, "_feather_blend")
        method = getattr(SceneMerger, "_feather_blend")
        assert "distance transform" in method.__doc__
        assert "smooth" in method.__doc__
        assert "seamless" in method.__doc__
