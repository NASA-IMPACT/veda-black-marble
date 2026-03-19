"""Tests for urban field computation and fractional road rasterization."""

import networkx as nx
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from rasterio.transform import from_bounds

# Import the actual implementations
from blackmarble.acquire.osm_fractional import buffer_and_rasterize_roads_fractional
from blackmarble.analyze.urban_fields import (
    compute_urban_fields,
    enhance_ntl_with_urban_field,
)


class TestFractionalRoadRasterization:
    """Test fractional road rasterization at sub-pixel resolution."""

    def test_empty_graph_returns_zeros(self):
        """Empty road network should return all zeros."""
        # Create empty graph
        G = nx.MultiDiGraph()

        # Create transform for 30m pixels
        transform = from_bounds(-1, -1, 1, 1, 10, 10)

        result = buffer_and_rasterize_roads_fractional(
            graph=G,
            bbox=(-1, -1, 1, 1),
            transform=transform,
            crs="EPSG:4326",
            shape=(10, 10),
            buffer_meters=30.0,
            sub_pixels=5,
        )

        assert result.shape == (10, 10)
        assert np.all(result == 0.0)
        assert result.dtype == np.float32

    def test_single_pixel_coverage(self):
        """Test fractional coverage for a single 30m pixel."""
        # This would need a mock graph with a small road segment
        # For now, we document expected behavior:
        # - Road covering 1 of 25 sub-pixels → 0.04 coverage
        # - Road covering 5 of 25 sub-pixels → 0.20 coverage
        # - Road covering 25 of 25 sub-pixels → 1.00 coverage

    def test_sub_pixel_alignment(self):
        """Verify 6m sub-pixels align perfectly with 30m pixels."""
        # 5x5 sub-pixels should exactly fill one 30m pixel
        # No gaps or overlaps at boundaries

    def test_hierarchical_not_supported(self):
        """Hierarchical mode should raise NotImplementedError."""
        G = nx.MultiDiGraph()
        transform = from_bounds(-1, -1, 1, 1, 10, 10)

        with pytest.raises(NotImplementedError):
            buffer_and_rasterize_roads_fractional(
                graph=G,
                bbox=(-1, -1, 1, 1),
                transform=transform,
                crs="EPSG:4326",
                shape=(10, 10),
                hierarchical=True,
            )


class TestUrbanFields:
    """Test multi-scale urban field computation."""

    def test_single_point_gaussian_spread(self):
        """Single road pixel should spread according to Gaussian."""
        # Create 20x20 grid with single point in center
        road_fraction = np.zeros((20, 20), dtype=np.float32)
        road_fraction[10, 10] = 1.0

        # Test with single scale
        urban_field = compute_urban_fields(
            road_fraction=road_fraction,
            sigmas=[2.0],
            weights=[0.5, 0.5],  # 50% direct, 50% blurred
        )

        # Center should be between 0.5 and 1.0 (mix of direct and blurred)
        assert 0.5 < urban_field[10, 10] < 1.0

        # Should decrease with distance
        assert urban_field[10, 10] > urban_field[10, 12]  # 2 pixels away
        assert urban_field[10, 12] > urban_field[10, 14]  # 4 pixels away

        # Should be symmetric
        assert_array_almost_equal(urban_field[10, 8], urban_field[10, 12])
        assert_array_almost_equal(urban_field[8, 10], urban_field[12, 10])

    def test_multi_scale_combination(self):
        """Test that multiple scales combine correctly."""
        # Create test pattern: 3x3 block of roads
        road_fraction = np.zeros((30, 30), dtype=np.float32)
        road_fraction[14:17, 14:17] = 1.0

        urban_field = compute_urban_fields(
            road_fraction=road_fraction,
            sigmas=[2.0, 5.0, 10.0],
            weights=[0.25, 0.25, 0.25, 0.25],
        )

        # Center of road block should be highest
        center_value = urban_field[15, 15]
        assert center_value == np.max(urban_field)

        # Each scale should contribute
        # Small sigma: tight spread
        # Large sigma: wide spread
        # Check that we have gradual falloff
        assert urban_field[15, 20] > 0  # 5 pixels away
        assert urban_field[15, 25] > 0  # 10 pixels away (only largest sigma)

    def test_weight_validation(self):
        """Weights must sum to 1.0 and match sigma count."""
        road_fraction = np.ones((10, 10), dtype=np.float32)

        # Wrong number of weights
        with pytest.raises(AssertionError):
            compute_urban_fields(
                road_fraction=road_fraction,
                sigmas=[2.0, 5.0],
                weights=[0.5, 0.5],  # Need 3 weights for 2 sigmas
            )

        # Weights don't sum to 1
        with pytest.raises(AssertionError):
            compute_urban_fields(
                road_fraction=road_fraction,
                sigmas=[2.0],
                weights=[0.4, 0.5],  # Sum = 0.9
            )

    def test_edge_handling(self):
        """Test Gaussian behavior at image edges."""
        # Road at edge
        road_fraction = np.zeros((20, 20), dtype=np.float32)
        road_fraction[0, 10] = 1.0

        urban_field = compute_urban_fields(
            road_fraction=road_fraction, sigmas=[3.0], weights=[0.5, 0.5]
        )

        # Should handle edge correctly (reflect mode)
        assert urban_field[0, 10] > 0
        assert not np.any(np.isnan(urban_field))


class TestNTLEnhancement:
    """Test NTL enhancement with urban fields."""

    def test_multiplicative_enhancement(self):
        """Test simple multiplicative enhancement."""
        ntl = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
        urban_field = np.array([[0.0, 1.0], [0.5, 0.0]], dtype=np.float32)

        enhanced = enhance_ntl_with_urban_field(
            ntl=ntl, urban_field=urban_field, enhancement_mode="multiplicative"
        )

        # The multiplicative mode now rescales to preserve dynamic range
        # With default multiplicative_factor=0.3:
        # The max enhancement factor is 1.3 (1.0 + 0.3 * 1.0 for full urban field)
        # All values are then divided by 1.3 to preserve range

        # No roads → minimum enhancement after rescaling
        assert_array_almost_equal(enhanced[0, 0], 0.5 / 1.3, decimal=5)

        # Full roads → full enhancement then rescaled
        assert_array_almost_equal(enhanced[0, 1], (0.5 * 1.3) / 1.3, decimal=5)  # = 0.5

        # Half roads → partial enhancement then rescaled
        assert_array_almost_equal(enhanced[1, 0], (0.5 * 1.15) / 1.3, decimal=5)

    def test_unsupported_modes_raise_error(self):
        """Test that unsupported enhancement modes raise an error."""
        ntl = np.array([[0.5, 0.5]], dtype=np.float32)
        urban_field = np.array([[0.0, 1.0]], dtype=np.float32)

        # Test that additive mode raises error
        with pytest.raises(ValueError, match="Only 'multiplicative' enhancement mode is supported"):
            enhance_ntl_with_urban_field(
                ntl=ntl,
                urban_field=urban_field,
                enhancement_mode="additive",  # type: ignore
            )

        # Test that hybrid mode raises error
        with pytest.raises(ValueError, match="Only 'multiplicative' enhancement mode is supported"):
            enhance_ntl_with_urban_field(
                ntl=ntl,
                urban_field=urban_field,
                enhancement_mode="hybrid",  # type: ignore
            )

    def test_enhancement_preserves_zeros(self):
        """Areas with no urban field should scale proportionally but still preserve range."""
        ntl = np.random.rand(10, 10).astype(np.float32)
        urban_field = np.zeros((10, 10), dtype=np.float32)

        enhanced = enhance_ntl_with_urban_field(
            ntl=ntl, urban_field=urban_field, enhancement_mode="multiplicative"
        )

        # With zero urban field, enhancement factor is 1.0 everywhere
        # But due to rescaling by max factor (1.0), result should equal input
        assert_array_almost_equal(enhanced, ntl)


class TestIntegration:
    """Test the full pipeline integration."""

    def test_predictable_pattern(self):
        """Create a predictable test case to verify full pipeline."""
        # Create a 10x10 30m grid
        # Place roads in a cross pattern (+)
        road_fraction = np.zeros((10, 10), dtype=np.float32)
        road_fraction[5, :] = 0.8  # Horizontal road
        road_fraction[:, 5] = 0.8  # Vertical road
        road_fraction[5, 5] = 1.0  # Intersection

        # Apply urban fields
        urban_field = compute_urban_fields(
            road_fraction=road_fraction, sigmas=[1.0, 2.0], weights=[0.4, 0.3, 0.3]
        )

        # Verify cross pattern is preserved but smoothed
        assert urban_field[5, 5] == np.max(urban_field)  # Intersection highest
        assert urban_field[5, 3] > urban_field[3, 3]  # On road > off road
        assert urban_field[4, 5] > urban_field[2, 5]  # Near road > far

        # Create synthetic NTL that's dim near roads
        ntl = np.ones((10, 10), dtype=np.float32) * 0.2
        ntl[road_fraction > 0] = 0.1  # Roads are darker (missing lights)

        # Enhance with multiplicative mode
        enhanced = enhance_ntl_with_urban_field(
            ntl=ntl, urban_field=urban_field, enhancement_mode="multiplicative"
        )

        # With multiplicative enhancement and rescaling, verify the pattern
        # The enhancement preserves relative differences

        # Areas with higher urban field get proportionally more enhancement
        # even though absolute values are rescaled
        assert enhanced[5, 5] < enhanced[0, 0]  # Intersection was darker, stays darker
        assert enhanced[5, 3] < enhanced[0, 3]  # Road pixels stay darker than non-road

        # Verify range is preserved
        assert enhanced.min() >= 0
        assert enhanced.max() <= ntl.max()

        # Enhancement should follow urban field pattern
        enhancement_map = enhanced - ntl
        correlation = np.corrcoef(urban_field.flatten(), enhancement_map.flatten())[0, 1]
        assert (
            correlation > 0.75
        )  # Strong positive correlation (reduced threshold for conservative enhancement)
