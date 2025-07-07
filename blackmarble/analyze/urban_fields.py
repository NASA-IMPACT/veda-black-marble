"""Multi-scale urban field computation from road networks.

This module implements anisotropic urban density fields that capture
the spatial influence of road infrastructure at multiple scales.
"""

from typing import Any, Literal
import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter

# Type aliases
ArrayLike = npt.NDArray[Any]


def compute_urban_fields(
    road_fraction: ArrayLike,
    sigmas: list[float] = [2.0, 5.0, 10.0],
    weights: list[float] = [0.2, 0.3, 0.3, 0.2],
) -> ArrayLike:
    """Compute anisotropic urban density fields.

    Creates a multi-scale representation of urban density by combining
    the direct road coverage with Gaussian-blurred versions at different
    scales. This captures both immediate road presence and the broader
    urban context.

    Args:
        road_fraction: Fractional road coverage (0.0-1.0) per pixel
        sigmas: Gaussian filter scales in pixels. Each sigma creates
            a density field at that scale. Default [2.0, 5.0, 10.0]
            corresponds to ~60m, ~150m, ~300m for 30m pixels.
        weights: Weights for combining [direct, scale1, scale2, ...].
            Must sum to 1.0 and have length = len(sigmas) + 1.

    Returns:
        Urban field combining multiple scales, values in [0.0, 1.0]

    Examples:
        >>> # Simple case with single scale
        >>> roads = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        >>> field = compute_urban_fields(roads, sigmas=[1.0], weights=[0.5, 0.5])
        >>> # Center column (roads) will have highest values
        >>> assert np.all(field[:, 1] > field[:, 0])
    """
    # Validate inputs
    assert len(weights) == len(sigmas) + 1, (
        f"Need {len(sigmas) + 1} weights for {len(sigmas)} sigmas, got {len(weights)}"
    )
    assert abs(sum(weights) - 1.0) < 1e-6, (
        f"Weights must sum to 1.0, got {sum(weights)}"
    )

    # Direct coverage (unblurred)
    fields = [road_fraction.astype(np.float32)]

    # Multi-scale densities
    for sigma in sigmas:
        # Use reflect mode to handle edges properly
        density = gaussian_filter(
            road_fraction.astype(np.float32), sigma=sigma, mode="reflect"
        )
        fields.append(density)

    # Weighted combination
    urban_field = np.zeros_like(road_fraction, dtype=np.float32)
    for field, weight in zip(fields, weights):
        urban_field += weight * field

    # Ensure output is in valid range
    urban_field = np.clip(urban_field, 0.0, 1.0)

    return urban_field


def enhance_ntl_with_urban_field(
    ntl: ArrayLike,
    urban_field: ArrayLike,
    enhancement_mode: Literal["additive", "multiplicative", "hybrid"] = "hybrid",
    multiplicative_factor: float = 0.15,
    additive_factor: float = 0.05,
    hybrid_expected_scale: float = 0.15,
    hybrid_expected_sqrt_scale: float = 0.05,
    hybrid_application_factor: float = 0.5,
) -> ArrayLike:
    """Enhance NTL based on urban density field.

    Applies enhancement to nighttime lights based on the urban density
    field, which represents the presence and influence of road infrastructure.

    Args:
        ntl: Nighttime lights data (can be in any range, e.g., 0-3000 nW/cm²/sr)
        urban_field: Urban density field from compute_urban_fields (0.0-1.0)
        enhancement_mode: How to apply enhancement:
            - "multiplicative": Simple multiplication (legacy approach)
            - "additive": Add urban glow independent of current brightness
            - "hybrid": Context-aware enhancement based on expected vs actual light
        multiplicative_factor: Factor for multiplicative mode (default: 0.15)
        additive_factor: Factor for additive mode (default: 0.05)
        hybrid_expected_scale: Scale for linear term in hybrid mode (default: 0.15)
        hybrid_expected_sqrt_scale: Scale for sqrt term in hybrid mode (default: 0.05)
        hybrid_application_factor: How much of the missing light to add (default: 0.5)

    Returns:
        Enhanced NTL array

    Notes:
        The hybrid mode implements the key insight that roads indicate where
        we expect urban activity, so areas with roads but low NTL are likely
        undermeasured (e.g., residential areas with tree cover, new LED lights).

        For VIIRS data with high values (0-3000), the enhancement uses percentage-based
        scaling to ensure visible enhancement regardless of the absolute NTL range.
    """
    if enhancement_mode == "multiplicative":
        # Multiplicative enhancement with dynamic range preservation
        enhancement_factor = 1.0 + multiplicative_factor * urban_field
        enhanced = ntl * enhancement_factor
        # Rescale to preserve dynamic range (like old 1.3x then /1.3 approach)
        max_factor = enhancement_factor.max()
        return enhanced / max_factor

    elif enhancement_mode == "additive":
        # Pure additive: NTL + (factor * urban_field)
        # Adds urban glow/scatter independent of current brightness
        return ntl + (additive_factor * urban_field)

    elif enhancement_mode == "hybrid":
        # Context-aware enhancement with proper dynamic range preservation

        # DEBUG: Print parameters being used
        print(f"\n🔍 HYBRID ENHANCEMENT PARAMETERS:")
        print(f"  - expected_scale: {hybrid_expected_scale}")
        print(f"  - expected_sqrt_scale: {hybrid_expected_sqrt_scale}")
        print(f"  - application_factor: {hybrid_application_factor}")

        # Calculate enhancement factor based on urban density
        # This will be used to create a multiplicative enhancement similar to the old 1.3x approach
        enhancement_factor = (
            1.0
            + (
                hybrid_expected_scale * urban_field
                + hybrid_expected_sqrt_scale * np.sqrt(urban_field)
            )
            * hybrid_application_factor
        )

        # Apply enhancement multiplicatively
        enhanced = ntl * enhancement_factor

        # Find the maximum enhancement factor applied
        max_enhancement_factor = enhancement_factor.max()

        # Rescale everything to preserve dynamic range
        # This is like the old approach: multiply by up to 1.3, then divide everything by 1.3
        result = enhanced / max_enhancement_factor

        # DEBUG: Print enhancement statistics
        print(f"\n📊 HYBRID ENHANCEMENT STATS:")
        print(f"  - NTL range before: [{ntl.min():.1f}, {ntl.max():.1f}]")
        print(f"  - NTL range after: [{result.min():.1f}, {result.max():.1f}]")
        print(f"  - Max enhancement factor: {max_enhancement_factor:.3f}")
        print(
            f"  - Enhancement factors range: [{enhancement_factor.min():.3f}, {enhancement_factor.max():.3f}]"
        )
        print(
            f"  - Pixels enhanced: {np.sum(enhancement_factor > 1.001)} ({np.sum(enhancement_factor > 1.001) / enhancement_factor.size * 100:.1f}%)"
        )

        return result

    else:
        raise ValueError(
            f"Unknown enhancement_mode: {enhancement_mode}. "
            f"Must be one of: 'multiplicative', 'additive', 'hybrid'"
        )
