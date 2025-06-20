"""Named constants for QA bit values used throughout the test suite.

These values are from official Landsat and VIIRS documentation and should
be used consistently across all tests.
"""

# Landsat QA_PIXEL bit values (16-bit)
# Reference: Landsat 8-9 Collection 2 Level 2 Science Product Guide

# Clear conditions
QA_CLEAR_LAND = 21824  # Clear land: bits 0,6,7,8,9,10 set
QA_CLEAR_WATER = 21952  # Clear water: bits 0,6,7,8,9,10 + water bit set
QA_CLEAR_SNOW = 22080  # Clear snow/ice: bits 0,6,7,8,9,10 + snow bit set

# Cloud conditions
QA_LOW_CLOUD = 22280  # Low confidence cloud
QA_MEDIUM_CLOUD = 22344  # Medium confidence cloud
QA_MEDIUM_CLOUD_ALT = 54596  # Alternative medium confidence cloud (used in CloudMasker)
QA_HIGH_CLOUD = 54852  # High confidence cloud: bits 0,2,6,7,8,10,14,15 set

# Cloud shadow
QA_LOW_SHADOW = 23888  # Low confidence cloud shadow
QA_MEDIUM_SHADOW = 24144  # Medium confidence cloud shadow
QA_HIGH_SHADOW = 24400  # High confidence cloud shadow

# Special conditions
QA_DILATED_CLOUD = 55052  # Dilated cloud
QA_CIRRUS = 56856  # Cirrus detected

# Landsat QA_AEROSOL bit values (8-bit)
QA_AEROSOL_VALID = 64  # Valid aerosol retrieval: bit 6 set
QA_AEROSOL_LOW = 66  # Low aerosol: bits 1,6 set
QA_AEROSOL_MEDIUM = 68  # Medium aerosol: bits 2,6 set
QA_AEROSOL_HIGH = 72  # High aerosol: bits 3,6 set
QA_AEROSOL_CLIMATOLOGY = 80  # Climatology aerosol: bits 4,6 set

# Additional aerosol values used in masking
QA_AEROSOL_192 = 192  # Medium aerosol level
QA_AEROSOL_194 = 194  # Medium aerosol level
QA_AEROSOL_196 = 196  # High aerosol level
QA_AEROSOL_224 = 224  # High aerosol level (strict threshold)
QA_AEROSOL_228 = 228  # High aerosol level

# Landsat QA_RADSAT bit values (16-bit) - per band saturation flags
QA_RADSAT_NONE = 0  # No saturation
QA_RADSAT_BAND1 = 1  # Band 1 saturated
QA_RADSAT_BAND2 = 2  # Band 2 saturated
QA_RADSAT_BAND3 = 4  # Band 3 saturated (Green)
QA_RADSAT_BAND4 = 8  # Band 4 saturated (Red)
QA_RADSAT_BAND5 = 16  # Band 5 saturated (NIR)
QA_RADSAT_BAND6 = 32  # Band 6 saturated
QA_RADSAT_BAND7 = 64  # Band 7 saturated

# VIIRS Quality Flag (QF) values
VIIRS_QF_GOOD = 0  # Good quality
VIIRS_QF_POOR = 1  # Poor quality
VIIRS_QF_NO_RETRIEVAL = 2  # No retrieval
VIIRS_QF_OUTSIDE_RANGE = 255  # Outside expected range

# Common bit masks for QA_PIXEL interpretation
QA_PIXEL_FILL_BIT = 0  # Bit 0: Fill value
QA_PIXEL_DILATED_CLOUD_BIT = 1  # Bit 1: Dilated cloud
QA_PIXEL_CIRRUS_BIT = 2  # Bit 2: Cirrus
QA_PIXEL_CLOUD_BIT = 3  # Bit 3: Cloud
QA_PIXEL_CLOUD_SHADOW_BIT = 4  # Bit 4: Cloud shadow
QA_PIXEL_SNOW_BIT = 5  # Bit 5: Snow/ice
QA_PIXEL_CLEAR_BIT = 6  # Bit 6: Clear
QA_PIXEL_WATER_BIT = 7  # Bit 7: Water

# Cloud confidence bits (8-9)
QA_PIXEL_CLOUD_CONFIDENCE_LOW = 0b01 << 8  # Low confidence
QA_PIXEL_CLOUD_CONFIDENCE_MEDIUM = 0b10 << 8  # Medium confidence
QA_PIXEL_CLOUD_CONFIDENCE_HIGH = 0b11 << 8  # High confidence


# Helper functions for bit manipulation
def has_bit(value: int, bit: int) -> bool:
    """Check if a specific bit is set in a QA value.

    Args:
        value: QA pixel value
        bit: Bit position to check (0-based)

    Returns:
        True if bit is set, False otherwise
    """
    return bool(value & (1 << bit))


def set_bit(value: int, bit: int) -> int:
    """Set a specific bit in a QA value.

    Args:
        value: QA pixel value
        bit: Bit position to set (0-based)

    Returns:
        Value with bit set
    """
    return value | (1 << bit)


def clear_bit(value: int, bit: int) -> int:
    """Clear a specific bit in a QA value.

    Args:
        value: QA pixel value
        bit: Bit position to clear (0-based)

    Returns:
        Value with bit cleared
    """
    return value & ~(1 << bit)


def get_cloud_confidence(qa_pixel: int) -> int:
    """Extract cloud confidence level from QA_PIXEL value.

    Args:
        qa_pixel: QA_PIXEL value

    Returns:
        Cloud confidence level (0=not set, 1=low, 2=medium, 3=high)
    """
    return (qa_pixel >> 8) & 0b11


def is_clear_land(qa_pixel: int) -> bool:
    """Check if pixel represents clear land conditions.

    Args:
        qa_pixel: QA_PIXEL value

    Returns:
        True if clear land, False otherwise
    """
    return qa_pixel == QA_CLEAR_LAND or (
        has_bit(qa_pixel, QA_PIXEL_CLEAR_BIT)
        and not has_bit(qa_pixel, QA_PIXEL_WATER_BIT)
        and not has_bit(qa_pixel, QA_PIXEL_CLOUD_BIT)
        and not has_bit(qa_pixel, QA_PIXEL_CLOUD_SHADOW_BIT)
    )


def is_clear_water(qa_pixel: int) -> bool:
    """Check if pixel represents clear water conditions.

    Args:
        qa_pixel: QA_PIXEL value

    Returns:
        True if clear water, False otherwise
    """
    return qa_pixel == QA_CLEAR_WATER or (
        has_bit(qa_pixel, QA_PIXEL_CLEAR_BIT)
        and has_bit(qa_pixel, QA_PIXEL_WATER_BIT)
        and not has_bit(qa_pixel, QA_PIXEL_CLOUD_BIT)
        and not has_bit(qa_pixel, QA_PIXEL_CLOUD_SHADOW_BIT)
    )


def is_cloudy(qa_pixel: int) -> bool:
    """Check if pixel has any cloud coverage.

    Args:
        qa_pixel: QA_PIXEL value

    Returns:
        True if cloudy, False otherwise
    """
    return (
        has_bit(qa_pixel, QA_PIXEL_CLOUD_BIT)
        or has_bit(qa_pixel, QA_PIXEL_DILATED_CLOUD_BIT)
        or has_bit(qa_pixel, QA_PIXEL_CIRRUS_BIT)
        or qa_pixel
        in [QA_LOW_CLOUD, QA_MEDIUM_CLOUD, QA_HIGH_CLOUD, QA_DILATED_CLOUD, QA_CIRRUS]
    )
