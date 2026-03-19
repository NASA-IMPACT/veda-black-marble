"""Type stubs for rasterio.enums."""

from enum import Enum

class Resampling(Enum):
    nearest = 0
    bilinear = 1
    cubic = 2
    cubic_spline = 3
    lanczos = 4
    average = 5
    mode = 6
    gauss = 7
    max = 8
    min = 9
    med = 10
    q1 = 11
    q3 = 12

class ColorInterp(Enum):
    undefined = 0
    gray = 1
    palette = 2
    red = 3
    green = 4
    blue = 5
    alpha = 6
    hue = 7
    saturation = 8
    lightness = 9
    cyan = 10
    magenta = 11
    yellow = 12
    black = 13

class Compression(Enum):
    none = "NONE"
    jpeg = "JPEG"
    lzw = "LZW"
    packbits = "PACKBITS"
    deflate = "DEFLATE"
    ccittrle = "CCITTRLE"
    ccittfax3 = "CCITTFAX3"
    ccittfax4 = "CCITTFAX4"
    lzma = "LZMA"
    zstd = "ZSTD"
    lerc = "LERC"
    lerc_deflate = "LERC_DEFLATE"
    lerc_zstd = "LERC_ZSTD"
    webp = "WEBP"

class Interleaving(Enum):
    pixel = "PIXEL"
    line = "LINE"
    band = "BAND"

class MaskFlags(Enum):
    all_valid = 1
    per_dataset = 2
    alpha = 4
    nodata = 8

__all__ = ["Resampling", "ColorInterp", "Compression", "Interleaving", "MaskFlags"]
