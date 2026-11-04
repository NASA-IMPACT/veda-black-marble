"""Functions for downloading VIIRS nighttime lights data from NASA LAADS."""

import logging
import warnings
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import earthaccess
import rasterio
from rasterio.errors import NotGeoreferencedWarning


logger = logging.getLogger(__name__)


ViirsProductName = Literal["VJ146A2", "VNP46A2"]


@dataclass(frozen=True)
class ViirsProduct:
    """A Black Marble daily gap-filled nighttime lights product."""

    short_name: str
    version: str
    platform: str
    start_date: datetime
    end_date: datetime | None = None

    @property
    def sensor(self) -> str:
        return f"VIIRS ({self.platform})"


# VJ146A2 (NOAA-20 / JPSS-1) and VNP46A2 (Suomi NPP) are structural twins: same CMR
# version, same LAADS provider, same h/v 2400x2400 tiling, same HDF-EOS5 grid path and
# Gap_Filled_DNB_BRDF-Corrected_NTL layer, same -999.9 fill. Only the platform and the
# temporal extent differ.
VIIRS_PRODUCTS: dict[str, ViirsProduct] = {
    "VJ146A2": ViirsProduct(
        short_name="VJ146A2",
        version="2",
        platform="NOAA-20",
        start_date=datetime(2018, 1, 19),
    ),
    # Suomi NPP product delivery ceases 2026-11-01. The archive remains available, so
    # historical dates still work, but there is no forward data stream after that date.
    "VNP46A2": ViirsProduct(
        short_name="VNP46A2",
        version="2",
        platform="Suomi NPP",
        start_date=datetime(2012, 1, 19),
        end_date=datetime(2026, 11, 1),
    ),
}

DEFAULT_VIIRS_PRODUCT: ViirsProductName = "VJ146A2"

# Backwards-compatible aliases for the default product.
BM_SHORT_NAME = VIIRS_PRODUCTS[DEFAULT_VIIRS_PRODUCT].short_name
BM_VERSION = VIIRS_PRODUCTS[DEFAULT_VIIRS_PRODUCT].version

NTL_DATASET_PATH = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data_Fields/Gap_Filled_DNB_BRDF-Corrected_NTL"


def convert_to_tiff(
    input_h5: str | Path, output_path: str | Path, dataset_path: str = NTL_DATASET_PATH
) -> Path:

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        subdatasets = rasterio.open(input_h5).subdatasets
    dnb_sds = [sds for sds in subdatasets if dataset_path in sds][0]

    with rasterio.open(dnb_sds, "r") as src:
        profile = src.profile
        data = src.read(1)

    dst_profile = deepcopy(profile)
    dst_profile.update(
        driver="GTiff", predictor=3, compress="deflate", blockxsize=256, blockysize=256
    )
    with rasterio.open(output_path, mode="w", **dst_profile) as dst:
        dst.write(data, 1)

    return Path(output_path)


def get_viirs_product(name: str) -> ViirsProduct:
    """Look up a supported VIIRS product by short name (case-insensitive)."""
    try:
        return VIIRS_PRODUCTS[name.strip().upper()]
    except KeyError:
        supported = ", ".join(VIIRS_PRODUCTS)
        raise ValueError(f"Unsupported VIIRS product {name!r}. Use one of: {supported}") from None


def validate_viirs_date(dt: datetime, product: ViirsProduct) -> None:
    """Fail fast on dates the product cannot serve.

    A date before the mission start returns zero granules from CMR, which otherwise
    surfaces much later as an obscure "no VIIRS files" failure.
    """
    if dt.date() < product.start_date.date():
        raise ValueError(
            f"{product.short_name} ({product.platform}) has no data before "
            f"{product.start_date:%Y-%m-%d}; requested {dt:%Y-%m-%d}."
        )
    if product.end_date is not None and dt.date() >= product.end_date.date():
        logger.warning(
            "%s (%s) product delivery ceased on %s; %s may have no data. Consider %s instead.",
            product.short_name,
            product.platform,
            f"{product.end_date:%Y-%m-%d}",
            f"{dt:%Y-%m-%d}",
            DEFAULT_VIIRS_PRODUCT,
        )


def download_viirs(
    dt: datetime,
    bbox: tuple[float, float, float, float],
    output_dir: str | Path,
    product: str = DEFAULT_VIIRS_PRODUCT,
) -> dict[str, list[Path]]:
    """Search and download VIIRS Black Marble (VJ146A2 / VNP46A2) files.

    Args:
        dt: Target date.
        bbox: (min_lon, min_lat, max_lon, max_lat) in WGS84.
        output_dir: Directory to download granules into.
        product: VIIRS product short name. Defaults to VJ146A2 (NOAA-20).
    """
    viirs_product = get_viirs_product(product)
    validate_viirs_date(dt, viirs_product)

    logger.info("Logging in to Earthdata...")
    earthaccess.login()

    logger.info("Searching for %s data...", viirs_product.short_name)
    results = earthaccess.search_data(
        short_name=viirs_product.short_name,
        version=viirs_product.version,
        temporal=(dt.strftime("%Y-%m-%d"), dt.strftime("%Y-%m-%dT23:59:59"), True),
        bounding_box=bbox,
    )

    logger.info("Downloading VIIRS data...")
    filelist = earthaccess.download(results, local_path=output_dir, show_progress=False)

    tiff_filelist = [convert_to_tiff(f, f.with_suffix(".tif")) for f in filelist]

    return {"gap_filled_ntl": tiff_filelist}
