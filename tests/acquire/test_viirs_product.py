"""VIIRS product selection: VJ146A2 (NOAA-20) by default, VNP46A2 (Suomi NPP) on request."""

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from blackmarble.acquire import viirs
from blackmarble.config import PipelineConfig


BBOX = (-122.55, 37.69, -122.32, 37.81)


@pytest.fixture
def cmr_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Stub earthaccess and record every CMR search."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(viirs.earthaccess, "login", lambda *a, **k: None)
    monkeypatch.setattr(
        viirs.earthaccess, "search_data", lambda **kwargs: calls.append(kwargs) or []
    )
    monkeypatch.setattr(viirs.earthaccess, "download", lambda *a, **k: [])
    return calls


def test_default_product_is_noaa20():
    assert viirs.DEFAULT_VIIRS_PRODUCT == "VJ146A2"
    config = PipelineConfig(bbox=BBOX, date=datetime(2023, 6, 15), output_path="x.tif")
    assert config.acquisition.viirs_product == "VJ146A2"


def test_download_searches_noaa20_by_default(cmr_calls: list[dict[str, Any]], tmp_path: Path):
    viirs.download_viirs(datetime(2023, 6, 15), BBOX, tmp_path)
    assert cmr_calls[0]["short_name"] == "VJ146A2"
    assert cmr_calls[0]["version"] == "2"


def test_download_can_search_suomi_npp(cmr_calls: list[dict[str, Any]], tmp_path: Path):
    viirs.download_viirs(datetime(2023, 6, 15), BBOX, tmp_path, product="vnp46a2")
    assert cmr_calls[0]["short_name"] == "VNP46A2"
    assert cmr_calls[0]["version"] == "2"


def test_date_before_mission_start_fails_before_login(
    cmr_calls: list[dict[str, Any]], tmp_path: Path
):
    with pytest.raises(ValueError, match="no data before 2018-01-19"):
        viirs.download_viirs(datetime(2017, 6, 15), BBOX, tmp_path, product="VJ146A2")
    assert cmr_calls == []
    # Suomi NPP covers that date.
    viirs.download_viirs(datetime(2017, 6, 15), BBOX, tmp_path, product="VNP46A2")
    assert cmr_calls[0]["short_name"] == "VNP46A2"


def test_suomi_npp_after_sunset_warns(
    cmr_calls: list[dict[str, Any]], tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    viirs.download_viirs(datetime(2026, 12, 1), BBOX, tmp_path, product="VNP46A2")
    assert "delivery ceased" in caplog.text


def test_unknown_product_rejected():
    with pytest.raises(ValueError, match="Unsupported VIIRS product"):
        viirs.get_viirs_product("VNP46A1")


def test_products_share_layer_and_version():
    """The two products are structural twins; only platform and dates differ."""
    noaa20, snpp = viirs.VIIRS_PRODUCTS["VJ146A2"], viirs.VIIRS_PRODUCTS["VNP46A2"]
    assert noaa20.version == snpp.version
    assert noaa20.sensor == "VIIRS (NOAA-20)"
    assert snpp.sensor == "VIIRS (Suomi NPP)"
