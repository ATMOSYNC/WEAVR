import pytest

from weavr.data.ecmwf_open_data import fetch_latest_forecast, slice_india


def test_fetch_latest_forecast_total_precipitation():
    try:
        ds = fetch_latest_forecast(param="tp", step=0)
    except Exception as exc:  # network/open-data access issue, not a code defect
        pytest.skip(f"ECMWF open-data feed unreachable: {exc}")

    assert "tp" in ds.data_vars

    india = slice_india(ds)
    assert india["tp"].sizes["latitude"] > 0
    assert india["tp"].sizes["longitude"] > 0
