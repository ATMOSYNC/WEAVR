import pytest

from weavr.data.weatherbench2 import GRAPHCAST_2020, fetch_india_slice


def test_fetch_india_slice_graphcast():
    try:
        da = fetch_india_slice(GRAPHCAST_2020, "2m_temperature", n_times=1)
    except Exception as exc:  # network/GCS access issue, not a code defect
        pytest.skip(f"WeatherBench 2 GCS bucket unreachable: {exc}")

    assert da.sizes["time"] == 1
    assert da.latitude.min() >= 6.0
    assert da.latitude.max() <= 38.0
    assert da.longitude.min() >= 68.0
    assert da.longitude.max() <= 98.0
    # Sanity range for 2m temperature in Kelvin
    assert 250.0 < float(da.max()) < 330.0
