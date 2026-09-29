import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_baseline_store import (  # noqa: E402
    _clear_encoding,
    _lat_slice_for,
    _weekly_init_times,
)


def _dataset_with_lat(values):
    da = xr.DataArray(values, coords={"latitude": values}, dims=["latitude"])
    return xr.Dataset({"x": da})


class TestLatSliceFor:
    def test_ascending_coordinate_uses_low_to_high(self):
        ds = _dataset_with_lat(np.arange(-90.0, 91.0, 1.0))
        result = _lat_slice_for(ds, "latitude", 6.0, 38.0)
        assert result == slice(6.0, 38.0)
        assert ds.sel(latitude=result).sizes["latitude"] > 0

    def test_descending_coordinate_uses_high_to_low(self):
        ds = _dataset_with_lat(np.arange(90.0, -91.0, -1.0))
        result = _lat_slice_for(ds, "latitude", 6.0, 38.0)
        assert result == slice(38.0, 6.0)
        # The real bug this guards against: a naive slice(lo, hi) on a
        # descending coordinate silently selects zero points.
        assert ds.sel(latitude=result).sizes["latitude"] > 0
        assert ds.sel(latitude=slice(6.0, 38.0)).sizes["latitude"] == 0


class TestWeeklyInitTimes:
    def test_samples_at_requested_cadence_from_12_hourly_source(self):
        times = pd.date_range("2020-06-01", "2020-06-14T12:00", freq="12h")
        x = xr.DataArray(np.zeros(len(times)), coords={"time": times}, dims=["time"])
        ds = xr.Dataset({"x": x})

        sampled = _weekly_init_times(ds, "2020-06-01", "2020-06-14", cadence_days=7)

        # 14 days at 12h cadence = 28 entries; a 7-day cadence should pick
        # roughly every 14th entry, landing on real timestamps from the source.
        assert len(sampled) >= 2
        assert set(sampled).issubset(set(times.values))
        assert (pd.to_datetime(sampled).hour == 0).all()

    def test_daily_cadence_picks_only_00_utc(self):
        times = pd.date_range("2020-06-01", "2020-06-05T12:00", freq="12h")
        x = xr.DataArray(np.zeros(len(times)), coords={"time": times}, dims=["time"])
        ds = xr.Dataset({"x": x})

        sampled = _weekly_init_times(ds, "2020-06-01", "2020-06-05", cadence_days=1)
        sampled_dt = pd.to_datetime(sampled)
        assert len(sampled) == 5
        assert (sampled_dt.hour == 0).all()
        assert list(sampled_dt.strftime("%Y-%m-%d")) == [
            "2020-06-01",
            "2020-06-02",
            "2020-06-03",
            "2020-06-04",
            "2020-06-05",
        ]

    def test_empty_source_returns_empty(self):
        empty_times = np.array([], dtype="datetime64[ns]")
        x = xr.DataArray(np.zeros(0), coords={"time": empty_times}, dims=["time"])
        ds = xr.Dataset({"x": x})

        sampled = _weekly_init_times(ds, "2020-06-01", "2020-06-14", cadence_days=7)
        assert len(sampled) == 0


class TestClearEncoding:
    def test_strips_encoding_from_all_variables_and_coords(self):
        da = xr.DataArray(np.zeros(3), coords={"time": [1, 2, 3]}, dims=["time"])
        da.encoding = {"chunks": (1,), "compressors": "fake"}
        da["time"].encoding = {"chunks": (1,)}
        ds = xr.Dataset({"v": da})

        out = _clear_encoding(ds)

        assert out["v"].encoding == {}
        assert out["time"].encoding == {}
