import numpy as np
import pandas as pd
import pytest
import xarray as xr

from weavr.grid import (
    COMMON_LAT,
    COMMON_LON,
    GRID_RESOLUTION_DEG,
    SourceTooCoarseError,
    imd_day_containing,
    imd_day_for,
    regrid_to_common,
    resample_to_imd_day,
)


def _synthetic_grid(lat_res: float, lon_res: float) -> xr.Dataset:
    lat = np.arange(6.0, 39.0, lat_res)
    lon = np.arange(66.0, 101.0, lon_res)
    data = np.random.default_rng(0).random((len(lat), len(lon)))
    da = xr.DataArray(
        data, coords={"latitude": lat, "longitude": lon}, dims=["latitude", "longitude"]
    )
    return xr.Dataset({"x": da})


class TestRegridToCommon:
    def test_finer_source_regrids_to_common_shape(self):
        ds = _synthetic_grid(lat_res=0.1, lon_res=0.1)
        out = regrid_to_common(ds)

        assert out["x"].shape == (len(COMMON_LAT), len(COMMON_LON))
        assert out.latitude.min().item() == pytest.approx(COMMON_LAT.min())
        assert out.latitude.max().item() == pytest.approx(COMMON_LAT.max())

    def test_matching_resolution_source_passes_through(self):
        ds = _synthetic_grid(lat_res=GRID_RESOLUTION_DEG, lon_res=GRID_RESOLUTION_DEG)
        out = regrid_to_common(ds)
        assert out["x"].shape == (len(COMMON_LAT), len(COMMON_LON))

    def test_coarser_source_raises_instead_of_upsampling(self):
        ds = _synthetic_grid(lat_res=0.5, lon_res=0.5)
        with pytest.raises(SourceTooCoarseError):
            regrid_to_common(ds)

    def test_missing_coords_raises_value_error(self):
        da = xr.DataArray(np.zeros((3, 3)), dims=["y", "x"])
        with pytest.raises(ValueError):
            regrid_to_common(xr.Dataset({"v": da}))


class TestResampleToImdDay:
    def test_full_days_sum_to_24_hours(self):
        times = pd.date_range("2024-06-14T03:00", "2024-06-16T02:00", freq="1h")
        precip = xr.DataArray(np.ones(len(times)), coords={"time": times}, dims=["time"])
        ds = xr.Dataset({"precip": precip})

        out = resample_to_imd_day(ds)

        # Two full IMD days: [06-14T03:00, 06-15T03:00) and [06-15T03:00, 06-16T03:00)
        assert list(out["precip"].values) == [24.0, 24.0]
        # ... labelled by the date each window ends on.
        assert list(pd.DatetimeIndex(out["time"].values)) == [
            pd.Timestamp("2024-06-15"),
            pd.Timestamp("2024-06-16"),
        ]

    def test_day_boundaries_are_labelled_by_end_date_not_midnight_utc(self):
        # A window starting exactly at 03:00 UTC on 06-14 ends 03:00 UTC on 06-15,
        # so IMD labels it 06-15 -- not the midnight-UTC day it starts in.
        times = pd.date_range("2024-06-14T03:00", periods=24, freq="1h")
        precip = xr.DataArray(np.ones(24), coords={"time": times}, dims=["time"])
        ds = xr.Dataset({"precip": precip})

        out = resample_to_imd_day(ds)

        assert len(out["time"]) == 1
        assert pd.Timestamp(out["time"].values[0]) == pd.Timestamp("2024-06-15")
        assert out["precip"].values[0] == 24.0

    def test_partial_window_at_data_edges_sums_only_available_hours(self):
        # Data starts mid-window: the window [06-13T03:00, 06-14T03:00) is IMD
        # day "2024-06-14"; only the last 3 hours of it have data.
        times = pd.date_range("2024-06-14T00:00", "2024-06-14T02:00", freq="1h")
        ds = xr.Dataset({"precip": xr.DataArray(np.ones(3), coords={"time": times}, dims=["time"])})

        out = resample_to_imd_day(ds)

        assert pd.Timestamp(out["time"].values[0]) == pd.Timestamp("2024-06-14")
        assert out["precip"].values[0] == 3.0

    def test_missing_time_dim_raises_value_error(self):
        ds = xr.Dataset({"v": xr.DataArray(np.zeros(3), dims=["x"])})
        with pytest.raises(ValueError):
            resample_to_imd_day(ds)


class TestImdDayContaining:
    def test_window_is_labelled_by_its_end_date(self):
        # IMD's page: "0830 IST of 21-10 to 0830 IST of 22-10" is the value for 22-10.
        assert imd_day_containing(pd.Timestamp("2026-10-21T03:00")) == pd.Timestamp("2026-10-22")
        assert imd_day_containing(pd.Timestamp("2026-10-22T02:59")) == pd.Timestamp("2026-10-22")

    def test_windows_are_half_open_at_03_utc(self):
        assert imd_day_containing(pd.Timestamp("2024-06-15T02:59:59")) == pd.Timestamp("2024-06-15")
        assert imd_day_containing(pd.Timestamp("2024-06-15T03:00")) == pd.Timestamp("2024-06-16")

    def test_midnight_utc_belongs_to_the_day_ending_that_morning(self):
        assert imd_day_containing(pd.Timestamp("2024-06-15T00:00")) == pd.Timestamp("2024-06-15")

    def test_accepts_a_datetime_index(self):
        out = imd_day_containing(pd.DatetimeIndex(["2024-06-15T00:00", "2024-06-15T12:00"]))
        assert list(out) == [pd.Timestamp("2024-06-15"), pd.Timestamp("2024-06-16")]


class TestImdDayFor:
    def test_24h_forecast_from_00utc_validates_on_the_next_imd_day(self):
        # Window 06-01T00 -> 06-02T00 overlaps IMD day 06-02 (06-01T03 -> 06-02T03)
        # for 21 h, and IMD day 06-01 (05-31T03 -> 06-01T03) for only 3 h.
        assert imd_day_for(pd.Timestamp("2020-06-01"), 24) == pd.Timestamp("2020-06-02")

    def test_each_extra_lead_day_moves_the_label_one_day(self):
        init = pd.Timestamp("2020-06-01")
        assert [imd_day_for(init, h) for h in (24, 48, 72, 96, 120)] == [
            pd.Timestamp(f"2020-06-0{d}") for d in (2, 3, 4, 5, 6)
        ]

    def test_a_window_ending_exactly_at_03_utc_is_that_days_window(self):
        # Init 03 UTC + 24 h: window ends 03:00 on 06-02, which is IMD day 06-02 exactly.
        assert imd_day_for(pd.Timestamp("2020-06-01T03:00"), 24) == pd.Timestamp("2020-06-02")

    def test_accepts_a_datetime_index(self):
        out = imd_day_for(pd.DatetimeIndex(["2020-06-01", "2020-06-02"]), 24)
        assert list(out) == [pd.Timestamp("2020-06-02"), pd.Timestamp("2020-06-03")]
