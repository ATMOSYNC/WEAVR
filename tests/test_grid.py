import numpy as np
import pandas as pd
import pytest
import xarray as xr

from weavr.grid import (
    COMMON_LAT,
    COMMON_LON,
    GRID_RESOLUTION_DEG,
    SourceTooCoarseError,
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

    def test_day_boundaries_are_labelled_by_start_date_not_midnight_utc(self):
        # A day boundary exactly at 03:00 UTC must fall in the day that starts there,
        # not the previous midnight-UTC day.
        times = pd.date_range("2024-06-14T03:00", periods=24, freq="1h")
        precip = xr.DataArray(np.ones(24), coords={"time": times}, dims=["time"])
        ds = xr.Dataset({"precip": precip})

        out = resample_to_imd_day(ds)

        assert len(out["time"]) == 1
        assert pd.Timestamp(out["time"].values[0]) == pd.Timestamp("2024-06-14")
        assert out["precip"].values[0] == 24.0

    def test_partial_window_at_data_edges_sums_only_available_hours(self):
        # Data starts mid-window: IMD day "2024-06-13" window is
        # [06-13T03:00, 06-14T03:00); only the last 3 hours of it have data.
        times = pd.date_range("2024-06-14T00:00", "2024-06-14T02:00", freq="1h")
        ds = xr.Dataset({"precip": xr.DataArray(np.ones(3), coords={"time": times}, dims=["time"])})

        out = resample_to_imd_day(ds)

        assert pd.Timestamp(out["time"].values[0]) == pd.Timestamp("2024-06-13")
        assert out["precip"].values[0] == 3.0

    def test_missing_time_dim_raises_value_error(self):
        ds = xr.Dataset({"v": xr.DataArray(np.zeros(3), dims=["x"])})
        with pytest.raises(ValueError):
            resample_to_imd_day(ds)
