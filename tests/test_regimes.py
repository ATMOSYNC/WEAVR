import numpy as np
import pandas as pd
import pytest
import xarray as xr

from weavr.regimes import (
    ACTIVE_BREAK_ANOMALY_THRESHOLD,
    MIN_CONSECUTIVE_DAYS,
    MONSOON_CORE_ZONE_LAT,
    MONSOON_CORE_ZONE_LON,
    EmptyCoreZoneError,
    _apply_min_run_length,
    classify_monsoon_active_break,
    core_zone_daily_mean,
)


def _daily_rain(values: np.ndarray, start: str = "2020-06-01") -> xr.DataArray:
    n_days, n_lat, n_lon = values.shape
    times = pd.date_range(start, periods=n_days, freq="D")
    lats = np.linspace(6.5, 38.5, n_lat)
    lons = np.linspace(66.5, 100.0, n_lon)
    return xr.DataArray(
        values,
        coords={"time": times, "latitude": lats, "longitude": lons},
        dims=["time", "latitude", "longitude"],
    )


class TestCoreZoneDailyMean:
    def test_averages_only_the_core_zone_gridpoints(self):
        lats = np.array([10.0, 20.0, 30.0])  # 20.0 is inside 18-28N; 10/30 are not
        lons = np.array([70.0, 90.0])  # both inside 65-88E boundary except 90
        times = pd.date_range("2020-06-01", periods=1)
        values = np.array([[[1.0, 100.0], [5.0, 100.0], [1.0, 100.0]]])
        da = xr.DataArray(
            values, coords={"time": times, "latitude": lats, "longitude": lons},
            dims=["time", "latitude", "longitude"],
        )

        result = core_zone_daily_mean(da)

        # Only latitude=20 (inside 18-28) and longitude=70 (inside 65-88)
        # should contribute -- core_zone_daily_mean's own slice keeps every
        # matching gridpoint, here just the single (20, 70) cell = 5.0.
        assert result.values[0] == pytest.approx(5.0)

    def test_core_zone_bounds_match_rajeevan_gadgil_bhate(self):
        assert MONSOON_CORE_ZONE_LAT == (18.0, 28.0)
        assert MONSOON_CORE_ZONE_LON == (65.0, 88.0)

    def test_raises_when_core_zone_selects_zero_gridpoints(self):
        # lats [6.5, 38.5] straddle but never land inside 18-28N.
        da = _daily_rain(np.zeros((1, 2, 2)))

        with pytest.raises(EmptyCoreZoneError):
            core_zone_daily_mean(da)


class TestApplyMinRunLength:
    def test_short_run_reverts_to_neutral(self):
        labels = np.array(["active", "active", "", "break", "break", "break"], dtype=object)

        result = _apply_min_run_length(labels, min_days=3)

        np.testing.assert_array_equal(
            result, np.array(["", "", "", "break", "break", "break"], dtype=object)
        )

    def test_run_at_exact_threshold_is_kept(self):
        labels = np.array(["active", "active", "active"], dtype=object)

        result = _apply_min_run_length(labels, min_days=3)

        np.testing.assert_array_equal(result, labels)

    def test_neutral_runs_are_never_relabelled(self):
        labels = np.array(["", ""], dtype=object)

        result = _apply_min_run_length(labels, min_days=3)

        np.testing.assert_array_equal(result, labels)


class TestClassifyMonsoonActiveBreak:
    def test_sustained_high_anomaly_is_labelled_active(self):
        n_days, n_lat, n_lon = 10, 3, 3
        # Climatology: constant, low-variance rain everywhere.
        clim_values = np.full((30, n_lat, n_lon), 5.0)
        climatology = _daily_rain(clim_values, start="2010-06-01")

        # Real season: a sustained high-rain spell (days 3-7, 5 days) well
        # above the climatological mean, surrounded by near-climatological days.
        real_values = np.full((n_days, n_lat, n_lon), 5.0)
        real_values[3:8] = 50.0
        rain = _daily_rain(real_values)

        result = classify_monsoon_active_break(rain, climatology)

        assert list(result.values[3:8]) == ["active"] * 5
        assert result.values[0] == ""
        assert result.values[9] == ""

    def test_sustained_low_anomaly_is_labelled_break(self):
        n_days, n_lat, n_lon = 10, 3, 3
        clim_values = np.full((30, n_lat, n_lon), 5.0)
        # Give the climatology some spread so std is nonzero.
        rng = np.random.default_rng(0)
        clim_values = clim_values + rng.normal(scale=1.0, size=clim_values.shape)
        climatology = _daily_rain(clim_values, start="2010-06-01")

        real_values = np.full((n_days, n_lat, n_lon), 5.0)
        real_values[2:6] = 0.0
        rain = _daily_rain(real_values)

        result = classify_monsoon_active_break(rain, climatology)

        assert all(label == "break" for label in result.values[2:6])

    def test_a_single_day_spike_does_not_count_as_active(self):
        n_days, n_lat, n_lon = 10, 3, 3
        clim_values = np.full((30, n_lat, n_lon), 5.0)
        climatology = _daily_rain(clim_values, start="2010-06-01")

        real_values = np.full((n_days, n_lat, n_lon), 5.0)
        real_values[5] = 500.0  # one extreme day, below MIN_CONSECUTIVE_DAYS
        rain = _daily_rain(real_values)

        result = classify_monsoon_active_break(rain, climatology)

        assert result.values[5] == ""

    def test_output_shares_the_real_season_time_coordinate(self):
        # n_lat/n_lon=3 (not 2): _daily_rain's linspace grid must include at
        # least one point inside the monsoon core zone (18-28N, 65-88E),
        # or core_zone_daily_mean's own spatial selection is empty and its
        # time series is all-NaN -- a real degenerate case, not something
        # this test means to exercise.
        n_days, n_lat, n_lon = 5, 3, 3
        rng = np.random.default_rng(1)
        clim_values = 5.0 + rng.normal(scale=1.0, size=(10, n_lat, n_lon))
        climatology = _daily_rain(clim_values, start="2010-06-01")
        rain = _daily_rain(np.full((n_days, n_lat, n_lon), 5.0))

        result = classify_monsoon_active_break(rain, climatology)

        np.testing.assert_array_equal(result["time"].values, rain["time"].values)


class TestConstants:
    def test_min_consecutive_days_and_threshold_match_the_cited_paper(self):
        assert MIN_CONSECUTIVE_DAYS == 3
        assert ACTIVE_BREAK_ANOMALY_THRESHOLD == 1.0
