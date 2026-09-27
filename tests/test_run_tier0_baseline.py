import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_tier0_baseline import PRECIP_M_TO_MM, _align_to_imd_day  # noqa: E402


class TestAlignToImdDay:
    def test_maps_valid_time_to_the_imd_day_it_falls_within(self):
        # init 2020-06-01T00:00 UTC + 24h lead -> valid at 2020-06-02T00:00 UTC,
        # which falls inside the IMD day window 06-01T03:00 -> 06-02T03:00,
        # so it must be labelled 2020-06-01, not 2020-06-02.
        times = pd.date_range("2020-06-01", periods=3, freq="7D")
        da = xr.DataArray(np.arange(3, dtype=float), coords={"time": times}, dims=["time"])

        result = _align_to_imd_day(da, lead_hours=24)

        assert result.dims == ("sample",)
        expected = pd.DatetimeIndex(times).normalize()
        assert list(pd.DatetimeIndex(result["sample"].values)) == list(expected)

    def test_longer_lead_can_cross_into_the_next_calendar_day(self):
        # init at 00:00 UTC + 96h (4 days) lands at 00:00 UTC 4 days later,
        # still within the previous day's 03-03 UTC window -> same-day label
        # as the 24h case, not shifted by the extra days' worth of hours.
        times = pd.date_range("2020-06-01", periods=1, freq="D")
        da = xr.DataArray([1.0], coords={"time": times}, dims=["time"])

        result_24h = _align_to_imd_day(da, lead_hours=24)
        result_96h = _align_to_imd_day(da, lead_hours=96)

        assert pd.Timestamp(result_96h["sample"].values[0]) - pd.Timestamp(
            result_24h["sample"].values[0]
        ) == pd.Timedelta(days=3)


class TestPrecipUnitConversion:
    def test_meters_to_millimeters_constant(self):
        # WeatherBench 2's total_precipitation_24hr is in meters; IMD's rain
        # is in millimeters -- caught by comparing real magnitudes
        # (forecast ~0.006, obs ~5.7) before trusting any score, not assumed.
        assert PRECIP_M_TO_MM == 1000.0
