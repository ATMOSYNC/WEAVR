import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_phase2_ensemble_baseline import (  # noqa: E402
    _exceedance_probability,
    _reconstruct_raw_source,
    _valid_time_to_imd_day,
)


class TestValidTimeToImdDay:
    def test_maps_valid_time_to_the_imd_day_it_falls_within(self):
        # init 2020-06-01T00:00 UTC + 24h lead -> valid at 2020-06-02T00:00 UTC,
        # inside the IMD day window 06-01T03:00 -> 06-02T03:00, so labelled
        # 2020-06-01 -- same convention scripts/run_tier0_baseline.py uses.
        result = _valid_time_to_imd_day(np.datetime64("2020-06-01T00:00:00"), lead_hours=24)
        assert pd.Timestamp(result) == pd.Timestamp("2020-06-01")

    def test_longer_lead_can_cross_into_the_next_calendar_day(self):
        result_24h = _valid_time_to_imd_day(np.datetime64("2020-06-01T00:00:00"), lead_hours=24)
        result_96h = _valid_time_to_imd_day(np.datetime64("2020-06-01T00:00:00"), lead_hours=96)
        assert pd.Timestamp(result_96h) - pd.Timestamp(result_24h) == pd.Timedelta(days=3)


class TestExceedanceProbability:
    def test_fraction_of_members_at_or_above_threshold(self):
        ensemble = xr.DataArray(
            np.array([0.0, 5.0, 10.0, 15.0]).reshape(1, 4), dims=["sample", "member"]
        )

        probability = _exceedance_probability(ensemble, threshold=5.0)

        # 3 of 4 members (5.0, 10.0, 15.0) are >= 5.0
        assert probability.item() == 0.75

    def test_ignores_nan_members(self):
        ensemble = xr.DataArray(
            np.array([10.0, 10.0, np.nan]).reshape(1, 3), dims=["sample", "member"]
        )

        probability = _exceedance_probability(ensemble, threshold=5.0)

        # both real members exceed -- the NaN (missing, short-window) member
        # must not silently count as a non-exceedance and drag this to 2/3.
        assert probability.item() == 1.0


def _make_dense(nominal_times, lead_hours, offsets, values) -> xr.Dataset:
    dims = ("nominal_time", "lead_hours", "member_offset_hours", "latitude", "longitude")
    return xr.Dataset(
        {"var": (dims, values)},
        coords={
            "nominal_time": nominal_times,
            "lead_hours": lead_hours,
            "member_offset_hours": offsets,
            "latitude": np.array([10.0]),
            "longitude": np.array([70.0]),
        },
    )


class TestReconstructRawSource:
    def test_round_trips_a_single_known_combo(self):
        nominal_times = np.array(["2020-06-01T00:00:00"], dtype="datetime64[ns]")
        lead_hours = [24]
        offsets = [0]
        values = np.array([[[[[42.0]]]]])  # (nominal, lead, offset, lat, lon)
        dense = _make_dense(nominal_times, lead_hours, offsets, values)

        raw = _reconstruct_raw_source(dense, "var")

        assert raw.sizes["time"] == 1
        assert raw.sizes["prediction_timedelta"] == 1
        assert raw["prediction_timedelta"].values[0] == 24
        assert raw["time"].values[0] == nominal_times[0]
        assert raw["var"].isel(time=0, prediction_timedelta=0).item() == 42.0

    def test_drops_cells_with_no_valid_lead_rather_than_inventing_a_source_time(self):
        # offset=+48 at lead=24 needs source lead -24h, which doesn't exist
        # (_valid_combos excludes it) -- the reconstructed source must not
        # contain a fabricated entry for it.
        nominal_times = np.array(["2020-06-01T00:00:00"], dtype="datetime64[ns]")
        lead_hours = [24]
        offsets = [0, 48]
        values = np.full((1, 1, 2, 1, 1), np.nan)
        values[0, 0, 0] = 1.0  # only offset=0 is a real, fetched value
        dense = _make_dense(nominal_times, lead_hours, offsets, values)

        raw = _reconstruct_raw_source(dense, "var")

        # Only the one valid (time, lead) combo (offset=0 -> lead=24) exists.
        assert raw.sizes["time"] == 1
        assert raw.sizes["prediction_timedelta"] == 1
        assert raw["prediction_timedelta"].values[0] == 24

    def test_reconstructed_source_feeds_build_lagged_ensemble_correctly(self):
        # Full integration check: a dense store built the normal way, then
        # reconstructed and run back through build_lagged_ensemble, must
        # reproduce the dense store's own already-assembled slice exactly.
        from weavr.ensemble import build_lagged_ensemble

        nominal_times = np.array(
            ["2020-06-01T00:00:00", "2020-06-08T00:00:00"], dtype="datetime64[ns]"
        )
        lead_hours = [72]  # a lead with a full 9-member window
        offsets = [-48, -36, -24, -12, 0, 12, 24, 36, 48]
        rng = np.random.default_rng(0)
        values = rng.normal(size=(2, 1, 9, 1, 1))
        dense = _make_dense(nominal_times, lead_hours, offsets, values)

        raw = _reconstruct_raw_source(dense, "var")
        ensemble = build_lagged_ensemble(
            raw, "var", nominal_times[0], 72, n_lags=4, lag_spacing_hours=12
        )

        direct = dense["var"].sel(nominal_time=nominal_times[0], lead_hours=72).rename(
            member_offset_hours="member"
        )
        np.testing.assert_allclose(
            ensemble.sortby("member").values, direct.sortby("member").values
        )
