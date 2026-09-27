import numpy as np
import pandas as pd
import pytest
import xarray as xr

from weavr import verify as V
from weavr.ensemble import NoLaggedMembersError, _lag_offsets, build_lagged_ensemble


def _synthetic_source(n_days: int = 20) -> xr.Dataset:
    """A WeatherBench-2-shaped synthetic source: 12-hourly init times, integer
    prediction_timedelta hours from 6..240 in 6h steps -- the same shape
    confirmed live for GraphCast/Pangu (docs/baseline-store.md).
    """
    times = pd.date_range("2020-06-01", periods=n_days * 2, freq="12h")
    leads = np.arange(6, 241, 6)
    lat = np.array([10.0, 10.25])
    lon = np.array([70.0, 70.25])

    rng = np.random.default_rng(0)
    data = rng.normal(
        loc=280.0, scale=1.0, size=(len(times), len(leads), len(lat), len(lon))
    )
    return xr.Dataset(
        {"2m_temperature": (("time", "prediction_timedelta", "latitude", "longitude"), data)},
        coords={
            "time": times,
            "prediction_timedelta": leads,
            "latitude": lat,
            "longitude": lon,
        },
    )


class TestLagOffsets:
    def test_default_nine_offsets_including_zero(self):
        assert _lag_offsets(4, 12) == [-48, -36, -24, -12, 0, 12, 24, 36, 48]


class TestBuildLaggedEnsembleFullWindow:
    def test_full_nine_member_window_at_a_long_lead(self):
        ds = _synthetic_source()
        nominal_init = pd.Timestamp("2020-06-05T00:00:00")

        ensemble = build_lagged_ensemble(ds, "2m_temperature", nominal_init, lead_hours=120)

        assert ensemble.sizes["member"] == 9
        assert ensemble.attrs["n_valid_members"] == 9
        assert not bool(ensemble.isnull().any())

    def test_every_member_targets_the_same_valid_time(self):
        ds = _synthetic_source()
        nominal_init = pd.Timestamp("2020-06-05T00:00:00")
        lead_hours = 120

        ensemble = build_lagged_ensemble(ds, "2m_temperature", nominal_init, lead_hours=lead_hours)

        nominal_valid_time = nominal_init + pd.Timedelta(hours=lead_hours)
        for offset in ensemble["member"].values:
            member = ensemble.sel(member=offset)
            source_init = pd.Timestamp(member["source_init_time"].item())
            source_lead = member["source_lead_hours"].item()
            assert source_init + pd.Timedelta(hours=source_lead) == nominal_valid_time

    def test_member_values_match_direct_selection(self):
        ds = _synthetic_source()
        nominal_init = pd.Timestamp("2020-06-05T00:00:00")

        ensemble = build_lagged_ensemble(ds, "2m_temperature", nominal_init, lead_hours=96)

        # offset=0 member must equal the plain deterministic forecast at the
        # nominal init time and lead -- the un-lagged case.
        direct = ds["2m_temperature"].sel(
            time=nominal_init.to_datetime64(), prediction_timedelta=96
        )
        np.testing.assert_array_equal(ensemble.sel(member=0).values, direct.values)


class TestBuildLaggedEnsembleShortWindow:
    def test_short_lead_drops_members_needing_a_lead_below_source_minimum(self):
        # Documented in docs/baseline-store.md: lead=24h keeps only 6 of 9
        # members (offsets +24/+36/+48 would need leads of 0h/-12h/-24h,
        # none of which exist).
        ds = _synthetic_source()
        nominal_init = pd.Timestamp("2020-06-05T00:00:00")

        ensemble = build_lagged_ensemble(ds, "2m_temperature", nominal_init, lead_hours=24)

        assert ensemble.sizes["member"] == 9
        assert ensemble.attrs["n_valid_members"] == 6
        nan_offsets = sorted(
            int(o)
            for o in ensemble["member"].values
            if bool(ensemble.sel(member=o).isnull().all())
        )
        assert nan_offsets == [24, 36, 48]

    def test_lead_48_keeps_eight_of_nine_members(self):
        ds = _synthetic_source()
        nominal_init = pd.Timestamp("2020-06-05T00:00:00")

        ensemble = build_lagged_ensemble(ds, "2m_temperature", nominal_init, lead_hours=48)

        assert ensemble.attrs["n_valid_members"] == 8
        nan_offsets = [
            int(o)
            for o in ensemble["member"].values
            if bool(ensemble.sel(member=o).isnull().all())
        ]
        assert nan_offsets == [48]

    def test_padded_short_window_matches_unpadded_crps_against_verifypy(self):
        # The actual claim this module's docstring relies on: a NaN-padded
        # 9-member array scores identically via weavr.verify.crps to an
        # unpadded array of just the real values -- checked directly against
        # xskillscore's crps_ensemble (which verify.py delegates to), not
        # assumed.
        ds = _synthetic_source()
        nominal_init = pd.Timestamp("2020-06-05T00:00:00")

        padded = build_lagged_ensemble(ds, "2m_temperature", nominal_init, lead_hours=24)
        real_values = padded.dropna(dim="member")
        assert real_values.sizes["member"] == 6

        obs = xr.full_like(padded.isel(member=0, drop=True), 280.0)

        padded_score = V.crps(padded, obs)
        unpadded_score = V.crps(real_values, obs)

        xr.testing.assert_allclose(padded_score, unpadded_score)

    def test_works_directly_as_input_to_verifypy_crps(self):
        ds = _synthetic_source()
        nominal_init = pd.Timestamp("2020-06-05T00:00:00")
        ensemble = build_lagged_ensemble(ds, "2m_temperature", nominal_init, lead_hours=120)
        obs = xr.full_like(ensemble.isel(member=0, drop=True), 280.0)

        score = V.crps(ensemble, obs)

        assert "member" not in score.dims
        assert not bool(score.isnull().any())

    def test_works_directly_as_input_to_verifypy_brier_score_via_exceedance_probability(self):
        ds = _synthetic_source()
        nominal_init = pd.Timestamp("2020-06-05T00:00:00")
        ensemble = build_lagged_ensemble(ds, "2m_temperature", nominal_init, lead_hours=120)

        probability = (ensemble > 280.0).mean(dim="member", skipna=True)
        obs_binary = xr.full_like(probability, 1.0)

        score = V.brier_score(probability, obs_binary)

        assert "member" not in score.dims
        assert not bool(score.isnull().any())


def _synthetic_precip_source(n_days: int = 20) -> xr.Dataset:
    """A source shaped like GraphCast's real total_precipitation_24hr archive:
    the (time, prediction_timedelta) index exists down to 6h, exactly like
    _synthetic_source, but every value at a lead below 24h is NaN -- a real,
    confirmed constraint (a 24-hour accumulation isn't defined until a full
    24h of forecast has elapsed), not a missing index entry. `.sel()` on
    these cells succeeds; the data itself is NaN.
    """
    times = pd.date_range("2020-06-01", periods=n_days * 2, freq="12h")
    leads = np.arange(6, 241, 6)
    lat = np.array([10.0, 10.25])
    lon = np.array([70.0, 70.25])

    rng = np.random.default_rng(0)
    data = rng.normal(
        loc=0.002, scale=0.001, size=(len(times), len(leads), len(lat), len(lon))
    )
    data[:, leads < 24, :, :] = np.nan
    dims = ("time", "prediction_timedelta", "latitude", "longitude")
    return xr.Dataset(
        {"total_precipitation_24hr": (dims, data)},
        coords={"time": times, "prediction_timedelta": leads, "latitude": lat, "longitude": lon},
    )


class TestBuildLaggedEnsembleAllNanValues:
    def test_a_successful_fetch_with_all_nan_data_is_not_counted_as_valid(self):
        # At lead_hours=24h, offset=+12h needs required_lead=12h -- above
        # MIN_SOURCE_LEAD_HOURS=6h (the index exists, .sel() succeeds) but
        # below precipitation's real 24h floor (the data is NaN there).
        # This must be excluded from n_valid_members, not miscounted as a
        # real member just because the fetch itself didn't raise.
        ds = _synthetic_precip_source()
        nominal_init = pd.Timestamp("2020-06-05T00:00:00")

        ensemble = build_lagged_ensemble(
            ds, "total_precipitation_24hr", nominal_init, lead_hours=24
        )

        # offsets -48,-36,-24,-12,0 need leads 72,60,48,36,24 -- all >=24h,
        # real data. offset +12 needs lead 12h -- index exists, data is NaN.
        # offsets +24,+36,+48 need leads 0,-12,-24 -- below MIN_SOURCE_LEAD_HOURS,
        # index doesn't exist either. Only 5 of 9 are genuinely usable.
        assert ensemble.attrs["n_valid_members"] == 5
        assert bool(ensemble.sel(member=12).isnull().all())

    def test_matches_the_real_non_null_count_exactly(self):
        ds = _synthetic_precip_source()
        nominal_init = pd.Timestamp("2020-06-05T00:00:00")

        ensemble = build_lagged_ensemble(
            ds, "total_precipitation_24hr", nominal_init, lead_hours=24
        )

        actual_non_null = int(ensemble.notnull().sum(dim="member").isel(latitude=0, longitude=0))
        assert ensemble.attrs["n_valid_members"] == actual_non_null

    def test_a_lead_with_no_all_nan_gap_is_unaffected(self):
        # At lead_hours=72h, every required lead is >=24h -- the
        # precipitation-specific NaN floor never applies, so this should
        # behave identically to a normal temperature-like source (9/9).
        ds = _synthetic_precip_source()
        nominal_init = pd.Timestamp("2020-06-05T00:00:00")

        ensemble = build_lagged_ensemble(
            ds, "total_precipitation_24hr", nominal_init, lead_hours=72
        )

        assert ensemble.attrs["n_valid_members"] == 9
        assert not bool(ensemble.isnull().any())


class TestNoLaggedMembersError:
    def test_raises_when_every_offset_falls_outside_source_lead_range(self):
        ds = _synthetic_source()
        nominal_init = pd.Timestamp("2020-06-05T00:00:00")

        # lead_hours=2h: even the nominal (0h offset) member needs a 2h lead,
        # below the source's confirmed 6h minimum -- and every lagged offset
        # needs an even more extreme lead. Nothing is fetchable.
        with pytest.raises(NoLaggedMembersError):
            build_lagged_ensemble(ds, "2m_temperature", nominal_init, lead_hours=2)
