import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from validate_daily_stores import (  # noqa: E402
    EXPECTED_MEMBERS_BY_LEAD,
    Report,
    check_init_times,
    check_not_all_nan,
    count_members_by_lead,
    validate_baseline,
)

LEADS = [24, 48, 72, 96, 120]
OFFSETS = [0, 12, 24, 36, 48, 60, 72, 84, 96]


def _daily_times(n=122, start="2020-06-01"):
    return pd.date_range(start, periods=n, freq="D").values


class TestCheckInitTimes:
    def test_a_complete_daily_season_passes(self):
        report = Report()
        check_init_times(_daily_times(), report, "x", 122)
        assert report.failures == []

    def test_a_short_fetch_fails(self):
        # The most likely real failure: the build stopped partway.
        report = Report()
        check_init_times(_daily_times(n=80), report, "x", 122)
        assert any("80 init times" in f for f in report.failures)

    def test_a_gap_in_the_sequence_fails(self):
        times = np.delete(_daily_times(), 50)
        report = Report()
        check_init_times(times, report, "x", 121)
        assert any("gaps" in f for f in report.failures)

    def test_the_wrong_season_fails(self):
        report = Report()
        check_init_times(_daily_times(start="2020-03-01"), report, "x", 122)
        assert any("spans" in f for f in report.failures)

    def test_an_empty_store_fails_without_crashing(self):
        report = Report()
        check_init_times(np.array([], dtype="datetime64[ns]"), report, "x", 122)
        assert report.failures


class TestCheckNotAllNan:
    def _dataset(self, values):
        return xr.Dataset(
            {"rain": (("time", "latitude", "longitude"), values)},
            coords={
                "time": _daily_times(n=values.shape[0]),
                "latitude": np.arange(values.shape[1], dtype=float),
                "longitude": np.arange(values.shape[2], dtype=float),
            },
        )

    def test_real_data_passes(self):
        report = Report()
        check_not_all_nan(self._dataset(np.ones((5, 3, 3))), report, "g")
        assert report.failures == []

    def test_an_entirely_nan_variable_fails(self):
        report = Report()
        check_not_all_nan(self._dataset(np.full((5, 3, 3), np.nan)), report, "g")
        assert any("finite" in f for f in report.failures)

    def test_one_all_nan_day_fails_even_when_the_rest_is_fine(self):
        # A store can be 99% populated and still have a broken day that would
        # silently drop out of every score.
        values = np.ones((5, 3, 3))
        values[2] = np.nan
        report = Report()
        check_not_all_nan(self._dataset(values), report, "g")
        assert any("all-NaN samples" in f for f in report.failures)

    def test_partial_nan_within_a_day_is_allowed(self):
        # IMD's grid is land-only, so ~73% NaN per day is expected and must
        # not be treated as a failure.
        values = np.full((5, 4, 4), np.nan)
        values[:, 0, 0] = 1.0
        report = Report()
        check_not_all_nan(self._dataset(values), report, "g")
        assert report.failures == []


class TestCountMembersByLead:
    def _lagged(self, members_by_lead):
        """A lagged array where each lead has exactly the requested members."""
        values = np.full((3, len(LEADS), len(OFFSETS), 2, 2), np.nan)
        for li, lead in enumerate(LEADS):
            for mi in range(members_by_lead[lead]):
                values[:, li, mi, :, :] = 1.0
        return xr.DataArray(
            values,
            coords={
                "nominal_time": _daily_times(n=3),
                "lead_hours": LEADS,
                "member_offset_hours": OFFSETS,
                "latitude": [0.0, 1.0],
                "longitude": [0.0, 1.0],
            },
            dims=["nominal_time", "lead_hours", "member_offset_hours", "latitude", "longitude"],
        )

    def test_recovers_the_documented_temperature_pattern(self):
        expected = EXPECTED_MEMBERS_BY_LEAD["2m_temperature"]
        assert count_members_by_lead(self._lagged(expected)) == expected

    def test_recovers_the_documented_precipitation_pattern(self):
        expected = EXPECTED_MEMBERS_BY_LEAD["total_precipitation_24hr"]
        assert count_members_by_lead(self._lagged(expected)) == expected

    def test_takes_the_minimum_across_nominal_times_not_a_sample(self):
        # One bad init with fewer members must be caught. Sampling the first
        # few nominal times (the previous implementation) would miss it.
        expected = EXPECTED_MEMBERS_BY_LEAD["2m_temperature"]
        array = self._lagged(expected)
        array[2, 4, 8, :, :] = np.nan  # last nominal time loses a member at 120h

        assert count_members_by_lead(array)[120] == 8
        assert expected[120] == 9


class TestValidateBaseline:
    def test_a_missing_store_fails_rather_than_raising(self, tmp_path):
        report = Report()
        validate_baseline(str(tmp_path / "absent.zarr"), report, 122)
        assert any("does not exist" in f for f in report.failures)


class TestReport:
    def test_collects_every_failure_not_just_the_first(self):
        report = Report()
        report.check(False, "first")
        report.check(True, "second")
        report.check(False, "third")
        assert report.failures == ["first", "third"]

    def test_check_returns_the_condition(self):
        report = Report()
        assert report.check(True, "ok") is True
        assert report.check(False, "no") is False


@pytest.mark.parametrize("variable", sorted(EXPECTED_MEMBERS_BY_LEAD))
def test_documented_patterns_cover_every_lead(variable):
    assert sorted(EXPECTED_MEMBERS_BY_LEAD[variable]) == LEADS
