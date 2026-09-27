import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_lagged_ensemble_store import (  # noqa: E402
    MAX_SOURCE_LEAD_HOURS,
    MIN_SOURCE_LEAD_HOURS,
    _lag_offsets,
    _valid_combos,
)


class TestLagOffsets:
    def test_default_four_lags_twelve_hour_spacing_gives_nine_offsets(self):
        offsets = _lag_offsets(n_lags=4, spacing_hours=12)
        assert offsets == [-48, -36, -24, -12, 0, 12, 24, 36, 48]

    def test_includes_zero_the_nominal_member(self):
        offsets = _lag_offsets(n_lags=2, spacing_hours=6)
        assert 0 in offsets
        assert len(offsets) == 5


class TestValidCombos:
    def test_short_lead_drops_members_needing_a_lead_below_the_source_minimum(self):
        # nominal lead 24h; offsets +24/+36/+48h would need source leads of
        # 0h/-12h/-24h, none of which exist (source leads start at 6h) --
        # these three offsets must be dropped, not substituted or clamped.
        nominal_times = np.array(["2020-06-01"], dtype="datetime64[ns]")
        combos = _valid_combos(nominal_times, lead_hours=[24], offsets=[-48, -24, 0, 24, 36, 48])

        kept_offsets = sorted(c[2] for c in combos)  # index into the offsets list
        offsets = [-48, -24, 0, 24, 36, 48]
        kept_values = sorted(offsets[k] for k in kept_offsets)
        assert kept_values == [-48, -24, 0]

    def test_long_lead_keeps_every_offset_within_source_range(self):
        nominal_times = np.array(["2020-06-01"], dtype="datetime64[ns]")
        offsets = [-48, -36, -24, -12, 0, 12, 24, 36, 48]
        combos = _valid_combos(nominal_times, lead_hours=[120], offsets=offsets)

        assert len(combos) == len(offsets)

    def test_required_source_lead_and_source_time_are_computed_correctly(self):
        nominal_times = np.array(["2020-06-01T00:00:00"], dtype="datetime64[ns]")
        combos = _valid_combos(nominal_times, lead_hours=[48], offsets=[12])

        assert len(combos) == 1
        i, j, k, source_time, source_lead = combos[0]
        assert (i, j, k) == (0, 0, 0)
        assert source_lead == 36  # 48h nominal lead - 12h offset
        assert source_time == np.datetime64("2020-06-01T12:00:00")

    def test_never_yields_a_source_lead_outside_the_confirmed_source_range(self):
        nominal_times = np.array(["2020-06-01", "2020-07-15"], dtype="datetime64[ns]")
        combos = _valid_combos(
            nominal_times,
            lead_hours=[24, 48, 72, 96, 120],
            offsets=[-48, -36, -24, -12, 0, 12, 24, 36, 48],
        )

        for _, _, _, _, source_lead in combos:
            assert MIN_SOURCE_LEAD_HOURS <= source_lead <= MAX_SOURCE_LEAD_HOURS

    def test_member_counts_by_lead_match_the_documented_pattern(self):
        # Documented in docs/baseline-store.md: lead=24h -> 6/9, lead=48h -> 8/9,
        # lead=72/96/120h -> 9/9, derived from the source's confirmed 6h..240h
        # prediction_timedelta range.
        nominal_times = np.array(["2020-06-01"], dtype="datetime64[ns]")
        offsets = [-48, -36, -24, -12, 0, 12, 24, 36, 48]
        expected = {24: 6, 48: 8, 72: 9, 96: 9, 120: 9}

        for lead, n_expected in expected.items():
            combos = _valid_combos(nominal_times, lead_hours=[lead], offsets=offsets)
            assert len(combos) == n_expected, f"lead={lead}"
