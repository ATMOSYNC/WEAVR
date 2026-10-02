"""Tests for scripts/run_tail_repair_evt.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_scorecard import _pooled_sedi  # noqa: E402
from run_tail_repair_evt import (  # noqa: E402
    compute_sedi,
    expected_counts,
    main,
    parse_args,
)


def test_parse_args() -> None:
    """parse_args accepts expected CLI flags."""
    args = parse_args(["--threshold-u", "70.0", "--out-csv", "results/test.csv"])
    assert args.threshold_u == 70.0
    assert args.out_csv == Path("results/test.csv")


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert args.threshold_u == 64.5
    assert args.results_dir == "results"
    assert args.ifs_ensemble_stores is None


class TestComputeSedi:
    """`compute_sedi` delegates to the canonical `_pooled_sedi`.

    The stub carried its own eps-guarded copy of SEDI. The two agree to ~9
    decimals on any non-degenerate contingency table, but the stub fudged the
    degenerate cases: it returned 1.0 for a perfectly separated forecast
    (H == 1) and 0.0 for an empty table, where `weavr.verify.sedi` returns NaN
    because one of the logarithms is undefined.

    Those are not cosmetic. F = 0 happens constantly at 204.5 mm -- it means
    the forecast never issued a false alarm, which happens whenever it never
    forecasts the event at all -- and `docs/preregistration.md` uses SEDI
    precisely because it stays non-degenerate where CSI and ETS collapse.
    One implementation, matching the documented NaN contract.
    """

    def test_matches_canonical_on_non_degenerate(self):
        for counts in [(10, 5, 2, 100), (6, 9, 4, 81), (30, 70, 10, 890)]:
            assert compute_sedi(*counts) == pytest.approx(_pooled_sedi(*counts))

    def test_strong_but_non_degenerate_forecast_scores_well(self):
        # h = 10/15 = 0.67, f = 2/102 = 0.0196 -> a genuinely skilful forecast.
        assert compute_sedi(10, 5, 2, 100) > 0.8

    def test_perfectly_separated_is_nan_not_one(self):
        # H == 1 makes ln(1 - H) undefined.
        assert np.isnan(compute_sedi(10, 0, 0, 100))

    def test_empty_table_is_nan_not_zero(self):
        assert np.isnan(compute_sedi(0, 0, 0, 0))


class TestExpectedCounts:
    """Probability forecasts feed SEDI through expected counts.

    Using `sum(p * obs)` rather than a hard 0.5 decision matters here: a 0.5
    rule would discard exactly the low-probability extreme cells this step
    exists to repair, and would silently make SEDI look worse than it is.
    """

    def test_perfect_probability(self):
        prob = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        obs = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        counts = expected_counts(prob, obs)
        assert counts["hits"].sum() == pytest.approx(2.0)
        assert counts["false_alarms"].sum() == pytest.approx(0.0)
        assert counts["misses"].sum() == pytest.approx(0.0)

    def test_climatology_conserves_total_counts(self):
        rng = np.random.default_rng(0)
        prob = rng.uniform(0, 1, size=(3, 4, 5))
        obs = (rng.uniform(size=(3, 4, 5)) > 0.7).astype(float)
        counts = expected_counts(prob, obs)
        per_day = sum(counts.values())
        assert np.allclose(per_day, 20.0)

    def test_nan_excluded(self):
        prob = np.array([[[np.nan, 0.5]]])
        obs = np.array([[[1.0, 1.0]]])
        counts = expected_counts(prob, obs)
        assert counts["hits"].sum() == pytest.approx(0.5)


class TestMainIsNotAStub:
    """The regression that matters: this runner used to evaluate nothing."""

    def test_main_is_callable(self):
        assert callable(main)

    def test_no_longer_prints_only_initialized_message(self):
        source = (SCRIPTS / "run_tail_repair_evt.py").read_text()
        assert "Tail repair EVT runner initialized" not in source


def test_csgd_only_arm_uses_point_mass_fallback() -> None:
    """The unfittable-bin arm must genuinely emit hard zeros.

    That is the behaviour the GPD tail exists to replace, so the comparison
    would be meaningless if this arm quietly repaired itself.

    The fitted cell is given a genuinely heavy forecast (members
    180/200/250 mm) so its CSGD probability of exceeding 115.6 mm is a small
    but real 0.03 -- a 5 mm forecast would legitimately score 0.0 there and
    the test would prove nothing about the fallback.
    """
    from run_tail_repair_evt import csgd_only_probabilities

    from weavr.emos import CensoredShiftedGammaResult

    coords = {
        "sample": [0, 1],
        "latitude": [10.0],
        "longitude": [70.0, 71.0],
        "member": [0, 1, 2],
    }
    per_member = np.array([[180.0, 10.0], [200.0, 12.0], [250.0, 8.0]])
    data = np.stack(
        [np.stack([np.stack([per_member[m] for m in range(3)], axis=-1)]) for _ in range(2)]
    )
    forecasts = xr.DataArray(
        data, coords=coords, dims=("sample", "latitude", "longitude", "member")
    )

    label_coords = {k: coords[k] for k in ("sample", "latitude", "longitude")}
    labels = xr.DataArray(
        np.array([[["light", "extremely_heavy"]]] * 2),
        coords=label_coords,
        dims=("sample", "latitude", "longitude"),
    )
    emos_results = {
        "light": CensoredShiftedGammaResult(
            bin_label="light",
            source="graphcast",
            shift=0.0,
            climatological_mean=2.0,
            climatological_std=1.0,
            coefficients={"a1": 1.0, "a2": 0.5, "a3": 0.2, "a4": 0.1},
            is_fallback=False,
        )
    }

    probabilities = csgd_only_probabilities(forecasts, labels, emos_results, 115.6)
    assert probabilities[0, 0, 1] == 0.0
    assert 0.0 < probabilities[0, 0, 0] < 1.0
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))

    # Well below the fitted bin's scale, the same bin saturates at 1.
    assert csgd_only_probabilities(forecasts, labels, emos_results, 64.5)[0, 0, 0] == 1.0


class TestMemberCountingBenchmark:
    """The IFS-ENS member-counting benchmark #74 recorded as NOT RUN.

    `--ifs-ensemble-stores` existed on the runner but was never resolved into
    paths, so `ifs_available` was decided on an argument nothing read, the
    runner printed "enabled", and no benchmark row was ever emitted. The
    benchmark is the comparison step 12 is measured *against*, so silently
    omitting it makes the spliced tail look unopposed.
    """

    def test_probability_is_the_empirical_member_fraction(self):
        import numpy as np
        import xarray as xr
        from run_tail_repair_evt import member_counting_probabilities

        # (sample=2, member=4, lat=1, lon=2) -- the shape `load_ifs_ensemble`
        # produces: one lead already selected, time renamed to `sample`.
        # Members 1 and 2 exceed 5 mm at lon 70; no member does at lon 71.
        members = np.array(
            [
                [[[10.0, 1.0]], [[10.0, 1.0]], [[1.0, 1.0]], [[1.0, 1.0]]],
                [[[20.0, 2.0]], [[20.0, 2.0]], [[2.0, 2.0]], [[2.0, 2.0]]],
            ]
        )
        ens = xr.DataArray(
            members,
            dims=("sample", "member", "latitude", "longitude"),
            coords={
                "sample": [0, 1],
                "member": [1, 2, 3, 4],
                "latitude": [10.0],
                "longitude": [70.0, 71.0],
            },
        )
        probs = member_counting_probabilities(
            ens, 5.0, np.array([True, False])
        )
        # Sample 0: 2 of 4 members above 5 mm at lon 70, none at lon 71.
        assert probs[0, 0, 0] == pytest.approx(0.5)
        assert probs[0, 0, 1] == pytest.approx(0.0)
        # Only the test sample is returned, so the result lines up with
        # `test_obs` (one row per test day) rather than the full series.
        assert probs.shape == (1, 1, 2)
        assert not np.isnan(probs).any()

    def test_benchmark_row_is_scored_with_the_shared_helpers(self):
        import numpy as np
        from run_tail_repair_evt import benchmark_row

        probs = np.full((2, 2, 2), 0.25)
        obs = np.array([[[0.0, 60.0], [0.0, 60.0]], [[0.0, 60.0], [0.0, 60.0]]])
        row = benchmark_row(
            probs, obs, climatology_grid=np.full((2, 2), 0.1), threshold=50.0,
            lead_hours=24, split_label="2018", split_kind="leave_one_year_out",
            n_train=8, n_test=8,
        )
        assert row["arm"] == "raw_ifs_ens_member_counting"
        # Half the cells are above the threshold and predicted 0.25.
        assert row["brier"] == pytest.approx(0.5 * 0.25**2 + 0.5 * 0.75**2)
        assert np.isfinite(row["sedi"])
        assert row["false_zero_cells"] == 0

    def test_no_false_zeros_when_members_exceed(self):
        import numpy as np
        from run_tail_repair_evt import benchmark_row

        probs = np.full((2, 2, 2), 0.5)  # no exact zeros
        obs = np.zeros((2, 2, 2))
        row = benchmark_row(
            probs, obs, climatology_grid=np.zeros((2, 2)), threshold=115.6,
            lead_hours=24, split_label="2018", split_kind="leave_one_year_out",
            n_train=8, n_test=8,
        )
        assert row["false_zero_cells"] == 0
