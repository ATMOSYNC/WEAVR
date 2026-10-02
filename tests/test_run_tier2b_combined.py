import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_tier2b_combined import (
    _accumulate_day_sums,
    _new_day_accumulator,
    cdf_at_threshold,
    cell_weighted_mean,
    combine_predictive_quantiles,
    combined_beats_parent,
    comparable_train_crps,
    compute_brier_score,
    fit_quantile_weight,
    pit_summary,
    predictive_mean_from_cdf,
    quantiles_from_cdf,
    score_quantiles_crps,
    select_emos_source,
)


class TestCombinePredictiveQuantiles:
    def test_combines_csgd_and_bma_samples(self):
        mean = np.array([2.0, 5.0])
        std = np.array([1.5, 2.5])
        shift = np.array([-0.5, -1.0])

        rng = np.random.default_rng(42)
        bma_samples = rng.uniform(0, 10, size=(2, 200))
        levels = np.array([0.25, 0.50, 0.75])

        q_csgd, q_bma, q_avg = combine_predictive_quantiles(
            mean, std, shift, bma_samples, levels=levels, weight=0.5
        )

        assert q_csgd.shape == (2, 3)
        assert q_bma.shape == (2, 3)
        assert q_avg.shape == (2, 3)
        assert np.all(q_avg >= 0.0)
        assert np.all(np.diff(q_avg, axis=-1) >= 0.0)


class TestComputeBrierScore:
    def test_perfect_forecast(self):
        prob = np.array([0.0, 1.0, 1.0])
        obs = np.array([2.0, 15.0, 20.0])
        # threshold 10.0 -> event is [0, 1, 1]
        assert compute_brier_score(prob, obs, threshold=10.0) == pytest.approx(0.0)

    def test_worst_forecast(self):
        prob = np.array([1.0, 0.0])
        obs = np.array([0.0, 50.0])
        # threshold 10.0 -> event is [0, 1]
        assert compute_brier_score(prob, obs, threshold=10.0) == pytest.approx(1.0)


class TestScoreQuantilesCrps:
    def test_computes_mean_crps(self):
        quantiles = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        levels = np.array([0.25, 0.50, 0.75])
        obs = np.array([2.0, 5.0])

        score = score_quantiles_crps(quantiles, levels, obs)
        assert score >= 0.0


class TestCdfAtThreshold:
    def test_matches_analytic_cdf_for_uniform_distribution(self):
        # Uniform on [0, 10] -> F(t) = t / 10, so the interpolated estimate
        # should agree with the closed form everywhere strictly inside.
        levels = np.linspace(0.01, 0.99, 99)
        q = np.tile(levels * 10.0, (3, 1))
        for t in (0.5, 3.0, 7.5, 9.9):
            got = cdf_at_threshold(q, levels, t)
            assert np.allclose(got, t / 10.0, atol=2e-2)

    def test_below_all_quantiles_saturates_to_zero(self):
        levels = np.array([0.25, 0.5, 0.75])
        q = np.array([[5.0, 6.0, 7.0]])
        # A threshold under the whole quantile range means the event is
        # certain, so F saturates at 0 rather than at the lowest *reported*
        # level: clamping to 0.25 would understate P(Y > 1.0).
        assert cdf_at_threshold(q, levels, 1.0)[0] == pytest.approx(0.0)

    def test_above_all_quantiles_saturates_to_one(self):
        levels = np.array([0.25, 0.5, 0.75])
        q = np.array([[5.0, 6.0, 7.0]])
        assert cdf_at_threshold(q, levels, 99.0)[0] == pytest.approx(1.0)

    def test_accepts_one_threshold_per_cell(self):
        # The PIT path needs this: each cell asks about its own observation.
        levels = np.linspace(0.01, 0.99, 99)
        q = np.tile(levels * 10.0, (4, 1))
        obs = np.array([1.0, 2.5, 7.5, 9.0])
        got = cdf_at_threshold(q, levels, obs)
        assert got.shape == (4,)
        assert np.allclose(got, obs / 10.0, atol=2e-2)

    def test_flat_point_mass_at_zero_does_not_divide_by_zero(self):
        # The CSGD point mass at zero puts many levels at exactly q = 0, so
        # the bracketing pair is flat. This must stay finite.
        levels = np.array([0.25, 0.5, 0.75])
        q = np.array([[0.0, 0.0, 0.0]])
        out = cdf_at_threshold(q, levels, 0.0)
        assert np.all(np.isfinite(out))

    def test_non_uniform_levels_are_respected(self):
        # Deliberately non-uniform: index-space interpolation would get this
        # wrong, level-space interpolation must not.
        levels = np.array([0.1, 0.5, 0.9])
        q = np.array([[10.0, 20.0, 30.0]])
        # halfway between q=10 (F=0.1) and q=20 (F=0.5) -> F=0.3
        assert cdf_at_threshold(q, levels, 15.0)[0] == pytest.approx(0.3)

    def test_output_is_bounded(self):
        levels = np.linspace(0.01, 0.99, 99)
        rng = np.random.default_rng(0)
        q = np.sort(rng.uniform(0, 50, size=(20, 99)), axis=-1)
        out = cdf_at_threshold(q, levels, 25.0)
        assert np.all((out >= 0.0) & (out <= 1.0))


class TestQuantilesFromCdf:
    def test_round_trips_a_uniform_distribution(self):
        # A uniform [0, 10] CDF inverted at the levels must give back 10*tau.
        levels = np.linspace(0.05, 0.95, 19)
        grid = np.arange(0.0, 20.0 + 0.5, 0.5)
        cdf = np.clip(grid / 10.0, 0.0, 1.0)
        out = quantiles_from_cdf(cdf[None, :], grid, levels)
        assert np.allclose(out[0], levels * 10.0, atol=1.0)

    def test_is_monotone_along_levels(self):
        rng = np.random.default_rng(1)
        grid = np.arange(0.0, 50.0 + 1.0, 1.0)
        cdf = np.sort(rng.uniform(0, 1, size=(7, grid.size)), axis=-1)[:, ::-1]
        cdf = np.minimum.accumulate(cdf, axis=-1)
        levels = np.linspace(0.05, 0.95, 19)
        out = quantiles_from_cdf(cdf, grid, levels)
        assert np.all(np.diff(out, axis=-1) >= 0.0)

    def test_cdf_never_reaching_tau_truncates_to_grid_top(self):
        # A CDF stuck at 0.5 never reaches tau=0.9, so that quantile lies
        # beyond the grid and must truncate to its top value.
        grid = np.arange(0.0, 10.0 + 1.0, 1.0)
        cdf = np.full((1, grid.size), 0.5)
        out = quantiles_from_cdf(cdf, grid, np.array([0.5, 0.9]))
        assert out[0, 0] == pytest.approx(grid[0])
        assert out[0, 1] == pytest.approx(grid[-1])

    def test_point_mass_at_zero_inverts_to_all_zeros(self):
        # F == 1 everywhere is a point mass at zero, whose every quantile is
        # zero. This is the case that catches an inverted crossing test.
        grid = np.arange(0.0, 10.0 + 1.0, 1.0)
        cdf = np.ones((1, grid.size))
        out = quantiles_from_cdf(cdf, grid, np.array([0.1, 0.5, 0.9]))
        assert np.allclose(out[0], 0.0)


class TestPredictiveMeanFromCdf:
    def test_uniform_distribution_mean(self):
        grid = np.arange(0.0, 20.0 + 0.5, 0.5)
        cdf = np.clip(grid / 10.0, 0.0, 1.0)
        got = predictive_mean_from_cdf(cdf[None, :], grid)
        assert got[0] == pytest.approx(5.0, abs=0.05)

    def test_point_mass_at_zero_has_zero_mean(self):
        grid = np.arange(0.0, 50.0 + 1.0, 1.0)
        cdf = np.ones((1, grid.size))
        assert predictive_mean_from_cdf(cdf, grid)[0] == pytest.approx(0.0)


class TestPitSummary:
    def test_perfectly_calibrated_gives_near_half(self):
        # A uniform predictive distribution is calibrated by construction, so
        # F(obs) is uniform and the MAD from 0.5 should be modest.
        levels = np.linspace(0.01, 0.99, 99)
        rng = np.random.default_rng(3)
        obs = rng.uniform(0, 10, size=20000)
        q = np.tile(levels * 10.0, (obs.size, 1))
        summary = pit_summary(q, levels, obs)
        assert summary["pit_mean"] == pytest.approx(0.5, abs=0.02)
        assert summary["pit_mad_from_half"] == pytest.approx(0.25, abs=0.02)
        assert summary["pit_central_90_coverage"] == pytest.approx(0.9, abs=0.02)
        assert summary["n_pit_cells"] == 20000

    def test_overconfident_forecast_is_flagged(self):
        # An over-dispersed predictive distribution (far too wide) puts most
        # observations near F=0, so the MAD from 0.5 must be large.
        levels = np.linspace(0.01, 0.99, 99)
        rng = np.random.default_rng(4)
        obs = rng.uniform(0, 1, size=5000)
        q = np.tile(levels * 100.0, (obs.size, 1))
        summary = pit_summary(q, levels, obs)
        assert summary["pit_mean"] < 0.1
        assert summary["pit_mad_from_half"] > 0.4


class TestFitQuantileWeight:
    def test_recovers_an_endpoint_when_bma_is_useless(self):
        # A BMA parent that predicts a constant zero is strictly worse than
        # CSGD everywhere, so the search must collapse onto w=0 (pure CSGD)
        # rather than inventing an interior mixture.
        levels = np.linspace(0.05, 0.95, 19)
        rng = np.random.default_rng(5)
        obs = rng.uniform(0, 20, size=500)
        # A CSGD parent centred well on the observations, against a BMA
        # parent that predicts a constant zero.
        q_csgd = np.tile(np.linspace(2.0, 18.0, levels.size), (obs.size, 1))
        q_bma = np.zeros((obs.size, levels.size))
        w, _ = fit_quantile_weight(q_csgd, q_bma, levels, obs)
        assert w == pytest.approx(0.0)

    def test_weight_is_within_grid(self):
        levels = np.linspace(0.05, 0.95, 19)
        rng = np.random.default_rng(6)
        q_a = np.sort(rng.uniform(0, 20, size=(50, levels.size)), axis=-1)
        q_b = np.sort(rng.uniform(0, 20, size=(50, levels.size)), axis=-1)
        obs = rng.uniform(0, 20, size=50)
        w, crps = fit_quantile_weight(q_a, q_b, levels, obs)
        assert 0.0 <= w <= 1.0
        assert np.isfinite(crps)


class TestSelectEmosSource:
    def test_picks_lower_train_crps(self):
        assert select_emos_source({"graphcast": 5.0, "ifs_ens": 4.0}) == "ifs_ens"

    def test_tie_goes_to_tie_breaker(self):
        assert select_emos_source({"graphcast": 4.0, "ifs_ens": 4.0}) == "graphcast"

    def test_nan_source_is_never_selected(self):
        # min() over a dict containing NaN would return the NaN entry's key
        # depending on order; a source with no trainable cells must be skipped.
        assert select_emos_source({"graphcast": float("nan"), "ifs_ens": 4.0}) == "ifs_ens"
        both_nan = {"graphcast": float("nan"), "ifs_ens": float("nan")}
        assert select_emos_source(both_nan) == "graphcast"


class TestCombinedBeatsParent:
    """The sign rule that decided H10, pinned with the case that got it wrong.

    Every number below is taken from a real Tier 2b run. The first case is the
    one that matters: `d = -0.0467` with CI `[-0.0743, -0.0222]` is the
    combination beating EMOS-CSG at lead 120 by a wide, clearly significant
    margin. Testing `ci_low > 0` called it a failure, so the run reported a FAIL
    whose actual cause was an inverted comparison rather than the methods.
    """

    def test_significant_improvement_counts_as_a_win(self):
        assert combined_beats_parent(-0.046671, -0.022159) is True

    def test_significant_improvement_over_bma_counts_as_a_win(self):
        assert combined_beats_parent(-0.015549, -0.008425) is True

    def test_an_interval_straddling_zero_is_not_a_win(self):
        # lead 48 against bma: negative point estimate, upper bound above 0.
        assert combined_beats_parent(-0.021209, 0.000426) is False

    def test_a_clear_loss_is_not_a_win(self):
        # lead 96 against bma, and lead 96 against emos_csg in the other
        # direction. Both are worse and both must stay losses.
        assert combined_beats_parent(0.034662, 0.052665) is False
        assert combined_beats_parent(0.006380, 0.031499) is False

    def test_an_exact_tie_is_not_a_win(self):
        # The collapsed per-bin arm: difference and both bounds identically 0.
        assert combined_beats_parent(0.0, 0.0) is False

    def test_degenerate_bootstrap_never_wins(self):
        assert combined_beats_parent(-0.5, -0.1, degenerate=True) is False

    def test_positive_difference_with_intervals_above_zero_is_not_a_win(self):
        # The exact shape the inverted test used to accept.
        assert combined_beats_parent(0.01, 0.02) is False


class TestCellWeightedMean:
    """The averaging bug that decided H10 on the wrong arm.

    `values.sum() / counts.sum()` divides a sum of per-group means by a sum of
    cell counts. It is not a mean, it has no stable scale, and here it reported
    `per_bin`'s train CRPS as 7e-5 mm against `quantile_avg`'s 4.8 mm, so the
    primary-arm nomination picked `per_bin` at four of five leads on units
    rather than skill. Every value was finite and no test failed.
    """

    def test_equals_the_plain_mean_when_groups_are_equal_sized(self):
        assert cell_weighted_mean(
            np.array([4.0, 6.0]), np.array([10.0, 10.0])
        ) == pytest.approx(5.0)

    def test_weights_toward_the_larger_group(self):
        # Two groups, wildly unequal counts: the big one must dominate.
        assert cell_weighted_mean(
            np.array([4.0, 6.0]), np.array([1000.0, 1.0])
        ) == pytest.approx((4.0 * 1000 + 6.0 * 1) / 1001.0)

    def test_result_lies_within_the_range_of_the_inputs(self):
        """The broken version falls orders of magnitude below the smallest input.

        This is the property that would have caught it: no average of per-group
        means can land outside the range of those means.
        """
        values = np.array([4.8, 4.9, 5.0])
        counts = np.array([20_000.0, 18_000.0, 19_000.0])
        got = cell_weighted_mean(values, counts)
        assert values.min() <= got <= values.max()
        # And specifically not the 7e-5 the sum-of-means version produced.
        assert got > 0.1

    def test_broken_shortcut_is_far_outside_the_range(self):
        values = np.array([4.8, 4.9, 5.0])
        counts = np.array([20_000.0, 18_000.0, 19_000.0])
        broken = float(values.sum() / counts.sum())
        assert broken < 1e-3, "the shortcut no longer reproduces the historical bug"

    def test_empty_input_is_nan_not_zero(self):
        assert np.isnan(cell_weighted_mean(np.array([]), np.array([])))

    def test_zero_counts_is_nan_not_a_division_error(self):
        assert np.isnan(cell_weighted_mean(np.array([4.0]), np.array([0.0])))


class TestComparableTrainCrps:
    """The guard that stops a units error deciding which arm is judged."""

    def test_plausible_values_pass_through_unchanged(self):
        got = comparable_train_crps(
            {"quantile_avg": 4.83, "per_bin": 4.79, "other": 5.01}
        )
        assert got == {"quantile_avg": 4.83, "per_bin": 4.79, "other": 5.01}

    def test_the_historical_bug_is_rejected(self):
        """7e-5 against 4.8 is a units error, and must raise rather than pick."""
        with pytest.raises(ValueError, match="not on a common scale"):
            comparable_train_crps({"quantile_avg": 4.83, "per_bin": 7.35e-5})

    def test_error_names_the_offending_values(self):
        with pytest.raises(ValueError) as excinfo:
            comparable_train_crps({"quantile_avg": 4.83, "per_bin": 7.35e-5})
        message = str(excinfo.value)
        assert "quantile_avg" in message and "per_bin" in message

    def test_non_finite_values_are_ignored(self):
        got = comparable_train_crps(
            {"quantile_avg": 4.83, "per_bin": float("nan")}
        )
        assert got == {"quantile_avg": 4.83}

    def test_a_single_candidate_cannot_be_mismatched(self):
        assert comparable_train_crps({"per_bin": 4.8}) == {"per_bin": 4.8}

    def test_no_candidates_is_not_an_error(self):
        assert comparable_train_crps({}) == {}

    def test_ratio_limit_is_configurable_for_tightening(self):
        with pytest.raises(ValueError):
            comparable_train_crps(
                {"a": 4.83, "b": 4.0}, ratio_limit=1.05
            )


class TestDayAccumulatorBrier:
    """The Brier columns were squared rainfall, not a Brier score.

    The binary observation was passed in as the raw rainfall value, so the
    score was `(probability - rainfall_mm) ** 2` -- about 285 mm^2, where a
    Brier score is bounded by 1. Every `brier_*` column in every written output
    carried it. These tests pin both the bound and the indicator's direction,
    which is what the raw-value version got wrong.
    """

    THRESHOLDS = [7.5, 115.6]

    def _accumulator(self):
        return _new_day_accumulator(n_days=1, thresholds=self.THRESHOLDS)

    def _run(self, exceed, obs):
        acc = self._accumulator()
        metrics = {
            "crps": np.ones_like(obs, dtype=float),
            "pit": np.full_like(obs, 0.5, dtype=float),
            "exceed": np.asarray(exceed, dtype=float).reshape(len(self.THRESHOLDS), -1),
        }
        _accumulate_day_sums(
            acc, metrics, np.zeros(len(obs), dtype=int),
            self.THRESHOLDS, np.asarray(obs, dtype=float),
        )
        # `brier_num` accumulates a sum over cells; the reported score divides
        # by the cell count, so the mean is what a caller sees.
        return acc["brier_num"][:, 0] / float(len(obs))

    def test_a_perfect_forecast_scores_zero(self):
        # Both cells stay below 7.5 mm and the forecast says so, at both
        # thresholds. (An earlier fixture used 20 mm here, which *does* exceed
        # 7.5 mm -- the assertion was wrong, not the code.)
        got = self._run([[0.0, 0.0], [0.0, 0.0]], [2.0, 5.0])
        assert np.allclose(got, 0.0)

    def test_a_certain_miss_scores_one(self):
        # Predicting no exceedance where it rained heavily is the worst case at
        # 7.5 mm, where both cells exceed. At 115.6 mm only the 250 mm cell
        # does, so that row scores 0.5 -- the threshold-relative part that the
        # raw-rainfall version could not have produced.
        got = self._run([[0.0, 0.0], [0.0, 0.0]], [30.0, 250.0])
        assert np.isclose(got[0], 1.0)
        assert np.isclose(got[1], 0.5)

    def test_the_score_is_bounded_by_one(self):
        got = self._run([[0.3, 0.7], [0.2, 0.9]], [30.0, 250.0])
        assert np.all(got >= 0.0) and np.all(got <= 1.0)

    def test_the_historical_defect_would_be_hundreds(self):
        """Rainfall in mm against a probability: the old number, for contrast."""
        obs_mm = np.array([30.0, 250.0])
        prob = np.array([0.3, 0.7])
        broken = float(np.mean((prob - obs_mm) ** 2))
        assert broken > 100.0

    def test_exceedance_uses_strictly_greater_than(self):
        # obs exactly at the threshold is not an exceedance.
        got = self._run([[0.0], [0.0]], [7.5])
        assert np.allclose(got, 0.0)
