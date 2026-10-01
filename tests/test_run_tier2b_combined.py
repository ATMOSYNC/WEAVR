import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_tier2b_combined import (
    cdf_at_threshold,
    combine_predictive_quantiles,
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
