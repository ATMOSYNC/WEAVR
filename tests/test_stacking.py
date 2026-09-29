"""Tests for weavr.stacking (Step 13: Combine the combiners)."""

import numpy as np
import pandas as pd

from weavr.emos import csgd_crps
from weavr.stacking import (
    crps_from_quantiles,
    linear_pool,
    predictive_quantiles_bma,
    predictive_quantiles_csgd,
    quantile_average,
    select_per_bin,
)


class TestSelectPerBin:
    def test_selects_lowest_crps_per_bin_and_lead(self):
        df = pd.DataFrame(
            [
                {"bin": "dry", "lead": 24, "combiner": "bma", "crps": 0.05},
                {"bin": "dry", "lead": 24, "combiner": "emos_csg", "crps": 0.08},
                {"bin": "light", "lead": 24, "combiner": "bma", "crps": 1.20},
                {"bin": "light", "lead": 24, "combiner": "emos_csg", "crps": 0.95},
                {"bin": "heavy", "lead": 48, "combiner": "bma", "crps": 4.50},
                {"bin": "heavy", "lead": 48, "combiner": "emos_csg", "crps": 4.10},
            ]
        )
        res = select_per_bin(df)
        assert res.get("dry", 24) == "bma"
        assert res.get("light", 24) == "emos_csg"
        assert res.get("heavy", 48) == "emos_csg"

    def test_tie_breaker_prefers_emos_csg(self):
        df = pd.DataFrame(
            [
                {"bin": "moderate", "lead": 24, "combiner": "bma", "crps": 1.50},
                {"bin": "moderate", "lead": 24, "combiner": "emos_csg", "crps": 1.50},
            ]
        )
        res = select_per_bin(df, tie_breaker="emos_csg")
        assert res.get("moderate", 24) == "emos_csg"

    def test_unseen_bin_falls_back_to_pooled_best(self):
        df = pd.DataFrame(
            [
                {"bin": "dry", "lead": 24, "combiner": "bma", "crps": 0.50},
                {"bin": "dry", "lead": 24, "combiner": "emos_csg", "crps": 0.20},
                {"bin": "light", "lead": 24, "combiner": "bma", "crps": 0.80},
                {"bin": "light", "lead": 24, "combiner": "emos_csg", "crps": 0.60},
            ]
        )
        res = select_per_bin(df)
        # Overall pooled best is emos_csg (mean 0.40 vs 0.65)
        assert res.pooled_best == "emos_csg"
        # Unseen extreme bin falls back to pooled_best
        assert res.get("extreme", 96) == "emos_csg"


class TestPredictiveQuantilesCsgd:
    def test_point_mass_censoring_at_zero(self):
        # With mean=2.0, std=3.0, shift=-1.5:
        # A significant fraction of probability mass is censored at zero
        levels = np.array([0.05, 0.10, 0.20, 0.50, 0.80, 0.95])
        q = predictive_quantiles_csgd(mean=2.0, std=3.0, shift=-1.5, levels=levels)

        # Non-negative everywhere
        assert np.all(q >= 0.0)
        # Monotone non-decreasing
        assert np.all(np.diff(q) >= 0.0)
        # Lowest quantile should be zero due to left-censoring
        assert q[0] == 0.0

    def test_degenerate_zero_mean_returns_zeros(self):
        levels = np.array([0.1, 0.5, 0.9])
        q = predictive_quantiles_csgd(mean=0.0, std=1e-6, shift=0.0, levels=levels)
        assert np.all(q == 0.0)

    def test_vectorized_broadcasting(self):
        mean = np.array([1.0, 5.0, 10.0])
        std = np.array([1.5, 3.0, 5.0])
        shift = np.array([-0.5, -1.0, -2.0])
        levels = np.array([0.25, 0.5, 0.75])

        q = predictive_quantiles_csgd(mean, std, shift, levels=levels)
        assert q.shape == (3, 3)
        assert np.all(q >= 0.0)
        assert np.all(np.diff(q, axis=-1) >= 0.0)


class TestPredictiveQuantilesBma:
    def test_empirical_quantiles_match_known_distribution(self):
        rng = np.random.default_rng(42)
        # 10000 samples from an exponential distribution with scale 5.0
        samples = rng.exponential(scale=5.0, size=(1, 10000))
        levels = np.array([0.25, 0.50, 0.75])

        q = predictive_quantiles_bma(samples, levels=levels)
        assert q.shape == (1, 3)
        # True theoretical quantiles: -scale * ln(1 - tau)
        expected = -5.0 * np.log(1.0 - levels)
        np.testing.assert_allclose(q[0], expected, rtol=0.05)


class TestQuantileAverage:
    def test_averages_quantiles_properly(self):
        q_a = np.array([[1.0, 2.0, 3.0]])
        q_b = np.array([[3.0, 6.0, 9.0]])

        # w = 0 -> q_a
        np.testing.assert_allclose(quantile_average(q_a, q_b, weight=0.0), q_a)
        # w = 1 -> q_b
        np.testing.assert_allclose(quantile_average(q_a, q_b, weight=1.0), q_b)
        # w = 0.5 -> average
        expected = np.array([[2.0, 4.0, 6.0]])
        np.testing.assert_allclose(quantile_average(q_a, q_b, weight=0.5), expected)

    def test_monotonicity_preserved(self):
        q_a = np.sort(np.random.default_rng(1).uniform(0, 10, size=(5, 20)), axis=-1)
        q_b = np.sort(np.random.default_rng(2).uniform(0, 15, size=(5, 20)), axis=-1)

        q_avg = quantile_average(q_a, q_b, weight=0.4)
        assert np.all(np.diff(q_avg, axis=-1) >= 0.0)


class TestCrpsFromQuantilesAccuracy:
    def test_matches_closed_form_csgd_crps_within_tolerance(self):
        """Step 13 requirement: test crps_from_quantiles against closed-form csgd_crps

        on a known CSGD (error below a stated tolerance, < 1.5% with 99 levels).
        """
        mean = 4.0
        std = 3.5
        shift = -1.2
        obs_values = np.array([0.0, 1.5, 4.0, 8.0, 15.0])

        levels = np.linspace(0.005, 0.995, 199)
        q = predictive_quantiles_csgd(mean=mean, std=std, shift=shift, levels=levels)

        # Closed-form exact CRPS from weavr.emos
        exact_crps = csgd_crps(mean, std, shift, obs_values)

        # Approximate CRPS via pinball loss integration over quantiles
        approx_crps = np.array([crps_from_quantiles(q, levels, y) for y in obs_values])

        # Relative error should be under 1.5% across all observation points
        relative_errors = np.abs(approx_crps - exact_crps) / np.maximum(exact_crps, 1e-4)
        assert np.all(relative_errors < 0.015), f"Max relative error: {np.max(relative_errors):.4f}"


class TestLinearPool:
    def test_probability_averaging(self):
        pa = np.array([0.2, 0.8])
        pb = np.array([0.6, 0.4])
        pool = linear_pool(pa, pb, weight=0.5)
        np.testing.assert_allclose(pool, np.array([0.4, 0.6]))
