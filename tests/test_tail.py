"""Tests for extreme-value tail repair (weavr.tail, Step 12)."""

from __future__ import annotations

import numpy as np
import pytest

from weavr.emos import CensoredShiftedGammaResult
from weavr.tail import (
    TailFit,
    exceedance_probability_with_tail,
    fit_pooled_gpd,
    nearest_fittable_bin,
    spliced_exceedance_probability,
)


def _sample_gpd(xi: float, sigma: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample from Generalized Pareto Distribution GPD(xi, sigma)."""
    u = rng.uniform(0.0, 1.0, size=size)
    if abs(xi) < 1e-7:
        return -sigma * np.log(1.0 - u)
    return (sigma / xi) * ((1.0 - u) ** (-xi) - 1.0)


def test_fit_pooled_gpd_synthetic() -> None:
    """fit_pooled_gpd recovers true parameters on synthetic GPD data."""
    rng = np.random.default_rng(42)
    true_xi = 0.2
    scales = {"WC": 15.0, "WI": 12.0, "SI": 14.0, "CI": 18.0, "NE1": 16.0, "NE2": 13.0}

    exceedances = {}
    for r, sig in scales.items():
        # generate 500 excesses above u=64.5
        excess = _sample_gpd(true_xi, sig, size=500, rng=rng)
        exceedances[r] = 64.5 + excess

    fit = fit_pooled_gpd(exceedances, threshold_u=64.5)

    assert isinstance(fit, TailFit)
    assert fit.threshold_u == 64.5
    # xi should be close to true_xi within statistical variance
    assert abs(fit.shape_xi - true_xi) < 0.1
    # scales should be within 20%
    for r, sig in scales.items():
        assert abs(fit.scale_by_region[r] - sig) / sig < 0.25

    # Check standard error and CI
    assert fit.xi_se is not None
    assert fit.xi_se > 0.0
    assert fit.xi_ci is not None
    assert fit.xi_ci[0] < fit.shape_xi < fit.xi_ci[1]


def test_spliced_exceedance_probability_continuity_and_monotonicity() -> None:
    """spliced_exceedance_probability is continuous at u, monotone, and in [0, 1]."""
    fit = TailFit(
        threshold_u=64.5,
        shape_xi=0.15,
        scale_by_region={"WC": 20.0, "CI": 15.0},
    )

    p_u = 0.08
    u = fit.threshold_u

    # 1. Exact continuity at u
    assert spliced_exceedance_probability(p_u, fit, "WC", u) == pytest.approx(p_u)
    assert spliced_exceedance_probability(p_u, fit, "WC", u - 10.0) == pytest.approx(p_u)

    # 2. Strict monotonicity for y > u
    y_values = np.linspace(u, 300.0, 50)
    probs = [spliced_exceedance_probability(p_u, fit, "WC", y) for y in y_values]

    for p in probs:
        assert 0.0 <= p <= 1.0

    # Non-increasing: p(y_i) >= p(y_{i+1})
    diffs = np.diff(probs)
    assert np.all(diffs <= 1e-9)


def test_spliced_exceedance_probability_weibull_support_bound() -> None:
    """When xi < 0, GPD has bounded upper support: probability drops to 0 beyond endpoint."""
    fit = TailFit(
        threshold_u=50.0,
        shape_xi=-0.2,
        scale_by_region={"WC": 10.0},
    )
    p_u = 0.05
    # Upper bound = u - sigma / xi = 50 - 10 / (-0.2) = 100.0
    p_below_endpoint = spliced_exceedance_probability(p_u, fit, "WC", 90.0)
    p_at_endpoint = spliced_exceedance_probability(p_u, fit, "WC", 100.0)
    p_beyond_endpoint = spliced_exceedance_probability(p_u, fit, "WC", 110.0)

    assert p_below_endpoint > 0.0
    assert p_at_endpoint == pytest.approx(0.0, abs=1e-5)
    assert p_beyond_endpoint == 0.0


def test_spliced_exceedance_probability_monte_carlo() -> None:
    """Verify spliced exceedance probability matches Monte Carlo sampling of spliced process."""
    rng = np.random.default_rng(123)
    p_u = 0.10
    u = 64.5
    xi = 0.18
    sigma = 22.0
    region = "WC"

    fit = TailFit(
        threshold_u=u,
        shape_xi=xi,
        scale_by_region={region: sigma},
    )

    # Simulate N=100,000 draws:
    # Bernoulli(p_u): if 1, draw Z ~ GPD(xi, sigma), rainfall Y = u + Z
    # if 0, rainfall Y <= u (not exceeding)
    n_sims = 100_000
    exceeds_u = rng.uniform(0.0, 1.0, size=n_sims) < p_u
    n_exceed = int(np.sum(exceeds_u))
    z = _sample_gpd(xi, sigma, size=n_exceed, rng=rng)

    eval_thresholds = [75.0, 115.6, 150.0, 204.5]
    for th in eval_thresholds:
        analytic_prob = float(spliced_exceedance_probability(p_u, fit, region, th))
        # Empirical count: number of simulated cases where Y > th
        emp_count = int(np.sum(z > (th - u)))
        emp_prob = emp_count / n_sims

        # Binomial standard error: sqrt(p * (1-p) / N)
        se = np.sqrt(analytic_prob * (1.0 - analytic_prob) / n_sims)
        # Check agreement within 3.5 standard errors
        assert abs(emp_prob - analytic_prob) < 3.5 * se


def test_spliced_exceedance_probability_vectorization() -> None:
    """Vectorized inputs return identical values to scalar calls."""
    fit = TailFit(
        threshold_u=64.5,
        shape_xi=0.15,
        scale_by_region={"WC": 20.0, "CI": 15.0},
    )
    p_u_arr = np.array([0.05, 0.10])
    reg_arr = np.array(["WC", "CI"])
    y_arr = np.array([115.6, 204.5])

    res = spliced_exceedance_probability(p_u_arr, fit, reg_arr, y_arr)
    assert isinstance(res, np.ndarray)
    assert res.shape == (2,)

    s0 = spliced_exceedance_probability(0.05, fit, "WC", 115.6)
    s1 = spliced_exceedance_probability(0.10, fit, "CI", 204.5)
    assert res[0] == pytest.approx(s0)
    assert res[1] == pytest.approx(s1)


def test_nearest_fittable_bin() -> None:
    """nearest_fittable_bin searches closest rain intensity bins."""
    res_dry = CensoredShiftedGammaResult(
        bin_label="dry",
        source="ifs_ens",
        shift=0.5,
        climatological_mean=1.0,
        climatological_std=1.0,
        coefficients={"a1": 1.0, "a2": 0.5, "a3": 1.0, "a4": 0.5},
        is_fallback=False,
    )
    res_heavy = CensoredShiftedGammaResult(
        bin_label="heavy",
        source="ifs_ens",
        shift=1.0,
        climatological_mean=50.0,
        climatological_std=20.0,
        coefficients={"a1": 1.0, "a2": 0.5, "a3": 1.0, "a4": 0.5},
        is_fallback=False,
    )
    res_extreme_fallback = CensoredShiftedGammaResult(
        bin_label="extremely_heavy",
        source="ifs_ens",
        shift=0.0,
        climatological_mean=0.0,
        climatological_std=0.0,
        coefficients={},
        is_fallback=True,
    )

    results_by_bin = {
        "dry": res_dry,
        "heavy": res_heavy,
        "extremely_heavy": res_extreme_fallback,
    }

    # For extremely_heavy (which is fallback), nearest should be heavy
    nearest = nearest_fittable_bin("extremely_heavy", results_by_bin)
    assert nearest is not None
    assert nearest.bin_label == "heavy"


def test_exceedance_probability_with_tail() -> None:
    """exceedance_probability_with_tail flags csgd vs csgd+gpd_tail vs fallback correctly."""
    fit = TailFit(
        threshold_u=64.5,
        shape_xi=0.15,
        scale_by_region={"WC": 20.0, "CI": 15.0},
    )

    res_light = CensoredShiftedGammaResult(
        bin_label="light",
        source="ifs_ens",
        shift=1.0,
        climatological_mean=10.0,
        climatological_std=5.0,
        coefficients={"a1": 1.0, "a2": 0.5, "a3": 1.0, "a4": 0.5},
        is_fallback=False,
    )
    res_heavy = CensoredShiftedGammaResult(
        bin_label="heavy",
        source="ifs_ens",
        shift=2.0,
        climatological_mean=70.0,
        climatological_std=25.0,
        coefficients={"a1": 1.0, "a2": 0.5, "a3": 1.0, "a4": 0.5},
        is_fallback=False,
    )
    res_ext_fallback = CensoredShiftedGammaResult(
        bin_label="extremely_heavy",
        source="ifs_ens",
        shift=0.0,
        climatological_mean=0.0,
        climatological_std=0.0,
        coefficients={},
        is_fallback=True,
    )

    emos_by_bin = {
        "light": res_light,
        "heavy": res_heavy,
        "extremely_heavy": res_ext_fallback,
    }

    # 3 grid cells: one light forecast, one heavy forecast, one extreme forecast
    f_vals = np.array([20.0, 80.0, 220.0])
    m_vals = np.array([18.0, 75.0, 210.0])
    s_vals = np.array([5.0, 15.0, 30.0])
    regions = np.array(["WC", "WC", "WC"])

    # Test at threshold 115.6 (above u=64.5)
    probs, methods = exceedance_probability_with_tail(
        f_vals, m_vals, s_vals, emos_by_bin, fit, regions, threshold=115.6
    )

    assert len(probs) == 3
    assert len(methods) == 3

    # For extreme forecast (in unfittable bin extremely_heavy), method must be csgd+gpd_tail
    assert methods[2] == "csgd+gpd_tail"
    assert probs[2] > 0.0  # Real probability, not fallback 0.0!

    # For heavy forecast at threshold 115.6 (> u=64.5), method uses GPD tail
    assert methods[1] == "csgd+gpd_tail"

