import numpy as np
import pandas as pd
import pytest
import xarray as xr

from weavr.verify import (
    acc,
    bias,
    brier_score,
    calibrated_spread_skill_ratio,
    contingency_scores,
    crps,
    ensemble_spread,
    fss,
    rmse,
    seeps,
    spread_skill_ratio,
)


def _da(values, dims="x"):
    return xr.DataArray(np.asarray(values, dtype=float), dims=dims)


class TestRmse:
    def test_perfect_forecast_is_zero(self):
        obs = _da([1.0, 2.0, 3.0, 4.0])
        assert float(rmse(obs, obs, dim="x")) == 0.0

    def test_known_constant_offset(self):
        obs = _da([0.0, 0.0, 0.0, 0.0])
        forecast = _da([3.0, -3.0, 3.0, -3.0])
        # every error is +/-3 -> RMSE is exactly 3
        assert float(rmse(forecast, obs, dim="x")) == pytest.approx(3.0)


class TestBias:
    def test_zero_for_identical(self):
        obs = _da([1.0, 2.0, 3.0])
        assert float(bias(obs, obs, dim="x")) == 0.0

    def test_positive_when_forecast_over_predicts(self):
        obs = _da([1.0, 2.0, 3.0])
        forecast = obs + 2.0
        assert float(bias(forecast, obs, dim="x")) == pytest.approx(2.0)

    def test_negative_when_forecast_under_predicts(self):
        obs = _da([1.0, 2.0, 3.0])
        forecast = obs - 3.0
        assert float(bias(forecast, obs, dim="x")) == pytest.approx(-3.0)


class TestAcc:
    def test_perfect_correlation(self):
        climatology = _da([0.0, 0.0, 0.0, 0.0])
        obs = _da([1.0, -2.0, 3.0, -4.0])
        assert float(acc(obs, obs, climatology, dim="x")) == pytest.approx(1.0)

    def test_anti_correlated(self):
        climatology = _da([0.0, 0.0, 0.0, 0.0])
        obs = _da([1.0, -2.0, 3.0, -4.0])
        forecast = -obs
        assert float(acc(forecast, obs, climatology, dim="x")) == pytest.approx(-1.0)


class TestCrps:
    def test_deterministic_ensemble_equals_absolute_error(self):
        # every member identical -> CRPS collapses to |member - obs|
        obs = _da([0.0])
        forecast = xr.DataArray([[2.0, 2.0, 2.0]], dims=["x", "member"])
        result = crps(forecast, obs, dim="x")
        assert float(result) == pytest.approx(2.0)

    def test_zero_for_perfect_deterministic_match(self):
        obs = _da([5.0])
        forecast = xr.DataArray([[5.0, 5.0]], dims=["x", "member"])
        assert float(crps(forecast, obs, dim="x")) == pytest.approx(0.0)


class TestBrierScore:
    def test_zero_for_perfect_probability_forecast(self):
        obs_binary = _da([1.0, 0.0, 1.0, 0.0])
        forecast = _da([1.0, 0.0, 1.0, 0.0])
        assert float(brier_score(forecast, obs_binary, dim="x")) == pytest.approx(0.0)

    def test_known_value_for_partial_confidence(self):
        obs_binary = _da([1.0])
        forecast = _da([0.5])
        assert float(brier_score(forecast, obs_binary, dim="x")) == pytest.approx(0.25)


class TestCalibratedSpreadSkillRatio:
    def test_matches_known_values(self):
        # sqrt((M+1)/M) for M=9,8,6 -- the real member counts this project's
        # leads produce (docs/baseline-store.md) -- not a flat 1.0.
        assert calibrated_spread_skill_ratio(9) == pytest.approx(np.sqrt(10 / 9))
        assert calibrated_spread_skill_ratio(8) == pytest.approx(np.sqrt(9 / 8))
        assert calibrated_spread_skill_ratio(6) == pytest.approx(np.sqrt(7 / 6))

    def test_approaches_one_as_members_grow(self):
        assert calibrated_spread_skill_ratio(10_000) == pytest.approx(1.0, abs=1e-3)

    def test_is_never_exactly_one_for_a_finite_ensemble(self):
        assert calibrated_spread_skill_ratio(9) > 1.0


class TestEnsembleSpread:
    def test_zero_spread_ensemble(self):
        ensemble = xr.DataArray(np.full((2, 5), 3.0), dims=["sample", "member"])
        assert float(ensemble_spread(ensemble, dim="sample")) == pytest.approx(0.0)

    def test_matches_ddof1_variance_directly(self):
        rng = np.random.default_rng(0)
        values = rng.normal(size=(1, 20))
        ensemble = xr.DataArray(values, dims=["sample", "member"])
        expected = float(np.sqrt(values.var(axis=1, ddof=1).mean()))
        assert float(ensemble_spread(ensemble, dim="sample")) == pytest.approx(expected)

    def test_ignores_nan_members_rather_than_treating_them_as_zero_spread(self):
        # A short lagged window pads missing members with NaN
        # (src/weavr/ensemble.py) -- those must be excluded from the
        # variance calculation, not silently pull the spread toward zero.
        with_nan = xr.DataArray(
            np.array([[1.0, 2.0, 3.0, np.nan, np.nan]]), dims=["sample", "member"]
        )
        without_nan = xr.DataArray(np.array([[1.0, 2.0, 3.0]]), dims=["sample", "member"])

        assert float(ensemble_spread(with_nan, dim="sample")) == pytest.approx(
            float(ensemble_spread(without_nan, dim="sample"))
        )


class TestSpreadSkillRatio:
    def test_calibrated_ensemble_scores_near_the_calibrated_ratio(self):
        # obs and members drawn iid from the same distribution -- the
        # definition of a perfectly reliable/calibrated ensemble -- should
        # score a ratio close to calibrated_spread_skill_ratio(M), not 1.0
        # and not far above it.
        rng = np.random.default_rng(0)
        m_members = 9
        n_samples = 20_000
        draws = rng.normal(size=(n_samples, m_members + 1))
        obs = xr.DataArray(draws[:, 0], dims=["sample"])
        ensemble = xr.DataArray(draws[:, 1:], dims=["sample", "member"])

        ratio = float(spread_skill_ratio(ensemble, obs, dim="sample"))
        calibrated = calibrated_spread_skill_ratio(m_members)

        assert ratio == pytest.approx(calibrated, rel=0.05)

    def test_under_dispersive_ensemble_scores_well_above_the_calibrated_ratio(self):
        # Members clustered tightly (small spread) while obs varies far more
        # than the ensemble does -- the real error is much larger than the
        # ensemble's own spread, the definition of under-dispersion.
        rng = np.random.default_rng(1)
        n_samples = 2_000
        obs = xr.DataArray(rng.normal(scale=10.0, size=n_samples), dims=["sample"])
        ensemble = xr.DataArray(
            rng.normal(loc=0.0, scale=0.1, size=(n_samples, 9)), dims=["sample", "member"]
        )

        ratio = float(spread_skill_ratio(ensemble, obs, dim="sample"))
        calibrated = calibrated_spread_skill_ratio(9)

        assert ratio > 3 * calibrated


def _climatology_for_seeps(n_dry=150, n_wet=150, dry_value=0.0, dry_threshold=1.0):
    """A synthetic climatology with an exact dry fraction and a large, unique wet range.

    n_dry days at `dry_value` (below `dry_threshold`) and n_wet days spanning
    1..n_wet (all above `dry_threshold`) give p1 = n_dry/(n_dry+n_wet) exactly,
    with the light/heavy split determined by the 2/3 quantile of the wet
    values -- enough resolution that the quantile isn't degenerate.
    """
    values = np.concatenate([np.full(n_dry, dry_value), np.arange(1, n_wet + 1, dtype=float)])
    times = pd.date_range("2000-01-01", periods=len(values))
    return xr.DataArray(values, coords={"time": times}, dims=["time"])


class TestSeeps:
    def test_perfect_forecast_is_zero(self):
        climatology = _climatology_for_seeps()
        obs = xr.DataArray([0.0, 50.0, 140.0], dims=["sample"])
        result = seeps(obs, obs, climatology, dim="sample")
        assert float(result) == pytest.approx(0.0)

    @pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
    def test_degenerate_all_dry_climatology_is_nan(self):
        # An all-dry climatology has no wet days at all, so the light/heavy
        # quantile is computed over an empty (all-NaN) slice -- numpy warns
        # about that, which is expected here, not a bug to chase.
        climatology = xr.DataArray(
            np.zeros(30), coords={"time": pd.date_range("2000-01-01", periods=30)}, dims=["time"]
        )
        obs = xr.DataArray([0.0, 0.0], dims=["sample"])
        forecast = xr.DataArray([0.0, 5.0], dims=["sample"])
        result = seeps(forecast, obs, climatology, dim="sample")
        assert np.isnan(float(result))

    def test_matrix_is_equitable(self):
        """A forecast drawn at the climatological category frequencies should
        score the same (=2) expectation regardless of the true category --
        the defining property of an "equitable" score (Rodwell et al. 2010).
        Checked directly against the public seeps() function, not just the
        formula in isolation, and at an asymmetric p1 (0.5) vs p3 (~1/6)
        specifically because that asymmetry is what earlier caught an error
        in a first draft of the scoring matrix (a naively-remembered
        1/(3*p3) term that is only equitable when p1 happens to equal p3).
        """
        climatology = _climatology_for_seeps(n_dry=150, n_wet=150)
        dry_value = 0.0
        light_value = 50.0  # below the 2/3 quantile of 1..150 (~100.3)
        heavy_value = 140.0  # above it
        category_values = {0: dry_value, 1: light_value, 2: heavy_value}

        # Empirical climatological frequencies, from the same categorize
        # logic seeps() itself uses -- exact fractions, not assumed 2:1.
        p1 = 150 / 300
        p3 = (1 - p1) / 3
        p2 = 1 - p1 - p3
        category_probs = {0: p1, 1: p2, 2: p3}

        n = 3000
        forecast_values = []
        for cat, prob in category_probs.items():
            forecast_values += [category_values[cat]] * round(n * prob)

        for true_cat, true_value in category_values.items():
            obs = xr.DataArray(np.full(len(forecast_values), true_value), dims=["sample"])
            forecast = xr.DataArray(forecast_values, dims=["sample"])
            result = float(seeps(forecast, obs, climatology, dim="sample"))
            assert result == pytest.approx(2.0, abs=0.05), f"true_cat={true_cat}"


class TestFss:
    def test_perfect_match_is_one(self):
        field = xr.DataArray(
            [[0.0, 10.0, 0.0], [10.0, 10.0, 0.0], [0.0, 0.0, 0.0]],
            dims=["latitude", "longitude"],
        )
        result = fss(field, field, threshold=5.0, neighborhood_size=3)
        assert float(result) == pytest.approx(1.0)

    def test_no_event_anywhere_is_nan(self):
        field = xr.DataArray(np.zeros((3, 3)), dims=["latitude", "longitude"])
        result = fss(field, field, threshold=5.0, neighborhood_size=3)
        assert np.isnan(float(result))


class TestContingencyScores:
    def test_perfect_forecast(self):
        rng = np.random.default_rng(0)
        obs = xr.DataArray(rng.random((10, 10)) * 20, dims=["x", "y"])
        result = contingency_scores(obs, obs, thresholds=[5.0], dim=["x", "y"])
        scores = result[5.0]
        assert float(scores["pod"]) == pytest.approx(1.0)
        assert float(scores["far"]) == pytest.approx(0.0)
        assert float(scores["csi"]) == pytest.approx(1.0)
        assert float(scores["ets"]) == pytest.approx(1.0)

    def test_forecast_never_predicts_event_gives_pod_zero(self):
        obs = xr.DataArray(np.full((5, 5), 10.0), dims=["x", "y"])
        forecast = xr.DataArray(np.zeros((5, 5)), dims=["x", "y"])
        result = contingency_scores(forecast, obs, thresholds=[5.0], dim=["x", "y"])
        scores = result[5.0]
        assert float(scores["pod"]) == pytest.approx(0.0)
        assert float(scores["csi"]) == pytest.approx(0.0)
        # zero forecast-yes events -> FAR is 0/0, must be NaN not silently 0.
        assert np.isnan(float(scores["far"]))

    def test_all_dry_gives_nan_across_the_board(self):
        obs = xr.DataArray(np.zeros((5, 5)), dims=["x", "y"])
        forecast = xr.DataArray(np.zeros((5, 5)), dims=["x", "y"])
        result = contingency_scores(forecast, obs, thresholds=[5.0], dim=["x", "y"])
        scores = result[5.0]
        for name in ("pod", "far", "csi", "ets"):
            assert np.isnan(float(scores[name])), name

    def test_default_thresholds_are_imd_categories(self):
        obs = xr.DataArray(np.array([[0.0, 10.0], [100.0, 250.0]]), dims=["x", "y"])
        result = contingency_scores(obs, obs, dim=["x", "y"])
        assert set(result.keys()) == {7.5, 64.5, 115.6, 204.5}
