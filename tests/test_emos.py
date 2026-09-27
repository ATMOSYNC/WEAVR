import numpy as np
import pytest
import xarray as xr
from scipy import integrate
from scipy.stats import gamma as gamma_dist

from weavr.emos import (
    MIN_TRAIN_DAYS_PER_BIN,
    CensoredShiftedGammaResult,
    csgd_crps,
    fit_emos_csg,
    predict_csgd_params,
    score_csgd,
)


def _csgd_cdf(t: np.ndarray, mean: float, std: float, shift: float) -> np.ndarray:
    kappa = mean**2 / std**2
    theta = std**2 / mean
    t = np.asarray(t, dtype=float)
    return np.where(t < 0, 0.0, gamma_dist.cdf((t - shift) / theta, a=kappa))


def _numerical_crps(mean: float, std: float, shift: float, y: float) -> float:
    """Brute-force numerical integration of CRPS's raw definition
    (Scheuerer & Hamill 2015 eq. 9): integral of (F(t) - H(t-y))^2 dt over
    the real line, for y>=0. For t<0, F(t)=0 (left-censored) and H(t-y)=0
    (since y>=0), so the integrand is exactly zero there and the integral
    can start at 0 instead of -infinity; it is truncated far enough into
    the upper tail that the integrand is negligible.
    """
    upper = max(y, mean) + 50 * (std + 1)

    def integrand_below(t: float) -> float:
        return _csgd_cdf(np.array([t]), mean, std, shift)[0] ** 2

    def integrand_above(t: float) -> float:
        return (_csgd_cdf(np.array([t]), mean, std, shift)[0] - 1) ** 2

    below, _ = integrate.quad(integrand_below, 0.0, y, limit=200)
    above, _ = integrate.quad(integrand_above, y, upper, limit=200)
    return below + above


class TestCsgdCrpsMatchesNumericalIntegration:
    @pytest.mark.parametrize(
        "mean,std,shift,y",
        [
            (5.0, 3.0, -1.0, 0.0),
            (5.0, 3.0, -1.0, 2.0),
            (5.0, 3.0, -1.0, 10.0),
            (20.0, 15.0, -5.0, 0.0),
            (20.0, 15.0, -5.0, 30.0),
            (1.0, 1.0, -0.1, 0.5),
            (50.0, 10.0, -2.0, 60.0),
        ],
    )
    def test_closed_form_matches_brute_force_integral(self, mean, std, shift, y):
        closed_form = csgd_crps(mean, std, shift, y)
        numerical = _numerical_crps(mean, std, shift, y)

        assert closed_form == pytest.approx(numerical, abs=1e-3)

    def test_vectorized_over_arrays(self):
        mean = np.array([5.0, 20.0])
        std = np.array([3.0, 15.0])
        shift = np.array([-1.0, -5.0])
        y = np.array([2.0, 30.0])

        result = csgd_crps(mean, std, shift, y)

        assert result.shape == (2,)
        for i in range(2):
            assert result[i] == pytest.approx(
                _numerical_crps(mean[i], std[i], shift[i], y[i]), abs=1e-3
            )


class TestCsgdCrpsProperties:
    def test_crps_is_nonnegative(self):
        for mean, std, shift, y in [(5.0, 3.0, -1.0, 0.0), (5.0, 3.0, -1.0, 100.0)]:
            assert csgd_crps(mean, std, shift, y) >= 0.0

    def test_crps_increases_with_distance_from_a_point_mass_like_fit(self):
        # A very small std approximates a near-deterministic forecast at
        # `mean` -- CRPS should be small near `mean` and grow further away.
        mean, std, shift = 10.0, 0.1, -0.01
        near = csgd_crps(mean, std, shift, 10.0)
        far = csgd_crps(mean, std, shift, 50.0)
        assert far > near


class TestCsgdCrpsHandlesNonPositiveMean:
    # Regression test for a real bug caught running Phase 4 step 6 against
    # the real stores: ~1% of GraphCast's own real forecast cells carry
    # tiny negative numerical-noise artifacts (a known ML-weather-model
    # characteristic). A fitted regression's `location = a1 + a2*
    # ensemble_mean` can then come out non-positive, and `mean`'s kappa
    # numerator using the raw value while theta's denominator used a
    # clipped one produced a near-zero-shape, huge-scale gamma -- an
    # uncorrected CRPS of ~1165mm for this exact real case (obs=0). Fixed
    # by clipping `mean` once, consistently, before computing both kappa
    # and theta.
    def test_slightly_negative_mean_gives_a_sane_bounded_crps(self):
        # The exact real (mean, std, shift, y) values recovered from a
        # dry-bin GraphCast cell where the fitted regression's location
        # came out negative.
        mean, std, shift, y = -0.05115537274117088, 0.05879086349718349, -0.44782659545748243, 0.0

        result = csgd_crps(mean, std, shift, y)

        assert result >= 0.0
        assert result < 5.0

    def test_zero_mean_gives_a_sane_bounded_crps(self):
        result = csgd_crps(0.0, 0.06, -0.45, 0.0)

        assert result >= 0.0
        assert result < 5.0


def _synthetic_ensemble(rng: np.random.Generator, n_sample: int, n_space: int, n_member: int):
    times = np.arange(n_sample)
    lats = np.arange(n_space)
    lons = np.array([0])

    # Draw "true" precipitation from a known CSGD, then build an ensemble
    # whose mean/spread are linearly related to it, so fit_emos_csg has a
    # real, recoverable signal to find.
    true_mean, true_std, true_shift = 8.0, 5.0, -1.5
    kappa = true_mean**2 / true_std**2
    theta = true_std**2 / true_mean
    raw = gamma_dist.rvs(a=kappa, scale=theta, size=(n_sample, n_space), random_state=rng)
    obs_values = np.clip(raw + true_shift, 0.0, None)

    member_noise = rng.normal(scale=1.0, size=(n_sample, n_space, n_member))
    ensemble_values = obs_values[:, :, None] + member_noise
    ensemble_values = np.clip(ensemble_values, 0.0, None)

    ensemble = xr.DataArray(
        ensemble_values[:, :, None, :],
        dims=("sample", "latitude", "longitude", "member"),
        coords={"sample": times, "latitude": lats, "longitude": lons},
    )
    obs = xr.DataArray(
        obs_values[:, :, None],
        dims=("sample", "latitude", "longitude"),
        coords={"sample": times, "latitude": lats, "longitude": lons},
    )
    return ensemble, obs


class TestFitEmosCsgRecoversRealSignal:
    def test_fit_reduces_crps_relative_to_climatology_only(self):
        rng = np.random.default_rng(0)
        ensemble, obs = _synthetic_ensemble(rng, n_sample=40, n_space=6, n_member=10)

        # A single rain bin covering everything, so the whole synthetic
        # dataset is fit as one bin (isolates the regression fit itself
        # from rain_bins.py's own classification, tested separately).
        rain_bin_labels = xr.full_like(obs, "light", dtype=object)
        train_mask = np.ones(ensemble.sizes["sample"], dtype=bool)

        results = fit_emos_csg(
            ensemble,
            obs,
            rain_bin_labels,
            train_mask,
            source="synthetic",
        )
        result = results["light"]

        assert not result.is_fallback
        assert result.n_train_days == 40

        ensemble_mean = ensemble.mean(dim="member")
        ensemble_spread = ensemble.std(dim="member", ddof=1)
        fitted_scores = score_csgd(result, ensemble_mean, ensemble_spread, obs)

        # Compare against a naive "climatology only" predictive CSGD (the
        # fixed unconditional fit, ignoring the ensemble forecast) -- the
        # real conditional fit should do at least as well on its own
        # training data, since it strictly generalizes the intercept-only
        # climatology model (a2=a4=0 recovers it).
        climatology_only = CensoredShiftedGammaResult(
            bin_label="light",
            source="synthetic",
            shift=result.shift,
            climatological_mean=result.climatological_mean,
            climatological_std=result.climatological_std,
            coefficients={
                "a1": result.climatological_mean,
                "a2": 0.0,
                "a3": 1.0,
                "a4": 0.0,
            },
        )
        climatology_scores = score_csgd(
            climatology_only, ensemble_mean, ensemble_spread, obs
        )

        assert float(fitted_scores.mean()) <= float(climatology_scores.mean()) + 1e-6


class TestDegenerateAllDryFallback:
    def test_all_zero_obs_falls_back_rather_than_fitting(self):
        n_sample, n_space, n_member = 10, 4, 6
        times = np.arange(n_sample)
        lats = np.arange(n_space)
        lons = np.array([0])

        obs = xr.DataArray(
            np.zeros((n_sample, n_space, 1)),
            dims=("sample", "latitude", "longitude"),
            coords={"sample": times, "latitude": lats, "longitude": lons},
        )
        ensemble = xr.DataArray(
            np.zeros((n_sample, n_space, 1, n_member)),
            dims=("sample", "latitude", "longitude", "member"),
            coords={"sample": times, "latitude": lats, "longitude": lons},
        )
        rain_bin_labels = xr.full_like(obs, "dry", dtype=object)
        train_mask = np.ones(n_sample, dtype=bool)

        results = fit_emos_csg(ensemble, obs, rain_bin_labels, train_mask, source="synthetic")
        result = results["dry"]

        assert result.is_fallback
        assert result.reason is not None
        assert "probability of precipitation" in result.reason

        mean, std, shift = predict_csgd_params(result, np.array([0.0]), np.array([0.0]))
        assert mean[0] == 0.0
        assert shift[0] == 0.0


class TestSparseBinFallback:
    def test_bin_with_too_few_train_days_falls_back(self):
        n_sample, n_space, n_member = 10, 4, 6
        times = np.arange(n_sample)
        lats = np.arange(n_space)
        lons = np.array([0])
        rng = np.random.default_rng(1)

        obs_values = rng.uniform(1.0, 5.0, size=(n_sample, n_space, 1))
        ensemble_values = obs_values[..., None] + rng.normal(
            scale=0.5, size=(n_sample, n_space, 1, n_member)
        )

        obs = xr.DataArray(
            obs_values,
            dims=("sample", "latitude", "longitude"),
            coords={"sample": times, "latitude": lats, "longitude": lons},
        )
        ensemble = xr.DataArray(
            np.clip(ensemble_values, 0.0, None),
            dims=("sample", "latitude", "longitude", "member"),
            coords={"sample": times, "latitude": lats, "longitude": lons},
        )

        # Only 3 of the 10 train days contribute any cell to the "heavy"
        # bin -- below MIN_TRAIN_DAYS_PER_BIN (5) -- matching the real
        # sparse-bin scenario docs/phase4-data-and-combiner-scope.md found.
        labels = np.full((n_sample, n_space, 1), "light", dtype=object)
        labels[:3, 0, 0] = "heavy"
        rain_bin_labels = xr.DataArray(
            labels,
            dims=("sample", "latitude", "longitude"),
            coords={"sample": times, "latitude": lats, "longitude": lons},
        )
        train_mask = np.ones(n_sample, dtype=bool)

        results = fit_emos_csg(ensemble, obs, rain_bin_labels, train_mask, source="synthetic")
        result = results["heavy"]

        assert result.is_fallback
        assert result.n_train_days == 3
        assert f"need >= {MIN_TRAIN_DAYS_PER_BIN}" in result.reason


class TestPredictCsgdParams:
    def test_fallback_predicts_a_point_mass_at_zero(self):
        result = CensoredShiftedGammaResult(
            bin_label="extremely_heavy",
            source="synthetic",
            shift=0.0,
            climatological_mean=0.0,
            climatological_std=1e-6,
            is_fallback=True,
            reason="never fittable",
        )
        mean, std, shift = predict_csgd_params(
            result, np.array([100.0, 200.0]), np.array([10.0, 20.0])
        )
        np.testing.assert_array_equal(mean, [0.0, 0.0])
        np.testing.assert_array_equal(shift, [0.0, 0.0])
        assert np.all(std > 0)

    def test_fitted_result_uses_its_own_coefficients(self):
        result = CensoredShiftedGammaResult(
            bin_label="light",
            source="synthetic",
            shift=-1.0,
            climatological_mean=5.0,
            climatological_std=3.0,
            coefficients={"a1": 1.0, "a2": 2.0, "a3": 0.5, "a4": 0.1},
        )
        mean, std, shift = predict_csgd_params(
            result, np.array([2.0]), np.array([1.0])
        )
        expected_mean = 1.0 + 2.0 * 2.0
        expected_std = 0.5 * np.sqrt(expected_mean) + 0.1 * 1.0
        assert mean[0] == pytest.approx(expected_mean)
        assert std[0] == pytest.approx(expected_std)
        assert shift[0] == -1.0

    def test_negative_ensemble_mean_gives_a_positive_returned_location(self):
        # Regression test: a fitted regression's own a1/a2 coefficients
        # keep location positive for any *nonnegative* ensemble_mean, but
        # real forecast data can carry tiny negative numerical-noise
        # artifacts (see TestCsgdCrpsHandlesNonPositiveMean). The returned
        # `location` must stay positive and consistent with the `scale`
        # formula's own internal clipping, not silently negative.
        result = CensoredShiftedGammaResult(
            bin_label="dry",
            source="synthetic",
            shift=-0.1,
            climatological_mean=1.0,
            climatological_std=1.0,
            coefficients={"a1": 0.01, "a2": 1.0, "a3": 1.0, "a4": 0.0},
        )
        location, scale, shift = predict_csgd_params(
            result, np.array([-0.5]), np.array([0.1])
        )
        assert location[0] > 0.0
        assert np.isfinite(scale[0])
