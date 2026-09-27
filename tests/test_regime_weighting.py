import numpy as np
import xarray as xr

from weavr.regime_weighting import (
    MIN_TRAIN_SAMPLES,
    blend_with_regime_weights,
    build_regime_weight_series,
    fit_regime_weights,
)


def _forecast_da(values: np.ndarray, sample_coords) -> xr.DataArray:
    return xr.DataArray(
        values,
        coords={"sample": sample_coords},
        dims=["sample", "latitude", "longitude"],
    )


def _regime_labels(labels: np.ndarray, sample_coords) -> xr.DataArray:
    return xr.DataArray(labels, coords={"sample": sample_coords}, dims=["sample"], name="regime")


class TestFitRegimeWeightsBetterPredictorPerCategory:
    def test_a_source_only_good_in_one_regime_category_gets_higher_weight_there(self):
        # "good_in_active" tracks obs almost exactly on active days, but is
        # unrelated noise on neutral days; "good_in_neutral" is the reverse.
        # A regime-stratified fit should recover this per-category split --
        # a single pooled (non-stratified) fit could not, since averaged
        # across both categories neither source looks clearly better.
        samples = np.arange(40)
        labels = np.array(["active"] * 20 + [""] * 20)
        regime_labels = _regime_labels(labels, samples)

        rng = np.random.default_rng(0)
        obs_values = rng.normal(loc=5.0, scale=2.0, size=(40, 2, 2))

        good_in_active = obs_values.copy()
        good_in_active[:20] += rng.normal(scale=0.05, size=(20, 2, 2))
        good_in_active[20:] = rng.normal(loc=5.0, scale=2.0, size=(20, 2, 2))

        good_in_neutral = obs_values.copy()
        good_in_neutral[20:] += rng.normal(scale=0.05, size=(20, 2, 2))
        good_in_neutral[:20] = rng.normal(loc=5.0, scale=2.0, size=(20, 2, 2))

        obs = _forecast_da(obs_values, samples)
        forecasts = {
            "good_in_active": _forecast_da(good_in_active, samples),
            "good_in_neutral": _forecast_da(good_in_neutral, samples),
        }
        train_mask = np.ones(40, dtype=bool)

        results = fit_regime_weights(forecasts, obs, regime_labels, train_mask)

        assert not results["active"].is_fallback
        assert not results[""].is_fallback
        assert results["active"].weights["good_in_active"] > 0.9
        assert results[""].weights["good_in_neutral"] > 0.9


class TestFitRegimeWeightsDegenerateFallback:
    def test_sparse_regime_category_triggers_documented_fallback(self):
        samples = np.arange(10)
        # Only 1 "active" day among 10 -- below MIN_TRAIN_SAMPLES.
        labels = np.array(["active"] + [""] * 9)
        regime_labels = _regime_labels(labels, samples)

        rng = np.random.default_rng(1)
        obs_values = rng.normal(size=(10, 2, 2))
        a = rng.normal(size=(10, 2, 2))
        b = rng.normal(size=(10, 2, 2))

        obs = _forecast_da(obs_values, samples)
        forecasts = {"a": _forecast_da(a, samples), "b": _forecast_da(b, samples)}
        train_mask = np.ones(10, dtype=bool)
        assert MIN_TRAIN_SAMPLES > 1

        results = fit_regime_weights(forecasts, obs, regime_labels, train_mask)

        result = results["active"]
        assert result.is_fallback
        assert result.reason is not None
        assert "regime category" in result.reason
        assert result.weights == {"a": 0.5, "b": 0.5}
        assert not results[""].is_fallback

    def test_rank_deficient_category_triggers_documented_fallback(self):
        samples = np.arange(10)
        labels = np.array(["active"] * 10)
        regime_labels = _regime_labels(labels, samples)

        rng = np.random.default_rng(2)
        obs_values = rng.normal(size=(10, 2, 2))
        identical = rng.normal(size=(10, 2, 2))

        obs = _forecast_da(obs_values, samples)
        forecasts = {
            "source1": _forecast_da(identical, samples),
            "source2": _forecast_da(identical.copy(), samples),
        }
        train_mask = np.ones(10, dtype=bool)

        results = fit_regime_weights(forecasts, obs, regime_labels, train_mask)

        result = results["active"]
        assert result.is_fallback
        assert "rank-deficient" in result.reason
        assert result.weights == {"source1": 0.5, "source2": 0.5}


class TestBuildRegimeWeightSeriesAndBlend:
    def test_blend_uses_the_correct_category_weight_per_day(self):
        samples = np.arange(4)
        labels = np.array(["active", "active", "", ""])
        regime_labels = _regime_labels(labels, samples)

        results = {
            "active": type(
                "R", (), {"regime": "active", "weights": {"a": 1.0, "b": 0.0}}
            )(),
            "": type("R", (), {"regime": "", "weights": {"a": 0.0, "b": 1.0}})(),
        }

        weight_series = build_regime_weight_series(results, regime_labels, ["a", "b"])

        np.testing.assert_array_equal(weight_series["a"].values, [1.0, 1.0, 0.0, 0.0])
        np.testing.assert_array_equal(weight_series["b"].values, [0.0, 0.0, 1.0, 1.0])

        lat = np.array([10.0])
        lon = np.array([70.0])
        coords = {"sample": samples, "latitude": lat, "longitude": lon}
        dims = ["sample", "latitude", "longitude"]
        a_values = np.array([[[1.0]], [[2.0]], [[3.0]], [[4.0]]])
        b_values = np.array([[[10.0]], [[20.0]], [[30.0]], [[40.0]]])
        forecasts = {
            "a": xr.DataArray(a_values, coords=coords, dims=dims),
            "b": xr.DataArray(b_values, coords=coords, dims=dims),
        }

        blend = blend_with_regime_weights(forecasts, weight_series)

        # Active days (0, 1) take "a" entirely; neutral days (2, 3) take "b" entirely.
        blended_point = blend.isel(latitude=0, longitude=0).values
        np.testing.assert_allclose(blended_point, [1.0, 2.0, 30.0, 40.0])
