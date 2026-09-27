import numpy as np
import pytest
import xarray as xr

from weavr.weighting import (
    MIN_TRAIN_SAMPLES,
    clip_and_renormalize,
    fit_region_weights,
    fit_weights_least_squares,
)


def _single_region_labels(lat, lon, label="R1") -> xr.DataArray:
    return xr.DataArray(
        np.full((len(lat), len(lon)), label),
        coords={"latitude": lat, "longitude": lon},
        dims=["latitude", "longitude"],
    )


def _forecast_da(values: np.ndarray, sample_coords) -> xr.DataArray:
    return xr.DataArray(
        values,
        coords={"sample": sample_coords},
        dims=["sample", "latitude", "longitude"],
    )


class TestFitWeightsLeastSquares:
    def test_recovers_known_weights_exactly_for_noiseless_data(self):
        rng = np.random.default_rng(0)
        n_points, n_sources = 200, 3
        design = rng.normal(size=(n_points, n_sources))
        true_weights = np.array([0.2, 0.5, 0.3])
        target = design @ true_weights

        fitted = fit_weights_least_squares(design, target)

        np.testing.assert_allclose(fitted, true_weights, atol=1e-8)


class TestClipAndRenormalize:
    def test_all_positive_weights_pass_through_renormalized(self):
        result = clip_and_renormalize(np.array([1.0, 1.0, 2.0]))
        np.testing.assert_allclose(result, [0.25, 0.25, 0.5])

    def test_negative_weight_is_clipped_then_remaining_renormalized_to_sum_one(self):
        # raw weights [-0.5, 1.0, 1.5] -> clipped [0, 1.0, 1.5] -> sum 2.5
        result = clip_and_renormalize(np.array([-0.5, 1.0, 1.5]))
        np.testing.assert_allclose(result, [0.0, 0.4, 0.6])
        assert result.sum() == pytest.approx(1.0)

    def test_all_negative_weights_return_none(self):
        assert clip_and_renormalize(np.array([-1.0, -2.0, -0.5])) is None


class TestFitRegionWeightsBetterPredictor:
    def test_clearly_better_source_gets_higher_weight(self):
        lat = np.array([10.0, 10.25])
        lon = np.array([70.0, 70.25])
        samples = np.arange(20)
        region_labels = _single_region_labels(lat, lon)

        rng = np.random.default_rng(1)
        obs_values = rng.normal(loc=5.0, scale=2.0, size=(20, 2, 2))
        # source "good" tracks obs almost exactly; source "bad" is unrelated noise.
        good = obs_values + rng.normal(scale=0.05, size=(20, 2, 2))
        bad = rng.normal(loc=5.0, scale=2.0, size=(20, 2, 2))

        obs = _forecast_da(obs_values, samples)
        forecasts = {"good": _forecast_da(good, samples), "bad": _forecast_da(bad, samples)}
        train_mask = np.ones(20, dtype=bool)

        results = fit_region_weights(forecasts, obs, region_labels, train_mask)

        result = results["R1"]
        assert not result.is_fallback
        assert result.weights["good"] > result.weights["bad"]
        assert result.weights["good"] > 0.9


class TestFitRegionWeightsClippingAndRenormalization:
    def test_source_with_negative_naive_fit_is_clipped_and_weights_still_sum_to_one(self):
        lat = np.array([10.0, 10.25])
        lon = np.array([70.0, 70.25])
        samples = np.arange(30)
        region_labels = _single_region_labels(lat, lon)

        rng = np.random.default_rng(2)
        # "anti" is obs's near-exact negative -- its naive OLS coefficient
        # against obs alone would come out negative.
        obs_values = rng.normal(loc=0.0, scale=3.0, size=(30, 2, 2))
        good = obs_values + rng.normal(scale=0.05, size=(30, 2, 2))
        anti = -obs_values + rng.normal(scale=0.05, size=(30, 2, 2))

        obs = _forecast_da(obs_values, samples)
        forecasts = {"good": _forecast_da(good, samples), "anti": _forecast_da(anti, samples)}
        train_mask = np.ones(30, dtype=bool)

        results = fit_region_weights(forecasts, obs, region_labels, train_mask)

        result = results["R1"]
        assert not result.is_fallback
        assert result.weights["anti"] == pytest.approx(0.0, abs=1e-6)
        assert sum(result.weights.values()) == pytest.approx(1.0)


class TestFitRegionWeightsDegenerateFallback:
    def test_too_few_train_samples_triggers_documented_fallback(self):
        lat = np.array([10.0, 10.25])
        lon = np.array([70.0, 70.25])
        samples = np.arange(5)
        region_labels = _single_region_labels(lat, lon)

        rng = np.random.default_rng(3)
        obs_values = rng.normal(size=(5, 2, 2))
        a = rng.normal(size=(5, 2, 2))
        b = rng.normal(size=(5, 2, 2))

        obs = _forecast_da(obs_values, samples)
        forecasts = {"a": _forecast_da(a, samples), "b": _forecast_da(b, samples)}
        # Only one sample marked train -- below MIN_TRAIN_SAMPLES.
        train_mask = np.array([True, False, False, False, False])
        assert MIN_TRAIN_SAMPLES > 1

        results = fit_region_weights(forecasts, obs, region_labels, train_mask)

        result = results["R1"]
        assert result.is_fallback
        assert result.reason is not None
        assert "train sample" in result.reason
        assert result.weights == {"a": 0.5, "b": 0.5}

    def test_rank_deficient_region_triggers_documented_fallback(self):
        # Two sources that are literally identical everywhere in this
        # region -- the design matrix has no way to distinguish them.
        lat = np.array([10.0, 10.25])
        lon = np.array([70.0, 70.25])
        samples = np.arange(10)
        region_labels = _single_region_labels(lat, lon)

        rng = np.random.default_rng(4)
        obs_values = rng.normal(size=(10, 2, 2))
        identical = rng.normal(size=(10, 2, 2))

        obs = _forecast_da(obs_values, samples)
        forecasts = {
            "source1": _forecast_da(identical, samples),
            "source2": _forecast_da(identical.copy(), samples),
        }
        train_mask = np.ones(10, dtype=bool)

        results = fit_region_weights(forecasts, obs, region_labels, train_mask)

        result = results["R1"]
        assert result.is_fallback
        assert "rank-deficient" in result.reason
        assert result.weights == {"source1": 0.5, "source2": 0.5}

    def test_nan_gridpoints_outside_a_region_do_not_count_toward_its_sample_requirement(self):
        # A region occupying only part of the grid must still fit correctly
        # using just its own gridpoints, not be starved by NaN cells
        # belonging to a different region.
        lat = np.array([10.0, 10.25, 20.0, 20.25])
        lon = np.array([70.0, 70.25])
        samples = np.arange(20)
        labels = np.array(
            [["R1", "R1"], ["R1", "R1"], ["R2", "R2"], ["R2", "R2"]]
        )
        region_labels = xr.DataArray(
            labels, coords={"latitude": lat, "longitude": lon}, dims=["latitude", "longitude"]
        )

        rng = np.random.default_rng(5)
        obs_values = rng.normal(loc=5.0, size=(20, 4, 2))
        good = obs_values + rng.normal(scale=0.05, size=(20, 4, 2))
        bad = rng.normal(loc=5.0, size=(20, 4, 2))

        obs = _forecast_da(obs_values, samples)
        forecasts = {"good": _forecast_da(good, samples), "bad": _forecast_da(bad, samples)}
        train_mask = np.ones(20, dtype=bool)

        results = fit_region_weights(forecasts, obs, region_labels, train_mask)

        assert not results["R1"].is_fallback
        assert not results["R2"].is_fallback
        assert results["R1"].weights["good"] > 0.9
        assert results["R2"].weights["good"] > 0.9
