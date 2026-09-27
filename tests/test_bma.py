import numpy as np
import pytest
import xarray as xr

from weavr.bma import (
    MIN_TRAIN_DAYS_PER_BIN,
    BmaFitResult,
    fit_hierarchical_bma,
    score_bma,
)


def _coords(n_sample, n_lat, n_lon):
    return {
        "sample": np.arange(n_sample),
        "latitude": np.arange(n_lat, dtype=float),
        "longitude": np.arange(n_lon, dtype=float),
    }


def _da(values, coords):
    dims = ["sample", "latitude", "longitude"]
    if values.ndim == 4:
        dims = [*dims, "member"]
    return xr.DataArray(values, coords=coords, dims=dims)


def _single_region_and_bin(n_sample, n_lat, n_lon, coords, bin_label="light"):
    region_labels = xr.DataArray(
        np.full((n_lat, n_lon), "R1"),
        coords={"latitude": coords["latitude"], "longitude": coords["longitude"]},
        dims=["latitude", "longitude"],
    )
    rain_bin_labels = xr.DataArray(
        np.full((n_sample, n_lat, n_lon), bin_label, dtype=object),
        coords=coords,
        dims=["sample", "latitude", "longitude"],
    )
    return region_labels, rain_bin_labels


class TestFitHierarchicalBmaRecoversBetterComponent:
    def test_clearly_better_source_gets_higher_weight(self):
        rng = np.random.default_rng(0)
        n_sample, n_lat, n_lon, n_member = 40, 3, 3, 8
        coords = _coords(n_sample, n_lat, n_lon)

        # Real precipitation-like obs: mostly small positive values, no
        # zeros, so both components' wet-regression path gets exercised.
        obs_values = rng.uniform(5.0, 30.0, size=(n_sample, n_lat, n_lon))

        # "good" ensemble tracks obs almost exactly across members; "bad"
        # is unrelated noise -- EM should assign "good" far more weight.
        good_members = obs_values[..., None] + rng.normal(
            scale=0.2, size=(n_sample, n_lat, n_lon, n_member)
        )
        bad_members = rng.uniform(5.0, 30.0, size=(n_sample, n_lat, n_lon, n_member))

        obs = _da(obs_values, coords)
        forecasts = {
            "good": _da(np.clip(good_members, 0.0, None), coords),
            "bad": _da(np.clip(bad_members, 0.0, None), coords),
        }
        region_labels, rain_bin_labels = _single_region_and_bin(n_sample, n_lat, n_lon, coords)
        train_mask = np.ones(n_sample, dtype=bool)

        results = fit_hierarchical_bma(
            forecasts, obs, rain_bin_labels, region_labels, train_mask
        )
        result = results[("light", "R1")]

        assert not result.is_fallback
        assert result.weights["good"] > result.weights["bad"]
        assert result.weights["good"] > 0.7
        assert sum(result.weights.values()) == pytest.approx(1.0)


class TestMixtureWeightsSumToOne:
    def test_weights_sum_to_one_for_every_fitted_cell(self):
        rng = np.random.default_rng(1)
        n_sample, n_lat, n_lon, n_member = 30, 2, 2, 6
        coords = _coords(n_sample, n_lat, n_lon)

        obs_values = rng.uniform(1.0, 20.0, size=(n_sample, n_lat, n_lon))
        members = obs_values[..., None] + rng.normal(
            scale=1.0, size=(n_sample, n_lat, n_lon, n_member)
        )
        deterministic = obs_values + rng.normal(scale=1.5, size=(n_sample, n_lat, n_lon))

        obs = _da(obs_values, coords)
        forecasts = {
            "ensemble_source": _da(np.clip(members, 0.0, None), coords),
            "hres": _da(np.clip(deterministic, 0.0, None), coords),
        }
        region_labels, rain_bin_labels = _single_region_and_bin(n_sample, n_lat, n_lon, coords)
        train_mask = np.ones(n_sample, dtype=bool)

        results = fit_hierarchical_bma(
            forecasts, obs, rain_bin_labels, region_labels, train_mask
        )
        result = results[("light", "R1")]

        assert not result.is_fallback
        assert sum(result.weights.values()) == pytest.approx(1.0)
        assert result.components["hres"].route == "kernel_dressing"
        assert result.components["ensemble_source"].route == "ensemble_dressing"


class TestSparseCellFallback:
    def test_cell_with_too_few_train_days_falls_back(self):
        n_sample, n_lat, n_lon, n_member = 10, 2, 2, 4
        coords = _coords(n_sample, n_lat, n_lon)
        rng = np.random.default_rng(2)

        obs_values = rng.uniform(1.0, 5.0, size=(n_sample, n_lat, n_lon))
        members = obs_values[..., None] + rng.normal(
            scale=0.5, size=(n_sample, n_lat, n_lon, n_member)
        )

        obs = _da(obs_values, coords)
        forecasts = {"good": _da(np.clip(members, 0.0, None), coords)}

        region_labels, _ = _single_region_and_bin(n_sample, n_lat, n_lon, coords)
        # Only 3 of 10 train days land in the "heavy" bin -- below
        # MIN_TRAIN_DAYS_PER_BIN (5).
        labels = np.full((n_sample, n_lat, n_lon), "light", dtype=object)
        labels[:3, 0, 0] = "heavy"
        rain_bin_labels = xr.DataArray(
            labels, coords=coords, dims=["sample", "latitude", "longitude"]
        )

        train_mask = np.ones(n_sample, dtype=bool)

        results = fit_hierarchical_bma(
            forecasts, obs, rain_bin_labels, region_labels, train_mask
        )
        result = results[("heavy", "R1")]

        assert result.is_fallback
        assert result.n_train_days == 3
        assert f"need >= {MIN_TRAIN_DAYS_PER_BIN}" in result.reason
        assert sum(result.weights.values()) == pytest.approx(1.0)


class TestScoreBma:
    def test_fallback_result_scores_against_a_point_mass_at_zero(self):
        n_lat, n_lon = 2, 2
        coords = {
            "latitude": np.arange(n_lat, dtype=float),
            "longitude": np.arange(n_lon, dtype=float),
        }
        obs = xr.DataArray(
            np.full((n_lat, n_lon), 3.0), coords=coords, dims=["latitude", "longitude"]
        )
        forecast_mean = {
            "good": xr.DataArray(
                np.full((n_lat, n_lon), 3.0), coords=coords, dims=["latitude", "longitude"]
            )
        }
        forecast_spread = {"good": None}

        result = BmaFitResult(
            bin_label="light", region="R1", weights={"good": 1.0}, is_fallback=True
        )

        scores = score_bma(
            result,
            forecast_mean,
            forecast_spread,
            obs,
            rng=np.random.default_rng(0),
            n_samples=50,
        )

        # A point-mass-at-zero forecast's CRPS against obs=3 is exactly |3-0|=3.
        assert np.allclose(scores.values, 3.0, atol=1e-6)

    def test_fitted_result_scores_lower_than_a_far_off_fallback(self):
        rng = np.random.default_rng(3)
        n_sample, n_lat, n_lon, n_member = 30, 2, 2, 6
        coords = _coords(n_sample, n_lat, n_lon)

        obs_values = rng.uniform(5.0, 15.0, size=(n_sample, n_lat, n_lon))
        members = obs_values[..., None] + rng.normal(
            scale=0.5, size=(n_sample, n_lat, n_lon, n_member)
        )

        obs = _da(obs_values, coords)
        forecasts = {"good": _da(np.clip(members, 0.0, None), coords)}
        region_labels, rain_bin_labels = _single_region_and_bin(n_sample, n_lat, n_lon, coords)
        train_mask = np.ones(n_sample, dtype=bool)

        results = fit_hierarchical_bma(
            forecasts, obs, rain_bin_labels, region_labels, train_mask
        )
        result = results[("light", "R1")]
        assert not result.is_fallback

        test_coords = {
            "latitude": coords["latitude"],
            "longitude": coords["longitude"],
        }
        test_obs = xr.DataArray(
            obs_values[0], coords=test_coords, dims=["latitude", "longitude"]
        )
        forecast_mean = {
            "good": xr.DataArray(
                members[0].mean(axis=-1), coords=test_coords, dims=["latitude", "longitude"]
            )
        }
        forecast_spread = {
            "good": xr.DataArray(
                members[0].std(axis=-1, ddof=1), coords=test_coords, dims=["latitude", "longitude"]
            )
        }

        fitted_scores = score_bma(
            result, forecast_mean, forecast_spread, test_obs, rng=np.random.default_rng(4)
        )

        fallback_result = BmaFitResult(
            bin_label="light", region="R1", weights={"good": 1.0}, is_fallback=True
        )
        fallback_scores = score_bma(
            fallback_result,
            forecast_mean,
            forecast_spread,
            test_obs,
            rng=np.random.default_rng(4),
        )

        assert float(fitted_scores.mean()) < float(fallback_scores.mean())
