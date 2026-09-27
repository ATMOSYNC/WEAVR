import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_tier2_hierarchical_baseline import (  # noqa: E402
    _clip_negative_precip,
    _csgd_predictive_mean,
    _ordered_values,
    _pool_rmse_from_mse,
    _pool_weighted_mean,
    score_emos_source,
)

from weavr.emos import CensoredShiftedGammaResult  # noqa: E402


class TestClipNegativePrecip:
    def test_negative_values_floored_to_zero(self):
        da = xr.DataArray([-0.5, 0.0, 3.2, -0.01])

        result = _clip_negative_precip(da)

        np.testing.assert_array_equal(result.values, [0.0, 0.0, 3.2, 0.0])

    def test_positive_values_untouched(self):
        da = xr.DataArray([1.0, 2.5, 100.0])

        result = _clip_negative_precip(da)

        np.testing.assert_array_equal(result.values, [1.0, 2.5, 100.0])


class TestOrderedValues:
    def test_reorders_to_sample_latitude_longitude(self):
        da = xr.DataArray(
            np.arange(2 * 3 * 4).reshape(3, 4, 2),
            dims=["latitude", "longitude", "sample"],
            coords={"latitude": np.arange(3), "longitude": np.arange(4), "sample": np.arange(2)},
        )

        result = _ordered_values(da)

        assert result.shape == (2, 3, 4)
        expected = da.transpose("sample", "latitude", "longitude").values
        np.testing.assert_array_equal(result, expected)

    def test_keeps_member_dim_last_when_requested(self):
        da = xr.DataArray(
            np.zeros((5, 2, 3, 4)),
            dims=["member", "sample", "latitude", "longitude"],
            coords={
                "member": np.arange(5),
                "sample": np.arange(2),
                "latitude": np.arange(3),
                "longitude": np.arange(4),
            },
        )

        result = _ordered_values(da, member_dim="member")

        assert result.shape == (2, 3, 4, 5)


class TestPoolWeightedMean:
    def test_reconstructs_exact_overall_mean(self):
        # Group A: 2 points averaging 10; Group B: 3 points averaging 20.
        # True pooled mean = (2*10 + 3*20)/5 = 16.
        values = np.array([10.0, 20.0])
        weights = np.array([2.0, 3.0])

        assert _pool_weighted_mean(values, weights) == pytest.approx(16.0)

    def test_ignores_nan_and_zero_weight_groups(self):
        values = np.array([10.0, np.nan, 30.0])
        weights = np.array([1.0, 5.0, 0.0])

        assert _pool_weighted_mean(values, weights) == pytest.approx(10.0)

    def test_all_invalid_returns_nan(self):
        values = np.array([np.nan, np.nan])
        weights = np.array([0.0, 0.0])

        assert np.isnan(_pool_weighted_mean(values, weights))


class TestPoolRmseFromMse:
    def test_pools_mse_before_sqrt(self):
        # Group A: mse=4 (rmse=2), n=1; Group B: mse=16 (rmse=4), n=3.
        # Correct pooled mse = (1*4 + 3*16)/4 = 13 -> rmse = sqrt(13),
        # which is NOT the same as averaging 2 and 4 directly.
        mse_values = np.array([4.0, 16.0])
        weights = np.array([1.0, 3.0])

        result = _pool_rmse_from_mse(mse_values, weights)

        assert result == pytest.approx(np.sqrt(13.0))
        assert result != pytest.approx((2.0 * 1 + 4.0 * 3) / 4)


class TestCsgdPredictiveMean:
    def test_recovers_known_mean_for_a_near_deterministic_fit(self):
        # A tiny scale relative to location approximates a point mass at
        # (location + shift) -- the Monte Carlo predictive mean should
        # land close to that value.
        location = np.array([10.0])
        scale = np.array([0.01])
        shift = np.array([-0.5])
        rng = np.random.default_rng(0)

        result = _csgd_predictive_mean(location, scale, shift, rng, n_samples=2000)

        assert result[0] == pytest.approx(9.5, abs=0.1)


class TestScoreEmosSource:
    def _forecast_and_obs(self):
        n_sample, n_lat, n_lon = 4, 1, 2
        coords = {
            "sample": np.arange(n_sample),
            "latitude": np.arange(n_lat, dtype=float),
            "longitude": np.arange(n_lon, dtype=float),
        }
        # Every cell forecasts a mean of 5.0 (member values 4, 5, 6);
        # obs matches closely so the fitted-looking result should score
        # near-zero CRPS and near-zero bias.
        member_values = np.stack(
            [np.full((n_sample, n_lat, n_lon), v) for v in (4.0, 5.0, 6.0)], axis=-1
        )
        forecast = xr.DataArray(
            member_values, dims=["sample", "latitude", "longitude", "member"], coords=coords
        )
        obs = xr.DataArray(
            np.full((n_sample, n_lat, n_lon), 5.0), dims=["sample", "latitude", "longitude"],
            coords=coords,
        )
        rain_bin_labels = xr.DataArray(
            np.full((n_sample, n_lat, n_lon), "light", dtype=object),
            dims=["sample", "latitude", "longitude"],
            coords=coords,
        )
        return forecast, obs, rain_bin_labels

    def test_known_good_fit_scores_low_crps_and_bias(self):
        forecast, obs, rain_bin_labels = self._forecast_and_obs()
        # A result whose location tracks the ensemble mean almost exactly,
        # with a small scale -- should predict close to 5.0 everywhere.
        result = CensoredShiftedGammaResult(
            bin_label="light",
            source="synthetic",
            shift=-0.01,
            climatological_mean=5.0,
            climatological_std=1.0,
            coefficients={"a1": 0.0, "a2": 1.0, "a3": 0.05, "a4": 0.0},
        )
        test_mask = np.ones(forecast.sizes["sample"], dtype=bool)
        rng = np.random.default_rng(0)

        per_bin = score_emos_source(
            {"light": result}, forecast, obs, rain_bin_labels, test_mask, rng, n_samples=2000
        )

        expected_n_cells = forecast.sizes["sample"] * forecast.sizes["longitude"]
        assert per_bin["light"]["n_test_cells"] == expected_n_cells
        assert per_bin["light"]["crps_mm"] < 1.0
        assert abs(per_bin["light"]["bias_mm"]) < 1.0
        assert per_bin["light"]["is_fallback"] is False

    def test_bin_with_zero_test_cells_reports_nan_not_a_crash(self):
        forecast, obs, rain_bin_labels = self._forecast_and_obs()
        result = CensoredShiftedGammaResult(
            bin_label="heavy",
            source="synthetic",
            shift=0.0,
            climatological_mean=0.0,
            climatological_std=1e-6,
            is_fallback=True,
            reason="never fittable",
        )
        test_mask = np.ones(forecast.sizes["sample"], dtype=bool)
        rng = np.random.default_rng(0)

        per_bin = score_emos_source(
            {"heavy": result}, forecast, obs, rain_bin_labels, test_mask, rng
        )

        assert per_bin["heavy"]["n_test_cells"] == 0
        assert np.isnan(per_bin["heavy"]["crps_mm"])
        assert per_bin["heavy"]["is_fallback"] is True
