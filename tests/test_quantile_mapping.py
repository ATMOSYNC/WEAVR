"""Unit tests for weavr.quantile_mapping (Step 11)."""

import numpy as np
import pytest
import xarray as xr

from weavr.quantile_mapping import (
    apply_quantile_map,
    apply_regional_quantile_maps,
    fit_quantile_map,
    fit_regional_quantile_maps,
)


class TestQuantileMapBasic:
    def test_identity_when_distributions_match(self):
        rng = np.random.default_rng(42)
        data = rng.exponential(scale=10.0, size=5000)

        qm = fit_quantile_map(data, data)
        # Apply to new test values
        test_vals = np.array([0.5, 2.0, 10.0, 25.0, 50.0])
        corrected = apply_quantile_map(qm, test_vals)

        np.testing.assert_allclose(corrected, test_vals, rtol=0.03)

    def test_shifted_gamma_distribution_recovered(self):
        rng = np.random.default_rng(123)
        # Forecast underestimates: Gamma(shape=2, scale=3) -> mean 6
        fcst_train = rng.gamma(shape=2.0, scale=3.0, size=10000)
        # Observed is larger: Gamma(shape=2, scale=6) -> mean 12
        obs_train = rng.gamma(shape=2.0, scale=6.0, size=10000)

        qm = fit_quantile_map(fcst_train, obs_train)

        # Applying QM to median of forecast should yield approx median of obs
        fcst_median = np.median(fcst_train)
        obs_median = np.median(obs_train)

        mapped_median = float(apply_quantile_map(qm, float(fcst_median)))
        assert mapped_median == pytest.approx(obs_median, rel=0.05)

    def test_dry_fraction_matching_eliminates_drizzle(self):
        # Forecast has 10% dry, but observed has 40% dry
        fcst = np.concatenate([np.zeros(100), np.linspace(0.01, 10.0, 900)])
        obs = np.concatenate([np.zeros(400), np.linspace(0.1, 20.0, 600)])

        qm = fit_quantile_map(fcst, obs, wet_threshold=0.1)
        assert qm.obs_dry_fraction == pytest.approx(0.40, abs=0.01)

        # Values in the lower 40% of forecast should now be mapped to 0.0
        low_vals = np.array([0.05, 0.1, 0.5])
        corrected = apply_quantile_map(qm, low_vals)
        assert np.all(corrected == 0.0)

    def test_never_negative(self):
        qm = fit_quantile_map(np.array([1.0, 2.0, 5.0]), np.array([0.0, 1.0, 4.0]))
        test_inputs = np.array([-5.0, 0.0, 0.1, 1.0])
        corrected = apply_quantile_map(qm, test_inputs)
        assert np.all(corrected >= 0.0)

    def test_monotonicity_preserved(self):
        rng = np.random.default_rng(999)
        fcst_train = rng.gamma(shape=1.5, scale=4.0, size=2000)
        obs_train = rng.gamma(shape=2.0, scale=5.0, size=2000)

        qm = fit_quantile_map(fcst_train, obs_train)

        test_sorted = np.linspace(0.0, 100.0, 500)
        corrected = apply_quantile_map(qm, test_sorted)

        assert np.all(np.diff(corrected) >= 0.0)

    def test_upper_extrapolation_additive_vs_ratio(self):
        fcst_train = np.array([0.0, 10.0, 20.0])
        obs_train = np.array([0.0, 15.0, 30.0])

        qm_add = fit_quantile_map(fcst_train, obs_train, extrapolation_rule="constant_additive")
        qm_rat = fit_quantile_map(fcst_train, obs_train, extrapolation_rule="ratio")

        # Test value beyond max (20.0): 40.0
        # Additive: 40 + (30 - 20) = 50
        # Ratio: 40 * (30 / 20) = 60
        res_add = float(apply_quantile_map(qm_add, 40.0))
        res_rat = float(apply_quantile_map(qm_rat, 40.0))

        assert res_add == pytest.approx(50.0)
        assert res_rat == pytest.approx(60.0)


class TestRegionalQuantileMapping:
    def test_fits_and_applies_by_region(self):
        # 2 samples, 2x2 grid
        regions_da = xr.DataArray(
            [["north", "north"], ["south", "south"]],
            dims=["latitude", "longitude"],
            coords={"latitude": [0, 1], "longitude": [0, 1]},
        )

        fcst_da = xr.DataArray(
            np.ones((2, 2, 2)) * 10.0,
            dims=["sample", "latitude", "longitude"],
            coords={"sample": [0, 1], "latitude": [0, 1], "longitude": [0, 1]},
            name="precip",
        )

        # Observations: north has 20 mm, south has 40 mm
        obs_vals = np.array([
            [[20.0, 20.0], [40.0, 40.0]],
            [[20.0, 20.0], [40.0, 40.0]],
        ])
        obs_da = xr.DataArray(
            obs_vals,
            dims=["sample", "latitude", "longitude"],
            coords={"sample": [0, 1], "latitude": [0, 1], "longitude": [0, 1]},
        )

        reg_maps = fit_regional_quantile_maps(fcst_da, obs_da, regions_da)
        assert "north" in reg_maps
        assert "south" in reg_maps

        corrected = apply_regional_quantile_maps(reg_maps, fcst_da, regions_da)
        assert isinstance(corrected, xr.DataArray)
        assert corrected.dims == fcst_da.dims
        # North cells (lat 0) should be mapped close to 20
        np.testing.assert_allclose(corrected.values[:, 0, :], 20.0, rtol=0.05)
        # South cells (lat 1) should be mapped close to 40
        np.testing.assert_allclose(corrected.values[:, 1, :], 40.0, rtol=0.05)
