import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_tier3_regime_conditioned_baseline import (  # noqa: E402
    _domain_summary_from_stats,
    load_monsoon_phase_for_samples,
    score_point_forecast,
    score_point_forecast_by_bin,
)


def _grid_da(values: np.ndarray, samples) -> xr.DataArray:
    lat = np.array([10.0, 10.25])
    lon = np.array([70.0, 70.25])
    return xr.DataArray(
        values,
        coords={"sample": samples, "latitude": lat, "longitude": lon},
        dims=["sample", "latitude", "longitude"],
    )


class TestScorePointForecast:
    def test_crps_equals_absolute_error_for_a_point_forecast(self):
        samples = np.arange(2)
        forecast = _grid_da(np.array([[[3.0, 5.0], [1.0, 0.0]], [[2.0, 2.0], [2.0, 2.0]]]), samples)
        obs = _grid_da(np.array([[[1.0, 5.0], [1.0, 4.0]], [[0.0, 6.0], [4.0, 2.0]]]), samples)

        stats = score_point_forecast(forecast, obs)

        expected_abs_errors = np.abs(forecast.values - obs.values)
        assert stats["n_test_cells"] == expected_abs_errors.size
        assert stats["crps_mm"] == np.mean(expected_abs_errors)
        assert stats["bias_mm"] == np.mean(forecast.values - obs.values)
        assert stats["mse_mm2"] == np.mean((forecast.values - obs.values) ** 2)

    def test_all_nan_forecast_returns_zero_cells_not_a_crash(self):
        samples = np.arange(1)
        forecast = _grid_da(np.full((1, 2, 2), np.nan), samples)
        obs = _grid_da(np.zeros((1, 2, 2)), samples)

        stats = score_point_forecast(forecast, obs)

        assert stats["n_test_cells"] == 0
        assert np.isnan(stats["crps_mm"])


class TestScorePointForecastByBin:
    def test_splits_scoring_by_the_supplied_bin_label(self):
        samples = np.arange(1)
        forecast = _grid_da(np.array([[[1.0, 100.0], [1.0, 100.0]]]), samples)
        obs = _grid_da(np.array([[[0.0, 90.0], [0.0, 90.0]]]), samples)
        bin_labels = xr.DataArray(
            np.array([[["dry", "heavy"], ["dry", "heavy"]]]),
            coords=forecast.coords,
            dims=forecast.dims,
        )

        per_bin = score_point_forecast_by_bin(forecast, obs, bin_labels)

        assert per_bin["dry"]["n_test_cells"] == 2
        assert per_bin["dry"]["crps_mm"] == 1.0
        assert per_bin["heavy"]["n_test_cells"] == 2
        assert per_bin["heavy"]["crps_mm"] == 10.0
        assert per_bin["light"]["n_test_cells"] == 0


class TestDomainSummaryFromStats:
    def test_computes_rmse_as_sqrt_of_mse(self):
        stats = {"crps_mm": 1.5, "mse_mm2": 4.0, "bias_mm": -0.5, "n_test_cells": 10}

        summary = _domain_summary_from_stats(stats, "regime")

        assert summary["regime_crps_mm"] == 1.5
        assert summary["regime_rmse_mm"] == 2.0
        assert summary["regime_bias_mm"] == -0.5
        assert summary["regime_n_test_cells"] == 10

    def test_nan_mse_produces_nan_rmse_not_a_crash(self):
        stats = {
            "crps_mm": float("nan"),
            "mse_mm2": float("nan"),
            "bias_mm": float("nan"),
            "n_test_cells": 0,
        }

        summary = _domain_summary_from_stats(stats, "regime")

        assert np.isnan(summary["regime_rmse_mm"])


class TestLoadMonsoonPhaseForSamples:
    def test_reindexes_full_season_classification_onto_the_aligned_sample_coordinate(self):
        # A minimal real-shaped case: core zone must have >=1 gridpoint
        # (see weavr.regimes.EmptyCoreZoneError), so use a 3-point lat grid
        # spanning 18-28N, matching tests/test_regimes.py's own fixture
        # pattern.
        lat = np.linspace(6.5, 38.5, 3)
        lon = np.linspace(66.5, 100.0, 3)
        times = pd.date_range("2020-06-01", periods=10, freq="D")
        clim_times = pd.date_range("2010-06-01", periods=10, freq="D")

        rng = np.random.default_rng(0)
        rain_values = 5.0 + rng.normal(scale=1.0, size=(10, 3, 3))
        clim_values = 5.0 + rng.normal(scale=1.0, size=(10, 3, 3))

        def _da(values, ts):
            return xr.DataArray(
                values, coords={"time": ts, "latitude": lat, "longitude": lon},
                dims=["time", "latitude", "longitude"],
            )

        obs = xr.Dataset({"rain": _da(rain_values, times)})
        climatology = xr.Dataset({"rain": _da(clim_values, clim_times)})

        # Only ask for a subset of the real days -- the aligned "sample"
        # coordinate a lead's own forecasts actually have.
        sample_values = times[[2, 5, 7]].values

        result = load_monsoon_phase_for_samples(obs, climatology, sample_values)

        assert result.dims == ("sample",)
        np.testing.assert_array_equal(result["sample"].values, sample_values)
