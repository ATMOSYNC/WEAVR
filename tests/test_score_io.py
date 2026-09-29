import numpy as np
import pandas as pd
import pytest
import xarray as xr

from weavr import verify as V
from weavr.score_io import per_day_scores, read_per_day_scores, write_per_day_scores


def _forecast_and_obs(n_days=6, n_lat=4, n_lon=5, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-07-01", periods=n_days, freq="D")
    coords = {
        "sample": dates,
        "latitude": np.arange(n_lat, dtype=float),
        "longitude": np.arange(n_lon, dtype=float),
    }
    obs = xr.DataArray(
        rng.uniform(0, 120, size=(n_days, n_lat, n_lon)), coords=coords, dims=list(coords)
    )
    forecast = obs + xr.DataArray(
        rng.normal(0, 10, size=(n_days, n_lat, n_lon)), coords=coords, dims=list(coords)
    )
    return forecast, obs


class TestPerDayScores:
    def test_one_row_per_day_with_the_right_dates(self):
        forecast, obs = _forecast_and_obs(n_days=6)

        frame = per_day_scores(forecast, obs)

        assert len(frame) == 6
        assert list(frame["date"]) == list(pd.DatetimeIndex(obs["sample"].values))
        assert set(frame["fold"]) == {"test"}

    def test_mse_is_the_spatial_mean_squared_error_for_that_day(self):
        forecast, obs = _forecast_and_obs()

        frame = per_day_scores(forecast, obs)

        expected = float(((forecast - obs) ** 2).isel(sample=2).mean())
        assert frame["mse_mm2"].iloc[2] == pytest.approx(expected)

    def test_writes_mse_not_rmse(self):
        # Writing daily RMSE would make sqrt(mean(.)) impossible later and
        # bias every bootstrapped RMSE low (Jensen's inequality).
        forecast, obs = _forecast_and_obs()

        frame = per_day_scores(forecast, obs)

        day_rmse = float(V.rmse(forecast.isel(sample=0), obs.isel(sample=0)))
        assert frame["mse_mm2"].iloc[0] == pytest.approx(day_rmse**2)
        assert frame["mse_mm2"].iloc[0] != pytest.approx(day_rmse)

    def test_records_counts_not_ratios_at_every_threshold(self):
        forecast, obs = _forecast_and_obs()

        frame = per_day_scores(forecast, obs)

        for threshold in V.IMD_RAIN_THRESHOLDS_MM:
            for name in ("hits", "misses", "false_alarms", "correct_negatives"):
                assert f"{name}_{threshold}" in frame.columns
            assert f"csi_{threshold}" not in frame.columns

    def test_counts_sum_to_the_cells_scored_each_day(self):
        forecast, obs = _forecast_and_obs(n_lat=4, n_lon=5)

        frame = per_day_scores(forecast, obs)

        total = (
            frame["hits_7.5"]
            + frame["misses_7.5"]
            + frame["false_alarms_7.5"]
            + frame["correct_negatives_7.5"]
        )
        assert (total == 20).all()

    def test_counts_rebuild_the_aggregate_csi_when_pooled_over_days(self):
        # This is the property the scorecard depends on: a categorical score
        # over a set of days must be computable from the summed counts, not
        # from the mean of daily scores.
        forecast, obs = _forecast_and_obs(n_days=8)
        threshold = 7.5

        frame = per_day_scores(forecast, obs)
        hits = frame[f"hits_{threshold}"].sum()
        misses = frame[f"misses_{threshold}"].sum()
        false_alarms = frame[f"false_alarms_{threshold}"].sum()
        pooled_csi = hits / (hits + misses + false_alarms)

        expected = float(
            V.contingency_scores(forecast, obs, thresholds=(threshold,))[threshold]["csi"]
        )
        assert pooled_csi == pytest.approx(expected)

    def test_ensemble_adds_crps_and_twcrps(self):
        forecast, obs = _forecast_and_obs()
        rng = np.random.default_rng(1)
        ensemble = xr.DataArray(
            rng.uniform(0, 120, size=(*forecast.shape, 4)),
            coords={**forecast.coords, "member": np.arange(4)},
            dims=[*forecast.dims, "member"],
        )

        frame = per_day_scores(forecast, obs, ensemble=ensemble)

        assert "crps_mm" in frame.columns
        assert "twcrps_64.5_mm" in frame.columns
        expected = float(V.crps(ensemble.isel(sample=0), obs.isel(sample=0)))
        assert frame["crps_mm"].iloc[0] == pytest.approx(expected)

    def test_probabilities_add_brier_columns(self):
        forecast, obs = _forecast_and_obs()
        prob = xr.DataArray(
            np.full(forecast.shape, 0.3), coords=forecast.coords, dims=forecast.dims
        )

        frame = per_day_scores(forecast, obs, probabilities={64.5: prob})

        assert "brier_64.5" in frame.columns
        expected = float(
            V.brier_score(
                prob.isel(sample=0), (obs >= 64.5).astype(float).isel(sample=0)
            )
        )
        assert frame["brier_64.5"].iloc[0] == pytest.approx(expected)

    def test_per_cell_scores_are_averaged_not_recomputed(self):
        # The route for EMOS/BMA, which already return per-cell CRPS.
        forecast, obs = _forecast_and_obs()
        per_cell = xr.DataArray(
            np.arange(forecast.size, dtype=float).reshape(forecast.shape),
            coords=forecast.coords,
            dims=forecast.dims,
        )

        frame = per_day_scores(forecast, obs, per_cell_scores={"crps_mm": per_cell})

        assert frame["crps_mm"].iloc[3] == pytest.approx(float(per_cell.isel(sample=3).mean()))

    def test_n_cells_counts_only_cells_finite_in_both(self):
        forecast, obs = _forecast_and_obs(n_lat=2, n_lon=2)
        obs = obs.copy()
        obs[0, 0, 0] = np.nan

        frame = per_day_scores(forecast, obs)

        assert frame["n_cells"].iloc[0] == pytest.approx(3.0)
        assert frame["n_cells"].iloc[1] == pytest.approx(4.0)

    def test_fold_is_recorded_for_leave_one_year_out(self):
        forecast, obs = _forecast_and_obs()

        frame = per_day_scores(forecast, obs, fold="2018")

        assert set(frame["fold"]) == {"2018"}


class TestWriteAndReadPerDayScores:
    def test_round_trips_through_the_expected_filename(self, tmp_path):
        forecast, obs = _forecast_and_obs()
        frame = per_day_scores(forecast, obs)

        path = write_per_day_scores("tier0", 24, frame, out_dir=tmp_path)

        assert path == tmp_path / "per_day" / "tier0__lead24.csv"
        assert path.is_file()

    def test_read_recovers_method_and_lead_from_the_filename(self, tmp_path):
        forecast, obs = _forecast_and_obs()
        frame = per_day_scores(forecast, obs)
        write_per_day_scores("tier2_emos_graphcast", 72, frame, out_dir=tmp_path)
        write_per_day_scores("tier0", 24, frame, out_dir=tmp_path)

        combined = read_per_day_scores(out_dir=tmp_path)

        assert set(combined["method"]) == {"tier2_emos_graphcast", "tier0"}
        assert set(combined["lead_hours"]) == {24, 72}
        # A method name containing "_" must survive the round trip.
        subset = combined[combined["method"] == "tier2_emos_graphcast"]
        assert set(subset["lead_hours"]) == {72}

    def test_values_survive_the_round_trip(self, tmp_path):
        forecast, obs = _forecast_and_obs()
        frame = per_day_scores(forecast, obs)
        write_per_day_scores("tier0", 24, frame, out_dir=tmp_path)

        combined = read_per_day_scores(out_dir=tmp_path)

        np.testing.assert_allclose(
            combined["mse_mm2"].to_numpy(), frame["mse_mm2"].to_numpy()
        )
        assert list(combined["date"]) == list(frame["date"])

    def test_missing_directory_returns_an_empty_frame(self, tmp_path):
        assert read_per_day_scores(out_dir=tmp_path / "nothing").empty

    def test_rejects_an_empty_method_name(self, tmp_path):
        forecast, obs = _forecast_and_obs()
        with pytest.raises(ValueError, match="non-empty"):
            write_per_day_scores("", 24, per_day_scores(forecast, obs), out_dir=tmp_path)
