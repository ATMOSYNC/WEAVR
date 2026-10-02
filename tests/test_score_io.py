from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from weavr import verify as V
from weavr.score_io import (
    PerDayScoreWriter,
    guard_result_overwrites,
    per_day_scores,
    read_per_day_scores,
    resolve_result_paths,
    write_per_day_scores,
    write_rows_csv,
)


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


class TestResolveResultPaths:
    """Every result CSV must follow --results-dir, not hardcode 'results/'.

    The bug these lock down: each output flag used to default to its own
    literal "results/..." path, so redirecting --out-csv and --results-dir to
    a scratch directory still wrote the per-bin and per-region CSVs over the
    committed results.
    """

    FILENAMES = {
        "out_csv": "tier2_hierarchical_baseline.csv",
        "bin_out_csv": "tier2_hierarchical_baseline_by_bin.csv",
        "region_out_csv": "tier2_hierarchical_baseline_by_region.csv",
    }

    def test_every_output_lands_in_the_given_directory(self, tmp_path):
        paths = resolve_result_paths(tmp_path, self.FILENAMES)
        assert set(paths) == set(self.FILENAMES)
        for dest, path in paths.items():
            assert Path(path).parent == tmp_path, dest
            assert Path(path).name == self.FILENAMES[dest]

    def test_an_explicit_path_wins_and_others_still_follow_results_dir(self, tmp_path):
        elsewhere = tmp_path / "elsewhere" / "custom.csv"
        paths = resolve_result_paths(
            tmp_path,
            self.FILENAMES,
            {"bin_out_csv": str(elsewhere), "out_csv": None},
        )
        assert paths["bin_out_csv"] == str(elsewhere)
        assert Path(paths["out_csv"]).parent == tmp_path
        assert Path(paths["region_out_csv"]).parent == tmp_path

    def test_nothing_resolves_under_a_bare_results_string(self, tmp_path):
        paths = resolve_result_paths(tmp_path, self.FILENAMES)
        assert not any(p.startswith("results/") for p in paths.values())


class TestGuardResultOverwrites:
    def test_is_silent_when_no_output_exists(self, tmp_path):
        guard_result_overwrites(
            [tmp_path / "a.csv", tmp_path / "nested" / "b.csv"]
        )

    def test_refuses_to_replace_an_existing_file(self, tmp_path):
        existing = tmp_path / "committed.csv"
        existing.write_text("reviewed numbers")
        with pytest.raises(SystemExit) as excinfo:
            guard_result_overwrites([existing, tmp_path / "absent.csv"])
        message = str(excinfo.value)
        assert "committed.csv" in message
        assert "--force" in message

    def test_names_every_offending_file(self, tmp_path):
        first, second = tmp_path / "a.csv", tmp_path / "b.csv"
        first.write_text("x")
        second.write_text("y")
        with pytest.raises(SystemExit) as excinfo:
            guard_result_overwrites([first, second, tmp_path / "absent.csv"])
        assert str(first) in str(excinfo.value)
        assert str(second) in str(excinfo.value)
        assert "absent.csv" not in str(excinfo.value)

    def test_force_allows_an_intentional_replacement(self, tmp_path):
        existing = tmp_path / "committed.csv"
        existing.write_text("reviewed numbers")
        guard_result_overwrites([existing], force=True)
        assert existing.read_text() == "reviewed numbers"

    def test_guard_never_deletes_or_truncates_what_it_protects(self, tmp_path):
        existing = tmp_path / "committed.csv"
        existing.write_text("reviewed numbers")
        with pytest.raises(SystemExit):
            guard_result_overwrites([existing])
        assert existing.read_text() == "reviewed numbers"


class TestPerDayScoreWriter:
    """Regression cover for the fold-overwrite that step 07 hit.

    `write_per_day_scores` names files by `(method, lead)`, so a LOYO runner
    that called it once per fold kept only the last fold. The aggregate CSVs
    stayed correct because folds were scored in memory, so nothing failed --
    the 2018 fold's days simply never reached the scorecard, and every paired
    CI built on them came back non-computable.
    """

    def _fold(self, dates, label):
        return pd.DataFrame(
            {
                "date": dates,
                "fold": label,
                "n_cells": 17_415,
                "mse_mm2": 1.0,
                "mae_mm": 0.9,
            }
        )

    def test_two_folds_survive_in_one_file(self, tmp_path):
        writer = PerDayScoreWriter(tmp_path)
        writer.add("tier0", 24, self._fold(pd.date_range("2018-07-01", periods=3), "2018"))
        writer.add("tier0", 24, self._fold(pd.date_range("2020-07-01", periods=3), "2020"))
        written = writer.flush()

        assert [p.name for p in written] == ["tier0__lead24.csv"]
        frame = pd.read_csv(written[0])
        assert len(frame) == 6
        assert sorted(frame["fold"].astype(str).unique()) == ["2018", "2020"]

    def test_direct_writes_in_a_loop_would_have_lost_a_fold(self, tmp_path):
        """Pins *why* the writer exists: the naive loop keeps one fold."""
        folds = [
            self._fold(pd.date_range("2018-07-01", periods=3), "2018"),
            self._fold(pd.date_range("2020-07-01", periods=3), "2020"),
        ]
        for frame in folds:
            write_per_day_scores("tier0", 24, frame, out_dir=tmp_path)

        frame = pd.read_csv(tmp_path / "per_day" / "tier0__lead24.csv")
        assert len(frame) == 3
        assert set(frame["fold"].astype(str)) == {"2020"}

    def test_add_after_flush_is_refused_rather_than_dropped(self, tmp_path):
        writer = PerDayScoreWriter(tmp_path)
        writer.add("tier0", 24, self._fold(pd.date_range("2018-07-01", periods=2), "2018"))
        writer.flush()
        with pytest.raises(RuntimeError):
            writer.add("tier0", 24, self._fold(pd.date_range("2020-07-01", periods=2), "2020"))

    def test_duplicate_dates_across_folds_are_rejected(self, tmp_path):
        """Overlapping folds would double-count days in the paired CI."""
        writer = PerDayScoreWriter(tmp_path)
        dates = pd.date_range("2018-07-01", periods=3)
        writer.add("tier0", 24, self._fold(dates, "2018"))
        writer.add("tier0", 24, self._fold(dates, "2020"))
        with pytest.raises(ValueError, match="duplicate dates"):
            writer.flush()

    def test_frame_without_a_fold_column_is_rejected(self, tmp_path):
        writer = PerDayScoreWriter(tmp_path)
        frame = self._fold(pd.date_range("2018-07-01", periods=2), "2018").drop(
            columns=["fold"]
        )
        with pytest.raises(ValueError, match="fold"):
            writer.add("tier0", 24, frame)

    def test_flush_is_a_noop_on_an_empty_run(self, tmp_path):
        assert PerDayScoreWriter(tmp_path).flush() == []


class TestWriteRowsCsv:
    """Regression: heterogeneous rows must not raise.

    A two-season Tier 2 run lost two hours of BMA scoring to
    `ValueError: dict contains fields not in fieldnames: 'rmse_mm'`. The
    per-fold rows carried no `rmse_mm`; the pooled rows appended later did;
    and the writer derived its fieldnames from `rows[0]` alone, so the
    by-bin CSV raised *after* every expensive number had been computed.
    """

    def test_later_row_with_extra_key_is_written_not_raised(self, tmp_path):
        rows = [
            {"lead_hours": 24, "fold": "2018", "crps_mm": 4.2, "mse_mm2": 30.1},
            {
                "lead_hours": 24,
                "fold": "pooled",
                "crps_mm": 4.3,
                "mse_mm2": 31.0,
                "rmse_mm": 5.57,  # pooled rows carry this; per-fold rows do not
            },
        ]
        out = write_rows_csv(tmp_path / "by_bin.csv", rows)
        frame = pd.read_csv(out)

        assert list(frame.columns) == ["lead_hours", "fold", "crps_mm", "mse_mm2", "rmse_mm"]
        # The row lacking the key gets an empty field, not a crash and not a
        # silent zero that would read as a real measurement.
        assert np.isnan(frame.loc[0, "rmse_mm"])
        assert frame.loc[1, "rmse_mm"] == pytest.approx(5.57)
        assert len(frame) == 2

    def test_fieldnames_follow_first_seen_order(self, tmp_path):
        rows = [{"b": 1, "a": 2}, {"c": 3, "a": 4}]
        out = write_rows_csv(tmp_path / "ordered.csv", rows)
        header = out.read_text().splitlines()[0]
        assert header == "b,a,c"

    def test_empty_row_list_writes_headerless_file(self, tmp_path):
        out = write_rows_csv(tmp_path / "empty.csv", [])
        assert out.exists()
        assert out.read_text() == ""

    def test_creates_missing_parent_directory(self, tmp_path):
        out = write_rows_csv(tmp_path / "nested" / "deep" / "x.csv", [{"a": 1}])
        assert out.exists()
