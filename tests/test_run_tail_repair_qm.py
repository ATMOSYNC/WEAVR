import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_tail_repair_qm import (  # noqa: E402
    VARIANTS,
    deterministic_per_day,
    main,
    paired_metrics,
    parse_args,
    pooled_categorical,
    summary_metrics,
)


class TestRunTailRepairQmArgs:
    def test_default_args(self):
        args = parse_args([])
        # Both variants by default: neither is selected by looking at test
        # score, so the caller has to narrow it deliberately.
        assert args.variants == VARIANTS
        assert args.out_csv is None
        # String defaults, matching the other eight scoring runners; the
        # path helpers coerce at use.
        assert args.results_dir == "results"

    def test_variant_selection(self):
        args = parse_args(["--variants", "climatology"])
        assert args.variants == ["climatology"]

    def test_unknown_variant_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["--variants", "nonsense"])

    def test_redirectable_outputs(self):
        args = parse_args(["--results-dir", "/tmp/scratch"])
        assert args.results_dir == "/tmp/scratch"
        assert args.out_csv is None

    def test_explicit_out_csv_wins(self):
        args = parse_args(["--out-csv", "/tmp/a.csv"])
        assert args.out_csv == "/tmp/a.csv"


class TestPooledCategorical:
    """`csi` consumes 3 count columns, `sedi` consumes 4.

    Getting that wrong was a real TypeError during the first run, and it
    would have been silent had the arity happened to match.
    """

    def _frame(self):
        return pd.DataFrame(
            {
                "hits_64.5": [10.0, 12.0],
                "misses_64.5": [5.0, 4.0],
                "false_alarms_64.5": [2.0, 1.0],
                "correct_negatives_64.5": [30.0, 33.0],
            }
        )

    def test_csi_uses_three_columns(self):
        assert pd.notna(pooled_categorical(self._frame(), 64.5, "csi"))

    def test_sedi_uses_four_columns(self):
        assert pd.notna(pooled_categorical(self._frame(), 64.5, "sedi"))

    def test_unknown_score_raises(self):
        with pytest.raises(ValueError, match="Unsupported categorical score"):
            pooled_categorical(self._frame(), 64.5, "not_a_score")


class TestDeterministicPerDay:
    def test_suppresses_duplicate_crps_column(self):
        coords = {
            "sample": pd.date_range("2020-06-01", periods=3, freq="D"),
            "latitude": [10.0, 11.0],
            "longitude": [70.0, 71.0],
        }
        shape = (3, 2, 2)
        rng = np.random.default_rng(0)
        dims = ("sample", "latitude", "longitude")
        da = {
            name: xr.DataArray(rng.random(shape), coords=coords, dims=dims)
            for name in ("forecast", "obs")
        }
        frame = deterministic_per_day(da["forecast"], da["obs"], "2020", 64.5)
        assert "crps_mm" not in frame.columns
        assert "twcrps_64.5_mm" in frame.columns
        assert "brier_7.5" in frame.columns
        assert len(frame) == 3


class TestMainIsNotAStub:
    """The regression that matters most: this runner used to evaluate nothing."""

    def test_main_exists_and_is_callable(self):
        assert callable(main)

    def test_no_longer_prints_only_initialized_message(self):
        assert "Tail repair QM runner initialized" not in (
            SCRIPTS / "run_tail_repair_qm.py"
        ).read_text()


def _per_day_frame(n_days: int, scale: float, hits: float, false_alarms: float) -> pd.DataFrame:
    """Synthetic per-day rows in `deterministic_per_day`'s schema."""
    rng = np.random.default_rng(int(scale * 100))
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-06-01", periods=n_days, freq="D"),
            "mse_mm2": scale * (1.0 + rng.random(n_days)),
            "mae_mm": scale * (0.5 + rng.random(n_days)),
            "twcrps_64.5_mm": scale * 0.1 * (1.0 + rng.random(n_days)),
            "brier_7.5": 0.2 + 0.01 * rng.random(n_days),
        }
    )
    for t in (7.5, 64.5, 115.6, 204.5):
        frame[f"hits_{t}"] = hits + rng.integers(0, 2, n_days)
        frame[f"misses_{t}"] = 10.0 - hits + rng.integers(0, 2, n_days)
        frame[f"false_alarms_{t}"] = false_alarms + rng.integers(0, 2, n_days)
        frame[f"correct_negatives_{t}"] = 500.0 + rng.integers(0, 2, n_days)
    return frame


class TestSummaryMetrics:
    def test_keeps_the_legacy_columns_first_and_adds_every_prompt_metric(self):
        row = summary_metrics(_per_day_frame(30, 1.0, 5.0, 3.0))
        assert list(row)[:6] == [
            "rmse_mm", "mae_mm", "twcrps_64.5_mm", "brier_7.5", "sedi_115.6", "csi_64.5",
        ]
        for t in (64.5, 115.6, 204.5):
            for score in ("pod", "far", "csi", "sedi"):
                assert np.isfinite(row[f"{score}_{t}"]), f"{score}_{t}"

    def test_crps_of_a_point_forecast_is_its_mae(self):
        row = summary_metrics(_per_day_frame(30, 1.0, 5.0, 3.0))
        assert row["crps_mm"] == row["mae_mm"]


class TestPairedMetrics:
    def test_a_better_qm_arm_reads_negative_on_every_loss_and_score(self):
        raw = _per_day_frame(60, 2.0, 3.0, 8.0)
        qm = _per_day_frame(60, 1.0, 8.0, 1.0)  # lower errors, more hits, fewer false alarms
        qm["date"] = raw["date"]
        row = paired_metrics(qm, raw, block_days=7, n_resamples=200, seed=0)
        for key in ("rmse_delta_mm", "crps_delta_mm", "twcrps_64.5_delta"):
            assert row[key] < 0, key
        for t in (64.5, 115.6, 204.5):
            for score in ("pod", "far", "csi", "sedi"):
                assert row[f"{score}_{t}_delta"] < 0, f"{score}_{t}"
        assert row["twcrps_64.5_improved"] and row["h7_lead_improved"]

    def test_h7_columns_keep_their_names(self):
        raw = _per_day_frame(30, 1.0, 5.0, 3.0)
        row = paired_metrics(raw, raw.copy(), block_days=7, n_resamples=100, seed=0)
        for key in (
            "twcrps_64.5_improved", "sedi_115.6_improved", "brier_7.5_degraded",
            "h7_lead_improved", "bootstrap_degenerate", "rmse_worse",
        ):
            assert key in row
        assert not row["h7_lead_improved"]
