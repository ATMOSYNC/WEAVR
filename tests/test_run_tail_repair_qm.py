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
    parse_args,
    pooled_categorical,
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
