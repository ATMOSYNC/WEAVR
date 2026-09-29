"""Tests for scripts/run_tail_repair_evt.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from run_tail_repair_evt import compute_sedi, parse_args  # noqa: E402


def test_parse_args() -> None:
    """parse_args accepts expected CLI flags."""
    args = parse_args(["--threshold-u", "70.0", "--out-csv", "results/test.csv"])
    assert args.threshold_u == 70.0
    assert args.out_csv == Path("results/test.csv")


def test_compute_sedi_edge_cases() -> None:
    """compute_sedi computes bounded SEDI scores."""
    # Perfect forecast
    sedi_perfect = compute_sedi(hits=10, misses=0, false_alarms=0, correct_negatives=100)
    assert sedi_perfect > 0.9

    # Empty
    assert compute_sedi(0, 0, 0, 0) == 0.0
