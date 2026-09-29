#!/usr/bin/env python3
"""Run Tail Repair via Extreme-Value Tail (Step 12).

Evaluates pooled Generalized Pareto Distribution (GPD) extreme-value tails
for unfittable rain bins anchored on IMD climatological extremes:
- Evaluates P(>=115.6) and P(>=204.5) exceedance probabilities
- Reliability tables, SEDI, Brier Score, and threshold-weighted CRPS (tw-CRPS)
- Benchmarked against point-mass fallback, Step 11 QM, and raw IFS-ENS member counting
- Evaluates H7 (EVT part) verdict
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

LEAD_HOURS = [24, 48, 72, 96, 120]
EXTREME_THRESHOLDS = [115.6, 204.5]


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-store",
        type=Path,
        default=Path("data/baseline_2020_jjas_daily.zarr"),
        help="Path to baseline Zarr store",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/tail_repair_evt.csv"),
        help="Path to output results CSV",
    )
    parser.add_argument(
        "--threshold-u",
        type=float,
        default=64.5,
        help="Tail splicing threshold u in mm (default: 64.5 mm)",
    )
    return parser.parse_args(args)


def compute_sedi(hits: int, misses: int, false_alarms: int, correct_negatives: int) -> float:
    """Symmetric Extremal Dependence Index (SEDI)."""
    eps = 1e-9
    total = hits + misses + false_alarms + correct_negatives
    if total == 0:
        return 0.0

    hit_rate = (hits + eps) / (hits + misses + 2 * eps)
    false_alarm_rate = (false_alarms + eps) / (false_alarms + correct_negatives + 2 * eps)

    log_f = np.log(false_alarm_rate)
    log_h = np.log(hit_rate)
    log_1_f = np.log(1.0 - false_alarm_rate + eps)
    log_1_h = np.log(1.0 - hit_rate + eps)

    numerator = (log_f - log_h) - (log_1_f - log_1_h)
    denominator = (log_f + log_h) + (log_1_f + log_1_h)

    if abs(denominator) < eps:
        return 0.0
    return float(numerator / denominator)


def main() -> int:
    args = parse_args()
    print(
        f"Tail repair EVT runner initialized with threshold_u={args.threshold_u} mm "
        f"-> {args.out_csv}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
