#!/usr/bin/env python3
"""Run Tail Repair via Quantile Mapping (Step 11).

Evaluates per-source, per-region quantile mapping (QM) against IMD observations:
a. Raw vs QM-corrected single sources (GraphCast, HRES, IFS-ENS mean)
b. Tier 1 regional blend built from QM-corrected sources
c. EMOS-CSG fit on QM-corrected ensemble members

Evaluates H7 (QM part): passes if tw-CRPS or SEDI@115.6 improves at >= 3/5 leads
with CI excluding 0, AND Brier@7.5 does not get significantly worse.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

LEAD_HOURS = [24, 48, 72, 96, 120]
IMD_THRESHOLDS = [7.5, 35.5, 64.5, 115.6, 204.5]


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-store",
        type=Path,
        default=Path("data/baseline_2020_jjas_daily.zarr"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/tail_repair_qm.csv"),
    )
    parser.add_argument(
        "--variant",
        choices=["same_period", "climatology"],
        default="same_period",
        help="Observed reference distribution source",
    )
    return parser.parse_args(args)


def main() -> int:
    args = parse_args()
    print(f"Tail repair QM runner initialized with variant={args.variant} -> {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
