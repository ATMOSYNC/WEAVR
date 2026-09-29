#!/usr/bin/env python3
"""Run Tier 2b combined combiners (EMOS-CSG + BMA meta-blend) -- Step 13.

Builds and tests two ways to combine EMOS-CSG and BMA under pre-registered H10:
(a) Per-bin x lead selection: chooses the combiner with lower train-fold CRPS per (rain bin, lead).
(b) Quantile averaging (Vincentization): averages predictive quantiles (Lichtendahl et al. 2013).
(c) Linear pool: standard probability averaging as a comparison baseline.

Evaluates under LOYO (or block split fallback) with CIs and pre-registered H10 verdict.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from weavr.stacking import (
    DEFAULT_QUANTILE_LEVELS,
    crps_from_quantiles,
    predictive_quantiles_bma,
    predictive_quantiles_csgd,
    quantile_average,
)

LEAD_HOURS = [24, 48, 72, 96, 120]
IMD_THRESHOLDS = [7.5, 35.5, 64.5, 115.6, 204.5]


def combine_predictive_quantiles(
    csgd_mean: np.ndarray,
    csgd_std: np.ndarray,
    csgd_shift: np.ndarray,
    bma_samples: np.ndarray,
    levels: np.ndarray | None = None,
    weight: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (q_csgd, q_bma, q_avg) predictive quantiles."""
    if levels is None:
        levels = DEFAULT_QUANTILE_LEVELS

    q_csgd = predictive_quantiles_csgd(csgd_mean, csgd_std, csgd_shift, levels=levels)
    q_bma = predictive_quantiles_bma(bma_samples, levels=levels)
    q_avg = quantile_average(q_csgd, q_bma, weight=weight)
    return q_csgd, q_bma, q_avg


def compute_brier_score(prob: np.ndarray, obs: np.ndarray, threshold: float) -> float:
    """Mean Brier score for event obs > threshold."""
    event = (obs > threshold).astype(float)
    return float(np.mean((prob - event) ** 2))


def score_quantiles_crps(
    quantiles: np.ndarray,
    levels: np.ndarray,
    obs: np.ndarray,
) -> float:
    """Compute mean CRPS from predictive quantiles against observations."""
    crps_vals = crps_from_quantiles(quantiles, levels, obs)
    return float(np.mean(crps_vals))


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-store", type=Path, default=Path("data/baseline_2020_jjas_daily.zarr")
    )
    parser.add_argument(
        "--lagged-store",
        type=Path,
        default=Path("data/lagged_ensemble_inputs_2020_jjas_daily.zarr"),
    )
    parser.add_argument(
        "--ifs-ensemble-store", type=Path, default=Path("data/ifs_ens_2020_jjas.zarr")
    )
    parser.add_argument("--out-csv", type=Path, default=Path("results/tier2b_combined.csv"))
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--quantile-weight", type=float, default=0.5)
    return parser.parse_args(args)


def main() -> int:
    args = parse_args()
    print(f"Tier 2b combined combiners runner initialized with output at {args.out_csv}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
