#!/usr/bin/env python3
"""Threshold-choice and shape-interval diagnostics for the Step 12 pooled GPD tail.

`scripts/run_tail_repair_evt.py` fits the tail at one threshold, u = 64.5 mm,
and reports xi with an interval from the observed information matrix. #74 asks
for two things that run cannot show:

1. **Evidence for the threshold.** A GPD is only appropriate above a u where
   the shape has stabilised. The standard check is to refit across candidate
   thresholds and look for the region where xi is flat within its uncertainty
   and the mean residual life `E[Y - u | Y > u]` is linear in u.
2. **Whether the interval can be believed.** The information-matrix interval
   treats every exceedance as independent. They are not: one monsoon day
   produces exceedances in dozens of neighbouring cells, and wet spells span
   consecutive days. So the fit is repeated on a moving-block bootstrap that
   resamples *whole days* in 7-day blocks, which carries that dependence into
   the resamples. The ratio of the two interval widths is the dependence
   inflation the information-matrix interval leaves out.

Everything is fitted on the IMD JJAS archive (2006-2020) minus each LOYO
fold's test year -- the exact data the runner fits on since 31 Oct (step 12:
"IMD archive exceedances above u, EXCLUDING the test year"; it used to be the
training season alone) -- and nothing here touches a forecast. The block
bootstrap resamples archive days in 7-day blocks; a block can span the gap
between two seasons, which only adds a little independence it should not.

Output: `results/tail_evt_diagnostics.csv`, one row per (fold, threshold).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_seeps_climatology import load_climatology  # noqa: E402

from weavr.quantile_mapping import climatology_excluding_years  # noqa: E402
from weavr.regions import assign_regions  # noqa: E402
from weavr.score_io import FORCE_HELP, guard_result_overwrites  # noqa: E402
from weavr.splits import iter_evaluation_folds  # noqa: E402
from weavr.stores import (  # noqa: E402
    DEFAULT_BASELINE_DAILY_STORES,
    open_multi_season,
    resolve_store_paths,
)
from weavr.tail import DEFAULT_TAIL_THRESHOLD_U, fit_pooled_gpd  # noqa: E402

#: Candidate thresholds, bracketing the runner's u = 64.5 mm (IMD "heavy") from
#: below and running up to the 115.6 mm "very heavy" boundary.
CANDIDATE_THRESHOLDS_MM = (35.5, 50.0, 64.5, 80.0, 100.0, 115.6)


def exceedances_by_region(
    rain: np.ndarray, region_labels: np.ndarray, threshold_u: float
) -> dict[str, np.ndarray]:
    """Per-region values above `threshold_u` from a `(day, lat, lon)` grid."""
    out: dict[str, np.ndarray] = {}
    for region in np.unique(region_labels):
        values = rain[:, region_labels == region].ravel()
        values = values[np.isfinite(values) & (values > threshold_u)]
        if values.size:
            out[str(region)] = values
    return out


def moving_block_day_indices(
    n_days: int, block_days: int, rng: np.random.Generator
) -> np.ndarray:
    """Day indices for one moving-block bootstrap resample of length `n_days`.

    Whole days are resampled, so every cell on a chosen day comes along
    together and spatial dependence is preserved; days come in runs of
    `block_days`, so wet spells are too.
    """
    block = max(1, min(block_days, n_days))
    n_blocks = -(-n_days // block)
    starts = rng.integers(0, n_days - block + 1, size=n_blocks)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n_days]


def diagnose_threshold(
    rain: np.ndarray,
    region_labels: np.ndarray,
    threshold_u: float,
    n_resamples: int,
    block_days: int,
    rng: np.random.Generator,
) -> dict:
    """xi with both intervals, and the mean residual life, at one threshold."""
    excess = exceedances_by_region(rain, region_labels, threshold_u)
    fit = fit_pooled_gpd(excess, threshold_u=threshold_u)
    pooled = np.concatenate(list(excess.values()))

    boot = np.full(n_resamples, np.nan)
    for k in range(n_resamples):
        days = moving_block_day_indices(rain.shape[0], block_days, rng)
        try:
            boot[k] = fit_pooled_gpd(
                exceedances_by_region(rain[days], region_labels, threshold_u),
                threshold_u=threshold_u,
            ).shape_xi
        except ValueError:
            # A resample with too few exceedances in every region cannot be
            # fitted; it is dropped and counted rather than imputed.
            continue
    ok = boot[np.isfinite(boot)]
    boot_lo, boot_hi = (np.quantile(ok, [0.025, 0.975]) if ok.size else (np.nan, np.nan))
    hessian_width = (
        fit.xi_ci[1] - fit.xi_ci[0] if fit.xi_ci is not None else float("nan")
    )
    return {
        "threshold_u": threshold_u,
        "n_exceedances": int(pooled.size),
        "n_regions_fitted": int(sum(n >= 10 for n in fit.n_samples_by_region.values())),
        "mean_residual_life_mm": float(np.mean(pooled - threshold_u)),
        "xi": fit.shape_xi,
        "xi_se_information": fit.xi_se if fit.xi_se is not None else float("nan"),
        "xi_ci_lo_information": fit.xi_ci[0] if fit.xi_ci is not None else float("nan"),
        "xi_ci_hi_information": fit.xi_ci[1] if fit.xi_ci is not None else float("nan"),
        "xi_se_block_bootstrap": float(np.std(ok, ddof=1)) if ok.size > 1 else float("nan"),
        "xi_ci_lo_block_bootstrap": float(boot_lo),
        "xi_ci_hi_block_bootstrap": float(boot_hi),
        "dependence_inflation": float((boot_hi - boot_lo) / hessian_width),
        "n_resamples_fitted": int(ok.size),
    }


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-stores", nargs="+", default=None)
    parser.add_argument("--climatology", default="data/imd_seeps_climatology_jjas.zarr")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--n-resamples", type=int, default=200)
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help=FORCE_HELP)
    return parser.parse_args(args)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_csv = args.out_csv or Path(args.results_dir) / "tail_evt_diagnostics.csv"
    guard_result_overwrites([out_csv], force=args.force)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    baseline_paths = resolve_store_paths(
        args.baseline_stores, None, DEFAULT_BASELINE_DAILY_STORES, None
    )
    obs = open_multi_season(baseline_paths, group="imd_observed")["rain"].load()
    obs = obs.transpose("time", "latitude", "longitude")
    region_labels = assign_regions(obs["latitude"].values, obs["longitude"].values)
    region_2d = region_labels.transpose("latitude", "longitude").values
    rng = np.random.default_rng(args.seed)

    archive = load_climatology(args.climatology).load()["rain"]
    archive = archive.transpose("time", "latitude", "longitude")
    rows = []
    times = pd.DatetimeIndex(obs["time"].values)
    for train_mask, _test_mask, fold in iter_evaluation_folds(
        times, test_fraction=args.test_fraction
    ):
        rain = climatology_excluding_years(archive, times[_test_mask]).values
        for threshold_u in CANDIDATE_THRESHOLDS_MM:
            # `n_train_days`: archive days the tail is fitted on (test year removed).
            row = {"fold": fold, "n_train_days": int(rain.shape[0])}
            row.update(
                diagnose_threshold(
                    rain, region_2d, threshold_u, args.n_resamples, args.block_days, rng
                )
            )
            row["is_runner_threshold"] = threshold_u == DEFAULT_TAIL_THRESHOLD_U
            rows.append(row)
            print(
                f"[fold {fold} | u={threshold_u:>5}mm] n={row['n_exceedances']:>6} "
                f"xi={row['xi']:+.3f} info CI[{row['xi_ci_lo_information']:+.3f},"
                f"{row['xi_ci_hi_information']:+.3f}] block CI["
                f"{row['xi_ci_lo_block_bootstrap']:+.3f},{row['xi_ci_hi_block_bootstrap']:+.3f}]"
                f" x{row['dependence_inflation']:.2f} MRL={row['mean_residual_life_mm']:.1f}mm"
            )

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
