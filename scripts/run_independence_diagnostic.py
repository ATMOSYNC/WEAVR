#!/usr/bin/env python3
"""Measure how independent the three rainfall sources' errors actually are.

Blending only helps when the sources make different mistakes. This script
tests that premise directly, with `src/weavr/independence.py`, and writes
`results/independence_diagnostic.csv`: the pairwise error correlations, the
effective number of models, and the participation ratio, for

- each lead, domain-wide, and
- each lead x region (`src/weavr/regions.py`'s Sreekala & Babu 6 zones, the
  same pooling Tier 1 fits its weights over).

**Computed on the TRAIN split only.** The correlation structure is an input
to how the blend is built (it is the reason Tier 1's fitted weights look the
way they do), so measuring it on test days would leak the test set into the
design, exactly what `src/weavr/splits.py` exists to prevent. The split
comes from `run_tier0_baseline.train_test_masks`, so it is literally the
tier scripts' split, not a re-derivation of it.

Two things this diagnostic is expected to explain:

1. Whether the equal-weight mean can help at all. If the effective number of
   models is near 1, averaging three sources is close to averaging one, and
   no weighting scheme recovers a diversity that isn't there.
2. Why Tier 1 gives HRES zero weight in most region x lead cells. A zero OLS
   weight under collinearity means "this source is redundant *given the
   others in this fit*", not "this source is bad" -- and the HRES/IFS-ENS
   correlation printed here is what makes that the right reading.

Unit note, and a real bug this script would otherwise hide: WeatherBench 2's
`total_precipitation_24hr` is in **metres**, IMD's `rain` is in
**millimetres**. Unconverted, every source's error is essentially `-obs`,
and every pairwise correlation comes out at exactly 1.0 -- which looks like
a dramatic finding rather than the unit bug it is. The conversion
(`run_tier0_baseline.PRECIP_M_TO_MM`) is applied by the shared alignment
helper this script reuses.

Usage:
    python scripts/run_independence_diagnostic.py [--store PATH]
        [--out-csv PATH] [--test-fraction F]
"""

from __future__ import annotations

import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tier1_regional_baseline import load_aligned_forecasts_and_obs  # noqa: E402

from weavr.independence import (  # noqa: E402
    effective_number_of_models,
    error_correlation_matrix,
    participation_ratio,
)
from weavr.regions import assign_regions  # noqa: E402
from weavr.splits import iter_evaluation_folds  # noqa: E402
from weavr.stores import (  # noqa: E402
    DEFAULT_BASELINE_DAILY_STORES,
    open_multi_season,
    resolve_store_paths,
)

PRECIP_VARIABLE = "total_precipitation_24hr"
LEAD_HOURS = [24, 48, 72, 96, 120]
FORECAST_SOURCE_NAMES = ["graphcast", "hres", "ifs_ens_mean"]
ERROR_DIMS = ("sample", "latitude", "longitude")


def diagnose(errors: dict[str, xr.DataArray]) -> dict[str, object]:
    """Correlations plus both effective-dimensionality summaries for one
    (lead, scope) slice. Returns a flat row, with one `corr_<a>_<b>` column
    per source pair. `main` then adds the identifying columns (lead, scope,
    region, split), which is why the value type is widened to `object`.
    """
    names, matrix = error_correlation_matrix(errors, ERROR_DIMS)
    row: dict[str, object] = {
        "n_sources": len(names),
        "n_eff": effective_number_of_models(matrix),
        "participation_ratio": participation_ratio(matrix),
        "mean_off_diagonal": float(np.mean(matrix[~np.eye(len(names), dtype=bool)])),
    }
    index = {name: i for i, name in enumerate(names)}
    for a, b in combinations(names, 2):
        row[f"corr_{a}_{b}"] = float(matrix[index[a], index[b]])
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-stores",
        nargs="+",
        default=None,
        help="One or more baseline store paths (multi-season; default: 2018 + 2020 daily)",
    )
    parser.add_argument("--baseline-store", default=None, help="Legacy single baseline store path")
    parser.add_argument("--store", default=None, help="Legacy single store path")
    parser.add_argument("--out-csv", default="results/independence_diagnostic.csv")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()

    baseline_stores = resolve_store_paths(
        args.baseline_stores or args.baseline_store or args.store or DEFAULT_BASELINE_DAILY_STORES
    )

    sources = {
        name: open_multi_season(baseline_stores, group=name)
        for name in FORECAST_SOURCE_NAMES
    }
    obs = open_multi_season(baseline_stores, group="imd_observed").load()
    region_labels = assign_regions(obs["latitude"].values, obs["longitude"].values)

    print(f"Independence diagnostic on {baseline_stores}, TRAIN split only")
    print(f"Sources: {', '.join(FORECAST_SOURCE_NAMES)}")

    rows = []
    for lead_hours in LEAD_HOURS:
        forecasts, obs_aligned = load_aligned_forecasts_and_obs(
            sources, obs, PRECIP_VARIABLE, lead_hours
        )
        sample_times = pd.DatetimeIndex(obs_aligned["sample"].values)
        folds = list(iter_evaluation_folds(sample_times, test_fraction=args.test_fraction))

        for train_mask, test_mask, split_label in folds:
            split_kind = (
                "leave_one_year_out"
                if split_label != "seasonal_block_split"
                else "seasonal_block_split"
            )
            train_obs = obs_aligned.isel(sample=train_mask)
            errors = {
                name: da.isel(sample=train_mask) - train_obs for name, da in forecasts.items()
            }

            domain_row = diagnose(errors)
            domain_row.update(
                {
                    "lead_hours": lead_hours,
                    "fold": split_label,
                    "scope": "domain",
                    "region": "",
                    "n_train": int(train_mask.sum()),
                    "split": split_kind,
                }
            )
            rows.append(domain_row)
            print(
                f"[lead {lead_hours:>3}h | fold {split_label}] domain "
                f"n_eff={domain_row['n_eff']:.3f} "
                f"pr={domain_row['participation_ratio']:.3f} "
                f"mean_corr={domain_row['mean_off_diagonal']:.3f}"
            )

            for region in sorted(np.unique(region_labels.values).tolist()):
                region_mask = (region_labels == region).values
                region_errors = {name: da.where(region_mask) for name, da in errors.items()}
                region_row = diagnose(region_errors)
                region_row.update(
                    {
                        "lead_hours": lead_hours,
                        "fold": split_label,
                        "scope": "region",
                        "region": region,
                        "n_train": int(train_mask.sum()),
                        "split": split_kind,
                    }
                )
                rows.append(region_row)
                print(
                    f"[lead {lead_hours:>3}h | fold {split_label}] {region:<7} "
                    f"n_eff={region_row['n_eff']:.3f} "
                    f"pr={region_row['participation_ratio']:.3f} "
                    f"mean_corr={region_row['mean_off_diagonal']:.3f}"
                )

        if len(folds) > 1:
            all_train = np.ones(obs_aligned.sizes["sample"], dtype=bool)
            train_obs = obs_aligned.isel(sample=all_train)
            errors = {
                name: da.isel(sample=all_train) - train_obs for name, da in forecasts.items()
            }
            op_domain_row = diagnose(errors)
            op_domain_row.update(
                {
                    "lead_hours": lead_hours,
                    "fold": "operational",
                    "scope": "domain",
                    "region": "",
                    "n_train": int(all_train.sum()),
                    "split": "operational_all_seasons",
                }
            )
            rows.append(op_domain_row)
            for region in sorted(np.unique(region_labels.values).tolist()):
                region_mask = (region_labels == region).values
                region_errors = {name: da.where(region_mask) for name, da in errors.items()}
                op_region_row = diagnose(region_errors)
                op_region_row.update(
                    {
                        "lead_hours": lead_hours,
                        "fold": "operational",
                        "scope": "region",
                        "region": region,
                        "n_train": int(all_train.sum()),
                        "split": "operational_all_seasons",
                    }
                )
                rows.append(op_region_row)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    leading = ["lead_hours", "fold", "scope", "region", "n_train", "split", "n_sources"]
    trailing = [k for k in rows[0] if k not in leading]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=leading + trailing)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
