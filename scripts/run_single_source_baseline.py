#!/usr/bin/env python3
"""Score every rainfall source ALONE, on the tier scripts' own test split.

Every tier in this project (Tier 0's equal-weight mean, Tier 1's regional
weights, Tier 2's hierarchical EMOS/BMA) is currently justified against
*another blend*. Nothing in `results/` records what the raw sources score on
their own -- so nothing records whether blending is worth doing at all. A
reviewer can answer that in five minutes with this repo's own functions, so
this script answers it first, in the repo, with the uncomfortable result
committed rather than left to be discovered.

What it produces, per lead and per source:

- the raw source's TEST scores, from exactly the same code path Tier 0 uses
  (`run_tier0_baseline.score_lead`), on exactly the same aligned samples and
  the same split (`run_tier1_regional_baseline.load_aligned_forecasts_and_obs`
  outer-aligns all three sources onto one shared `sample` axis first, so
  every source and every blend is scored on literally the same held-out
  days -- not on per-source splits that merely "should" line up).
- `best_single_member_on_train`: the source with the lowest RMSE on the
  **train** split, and that source's test scores. Chosen on train only --
  picking the best source on test would be exactly the leak
  `src/weavr/splits.py` exists to prevent, and would make the baseline
  unbeatable for dishonest reasons.

Scope, inherited from the tier scripts rather than re-litigated:

- **Precipitation only**, and only the 3 of 4 baseline-store sources that
  carry `total_precipitation_24hr`. **Pangu has no precipitation variable at
  all** in its WeatherBench 2 archive (docs/baseline-store.md) -- it is
  named in the output as absent, and is *not* zero-filled. A zero-filled
  Pangu would score like a perfect dry forecast on dry days and would
  silently drag any mean it entered.
- WeatherBench 2 precipitation is in **metres**, IMD's rain is in
  millimetres; the conversion is `run_tier0_baseline.PRECIP_M_TO_MM`,
  applied by the shared alignment helper.
- RMSE alone is not the verdict. The research brief's section 1.3 point
  applies directly here: RMSE rewards smooth, MSE-trained AI output, so a
  source that wins on RMSE can still lose on CRPS, heavy-rain POD/ETS and
  reliability. This script emits the full Tier 0 metric set (bias, ACC,
  SEEPS, FSS, POD/FAR/CSI/ETS at every IMD threshold) for that reason.
  See docs/single-source-and-independence-results.md.

Usage:
    python scripts/run_single_source_baseline.py [--store PATH]
        [--climatology PATH] [--out-csv PATH] [--test-fraction F]
        [--neighborhood-size N]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_seeps_climatology import load_climatology  # noqa: E402
from run_tier0_baseline import score_lead  # noqa: E402
from run_tier1_regional_baseline import load_aligned_forecasts_and_obs  # noqa: E402

from weavr import verify as V  # noqa: E402
from weavr.climatology import climatological_ensemble  # noqa: E402
from weavr.score_io import (  # noqa: E402
    FORCE_HELP,
    guard_result_overwrites,
    per_day_scores,
    resolve_result_paths,
    write_per_day_scores,
)
from weavr.splits import iter_evaluation_folds  # noqa: E402
from weavr.stores import (  # noqa: E402
    open_multi_season,
    resolve_store_paths,
)

PRECIP_VARIABLE = "total_precipitation_24hr"
LEAD_HOURS = [24, 48, 72, 96, 120]
# The 3 sources that carry total_precipitation_24hr, matching Tier 0/Tier 1.
FORECAST_SOURCE_NAMES = ["graphcast", "hres", "ifs_ens_mean"]
# Named so the output states its absence rather than leaving a reader to
# wonder why a 4-source store produced 3 rows.
SOURCES_WITHOUT_PRECIPITATION = ["pangu"]
NEIGHBORHOOD_SIZE = 5  # grid cells (~1.25 deg edge) -- matches Tier 0.


def train_rmse_by_source(
    forecasts: dict[str, xr.DataArray],
    obs: xr.DataArray,
    train_mask,
) -> dict[str, float]:
    """RMSE of each raw source on the TRAIN split only.

    This is the selection statistic for `best_single_member_on_train`. It is
    deliberately the only thing computed on train: the reported scores all
    come from `score_lead`'s test split.
    """
    return {
        name: float(V.rmse(da.isel(sample=train_mask), obs.isel(sample=train_mask)))
        for name, da in forecasts.items()
    }


def best_single_member_on_train(train_rmse: dict[str, float]) -> str:
    """The lowest-train-RMSE source, ties broken by name for reproducibility.

    Pure, so the choice rule is testable without a store.
    """
    if not train_rmse:
        raise ValueError("train_rmse is empty -- no source to choose from.")
    return min(sorted(train_rmse), key=lambda name: train_rmse[name])


def write_climatology_per_day(
    obs_aligned: xr.DataArray,
    climatology: xr.Dataset,
    test_mask,
    lead_hours: int,
    thresholds: tuple[float, ...],
    results_dir: str,
    fold: str = "test",
) -> None:
    """Score the climatological reference on the same test days, as a method."""
    test_obs = obs_aligned.isel(sample=test_mask)
    test_dates = pd.DatetimeIndex(test_obs["sample"].values)
    test_years = sorted(set(test_dates.year))

    frames = []
    for position, date in enumerate(test_dates):
        ensemble = climatological_ensemble(
            climatology, pd.DatetimeIndex([date]), exclude_years=test_years
        ).transpose("sample", "member", "latitude", "longitude")
        frames.append(
            per_day_scores(
                ensemble.mean(dim="member", skipna=True),
                test_obs.isel(sample=[position]),
                fold=fold,
                ensemble=ensemble,
                thresholds=thresholds,
                probabilities={
                    t: (ensemble >= t).mean(dim="member", skipna=True) for t in thresholds
                },
            )
        )

    write_per_day_scores(
        "climatology",
        lead_hours,
        pd.concat(frames, ignore_index=True),
        out_dir=results_dir,
    )


def score_lead_all_sources(
    sources: dict[str, xr.Dataset],
    obs: xr.Dataset,
    climatology: xr.Dataset,
    lead_hours: int,
    test_fraction: float,
    thresholds: tuple[float, ...],
    neighborhood_size: int,
    results_dir: str | None = None,
) -> list[dict]:
    """One row per raw source at one lead, plus a `best_single_member_on_train`
    row repeating the winning source's test scores under a stable label.

    When `results_dir` is given, also writes per-day test scores for each
    source and for the climatological reference (step 04, additive).
    """
    forecasts, obs_aligned = load_aligned_forecasts_and_obs(
        sources, obs, PRECIP_VARIABLE, lead_hours
    )
    sample_times = pd.DatetimeIndex(obs_aligned["sample"].values)
    folds = list(iter_evaluation_folds(sample_times, test_fraction=test_fraction))

    fold_winners: dict[str, str] = {}
    for train_mask, test_mask, split_label in folds:
        train_rmse = train_rmse_by_source(forecasts, obs_aligned, train_mask)
        winner = best_single_member_on_train(train_rmse)
        fold_winners[split_label] = winner

        if results_dir is not None:
            test_obs = obs_aligned.isel(sample=test_mask)
            for name, da in forecasts.items():
                write_per_day_scores(
                    name,
                    lead_hours,
                    per_day_scores(
                        da.isel(sample=test_mask), test_obs, fold=split_label, thresholds=thresholds
                    ),
                    out_dir=results_dir,
                )
            write_per_day_scores(
                "best_single_member_on_train",
                lead_hours,
                per_day_scores(
                    forecasts[winner].isel(sample=test_mask),
                    test_obs,
                    fold=split_label,
                    thresholds=thresholds,
                ),
                out_dir=results_dir,
            )
            write_climatology_per_day(
                obs_aligned,
                climatology,
                test_mask,
                lead_hours,
                thresholds,
                results_dir,
                fold=split_label,
            )

    rows: list[dict] = []
    scored_by_source: dict[str, list[dict]] = {}
    for name in FORECAST_SOURCE_NAMES:
        lead_results = score_lead(
            forecasts[name], obs_aligned, climatology, test_fraction, thresholds, neighborhood_size
        )
        scored_by_source[name] = lead_results
        for res in lead_results:
            fold_lbl = res.get("fold", "seasonal_block_split")
            res["lead_hours"] = lead_hours
            res["source"] = name
            if fold_lbl in fold_winners:
                winner_for_fold = fold_winners[fold_lbl]
                matching_train = [f[0] for f in folds if f[2] == fold_lbl][0]
                train_rmses = train_rmse_by_source(forecasts, obs_aligned, matching_train)
                res["train_rmse_mm"] = train_rmses[name]
                res["selected_on_train"] = (name == winner_for_fold)
            else:
                fold_train_rmses = [
                    train_rmse_by_source(forecasts, obs_aligned, f[0])[name] for f in folds
                ]
                res["train_rmse_mm"] = float(np.mean(fold_train_rmses))
                res["selected_on_train"] = False
            rows.append(res)

    best_rows: list[dict] = []
    test_best_forecasts = []
    test_obs_list = []
    for train_mask, test_mask, split_label in folds:
        winner = fold_winners[split_label]
        winner_row = [r for r in scored_by_source[winner] if r.get("fold") == split_label][0]
        best_fold_row = dict(winner_row)
        best_fold_row["source"] = "best_single_member_on_train"
        best_fold_row["selected_source"] = winner
        best_fold_row["selected_on_train"] = False
        best_rows.append(best_fold_row)

        test_best_forecasts.append(forecasts[winner].isel(sample=test_mask))
        test_obs_list.append(obs_aligned.isel(sample=test_mask))

    if len(folds) > 1:
        pooled_best = xr.concat(test_best_forecasts, dim="sample")
        pooled_obs = xr.concat(test_obs_list, dim="sample")
        climatology_mean = climatology["rain"].mean(dim="time", skipna=True)
        fss_pooled = {
            t: float(
                V.fss(
                    pooled_best,
                    pooled_obs,
                    threshold=t,
                    neighborhood_size=neighborhood_size,
                )
            )
            for t in thresholds
        }
        contingency_pooled = V.contingency_scores(
            pooled_best, pooled_obs, thresholds=thresholds
        )
        best_pooled_row = {
            "lead_hours": lead_hours,
            "source": "best_single_member_on_train",
            "selected_source": "pooled",
            "selected_on_train": False,
            "fold": "pooled",
            "split": "leave_one_year_out",
            "n_samples": int(obs_aligned.sizes["sample"]),
            "n_train": int(obs_aligned.sizes["sample"]),
            "n_test": int(pooled_best.sizes["sample"]),
            "train_rmse_mm": float(np.mean([r["train_rmse_mm"] for r in best_rows])),
            "rmse_mm": float(V.rmse(pooled_best, pooled_obs)),
            "bias_mm": float(V.bias(pooled_best, pooled_obs)),
            "acc": float(V.acc(pooled_best, pooled_obs, climatology_mean)),
            "seeps": float(
                V.seeps(pooled_best, pooled_obs, climatology["rain"], climatology_dim="time")
            ),
            "fss": fss_pooled,
            "contingency": {
                t: {k: float(v) for k, v in s.items()}
                for t, s in contingency_pooled.items()
            },
        }
        best_rows.append(best_pooled_row)

    rows.extend(best_rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-stores",
        nargs="+",
        default=None,
        help="One or more baseline store paths (multi-season; default: 2018 + 2020 daily)",
    )
    parser.add_argument("--store", default=None, help="Legacy single store path")
    parser.add_argument("--climatology", default="data/imd_seeps_climatology_jjas.zarr")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for the per-day files, and the default parent for the output CSV(s).",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Aggregated CSV path (default: <results-dir>/single_source_baseline.csv)",
    )
    parser.add_argument("--force", action="store_true", help=FORCE_HELP)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--neighborhood-size", type=int, default=NEIGHBORHOOD_SIZE)
    args = parser.parse_args()
    _paths = resolve_result_paths(
        args.results_dir,
        {
            "out_csv": "single_source_baseline.csv",
        },
        {
            "out_csv": args.out_csv,
        },
    )
    args.out_csv = _paths["out_csv"]
    guard_result_overwrites(_paths.values(), force=args.force)

    baseline_paths = resolve_store_paths(args.baseline_stores, args.store)
    sources = {
        name: open_multi_season(baseline_paths, group=name)
        for name in FORECAST_SOURCE_NAMES
    }
    obs = open_multi_season(baseline_paths, group="imd_observed").load()
    climatology = load_climatology(args.climatology).load()

    print(f"Single-source baseline, scored against {baseline_paths}'s imd_observed group")
    print(f"Sources scored alone: {', '.join(FORECAST_SOURCE_NAMES)}")
    print(
        f"Not scored (no {PRECIP_VARIABLE} in its archive, not zero-filled): "
        f"{', '.join(SOURCES_WITHOUT_PRECIPITATION)}"
    )

    rows = []
    for lead_hours in LEAD_HOURS:
        lead_rows = score_lead_all_sources(
            sources,
            obs,
            climatology,
            lead_hours,
            args.test_fraction,
            V.IMD_RAIN_THRESHOLDS_MM,
            args.neighborhood_size,
            results_dir=args.results_dir,
        )
        rows.extend(lead_rows)
        for row in lead_rows:
            print(
                f"[lead {lead_hours:>3}h | fold {row.get('fold', 'n/a')}] {row['source']:<28} "
                f"train_rmse={row['train_rmse_mm']:.2f}mm test_rmse={row['rmse_mm']:.2f}mm"
            )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "lead_hours",
        "source",
        "fold",
        "selected_source",
        "selected_on_train",
        "split",
        "n_samples",
        "n_train",
        "n_test",
        "train_rmse_mm",
        "rmse_mm",
        "bias_mm",
        "acc",
        "seeps",
    ]
    for t in V.IMD_RAIN_THRESHOLDS_MM:
        fieldnames += [f"fss_{t}mm", f"pod_{t}mm", f"far_{t}mm", f"csi_{t}mm", f"ets_{t}mm"]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {k: row[k] for k in fieldnames if k in row}
            for t in V.IMD_RAIN_THRESHOLDS_MM:
                flat[f"fss_{t}mm"] = row["fss"][t]
                flat[f"pod_{t}mm"] = row["contingency"][t]["pod"]
                flat[f"far_{t}mm"] = row["contingency"][t]["far"]
                flat[f"csi_{t}mm"] = row["contingency"][t]["csi"]
                flat[f"ets_{t}mm"] = row["contingency"][t]["ets"]
            writer.writerow(flat)

    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
