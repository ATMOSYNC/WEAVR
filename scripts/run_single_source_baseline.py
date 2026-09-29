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

import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_seeps_climatology import load_climatology  # noqa: E402
from run_tier0_baseline import score_lead, train_test_masks  # noqa: E402
from run_tier1_regional_baseline import load_aligned_forecasts_and_obs  # noqa: E402

from weavr import verify as V  # noqa: E402
from weavr.climatology import climatological_ensemble  # noqa: E402
from weavr.score_io import per_day_scores, write_per_day_scores  # noqa: E402

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
) -> None:
    """Score the climatological reference on the same test days, as a method.

    Every skill score in `docs/preregistration.md` is stated against
    climatology (H3's Brier skill score explicitly), so climatology has to
    appear in `results/per_day/` like any other method for the scorecard to
    pair against it. The test year is excluded from the reference, which is
    the whole point -- see `weavr.climatology`.

    It is written here, in the single-source script, because this is where
    the "what would you have said knowing nothing" baselines belong; it does
    not depend on the lead time at all, but is written per lead so every
    comparison has a same-shaped partner file.
    """
    test_obs = obs_aligned.isel(sample=test_mask)
    test_dates = pd.DatetimeIndex(test_obs["sample"].values)
    test_years = sorted(set(test_dates.year))

    # Built one day at a time, deliberately. A 15-day window over a 14-year
    # archive gives 434 members, so a single day's ensemble on the full
    # 129x135 grid is already ~60 MB; materialising every test day at once
    # exhausted memory on the first attempt here, and step 07's daily,
    # two-season split has ~30x more test days. Looping keeps the footprint
    # flat regardless of how many days are scored.
    frames = []
    for position, date in enumerate(test_dates):
        ensemble = climatological_ensemble(
            climatology, pd.DatetimeIndex([date]), exclude_years=test_years
        ).transpose("sample", "member", "latitude", "longitude")
        # The climatological point forecast is the ensemble mean -- the
        # honest deterministic answer from climate alone. Probabilities come
        # from this same ensemble rather than from a second call per
        # threshold, which would rebuild it once per IMD category.
        frames.append(
            per_day_scores(
                ensemble.mean(dim="member", skipna=True),
                test_obs.isel(sample=[position]),
                fold="test",
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
    _, train_mask, test_mask = train_test_masks(sample_times, test_fraction)
    train_rmse = train_rmse_by_source(forecasts, obs_aligned, train_mask)
    winner = best_single_member_on_train(train_rmse)

    if results_dir is not None:
        test_obs = obs_aligned.isel(sample=test_mask)
        for name, da in forecasts.items():
            write_per_day_scores(
                name,
                lead_hours,
                per_day_scores(
                    da.isel(sample=test_mask), test_obs, fold="test", thresholds=thresholds
                ),
                out_dir=results_dir,
            )
        # The train-chosen baseline is a method in its own right: it is what
        # docs/preregistration.md's H1 compares against, and it is not always
        # the same source at every lead.
        write_per_day_scores(
            "best_single_member_on_train",
            lead_hours,
            per_day_scores(
                forecasts[winner].isel(sample=test_mask),
                test_obs,
                fold="test",
                thresholds=thresholds,
            ),
            out_dir=results_dir,
        )
        write_climatology_per_day(
            obs_aligned, climatology, test_mask, lead_hours, thresholds, results_dir
        )

    rows = []
    scored: dict[str, dict] = {}
    for name in FORECAST_SOURCE_NAMES:
        result = score_lead(
            forecasts[name], obs_aligned, climatology, test_fraction, thresholds, neighborhood_size
        )
        result["lead_hours"] = lead_hours
        result["source"] = name
        result["train_rmse_mm"] = train_rmse[name]
        result["selected_on_train"] = name == winner
        scored[name] = result
        rows.append(result)

    best_row = dict(scored[winner])
    best_row["source"] = "best_single_member_on_train"
    best_row["selected_source"] = winner
    # This row repeats the winner's scores under a stable label; the flag
    # belongs to the raw-source row only, so filtering the CSV on
    # selected_on_train returns one row per lead, not two.
    best_row["selected_on_train"] = False
    rows.append(best_row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="data/baseline_2020_jjas.zarr")
    parser.add_argument("--climatology", default="data/imd_seeps_climatology_jjas.zarr")
    parser.add_argument("--out-csv", default="results/single_source_baseline.csv")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--neighborhood-size", type=int, default=NEIGHBORHOOD_SIZE)
    args = parser.parse_args()

    sources = {
        name: xr.open_zarr(args.store, group=name, consolidated=True)
        for name in FORECAST_SOURCE_NAMES
    }
    obs = xr.open_zarr(args.store, group="imd_observed", consolidated=True).load()
    climatology = load_climatology(args.climatology).load()

    print(f"Single-source baseline, scored against {args.store}'s imd_observed group")
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
                f"[lead {lead_hours:>3}h] {row['source']:<28} "
                f"train_rmse={row['train_rmse_mm']:.2f}mm test_rmse={row['rmse_mm']:.2f}mm"
            )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "lead_hours",
        "source",
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
