#!/usr/bin/env python3
"""Run EMOS-CSG on quantile-mapped ensemble members -- Step 11, LOYO evaluation.

`scripts/run_tail_repair_qm.py` compares raw against quantile-mapped
*deterministic* sources (`graphcast`, `hres`, `ifs_ens_mean`) and against the
Tier 1 blend. That leaves the question Step 11 actually exists to answer
unanswered: does correcting the **members** before EMOS sees them change the
calibrated distribution?

This runner closes that gap. Under leave-one-year-out, per lead:

- `raw_members`           -- EMOS-CSG fitted on the raw lagged GraphCast
                             members. Today's behaviour, and the baseline both
                             QM arms are measured against.
- `qm_members_same_period` -- regional quantile maps fitted on the training
                             fold's own members, applied member by member,
                             then EMOS-CSG refitted on the corrected members.
- `qm_members_climatology` -- the same, but the maps are fitted against the
                             full JJAS climatological distribution rather than
                             the training fold, so the maps do not depend on
                             which year is being predicted.

Members come from the **lagged GraphCast store**, which exists for both
seasons, so this evaluation does not wait on the daily IFS-ENS build.

The comparison is deliberately paired. Both arms score the same held-out days
against the same observations, and the only change between them is whether EMOS
saw mapped members, so the CI is on the per-day difference rather than on two
independently noisy means. EMOS-CSG is fitted to minimise CRPS, so CRPS is the
headline metric; Brier and SEDI at the IMD thresholds are recorded alongside to
show whether a CRPS change came with a calibration change or only a sharpness
one.

A QM arm that raises CRPS is a result, not a failure of the runner, and is
written out as such -- `improves` in the paired table is False in that case
rather than the arm being quietly dropped.

Outputs
-------
`results/tail_repair_qm_emos.csv`
    Per `(lead, fold, arm, threshold)`: CRPS, Brier, SEDI, RMSE, cell counts.
`results/tail_repair_qm_emos_paired.csv`
    Each QM arm minus `raw_members`, per lead, with 95% CIs.
`results/per_day/qmemos_{method}__lead<L>.csv`
    Per-day rows in `weavr.score_io.per_day_scores` schema, so
    `scripts/run_scorecard.py` can bootstrap them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_seeps_climatology import load_climatology  # noqa: E402
from run_scorecard import _pooled_sedi  # noqa: E402
from run_tail_repair_evt import expected_counts  # noqa: E402
from run_tier2_hierarchical_baseline import load_graphcast_ensemble  # noqa: E402

from weavr.emos import (  # noqa: E402
    csgd_crps,
    exceedance_probability_csgd,
    fit_emos_csg,
    predict_csgd_params,
)
from weavr.quantile_mapping import (  # noqa: E402
    apply_regional_quantile_maps,
    climatology_excluding_years,
    fit_regional_quantile_maps,
)
from weavr.rain_bins import RAIN_BIN_LABELS, classify_rain_bin  # noqa: E402
from weavr.regions import assign_regions  # noqa: E402
from weavr.score_io import (  # noqa: E402
    FORCE_HELP,
    guard_result_overwrites,
    per_day_scores,
    resolve_result_paths,
    write_per_day_scores,
)
from weavr.significance import paired_difference_ci  # noqa: E402
from weavr.splits import iter_evaluation_folds  # noqa: E402
from weavr.stores import (  # noqa: E402
    DEFAULT_BASELINE_DAILY_STORES,
    DEFAULT_LAGGED_DAILY_STORES,
    open_multi_season,
    resolve_store_paths,
)

LEAD_HOURS = [24, 48, 72, 96, 120]
THRESHOLDS = (7.5, 35.5, 64.5, 115.6, 204.5)
ARMS = ["raw_members", "qm_members_same_period", "qm_members_climatology"]
ENSEMBLE_SOURCE = "graphcast"


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-stores", nargs="+", type=Path, default=None)
    parser.add_argument("--lagged-stores", nargs="+", type=Path, default=None)
    parser.add_argument(
        "--climatology",
        type=Path,
        default=Path("data/imd_seeps_climatology_jjas.zarr"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-paired-csv", type=Path, default=None)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--force", action="store_true", help=FORCE_HELP)
    return parser.parse_args(args)


def quantile_mapped_members(
    forecasts: xr.DataArray,
    train_mask: np.ndarray,
    reference: xr.DataArray,
    region_labels: xr.DataArray,
    lead_hours: int,
) -> xr.DataArray:
    """Return the ensemble with each member quantile-mapped.

    The maps are fitted on the training days' forecasts only and then applied
    to every day. Fitting on the whole series would put the test days'
    forecast distribution inside the map scored on them.

    `member_dim="member"` keeps the member dimension when fitting, so each
    member is corrected against the reference distribution instead of the
    ensemble average being corrected once. That matters because EMOS uses the
    ensemble spread as a predictor; correcting a single averaged field would
    leave the spread untouched and change the two arms in a way that is not
    the one under test.
    """
    maps = fit_regional_quantile_maps(
        forecasts.isel(sample=np.asarray(train_mask, dtype=bool)),
        reference,
        region_labels,
        sample_dim="sample",
        member_dim="member",
        source=ENSEMBLE_SOURCE,
        lead=lead_hours,
    )
    return apply_regional_quantile_maps(maps, forecasts, region_labels)


def fitted_probability_grid(
    emos_results: dict,
    labels: np.ndarray,
    mean_values: np.ndarray,
    spread_values: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Exceedance probability per cell, with unfittable bins at 0.0.

    The 0.0 is Tier 2's existing point-mass fallback, kept identical across
    arms so the QM effect is not confounded with a change in fallback policy.
    """
    probabilities = np.full(mean_values.shape, np.nan, dtype=float)
    for bin_label in RAIN_BIN_LABELS:
        mask = labels == bin_label
        if not mask.any():
            continue
        result = emos_results.get(str(bin_label))
        if result is None:
            probabilities[mask] = 0.0
            continue
        mean, std, shift = predict_csgd_params(
            result, mean_values[mask], spread_values[mask]
        )
        probabilities[mask] = exceedance_probability_csgd(mean, std, shift, threshold)
    return probabilities


def probability_sedi(probability: np.ndarray, obs: np.ndarray, threshold: float) -> float:
    """SEDI of a probability forecast, from expected contingency counts.

    `weavr.verify.sedi` thresholds forecast *values*, so a probability grid
    passed to it with a threshold in mm never exceeds it and the score is
    meaningless. This is the convention `run_tail_repair_evt.py` uses for its
    probabilistic arms, so the two runners' SEDI values are comparable.
    """
    counts = expected_counts(probability, (obs >= threshold).astype(float))
    return _pooled_sedi(*[float(np.sum(counts[k])) for k in counts])


def crps_grid(
    emos_results: dict,
    labels: np.ndarray,
    mean_values: np.ndarray,
    spread_values: np.ndarray,
    obs_values: np.ndarray,
) -> np.ndarray:
    """Per-cell CRPS of the fitted predictive CSGD, unreduced.

    Binned by the same rain-bin labels `fit_emos_csg` was fitted on, so each
    cell is scored against the distribution EMOS actually assigned to it.
    """
    scores = np.full(mean_values.shape, np.nan, dtype=float)
    for bin_label in RAIN_BIN_LABELS:
        result = emos_results.get(str(bin_label))
        if result is None:
            continue
        mask = (labels == bin_label) & ~np.isnan(obs_values) & ~np.isnan(mean_values)
        if not mask.any():
            continue
        mean, std, shift = predict_csgd_params(
            result, mean_values[mask], spread_values[mask]
        )
        scores[mask] = csgd_crps(mean, std, shift, obs_values[mask])
    return scores


def main() -> int:
    args = parse_args()
    paths = resolve_result_paths(
        results_dir=args.results_dir,
        filenames={
            "scores": "tail_repair_qm_emos.csv",
            "paired": "tail_repair_qm_emos_paired.csv",
        },
        overrides={
            "scores": str(args.out_csv) if args.out_csv else None,
            "paired": str(args.out_paired_csv) if args.out_paired_csv else None,
        },
    )
    guard_result_overwrites(list(paths.values()), force=args.force)

    baseline_paths = resolve_store_paths(
        args.baseline_stores, None, DEFAULT_BASELINE_DAILY_STORES, None
    )
    lagged_paths = resolve_store_paths(
        args.lagged_stores, None, DEFAULT_LAGGED_DAILY_STORES, None
    )

    obs = open_multi_season(baseline_paths, group="imd_observed").load()
    climatology_rain = load_climatology(args.climatology).load()["rain"]
    region_labels = assign_regions(obs["latitude"].values, obs["longitude"].values)

    print(f"EMOS-CSG on QM-corrected members (Step 11), leads {LEAD_HOURS}")
    print(f"  baseline stores: {len(baseline_paths)} season(s)")
    print(f"  lagged stores  : {len(lagged_paths)} season(s)")

    rows: list[dict] = []
    per_day: dict[tuple[str, int], pd.DataFrame] = {}

    for lead_hours in LEAD_HOURS:
        forecasts = load_graphcast_ensemble(lagged_paths, lead_hours)
        sample_times = pd.DatetimeIndex(forecasts["sample"].values)
        obs_aligned = obs["rain"].reindex(time=sample_times).rename(time="sample")

        # A day with no observation anywhere in the box contributes no score,
        # and keeping it would let the EMOS regression train on NaNs.
        has_obs = ~obs_aligned.isnull().all(dim=["latitude", "longitude"])
        forecasts = forecasts.isel(sample=has_obs.values)
        obs_aligned = obs_aligned.isel(sample=has_obs.values)
        sample_times = sample_times[has_obs.values]

        rain_bin_labels = classify_rain_bin(
            forecasts.mean(dim="member", skipna=True)
        )
        folds = list(
            iter_evaluation_folds(sample_times, test_fraction=args.test_fraction)
        )

        for train_mask, test_mask, split_label in folds:
            train_obs = obs_aligned.isel(sample=train_mask)
            arm_forecasts = {"raw_members": forecasts}
            # The climatology covers both test seasons, so the test year is
            # removed before it is used as a reference.
            fold_climatology = climatology_excluding_years(
                climatology_rain, sample_times[test_mask]
            )
            for variant, reference in (
                ("same_period", train_obs),
                ("climatology", fold_climatology),
            ):
                arm_forecasts[f"qm_members_{variant}"] = quantile_mapped_members(
                    forecasts, train_mask, reference, region_labels, lead_hours
                )

            fold_crps: dict[str, float] = {}
            for arm, arm_forecast in arm_forecasts.items():
                emos_results = fit_emos_csg(
                    arm_forecast,
                    obs_aligned,
                    rain_bin_labels,
                    train_mask,
                    source=ENSEMBLE_SOURCE,
                )
                test_forecast = arm_forecast.isel(sample=test_mask)
                test_labels = rain_bin_labels.isel(sample=test_mask)
                test_obs = obs_aligned.isel(sample=test_mask)
                mean_values = test_forecast.mean(dim="member", skipna=True).values
                # ddof=1, matching the spread `fit_emos_csg` was fitted on.
                spread_values = test_forecast.std(dim="member", ddof=1, skipna=True).values
                obs_values = test_obs.values
                labels = test_labels.values

                cells = crps_grid(
                    emos_results, labels, mean_values, spread_values, obs_values
                )
                crps_mm = float(np.nanmean(cells))
                fold_crps[arm] = crps_mm

                probability_grids = {
                    threshold: fitted_probability_grid(
                        emos_results, labels, mean_values, spread_values, threshold
                    )
                    for threshold in THRESHOLDS
                }

                template = test_obs
                per_day_frame = per_day_scores(
                    forecast=test_forecast.mean(dim="member", skipna=True),
                    obs=test_obs,
                    fold=split_label,
                    probabilities={
                        threshold: _as_data_array(grid, template)
                        for threshold, grid in probability_grids.items()
                    },
                    per_cell_scores={
                        "crps_mm": _as_data_array(cells, template, "crps_mm")
                    },
                )
                per_day[(arm, lead_hours)] = pd.concat(
                    [per_day.get((arm, lead_hours), pd.DataFrame()), per_day_frame],
                    ignore_index=True,
                )

                n_cells = int(np.sum(~np.isnan(cells)))
                rmse_mm = float(
                    np.sqrt(np.nanmean((mean_values - obs_values) ** 2))
                )
                for threshold in THRESHOLDS:
                    grid = probability_grids[threshold]
                    valid = ~np.isnan(grid) & ~np.isnan(obs_values)
                    brier = float(
                        np.nanmean(
                            (grid[valid] - (obs_values[valid] >= threshold).astype(float))
                            ** 2
                        )
                    )
                    sedi_value = probability_sedi(grid, obs_values, threshold)
                    rows.append(
                        {
                            "lead_hours": lead_hours,
                            "split_label": split_label,
                            "arm": arm,
                            "threshold_mm": threshold,
                            "n_test_days": int(test_mask.sum()),
                            "n_cells": n_cells,
                            "crps_mm": crps_mm,
                            "brier": brier,
                            "sedi": sedi_value,
                            "rmse_mm": rmse_mm,
                        }
                    )

            print(
                f"  [lead {lead_hours:>3}h | fold {split_label}] "
                + "  ".join(f"{arm}={fold_crps[arm]:.4f}" for arm in ARMS),
                flush=True,
            )

    frame = pd.DataFrame(rows)
    for path in paths.values():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths["scores"], index=False)

    paired_rows = []
    for lead_hours in LEAD_HOURS:
        for arm in ARMS:
            if arm == "raw_members":
                continue
            base = _crps_by_day(per_day, "raw_members", lead_hours)
            arm_days = _crps_by_day(per_day, arm, lead_hours)
            if len(base) != len(arm_days) or not len(base):
                continue
            # Paired A-vs-B form: the QM arm as A, raw members as B, on the
            # same days in the same order. Lower CRPS is better, so a wholly
            # negative interval means the QM arm improved.
            ci = paired_difference_ci(arm_days, base)
            paired_rows.append(
                {
                    "lead_hours": lead_hours,
                    "arm": arm,
                    "n_days": len(arm_days),
                    "mean_crps_delta_mm": ci.estimate,
                    "ci_low": ci.ci_lo,
                    "ci_high": ci.ci_hi,
                    "degenerate": bool(ci.degenerate),
                    "improves": bool(ci.significant and ci.ci_hi < 0),
                }
            )
    paired_frame = pd.DataFrame(paired_rows)
    paired_frame.to_csv(paths["paired"], index=False)

    for (arm, lead_hours), per_day_frame in per_day.items():
        write_per_day_scores(
            f"qmemos_{arm}", lead_hours, per_day_frame, out_dir=args.results_dir
        )

    print(f"\nWrote {paths['scores']} and {paths['paired']}")
    return 0


def _as_data_array(values: np.ndarray, template: xr.DataArray, name=None) -> xr.DataArray:
    return xr.DataArray(values, coords=template.coords, dims=template.dims, name=name)


def _crps_by_day(per_day: dict, arm: str, lead_hours: int) -> np.ndarray:
    frame = per_day.get((arm, lead_hours))
    if frame is None or "crps_mm" not in frame:
        return np.array([])
    return frame["crps_mm"].to_numpy(dtype=float)


if __name__ == "__main__":
    sys.exit(main())
