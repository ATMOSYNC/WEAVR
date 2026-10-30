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

Two member sources, each its own EMOS-CSG, as in Tier 2 (step 11 item 4c
asks for both):

- `graphcast` -- the lagged GraphCast pseudo-ensemble.
- `ifs_ens`   -- the real IFS 50-member ensemble (daily stores, both seasons).

Every lead is loaded through Tier 2's own `align_all_sources`, and rain bins
come from the GraphCast ensemble mean for both sources, as Tier 2 classifies
them. So each source's `raw_members` arm is Tier 2's EMOS for that source, cell
for cell, which the post-run checks assert day by day.

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

H7 is scored here too, per source and arm, by the same rule as
`run_tail_repair_qm.py`: tw-CRPS at 64.5 mm (closed form, `weavr.emos.csgd_twcrps`)
or SEDI at 115.6 mm improves with the CI excluding 0, and Brier at 7.5 mm is
not significantly worse. SEDI, POD, FAR and CSI of a probability forecast come
from its expected contingency counts (`weavr.verify.expected_contingency_counts`).

Outputs
-------
`results/tail_repair_qm_emos.csv`
    Per `(lead, fold, source, arm, threshold)`: CRPS, tw-CRPS@64.5, Brier,
    SEDI, POD, FAR, CSI, RMSE of the member mean EMOS saw, cell counts.
`results/tail_repair_qm_emos_paired.csv`
    Each QM arm minus that source's `raw_members`, per lead, with 95% CIs on
    every metric and the per-lead H7 outcome.
`results/per_day/qmemos_{arm}__lead<L>.csv` (graphcast) and
`results/per_day/qmemos_ifs_ens_{arm}__lead<L>.csv`
    Per-day rows in `weavr.score_io.per_day_scores` schema, so
    `scripts/run_scorecard.py` can bootstrap them, plus the expected counts
    (`prob_hits_<t>` ...) the probability SEDI is pooled from.
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
from run_scorecard import (  # noqa: E402
    _bootstrap_categorical_difference,
    _pooled_categorical,
)
from run_tier2_hierarchical_baseline import (  # noqa: E402
    align_all_sources,
    load_graphcast_ensemble,
    load_hres_forecast,
    load_ifs_ensemble,
)

from weavr.emos import (  # noqa: E402
    csgd_crps,
    csgd_twcrps,
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
    DEFAULT_IFS_DAILY_STORES,
    DEFAULT_LAGGED_DAILY_STORES,
    open_multi_season,
    resolve_store_paths,
)
from weavr.verify import expected_contingency_counts  # noqa: E402

LEAD_HOURS = [24, 48, 72, 96, 120]
THRESHOLDS = (7.5, 35.5, 64.5, 115.6, 204.5)
ARMS = ["raw_members", "qm_members_same_period", "qm_members_climatology"]
ENSEMBLE_SOURCE = "graphcast"
SOURCES = ["graphcast", "ifs_ens"]
TWCRPS_THRESHOLD = 64.5
BRIER_THRESHOLD = 7.5
SEDI_THRESHOLD = 115.6
MIN_LEADS_FOR_PASS = 3
# POD/FAR/CSI/SEDI thresholds step 11's prompt names.
CATEGORICAL_THRESHOLDS = (64.5, 115.6, 204.5)
CATEGORICAL_SCORE_NAMES = ("pod", "far", "csi", "sedi")
COUNT_KEYS = ("hits", "misses", "false_alarms", "correct_negatives")


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-stores", nargs="+", type=Path, default=None)
    parser.add_argument("--lagged-stores", nargs="+", type=Path, default=None)
    parser.add_argument("--ifs-ensemble-stores", nargs="+", type=Path, default=None)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=SOURCES,
        default=SOURCES,
        help="Member sources to evaluate, each with its own EMOS-CSG (default: both)",
    )
    parser.add_argument(
        "--leads",
        nargs="+",
        type=int,
        choices=LEAD_HOURS,
        default=None,
        help="Subset of leads (default: all). For timing one lead before a full run.",
    )
    parser.add_argument(
        "--climatology",
        type=Path,
        default=Path("data/imd_seeps_climatology_jjas.zarr"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-paired-csv", type=Path, default=None)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help=FORCE_HELP)
    return parser.parse_args(args)


def quantile_mapped_members(
    forecasts: xr.DataArray,
    train_mask: np.ndarray,
    reference: xr.DataArray,
    region_labels: xr.DataArray,
    lead_hours: int,
    source: str = ENSEMBLE_SOURCE,
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
        source=source,
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


def probability_score(
    probability: np.ndarray, obs: np.ndarray, threshold: float, score: str
) -> float:
    """POD, FAR, CSI or SEDI of a probability forecast, from expected counts.

    `weavr.verify.sedi` thresholds forecast *values*, so a probability grid
    passed to it with a threshold in mm never exceeds it and the score is
    meaningless. Expected counts avoid inventing a decision threshold.
    """
    counts = expected_contingency_counts(probability, (obs >= threshold).astype(float))
    return _pooled_categorical(tuple(counts[k] for k in COUNT_KEYS), score)


def probability_sedi(probability: np.ndarray, obs: np.ndarray, threshold: float) -> float:
    """SEDI of a probability forecast (see `probability_score`).

    Until 31 Oct this pooled `run_tail_repair_evt.expected_counts`, which
    swapped misses and false alarms; every probability SEDI written before
    then is wrong.
    """
    return probability_score(probability, obs, threshold, "sedi")


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


def twcrps_grid(
    emos_results: dict,
    labels: np.ndarray,
    mean_values: np.ndarray,
    spread_values: np.ndarray,
    obs_values: np.ndarray,
    threshold: float = TWCRPS_THRESHOLD,
) -> np.ndarray:
    """Per-cell tw-CRPS of the fitted CSGD, on exactly `crps_grid`'s cells."""
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
        scores[mask] = csgd_twcrps(mean, std, shift, obs_values[mask], threshold)
    return scores


def evaluate_arm(
    arm_forecast: xr.DataArray,
    obs_aligned: xr.DataArray,
    rain_bin_labels: xr.DataArray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    source: str,
    split_label: str,
) -> tuple[pd.DataFrame, dict]:
    """Fit EMOS-CSG on one arm's training days and score its test days.

    Returns the per-day frame (the `per_day_scores` schema plus the expected
    counts `prob_<count>_<t>`) and a summary per threshold.
    """
    emos_results = fit_emos_csg(
        arm_forecast,
        obs_aligned,
        rain_bin_labels,
        train_mask,
        source=source,
    )
    test_forecast = arm_forecast.isel(sample=test_mask)
    test_labels = rain_bin_labels.isel(sample=test_mask)
    test_obs = obs_aligned.isel(sample=test_mask)
    member_mean = test_forecast.mean(dim="member", skipna=True)
    mean_values = member_mean.values
    # ddof=1, matching the spread `fit_emos_csg` was fitted on.
    spread_values = test_forecast.std(dim="member", ddof=1, skipna=True).values
    obs_values = test_obs.values
    labels = test_labels.values

    cells = crps_grid(emos_results, labels, mean_values, spread_values, obs_values)
    tw_cells = twcrps_grid(emos_results, labels, mean_values, spread_values, obs_values)
    probability_grids = {
        threshold: fitted_probability_grid(
            emos_results, labels, mean_values, spread_values, threshold
        )
        for threshold in THRESHOLDS
    }

    frame = per_day_scores(
        forecast=member_mean,
        obs=test_obs,
        fold=split_label,
        probabilities={
            threshold: _as_data_array(grid, test_obs)
            for threshold, grid in probability_grids.items()
        },
        per_cell_scores={
            "crps_mm": _as_data_array(cells, test_obs, "crps_mm"),
            f"twcrps_{TWCRPS_THRESHOLD}_mm": _as_data_array(
                tw_cells, test_obs, f"twcrps_{TWCRPS_THRESHOLD}_mm"
            ),
        },
    )
    # Daily expected counts, in day order, so the probability SEDI/POD/FAR/CSI
    # can be bootstrapped like any other categorical score. `per_day_scores`
    # emits one row per test sample in sample order; checked, not assumed.
    sample_dates = pd.DatetimeIndex(test_obs["sample"].values)
    if not np.array_equal(pd.DatetimeIndex(frame["date"]).values, sample_dates.values):
        raise RuntimeError("per_day_scores rows are not in test-sample order")
    for threshold in CATEGORICAL_THRESHOLDS:
        counts = expected_contingency_counts(
            probability_grids[threshold], (obs_values >= threshold).astype(float), axis=(1, 2)
        )
        for key in COUNT_KEYS:
            frame[f"prob_{key}_{threshold}"] = counts[key]

    summary = {
        "n_test_days": int(test_mask.sum()),
        "n_cells": int(np.sum(~np.isnan(cells))),
        "crps_mm": float(np.nanmean(cells)),
        f"twcrps_{TWCRPS_THRESHOLD}_mm": float(np.nanmean(tw_cells)),
        # RMSE of the member mean EMOS was given: what QM does to the inputs.
        "rmse_mm": float(np.sqrt(np.nanmean((mean_values - obs_values) ** 2))),
    }
    per_threshold = {}
    for threshold in THRESHOLDS:
        grid = probability_grids[threshold]
        valid = ~np.isnan(grid) & ~np.isnan(obs_values)
        per_threshold[threshold] = {
            "brier": float(
                np.nanmean(
                    (grid[valid] - (obs_values[valid] >= threshold).astype(float)) ** 2
                )
            ),
            **{
                score: probability_score(grid, obs_values, threshold, score)
                for score in CATEGORICAL_SCORE_NAMES
            },
        }
    return frame, {"summary": summary, "per_threshold": per_threshold}


def _prob_counts(frame: pd.DataFrame, threshold: float) -> tuple[np.ndarray, ...]:
    return tuple(frame[f"prob_{key}_{threshold}"].to_numpy(dtype=float) for key in COUNT_KEYS)


def paired_arm_metrics(
    arm_days: pd.DataFrame, base_days: pd.DataFrame, block_days: int, n_resamples: int, seed: int
) -> dict:
    """One QM arm minus raw members on the same days: every metric with a CI.

    `mean_crps_delta_mm` ... `improves` keep the names and meaning this file
    always wrote. Negative is better for every difference (categorical scores
    use the scorecard's orientation).
    """
    if not arm_days["date"].reset_index(drop=True).equals(
        base_days["date"].reset_index(drop=True)
    ):
        raise RuntimeError("paired arms are not on the same days in the same order")

    def mean_ci(column: str):
        return paired_difference_ci(
            arm_days[column].to_numpy(dtype=float),
            base_days[column].to_numpy(dtype=float),
            block_days=block_days,
            n_resamples=n_resamples,
            seed=seed,
        )

    def categorical_ci(threshold: float, score: str):
        return _bootstrap_categorical_difference(
            _prob_counts(arm_days, threshold),
            _prob_counts(base_days, threshold),
            score,
            block_days,
            n_resamples,
            seed,
        )

    def below_zero(ci) -> bool:
        return bool(np.isfinite(ci.ci_lo) and np.isfinite(ci.ci_hi) and ci.ci_hi < 0.0)

    crps_ci = mean_ci("crps_mm")
    tw_ci = mean_ci(f"twcrps_{TWCRPS_THRESHOLD}_mm")
    brier_ci = mean_ci(f"brier_{BRIER_THRESHOLD}")
    sedi_ci = categorical_ci(SEDI_THRESHOLD, "sedi")
    row = {
        "n_days": len(arm_days),
        "mean_crps_delta_mm": crps_ci.estimate,
        "ci_low": crps_ci.ci_lo,
        "ci_high": crps_ci.ci_hi,
        "degenerate": bool(crps_ci.degenerate),
        "improves": bool(crps_ci.significant and crps_ci.ci_hi < 0),
        f"twcrps_{TWCRPS_THRESHOLD}_delta": tw_ci.estimate,
        f"twcrps_{TWCRPS_THRESHOLD}_ci_lo": tw_ci.ci_lo,
        f"twcrps_{TWCRPS_THRESHOLD}_ci_hi": tw_ci.ci_hi,
        f"twcrps_{TWCRPS_THRESHOLD}_improved": below_zero(tw_ci),
        f"sedi_{SEDI_THRESHOLD}_delta": sedi_ci.estimate,
        f"sedi_{SEDI_THRESHOLD}_ci_lo": sedi_ci.ci_lo,
        f"sedi_{SEDI_THRESHOLD}_ci_hi": sedi_ci.ci_hi,
        f"sedi_{SEDI_THRESHOLD}_improved": below_zero(sedi_ci),
        f"brier_{BRIER_THRESHOLD}_delta": brier_ci.estimate,
        f"brier_{BRIER_THRESHOLD}_ci_lo": brier_ci.ci_lo,
        f"brier_{BRIER_THRESHOLD}_ci_hi": brier_ci.ci_hi,
        f"brier_{BRIER_THRESHOLD}_degraded": bool(
            np.isfinite(brier_ci.ci_lo) and brier_ci.ci_lo > 0.0
        ),
    }
    row["h7_lead_improved"] = bool(
        row[f"twcrps_{TWCRPS_THRESHOLD}_improved"] or row[f"sedi_{SEDI_THRESHOLD}_improved"]
    )
    for threshold in CATEGORICAL_THRESHOLDS:
        for score in CATEGORICAL_SCORE_NAMES:
            key = f"{score}_{threshold}"
            if f"{key}_delta" in row:
                continue
            ci = categorical_ci(threshold, score)
            row.update(
                {f"{key}_delta": ci.estimate, f"{key}_ci_lo": ci.ci_lo, f"{key}_ci_hi": ci.ci_hi}
            )
    return row


def per_day_method_name(source: str, arm: str) -> str:
    """`qmemos_<arm>` for GraphCast (the names this file always wrote), else
    `qmemos_<source>_<arm>`."""
    return f"qmemos_{arm}" if source == ENSEMBLE_SOURCE else f"qmemos_{source}_{arm}"


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
    ifs_paths = resolve_store_paths(
        args.ifs_ensemble_stores, None, DEFAULT_IFS_DAILY_STORES, None
    )

    obs = open_multi_season(baseline_paths, group="imd_observed").load()
    climatology_rain = load_climatology(args.climatology).load()["rain"]
    region_labels = assign_regions(obs["latitude"].values, obs["longitude"].values)
    leads = list(args.leads or LEAD_HOURS)

    print(f"EMOS-CSG on QM-corrected members (Step 11), leads {leads}, sources {args.sources}")
    print(f"  baseline stores: {len(baseline_paths)} season(s)")
    print(f"  lagged stores  : {len(lagged_paths)} season(s)")
    print(f"  IFS stores     : {len(ifs_paths)} season(s)")

    rows: list[dict] = []
    per_day: dict[tuple[str, str, int], list[pd.DataFrame]] = {}

    for lead_hours in leads:
        # Tier 2's own loaders and alignment: the raw arms are then Tier 2's
        # EMOS runs exactly, which the post-run checks compare day by day.
        aligned, obs_aligned = align_all_sources(
            load_graphcast_ensemble(lagged_paths, lead_hours),
            load_ifs_ensemble(ifs_paths, lead_hours),
            load_hres_forecast(baseline_paths, lead_hours),
            obs,
        )
        sample_times = pd.DatetimeIndex(obs_aligned["sample"].values)
        # Bins from the GraphCast ensemble mean, for both sources: Tier 2's rule.
        rain_bin_labels = classify_rain_bin(aligned["graphcast"].mean(dim="member", skipna=True))
        forecasts = {source: aligned[source] for source in args.sources}
        del aligned
        folds = list(iter_evaluation_folds(sample_times, test_fraction=args.test_fraction))

        for train_mask, test_mask, split_label in folds:
            train_obs = obs_aligned.isel(sample=train_mask)
            # The climatology covers both test seasons, so the test year is
            # removed before it is used as a reference.
            fold_climatology = climatology_excluding_years(
                climatology_rain, sample_times[test_mask]
            )
            references = {
                "qm_members_same_period": train_obs,
                "qm_members_climatology": fold_climatology,
            }
            for source, members in forecasts.items():
                fold_crps: dict[str, float] = {}
                for arm in ARMS:
                    # One arm in memory at a time: a mapped copy of the
                    # 50-member IFS ensemble is ~1.7 GB per lead.
                    arm_forecast = (
                        members
                        if arm == "raw_members"
                        else quantile_mapped_members(
                            members, train_mask, references[arm], region_labels,
                            lead_hours, source,
                        )
                    )
                    frame, scores = evaluate_arm(
                        arm_forecast, obs_aligned, rain_bin_labels, train_mask, test_mask,
                        source, split_label,
                    )
                    del arm_forecast
                    per_day.setdefault((source, arm, lead_hours), []).append(frame)
                    fold_crps[arm] = scores["summary"]["crps_mm"]
                    for threshold in THRESHOLDS:
                        rows.append(
                            {
                                "lead_hours": lead_hours,
                                "split_label": split_label,
                                "arm": arm,
                                "threshold_mm": threshold,
                                "n_test_days": scores["summary"]["n_test_days"],
                                "n_cells": scores["summary"]["n_cells"],
                                "crps_mm": scores["summary"]["crps_mm"],
                                "brier": scores["per_threshold"][threshold]["brier"],
                                "sedi": scores["per_threshold"][threshold]["sedi"],
                                "rmse_mm": scores["summary"]["rmse_mm"],
                                "emos_source": source,
                                f"twcrps_{TWCRPS_THRESHOLD}_mm": scores["summary"][
                                    f"twcrps_{TWCRPS_THRESHOLD}_mm"
                                ],
                                "pod": scores["per_threshold"][threshold]["pod"],
                                "far": scores["per_threshold"][threshold]["far"],
                                "csi": scores["per_threshold"][threshold]["csi"],
                            }
                        )
                print(
                    f"  [lead {lead_hours:>3}h | fold {split_label} | {source:>9}] "
                    + "  ".join(f"{arm}={fold_crps[arm]:.4f}" for arm in ARMS),
                    flush=True,
                )
        del forecasts

    frame = pd.DataFrame(rows)
    for path in paths.values():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths["scores"], index=False)

    pooled = {key: pd.concat(frames, ignore_index=True) for key, frames in per_day.items()}
    paired_rows = []
    for source in args.sources:
        for lead_hours in leads:
            base = pooled.get((source, "raw_members", lead_hours))
            for arm in ARMS[1:]:
                arm_days = pooled.get((source, arm, lead_hours))
                if base is None or arm_days is None or not len(base):
                    continue
                paired_rows.append(
                    {
                        "lead_hours": lead_hours,
                        "arm": arm,
                        "emos_source": source,
                        **paired_arm_metrics(
                            arm_days, base, args.block_days, args.n_resamples, args.seed
                        ),
                    }
                )
    paired_frame = pd.DataFrame(paired_rows)
    paired_frame.to_csv(paths["paired"], index=False)

    for (source, arm, lead_hours), per_day_frame in pooled.items():
        write_per_day_scores(
            per_day_method_name(source, arm), lead_hours, per_day_frame,
            out_dir=args.results_dir,
        )

    print("\n" + "=" * 78)
    print("H7 on EMOS-CSG fitted to QM-corrected members -- tw-CRPS@64.5 or SEDI@115.6")
    print("improves at >= 3 of 5 leads (CI excluding 0) and Brier@7.5 does not degrade.")
    print("=" * 78)
    for (source, arm), subset in paired_frame.groupby(["emos_source", "arm"], sort=False):
        improved = int(subset["h7_lead_improved"].sum())
        degraded = int(subset[f"brier_{BRIER_THRESHOLD}_degraded"].sum())
        passes = improved >= MIN_LEADS_FOR_PASS and degraded == 0
        print(
            f"  {source:>9} {arm:>24}: {improved}/{len(subset)} leads improved, "
            f"{degraded} with degraded Brier -> H7 {'PASS' if passes else 'FAIL'}"
        )
    print(f"\nWrote {paths['scores']} and {paths['paired']}")
    return 0


def _as_data_array(values: np.ndarray, template: xr.DataArray, name=None) -> xr.DataArray:
    return xr.DataArray(values, coords=template.coords, dims=template.dims, name=name)


if __name__ == "__main__":
    sys.exit(main())
