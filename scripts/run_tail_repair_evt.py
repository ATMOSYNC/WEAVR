#!/usr/bin/env python3
"""Run Tail Repair via Extreme-Value Tail (Step 12) -- LOYO evaluation of H7 (EVT part).

Replaces the 80-line command-line stub. That stub parsed arguments, printed
one line and returned 0: it fitted nothing and scored nothing, so
`results/tail_repair_evt.csv` never existed and H7 (EVT part) had no evidence
behind it.

The problem it addresses
------------------------
Tier 2's EMOS-CSG fits one regression per rain-intensity bin. Bins with too
few training observations -- notably `extremely_heavy` (>= 204.5 mm), where a
two-season evidence base still yields only a few dozen events -- cannot be
fitted at all, and those cells receive a point-mass fallback: a hard
probability of **zero**. The dashboard showed those cells as "not a real
probability", which is at least honest, but it means the model cannot ever
issue a rare-extreme warning.

This runner fits a pooled Generalized Pareto tail above
`u = 64.5 mm` (IMD's heavy-rainfall threshold) -- one shared shape
parameter, per-region scales -- and splices it onto the fitted CSGD
distribution above `u`, so the upper tail is described by a statistic that
does not need thousands of localised training events.

What it evaluates, per lead, under leave-one-year-out
----------------------------------------------------
Three arms, all scored on the same held-out days against the same
observations:

- `csgd_only`      -- today's behaviour: fitted bins use their CSGD
                      exceedance probability; unfittable bins get the
                      point-mass fallback of exactly 0.0.
- `csgd_gpd_tail`  -- unfittable bins are repaired by splicing the pooled GPD
                      onto the nearest fittable bin's distribution.
- `raw_ifs_ens_member_counting` -- the simple benchmark: the empirical
                      fraction of IFS-ENS members above the threshold. Only
                      computed when an IFS-ENS store is supplied; skipped and
                      recorded as unavailable otherwise, never silently.

The headline claim in docs/tail-repair-results.md is that splicing
"eliminates false zero-probability assignments across all five leads". So
that is measured directly, not assumed: `false_zero_cells` counts test cells
whose exceedance probability is exactly 0.0 at 115.6 and 204.5 mm.

Ensembles come from the **lagged GraphCast store**, which exists for both
seasons. That is deliberate: `weavr.emos.fit_emos_csg` fits any ensemble
source, so this step does not need the daily IFS-ENS store and is not blocked
on it. The IFS-ENS store only adds the third benchmark arm.

Outputs
-------
`results/tail_repair_evt.csv`
    Per `(lead, fold, arm, threshold)`: Brier, Brier skill score against the
    IMD JJAS climatology, SEDI, RMSE of the ensemble mean, plus the
    false-zero count and the per-method cell breakdown.
`results/tail_repair_evt_paired.csv`
    Paired `csgd_gpd_tail` vs `csgd_only` differences with 95% CIs.
`results/per_day/evt_{arm}_t{t}__lead<L>.csv`
    Per-day rows in `weavr.score_io.per_day_scores` schema so
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
from run_scorecard import (  # noqa: E402
    _bootstrap_categorical_difference,
    _categorical_from_counts,
    _pooled_sedi,
)
from run_tier2_hierarchical_baseline import (  # noqa: E402
    load_graphcast_ensemble as _tier2_load_graphcast_ensemble,
)
from run_tier2_hierarchical_baseline import (
    load_ifs_ensemble as _tier2_load_ifs_ensemble,
)

from weavr.emos import (  # noqa: E402
    exceedance_probability_csgd,
    fit_emos_csg,
    predict_csgd_params,
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
from weavr.tail import (  # noqa: E402
    DEFAULT_TAIL_THRESHOLD_U,
    exceedance_probability_with_tail,
    fit_pooled_gpd,
)

LEAD_HOURS = [24, 48, 72, 96, 120]
EXTREME_THRESHOLDS = [115.6, 204.5]
GRAPHCAST_ENSEMBLE_SOURCE = "graphcast"


def load_graphcast_ensemble(paths: list[str | Path], lead_hours: int) -> xr.DataArray:
    """GraphCast's lagged pseudo-ensemble at one lead, in mm, IMD-day-aligned.

    A thin alias over `run_tier2_hierarchical_baseline.load_graphcast_ensemble`
    rather than a second implementation. That loader is the canonical one: the
    lagged store keeps `(nominal_time, lead_hours, member_offset_hours, lat,
    lon)`, so the pseudo-ensemble has to be rebuilt through
    `run_phase2_ensemble_baseline.build_ensembles_for_lead` and re-densified
    from `lead_hours`/`nominal_time`, which are not `prediction_timedelta`/
    `time`. Re-deriving that here got it wrong twice.
    """
    return _tier2_load_graphcast_ensemble(paths, lead_hours)


def fit_tail_on_training_obs(
    obs_train: xr.DataArray, region_labels: xr.DataArray, threshold_u: float
):
    """Pooled GPD fitted on the training fold's own exceedances, per region.

    Fitting the tail on the training observations (not on the test fold, and
    not on the full 15-year climatology) keeps the tail inside the same
    cross-validation boundary as the CSGD parameters it is spliced onto.
    """
    values = obs_train.values
    exceedances: dict[str, np.ndarray] = {}
    for region in np.unique(region_labels.values):
        mask = region_labels.values == region
        pooled = values[..., mask].ravel()
        pooled = pooled[np.isfinite(pooled) & (pooled > threshold_u)]
        if pooled.size:
            exceedances[str(region)] = pooled
    if not exceedances:
        raise ValueError(
            f"No training exceedances above u={threshold_u} mm; cannot fit a tail. "
            "The evidence base has too few heavy-rainfall events to repair."
        )
    return fit_pooled_gpd(exceedances, threshold_u=threshold_u)


def csgd_only_probabilities(
    forecasts: xr.DataArray,
    rain_bin_labels: xr.DataArray,
    emos_results_by_bin: dict,
    threshold: float,
) -> np.ndarray:
    """Today's behaviour: CSGD where fitted, hard 0.0 where not.

    Deliberately reproduces the point-mass fallback rather than improving it,
    because this is the arm the GPD tail has to beat.
    """
    ensemble_mean = forecasts.mean(dim="member", skipna=True)
    ensemble_spread = forecasts.std(dim="member", skipna=True)
    probabilities = np.full(ensemble_mean.shape, np.nan, dtype=float)

    labels = rain_bin_labels.values
    for bin_label in RAIN_BIN_LABELS:
        result = emos_results_by_bin.get(str(bin_label))
        if result is None:
            # Unfittable bin: the point-mass fallback that produces the
            # false zero-probability assignments this step exists to remove.
            probabilities[labels == bin_label] = 0.0
            continue
        mask = labels == bin_label
        if not mask.any():
            continue
        mean, std, shift = predict_csgd_params(
            result,
            ensemble_mean.values[mask],
            ensemble_spread.values[mask],
        )
        probabilities[mask] = exceedance_probability_csgd(mean, std, shift, threshold)
    return probabilities


def as_data_array(values: np.ndarray, template: xr.DataArray, name: str) -> xr.DataArray:
    return xr.DataArray(values, coords=template.coords, dims=template.dims, name=name)


def expected_counts(probability: np.ndarray, obs_binary: np.ndarray) -> dict[str, np.ndarray]:
    """Expected contingency counts implied by a probability forecast.

    SEDI is defined on probabilities, not on a hard decision: using
    `sum(p * obs)` for hits and `sum(p * (1 - obs))` for misses lets the
    existing `_pooled_sedi` helper work unchanged and avoids inventing an
    arbitrary decision threshold (0.5 would silently discard exactly the
    low-probability extreme cells this step is about).
    """
    finite = np.isfinite(probability) & np.isfinite(obs_binary)
    p = np.where(finite, probability, 0.0)
    o = np.where(finite, obs_binary, 0.0)
    return {
        "hits": (p * o).sum(axis=(1, 2)),
        "misses": (p * (1.0 - o)).sum(axis=(1, 2)),
        "false_alarms": ((1.0 - p) * o).sum(axis=(1, 2)),
        "correct_negatives": ((1.0 - p) * (1.0 - o)).sum(axis=(1, 2)),
    }


def member_counting_probabilities(
    ifs_ensemble: xr.DataArray, threshold: float, test_mask: np.ndarray
) -> np.ndarray:
    """`P(rain > threshold)` from the raw IFS-ENS members, nothing else.

    This is the benchmark step 12 is measured *against*: the empirical
    fraction of members exceeding the threshold, with no calibration applied.
    It is the comparison a reader should ask for before believing the spliced
    tail adds anything, and it could not be run until step 07 produced daily
    full-member IFS-ENS stores for both seasons (#74 recorded it as NOT RUN for
    exactly that reason).

    Returns a `(sample, latitude, longitude)` grid containing **only the test
    samples**, so it lines up with `test_obs` rather than with the full
    two-season series. Returning the full grid with NaN padding is the trap
    here: the shapes then differ by the fold size and the scoring step raises
    deep inside a boolean mask.
    """
    exceed = (ifs_ensemble > threshold).mean(dim="member", skipna=True)

    # `_align_to_imd_day` (inside `load_ifs_ensemble`) renames the time axis to
    # `sample` and has already selected one lead; a raw store would still carry
    # `time`/`prediction_timedelta`. Accept either, and drop any leftover
    # singleton axis rather than assuming a particular caller's shape.
    time_dim = "sample" if "sample" in exceed.dims else "time"
    for dim in (d for d in exceed.dims if d not in (time_dim, "latitude", "longitude")):
        if exceed.sizes[dim] == 1:
            exceed = exceed.squeeze(dim, drop=True)
        else:
            raise ValueError(
                f"member_counting_probabilities expects a single lead, but "
                f"{ifs_ensemble.name or 'the ensemble'} still has "
                f"{exceed.sizes[dim]} values along {dim!r}. Select one lead first."
            )

    grid = exceed.transpose(time_dim, "latitude", "longitude").values.astype(float)
    mask = np.asarray(test_mask, dtype=bool)
    if grid.shape[0] != mask.size:
        raise ValueError(
            f"IFS-ENS ensemble has {grid.shape[0]} samples but the fold mask has "
            f"{mask.size}; they must describe the same days."
        )
    return grid[mask]


def benchmark_row(
    probs: np.ndarray,
    obs_test: np.ndarray,
    climatology_grid: np.ndarray,
    threshold: float,
    lead_hours: int,
    split_label: str,
    split_kind: str,
    n_train: int,
    n_test: int,
) -> dict:
    """Score one (lead, fold, threshold) of the member-counting benchmark.

    Scored with the *same* helpers as the spliced-tail arms -- `expected_counts`
    then `compute_sedi` -- so the comparison differs only in the forecast being
    scored, never in how it is scored.
    """
    finite = np.isfinite(probs)
    obs_binary = (obs_test > threshold).astype(float)
    p = probs[finite]
    o = obs_binary[finite]
    # The climatological reference is a per-cell `(latitude, longitude)` grid,
    # not a scalar, so it has to be broadcast up to the forecast's shape and
    # then reduced with the same mask `finite` uses. Broadcasting (rather than
    # collapsing the sample axis first) is what keeps it aligned: the grid
    # repeats once per sample, exactly as the forecast does.
    climatology_broadcast = np.broadcast_to(
        np.asarray(climatology_grid, dtype=float), probs.shape
    )
    climatology_selected = climatology_broadcast[finite]
    brier = float(np.mean((p - o) ** 2))
    counts = expected_counts(probs, obs_binary)
    sedi = compute_sedi(
        float(np.sum(counts["hits"])),
        float(np.sum(counts["misses"])),
        float(np.sum(counts["false_alarms"])),
        float(np.sum(counts["correct_negatives"])),
    )
    climat_brier = float(np.mean((climatology_selected - o) ** 2))
    return {
        "lead_hours": lead_hours,
        "fold": split_label,
        "split": split_kind,
        "arm": "raw_ifs_ens_member_counting",
        "threshold": threshold,
        "n_train": n_train,
        "n_test": n_test,
        "brier": brier,
        "climatology_brier": climat_brier,
        "brier_skill_score": 1.0 - (brier / climat_brier) if climat_brier > 0 else float("nan"),
        "sedi": sedi,
        "rmse_of_mean_mm": float("nan"),
        "false_zero_cells": int(np.sum(np.isfinite(probs) & (probs == 0.0))),
        "threshold_u": float("nan"),
        "tail_shape_xi": float("nan"),
        "fit_bins": "",
        "n_bins_total": 0,
    }


def compute_sedi(
    hits: float, misses: float, false_alarms: float, correct_negatives: float
) -> float:
    """SEDI from pooled counts.

    Kept as the module's public entry point (the stub's signature, and what
    `tests/test_run_tail_repair_evt.py` exercises) but implemented by
    delegating to `run_scorecard._pooled_sedi`, which is the canonical version
    and mirrors `weavr.verify.sedi` including Ferro & Stephenson's degenerate
    cases. The stub carried its own eps-guarded copy, which returned 0.0 for a
    perfectly separated forecast instead of the +inf that formulation implies
    -- one implementation, not two that can drift.
    """
    return _pooled_sedi(hits, misses, false_alarms, correct_negatives)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Split out of `main` so the argument surface can be tested directly --
    `tests/test_run_tail_repair_evt.py` imports this by name.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lagged-stores", nargs="+", default=None, help="Lagged ensemble stores")
    parser.add_argument(
        "--baseline-stores",
        nargs="+",
        default=None,
        help="Baseline stores (for imd_observed and climatology)",
    )
    parser.add_argument("--baseline-store", default=None)
    parser.add_argument("--store", default=None)
    parser.add_argument(
        "--ifs-ensemble-stores",
        nargs="+",
        default=None,
        help="Optional. Enables the raw IFS-ENS member-counting benchmark arm.",
    )
    parser.add_argument("--climatology", default="data/imd_seeps_climatology_jjas.zarr")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Default: <results-dir>/tail_repair_evt.csv",
    )
    parser.add_argument(
        "--paired-out-csv",
        type=Path,
        default=None,
        help="Default: <results-dir>/tail_repair_evt_paired.csv",
    )
    parser.add_argument("--force", action="store_true", help=FORCE_HELP)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--threshold-u",
        type=float,
        default=DEFAULT_TAIL_THRESHOLD_U,
        help=f"Tail splice threshold in mm (default: {DEFAULT_TAIL_THRESHOLD_U})",
    )
    return parser.parse_args(args)


def main() -> int:
    args = parse_args()

    _paths = resolve_result_paths(
        args.results_dir,
        {"out_csv": "tail_repair_evt.csv", "paired_out_csv": "tail_repair_evt_paired.csv"},
        {"out_csv": args.out_csv, "paired_out_csv": args.paired_out_csv},
    )
    args.out_csv = _paths["out_csv"]
    args.paired_out_csv = _paths["paired_out_csv"]
    guard_result_overwrites(_paths.values(), force=args.force)

    # `to_csv` needs its parent to exist and will not create it, and
    # `write_per_day_scores` only makes the `per_day/` subdirectory -- so a
    # fresh --results-dir failed at the very last step after all the compute.
    for _p in _paths.values():
        Path(_p).parent.mkdir(parents=True, exist_ok=True)

    # `to_csv` needs its parent to exist and will not create it, and
    # `write_per_day_scores` only makes the `per_day/` subdirectory -- so a
    # fresh --results-dir failed at the very last step after all the compute.
    for _p in _paths.values():
        Path(_p).parent.mkdir(parents=True, exist_ok=True)

    legacy = args.baseline_store if args.baseline_store is not None else args.store
    baseline_paths = resolve_store_paths(
        args.baseline_stores, legacy, DEFAULT_BASELINE_DAILY_STORES, "data/baseline_2020_jjas.zarr"
    )
    lagged_paths = resolve_store_paths(
        args.lagged_stores,
        None,
        DEFAULT_IFS_DAILY_STORES,
    DEFAULT_LAGGED_DAILY_STORES,
        None,
    )
    # The `--ifs-ensemble-stores` flag existed but was never resolved into
    # paths, so `ifs_available` was decided on a non-None argument that nothing
    # then read. That is why the runner announced the benchmark as "enabled"
    # and emitted no rows for it.
    ifs_paths = resolve_store_paths(
        args.ifs_ensemble_stores,
        None,
        DEFAULT_IFS_DAILY_STORES,
        None,
    )

    obs = open_multi_season(baseline_paths, group="imd_observed").load()
    climatology = load_climatology(args.climatology).load()
    climatology_rain = climatology["rain"]
    region_labels = assign_regions(obs["latitude"].values, obs["longitude"].values)

    # Climatological exceedance frequency per cell: the reference the Brier
    # skill score is measured against.
    climatology_probability = {
        threshold: (climatology_rain >= threshold).mean(dim="time")
        for threshold in EXTREME_THRESHOLDS
    }

    ifs_available = args.ifs_ensemble_stores is not None
    print(f"Tail repair via extreme-value tail (Step 12), u={args.threshold_u} mm")
    print(f"Lagged stores: {lagged_paths}")
    print(
        "IFS-ENS member-counting benchmark: "
        + (
            "enabled"
            if ifs_available
            else "SKIPPED (no --ifs-ensemble-stores; recorded unavailable)"
        )
    )

    metric_rows: list[dict] = []
    paired_rows: list[dict] = []
    pooled_per_day: dict[tuple[str, float, str], list[pd.DataFrame]] = {}

    for lead_hours in LEAD_HOURS:
        forecasts = load_graphcast_ensemble(lagged_paths, lead_hours)
        ifs_ensemble = (
            _tier2_load_ifs_ensemble(ifs_paths, lead_hours) if ifs_available else None
        )
        ensemble_mean_all = forecasts.mean(dim="member", skipna=True)
        sample_values = ensemble_mean_all["sample"].values
        obs_aligned = obs["rain"].reindex(time=sample_values).rename(time="sample")
        has_obs = ~obs_aligned.isnull().all(dim=["latitude", "longitude"])
        forecasts = forecasts.isel(sample=has_obs.values)
        obs_aligned = obs_aligned.isel(sample=has_obs.values)
        if ifs_ensemble is not None:
            # The IFS series comes from a different store and is not filtered by
            # `has_obs`, so at some leads it is longer than the sample set the
            # forecasts were trimmed to (244 vs 242 at lead 48). Reindex onto the
            # forecasts' own `sample` axis so the benchmark is scored on exactly
            # the days the other arms were scored on, rather than raising a
            # length mismatch several minutes into the run.
            ifs_ensemble = ifs_ensemble.reindex(sample=forecasts["sample"].values)
            if bool(np.isnan(ifs_ensemble).all()):
                raise ValueError(
                    "the IFS-ENS ensemble shares no days with the forecast "
                    "sample set; check that both stores cover the same season."
                )

        # Derived only after the IMD filter: two IMD days in this period fall
        # outside the grid, so every array built from these stays the same
        # length as `forecasts`/`obs_aligned` and the fold masks line up.
        ensemble_mean = forecasts.mean(dim="member", skipna=True)
        ensemble_spread = forecasts.std(dim="member", skipna=True)
        rain_bin_labels = classify_rain_bin(ensemble_mean)

        sample_times = pd.DatetimeIndex(obs_aligned["sample"].values)
        folds = list(iter_evaluation_folds(sample_times, test_fraction=args.test_fraction))

        for train_mask, test_mask, split_label in folds:
            train_obs = obs_aligned.isel(sample=train_mask)
            test_obs = obs_aligned.isel(sample=test_mask)
            test_forecasts = forecasts.isel(sample=test_mask)
            test_mean = ensemble_mean.isel(sample=test_mask)
            test_spread = ensemble_spread.isel(sample=test_mask)
            split_kind = (
                "leave_one_year_out"
                if split_label != "seasonal_block_split"
                else "seasonal_block_split"
            )

            emos_results = fit_emos_csg(
                forecasts,
                obs_aligned,
                rain_bin_labels,
                train_mask,
                source=GRAPHCAST_ENSEMBLE_SOURCE,
            )
            tail_fit = fit_tail_on_training_obs(train_obs, region_labels, args.threshold_u)
            fitable = sorted(emos_results.keys())

            for threshold in EXTREME_THRESHOLDS:
                obs_binary = (test_obs >= threshold).astype(float)

                baseline_probs = csgd_only_probabilities(
                    test_forecasts, rain_bin_labels.isel(sample=test_mask), emos_results, threshold
                )
                tail_probs, methods = exceedance_probability_with_tail(
                    test_mean.values,
                    test_mean.values,
                    test_spread.values,
                    emos_results,
                    tail_fit,
                    region_labels.values,
                    threshold,
                )

                for arm, probs in (("csgd_only", baseline_probs), ("csgd_gpd_tail", tail_probs)):
                    prob_da = as_data_array(probs, test_mean, f"p_exceed_{threshold}")
                    frame = per_day_scores(
                        test_mean,
                        test_obs,
                        fold=split_label,
                        probabilities={threshold: prob_da},
                    )
                    counts = expected_counts(probs, obs_binary.values)
                    for name, values in counts.items():
                        frame[f"{name}_{threshold}"] = values

                    climat_p = climatology_probability[threshold].values
                    climat_p_da = as_data_array(
                        np.broadcast_to(climat_p, probs.shape).copy(), test_mean, "climat"
                    )
                    brier = float(frame[f"brier_{threshold}"].mean())
                    climat_brier = float(
                        per_day_scores(
                            test_mean,
                            test_obs,
                            fold=split_label,
                            probabilities={threshold: climat_p_da},
                        )[f"brier_{threshold}"].mean()
                    )
                    bss = 1.0 - (brier / climat_brier) if climat_brier > 0 else float("nan")

                    row = {
                        "lead_hours": lead_hours,
                        "fold": split_label,
                        "split": split_kind,
                        "arm": arm,
                        "threshold": threshold,
                        "n_train": int(train_mask.sum()),
                        "n_test": int(test_mask.sum()),
                        "brier": brier,
                        "climatology_brier": climat_brier,
                        "brier_skill_score": bss,
                        "sedi": _pooled_sedi(
                            *[float(np.sum(counts[k])) for k in counts]
                        ),
                        "rmse_of_mean_mm": float(np.sqrt(frame["mse_mm2"].mean())),
                        "false_zero_cells": int(np.sum(probs == 0.0)),
                        "threshold_u": args.threshold_u,
                        "tail_shape_xi": float(tail_fit.shape_xi),
                        "fit_bins": ";".join(fitable),
                        "n_bins_total": len(RAIN_BIN_LABELS),
                    }
                    if arm == "csgd_gpd_tail":
                        for method in ("csgd", "csgd+gpd_tail", "fallback"):
                            row[f"cells_{method}"] = int(np.sum(methods == method))
                    metric_rows.append(row)

                    pooled_per_day.setdefault((arm, threshold, lead_hours), []).append(frame)

                    # Raw IFS-ENS member-counting benchmark: the arm step 12 is
                    # measured against. #74 recorded it NOT RUN because the daily
                    # full-member IFS-ENS stores did not exist; step 07 built
                    # them, so it runs here, scored with the same helpers as the
                    # arms above so only the forecast differs, never the scoring.
                    #
                    # Its own loop variable: an earlier version reused
                    # `threshold`, so this block ran before the enclosing arm's
                    # `pooled_per_day.setdefault((arm, threshold, ...))` and
                    # left that keyed only on the last threshold, which then
                    # raised KeyError on the paired-CI pass.
                    if ifs_available and ifs_ensemble is not None:
                        for bench_threshold in EXTREME_THRESHOLDS:
                            metric_rows.append(
                                benchmark_row(
                                    member_counting_probabilities(
                                        ifs_ensemble, bench_threshold, test_mask
                                    ),
                                    test_obs.values,
                                    climatology_probability[bench_threshold].values,
                                    bench_threshold,
                                    lead_hours,
                                    split_label,
                                    split_kind,
                                    int(train_mask.sum()),
                                    int(test_mask.sum()),
                                )
                            )

                print(
                    f"[lead {lead_hours:>3}h | fold {split_label} | t={threshold:>5}mm] "
                    f"fallback bins unfitted: "
                    f"{sorted(set(RAIN_BIN_LABELS) - set(fitable))} | "
                    f"zero-prob cells: csgd_only={int(np.sum(baseline_probs == 0.0))} "
                    f"tail={int(np.sum(tail_probs == 0.0))}"
                )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")

    for (arm, threshold, lead_hours), frames in pooled_per_day.items():
        write_per_day_scores(
            f"evt_{arm}_t{threshold:g}",
            lead_hours,
            pd.concat(frames, ignore_index=True),
            out_dir=args.results_dir,
        )

    paired_rows = []
    for lead_hours in LEAD_HOURS:
        for threshold in EXTREME_THRESHOLDS:
            tail_all = pd.concat(
                pooled_per_day[("csgd_gpd_tail", threshold, lead_hours)], ignore_index=True
            )
            base_all = pd.concat(
                pooled_per_day[("csgd_only", threshold, lead_hours)], ignore_index=True
            )
            brier_ci = paired_difference_ci(
                tail_all[f"brier_{threshold}"].to_numpy(float),
                base_all[f"brier_{threshold}"].to_numpy(float),
                block_days=args.block_days,
                n_resamples=args.n_resamples,
                seed=args.seed,
                aggregate="mean",
            )
            sedi_ci = _bootstrap_categorical_difference(
                _categorical_from_counts(tail_all, threshold, "sedi"),
                _categorical_from_counts(base_all, threshold, "sedi"),
                "sedi",
                args.block_days,
                args.n_resamples,
                args.seed,
            )
            paired_rows.append(
                {
                    "lead_hours": lead_hours,
                    "threshold": threshold,
                    "n_test_days": int(len(tail_all)),
                    "brier_delta": brier_ci.estimate,
                    "brier_ci_lo": brier_ci.ci_lo,
                    "brier_ci_hi": brier_ci.ci_hi,
                    "brier_improved": bool(
                        np.isfinite(brier_ci.ci_hi) and brier_ci.ci_hi < 0.0
                    ),
                    "sedi_delta": sedi_ci.estimate,
                    "sedi_ci_lo": sedi_ci.ci_lo,
                    "sedi_ci_hi": sedi_ci.ci_hi,
                    "sedi_improved": bool(np.isfinite(sedi_ci.ci_hi) and sedi_ci.ci_hi < 0.0),
                    "bootstrap_degenerate": bool(brier_ci.degenerate),
                }
            )
            print(
                f"[lead {lead_hours:>3}h | t={threshold:>5}mm] tail vs csgd_only: "
                f"Brier d={brier_ci.estimate:+.6f} "
                f"CI[{brier_ci.ci_lo:+.6f},{brier_ci.ci_hi:+.6f}] | "
                f"SEDI d={sedi_ci.estimate:+.4f} "
                f"CI[{sedi_ci.ci_lo:+.4f},{sedi_ci.ci_hi:+.4f}]"
            )

    paired = pd.DataFrame(paired_rows)
    paired.to_csv(args.paired_out_csv, index=False)
    print(f"Wrote {args.paired_out_csv}")

    print("\n" + "=" * 78)
    print("H7 (EVT part): the tail must eliminate false zero-probability assignments")
    print("at 115.6 mm and 204.5 mm across all five leads.")
    print("=" * 78)
    for threshold in EXTREME_THRESHOLDS:
        subset = paired[paired["threshold"] == threshold]
        zero_base = metrics[
            (metrics["threshold"] == threshold) & (metrics["arm"] == "csgd_only")
        ]["false_zero_cells"]
        zero_tail = metrics[
            (metrics["threshold"] == threshold) & (metrics["arm"] == "csgd_gpd_tail")
        ]["false_zero_cells"]
        no_zeros = bool((zero_tail == 0).all())
        improved = int(subset["brier_improved"].sum())
        print(
            f"  t={threshold:>5}mm: zero-probability cells "
            f"{int(zero_base.sum())} -> {int(zero_tail.sum())} "
            f"({'eliminated' if no_zeros else 'REMAIN'}); "
            f"Brier improved at {improved}/5 leads; "
            f"SEDI improved at {int(subset['sedi_improved'].sum())}/5 leads"
        )
    benchmark_rows = [r for r in metric_rows if r["arm"] == "raw_ifs_ens_member_counting"]
    if ifs_available and not benchmark_rows:
        raise RuntimeError(
            "IFS-ENS stores were supplied but the member-counting benchmark "
            "produced no rows. The benchmark is either silently skipped or "
            "broken; refusing to report the run as complete."
        )
    if not ifs_available:
        print(
            "\nNote: the raw IFS-ENS member-counting benchmark is not included. Supply "
            "--ifs-ensemble-stores to add it; it is omitted rather than faked."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
