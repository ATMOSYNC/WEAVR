#!/usr/bin/env python3
"""Run Tail Repair via Quantile Mapping (Step 11) -- LOYO evaluation of H7 (QM part).

Replaces the 51-line command-line stub. That stub parsed arguments, printed
one line and returned 0: it evaluated nothing, so `results/tail_repair_qm.csv`
was never produced and H7 (QM part) had no evidence behind it. This is the
real runner.

What it evaluates, per lead time, under leave-one-year-out cross-validation
(`weavr.splits.iter_evaluation_folds`):

- **Raw vs QM, each single source.** The three baseline-store sources that
  carry `total_precipitation_24hr` -- graphcast, hres, ifs_ens_mean (pangu
  has no precipitation at all, per docs/baseline-store.md).
- **Raw vs QM, the Tier 1 regional-weights blend.** Weights are refitted on
  the QM-corrected training forecasts so the comparison isolates the effect of
  the mapping rather than confounding it with a change of weights.
- **Both reference variants**, as the issue asks:
  - `same_period`  -- the map is trained against the training period's own
    IMD observations.
  - `climatology`  -- the map is trained against the 15-year IMD JJAS
    climatology, which is what you would use for a day with no observations
    of its own.

Two things this deliberately does not do:

- It does not fit on any test day. Maps are fitted on the training fold only
  and then applied to the held-out year.
- It does not select a variant by looking at the test score. Both are
  reported, and the H7 verdict is computed per variant.

H7 (QM part) is the pre-registered criterion in
docs/tail-repair-results.md: tw-CRPS at 64.5 mm **or** SEDI at 115.6 mm must
improve at >= 3 of 5 lead times with the 95% paired block-bootstrap CI
strictly excluding 0, while Brier at 7.5 mm does not degrade significantly.
The RMSE trade-off is reported alongside, because lifting the upper tail
raises MSE slightly by construction and that should be visible rather than
buried.

Outputs
-------
`results/tail_repair_qm.csv`
    Per `(variant, lead, fold, scope, source, method)`, raw rows included for
    every single source as well as the blend: RMSE, MAE, CRPS, tw-CRPS@64.5,
    Brier@7.5, and POD/FAR/CSI/SEDI pooled over the fold at 64.5, 115.6 and
    204.5 mm. `crps_mm` equals `mae_mm` by identity: every source here is a
    point forecast, and the CRPS of a point mass at x is |y - x|.
`results/tail_repair_qm_paired.csv`
    QM minus raw, for the blend and for each single source, pooled over both
    test folds: every metric above with its 95% paired block-bootstrap CI,
    plus the per-lead H7 outcome. The pre-registered H7 verdict is the
    blend's; the single-source rows show where the effect comes from.
`results/per_day/<method>__lead<lead>.csv`
    Per-day rows for both arms, in the schema `weavr.score_io.per_day_scores`
    emits so `scripts/run_scorecard.py` can bootstrap them.

Scope: precipitation only. Every source carries `2m_temperature` but the
store's `imd_observed` group holds `rain` alone, so there is no
matching-resolution temperature ground truth to score against -- the same
limitation run_tier0_baseline.py and run_phase2_ensemble_baseline.py state.
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
    _pooled_categorical,
    _rmse_from_mse,
)
from run_tier0_baseline import PRECIP_M_TO_MM, _align_to_imd_day  # noqa: E402
from run_tier1_regional_baseline import (  # noqa: E402
    FORECAST_SOURCE_NAMES,
    blend_with_region_weights,
    build_region_weight_grid,
)

from weavr.quantile_mapping import (  # noqa: E402
    apply_regional_quantile_maps,
    climatology_excluding_years,
    fit_regional_quantile_maps,
)
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
    open_multi_season,
    resolve_store_paths,
)
from weavr.weighting import fit_region_weights  # noqa: E402

LEAD_HOURS = [24, 48, 72, 96, 120]
VARIANTS = ["same_period", "climatology"]
TWCRPS_THRESHOLD = 64.5
BRIER_THRESHOLD = 7.5
SEDI_THRESHOLD = 115.6
MIN_LEADS_FOR_PASS = 3
# The heavy, very heavy and extremely heavy IMD thresholds, which step 11's
# prompt asks POD/FAR/CSI/SEDI at.
CATEGORICAL_THRESHOLDS = (64.5, 115.6, 204.5)
CATEGORICAL_SCORE_NAMES = ("pod", "far", "csi", "sedi")


def load_aligned(
    sources: dict[str, xr.Dataset],
    obs: xr.Dataset,
    lead_hours: int,
) -> tuple[dict[str, xr.DataArray], xr.DataArray]:
    """Unit-convert, IMD-day-align and reindex every source at one lead.

    Identical convention to `run_tier1_regional_baseline.load_aligned_forecasts_and_obs`,
    reused rather than reimplemented: an outer join so a source missing a
    sample gets NaN there instead of being silently dropped, then restricted
    to samples with real IMD ground truth.
    """
    raw_aligned = {
        name: _align_to_imd_day(
            sources[name]["total_precipitation_24hr"]
            .sel(prediction_timedelta=lead_hours)
            .load()
            * PRECIP_M_TO_MM,
            lead_hours,
        )
        for name in FORECAST_SOURCE_NAMES
    }
    aligned = xr.align(*raw_aligned.values(), join="outer")
    forecasts = dict(zip(raw_aligned.keys(), aligned, strict=True))

    sample_values = forecasts[FORECAST_SOURCE_NAMES[0]]["sample"].values
    obs_aligned = obs["rain"].reindex(time=sample_values).rename(time="sample")
    has_obs = ~obs_aligned.isnull().all(dim=["latitude", "longitude"])
    forecasts = {name: da.isel(sample=has_obs.values) for name, da in forecasts.items()}
    obs_aligned = obs_aligned.isel(sample=has_obs.values)
    return forecasts, obs_aligned


def fit_maps_for_variant(
    forecast_train: xr.DataArray,
    train_obs: xr.DataArray,
    region_labels: xr.DataArray,
    climatology_rain: xr.DataArray,
    variant: str,
    lead_hours: int,
    source: str,
) -> dict:
    """Fit per-region quantile maps on the training fold only.

    `fit_regional_quantile_maps` ravels the forecast and reference arrays
    independently, so the reference needs matching spatial dims but need not
    match the forecast's sample count. That is what lets the climatology
    variant pass all 1830 JJAS days as the reference distribution rather
    than forcing an artificial one-value-per-cell reference.
    """
    reference = train_obs if variant == "same_period" else climatology_rain
    return fit_regional_quantile_maps(
        forecast_train,
        reference,
        region_labels,
        sample_dim="sample",
        source=source,
        lead=lead_hours,
    )


def pooled_categorical(frame: pd.DataFrame, threshold: float, score: str) -> float:
    """A pooled categorical score from this frame's daily contingency counts.

    Delegates to the scorecard's own `CATEGORICAL_SCORES` table, which
    `_bootstrap_categorical_difference` also reads -- that shared table is
    what lets the point estimate here and the bootstrap replicates agree. An
    unknown score is rejected explicitly rather than allowed to raise a bare
    TypeError from inside the scoring library.
    """
    from run_scorecard import CATEGORICAL_SCORES

    if score not in CATEGORICAL_SCORES:
        known = sorted(CATEGORICAL_SCORES)
        raise ValueError(f"Unsupported categorical score: {score!r} (expected one of {known})")
    return _pooled_categorical(_categorical_from_counts(frame, threshold, score), score)


def summary_metrics(day: pd.DataFrame) -> dict:
    """Every metric step 11 reports, from one arm's per-day frame.

    The first six keys are the columns this file has always written, in the
    same order, so existing readers of `tail_repair_qm.csv` are unaffected.
    """
    row = {
        "rmse_mm": float(np.sqrt(day["mse_mm2"].mean())),
        "mae_mm": float(day["mae_mm"].mean()),
        "twcrps_64.5_mm": float(day["twcrps_64.5_mm"].mean()),
        "brier_7.5": float(day["brier_7.5"].mean()),
        "sedi_115.6": pooled_categorical(day, SEDI_THRESHOLD, "sedi"),
        "csi_64.5": pooled_categorical(day, 64.5, "csi"),
        # A point forecast's CRPS is its absolute error (see deterministic_per_day).
        "crps_mm": float(day["mae_mm"].mean()),
    }
    for threshold in CATEGORICAL_THRESHOLDS:
        for score in CATEGORICAL_SCORE_NAMES:
            key = f"{score}_{threshold}"
            if key not in row:
                row[key] = pooled_categorical(day, threshold, score)
    return row


def _sig(ci) -> bool:
    """CI wholly below 0: the QM arm is better (every difference is a loss)."""
    return bool(np.isfinite(ci.ci_lo) and np.isfinite(ci.ci_hi) and ci.ci_hi < 0.0)


def _excludes_zero(ci) -> bool:
    # Direction-agnostic: for a cost metric like RMSE the question is whether
    # the CI clears zero at all, not which side it lands on. Reusing `_sig`
    # here would report a CI of [+3.76, +4.51] as "not significant" purely
    # because it is an increase.
    return bool(
        np.isfinite(ci.ci_lo) and np.isfinite(ci.ci_hi) and (ci.ci_hi < 0.0 or ci.ci_lo > 0.0)
    )


def paired_metrics(
    qm_all: pd.DataFrame, raw_all: pd.DataFrame, block_days: int, n_resamples: int, seed: int
) -> dict:
    """QM minus raw on the same days, every metric with a 95% paired CI.

    The H7 columns (`twcrps_64.5_*`, `sedi_115.6_*`, `brier_7.5_*`,
    `h7_lead_improved`) keep the names and rules they always had. Categorical
    differences use the scorecard's orientation: negative means QM is better,
    for FAR as for POD/CSI/SEDI.
    """
    def mean_ci(column: str):
        return paired_difference_ci(
            qm_all[column].to_numpy(float),
            raw_all[column].to_numpy(float),
            block_days=block_days,
            n_resamples=n_resamples,
            seed=seed,
            aggregate="mean",
        )

    def categorical_ci(threshold: float, score: str):
        return _bootstrap_categorical_difference(
            _categorical_from_counts(qm_all, threshold, score),
            _categorical_from_counts(raw_all, threshold, score),
            score,
            block_days,
            n_resamples,
            seed,
        )

    rmse_ci = paired_difference_ci(
        _rmse_from_mse(qm_all),
        _rmse_from_mse(raw_all),
        block_days=block_days,
        n_resamples=n_resamples,
        seed=seed,
        aggregate="rmse",
    )
    tw_ci = mean_ci("twcrps_64.5_mm")
    brier_ci = mean_ci("brier_7.5")
    sedi_ci = categorical_ci(SEDI_THRESHOLD, "sedi")
    tw_improved = _sig(tw_ci)
    sedi_improved = _sig(sedi_ci)
    brier_worse = bool(np.isfinite(brier_ci.ci_lo) and brier_ci.ci_lo > 0.0)

    row = {
        "n_test_days": int(len(qm_all)),
        "rmse_delta_mm": rmse_ci.estimate,
        "rmse_ci_lo": rmse_ci.ci_lo,
        "rmse_ci_hi": rmse_ci.ci_hi,
        "rmse_significant": _excludes_zero(rmse_ci),
        "rmse_worse": bool(np.isfinite(rmse_ci.ci_lo) and rmse_ci.ci_lo > 0.0),
        "twcrps_64.5_delta": tw_ci.estimate,
        "twcrps_64.5_ci_lo": tw_ci.ci_lo,
        "twcrps_64.5_ci_hi": tw_ci.ci_hi,
        "twcrps_64.5_improved": tw_improved,
        "sedi_115.6_delta": sedi_ci.estimate,
        "sedi_115.6_ci_lo": sedi_ci.ci_lo,
        "sedi_115.6_ci_hi": sedi_ci.ci_hi,
        "sedi_115.6_improved": sedi_improved,
        "brier_7.5_delta": brier_ci.estimate,
        "brier_7.5_ci_lo": brier_ci.ci_lo,
        "brier_7.5_ci_hi": brier_ci.ci_hi,
        "brier_7.5_degraded": brier_worse,
        "h7_lead_improved": tw_improved or sedi_improved,
        "bootstrap_degenerate": bool(rmse_ci.degenerate),
    }
    # CRPS = MAE for these point forecasts, so one interval serves both.
    crps_ci = mean_ci("mae_mm")
    row.update(
        {
            "crps_delta_mm": crps_ci.estimate,
            "crps_ci_lo": crps_ci.ci_lo,
            "crps_ci_hi": crps_ci.ci_hi,
            "crps_significant": _excludes_zero(crps_ci),
        }
    )
    for threshold in CATEGORICAL_THRESHOLDS:
        for score in CATEGORICAL_SCORE_NAMES:
            key = f"{score}_{threshold}"
            if f"{key}_delta" in row:
                continue
            ci = categorical_ci(threshold, score)
            row.update(
                {
                    f"{key}_delta": ci.estimate,
                    f"{key}_ci_lo": ci.ci_lo,
                    f"{key}_ci_hi": ci.ci_hi,
                    f"{key}_significant": _excludes_zero(ci),
                }
            )
    return row


def deterministic_per_day(
    forecast: xr.DataArray, obs: xr.DataArray, fold: str, twcrps_threshold: float
) -> pd.DataFrame:
    """Per-day rows for a *deterministic* forecast field.

    A deterministic forecast is a one-member ensemble, so the ensemble-scored
    metrics go through `expand_dims(member=...)` rather than being dropped.
    That keeps tw-CRPS on exactly the same definition the ensemble tiers use,
    so the H7 comparison is like-for-like; for a point forecast it reduces to
    the threshold-weighted error measure. `crps_mm` is suppressed for the
    same reason: `crps_large_ensemble` on one member is just MAE and would
    only duplicate `mae_mm`.
    """
    frame = per_day_scores(
        forecast,
        obs,
        fold=fold,
        ensemble=forecast.expand_dims(member=[0]),
        probabilities={
            BRIER_THRESHOLD: (forecast >= BRIER_THRESHOLD).astype(float),
        },
        twcrps_threshold=twcrps_threshold,
    )
    return frame.drop(columns=[c for c in frame.columns if c == "crps_mm"])


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Split out of `main` so the argument surface can be tested directly --
    `tests/test_run_tail_repair_qm.py` imports this by name.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-stores",
        nargs="+",
        default=None,
        help="One or more baseline store paths (multi-season; default: 2018 + 2020 daily)",
    )
    parser.add_argument("--baseline-store", default=None, help="Legacy single baseline store path")
    parser.add_argument("--store", default=None, help="Legacy single store store path")
    parser.add_argument("--climatology", default="data/imd_seeps_climatology_jjas.zarr")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-csv", default=None, help="Default: <results-dir>/tail_repair_qm.csv")
    parser.add_argument(
        "--paired-out-csv",
        default=None,
        help="Default: <results-dir>/tail_repair_qm_paired.csv",
    )
    parser.add_argument("--force", action="store_true", help=FORCE_HELP)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=VARIANTS,
        help="Reference variant(s) to evaluate (default: both)",
    )
    return parser.parse_args(args)


def main() -> int:
    args = parse_args()

    _paths = resolve_result_paths(
        args.results_dir,
        {"out_csv": "tail_repair_qm.csv", "paired_out_csv": "tail_repair_qm_paired.csv"},
        {"out_csv": args.out_csv, "paired_out_csv": args.paired_out_csv},
    )
    args.out_csv = _paths["out_csv"]
    args.paired_out_csv = _paths["paired_out_csv"]
    guard_result_overwrites(_paths.values(), force=args.force)

    legacy_path = args.baseline_store if args.baseline_store is not None else args.store
    baseline_paths = resolve_store_paths(
        args.baseline_stores,
        legacy_path,
        DEFAULT_BASELINE_DAILY_STORES,
        "data/baseline_2020_jjas.zarr",
    )

    sources = {
        name: open_multi_season(baseline_paths, group=name) for name in FORECAST_SOURCE_NAMES
    }
    obs = open_multi_season(baseline_paths, group="imd_observed").load()
    climatology = load_climatology(args.climatology).load()
    # The full climatological distribution, not its spatial mean: a quantile
    # map is fitted by pairing forecast and reference samples, so collapsing
    # ~1830 IMD JJAS days to one value per cell would leave every reference
    # sample identical and make the fitted map degenerate.
    climatology_rain = climatology["rain"]
    region_labels = assign_regions(obs["latitude"].values, obs["longitude"].values)
    region_names = sorted(np.unique(region_labels.values).tolist())

    print(f"Tail repair via quantile mapping (Step 11), stores={baseline_paths}")
    print(f"Variants: {args.variants}  |  regions: {region_names}")

    metric_rows: list[dict] = []
    paired_rows: list[dict] = []
    # (variant, lead, arm) -> per-day frame, pooled over both test folds, for CI
    pooled_per_day: dict[tuple[str, int, str], list[pd.DataFrame]] = {}

    for lead_hours in LEAD_HOURS:
        forecasts, obs_aligned = load_aligned(sources, obs, lead_hours)
        sample_times = pd.DatetimeIndex(obs_aligned["sample"].values)
        folds = list(iter_evaluation_folds(sample_times, test_fraction=args.test_fraction))

        raw_blend_frames: list[pd.DataFrame] = []
        raw_blend_obs: list[xr.DataArray] = []
        # Raw single sources, per source, pooled over folds: the "raw" arm of
        # step 11 item 4a's raw-vs-QM comparison, which used to be missing.
        raw_source_frames: dict[str, list[pd.DataFrame]] = {n: [] for n in FORECAST_SOURCE_NAMES}

        for train_mask, test_mask, split_label in folds:
            train_obs = obs_aligned.isel(sample=train_mask)
            test_obs = obs_aligned.isel(sample=test_mask)
            forecast_train = {n: da.isel(sample=train_mask) for n, da in forecasts.items()}
            forecast_test = {n: da.isel(sample=test_mask) for n, da in forecasts.items()}
            # The climatology spans both test seasons; the test year is removed
            # so the `climatology` map never sees the observations it is scored on.
            fold_climatology = climatology_excluding_years(
                climatology_rain, sample_times[test_mask]
            )
            split_kind = (
                "leave_one_year_out"
                if split_label != "seasonal_block_split"
                else "seasonal_block_split"
            )

            # --- raw Tier 1 blend (the reference arm) -------------------------
            weight_results = fit_region_weights(
                forecast_train, obs_aligned, region_labels, train_mask, sample_dim="sample"
            )
            weight_grids = build_region_weight_grid(
                weight_results, region_labels, FORECAST_SOURCE_NAMES
            )
            raw_blend = blend_with_region_weights(forecasts, weight_grids).isel(sample=test_mask)
            raw_day = deterministic_per_day(raw_blend, test_obs, split_label, TWCRPS_THRESHOLD)
            raw_blend_frames.append(raw_day)
            raw_blend_obs.append(test_obs)

            row_key = {
                "lead_hours": lead_hours,
                "fold": split_label,
                "split": split_kind,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
            }
            metric_rows.append(
                {
                    "variant": "raw",
                    **row_key,
                    "scope": "tier1_blend",
                    "source": "blend",
                    "method": "raw",
                    **summary_metrics(raw_day),
                }
            )
            for name in FORECAST_SOURCE_NAMES:
                source_day = deterministic_per_day(
                    forecast_test[name], test_obs, split_label, TWCRPS_THRESHOLD
                )
                raw_source_frames[name].append(source_day)
                metric_rows.append(
                    {
                        "variant": "raw",
                        **row_key,
                        "scope": "single_source",
                        "source": name,
                        "method": "raw",
                        **summary_metrics(source_day),
                    }
                )

            for variant in args.variants:
                qm_train: dict[str, xr.DataArray] = {}
                qm_test: dict[str, xr.DataArray] = {}
                for name in FORECAST_SOURCE_NAMES:
                    maps = fit_maps_for_variant(
                        forecast_train[name],
                        train_obs,
                        region_labels,
                        fold_climatology,
                        variant,
                        lead_hours,
                        name,
                    )
                    qm_train[name] = apply_regional_quantile_maps(
                        maps, forecast_train[name], region_labels
                    )
                    qm_test[name] = apply_regional_quantile_maps(
                        maps, forecast_test[name], region_labels
                    )

                for name in FORECAST_SOURCE_NAMES:
                    day = deterministic_per_day(
                        qm_test[name], test_obs, split_label, TWCRPS_THRESHOLD
                    )
                    metric_rows.append(
                        {
                            "variant": variant,
                            **row_key,
                            "scope": "single_source",
                            "source": name,
                            "method": "qm",
                            **summary_metrics(day),
                        }
                    )
                    pooled_per_day.setdefault((variant, lead_hours, f"qm_{name}"), []).append(day)

                # --- QM-corrected Tier 1 blend --------------------------------
                qm_weight_results = fit_region_weights(
                    qm_train, obs_aligned, region_labels, train_mask, sample_dim="sample"
                )
                qm_weight_grids = build_region_weight_grid(
                    qm_weight_results, region_labels, FORECAST_SOURCE_NAMES
                )
                qm_blend = blend_with_region_weights(qm_test, qm_weight_grids)
                qm_day = deterministic_per_day(
                    qm_blend, test_obs, split_label, TWCRPS_THRESHOLD
                )

                metric_rows.append(
                    {
                        "variant": variant,
                        **row_key,
                        "scope": "tier1_blend",
                        "source": "blend",
                        "method": "qm",
                        **summary_metrics(qm_day),
                    }
                )
                pooled_per_day.setdefault((variant, lead_hours, "qm_blend"), []).append(qm_day)

            print(f"[lead {lead_hours:>3}h | fold {split_label}] raw tier1 blend evaluated")

        pooled_per_day.setdefault(("raw", lead_hours, "raw_blend"), []).extend(raw_blend_frames)
        write_per_day_scores(
            "qm_raw_blend",
            lead_hours,
            pd.concat(raw_blend_frames, ignore_index=True),
            out_dir=args.results_dir,
        )
        for variant in args.variants:
            frames = pooled_per_day[(variant, lead_hours, "qm_blend")]
            write_per_day_scores(
                f"qm_{variant}_blend",
                lead_hours,
                pd.concat(frames, ignore_index=True),
                out_dir=args.results_dir,
            )

        # --- paired comparisons, pooled over both test folds -----------------
        raw_all = pd.concat(raw_blend_frames, ignore_index=True)
        for variant in args.variants:
            qm_all = pd.concat(pooled_per_day[(variant, lead_hours, "qm_blend")], ignore_index=True)
            row = paired_metrics(qm_all, raw_all, args.block_days, args.n_resamples, args.seed)
            paired_rows.append(
                {"variant": variant, "lead_hours": lead_hours, "scope": "tier1_blend",
                 "source": "blend", **row}
            )
            print(
                f"[lead {lead_hours:>3}h | {variant:>12}] "
                f"twCRPS d={row['twcrps_64.5_delta']:+.4f} "
                f"CI[{row['twcrps_64.5_ci_lo']:+.4f},{row['twcrps_64.5_ci_hi']:+.4f}] "
                f"{'IMPROVED' if row['twcrps_64.5_improved'] else 'ns'} | "
                f"SEDI@115.6 d={row['sedi_115.6_delta']:+.4f} "
                f"{'IMPROVED' if row['sedi_115.6_improved'] else 'ns'} | "
                f"RMSE d={row['rmse_delta_mm']:+.3f}mm | "
                f"Brier@7.5 {'DEGRADED' if row['brier_7.5_degraded'] else 'ok'}"
            )
            for name in FORECAST_SOURCE_NAMES:
                source_qm = pd.concat(
                    pooled_per_day[(variant, lead_hours, f"qm_{name}")], ignore_index=True
                )
                source_raw = pd.concat(raw_source_frames[name], ignore_index=True)
                paired_rows.append(
                    {"variant": variant, "lead_hours": lead_hours, "scope": "single_source",
                     "source": name,
                     **paired_metrics(
                         source_qm, source_raw, args.block_days, args.n_resamples, args.seed
                     )}
                )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")

    paired = pd.DataFrame(paired_rows)
    paired.to_csv(args.paired_out_csv, index=False)
    print(f"Wrote {args.paired_out_csv}")

    print("\n" + "=" * 78)
    print("H7 (QM part) -- tw-CRPS@64.5 or SEDI@115.6 must improve at >= 3 of 5 leads")
    print("with the 95% paired CI excluding 0, and Brier@7.5 must not degrade.")
    print("=" * 78)
    blend = paired[paired["scope"] == "tier1_blend"]
    for variant in args.variants:
        subset = blend[blend["variant"] == variant]
        improved = int(subset["h7_lead_improved"].sum())
        degraded = int(subset["brier_7.5_degraded"].sum())
        degenerate = bool(subset["bootstrap_degenerate"].any())
        passes = improved >= MIN_LEADS_FOR_PASS and degraded == 0
        note = "  [bootstrap degenerate -- CI is undefined]" if degenerate else ""
        print(
            f"  {variant:>12}: {improved}/5 leads improved, "
            f"{degraded}/5 leads with degraded Brier -> "
            f"H7(QM) {'PASS' if passes else 'FAIL'}{note}"
        )
    print(
        "\nRMSE trade-off (positive = QM costs accuracy):\n"
        + "\n".join(
            f"  {r.variant:>12} lead {r.lead_hours:>3}h: {r.rmse_delta_mm:+.3f} mm "
            f"CI[{r.rmse_ci_lo:+.3f},{r.rmse_ci_hi:+.3f}]"
            for r in blend.itertuples()
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
