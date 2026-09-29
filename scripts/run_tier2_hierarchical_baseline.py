#!/usr/bin/env python3
"""Run the Tier 2 hierarchical baseline: Phase 4's actual exit criterion.

The combiner-comparison counterpart to Tier 0/Tier 1's own runs
(scripts/run_tier0_baseline.py, scripts/run_tier1_regional_baseline.py):
fits both of Phase 4's real combiners -- EMOS-CSG (src/weavr/emos.py) and
hierarchical BMA (src/weavr/bma.py) -- per bin/region/lead against the real
stores, scores each with CRPS (the proper score both combiners are designed
to optimize) plus RMSE/bias of each combiner's predictive mean, and reports
an honest comparison against each other and against Tier 0's equal-weight
mean and Tier 1's regional-weight blend, all recomputed on the identical
held-out split (Tier 1's own precedent).

Scope, decided by what steps 1-5 already established, not re-litigated here:

- **EMOS-CSG sources: GraphCast's Phase 2 lagged pseudo-ensemble and step
  2's real IFS 50-member ensemble, fit independently** (never HRES, which
  stays deterministic in every store) -- per
  docs/phase4-data-and-combiner-scope.md's own decision. Reported as two
  separate combiners (`emos_graphcast`, `emos_ifs_ens`), never blended
  together into a further meta-combiner (issue #6 names no such step).
- **BMA sources: GraphCast + IFS (ensemble dressing) + HRES (kernel
  dressing), fit jointly into one mixture** -- per step 5's own per-source
  route decision.
- **Rain-intensity bins are classified from GraphCast's own ensemble-mean
  forecast, once per lead, shared by both combiners** -- the same
  forecast `docs/phase4-data-and-combiner-scope.md`'s own per-bin
  sample-count table used, not a new per-source convention; this keeps
  EMOS-CSG (fit per source) and BMA (fit jointly, needs one shared bin
  label per cell) comparable on identical strata.
- **CV strategy: one `seasonal_block_split`, shared by every method at
  each lead** -- Tier 0, Tier 1, both EMOS-CSG sources, and BMA are all
  fit and scored against the *same* train/test sample split, a stronger
  fairness guarantee than Tier 1's own script (which recomputed Tier 0's
  equal-weight mean on its own split; this script recomputes Tier 0
  *and* Tier 1 on the same split used for the Phase 4 combiners).
- **Tier 0/Tier 1's "ifs_ens_mean" source is this run's own IFS ensemble
  mean**, not a separately-opened `baseline_2020_jjas.zarr` group -- both
  ultimately come from the same real WeatherBench 2 archive, and deriving
  it from the already-loaded 50-member ensemble guarantees identical
  sample alignment with the Phase 4 combiners' own IFS source, rather than
  introducing a second, independently-aligned copy.
- **HEPPI stays a validation/methodology reference only** (step 1's
  decision) -- see the "HEPPI cross-check" section of
  docs/tier2-hierarchical-baseline-results.md this script's docstring
  points to, not a training source mixed in here.
- **Small sample sizes are still the same constraint every prior tier
  named** -- per-bin (and per-bin-region, for BMA) splitting can only
  shrink effective per-cell counts further than Tier 0/1's already-small
  numbers; `MIN_TRAIN_DAYS_PER_BIN`'s fallback is expected to trigger
  often, especially for "extreme" (never fittable, per step 1) and
  "heavy" beyond 24h/48h, and this is reported plainly via each result's
  `is_fallback` flag rather than hidden.

Usage:
    python scripts/run_tier2_hierarchical_baseline.py
        [--baseline-store PATH] [--lagged-store PATH] [--ifs-ensemble-store PATH]
        [--climatology PATH] [--out-csv PATH] [--bin-out-csv PATH]
        [--region-out-csv PATH] [--test-fraction F] [--n-monte-carlo N]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_phase2_ensemble_baseline import (  # noqa: E402
    _reconstruct_raw_source,
    build_ensembles_for_lead,
)
from run_tier0_baseline import PRECIP_M_TO_MM, _align_to_imd_day  # noqa: E402
from run_tier1_regional_baseline import (  # noqa: E402
    blend_with_region_weights,
    build_region_weight_grid,
    equal_weight_blend,
)

from weavr import verify as V  # noqa: E402
from weavr.bma import fit_hierarchical_bma, sample_bma_mixture, score_bma  # noqa: E402
from weavr.emos import csgd_crps, fit_emos_csg, predict_csgd_params  # noqa: E402
from weavr.rain_bins import classify_rain_bin  # noqa: E402
from weavr.regions import assign_regions  # noqa: E402
from weavr.score_io import per_day_scores, write_per_day_scores  # noqa: E402
from weavr.splits import (  # noqa: E402
    iter_evaluation_folds,
)
from weavr.stores import (  # noqa: E402
    DEFAULT_BASELINE_DAILY_STORES,
    DEFAULT_IFS_DAILY_STORES,
    DEFAULT_LAGGED_DAILY_STORES,
    open_multi_season,
    resolve_store_paths,
)
from weavr.weighting import fit_region_weights  # noqa: E402

PRECIP_VARIABLE = "total_precipitation_24hr"
LEAD_HOURS = [24, 48, 72, 96, 120]
# The 3 deterministic sources Tier 0/Tier 1 already use -- "ifs_ens_mean"
# here is this run's own IFS ensemble's mean (see this module's docstring),
# not a separately-opened baseline-store group.
TIER0_TIER1_SOURCE_NAMES = ["graphcast", "hres", "ifs_ens_mean"]
N_MONTE_CARLO_SAMPLES = 500
_TINY = 1e-6


def _clip_negative_precip(da: xr.DataArray) -> xr.DataArray:
    """Floors tiny negative precipitation values to exactly zero."""
    return da.clip(min=0.0)


def load_graphcast_ensemble(
    lagged_store: Sequence[str | Path] | str | Path, lead_hours: int
) -> xr.DataArray:
    """GraphCast's Phase 2 lagged pseudo-ensemble at one lead, in mm, aligned
    to the IMD day each forecast validates on -- reuses
    scripts/run_phase2_ensemble_baseline.py's own construction rather than
    reimplementing it (see that module for why this doesn't re-fetch from GCS).
    """
    paths = [lagged_store] if isinstance(lagged_store, (str, Path)) else list(lagged_store)
    dense = open_multi_season(paths, group="graphcast").load()
    raw_ds = _reconstruct_raw_source(dense, PRECIP_VARIABLE)
    ensemble = build_ensembles_for_lead(
        raw_ds, PRECIP_VARIABLE, dense["nominal_time"].values, lead_hours
    )
    return _clip_negative_precip(ensemble * PRECIP_M_TO_MM)


def load_ifs_ensemble(
    ifs_store: Sequence[str | Path] | str | Path, lead_hours: int
) -> xr.DataArray:
    """The real IFS 50-member ensemble at one lead, in mm, aligned to the IMD day."""
    paths = [ifs_store] if isinstance(ifs_store, (str, Path)) else list(ifs_store)
    ds = open_multi_season(paths)
    da = ds[PRECIP_VARIABLE].sel(prediction_timedelta=lead_hours).load() * PRECIP_M_TO_MM
    return _clip_negative_precip(_align_to_imd_day(da, lead_hours))


def load_hres_forecast(
    baseline_store: Sequence[str | Path] | str | Path, lead_hours: int
) -> xr.DataArray:
    """HRES's deterministic forecast at one lead, in mm, aligned to the IMD day."""
    paths = [baseline_store] if isinstance(baseline_store, (str, Path)) else list(baseline_store)
    ds = open_multi_season(paths, group="hres")
    da = ds[PRECIP_VARIABLE].sel(prediction_timedelta=lead_hours).load() * PRECIP_M_TO_MM
    return _clip_negative_precip(_align_to_imd_day(da, lead_hours))


def align_all_sources(
    graphcast_ensemble: xr.DataArray,
    ifs_ensemble: xr.DataArray,
    hres_forecast: xr.DataArray,
    obs: xr.Dataset,
) -> tuple[dict[str, xr.DataArray], xr.DataArray]:
    """Reindexes every source onto one shared `sample` coordinate (an outer
    join -- a source missing a particular sample gets NaN there, matching
    scripts/run_tier1_regional_baseline.py's own convention), then restricts
    to samples with real IMD ground truth. `member` (present only on the
    two ensemble sources) is untouched by the outer join, which only acts
    on the shared `sample` dimension.
    """
    graphcast_aligned, ifs_aligned, hres_aligned = xr.align(
        graphcast_ensemble, ifs_ensemble, hres_forecast, join="outer"
    )
    forecasts = {
        "graphcast": graphcast_aligned,
        "ifs_ens": ifs_aligned,
        "hres": hres_aligned,
    }

    sample_values = forecasts["hres"]["sample"].values
    obs_aligned = obs["rain"].reindex(time=sample_values).rename(time="sample")
    has_obs = ~obs_aligned.isnull().all(dim=["latitude", "longitude"])
    forecasts = {name: da.isel(sample=has_obs.values) for name, da in forecasts.items()}
    obs_aligned = obs_aligned.isel(sample=has_obs.values)
    return forecasts, obs_aligned


def _ordered_values(da: xr.DataArray, member_dim: str | None = None) -> np.ndarray:
    """`da`'s values with dims forced to `(sample, latitude, longitude[,
    member])` -- so the plain boolean-array cell masking below (the same
    idiom src/weavr/emos.py and src/weavr/bma.py already use internally)
    is safe regardless of the array's original dimension order.
    """
    dims = ["sample", "latitude", "longitude"]
    if member_dim is not None and member_dim in da.dims:
        dims.append(member_dim)
    return da.transpose(*dims).values


def _pool_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Cell-count-weighted average of several bins'/cells' own mean
    statistics -- exact for any pooled *mean* quantity (e.g. mean CRPS,
    mean bias, or mean squared error before its final sqrt), since a
    weighted average of per-group means with weights equal to each
    group's own count reconstructs the true overall mean exactly.
    """
    valid = ~np.isnan(values) & (weights > 0)
    if not np.any(valid):
        return float("nan")
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def _pool_rmse_from_mse(mse_values: np.ndarray, weights: np.ndarray) -> float:
    """Domain-wide RMSE from several bins'/cells' own mean squared errors --
    pools the mean-squared-error (an averageable quantity) via
    `_pool_weighted_mean`, then takes the square root once at the end,
    rather than (incorrectly) averaging each group's own RMSE directly.
    """
    pooled_mse = _pool_weighted_mean(mse_values, weights)
    return float(np.sqrt(pooled_mse)) if not np.isnan(pooled_mse) else float("nan")


def _csgd_predictive_mean(
    location: np.ndarray,
    scale: np.ndarray,
    shift: np.ndarray,
    rng: np.random.Generator,
    n_samples: int = N_MONTE_CARLO_SAMPLES,
) -> np.ndarray:
    """Monte Carlo estimate of the fitted CSGD's own predictive mean (the
    exact analytic mean of a censored-shifted-gamma needs machinery beyond
    src/weavr/emos.py's own scope) -- draws from the same shape/scale/shift
    parameterization that module documents, shifted and censored at zero
    exactly as `weavr.emos.csgd_crps` assumes, and reused here only for
    reporting RMSE/bias of the combiner's point-forecast summary, not for
    CRPS itself (which has the real closed form already).
    """
    kappa = location**2 / np.clip(scale**2, _TINY, None)
    theta = np.clip(scale**2, _TINY, None) / np.clip(location, _TINY, None)
    draws = rng.gamma(
        np.clip(kappa, _TINY, None)[..., None],
        np.clip(theta, _TINY, None)[..., None],
        size=(*location.shape, n_samples),
    )
    censored = np.clip(draws + shift[..., None], 0.0, None)
    return censored.mean(axis=-1)


def score_emos_source(
    results: dict,
    forecast: xr.DataArray,
    obs: xr.DataArray,
    rain_bin_labels: xr.DataArray,
    test_mask: np.ndarray,
    rng: np.random.Generator,
    n_samples: int = N_MONTE_CARLO_SAMPLES,
    member_dim: str = "member",
    per_cell_out: dict[str, np.ndarray] | None = None,
) -> dict[str, dict]:
    """Per-bin CRPS/RMSE/bias of one EMOS-CSG source's fitted results,
    scored at test time by looking up each test cell's own rain-bin label
    (classified from its own forecast, per this module's docstring) and
    that bin's fitted (or fallback) `CensoredShiftedGammaResult`.

    `per_cell_out`, when given, is additionally filled with full
    `(sample, latitude, longitude)` grids of `"crps"` and
    `"predictive_mean"`, NaN outside the scored cells. Step 04 needs the
    unreduced fields to write per-day scores; collecting them here reuses
    the numbers this function already computes rather than re-predicting
    them in a second, possibly divergent, code path. The returned summary is
    unchanged either way.
    """
    forecast_mean = forecast.mean(dim=member_dim, skipna=True)
    forecast_spread = forecast.std(dim=member_dim, ddof=1, skipna=True)

    obs_v = _ordered_values(obs)
    mean_v = _ordered_values(forecast_mean)
    spread_v = _ordered_values(forecast_spread)
    bin_v = _ordered_values(rain_bin_labels)

    test_mask_3d = np.broadcast_to(test_mask[:, None, None], obs_v.shape)
    valid = ~np.isnan(obs_v) & ~np.isnan(mean_v) & ~np.isnan(spread_v) & test_mask_3d

    if per_cell_out is not None:
        per_cell_out["crps"] = np.full(obs_v.shape, np.nan)
        per_cell_out["predictive_mean"] = np.full(obs_v.shape, np.nan)

    per_bin: dict[str, dict] = {}
    for bin_label, result in results.items():
        cell_mask = (bin_v == bin_label) & valid
        n_cells = int(cell_mask.sum())
        if n_cells == 0:
            per_bin[bin_label] = {
                "n_test_cells": 0,
                "crps_mm": float("nan"),
                "mse_mm2": float("nan"),
                "bias_mm": float("nan"),
                "is_fallback": result.is_fallback,
            }
            continue

        mean_cells = mean_v[cell_mask]
        spread_cells = spread_v[cell_mask]
        obs_cells = obs_v[cell_mask]

        location, scale, shift = predict_csgd_params(result, mean_cells, spread_cells)
        crps_values = csgd_crps(location, scale, shift, obs_cells)
        predictive_mean = _csgd_predictive_mean(location, scale, shift, rng, n_samples=n_samples)

        if per_cell_out is not None:
            per_cell_out["crps"][cell_mask] = crps_values
            per_cell_out["predictive_mean"][cell_mask] = predictive_mean

        per_bin[bin_label] = {
            "n_test_cells": n_cells,
            "crps_mm": float(np.mean(crps_values)),
            "mse_mm2": float(np.mean((predictive_mean - obs_cells) ** 2)),
            "bias_mm": float(np.mean(predictive_mean - obs_cells)),
            "is_fallback": result.is_fallback,
        }
    return per_bin


def score_bma_cells(
    results: dict,
    forecasts: dict[str, xr.DataArray],
    obs: xr.DataArray,
    rain_bin_labels: xr.DataArray,
    region_labels: xr.DataArray,
    test_mask: np.ndarray,
    rng: np.random.Generator,
    n_samples: int = N_MONTE_CARLO_SAMPLES,
    member_dim: str = "member",
    per_cell_out: dict[str, np.ndarray] | None = None,
) -> dict[tuple[str, str], dict]:
    """Per (bin, region) CRPS/RMSE/bias of the fitted BMA mixture, scored at
    test time -- the same cell-lookup idea as `score_emos_source`, one
    level more stratified (bin and region both determine which
    `BmaFitResult` applies).

    `per_cell_out` behaves exactly as in `score_emos_source`: when given, it
    is filled with full `(sample, latitude, longitude)` grids of `"crps"`
    and `"predictive_mean"` for step 04's per-day scores.
    """
    obs_v = _ordered_values(obs)
    bin_v = _ordered_values(rain_bin_labels)
    region_2d = region_labels.transpose("latitude", "longitude").values
    region_v = np.broadcast_to(region_2d[None, :, :], obs_v.shape)

    mean_arrays: dict[str, np.ndarray] = {}
    spread_arrays: dict[str, np.ndarray | None] = {}
    valid = ~np.isnan(obs_v)
    for source, da in forecasts.items():
        if member_dim in da.dims:
            mean_v = _ordered_values(da.mean(dim=member_dim, skipna=True))
            spread_v = _ordered_values(da.std(dim=member_dim, ddof=1, skipna=True))
            spread_arrays[source] = spread_v
            valid = valid & ~np.isnan(spread_v)
        else:
            mean_v = _ordered_values(da)
            spread_arrays[source] = None
        mean_arrays[source] = mean_v
        valid = valid & ~np.isnan(mean_v)

    test_mask_3d = np.broadcast_to(test_mask[:, None, None], obs_v.shape)
    valid = valid & test_mask_3d

    if per_cell_out is not None:
        per_cell_out["crps"] = np.full(obs_v.shape, np.nan)
        per_cell_out["predictive_mean"] = np.full(obs_v.shape, np.nan)

    per_cell: dict[tuple[str, str], dict] = {}
    for (bin_label, region), result in results.items():
        cell_mask = (bin_v == bin_label) & (region_v == region) & valid
        n_cells = int(cell_mask.sum())
        if n_cells == 0:
            per_cell[(bin_label, region)] = {
                "n_test_cells": 0,
                "crps_mm": float("nan"),
                "mse_mm2": float("nan"),
                "bias_mm": float("nan"),
                "is_fallback": result.is_fallback,
            }
            continue

        cell_mean = {s: mean_arrays[s][cell_mask] for s in mean_arrays}
        # Bound to a local before the None check so the narrowing sticks;
        # mypy cannot narrow `spread_arrays[s]` through a dict subscript
        # inside a comprehension. Behaviour is unchanged.
        cell_spread: dict[str, np.ndarray | None] = {}
        for source in mean_arrays:
            spread = spread_arrays[source]
            cell_spread[source] = None if spread is None else spread[cell_mask]
        obs_cells = obs_v[cell_mask]

        cell_mean_da = {s: xr.DataArray(v, dims="cell") for s, v in cell_mean.items()}
        cell_spread_da = {
            s: (xr.DataArray(v, dims="cell") if v is not None else None)
            for s, v in cell_spread.items()
        }
        obs_cells_da = xr.DataArray(obs_cells, dims="cell")

        crps_values = score_bma(
            result, cell_mean_da, cell_spread_da, obs_cells_da, rng=rng,
            n_samples=n_samples,
        )
        samples = sample_bma_mixture(
            result, cell_mean, cell_spread, rng, n_samples=n_samples
        )
        predictive_mean = samples.mean(axis=-1)

        if per_cell_out is not None:
            per_cell_out["crps"][cell_mask] = crps_values.values
            per_cell_out["predictive_mean"][cell_mask] = predictive_mean

        per_cell[(bin_label, region)] = {
            "n_test_cells": n_cells,
            "crps_mm": float(np.mean(crps_values.values)),
            "mse_mm2": float(np.mean((predictive_mean - obs_cells) ** 2)),
            "bias_mm": float(np.mean(predictive_mean - obs_cells)),
            "is_fallback": result.is_fallback,
        }
    return per_cell


def _domain_summary(per_group: dict, prefix: str) -> dict:
    n = np.array([v["n_test_cells"] for v in per_group.values()], dtype=float)
    crps = np.array([v["crps_mm"] for v in per_group.values()])
    mse = np.array([v["mse_mm2"] for v in per_group.values()])
    bias = np.array([v["bias_mm"] for v in per_group.values()])
    return {
        f"{prefix}_crps_mm": _pool_weighted_mean(crps, n),
        f"{prefix}_rmse_mm": _pool_rmse_from_mse(mse, n),
        f"{prefix}_bias_mm": _pool_weighted_mean(bias, n),
        f"{prefix}_n_test_cells": int(n.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-stores",
        nargs="+",
        default=None,
        help="One or more baseline store paths (multi-season; default: 2018 + 2020 daily)",
    )
    parser.add_argument("--baseline-store", default=None, help="Legacy single baseline store path")
    parser.add_argument(
        "--lagged-stores",
        nargs="+",
        default=None,
        help="One or more lagged ensemble store paths (multi-season; default: 2018 + 2020 daily)",
    )
    parser.add_argument("--lagged-store", default=None, help="Legacy single lagged store path")
    parser.add_argument(
        "--ifs-ensemble-stores",
        nargs="+",
        default=None,
        help="One or more IFS ensemble store paths (multi-season; default: 2018 + 2020 daily)",
    )
    parser.add_argument("--ifs-ensemble-store", default=None, help="Legacy single IFS store path")
    parser.add_argument("--climatology", default="data/imd_seeps_climatology_jjas.zarr")
    parser.add_argument("--out-csv", default="results/tier2_hierarchical_baseline.csv")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--bin-out-csv", default="results/tier2_hierarchical_baseline_by_bin.csv")
    parser.add_argument(
        "--region-out-csv", default="results/tier2_hierarchical_baseline_by_region.csv"
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--n-monte-carlo", type=int, default=N_MONTE_CARLO_SAMPLES)
    args = parser.parse_args()
    n_samples = args.n_monte_carlo

    baseline_paths = resolve_store_paths(
        args.baseline_stores,
        args.baseline_store,
        DEFAULT_BASELINE_DAILY_STORES,
        "data/baseline_2020_jjas.zarr",
    )
    lagged_paths = resolve_store_paths(
        args.lagged_stores,
        args.lagged_store,
        DEFAULT_LAGGED_DAILY_STORES,
        "data/lagged_ensemble_inputs_2020_jjas.zarr",
    )
    ifs_paths = resolve_store_paths(
        args.ifs_ensemble_stores,
        args.ifs_ensemble_store,
        DEFAULT_IFS_DAILY_STORES,
        "data/ifs_ens_2020_jjas.zarr",
    )

    obs = open_multi_season(baseline_paths, group="imd_observed").load()
    region_labels = assign_regions(obs["latitude"].values, obs["longitude"].values)
    rng = np.random.default_rng(0)

    print("Tier 2 hierarchical baseline: EMOS-CSG and BMA scored against real stores")
    print(f"Baseline store(s): {baseline_paths}")
    print(f"Lagged store(s): {lagged_paths}")
    print(f"IFS ensemble store(s): {ifs_paths}")

    domain_rows: list[dict] = []
    bin_rows: list[dict] = []
    region_rows: list[dict] = []

    for lead_hours in LEAD_HOURS:
        graphcast_ensemble = load_graphcast_ensemble(lagged_paths, lead_hours)
        ifs_ensemble = load_ifs_ensemble(ifs_paths, lead_hours)
        hres_forecast = load_hres_forecast(baseline_paths, lead_hours)
        forecasts, obs_aligned = align_all_sources(
            graphcast_ensemble, ifs_ensemble, hres_forecast, obs
        )

        sample_times = pd.DatetimeIndex(obs_aligned["sample"].values)
        folds = list(iter_evaluation_folds(sample_times, test_fraction=args.test_fraction))

        test_tier0_list = []
        test_tier1_list = []
        test_obs_list = []
        fold_emos_per_bin: dict[str, list[dict]] = {"graphcast": [], "ifs_ens": []}
        fold_bma_cells: list[dict] = []

        for train_mask, test_mask, split_label in folds:
            rain_bin_labels = classify_rain_bin(
                forecasts["graphcast"].mean(dim="member", skipna=True)
            )

            emos_results = {
                "graphcast": fit_emos_csg(
                    forecasts["graphcast"],
                    obs_aligned,
                    rain_bin_labels,
                    train_mask,
                    source="graphcast",
                ),
                "ifs_ens": fit_emos_csg(
                    forecasts["ifs_ens"],
                    obs_aligned,
                    rain_bin_labels,
                    train_mask,
                    source="ifs_ens",
                ),
            }
            bma_results = fit_hierarchical_bma(
                forecasts, obs_aligned, rain_bin_labels, region_labels, train_mask
            )

            tier01_sources = {
                "graphcast": forecasts["graphcast"].mean(dim="member", skipna=True),
                "hres": forecasts["hres"],
                "ifs_ens_mean": forecasts["ifs_ens"].mean(dim="member", skipna=True),
            }
            weight_results = fit_region_weights(
                tier01_sources, obs_aligned, region_labels, train_mask, sample_dim="sample"
            )
            weight_grids = build_region_weight_grid(
                weight_results, region_labels, TIER0_TIER1_SOURCE_NAMES
            )
            tier1_blend = blend_with_region_weights(tier01_sources, weight_grids)
            tier0_blend = equal_weight_blend(tier01_sources)

            test_tier0 = tier0_blend.isel(sample=test_mask)
            test_tier1 = tier1_blend.isel(sample=test_mask)
            test_obs = obs_aligned.isel(sample=test_mask)
            test_tier0_list.append(test_tier0)
            test_tier1_list.append(test_tier1)
            test_obs_list.append(test_obs)

            split_kind = (
                "leave_one_year_out"
                if split_label != "seasonal_block_split"
                else "seasonal_block_split"
            )
            domain_row = {
                "lead_hours": lead_hours,
                "fold": split_label,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "split": split_kind,
                "tier0_rmse_mm": float(V.rmse(test_tier0, test_obs)),
                "tier0_bias_mm": float(V.bias(test_tier0, test_obs)),
                "tier1_rmse_mm": float(V.rmse(test_tier1, test_obs)),
                "tier1_bias_mm": float(V.bias(test_tier1, test_obs)),
            }

            def _write_per_day(method: str, grids: dict[str, np.ndarray], fold_lbl: str) -> None:
                template = obs_aligned.transpose("sample", "latitude", "longitude")
                predictive_mean = xr.DataArray(
                    grids["predictive_mean"], coords=template.coords, dims=template.dims
                ).isel(sample=test_mask)
                crps_grid = xr.DataArray(
                    grids["crps"], coords=template.coords, dims=template.dims
                ).isel(sample=test_mask)
                write_per_day_scores(
                    method,
                    lead_hours,
                    per_day_scores(
                        predictive_mean,
                        test_obs,
                        fold=fold_lbl,
                        per_cell_scores={"crps_mm": crps_grid},
                    ),
                    out_dir=args.results_dir,
                )

            for source_key, results in emos_results.items():
                emos_grids: dict[str, np.ndarray] = {}
                per_bin = score_emos_source(
                    results,
                    forecasts[source_key],
                    obs_aligned,
                    rain_bin_labels,
                    test_mask,
                    rng,
                    n_samples=n_samples,
                    per_cell_out=emos_grids,
                )
                _write_per_day(f"tier2_emos_{source_key}", emos_grids, split_label)
                fold_emos_per_bin[source_key].append(per_bin)
                domain_row.update(_domain_summary(per_bin, f"emos_{source_key}"))
                for bin_label, stats in per_bin.items():
                    bin_rows.append(
                        {
                            "lead_hours": lead_hours,
                            "fold": split_label,
                            "combiner": f"emos_{source_key}",
                            "bin": bin_label,
                            "region": "",
                            **stats,
                        }
                    )

            bma_grids: dict[str, np.ndarray] = {}
            bma_per_cell = score_bma_cells(
                bma_results,
                forecasts,
                obs_aligned,
                rain_bin_labels,
                region_labels,
                test_mask,
                rng,
                n_samples=n_samples,
                per_cell_out=bma_grids,
            )
            _write_per_day("tier2_bma", bma_grids, split_label)
            fold_bma_cells.append(bma_per_cell)
            domain_row.update(_domain_summary(bma_per_cell, "bma"))
            for (bin_label, region), stats in bma_per_cell.items():
                bin_rows.append(
                    {
                        "lead_hours": lead_hours,
                        "fold": split_label,
                        "combiner": "bma",
                        "bin": bin_label,
                        "region": region,
                        **stats,
                    }
                )
                region_rows.append(
                    {
                        "lead_hours": lead_hours,
                        "fold": split_label,
                        "region": region,
                        "bin": bin_label,
                        **stats,
                    }
                )

            domain_rows.append(domain_row)
            eg_crps = domain_row.get("emos_graphcast_crps_mm", float("nan"))
            ei_crps = domain_row.get("emos_ifs_ens_crps_mm", float("nan"))
            bma_crps = domain_row.get("bma_crps_mm", float("nan"))
            print(
                f"[lead {lead_hours:>3}h | fold {split_label}] "
                f"tier0 rmse={domain_row['tier0_rmse_mm']:.2f}mm "
                f"tier1 rmse={domain_row['tier1_rmse_mm']:.2f}mm "
                f"emos_graphcast crps={eg_crps:.2f}mm "
                f"emos_ifs_ens crps={ei_crps:.2f}mm "
                f"bma crps={bma_crps:.2f}mm"
            )

        if len(folds) > 1:
            pooled_tier0 = xr.concat(test_tier0_list, dim="sample")
            pooled_tier1 = xr.concat(test_tier1_list, dim="sample")
            pooled_obs = xr.concat(test_obs_list, dim="sample")

            pooled_domain_row = {
                "lead_hours": lead_hours,
                "fold": "pooled",
                "n_train": int(obs_aligned.sizes["sample"]),
                "n_test": int(pooled_obs.sizes["sample"]),
                "split": "leave_one_year_out",
                "tier0_rmse_mm": float(V.rmse(pooled_tier0, pooled_obs)),
                "tier0_bias_mm": float(V.bias(pooled_tier0, pooled_obs)),
                "tier1_rmse_mm": float(V.rmse(pooled_tier1, pooled_obs)),
                "tier1_bias_mm": float(V.bias(pooled_tier1, pooled_obs)),
            }

            for source_key in ("graphcast", "ifs_ens"):
                pooled_per_bin = {}
                all_bins = {b for p in fold_emos_per_bin[source_key] for b in p}
                for b in all_bins:
                    b_stats = [p[b] for p in fold_emos_per_bin[source_key] if b in p]
                    n = np.array([s["n_test_cells"] for s in b_stats], dtype=float)
                    crps = np.array([s["crps_mm"] for s in b_stats])
                    mse = np.array([s["mse_mm2"] for s in b_stats])
                    bias = np.array([s["bias_mm"] for s in b_stats])
                    pooled_per_bin[b] = {
                        "n_test_cells": int(n.sum()),
                        "crps_mm": _pool_weighted_mean(crps, n),
                        "rmse_mm": _pool_rmse_from_mse(mse, n),
                        "bias_mm": _pool_weighted_mean(bias, n),
                        "mse_mm2": _pool_weighted_mean(mse, n),
                        "is_fallback": any(s.get("is_fallback", False) for s in b_stats),
                    }
                    bin_rows.append(
                        {
                            "lead_hours": lead_hours,
                            "fold": "pooled",
                            "combiner": f"emos_{source_key}",
                            "bin": b,
                            "region": "",
                            **pooled_per_bin[b],
                        }
                    )
                pooled_domain_row.update(_domain_summary(pooled_per_bin, f"emos_{source_key}"))

            pooled_bma_per_cell = {}
            all_bma_keys = {k for p in fold_bma_cells for k in p}
            for (bin_label, region) in all_bma_keys:
                cell_stats = [
                    p[(bin_label, region)]
                    for p in fold_bma_cells
                    if (bin_label, region) in p
                ]
                n = np.array([s["n_test_cells"] for s in cell_stats], dtype=float)
                crps = np.array([s["crps_mm"] for s in cell_stats])
                mse = np.array([s["mse_mm2"] for s in cell_stats])
                bias = np.array([s["bias_mm"] for s in cell_stats])
                pooled_bma_per_cell[(bin_label, region)] = {
                    "n_test_cells": int(n.sum()),
                    "crps_mm": _pool_weighted_mean(crps, n),
                    "rmse_mm": _pool_rmse_from_mse(mse, n),
                    "bias_mm": _pool_weighted_mean(bias, n),
                    "mse_mm2": _pool_weighted_mean(mse, n),
                    "is_fallback": any(s.get("is_fallback", False) for s in cell_stats),
                }
                bin_rows.append(
                    {
                        "lead_hours": lead_hours,
                        "fold": "pooled",
                        "combiner": "bma",
                        "bin": bin_label,
                        "region": region,
                        **pooled_bma_per_cell[(bin_label, region)],
                    }
                )
                region_rows.append(
                    {
                        "lead_hours": lead_hours,
                        "fold": "pooled",
                        "region": region,
                        "bin": bin_label,
                        **pooled_bma_per_cell[(bin_label, region)],
                    }
                )
            pooled_domain_row.update(_domain_summary(pooled_bma_per_cell, "bma"))
            domain_rows.append(pooled_domain_row)

    def _write_csv(path_str: str, rows: list[dict]) -> None:
        out_path = Path(path_str)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys()) if rows else []
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {out_path}")

    _write_csv(args.out_csv, domain_rows)
    _write_csv(args.bin_out_csv, bin_rows)
    _write_csv(args.region_out_csv, region_rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())
