#!/usr/bin/env python3
"""Export small, committed example grids for the dashboard.

Per `docs/phase7-dashboard-scope.md`'s decision (routed to the user via
`AskUserQuestion`): the blended-map and extreme-probability views read
small, committed example exports rather than requiring the local, gitignored
Zarr stores to exist on whoever opens the dashboard.

**Blended-map grid** (`example_blend_grid.npz`): reuses
`run_tier1_regional_baseline.py`'s own `load_aligned_forecasts_and_obs` /
`build_region_weight_grid` / `blend_with_region_weights` (already real,
already tested) rather than reimplementing the blend a third time --
`run_daily_pipeline.py` (Phase 6, PR #39) already does the same. Reuses
`results/tier1_regional_weights.csv`'s already-fitted weights rather than
re-fitting them here: this script only renders a snapshot of that real fit.
For each of the 5 lead times, blends the single most recent real aligned
sample (not a synthetic one, and not an average across samples, which would
blur the real spatial pattern a viewer should see).

**Extreme-probability grid** (`example_probability_grid.npz`): per step 4's
own decision (routed via `AskUserQuestion`), uses EMOS-CSG's `ifs_ens`
combiner (the real 50-member IFS ensemble, not GraphCast's lagged
pseudo-ensemble) -- its censored-shifted-gamma has a real closed-form
survival function (`weavr.emos.exceedance_probability_csgd`), unlike BMA's
mixture (no closed form, per `weavr.bma`'s own docstring). This diverges
from the blended map's own Tier 1 combiner, stated plainly in the view
itself, because Tier 1's deterministic blend has no predictive distribution
to compute an exceedance probability from at all. Reuses
`run_tier2_hierarchical_baseline.py`'s own real data-loading functions
(`load_graphcast_ensemble`, `load_ifs_ensemble`, `load_hres_forecast`,
`align_all_sources`) and `weavr.emos.fit_emos_csg`/`predict_csgd_params`
rather than reimplementing EMOS-CSG fitting a second time. Unlike
`run_tier2_hierarchical_baseline.py`'s own held-out evaluation split, this
script fits on *all* available real samples (a display snapshot, not a
skill evaluation) -- stated explicitly since it is a real difference from
that script's own convention.

Usage:
    python scripts/export_dashboard_example_grids.py
        [--store PATH] [--weights-csv PATH]
        [--lagged-store PATH] [--ifs-ensemble-store PATH]
        [--out-blend PATH] [--out-probability PATH]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tier1_regional_baseline import (  # noqa: E402
    FORECAST_SOURCE_NAMES,
    LEAD_HOURS,
    PRECIP_VARIABLE,
    blend_with_region_weights,
    build_region_weight_grid,
    load_aligned_forecasts_and_obs,
)
from run_tier2_hierarchical_baseline import (  # noqa: E402
    align_all_sources,
    load_graphcast_ensemble,
    load_hres_forecast,
    load_ifs_ensemble,
)

from weavr.emos import exceedance_probability_csgd, fit_emos_csg, predict_csgd_params  # noqa: E402
from weavr.rain_bins import RAIN_BIN_LABELS, classify_rain_bin  # noqa: E402
from weavr.regions import assign_regions  # noqa: E402
from weavr.verify import IMD_RAIN_THRESHOLDS_MM  # noqa: E402
from weavr.weighting import RegionWeightResult  # noqa: E402

DEFAULT_STORE = "data/baseline_2020_jjas.zarr"
DEFAULT_WEIGHTS_CSV = "results/tier1_regional_weights.csv"
DEFAULT_LAGGED_STORE = "data/lagged_ensemble_inputs_2020_jjas.zarr"
DEFAULT_IFS_ENSEMBLE_STORE = "data/ifs_ens_2020_jjas.zarr"
DEFAULT_OUT_BLEND = "dashboard/data/example_blend_grid.npz"
DEFAULT_OUT_PROBABILITY = "dashboard/data/example_probability_grid.npz"
EXTREME_THRESHOLD_MM = IMD_RAIN_THRESHOLDS_MM[-1]  # 204.5mm, IMD's own "extremely heavy" boundary


def load_fitted_weights(csv_path: str | Path, lead_hours: int) -> dict[str, dict[str, float]]:
    """Read `results/tier1_regional_weights.csv`'s already-fitted weights for one lead.

    Returns `{region: {source: weight}}` -- this reuses the real,
    already-committed fit; it does not re-fit anything.
    """
    weights: dict[str, dict[str, float]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["lead_hours"]) != lead_hours:
                continue
            weights[row["region"]] = {
                source: float(row[f"weight_{source}"]) for source in FORECAST_SOURCE_NAMES
            }
    if not weights:
        raise ValueError(f"no fitted weights found for lead_hours={lead_hours} in {csv_path}")
    return weights


def _weight_results_from_csv(
    weights_by_region: dict[str, dict[str, float]],
) -> dict[str, RegionWeightResult]:
    """Wrap CSV-sourced weights in `RegionWeightResult` so
    `build_region_weight_grid` (which only reads `.weights`) can reuse them
    unmodified, without re-fitting."""
    return {
        region: RegionWeightResult(region=region, weights=w, is_fallback=False)
        for region, w in weights_by_region.items()
    }


def build_example_blend_grid(store_path: str, weights_csv: str, lead_hours: int) -> xr.DataArray:
    """The real Tier 1 blend, for one lead time, at the most recent real sample."""
    sources = {
        name: xr.open_zarr(store_path, group=name, consolidated=True)
        for name in FORECAST_SOURCE_NAMES
    }
    obs = xr.open_zarr(store_path, group="imd_observed", consolidated=True).load()
    region_labels = assign_regions(obs["latitude"].values, obs["longitude"].values)

    forecasts, _obs_aligned = load_aligned_forecasts_and_obs(
        sources, obs, PRECIP_VARIABLE, lead_hours
    )

    weights_by_region = load_fitted_weights(weights_csv, lead_hours)
    weight_results = _weight_results_from_csv(weights_by_region)
    weight_grids = build_region_weight_grid(weight_results, region_labels, FORECAST_SOURCE_NAMES)

    blend = blend_with_region_weights(forecasts, weight_grids)
    return blend.isel(sample=-1)


def build_example_probability_grid(
    baseline_store: str, lagged_store: str, ifs_store: str, lead_hours: int
) -> dict[str, np.ndarray | str]:
    """The real EMOS-CSG (`ifs_ens` source) exceedance-probability grid,
    P(rain > 204.5mm), for one lead time, at the most recent real aligned
    sample.

    Fits on *all* available real samples (a display snapshot, not the
    held-out skill evaluation `run_tier2_hierarchical_baseline.py` itself
    performs) -- a real, stated difference from that script's own
    convention. `rain_bin_labels` are classified from GraphCast's own
    ensemble-mean forecast, the same shared convention
    `run_tier2_hierarchical_baseline.py` uses so EMOS-CSG's per-cell bin
    lookup matches this project's one established rule, even though the
    probability itself comes from the `ifs_ens` combiner.

    Returns `{"probability": (lat, lon) float array, "is_fallback": (lat,
    lon) bool array}` -- a cell is `True` in `is_fallback` when its own
    forecast fell in a rain-intensity bin EMOS-CSG could not fit for real
    at this lead (checked against `docs/phase4-data-and-combiner-scope.md`'s
    own finding: the extremely_heavy bin is never fittable at any lead).
    """
    obs = xr.open_zarr(baseline_store, group="imd_observed", consolidated=True).load()

    graphcast_ensemble = load_graphcast_ensemble(lagged_store, lead_hours)
    ifs_ensemble = load_ifs_ensemble(ifs_store, lead_hours)
    hres_forecast = load_hres_forecast(baseline_store, lead_hours)
    forecasts, obs_aligned = align_all_sources(
        graphcast_ensemble, ifs_ensemble, hres_forecast, obs
    )

    all_train_mask = np.ones(obs_aligned.sizes["sample"], dtype=bool)
    rain_bin_labels = classify_rain_bin(forecasts["graphcast"].mean(dim="member", skipna=True))
    emos_results = fit_emos_csg(
        forecasts["ifs_ens"], obs_aligned, rain_bin_labels, all_train_mask, source="ifs_ens"
    )

    ifs_mean_sample = forecasts["ifs_ens"].mean(dim="member", skipna=True).isel(sample=-1)
    ifs_spread_sample = forecasts["ifs_ens"].std(dim="member", ddof=1, skipna=True).isel(
        sample=-1
    )
    bin_labels_sample = rain_bin_labels.isel(sample=-1)

    shape = ifs_mean_sample.shape
    probability = np.full(shape, np.nan, dtype=float)
    is_fallback = np.zeros(shape, dtype=bool)

    for bin_label in RAIN_BIN_LABELS:
        result = emos_results[bin_label]
        cell_mask = (bin_labels_sample.values == bin_label) & ~np.isnan(ifs_mean_sample.values)
        if not cell_mask.any():
            continue
        mean_cells = ifs_mean_sample.values[cell_mask]
        spread_cells = ifs_spread_sample.values[cell_mask]
        location, scale, shift = predict_csgd_params(result, mean_cells, spread_cells)
        probability[cell_mask] = exceedance_probability_csgd(
            location, scale, shift, EXTREME_THRESHOLD_MM
        )
        is_fallback[cell_mask] = result.is_fallback

    return {
        "latitude": ifs_mean_sample["latitude"].values,
        "longitude": ifs_mean_sample["longitude"].values,
        "probability": probability,
        "is_fallback": is_fallback,
        "sample_time": str(ifs_mean_sample["sample"].values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=DEFAULT_STORE)
    parser.add_argument("--weights-csv", default=DEFAULT_WEIGHTS_CSV)
    parser.add_argument("--lagged-store", default=DEFAULT_LAGGED_STORE)
    parser.add_argument("--ifs-ensemble-store", default=DEFAULT_IFS_ENSEMBLE_STORE)
    parser.add_argument("--out-blend", default=DEFAULT_OUT_BLEND)
    parser.add_argument("--out-probability", default=DEFAULT_OUT_PROBABILITY)
    args = parser.parse_args()

    blend_out_path = Path(args.out_blend)
    blend_out_path.parent.mkdir(parents=True, exist_ok=True)

    blend_arrays: dict[str, np.ndarray] = {"lead_hours": np.array(LEAD_HOURS)}
    for lead_hours in LEAD_HOURS:
        print(f"[lead {lead_hours:>3}h] building example blend grid...")
        grid = build_example_blend_grid(args.store, args.weights_csv, lead_hours)
        if "latitude" not in blend_arrays:
            blend_arrays["latitude"] = grid["latitude"].values
            blend_arrays["longitude"] = grid["longitude"].values
        blend_arrays[f"blend_lead_{lead_hours}"] = grid.values
        blend_arrays[f"sample_time_lead_{lead_hours}"] = np.array(str(grid["sample"].values))

    # numpy's own stub for savez(file, *args, **kwds) can't type-check a
    # dynamic **dict unpack (the dict's keys are lead-time-dependent, not
    # literal keyword names) -- this is a real numpy typing-stub
    # limitation, not a real type error (savez's runtime behavior handles
    # arbitrary **kwargs of arrays correctly).
    np.savez(blend_out_path, **blend_arrays)  # type: ignore[arg-type]
    print(f"Wrote {blend_out_path}")

    probability_out_path = Path(args.out_probability)
    probability_out_path.parent.mkdir(parents=True, exist_ok=True)

    probability_arrays: dict[str, np.ndarray] = {"lead_hours": np.array(LEAD_HOURS)}
    for lead_hours in LEAD_HOURS:
        print(f"[lead {lead_hours:>3}h] building example extreme-probability grid...")
        probability_grid = build_example_probability_grid(
            args.store, args.lagged_store, args.ifs_ensemble_store, lead_hours
        )
        if "latitude" not in probability_arrays:
            probability_arrays["latitude"] = np.asarray(probability_grid["latitude"])
            probability_arrays["longitude"] = np.asarray(probability_grid["longitude"])
        probability_arrays[f"probability_lead_{lead_hours}"] = np.asarray(
            probability_grid["probability"]
        )
        probability_arrays[f"is_fallback_lead_{lead_hours}"] = np.asarray(
            probability_grid["is_fallback"]
        )
        probability_arrays[f"sample_time_lead_{lead_hours}"] = np.array(
            probability_grid["sample_time"]
        )

    np.savez(probability_out_path, **probability_arrays)  # type: ignore[arg-type]
    print(f"Wrote {probability_out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
