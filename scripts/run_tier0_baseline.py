#!/usr/bin/env python3
"""Run the Tier 0 equal-weight baseline: Phase 1's actual exit criterion.

Computes the simplest possible blend -- an unweighted mean across all
forecast sources that carry a given variable -- and scores it with every
function in src/weavr/verify.py, against IMD gauges, using the split
strategy from src/weavr/splits.py. This is the number every later tier
(Phase 3's regional weights, Phase 4's hierarchical BMA/EMOS, ...) must beat
to justify its own added complexity. See docs/tier0-baseline-results.md for
the actual numbers this produces and every limitation found running it.

Scope, decided by what the data actually supports, not assumed:

- **Precipitation only.** The baseline store's only IMD ground truth is
  `imd_observed.rain` -- there is no IMD temperature observation in the
  store at all (build_baseline_store.py only ever pulled IMD's `rain`
  product). IMD does publish a gridded temperature product, but at 1 deg
  native resolution -- coarser than weavr's locked 0.25 deg grid, so
  `regrid_to_common` would (correctly) refuse it as upsampling, the same
  guard that already blocks a temperature ACC climatology (see
  docs/phase-1-data-requirements.md). Verifying 2m_temperature against real
  IMD ground truth is therefore not possible with data available today;
  Tier 0 scores precipitation (`total_precipitation_24hr`) only, and this is
  documented rather than silently narrowing scope.
- Of the 4 forecast sources, only graphcast/hres/ifs_ens_mean carry
  `total_precipitation_24hr` (pangu's archive has no precipitation variable
  at all, per docs/baseline-store.md) -- the equal-weight mean is over these
  3, not all 4.
- CRPS/Brier are not computed: no ensemble-shaped source exists in the store
  (deferred to Phase 2's lagged AI ensembles, per
  docs/phase-1-data-requirements.md).
- leave_one_year_out is not usable (the store is one season); every lead
  time uses seasonal_block_split instead, which this script states
  explicitly rather than silently falling back.

A real unit bug was caught running this, not assumed away: WeatherBench 2's
`total_precipitation_24hr` is in **meters** (the ECMWF/GRIB convention),
while IMD's `rain` is in **millimeters**. Scoring them directly against each
other without converting produces a "bias" of roughly -obs_mean -- which
looks like a plausible negative number, not an obvious crash -- because the
forecast is effectively ~0 in the wrong units. Caught by checking the actual
magnitudes (`fc.mean() ~ 0.006`, `obs.mean() ~ 5.7`) before trusting any
score, not by assuming matching units. Fixed by multiplying forecast
precipitation by 1000 before scoring (`PRECIP_M_TO_MM` below).

Usage:
    python scripts/run_tier0_baseline.py [--store PATH] [--climatology PATH]
                                          [--out-csv PATH]
                                          [--test-fraction F]
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

from weavr import verify as V  # noqa: E402
from weavr.grid import IMD_DAY_START_HOUR_UTC  # noqa: E402
from weavr.splits import (  # noqa: E402
    InsufficientTimeBlocksError,
    leave_one_year_out,
    seasonal_block_split,
)

PRECIP_VARIABLE = "total_precipitation_24hr"
PRECIP_M_TO_MM = 1000.0  # WeatherBench 2 precip is in meters; IMD's is in mm.
LEAD_HOURS = [24, 48, 72, 96, 120]
FORECAST_SOURCE_NAMES = ["graphcast", "pangu", "hres", "ifs_ens_mean"]
NEIGHBORHOOD_SIZE = 5  # grid cells (~1.25 deg edge) -- a representative mesoscale window.


def _align_to_imd_day(da: xr.DataArray, lead_hours: int) -> xr.DataArray:
    """Reindex a forecast DataArray from init_time to the IMD day its lead validates on.

    valid_time = init_time + lead_hours; the IMD day it falls in is
    `(valid_time - 3h).normalize()` -- the same 03-03 UTC window
    `weavr.grid.resample_to_imd_day` uses, applied here to a forecast's
    lead-adjusted valid time instead of a continuous observation series
    (a genuinely different problem, see docs/baseline-store.md).
    """
    init_times = pd.DatetimeIndex(da["time"].values)
    valid_time = init_times + pd.Timedelta(hours=lead_hours)
    imd_day = (valid_time - pd.Timedelta(hours=IMD_DAY_START_HOUR_UTC)).normalize()
    return da.assign_coords(time=imd_day).rename(time="sample")


def equal_weight_mean_and_obs(
    sources: dict[str, xr.Dataset],
    obs: xr.Dataset,
    variable: str,
    lead_hours: int,
) -> tuple[xr.DataArray, xr.DataArray, list[str]]:
    """Build the equal-weight mean forecast and its matching obs, aligned by IMD day.

    Returns (mean_forecast, obs_aligned, contributing_source_names), both
    DataArrays indexed by a shared `sample` dim, with any sample missing IMD
    ground truth already dropped.
    """
    contributing = [name for name, ds in sources.items() if variable in ds.data_vars]
    aligned = []
    for name in contributing:
        da = sources[name][variable].sel(prediction_timedelta=lead_hours).load()
        if variable == PRECIP_VARIABLE:
            da = da * PRECIP_M_TO_MM
        aligned.append(_align_to_imd_day(da, lead_hours))

    stacked = xr.concat(aligned, dim="source")
    mean_forecast = stacked.mean(dim="source", skipna=True)

    obs_aligned = obs["rain"].reindex(time=mean_forecast["sample"].values).rename(time="sample")
    has_obs = ~obs_aligned.isnull().all(dim=["latitude", "longitude"])
    return (
        mean_forecast.isel(sample=has_obs.values),
        obs_aligned.isel(sample=has_obs.values),
        contributing,
    )


def score_lead(
    mean_forecast: xr.DataArray,
    obs: xr.DataArray,
    climatology: xr.Dataset,
    test_fraction: float,
    thresholds: tuple[float, ...],
    neighborhood_size: int,
) -> dict:
    n_samples = mean_forecast.sizes["sample"]
    sample_times = pd.DatetimeIndex(mean_forecast["sample"].values)

    split_kind = "leave_one_year_out"
    try:
        # Only meaningful with 2+ distinct years -- checked, not assumed:
        # this raises against today's single-season store, so the except
        # branch is what actually runs. Kept so this script automatically
        # starts using the more rigorous split the day a second season is
        # added to the store, with no code change needed here.
        train_mask, test_mask = next(iter(leave_one_year_out(sample_times)))
    except InsufficientTimeBlocksError:
        split_kind = "seasonal_block_split (single-season store; leave_one_year_out not usable)"
        train_mask, test_mask = seasonal_block_split(sample_times, test_fraction=test_fraction)

    test_forecast = mean_forecast.isel(sample=test_mask)
    test_obs = obs.isel(sample=test_mask)

    climatology_mean = climatology["rain"].mean(dim="time", skipna=True)

    fss_by_threshold = {
        t: float(V.fss(test_forecast, test_obs, threshold=t, neighborhood_size=neighborhood_size))
        for t in thresholds
    }
    contingency = V.contingency_scores(test_forecast, test_obs, thresholds=thresholds)

    return {
        "n_samples": n_samples,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "split": split_kind,
        "rmse_mm": float(V.rmse(test_forecast, test_obs)),
        "bias_mm": float(V.bias(test_forecast, test_obs)),
        "acc": float(V.acc(test_forecast, test_obs, climatology_mean)),
        "seeps": float(
            V.seeps(test_forecast, test_obs, climatology["rain"], climatology_dim="time")
        ),
        "fss": fss_by_threshold,
        "contingency": {t: {k: float(v) for k, v in s.items()} for t, s in contingency.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="data/baseline_2020_jjas.zarr")
    parser.add_argument("--climatology", default="data/imd_seeps_climatology_jjas.zarr")
    parser.add_argument("--out-csv", default="results/tier0_baseline.csv")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--neighborhood-size", type=int, default=NEIGHBORHOOD_SIZE)
    args = parser.parse_args()

    sources = {
        name: xr.open_zarr(args.store, group=name, consolidated=True)
        for name in FORECAST_SOURCE_NAMES
    }
    obs = xr.open_zarr(args.store, group="imd_observed", consolidated=True).load()
    climatology = load_climatology(args.climatology).load()

    print(f"Tier 0 baseline: equal-weight mean, scored against {args.store}'s imd_observed group")
    print(f"Climatology: {args.climatology} ({climatology.sizes['time']} JJAS days)")
    print("Scope: precipitation only -- no matching-resolution IMD temperature ground truth")
    print("CRPS/Brier: not computed -- no ensemble-shaped source in this store (deferred, Phase 2)")

    rows = []
    for lead_hours in LEAD_HOURS:
        mean_forecast, obs_aligned, contributing = equal_weight_mean_and_obs(
            sources, obs, PRECIP_VARIABLE, lead_hours
        )
        result = score_lead(
            mean_forecast,
            obs_aligned,
            climatology,
            args.test_fraction,
            V.IMD_RAIN_THRESHOLDS_MM,
            args.neighborhood_size,
        )
        result["lead_hours"] = lead_hours
        result["sources"] = "+".join(contributing)
        rows.append(result)
        print(f"[lead {lead_hours:>3}h] {result}")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "lead_hours",
        "sources",
        "split",
        "n_samples",
        "n_train",
        "n_test",
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
