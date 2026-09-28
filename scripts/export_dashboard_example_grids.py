#!/usr/bin/env python3
"""Export a small, committed example blended-forecast grid for the dashboard.

Per `docs/phase7-dashboard-scope.md`'s decision (routed to the user via
`AskUserQuestion`): the blended-map view reads a small, committed example
export rather than requiring `data/baseline_2020_jjas.zarr` to exist
locally on whoever opens the dashboard.

Reuses `run_tier1_regional_baseline.py`'s own `load_aligned_forecasts_and_obs`
/ `build_region_weight_grid` / `blend_with_region_weights` (already real,
already tested) rather than reimplementing the blend a third time --
`run_daily_pipeline.py` (Phase 6, PR #39) already does the same. Reuses
`results/tier1_regional_weights.csv`'s already-fitted weights rather than
re-fitting them here: this script only renders a snapshot of that real fit,
it does not produce a new one.

For each of the 5 lead times, blends the single most recent real aligned
sample (not a synthetic one, and not an average across samples, which would
blur the real spatial pattern a viewer should see) using that lead's real
fitted regional weights, and writes every lead's grid to one committed
`dashboard/data/example_blend_grid.npz`.

Usage:
    python scripts/export_dashboard_example_grids.py
        [--store PATH] [--weights-csv PATH] [--out PATH]
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

from weavr.regions import assign_regions  # noqa: E402
from weavr.weighting import RegionWeightResult  # noqa: E402

DEFAULT_STORE = "data/baseline_2020_jjas.zarr"
DEFAULT_WEIGHTS_CSV = "results/tier1_regional_weights.csv"
DEFAULT_OUT = "dashboard/data/example_blend_grid.npz"


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=DEFAULT_STORE)
    parser.add_argument("--weights-csv", default=DEFAULT_WEIGHTS_CSV)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {"lead_hours": np.array(LEAD_HOURS)}
    for lead_hours in LEAD_HOURS:
        print(f"[lead {lead_hours:>3}h] building example blend grid...")
        grid = build_example_blend_grid(args.store, args.weights_csv, lead_hours)
        if "latitude" not in arrays:
            arrays["latitude"] = grid["latitude"].values
            arrays["longitude"] = grid["longitude"].values
        arrays[f"blend_lead_{lead_hours}"] = grid.values
        arrays[f"sample_time_lead_{lead_hours}"] = np.array(str(grid["sample"].values))

    # numpy's own stub for savez(file, *args, **kwds) can't type-check a
    # dynamic **dict unpack (the dict's keys are lead-time-dependent, not
    # literal keyword names) -- this is a real numpy typing-stub
    # limitation, not a real type error (savez's runtime behavior handles
    # arbitrary **kwargs of arrays correctly).
    np.savez(out_path, **arrays)  # type: ignore[arg-type]
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
