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

**Extreme-probability grid** (`example_probability_grid.npz`): step 4/12
deliverable. Exports P(rain > 115.6 mm) and P(rain > 204.5 mm) for all
five lead times, each with per-cell method flags
(``"csgd"`` | ``"csgd+gpd_tail"`` | ``"fallback"``) from
``weavr.tail.exceedance_probability_with_tail``.

Output NPZ keys for each lead time L and threshold T:

- ``probability_{slug}_lead_{L}`` -- float (lat, lon), P(rain > T mm)
- ``is_fallback_{slug}_lead_{L}`` -- bool (lat, lon), True only when no
  fittable EMOS-CSG bin is available (true fallback, not GPD tail)
- ``method_{slug}_lead_{L}`` -- str (lat, lon), method flag

where slug is ``115p6`` for 115.6 mm and ``204p5`` for 204.5 mm.
Legacy keys ``probability_lead_{L}`` / ``is_fallback_lead_{L}`` are also
written, pointing to the 204.5 mm results, for backward compatibility.

The extreme-value GPD tail is fitted from IMD observed precipitation
(all available samples in the baseline store) pooled across Sreekala &
Babu zones above u = 64.5 mm via ``weavr.tail.fit_pooled_gpd``. EMOS-CSG
is also fitted on all samples -- a display snapshot, not the held-out
skill evaluation ``run_tier2_hierarchical_baseline.py`` performs (a real,
stated difference from that script's own convention). Bin classification
uses the GraphCast ensemble mean, the same convention the rest of the
pipeline uses.

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

from weavr.emos import fit_emos_csg  # noqa: E402
from weavr.rain_bins import classify_rain_bin  # noqa: E402
from weavr.regions import SREEKALA_BABU_ZONES, assign_regions  # noqa: E402
from weavr.tail import (  # noqa: E402
    DEFAULT_TAIL_THRESHOLD_U,
    TailFit,
    exceedance_probability_with_tail,
    fit_pooled_gpd,
)
from weavr.weighting import RegionWeightResult  # noqa: E402

DEFAULT_STORE = "data/baseline_2020_jjas.zarr"
DEFAULT_WEIGHTS_CSV = "results/tier1_regional_weights.csv"
DEFAULT_LAGGED_STORE = "data/lagged_ensemble_inputs_2020_jjas.zarr"
DEFAULT_IFS_ENSEMBLE_STORE = "data/ifs_ens_2020_jjas.zarr"
DEFAULT_OUT_BLEND = "dashboard/data/example_blend_grid.npz"
DEFAULT_OUT_PROBABILITY = "dashboard/data/example_probability_grid.npz"
# Both extreme thresholds exported (step 12 deliverable).
EXTREME_THRESHOLDS_MM: tuple[float, float] = (115.6, 204.5)
_THRESHOLD_SLUG: dict[float, str] = {115.6: "115p6", 204.5: "204p5"}


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


def latest_complete_sample_index(forecasts: dict[str, xr.DataArray]) -> int:
    """Index of the last sample on which *every* forecast source has data.

    The display snapshot cannot simply be `sample=-1`. The weekly IFS-ENS store
    holds 18 initialisations, while the daily baseline/lagged stores hold 122,
    so `align_all_sources` outer-joins onto 122 samples and leaves 104 of them
    entirely NaN for IFS-ENS. Taking `sample=-1` therefore selects a timestep
    where IFS-ENS is all-NaN: the blend silently loses one of its three inputs,
    and `build_example_probability_grids` -- which is driven by IFS-ENS alone --
    gets an empty array, so every cell is classified "fallback" with probability
    0 and the extreme-probability map comes out uniformly grey.

    Returns the largest index present in every source. Raises rather than
    falling back to a partial timestep: a grid that looks real but is built
    from missing data is the worst outcome here.
    """
    n_samples = min(int(ds.sizes["sample"]) for ds in forecasts.values())

    def has_data(index: int) -> bool:
        for ds in forecasts.values():
            arr = ds.isel(sample=index)
            if "member" in arr.dims:
                arr = arr.mean(dim="member", skipna=True)
            if not np.isfinite(np.asarray(arr.values, dtype=float)).any():
                return False
        return True

    for index in range(n_samples - 1, -1, -1):
        if has_data(index):
            return index

    raise ValueError(
        "no sample on which every forecast source has data: the provided stores "
        "have no initialisation in common, so no example grid can be built"
    )


def build_example_blend_grid(store_path: str, weights_csv: str, lead_hours: int) -> xr.DataArray:
    """The real Tier 1 blend, for one lead time, at the most recent sample on
    which every source has data (see `latest_complete_sample_index`)."""
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
    return blend.isel(sample=latest_complete_sample_index(forecasts))


def _collect_exceedances_by_region(
    obs: xr.Dataset,
    precip_var: str,
    threshold_u: float,
    region_labels: np.ndarray,
) -> dict[str, np.ndarray]:
    """Collect IMD precipitation exceedances above `threshold_u` per zone.

    Returns a mapping ``{region: 1D array of raw observations > threshold_u}``
    used as input to ``fit_pooled_gpd``. The exceedances are the raw observed
    values (not the excess values z = y - u); ``fit_pooled_gpd`` handles the
    subtraction internally.
    """
    precip = obs[precip_var].values  # shape (sample, lat, lon) or (lat, lon, sample)
    # Ensure shape is (n_samples, n_lat, n_lon) using xarray's standard dim ordering.
    lat_dim = obs[precip_var].dims.index("latitude") if "latitude" in obs[precip_var].dims else 1
    lon_dim = obs[precip_var].dims.index("longitude") if "longitude" in obs[precip_var].dims else 2
    # Flatten samples, keep spatial dims as (lat, lon).
    spatial_shape = (obs.sizes.get("latitude", precip.shape[lat_dim]),
                     obs.sizes.get("longitude", precip.shape[lon_dim]))
    precip_flat = precip.reshape(-1, *spatial_shape)  # (sample, lat, lon)

    exceedances: dict[str, list[float]] = {r: [] for r in SREEKALA_BABU_ZONES}
    for r in SREEKALA_BABU_ZONES:
        mask = (region_labels == r)  # (lat, lon)
        for s in range(precip_flat.shape[0]):
            vals = precip_flat[s][mask]  # 1D, one cell per region gridpoint
            vals = vals[~np.isnan(vals)]
            vals = vals[vals > threshold_u]
            exceedances[r].extend(vals.tolist())
    return {r: np.asarray(v, dtype=float) for r, v in exceedances.items()}


def build_example_probability_grids(
    baseline_store: str, lagged_store: str, ifs_store: str, lead_hours: int
) -> dict[str, np.ndarray | str]:
    """EMOS-CSG (`ifs_ens`) exceedance-probability grids for both thresholds.

    Returns results for P(rain > 115.6 mm) and P(rain > 204.5 mm) with
    per-cell method flags ("csgd" | "csgd+gpd_tail" | "fallback").

    The GPD tail is fitted from IMD observations (all available samples,
    not just the held-out test split) pooled across Sreekala & Babu zones.
    EMOS-CSG is also fitted on all available samples (a display snapshot,
    not a held-out skill evaluation -- a real, documented difference from
    ``run_tier2_hierarchical_baseline.py``).

    Bin classification uses the GraphCast ensemble mean, the same convention
    the rest of the pipeline uses (see ``run_tier2_hierarchical_baseline.py``).

    Returns
    -------
    dict with keys:
        ``latitude``, ``longitude``, ``sample_time`` -- grid coordinates.
        For each threshold slug ("115p6", "204p5"):
            ``probability_{slug}`` -- float (lat, lon), P(rain > T mm).
            ``is_fallback_{slug}`` -- bool (lat, lon), True only for true fallback.
            ``method_{slug}`` -- str (lat, lon), one of the three method flags.
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

    ifs_mean_all = forecasts["ifs_ens"].mean(dim="member", skipna=True)
    # Not `sample=-1`: the weekly IFS-ENS store covers 18 of the 122 aligned
    # samples, so the last one is all-NaN (see latest_complete_sample_index).
    display_index = latest_complete_sample_index(forecasts)
    ifs_mean_sample = ifs_mean_all.isel(sample=display_index)
    ifs_spread_sample = forecasts["ifs_ens"].std(dim="member", ddof=1, skipna=True).isel(
        sample=display_index
    )

    # Region labels for each gridpoint (shape: lat x lon).
    # assign_regions already returns a (latitude, longitude) DataArray, so the
    # meshgrid/ravel/reshape round-trip this used to do was both redundant and
    # broken: xr.DataArray has no .reshape(). The consumers below
    # (_collect_exceedances_by_region masks a numpy array with it, and
    # weavr.tail.exceedance_probability_with_tail indexes it as np.ndarray)
    # want a plain numpy array.
    lat_vals = ifs_mean_sample["latitude"].values
    lon_vals = ifs_mean_sample["longitude"].values
    region_labels = np.asarray(assign_regions(lat_vals, lon_vals).values)

    # Fit pooled GPD on all available IMD observations.
    precip_var = next(
        (v for v in obs.data_vars if "precip" in v.lower() or "rain" in v.lower()),
        next(iter(obs.data_vars)),  # fallback: first variable
    )
    exceedances_by_region = _collect_exceedances_by_region(
        obs, precip_var, DEFAULT_TAIL_THRESHOLD_U, region_labels
    )
    tail_fit: TailFit = fit_pooled_gpd(exceedances_by_region, DEFAULT_TAIL_THRESHOLD_U)

    out: dict[str, np.ndarray | str] = {
        "latitude": lat_vals,
        "longitude": lon_vals,
        "sample_time": str(ifs_mean_sample["sample"].values),
    }
    for threshold in EXTREME_THRESHOLDS_MM:
        slug = _THRESHOLD_SLUG[threshold]
        probs, methods = exceedance_probability_with_tail(
            forecast_values=ifs_mean_sample,
            ensemble_mean=ifs_mean_sample,
            ensemble_spread=ifs_spread_sample,
            emos_results_by_bin=emos_results,
            tail_fit=tail_fit,
            region=region_labels,
            threshold=threshold,
        )
        is_fallback = methods == "fallback"
        out[f"probability_{slug}"] = probs
        out[f"is_fallback_{slug}"] = is_fallback
        out[f"method_{slug}"] = methods
    return out


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
        print(f"[lead {lead_hours:>3}h] building example extreme-probability grids "
              f"(thresholds: {list(EXTREME_THRESHOLDS_MM)} mm)...")
        probability_grid = build_example_probability_grids(
            args.store, args.lagged_store, args.ifs_ensemble_store, lead_hours
        )
        if "latitude" not in probability_arrays:
            probability_arrays["latitude"] = np.asarray(probability_grid["latitude"])
            probability_arrays["longitude"] = np.asarray(probability_grid["longitude"])
        for threshold in EXTREME_THRESHOLDS_MM:
            slug = _THRESHOLD_SLUG[threshold]
            probability_arrays[f"probability_{slug}_lead_{lead_hours}"] = np.asarray(
                probability_grid[f"probability_{slug}"]
            )
            probability_arrays[f"is_fallback_{slug}_lead_{lead_hours}"] = np.asarray(
                probability_grid[f"is_fallback_{slug}"]
            )
            # Cast to a fixed-width unicode dtype rather than leaving the
            # labels as an object array: `np.save` can only read object arrays
            # back with `allow_pickle=True`, and the dashboard loader (like
            # `np.load`'s default) does not pass that, so an object-dtype
            # `method_*` array makes `load_probability_grid` raise
            # "Object arrays cannot be loaded" for *every* threshold.
            probability_arrays[f"method_{slug}_lead_{lead_hours}"] = np.asarray(
                probability_grid[f"method_{slug}"], dtype=str
            )
        # Backward-compatible legacy keys: point to the 204.5 mm results so that
        # existing code reading probability_lead_N / is_fallback_lead_N still works.
        probability_arrays[f"probability_lead_{lead_hours}"] = np.asarray(
            probability_grid[f"probability_{_THRESHOLD_SLUG[204.5]}"]
        )
        probability_arrays[f"is_fallback_lead_{lead_hours}"] = np.asarray(
            probability_grid[f"is_fallback_{_THRESHOLD_SLUG[204.5]}"]
        )
        probability_arrays[f"sample_time_lead_{lead_hours}"] = np.array(
            probability_grid["sample_time"]
        )

    np.savez(probability_out_path, **probability_arrays)  # type: ignore[arg-type]
    print(f"Wrote {probability_out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
