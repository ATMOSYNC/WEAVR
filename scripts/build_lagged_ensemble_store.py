#!/usr/bin/env python3
"""Build the Phase 2 lagged-ensemble input store: the extra AI-model init times
a +/-4-starts, 12h-spaced lagged ensemble needs, on top of Phase 1's baseline store.

A separate script/store, not an extension of build_baseline_store.py -- checked
live (see docs/baseline-store.md's Phase 2 addition section), the shape of what's
needed here (a per-nominal-forecast cluster of extra init times, each queried at
a lead adjusted to land on a shared valid time) doesn't fit build_baseline_store's
flat (time, prediction_timedelta) sampling grid without distortion, the same
reasoning that made scripts/build_seeps_climatology.py a separate script rather
than an extension in Phase 1.

Scope, decided by what was checked, not assumed:

- **GraphCast and Pangu only.** These are the phase-plan's named "AI models" --
  the ones with a single deterministic forecast per init time and no ensemble
  spread of their own. HRES (NWP deterministic) and ifs_ens_mean (already a
  collapsed NWP ensemble mean) are untouched; giving a real NWP ensemble a
  pseudo-spread via lagging is a different problem this script doesn't solve.
- **Both archives are confirmed 12-hourly at their native resolution** (checked
  live by listing the actual GCS zarr stores, not by trusting the docstring
  claim in build_baseline_store.py) and both span well outside the +/-48h
  window needed around every one of Phase 1's 18 weekly JJAS-2020 sample
  points -- GraphCast covers 2019-11-16..2021-01-31, Pangu covers
  2018-01-01..2022-12-31. **No sample point runs off either archive's edge**,
  so every nominal forecast gets a full 9-member window at the *time* axis.
  This is a checked finding, not an assumption -- see docs/baseline-store.md.
- **Member availability is limited by *lead*, not by archive edge, and only at
  the shortest lead times -- differently per variable.** Both archives'
  `prediction_timedelta` axis starts at 6h (there is no 0h or negative lead).
  A lagged member initialized *after* the nominal init time needs a *shorter*
  lead to reach the same valid time (required_lead = nominal_lead -
  offset_hours); at nominal_lead=24h, an offset of +36h or +48h would need a
  lead of -12h or -24h, which does not exist for either variable. This alone
  gives **`2m_temperature`: lead=24h gets 6 of 9 members, lead=48h gets 8 of
  9, lead=72h+ gets the full 9.** `total_precipitation_24hr` has a second,
  stricter constraint on top of that, found only by checking the actual
  fetched values (not assumed to match temperature's pattern): the source
  index exists at every 6h step, but the *data itself* is NaN for any lead
  below 24h -- a 24-hour accumulation isn't defined until a full 24h of
  forecast has elapsed. This gives **`total_precipitation_24hr`: lead=24h
  gets 5 of 9 members, lead=48h gets 7 of 9, lead=72h+ gets the full 9** (see
  `src/weavr/ensemble.py`'s `build_lagged_ensemble`, which checks fetched
  values for this, not just index existence). Documented here and in
  docs/baseline-store.md rather than padding the missing members with a
  substitute value.
- **Fetch cost, measured live**: a single-chunk fetch (one (time,
  prediction_timedelta) pair, full India-sliced lat/lon) measured ~2.3-2.6s
  for both sources -- consistent with Phase 0's ~0.75-2s estimate for this
  chunk shape. The exact (not cartesian-superset) set of new chunks this
  script fetches is 41 valid (nominal_time, lead, offset) combinations per
  variable per weekly sample (6+8+9+9+9, including the already-in-baseline-
  store offset=0 case, refetched here for a self-contained store rather than
  cross-referencing two stores) x 18 weekly samples x 3 (source, variable)
  pairs (GraphCast: 2m_temperature + total_precipitation_24hr, Pangu:
  2m_temperature only) = 2214 chunk fetches, ~=92 minutes serial at the
  measured per-chunk latency. This is well short of the ~2-2.5h/~30-35GB IFS
  full-ensemble pull Phase 1 deferred to Phase 2 -- tractable to just run,
  not a tradeoff requiring a scope cut. Vectorized xarray selection (a single
  `.sel(time=.., prediction_timedelta=..)` call with array indexers, rather
  than one Python-level fetch per combination) lets dask parallelize the
  actual GCS requests instead of paying the full serial estimate.

Usage:
    python scripts/build_lagged_ensemble_store.py [--out PATH]
        [--start DATE] [--end DATE] [--lead-hours H [H ...]]
        [--init-cadence-days N] [--n-lags N] [--lag-spacing-hours H]

Idempotent: writes a JSON manifest next to the store recording which source
succeeded; a source already marked "ok" is skipped on re-run unless --force
is passed -- same pattern as build_baseline_store.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Literal

import dask
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_baseline_store import (  # noqa: E402
    DEFAULT_END,
    DEFAULT_INIT_CADENCE_DAYS,
    DEFAULT_LEAD_HOURS,
    DEFAULT_START,
    FORECAST_SOURCES,
    GCS_ANON,
    INDIA_LAT_SLICE,
    INDIA_LON_SLICE_0_360,
    ForecastSource,
    _clear_encoding,
    _lat_slice_for,
    _weekly_init_times,
)

from weavr.grid import SourceTooCoarseError, regrid_to_common  # noqa: E402

# The AI models the phase-plan names for lagged-ensemble treatment -- not
# HRES/ifs_ens_mean, which are NWP sources, not deterministic AI models.
AI_SOURCE_NAMES = ("graphcast", "pangu")

DEFAULT_N_LAGS = 4
DEFAULT_LAG_SPACING_HOURS = 12

# A source's prediction_timedelta axis: confirmed live (both GraphCast's and
# Pangu's WeatherBench 2 archives) to run 6h..240h in 6h steps -- no 0h or
# negative lead exists. A lagged member needing a lead outside this range is
# not fetchable, not a bug to route around.
MIN_SOURCE_LEAD_HOURS = 6
MAX_SOURCE_LEAD_HOURS = 240


def _lag_offsets(n_lags: int, spacing_hours: int) -> list[int]:
    """+/-n_lags starts, spacing_hours apart, including the nominal (0h) member."""
    return [i * spacing_hours for i in range(-n_lags, n_lags + 1)]


def _valid_combos(
    nominal_times: np.ndarray,
    lead_hours: list[int],
    offsets: list[int],
) -> list[tuple[int, int, int, np.datetime64, int]]:
    """Every (i_nominal, i_lead, i_offset, source_time, source_lead) combo whose
    required lead actually exists in a source's prediction_timedelta axis.
    """
    combos = []
    for i, nt in enumerate(nominal_times):
        for j, lead in enumerate(lead_hours):
            for k, offset in enumerate(offsets):
                source_lead = lead - offset
                if MIN_SOURCE_LEAD_HOURS <= source_lead <= MAX_SOURCE_LEAD_HOURS:
                    source_time = nt + np.timedelta64(offset, "h")
                    combos.append((i, j, k, source_time, source_lead))
    return combos


def deduplicate_fetch_pairs(
    combos: list[tuple[int, int, int, np.datetime64, int]],
) -> list[tuple[np.datetime64, int]]:
    """Return sorted unique (source_time, source_lead) pairs across all combos."""
    seen = set()
    unique = []
    for _, _, _, source_time, source_lead in combos:
        key = (source_time, int(source_lead))
        if key not in seen:
            seen.add(key)
            unique.append(key)
    unique.sort(key=lambda p: (p[0], p[1]))
    return unique


def _get_nominal_times(
    ds: xr.Dataset,
    start: str,
    end: str,
    init_cadence_days: int,
    nominal_times_from: str | Path | None = None,
) -> np.ndarray:
    """Resolve nominal init times either from an existing store or using cadence."""
    if nominal_times_from:
        p = Path(nominal_times_from)
        if not p.exists():
            raise FileNotFoundError(f"Store for nominal times not found: {p}")
        try:
            import zarr

            r = zarr.open_group(p, mode="r")
            keys = list(r.group_keys())
            group = "graphcast" if "graphcast" in keys else (keys[0] if keys else None)
            if group is not None:
                store_ds = xr.open_zarr(p, group=group, consolidated=True)
            else:
                store_ds = xr.open_zarr(p, consolidated=True)
            coord = "time" if "time" in store_ds.coords else "nominal_time"
            times = store_ds[coord].values
        except Exception:
            store_ds = xr.open_zarr(p, consolidated=True)
            coord = "time" if "time" in store_ds.coords else "nominal_time"
            times = store_ds[coord].values
        times = times[(times >= np.datetime64(start)) & (times <= np.datetime64(end))]
        return times
    return _weekly_init_times(ds, start, end, init_cadence_days)


def build_lagged_group(
    source: ForecastSource,
    start: str,
    end: str,
    lead_hours: list[int],
    init_cadence_days: int,
    n_lags: int,
    lag_spacing_hours: int,
    nominal_times_from: str | Path | None = None,
    batch_size: int = 50,
    workers: int = 6,
) -> tuple[xr.Dataset, dict]:
    ds = xr.open_zarr(source.zarr_path, storage_options=GCS_ANON, consolidated=True)
    ds = ds[[v for v in source.variables if v in ds.data_vars]]

    nominal_times = _get_nominal_times(
        ds, start, end, init_cadence_days, nominal_times_from=nominal_times_from
    )

    lat_slice = _lat_slice_for(ds, source.lat_dim, INDIA_LAT_SLICE.start, INDIA_LAT_SLICE.stop)
    ds = ds.sel({source.lat_dim: lat_slice, source.lon_dim: INDIA_LON_SLICE_0_360})
    if source.lat_dim != "latitude":
        ds = ds.rename({source.lat_dim: "latitude"})
    if source.lon_dim != "longitude":
        ds = ds.rename({source.lon_dim: "longitude"})

    offsets = _lag_offsets(n_lags, lag_spacing_hours)
    combos = _valid_combos(nominal_times, lead_hours, offsets)

    n_nominal, n_leads, n_offsets = len(nominal_times), len(lead_hours), len(offsets)
    n_lat = ds.sizes["latitude"]
    n_lon = ds.sizes["longitude"]

    out_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        var: (
            ("nominal_time", "lead_hours", "member_offset_hours", "latitude", "longitude"),
            np.full((n_nominal, n_leads, n_offsets, n_lat, n_lon), np.nan, dtype=np.float64),
        )
        for var in ds.data_vars
    }

    # Deduplicate fetch pairs across all nominal inits and leads
    unique_pairs = deduplicate_fetch_pairs(combos)
    pair_to_combos: dict[tuple[np.datetime64, int], list[tuple[int, int, int]]] = {}
    for i, j, k, source_time, source_lead in combos:
        pair_to_combos.setdefault((source_time, int(source_lead)), []).append((i, j, k))

    n_unique = len(unique_pairs)
    n_batches = (n_unique + batch_size - 1) // batch_size
    pct_reduction = 100.0 * (1.0 - n_unique / max(len(combos), 1))
    print(
        f"  [{source.name}] {n_nominal} nominal inits ({len(combos)} total combos) "
        f"deduplicated to {n_unique} unique (source_time, source_lead) pairs "
        f"({pct_reduction:.1f}% reduction). "
        f"Fetching in {n_batches} batches (batch size {batch_size}) ...",
        flush=True,
    )

    t0_fetch = time.time()
    for batch_idx in range(n_batches):
        batch = unique_pairs[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        time_indexer = xr.DataArray([p[0] for p in batch], dims="fetch_pair")
        lead_indexer = xr.DataArray([p[1] for p in batch], dims="fetch_pair")

        t_batch_0 = time.time()
        # The concurrency here is dask's, not an explicit thread pool: one
        # vectorised .sel() over the whole batch becomes `batch_size`
        # independent chunk reads, which dask fetches in parallel. `workers`
        # sets how many at once. Wiring it to the scheduler is what makes the
        # --workers flag (and the value recorded in the manifest) mean
        # something -- it was previously accepted and stored but unused.
        with dask.config.set(scheduler="threads", num_workers=workers):
            fetched = ds.sel(time=time_indexer, prediction_timedelta=lead_indexer).load()
        dt_batch = time.time() - t_batch_0

        for var in ds.data_vars:
            values = fetched[var].values
            for pair_idx, p in enumerate(batch):
                val_slice = values[pair_idx]
                for (i, j, k) in pair_to_combos[p]:
                    out_vars[var][1][i, j, k] = val_slice

        print(
            f"    batch {batch_idx + 1}/{n_batches} done "
            f"({len(batch)} pairs in {dt_batch:.2f}s, "
            f"elapsed: {time.time() - t0_fetch:.1f}s)",
            flush=True,
        )

    lat_values = ds["latitude"].values
    lon_values = ds["longitude"].values

    out = xr.Dataset(
        out_vars,
        coords={
            "nominal_time": nominal_times,
            "lead_hours": lead_hours,
            "member_offset_hours": offsets,
            "latitude": lat_values,
            "longitude": lon_values,
        },
    )
    out = _clear_encoding(out)

    try:
        out = regrid_to_common(out)
    except SourceTooCoarseError:
        raise

    n_valid_by_lead = {
        int(lead): sum(1 for c in combos if lead_hours[c[1]] == lead) // n_nominal
        for lead in lead_hours
    }
    info = {
        "status": "ok",
        "variables": list(out.data_vars),
        "n_nominal_times": n_nominal,
        "n_lead_hours": n_leads,
        "n_offsets": n_offsets,
        "n_members_by_lead": n_valid_by_lead,
        "n_chunks_fetched": len(unique_pairs) * len(list(ds.data_vars)),
        "n_nominal_combos": len(combos),
        "n_unique_pairs": len(unique_pairs),
        "init_cadence_days": init_cadence_days,
        "nominal_times_from": str(nominal_times_from) if nominal_times_from else None,
        "known_gaps": source.known_gaps,
        "source_archive_path": source.zarr_path,
    }
    return out, info


def _load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, default=str))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=os.environ.get(
            "WEAVR_LAGGED_ENSEMBLE_STORE_PATH", "data/lagged_ensemble_inputs_2020_jjas.zarr"
        ),
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--lead-hours", type=int, nargs="+", default=DEFAULT_LEAD_HOURS)
    parser.add_argument("--init-cadence-days", type=int, default=DEFAULT_INIT_CADENCE_DAYS)
    parser.add_argument("--n-lags", type=int, default=DEFAULT_N_LAGS)
    parser.add_argument("--lag-spacing-hours", type=int, default=DEFAULT_LAG_SPACING_HOURS)
    parser.add_argument(
        "--nominal-times-from",
        default=None,
        help="Path to baseline zarr store to read nominal init times from",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of unique (source_time, source_lead) pairs per fetch batch",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of concurrent worker threads for batch fetching (default: 6)",
    )
    parser.add_argument("--force", action="store_true", help="re-pull sources already marked ok")
    args = parser.parse_args()

    store_path = Path(args.out)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = store_path.with_suffix(".manifest.json")
    manifest = _load_manifest(manifest_path)

    ai_sources = [s for s in FORECAST_SOURCES if s.name in AI_SOURCE_NAMES]

    print(f"Building lagged-ensemble input store at {store_path}")
    # --nominal-times-from overrides the cadence entirely, so printing the
    # cadence flag alongside it would claim a 7-day spacing while actually
    # building 122 daily inits -- misleading in exactly the log a reader
    # checks to confirm what was built.
    cadence_description = (
        f"nominal times from {args.nominal_times_from}"
        if args.nominal_times_from
        else f"init cadence {args.init_cadence_days}d"
    )
    print(
        f"Window: {args.start} .. {args.end}, {cadence_description}, "
        f"lead hours {args.lead_hours}, +/-{args.n_lags} lags @ {args.lag_spacing_hours}h, "
        f"workers {args.workers}"
    )

    for source in ai_sources:
        group_name = source.name
        if not args.force and manifest.get(group_name, {}).get("status") == "ok":
            print(f"[skip] {group_name}: already marked ok in manifest")
            continue

        print(f"[fetch] {group_name} ...")
        try:
            ds, info = build_lagged_group(
                source,
                args.start,
                args.end,
                args.lead_hours,
                args.init_cadence_days,
                args.n_lags,
                args.lag_spacing_hours,
                nominal_times_from=args.nominal_times_from,
                batch_size=args.batch_size,
                workers=args.workers,
            )
            mode: Literal["w", "a"] = "w" if not store_path.exists() else "a"
            ds.to_zarr(store_path, group=group_name, mode=mode)
            info["fetched_at"] = datetime.utcnow().isoformat() + "Z"
            manifest[group_name] = info
            print(f"[ok]   {group_name}: {info}")
        except Exception as exc:  # a source failing must not abort the others
            manifest[group_name] = {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "fetched_at": datetime.utcnow().isoformat() + "Z",
            }
            print(f"[FAIL] {group_name}: {exc}", file=sys.stderr)
        finally:
            manifest["_meta"] = {
                "init_cadence_days": args.init_cadence_days,
                "nominal_times_from": (
                    str(args.nominal_times_from) if args.nominal_times_from else None
                ),
                "start": args.start,
                "end": args.end,
                "lead_hours": args.lead_hours,
                "n_lags": args.n_lags,
                "lag_spacing_hours": args.lag_spacing_hours,
                "batch_size": args.batch_size,
                "workers": args.workers,
            }
            _save_manifest(manifest_path, manifest)

    print("\n--- Summary ---")
    for group_name, info in manifest.items():
        if group_name.startswith("_"):
            continue
        status = info.get("status")
        if status == "ok":
            print(
                f"  {group_name}: OK, variables={info.get('variables')}, "
                f"members_by_lead={info.get('n_members_by_lead')}"
            )
        else:
            print(f"  {group_name}: FAILED - {info.get('error')}")

    n_failed = sum(
        1 for k, i in manifest.items() if not k.startswith("_") and i.get("status") != "ok"
    )
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
