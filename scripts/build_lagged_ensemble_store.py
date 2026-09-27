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
import traceback
from datetime import datetime
from pathlib import Path
from typing import Literal

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


def build_lagged_group(
    source: ForecastSource,
    start: str,
    end: str,
    lead_hours: list[int],
    init_cadence_days: int,
    n_lags: int,
    lag_spacing_hours: int,
) -> tuple[xr.Dataset, dict]:
    ds = xr.open_zarr(source.zarr_path, storage_options=GCS_ANON, consolidated=True)
    ds = ds[[v for v in source.variables if v in ds.data_vars]]

    nominal_times = _weekly_init_times(ds, start, end, init_cadence_days)

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

    # Fetched in one batch per nominal week (~41 combos each), not one giant
    # vectorized .sel() over all ~740 combos at once -- a single mega-call
    # was measured live to stall indefinitely (bytes-in flat for 45+s, the
    # async I/O thread parked in a socket wait with no forward progress)
    # rather than merely being slow. Batching per week keeps each vectorized
    # selection small enough to actually complete, at the cost of one dask
    # `.load()` call per nominal time instead of a single call overall.
    out_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        var: (
            ("nominal_time", "lead_hours", "member_offset_hours", "latitude", "longitude"),
            np.full((n_nominal, n_leads, n_offsets, n_lat, n_lon), np.nan, dtype=np.float64),
        )
        for var in ds.data_vars
    }

    combos_by_nominal: dict[int, list[tuple[int, int, int, np.datetime64, int]]] = {}
    for combo in combos:
        combos_by_nominal.setdefault(combo[0], []).append(combo)

    for i in sorted(combos_by_nominal):
        week_combos = combos_by_nominal[i]
        time_indexer = xr.DataArray([c[3] for c in week_combos], dims="combo")
        lead_indexer = xr.DataArray([c[4] for c in week_combos], dims="combo")
        fetched = ds.sel(time=time_indexer, prediction_timedelta=lead_indexer).load()
        print(
            f"  [{source.name}] nominal week {i + 1}/{n_nominal} "
            f"({nominal_times[i]}): {len(week_combos)} combos fetched"
        )

        for var in ds.data_vars:
            values = fetched[var].values
            for combo_idx, (_, j, k, _, _) in enumerate(week_combos):
                out_vars[var][1][i, j, k] = values[combo_idx]

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
        "n_chunks_fetched": len(combos) * len(list(ds.data_vars)),
        "known_gaps": source.known_gaps,
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
    parser.add_argument("--force", action="store_true", help="re-pull sources already marked ok")
    args = parser.parse_args()

    store_path = Path(args.out)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = store_path.with_suffix(".manifest.json")
    manifest = _load_manifest(manifest_path)

    ai_sources = [s for s in FORECAST_SOURCES if s.name in AI_SOURCE_NAMES]

    print(f"Building lagged-ensemble input store at {store_path}")
    print(
        f"Window: {args.start} .. {args.end}, init cadence {args.init_cadence_days}d, "
        f"lead hours {args.lead_hours}, +/-{args.n_lags} lags @ {args.lag_spacing_hours}h"
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
            _save_manifest(manifest_path, manifest)

    print("\n--- Summary ---")
    for group_name, info in manifest.items():
        status = info.get("status")
        if status == "ok":
            print(
                f"  {group_name}: OK, variables={info.get('variables')}, "
                f"members_by_lead={info.get('n_members_by_lead')}"
            )
        else:
            print(f"  {group_name}: FAILED - {info.get('error')}")

    n_failed = sum(1 for i in manifest.values() if i.get("status") != "ok")
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
