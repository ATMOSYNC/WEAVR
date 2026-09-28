#!/usr/bin/env python3
"""Build the Phase 4 real IFS 50-member ensemble store.

Phase 0/1 deliberately deferred this exact pull: `docs/baseline-store.md`'s
`ifs_ens_mean` row already flagged "full-ensemble access is a Phase 1+
decision if BMA/EMOS needs individual members," and
`docs/phase-1-data-requirements.md` measured the real cost of getting it
(~47s/chunk, ~2-2.5h/~30-35GB at Phase 0's sampling density) before
deferring it to whichever later phase actually needed individual members.
Phase 4 is that phase -- see `docs/phase4-data-and-combiner-scope.md` for
the full decision (a genuine cost/benefit tradeoff, put to the user via
`AskUserQuestion` rather than decided silently).

Real chunking confirmed live before writing this script (not re-quoted
from the earlier estimate): `gs://weatherbench2/datasets/ifs_ens/
2018-2022-1440x721.zarr` chunks every variable as one `(1, 50, 1, 721,
1440)` block per `(time, prediction_timedelta)` pair -- the full 50-member,
full-global-grid slab in a single I/O chunk, matching
docs/phase-1-data-requirements.md's own finding. A live single-chunk probe
measured **~34s and ~208MB** for one `(time, prediction_timedelta)` pair of
`total_precipitation_24hr` alone -- close to the earlier ~47s/chunk
estimate (that estimate covered 2 variables per chunk-pair; this store
pulls precipitation only, per this project's precipitation-only scope
throughout every prior tier, so real cost here is smaller: 18 timestamps x
5 leads = 90 chunks x ~208MB = ~18.7GB, ~50 minutes if fetched serially).

This project's exact 18 weekly JJAS-2020 timestamps are confirmed (checked
live, not re-derived) to all exist verbatim in the raw archive's own time
index -- selected directly by real value here (`.sel(time=<exact list>)`)
rather than re-running `build_baseline_store.py`'s `_weekly_init_times`
cadence logic against this archive independently, which risks a
subtly-different stride/anchor picking different timestamps. This
guarantees the new store's samples align 1:1 with
`data/baseline_2020_jjas.zarr` with no separate alignment logic needed
downstream.

Fetched in one batch per timestamp (5 leads x 50 members x full lat/lon
each, ~1GB/batch), not one giant vectorized `.sel()` over all 90 combos at
once -- following `scripts/build_lagged_ensemble_store.py`'s own
documented fix for the real stall a single mega-call caused there. Also
resumable: each fetched timestamp is cached to a small per-timestamp
NetCDF file in a staging directory as soon as it arrives, with an
idempotent manifest tracking which timestamps succeeded -- so a failure
partway through a ~50-minute fetch only needs to retry the timestamps that
actually failed (re-reading the rest from the staging cache, not
re-fetching from GCS), rather than wasting the whole run.

Usage:
    python scripts/build_ifs_ensemble_store.py [--out PATH]
        [--baseline-store PATH] [--lead-hours H [H ...]]
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

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_baseline_store import (  # noqa: E402
    DEFAULT_LEAD_HOURS,
    GCS_ANON,
    INDIA_LAT_SLICE,
    INDIA_LON_SLICE_0_360,
    _clear_encoding,
    _lat_slice_for,
)

from weavr.grid import SourceTooCoarseError, regrid_to_common  # noqa: E402

IFS_ENS_ZARR_PATH = "gs://weatherbench2/datasets/ifs_ens/2018-2022-1440x721.zarr"
PRECIP_VARIABLE = "total_precipitation_24hr"
N_MEMBERS = 50


def _load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, default=str))


def real_baseline_timestamps(baseline_store: str, group: str = "ifs_ens_mean") -> np.ndarray:
    """The exact 18 weekly JJAS-2020 timestamps the baseline store already
    uses for this source family, read directly from the real store -- not
    re-derived from a cadence formula, so this new store's samples are
    guaranteed to align 1:1 with `data/baseline_2020_jjas.zarr` by
    construction rather than by coincidence.
    """
    ds = xr.open_zarr(baseline_store, group=group, consolidated=True)
    return ds["time"].values


def fetch_one_timestamp(
    ds: xr.Dataset, timestamp: np.datetime64, lead_hours: list[int]
) -> xr.Dataset:
    """Fetch every requested lead's full 50-member, full-global-grid slab
    for one timestamp in a single vectorized `.sel()`/`.load()` call --
    small enough (5 leads here) to complete without the stall a much
    larger single call caused in scripts/build_lagged_ensemble_store.py.
    """
    lead_indexer = xr.DataArray(lead_hours, dims="prediction_timedelta")
    return ds.sel(time=timestamp, prediction_timedelta=lead_indexer).load()


def staging_path(staging_dir: Path, ts: np.datetime64) -> Path:
    ts_key = str(np.datetime_as_string(ts, unit="s")).replace(":", "")
    return staging_dir / f"{ts_key}.nc"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_out = os.environ.get("WEAVR_IFS_ENSEMBLE_STORE_PATH", "data/ifs_ens_2020_jjas.zarr")
    parser.add_argument("--out", default=default_out)
    parser.add_argument("--baseline-store", default="data/baseline_2020_jjas.zarr")
    parser.add_argument("--lead-hours", type=int, nargs="+", default=DEFAULT_LEAD_HOURS)
    parser.add_argument(
        "--force", action="store_true", help="re-fetch timestamps already marked ok"
    )
    args = parser.parse_args()

    store_path = Path(args.out)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = store_path.with_suffix(".manifest.json")
    manifest = _load_manifest(manifest_path)
    staging_dir = store_path.parent / f"{store_path.name}.staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building real IFS 50-member ensemble store at {store_path}")
    print(f"Source: {IFS_ENS_ZARR_PATH}")

    ds = xr.open_zarr(IFS_ENS_ZARR_PATH, storage_options=GCS_ANON, consolidated=True)
    ds = ds[[PRECIP_VARIABLE]]
    timestamps = real_baseline_timestamps(args.baseline_store)
    lat_slice = _lat_slice_for(ds, "latitude", INDIA_LAT_SLICE.start, INDIA_LAT_SLICE.stop)
    ds = ds.sel(latitude=lat_slice, longitude=INDIA_LON_SLICE_0_360)
    ds = ds.rename({"number": "member"})

    print(f"Window: {len(timestamps)} timestamps (matching {args.baseline_store}'s own), "
          f"lead hours {args.lead_hours}")

    # WeatherBench 2's closest thing to a model-version identifier, recorded
    # once at the manifest's top level (this script has one source, unlike
    # build_baseline_store.py's per-source-group manifest) -- see
    # build_baseline_store.py's own build_forecast_group for why (Phase 6's
    # model-version-metadata check).
    manifest["_source_archive_path"] = IFS_ENS_ZARR_PATH
    _save_manifest(manifest_path, manifest)

    fetch_start = time.time()
    per_timestamp_seconds: dict[str, float] = manifest.get("_per_timestamp_seconds", {})

    for i, ts in enumerate(timestamps):
        ts_key = str(np.datetime_as_string(ts, unit="s"))
        staging_file = staging_path(staging_dir, ts)
        already_ok = manifest.get(ts_key, {}).get("status") == "ok" and staging_file.exists()
        if not args.force and already_ok:
            print(f"[skip] {ts_key}: already fetched and cached in staging")
            continue

        print(f"[fetch] timestamp {i + 1}/{len(timestamps)} ({ts_key}) ...")
        t0 = time.time()
        try:
            fetched = fetch_one_timestamp(ds, ts, args.lead_hours)
            elapsed = time.time() - t0
            fetched.assign_coords(time=ts).to_netcdf(staging_file)
            per_timestamp_seconds[ts_key] = elapsed
            manifest[ts_key] = {
                "status": "ok",
                "elapsed_seconds": elapsed,
                "fetched_at": datetime.utcnow().isoformat() + "Z",
            }
            print(f"[ok]   {ts_key}: {elapsed:.1f}s")
        except Exception as exc:  # one timestamp failing must not abort the rest
            manifest[ts_key] = {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "fetched_at": datetime.utcnow().isoformat() + "Z",
            }
            print(f"[FAIL] {ts_key}: {exc}", file=sys.stderr)
        finally:
            manifest["_per_timestamp_seconds"] = per_timestamp_seconds
            _save_manifest(manifest_path, manifest)

    total_elapsed = time.time() - fetch_start

    n_failed = sum(
        1 for k, v in manifest.items() if not k.startswith("_") and v.get("status") != "ok"
    )
    if n_failed:
        print(
            f"\n{n_failed} timestamp(s) failed -- not writing the final store. "
            "Re-run this script (successfully-fetched timestamps are cached in "
            f"{staging_dir} and will be skipped) to retry only the failures."
        )
        return 1

    # Every timestamp is now cached in staging (this run's fetches plus any
    # from a prior partial run) -- combine all of them, not just this run's
    # subset, so a resumed run doesn't silently drop earlier successes.
    fetched_datasets = [xr.open_dataset(staging_path(staging_dir, ts)) for ts in timestamps]
    combined = xr.concat(fetched_datasets, dim="time")
    combined = _clear_encoding(combined)

    try:
        combined = regrid_to_common(combined)
    except SourceTooCoarseError:
        raise

    if store_path.exists():
        import shutil

        shutil.rmtree(store_path)
    combined.to_zarr(store_path, mode="w")

    real_total_bytes = sum(d[PRECIP_VARIABLE].nbytes for d in fetched_datasets)
    summary = {
        "n_timestamps": len(fetched_datasets),
        "n_lead_hours": len(args.lead_hours),
        "n_members": N_MEMBERS,
        "total_elapsed_seconds": total_elapsed,
        "total_bytes_fetched": real_total_bytes,
        "mean_seconds_per_timestamp": total_elapsed / max(len(timestamps), 1),
    }
    manifest["_summary"] = summary
    _save_manifest(manifest_path, manifest)

    print("\n--- Summary ---")
    print(
        f"Fetched {len(fetched_datasets)} timestamps this run in {total_elapsed / 60:.1f} min "
        f"({real_total_bytes / 1e9:.2f} GB from staging) -- "
        f"{summary['mean_seconds_per_timestamp']:.1f}s/timestamp average "
        f"(vs. the ~34s/single-chunk x 5 leads = ~170s/timestamp serial estimate; "
        f"per-timestamp batching lets dask fetch a timestamp's 5 leads concurrently)."
    )
    print(f"Wrote {store_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
