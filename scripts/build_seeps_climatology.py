#!/usr/bin/env python3
"""Build a multi-year IMD-only JJAS rainfall archive for a SEEPS climatology.

SEEPS needs a per-gridpoint local climatology (a distribution of historical
daily rainfall at that point) to define its dry/light/heavy terciles. One
season (Phase 0's `data/baseline_2020_jjas.zarr`) is far too thin for that.
IMD-only data has none of WeatherBench 2's per-chunk GCS latency (~22s for a
*full year*, confirmed live, vs. ~2-3s *per chunk* for a single forecast
source/lead/init combination) — see docs/baseline-store.md and
docs/phase-1-data-requirements.md — so pulling many extra years is cheap and
needs no sampling-density tradeoff, unlike the forecast sources.

Only JJAS days are kept per year: every forecast this climatology will ever
score is JJAS (see docs/phase-plan.md), so there's no reason to pay for or
store the other eight months.

This is a separate script from build_baseline_store.py, not an extra source
folded into it: the shape of what's being built is different (many years of
one IMD variable, no forecast pairing, no lead-time dimension) and forcing
it into the same script/store layout would obscure that difference rather
than clarify it.

Each year is written to its own zarr group (y<year>), the same pattern
build_baseline_store.py uses per-source -- not concatenated along a shared
`time` axis via `append_dim`. That was tried first and found broken by
testing: this script's idempotent retry-on-failure design (a year that
fails on one run gets retried, in order, on the next) means a year can be
appended well after later years already succeeded, landing it out of
chronological order in the underlying store -- confirmed live when 2009 and
2018 failed on transient network timeouts, got retried after 2019/2020 had
already succeeded, and landed at the *end* of the time axis instead of
where they belong. `load_climatology()` below opens every year-group and
concatenates+sorts them into one chronological array, so this ordering
detail never leaks to callers.

Usage:
    python scripts/build_seeps_climatology.py [--out PATH] [--start-year Y]
                                               [--end-year Y]

    WEAVR_SEEPS_CLIMATOLOGY_PATH can also set the output path.

Idempotent: writes a JSON manifest next to the store recording which years
succeeded; a year already marked "ok" is skipped on re-run unless --force
is passed.
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

import xarray as xr

from weavr.data.imd_gridded import fetch_year
from weavr.grid import regrid_to_common

# 15 years (2006-2020 inclusive): enough for a meaningful per-gridpoint
# tercile climatology without an unbounded pull -- IMD's own gridded product
# is well-established over this window. Extend if a longer baseline turns
# out to matter once weavr.verify.seeps is actually built and validated.
DEFAULT_END_YEAR = 2020
DEFAULT_START_YEAR = 2006

JJAS_START_MONTH_DAY = "06-01"
JJAS_END_MONTH_DAY = "09-30"


def _clear_encoding(ds: xr.Dataset) -> xr.Dataset:
    """Strip zarr-v2-era encoding copied from imdlib's own xarray output.

    Same issue as build_baseline_store.py's `_clear_encoding`: writing a
    source's original `.encoding` back out via zarr-python v3 fails on the
    codec spec. Duplicated here rather than imported -- this is a private,
    four-line helper in a sibling script, not shared library code.
    """
    for name in list(ds.data_vars) + list(ds.coords):
        ds[name].encoding = {}
    return ds


def build_year(year: int) -> tuple[xr.Dataset, dict]:
    ds = fetch_year(year, var_type="rain")
    ds = ds.sel(time=slice(f"{year}-{JJAS_START_MONTH_DAY}", f"{year}-{JJAS_END_MONTH_DAY}"))
    ds = ds.rename({"lat": "latitude", "lon": "longitude"})
    ds = _clear_encoding(ds)
    ds = regrid_to_common(ds)

    info = {
        "status": "ok",
        "variables": list(ds.data_vars),
        "n_days": int(ds.sizes.get("time", 0)),
    }
    return ds, info


def load_climatology(path: str | Path) -> xr.Dataset:
    """Open every year-group in the archive and return one chronological Dataset.

    Years are stored one-per-group (see the module docstring for why: an
    append-along-time layout landed retried years out of order). This
    concatenates them and sorts by time, so callers never need to know
    about the per-year group layout or the order groups were written in.
    """
    import zarr

    root = zarr.open_group(str(path), mode="r")
    year_groups = sorted(name for name in root.keys() if name.startswith("y"))
    datasets = [xr.open_zarr(path, group=name, consolidated=True) for name in year_groups]
    combined = xr.concat(datasets, dim="time")
    return combined.sortby("time")


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
            "WEAVR_SEEPS_CLIMATOLOGY_PATH", "data/imd_seeps_climatology_jjas.zarr"
        ),
    )
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--force", action="store_true", help="re-pull years already marked ok")
    args = parser.parse_args()

    store_path = Path(args.out)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = store_path.with_suffix(".manifest.json")
    manifest = _load_manifest(manifest_path)

    years = list(range(args.start_year, args.end_year + 1))
    print(f"Building SEEPS climatology archive at {store_path}")
    print(f"Years: {years[0]}-{years[-1]} (JJAS only)")

    for year in years:
        key = str(year)
        if not args.force and manifest.get(key, {}).get("status") == "ok":
            print(f"[skip] {year}: already marked ok in manifest")
            continue

        print(f"[fetch] {year} ...")
        try:
            ds, info = build_year(year)
            mode: Literal["w", "a"] = "w" if not store_path.exists() else "a"
            ds.to_zarr(store_path, group=f"y{year}", mode=mode)
            info["fetched_at"] = datetime.utcnow().isoformat() + "Z"
            manifest[key] = info
            print(f"[ok]   {year}: {info}")
        except Exception as exc:  # a single year failing must not abort the rest
            manifest[key] = {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "fetched_at": datetime.utcnow().isoformat() + "Z",
            }
            print(f"[FAIL] {year}: {exc}", file=sys.stderr)
        finally:
            _save_manifest(manifest_path, manifest)

    print("\n--- Summary ---")
    n_ok = sum(1 for i in manifest.values() if i.get("status") == "ok")
    n_failed = len(manifest) - n_ok
    print(f"Years ok: {n_ok}, failed: {n_failed}")
    for year, info in sorted(manifest.items()):
        status = info.get("status")
        if status == "ok":
            print(f"  {year}: OK, n_days={info.get('n_days')}")
        else:
            print(f"  {year}: FAILED - {info.get('error')}")

    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
