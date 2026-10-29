#!/usr/bin/env python3
"""Pre-flight: prove every input store carries data BEFORE a multi-hour run.

Written after `data/ifs_ens_2020_jjas_daily.zarr` turned out to be 3.4 MB of
zeros -- correct shape, correct dates, correct dtype, no data -- and a
two-season Tier 2 run spent 2h15m of CPU scoring against it. Every structural
check passed, because zeros are not NaN and not the wrong shape.

This checks the thing that actually matters: the *values*. A store earns the
right to be read by an expensive run only by holding non-zero data, and this
costs seconds instead of hours.

Exits 0 only if every store is usable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

DATA = Path(os.environ.get("WEAVR_DATA_DIR", "data")).resolve()

#: Below this fraction of non-zero values a precipitation forecast is not
#: data. Real JJAS seasons measure 0.8-0.9. Kept far from both: the failure
#: being caught is 0.0, and a threshold near the observed value would reject
#: a legitimately dry grid.
MIN_NONZERO = 0.01

#: (path, group, variable, expect_nonzero)
#: `None` for expect_nonzero means "presence is enough" (e.g. the
#: climatology, whose values are normalised probabilities and can be 0).
CHECKS: list[tuple[str, str, str, bool]] = [
    ("baseline_2018_jjas_daily.zarr", "graphcast", "total_precipitation_24hr", True),
    ("baseline_2020_jjas_daily.zarr", "graphcast", "total_precipitation_24hr", True),
    ("baseline_2018_jjas_daily.zarr", "hres", "total_precipitation_24hr", True),
    ("baseline_2020_jjas_daily.zarr", "hres", "total_precipitation_24hr", True),
    ("baseline_2018_jjas_daily.zarr", "ifs_ens_mean", "total_precipitation_24hr", True),
    ("baseline_2020_jjas_daily.zarr", "ifs_ens_mean", "total_precipitation_24hr", True),
    ("baseline_2018_jjas_daily.zarr", "pangu", "2m_temperature", True),
    ("baseline_2020_jjas_daily.zarr", "pangu", "2m_temperature", True),
    ("baseline_2018_jjas_daily.zarr", "imd_observed", "rain", True),
    ("baseline_2020_jjas_daily.zarr", "imd_observed", "rain", True),
    (
        "lagged_ensemble_inputs_2018_jjas_daily.zarr",
        "graphcast",
        "total_precipitation_24hr",
        True,
    ),
    (
        "lagged_ensemble_inputs_2020_jjas_daily.zarr",
        "graphcast",
        "total_precipitation_24hr",
        True,
    ),
    ("lagged_ensemble_inputs_2018_jjas_daily.zarr", "pangu", "2m_temperature", True),
    ("lagged_ensemble_inputs_2020_jjas_daily.zarr", "pangu", "2m_temperature", True),
    ("ifs_ens_2018_jjas_daily.zarr", None, "total_precipitation_24hr", True),
    ("ifs_ens_2020_jjas_daily.zarr", None, "total_precipitation_24hr", True),
    ("imd_seeps_climatology_jjas.zarr", "y2015", "rain", True),
]

#: Forecast precipitation must be in METRES (WeatherBench 2 / ECMWF), because
#: every runner multiplies by `PRECIP_M_TO_MM` before scoring. Observations are
#: in MILLIMETRES (IMD) and are never converted.
#:
#: This check exists because a store in the wrong unit is invisible: the
#: multiply-by-1000 produces finite, plausible-looking numbers that are simply
#: 1000x too large, and Tier 2's `tier0` columns read ~4000 mm while every
#: other column looks fine. A daily accumulation in metres is O(1e-4 .. 5e-1);
#: in millimetres it is O(1 .. 400). The two ranges do not overlap.
METRES_RANGE = (1e-5, 0.5)
MILLIMETRES_RANGE = (0.1, 500.0)

#: Seasons must not silently collapse to one. A single year here is exactly
#: how a "two-season LOYO" run becomes a one-season run that still succeeds.
EXPECTED_TIMES = {"2018": 122, "2020": 122}

#: Variable names that carry the metres/millimetres unit contract.
PRECIP_VARS = {"total_precipitation_24hr"}


def main() -> int:
    print("=== pre-flight store integrity ===")
    problems: list[str] = []

    for store, group, var, expect_nonzero in CHECKS:
        path = DATA / store
        label = f"{store}{'/' + group if group else ''}:{var}"
        if not path.exists():
            problems.append(f"{label}: MISSING ({path})")
            continue
        try:
            ds = xr.open_zarr(path, group=group) if group else xr.open_zarr(path)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{label}: cannot open -- {type(exc).__name__}: {exc}")
            continue

        if var is None:
            # No specific variable required: presence of readable data is enough.
            var = list(ds.data_vars)[0] if ds.data_vars else None
            if var is None:
                problems.append(f"{store}: no data variables at all")
                continue
        if var not in ds.data_vars:
            problems.append(
                f"{label}: variable absent, found {list(ds.data_vars)} "
                "(this is expected for pangu precipitation, not for these)"
            )
            continue

        da = ds[var]
        # The lagged stores index on `nominal_time`; the rest on `time`.
        tdim = "time" if "time" in da.dims else (
            "nominal_time" if "nominal_time" in da.dims else None
        )
        if tdim is None:
            problems.append(
                f"{label}: no time dimension (dims {da.dims}) -- cannot confirm cadence"
            )
            continue
        n_time = da.sizes.get(tdim, 1)
        picks = np.unique(np.linspace(0, n_time - 1, min(4, n_time), dtype=int))
        finite = nonzero = 0
        for t in picks:
            block = da.isel({tdim: int(t)}).values
            m = np.isfinite(block)
            finite += int(m.sum())
            nonzero += int((block[m] > 0).sum())
        frac = nonzero / max(finite, 1)

        # Units, for precipitation from a forecast store only.
        if PRECIP_VARS & {var} and store.startswith(("baseline", "lagged", "ifs_ens")):
            unit_mean = float(np.nanmean(
                da.isel({da.dims[0]: 0}).values if da.ndim else da.values
            ))
            if not (METRES_RANGE[0] <= abs(unit_mean) <= METRES_RANGE[1]):
                problems.append(
                    f"{label}: mean |{unit_mean:.4g}| is outside the metres range "
                    f"{METRES_RANGE}. Every runner multiplies forecast "
                    "precipitation by 1000 (PRECIP_M_TO_MM), so a store written "
                    "in millimetres is silently scored 1000x too large and still "
                    "looks like a plausible number."
                )

        status = "ok"
        if expect_nonzero and frac < MIN_NONZERO:
            problems.append(
                f"{label}: only {frac:.3%} non-zero (need >= {MIN_NONZERO:.0%}) -- "
                "structurally valid but empty; scoring against this is meaningless"
            )
            status = "EMPTY"
        print(f"  [{status:>5}] {label:<62} n_time={n_time:<4} nonzero={frac:.1%}")

    # Both seasons must actually be present for the daily stores a LOYO run
    # folds over.
    for store in (
        "baseline_2018_jjas_daily.zarr",
        "baseline_2020_jjas_daily.zarr",
        "lagged_ensemble_inputs_2018_jjas_daily.zarr",
        "lagged_ensemble_inputs_2020_jjas_daily.zarr",
        "ifs_ens_2018_jjas_daily.zarr",
        "ifs_ens_2020_jjas_daily.zarr",
    ):
        kind = next(k for k in ("baseline", "lagged", "ifs_ens") if k in store)
        years = [y for y in EXPECTED_TIMES if store.startswith(f"{kind}_{y}")]
        if not years:
            continue
        for y in years:
            expected = EXPECTED_TIMES[y]
            probe_group = (
                "graphcast" if "baseline" in store
                else "graphcast" if "lagged" in store
                else None
            )
            try:
                ds = (
                    xr.open_zarr(DATA / store, group=probe_group)
                    if probe_group
                    else xr.open_zarr(DATA / store)
                )
                tdim = "time" if "time" in ds.coords else "nominal_time"
                n = ds[tdim].size if tdim in ds.coords else -1
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{store}: cannot read time axis -- {type(exc).__name__}: {exc}")
                continue
            if n != expected:
                problems.append(
                    f"{store}: {n} times, expected {expected} for the {y} season -- "
                    "a short store silently truncates the fold"
                )

    if problems:
        print(f"\n  PREFLIGHT FAIL ({len(problems)} problem(s)) -- do NOT start a long run:")
        for p in problems:
            print(f"    - {p}")
        return 1
    print("\n  PREFLIGHT PASS: every store is present, two-season, and carries data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
