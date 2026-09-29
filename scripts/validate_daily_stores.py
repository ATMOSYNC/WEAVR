#!/usr/bin/env python3
"""Validate the 2020 daily-cadence stores against what they are supposed to be.

Step 05 rebuilds the 2020 stores at daily init cadence. A store that looks
plausible but is quietly short of init times, or missing lagged members at
short leads, would not crash anything -- it would just make every downstream
result wrong in a way nobody notices. So each expectation is checked and
**failed on**, not printed for a human to eyeball.

What is checked, and why each one:

- **122 init times, 2020-06-01 to 2020-09-30, no gaps.** The whole point of
  the step is 18 weekly -> 122 daily samples. A partial fetch is the most
  likely failure and the easiest to miss.
- **Leads [24, 48, 72, 96, 120].** What every tier script asks for.
- **No all-NaN variable, and no all-NaN sample.** A group can open fine and
  contain nothing.
- **Lagged member counts per lead**, against the pattern the weekly store
  documented (docs/baseline-store.md): `2m_temperature` gets 6/8/9/9/9 and
  `total_precipitation_24hr` gets 5/7/9/9/9 at leads 24/48/72/96/120. Short
  leads legitimately have fewer members -- a lagged member needs a *shorter*
  source lead to reach the same valid time, and accumulated precipitation is
  undefined below 24 h. Checking this is what distinguishes "the daily build
  reproduced the documented structure" from "the daily build produced
  something".
- **IFS-ENS: 50 members.**

Usage:
    python scripts/validate_daily_stores.py [--baseline PATH] [--lagged PATH]
        [--ifs-ens PATH] [--expected-inits N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

EXPECTED_INITS = 122
EXPECTED_START = "2020-06-01"
EXPECTED_END = "2020-09-30"
EXPECTED_LEADS = (24, 48, 72, 96, 120)

BASELINE_GROUPS = ("graphcast", "pangu", "hres", "ifs_ens_mean", "imd_observed")
LAGGED_GROUPS = ("graphcast", "pangu")
IFS_ENS_MEMBERS = 50

# docs/baseline-store.md's documented weekly-store structure, which the daily
# build must reproduce. A lagged member needs a shorter source lead to reach
# the same valid time, so short leads lose members; precipitation loses one
# more because a 24 h accumulation is undefined below 24 h.
EXPECTED_MEMBERS_BY_LEAD = {
    "2m_temperature": {24: 6, 48: 8, 72: 9, 96: 9, 120: 9},
    "total_precipitation_24hr": {24: 5, 48: 7, 72: 9, 96: 9, 120: 9},
}


class Report:
    """Collects failures so one run reports every problem, not just the first."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, condition: bool, message: str) -> bool:
        if condition:
            print(f"    ok   {message}")
        else:
            print(f"    FAIL {message}")
            self.failures.append(message)
        return condition

    def note(self, message: str) -> None:
        print(f"    --   {message}")


def check_init_times(times: np.ndarray, report: Report, label: str, expected: int) -> None:
    """Init count, span and gap-freeness in one place, for every store."""
    index = pd.DatetimeIndex(times)
    report.check(len(index) == expected, f"{label}: {len(index)} init times (expected {expected})")
    if len(index) == 0:
        return
    report.check(
        str(index.min().date()) == EXPECTED_START and str(index.max().date()) == EXPECTED_END,
        f"{label}: spans {index.min().date()}..{index.max().date()} "
        f"(expected {EXPECTED_START}..{EXPECTED_END})",
    )
    gaps = (pd.Series(index).diff().dt.days.dropna() != 1).sum()
    report.check(gaps == 0, f"{label}: {gaps} gaps in the daily sequence (expected 0)")


def check_not_all_nan(data: xr.Dataset, report: Report, label: str) -> None:
    """Every variable has some real data, and no single sample is entirely NaN.

    Uses lazy reductions rather than `.values`: the IFS-ENS store is about
    2 GB, and materialising it to count NaNs is both slow and needless.
    """
    for name, array in data.data_vars.items():
        finite_fraction = float((~array.isnull()).mean())
        report.check(finite_fraction > 0.0, f"{label}.{name}: {finite_fraction:.1%} finite")

        sample_dim = next((d for d in ("time", "nominal_time") if d in array.dims), None)
        if sample_dim is not None:
            other_dims = [d for d in array.dims if d != sample_dim]
            per_sample_finite = (~array.isnull()).any(dim=other_dims)
            n_empty = int((~per_sample_finite).sum())
            report.check(n_empty == 0, f"{label}.{name}: {n_empty} all-NaN samples (expected 0)")


def count_members_by_lead(data: xr.DataArray) -> dict[int, int]:
    """Members that carry real data at each lead, minimised over nominal times.

    The *minimum* across nominal times, not a sample of five: a structural
    member count is only meaningful if it holds for every init. Sampling
    would miss exactly the partial-fetch failure this validator exists for.
    """
    present = (~data.isnull()).any(dim=[d for d in data.dims if d not in
                                        ("lead_hours", "member_offset_hours", "nominal_time")])
    per_lead: dict[int, int] = {}
    for lead in data["lead_hours"].values:
        at_lead = present.sel(lead_hours=lead)
        members_per_nominal = at_lead.sum(dim="member_offset_hours")
        per_lead[int(lead)] = int(members_per_nominal.min())
    return per_lead


def validate_baseline(path: str, report: Report, expected_inits: int) -> None:
    print(f"\n=== baseline store: {path}")
    if not Path(path).exists():
        report.check(False, f"{path} does not exist")
        return
    for group in BASELINE_GROUPS:
        try:
            data = xr.open_zarr(path, group=group, consolidated=True)
        except Exception as exc:  # noqa: BLE001 - report, don't abort the whole run
            report.check(False, f"{group}: cannot open ({type(exc).__name__}: {exc})")
            continue
        print(f"  [{group}] {dict(data.sizes)} vars={list(data.data_vars)}")
        check_init_times(data["time"].values, report, group, expected_inits)
        if "prediction_timedelta" in data.dims:
            leads = tuple(int(x) for x in data["prediction_timedelta"].values)
            report.check(leads == EXPECTED_LEADS, f"{group}: leads {leads}")
        check_not_all_nan(data, report, group)


def validate_lagged(path: str, report: Report, expected_inits: int) -> None:
    print(f"\n=== lagged-ensemble store: {path}")
    if not Path(path).exists():
        report.check(False, f"{path} does not exist")
        return
    for group in LAGGED_GROUPS:
        try:
            data = xr.open_zarr(path, group=group, consolidated=True)
        except Exception as exc:  # noqa: BLE001
            report.check(False, f"{group}: cannot open ({type(exc).__name__}: {exc})")
            continue
        print(f"  [{group}] {dict(data.sizes)} vars={list(data.data_vars)}")
        check_init_times(data["nominal_time"].values, report, group, expected_inits)
        check_not_all_nan(data, report, group)

        for name, array in data.data_vars.items():
            expected = EXPECTED_MEMBERS_BY_LEAD.get(name)
            if expected is None:
                report.note(f"{group}.{name}: no documented member pattern, not checked")
                continue
            actual = count_members_by_lead(array)
            report.check(
                actual == expected,
                f"{group}.{name}: members by lead {actual} (documented {expected})",
            )


def validate_ifs_ens(path: str, report: Report, expected_inits: int, required: bool) -> None:
    print(f"\n=== IFS-ENS store: {path}")
    if not Path(path).exists():
        if not required:
            # Deliberately not built -- see docs/baseline-store.md. Reporting
            # this as a failure would train readers to ignore the validator's
            # output, which is worse than not checking it at all.
            report.note(f"{path} not built (--no-require-ifs-ens); skipping")
            return
        report.check(False, f"{path} does not exist")
        return
    try:
        data = xr.open_zarr(path, consolidated=True)
    except Exception as exc:  # noqa: BLE001
        report.check(False, f"cannot open ({type(exc).__name__}: {exc})")
        return
    print(f"  {dict(data.sizes)} vars={list(data.data_vars)}")
    check_init_times(data["time"].values, report, "ifs_ens", expected_inits)
    report.check(
        data.sizes.get("member") == IFS_ENS_MEMBERS,
        f"ifs_ens: {data.sizes.get('member')} members (expected {IFS_ENS_MEMBERS})",
    )
    leads = tuple(int(x) for x in data["prediction_timedelta"].values)
    report.check(leads == EXPECTED_LEADS, f"ifs_ens: leads {leads}")
    check_not_all_nan(data, report, "ifs_ens")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="data/baseline_2020_jjas_daily.zarr")
    parser.add_argument("--lagged", default="data/lagged_ensemble_inputs_2020_jjas_daily.zarr")
    parser.add_argument("--ifs-ens", default="data/ifs_ens_2020_jjas_daily.zarr")
    parser.add_argument("--expected-inits", type=int, default=EXPECTED_INITS)
    parser.add_argument(
        "--no-require-ifs-ens",
        dest="require_ifs_ens",
        action="store_false",
        help=(
            "Treat a missing daily IFS-ENS store as a documented skip rather than a "
            "failure. The 50-member daily pull was deliberately not run (126 GB); see "
            "docs/baseline-store.md."
        ),
    )
    args = parser.parse_args()

    report = Report()
    validate_baseline(args.baseline, report, args.expected_inits)
    validate_lagged(args.lagged, report, args.expected_inits)
    validate_ifs_ens(args.ifs_ens, report, args.expected_inits, args.require_ifs_ens)

    print()
    if report.failures:
        print(f"VALIDATION FAILED: {len(report.failures)} check(s) did not pass:")
        for failure in report.failures:
            print(f"  - {failure}")
        return 1
    print("VALIDATION PASSED: every check above holds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
