#!/usr/bin/env python3
"""Gate: refuse to start follow-on work unless Step 07's outputs are actually sound.

Step 07's own history is a list of ways a run can look successful and not be:
a fold silently overwritten (244 -> 122 rows), a season scoring `nan` because
its store was missing, a BMA CRPS of 1540 where 4.9 belongs, a CSV whose
header does not match its rows. Each of those passed a "did it write a file"
check. This checks the numbers instead.

Exits 0 only if every check passes. Any failure prints the specific reason.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

RESULTS = Path(
    os.environ.get("WEAVR_STEP07_RESULTS_DIR", ".step07_v2")
).resolve()
PER_DAY = RESULTS / "per_day"
LEADS = [24, 48, 72, 96, 120]

#: Init times per season on the daily evidence base. Overridable so the gate
#: can be exercised against the miniature stores in
#: `tests/test_runners_end_to_end.py`, which use 8. The real value is asserted
#: directly by a test so it cannot drift silently.
SEASON_DAYS = int(os.environ.get("WEAVR_GATE_SEASON_DAYS", "122"))

#: Any CRPS/RMSE above this is a numerical failure, not weather. The
#: historical lead-72 BMA outlier was ~1540 against a true value near 4.9.
ABSURD = 200.0

failures: list[str] = []
notes: list[str] = []


def expected_rows(lead_hours: int) -> int:
    """Per-day rows across both folds at this lead.

    A longer lead pushes the last init times' valid times past the end of the
    observed window, so each extra lead day costs one day per season. Verified
    against the tier0/tier1/single-source/phase2 runs: 244/242/240/238/236.
    """
    lead_days = lead_hours // 24
    return 2 * (SEASON_DAYS - (lead_days - 1))


def check_domain_csv(name: str) -> pd.DataFrame | None:
    path = RESULTS / name
    if not path.exists():
        failures.append(f"missing domain CSV: {name}")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        failures.append(f"unreadable domain CSV {name}: {type(exc).__name__}: {exc}")
        return None

    if "fold" not in df.columns:
        failures.append(f"{name}: no 'fold' column")
        return None
    folds = set(df["fold"].astype(str))
    if not {"2018", "2020"} <= folds:
        failures.append(
            f"{name}: not two-season -- folds present {sorted(folds)}; "
            "a single fold here means the other season silently failed"
        )
    if "lead_hours" in df.columns:
        leads = set(df["lead_hours"].astype(int))
        missing = sorted(set(LEADS) - leads)
        if missing:
            failures.append(f"{name}: missing leads {missing}")
    return df


def check_finite(df: pd.DataFrame | None, name: str, prefixes: tuple[str, ...]) -> None:
    if df is None:
        return
    checked = 0
    for col in df.columns:
        if not any(col.startswith(p) for p in prefixes):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        checked += 1
        if vals.isna().any():
            n = int(vals.isna().sum())
            failures.append(f"{name}: column '{col}' has {n} NaN of {len(vals)} rows")
        finite = vals.dropna().abs()
        if not finite.empty and float(finite.max()) > ABSURD:
            worst = float(finite.max())
            failures.append(
                f"{name}: column '{col}' max |value| = {worst:.1f} > {ABSURD}; "
                "looks like the known BMA/EMOS blow-up, not weather"
            )
    if checked == 0:
        notes.append(f"{name}: no {prefixes} columns found to check")


def check_per_day(pattern: str) -> None:
    files = sorted(PER_DAY.glob(pattern))
    if not files:
        failures.append(f"no per-day files matching {pattern}")
        return
    for lead in LEADS:
        suffix = f"__lead{lead}.csv"
        match = [f for f in files if f.name.endswith(suffix)]
        if not match:
            failures.append(f"{pattern}: no per-day file for lead {lead}")
            continue
        for f in match:
            try:
                df = pd.read_csv(f)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"unreadable per-day {f.name}: {type(exc).__name__}")
                continue
            folds = set(df["fold"].astype(str)) if "fold" in df.columns else set()
            if not {"2018", "2020"} <= folds:
                failures.append(
                    f"{f.name}: folds {sorted(folds)} -- PerDayScoreWriter "
                    "overwrite bug, or one fold failed"
                )
            exp = expected_rows(lead)
            got = len(df)
            if got > exp:
                failures.append(f"{f.name}: {got} rows > expected {exp}")
            elif exp - got > 8:
                failures.append(
                    f"{f.name}: {got} rows, expected ~{exp} "
                    f"({exp - got} short -- a season may be missing)"
                )


def main() -> int:
    if not RESULTS.exists():
        print(f"GATE FAIL: {RESULTS} does not exist")
        return 1

    print("=== Step 07 output gate ===")

    # 1. The slow tiers must exist and be two-season.
    t2 = check_domain_csv("tier2_hierarchical_baseline.csv")
    t3 = check_domain_csv("tier3_regime_conditioned_baseline.csv")
    check_finite(t2, "tier2", ("emos_", "bma_"))
    check_finite(t3, "tier3", ("regime_",))

    # 2. Per-day files must carry BOTH folds at every lead. This is the check
    #    that would have caught the original bug.
    check_per_day("tier2_emos_graphcast__lead*.csv")
    check_per_day("tier2_bma__lead*.csv")
    check_per_day("tier3_*.csv")

    # 3. Sanity: a completed run leaves no stray staging or partial files.
    for stray in ("tier2_hierarchical_baseline.csv.tmp",):
        if (RESULTS / stray).exists():
            failures.append(f"stray partial file present: {stray}")

    for n in notes:
        print(f"  note:  {n}")
    if failures:
        print(f"  GATE FAIL ({len(failures)} problem(s)):")
        for f in failures:
            print(f"    - {f}")
        return 1

    print("  GATE PASS: tier2/tier3 are two-season, finite, plausible, and per-day complete.")
    print("  Safe to start follow-on work.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
