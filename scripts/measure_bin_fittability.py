#!/usr/bin/env python3
"""Re-measure the per-bin fittability table on the v2 train folds (step 07 item 5).

The v1 table in `docs/phase4-data-and-combiner-scope.md` counted distinct
*contributing train days* per bin over a 14-day `seasonal_block_split` window
and concluded that `extremely_heavy` was never fittable. Step 07 asks for the
same table re-measured on the v2 base, at
`MIN_TRAIN_DAYS_PER_BIN = 5` (which this script does not change).

Rather than recount days here, this reads the *outcome* of the fits Tier 2
already performed on each training fold: `is_fallback` in the by-bin and
by-region breakdowns is the combiner's own record of whether it fit a bin for
real or substituted a climatological point mass. That is the quantity the
pre-registration is about, and it comes from the same fits that produced the
published scores, so the table cannot drift from them.

Two granularities are reported, because they answer different questions and
conflating them is how a false "everything fits" claim happens:

* **per bin** (`tier2_*_by_bin.csv`) -- EMOS-CSG fits one distribution per
  rain bin, pooled over regions.
* **per (bin, region)** (`tier2_*_by_region.csv`) -- BMA fits per bin *and*
  region, so each bin is asked to fit six times per fold and can fail in some
  regions while succeeding in others.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

BINS = ("dry", "light", "heavy", "very_heavy", "extremely_heavy")
LEADS = (24, 48, 72, 96, 120)


def _cells(grouped: pd.DataFrame, rain_bin: str) -> tuple[int, int] | None:
    if rain_bin not in set(grouped["bin"]):
        return None
    rows = grouped[grouped["bin"] == rain_bin]
    return int((rows["is_fallback"] == False).sum()), len(rows)  # noqa: E712


def table(by_bin: pd.DataFrame, combiner_prefix: str, granularity: str) -> list[dict]:
    """Fittability per (lead, bin) for one granularity and combiner family."""
    frame = by_bin
    if "combiner" in frame.columns:
        frame = frame[frame["combiner"].str.startswith(combiner_prefix)]
    frame = frame[frame["fold"].astype(str) != "pooled"]
    rows: list[dict] = []
    for lead in LEADS:
        at_lead = frame[frame["lead_hours"] == lead]
        if at_lead.empty:
            continue
        for rain_bin in BINS:
            counts = _cells(at_lead, rain_bin)
            if counts is None:
                continue
            fitted, total = counts
            rows.append(
                {
                    "granularity": granularity,
                    "combiner": combiner_prefix,
                    "lead_hours": lead,
                    "bin": rain_bin,
                    "fits": fitted,
                    "fits_total": total,
                    "fraction": fitted / total if total else float("nan"),
                }
            )
    return rows


def render(rows: list[dict]) -> None:
    for granularity, combiner in (
        ("per bin (EMOS-CSG, pooled over regions)", "emos_"),
        ("per (bin, region) (BMA)", "bma"),
    ):
        subset = [
            r for r in rows if r["granularity"] == granularity and r["combiner"] == combiner
        ]
        if not subset:
            continue
        print(f"\n### {granularity}")
        header = "  lead |" + "".join(f" {b:>18}" for b in BINS)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for lead in LEADS:
            cells = []
            for rain_bin in BINS:
                match = [
                    r for r in subset if r["lead_hours"] == lead and r["bin"] == rain_bin
                ]
                text = (
                    f"{match[0]['fits']}/{match[0]['fits_total']}" if match else "--"
                )
                cells.append(text.rjust(18))
            print(f"  {lead:>4}h |" + "".join(cells))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory holding tier2_hierarchical_baseline_by_{bin,region}.csv",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Default: <results-dir>/bin_fittability.csv",
    )
    args = parser.parse_args()

    root = Path(args.results_dir)
    by_bin_path = root / "tier2_hierarchical_baseline_by_bin.csv"
    by_region_path = root / "tier2_hierarchical_baseline_by_region.csv"
    for path in (by_bin_path, by_region_path):
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 1

    by_bin = pd.read_csv(by_bin_path)
    by_region = pd.read_csv(by_region_path)

    rows: list[dict] = []
    rows += table(by_bin, "emos_", "per bin (EMOS-CSG, pooled over regions)")
    # by_region carries no `combiner` column -- it *is* the BMA breakdown --
    # so label it here rather than filtering on a column that is not there.
    rows += table(by_region, "bma", "per (bin, region) (BMA)")

    print("Per-bin fittability on the v2 train folds")
    print("(fits / fits attempted; is_fallback=False means fit for real, not a")
    print(" climatological point-mass substitution. MIN_TRAIN_DAYS_PER_BIN = 5 unchanged.)")
    render(rows)

    out = Path(args.out_csv) if args.out_csv else root / "bin_fittability.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())