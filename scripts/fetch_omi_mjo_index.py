#!/usr/bin/env python3
"""Fetch the real OMI (OLR-based MJO Index) and store 2020 JJAS's daily phase.

Kiladis, Dias, Straub, Wheeler, Tulich, Kikuchi, Weickmann & Ventrice
(2014), "A Comparison of OLR and Circulation-Based Indices for Tracking
the MJO," Monthly Weather Review 142(5), 1697-1715 -- chosen over the more
commonly cited Wheeler & Hendon RMM index because this project's own
fetch attempt against RMM's host (the Australian Bureau of Meteorology,
bom.gov.au) returned HTTP 403, while OMI's file, hosted directly on a
NOAA government domain (psl.noaa.gov), was fetched successfully. See
docs/phase5-regime-covariate-scope.md.

The real file's actual column layout was checked directly (not assumed
from NOAA's own separate, generic format-description page, which
describes a 7-column year/month/day/hour/PC1/PC2/amplitude layout that
does NOT match what this file itself contains): `omi.1x.txt` is
`year month day PC1 PC2 amplitude` -- 6 columns, no hour column, one row
per day, 1991 through the near-present (confirmed to include all of 2020,
a leap year, at exactly 366 real rows).

MJO phase (1-8) is not a stored column -- it is the standard, well-known
derived quantity from the two principal components:
`phase = floor(atan2(PC2, PC1) in degrees, shifted to [0, 360)) / 45 + 1`,
matching the convention Wheeler & Hendon (2004)/Kiladis et al. (2014) both
use for RMM/OMI alike (8 phases of 45 degrees each around the PC1-PC2
plane).

Usage:
    python scripts/fetch_omi_mjo_index.py [--out PATH] [--year Y]

    Restricts the stored output to one JJAS season (default: 2020, this
    project's only real data year) -- the full multi-decade series is
    fetched (cheap: ~13,000 rows of plain text) but only the relevant
    season is written out, matching every other store in this project
    keeping only what it actually needs (docs/baseline-store.md).
"""

from __future__ import annotations

import argparse
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

OMI_URL = "https://psl.noaa.gov/mjo/mjoindex/omi.1x.txt"
OMI_COLUMNS = ["year", "month", "day", "pc1", "pc2", "amplitude"]


def fetch_omi_raw(url: str = OMI_URL, timeout_seconds: float = 30.0) -> pd.DataFrame:
    """Downloads the real OMI file and parses it into a DataFrame, checking
    the column count against what the file actually contains rather than
    assuming a fixed layout from NOAA's own separate format-description
    page (which does not match, per this module's docstring).
    """
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()

    frame = pd.read_csv(StringIO(response.text), sep=r"\s+", header=None)
    if frame.shape[1] != len(OMI_COLUMNS):
        raise ValueError(
            f"Expected {len(OMI_COLUMNS)} columns in the real OMI file, got "
            f"{frame.shape[1]} -- NOAA may have changed the file format; "
            "re-check column meaning before trusting the parse."
        )
    frame.columns = OMI_COLUMNS
    return frame


def compute_mjo_phase(pc1: np.ndarray, pc2: np.ndarray) -> np.ndarray:
    """The standard 8-phase MJO convention (Wheeler & Hendon 2004; Kiladis
    et al. 2014): phase 1-8, each a 45-degree sector of the (PC1, PC2)
    plane, going counter-clockwise from due east.
    """
    angle_deg = np.degrees(np.arctan2(pc2, pc1))
    angle_deg = np.mod(angle_deg, 360.0)
    return np.floor(angle_deg / 45.0).astype(int) + 1


def build_omi_jjas_dataset(frame: pd.DataFrame, year: int) -> xr.Dataset:
    """Restricts the real OMI record to one JJAS season and computes phase."""
    season = frame[
        (frame["year"] == year) & (frame["month"] >= 6) & (frame["month"] <= 9)
    ].copy()
    if season.empty:
        raise ValueError(f"No OMI rows found for {year} JJAS -- check real file coverage.")

    times = pd.to_datetime(season[["year", "month", "day"]])
    phase = compute_mjo_phase(season["pc1"].to_numpy(), season["pc2"].to_numpy())

    return xr.Dataset(
        {
            "phase": (("time",), phase),
            "amplitude": (("time",), season["amplitude"].to_numpy()),
            "pc1": (("time",), season["pc1"].to_numpy()),
            "pc2": (("time",), season["pc2"].to_numpy()),
        },
        coords={"time": times.values},
    ).sortby("time")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/mjo_phase_2020_jjas.zarr")
    parser.add_argument("--year", type=int, default=2020)
    args = parser.parse_args()

    print(f"Fetching real OMI index from {OMI_URL}")
    frame = fetch_omi_raw()
    print(f"Fetched {len(frame)} real daily rows ({frame['year'].min()}-{frame['year'].max()})")

    dataset = build_omi_jjas_dataset(frame, args.year)
    print(f"{args.year} JJAS: {dataset.sizes['time']} real days")
    print(f"Phase counts:\n{pd.Series(dataset['phase'].values).value_counts().sort_index()}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_zarr(out_path, mode="w")
    print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
