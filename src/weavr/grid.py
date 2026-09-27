"""The one common grid and time convention every data source gets aligned to.

Locked decision (see docs/grid-and-time-convention.md for the full
rationale): 0.25 degree lat/lon over India, and IMD's 03 UTC -> 03 UTC
(next day) 24-hour accumulation window — not UTC midnight-to-midnight.

Every ingestion script must call `regrid_to_common` / `resample_to_imd_day`
rather than reimplementing regridding or resampling inline, so alignment
bugs can't silently diverge between sources.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

# --- Common grid ------------------------------------------------------------

# Resolution, in degrees. Matches IMD gridded rainfall's native resolution
# and the resolution WeatherBench 2's regridded products are served at.
GRID_RESOLUTION_DEG = 0.25

# Bounds confirmed against a real IMD gridded-rainfall pull (imdlib,
# var_type="rain", 2023): lat 6.5-38.5, lon 66.5-100.0 at 0.25 deg spacing.
# This is IMD's actual native grid, not the rough 6-38N/68-98E estimate a
# first pass at this decision might reach for — use these exact bounds so
# regridding onto IMD's own rainfall grid needs no further interpolation at
# the domain edges.
GRID_LAT_MIN = 6.5
GRID_LAT_MAX = 38.5
GRID_LON_MIN = 66.5
GRID_LON_MAX = 100.0

COMMON_LAT = np.round(
    np.arange(GRID_LAT_MIN, GRID_LAT_MAX + GRID_RESOLUTION_DEG / 2, GRID_RESOLUTION_DEG), 2
)
COMMON_LON = np.round(
    np.arange(GRID_LON_MIN, GRID_LON_MAX + GRID_RESOLUTION_DEG / 2, GRID_RESOLUTION_DEG), 2
)


class SourceTooCoarseError(ValueError):
    """Raised when a source dataset's native resolution is coarser than the common grid.

    Regridding a coarser source onto a finer grid is upsampling: it produces
    extra grid points that carry no additional information, which can hide
    real skill differences behind an illusion of matching resolution. Fail
    loudly instead.
    """


def _native_resolution(coord: xr.DataArray) -> float:
    diffs = np.abs(np.diff(np.sort(coord.values)))
    return float(np.median(diffs))


def regrid_to_common(
    ds: xr.Dataset,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
) -> xr.Dataset:
    """Regrid a source dataset onto the common 0.25 deg India grid.

    Uses linear interpolation (xarray's built-in `interp`), which is
    sufficient for a common evaluation grid at matched-order resolution.
    Swap for a conservative-remapping tool (e.g. xesmf) if a source needs
    area-weighted regridding (e.g. from a much coarser native grid) rather
    than point interpolation.

    Raises SourceTooCoarseError if the source's native resolution is coarser
    than the common grid, rather than silently upsampling.
    """
    if lat_dim not in ds.coords or lon_dim not in ds.coords:
        raise ValueError(f"Dataset missing expected coords: {lat_dim!r}, {lon_dim!r}")

    native_lat_res = _native_resolution(ds[lat_dim])
    native_lon_res = _native_resolution(ds[lon_dim])
    if native_lat_res > GRID_RESOLUTION_DEG or native_lon_res > GRID_RESOLUTION_DEG:
        raise SourceTooCoarseError(
            f"Source resolution (lat={native_lat_res:.3f}, lon={native_lon_res:.3f} deg) "
            f"is coarser than the common grid ({GRID_RESOLUTION_DEG} deg). "
            "Upsampling would hide skill differences due to resolution, not model quality — "
            "regrid conservatively with an area-weighted tool instead, or exclude this source."
        )

    return ds.interp({lat_dim: COMMON_LAT, lon_dim: COMMON_LON}, method="linear")


# --- IMD day convention ------------------------------------------------------

# IMD's daily accumulation window: 03:00 UTC to 03:00 UTC the next day (this
# is 08:30 IST to 08:30 IST, IMD's own observation-day convention), not UTC
# midnight-to-midnight. A day labelled "2024-06-15" in IMD data covers
# 2024-06-15 03:00 UTC through 2024-06-16 03:00 UTC.
IMD_DAY_START_HOUR_UTC = 3


def resample_to_imd_day(ds: xr.Dataset, time_dim: str = "time") -> xr.Dataset:
    """Resample/accumulate a sub-daily or differently-aligned dataset onto IMD's day.

    Sums over each 24h window from 03:00 UTC to the next 03:00 UTC, and
    labels the resulting day with the date the window *starts* on (matching
    IMD's own convention of labelling a day by its 03 UTC start).

    Implementation note: `Dataset.resample(..., offset=...)` does not shift
    calendar-day ("1D") bin boundaries in the xarray/pandas versions this was
    tested against — the offset is silently ignored, which would produce
    midnight-aligned bins while claiming to be 03 UTC-aligned. Shift the time
    coordinate back by the offset before resampling instead, so the
    calendar-day boundaries fall exactly on the desired 03 UTC/day.
    """
    if time_dim not in ds.dims:
        raise ValueError(f"Dataset missing expected time dimension: {time_dim!r}")

    offset = pd.Timedelta(hours=IMD_DAY_START_HOUR_UTC)
    shifted = ds.assign_coords({time_dim: ds[time_dim] - offset})
    resampled = shifted.resample({time_dim: "1D"}).sum(skipna=True)

    # Bin labels are now midnight in shifted-time, i.e. exactly 03:00 UTC in
    # real time for the window's start; normalise to a plain date label
    # (matching IMD's own day-by-start-date convention) rather than exposing
    # the shifted timestamp.
    new_labels = pd.DatetimeIndex(resampled[time_dim].values).normalize()
    return resampled.assign_coords({time_dim: new_labels})
