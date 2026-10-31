"""The one common grid and time convention every data source gets aligned to.

Locked decision (see docs/grid-and-time-convention.md for the full
rationale): 0.25 degree lat/lon over India, and IMD's 03 UTC -> 03 UTC
(next day) 24-hour accumulation window, labelled by the date it ends on —
not UTC midnight-to-midnight.

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

# IMD's daily accumulation window runs from 03:00 UTC to the next 03:00 UTC
# (08:30 IST to 08:30 IST), not UTC midnight-to-midnight. IMD labels a day by
# the date the window **ends** on: the total recorded "from 0830 hrs of
# 21-10-2026 to 0830 hrs of 22-10-2026" is the value for 22-10-2026
# (https://mausam.imd.gov.in/responsive/rainfallinformation_msd.php). So a day
# labelled "2024-06-15" covers 2024-06-14 03:00 UTC through 2024-06-15 03:00
# UTC. This was previously read the other way round (labelled by the start),
# which paired every forecast with the IMD day before the one it covers -- see
# docs/grid-and-time-convention.md and issue #106.
IMD_DAY_START_HOUR_UTC = 3

_ONE_NANOSECOND = pd.Timedelta(1, unit="ns")


def imd_day_containing(instant):
    """The IMD day label (by end date) of the window that contains `instant`.

    Accepts a `pd.Timestamp` or a `pd.DatetimeIndex`. Windows are half-open,
    `[03:00 UTC, next 03:00 UTC)`, so an instant at exactly 03:00 UTC starts
    the next day's window: 2024-06-15 03:00 UTC is in day 2024-06-16, and
    2024-06-15 02:59 UTC is in day 2024-06-15.
    """
    shifted = pd.to_datetime(instant) - pd.Timedelta(hours=IMD_DAY_START_HOUR_UTC)
    return shifted.normalize() + pd.Timedelta(days=1)


def imd_day_for(init_time, lead_hours: int):
    """The IMD day a forecast's accumulation window `(init, init + lead]` validates on.

    This is the one place a forecast is paired with an observed day; every
    runner calls it so the convention cannot drift between scripts.

    `total_precipitation_24hr` valid at `init + lead` is the accumulation over
    the 24 h *ending* at that instant. The IMD day that matches it is the one
    whose window ends at or just after that instant, so the last nanosecond of
    the forecast window is what is placed: a 24 h forecast initialised at
    00 UTC on day `t` covers 00 UTC `t` to 00 UTC `t+1`, which overlaps IMD day
    `t+1` (03 UTC `t` to 03 UTC `t+1`) for 21 of 24 hours. IMD day `t` shares
    only 3 hours with it.

    Accepts a `pd.Timestamp` or a `pd.DatetimeIndex` of init times.
    """
    valid_time = pd.to_datetime(init_time) + pd.Timedelta(hours=lead_hours)
    return imd_day_containing(valid_time - _ONE_NANOSECOND)


def resample_to_imd_day(ds: xr.Dataset, time_dim: str = "time") -> xr.Dataset:
    """Resample/accumulate a sub-daily or differently-aligned dataset onto IMD's day.

    Sums over each 24h window from 03:00 UTC to the next 03:00 UTC, and
    labels the resulting day with the date the window *ends* on (matching
    IMD's own convention of labelling a day by the date its 08:30 IST total
    ends -- see `imd_day_containing`).

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
    # real time for the window's *start*. IMD labels a day by the date its
    # window *ends* on, so move each label forward one day and normalise to a
    # plain date rather than exposing the shifted timestamp.
    new_labels = pd.DatetimeIndex(resampled[time_dim].values).normalize() + pd.Timedelta(days=1)
    return resampled.assign_coords({time_dim: new_labels})
