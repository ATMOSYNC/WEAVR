"""Climatological reference forecasts, for skill scores that mean something.

A skill score needs a reference, and the only honest reference for "is this
forecast useful?" is what you would have said knowing nothing but the
climate. `docs/preregistration.md`'s H3 is stated against exactly this:
P(>= 64.5 mm) must have a Brier skill score above zero *against IMD
climatology*, not merely a low Brier score in absolute terms. A rare event
gets a low Brier score from forecasting "never", so an absolute score proves
nothing.

The climatological ensemble for a target date is built from the IMD archive
itself: for each grid cell, the members are the observed rainfall on nearby
days of the season, across the archive's years. That gives a full
distribution per cell -- so it can be scored with CRPS and with Brier at any
threshold, using the same functions as any other ensemble.

**Two things here would silently invalidate every skill score, so both are
explicit:**

- **The test year is excluded.** Including the year being scored would put
  the answer inside the reference forecast, making the reference
  artificially good and WEAVR's skill against it artificially bad. Under
  leave-one-year-out, `exclude_years` is the held-out fold.
- **The member count is constant across target dates.** The obvious
  implementation -- a symmetric +/- window on day-of-season -- truncates near
  1 June and 30 September, giving fewer members there. A ragged ensemble
  either needs NaN padding (which propagates through `crps`) or makes the
  reference quietly sharper mid-season than at the edges. Instead the window
  slides inward at the season boundaries, keeping `2 * window_days + 1`
  distinct days of season everywhere. It is the nearest available stretch of
  season, and it is a shift rather than a truncation.

The archive is `data/imd_seeps_climatology_jjas.zarr` (IMD-only, JJAS
2006-2020, 15 years), loaded by
`scripts/build_seeps_climatology.load_climatology` -- which is the caller's
job, so that everything in this module stays pure and testable against a
synthetic store.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_WINDOW_DAYS = 15

# Days are matched across years by their position in a fixed NON-LEAP
# reference year, not by raw day-of-year. Day-of-year is off by one after
# February in a leap year -- 1 June is day 153 in 2020 but day 152 in 2018 --
# so matching on it silently misaligns leap years against the rest of the
# archive and produces ragged member counts. Caught by
# tests/test_climatology.py::test_member_count_is_constant_at_the_season_edges
# against a real 2018/2019/2020 archive, not reasoned about in advance.
_SEASONAL_REFERENCE_YEAR = 2001  # any non-leap year


def _seasonal_index(times: pd.DatetimeIndex) -> np.ndarray:
    """Position of each date within the year, comparable across leap years.

    29 February has no counterpart in the reference year and maps to 1 March.
    That is irrelevant for the monsoon seasons this project covers (JJAS, and
    OND for step 25), and is noted rather than special-cased.
    """
    days = np.where((times.month == 2) & (times.day == 29), 28, times.day)
    return np.array(
        [
            pd.Timestamp(_SEASONAL_REFERENCE_YEAR, int(m), int(d)).dayofyear
            for m, d in zip(times.month, days, strict=True)
        ]
    )


def _window_days_of_year(
    target_doy: int, available_doys: np.ndarray, window_days: int
) -> np.ndarray:
    """The `2 * window_days + 1` days of season nearest `target_doy`.

    Centred on the target where possible, slid inward at the season edges so
    the count never changes. If the season itself is shorter than the
    requested width, every available day is returned.
    """
    width = 2 * window_days + 1
    if available_doys.size <= width:
        return available_doys

    low, high = available_doys.min(), available_doys.max()
    start = target_doy - window_days
    end = target_doy + window_days
    if start < low:
        start, end = low, low + width - 1
    elif end > high:
        start, end = high - width + 1, high
    return available_doys[(available_doys >= start) & (available_doys <= end)]


def climatological_ensemble(
    climatology: xr.Dataset | xr.DataArray,
    target_dates: pd.DatetimeIndex | np.ndarray,
    exclude_years: int | list[int] | tuple[int, ...] = (),
    window_days: int = DEFAULT_WINDOW_DAYS,
    variable: str = "rain",
    sample_dim: str = "sample",
    member_dim: str = "member",
) -> xr.DataArray:
    """A per-cell climatological ensemble for each target date.

    Returns a `(sample, member, latitude, longitude)` DataArray whose members
    are observed IMD values from the nearest `2 * window_days + 1` days of
    season, across every archive year **except** those in `exclude_years`.

    With the default 15-day window and a 15-year archive, a cell gets
    31 days x 15 years = 465 members -- minus the excluded year, so 434 under
    leave-one-year-out. That is a well-resolved reference distribution, which
    matters because it is the denominator of every skill score.

    Raises `ValueError` if excluding those years leaves no data, rather than
    returning an empty ensemble that would produce silent NaN skill scores.
    """
    data = climatology[variable] if isinstance(climatology, xr.Dataset) else climatology
    if isinstance(exclude_years, int):
        exclude_years = (exclude_years,)
    excluded = set(exclude_years)

    times = pd.DatetimeIndex(data["time"].values)
    keep = ~np.isin(times.year, list(excluded)) if excluded else np.ones(len(times), bool)
    if not keep.any():
        raise ValueError(
            f"Excluding years {sorted(excluded)} removes every day in the climatology "
            f"archive (which covers {sorted(set(times.year))}). A skill score needs a "
            "non-empty reference."
        )
    usable = data.isel(time=keep)
    usable_times = times[keep]
    usable_doys = _seasonal_index(usable_times)
    available_doys = np.unique(usable_doys)

    targets = pd.DatetimeIndex(target_dates)
    target_doys = _seasonal_index(targets)
    members_per_date = []
    for target_doy in target_doys:
        chosen = _window_days_of_year(int(target_doy), available_doys, window_days)
        selected = usable.isel(time=np.flatnonzero(np.isin(usable_doys, chosen)))
        members_per_date.append(
            selected.rename({"time": member_dim}).drop_vars(member_dim, errors="ignore")
        )

    counts = {int(m.sizes[member_dim]) for m in members_per_date}
    if len(counts) > 1:
        # The sliding window is supposed to make this impossible; if it ever
        # happens, say so loudly rather than padding with NaN and letting a
        # ragged reference quietly distort the skill scores.
        raise ValueError(
            f"Climatological ensembles have inconsistent member counts {sorted(counts)}. "
            "The archive is probably missing days for some years."
        )

    ensemble = xr.concat(members_per_date, dim=sample_dim)
    ensemble = ensemble.assign_coords({sample_dim: targets})
    return ensemble.assign_coords({member_dim: np.arange(ensemble.sizes[member_dim])})


def climatological_probability(
    climatology: xr.Dataset | xr.DataArray,
    target_dates: pd.DatetimeIndex | np.ndarray,
    threshold: float,
    exclude_years: int | list[int] | tuple[int, ...] = (),
    window_days: int = DEFAULT_WINDOW_DAYS,
    variable: str = "rain",
    sample_dim: str = "sample",
    member_dim: str = "member",
) -> xr.DataArray:
    """The climatological probability of exceeding `threshold`, per cell and date.

    The event frequency of the same ensemble `climatological_ensemble`
    returns: the fraction of nearby-day, other-year observations at or above
    the threshold. This is the reference probability forecast that
    `weavr.verify.brier_skill_score` is computed against, and the honest
    answer to "what would you have said knowing only the climate?".

    Uses `>=`, matching `weavr.verify.contingency_scores`' event convention
    (an event is `value >= threshold`) so a probability and a contingency
    table at the same threshold describe the same event.
    """
    ensemble = climatological_ensemble(
        climatology,
        target_dates,
        exclude_years=exclude_years,
        window_days=window_days,
        variable=variable,
        sample_dim=sample_dim,
        member_dim=member_dim,
    )
    return (ensemble >= threshold).mean(dim=member_dim, skipna=True)
