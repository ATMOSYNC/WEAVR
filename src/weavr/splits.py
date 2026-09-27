"""Block-by-time train/verification splits -- the only sanctioned way to split
time series data anywhere in this codebase.

Random splits (`sklearn`'s default, `np.random.shuffle`) leak information on
time-series weather data: a random split puts days from the same monsoon
spell, sometimes the same week, on both sides, so a model can effectively
"see" conditions adjacent to what it's evaluated on. That inflates apparent
skill in a way that won't reproduce operationally, where you only ever have
the past. Every later phase (Phase 1's Tier 0 baseline, Phase 3's regional
weights, Phase 4's BMA/EMOS fitting) must split time via one of these two
functions instead.

Which one to use: `leave_one_year_out` needs at least two distinct years of
data and gives the more rigorous split (train on every other year, test on
one). Checked directly against the real Phase 1 data (not assumed): the
paired forecast+obs baseline store (`data/baseline_2020_jjas.zarr`) still
covers a single season, 2020 JJAS only -- the 15-year archive built for
SEEPS/ACC climatology (`data/imd_seeps_climatology_jjas.zarr`) is IMD-only,
with no paired forecasts, so it isn't a dataset you train/test a forecast
blend against. `leave_one_year_out` will therefore raise
`InsufficientTimeBlocksError` against today's baseline store, by design --
use `seasonal_block_split` until a multi-year paired store exists.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
import xarray as xr

TimesLike: TypeAlias = xr.Dataset | xr.DataArray | pd.DatetimeIndex | np.ndarray


class InsufficientTimeBlocksError(ValueError):
    """Raised when a time-based split is requested but the input doesn't have
    enough distinct time blocks (years, or timestamps) to produce a
    meaningful split -- rather than silently degenerating into a
    random-looking split of a single block.
    """


def _extract_times(dataset_or_times: TimesLike, time_dim: str) -> pd.DatetimeIndex:
    if isinstance(dataset_or_times, xr.Dataset | xr.DataArray):
        if time_dim not in dataset_or_times.coords:
            raise ValueError(f"Input is missing expected time coordinate: {time_dim!r}")
        return pd.DatetimeIndex(dataset_or_times[time_dim].values)
    return pd.DatetimeIndex(dataset_or_times)


def leave_one_year_out(
    dataset_or_times: TimesLike,
    time_dim: str = "time",
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_mask, test_mask) boolean arrays, one per distinct year present.

    Each iteration holds out exactly one calendar year as the test block and
    uses every other year as train -- e.g. with data spanning 2018-2021,
    yields 4 folds, each testing on one year and training on the other 3.
    Masks are boolean arrays aligned to `dataset_or_times`'s time axis (or
    the input array itself, if a plain time array/index was passed), usable
    directly with `.isel({time_dim: mask})`.

    Raises `InsufficientTimeBlocksError` if fewer than 2 distinct years are
    present -- a single year of data cannot produce a meaningful
    leave-one-year-out split (there would be nothing left to train on for
    the one held-out year). Use `seasonal_block_split` instead in that case.
    """
    times = _extract_times(dataset_or_times, time_dim)
    years = times.year.values
    distinct_years = np.unique(years)

    if len(distinct_years) < 2:
        raise InsufficientTimeBlocksError(
            f"leave_one_year_out needs at least 2 distinct years, got "
            f"{len(distinct_years)} ({list(distinct_years)}). A single-year "
            "store cannot produce a meaningful leave-one-year-out split -- "
            "use seasonal_block_split instead."
        )

    for year in sorted(distinct_years):
        test_mask = years == year
        train_mask = ~test_mask
        yield train_mask, test_mask


def seasonal_block_split(
    dataset_or_times: TimesLike,
    test_fraction: float = 0.2,
    time_dim: str = "time",
    position: Literal["trailing", "leading"] = "trailing",
) -> tuple[np.ndarray, np.ndarray]:
    """Split by holding out one contiguous chronological block, not a random subset.

    Sorts the distinct timestamps in `dataset_or_times`, then holds out the
    last (`position="trailing"`, the default -- train on the past, test on
    the future, matching how this would run operationally) or first
    (`position="leading"`) `test_fraction` of them as the test block. The
    boundary is computed from the actual sorted timestamps present, not an
    assumed contiguous calendar range, so missing days (e.g. IMD's own
    documented gaps) don't shift it -- and because it's a single cut through
    sorted time, every train timestamp is strictly before every test
    timestamp (or after, for `position="leading"`), never interleaved.

    Use this instead of `leave_one_year_out` when fewer than 2 distinct
    years of data exist -- true for Phase 1's current baseline store (one
    JJAS season only, see this module's docstring).

    Raises `InsufficientTimeBlocksError` if there are fewer than 2 distinct
    timestamps, or if `test_fraction` is so extreme relative to the data
    that either the train or test block would be empty -- rather than
    silently ignoring the requested fraction and returning a degenerate
    split.
    """
    if not (0.0 < test_fraction < 1.0):
        raise ValueError(f"test_fraction must be strictly between 0 and 1, got {test_fraction}")

    times = _extract_times(dataset_or_times, time_dim)
    n = len(times)
    if n < 2:
        raise InsufficientTimeBlocksError(
            f"seasonal_block_split needs at least 2 distinct timestamps, got {n}."
        )

    order = np.argsort(times.values)
    n_test = round(n * test_fraction)
    if n_test <= 0 or n_test >= n:
        raise InsufficientTimeBlocksError(
            f"test_fraction={test_fraction} over {n} timestamps would produce an "
            f"empty train or test block ({n_test} of {n}) -- choose a less extreme "
            "test_fraction rather than silently dropping one side of the split."
        )

    if position == "trailing":
        test_positions, train_positions = order[n - n_test :], order[: n - n_test]
    elif position == "leading":
        test_positions, train_positions = order[:n_test], order[n_test:]
    else:
        raise ValueError(f"position must be 'trailing' or 'leading', got {position!r}")

    train_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[train_positions] = True
    test_mask[test_positions] = True
    return train_mask, test_mask
