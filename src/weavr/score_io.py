"""Per-day domain-wide scores: the input every confidence interval needs.

Every scoring script in this project writes an *aggregated* CSV -- one RMSE
per lead, one CRPS per bin. Those numbers cannot be given an error bar after
the fact: a bootstrap resamples days, so it needs the score of each day
separately. `weavr.significance` therefore has nothing to work with unless
the scripts also record per-day scores, which is what this module adds.

Two details make the difference between a CI that means something and one
that does not:

- **Ratios are not written; counts are.** The CSI of a resampled week is not
  the mean of its daily CSIs, because a ratio of sums is not the sum of
  ratios. Recording hits/misses/false_alarms/correct_negatives per day lets
  `scripts/run_scorecard.py` rebuild CSI, ETS or SEDI *inside* each bootstrap
  replicate, which is the only way to put an interval on a categorical
  score. The same reasoning applies to RMSE: this module writes daily **MSE**
  (`mse_mm2`), never daily RMSE, so the square root can be taken after
  averaging (see `weavr.significance.paired_difference_ci`).
- **Every method writes the same schema.** The scorecard pairs any two
  methods on their shared dates, so `graphcast` and `tier2_bma` must agree on
  column names and on what a row means. One row is one IMD day, scored over
  every grid cell in the domain.

Output layout: `results/per_day/<method>__lead<hours>.csv`, one file per
(method, lead). Flat, greppable, and diffable -- a directory of small CSVs
rather than one wide file, so a step that only regenerates Tier 2 doesn't
rewrite Tier 0's numbers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from weavr import verify as V

PER_DAY_SUBDIR = "per_day"
SPATIAL_DIMS = ("latitude", "longitude")


def _spatial_mean(values: xr.DataArray, dims: Sequence[str]) -> np.ndarray:
    present = [d for d in dims if d in values.dims]
    reduced = values.mean(dim=present, skipna=True) if present else values
    return np.asarray(reduced.values, dtype=float).ravel()


def per_day_scores(
    forecast: xr.DataArray,
    obs: xr.DataArray,
    *,
    fold: str = "test",
    sample_dim: str = "sample",
    spatial_dims: Sequence[str] = SPATIAL_DIMS,
    ensemble: xr.DataArray | None = None,
    member_dim: str = "member",
    thresholds: Sequence[float] = V.IMD_RAIN_THRESHOLDS_MM,
    twcrps_threshold: float = 64.5,
    probabilities: Mapping[float, xr.DataArray] | None = None,
    per_cell_scores: Mapping[str, xr.DataArray] | None = None,
) -> pd.DataFrame:
    """One row per day, with every score the scorecard can bootstrap.

    `forecast` and `obs` are `(sample, latitude, longitude)`. Optional extras:

    - `ensemble` `(sample, member, lat, lon)` adds `crps_mm` and
      `twcrps_<t>_mm`.
    - `probabilities` maps a threshold to a `(sample, lat, lon)` probability
      field and adds `brier_<t>`.
    - `per_cell_scores` maps a column name to an already-computed per-cell
      score, averaged here over the spatial dims. This is the route for
      `weavr.emos.score_csgd` and `weavr.bma.score_bma`, which return
      unreduced per-cell CRPS -- reusing their numbers rather than
      recomputing a second, possibly inconsistent, version.

    Columns always present: `date`, `fold`, `n_cells`, `mse_mm2`, `mae_mm`,
    and `hits_<t>` / `misses_<t>` / `false_alarms_<t>` / `correct_negatives_<t>`
    for each threshold.

    `n_cells` is the number of cells finite in both forecast and obs that
    day. The scorecard needs it because a day scored over 200 cells and one
    scored over 17,000 are not equally informative, and because a day with
    zero valid cells must be dropped rather than contributing a NaN.
    """
    dims = list(spatial_dims)
    error = forecast - obs
    valid = forecast.notnull() & obs.notnull()
    n_spatial_cells = int(np.prod([forecast.sizes[d] for d in dims if d in forecast.dims]))

    frame = pd.DataFrame(
        {
            "date": pd.DatetimeIndex(forecast[sample_dim].values),
            "fold": fold,
            "n_cells": _spatial_mean(valid.astype(float), dims) * n_spatial_cells,
            "mse_mm2": _spatial_mean(error**2, dims),
            "mae_mm": _spatial_mean(abs(error), dims),
        }
    )

    if ensemble is not None:
        # `crps_large_ensemble`, not `crps`: identical answer, but O(m)
        # memory instead of O(m^2). The 434-member climatological reference
        # needs ~83 GB by the plain route and was killed by the OS.
        frame["crps_mm"] = _spatial_mean(
            V.crps_large_ensemble(ensemble, obs, dim=dims, member_dim=member_dim), dims=[]
        )
        frame[f"twcrps_{twcrps_threshold}_mm"] = _spatial_mean(
            V.twcrps_ensemble(
                ensemble, obs, twcrps_threshold, dim=dims, member_dim=member_dim
            ),
            dims=[],
        )

    if per_cell_scores:
        for name, values in per_cell_scores.items():
            frame[name] = _spatial_mean(values, dims)

    if probabilities:
        for threshold, prob in probabilities.items():
            frame[f"brier_{threshold}"] = _spatial_mean(
                V.brier_score(prob, (obs >= threshold).astype(float), dim=dims), dims=[]
            )

    counts = V.contingency_counts(forecast, obs, thresholds=thresholds, dim=dims)
    for threshold, table in counts.items():
        for name, values in table.items():
            frame[f"{name}_{threshold}"] = np.asarray(values.values, dtype=float).ravel()

    return frame


def write_per_day_scores(
    method: str,
    lead: int,
    per_day: pd.DataFrame,
    out_dir: str | Path = "results",
) -> Path:
    """Write `results/per_day/<method>__lead<lead>.csv` and return its path.

    `method` is the name the scorecard will compare by, so it must be stable
    across runs and unique per scored configuration (`graphcast`, `tier0`,
    `tier2_emos_graphcast`, `climatology`, ...). `__` separates the method
    from the lead so a method name containing an underscore stays readable.
    """
    if not method:
        raise ValueError("method must be a non-empty name; the scorecard pairs on it.")

    directory = Path(out_dir) / PER_DAY_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{method}__lead{lead}.csv"
    per_day.to_csv(path, index=False)
    return path


def read_per_day_scores(out_dir: str | Path = "results") -> pd.DataFrame:
    """Read every per-day CSV back into one long frame.

    Adds `method` and `lead_hours` columns parsed from each filename, so
    `scripts/run_scorecard.py` can pair methods without a manifest. Returns
    an empty frame when the directory does not exist yet, rather than
    raising -- a fresh checkout has no results.
    """
    directory = Path(out_dir) / PER_DAY_SUBDIR
    if not directory.is_dir():
        return pd.DataFrame()

    frames = []
    for path in sorted(directory.glob("*__lead*.csv")):
        method, _, lead_part = path.stem.partition("__lead")
        frame = pd.read_csv(path, parse_dates=["date"])
        frame["method"] = method
        frame["lead_hours"] = int(lead_part)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
