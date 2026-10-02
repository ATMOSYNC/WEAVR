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

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from weavr import verify as V

PER_DAY_SUBDIR = "per_day"
SPATIAL_DIMS = ("latitude", "longitude")

# Shared --force help so every scoring script offers the same escape hatch from
# guard_result_overwrites in the same words.
FORCE_HELP = (
    "Overwrite existing result CSVs. Without this the script refuses to "
    "replace a file that already exists, so a scratch run cannot silently "
    "clobber the committed results."
)


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


class PerDayScoreWriter:
    """Collect per-day frames across folds, then write one file per method/lead.

    `write_per_day_scores` names a file by `(method, lead)` alone, so calling it
    once per fold overwrites the previous fold: a two-season LOYO run kept only
    the *last* fold's days, and the 2018 fold vanished from the evidence without
    an error. The aggregate CSVs still looked right, because each fold was
    scored in memory, so the loss only showed up as scorecard CIs that could not
    be computed -- silently, on the numbers that decide the pre-registered
    claims.

    Buffering here makes the fold dimension structural rather than a convention
    each runner has to remember: `add` as many folds as there are, `flush`
    once. Forgetting `flush` writes nothing at all, which fails loudly, instead
    of writing a plausible-looking half-complete file.

    Use it as a context manager to make forgetting `flush` impossible.
    """

    def __init__(self, out_dir: str | Path = "results") -> None:
        self.out_dir = out_dir
        self._frames: dict[tuple[str, int], list[pd.DataFrame]] = {}
        self._flushed = False

    def add(self, method: str, lead: int, per_day: pd.DataFrame) -> None:
        """Buffer one fold's rows for `(method, lead)`."""
        if self._flushed:
            raise RuntimeError(
                "add() after flush(): the files are already written, so later "
                "folds would be dropped. Move flush() to the end of the folds."
            )
        if not method:
            raise ValueError("method must be a non-empty name; the scorecard pairs on it.")
        if "fold" not in per_day.columns:
            raise ValueError(
                f"per-day frame for {method!r} lead {lead} has no 'fold' column; "
                "the fold label is what keeps folds from overwriting each other."
            )
        self._frames.setdefault((method, lead), []).append(per_day)

    def flush(self) -> list[Path]:
        """Write every buffered `(method, lead)` and return the paths written."""
        written: list[Path] = []
        for (method, lead), frames in sorted(self._frames.items()):
            combined = pd.concat(frames, ignore_index=True)
            duplicates = combined["date"].duplicated().sum()
            if duplicates:
                raise ValueError(
                    f"{method!r} lead {lead} produced {duplicates} duplicate dates "
                    "across folds. Two folds must score disjoint days; otherwise "
                    "the paired comparison would double-count them."
                )
            written.append(
                write_per_day_scores(method, lead, combined, out_dir=self.out_dir)
            )
        self._frames.clear()
        self._flushed = True
        return written

    def __enter__(self) -> PerDayScoreWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        if not self._flushed and exc_info[0] is not None:
            self.flush()


def resolve_result_paths(
    results_dir: str | Path,
    filenames: Mapping[str, str],
    overrides: Mapping[str, str | None] | None = None,
) -> dict[str, str]:
    """Place every result CSV in one directory unless told otherwise.

    `filenames` maps an argparse dest to the filename it defaults to, so all
    of a script's outputs share the single `--results-dir` the caller chose.

    Scripts used to hardcode `"results/..."` as each argparse default,
    independently of one another. That made `--out-csv`/`--results-dir`
    unreliable: pointing them at a scratch directory redirected the main CSV
    and the per-day files, but the per-bin and per-region CSVs still resolved
    to `results/` and overwrote the committed, reviewed numbers in place. A
    silent wrong-number bug, which is worse than a crash.

    An explicit path in `overrides` still wins, so a caller can scatter
    outputs if it genuinely needs to.
    """
    directory = Path(results_dir)
    resolved = {
        dest: str(directory / name) for dest, name in filenames.items()
    }
    for dest, override in (overrides or {}).items():
        if override:
            resolved[dest] = str(override)
    return resolved


def guard_result_overwrites(
    paths: Sequence[str | Path],
    force: bool = False,
) -> None:
    """Refuse to replace existing result CSVs unless `force` is set.

    `results/` holds committed, reviewed numbers. Overwriting them with a
    differently-scoped run -- a single-season one, or a partial one that
    fails halfway -- is the silent-wrong-number failure this project cares
    about, so it raises `SystemExit` rather than warning.

    `SystemExit` (not `ValueError`) because the intended response is "the
    operator picks different paths or passes `--force`", which is what an
    uncaught `SystemExit` in a script produces: a non-zero exit and a message.
    """
    if force:
        return
    existing = [str(p) for p in paths if Path(p).exists()]
    if not existing:
        return
    listed = "\n  ".join(existing)
    raise SystemExit(
        f"Refusing to overwrite existing result file(s):\n  {listed}\n"
        "These paths hold committed results. Write somewhere else "
        "(--results-dir, or the per-file output flags), or pass --force if "
        "replacing them is genuinely intended."
    )


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


def write_rows_csv(path: str | Path, rows: Sequence[Mapping[str, object]]) -> Path:
    """Write a list of row dicts to `path`, tolerating heterogeneous keys.

    The fieldnames come from the **union** of every row's keys, in first-seen
    order, rather than from `rows[0]`. Deriving them from the first row alone
    is a trap that fires only after all the expensive work is done: a runner
    that appends per-fold rows and then pooled rows, where the pooled rows
    carry an extra column, produces a perfectly valid first row and a later
    row that `csv.DictWriter` rejects with "dict contains fields not in
    fieldnames". That is exactly what killed a two-season Tier 2 run: after
    two hours of BMA sampling, writing the per-bin CSV raised, the per-day
    scores were still buffered in memory, and the whole run was lost.

    Union semantics also mean a caller cannot lose a column by accident, and
    a missing key is written as an empty field rather than crashing -- which
    is the honest representation for a row that genuinely does not carry a
    per-region or per-bin statistic.

    Returns the path written.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        if fieldnames:
            writer.writeheader()
        writer.writerows(rows)
    return out_path
