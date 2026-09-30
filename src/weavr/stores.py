"""Read per-season Zarr stores as one dataset for leave-one-year-out work."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import xarray as xr

DEFAULT_BASELINE_DAILY_STORES: tuple[str, ...] = (
    "data/baseline_2018_jjas_daily.zarr",
    "data/baseline_2020_jjas_daily.zarr",
)
DEFAULT_LAGGED_DAILY_STORES: tuple[str, ...] = (
    "data/lagged_ensemble_inputs_2018_jjas_daily.zarr",
    "data/lagged_ensemble_inputs_2020_jjas_daily.zarr",
)
DEFAULT_IFS_DAILY_STORES: tuple[str, ...] = (
    "data/ifs_ens_2018_jjas_daily.zarr",
    "data/ifs_ens_2020_jjas_daily.zarr",
)


def resolve_store_paths(
    specified_paths: Sequence[str | Path] | None = None,
    legacy_single_path: str | Path | None = None,
    default_multi_paths: Sequence[str | Path] = DEFAULT_BASELINE_DAILY_STORES,
    legacy_fallback_path: str | Path | None = "data/baseline_2020_jjas.zarr",
    allow_legacy_fallback: bool = False,
) -> list[str | Path]:
    """Resolve store paths, refusing to run on a partial or substituted store set.

    Order of precedence:
    1. If `legacy_single_path` is explicitly provided, return `[legacy_single_path]`.
    2. If `specified_paths` is provided and non-empty, return list of those paths.
    3. If every default multi-season path exists, return them all.
    4. If *no* default exists, and `allow_legacy_fallback` is set, return
       `[legacy_fallback_path]` if that legacy store is on disk.
    5. Return the full `default_multi_paths` so the caller fails on the open.

    Two substitutions used to happen silently here, and both turned missing data
    into plausible-looking output rather than an error:

    - A **partially** present default returned the subset that happened to
      exist. With only the 2020 IFS-ENS store on disk, a two-season Tier 2 run
      bound IFS-ENS to 2020 alone, and every method drawing on it -- both
      EMOS-CSG variants and BMA -- scored `nan` in the fold testing on 2018.
    - The **legacy fallback** fired when *no* default existed. This is the
      subtler one: `DEFAULT_IFS_DAILY_STORES` names two daily stores and
      neither is on disk, but the weekly `data/ifs_ens_2020_jjas.zarr` is, so
      the run silently received 18 weekly samples in place of 122 daily ones
      for both seasons. The first guard alone does not catch this, because
      nothing was partially present -- everything was absent.

    So the legacy fallback is now opt-in via `allow_legacy_fallback`. Pass
    `specified_paths` to scope a run deliberately to the seasons you have.
    """
    if legacy_single_path is not None:
        return [legacy_single_path]
    if specified_paths is not None and len(specified_paths) > 0:
        return list(specified_paths)

    existing = [p for p in default_multi_paths if Path(p).exists()]
    if len(existing) == len(default_multi_paths):
        return existing

    missing = [p for p in default_multi_paths if not Path(p).exists()]
    if existing:
        raise FileNotFoundError(
            "Refusing to run on a partial multi-season store set: found "
            f"{[str(p) for p in existing]} but missing "
            f"{[str(p) for p in missing]}.\n"
            "Running anyway scores the missing season as NaN, silently. Either "
            "build the missing store, or pass the paths you intend explicitly "
            "(e.g. --baseline-stores / --ifs-ensemble-stores) to scope the run "
            "to the seasons you actually have."
        )

    if legacy_fallback_path is not None and Path(legacy_fallback_path).exists():
        if not allow_legacy_fallback:
            raise FileNotFoundError(
                f"None of the {len(default_multi_paths)} default multi-season "
                f"stores exist, but the legacy single-season store "
                f"{legacy_fallback_path} does.\n"
                "Substituting it would run a single-season evaluation where a "
                "multi-season one was requested, silently. Pass "
                "allow_legacy_fallback=True if that is genuinely intended, or "
                "pass the store paths explicitly to scope the run."
            )
        return [legacy_fallback_path]
    return list(default_multi_paths)


def open_multi_season(paths: Sequence[str | Path], group: str | None = None) -> xr.Dataset:
    """Open matching source groups and concatenate their disjoint seasons.

    Each season remains an independent, resumable store with its own manifest.
    The loader refuses mismatched grids, variables, time axes, or overlapping
    dates so a later training run cannot silently align incompatible stores.
    """
    if not paths:
        raise ValueError("At least one season path is required")

    datasets: list[xr.Dataset] = []
    try:
        for path in paths:
            if group is not None:
                datasets.append(xr.open_zarr(path, group=group, consolidated=True))
            else:
                datasets.append(xr.open_zarr(path, consolidated=True))

        reference = datasets[0]
        for coord in ("latitude", "longitude"):
            if coord not in reference.coords:
                raise ValueError(f"First season is missing {coord!r}")
        time_dim = "time" if "time" in reference.dims else "nominal_time"
        if time_dim not in reference.dims:
            raise ValueError("First season has neither time nor nominal_time")

        previous_end: np.datetime64 | None = None
        for path, ds in zip(paths, datasets, strict=True):
            if time_dim not in ds.dims:
                raise ValueError(f"{path}: missing {time_dim!r} dimension")
            if set(ds.data_vars) != set(reference.data_vars):
                raise ValueError(f"{path}: data variables differ from the first season")
            for coord in ("latitude", "longitude"):
                if coord not in ds.coords or not np.array_equal(ds[coord], reference[coord]):
                    raise ValueError(f"{path}: {coord} grid differs from the first season")
            times = np.asarray(ds[time_dim].values)
            if not len(times) or np.any(np.diff(times) <= np.timedelta64(0, "ns")):
                raise ValueError(f"{path}: {time_dim} must be nonempty and strictly increasing")
            if previous_end is not None and times[0] <= previous_end:
                raise ValueError(f"{path}: seasons overlap or are out of order")
            previous_end = times[-1]

        return xr.concat(datasets, dim=time_dim, join="exact", compat="equals")
    except Exception:
        for ds in datasets:
            ds.close()
        raise
