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
) -> list[str | Path]:
    """Resolve store paths, falling back to existing files if defaults are partially present.

    Order of precedence:
    1. If `legacy_single_path` is explicitly provided, return `[legacy_single_path]`.
    2. If `specified_paths` is provided and non-empty, return list of those paths.
    3. If default multi-season paths exist on disk, return existing default paths.
    4. If legacy fallback path exists on disk, return `[legacy_fallback_path]`.
    5. Return the full `default_multi_paths`.
    """
    if legacy_single_path is not None:
        return [legacy_single_path]
    if specified_paths is not None and len(specified_paths) > 0:
        return list(specified_paths)
    existing_defaults = [p for p in default_multi_paths if Path(p).exists()]
    if existing_defaults:
        return existing_defaults
    if legacy_fallback_path is not None and Path(legacy_fallback_path).exists():
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
