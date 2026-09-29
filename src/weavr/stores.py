"""Read per-season Zarr stores as one dataset for leave-one-year-out work."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import xarray as xr


def open_multi_season(paths: Sequence[str | Path], group: str) -> xr.Dataset:
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
            datasets.append(xr.open_zarr(path, group=group, consolidated=True))

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
