"""Thin client for IMD's 0.25 degree gridded daily rainfall, via imdlib.

No authentication needed — imdlib downloads IMD's public binary (.grd)
files directly from IMD's server and caches them on disk. `get_xarray()`
already converts IMD's internal missing-data fill to NaN, so callers get a
clean array with no extra masking step needed.

See https://github.com/iamsaswata/imdlib.
"""

from __future__ import annotations

from pathlib import Path

import imdlib
import xarray as xr


def fetch_year(
    year: int,
    var_type: str = "rain",
    cache_dir: str | Path = "/tmp/imdlib_cache",
) -> xr.Dataset:
    """Download (or reuse cached) IMD gridded data for one full year and return as xarray.

    var_type: "rain" (daily rainfall, 0.25 deg), "tmin"/"tmax" (daily temperature, 1 deg).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    data = imdlib.get_data(
        var_type,
        year,
        year,
        fn_format="yearwise",
        file_dir=str(cache_dir) + "/",
    )
    return data.get_xarray()
