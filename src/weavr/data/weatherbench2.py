"""Thin client for the WeatherBench 2 public Zarr archive on GCS.

Serves IFS HRES/ENS and AI-model (GraphCast, Pangu, ...) forecasts on a
common grid. See https://weatherbench2.readthedocs.io/en/latest/data-guide.html
for the full catalog.

No authentication needed — the bucket is public, accessed anonymously.
"""

from __future__ import annotations

import xarray as xr

GCS_ANON_STORAGE_OPTIONS = {"token": "anon"}

# India bounding box (approx.), used to slice test/working sets down to a
# manageable size. WeatherBench 2 longitudes run 0-360, not -180..180.
INDIA_LAT_SLICE = slice(6.0, 38.0)
INDIA_LON_SLICE = slice(68.0, 98.0)

# Known dataset paths as of the 2020 GraphCast eval period. WeatherBench 2's
# catalog is organised as gs://weatherbench2/datasets/<model>/<year>/<range
# descriptor>.zarr — list the bucket with gcsfs to find other years/models:
#   gcsfs.GCSFileSystem(token="anon").ls("gs://weatherbench2/datasets/<model>")
GRAPHCAST_2020 = (
    "gs://weatherbench2/datasets/graphcast/2020/"
    "date_range_2019-11-16_2021-02-01_12_hours-240x121_equiangular_with_poles_conservative.zarr"
)


def open_dataset(zarr_path: str) -> xr.Dataset:
    """Lazily open a WeatherBench 2 Zarr store (dask-backed, no data pulled yet)."""
    return xr.open_zarr(zarr_path, storage_options=GCS_ANON_STORAGE_OPTIONS, consolidated=True)


def fetch_india_slice(
    zarr_path: str,
    variable: str,
    n_times: int = 1,
) -> xr.DataArray:
    """Pull a small India-bounded slice of one variable, for smoke-testing access.

    Only the first `n_times` forecast initialisations and the first
    prediction_timedelta step are materialised, to keep this cheap.
    """
    ds = open_dataset(zarr_path)
    da = ds[variable]

    if "time" in da.dims:
        da = da.isel(time=slice(0, n_times))
    if "prediction_timedelta" in da.dims:
        da = da.isel(prediction_timedelta=0)

    return da.sel(latitude=INDIA_LAT_SLICE, longitude=INDIA_LON_SLICE).load()
