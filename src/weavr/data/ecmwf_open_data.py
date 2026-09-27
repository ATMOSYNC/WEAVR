"""Thin client for ECMWF's open-data feed (IFS HRES + AIFS).

No authentication needed. Data is served as GRIB2; reading it needs
cfgrib + eccodes, which are pinned as project dependencies.

ECMWF's own direct portal (`source="ecmwf"`) enforces a 500-simultaneous-
connection cap and returned HTTP 429 repeatedly when this was tested — the
underlying client retries with backoff, but it's slow under load. ECMWF
publishes the same feed on AWS/Azure/GCS specifically as relief for this
("For added reliability, the open-data is replicated across AWS, Azure, and
Google Cloud" — from the client's own startup message). Confirmed working
here: `source="aws"` returns immediately with no rate-limit errors, so it is
the default.

See https://github.com/ecmwf/ecmwf-opendata for request parameters.
Data is licensed CC BY 4.0 — attribute ECMWF when using it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import xarray as xr
from ecmwf.opendata import Client

INDIA_LAT_SLICE = slice(38.0, 6.0)  # ECMWF grib latitudes run north->south
INDIA_LON_SLICE = slice(68.0, 98.0)


def fetch_latest_forecast(
    param: str = "tp",
    step: int = 0,
    model: str = "ifs",
    forecast_type: str = "fc",
    source: str = "aws",
) -> xr.Dataset:
    """Retrieve the latest available forecast for one parameter/step as an xr.Dataset.

    Downloads a single GRIB2 message to a temp file, then opens it with
    cfgrib. Kept deliberately small (one param, one step) — this proves
    access, it is not the bulk puller. Defaults to the AWS mirror (see
    module docstring) rather than ECMWF's own rate-limited portal.
    """
    client = Client(source=source, model=model)
    with tempfile.TemporaryDirectory() as tmpdir:
        target = str(Path(tmpdir) / "fetch.grib2")
        client.retrieve(type=forecast_type, step=step, param=param, target=target)
        ds = xr.open_dataset(target, engine="cfgrib")
        return ds.load()


def slice_india(ds: xr.Dataset) -> xr.Dataset:
    """Slice an ECMWF open-data dataset down to the India bounding box."""
    return ds.sel(latitude=INDIA_LAT_SLICE, longitude=INDIA_LON_SLICE)
