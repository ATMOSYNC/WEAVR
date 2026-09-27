"""Loader for the HEPPI (WCSSP India) reference dataset.

HEPPI (Michael Angus, WCSSP India, 2020-2021) is a third-party research
output: NCMRWF NEPS-G ensemble forecasts (raw, plus two already-applied
bias-correction variants: univariate quantile mapping and EMOS) paired with
IMD observed rainfall, all pre-regridded onto IMD's native 0.25 degree grid.
It is not redistributed with this repository — obtain it yourself from
https://edata.bham.ac.uk/698/ and point these loaders at your local copy.

Why this is worth loading at all: the grid the authors used is, point for
point, weavr's own locked common grid (see weavr.grid) — confirmed below by
`_assign_grid_coords`, not assumed. That makes this dataset usable without
any regridding as a methodology reference for Phase 1 bias-correction work
(compare against UQM/EMOS with a known-good pair) and later verification
work, once its date caveat (below) is accounted for.

Known gap: the original release paired these files with a calendar-date
lookup (`IMD_dates.mat`) that is not part of this download. The `sample`
dimension exposed here is therefore a plain index, not a calendar date —
do not assume `sample=0` is any particular day without recovering that
mapping first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from weavr.grid import COMMON_LAT, COMMON_LON

FORECAST_VARIANTS = {
    "orig": ("NCMRWF_orig_forecast.nc", "NCMRWF_frcst"),
    "uqm": ("NCMRWF_UQM_forecast.nc", "NCMRWF_UQM_frcst"),
    "emos": ("NCMRWF_EMOS_forecast.nc", "NCMRWF_EMOS_frcst"),
}

SAMPLE_DATE_CAVEAT = (
    "Index-based 'sample' dimension: the original calendar-date lookup "
    "(IMD_dates.mat) is not part of this download. Do not assume sample "
    "order corresponds to a specific date range without recovering it."
)


class HeppiGridMismatchError(ValueError):
    """Raised when a HEPPI file's lat/lon grid doesn't match weavr's locked common grid.

    HEPPI ships lat/lon as plain data variables (IMD_lat, IMD_lon), not
    coordinates, so the files carry no built-in guarantee they still match
    weavr.grid.COMMON_LAT/COMMON_LON. Verify on every load rather than
    trusting the one-time check in this module's docstring.
    """


def _assign_grid_coords(ds: xr.Dataset) -> xr.Dataset:
    lat = ds["IMD_lat"].values
    lon = ds["IMD_lon"].values
    if lat.shape != COMMON_LAT.shape or not np.allclose(lat, COMMON_LAT):
        raise HeppiGridMismatchError(
            "HEPPI latitude grid does not match weavr.grid.COMMON_LAT; "
            "regrid before use or investigate why the source grid changed."
        )
    if lon.shape != COMMON_LON.shape or not np.allclose(lon, COMMON_LON):
        raise HeppiGridMismatchError(
            "HEPPI longitude grid does not match weavr.grid.COMMON_LON; "
            "regrid before use or investigate why the source grid changed."
        )
    return ds.assign_coords(lat=("lat", lat), lon=("lon", lon))


def _standardize(da: xr.DataArray, note: str) -> xr.DataArray:
    da = da.rename({"forecast": "sample"})
    order = [d for d in ("sample", "member", "lat", "lon") if d in da.dims]
    da = da.transpose(*order)
    da.attrs["note"] = note
    return da


def load_imd_observed(path: str | Path) -> xr.DataArray:
    """Load HEPPI's IMD observed rainfall as a (sample, lat, lon) DataArray."""
    ds = xr.open_dataset(path)
    ds = _assign_grid_coords(ds)
    da = ds["IMD_rainfall_observed"]
    return _standardize(da, SAMPLE_DATE_CAVEAT).rename("rainfall_observed")


def load_ncmrwf_forecast(path_dir: str | Path, variant: str = "orig") -> xr.DataArray:
    """Load an NCMRWF ensemble forecast variant as a (sample, member, lat, lon) DataArray.

    `variant` selects which of the three HEPPI forecast files to load:
    "orig" (raw ensemble), "uqm" (univariate quantile mapping applied), or
    "emos" (ensemble MOS applied) — the two bias-correction methods the
    original HEPPI project already ran, useful as a reference to validate
    weavr's own Phase 1 bias-correction implementation against.
    """
    if variant not in FORECAST_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {sorted(FORECAST_VARIANTS)}")
    filename, var_name = FORECAST_VARIANTS[variant]
    ds = xr.open_dataset(Path(path_dir) / filename)
    ds = _assign_grid_coords(ds)
    da = ds[var_name].rename({"ens": "member"})
    note = f"{SAMPLE_DATE_CAVEAT} Bias-correction variant: {variant!r}."
    return _standardize(da, note).rename(f"ncmrwf_{variant}_forecast")
