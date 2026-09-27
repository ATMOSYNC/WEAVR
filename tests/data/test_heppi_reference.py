from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from weavr.data.heppi_reference import (
    HeppiGridMismatchError,
    load_imd_observed,
    load_ncmrwf_forecast,
)
from weavr.grid import COMMON_LAT, COMMON_LON

N_SAMPLES = 3
N_MEMBERS = 2


def _write_imd_observed(path: Path, lat=COMMON_LAT, lon=COMMON_LON) -> None:
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {
            "IMD_lat": ("lat", lat),
            "IMD_lon": ("lon", lon),
            "IMD_rainfall_observed": (
                ("forecast", "lon", "lat"),
                rng.random((N_SAMPLES, len(lon), len(lat))),
            ),
        }
    )
    ds.to_netcdf(path)


def _write_ncmrwf_forecast(path: Path, var_name: str, lat=COMMON_LAT, lon=COMMON_LON) -> None:
    rng = np.random.default_rng(1)
    ds = xr.Dataset(
        {
            "IMD_lat": ("lat", lat),
            "IMD_lon": ("lon", lon),
            "ensemble_members": ("ens", np.arange(N_MEMBERS, dtype=float)),
            var_name: (
                ("forecast", "ens", "lon", "lat"),
                rng.random((N_SAMPLES, N_MEMBERS, len(lon), len(lat))),
            ),
        }
    )
    ds.to_netcdf(path)


class TestLoadImdObserved:
    def test_loads_and_standardizes_dims(self, tmp_path):
        f = tmp_path / "IMD_observed.nc"
        _write_imd_observed(f)

        da = load_imd_observed(f)

        assert da.dims == ("sample", "lat", "lon")
        assert da.sizes["sample"] == N_SAMPLES
        assert np.allclose(da["lat"].values, COMMON_LAT)
        assert np.allclose(da["lon"].values, COMMON_LON)
        assert "note" in da.attrs

    def test_grid_mismatch_raises(self, tmp_path):
        f = tmp_path / "IMD_observed.nc"
        _write_imd_observed(f, lat=COMMON_LAT + 1.0)

        with pytest.raises(HeppiGridMismatchError):
            load_imd_observed(f)


class TestLoadNcmrwfForecast:
    def test_loads_and_standardizes_dims(self, tmp_path):
        f = tmp_path / "NCMRWF_orig_forecast.nc"
        _write_ncmrwf_forecast(f, "NCMRWF_frcst")

        da = load_ncmrwf_forecast(tmp_path, variant="orig")

        assert da.dims == ("sample", "member", "lat", "lon")
        assert da.sizes["member"] == N_MEMBERS
        assert "orig" in da.attrs["note"]

    def test_unknown_variant_raises(self, tmp_path):
        with pytest.raises(ValueError, match="unknown variant"):
            load_ncmrwf_forecast(tmp_path, variant="bogus")

    def test_uqm_and_emos_variants_use_correct_filenames(self, tmp_path):
        _write_ncmrwf_forecast(tmp_path / "NCMRWF_UQM_forecast.nc", "NCMRWF_UQM_frcst")
        _write_ncmrwf_forecast(tmp_path / "NCMRWF_EMOS_forecast.nc", "NCMRWF_EMOS_frcst")

        uqm = load_ncmrwf_forecast(tmp_path, variant="uqm")
        emos = load_ncmrwf_forecast(tmp_path, variant="emos")

        assert uqm.name == "ncmrwf_uqm_forecast"
        assert emos.name == "ncmrwf_emos_forecast"
